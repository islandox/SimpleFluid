/** @file ConservativeAMRRemap.cc
 * @brief Sparse native transfer with independently retained microscopic phase carriers.
 */
#include "FVM/ConservativeAMRRemap.hh"
#include "FVM/details/AMRRemapBudget.hh"
#include "FVM/details/GeometryWireProtocol.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Import.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace SimpleFluid::VoF
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Wide = long double;
constexpr Wide epsilon = std::numeric_limits<double>::epsilon();

void require(bool valid, const char* message)
{
    int local = valid, global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::invalid_argument(message);
}
void require_diagnostic(bool valid, const std::string& detail, const char* context)
{
    const auto comm = Tpetra::getDefaultComm();
    int local = valid ? std::numeric_limits<int>::max() : comm->getRank(), failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &failed);
    if (failed == std::numeric_limits<int>::max())
        return;
    std::string message = comm->getRank() == failed ? detail : std::string{};
    int length = static_cast<int>(std::min(message.size(), static_cast<std::size_t>(std::numeric_limits<int>::max())));
    Teuchos::broadcast(*comm, failed, 1, &length);
    message.resize(length);
    Teuchos::broadcast(*comm, failed, length, message.data());
    throw std::invalid_argument(std::string(context) + " [rank " + std::to_string(failed) + "] " + message);
}

void identical(double value, const char* message)
{
    double lo = 0, hi = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &value, &lo);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &value, &hi);
    require(std::isfinite(value) && lo == hi, message);
}
SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> owner_geometry(
    const ConservativeAMRRemap::Handle& mesh, LO cell)
{
    const auto* native = std::get_if<ConservativeAMRRemap::Handle::MultiRegionPtr>(&mesh.variant());
    if (!native || !*native)
        throw std::invalid_argument("AMR certificate requires native MultiRegion geometry.");
    return SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::from_native_cell(
        **native, mesh.cell_global_id(cell));
}

void validate_catalog_shells(const ConservativeAMRRemap::Handle& mesh,
    const std::vector<SimpleFluid::Meshes::AMR::AdaptiveCellBinding>& catalog, std::size_t byte_budget)
{
    namespace wire = SimpleFluid::Meshes::AMR::adaptive_detail;
    const auto comm = Tpetra::getDefaultComm();
    const auto owned_map = mesh.owned_cell_map();
    std::vector<GO> requested;
    std::map<GO, const SimpleFluid::Meshes::AMR::AdaptiveCellBinding*> owned_bindings;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{byte_budget};
            for (const auto& cell : catalog)
            {
                if (cell.gid > static_cast<std::uint64_t>(std::numeric_limits<GO>::max()))
                    throw std::invalid_argument("AMR catalog GID exceeds native map ordinal capacity.");
                budget.add(sizeof(GO) + sizeof(int));
                requested.push_back(static_cast<GO>(cell.gid));
                if (owned_map->isNodeGlobalElement(static_cast<GO>(cell.gid)))
                    owned_bindings.emplace(static_cast<GO>(cell.gid), &cell);
            }
        });
    std::vector<int> owners(requested.size(), -1);
    (void) owned_map->getRemoteIndexList(Teuchos::arrayViewFromVector(requested), Teuchos::arrayViewFromVector(owners));
    require(std::all_of(owners.begin(), owners.end(), [](int rank) { return rank >= 0; }),
        "AMR certificate catalog references a cell without a native owner.");
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{byte_budget};
            for (std::size_t i = 0; i < requested.size(); ++i)
            {
                wire::Writer w;
                geometry_wire::write_header(w, geometry_wire::Record::NativeShellRequest);
                w.word(i);
                w.word(static_cast<std::uint64_t>(requested[i]));
                budget.packet(w.data.size());
                requests.push_back({owners[i], std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, comm, byte_budget);
    requests.clear();
    requests.shrink_to_fit();
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{byte_budget};
            // Only requested certificates are materialized; no global maximum width,
            // global geometry catalog, or solution-field import is involved.
            std::map<GO, std::shared_ptr<const std::vector<char>>> certificates;
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                geometry_wire::read_header(r, geometry_wire::Record::NativeShellRequest);
                const auto ordinal = r.word(), gid = r.word();
                r.done();
                if (gid > static_cast<std::uint64_t>(std::numeric_limits<GO>::max()))
                    throw std::invalid_argument("Invalid owner certificate GID.");
                const auto binding = owned_bindings.at(static_cast<GO>(gid));
                auto& certificate = certificates[static_cast<GO>(gid)];
                if (!certificate)
                {
                    auto bytes = owner_geometry(mesh, owned_map->getLocalElement(static_cast<GO>(gid)))
                                     ->encode_certificate(byte_budget);
                    budget.add(bytes.size());
                    certificate = std::make_shared<const std::vector<char>>(std::move(bytes));
                }
                wire::Writer w;
                geometry_wire::write_header(w, geometry_wire::Record::NativeShellReply);
                w.word(ordinal);
                w.word(gid);
                w.bytes(SimpleFluid::Meshes::AMR::encode_adaptive_key(binding->key));
                w.word(binding->material);
                w.bytes(*certificate);
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    incoming.packets.clear();
    incoming.packets.shrink_to_fit();
    auto returned = exchange_owner_packets(replies, comm, byte_budget);
    wire::collective(*comm,
        [&]
        {
            std::vector<bool> seen(catalog.size());
            for (const auto& packet : returned.packets)
            {
                wire::Reader r{packet.bytes};
                geometry_wire::read_header(r, geometry_wire::Record::NativeShellReply);
                const auto ordinal = r.word(), gid = r.word();
                if (ordinal >= catalog.size() || seen[ordinal] || packet.peer != owners[ordinal] ||
                    gid != catalog[ordinal].gid)
                    throw std::invalid_argument("Unexpected or duplicate native owner certificate response.");
                const auto& expected = catalog[ordinal];
                const auto key = SimpleFluid::Meshes::AMR::decode_adaptive_key(r.bytes());
                const auto material = r.word();
                if (key != expected.key || material != expected.material)
                    throw std::invalid_argument(
                        "Catalog GID " + std::to_string(gid) + " ancestry/material differs from its owner binding.");
                const auto geometry =
                    SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::decode_certificate(
                        r.bytes(), byte_budget);
                r.done();
                if (!expected.geometry->equivalent(*geometry))
                    throw std::invalid_argument(
                        "Catalog GID " + std::to_string(gid) + " differs from its native owner's shell.");
                seen[ordinal] = true;
            }
            if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
                throw std::invalid_argument("Missing native owner certificate response.");
        });
}

Wide measure(const SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>& geometry)
{
    return geometry ? geometry->moments().volume.value : 0.0L;
}
double represent(Wide value)
{
    const double result = static_cast<double>(value);
    if (!std::isfinite(result) || (value != 0 && result == 0))
        throw std::invalid_argument("AMR remap result is not representable without losing a nonzero amount.");
    return result;
}
Wide carrier(InventoryCarrier kind, Wide bulk, Wide gas, Wide liquid, double rl, double rg)
{
    switch (kind)
    {
        case InventoryCarrier::BulkVolume:
            return bulk;
        case InventoryCarrier::GasVolume:
            return gas;
        case InventoryCarrier::LiquidVolume:
            return liquid;
        case InventoryCarrier::MixtureMass:
            return detail::weighted_amount(liquid, static_cast<Wide>(rl), 1.0L) +
                   detail::weighted_amount(gas, static_cast<Wide>(rg), 1.0L);
    }
    throw std::invalid_argument("Unknown AMR inventory carrier.");
}
} // namespace

struct ConservativeAMRRemap::Impl
{
    SP<const Handle> source, target;
    std::shared_ptr<const SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> plan;
    Teuchos::RCP<const Pack::map_type> source_map, target_map, donors;
    std::unique_ptr<Pack::import_type> importer;
    std::vector<const SimpleFluid::Meshes::AMR::AdaptiveCellBinding*> source_cells;
    std::map<std::uint64_t, const SimpleFluid::Meshes::AMR::AdaptiveCellBinding*> source_catalog;
    std::vector<const SimpleFluid::Meshes::AMR::AdaptiveOverlapRecord*> records;
    std::vector<Wide> target_volumes;
    std::uint64_t source_epoch = 0, target_epoch = 0;
    const void *source_identity = nullptr, *target_identity = nullptr;

    void validate() const
    {
        bool valid = false;
        try
        {
            valid = source->geometry_identity() == source_identity && target->geometry_identity() == target_identity &&
                    source->geometry_epoch() == source_epoch && target->geometry_epoch() == target_epoch &&
                    source->owned_cell_map() == source_map && target->owned_cell_map() == target_map;
        }
        catch (...)
        {
        }
        require(valid, "AMR remap endpoint geometry or owned map changed.");
    }
    std::vector<std::vector<double>> import(const std::vector<std::vector<double>>& values) const
    {
        Pack::multi_vector_type owned(source_map, values.size(), false), halo(donors, values.size(), false);
        for (std::size_t a = 0; a < values.size(); ++a)
        {
            auto data = owned.getDataNonConst(a);
            std::copy(values[a].begin(), values[a].end(), data.begin());
        }
        halo.doImport(owned, *importer, Tpetra::INSERT);
        std::vector<std::vector<double>> result(values.size());
        for (std::size_t a = 0; a < values.size(); ++a)
        {
            const auto data = halo.getData(a);
            result[a].assign(data.begin(), data.end());
        }
        return result;
    }
    std::vector<double> reverse(const std::vector<double>& values) const
    {
        Pack::multi_vector_type partial(donors, 1, false), owned(source_map, 1, true);
        auto data = partial.getDataNonConst(0);
        std::copy(values.begin(), values.end(), data.begin());
        owned.doExport(partial, *importer, Tpetra::ADD);
        const auto result = owned.getData(0);
        return {result.begin(), result.end()};
    }
};

ConservativeAMRRemap::ConservativeAMRRemap(SP<const Handle> source, SP<const Handle> target,
    std::shared_ptr<const SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> overlaps)
    : ConservativeAMRRemap(std::move(source), std::move(target), std::move(overlaps), 64 * 1024 * 1024)
{
}

ConservativeAMRRemap::ConservativeAMRRemap(SP<const Handle> source, SP<const Handle> target,
    std::shared_ptr<const SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> overlaps, std::size_t certificate_byte_budget)
    : impl_(std::make_unique<Impl>())
{
    require(source && target && overlaps, "AMR remap requires native endpoints and a certified overlap plan.");
    impl_->source = std::move(source);
    impl_->target = std::move(target);
    impl_->plan = std::move(overlaps);
    auto& p = *impl_;
    p.source_map = p.source->owned_cell_map();
    p.target_map = p.target->owned_cell_map();
    const auto comm = Tpetra::getDefaultComm();
    bool valid = false;
    try
    {
        p.source_epoch = p.source->geometry_epoch();
        p.target_epoch = p.target->geometry_epoch();
        p.source_identity = p.source->geometry_identity();
        p.target_identity = p.target->geometry_identity();
        const auto& a = p.plan->source_observation();
        const auto& b = p.plan->target_observation();
        valid = a.identity && b.identity && a.topology_identity && b.topology_identity && a.partition_identity &&
                b.partition_identity && a.native_geometry == p.source_identity &&
                b.native_geometry == p.target_identity && a.geometry == p.source_epoch &&
                b.geometry == p.target_epoch && a.communicator == comm.getRawPtr() &&
                b.communicator == comm.getRawPtr() && p.source_map->getComm().getRawPtr() == comm.getRawPtr() &&
                p.target_map->getComm().getRawPtr() == comm.getRawPtr();
    }
    catch (...)
    {
    }
    require(valid, "AMR remap plan observations do not match the exact native endpoints.");
    require(p.source_map->getGlobalNumElements() > 0 && p.target_map->getGlobalNumElements() > 0,
        "AMR remap requires nonempty global source and target domains.");
    require(p.source_map->isOneToOne() && p.target_map->isOneToOne(), "AMR remap requires unique owned cell maps.");
    p.source_cells.resize(p.source->num_owned_cells());
    p.target_volumes.resize(p.target->num_owned_cells());
    std::set<GO> donor_ids;
    std::string diagnostic;
    valid = true;
    try
    {
        for (const auto& cell : p.plan->source_cells())
        {
            p.source_catalog.emplace(cell.gid, &cell);
            const auto local = p.source_map->getLocalElement(static_cast<GO>(cell.gid));
            if (local < 0)
                continue;
            p.source_cells[local] = &cell;
            const Wide full = measure(cell.geometry), native = p.source->cell_volume(local);
            if (!(full > 0) || std::abs(full - native) > 64 * epsilon * full)
                throw std::invalid_argument(
                    "Source GID " + std::to_string(cell.gid) + " volume differs from its native cell.");
            if (!cell.geometry->equivalent(*owner_geometry(*p.source, local)))
                throw std::invalid_argument(
                    "Source GID " + std::to_string(cell.gid) + " shell differs from its native cell.");
        }
        for (const auto& cell : p.plan->target_cells())
        {
            const auto local = p.target_map->getLocalElement(static_cast<GO>(cell.gid));
            if (local < 0)
                continue;
            const Wide full = measure(cell.geometry), native = p.target->cell_volume(local);
            p.target_volumes[local] = full;
            if (!(full > 0) || std::abs(full - native) > 64 * epsilon * full)
                throw std::invalid_argument(
                    "Target GID " + std::to_string(cell.gid) + " volume differs from its native cell.");
            if (!cell.geometry->equivalent(*owner_geometry(*p.target, local)))
                throw std::invalid_argument(
                    "Target GID " + std::to_string(cell.gid) + " shell differs from its native cell.");
        }
        for (const auto& record : p.plan->records())
            if (p.target_map->isNodeGlobalElement(static_cast<GO>(record.target_gid)))
            {
                p.records.push_back(&record);
                donor_ids.insert(static_cast<GO>(record.source_gid));
            }
        valid = valid && std::all_of(p.source_cells.begin(), p.source_cells.end(), [](auto x) { return x; }) &&
                std::all_of(p.target_volumes.begin(), p.target_volumes.end(), [](auto x) { return x > 0; });
    }
    catch (const std::exception& error)
    {
        valid = false;
        diagnostic = error.what();
    }
    catch (...)
    {
        valid = false;
        diagnostic = "Unknown native shell validation failure.";
    }
    if (!valid && diagnostic.empty())
        diagnostic = "An owned cell is absent from its local ancestry catalog.";
    require_diagnostic(valid, diagnostic, "AMR certified cells do not match native owned geometry.");
    // Exact certificate inputs include remote siblings beyond active solution
    // donors. Bind every supplied source AND target shell to its native owner.
    validate_catalog_shells(*p.source, p.plan->source_cells(), certificate_byte_budget);
    validate_catalog_shells(*p.target, p.plan->target_cells(), certificate_byte_budget);
    p.donors = Teuchos::rcp(new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        Teuchos::arrayViewFromVector(std::vector<GO>(donor_ids.begin(), donor_ids.end())), GO{}, comm));
    p.importer = std::make_unique<Pack::import_type>(p.source_map, p.donors);
}
ConservativeAMRRemap::~ConservativeAMRRemap() = default;
ConservativeAMRRemap::ConservativeAMRRemap(ConservativeAMRRemap&&) noexcept = default;
ConservativeAMRRemap& ConservativeAMRRemap::operator=(ConservativeAMRRemap&&) noexcept = default;

AMRRemapReport ConservativeAMRRemap::apply(const AMRRemapPhases& phases,
    const std::vector<ConservedQuantity>& inventories, AMRRemapOutput& output, double rl, double rg) const
{
    const auto& p = *impl_;
    p.validate();
    identical(rl, "AMR liquid densities disagree.");
    identical(rg, "AMR gas densities disagree.");
    identical(static_cast<double>(inventories.size()), "AMR inventory counts disagree.");
    require(rl > 0 && rg > 0, "AMR remap requires positive phase densities.");
    const auto count = p.source_cells.size();
    bool valid = phases.gas_volume.size() == count && phases.carrier_volume.size() == count &&
                 phases.gas_halfspace.size() == count &&
                 (phases.geometry_declaration.empty() || phases.geometry_declaration.size() == count);
    std::set<std::string> names;
    for (const auto& quantity : inventories)
    {
        identical(static_cast<double>(quantity.carrier), "AMR inventory carriers disagree.");
        identical(quantity.nonnegative ? 1.0 : 0.0, "AMR inventory sign policies disagree.");
        identical(static_cast<double>(quantity.name.size()), "AMR inventory name lengths disagree.");
        require(quantity.name.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "AMR inventory name exceeds collective byte capacity.");
        std::vector<char> name(quantity.name.begin(), quantity.name.end());
        Teuchos::broadcast(*Tpetra::getDefaultComm(), 0, static_cast<int>(name.size()), name.data());
        require(std::equal(name.begin(), name.end(), quantity.name.begin()), "AMR inventory names disagree.");
        valid = valid && !quantity.name.empty() && names.insert(quantity.name).second &&
                static_cast<unsigned>(quantity.carrier) <= static_cast<unsigned>(InventoryCarrier::MixtureMass) &&
                quantity.amount.size() == count &&
                std::all_of(quantity.amount.begin(), quantity.amount.end(),
                    [&](double x) { return std::isfinite(x) && (!quantity.nonnegative || x >= 0); });
    }
    require(valid, "AMR remap requires finite named inventories and explicit owned phase arrays.");
    // Imported donor columns: bulk, physical gas/carrier, fallback markers, plane, amounts.
    std::vector<std::vector<double>> data(12 + inventories.size(), std::vector<double>(count));
    valid = true;
    try
    {
        for (std::size_t c = 0; c < count; ++c)
        {
            const auto& geometry = p.source_cells[c]->geometry;
            const Wide full = measure(geometry), gas = phases.gas_volume[c], liquid = phases.carrier_volume[c];
            const Wide bound = 64 * epsilon * full + geometry->moments().volume.binary64_absolute_error();
            if (!std::isfinite(gas) || !std::isfinite(liquid) || gas < 0 || liquid < 0 ||
                std::abs(gas + liquid - full) > bound || gas > full + bound || liquid > full + bound)
            {
                valid = false;
                continue;
            }
            Wide gg = gas == 0 ? 0 : full, gl = liquid == 0 ? 0 : full;
            const auto& plane = phases.gas_halfspace[c];
            if (!std::isfinite(plane.offset))
                throw std::invalid_argument("AMR gas halfspace offset must be finite.");
            const auto declaration =
                phases.geometry_declaration.empty() ? AMRPhaseGeometry::Plic : phases.geometry_declaration[c];
            if (declaration == AMRPhaseGeometry::PureGas)
            {
                gg = full;
                gl = 0;
            }
            else if (declaration == AMRPhaseGeometry::PureCarrier)
            {
                gg = 0;
                gl = full;
            }
            else if (declaration != AMRPhaseGeometry::Plic)
                throw std::invalid_argument("Unknown accepted AMR phase geometry declaration.");
            else if (gas > 0 && liquid > 0)
            {
                const auto clipped = geometry->clip(plane);
                gg = measure(clipped.inside);
                gl = measure(clipped.outside);
            }
            const bool fallback_gas = gas > 0 && gg == 0, fallback_liquid = liquid > 0 && gl == 0;
            valid = valid && std::abs(gg - gas) <= bound && std::abs(gl - liquid) <= bound &&
                    (!fallback_gas || (declaration == AMRPhaseGeometry::PureCarrier && gas <= bound)) &&
                    (!fallback_liquid || (declaration == AMRPhaseGeometry::PureGas && liquid <= bound));
            data[0][c] = represent(full);
            data[1][c] = static_cast<double>(gas);
            data[2][c] = static_cast<double>(liquid);
            data[3][c] = fallback_gas ? 1.0 : 0.0;
            data[4][c] = fallback_liquid ? 1.0 : 0.0;
            for (std::size_t a = 0; a < 3; ++a)
            {
                data[5 + a][c] = plane.normal[a];
                data[8 + a][c] = plane.anchor[a];
            }
            data[11][c] = plane.offset;
            for (std::size_t q = 0; q < inventories.size(); ++q)
                data[12 + q][c] = inventories[q].amount[c];
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(valid, "AMR physical phase volumes disagree materially with actual donor geometry.");
    const auto halo = p.import(data);
    std::vector<Wide> bulk(p.records.size()), raw_gas(p.records.size()), raw_liquid(p.records.size());
    std::vector<SimpleFluid::detail::CompensatedSum<Wide>> bulk_sum(p.donors->getLocalNumElements()),
        gas_sum(bulk_sum.size()), liquid_sum(bulk_sum.size()), occurrences(bulk_sum.size());
    valid = true;
    try
    {
        for (std::size_t r = 0; r < p.records.size(); ++r)
        {
            const auto& record = *p.records[r];
            const auto donor = p.donors->getLocalElement(static_cast<GO>(record.source_gid));
            bulk[r] = record.moments.volume.value;
            const Wide full = halo[0][donor], g = halo[1][donor], l = halo[2][donor];
            SimpleFluid::PLIC::GeometryHalfspace plane;
            for (std::size_t a = 0; a < 3; ++a)
            {
                plane.normal[a] = halo[5 + a][donor];
                plane.anchor[a] = halo[8 + a][donor];
            }
            plane.offset = halo[11][donor];
            Wide source_gas = g == 0 ? 0 : full, source_liquid = l == 0 ? 0 : full;
            Wide intersection_gas = g == 0 ? 0 : bulk[r], intersection_liquid = l == 0 ? 0 : bulk[r];
            if (g > 0 && l > 0 && halo[3][donor] == 0 && halo[4][donor] == 0)
            {
                const auto source_cut = p.source_catalog.at(record.source_gid)->geometry->clip(plane);
                const auto cut = record.intersection->clip(plane);
                source_gas = measure(source_cut.inside);
                source_liquid = measure(source_cut.outside);
                intersection_gas = measure(cut.inside);
                intersection_liquid = measure(cut.outside);
            }
            // Ratios retain actual tiny geometry without narrowing phase measures to double.
            raw_gas[r] = g == 0                ? 0
                         : halo[3][donor] != 0 ? detail::weighted_amount(1.0L, bulk[r], full)
                                               : detail::weighted_amount(1.0L, intersection_gas, source_gas);
            raw_liquid[r] = l == 0                ? 0
                            : halo[4][donor] != 0 ? detail::weighted_amount(1.0L, bulk[r], full)
                                                  : detail::weighted_amount(1.0L, intersection_liquid, source_liquid);
            bulk_sum[donor] += bulk[r];
            gas_sum[donor] += raw_gas[r];
            liquid_sum[donor] += raw_liquid[r];
            occurrences[donor] += 1;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(valid, "AMR actual overlap phase clipping failed.");
    auto reverse = [&](const auto& sums)
    {
        std::vector<double> local(sums.size());
        for (std::size_t c = 0; c < sums.size(); ++c)
            local[c] = static_cast<double>(sums[c].value());
        return p.reverse(local);
    };
    const auto covered = reverse(bulk_sum), clipped_gas = reverse(gas_sum), clipped_liquid = reverse(liquid_sum),
               terms = reverse(occurrences);
    valid = true;
    for (std::size_t c = 0; c < count; ++c)
    {
        const Wide full = data[0][c], arithmetic = (16 + 4 * terms[c]) * epsilon * full;
        valid = valid && std::abs(static_cast<Wide>(covered[c]) - full) <= arithmetic;
        const Wide weight_bound = (16 + 4 * terms[c]) * epsilon;
        if (data[1][c] > 0)
            valid = valid && std::abs(static_cast<Wide>(clipped_gas[c]) - 1) <= weight_bound;
        if (data[2][c] > 0)
            valid = valid && std::abs(static_cast<Wide>(clipped_liquid[c]) - 1) <= weight_bound;
    }
    require(valid, "AMR donor geometric coverage or phase closure exceeds its arithmetic/physical bound.");
    AMRRemapOutput candidate;
    candidate.inventories = inventories;
    const auto targets = p.target_volumes.size();
    std::vector<SimpleFluid::detail::CompensatedSum<Wide>> target_gas(targets), target_liquid(targets);
    std::vector<std::vector<SimpleFluid::detail::CompensatedSum<Wide>>> amounts(
        inventories.size(), std::vector<SimpleFluid::detail::CompensatedSum<Wide>>(targets));
    valid = true;
    try
    {
        for (std::size_t r = 0; r < p.records.size(); ++r)
        {
            const auto& record = *p.records[r];
            const auto donor = p.donors->getLocalElement(static_cast<GO>(record.source_gid));
            const auto target = p.target_map->getLocalElement(static_cast<GO>(record.target_gid));
            const Wide full = halo[0][donor], gas = halo[1][donor], liquid = halo[2][donor];
            const Wide gw = raw_gas[r];
            const Wide lw = raw_liquid[r];
            const Wide gv = detail::weighted_amount(gas, gw, 1.0L), lv = detail::weighted_amount(liquid, lw, 1.0L);
            target_gas[target] += gv;
            target_liquid[target] += lv;
            for (std::size_t q = 0; q < inventories.size(); ++q)
            {
                const auto kind = inventories[q].carrier;
                const Wide denominator = carrier(kind, full, gas, liquid, rl, rg);
                const Wide numerator = carrier(kind, bulk[r], gv, lv, rl, rg);
                if (!(denominator > 0))
                {
                    if (halo[12 + q][donor] != 0)
                        valid = false;
                }
                else
                    amounts[q][target] +=
                        detail::weighted_amount(static_cast<Wide>(halo[12 + q][donor]), numerator, denominator);
            }
        }
        candidate.gas_volume.resize(targets);
        candidate.carrier_volume.resize(targets);
        for (auto& quantity : candidate.inventories)
            quantity.amount.resize(targets);
        for (std::size_t c = 0; c < targets; ++c)
        {
            candidate.gas_volume[c] = represent(target_gas[c].value());
            candidate.carrier_volume[c] = represent(target_liquid[c].value());
            valid = valid && std::abs(target_gas[c].value() + target_liquid[c].value() - p.target_volumes[c]) <=
                                 128 * epsilon * p.target_volumes[c];
            for (std::size_t q = 0; q < inventories.size(); ++q)
            {
                const Wide value = amounts[q][c].value();
                valid = valid && (!inventories[q].nonnegative || value >= 0);
                candidate.inventories[q].amount[c] = represent(value);
            }
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(valid, "AMR transfer has an exhausted carrier, unrepresentable amount or invalid target phase budget.");
    AMRRemapReport report;
    const auto budgets = 2 + inventories.size();
    report.binary_exponent.resize(budgets);
    report.scaled_before.resize(budgets);
    report.scaled_after.resize(budgets);
    report.scaled_absolute_before.resize(budgets);
    report.scaled_absolute_after.resize(budgets);
    const auto comm = Tpetra::getDefaultComm();
    for (std::size_t q = 0; q < budgets; ++q)
    {
        const auto& before = q == 0 ? phases.gas_volume : q == 1 ? phases.carrier_volume : inventories[q - 2].amount;
        const auto& after = q == 0   ? candidate.gas_volume
                            : q == 1 ? candidate.carrier_volume
                                     : candidate.inventories[q - 2].amount;
        double maximum = 0;
        for (const auto x : before)
            maximum = std::max(maximum, std::abs(x));
        for (const auto x : after)
            maximum = std::max(maximum, std::abs(x));
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        if (maximum > 0)
            (void) std::frexp(maximum, &local_exponent);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        if (exponent == std::numeric_limits<int>::min())
            exponent = 0;
        // Scale each physical double before summation or communication. No physical
        // total is narrowed, and an entirely subnormal budget is normalized to O(1).
        SimpleFluid::detail::CompensatedSum<double> sb, sa, nb, na;
        for (const auto x : before)
        {
            const auto scaled = std::ldexp(x, -exponent);
            sb += scaled;
            nb += std::abs(scaled);
        }
        for (const auto x : after)
        {
            const auto scaled = std::ldexp(x, -exponent);
            sa += scaled;
            na += std::abs(scaled);
        }
        const std::array<double, 4> local{sb.value(), sa.value(), nb.value(), na.value()};
        require(comm->getSize() <= std::numeric_limits<int>::max() / 4,
            "AMR scalar diagnostic count exceeds MPI capacity.");
        std::vector<double> gathered(4 * static_cast<std::size_t>(comm->getSize()));
        Teuchos::gatherAll(*comm, 4, local.data(), static_cast<int>(gathered.size()), gathered.data());
        std::vector<std::array<double, 4>> rank_budgets(comm->getSize());
        for (std::size_t rank = 0; rank < rank_budgets.size(); ++rank)
            std::copy_n(gathered.begin() + 4 * rank, 4, rank_budgets[rank].begin());
        const Wide cells =
            static_cast<Wide>(p.source_map->getGlobalNumElements()) + p.target_map->getGlobalNumElements();
        // The only gathered data are four bounded scalar diagnostics per rank.
        const auto reduced = detail::sum_rank_budgets(rank_budgets, cells);
        const auto& global = reduced.values;
        const Wide norm = std::max(global[2], global[3]);
        // Underflow in normalization and final output rounding are bounded in the
        // same exponent as the physical gate. Neither term is a carrier/norm floor.
        const Wide scaled_quantum = std::ldexp(static_cast<Wide>(std::numeric_limits<double>::denorm_min()), -exponent);
        const Wide representability = cells * scaled_quantum;
        const Wide arithmetic = reduced.arithmetic;
        const Wide residual = std::abs(static_cast<Wide>(global[1]) - global[0]);
        require(std::all_of(global.begin(), global.end(), [](double x) { return std::isfinite(x); }) && norm >= 0 &&
                    residual + arithmetic <= 128 * epsilon * norm + representability,
            "AMR remap failed its exponent-scaled physical signed L1 conservation gate.");
        report.binary_exponent[q] = exponent;
        report.scaled_before[q] = global[0];
        report.scaled_after[q] = global[1];
        report.scaled_absolute_before[q] = global[2];
        report.scaled_absolute_after[q] = global[3];
    }
    output = std::move(candidate);
    return report;
}
} // namespace SimpleFluid::VoF
