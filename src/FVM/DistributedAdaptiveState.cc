/** @file DistributedAdaptiveState.cc
 * @brief Source-pinned named state transactions over actual sparse native maps.
 */
#include "FVM/DistributedAdaptiveState.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif
namespace SimpleFluid::VoF
{
using namespace Meshes;
using ::SimpleFluid::Meshes::AMR::adaptive_detail::Reader;
struct Writer
{
    ::SimpleFluid::Meshes::AMR::AdaptiveBytes data;
    std::size_t limit = std::numeric_limits<std::size_t>::max();
    void charge(std::size_t n) const
    {
        if (n > limit - data.size())
            throw std::invalid_argument("Adaptive state wire budget exhausted.");
    }
    void word(std::uint64_t value)
    {
        charge(8);
        for (unsigned i = 0; i < 8; ++i)
            data.push_back(char(value >> (8 * i)));
    }
    void bytes(std::span<const char> value)
    {
        charge(8);
        if (value.size() > limit - data.size() - 8)
            throw std::invalid_argument("Adaptive state wire budget exhausted.");
        word(value.size());
        data.insert(data.end(), value.begin(), value.end());
    }
};
struct DistributedAdaptiveState::Data
{
    std::shared_ptr<const Hierarchy> hierarchy;
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenNativeModel> frozen;
    AdaptiveStateOptions options;
    ::SimpleFluid::Meshes::AMR::AdaptiveNativePublication native;
    std::vector<Key> order;
    std::vector<AdaptiveHistoryFrame> histories;
    ::SimpleFluid::Meshes::AMR::AdaptiveBytes schema;
    std::map<std::pair<std::string, std::string>, std::shared_ptr<const Field>> fields;
    ::SimpleFluid::Meshes::AMR::HierarchyObservation observation;
    std::uint64_t collective_id = 0;
};
namespace
{
using Data = DistributedAdaptiveState::Data;
using Key = ::SimpleFluid::Meshes::AMR::AdaptiveCellKey;
using LO = DefaultTpetraTypes::local_ordinal_type;
auto model_comm(const Data& data)
{
    return data.hierarchy ? data.hierarchy->communicator() : data.frozen->communicator();
}
bool owns(const Data& data, const Key& key)
{
    return data.hierarchy ? data.hierarchy->owned().contains(key) : data.frozen->owned().contains(key);
}
bool native_valid(const Data& data)
{
    if (data.hierarchy)
        return data.hierarchy->geometry_unchanged() && data.hierarchy->observation() == data.observation &&
               data.hierarchy->native_partition() == data.native.partition;
    return data.frozen && data.native.partition->geometry_unchanged() &&
           data.native.partition->geometry()->geometry_epoch() == data.observation.geometry &&
           data.observation.identity == data.frozen->identity();
}

std::uint64_t new_state_id(Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    static std::atomic<std::uint64_t> sequence{1};
    std::uint64_t id = 0;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                id = sequence.fetch_add(1);
                if (!id)
                    throw std::overflow_error("Adaptive state identity exhausted.");
            }
        });
    Teuchos::broadcast(*comm, 0, 1, &id);
    return id;
}
void agree_id(std::uint64_t id, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::uint64_t low = 0, high = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &id, &low);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &id, &high);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (!id || low != high)
                throw std::invalid_argument("Rank-divergent adaptive state/control transaction.");
        });
}
::SimpleFluid::Meshes::AMR::AdaptiveBytes schema(const std::vector<AdaptiveHistoryFrame>& histories, std::size_t limit)
{
    Writer w{{}, limit};
    w.word(0x5346414d52535431ULL);
    w.word(histories.size());
    std::set<std::string> names;
    for (const auto& frame : histories)
    {
        if (frame.name.empty() || !names.insert(frame.name).second || !(frame.carrier_density > 0) ||
            !std::isfinite(frame.carrier_density) || !(frame.gas_density > 0) || !std::isfinite(frame.gas_density))
            throw std::invalid_argument("Invalid or duplicated adaptive history schema.");
        w.bytes(std::span(frame.name));
        w.word(std::bit_cast<std::uint64_t>(frame.carrier_density));
        w.word(std::bit_cast<std::uint64_t>(frame.gas_density));
        w.word(frame.reconstruction_ready);
        w.word(frame.inventories.size());
        std::set<std::string> quantities;
        for (const auto& q : frame.inventories)
        {
            if (q.name.empty() || q.name == "gas_volume" || q.name == "carrier_volume" ||
                !quantities.insert(q.name).second ||
                std::uint64_t(q.carrier) > std::uint64_t(InventoryCarrier::MixtureMass))
                throw std::invalid_argument("Invalid or duplicated adaptive quantity schema.");
            w.bytes(std::span(q.name));
            w.word(std::uint64_t(q.carrier));
            w.word(q.nonnegative);
        }
        w.word(frame.pairs.size());
        std::set<std::string> paired;
        for (const auto& pair : frame.pairs)
        {
            const auto number = std::find_if(frame.inventories.begin(), frame.inventories.end(),
                [&](const auto& q) { return q.name == pair.number; });
            const auto moles = std::find_if(frame.inventories.begin(), frame.inventories.end(),
                [&](const auto& q) { return q.name == pair.moles; });
            if (number == frame.inventories.end() || moles == frame.inventories.end() || pair.number == pair.moles ||
                !paired.insert(pair.number).second || !paired.insert(pair.moles).second ||
                number->carrier != moles->carrier || !number->nonnegative || !moles->nonnegative)
                throw std::invalid_argument("Invalid paired number/moles history metadata.");
            w.bytes(std::span(pair.number));
            w.bytes(std::span(pair.moles));
        }
    }
    return std::move(w.data);
}
void agree_schema(const ::SimpleFluid::Meshes::AMR::AdaptiveBytes& local, Teuchos::RCP<const Teuchos::Comm<int>> comm,
    std::size_t limit)
{
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (local.size() > limit || local.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Adaptive state schema exceeds byte budget.");
        });
    int size = comm->getRank() == 0 ? static_cast<int>(local.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &size);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (size < 0 || std::size_t(size) > limit)
                throw std::invalid_argument("Adaptive schema exceeds receiver budget.");
        });
    std::vector<char> canonical;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            canonical.resize(static_cast<std::size_t>(size));
            if (comm->getRank() == 0)
                canonical = local;
        });
    Teuchos::broadcast(*comm, 0, size, canonical.data());
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (canonical != local)
                throw std::invalid_argument("Adaptive state/history schemas differ across ranks.");
        });
}
void source_valid(const Data& data)
{
    agree_id(data.collective_id, model_comm(data));
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*model_comm(data),
        [&]
        {
            if (!native_valid(data) ||
                data.native.mesh->owned_cell_map().getRawPtr() != data.observation.partition_identity.get())
                throw std::invalid_argument("Adaptive state source native geometry/map observation is stale.");
            const auto map = data.native.mesh->owned_cell_map();
            if (map->getLocalNumElements() != data.order.size())
                throw std::invalid_argument("Adaptive state map row count changed.");
            for (std::size_t i = 0; i < data.order.size(); ++i)
                if (static_cast<std::uint64_t>(map->getGlobalElement(LO(i))) !=
                    data.native.owned_gids.at(data.order[i]))
                    throw std::invalid_argument("Adaptive state map GID order changed.");
        });
}
void actual_order(Data& data)
{
    data.order.clear();
    const auto map = data.native.mesh->owned_cell_map();
    if (map->getLocalNumElements() != data.native.owned_gids.size())
        throw std::invalid_argument("Named state owned registry/map count mismatch.");
    for (std::size_t i = 0; i < map->getLocalNumElements(); ++i)
    {
        const auto gid = static_cast<std::uint64_t>(map->getGlobalElement(LO(i)));
        const auto& key = data.native.local_keys.at(gid);
        if (data.native.owned_gids.at(key) != gid || !owns(data, key))
            throw std::invalid_argument("Named state key order differs from actual owned map.");
        data.order.push_back(key);
    }
}
void normalize(AdaptiveHistoryFrame& frame, std::size_t count)
{
    if (frame.phases.gas_volume.size() != count || frame.phases.carrier_volume.size() != count)
        throw std::invalid_argument("Adaptive history carrier arrays do not match owned map.");
    if (frame.phases.gas_halfspace.empty())
        frame.phases.gas_halfspace.resize(count);
    if (frame.phases.geometry_declaration.empty())
        frame.phases.geometry_declaration.assign(count, AMRPhaseGeometry::Plic);
    if (frame.phases.gas_halfspace.size() != count || frame.phases.geometry_declaration.size() != count)
        throw std::invalid_argument("Adaptive history reconstruction arrays do not match owned map.");
    for (const auto& q : frame.inventories)
        if (q.amount.size() != count)
            throw std::invalid_argument("Adaptive history amount array does not match owned map.");
}
void validate_values(const Data& data)
{
    const auto n = data.native.mesh->num_owned_cells();
    for (const auto& frame : data.histories)
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto gas = frame.phases.gas_volume[i], carrier = frame.phases.carrier_volume[i];
            const auto volume = data.native.mesh->cell_volume(LO(i));
            if (!std::isfinite(gas) || !std::isfinite(carrier) || gas < 0 || carrier < 0 ||
                std::abs((static_cast<long double>(gas) + carrier) - volume) >
                    64 * static_cast<long double>(std::numeric_limits<double>::epsilon()) * volume)
                throw std::invalid_argument(
                    "Adaptive history physical phase volumes are inconsistent with native geometry.");
            const auto declaration = frame.phases.geometry_declaration[i];
            if (std::uint64_t(declaration) > std::uint64_t(AMRPhaseGeometry::PureCarrier))
                throw std::invalid_argument("Invalid adaptive phase declaration.");
            if (frame.reconstruction_ready && gas > 0 && carrier > 0 && declaration == AMRPhaseGeometry::Plic)
            {
                const auto& plane = frame.phases.gas_halfspace[i];
                bool nonzero = false;
                for (auto x : plane.normal)
                {
                    if (!std::isfinite(x))
                        throw std::invalid_argument("Nonfinite adaptive reconstruction normal.");
                    nonzero = nonzero || x != 0;
                }
                for (auto x : plane.anchor)
                    if (!std::isfinite(x))
                        throw std::invalid_argument("Nonfinite adaptive reconstruction anchor.");
                if (!nonzero || !std::isfinite(plane.offset))
                    throw std::invalid_argument("Mixed adaptive history lacks qualified reconstruction.");
            }
            for (const auto& q : frame.inventories)
                if (!std::isfinite(q.amount[i]) || (q.nonnegative && q.amount[i] < 0) ||
                    ((q.carrier == InventoryCarrier::GasVolume && gas == 0) ||
                        (q.carrier == InventoryCarrier::LiquidVolume && carrier == 0)) &&
                        q.amount[i] != 0)
                    throw std::invalid_argument("Adaptive history contains invalid or exhausted-carrier inventory.");
            for (const auto& pair : frame.pairs)
            {
                const auto& number = *std::find_if(frame.inventories.begin(), frame.inventories.end(),
                    [&](const auto& q) { return q.name == pair.number; });
                const auto& moles = *std::find_if(frame.inventories.begin(), frame.inventories.end(),
                    [&](const auto& q) { return q.name == pair.moles; });
                if ((number.amount[i] == 0) != (moles.amount[i] == 0))
                    throw std::invalid_argument("Inconsistent paired number/moles zero state.");
            }
        }
}
void certify_reconstructions(const Data& data)
{
    // Frozen copies make no clipping claim; their model rejects every topology
    // change. Caller history readiness flags survive unchanged ownership moves.
    if (data.frozen)
        return;
    if (std::none_of(
            data.histories.begin(), data.histories.end(), [](const auto& frame) { return frame.reconstruction_ready; }))
        return;
    const auto comm = data.hierarchy->communicator();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> cells;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{data.hierarchy->options().byte_budget};
            for (const auto& key : data.order)
            {
                const auto gid = data.native.owned_gids.at(key);
                const auto& leaf = data.hierarchy->owned().at(key);
                auto geometry = data.hierarchy->geometry(leaf);
                if (!geometry)
                    throw std::invalid_argument("Qualified reconstruction requires every owned cell certificate.");
                budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveCellBinding) + geometry->storage_bytes());
                cells.push_back({key, gid, leaf.material, std::move(geometry)});
                relations.push_back({::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind::Unchanged, gid, gid});
            }
        });
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> plan;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            plan = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(
                data.hierarchy->observation(), data.hierarchy->observation(), cells, cells, relations);
        });
    ConservativeAMRRemap identity(data.native.mesh, data.native.mesh, plan, data.hierarchy->options().byte_budget);
    for (const auto& frame : data.histories)
        if (frame.reconstruction_ready)
        {
            AMRRemapOutput validation;
            (void) identity.apply(
                frame.phases, frame.inventories, validation, frame.carrier_density, frame.gas_density);
        }
}
void publish_fields(Data& data)
{
    const auto comm = model_comm(data);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm, [&] { validate_values(data); });
    certify_reconstructions(data);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{data.options.byte_budget};
            const auto owned = data.native.mesh->num_owned_cells(), overlap = data.native.mesh->num_cells();
            if (overlap > std::numeric_limits<std::size_t>::max() - owned)
                throw std::invalid_argument("Adaptive state field row count overflows.");
            const auto rows = owned + overlap;
            if (rows > std::numeric_limits<std::size_t>::max() / sizeof(double))
                throw std::invalid_argument("Adaptive state field payload overflows.");
            for (const auto& frame : data.histories)
            {
                for (std::size_t i = 0; i < frame.inventories.size() + 2; ++i)
                    budget.add(rows * sizeof(double));
            }
        });
    const auto add = [&](const std::string& history, const std::string& name, const std::vector<double>& values)
    {
        std::shared_ptr<DistributedAdaptiveState::Field> field;
        ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
            [&]
            {
                field = std::make_shared<DistributedAdaptiveState::Field>(data.native.mesh, history + "/" + name);
                for (std::size_t i = 0; i < values.size(); ++i)
                    field->set_owned_value(LO(i), values[i]);
            });
        field->sync_ghosts();
        ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(
            *comm, [&] { data.fields.emplace(std::pair{history, name}, std::move(field)); });
    };
    for (const auto& frame : data.histories)
    {
        add(frame.name, "gas_volume", frame.phases.gas_volume);
        add(frame.name, "carrier_volume", frame.phases.carrier_volume);
        for (const auto& q : frame.inventories)
            add(frame.name, q.name, q.amount);
    }
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (data.hierarchy)
                data.observation = data.hierarchy->observation();
            else
            {
                data.observation.identity = data.frozen->identity();
                data.observation.topology = data.frozen->topology_generation();
                data.observation.partition = data.frozen->partition_generation();
                data.observation.geometry = data.native.partition->geometry()->geometry_epoch();
                data.observation.native_geometry = data.native.partition->geometry().get();
                const auto map = data.native.mesh->owned_cell_map();
                auto holder = std::make_shared<const decltype(map)>(map);
                data.observation.partition_identity = std::shared_ptr<const void>(holder, map.getRawPtr());
                data.observation.communicator = comm.getRawPtr();
            }
        });
}
void write_cell(Writer& w, const Data& data, const Key& key, std::size_t i)
{
    w.bytes(::SimpleFluid::Meshes::AMR::encode_adaptive_key(key));
    for (const auto& frame : data.histories)
    {
        for (auto value : {frame.phases.gas_volume[i], frame.phases.carrier_volume[i]})
            w.word(std::bit_cast<std::uint64_t>(value));
        const auto& plane = frame.phases.gas_halfspace[i];
        for (auto value : plane.normal)
            w.word(std::bit_cast<std::uint64_t>(value));
        for (auto value : plane.anchor)
            w.word(std::bit_cast<std::uint64_t>(value));
        w.word(std::bit_cast<std::uint64_t>(plane.offset));
        w.word(std::uint64_t(frame.phases.geometry_declaration[i]));
        for (const auto& q : frame.inventories)
            w.word(std::bit_cast<std::uint64_t>(q.amount[i]));
    }
}
void check_frame_budget(const Data& source, std::size_t n)
{
    ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{source.options.byte_budget};
    const auto product = [&](std::size_t rows, std::size_t width)
    {
        if (width && rows > std::numeric_limits<std::size_t>::max() / width)
            throw std::invalid_argument("Adaptive state payload size overflows.");
        budget.add(rows * width);
    };
    product(source.histories.size(), sizeof(AdaptiveHistoryFrame));
    for (const auto& frame : source.histories)
    {
        budget.add(frame.name.size());
        product(frame.inventories.size(), sizeof(ConservedQuantity));
        product(frame.pairs.size(), sizeof(AdaptiveInventoryPair));
        product(n, 2 * sizeof(double) + sizeof(GeometryHalfspace) + sizeof(AMRPhaseGeometry));
        for (const auto& q : frame.inventories)
        {
            budget.add(q.name.size());
            product(n, sizeof(double));
        }
        for (const auto& pair : frame.pairs)
            budget.add(pair.number.size() + pair.moles.size());
    }
}
std::vector<AdaptiveHistoryFrame> empty_frames(const Data& source, std::size_t n)
{
    check_frame_budget(source, n);
    std::vector<AdaptiveHistoryFrame> frames;
    frames.reserve(source.histories.size());
    for (const auto& metadata : source.histories)
    {
        AdaptiveHistoryFrame frame;
        frame.name = metadata.name;
        frame.pairs = metadata.pairs;
        frame.carrier_density = metadata.carrier_density;
        frame.gas_density = metadata.gas_density;
        frame.reconstruction_ready = metadata.reconstruction_ready;
        for (const auto& q : metadata.inventories)
            frame.inventories.push_back({q.name, q.carrier, {}, q.nonnegative});
        frame.phases.gas_volume.assign(n, 0);
        frame.phases.carrier_volume.assign(n, 0);
        frame.phases.gas_halfspace.assign(n, {});
        frame.phases.geometry_declaration.assign(n, AMRPhaseGeometry::Plic);
        for (auto& q : frame.inventories)
            q.amount.assign(n, 0);
        frames.push_back(std::move(frame));
    }
    return frames;
}
void read_cell(Reader& r, std::vector<AdaptiveHistoryFrame>& frames, std::size_t i)
{
    for (auto& frame : frames)
    {
        frame.phases.gas_volume[i] = std::bit_cast<double>(r.word());
        frame.phases.carrier_volume[i] = std::bit_cast<double>(r.word());
        auto& plane = frame.phases.gas_halfspace[i];
        for (auto& x : plane.normal)
            x = std::bit_cast<double>(r.word());
        for (auto& x : plane.anchor)
            x = std::bit_cast<double>(r.word());
        plane.offset = std::bit_cast<double>(r.word());
        const auto declaration = r.word();
        if (declaration > 2)
            throw std::invalid_argument("Invalid migrated phase declaration.");
        frame.phases.geometry_declaration[i] = AMRPhaseGeometry(declaration);
        for (auto& q : frame.inventories)
            q.amount[i] = std::bit_cast<double>(r.word());
    }
}
} // namespace
DistributedAdaptiveState::DistributedAdaptiveState(std::shared_ptr<const Hierarchy> hierarchy,
    std::vector<Key> input_keys, std::vector<AdaptiveHistoryFrame> histories, AdaptiveStateOptions options)
{
    if (!hierarchy)
        throw std::invalid_argument("Adaptive state requires distributed hierarchy.");
    const auto comm = hierarchy->communicator();
    std::shared_ptr<Data> data;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            data = std::make_shared<Data>();
            if (!options.byte_budget)
                throw std::invalid_argument("Adaptive state payload budget is zero.");
            data->options = std::move(options);
            data->hierarchy = hierarchy;
            if (hierarchy->global_cells() == 0)
                throw std::invalid_argument("Adaptive state requires a nonempty global native domain.");
            if (input_keys.size() != hierarchy->owned().size() || histories.empty())
                throw std::invalid_argument("Adaptive state requires complete owned histories.");
            Data layout;
            layout.hierarchy = hierarchy;
            layout.options = data->options;
            layout.histories = std::move(histories);
            // Charge the complete requested row layout before adding missing
            // reconstruction arrays or creating an output copy.
            check_frame_budget(layout, input_keys.size());
            histories = std::move(layout.histories);
            for (auto& frame : histories)
                normalize(frame, input_keys.size());
            data->schema = schema(histories, std::min(hierarchy->options().byte_budget, data->options.byte_budget));
        });
    agree_schema(data->schema, comm, hierarchy->options().byte_budget);
    data->collective_id = new_state_id(comm);
    data->native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(
        *hierarchy, hierarchy->configuration_descriptor(), data->options.additional_full_ghosts);
    data->hierarchy = hierarchy->with_partition(data->native.partition, data->native.mesh->owned_cell_map());
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            std::map<Key, std::size_t> input;
            for (std::size_t i = 0; i < input_keys.size(); ++i)
                if (!hierarchy->owned().contains(input_keys[i]) || !input.emplace(input_keys[i], i).second)
                    throw std::invalid_argument("Adaptive history key order is foreign or duplicated.");
            Data source;
            source.hierarchy = hierarchy;
            source.options = data->options;
            source.histories = std::move(histories);
            data->histories = empty_frames(source, input.size());
            actual_order(*data);
            for (std::size_t i = 0; i < data->order.size(); ++i)
            {
                Writer w;
                write_cell(w, source, data->order[i], input.at(data->order[i]));
                Reader r{w.data};
                (void) ::SimpleFluid::Meshes::AMR::decode_adaptive_key(r.bytes());
                read_cell(r, data->histories, i);
                r.done();
            }
        });
    publish_fields(*data);
    data_ = std::move(data);
}
DistributedAdaptiveState::DistributedAdaptiveState(
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenNativeModel> model, std::vector<Key> input_keys,
    std::vector<AdaptiveHistoryFrame> histories, AdaptiveStateOptions options)
{
    if (!model)
        throw std::invalid_argument("Frozen state requires an accepted native model.");
    auto comm = model->communicator();
    std::shared_ptr<Data> data;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (!options.byte_budget || !model->global_cells() || histories.empty() ||
                input_keys.size() != model->owned().size())
                throw std::invalid_argument("Invalid frozen native state layout or payload budget.");
            data = std::make_shared<Data>();
            data->frozen = model;
            data->options = std::move(options);
            Data layout;
            layout.options = data->options;
            layout.histories = std::move(histories);
            check_frame_budget(layout, input_keys.size());
            histories = std::move(layout.histories);
            for (auto& frame : histories)
                normalize(frame, input_keys.size());
            data->schema = schema(histories, std::min(model->options().byte_budget, data->options.byte_budget));
        });
    agree_schema(data->schema, comm, model->options().byte_budget);
    auto publication = model->publication(data->options.additional_full_ghosts);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            data->native = {std::move(publication.partition), std::move(publication.mesh),
                std::move(publication.owned_gids), std::move(publication.local_keys), publication.storage};
            std::map<Key, std::size_t> input;
            for (std::size_t i = 0; i < input_keys.size(); ++i)
                if (!model->owned().contains(input_keys[i]) || !input.emplace(input_keys[i], i).second)
                    throw std::invalid_argument("Frozen state keys are foreign or duplicated.");
            Data original;
            original.options = data->options;
            original.histories = std::move(histories);
            actual_order(*data);
            data->histories = empty_frames(original, data->order.size());
            for (std::size_t i = 0; i < data->order.size(); ++i)
            {
                Writer w{{}, data->options.byte_budget};
                write_cell(w, original, data->order[i], input.at(data->order[i]));
                Reader r{w.data};
                (void) ::SimpleFluid::Meshes::AMR::decode_adaptive_key(r.bytes());
                read_cell(r, data->histories, i);
                r.done();
            }
        });
    data->collective_id = new_state_id(comm);
    publish_fields(*data);
    data_ = std::move(data);
}
const std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenNativeModel>&
DistributedAdaptiveState::frozen_model() const noexcept
{
    return data_->frozen;
}
DistributedAdaptiveState::Plan DistributedAdaptiveState::migrate_frozen(
    std::span<const int> destinations, const ::SimpleFluid::Meshes::AMR::AdaptiveBalancePlan* proposal) const
{
    source_valid(*data_);
    const auto model = data_->frozen;
    const auto comm = model->communicator();
    const auto limit = std::min(model->options().byte_budget, data_->options.byte_budget);
    std::vector<OwnerPacket> packets;
    std::uint64_t state_wire = 0;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (destinations.size() != data_->order.size())
                throw std::invalid_argument("Frozen state destination count differs from actual map.");
            if (proposal)
            {
                if (proposal->source.identity != model->identity() ||
                    proposal->source.topology != model->topology_generation() ||
                    proposal->source.partition != model->partition_generation() || proposal->keys != data_->order ||
                    proposal->revisions.size() != data_->order.size())
                    throw std::invalid_argument("Frozen state balance proposal belongs to a foreign source/map.");
                for (std::size_t i = 0; i < data_->order.size(); ++i)
                    if (proposal->revisions[i] != model->owned().at(data_->order[i]).revision())
                        throw std::invalid_argument("Frozen state balance proposal has stale cell revisions.");
            }
            std::map<int, std::vector<std::size_t>> groups;
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            for (std::size_t i = 0; i < destinations.size(); ++i)
            {
                if (destinations[i] < 0 || destinations[i] >= comm->getSize())
                    throw std::invalid_argument("Invalid frozen state destination.");
                budget.add(sizeof(std::size_t));
                groups[destinations[i]].push_back(i);
            }
            for (const auto& [rank, indices] : groups)
            {
                Writer w{{}, limit - budget.used};
                w.word(data_->collective_id);
                w.bytes(data_->schema);
                w.word(indices.size());
                for (auto i : indices)
                    write_cell(w, *data_, data_->order[i], i);
                budget.packet(w.data.size());
                if (rank != comm->getRank())
                {
                    if (w.data.size() > std::numeric_limits<std::uint64_t>::max() - state_wire - 24)
                        throw std::overflow_error("Frozen state application wire counter overflows.");
                    state_wire += w.data.size() + 24;
                }
                packets.push_back({rank, std::move(w.data)});
            }
        });
    auto prepared = model->prepare_migration(destinations);
    Plan result;
    result.source_ = data_;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (prepared.local_application_bytes() > std::numeric_limits<std::uint64_t>::max() - state_wire)
                throw std::overflow_error("Combined frozen native/state wire counter overflows.");
            result.migration_wire_bytes_ = prepared.local_application_bytes() + state_wire;
        });
    if (proposal)
    {
        result.balance_ = ::SimpleFluid::Meshes::AMR::qualify_balance_wire_bytes(
            *proposal, result.migration_wire_bytes_, comm, proposal->objective_options);
        if (!result.balance_->accepted)
        {
            result.candidate_ = data_;
            result.migration_wire_bytes_ = 0;
            return result;
        }
    }
    auto migration = model->plan_migration(prepared);
    result.native_packet_phase_peak_bytes_ = migration.native_packet_phase_peak_bytes();
    result.native_application_bytes_exchanged_ = migration.local_application_bytes();
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (migration.local_application_bytes() != prepared.local_application_bytes())
                throw std::invalid_argument("Frozen prepared native wire differs from actual migration.");
        });
    auto candidate = model->apply(migration);
    if (candidate->identity() == model->identity())
    {
        result.candidate_ = data_;
        result.migration_wire_bytes_ = 0;
        return result;
    }
    auto publication = candidate->publication(data_->options.additional_full_ghosts);
    auto incoming = exchange_owner_packets(packets, comm, limit);
    result.state_exchange_peak_bytes_ = incoming.stats.peak_scratch_bytes;
    std::shared_ptr<Data> next;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            next = std::make_shared<Data>();
            next->frozen = candidate;
            next->options = data_->options;
            next->schema = data_->schema;
            next->native = {std::move(publication.partition), std::move(publication.mesh),
                std::move(publication.owned_gids), std::move(publication.local_keys), publication.storage};
            actual_order(*next);
            next->histories = empty_frames(*data_, next->order.size());
        });
    std::vector<Key> received;
    std::vector<int> senders;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            std::map<Key, std::size_t> index;
            for (std::size_t i = 0; i < next->order.size(); ++i)
                index.emplace(next->order[i], i);
            std::vector<bool> seen(next->order.size());
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                if (r.word() != data_->collective_id || r.bytes() != data_->schema)
                    throw std::invalid_argument("Frozen named state source/schema mismatch.");
                const auto count = r.word();
                if (count > (r.data.size() - r.offset) / 8)
                    throw std::invalid_argument("Invalid frozen state row count.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto key = ::SimpleFluid::Meshes::AMR::decode_adaptive_key(r.bytes());
                    const auto row = index.at(key);
                    if (seen[row])
                        throw std::invalid_argument("Duplicate frozen named state row.");
                    seen[row] = true;
                    received.push_back(key);
                    senders.push_back(packet.peer);
                    read_cell(r, next->histories, row);
                }
                r.done();
            }
            if (!std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }))
                throw std::invalid_argument("Missing frozen named state row.");
        });
    const auto owners = model->lookup_cells(received);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < owners.size(); ++i)
                if (owners[i].owner != senders[i])
                    throw std::invalid_argument("Frozen state row supplied by a foreign owner.");
        });
    next->collective_id = new_state_id(comm);
    publish_fields(*next);
    result.candidate_ = std::move(next);
    return result;
}

const std::shared_ptr<const DistributedAdaptiveState::Hierarchy>& DistributedAdaptiveState::hierarchy() const noexcept
{
    return data_->hierarchy;
}
const std::shared_ptr<const MeshHandle<>>& DistributedAdaptiveState::mesh() const noexcept
{
    return data_->native.mesh;
}
const std::vector<DistributedAdaptiveState::Key>& DistributedAdaptiveState::owned_order() const noexcept
{
    return data_->order;
}
const std::vector<AdaptiveHistoryFrame>& DistributedAdaptiveState::histories() const noexcept
{
    return data_->histories;
}
std::shared_ptr<const DistributedAdaptiveState::Field> DistributedAdaptiveState::field(
    const std::string& history, const std::string& quantity) const
{
    return data_->fields.at({history, quantity});
}
std::shared_ptr<const DistributedAdaptiveState> DistributedAdaptiveState::Plan::candidate() const
{
    return candidate_ ? std::shared_ptr<const DistributedAdaptiveState>(new DistributedAdaptiveState(candidate_))
                      : nullptr;
}
std::shared_ptr<const DistributedAdaptiveState> DistributedAdaptiveState::apply(const Plan& plan) const
{
    source_valid(*data_);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*model_comm(*data_),
        [&]
        {
            if (plan.source_ != data_ || !plan.candidate_ || !native_valid(*plan.candidate_))
                throw std::invalid_argument("Foreign or stale adaptive state transaction.");
        });
    agree_id(plan.candidate_->collective_id, model_comm(*data_));
    std::shared_ptr<const DistributedAdaptiveState> accepted;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(
        *model_comm(*data_), [&] { accepted.reset(new DistributedAdaptiveState(plan.candidate_)); });
    return accepted;
}
DistributedAdaptiveState::Plan DistributedAdaptiveState::plan_migration(std::span<const int> destinations) const
{
    return data_->frozen ? migrate_frozen(destinations, nullptr) : migrate(destinations, nullptr);
}
DistributedAdaptiveState::Plan DistributedAdaptiveState::plan_migration(
    const ::SimpleFluid::Meshes::AMR::AdaptiveBalancePlan& proposal) const
{
    return data_->frozen ? migrate_frozen(proposal.localdestinations, &proposal)
                         : migrate(proposal.localdestinations, &proposal);
}
DistributedAdaptiveState::Plan DistributedAdaptiveState::migrate(
    std::span<const int> destinations, const ::SimpleFluid::Meshes::AMR::AdaptiveBalancePlan* proposal) const
{
    source_valid(*data_);
    const auto& hierarchy = *data_->hierarchy;
    const auto comm = hierarchy.communicator();
    const auto limit = std::min(hierarchy.options().byte_budget, data_->options.byte_budget);
    struct Group
    {
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> leaves;
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveFamily> families;
        std::vector<std::size_t> indices;
    };
    std::map<int, Group> groups;
    std::vector<OwnerPacket> packets;
    std::uint64_t moved_bytes = 0, state_application_bytes = 0, estimated_native_bytes = 0;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (destinations.size() != data_->order.size())
                throw std::invalid_argument("Adaptive state migration layout mismatch.");
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::map<::SimpleFluid::Meshes::AMR::AdaptiveBytes, int> routes;
            for (std::size_t i = 0; i < destinations.size(); ++i)
            {
                const auto rank = destinations[i];
                if (rank < 0 || rank >= comm->getSize())
                    throw std::invalid_argument("Invalid adaptive state destination.");
                budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveLeaf) + sizeof(std::size_t) +
                           sizeof(::SimpleFluid::Meshes::AMR::AdaptiveBytes) + 8 * sizeof(void*) +
                           ::SimpleFluid::Meshes::AMR::encode_adaptive_key(data_->order[i]).size());
                auto& group = groups[rank];
                group.indices.push_back(i);
                const auto& key = data_->order[i];
                group.leaves.push_back(hierarchy.owned().at(key));
                routes.emplace(::SimpleFluid::Meshes::AMR::encode_adaptive_key(key), rank);
            }
            for (const auto& [key, family] : hierarchy.families())
            {
                budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveFamily));
                groups.at(routes.at(family.anchor)).families.push_back(family);
            }
            for (const auto& [rank, group] : groups)
            {
                Writer w{{}, limit - budget.used};
                w.word(data_->collective_id);
                w.bytes(data_->schema);
                w.word(group.indices.size());
                for (auto i : group.indices)
                    write_cell(w, *data_, data_->order[i], i);
                budget.packet(w.data.size());
                if (rank != comm->getRank())
                {
                    const auto native =
                        ::SimpleFluid::Meshes::AMR::encode_adaptive_batch(group.leaves, group.families, limit);
                    const auto bytes = native.size() + w.data.size() + 48;
                    if (bytes > std::numeric_limits<std::uint64_t>::max() - moved_bytes)
                        throw std::overflow_error("Adaptive migration wire accounting overflow.");
                    moved_bytes += bytes;
                    state_application_bytes += w.data.size() + 24;
                    estimated_native_bytes += native.size() + 24;
                }
                packets.push_back({rank, std::move(w.data)});
            }
        });
    std::optional<::SimpleFluid::Meshes::AMR::AdaptiveBalancePlan> qualified;
    if (proposal)
        qualified = ::SimpleFluid::Meshes::AMR::qualify_balance_wire_bytes(
            *proposal, moved_bytes, comm, proposal->objective_options);
    auto migration = qualified ? hierarchy.plan_migration(*qualified) : hierarchy.plan_migration(destinations);
    Plan plan;
    plan.source_ = data_;
    plan.balance_ = qualified;
    plan.migration_wire_bytes_ = moved_bytes;
    plan.native_packet_phase_peak_bytes_ = migration.native_migration().peak_native_packet_buffers;
    plan.native_application_bytes_exchanged_ = migration.actual_migration_application_bytes();
    auto candidate = hierarchy.apply(migration);
    if (candidate->identity() == hierarchy.identity())
    {
        plan.candidate_ = data_;
        plan.migration_wire_bytes_ = 0;
        return plan;
    }
    auto native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(
        *candidate, candidate->configuration_descriptor(), data_->options.additional_full_ghosts);
    auto bound = candidate->with_partition(native.partition, native.mesh->owned_cell_map());
    bound = hierarchy.apply(migration, bound);
    auto incoming = exchange_owner_packets(packets, comm, limit);
    plan.state_exchange_peak_bytes_ = incoming.stats.peak_scratch_bytes;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (migration.actual_migration_application_bytes() != estimated_native_bytes ||
                moved_bytes != migration.actual_migration_application_bytes() + state_application_bytes)
                throw std::invalid_argument(
                    "Native/state packed migration accounting differs from actual application wire.");
            std::uint64_t measured_state = 0;
            std::set<int> peers;
            for (const auto& packet : packets)
                if (packet.peer != comm->getRank())
                {
                    measured_state += packet.bytes.size() + 8;
                    peers.insert(packet.peer);
                }
            measured_state += 16 * peers.size();
            if (measured_state != state_application_bytes)
                throw std::invalid_argument("State packet framing accounting mismatch.");
        });
    std::shared_ptr<Data> next;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            next = std::make_shared<Data>();
            next->options = data_->options;
            next->hierarchy = std::move(bound);
            next->native = std::move(native);
            next->schema = data_->schema;
            actual_order(*next);
            next->histories = empty_frames(*data_, next->order.size());
        });
    next->collective_id = new_state_id(comm);
    std::vector<Key> received_keys;
    std::vector<int> senders;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            std::map<Key, std::size_t> indices;
            for (std::size_t i = 0; i < next->order.size(); ++i)
                indices.emplace(next->order[i], i);
            std::vector<bool> seen(next->order.size());
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                if (r.word() != data_->collective_id)
                    throw std::invalid_argument("Migrated state belongs to a stale source control.");
                if (r.bytes() != data_->schema)
                    throw std::invalid_argument("Migrated state/history schema differs from source.");
                const auto count = r.word();
                if (count > (packet.bytes.size() - r.offset) / 8)
                    throw std::invalid_argument("Invalid migrated state row count.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto key = ::SimpleFluid::Meshes::AMR::decode_adaptive_key(r.bytes());
                    const auto index = indices.at(key);
                    if (seen[index])
                        throw std::invalid_argument("Duplicate migrated named state row.");
                    seen[index] = true;
                    received_keys.push_back(key);
                    senders.push_back(packet.peer);
                    read_cell(r, next->histories, index);
                }
                r.done();
            }
            if (!std::all_of(seen.begin(), seen.end(), [](bool seen) { return seen; }))
                throw std::invalid_argument("Missing migrated named state row.");
        });
    const auto owners = hierarchy.lookup_cells(received_keys);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < owners.size(); ++i)
                if (owners[i].owner != senders[i])
                    throw std::invalid_argument("Migrated named state came from a foreign owner.");
        });
    publish_fields(*next);
    plan.candidate_ = std::move(next);
    return plan;
}
std::shared_ptr<const DistributedAdaptiveState> DistributedAdaptiveState::with_reconstruction(
    std::span<const AdaptiveHistoryReconstruction> reconstructions) const
{
    source_valid(*data_);
    std::shared_ptr<Data> next;
    const auto comm = model_comm(*data_);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            if (data_->frozen)
                throw std::invalid_argument("Frozen native copies cannot certify new PLIC reconstruction.");
            next = std::make_shared<Data>(*data_);
            if (reconstructions.size() != next->histories.size())
                throw std::invalid_argument("Every adaptive history needs its own qualified reconstruction.");
            for (std::size_t i = 0; i < reconstructions.size(); ++i)
            {
                const auto& reconstruction = reconstructions[i];
                auto& frame = next->histories[i];
                if (reconstruction.name != frame.name || reconstruction.gas_halfspace.size() != next->order.size() ||
                    (!reconstruction.geometry_declaration.empty() &&
                        reconstruction.geometry_declaration.size() != next->order.size()))
                    throw std::invalid_argument("Adaptive target reconstruction does not match history/map schema.");
                frame.phases.gas_halfspace = reconstruction.gas_halfspace;
                frame.phases.geometry_declaration = reconstruction.geometry_declaration;
                frame.reconstruction_ready = true;
                normalize(frame, next->order.size());
            }
            validate_values(*next);
            next->schema =
                schema(next->histories, std::min(next->hierarchy->options().byte_budget, next->options.byte_budget));
        });
    agree_schema(next->schema, comm, data_->hierarchy->options().byte_budget);
    certify_reconstructions(*next);
    next->collective_id = new_state_id(comm);
    std::shared_ptr<const DistributedAdaptiveState> result;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(
        *comm, [&] { result.reset(new DistributedAdaptiveState(std::move(next))); });
    return result;
}

namespace
{
bool prefix(const std::vector<std::uint64_t>& first, const std::vector<std::uint64_t>& second)
{
    return first.size() <= second.size() && std::equal(first.begin(), first.end(), second.begin());
}
std::vector<Key> parents(const Key& key)
{
    std::vector<Key> result;
    if (const auto* cart = std::get_if<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(&key.ancestry))
    {
        if (cart->level)
        {
            auto parent = key;
            parent.ancestry = cart->parent();
            result.push_back(std::move(parent));
        }
    }
    else if (const auto* convex = std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&key.ancestry))
    {
        if (!convex->path.empty())
        {
            auto parent = key;
            std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(parent.ancestry).path.pop_back();
            result.push_back(std::move(parent));
        }
    }
    else if (const auto* product = std::get_if<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(&key.ancestry))
    {
        if (!product->base.path.empty())
        {
            auto parent = key;
            std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(parent.ancestry).base.path.pop_back();
            result.push_back(std::move(parent));
        }
        if (!product->axial.path.empty())
        {
            auto parent = key;
            std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(parent.ancestry).axial.path.pop_back();
            result.push_back(std::move(parent));
        }
        if (!product->base.path.empty() && !product->axial.path.empty())
        {
            auto parent = key;
            auto& p = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(parent.ancestry);
            p.base.path.pop_back();
            p.axial.path.pop_back();
            result.push_back(std::move(parent));
        }
    }
    return result;
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveSplit> splits(const Key& key)
{
    if (std::holds_alternative<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry))
        return {::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial};
    if (std::holds_alternative<::SimpleFluid::Meshes::AMR::StaticCellAncestry>(key.ancestry))
        return {};
    return {::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic};
}
std::optional<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind> relation(const Key& source, const Key& target)
{
    using Kind = ::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation::Kind;
    if (source == target)
        return Kind::Unchanged;
    if (source.region != target.region || source.ancestry.index() != target.ancestry.index())
        return {};
    const auto source_parents = parents(source), target_parents = parents(target);
    if (std::find(target_parents.begin(), target_parents.end(), source) != target_parents.end())
        return Kind::Refinement;
    if (std::find(source_parents.begin(), source_parents.end(), target) != source_parents.end())
        return Kind::Coarsening;
    if (const auto* a = std::get_if<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(&source.ancestry))
    {
        const auto& b = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(target.ancestry);
        if (a->base.root != b.base.root || a->base.root_template != b.base.root_template ||
            a->axial.root != b.axial.root ||
            std::abs(static_cast<long long>(a->base.path.size()) - static_cast<long long>(b.base.path.size())) > 1 ||
            std::abs(static_cast<long long>(a->axial.path.size()) - static_cast<long long>(b.axial.path.size())) > 1)
            return {};
        const bool base_forward = prefix(a->base.path, b.base.path), base_reverse = prefix(b.base.path, a->base.path);
        const bool z_forward = prefix(a->axial.path, b.axial.path), z_reverse = prefix(b.axial.path, a->axial.path);
        if ((base_forward && z_reverse) || (base_reverse && z_forward))
            return Kind::ProductIntersection;
    }
    return {};
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> catalog(
    const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& hierarchy, const std::vector<Key>& probes,
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes>& family_probes, std::size_t limit)
{
    const auto comm = hierarchy.communicator();
    const auto located = hierarchy.lookup_cells(probes, false);
    const auto family_locations = hierarchy.lookup_families(family_probes, false);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> existing;
    for (std::size_t i = 0; i < family_probes.size(); ++i)
        if (family_locations[i].owner >= 0)
            existing.push_back(family_probes[i]);
    const auto families = hierarchy.fetch_families(existing);
    std::vector<Key> child_queries;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            for (const auto& family : families)
                for (const auto& child : family.children)
                {
                    budget.add(sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(child).size());
                    child_queries.push_back(child);
                }
        });
    const auto children = hierarchy.lookup_cells(child_queries, false);
    std::vector<Key> keys;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::set<Key> complete;
            const auto add = [&](const Key& key)
            {
                if (!complete.contains(key))
                {
                    budget.add(
                        sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*));
                    complete.insert(key);
                }
            };
            for (const auto& [key, leaf] : hierarchy.owned())
                add(key);
            for (std::size_t i = 0; i < probes.size(); ++i)
                if (located[i].owner >= 0)
                    add(probes[i]);
            std::size_t offset = 0;
            for (const auto& family : families)
            {
                bool active = true;
                for (std::size_t i = 0; i < family.children.size(); ++i)
                    active = active && children[offset + i].owner >= 0;
                if (active)
                    for (const auto& child : family.children)
                        add(child);
                offset += family.children.size();
            }
            keys.assign(complete.begin(), complete.end());
        });
    const auto locations = hierarchy.lookup_cells(keys);
    const auto leaves = hierarchy.fetch_leaves(keys);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellBinding> result;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (leaves[i].key != keys[i] || leaves[i].revision != locations[i].revision)
                    throw std::invalid_argument("Overlap family owner/revision authentication failed.");
                auto geometry = hierarchy.geometry(leaves[i]);
                if (!geometry)
                    throw std::invalid_argument(
                        "Every topology-remap participant requires certified planar convex geometry.");
                budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveCellBinding) +
                           ::SimpleFluid::Meshes::AMR::encode_adaptive_key(keys[i]).size() + geometry->storage_bytes());
                result.push_back({keys[i], locations[i].address, leaves[i].material, std::move(geometry)});
            }
        });
    return result;
}
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> overlap(const Data& source, const Data& target)
{
    const auto comm = source.hierarchy->communicator();
    const auto limit = source.options.byte_budget;
    std::vector<Key> probes;
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> family_probes;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::set<Key> roots;
            const auto add = [&](const Key& key)
            {
                if (!roots.contains(key))
                {
                    budget.add(
                        sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*));
                    roots.insert(key);
                }
            };
            for (const auto* hierarchy : {source.hierarchy.get(), target.hierarchy.get()})
                for (const auto& [key, leaf] : hierarchy->owned())
                {
                    add(key);
                    for (const auto& parent : parents(key))
                        add(parent);
                }
            probes.assign(roots.begin(), roots.end());
            std::set<::SimpleFluid::Meshes::AMR::AdaptiveBytes> families;
            for (const auto& key : probes)
                for (auto split : splits(key))
                {
                    auto family = ::SimpleFluid::Meshes::AMR::adaptive_family_key(key, split);
                    if (!families.contains(family))
                    {
                        budget.add(
                            family.size() + sizeof(::SimpleFluid::Meshes::AMR::AdaptiveBytes) + 4 * sizeof(void*));
                        families.insert(std::move(family));
                    }
                }
            family_probes.assign(families.begin(), families.end());
        });
    auto from = catalog(*source.hierarchy, probes, family_probes, limit),
         to = catalog(*target.hierarchy, probes, family_probes, limit);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation> relations;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            ::SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{limit};
            std::map<Key, std::vector<std::size_t>> target_index;
            for (std::size_t i = 0; i < to.size(); ++i)
            {
                auto ancestors = parents(to[i].key);
                ancestors.push_back(to[i].key);
                for (const auto& ancestor : ancestors)
                {
                    budget.add(sizeof(Key) + ::SimpleFluid::Meshes::AMR::encode_adaptive_key(ancestor).size() +
                               sizeof(std::size_t) + 4 * sizeof(void*));
                    target_index[ancestor].push_back(i);
                }
            }
            for (const auto& cell : from)
            {
                std::set<std::size_t> candidates;
                auto ancestors = parents(cell.key);
                ancestors.push_back(cell.key);
                for (const auto& ancestor : ancestors)
                    if (auto found = target_index.find(ancestor); found != target_index.end())
                        candidates.insert(found->second.begin(), found->second.end());
                for (auto i : candidates)
                    if (auto kind = relation(cell.key, to[i].key))
                    {
                        if (cell.material != to[i].material)
                            throw std::invalid_argument("Topology transfer changes physical material membership.");
                        budget.add(sizeof(::SimpleFluid::Meshes::AMR::AdaptiveAncestryRelation));
                        relations.push_back({*kind, cell.gid, to[i].gid});
                    }
            }
        });
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan> result;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            result = ::SimpleFluid::Meshes::AMR::AdaptiveOverlapPlan::build(source.hierarchy->observation(),
                target.hierarchy->observation(), std::move(from), std::move(to), relations);
        });
    return result;
}
} // namespace
DistributedAdaptiveState::Plan DistributedAdaptiveState::plan_adaptation(
    std::span<const ::SimpleFluid::Meshes::AMR::AdaptiveMark> refine,
    std::span<const ::SimpleFluid::Meshes::AMR::AdaptiveBytes> coarsen,
    std::span<const Key> target_additional_full_ghosts) const
{
    source_valid(*data_);
    if (data_->frozen)
    {
        auto no_topology = data_->frozen->plan_adaptation(refine, coarsen);
        (void) data_->frozen->apply(no_topology);
        Plan plan;
        plan.source_ = plan.candidate_ = data_;
        return plan;
    }
    const auto comm = data_->hierarchy->communicator();
    auto geometry_plan = data_->hierarchy->plan_adaptation(refine, coarsen);
    auto candidate = data_->hierarchy->apply(geometry_plan);
    Plan plan;
    plan.source_ = data_;
    if (candidate->identity() == data_->hierarchy->identity())
    {
        plan.candidate_ = data_;
        return plan;
    }
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            for (const auto& frame : data_->histories)
                if (!frame.reconstruction_ready)
                    throw std::invalid_argument(
                        "Topology remap requires qualified target reconstruction for every history.");
        });
    auto native = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(
        *candidate, candidate->configuration_descriptor(), target_additional_full_ghosts);
    auto bound = candidate->with_partition(native.partition, native.mesh->owned_cell_map());
    bound = data_->hierarchy->apply(geometry_plan, bound);
    std::shared_ptr<Data> next;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            next = std::make_shared<Data>();
            next->options = data_->options;
            next->options.additional_full_ghosts.assign(
                target_additional_full_ghosts.begin(), target_additional_full_ghosts.end());
            next->hierarchy = std::move(bound);
            next->native = std::move(native);
            actual_order(*next);
            next->histories = empty_frames(*data_, next->order.size());
        });
    const auto certificate = overlap(*data_, *next);
    ConservativeAMRRemap transfer(
        data_->native.mesh, next->native.mesh, certificate, data_->hierarchy->options().byte_budget);
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(
        *comm, [&] { plan.conservation_.resize(data_->histories.size()); });
    for (std::size_t i = 0; i < data_->histories.size(); ++i)
    {
        const auto& source = data_->histories[i];
        auto& target = next->histories[i];
        AMRRemapOutput output;
        plan.conservation_[i] =
            transfer.apply(source.phases, source.inventories, output, source.carrier_density, source.gas_density);
        target.phases.gas_volume = std::move(output.gas_volume);
        target.phases.carrier_volume = std::move(output.carrier_volume);
        target.inventories = std::move(output.inventories);
        target.reconstruction_ready = false;
    }
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            next->schema =
                schema(next->histories, std::min(next->hierarchy->options().byte_budget, next->options.byte_budget));
        });
    agree_schema(next->schema, comm, data_->hierarchy->options().byte_budget);
    next->collective_id = new_state_id(comm);
    publish_fields(*next);
    plan.candidate_ = std::move(next);
    return plan;
}
AdaptiveStateStorage DistributedAdaptiveState::storage() const
{
    AdaptiveStateStorage report;
    report.native = data_->native.storage;
    if (data_->hierarchy)
        report.planner = data_->hierarchy->storage();
    else
        report.frozen_planner = data_->frozen->storage();
    report.schema_bytes = data_->schema.capacity();
    report.owned_cells = mesh()->num_owned_cells();
    report.full_ghost_cells = mesh()->num_cells() - report.owned_cells;
    report.outer_metric_cells = data_->native.local_keys.size() - mesh()->num_cells();
    for (const auto& map : {mesh()->owned_cell_map(), mesh()->overlap_cell_map(), mesh()->owned_face_map(),
             mesh()->overlap_face_map(), mesh()->boundary_face_map()})
        report.map_gid_payload_bytes += map->getLocalNumElements() * sizeof(DefaultTpetraTypes::global_ordinal_type);
    report.importer_routing_estimate_bytes =
        (mesh()->num_cells() - mesh()->num_owned_cells() + mesh()->overlap_face_map()->getLocalNumElements() -
            mesh()->owned_face_map()->getLocalNumElements()) *
        (sizeof(DefaultTpetraTypes::global_ordinal_type) + 2 * sizeof(int));
    report.state_metadata_estimate_bytes = sizeof(Data) + data_->order.capacity() * sizeof(Key) +
                                           data_->histories.capacity() * sizeof(AdaptiveHistoryFrame) +
                                           data_->options.additional_full_ghosts.capacity() * sizeof(Key);
    for (const auto& key : data_->order)
        report.state_metadata_estimate_bytes += ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size();
    report.native_registry_estimate_bytes =
        data_->native.owned_gids.size() * (sizeof(std::pair<const Key, std::uint64_t>) + 4 * sizeof(void*)) +
        data_->native.local_keys.size() * (sizeof(std::pair<const std::uint64_t, Key>) + 4 * sizeof(void*));
    for (const auto& [key, gid] : data_->native.owned_gids)
        report.native_registry_estimate_bytes += ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size();
    for (const auto& [gid, key] : data_->native.local_keys)
        report.native_registry_estimate_bytes += ::SimpleFluid::Meshes::AMR::encode_adaptive_key(key).size();
    for (const auto& [name, field] : data_->fields)
        report.state_metadata_estimate_bytes += sizeof(decltype(data_->fields)::value_type) + 4 * sizeof(void*) +
                                                name.first.capacity() + name.second.capacity();
    for (const auto& frame : data_->histories)
    {
        report.state_metadata_estimate_bytes += frame.name.capacity() +
                                                frame.inventories.capacity() * sizeof(ConservedQuantity) +
                                                frame.pairs.capacity() * sizeof(AdaptiveInventoryPair);
        for (const auto& pair : frame.pairs)
            report.state_metadata_estimate_bytes += pair.number.capacity() + pair.moles.capacity();
        report.frame_capacity_bytes +=
            (frame.phases.gas_volume.capacity() + frame.phases.carrier_volume.capacity()) * sizeof(double) +
            frame.phases.gas_halfspace.capacity() * sizeof(GeometryHalfspace) +
            frame.phases.geometry_declaration.capacity() * sizeof(AMRPhaseGeometry);
        for (const auto& q : frame.inventories)
            report.frame_capacity_bytes += q.amount.capacity() * sizeof(double) + q.name.capacity();
    }
    report.field_scalar_payload_bytes =
        data_->fields.size() * (mesh()->num_owned_cells() + mesh()->num_cells()) * sizeof(double);
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
#if defined(__APPLE__)
        report.process_peak_resident_bytes = static_cast<std::size_t>(usage.ru_maxrss);
#else
        report.process_peak_resident_bytes = static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
#endif
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::size_t total = 0, resident = 0;
    const auto page = sysconf(_SC_PAGESIZE);
    if (page > 0 && statm >> total >> resident &&
        resident <= std::numeric_limits<std::size_t>::max() / std::size_t(page))
        report.resident_bytes = resident * std::size_t(page);
#endif
    return report;
}
} // namespace SimpleFluid::VoF
