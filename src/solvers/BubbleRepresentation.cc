/** @file BubbleRepresentation.cc
 * @brief Sparse deterministic birth planning and complete native gas components.
 */
#include "solvers/BubbleRepresentationInternal.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using ID = std::uint64_t;
using Box = SimpleFluid::Meshes::AMR::AMRBox;
using Vector = VoF::Vector;
using State = VoF::LowMachState;
using Reason = BubbleConversionReason;
using Sum = CompensatedSum<double>;
using LedgerSum = CompensatedSum<>;

void require_all(bool valid, const char* message)
{
    int local = valid ? 1 : 0, all = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &local, &all);
    if (!all)
        throw std::invalid_argument(message);
}
template<class Work> void collective(Work&& work, const char* message)
{
    std::exception_ptr error;
    try
    {
        work();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = error ? 1 : 0, any = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &any);
    if (any)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error(message);
    }
}
template<class T> T reduce(T local, Teuchos::EReductionType operation)
{
    T global{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), operation, 1, &local, &global);
    return global;
}
void identical(ID value)
{
    require_all(reduce(value, Teuchos::REDUCE_MIN) == reduce(value, Teuchos::REDUCE_MAX),
        "Representation integer metadata differ across ranks.");
}
void identical(double value)
{
    require_all(std::isfinite(value), "Representation controls must be finite.");
    require_all(reduce(value, Teuchos::REDUCE_MIN) == reduce(value, Teuchos::REDUCE_MAX),
        "Representation controls differ across ranks.");
}
void identical(const std::string& value)
{
    require_all(value.size() < 1000000, "Representation name is too long.");
    const auto comm = Tpetra::getDefaultComm();
    int length = comm->getRank() == 0 ? static_cast<int>(value.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &length);
    std::string copy = comm->getRank() == 0 ? value : std::string(static_cast<std::size_t>(length), '\0');
    if (length)
        Teuchos::broadcast(*comm, 0, length, copy.data());
    require_all(copy == value, "Representation names differ across ranks.");
}
double volume(const Box& b)
{
    return (b.upper[0] - b.lower[0]) * (b.upper[1] - b.lower[1]) * (b.upper[2] - b.lower[2]);
}
Vector center(const Box& b)
{
    return {std::midpoint(b.lower[0], b.upper[0]), std::midpoint(b.lower[1], b.upper[1]),
        std::midpoint(b.lower[2], b.upper[2])};
}
double diameter(double v)
{
    return std::cbrt(6 * v / std::numbers::pi);
}
double area(double n, double v)
{
    return n > 0 && v > 0 ? 4 * std::numbers::pi * n * std::pow(3 * v / (4 * std::numbers::pi * n), 2.0 / 3.0) : 0;
}
bool inside(const Vector& x, const Box& box)
{
    for (std::size_t a = 0; a < 3; ++a)
        if (x[a] < box.lower[a] || x[a] > box.upper[a])
            return false;
    return true;
}
bool intersects(const Box& a, const Box& b)
{
    for (std::size_t q = 0; q < 3; ++q)
        if (a.upper[q] <= b.lower[q] || b.upper[q] <= a.lower[q])
            return false;
    return true;
}
double finest_width(const SimpleFluid::Meshes::AMR::CartesianAMR& hierarchy)
{
    const auto& g = hierarchy.config();
    double h = 0;
    for (std::size_t a = 0; a < 3; ++a)
        h = std::max(
            h, g.extent[a] / (static_cast<double>(g.root_blocks[a]) * g.block_cells[a] * std::ldexp(1.0, g.max_level)));
    return h;
}
std::vector<ID> exchange_labels(const VoF::AdaptiveVoF& geometry, const std::vector<ID>& values)
{
    // Two exactly representable 32-bit words retain every uint64 label through
    // the native double-valued import without a lossy full-ID conversion.
    std::vector<double> lo(values.size()), hi(values.size());
    for (std::size_t c = 0; c < values.size(); ++c)
    {
        lo[c] = static_cast<std::uint32_t>(values[c]);
        hi[c] = values[c] >> 32;
    }
    lo = geometry.exchange_cells(lo);
    hi = geometry.exchange_cells(hi);
    std::vector<ID> result(lo.size());
    for (std::size_t c = 0; c < result.size(); ++c)
        result[c] = static_cast<ID>(lo[c]) | (static_cast<ID>(hi[c]) << 32);
    return result;
}
std::vector<ID> all_keys(const std::set<ID>& local)
{
    const auto comm = Tpetra::getDefaultComm();
    require_all(local.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()), "Too many gas components.");
    const int width = reduce(static_cast<int>(local.size()), Teuchos::REDUCE_MAX);
    if (width == 0)
        return {};
    require_all(
        width <= std::numeric_limits<int>::max() / comm->getSize(), "Gas component metadata exceed MPI capacity.");
    std::vector<ID> send(static_cast<std::size_t>(width), std::numeric_limits<ID>::max());
    std::copy(local.begin(), local.end(), send.begin());
    std::vector<ID> receive(send.size() * comm->getSize());
    Teuchos::gatherAll(*comm, width, send.data(), static_cast<int>(receive.size()), receive.data());
    std::sort(receive.begin(), receive.end());
    receive.erase(std::unique(receive.begin(), receive.end()), receive.end());
    if (!receive.empty() && receive.back() == std::numeric_limits<ID>::max())
        receive.pop_back();
    return receive;
}

struct Components
{
    std::vector<ResolvedBubbleComponent> summaries;
    std::vector<ID> owned_labels;
};
Components components(
    const AdaptiveVoFSolver& flow, const VoF::AdaptiveVoF& geometry, const std::vector<Box>& headspace)
{
    const auto& mesh = *geometry.mesh();
    const auto& state = flow.low_mach_state();
    const auto n = mesh.num_owned_cells();
    const auto aperture = geometry.gas_face_aperture(flow.gas_fraction());
    Components result;
    result.owned_labels.resize(n);
    for (std::size_t c = 0; c < n; ++c)
        if (state.resolved_fraction[c] > 0)
            result.owned_labels[c] = static_cast<ID>(mesh.cell_global_id(static_cast<LO>(c))) + 1;
    std::vector<ID> halo;
    ID iterations = 0;
    for (;;)
    {
        halo = exchange_labels(geometry, result.owned_labels);
        int changed = 0;
        {
            const auto execution = mesh.acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
                if (result.owned_labels[c])
                {
                    ID label = result.owned_labels[c];
                    mesh.visit_cell_faces(static_cast<LO>(c),
                        [&](LO f)
                        {
                            if (mesh.is_geometry_exterior_face(f) || aperture[f] <= 0)
                                return;
                            const auto other = mesh.opposite_cell(f, static_cast<LO>(c));
                            if (halo[other])
                                label = std::min(label, halo[other]);
                        });
                    changed = changed || label != result.owned_labels[c];
                    result.owned_labels[c] = label;
                }
        }
        if (reduce(changed, Teuchos::REDUCE_MAX) == 0)
            break;
        require_all(
            ++iterations <= mesh.owned_cell_map()->getGlobalNumElements(), "Gas component labeling did not converge.");
    }
    std::set<ID> local_keys;
    for (ID key : result.owned_labels)
        if (key)
            local_keys.insert(key);
    const auto keys = all_keys(local_keys);
    std::map<ID, std::size_t> index;
    for (std::size_t i = 0; i < keys.size(); ++i)
        index.emplace(keys[i], i);
    struct Local
    {
        std::array<Sum, 7> sum;
        std::array<double, 3> lower, upper;
        ID cells = 0;
        double maximum_width = 0;
        int wall = 0, head = 0, periodic = 0;
        Local()
        {
            lower.fill(std::numeric_limits<double>::infinity());
            upper.fill(-std::numeric_limits<double>::infinity());
        }
    };
    std::vector<Local> data(keys.size());
    {
        const auto execution = mesh.acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
            if (result.owned_labels[c])
            {
                auto& d = data[index.at(result.owned_labels[c])];
                ++d.cells;
                const auto box = flow.hierarchy()->cell_box(mesh.cell_global_id(static_cast<LO>(c)));
                const auto widths = flow.hierarchy()->block_spacing(
                    flow.hierarchy()->cell_block(mesh.cell_global_id(static_cast<LO>(c))));
                d.maximum_width = std::max(d.maximum_width, *std::max_element(widths.begin(), widths.end()));
                const auto x = center(box);
                const double v = state.resolved_fraction[c] * mesh.cell_volume(static_cast<LO>(c));
                d.sum[0] += v;
                d.sum[1] += state.resolved_hydrogen_moles[c];
                d.sum[2] += state.resolved_inert_moles[c];
                for (std::size_t a = 0; a < 3; ++a)
                {
                    d.sum[3 + a] += v * x[a];
                    d.lower[a] = std::min(d.lower[a], box.lower[a]);
                    d.upper[a] = std::max(d.upper[a], box.upper[a]);
                }
                for (const auto& region : headspace)
                    d.head = d.head || inside(x, region);
                mesh.visit_cell_faces(static_cast<LO>(c),
                    [&](LO f)
                    {
                        if (aperture[f] <= 0)
                            return;
                        if (mesh.is_geometry_exterior_face(f))
                        {
                            d.wall = 1;
                            return;
                        }
                        const auto other = mesh.opposite_cell(f, static_cast<LO>(c));
                        if (halo[other] != result.owned_labels[c])
                            return;
                        const auto ox = center(flow.hierarchy()->cell_box(mesh.cell_global_id(other)));
                        const auto image = mesh.cell_center_vector(f, static_cast<LO>(c));
                        for (std::size_t a = 0; a < 3; ++a)
                            if (flow.hierarchy()->config().periodic[a] &&
                                std::abs(ox[a] - x[a] - image.component(a)) >
                                    0.5 * flow.hierarchy()->config().extent[a])
                                d.periodic = 1;
                    });
            }
    }
    const auto patches = geometry.interface_patches(flow.gas_fraction());
    for (const auto& patch : patches)
    {
        const auto cell = mesh.overlap_cell_map()->getLocalElement(static_cast<GO>(patch.gas_cell));
        if (cell >= 0 && halo[cell])
            data[index.at(halo[cell])].sum[6] += patch.area;
    }
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        ResolvedBubbleComponent c;
        c.label = keys[i];
        c.topology_generation = flow.topology_generation();
        auto& d = data[i];
        std::array<double, 7> values{};
        for (std::size_t a = 0; a < values.size(); ++a)
            values[a] = reduce(d.sum[a].value(), Teuchos::REDUCE_SUM);
        c.cells = static_cast<std::size_t>(reduce(d.cells, Teuchos::REDUCE_SUM));
        c.gas_volume = values[0];
        c.hydrogen_moles = values[1];
        c.inert_moles = values[2];
        c.surface_area = values[6];
        c.diameter = diameter(c.gas_volume);
        c.maximum_cell_width = reduce(d.maximum_width, Teuchos::REDUCE_MAX);
        for (std::size_t a = 0; a < 3; ++a)
        {
            c.centroid[a] = values[3 + a] / c.gas_volume;
            c.bounds.lower[a] = reduce(d.lower[a], Teuchos::REDUCE_MIN);
            c.bounds.upper[a] = reduce(d.upper[a], Teuchos::REDUCE_MAX);
        }
        c.touches_wall = reduce(d.wall, Teuchos::REDUCE_MAX) != 0;
        c.touches_headspace = reduce(d.head, Teuchos::REDUCE_MAX) != 0;
        c.crosses_periodic_seam = reduce(d.periodic, Teuchos::REDUCE_MAX) != 0;
        result.summaries.push_back(std::move(c));
    }
    return result;
}

/** All ranks participate; only the event coordinator imports bounded stencil fields. */
class Samples
{
public:
    Teuchos::RCP<const Pack::map_type> target;
    std::unique_ptr<Pack::import_type> importer;
    std::map<ID, std::size_t> index;
    std::vector<ID> ids;
    Samples(const AdaptiveVoFSolver& flow, const std::set<ID>& requested)
    {
        const auto source = flow.mesh()->owned_cell_map();
        std::vector<GO> keys;
        if (source->getComm()->getRank() == 0)
            for (ID id : requested)
            {
                index.emplace(id, keys.size());
                ids.push_back(id);
                keys.push_back(static_cast<GO>(id));
            }
        target = Teuchos::rcp(new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            Teuchos::arrayViewFromVector(keys), GO{}, source->getComm()));
        importer = std::make_unique<Pack::import_type>(source, target);
    }
    std::vector<double> field(const AdaptiveVoFSolver& flow, const std::vector<double>& values) const
    {
        Pack::vector_type source(flow.mesh()->owned_cell_map(), true), out(target, true);
        {
            auto v = source.getDataNonConst();
            std::copy(values.begin(), values.end(), v.begin());
        }
        out.doImport(source, *importer, Tpetra::INSERT);
        const auto data = out.getData();
        return {data.begin(), data.end()};
    }
    std::vector<double> to_owned(const AdaptiveVoFSolver& flow, const std::vector<double>& values) const
    {
        Pack::vector_type source(target, true), out(flow.mesh()->owned_cell_map(), true);
        {
            auto v = source.getDataNonConst();
            std::copy(values.begin(), values.end(), v.begin());
        }
        out.doExport(source, *importer, Tpetra::ADD);
        const auto data = out.getData();
        return {data.begin(), data.end()};
    }
};

Box expanded(const Box& box, double factor)
{
    Box result;
    const auto middle = center(box);
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double radius = 0.5 * factor * (box.upper[a] - box.lower[a]);
        result.lower[a] = middle[a] - radius;
        result.upper[a] = middle[a] + radius;
    }
    return result;
}
Reason contain_box(Box& box, const SimpleFluid::Meshes::AMR::AMRGrid& grid, bool clip_wall)
{
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double lo = grid.origin[a], hi = lo + grid.extent[a];
        const double allowance =
            16 * std::numeric_limits<double>::epsilon() * std::max({std::abs(lo), std::abs(hi), grid.extent[a]});
        if (box.lower[a] < lo && box.lower[a] >= lo - allowance)
            box.lower[a] = lo;
        if (box.upper[a] > hi && box.upper[a] <= hi + allowance)
            box.upper[a] = hi;
        if (box.lower[a] < lo || box.upper[a] > hi)
        {
            if (grid.periodic[a])
                return Reason::PeriodicStencilUnsupported;
            if (!clip_wall)
                return Reason::AttachedToBoundary;
            box.lower[a] = std::max(box.lower[a], lo);
            box.upper[a] = std::min(box.upper[a], hi);
        }
    }
    return Reason::Ready;
}

/** Volume-exact piecewise-planar spherical birth, without an analytic curvature input. */
std::map<ID, double> sphere(const SimpleFluid::Meshes::AMR::CartesianAMR& hierarchy, const Box& stencil,
    const Vector& seed, double desired, const std::map<ID, double>& native_volume, const std::map<ID, double>& capacity)
{
    const auto cells = hierarchy.overlaps(stencil);
    std::map<ID, double> result;
    const double radius = 0.5 * diameter(desired);
    const auto evaluate = [&](double square, std::map<ID, double>* output)
    {
        LedgerSum sum;
        for (const auto& cell : cells)
        {
            const auto box = hierarchy.cell_box(cell.cell);
            const auto x = center(box);
            Vector normal{}, lo{}, hi{};
            double r2 = 0;
            for (std::size_t a = 0; a < 3; ++a)
            {
                normal[a] = x[a] - seed[a];
                r2 += normal[a] * normal[a];
                lo[a] = box.lower[a] - x[a];
                hi[a] = box.upper[a] - x[a];
            }
            const double distance = std::sqrt(r2);
            const double box_volume = volume(box);
            double v = capacity.at(cell.cell);
            if (distance > 0)
            {
                for (double& n : normal)
                    n /= distance;
                const double fraction =
                    VoF::GeometricVoF::box_plane_volume(lo, hi, normal, (square - r2) / (2 * distance)) / box_volume;
                // The PLIC box gives a geometric fraction; native phase packets
                // carry volumes. A saturated birth cannot borrow an inherited
                // EOS occupancy remainder to reach the nominal box volume.
                v = std::min(capacity.at(cell.cell), fraction * native_volume.at(cell.cell));
            }
            sum += v;
            if (output && v > 0)
                (*output)[cell.cell] = v;
        }
        return static_cast<double>(sum.value());
    };
    double lower = 0, upper = 4 * radius * radius;
    if (evaluate(lower, nullptr) > desired || evaluate(upper, nullptr) < desired)
        throw std::runtime_error("Birth stencil cannot geometrically represent its whole-bubble volume.");
    for (int step = 0; step < 96; ++step)
    {
        const double middle = std::midpoint(lower, upper);
        if (middle == lower || middle == upper)
            break;
        if (evaluate(middle, nullptr) > desired)
            upper = middle;
        else
            lower = middle;
    }
    const double measured = evaluate(std::midpoint(lower, upper), &result);
    const double residual = desired - measured;
    if (std::abs(residual) > 256 * std::numeric_limits<double>::epsilon() * desired)
        throw std::runtime_error("Birth geometry volume inversion did not reach floating-point closure.");
    if (residual != 0)
    {
        const auto receiver = std::max_element(result.begin(), result.end(),
            [&](const auto& a, const auto& b)
            {
                const double sa = residual > 0 ? capacity.at(a.first) - a.second : a.second;
                const double sb = residual > 0 ? capacity.at(b.first) - b.second : b.second;
                return sa < sb;
            });
        if (receiver == result.end() || receiver->second + residual < 0 ||
            receiver->second + residual > capacity.at(receiver->first))
            throw std::runtime_error("Birth geometry has no roundoff-correction support.");
        receiver->second += residual;
    }
    return result;
}
} // namespace

BubbleRepresentationEngine::BubbleRepresentationEngine(BubbleRepresentationOptions options,
    BubbleGasPhysics::MolarPivotGrid pivots, std::vector<std::string> populations)
    : options_(options), pivots_(std::move(pivots)), populations_(std::move(populations)),
      state_(std::make_shared<const Snapshot>())
{
    for (double x : {options.promotion_diameter, options.demotion_diameter, options.demotion_neighborhood_factor,
             options.maximum_dispersed_fraction, options.maximum_demoted_extent_ratio})
        identical(x);
    for (ID x : std::array<ID, 5>{static_cast<ID>(options.enabled), options.promotion_cells_per_diameter,
             options.demotion_cells_per_diameter, options.interface_guard_cells,
             static_cast<ID>(options.maximum_event_cells)})
        identical(x);
    require_all(options.promotion_diameter > options.demotion_diameter && options.demotion_diameter > 0 &&
                    options.promotion_cells_per_diameter >= 2 && options.demotion_cells_per_diameter > 0 &&
                    options.interface_guard_cells > 0 && options.demotion_neighborhood_factor > 1 &&
                    options.maximum_dispersed_fraction > 0 && options.maximum_dispersed_fraction < 1 &&
                    options.maximum_demoted_extent_ratio >= 1 && options.maximum_event_cells > 0,
        "Invalid representation thresholds, hysteresis, or local redistribution controls.");
    require_all(pivots_.mol_per_bubble.size() == populations_.size(), "Representation molar bins and names disagree.");
}
const BubbleRepresentationEngine::Snapshot& BubbleRepresentationEngine::snapshot() const noexcept
{
    return *state_;
}
std::shared_ptr<const BubbleRepresentationEngine::Snapshot> BubbleRepresentationEngine::snapshot_handle() const noexcept
{
    return state_;
}
std::shared_ptr<const BubbleRepresentationEngine::Snapshot> BubbleRepresentationEngine::invalidated_snapshot() const
{
    auto next = std::make_shared<Snapshot>(*state_);
    next->origins.clear();
    for (auto& record : next->births)
        if (record.status != BubbleBirthStatus::Promoted)
        {
            record.status = BubbleBirthStatus::Pending;
            record.reserved_number = 0;
            record.source_revision = 0;
            record.source_topology = 0;
        }
    return next;
}
void BubbleRepresentationEngine::publish(std::shared_ptr<const Snapshot> state, ID revision) noexcept
{
    state_ = std::move(state);
    binding_revision_ = revision;
}
void BubbleRepresentationEngine::request_births(const std::vector<BubbleBirthSeed>& seeds)
{
    identical(static_cast<ID>(seeds.size()));
    auto next = std::make_shared<Snapshot>(*state_);
    for (const auto& seed : seeds)
    {
        identical(seed.id);
        identical(seed.population);
        for (std::size_t a = 0; a < 3; ++a)
        {
            identical(seed.center[a]);
            identical(seed.donor_region.lower[a]);
            identical(seed.donor_region.upper[a]);
        }
        require_all(std::find(populations_.begin(), populations_.end(), seed.population) != populations_.end(),
            "A birth requires a registered eligible fixed-molar population.");
        bool valid = true;
        for (std::size_t a = 0; a < 3; ++a)
            valid = valid && seed.donor_region.upper[a] > seed.donor_region.lower[a];
        require_all(valid, "Birth donor regions must have finite positive extent.");
        const auto old =
            std::find_if(next->births.begin(), next->births.end(), [&](const auto& b) { return b.seed.id == seed.id; });
        if (old != next->births.end())
            require_all(old->seed.center == seed.center && old->seed.population == seed.population &&
                            old->seed.donor_region.lower == seed.donor_region.lower &&
                            old->seed.donor_region.upper == seed.donor_region.upper,
                "A persistent birth ID cannot be reused with a different payload.");
        else
            next->births.push_back({seed});
    }
    std::sort(
        next->births.begin(), next->births.end(), [](const auto& a, const auto& b) { return a.seed.id < b.seed.id; });
    state_ = std::move(next);
}

BubbleRepresentationEngine::Prepared BubbleRepresentationEngine::prepare(
    const AdaptiveVoFSolver& flow, const std::vector<Box>& headspace) const
{
    require_all(flow.low_mach_enabled(), "Representation changes require initialized low-Mach material inventories.");
    VoF::AdaptiveVoF geometry(flow.hierarchy());
    auto catalog = components(flow, geometry, headspace);
    Prepared prepared;
    auto next = std::make_shared<Snapshot>(*state_);
    prepared.request.source_hierarchy = flow.hierarchy();
    prepared.request.accepted_state_revision = flow.state_revision();
    prepared.inspection.components = std::move(catalog.summaries);
    const auto& state = flow.low_mach_state();
    const auto& hierarchy = *flow.hierarchy();
    const auto& config = hierarchy.config();
    const auto rank = Tpetra::getDefaultComm()->getRank();
    const double finest = finest_width(hierarchy);
    const bool provenance_current =
        binding_revision_ == flow.state_revision() && state_->topology == flow.topology_generation();
    if (!provenance_current)
        next->origins.clear();
    for (auto& component : prepared.inspection.components)
        if (const auto found = next->origins.find(component.label); found != next->origins.end())
            component.originating_birth_ids = found->second;
    std::vector<int> populations;
    for (const auto& name : populations_)
    {
        const auto found =
            std::find_if(state.dispersed.begin(), state.dispersed.end(), [&](const auto& p) { return p.name == name; });
        require_all(
            found != state.dispersed.end(), "Representation population registry no longer matches the flow state.");
        populations.push_back(static_cast<int>(found - state.dispersed.begin()));
    }
    std::vector<double> maximum_radius(state.dispersed.size());
    for (std::size_t p = 0; p < maximum_radius.size(); ++p)
    {
        double local = 0;
        for (double r : state.dispersed[p].radius)
            local = std::max(local, r);
        maximum_radius[p] = reduce(local, Teuchos::REDUCE_MAX);
    }
    std::set<ID> requested;
    std::vector<Box> birth_stencils(next->births.size()), component_stencils(prepared.inspection.components.size());
    const auto collect = [&](const Box& box, std::set<ID>& into)
    {
        for (const auto& item : hierarchy.overlaps(box))
            into.insert(item.cell);
    };
    collective(
        [&]
        {
            for (std::size_t b = 0; b < next->births.size(); ++b)
            {
                auto& record = next->births[b];
                record.source_revision = flow.state_revision();
                record.source_topology = flow.topology_generation();
                record.reserved_number = 0;
                if (record.status == BubbleBirthStatus::Promoted)
                {
                    record.reason = Reason::AlreadyPromoted;
                    continue;
                }
                record.status = BubbleBirthStatus::Pending;
                if (flow.pending_representation_relaxation())
                {
                    record.reason = Reason::PendingRelaxation;
                    continue;
                }
                const auto p = static_cast<std::size_t>(
                    std::find(populations_.begin(), populations_.end(), record.seed.population) - populations_.begin());
                Box donor = record.seed.donor_region;
                record.reason = contain_box(donor, config, false);
                if (record.reason != Reason::Ready)
                    continue;
                const double reach =
                    maximum_radius[static_cast<std::size_t>(populations[p])] + options_.interface_guard_cells * finest;
                auto& stencil = birth_stencils[b];
                for (std::size_t a = 0; a < 3; ++a)
                {
                    const double lo = config.origin[a], hi = lo + config.extent[a];
                    const double pad = 32 * std::numeric_limits<double>::epsilon() *
                                       std::max({std::abs(lo), std::abs(hi), config.extent[a]});
                    stencil.lower[a] = std::max(lo, record.seed.center[a] - reach - pad);
                    stencil.upper[a] = std::min(hi, record.seed.center[a] + reach + pad);
                }
                // This is only an import bound. Actual eligibility uses the local
                // whole-bubble volume, not a remote maximum particle radius.
                auto candidate = requested;
                collect(donor, candidate);
                collect(stencil, candidate);
                if (candidate.size() > options_.maximum_event_cells)
                {
                    record.reason = Reason::StencilTooLarge;
                    continue;
                }
                requested.swap(candidate);
            }
            for (std::size_t k = 0; k < prepared.inspection.components.size(); ++k)
            {
                auto& c = prepared.inspection.components[k];
                double extent = 0;
                for (std::size_t a = 0; a < 3; ++a)
                    extent = std::max(extent, c.bounds.upper[a] - c.bounds.lower[a]);
                if (flow.pending_representation_relaxation())
                    c.reason = Reason::PendingRelaxation;
                else if (c.touches_wall)
                    c.reason = Reason::AttachedToBoundary;
                else if (c.touches_headspace)
                    c.reason = Reason::HeadspaceConnected;
                else if (c.inert_moles > 0)
                    c.reason = Reason::ContainsInert;
                else if (c.crosses_periodic_seam)
                    c.reason = Reason::PeriodicStencilUnsupported;
                else if (c.diameter > options_.demotion_diameter)
                    c.reason = Reason::AboveDemotionDiameter;
                else if (c.diameter / c.maximum_cell_width >
                         options_.demotion_cells_per_diameter * (1 + 32 * std::numeric_limits<double>::epsilon()))
                    c.reason = Reason::WellResolved;
                else if (extent > options_.maximum_demoted_extent_ratio * c.diameter)
                    c.reason = Reason::ElongatedComponent;
                else if (pivots_.mol_per_bubble.empty() || c.hydrogen_moles < pivots_.mol_per_bubble.front() ||
                         c.hydrogen_moles > pivots_.mol_per_bubble.back())
                    c.reason = Reason::OutsideMolarGrid;
                else
                {
                    auto& stencil = component_stencils[k];
                    stencil = expanded(c.bounds, options_.demotion_neighborhood_factor);
                    c.reason = contain_box(stencil, config, true);
                    if (c.reason != Reason::Ready)
                        continue;
                    auto candidate = requested;
                    collect(stencil, candidate);
                    if (candidate.size() > options_.maximum_event_cells)
                        c.reason = Reason::StencilTooLarge;
                    else
                        requested.swap(candidate);
                }
            }
        },
        "Representation stencil construction failed on another rank.");
    Samples samples(flow, requested);
    const auto n = flow.mesh()->num_owned_cells();
    std::vector<double> own_volume(n), own_gas(n), own_liquid(n), own_carrier(n), own_dispersed(n);
    for (std::size_t c = 0; c < n; ++c)
    {
        own_volume[c] = flow.mesh()->cell_volume(static_cast<LO>(c));
        own_gas[c] = state.resolved_fraction[c] * own_volume[c];
        own_liquid[c] = state.liquid_fraction[c] * own_volume[c];
        LedgerSum carrier_sum, dispersed_sum;
        carrier_sum += own_liquid[c];
        for (const auto& population : state.dispersed)
        {
            carrier_sum += population.volume[c];
            dispersed_sum += population.volume[c];
        }
        own_carrier[c] = static_cast<double>(carrier_sum.value());
        own_dispersed[c] = static_cast<double>(dispersed_sum.value());
    }
    const auto cell_volume = samples.field(flow, own_volume), gas = samples.field(flow, own_gas),
               liquid = samples.field(flow, own_liquid), carrier = samples.field(flow, own_carrier),
               dispersed = samples.field(flow, own_dispersed);
    const auto hydrogen = samples.field(flow, state.resolved_hydrogen_moles);
    std::vector<std::vector<double>> number, moles, bubble_volume, radius;
    for (const auto& p : state.dispersed)
    {
        number.push_back(samples.field(flow, p.number));
        moles.push_back(samples.field(flow, p.hydrogen_moles));
        bubble_volume.push_back(samples.field(flow, p.volume));
        radius.push_back(samples.field(flow, p.radius));
    }
    std::vector<double> low(n), high(n);
    for (std::size_t c = 0; c < n; ++c)
    {
        low[c] = static_cast<std::uint32_t>(catalog.owned_labels[c]);
        high[c] = catalog.owned_labels[c] >> 32;
    }
    low = samples.field(flow, low);
    high = samples.field(flow, high);
    std::vector<ID> labels(low.size());
    for (std::size_t c = 0; c < labels.size(); ++c)
        labels[c] = static_cast<ID>(low[c]) | (static_cast<ID>(high[c]) << 32);

    std::vector<std::vector<std::size_t>> event_rows;
    const auto below = [](double a, double b)
    { return a < b - 32 * std::numeric_limits<double>::epsilon() * std::max(std::abs(a), std::abs(b)); };
    collective(
        [&]
        {
            if (rank != 0)
                return;
            auto available = number;
            std::set<ID> occupied;
            std::vector<Box> accepted_stencils;
            std::set<ID> selected_support;
            struct Part
            {
                ID cell;
                double volume;
            };
            // Match the core's compensated sparse debit sum and one final
            // cast. Close an allocation by changing its split, never its donor
            // inventory or the strict core capacity gate.
            const auto close_quota = [](const std::vector<double*>& pieces, double quota)
            {
                if (pieces.empty())
                {
                    if (quota != 0)
                        throw std::runtime_error("Representation volume quota has no parcel support.");
                    return;
                }
                const auto largest = *std::max_element(
                    pieces.begin(), pieces.end(), [](const auto* a, const auto* b) { return *a < *b; });
                const double largest_original = *largest;
                LedgerSum measured;
                for (const auto* value : pieces)
                    measured += *value;
                const double original = static_cast<double>(measured.value());
                if (original == quota)
                    return;
                const double allowance =
                    512 * std::numeric_limits<double>::epsilon() * std::max(quota, std::numeric_limits<double>::min());
                if (std::abs(original - quota) > allowance)
                    throw std::runtime_error("Representation source split does not close its physical volume quota.");
                double* second = nullptr;
                for (auto* value : pieces)
                    if (value != largest && (!second || *value > *second))
                        second = value;
                const double second_original = second ? *second : 0;
                const double second_step =
                    second ? std::max(std::nextafter(second_original, std::numeric_limits<double>::infinity()) -
                                          second_original,
                                 static_cast<double>(4 * std::numeric_limits<long double>::epsilon() * quota))
                           : 0;
                for (int perturb : {0, -1, 1})
                {
                    // At an exact midpoint, neither adjacent largest piece
                    // need round to an odd-ulp quota. Perturb the second piece
                    // at a scale visible when the compensated accumulator is
                    // rounded to long double, then close the same partition.
                    if (second && perturb)
                    {
                        const double candidate = second_original + perturb * second_step;
                        if (!(candidate >= 0) || !std::isfinite(candidate))
                            continue;
                        *second = candidate;
                    }
                    LedgerSum others;
                    for (const auto* value : pieces)
                        if (value != largest)
                            others += *value;
                    *largest = static_cast<double>(static_cast<long double>(quota) - others.value());
                    for (int attempt = 0; attempt < 4; ++attempt)
                    {
                        LedgerSum sum;
                        for (const auto* value : pieces)
                            sum += *value;
                        const double actual = static_cast<double>(sum.value());
                        const long double partition_change =
                            std::abs(static_cast<long double>(*largest) - largest_original) +
                            (second ? std::abs(static_cast<long double>(*second) - second_original) : 0);
                        if (*largest >= 0 && std::isfinite(*largest) && actual == quota &&
                            partition_change <= allowance)
                            return;
                        *largest =
                            std::nextafter(*largest, actual < quota ? std::numeric_limits<double>::infinity() : 0.0);
                    }
                }
                LedgerSum final_sum;
                for (const auto* value : pieces)
                    final_sum += *value;
                std::ostringstream message;
                message << std::setprecision(21)
                        << "Representation source split cannot reach its exact rounded quota: quota=" << quota
                        << ", original=" << original << ", measured=" << final_sum.value() << ", largest=" << *largest
                        << ", original_largest=" << largest_original << ", second=" << (second ? *second : 0)
                        << ", original_second=" << second_original << ", second_step=" << second_step
                        << ", L1_allowance=" << allowance << '.';
                throw std::runtime_error(message.str());
            };
            const auto match =
                [&](const std::string& context, std::vector<Part> from, std::vector<Part> to, auto&& emit)
            {
                struct Transfer
                {
                    std::size_t source;
                    ID target;
                    double volume;
                };
                const auto quotas = from;
                std::vector<Transfer> transfers;
                std::size_t a = 0, b = 0;
                LedgerSum original, target;
                for (const auto& p : from)
                    original += p.volume;
                for (const auto& p : to)
                    target += p.volume;
                while (a < from.size() && b < to.size())
                {
                    const double amount = std::min(from[a].volume, to[b].volume);
                    if (amount > 0)
                        transfers.push_back({a, to[b].cell, amount});
                    from[a].volume -= amount;
                    to[b].volume -= amount;
                    if (from[a].volume == 0)
                        ++a;
                    if (to[b].volume == 0)
                        ++b;
                }
                LedgerSum remainder, source_remainder, target_remainder;
                for (std::size_t k = a; k < from.size(); ++k)
                {
                    remainder += from[k].volume;
                    source_remainder += from[k].volume;
                }
                for (std::size_t k = b; k < to.size(); ++k)
                {
                    remainder += to[k].volume;
                    target_remainder += to[k].volume;
                }
                const long double allowance =
                    512 * std::numeric_limits<double>::epsilon() *
                    std::max(original.value(), static_cast<long double>(std::numeric_limits<double>::min()));
                if (remainder.value() > allowance)
                {
                    std::ostringstream message;
                    message << std::setprecision(21)
                            << "Representation parcel source and target volumes do not balance: context=" << context
                            << ", source_sum=" << original.value() << ", target_sum=" << target.value()
                            << ", remainder=" << remainder.value() << ", allowance=" << allowance
                            << ", from_count=" << from.size() << ", to_count=" << to.size()
                            << ", source_remainder=" << source_remainder.value()
                            << ", target_remainder=" << target_remainder.value() << ", source_index=" << a
                            << ", target_index=" << b << '.';
                    throw std::runtime_error(message.str());
                }
                // Recipient quotas are geometric floating-point partitions.
                // Spend every declared source quota, including a bounded final
                // partition remainder, instead of losing it in subtraction.
                if (!to.empty())
                    for (std::size_t k = a; k < from.size(); ++k)
                        if (from[k].volume > 0)
                            transfers.push_back({k, to.back().cell, from[k].volume});
                std::vector<std::vector<double*>> pieces(from.size());
                for (auto& transfer : transfers)
                    pieces[transfer.source].push_back(&transfer.volume);
                for (std::size_t k = 0; k < quotas.size(); ++k)
                    close_quota(pieces[k], quotas[k].volume);
                for (const auto& transfer : transfers)
                    if (transfer.volume > 0)
                        emit(quotas[transfer.source].cell, transfer.target, transfer.volume);
            };
            for (std::size_t b = 0; b < next->births.size(); ++b)
            {
                auto& record = next->births[b];
                if (record.reason != Reason::Ready)
                    continue;
                const auto slot = static_cast<std::size_t>(
                    std::find(populations_.begin(), populations_.end(), record.seed.population) - populations_.begin());
                const auto p = static_cast<std::size_t>(populations[slot]);
                std::vector<std::pair<std::size_t, double>> taken;
                Sum count, total_volume, total_moles, old_area;
                for (const auto& donor : hierarchy.overlaps(record.seed.donor_region))
                {
                    const auto i = samples.index.at(donor.cell);
                    const double remaining = 1 - count.value();
                    if (remaining <= 0)
                        break;
                    const double amount = std::min(available[p][i], remaining);
                    if (amount <= 0)
                        continue;
                    const double pivot = pivots_.mol_per_bubble[slot];
                    if (std::abs(moles[p][i] - pivot * number[p][i]) >
                        256 * std::numeric_limits<double>::epsilon() * std::max(moles[p][i], pivot * number[p][i]))
                        throw std::runtime_error("A whole-bubble donor does not satisfy its fixed molar pivot.");
                    taken.emplace_back(i, amount);
                    count += amount;
                    total_volume += bubble_volume[p][i] * (amount / number[p][i]);
                    total_moles += moles[p][i] * (amount / number[p][i]);
                    old_area += 4 * std::numbers::pi * radius[p][i] * radius[p][i] * amount;
                }
                if (count.value() < 1 - 256 * std::numeric_limits<double>::epsilon())
                {
                    record.reason = Reason::InsufficientDonors;
                    continue;
                }
                record.reserved_number = count.value();
                record.hydrogen_moles = total_moles.value();
                record.gas_volume = total_volume.value();
                record.diameter = diameter(record.gas_volume);
                if (below(record.diameter, options_.promotion_diameter))
                {
                    record.reason = Reason::BelowPromotionDiameter;
                    continue;
                }
                if (below(record.diameter / finest, options_.promotion_cells_per_diameter))
                {
                    record.reason = Reason::FinestResolutionLimited;
                    continue;
                }
                Box stencil;
                const double reach = 0.5 * record.diameter + options_.interface_guard_cells * finest;
                for (std::size_t a = 0; a < 3; ++a)
                {
                    stencil.lower[a] = record.seed.center[a] - reach;
                    stencil.upper[a] = record.seed.center[a] + reach;
                }
                record.reason = contain_box(stencil, config, false);
                if (record.reason != Reason::Ready)
                    continue;
                bool refined = true, clear = true;
                for (const auto& cell : hierarchy.overlaps(stencil))
                {
                    refined = refined && hierarchy.blocks()[hierarchy.cell_block(cell.cell)].level == config.max_level;
                    clear = clear && gas[samples.index.at(cell.cell)] == 0 && !occupied.contains(cell.cell);
                }
                for (const auto& other : accepted_stencils)
                    clear = clear && !intersects(stencil, other);
                if (!clear)
                {
                    record.reason = Reason::OverlappingResolvedGas;
                    continue;
                }
                if (!refined)
                {
                    record.reason = Reason::NeedsRefinement;
                    record.status = BubbleBirthStatus::Reserved;
                    for (const auto& [i, amount] : taken)
                        available[p][i] -= amount;
                    continue;
                }
                std::set<ID> support;
                collect(record.seed.donor_region, support);
                collect(stencil, support);
                if (std::any_of(support.begin(), support.end(), [&](ID id) { return selected_support.contains(id); }))
                {
                    record.reason = Reason::OverlappingReservation;
                    continue;
                }
                std::map<ID, double> insertion_volume, insertion_capacity;
                for (const auto& cell : hierarchy.overlaps(stencil))
                {
                    const auto i = samples.index.at(cell.cell);
                    insertion_volume.emplace(cell.cell, cell_volume[i]);
                    insertion_capacity.emplace(cell.cell, std::min(cell_volume[i], carrier[i]));
                }
                const auto inserted = sphere(
                    hierarchy, stencil, record.seed.center, record.gas_volume, insertion_volume, insertion_capacity);
                std::vector<Part> from, to;
                for (const auto& [i, amount] : taken)
                    from.push_back({samples.ids[i], bubble_volume[p][i] * (amount / number[p][i])});
                for (const auto& [id, v] : inserted)
                    if (v > 0)
                        to.push_back({id, v});
                std::vector<std::size_t> rows;
                match("promotion seed=" + std::to_string(record.seed.id) + " population=" + record.seed.population,
                    from, to,
                    [&](ID a, ID z, double v)
                    {
                        rows.push_back(prepared.request.gas_parcels.size());
                        prepared.request.gas_parcels.push_back({a, z, static_cast<int>(p), -1, v, 0});
                    });
                for (const auto& [i, amount] : taken)
                    available[p][i] -= amount;
                for (const auto& [id, v] : inserted)
                    if (v > 0)
                        occupied.insert(id);
                accepted_stencils.push_back(stencil);
                selected_support.insert(support.begin(), support.end());
                record.status = BubbleBirthStatus::Promoted;
                BubbleConversionEvent event;
                event.kind = BubbleConversionKind::Promotion;
                event.birth_seed_id = record.seed.id;
                event.component_label = inserted.begin()->first + 1;
                event.center = record.seed.center;
                event.gas_volume = record.gas_volume;
                event.hydrogen_moles = record.hydrogen_moles;
                event.dispersed_number_change = -1;
                event.resolved_number_change = 1;
                // The actual new PLIC area is added collectively below.
                event.surface_area_change = -old_area.value();
                event.surface_energy_change = flow.material().surface_tension * event.surface_area_change;
                event.originating_birth_ids = {record.seed.id};
                prepared.inspection.events.push_back(std::move(event));
                event_rows.push_back(std::move(rows));
            }
            for (std::size_t k = 0; k < prepared.inspection.components.size(); ++k)
            {
                auto& component = prepared.inspection.components[k];
                if (component.reason != Reason::Ready)
                    continue;
                std::vector<Part> source;
                std::vector<std::pair<ID, double>> hosts;
                Sum capacity;
                double maximum_void = 0;
                bool clear = true;
                for (const auto& cell : hierarchy.overlaps(component_stencils[k]))
                {
                    const auto i = samples.index.at(cell.cell);
                    const double removed = labels[i] == component.label ? gas[i] : 0;
                    if (removed > 0)
                        source.push_back({cell.cell, removed});
                    const double host = carrier[i] + removed;
                    if (host > 0)
                    {
                        hosts.emplace_back(cell.cell, host);
                        capacity += host;
                    }
                    if (carrier[i] > 0)
                        maximum_void = std::max(maximum_void, dispersed[i] / carrier[i]);
                    clear = clear && !selected_support.contains(cell.cell);
                }
                if (!clear)
                {
                    component.reason = Reason::OverlappingReservation;
                    continue;
                }
                const double beta = component.gas_volume / capacity.value();
                if (!(capacity.value() > 0) || beta + (1 - beta) * maximum_void > options_.maximum_dispersed_fraction)
                {
                    component.reason = Reason::InsufficientCarrier;
                    continue;
                }
                const auto hi_it = std::lower_bound(
                    pivots_.mol_per_bubble.begin(), pivots_.mol_per_bubble.end(), component.hydrogen_moles);
                const auto hi = static_cast<std::size_t>(hi_it - pivots_.mol_per_bubble.begin());
                std::vector<std::pair<std::size_t, double>> bins;
                if (*hi_it == component.hydrogen_moles)
                    bins.emplace_back(hi, 1.0);
                else
                {
                    const auto lo = hi - 1;
                    const double gap = pivots_.mol_per_bubble[hi] - pivots_.mol_per_bubble[lo];
                    const double lower = (pivots_.mol_per_bubble[hi] - component.hydrogen_moles) / gap;
                    const double upper = (component.hydrogen_moles - pivots_.mol_per_bubble[lo]) / gap;
                    bins = {{lo, lower}, {hi, upper}};
                    // Resolve the smaller physical share directly. Computing
                    // a tiny secondary bin as V minus an almost full first
                    // bin loses its relative accuracy independently in the
                    // source and host partitions, especially near a pivot.
                    std::stable_sort(bins.begin(), bins.end(),
                        [&](const auto& a, const auto& b)
                        {
                            return pivots_.mol_per_bubble[a.first] * a.second <
                                   pivots_.mol_per_bubble[b.first] * b.second;
                        });
                }
                double new_area = 0;
                const auto row_begin = prepared.request.gas_parcels.size();
                std::map<ID, double> source_assigned, target_assigned;
                for (std::size_t split = 0; split < bins.size(); ++split)
                {
                    const auto [bin, count] = bins[split];
                    const double mass_fraction = pivots_.mol_per_bubble[bin] * count / component.hydrogen_moles;
                    std::vector<Part> from, to;
                    for (const auto& cell : source)
                    {
                        const double amount = split + 1 == bins.size() ? cell.volume - source_assigned[cell.cell]
                                                                       : cell.volume * mass_fraction;
                        from.push_back({cell.cell, amount});
                        source_assigned[cell.cell] += amount;
                    }
                    for (const auto& [id, host] : hosts)
                    {
                        const double amount =
                            split + 1 == bins.size() ? beta * host - target_assigned[id] : beta * host * mass_fraction;
                        to.push_back({id, amount});
                        target_assigned[id] += amount;
                    }
                    match("demotion component=" + std::to_string(component.label) + " bin=" + std::to_string(bin), from,
                        to,
                        [&](ID a, ID z, double v)
                        {
                            const auto i = samples.index.at(a);
                            prepared.request.gas_parcels.push_back({a, z, -1, populations[bin], v,
                                v * hydrogen[i] / gas[i] / pivots_.mol_per_bubble[bin]});
                        });
                    new_area += area(count, component.gas_volume * mass_fraction);
                }
                // The complete component is withdrawn once. Closing the
                // combined two-pivot split is necessary even if each bin's
                // separately rounded subquota is already closed.
                std::map<ID, std::vector<double*>> source_pieces;
                for (std::size_t row = row_begin; row < prepared.request.gas_parcels.size(); ++row)
                {
                    auto& parcel = prepared.request.gas_parcels[row];
                    source_pieces[parcel.source_cell].push_back(&parcel.volume);
                }
                for (const auto& cell : source)
                    close_quota(source_pieces[cell.cell], cell.volume);
                for (std::size_t row = row_begin; row < prepared.request.gas_parcels.size(); ++row)
                {
                    auto& parcel = prepared.request.gas_parcels[row];
                    const auto i = samples.index.at(parcel.source_cell);
                    const auto bin = static_cast<std::size_t>(
                        std::find(populations.begin(), populations.end(), parcel.target_population) -
                        populations.begin());
                    parcel.target_number = parcel.volume * hydrogen[i] / gas[i] / pivots_.mol_per_bubble[bin];
                }
                for (const auto& [id, host] : hosts)
                    occupied.insert(id);
                for (const auto& cell : hierarchy.overlaps(component_stencils[k]))
                    selected_support.insert(cell.cell);
                BubbleConversionEvent event;
                event.kind = BubbleConversionKind::Demotion;
                event.component_label = component.label;
                event.center = component.centroid;
                event.gas_volume = component.gas_volume;
                event.hydrogen_moles = component.hydrogen_moles;
                event.dispersed_number_change = 1;
                event.resolved_number_change = -1;
                event.surface_area_change = new_area - component.surface_area;
                event.surface_energy_change = flow.material().surface_tension * event.surface_area_change;
                event.originating_birth_ids = component.originating_birth_ids;
                prepared.inspection.events.push_back(std::move(event));
                std::vector<std::size_t> rows(prepared.request.gas_parcels.size() - row_begin);
                std::iota(rows.begin(), rows.end(), row_begin);
                event_rows.push_back(std::move(rows));
            }
            for (std::size_t event_index = 0; event_index < event_rows.size(); ++event_index)
            {
                const auto& rows = event_rows[event_index];
                const auto& event = prepared.inspection.events[event_index];
                std::map<ID, LedgerSum> gap;
                std::map<std::pair<ID, std::size_t>, LedgerSum> removed_from_carrier;
                for (const auto row : rows)
                {
                    const auto& parcel = prepared.request.gas_parcels[row];
                    gap[parcel.source_cell] += parcel.volume;
                    gap[parcel.target_cell] += -parcel.volume;
                    if (parcel.source_population >= 0)
                        removed_from_carrier[{
                            parcel.source_cell, static_cast<std::size_t>(parcel.source_population)}] += parcel.volume;
                }
                std::vector<Part> from, to;
                for (const auto& [id, amount] : gap)
                {
                    const double value = static_cast<double>(amount.value());
                    if (value < 0)
                    {
                        const auto i = samples.index.at(id);
                        LedgerSum capacity;
                        capacity += liquid[i];
                        for (std::size_t p = 0; p < bubble_volume.size(); ++p)
                        {
                            const double debit = static_cast<double>(removed_from_carrier[{id, p}].value());
                            if (debit > bubble_volume[p][i])
                                throw std::runtime_error(
                                    "Representation gas split exceeds its actual population volume.");
                            capacity += bubble_volume[p][i] - debit;
                        }
                        const double remaining = static_cast<double>(capacity.value());
                        if (-value > remaining + 256 * std::numeric_limits<double>::epsilon() * cell_volume[i])
                            throw std::runtime_error(
                                "Representation displacement exhausts the remaining carrier packet.");
                        // Geometric and constituent sums may differ by a few
                        // ulps at complete displacement. The true packet is
                        // authoritative; never borrow the occupancy remainder.
                        // match() conserves this exact source quota and leaves
                        // only the checked recipient-partition roundoff.
                        from.push_back({id, std::min(-value, remaining)});
                    }
                    else if (value > 0)
                        to.push_back({id, value});
                }
                match("carrier event=" + std::to_string(event_index) +
                          (event.birth_seed_id ? " seed=" + std::to_string(*event.birth_seed_id)
                                               : " component=" + std::to_string(event.component_label)),
                    from, to, [&](ID a, ID z, double v) { prepared.request.carrier_parcels.push_back({a, z, v}); });
            }
        },
        "Representation donor aggregation or parcel planning failed on another rank.");

    // Only compact decisions/events are replicated; source field samples and
    // cellwise parcels stay on their unique coordinator.
    const auto comm = Tpetra::getDefaultComm();
    for (auto& record : next->births)
    {
        std::array<ID, 2> codes{static_cast<ID>(record.status), static_cast<ID>(record.reason)};
        std::array<double, 4> values{record.reserved_number, record.hydrogen_moles, record.gas_volume, record.diameter};
        Teuchos::broadcast(*comm, 0, 2, codes.data());
        Teuchos::broadcast(*comm, 0, 4, values.data());
        record.status = static_cast<BubbleBirthStatus>(codes[0]);
        record.reason = static_cast<Reason>(codes[1]);
        record.reserved_number = values[0];
        record.hydrogen_moles = values[1];
        record.gas_volume = values[2];
        record.diameter = values[3];
    }
    for (auto& component : prepared.inspection.components)
    {
        ID code = static_cast<ID>(component.reason);
        Teuchos::broadcast(*comm, 0, 1, &code);
        component.reason = static_cast<Reason>(code);
    }
    ID count = prepared.inspection.events.size();
    Teuchos::broadcast(*comm, 0, 1, &count);
    require_all(count < std::numeric_limits<ID>::max() - next->next_event, "Representation event IDs are exhausted.");
    prepared.inspection.events.resize(static_cast<std::size_t>(count));
    std::size_t event_index = 0;
    for (auto& event : prepared.inspection.events)
    {
        std::array<ID, 4> codes{static_cast<ID>(event.kind), event.birth_seed_id.has_value() ? 1U : 0U,
            event.birth_seed_id.value_or(0), event.component_label};
        std::array<double, 9> values{event.center[0], event.center[1], event.center[2], event.gas_volume,
            event.hydrogen_moles, event.dispersed_number_change, event.resolved_number_change,
            event.surface_area_change, event.surface_energy_change};
        Teuchos::broadcast(*comm, 0, 4, codes.data());
        Teuchos::broadcast(*comm, 0, 9, values.data());
        event.kind = static_cast<BubbleConversionKind>(codes[0]);
        event.birth_seed_id = codes[1] ? std::optional<ID>(codes[2]) : std::nullopt;
        event.component_label = codes[3];
        event.center = {values[0], values[1], values[2]};
        event.gas_volume = values[3];
        event.hydrogen_moles = values[4];
        event.dispersed_number_change = values[5];
        event.resolved_number_change = values[6];
        event.surface_area_change = values[7];
        event.surface_energy_change = values[8];
        ID origins = event.originating_birth_ids.size();
        Teuchos::broadcast(*comm, 0, 1, &origins);
        event.originating_birth_ids.resize(static_cast<std::size_t>(origins));
        if (origins)
            Teuchos::broadcast(*comm, 0, static_cast<int>(origins), event.originating_birth_ids.data());
        if (event.kind == BubbleConversionKind::Promotion)
        {
            std::vector<double> partial(samples.ids.size(), 0);
            if (rank == 0)
                for (const auto row : event_rows[event_index])
                {
                    const auto& parcel = prepared.request.gas_parcels[row];
                    partial[samples.index.at(parcel.target_cell)] += parcel.volume;
                }
            auto fraction = samples.to_owned(flow, partial);
            for (std::size_t c = 0; c < fraction.size(); ++c)
                fraction[c] /= own_volume[c];
            Sum local_area;
            for (const auto& patch : geometry.interface_patches(fraction))
                local_area += patch.area;
            event.surface_area_change += reduce(local_area.value(), Teuchos::REDUCE_SUM);
            event.surface_energy_change = flow.material().surface_tension * event.surface_area_change;
        }
        event.event_id = next->next_event++;
        if (event.kind == BubbleConversionKind::Promotion)
        {
            const auto birth = std::find_if(next->births.begin(), next->births.end(),
                [&](const auto& b) { return b.seed.id == *event.birth_seed_id; });
            Box box;
            const double reach = 0.5 * birth->diameter + options_.interface_guard_cells * finest;
            for (std::size_t a = 0; a < 3; ++a)
            {
                box.lower[a] = birth->seed.center[a] - reach;
                box.upper[a] = birth->seed.center[a] + reach;
            }
            (void) contain_box(box, config, false);
            prepared.supports.push_back({birth->seed.donor_region, box});
        }
        else
        {
            const auto comp = std::find_if(prepared.inspection.components.begin(), prepared.inspection.components.end(),
                [&](const auto& c) { return c.label == event.component_label; });
            const auto index = static_cast<std::size_t>(comp - prepared.inspection.components.begin());
            prepared.supports.push_back({component_stencils[index]});
        }
        if (event.kind == BubbleConversionKind::Promotion)
            next->origins[event.component_label] = event.originating_birth_ids;
        else
            next->origins.erase(event.component_label);
        next->history.push_back(event);
        ++event_index;
    }
    prepared.inspection.births = next->births;
    prepared.inspection.refinement.source_hierarchy = flow.hierarchy();
    prepared.inspection.refinement.source_observation = flow.hierarchy()->observation();
    VoF::RefinementHints hints;
    for (const auto& birth : next->births)
    {
        if (birth.reason == Reason::FinestResolutionLimited)
            prepared.inspection.refinement.resolution_limited = true;
        if (birth.reason == Reason::NeedsRefinement)
        {
            Box box;
            const double reach = 0.5 * birth.diameter + options_.interface_guard_cells * finest;
            for (std::size_t a = 0; a < 3; ++a)
            {
                box.lower[a] = birth.seed.center[a] - reach;
                box.upper[a] = birth.seed.center[a] + reach;
            }
            hints.promotion_regions.push_back(box);
        }
    }
    if (!hints.promotion_regions.empty())
    {
        VoF::RefinementOptions refine;
        refine.minimum_resolved_diameter = options_.promotion_diameter;
        refine.cells_per_diameter = options_.promotion_cells_per_diameter;
        prepared.inspection.refinement =
            geometry.refinement_plan(flow.gas_fraction(), flow.face_velocity(), refine, std::move(hints));
        prepared.inspection.refinement.coarsen.clear();
    }
    next->source_revision = flow.state_revision();
    next->topology = flow.topology_generation();
    prepared.next = std::move(next);
    return prepared;
}
} // namespace SimpleFluid::detail
