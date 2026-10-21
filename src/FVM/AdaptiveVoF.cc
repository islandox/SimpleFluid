/** @file AdaptiveVoF.cc
 * @brief Sparse geometric reconstruction, subface transport and conservative AMR remapping.
 */
#include "FVM/AdaptiveVoF.hh"
#include "FVM/details/VoFPhaseBudget.hh"
#include "FVM/details/VoFReconstructionCoverage.hh"

#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::VoF
{
namespace
{
using Pack = DefaultTpetraTypes;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;
using Box = SimpleFluid::Meshes::AMR::AMRBox;
constexpr double roundoff = 2.0e-12;
// Youngs gradients need one layer. Height functions need three axial layers;
// the radius-two surface fit needs one further layer for its PLIC gradients.
constexpr std::size_t padding = 3;
constexpr std::size_t protected_stencil = 4;

void require_collective(bool valid, const char* message)
{
    int local = valid ? 1 : 0, global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::invalid_argument(message);
}

double maximum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &result);
    return result;
}

void identical(double value, const char* message)
{
    require_collective(std::isfinite(value), message);
    double lo = 0.0, hi = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &value, &lo);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &value, &hi);
    require_collective(lo == hi, message);
}

double volume(const Box& b)
{
    return (b.upper[0] - b.lower[0]) * (b.upper[1] - b.lower[1]) * (b.upper[2] - b.lower[2]);
}

Vector center(const Box& b)
{
    return {0.5 * (b.lower[0] + b.upper[0]), 0.5 * (b.lower[1] + b.upper[1]), 0.5 * (b.lower[2] + b.upper[2])};
}

bool intersects(const Box& a, const Box& b)
{
    for (std::size_t d = 0; d < 3; ++d)
        if (a.upper[d] < b.lower[d] || b.upper[d] < a.lower[d])
            return false;
    return true;
}

void validate_f(const std::vector<double>& f, std::size_t n)
{
    require_collective(f.size() == n && std::all_of(f.begin(), f.end(), [](double v)
                                            { return std::isfinite(v) && v >= -roundoff && v <= 1.0 + roundoff; }),
        "AdaptiveVoF requires owned-cell finite fractions in [0,1] within retained roundoff.");
}

/** Only requested donor cells are imported; source maps remain one-to-one. */
struct Exchange
{
    struct Plan
    {
        Teuchos::RCP<const Pack::map_type> source, target;
        Pack::import_type importer;
        Plan(Teuchos::RCP<const Pack::map_type> from, Teuchos::RCP<const Pack::map_type> to)
            : source(std::move(from)), target(std::move(to)), importer(source, target)
        {
        }
    };
    std::shared_ptr<const Plan> plan;
    Teuchos::RCP<const Pack::map_type> source, target;
    // Workspaces belong to this AdaptiveVoF instance. Only the immutable Plan is shared.
    using Vectors = std::pair<std::unique_ptr<Pack::multi_vector_type>, std::unique_ptr<Pack::multi_vector_type>>;
    mutable std::map<std::size_t, Vectors> workspaces;
    mutable Pack::multi_vector_type *from = nullptr, *to = nullptr;
    mutable std::size_t columns = 0;

    explicit Exchange(std::shared_ptr<const Plan> p) : plan(std::move(p)), source(plan->source), target(plan->target) {}
    Exchange(Teuchos::RCP<const Pack::map_type> source, Teuchos::RCP<const Pack::map_type> target)
        : Exchange(std::make_shared<Plan>(std::move(source), std::move(target)))
    {
    }
    Exchange(Teuchos::RCP<const Pack::map_type> source, const std::set<GO>& ids)
        : Exchange(source,
              Teuchos::rcp(new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                  Teuchos::arrayViewFromVector(std::vector<GO>(ids.begin(), ids.end())), GO{}, source->getComm())))
    {
    }

    void workspace(std::size_t count) const
    {
        if (columns == count)
            return;
        auto& pair = workspaces[count];
        if (!pair.first)
        {
            pair.first = std::make_unique<Pack::multi_vector_type>(source, count, false);
            pair.second = std::make_unique<Pack::multi_vector_type>(target, count, false);
        }
        from = pair.first.get();
        to = pair.second.get();
        columns = count;
    }

    std::vector<std::vector<double>> batch(const std::vector<std::vector<double>>& owned) const
    {
        workspace(owned.size());
        for (std::size_t a = 0; a < columns; ++a)
        {
            auto values = from->getDataNonConst(a);
            std::copy(owned[a].begin(), owned[a].end(), values.begin());
        }
        to->doImport(*from, plan->importer, Tpetra::INSERT);
        std::vector<std::vector<double>> result(columns);
        for (std::size_t a = 0; a < columns; ++a)
        {
            const auto values = to->getData(a);
            result[a].assign(values.begin(), values.end());
        }
        return result;
    }

    std::vector<double> operator()(const std::vector<double>& owned) const
    {
        workspace(1);
        {
            auto values = from->getDataNonConst(0);
            std::copy(owned.begin(), owned.end(), values.begin());
        }
        to->doImport(*from, plan->importer, Tpetra::INSERT);
        const auto values = to->getData(0);
        return {values.begin(), values.end()};
    }

    std::vector<double> sum_to_source(const std::vector<double>& partial) const
    {
        workspace(1);
        {
            auto values = to->getDataNonConst(0);
            std::copy(partial.begin(), partial.end(), values.begin());
        }
        from->putScalar(0.0);
        from->doExport(*to, plan->importer, Tpetra::ADD);
        const auto values = from->getData(0);
        return {values.begin(), values.end()};
    }
};

double gas_intersection(const Box& donor, const Box& intersection, double f, const InterfacePlane& plane)
{
    const double v = volume(intersection), full = volume(donor);
    // Keep raw pure-cell roundoff and exact identity restriction visible.
    if (f <= 0.0 || f >= 1.0 || std::abs(v - full) <= 8.0 * std::numeric_limits<double>::epsilon() * full)
        return f * v;
    auto lo = intersection.lower, hi = intersection.upper;
    const auto c = center(donor);
    for (std::size_t a = 0; a < 3; ++a)
    {
        lo[a] -= c[a];
        hi[a] -= c[a];
    }
    return GeometricVoF::box_plane_volume(lo, hi, plane.normal, plane.offset);
}

double carrier_volume(InventoryCarrier carrier, double total, double gas, double rl, double rg)
{
    switch (carrier)
    {
        case InventoryCarrier::BulkVolume:
            return total;
        case InventoryCarrier::GasVolume:
            return gas;
        case InventoryCarrier::LiquidVolume:
            return total - gas;
        case InventoryCarrier::MixtureMass:
            return rl * (total - gas) + rg * gas;
    }
    throw std::invalid_argument("Unknown adaptive inventory carrier.");
}

using FacePoint = std::array<double, 2>;

std::vector<FacePoint> clip_face_polygon(const std::vector<FacePoint>& polygon, double a, double b, double threshold)
{
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(threshold)});
    if (polygon.empty() || scale == 0.0)
        return polygon;
    a /= scale;
    b /= scale;
    threshold /= scale;
    std::vector<FacePoint> result;
    for (std::size_t i = 0; i < polygon.size(); ++i)
    {
        const auto& start = polygon[i];
        const auto& end = polygon[(i + 1) % polygon.size()];
        const double first = a * start[0] + b * start[1] - threshold;
        const double second = a * end[0] + b * end[1] - threshold;
        if ((first >= 0.0) != (second >= 0.0))
        {
            const double fraction = first / (first - second);
            result.push_back({start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])});
        }
        if (second >= 0.0)
            result.push_back(end);
    }
    return result;
}
} // namespace

struct AdaptiveVoF::Impl
{
    struct Sample
    {
        GO cell;
        Box box;
        LO halo = -1;
    };
    struct Range
    {
        GO direct = -1; ///< Full same-level donor; converted from GID to halo ordinal at construction.
        std::size_t begin = 0, end = 0;
    };
    struct Patch
    {
        std::size_t block;
        UniformGrid grid;
        std::vector<Range> ranges;
        std::vector<Sample> samples; ///< Geometric intersections only for nonregular donors.
        std::vector<std::size_t> owned;
    };
    struct Geometry
    {
        std::shared_ptr<const Hierarchy> hierarchy;
        SimpleFluid::Meshes::AMR::HierarchyObservation observation;
        SP<const Handle> mesh;
        std::shared_ptr<const Exchange::Plan> cells, faces, stencil;
        std::vector<Patch> patches;
        std::vector<std::pair<std::size_t, std::size_t>> owned_patch_cell;

        explicit Geometry(std::shared_ptr<const Hierarchy> h) : hierarchy(std::move(h))
        {
            require_collective(bool(hierarchy), "AdaptiveVoF requires a hierarchy on every rank.");
            observation = hierarchy->observation();
            mesh = hierarchy->native_partition() ? std::make_shared<Handle>(*hierarchy->native_partition())
                                                 : std::make_shared<Handle>(hierarchy->geometry());
            cells = std::make_shared<Exchange::Plan>(mesh->owned_cell_map(), mesh->overlap_cell_map());
            faces = std::make_shared<Exchange::Plan>(mesh->owned_face_map(), mesh->overlap_face_map());
            std::set<std::size_t> blocks;
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                blocks.insert(hierarchy->cell_block(mesh->cell_global_id(static_cast<LO>(c))));
            std::set<GO> ids;
            std::map<std::size_t, std::size_t> patch_index;
            const auto& config = hierarchy->config();
            bool valid = true;
            try
            {
                for (const auto block : blocks)
                {
                    Patch patch;
                    patch.block = block;
                    patch.grid.spacing = hierarchy->block_spacing(block);
                    const auto bounds = hierarchy->block_box(block);
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        patch.grid.cells[a] = config.block_cells[a] + 2 * padding;
                        patch.grid.origin[a] = bounds.lower[a] - padding * patch.grid.spacing[a];
                    }
                    patch.ranges.resize(patch.grid.cell_count());
                    for (std::size_t c = 0; c < patch.ranges.size(); ++c)
                    {
                        Box box;
                        auto p = patch.grid.cell_center(c);
                        for (std::size_t a = 0; a < 3; ++a)
                        {
                            const double width = patch.grid.spacing[a], start = config.origin[a],
                                         length = config.extent[a];
                            if (config.periodic[a])
                                p[a] = start + (p[a] - start - std::floor((p[a] - start) / length) * length);
                            else
                            {
                                // Even scalar extension across slip walls, including
                                // deeper ghost layers and domains narrower than a halo.
                                const double period = 2.0 * length;
                                const double r = p[a] - start - std::floor((p[a] - start) / period) * period;
                                p[a] = start + (r > length ? period - r : r);
                                p[a] = std::clamp(p[a], start + 0.5 * width, start + length - 0.5 * width);
                            }
                            box.lower[a] = p[a] - 0.5 * width;
                            box.upper[a] = p[a] + 0.5 * width;
                        }
                        const auto overlaps = hierarchy->overlaps(box);
                        auto& range = patch.ranges[c];
                        range.begin = patch.samples.size();
                        for (const auto& overlap : overlaps)
                        {
                            ids.insert(static_cast<GO>(overlap.cell));
                            const auto donor = hierarchy->cell_box(overlap.cell);
                            if (overlaps.size() == 1 && overlap.box.lower == donor.lower &&
                                overlap.box.upper == donor.upper)
                                range.direct = static_cast<GO>(overlap.cell);
                            else
                                patch.samples.push_back({static_cast<GO>(overlap.cell), overlap.box});
                        }
                        range.end = patch.samples.size();
                        // overlaps() already validated any boundary excursion
                        // and clamped it. Use that same represented query here:
                        // large translations can round its width away from the
                        // nominal spacing even though coverage is complete.
                        for (std::size_t a = 0; a < 3; ++a)
                        {
                            const double end = config.origin[a] + config.extent[a];
                            box.lower[a] = std::clamp(box.lower[a], config.origin[a], end);
                            box.upper[a] = std::clamp(box.upper[a], config.origin[a], end);
                        }
                        if (!geometry_detail::has_complete_reconstruction_coverage(box, overlaps))
                            throw std::runtime_error(
                                "AdaptiveVoF reconstruction stencil has incomplete geometric coverage.");
                    }
                    patch_index[block] = patches.size();
                    patches.push_back(std::move(patch));
                }
            }
            catch (const std::exception&)
            {
                valid = false;
            }
            require_collective(valid, "AdaptiveVoF could not construct complete local reconstruction stencils.");
            stencil = Exchange(mesh->owned_cell_map(), ids).plan;
            for (auto& patch : patches)
            {
                for (auto& s : patch.samples)
                    s.halo = stencil->target->getLocalElement(s.cell);
                for (auto& range : patch.ranges)
                    if (range.direct >= 0)
                        range.direct = stencil->target->getLocalElement(range.direct);
            }
            const auto n = config.block_cells[0] * config.block_cells[1] * config.block_cells[2];
            owned_patch_cell.resize(mesh->num_owned_cells());
            for (std::size_t c = 0; c < owned_patch_cell.size(); ++c)
            {
                const auto gid = static_cast<std::size_t>(mesh->cell_global_id(static_cast<LO>(c)));
                const auto pi = patch_index.at(hierarchy->cell_block(gid));
                const auto local = gid % n;
                const auto i = local % config.block_cells[0],
                           j = (local / config.block_cells[0]) % config.block_cells[1],
                           k = local / (config.block_cells[0] * config.block_cells[1]);
                owned_patch_cell[c] = {pi, patches[pi].owned.size()};
                patches[pi].owned.push_back(patches[pi].grid.cell_index(i + padding, j + padding, k + padding));
            }
        }

    }; // Immutable hierarchy-derived geometry and communication plans.
    std::shared_ptr<const Geometry> geometry;
    const std::shared_ptr<const Hierarchy>& hierarchy;
    const SP<const Handle>& mesh;
    const std::vector<Patch>& patches;
    const std::vector<std::pair<std::size_t, std::size_t>>& owned_patch_cell;
    std::unique_ptr<Exchange> cells, faces, stencil;

    static std::shared_ptr<const Geometry> acquire(std::shared_ptr<const Hierarchy> hierarchy)
    {
        require_collective(bool(hierarchy), "AdaptiveVoF requires a hierarchy on every rank.");
        require_collective(hierarchy->geometry_unchanged(),
            "AdaptiveVoF cannot bind a hierarchy whose resident partition geometry has moved.");
        // Hierarchies are immutable, including their communicator ownership layout.
        // A hit must occur on every rank before any rank skips collective construction.
        static std::map<const Hierarchy*, std::weak_ptr<const Geometry>> cache;
        static std::mutex mutex;
        std::shared_ptr<const Geometry> cached;
        {
            const std::lock_guard lock(mutex);
            for (auto it = cache.begin(); it != cache.end();)
                if (it->second.expired())
                    it = cache.erase(it);
                else
                    ++it;
            const auto found = cache.find(hierarchy.get());
            if (found != cache.end())
                cached = found->second.lock();
        }
        int hit = cached && cached->observation == hierarchy->observation(), all_hit = 0;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &hit, &all_hit);
        if (all_hit)
            return cached;
        auto result = std::make_shared<const Geometry>(std::move(hierarchy));
        {
            const std::lock_guard lock(mutex);
            cache[result->hierarchy.get()] = result;
        }
        return result;
    }

    explicit Impl(std::shared_ptr<const Hierarchy> h)
        : geometry(acquire(std::move(h))), hierarchy(geometry->hierarchy), mesh(geometry->mesh),
          patches(geometry->patches), owned_patch_cell(geometry->owned_patch_cell),
          cells(std::make_unique<Exchange>(geometry->cells)), faces(std::make_unique<Exchange>(geometry->faces)),
          stencil(std::make_unique<Exchange>(geometry->stencil))
    {
    }

    bool geometry_unchanged() const noexcept
    {
        return hierarchy->geometry_unchanged() && geometry->observation == hierarchy->observation();
    }

    void validate_geometry() const
    {
        require_collective(geometry_unchanged(),
            "AdaptiveVoF requires its exact frozen hierarchy and resident partition geometry.");
    }

    UniformGrid operator_grid(const Patch& patch, std::size_t width) const
    {
        auto grid = patch.grid;
        for (std::size_t a = 0; a < 3; ++a)
        {
            grid.cells[a] -= 2 * (padding - width);
            grid.origin[a] += (padding - width) * grid.spacing[a];
        }
        return grid;
    }

    std::vector<double> sample(const Patch& patch, const std::vector<double>& halo,
        const std::vector<InterfacePlane>* planes = nullptr, std::size_t width = padding) const
    {
        const auto grid = operator_grid(patch, width);
        std::vector<double> f(grid.cell_count());
        for (std::size_t c = 0; c < f.size(); ++c)
        {
            auto p = grid.cell_coordinates(c);
            for (auto& x : p)
                x += padding - width;
            const auto& range = patch.ranges[patch.grid.cell_index(p[0], p[1], p[2])];
            if (range.direct >= 0)
            {
                f[c] = halo[range.direct];
                continue;
            }
            double gas = 0.0, covered = 0.0;
            for (std::size_t i = range.begin; i < range.end; ++i)
            {
                const auto& s = patch.samples[i];
                const double v = volume(s.box);
                gas += planes ? gas_intersection(hierarchy->cell_box(s.cell), s.box, halo[s.halo], (*planes)[s.halo])
                              : v * halo[s.halo];
                covered += v;
            }
            f[c] = gas / covered;
        }
        return f;
    }

    std::vector<InterfacePlane> reconstruct(const std::vector<double>& f) const
    {
        const auto halo = (*stencil)(f);
        std::vector<std::vector<InterfacePlane>> values;
        bool valid = true;
        try
        {
            for (const auto& patch : patches)
            {
                const auto grid = operator_grid(patch, 1);
                std::vector<std::size_t> requested;
                requested.reserve(patch.owned.size());
                for (const auto index : patch.owned)
                {
                    auto p = patch.grid.cell_coordinates(index);
                    for (auto& x : p)
                        x -= padding - 1;
                    requested.push_back(grid.cell_index(p[0], p[1], p[2]));
                }
                values.push_back(GeometricVoF(grid).reconstruct(sample(patch, halo, nullptr, 1), requested));
            }
        }
        catch (const std::exception&)
        {
            valid = false;
        }
        require_collective(valid, "AdaptiveVoF could not reconstruct its local geometric stencils.");
        std::vector<InterfacePlane> result(f.size());
        for (std::size_t c = 0; c < result.size(); ++c)
        {
            const auto [p, i] = owned_patch_cell[c];
            result[c] = values[p][i];
        }
        return result;
    }

    std::vector<InterfacePlane> exchange_planes(
        const std::vector<InterfacePlane>& owned, const Exchange& exchange) const
    {
        std::vector<InterfacePlane> result(exchange.target->getLocalNumElements());
        std::vector<std::vector<double>> components(4, std::vector<double>(owned.size()));
        for (std::size_t c = 0; c < owned.size(); ++c)
        {
            for (std::size_t a = 0; a < 3; ++a)
                components[a][c] = owned[c].normal[a];
            components[3][c] = owned[c].offset;
        }
        const auto halo = exchange.batch(components);
        for (std::size_t c = 0; c < result.size(); ++c)
        {
            for (std::size_t a = 0; a < 3; ++a)
                result[c].normal[a] = halo[a][c];
            result[c].offset = halo[3][c];
        }
        return result;
    }
};

AdaptiveVoF::AdaptiveVoF(std::shared_ptr<const Hierarchy> h) : impl_(std::make_unique<Impl>(std::move(h))) {}
AdaptiveVoF::~AdaptiveVoF() = default;
AdaptiveVoF::AdaptiveVoF(AdaptiveVoF&&) noexcept = default;
AdaptiveVoF& AdaptiveVoF::operator=(AdaptiveVoF&&) noexcept = default;
const std::shared_ptr<const AdaptiveVoF::Hierarchy>& AdaptiveVoF::hierarchy() const noexcept
{
    return impl_->hierarchy;
}
const SP<const AdaptiveVoF::Handle>& AdaptiveVoF::mesh() const noexcept
{
    return impl_->mesh;
}

AdaptiveGeometryStorage AdaptiveVoF::geometry_storage() const noexcept
{
    AdaptiveGeometryStorage result;
    result.patches = impl_->patches.size();
    for (const auto& patch : impl_->patches)
    {
        result.virtual_cells += patch.ranges.size();
        result.explicit_intersections += patch.samples.size();
        result.sample_payload_bytes +=
            patch.ranges.capacity() * sizeof(Impl::Range) + patch.samples.capacity() * sizeof(Impl::Sample);
        for (const auto& range : patch.ranges)
            result.direct_cells += range.direct >= 0;
    }
    return result;
}

std::vector<double> AdaptiveVoF::exchange_cells(const std::vector<double>& owned) const
{
    impl_->validate_geometry();
    require_collective(owned.size() == mesh()->num_owned_cells(), "AdaptiveVoF cell exchange requires owned values.");
    return (*impl_->cells)(owned);
}

std::vector<Vector> AdaptiveVoF::exchange_cell_vectors(const std::vector<Vector>& owned) const
{
    impl_->validate_geometry();
    require_collective(owned.size() == mesh()->num_owned_cells(), "AdaptiveVoF vector exchange requires owned values.");
    std::vector<std::vector<double>> components(3, std::vector<double>(owned.size()));
    for (std::size_t c = 0; c < owned.size(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
            components[a][c] = owned[c][a];
    const auto halo = impl_->cells->batch(components);
    std::vector<Vector> result(mesh()->num_cells());
    for (std::size_t c = 0; c < result.size(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
            result[c][a] = halo[a][c];
    return result;
}

std::vector<double> AdaptiveVoF::exchange_faces(const std::vector<double>& overlap) const
{
    impl_->validate_geometry();
    require_collective(overlap.size() == mesh()->num_faces(), "AdaptiveVoF face exchange requires overlap values.");
    return (*impl_->faces)(std::vector<double>(overlap.begin(), overlap.begin() + mesh()->num_owned_faces()));
}

std::vector<Vector> AdaptiveVoF::exchange_face_vectors(const std::vector<Vector>& overlap) const
{
    impl_->validate_geometry();
    require_collective(
        overlap.size() == mesh()->num_faces(), "AdaptiveVoF vector exchange requires overlap face values.");
    std::vector<std::vector<double>> components(3, std::vector<double>(mesh()->num_owned_faces()));
    for (std::size_t c = 0; c < mesh()->num_owned_faces(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
            components[a][c] = overlap[c][a];
    const auto halo = impl_->faces->batch(components);
    std::vector<Vector> result(mesh()->num_faces());
    for (std::size_t c = 0; c < result.size(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
            result[c][a] = halo[a][c];
    return result;
}

std::vector<InterfacePlane> AdaptiveVoF::reconstruct(const std::vector<double>& fraction) const
{
    impl_->validate_geometry();
    validate_f(fraction, mesh()->num_owned_cells());
    return impl_->reconstruct(fraction);
}

std::vector<double> AdaptiveVoF::curvature(const std::vector<double>& fraction) const
{
    const auto owned_planes = reconstruct(fraction);
    const auto planes = impl_->exchange_planes(owned_planes, *impl_->stencil);
    const auto halo = (*impl_->stencil)(fraction);
    std::vector<std::vector<double>> values;
    bool valid = true;
    try
    {
        for (const auto& patch : impl_->patches)
            values.push_back(GeometricVoF(patch.grid).curvature(impl_->sample(patch, halo, &planes), patch.owned));
    }
    catch (const std::exception&)
    {
        valid = false;
    }
    require_collective(valid, "AdaptiveVoF could not compute local curvature stencils.");
    std::vector<double> result(fraction.size());
    for (std::size_t c = 0; c < result.size(); ++c)
    {
        const auto [p, i] = impl_->owned_patch_cell[c];
        result[c] = values[p][i];
    }
    return result;
}

std::vector<InterfacePatch> AdaptiveVoF::interface_patches(const std::vector<double>& fraction) const
{
    const auto planes = reconstruct(fraction);
    const auto halo = exchange_cells(fraction);
    std::vector<InterfacePatch> patches;
    bool valid = true;
    try
    {
        const auto execution = mesh()->acquire_execution_view();
        const int rank = mesh()->owned_cell_map()->getComm()->getRank();
        const auto prepare = [&](InterfacePatch& patch)
        {
            const auto box = hierarchy()->cell_box(patch.carrier_cell);
            const auto origin = center(box);
            patch.source_hierarchy_ = hierarchy();
            patch.owner_rank_ = rank;
            for (std::size_t a = 0; a < 3; ++a)
            {
                patch.lower_[a] = box.lower[a] - origin[a];
                patch.upper_[a] = box.upper[a] - origin[a];
            }
            return origin;
        };
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            if (!(fraction[c] > 0.0 && fraction[c] < 1.0))
                continue;
            InterfacePatch patch;
            patch.gas_cell = patch.carrier_cell =
                static_cast<std::uint64_t>(mesh()->cell_global_id(static_cast<LO>(c)));
            patch.source_entity_ = patch.gas_cell;
            const auto origin = prepare(patch);
            const auto section =
                GeometricVoF::plane_section(patch.lower_, patch.upper_, planes[c].normal, planes[c].offset);
            if (!(section.area > 0.0))
                continue;
            patch.normal = patch.plane_normal_ = section.normal;
            const double magnitude = std::hypot(planes[c].normal[0], planes[c].normal[1], planes[c].normal[2]);
            patch.plane_offset_ = planes[c].offset / magnitude;
            patch.area = section.area;
            for (std::size_t a = 0; a < 3; ++a)
                patch.centroid[a] = origin[a] + section.centroid[a];
            patches.push_back(std::move(patch));
        }
        for (std::size_t face = 0; face < mesh()->num_owned_faces(); ++face)
        {
            const auto fid = static_cast<LO>(face);
            if (mesh()->is_geometry_exterior_face(fid))
                continue;
            const auto owner = mesh()->owner_cell(fid), neighbor = mesh()->neighbor_cell(fid);
            const bool owner_gas = halo[owner] >= 1.0 && halo[neighbor] <= 0.0;
            const bool neighbor_gas = halo[neighbor] >= 1.0 && halo[owner] <= 0.0;
            if (!owner_gas && !neighbor_gas)
                continue;
            InterfacePatch patch;
            patch.kind = InterfacePatch::Kind::MeshFace;
            const auto gas = owner_gas ? owner : neighbor;
            const auto carrier = owner_gas ? neighbor : owner;
            patch.gas_cell = static_cast<std::uint64_t>(mesh()->cell_global_id(gas));
            patch.carrier_cell = static_cast<std::uint64_t>(mesh()->cell_global_id(carrier));
            patch.source_entity_ = static_cast<std::uint64_t>(mesh()->face_global_id(fid));
            (void) prepare(patch);
            const auto normal = mesh()->face_normal(fid);
            const auto centroid = mesh()->face_centroid(fid);
            const auto from_carrier = mesh()->face_center_vector(fid, carrier);
            for (std::size_t a = 0; a < 3; ++a)
            {
                patch.normal[a] = patch.plane_normal_[a] = (owner_gas ? 1.0 : -1.0) * normal.component(a);
                patch.centroid[a] = centroid.component(a);
                patch.plane_offset_ += patch.normal[a] * from_carrier.component(a);
            }
            patch.area = mesh()->face_area(fid);
            patches.push_back(std::move(patch));
        }
    }
    catch (const std::exception&)
    {
        valid = false;
    }
    require_collective(valid, "AdaptiveVoF could not construct its interface geometry.");
    return patches;
}

std::vector<double> AdaptiveVoF::carrier_face_aperture(const std::vector<double>& fraction) const
{
    return phase_face_aperture(fraction, false);
}

std::vector<double> AdaptiveVoF::gas_face_aperture(const std::vector<double>& fraction) const
{
    return phase_face_aperture(fraction, true);
}

std::vector<double> AdaptiveVoF::phase_face_aperture(const std::vector<double>& fraction, bool gas) const
{
    const auto owned_planes = reconstruct(fraction);
    const auto planes = impl_->exchange_planes(owned_planes, *impl_->cells);
    const auto halo = exchange_cells(fraction);
    std::vector<double> aperture(mesh()->num_faces(), 0.0);
    bool valid = true;
    try
    {
        const auto execution = mesh()->acquire_execution_view();
        for (std::size_t face = 0; face < mesh()->num_owned_faces(); ++face)
        {
            const auto fid = static_cast<LO>(face);
            const bool boundary = mesh()->is_geometry_exterior_face(fid);
            if (boundary && !gas)
                continue;
            const auto owner = mesh()->owner_cell(fid), neighbor = boundary ? owner : mesh()->neighbor_cell(fid);
            if (gas ? (halo[owner] <= 0.0 || halo[neighbor] <= 0.0) : (halo[owner] >= 1.0 || halo[neighbor] >= 1.0))
                continue;
            const auto normal = mesh()->face_normal(fid);
            std::size_t axis = 0;
            while (axis < 2 && std::abs(normal.component(axis)) < 0.5)
                ++axis;
            const auto b = (axis + 1) % 3, c = (axis + 2) % 3;
            Vector lower{}, upper{};
            lower.fill(std::numeric_limits<double>::infinity());
            upper.fill(-std::numeric_limits<double>::infinity());
            const auto gid = static_cast<std::uint64_t>(mesh()->face_global_id(fid));
            for (const auto node : hierarchy()->geometry()->face_nodes(gid))
            {
                const auto point = hierarchy()->geometry()->node_coordinates(node);
                for (std::size_t a = 0; a < 3; ++a)
                {
                    lower[a] = std::min(lower[a], point.component(a));
                    upper[a] = std::max(upper[a], point.component(a));
                }
            }
            const double width_b = upper[b] - lower[b], width_c = upper[c] - lower[c];
            if (!(width_b > 0.0 && width_c > 0.0))
                throw std::runtime_error("Carrier aperture requires a positive Cartesian subface.");
            std::vector<FacePoint> polygon{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
            for (const auto cell : {owner, neighbor})
            {
                if (gas ? halo[cell] >= 1.0 : halo[cell] <= 0.0)
                    continue;
                const auto displacement = mesh()->face_center_vector(fid, cell);
                Vector origin{};
                for (std::size_t a = 0; a < 3; ++a)
                    origin[a] = displacement.component(a);
                origin[b] -= 0.5 * width_b;
                origin[c] -= 0.5 * width_c;
                const auto& plane = planes[cell];
                const double threshold =
                    plane.offset - std::inner_product(plane.normal.begin(), plane.normal.end(), origin.begin(), 0.0);
                const double sign = gas ? -1.0 : 1.0;
                polygon = clip_face_polygon(
                    polygon, sign * plane.normal[b] * width_b, sign * plane.normal[c] * width_c, sign * threshold);
            }
            double area = 0.0;
            for (std::size_t i = 0; i < polygon.size(); ++i)
            {
                const auto& a = polygon[i];
                const auto& z = polygon[(i + 1) % polygon.size()];
                area += a[0] * z[1] - a[1] * z[0];
            }
            area = 0.5 * std::abs(area);
            if (!std::isfinite(area) || area > 1.0 + 2.0e-12)
                throw std::runtime_error("Carrier aperture clipping produced an invalid fractional area.");
            aperture[face] = std::min(1.0, area);
        }
    }
    catch (const std::exception&)
    {
        valid = false;
    }
    require_collective(valid, "AdaptiveVoF could not construct shared carrier face apertures.");
    return exchange_faces(aperture);
}

double AdaptiveVoF::film_length(const InterfacePatch& patch, double delta) const
{
    if (!impl_->geometry_unchanged())
        throw std::invalid_argument("Interface film integration requires frozen source geometry.");
    if (patch.source_hierarchy_.get() != hierarchy().get() || !(patch.area > 0.0) || !std::isfinite(delta) ||
        !(delta > 0.0))
        throw std::invalid_argument(
            "Interface film integration requires a current patch and positive finite thickness.");
    Vector reversed{};
    for (std::size_t a = 0; a < 3; ++a)
        reversed[a] = -patch.plane_normal_[a];
    const auto carrier_volume = [&](double distance)
    { return GeometricVoF::box_plane_volume(patch.lower_, patch.upper_, reversed, -(patch.plane_offset_ + distance)); };
    const double carrier = carrier_volume(0.0);
    if (!(carrier > 0.0) || !std::isfinite(carrier))
        throw std::invalid_argument("Interface patch has no finite reconstructed carrier volume.");
    std::array<double, 8> vertex_distances{};
    double maximum_distance = 0.0;
    for (std::size_t corner = 0; corner < 8; ++corner)
    {
        double distance = -patch.plane_offset_;
        for (std::size_t a = 0; a < 3; ++a)
            distance += patch.plane_normal_[a] * ((corner >> a) & 1U ? patch.upper_[a] : patch.lower_[a]);
        vertex_distances[corner] = distance;
        maximum_distance = std::max(maximum_distance, distance);
    }
    const double end = std::min(delta, maximum_distance);
    if (!(end > 0.0))
        throw std::invalid_argument("Interface patch has no positive carrier-side depth.");
    std::vector<double> knots{0.0, end};
    for (const double distance : vertex_distances)
        if (distance > 0.0 && distance < end)
            knots.push_back(distance);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    detail::CompensatedSum<> integral;
    // Layer-cake identity: integral min(s,delta)dV = integral_0^delta V(s>t)dt.
    // Plane/box volume is piecewise cubic, with knots at vertex distances, so
    // two-point Gauss integration is exact on each piece up to roundoff.
    constexpr double abscissa = 0.57735026918962576451;
    for (std::size_t i = 1; i < knots.size(); ++i)
    {
        const double middle = std::midpoint(knots[i - 1], knots[i]);
        const double half = 0.5 * (knots[i] - knots[i - 1]);
        const double average = 0.5 * (carrier_volume(middle - abscissa * half) / carrier +
                                         carrier_volume(middle + abscissa * half) / carrier);
        integral += (static_cast<long double>(knots[i]) - knots[i - 1]) * average;
    }
    const double result = static_cast<double>(integral.value());
    if (!std::isfinite(result) || !(result > 0.0) || result > delta * (1.0 + 2.0e-12))
        throw std::runtime_error("Interface film integration produced an invalid geometric length.");
    return result;
}

void AdaptiveVoF::validate_patches(const std::vector<InterfacePatch>& patches, InterfaceSide side) const
{
    impl_->validate_geometry();
    identical(static_cast<double>(side), "Interface side selection differs between ranks.");
    bool valid = side == InterfaceSide::Gas || side == InterfaceSide::Carrier;
    const int rank = mesh()->owned_cell_map()->getComm()->getRank();
    const auto cells = hierarchy()->geometry()->num_cells();
    std::set<std::pair<InterfacePatch::Kind, std::uint64_t>> identities;
    for (const auto& patch : patches)
        valid = valid && patch.source_hierarchy_.get() == hierarchy().get() && patch.owner_rank_ == rank &&
                patch.gas_cell < cells && patch.carrier_cell < cells && std::isfinite(patch.area) && patch.area > 0.0 &&
                (patch.kind == InterfacePatch::Kind::Plic || patch.kind == InterfacePatch::Kind::MeshFace) &&
                identities.emplace(patch.kind, patch.source_entity_).second;
    require_collective(
        valid, "Interface operations require unique, locally owned patches from the exact source hierarchy.");
}

std::vector<double> AdaptiveVoF::scatter_patch_amounts(
    const std::vector<InterfacePatch>& patches, const std::vector<double>& amounts, InterfaceSide side) const
{
    validate_patches(patches, side);
    require_collective(amounts.size() == patches.size() && std::all_of(amounts.begin(), amounts.end(),
                                                               [](double value) { return std::isfinite(value); }),
        "Interface deposition requires one finite signed amount per local patch.");
    std::set<GO> ids;
    for (const auto& patch : patches)
        ids.insert(static_cast<GO>(side == InterfaceSide::Gas ? patch.gas_cell : patch.carrier_cell));
    const Exchange exchange(mesh()->owned_cell_map(), ids);
    std::vector<detail::CompensatedSum<double>> sums(ids.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        const auto cell = side == InterfaceSide::Gas ? patches[i].gas_cell : patches[i].carrier_cell;
        sums[exchange.target->getLocalElement(static_cast<GO>(cell))] += amounts[i];
    }
    std::vector<double> local(sums.size());
    for (std::size_t i = 0; i < sums.size(); ++i)
        local[i] = sums[i].value();
    require_collective(std::all_of(local.begin(), local.end(), [](double value) { return std::isfinite(value); }),
        "Interface patch amounts overflowed their destination inventory.");
    auto result = exchange.sum_to_source(local);
    require_collective(std::all_of(result.begin(), result.end(), [](double value) { return std::isfinite(value); }),
        "Interface deposition overflowed after collective accumulation.");
    return result;
}

std::vector<double> AdaptiveVoF::sample_patch_values(
    const std::vector<InterfacePatch>& patches, const std::vector<double>& owned_values, InterfaceSide side) const
{
    validate_patches(patches, side);
    require_collective(
        owned_values.size() == mesh()->num_owned_cells() &&
            std::all_of(owned_values.begin(), owned_values.end(), [](double value) { return std::isfinite(value); }),
        "Interface sampling requires a finite owned-cell scalar.");
    std::set<GO> ids;
    for (const auto& patch : patches)
        ids.insert(static_cast<GO>(side == InterfaceSide::Gas ? patch.gas_cell : patch.carrier_cell));
    const Exchange exchange(mesh()->owned_cell_map(), ids);
    const auto imported = exchange(owned_values);
    std::vector<double> result(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        const auto cell = side == InterfaceSide::Gas ? patches[i].gas_cell : patches[i].carrier_cell;
        result[i] = imported[exchange.target->getLocalElement(static_cast<GO>(cell))];
    }
    return result;
}

AdaptiveAdvection AdaptiveVoF::advect(const std::vector<double>& fraction, const std::vector<double>& face_velocity,
    double dt, std::size_t first_axis) const
{
    return advect_with_sources(fraction, face_velocity, dt, nullptr, first_axis);
}

AdaptiveAdvection AdaptiveVoF::advect(const std::vector<double>& fraction, const std::vector<double>& face_velocity,
    double dt, const PhaseVolumeIncrement& source, std::size_t first_axis) const
{
    return advect_with_sources(fraction, face_velocity, dt, &source, first_axis);
}

AdaptiveAdvection AdaptiveVoF::advect_with_sources(const std::vector<double>& fraction,
    const std::vector<double>& face_velocity, double dt, const PhaseVolumeIncrement* source,
    std::size_t first_axis) const
{
    impl_->validate_geometry();
    validate_f(fraction, mesh()->num_owned_cells());
    identical(dt, "AdaptiveVoF timestep must be identical and finite on every rank.");
    identical(static_cast<double>(first_axis), "AdaptiveVoF sweep order differs between ranks.");
    identical(source ? 1.0 : 0.0, "AdaptiveVoF physical-source selection differs between ranks.");
    require_collective(dt >= 0.0 && first_axis < 3 && face_velocity.size() == mesh()->num_faces(),
        "AdaptiveVoF requires a nonnegative timestep, sweep axis and overlap face velocity.");
    std::vector<double> total_source;
    if (source)
    {
        require_collective(source->gas.size() == fraction.size() && source->carrier.size() == fraction.size(),
            "AdaptiveVoF phase-volume increments require owned-cell ordering.");
        bool valid_source = true;
        total_source.resize(fraction.size());
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            total_source[c] = source->gas[c] + source->carrier[c];
            valid_source = valid_source && std::isfinite(source->gas[c]) && std::isfinite(source->carrier[c]) &&
                           std::isfinite(total_source[c]) &&
                           (dt > 0.0 || (source->gas[c] == 0.0 && source->carrier[c] == 0.0));
        }
        require_collective(valid_source,
            "AdaptiveVoF phase-volume increments must be finite; nonzero increments require positive dt.");
    }
    auto u = exchange_faces(face_velocity);
    bool valid = true;
    {
        const auto execution = mesh()->acquire_execution_view();
        for (std::size_t face = 0; face < u.size(); ++face)
            valid = valid && std::isfinite(u[face]) &&
                    (!mesh()->is_geometry_exterior_face(static_cast<LO>(face)) || u[face] == 0.0);
    }
    require_collective(valid, "AdaptiveVoF requires finite velocities and impermeable wall flux.");
    AdaptiveAdvection result{fraction, std::vector<double>(u.size()), std::vector<double>(u.size()), 0.0};
    double maxdiv = 0.0, maxresidual = 0.0, maxcfl = 0.0;
    detail::CompensatedSum<double> local_source, local_volume;
    {
        const auto execution = mesh()->acquire_execution_view();
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            double sum = 0.0;
            std::array<double, 3> axis_cfl{};
            mesh()->visit_cell_faces(static_cast<LO>(c),
                [&](LO f)
                {
                    const double sign = mesh()->owner_cell(f) == static_cast<LO>(c) ? 1.0 : -1.0;
                    const auto n = mesh()->face_normal(f);
                    std::size_t a = 0;
                    while (a < 2 && std::abs(n.component(a)) < 0.5)
                        ++a;
                    const double flux = dt * u[f] * mesh()->face_area(f);
                    sum += sign * flux;
                    // Half the absolute two-sided throughput; independently bound each subface width below.
                    axis_cfl[a] += 0.5 * std::abs(flux) / mesh()->cell_volume(static_cast<LO>(c));
                    const auto h =
                        hierarchy()->block_spacing(hierarchy()->cell_block(mesh()->cell_global_id(static_cast<LO>(c))));
                    if (std::abs(dt * u[f]) / h[a] > 0.5 + 1e-14)
                        valid = false;
                });
            maxdiv = std::max(maxdiv, std::abs(sum) / mesh()->cell_volume(static_cast<LO>(c)));
            maxresidual = std::max(maxresidual,
                std::abs(sum - (source ? total_source[c] : 0.0)) / mesh()->cell_volume(static_cast<LO>(c)));
            if (source)
            {
                local_source += total_source[c];
                local_volume += mesh()->cell_volume(static_cast<LO>(c));
            }
            maxcfl = std::max(maxcfl, std::accumulate(axis_cfl.begin(), axis_cfl.end(), 0.0));
        }
    }
    require_collective(valid, "AdaptiveVoF subface swept width exceeds half its donor cell.");
    result.maximum_volume_divergence = maximum(maxdiv);
    result.maximum_volume_source_residual = maximum(maxresidual);
    if (source)
    {
        const std::array<double, 2> local{local_source.value(), local_volume.value()};
        std::array<double, 2> global{};
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local.data(), global.data());
        require_collective(std::isfinite(global[0]) && std::isfinite(global[1]) && global[1] > 0.0 &&
                               std::abs(global[0]) <= 1.0e-10 * global[1],
            "AdaptiveVoF phase-volume sources are incompatible with its closed fixed domain.");
    }
    require_collective(maximum(maxcfl) <= 0.5 + 1e-14 && result.maximum_volume_source_residual <= 1e-10,
        "AdaptiveVoF requires CFL <= 0.5 and phase-source/face-volume continuity residual <= 1e-10.");
    const auto frozen = fraction;
    for (std::size_t sweep = 0; sweep < 3; ++sweep)
    {
        const auto a = (first_axis + sweep) % 3;
        double active = 0.0;
        {
            const auto execution = mesh()->acquire_execution_view();
            for (std::size_t face = 0; face < mesh()->num_owned_faces(); ++face)
                if (std::abs(mesh()->face_normal(static_cast<LO>(face)).component(a)) > 0.5)
                    active = std::max(active, std::abs(u[face]));
        }
        if (dt == 0.0 || maximum(active) == 0.0)
            continue;
        const auto owned_planes = reconstruct(result.fraction);
        const auto planes = impl_->exchange_planes(owned_planes, *impl_->cells);
        const auto fhalo = exchange_cells(result.fraction);
        std::vector<double> gas(u.size(), 0.0), total(u.size(), 0.0);
        valid = true;
        try
        {
            const auto execution = mesh()->acquire_execution_view();
            for (std::size_t face = 0; face < mesh()->num_owned_faces(); ++face)
            {
                const auto fid = static_cast<LO>(face);
                if (mesh()->is_geometry_exterior_face(fid) || std::abs(mesh()->face_normal(fid).component(a)) < 0.5 ||
                    u[face] == 0.0)
                    continue;
                const auto donor = u[face] > 0.0 ? mesh()->owner_cell(fid) : mesh()->neighbor_cell(fid);
                total[face] = u[face] * dt * mesh()->face_area(fid);
                if (fhalo[donor] <= 0.0 || fhalo[donor] >= 1.0)
                {
                    // Retain accepted inventory roundoff in f, but keep the
                    // physical face partition one-sided at a pure endpoint.
                    // Multiplying a retained f=1+ulp creates a negative carrier
                    // flux into a cell with exactly zero carrier capacity.
                    gas[face] = fhalo[donor] >= 1.0 ? total[face] : 0.0;
                    continue;
                }
                const auto gid = mesh()->cell_global_id(donor);
                const auto h = hierarchy()->block_spacing(hierarchy()->cell_block(gid));
                const auto other = mesh()->opposite_cell(fid, donor);
                const auto ho = hierarchy()->block_spacing(hierarchy()->cell_block(mesh()->cell_global_id(other)));
                const auto dc = mesh()->face_center_vector(fid, donor);
                // Scale the swept prism to a unit box. Subtracting a tiny
                // displacement from a global coordinate could otherwise
                // collapse its width, or relabel pure gas flux as liquid.
                Vector lower{}, widths{}, normal{};
                for (std::size_t b = 0; b < 3; ++b)
                {
                    const double w = std::min(h[b], ho[b]);
                    lower[b] = dc.component(b) - 0.5 * w;
                    widths[b] = w;
                }
                const double width = std::abs(u[face]) * dt;
                widths[a] = width;
                lower[a] = dc.component(a) > 0.0 ? 0.5 * h[a] : -0.5 * h[a];
                double offset = planes[donor].offset;
                for (std::size_t b = 0; b < 3; ++b)
                {
                    normal[b] = planes[donor].normal[b] * widths[b];
                    offset -= planes[donor].normal[b] * lower[b];
                }
                if (dc.component(a) > 0.0)
                    offset += planes[donor].normal[a] * width;
                gas[face] = total[face] * GeometricVoF::box_plane_volume({0, 0, 0}, {1, 1, 1}, normal, offset);
            }
        }
        catch (const std::exception&)
        {
            valid = false;
        }
        require_collective(valid, "AdaptiveVoF failed to integrate a swept subface volume.");
        gas = exchange_faces(gas);
        total = exchange_faces(total);
        // A quantized microscopic PLIC plane can export more phase than its
        // exact donor inventory. Correct only roundoff-sized excess, on the
        // canonical faces; never discard a positive cell inventory or clip f.
        for (unsigned correction = 0; correction < 4; ++correction)
        {
            std::vector<Vector> factors(fraction.size(), Vector{1.0, 1.0, 0.0});
            bool needs_correction = false;
            valid = true;
            try
            {
                const auto execution = mesh()->acquire_execution_view();
                for (std::size_t c = 0; c < fraction.size(); ++c)
                {
                    long double gas_out = 0.0L, carrier_out = 0.0L;
                    mesh()->visit_cell_faces(static_cast<LO>(c),
                        [&](LO face)
                        {
                            const double sign = mesh()->owner_cell(face) == static_cast<LO>(c) ? 1.0 : -1.0;
                            if (sign * total[face] > 0.0)
                            {
                                gas_out += sign * static_cast<long double>(gas[face]);
                                carrier_out += sign * static_cast<long double>(total[face] - gas[face]);
                            }
                        });
                    const auto factor = transport_detail::phase_budget(
                        gas_out, carrier_out, result.fraction[c], mesh()->cell_volume(static_cast<LO>(c)), roundoff);
                    factors[c] = {factor.gas, factor.carrier, 0.0};
                    needs_correction = needs_correction || factor.changes_flux();
                }
            }
            catch (const std::exception&)
            {
                valid = false;
            }
            require_collective(valid, "AdaptiveVoF directional outflow exceeds its donor phase budget.");
            if (maximum(needs_correction ? 1.0 : 0.0) == 0.0)
                break;
            require_collective(correction < 3, "AdaptiveVoF could not resolve a roundoff-only phase budget.");
            const auto halo_factors = exchange_cell_vectors(factors);
            {
                const auto execution = mesh()->acquire_execution_view();
                for (std::size_t face = 0; face < mesh()->num_owned_faces(); ++face)
                {
                    if (total[face] == 0.0)
                        continue;
                    const auto fid = static_cast<LO>(face);
                    const auto donor = total[face] > 0.0 ? mesh()->owner_cell(fid) : mesh()->neighbor_cell(fid);
                    gas[face] = transport_detail::budgeted_gas(
                        total[face], gas[face], {halo_factors[donor][0], halo_factors[donor][1]});
                }
            }
            gas = exchange_faces(gas);
        }
        {
            const auto execution = mesh()->acquire_execution_view();
            for (std::size_t c = 0; c < fraction.size(); ++c)
            {
                long double gas_in = 0.0L, gas_out = 0.0L, carrier_in = 0.0L, carrier_out = 0.0L;
                mesh()->visit_cell_faces(static_cast<LO>(c),
                    [&](LO face)
                    {
                        const double sign = mesh()->owner_cell(face) == static_cast<LO>(c) ? 1.0 : -1.0;
                        const long double g = sign * static_cast<long double>(gas[face]);
                        const long double carrier = sign * static_cast<long double>(total[face] - gas[face]);
                        gas_out += std::max(0.0L, g);
                        gas_in += std::max(0.0L, -g);
                        carrier_out += std::max(0.0L, carrier);
                        carrier_in += std::max(0.0L, -carrier);
                    });
                const double v = mesh()->cell_volume(static_cast<LO>(c));
                // These minority-phase forms are the frozen-WY update:
                // G'=G-netG+chi*netV, C'=C-netC+(1-chi)*netV.
                // Keeping (old-out)+in avoids cancellation of a positive tail.
                if (frozen[c] < 0.5)
                    result.fraction[c] = static_cast<double>(
                        ((static_cast<long double>(result.fraction[c]) * v - gas_out) + gas_in) / v);
                else
                    result.fraction[c] =
                        static_cast<double>(1.0L - (((1.0L - result.fraction[c]) * v - carrier_out) + carrier_in) / v);
            }
        }
        validate_f(result.fraction, fraction.size());
        for (std::size_t face = 0; face < u.size(); ++face)
        {
            result.gas_volume[face] += gas[face];
            result.liquid_volume[face] += total[face] - gas[face];
        }
    }
    if (source)
    {
        const auto execution = mesh()->acquire_execution_view();
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            const double correction = source->gas[c] - (frozen[c] >= 0.5 ? total_source[c] : 0.0);
            if (correction != 0.0)
                result.fraction[c] += correction / mesh()->cell_volume(static_cast<LO>(c));
        }
    }
    validate_f(result.fraction, fraction.size());
    return result;
}

AdaptiveRemapResult AdaptiveVoF::remap(const AdaptiveVoF& target, const std::vector<double>& fraction,
    const std::vector<ConservedQuantity>& inventories, double rl, double rg) const
{
    impl_->validate_geometry();
    target.impl_->validate_geometry();
    validate_f(fraction, mesh()->num_owned_cells());
    identical(rl, "Adaptive remap liquid density must agree on all ranks.");
    identical(rg, "Adaptive remap gas density must agree on all ranks.");
    identical(static_cast<double>(inventories.size()), "Adaptive remap inventory counts differ between ranks.");
    const auto& old = hierarchy()->config();
    const auto& next = target.hierarchy()->config();
    require_collective(
        rl > 0.0 && rg > 0.0 && old.origin == next.origin && old.extent == next.extent && old.periodic == next.periodic,
        "Adaptive remap requires positive densities and the same physical domain and boundaries.");
    bool valid = true;
    std::set<std::string> names;
    for (const auto& inventory : inventories)
    {
        identical(static_cast<double>(inventory.carrier), "Adaptive inventory carriers differ between ranks.");
        identical(inventory.nonnegative ? 1.0 : 0.0, "Adaptive inventory sign policies differ between ranks.");
        // An exact byte comparison avoids implementation-dependent string hashes.
        identical(static_cast<double>(inventory.name.size()), "Adaptive inventory names differ between ranks.");
        require_collective(inventory.name.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Adaptive inventory names exceed the collective byte-count limit.");
        std::vector<unsigned char> reference(inventory.name.begin(), inventory.name.end());
        Teuchos::broadcast(
            *mesh()->owned_cell_map()->getComm(), 0, static_cast<int>(reference.size()), reference.data());
        require_collective(std::equal(reference.begin(), reference.end(), inventory.name.begin(),
                               [](unsigned char a, char b) { return a == static_cast<unsigned char>(b); }),
            "Adaptive inventory names differ between ranks.");
        valid = valid && !inventory.name.empty() && names.insert(inventory.name).second &&
                static_cast<unsigned>(inventory.carrier) <= static_cast<unsigned>(InventoryCarrier::MixtureMass) &&
                inventory.amount.size() == fraction.size() &&
                std::all_of(inventory.amount.begin(), inventory.amount.end(),
                    [&](double x) { return std::isfinite(x) && (!inventory.nonnegative || x >= 0.0); });
    }
    require_collective(valid, "Adaptive remap requires unique named finite extensive inventories.");
    std::vector<std::vector<SimpleFluid::Meshes::AMR::AMROverlap>> overlaps(target.mesh()->num_owned_cells());
    std::set<GO> ids;
    for (std::size_t c = 0; c < overlaps.size(); ++c)
    {
        overlaps[c] =
            hierarchy()->overlaps(target.hierarchy()->cell_box(target.mesh()->cell_global_id(static_cast<LO>(c))));
        for (const auto& item : overlaps[c])
            ids.insert(static_cast<GO>(item.cell));
    }
    Exchange donors(mesh()->owned_cell_map(), ids);
    const auto f = donors(fraction);
    const auto planes = impl_->exchange_planes(reconstruct(fraction), donors);
    // Close each donor's PLIC roundoff once, across all receiving ranks.
    // This preserves tiny gas inventories whose subcell plane offset is near
    // floating-point resolution; it is not a post-remap fraction clipping.
    std::vector<std::vector<double>> raw_gas(overlaps.size());
    std::vector<double> local_gas(f.size()), local_volume(f.size());
    for (std::size_t c = 0; c < overlaps.size(); ++c)
        for (const auto& overlap : overlaps[c])
        {
            const auto lid = donors.target->getLocalElement(overlap.cell);
            const auto gas = gas_intersection(hierarchy()->cell_box(overlap.cell), overlap.box, f[lid], planes[lid]);
            raw_gas[c].push_back(gas);
            local_gas[lid] += gas;
            local_volume[lid] += volume(overlap.box);
        }
    const auto donor_gas = donors.sum_to_source(local_gas), donor_volume = donors.sum_to_source(local_volume);
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const double full = mesh()->cell_volume(static_cast<LO>(c));
        valid = valid && std::abs(donor_volume[c] - full) <= 2e-12 * full &&
                std::abs(donor_gas[c] - fraction[c] * full) <= 2e-12 * full;
    }
    require_collective(valid, "Adaptive remap geometric donor coverage or PLIC closure exceeds roundoff tolerance.");
    const auto total_gas = donors(donor_gas);
    std::vector<std::vector<double>> amounts;
    for (const auto& inventory : inventories)
        amounts.push_back(donors(inventory.amount));
    AdaptiveRemapResult result;
    result.fraction.resize(overlaps.size());
    result.inventories = inventories;
    for (auto& inventory : result.inventories)
        inventory.amount.assign(overlaps.size(), 0.0);
    valid = true;
    try
    {
        for (std::size_t c = 0; c < overlaps.size(); ++c)
        {
            double gas = 0.0, covered = 0.0;
            std::size_t index = 0;
            for (const auto& overlap : overlaps[c])
            {
                const auto lid = donors.target->getLocalElement(overlap.cell);
                const auto box = hierarchy()->cell_box(overlap.cell);
                const double v = volume(overlap.box), full = volume(box);
                const double raw = raw_gas[c][index++], desired = f[lid] * full;
                double gv = raw;
                if (total_gas[lid] > desired && total_gas[lid] > 0.0)
                    gv = raw * (desired / total_gas[lid]);
                else if (total_gas[lid] < desired && full > total_gas[lid])
                    gv = raw + (v - raw) * ((desired - total_gas[lid]) / (full - total_gas[lid]));
                gas += gv;
                covered += v;
                for (std::size_t q = 0; q < inventories.size(); ++q)
                {
                    const double denominator = carrier_volume(inventories[q].carrier, full, f[lid] * full, rl, rg);
                    const double numerator = carrier_volume(inventories[q].carrier, v, gv, rl, rg);
                    if (denominator <= 0.0)
                    {
                        if (amounts[q][lid] != 0.0)
                            valid = false;
                    }
                    else
                        result.inventories[q].amount[c] += amounts[q][lid] * (numerator / denominator);
                }
            }
            const double v = target.mesh()->cell_volume(static_cast<LO>(c));
            if (std::abs(covered - v) > 1e-9 * v)
                valid = false;
            result.fraction[c] = gas / v;
            for (const auto& inventory : result.inventories)
                if (!std::isfinite(inventory.amount[c]) || (inventory.nonnegative && inventory.amount[c] < -roundoff))
                    valid = false;
        }
    }
    catch (const std::exception&)
    {
        valid = false;
    }
    require_collective(valid, "Adaptive remap has missing coverage, exhausted carrier or a nonfinite inventory.");
    validate_f(result.fraction, result.fraction.size());
    // Inventory checks use extensive norms, including signed momentum whose
    // net sum can vanish. They are independent of pressure-solver tolerances.
    for (std::size_t q = 0; q <= inventories.size(); ++q)
    {
        detail::CompensatedSum<double> before, after, norm_before, norm_after;
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            const double x =
                q == 0 ? fraction[c] * mesh()->cell_volume(static_cast<LO>(c)) : inventories[q - 1].amount[c];
            before += x;
            norm_before += std::abs(x);
        }
        for (std::size_t c = 0; c < result.fraction.size(); ++c)
        {
            const double x = q == 0 ? result.fraction[c] * target.mesh()->cell_volume(static_cast<LO>(c))
                                    : result.inventories[q - 1].amount[c];
            after += x;
            norm_after += std::abs(x);
        }
        const std::array<double, 4> local{before.value(), after.value(), norm_before.value(), norm_after.value()};
        std::array<double, 4> global{};
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 4, local.data(), global.data());
        require_collective(std::isfinite(global[0]) && std::isfinite(global[1]) &&
                               std::abs(global[1] - global[0]) <=
                                   2e-11 * std::max({global[2], global[3], std::numeric_limits<double>::min()}),
            "Adaptive remap failed its global extensive conservation gate.");
    }
    return result;
}

RefinementPlan AdaptiveVoF::refinement_plan(const std::vector<double>& fraction,
    const std::vector<double>& face_velocity, RefinementOptions options, RefinementHints hints) const
{
    impl_->validate_geometry();
    validate_f(fraction, mesh()->num_owned_cells());
    identical(options.minimum_resolved_diameter, "Refinement diameter differs between ranks.");
    identical(options.cells_per_diameter, "Refinement resolution differs between ranks.");
    identical(options.stencil_cells, "Refinement stencil differs between ranks.");
    identical(options.forecast_time, "Refinement forecast differs between ranks.");
    identical(options.curvature_resolution, "Refinement curvature threshold differs between ranks.");
    identical(options.wake_threshold, "Refinement wake threshold differs between ranks.");
    require_collective(options.minimum_resolved_diameter > 0 && options.cells_per_diameter >= 4 &&
                           options.stencil_cells >= protected_stencil && options.forecast_time >= 0 &&
                           options.curvature_resolution >= 0 && options.wake_threshold > 0 &&
                           (hints.wake_indicator.empty() || hints.wake_indicator.size() == fraction.size()) &&
                           std::all_of(hints.wake_indicator.begin(), hints.wake_indicator.end(),
                               [](double x) { return std::isfinite(x) && x >= 0; }),
        "Invalid adaptive refinement controls or wake field.");
    identical(static_cast<double>(hints.promotion_regions.size()), "Promotion region counts differ between ranks.");
    for (const auto& b : hints.promotion_regions)
        for (std::size_t a = 0; a < 3; ++a)
        {
            identical(b.lower[a], "Promotion region bounds differ between ranks.");
            identical(b.upper[a], "Promotion region bounds differ between ranks.");
            require_collective(b.upper[a] >= b.lower[a], "Promotion region bounds must be ordered.");
        }
    const auto u = exchange_faces(face_velocity);
    require_collective(
        std::all_of(u.begin(), u.end(), [](double x) { return std::isfinite(x); }), "Nonfinite refinement velocity.");
    double speed = 0.0;
    for (const double x : u)
        speed = std::max(speed, std::abs(x));
    const double travel = maximum(speed) * options.forecast_time;
    require_collective(std::isfinite(travel), "Adaptive motion buffer overflows physical coordinates.");
    const auto f = exchange_cells(fraction);
    const auto k = options.curvature_resolution > 0.0 ? curvature(fraction) : std::vector<double>(fraction.size(), 0.0);
    const auto& blocks = hierarchy()->blocks();
    const auto& config = hierarchy()->config();
    std::vector<int> protected_blocks(blocks.size()), tagged(blocks.size()), phase_min(blocks.size(), 2),
        phase_max(blocks.size(), -1);
    const double target = options.minimum_resolved_diameter / options.cells_per_diameter;
    struct Guard
    {
        Box box;
        bool force_refinement;
    };
    std::vector<Guard> guards;
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto gid = mesh()->cell_global_id(static_cast<LO>(c));
        const auto block = hierarchy()->cell_block(gid);
        // Positive trace gas is still an inventory: never classify it as an
        // empty cell merely to make coarsening easier.
        const int phase = fraction[c] <= 0.0 ? 0 : (fraction[c] >= 1.0 ? 1 : 2);
        phase_min[block] = std::min(phase_min[block], phase);
        phase_max[block] = std::max(phase_max[block], phase);
        bool interface = phase == 2;
        mesh()->visit_cell_faces(static_cast<LO>(c),
            [&](LO face)
            {
                const auto other = mesh()->opposite_cell(face, static_cast<LO>(c));
                if (other != Handle::invalid_local_id() && ((fraction[c] >= 0.5) != (f[other] >= 0.5)))
                    interface = true;
            });
        const bool wake = !hints.wake_indicator.empty() && hints.wake_indicator[c] > options.wake_threshold;
        if (!interface && !wake)
            continue;
        auto guard = hierarchy()->cell_box(gid);
        const auto h = hierarchy()->block_spacing(block);
        const double maxh = *std::max_element(h.begin(), h.end());
        for (std::size_t a = 0; a < 3; ++a)
        {
            const double r = options.stencil_cells * maxh + travel;
            guard.lower[a] -= r;
            guard.upper[a] += r;
        }
        guards.push_back({guard,
            wake || (options.curvature_resolution > 0.0 && std::abs(k[c]) * maxh > options.curvature_resolution)});
    }
    // Merge only exact rectangular unions: the other two axes and refinement
    // criterion must coincide. This preserves the original protected/tagged sets.
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto same_strip = [&](const Guard& a, const Guard& b)
        {
            if (a.force_refinement != b.force_refinement)
                return false;
            for (std::size_t d = 0; d < 3; ++d)
                if (d != axis && (a.box.lower[d] != b.box.lower[d] || a.box.upper[d] != b.box.upper[d]))
                    return false;
            return true;
        };
        std::sort(guards.begin(), guards.end(),
            [&](const Guard& a, const Guard& b)
            {
                if (a.force_refinement != b.force_refinement)
                    return a.force_refinement < b.force_refinement;
                for (std::size_t d = 0; d < 3; ++d)
                    if (d != axis)
                    {
                        if (a.box.lower[d] != b.box.lower[d])
                            return a.box.lower[d] < b.box.lower[d];
                        if (a.box.upper[d] != b.box.upper[d])
                            return a.box.upper[d] < b.box.upper[d];
                    }
                return a.box.lower[axis] < b.box.lower[axis];
            });
        std::size_t count = 0;
        for (const auto& guard : guards)
            if (count && same_strip(guards[count - 1], guard) &&
                guards[count - 1].box.upper[axis] >= guard.box.lower[axis])
                guards[count - 1].box.upper[axis] = std::max(guards[count - 1].box.upper[axis], guard.box.upper[axis]);
            else
                guards[count++] = guard;
        guards.resize(count);
    }
    for (const auto& entry : guards)
    {
        const auto& guard = entry.box;
        // Query only permitted periodic images. Translating the query instead of
        // every leaf retains the exact closed-box/corner predicate.
        const int xmin = config.periodic[0] ? -1 : 0, xmax = config.periodic[0] ? 1 : 0;
        const int ymin = config.periodic[1] ? -1 : 0, ymax = config.periodic[1] ? 1 : 0;
        const int zmin = config.periodic[2] ? -1 : 0, zmax = config.periodic[2] ? 1 : 0;
        for (int z = zmin; z <= zmax; ++z)
            for (int y = ymin; y <= ymax; ++y)
                for (int x = xmin; x <= xmax; ++x)
                {
                    auto query = guard;
                    const std::array<int, 3> shift{x, y, z};
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        query.lower[a] -= shift[a] * config.extent[a];
                        query.upper[a] -= shift[a] * config.extent[a];
                        // Candidate padding only: translating floating coordinates
                        // must not exclude a leaf accepted by the original image test.
                        const double margin = 16.0 * std::numeric_limits<double>::epsilon() *
                                              std::max({std::abs(query.lower[a]), std::abs(query.upper[a]),
                                                  std::abs(config.origin[a]), config.extent[a]});
                        query.lower[a] -= margin;
                        query.upper[a] += margin;
                    }
                    for (const auto b : hierarchy()->intersecting_blocks(query))
                    {
                        auto image = hierarchy()->block_box(b);
                        for (std::size_t a = 0; a < 3; ++a)
                        {
                            image.lower[a] += shift[a] * config.extent[a];
                            image.upper[a] += shift[a] * config.extent[a];
                        }
                        if (intersects(guard, image))
                        {
                            protected_blocks[b] = 1;
                            const auto hb = hierarchy()->block_spacing(b);
                            if (entry.force_refinement ||
                                *std::max_element(hb.begin(), hb.end()) > target * (1.0 + 1e-12))
                                tagged[b] = 1;
                        }
                    }
                }
    }
    for (const auto& region : hints.promotion_regions)
        for (const auto b : hierarchy()->intersecting_blocks(region))
        {
            tagged[b] = 1;
            protected_blocks[b] = 1;
        }
    const auto reduce = [&](std::vector<int>& v, Teuchos::EReductionType operation)
    {
        std::vector<int> global(v.size());
        Teuchos::reduceAll(
            *mesh()->owned_cell_map()->getComm(), operation, static_cast<int>(v.size()), v.data(), global.data());
        v = std::move(global);
    };
    reduce(protected_blocks, Teuchos::REDUCE_MAX);
    reduce(tagged, Teuchos::REDUCE_MAX);
    reduce(phase_min, Teuchos::REDUCE_MIN);
    reduce(phase_max, Teuchos::REDUCE_MAX);
    RefinementPlan plan;
    plan.source_hierarchy = hierarchy();
    plan.source_observation = hierarchy()->observation();
    std::map<SimpleFluid::Meshes::AMR::AMRBlock, std::vector<std::size_t>> siblings;
    for (std::size_t b = 0; b < blocks.size(); ++b)
    {
        if (tagged[b] && blocks[b].level < config.max_level)
            plan.refine.push_back(b);
        const auto h = hierarchy()->block_spacing(b);
        if (protected_blocks[b] && blocks[b].level == config.max_level &&
            *std::max_element(h.begin(), h.end()) > target * (1.0 + 1e-12))
            plan.resolution_limited = true;
        if (blocks[b].level)
        {
            auto parent = blocks[b];
            --parent.level;
            for (auto& i : parent.index)
                i /= 2;
            siblings[parent].push_back(b);
        }
    }
    for (const auto& [parent, children] : siblings)
    {
        if (children.size() != 8)
            continue;
        const int phase = phase_min[children.front()];
        bool coarsen = phase < 2;
        for (auto b : children)
            coarsen = coarsen && !protected_blocks[b] && !tagged[b] && phase_min[b] == phase && phase_max[b] == phase;
        if (coarsen)
            plan.coarsen.push_back(parent);
    }
    return plan;
}

} // namespace SimpleFluid::VoF
