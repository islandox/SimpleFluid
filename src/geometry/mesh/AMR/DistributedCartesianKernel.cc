/** @file DistributedCartesianKernel.cc
 * @brief Native block grouping, Cartesian dyadic splits and detached coordinates.
 */
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using ID = std::uint64_t;
ID mul(ID a, ID b)
{
    if (b && a > std::numeric_limits<ID>::max() / b)
        throw std::overflow_error("Cartesian recipe index overflow.");
    return a * b;
}
ID scale(unsigned level)
{
    if (level >= 63)
        throw std::invalid_argument("Cartesian recipe level exceeds precision.");
    return ID(1) << level;
}
void validate_grid(const AMRGrid& grid)
{
    auto finest = scale(grid.max_level);
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (!grid.root_blocks[a] || grid.block_cells[a] < 2 || grid.block_cells[a] % 2 ||
            grid.block_cells[a] >= std::numeric_limits<unsigned>::max() || !std::isfinite(grid.origin[a]) ||
            !std::isfinite(grid.extent[a]) || grid.extent[a] <= 0)
            throw std::invalid_argument("Invalid detached Cartesian grid.");
        (void) mul(mul(grid.root_blocks[a], grid.block_cells[a]), finest);
    }
}
const CartesianCellAncestry& ancestry(const AMRGrid& grid, const AdaptiveCellKey& key)
{
    auto* value = std::get_if<CartesianCellAncestry>(&key.ancestry);
    if (!value || key.region != grid.region_id || value->level > grid.max_level)
        throw std::invalid_argument("Foreign Cartesian cell ancestry.");
    for (std::size_t a = 0; a < 3; ++a)
        if (value->index[a] >= mul(mul(grid.root_blocks[a], grid.block_cells[a]), scale(value->level)))
            throw std::invalid_argument("Cartesian cell lies outside root grid.");
    return *value;
}
double coordinate(const AMRGrid& grid, std::size_t axis, ID index, unsigned level)
{
    auto finest = mul(index, scale(grid.max_level - level));
    auto total = mul(mul(grid.root_blocks[axis], grid.block_cells[axis]), scale(grid.max_level));
    return grid.origin[axis] + grid.extent[axis] * (double(finest) / double(total));
}
AMRBlock block(const AMRGrid& grid, const CartesianCellAncestry& cell)
{
    AMRBlock result{cell.level, {}};
    for (std::size_t a = 0; a < 3; ++a)
        result.index[a] = cell.index[a] / grid.block_cells[a];
    return result;
}
std::pair<GeometryPoint, GeometryPoint> root_bounds(const AMRGrid& grid)
{
    GeometryPoint lower = grid.origin, upper{};
    for (std::size_t a = 0; a < 3; ++a)
        upper[a] = coordinate(grid, a, mul(grid.root_blocks[a], grid.block_cells[a]), 0);
    return {lower, upper};
}
std::set<AMRBlock> partner_blocks(const AMRGrid& grid, const AMRBlock& source)
{
    std::set<AMRBlock> result{source};
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (grid.periodic[axis])
        {
            const auto count = mul(grid.root_blocks[axis], scale(source.level));
            auto snapshot = result;
            for (auto partner : snapshot)
                if (partner.index[axis] == 0 || partner.index[axis] + 1 == count)
                {
                    partner.index[axis] = count - 1 - partner.index[axis];
                    result.insert(partner);
                }
        }
    return result;
}
std::shared_ptr<const NativeFragmentDescriptor> recipe(
    const AMRGrid& grid, const AMRBlock& block, NativeFragmentCache& cache)
{
    const auto& cells = grid.block_cells;
    auto topology = std::make_shared<const RectilinearTopology>(
        std::make_shared<const OrthoMeshTopo>(unsigned(cells[0]), unsigned(cells[1]), unsigned(cells[2]), false, false,
            false, OrthoMeshTopo::BoundaryNames{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}));
    Vec3D<ArrReal> edges;
    for (std::size_t a = 0; a < 3; ++a)
        for (ID i = 0; i <= cells[a]; ++i)
            edges[a].push_back(coordinate(grid, a, mul(block.index[a], cells[a]) + i, block.level));
    auto name = "block_" + std::to_string(block.level) + "_" + std::to_string(block.index[0]) + "_" +
                std::to_string(block.index[1]) + "_" + std::to_string(block.index[2]);
    return extract_native_fragment(MultiRegionMesh::Region(CartesianRegion(
                                       name, topology, std::make_shared<const RectilinearGeometry>(std::move(edges)))),
        cache);
}
ID local_cell(const AMRGrid& grid, const CartesianCellAncestry& cell)
{
    return cell.index[0] % grid.block_cells[0] +
           grid.block_cells[0] *
               (cell.index[1] % grid.block_cells[1] + grid.block_cells[1] * (cell.index[2] % grid.block_cells[2]));
}
class CartesianKernel final : public AdaptiveProviderKernel
{
    AMRGrid grid_;

public:
    explicit CartesianKernel(AMRGrid grid) : grid_(std::move(grid)) { validate_grid(grid_); }
    void validate(const AdaptiveLeaf& leaf) const override
    {
        const auto& cell = ancestry(grid_, leaf.key);
        if (leaf.provider != AdaptiveProviderKind::Cartesian || leaf.material != grid_.material_id ||
            leaf.domain.empty() || !leaf.native_recipe || leaf.native_cell != local_cell(grid_, cell))
            throw std::invalid_argument("Cartesian leaf membership or native ordinal differs from typed ancestry.");
        NativeFragmentCache cache;
        auto expected = recipe(grid_, block(grid_, cell), cache);
        if (leaf.native_recipe->kind != expected->kind || leaf.native_recipe->view != expected->view ||
            leaf.native_recipe->parts.size() != expected->parts.size())
            throw std::invalid_argument("Cartesian native block recipe differs from typed ancestry.");
        for (std::size_t i = 0; i < expected->parts.size(); ++i)
            if (!leaf.native_recipe->parts[i] || !expected->parts[i] ||
                *leaf.native_recipe->parts[i] != *expected->parts[i])
                throw std::invalid_argument("Cartesian native block axes or topology differ from typed ancestry.");
        if (leaf.geometry && !leaf.geometry->equivalent(*geometry(leaf)))
            throw std::invalid_argument("Cartesian retained certificate differs from typed ancestry.");
    }
    AdaptiveBytes descriptor() const override
    {
        adaptive_detail::Writer w;
        w.word(0x5346434152544b31ULL);
        for (std::size_t a = 0; a < 3; ++a)
        {
            w.word(grid_.root_blocks[a]);
            w.word(grid_.block_cells[a]);
            w.word(std::bit_cast<ID>(grid_.origin[a]));
            w.word(std::bit_cast<ID>(grid_.extent[a]));
            w.word(grid_.periodic[a]);
        }
        w.word(grid_.max_level);
        w.word(grid_.region_id);
        w.word(grid_.material_id);
        return std::move(w.data);
    }
    AdaptiveTrace traces(const AdaptiveLeaf& leaf) const override
    {
        const auto& cell = ancestry(grid_, leaf.key);
        return {leaf.key, {{descriptor(), cell.level, AdaptiveSplit::Isotropic, {}}}};
    }
    AdaptiveBytes contact_point(const AdaptiveLeaf& leaf, std::span<const char> point) const override
    {
        (void) ancestry(grid_, leaf.key);
        const auto [lower, upper] = root_bounds(grid_);
        return exact_periodic_point(point, lower, upper, grid_.periodic);
    }
    std::vector<AdaptiveCellKey> matching_partners(const AdaptiveLeaf& leaf) const override
    {
        const auto& cell = ancestry(grid_, leaf.key);
        auto source = block(grid_, cell);
        std::vector<AdaptiveCellKey> result;
        for (const auto& partner : partner_blocks(grid_, source))
            if (partner != source)
            {
                CartesianCellAncestry next{cell.level, {}};
                for (std::size_t a = 0; a < 3; ++a)
                    next.index[a] = partner.index[a] * grid_.block_cells[a] + cell.index[a] % grid_.block_cells[a];
                result.push_back({leaf.key.region, next});
            }
        return result;
    }
    AdaptiveCellMetrics metrics(const AdaptiveLeaf& leaf) const override
    {
        (void) ancestry(grid_, leaf.key);
        if (!leaf.native_recipe)
            throw std::invalid_argument("Native Cartesian metrics need a block recipe.");
        NativeFragmentCache cache;
        auto fragment = materialize_native_fragment(*leaf.native_recipe, cache);
        const auto& region = std::get<CartesianRegion>(fragment);
        auto center = region.geometry().cell_centroid(leaf.native_cell);
        return {region.geometry().cell_volume(leaf.native_cell), {center.x, center.y, center.z}};
    }
    ConvexGeometryPtr geometry(const AdaptiveLeaf& leaf) const override
    {
        const auto& cell = ancestry(grid_, leaf.key);
        GeometryPoint lower{}, upper{};
        for (std::size_t a = 0; a < 3; ++a)
        {
            lower[a] = coordinate(grid_, a, cell.index[a], cell.level);
            upper[a] = coordinate(grid_, a, cell.index[a] + 1, cell.level);
        }
        return ConvexCellGeometry::box(lower, upper);
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override
    {
        const auto& cell = ancestry(grid_, leaf.key);
        auto shape = geometry(leaf);
        std::vector<AdaptiveSurface> result;
        for (std::size_t f = 0; f < shape->num_faces(); ++f)
        {
            AdaptiveSurface surface;
            surface.geometry = shape->face_geometry(f);
            surface.require_native_whole_tile=true;surface.native_whole_geometry=surface.geometry;
            auto normal = surface.geometry->area_vector();
            std::size_t axis = 0;
            while (axis < 3 && normal.component(axis) == 0)
                ++axis;
            if (axis == 3)
                throw std::invalid_argument("Zero Cartesian face normal.");
            const bool upper = normal.component(axis) > 0;
            surface.orientation = upper ? 1 : -1;
            const auto& counts = grid_.block_cells;
            RectilinearTopology native(
                std::make_shared<const OrthoMeshTopo>(unsigned(counts[0]), unsigned(counts[1]), unsigned(counts[2]),
                    false, false, false, OrthoMeshTopo::BoundaryNames{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}));
            std::array<unsigned, 3> location{unsigned(cell.index[0] % counts[0]), unsigned(cell.index[1] % counts[1]),
                unsigned(cell.index[2] % counts[2])};
            location[axis] += upper;
            surface.native_face =
                native.indexer().face_ordinal({location[0], location[1], location[2], std::uint8_t(axis)});

            const auto boundary = cell.index[axis] + upper;
            const auto domain_cells = mul(mul(grid_.root_blocks[axis], grid_.block_cells[axis]), scale(cell.level));
            ID plane = grid_.periodic[axis] && boundary == domain_cells ? 0 : boundary;
            if (grid_.periodic[axis] && boundary == domain_cells)
            {
                const auto [lower, root_upper] = root_bounds(grid_);
                auto from = lower;
                from[axis] = root_upper[axis];
                surface.authority_image = exact_displacement(from, lower);
            }
            unsigned level = cell.level;
            while (level && plane % 2 == 0)
            {
                plane /= 2;
                --level;
            }
            adaptive_detail::Writer w;
            w.word(0x5346434152543631ULL);
            w.bytes(leaf.domain);
            w.word(grid_.region_id);
            w.word(axis);
            w.word(level);
            w.word(plane);
            for (std::size_t a = 0; a < 3; ++a)
                if (a != axis)
                    w.word(cell.index[a] / scale(cell.level - level));
            surface.key = std::move(w.data);
            surface.exterior = !grid_.periodic[axis] && (boundary == 0 || boundary == domain_cells);
            if (surface.exterior)
                surface.boundary = std::string(1, char('x' + axis)) + (upper ? "max" : "min");
            result.push_back(std::move(surface));
        }
        return result;
    }
    std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const override
    {
        NativeFragmentCache cache;
        std::map<AMRBlock, std::shared_ptr<const NativeFragmentDescriptor>> recipes;
        std::vector<AdaptiveLeaf> result;
        result.reserve(leaves.size());
        for (const auto& source : leaves)
        {
            auto leaf = source;
            const auto& cell = ancestry(grid_, leaf.key);
            auto key = block(grid_, cell);
            auto [it, inserted] = recipes.try_emplace(key);
            if (inserted)
                it->second = recipe(grid_, key, cache);
            leaf.native_recipe = it->second;
            leaf.native_cell = local_cell(grid_, cell);
            leaf.geometry.reset();
            leaf.surfaces.clear();
            result.push_back(std::move(leaf));
        }
        return result;
    }
    std::vector<AdaptiveCellKey> refinement_group(const AdaptiveCellKey& key, AdaptiveSplit split) const override
    {
        if (split != AdaptiveSplit::Isotropic)
            throw std::invalid_argument("Cartesian blocks require isotropic refinement.");
        const auto& cell = ancestry(grid_, key);
        const auto parent_block = block(grid_, cell);
        std::vector<AdaptiveCellKey> result;
        for (const auto& grouped_block : partner_blocks(grid_, parent_block))
            for (ID k = 0; k < grid_.block_cells[2]; ++k)
                for (ID j = 0; j < grid_.block_cells[1]; ++j)
                    for (ID i = 0; i < grid_.block_cells[0]; ++i)
                        result.push_back({key.region,
                            CartesianCellAncestry{cell.level, {grouped_block.index[0] * grid_.block_cells[0] + i,
                                                                  grouped_block.index[1] * grid_.block_cells[1] + j,
                                                                  grouped_block.index[2] * grid_.block_cells[2] + k}}});
        return result;
    }
    std::vector<AdaptiveBytes> coarsening_group(const AdaptiveFamily& family) const override
    {
        std::vector<AdaptiveBytes> result;
        for (const auto& key : refinement_group(family.parent.key, AdaptiveSplit::Isotropic))
            result.push_back(adaptive_family_key(key, AdaptiveSplit::Isotropic));
        return result;
    }
    AdaptiveSplitResult split(const AdaptiveLeaf& parent, AdaptiveSplit kind) const override
    {
        const auto& cell = ancestry(grid_, parent.key);
        if (kind != AdaptiveSplit::Isotropic || parent.provider != AdaptiveProviderKind::Cartesian ||
            cell.level >= grid_.max_level || parent.revision == std::numeric_limits<ID>::max())
            throw std::invalid_argument("Unsupported Cartesian cell split.");
        AdaptiveSplitResult result;
        result.family.key = adaptive_family_key(parent.key, kind);
        result.family.parent = parent;
        result.family.split = kind;
        result.family.revision = parent.revision;
        for (ID octant = 0; octant < 8; ++octant)
        {
            auto child = parent;
            CartesianCellAncestry next{cell.level + 1, {}};
            for (std::size_t a = 0; a < 3; ++a)
                next.index[a] = 2 * cell.index[a] + ((octant >> a) & 1);
            child.key.ancestry = next;
            child.revision++;
            child.parent_family = result.family.key;
            child.geometry.reset();
            child.surfaces.clear();
            child.native_recipe.reset();
            child.native_cell = local_cell(grid_, next);
            result.family.children.push_back(child.key);
            result.children.push_back(std::move(child));
        }
        result.children = repack(result.children);
        result.family.anchor =
            encode_adaptive_key(*std::min_element(result.family.children.begin(), result.family.children.end()));
        return result;
    }
};
} // namespace
std::shared_ptr<const AdaptiveProviderKernel> cartesian_adaptive_kernel(AMRGrid grid)
{
    return std::make_shared<const CartesianKernel>(std::move(grid));
}
AdaptiveLeaf cartesian_adaptive_leaf(const AMRGrid& grid, AdaptiveCellKey key, AdaptiveBytes domain)
{
    validate_grid(grid);
    if (domain.empty())
        throw std::invalid_argument("Cartesian connected domain is required.");
    const auto& cell = ancestry(grid, key);
    NativeFragmentCache cache;
    AdaptiveLeaf leaf;
    leaf.key = key;
    leaf.domain = std::move(domain);
    leaf.provider = AdaptiveProviderKind::Cartesian;
    leaf.material = grid.material_id;
    leaf.native_recipe = recipe(grid, block(grid, cell), cache);
    leaf.native_cell = local_cell(grid, cell);
    if (cell.level)
        leaf.parent_family = adaptive_family_key({leaf.key.region, cell.parent()}, AdaptiveSplit::Isotropic);
    return leaf;
}
} // namespace SimpleFluid::Meshes::AMR
