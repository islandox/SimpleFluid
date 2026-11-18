/** @file DistributedCartesianKernel.cc
 * @brief Native block grouping, Cartesian dyadic splits and detached coordinates.
 */
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/PLIC/details/NativeBoxSourceAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/mesh/NativeFragmentAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using ID = std::uint64_t;
std::size_t capacity_add(std::size_t first, std::size_t second)
{
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw std::invalid_argument("Cartesian materialization capacity overflows.");
    return first + second;
}
std::size_t capacity_product(std::size_t count, std::size_t size)
{
    if (size && count > std::numeric_limits<std::size_t>::max() / size)
        throw std::invalid_argument("Cartesian materialization capacity overflows.");
    return count * size;
}
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
std::pair<PLIC::GeometryPoint, PLIC::GeometryPoint> root_bounds(const AMRGrid& grid)
{
    PLIC::GeometryPoint lower = grid.origin, upper{};
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
/** Compare the existing homogeneous native descriptor without constructing an
 * expected provider, packet, name or axis array. Never trust a supplied recipe
 * as the expected geometry of its own cell. */
struct CartesianRecipeCheck
{
    std::span<const char> bytes;
    template<class T> void value(const T& expected)
    {
        if (bytes.size() < sizeof(T) || std::memcmp(bytes.data(), &expected, sizeof(T)))
            throw std::invalid_argument("Cartesian native recipe differs from typed source semantics.");
        bytes = bytes.subspan(sizeof(T));
    }
    void text(std::string_view expected)
    {
        value(std::uint64_t(expected.size()));
        if (bytes.size() < expected.size() || !std::equal(expected.begin(), expected.end(), bytes.begin()))
            throw std::invalid_argument("Cartesian native recipe label differs from typed source semantics.");
        bytes = bytes.subspan(expected.size());
    }
    void done() const
    {
        if (!bytes.empty())
            throw std::invalid_argument("Cartesian native recipe has trailing bytes.");
    }
};
constexpr std::array<std::string_view, 6> cartesian_names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
struct CartesianBlockName
{
    std::array<char, 128> bytes{};
    std::size_t size = 0;
    explicit CartesianBlockName(const AMRBlock& block)
    {
        constexpr std::string_view prefix = "block_";
        std::copy(prefix.begin(), prefix.end(), bytes.begin());
        char* cursor = bytes.data() + prefix.size();
        const auto append = [&](std::uint64_t value, bool separator)
        {
            if (separator) *cursor++ = '_';
            const auto encoded = std::to_chars(cursor, bytes.data() + bytes.size(), value);
            if (encoded.ec != std::errc{})
                throw std::invalid_argument("Cartesian block name exceeds its fixed representation.");
            cursor = encoded.ptr;
        };
        append(block.level, false);
        for (const auto index : block.index) append(index, true);
        size = static_cast<std::size_t>(cursor - bytes.data());
    }
    std::string_view view() const noexcept { return {bytes.data(), size}; }
};
class CartesianKernel final : public AdaptiveProviderKernel
{
    AMRGrid grid_;
    void write_surface_key(adaptive_detail::Writer& writer, const AdaptiveLeaf& leaf,
        const CartesianCellAncestry& cell, std::size_t axis, ID plane) const
    {
        unsigned level = cell.level;
        while (level && plane % 2 == 0)
        {
            plane /= 2;
            --level;
        }
        writer.word(0x5346434152543631ULL);
        writer.bytes(leaf.domain);
        writer.word(grid_.region_id);
        writer.word(axis);
        writer.word(level);
        writer.word(plane);
        for (std::size_t a = 0; a < 3; ++a)
            if (a != axis)
                writer.word(cell.index[a] / scale(cell.level - level));
    }

public:
    explicit CartesianKernel(AMRGrid grid) : grid_(std::move(grid)) { validate_grid(grid_); }
    void visit_native_publication_children(const geometry_detail::NativeConstructorOwnerSink&,
        adaptive_detail::NativeKernelAuthority::IdentityVisit,
        adaptive_detail::NativeKernelAuthority::GeometryVisit) const {}
    bool native_cartesian_incidence_grid(AMRGrid& output) const noexcept override
    {
        output = grid_;
        return true;
    }
    bool supports_native_source(const AdaptiveLeaf& leaf) const noexcept
    {
        return leaf.provider == AdaptiveProviderKind::Cartesian && !leaf.geometry && leaf.surfaces.empty() &&
               !leaf.factor_recipe && !leaf.join_recipe;
    }
    static constexpr bool native_source_publication_ready = true;
    std::size_t native_source_face_count(const AdaptiveLeaf&) const noexcept { return 6; }
    AdaptiveCellMetrics validate_native_source(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
        const geometry_detail::NativeConstructionAdmission& admission,
        std::span<adaptive_detail::NativeKernelSourceFace> routes, void* context = nullptr,
        adaptive_detail::NativeKernelAuthority::SupportVisit support = nullptr) const
    {
        const auto& cell = ancestry(grid_, leaf.key);
        if (routes.size() != 6 || leaf.provider != AdaptiveProviderKind::Cartesian || leaf.material != grid_.material_id ||
            leaf.domain.empty() || !leaf.native_recipe || leaf.native_cell != local_cell(grid_, cell) ||
            leaf.geometry || !leaf.surfaces.empty() || leaf.factor_recipe || leaf.join_recipe)
            throw std::invalid_argument("Cartesian source requires its original procedural leaf and typed ancestry.");
        const auto* region = std::get_if<CartesianRegion>(&fragment.region());
        if (!region || !fragment.allocation() || !fragment.allocation()->authentic_owner(fragment.allocation()))
            throw std::invalid_argument("Cartesian source lacks its admitted immutable provider.");
        const auto source_block = block(grid_, cell);
        const CartesianBlockName name(source_block);
        const auto& descriptor = *leaf.native_recipe;
        if (descriptor.kind != NativeFragmentDescriptor::Kind::Cartesian || descriptor.parts.size() != 4)
            throw std::invalid_argument("Cartesian source has a foreign fragment schema.");
        for (const auto& part : descriptor.parts)
            if (!part) throw std::invalid_argument("Cartesian source has a null fragment part.");
        CartesianRecipeCheck topology{*descriptor.parts[0]};
        for (const auto count : grid_.block_cells) topology.value(std::uint64_t(count));
        for (unsigned axis = 0; axis < 3; ++axis) topology.value(std::uint8_t(0));
        for (const auto label : cartesian_names) topology.text(label);
        topology.done();
        CartesianRecipeCheck view{descriptor.view}; view.text(name.view()); view.done();
        if (region->name() != name.view())
            throw std::invalid_argument("Cartesian source provider has a foreign block name.");
        const auto layout = region->layout();
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (layout.extents[axis] != grid_.block_cells[axis] || layout.periodic[axis] ||
                region->geometry().cell_edges()[axis].size() != grid_.block_cells[axis] + 1)
                throw std::invalid_argument("Cartesian source provider has a foreign block layout.");
            CartesianRecipeCheck coordinates{*descriptor.parts[axis + 1]};
            coordinates.value(std::uint64_t(grid_.block_cells[axis] + 1));
            for (ID i = 0; i <= grid_.block_cells[axis]; ++i)
            {
                const auto expected = coordinate(grid_, axis, mul(source_block.index[axis], grid_.block_cells[axis]) + i,
                    source_block.level);
                coordinates.value(expected);
                if (std::bit_cast<ID>(region->geometry().cell_edges()[axis][i]) != std::bit_cast<ID>(expected))
                    throw std::invalid_argument("Cartesian provider axes differ from original source expressions.");
            }
            coordinates.done();
        }
        for (int boundary = 0; boundary < 6; ++boundary)
            if (region->topology().boundary_batch_name(boundary) != cartesian_names[boundary])
                throw std::invalid_argument("Cartesian provider boundary aliases differ from original source.");
        const auto center = region->geometry().cell_centroid(leaf.native_cell);
        const AdaptiveCellMetrics metrics{region->geometry().cell_volume(leaf.native_cell), {center.x, center.y, center.z}};
        const auto bounds = geometry_bounds(leaf);
        struct Verification
        {
            const CartesianRegion& region;
            const AdaptiveLeaf& leaf;
            const geometry_detail::NativeConstructionAdmission& admission;
            std::span<adaptive_detail::NativeKernelSourceFace> routes;
            AdaptiveCartesianGeometryBounds bounds;
        } verification{*region, leaf, admission, routes, bounds};
        adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::CellGeometryRequest);
        PLIC::details::NativeBoxSourceAccess::verify(bounds.lower, bounds.upper, admission, &verification,
            [](void* raw, const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>& canonical)
            {
                auto& v = *static_cast<Verification*>(raw);
                const auto& topology = v.region.topology();
                const auto& indexer = topology.indexer();
                const auto location = indexer.cell_id(v.leaf.native_cell);
                std::array<unsigned, 3> base{location.i, location.j, location.k};
                if (canonical.num_faces() != 6 || topology.cell_faces(v.leaf.native_cell).size() != 6)
                    throw std::invalid_argument("Cartesian source has an incomplete original convex shell.");
                const auto nodes = topology.cell_nodes(v.leaf.native_cell);
                if (nodes.size() != 8)
                    throw std::invalid_argument("Cartesian source has an incomplete original corner set.");
                for (std::size_t corner = 0; corner < 8; ++corner)
                {
                    const auto node = indexer.node_ordinal({base[0] + unsigned(corner & 1),
                        base[1] + unsigned((corner >> 1) & 1), base[2] + unsigned((corner >> 2) & 1)});
                    if (std::count(nodes.begin(), nodes.end(), node) != 1)
                        throw std::invalid_argument("Cartesian source corner route is missing or repeated.");
                    const auto point = v.region.geometry().node_coordinates(node);
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        const auto expected = (corner >> axis) & 1 ? v.bounds.upper[axis] : v.bounds.lower[axis];
                        if (std::bit_cast<ID>(point.component(axis)) != std::bit_cast<ID>(expected))
                            throw std::invalid_argument("Cartesian source corner differs from original source bits.");
                    }
                }
                for (std::size_t surface = 0; surface < 6; ++surface)
                {
                    auto face_location = base;
                    const auto axis = surface / 2;
                    face_location[axis] += surface % 2;
                    const auto face = indexer.face_ordinal({face_location[0], face_location[1], face_location[2],
                        static_cast<std::uint8_t>(axis)});
                    const auto faces = topology.cell_faces(v.leaf.native_cell);
                    if (std::count(faces.begin(), faces.end(), face) != 1)
                        throw std::invalid_argument("Cartesian source face route is missing or repeated.");
                    std::array<PLIC::GeometryPoint, 4> points;
                    std::size_t count = 0;
                    for (const auto node : topology.face_nodes(face))
                    {
                        if (count >= points.size())
                            throw std::invalid_argument("Cartesian native source face has excess corners.");
                        const auto p = v.region.geometry().node_coordinates(node);
                        points[count++] = {p.x, p.y, p.z};
                    }
                    if (count != points.size())
                        throw std::invalid_argument("Cartesian native source face has missing corners.");
                    auto native = PLIC::details::NativePlanarOwnerAccess::make(points, v.admission);
                    const auto normal = v.region.geometry().face_area_vector(face);
                    const int outward = topology.owner_cell(face) == v.leaf.native_cell ? 1 : -1;
                    native = PLIC::details::NativePlanarOwnerAccess::oriented_like(native,
                        {outward * normal.x, outward * normal.y, outward * normal.z}, v.admission);
                    const auto source = PLIC::details::NativePlanarOwnerAccess::cell_face(canonical, surface, v.admission);
                    if (!native->contains(*source) || !source->contains(*native) || native->oppositely_oriented(*source))
                        throw std::invalid_argument("Native Cartesian convex boundary differs from original source.");
                    v.routes[surface] = {surface, face, outward};
                }
            });
        if (support)
        {
            const auto scratch = geometry_detail::native_constructor_storage(admission);
            adaptive_detail::Writer writer;
            scratch.owner->reserve(writer.data, capacity_add(64, leaf.domain.size()));
            for (std::size_t surface = 0; surface < routes.size(); ++surface)
            {
                const auto axis = surface / 2;
                const auto boundary = cell.index[axis] + surface % 2;
                const auto domain_cells = mul(mul(grid_.root_blocks[axis], grid_.block_cells[axis]), scale(cell.level));
                const bool wrap = grid_.periodic[axis] && boundary == domain_cells;
                const bool exterior = !grid_.periodic[axis] && (boundary == 0 || boundary == domain_cells);
                writer.data.clear();
                write_surface_key(writer, leaf, cell, axis, wrap ? 0 : boundary);
                support(context, surface, writer.data, exterior);
            }
        }
        return metrics;
    }
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
    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf& leaf) const override
    {
        (void) ancestry(grid_, leaf.key);
        const bool periodic = grid_.periodic[0] || grid_.periodic[1] || grid_.periodic[2];
        return {
            periodic ? AdaptiveContactNormalization::PeriodicCartesian : AdaptiveContactNormalization::PhysicalIdentity,
            false, false};
    }
    std::vector<AdaptiveContactImage> closed_contact_images(const AdaptiveLeaf& leaf) const override
    {
        const auto& cell = ancestry(grid_, leaf.key);
        const auto [root_lower, root_upper] = root_bounds(grid_);
        AdaptiveContactImage initial;
        initial.logical.emplace();
        initial.logical->frame = descriptor();
        initial.logical->exact_embedding = true;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            initial.bounds.lower[axis] = coordinate(grid_, axis, cell.index[axis], cell.level);
            initial.bounds.upper[axis] = coordinate(grid_, axis, cell.index[axis] + 1, cell.level);
            initial.logical->lower[axis] = mul(cell.index[axis], scale(grid_.max_level - cell.level));
            initial.logical->upper[axis] = mul(cell.index[axis] + 1, scale(grid_.max_level - cell.level));
            initial.logical->period[axis] =
                mul(mul(grid_.root_blocks[axis], grid_.block_cells[axis]), scale(grid_.max_level));
            int exponent = 0;
            const auto total = initial.logical->period[axis];
            // This sufficient condition proves that every coordinate operation
            // is an exact binary scaling of a representable integer. Other
            // admitted grids retain their exact ancestry metadata, but use the
            // physical closed enclosure for separation.
            initial.logical->exact_embedding = initial.logical->exact_embedding && grid_.origin[axis] == 0 &&
                                               std::frexp(grid_.extent[axis], &exponent) == 0.5 &&
                                               std::has_single_bit(total) && total <= (ID(1) << 52) &&
                                               std::isnormal(grid_.extent[axis] / double(total));
        }
        if (!initial.logical->exact_embedding)
        {
            const auto bounds = geometry(leaf)->closed_contact_bounds();
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                initial.bounds.lower[axis] = bounds[axis][0];
                initial.bounds.upper[axis] = bounds[axis][1];
            }
        }
        std::vector<AdaptiveContactImage> result{std::move(initial)};
        // contact_point maps the upper seam to the lower seam. Retain the full
        // physical box (safe overinclusion), plus every seam/corner singleton
        // image. In particular, no seam endpoint is lost by a half-open box.
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (grid_.periodic[axis] && result.front().logical->upper[axis] == result.front().logical->period[axis])
            {
                const auto count = result.size();
                for (std::size_t i = 0; i < count; ++i)
                {
                    auto image = result[i];
                    image.bounds.lower[axis] = image.bounds.upper[axis] = root_lower[axis];
                    image.logical->lower[axis] = image.logical->upper[axis] = 0;
                    auto physical = root_lower, canonical = root_lower;
                    for (std::size_t a = 0; a < 3; ++a)
                        if (image.logical->lower[a] == 0 && image.logical->upper[a] == 0 && grid_.periodic[a])
                            physical[a] = root_upper[a];
                    image.physical_to_contact = PLIC::displacement<ArithmeticMode::Ordinary>(physical, canonical);
                    result.push_back(std::move(image));
                }
            }
        return result;
    }
    PLIC::PointIdentity<ArithmeticMode::Ordinary> contact_point(
        const AdaptiveLeaf& leaf, const PLIC::PointIdentity<ArithmeticMode::Ordinary>& point) const override
    {
        (void) ancestry(grid_, leaf.key);
        const auto [lower, upper] = root_bounds(grid_);
        return PLIC::periodic_point<ArithmeticMode::Ordinary>(point, lower, upper, grid_.periodic);
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
    bool has_native_cartesian_geometry() const noexcept override { return true; }
    std::optional<AdaptiveCartesianGeometryBounds> native_cartesian_geometry_bounds(
        const AdaptiveLeaf& leaf) const override
    {
        return geometry_bounds(leaf);
    }
    AdaptiveCartesianGeometryBounds geometry_bounds(const AdaptiveLeaf& leaf) const
    {
        const auto& cell = ancestry(grid_, leaf.key);
        AdaptiveCartesianGeometryBounds result;
        for (std::size_t a = 0; a < 3; ++a)
        {
            result.lower[a] = coordinate(grid_, a, cell.index[a], cell.level);
            result.upper[a] = coordinate(grid_, a, cell.index[a] + 1, cell.level);
        }
        return result;
    }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        const auto bounds = geometry_bounds(leaf);
        return PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box(bounds.lower, bounds.upper);
    }
    std::vector<AdaptiveSurface> surfaces_from_shape(const AdaptiveLeaf& leaf,
        const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& shape, bool bounded_capacity = false) const
    {
        const auto& cell = ancestry(grid_, leaf.key);
        std::vector<AdaptiveSurface> result;
        result.reserve(shape->num_faces());
        for (std::size_t f = 0; f < shape->num_faces(); ++f)
        {
            AdaptiveSurface surface;
            surface.geometry = shape->face_geometry(f);
            surface.require_native_whole_tile = true;
            surface.native_whole_geometry = surface.geometry;
            auto normal = surface.geometry->area_vector();
            std::size_t axis = 0;
            while (axis < 3 && normal.component(axis) == 0)
                ++axis;
            if (axis == 3)
                throw std::invalid_argument("Zero Cartesian face normal.");
            const bool upper = normal.component(axis) > 0;
            surface.orientation = upper ? 1 : -1;
            const auto& counts = grid_.block_cells;
            const OrthogonalIndexer native(
                unsigned(counts[0]), unsigned(counts[1]), unsigned(counts[2]), false, false, false);
            std::array<unsigned, 3> location{unsigned(cell.index[0] % counts[0]), unsigned(cell.index[1] % counts[1]),
                unsigned(cell.index[2] % counts[2])};
            location[axis] += upper;
            surface.native_face = native.face_ordinal({location[0], location[1], location[2], std::uint8_t(axis)});

            const auto boundary = cell.index[axis] + upper;
            const auto domain_cells = mul(mul(grid_.root_blocks[axis], grid_.block_cells[axis]), scale(cell.level));
            ID plane = grid_.periodic[axis] && boundary == domain_cells ? 0 : boundary;
            if (grid_.periodic[axis] && boundary == domain_cells)
            {
                const auto [lower, root_upper] = root_bounds(grid_);
                auto from = lower;
                from[axis] = root_upper[axis];
                surface.authority_image = PLIC::displacement<ArithmeticMode::Ordinary>(from, lower);
            }
            adaptive_detail::Writer w;
            const auto key_capacity = capacity_add(64, leaf.domain.size());
            w.data.reserve(key_capacity);
            if (bounded_capacity && w.data.capacity() != key_capacity)
                throw std::invalid_argument("Cartesian surface key capacity differs from admission.");
            write_surface_key(w, leaf, cell, axis, plane);
            surface.key = std::move(w.data);
            surface.exterior = !grid_.periodic[axis] && (boundary == 0 || boundary == domain_cells);
            if (surface.exterior)
                surface.boundary = std::string(1, char('x' + axis)) + (upper ? "max" : "min");
            result.push_back(std::move(surface));
        }
        return result;
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override
    {
        return surfaces_from_shape(leaf, geometry(leaf));
    }
    std::size_t compact_native_materialization_bound(const AdaptiveLeaf& leaf) const override
    {
        // This envelope covers only this fixed Cartesian construction. Supplied
        // certificates/surfaces and arbitrary provider metadata keep the legacy
        // expanded guard, even if their public provider enum says Cartesian.
        if (!PLIC::geometry_encoding_capacity_profile_supported() || leaf.geometry || !leaf.surfaces.empty() ||
            !leaf.native_recipe || leaf.factor_recipe || leaf.join_recipe ||
            (grid_.periodic[0] || grid_.periodic[1] || grid_.periodic[2]))
            return 0;
        (void) ancestry(grid_, leaf.key);
        std::size_t bytes = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box_construction_capacity_bound();
        auto add = [&](std::size_t n) { bytes = capacity_add(bytes, n); };
        add(sizeof(AdaptiveLeaf));
        add(leaf.domain.capacity());
        add(leaf.parent_family.capacity());
        add(capacity_product(6, sizeof(AdaptiveSurface)));
        add(capacity_product(6, PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(4)));
        add(capacity_product(6, capacity_add(64, leaf.domain.size())));
        add(capacity_product(6, std::string{}.capacity()));
        // Include borrowed immutable recipe payloads too; the retained ledger
        // will subsequently deduplicate them across leaves and shared parts.
        add(sizeof(NativeFragmentDescriptor));
        add(leaf.native_recipe->view.capacity());
        add(capacity_product(leaf.native_recipe->parts.capacity(), sizeof(std::shared_ptr<const AdaptiveBytes>)));
        for (const auto& part : leaf.native_recipe->parts)
        {
            if (!part)
                throw std::invalid_argument("Null Cartesian native descriptor part.");
            add(sizeof(AdaptiveBytes));
            add(part->capacity());
        }
        return bytes;
    }
    AdaptiveLeaf compact_native_materialize(const AdaptiveLeaf& leaf) const override
    {
        if (!compact_native_materialization_bound(leaf))
            throw std::invalid_argument("Cartesian native record needs legacy admission.");
        auto result = leaf;
        result.geometry = geometry(leaf);
        result.surfaces = surfaces_from_shape(leaf, result.geometry, true);
        return result;
    }
    bool supports_leaf_local_repack() const noexcept override { return true; }
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
bool has_native_cartesian_geometry(const AdaptiveProviderKernel& kernel) noexcept
{
    return kernel.has_native_cartesian_geometry();
}
std::optional<AdaptiveCartesianGeometryBounds> native_cartesian_geometry_bounds(
    const AdaptiveProviderKernel& kernel, const AdaptiveLeaf& leaf)
{
    return kernel.native_cartesian_geometry_bounds(leaf);
}
std::shared_ptr<const AdaptiveProviderKernel> cartesian_adaptive_kernel(AMRGrid grid)
{
    return adaptive_detail::NativeKernelAuthority::create<CartesianKernel>(std::move(grid));
}
std::shared_ptr<const AdaptiveProviderKernel> adaptive_detail::cartesian_incidence_replay_kernel(AMRGrid grid)
{
    return std::make_shared<const CartesianKernel>(std::move(grid));
}
geometry_detail::NativeTrackedOwner<const AdaptiveProviderKernel> adaptive_detail::cartesian_incidence_replay_kernel(
    AMRGrid grid, const geometry_detail::NativeConstructionAdmission& admission)
{
    auto replay = geometry_detail::NativeAllocationEnvelopeAccess::make<CartesianKernel>(admission, std::move(grid));
    return {std::move(replay.owner), std::move(replay.envelope)};
}
std::size_t adaptive_detail::native_cartesian_kernel_payload_bytes() noexcept
{
    return sizeof(CartesianKernel);
}
std::optional<AMRGrid> trusted_cartesian_incidence_grid(const AdaptiveProviderKernel& kernel) noexcept
{
    AMRGrid grid;
    if (!kernel.native_cartesian_incidence_grid(grid))
        return std::nullopt;
    return grid;
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
