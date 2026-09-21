/** @file AnnularSemiStructuredMeshFactory.cc */
#include "geometry/AnnularSemiStructuredMeshFactory.hh"
#include "geometry/BoundaryLayerMeshFactory.hh"
#include "geometry/mesh/FrontalDelaunay2D.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{
namespace
{
constexpr size_t index_limit = std::numeric_limits<unsigned>::max();

size_t checked_product(size_t left, size_t right, size_t limit, const char* description)
{
    if (right != 0 && left > limit / right)
        throw std::overflow_error(std::string("Annular mesh ") + description + " exceeds supported IDs.");
    return left * right;
}

void checked_add(size_t& total, size_t value, const char* description)
{
    if (value > std::numeric_limits<size_t>::max() - total)
        throw std::overflow_error(std::string("Annular mesh ") + description + " exceeds supported IDs.");
    total += value;
}

size_t interval_count(long double length, real_t spacing)
{
    const auto count = std::ceil(length / static_cast<long double>(spacing));
    if (!std::isfinite(count) || count > static_cast<long double>(index_limit - 1))
        throw std::overflow_error("Annular mesh spacing requests too many intervals.");
    return std::max(size_t{1}, static_cast<size_t>(count));
}

/** Subdivide only the single bulk interval, preserving all graded wall widths. */
ArrReal subdivide_bulk(const ArrReal& graded, size_t lower_layers, size_t upper_layers, real_t spacing)
{
    if (graded.size() != lower_layers + upper_layers + 2)
        throw std::logic_error("Annular boundary-layer construction lost the bulk interval.");
    const auto low = graded[lower_layers];
    const auto high = graded[lower_layers + 1];
    const auto bulk = interval_count(static_cast<long double>(high) - low, spacing);
    if (lower_layers > index_limit - upper_layers - 1
        || bulk > index_limit - lower_layers - upper_layers - 1)
        throw std::overflow_error("Annular mesh axis exceeds supported IDs.");
    ArrReal result;
    result.reserve(lower_layers + upper_layers + bulk + 1);
    result.insert(result.end(), graded.begin(), graded.begin() + lower_layers + 1);
    for (size_t i = 1; i < bulk; ++i)
    {
        const auto value = static_cast<real_t>(static_cast<long double>(low)
            + (static_cast<long double>(high) - low) * i / bulk);
        if (!(value > result.back() && value < high))
            throw std::invalid_argument("Annular bulk coordinates cannot be represented distinctly.");
        result.push_back(value);
    }
    result.insert(result.end(), graded.begin() + lower_layers + 1, graded.end());
    return result;
}
} // namespace

void AnnularSemiStructuredMeshFactory::validate_options(const Options& options)
{
    const auto positive = [](real_t value) { return std::isfinite(value) && value > 0; };
    if (!positive(options.inner_radius) || !positive(options.outer_radius)
        || !(options.inner_radius < options.outer_radius)
        || !std::isfinite(options.bottom) || !std::isfinite(options.top)
        || !(options.bottom < options.top)
        || !std::isfinite(options.top - options.bottom))
        throw std::invalid_argument("Annular mesh requires 0 < inner_radius < outer_radius and finite bottom < top.");
    if (!positive(options.xy_spacing) || !positive(options.z_spacing)
        || !positive(options.first_layer_height) || !std::isfinite(options.growth_ratio)
        || options.growth_ratio < 1)
        throw std::invalid_argument("Annular mesh spacing and first layer height must be positive; growth ratio must be >= 1.");
    if (options.wall_layers > (index_limit - 2) / 2)
        throw std::overflow_error("Annular mesh boundary-layer count exceeds supported IDs.");
    for (const auto count : {options.inner_angular_cells, options.outer_angular_cells})
    {
        if (count != 0 && count < 8)
            throw std::invalid_argument("Annular angular counts must be zero (automatic) or at least eight.");
        if (count > index_limit - 1)
            throw std::overflow_error("Annular angular count exceeds supported IDs.");
    }
    const auto area = std::numbers::pi_v<long double>
        * (static_cast<long double>(options.outer_radius) - options.inner_radius)
        * (static_cast<long double>(options.outer_radius) + options.inner_radius);
    if (!std::isfinite(area) || area > std::numeric_limits<real_t>::max()
        || static_cast<real_t>(area) <= 0)
        throw std::invalid_argument("Annular cross-section area must be finite and representable.");
}

AnnularSemiStructuredMeshFactory::AnnularSemiStructuredMeshFactory(Options options)
    : d_options(std::move(options))
{
    validate_options(d_options);
}

auto AnnularSemiStructuredMeshFactory::plan_layout() const -> Layout
{
    const auto& o = d_options;
    auto angular = o.outer_angular_cells;
    if (!angular)
    {
        angular = std::max(size_t{8}, interval_count(
            2 * std::numbers::pi_v<long double> * o.outer_radius, o.xy_spacing));
        if (angular % 2) ++angular;
    }
    const auto half_angle = std::numbers::pi_v<real_t> / angular;
    const auto outer_apothem = o.outer_radius * std::cos(half_angle);
    if (!(outer_apothem > o.inner_radius))
        throw std::invalid_argument("Annular outer angular resolution is too coarse for a polygon outside the inner circle.");

    using Cylindrical = Meshes::OrthogonalCylindrial3D;
    // R temporarily stores polygon apothems. The existing factory therefore
    // supplies the exact normal wall widths without a second grading algorithm.
    Cylindrical axes({{{o.inner_radius, outer_apothem},
        {0, std::numbers::pi_v<real_t>, 2 * std::numbers::pi_v<real_t>}, {o.bottom, o.top}}});
    Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec> specs;
    const auto layers = [&](bool selected) { return selected ? o.wall_layers : size_t{0}; };
    const auto add = [&](const char* name, bool selected)
    {
        if (layers(selected)) specs.push_back({name, o.wall_layers, o.first_layer_height, o.growth_ratio});
    };
    add("rmin", o.refine_inner);
    add("rmax", o.refine_outer);
    add("zmin", o.refine_bottom);
    add("zmax", o.refine_top);
    BoundaryLayerMeshFactory(specs).build(axes);

    Layout layout;
    const auto& radial = axes.cell_edges()[Cylindrical::R];
    const auto& axial = axes.cell_edges()[Cylindrical::AXIAL];
    auto& count = layout.counts;
    count.angular_cells = angular;
    count.outer_angular_cells = angular;
    count.radial_cells = layers(o.refine_inner) + layers(o.refine_outer)
        + interval_count(static_cast<long double>(radial[layers(o.refine_inner) + 1])
            - radial[layers(o.refine_inner)], o.xy_spacing);
    count.axial_cells = layers(o.refine_bottom) + layers(o.refine_top)
        + interval_count(static_cast<long double>(axial[layers(o.refine_bottom) + 1])
            - axial[layers(o.refine_bottom)], o.z_spacing);
    if (count.axial_cells >= index_limit)
        throw std::overflow_error("Annular axial node count exceeds supported IDs.");
    count.coarse_xy_cells = checked_product(angular, count.radial_cells, index_limit, "XY sector count");
    layout.radial_apothems = subdivide_bulk(radial,
        layers(o.refine_inner), layers(o.refine_outer), o.xy_spacing);
    layout.z_edges = subdivide_bulk(axial,
        layers(o.refine_bottom), layers(o.refine_top), o.z_spacing);

    // Keep only the structured wall fronts. Bulk guide rings do not become
    // nodes: FrontalDelaunay2D supplies the complete interior point placement.
    const auto inner_layers = layers(o.refine_inner);
    const auto outer_first = count.radial_cells - layers(o.refine_outer);
    auto inner_angular = o.inner_angular_cells;
    if (!inner_angular)
    {
        // The inner polygon circumscribes its circle. Size from the largest
        // ring in that stack, including its circumradius correction, so its
        // bulk-facing edges also respect the requested circumferential size.
        const long double apothem = layout.radial_apothems[inner_layers];
        inner_angular = std::max(size_t{8}, interval_count(
            2 * std::numbers::pi_v<long double> * apothem, o.xy_spacing));
        if (inner_angular % 2) ++inner_angular;
        while (2 * std::numbers::pi_v<long double> * apothem /
               (inner_angular * std::cos(std::numbers::pi_v<long double> / inner_angular)) > o.xy_spacing)
        {
            if (inner_angular > index_limit - 3)
                throw std::overflow_error("Annular inner angular count exceeds supported IDs.");
            inner_angular += 2;
        }
    }
    count.inner_angular_cells = inner_angular;
    auto wall_xy = checked_product(inner_angular, inner_layers, index_limit, "inner wall XY cell count");
    checked_add(wall_xy, checked_product(angular, layers(o.refine_outer), index_limit,
        "outer wall XY cell count"), "wall XY cell count");
    auto wall_nodes = checked_product(inner_angular, inner_layers + 1, index_limit, "inner wall XY node count");
    checked_add(wall_nodes, checked_product(angular, layers(o.refine_outer) + 1, index_limit,
        "outer wall XY node count"), "wall XY node count");
    if (wall_xy > index_limit || wall_nodes > index_limit)
        throw std::overflow_error("Annular wall topology exceeds supported XY IDs.");
    layout.xy_nodes.reserve(wall_nodes);
    std::map<size_t, unsigned> ring_start;
    const auto ring_count = [&](size_t ring) { return ring <= inner_layers ? inner_angular : angular; };
    const auto add_ring = [&](size_t ring)
    {
        const auto n = ring_count(ring);
        ring_start.emplace(ring, static_cast<unsigned>(layout.xy_nodes.size()));
        const auto radius = layout.radial_apothems[ring] / std::cos(std::numbers::pi_v<real_t> / n);
        for (size_t j = 0; j < n; ++j)
        {
            const auto angle = 2 * std::numbers::pi_v<real_t> * j / n;
            layout.xy_nodes.push_back({radius * std::cos(angle), radius * std::sin(angle), 0});
        }
    };
    for (size_t ring = 0; ring <= inner_layers; ++ring) add_ring(ring);
    for (size_t ring = outer_first; ring <= count.radial_cells; ++ring) add_ring(ring);
    const auto node = [&](size_t ring, size_t angle)
    { return static_cast<unsigned>(ring_start.at(ring) + angle % ring_count(ring)); };
    const auto add_wall_band = [&](size_t ring)
    {
        for (size_t j = 0; j < ring_count(ring); ++j)
            layout.xy_cells.push_back({node(ring, j), node(ring + 1, j),
                                       node(ring + 1, j + 1), node(ring, j + 1)});
    };
    for (size_t ring = 0; ring < inner_layers; ++ring) add_wall_band(ring);
    for (size_t ring = outer_first; ring < count.radial_cells; ++ring) add_wall_band(ring);

    Arr<Meshes::SemiStructuredXY_Z::Vec3> outer_interface, inner_interface;
    outer_interface.reserve(angular);
    inner_interface.reserve(inner_angular);
    for (size_t j = 0; j < angular; ++j)
        outer_interface.push_back(layout.xy_nodes[node(outer_first, j)]);
    for (size_t j = 0; j < inner_angular; ++j)
        inner_interface.push_back(layout.xy_nodes[node(inner_layers, j)]);
    const auto bulk = Meshes::FrontalDelaunay2D::triangulate_annulus(
        outer_interface, inner_interface, o.xy_spacing);
    const auto interface_nodes = angular + inner_angular;
    if (bulk.nodes.size() < interface_nodes)
        throw std::logic_error("Frontal bulk lost its prescribed interface vertices.");
    const auto extra_nodes = bulk.nodes.size() - interface_nodes;
    if (extra_nodes > index_limit - layout.xy_nodes.size()
        || bulk.triangles.size() > index_limit - wall_xy)
        throw std::overflow_error("Annular frontal mesh exceeds supported XY IDs.");
    std::vector<unsigned> bulk_node_ids(bulk.nodes.size());
    for (size_t j = 0; j < angular; ++j)
    {
        if (bulk.nodes[j] != outer_interface[j])
            throw std::logic_error("Frontal bulk changed a prescribed interface vertex.");
        bulk_node_ids[j] = node(outer_first, j);
    }
    for (size_t j = 0; j < inner_angular; ++j)
    {
        if (bulk.nodes[angular + j] != inner_interface[j])
            throw std::logic_error("Frontal bulk changed a prescribed interface vertex.");
        bulk_node_ids[angular + j] = node(inner_layers, j);
    }
    for (size_t j = interface_nodes; j < bulk.nodes.size(); ++j)
    {
        bulk_node_ids[j] = static_cast<unsigned>(layout.xy_nodes.size());
        layout.xy_nodes.push_back(bulk.nodes[j]);
    }
    for (const auto& triangle : bulk.triangles)
        layout.xy_cells.push_back({bulk_node_ids.at(triangle[0]), bulk_node_ids.at(triangle[1]),
                                   bulk_node_ids.at(triangle[2])});
    for (size_t j = 0; j < inner_angular; ++j)
        layout.boundaries.push_back({node(0, j + 1), node(0, j), "rmin"});
    for (size_t j = 0; j < angular; ++j)
        layout.boundaries.push_back({node(count.radial_cells, j), node(count.radial_cells, j + 1), "rmax"});
    count.wall_xy_cells = wall_xy;
    count.bulk_xy_cells = bulk.triangles.size();
    count.mixed_xy_cells = count.xy_cells = layout.xy_cells.size();
    count.xy_nodes = layout.xy_nodes.size();
    std::set<std::pair<unsigned, unsigned>> edges;
    for (const auto& cell : layout.xy_cells)
        for (size_t j = 0; j < cell.size(); ++j)
            edges.insert(std::minmax(cell[j], cell[(j + 1) % cell.size()]));
    const auto mixed_edges = edges.size();
    if (mixed_edges > index_limit)
        throw std::overflow_error("Annular frontal mesh exceeds supported XY edge IDs.");
    count.bottom_axial_cells = layers(o.refine_bottom);
    count.top_axial_cells = layers(o.refine_top);
    count.bulk_axial_cells = count.axial_cells - count.bottom_axial_cells - count.top_axial_cells;
    count.hex_cells = checked_product(wall_xy, count.axial_cells,
        std::numeric_limits<size_t>::max(), "radial wall cell count");
    count.prism_cells = checked_product(count.bulk_xy_cells, count.axial_cells,
        std::numeric_limits<size_t>::max(), "prism count");
    count.cells = count.hex_cells;
    checked_add(count.cells, count.prism_cells, "cell count");
    const auto add_slab = [&](size_t xy, size_t edges, size_t nz)
    {
        if (!nz) return;
        ++count.regions;
        checked_add(count.nodes, checked_product(count.xy_nodes, nz + 1,
            std::numeric_limits<size_t>::max(), "region node count"), "node count");
        checked_add(count.faces, checked_product(xy, nz + 1,
            std::numeric_limits<size_t>::max(), "axial face count"), "face count");
        checked_add(count.faces, checked_product(edges, nz,
            std::numeric_limits<size_t>::max(), "side face count"), "face count");
    };
    add_slab(count.mixed_xy_cells, mixed_edges, count.bottom_axial_cells);
    add_slab(count.mixed_xy_cells, mixed_edges, count.bulk_axial_cells);
    add_slab(count.mixed_xy_cells, mixed_edges, count.top_axial_cells);
    // Matching seam faces are merged one-to-one. Region-qualified nodes
    // deliberately remain distinct.
    count.faces -= checked_product(count.mixed_xy_cells, count.regions - 1,
        std::numeric_limits<size_t>::max(), "interface face count");
    return layout;
}

auto AnnularSemiStructuredMeshFactory::planned_counts() const -> Counts
{
    return plan_layout().counts;
}

auto AnnularSemiStructuredMeshFactory::build() const -> Result
{
    Layout layout;
    using Native = Meshes::SemiStructuredXY_Z;
    using Composite = Meshes::MultiRegionMesh;
    std::vector<SP<Native>> natives;
    std::vector<Composite::Region> regions;
    std::vector<Composite::Interface> interfaces;
    size_t middle_region = 0;
    std::exception_ptr local_error;
    try
    {
        layout = plan_layout();
        const auto& nodes = layout.xy_nodes;
        const auto& mixed_cells = layout.xy_cells;
        const auto& boundaries = layout.boundaries;

        const auto add_region = [&](const char* name, const Arr<Arr<unsigned>>& cells, size_t begin, size_t end)
        {
            ArrReal z(layout.z_edges.begin() + begin, layout.z_edges.begin() + end + 1);
            auto native = std::make_shared<Native>(nodes, cells, z, boundaries);
            natives.push_back(native);
            regions.emplace_back(Meshes::native_region(name, std::move(native)));
        };
        if (layout.counts.bottom_axial_cells)
            add_region("bottom_layers", mixed_cells, 0, layout.counts.bottom_axial_cells);
        middle_region = regions.size();
        add_region("bulk", mixed_cells, layout.counts.bottom_axial_cells,
            layout.counts.axial_cells - layout.counts.top_axial_cells);
        if (layout.counts.top_axial_cells)
            add_region("top_layers", mixed_cells, layout.counts.axial_cells - layout.counts.top_axial_cells,
                layout.counts.axial_cells);

        for (size_t upper_region = 1; upper_region < natives.size(); ++upper_region)
        {
            const auto& lower = *natives[upper_region - 1];
            const auto& upper = *natives[upper_region];
            const auto lower_z = static_cast<unsigned>(lower.z_edges().size() - 1);
            Meshes::ExplicitConformingInterface interface;
            interface.first_region = upper_region - 1;
            interface.second_region = upper_region;
            interface.first_boundary = lower.boundary_id({0, lower_z, Native::Z_FACE});
            interface.second_boundary = upper.boundary_id({0, 0, Native::Z_FACE});
            interface.faces.reserve(mixed_cells.size());
            for (size_t xy = 0; xy < mixed_cells.size(); ++xy)
            {
                interface.faces.emplace_back(
                    lower.indexer().face_ordinal({static_cast<unsigned>(xy), lower_z, Native::Z_FACE}),
                    upper.indexer().face_ordinal({static_cast<unsigned>(xy), 0, Native::Z_FACE}));
            }
            interfaces.emplace_back(std::move(interface));
        }
    }
    catch (...) { local_error = std::current_exception(); }
    const auto comm = Tpetra::getDefaultComm();
    const int local_failed = local_error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_failed, &any_failed);
    if (any_failed)
    {
        if (local_error) std::rethrow_exception(local_error);
        throw std::invalid_argument("Annular mesh preparation failed on another rank.");
    }

    Result result;
    result.mesh = std::make_shared<Composite>(std::move(regions), std::move(interfaces),
        Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames);
    // Use the mesh's own polygon metrics as the coupling reference area.
    long double area = 0;
    const auto& middle = *natives[middle_region];
    const auto first_height = middle.z_edges()[1] - middle.z_edges()[0];
    for (size_t xy = 0; xy < layout.counts.mixed_xy_cells; ++xy)
        area += middle.cell_volume({static_cast<unsigned>(xy), 0}) / first_height;
    const auto& o = d_options;
    result.cross_section_area = static_cast<real_t>(area);
    result.analytic_cross_section_area = static_cast<real_t>(std::numbers::pi_v<long double>
        * (static_cast<long double>(o.outer_radius) - o.inner_radius)
        * (static_cast<long double>(o.outer_radius) + o.inner_radius));
    result.relative_area_deficit = 1 - result.cross_section_area / result.analytic_cross_section_area;
    result.axial_fractions.reserve(layout.z_edges.size());
    for (const auto z : layout.z_edges) result.axial_fractions.push_back((z - o.bottom) / (o.top - o.bottom));
    result.axial_fractions.front() = 0;
    result.axial_fractions.back() = 1;
    result.radial_apothems = std::move(layout.radial_apothems);
    result.angular_cells = layout.counts.angular_cells;
    result.counts = layout.counts;
    return result;
}

} // namespace SimpleFluid
