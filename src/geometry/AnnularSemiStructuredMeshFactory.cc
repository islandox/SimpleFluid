/** @file AnnularSemiStructuredMeshFactory.cc */
#include "geometry/AnnularSemiStructuredMeshFactory.hh"
#include "geometry/BoundaryLayerMeshFactory.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
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
    auto angular = std::max(size_t{8}, interval_count(
        2 * std::numbers::pi_v<long double> * o.outer_radius, o.xy_spacing));
    if (angular % 2) ++angular;
    const auto half_angle = std::numbers::pi_v<real_t> / angular;
    const auto outer_apothem = o.outer_radius * std::cos(half_angle);
    if (!(outer_apothem > o.inner_radius))
        throw std::invalid_argument("Annular XY spacing is too coarse for an inscribed outer polygon outside the inner circle.");

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
    count.radial_cells = layers(o.refine_inner) + layers(o.refine_outer)
        + interval_count(static_cast<long double>(radial[layers(o.refine_inner) + 1])
            - radial[layers(o.refine_inner)], o.xy_spacing);
    count.axial_cells = layers(o.refine_bottom) + layers(o.refine_top)
        + interval_count(static_cast<long double>(axial[layers(o.refine_bottom) + 1])
            - axial[layers(o.refine_bottom)], o.z_spacing);
    if (count.axial_cells >= index_limit)
        throw std::overflow_error("Annular axial node count exceeds supported IDs.");
    count.xy_cells = checked_product(angular, count.radial_cells, index_limit, "XY cell count");
    count.xy_nodes = checked_product(angular, count.radial_cells + 1, index_limit, "XY node count");
    // Side faces number N*(2*radial_cells+1), before axial extrusion.
    const auto xy_edges = checked_product(angular, 2 * count.radial_cells + 1, index_limit, "XY edge count");
    count.cells = checked_product(count.xy_cells, count.axial_cells,
        std::numeric_limits<size_t>::max(), "cell count");
    count.nodes = checked_product(count.xy_nodes, count.axial_cells + 1,
        std::numeric_limits<size_t>::max(), "node count");
    const auto axial_faces = checked_product(count.xy_cells, count.axial_cells + 1,
        std::numeric_limits<size_t>::max(), "axial face count");
    const auto side_faces = checked_product(xy_edges, count.axial_cells,
        std::numeric_limits<size_t>::max(), "side face count");
    if (axial_faces > std::numeric_limits<size_t>::max() - side_faces)
        throw std::overflow_error("Annular mesh total face count exceeds supported IDs.");
    count.faces = axial_faces + side_faces;
    layout.radial_apothems = subdivide_bulk(radial,
        layers(o.refine_inner), layers(o.refine_outer), o.xy_spacing);
    layout.z_edges = subdivide_bulk(axial,
        layers(o.refine_bottom), layers(o.refine_top), o.z_spacing);
    return layout;
}

auto AnnularSemiStructuredMeshFactory::planned_counts() const -> Counts
{
    return plan_layout().counts;
}

auto AnnularSemiStructuredMeshFactory::build() const -> Result
{
    auto layout = plan_layout();
    const auto& o = d_options;
    const auto n = layout.counts.angular_cells;
    const auto cosine = std::cos(std::numbers::pi_v<real_t> / n);
    Arr<Meshes::SemiStructuredXY_Z::Vec3> nodes;
    nodes.reserve(layout.counts.xy_nodes);
    for (const auto apothem : layout.radial_apothems)
    {
        const auto radius = apothem / cosine;
        for (size_t j = 0; j < n; ++j)
        {
            const auto theta = 2 * std::numbers::pi_v<real_t> * j / n;
            nodes.push_back({radius * std::cos(theta), radius * std::sin(theta), 0});
        }
    }
    const auto id = [n](size_t ring, size_t angle) { return static_cast<unsigned>(ring * n + angle % n); };
    Arr<Arr<unsigned>> cells;
    cells.reserve(layout.counts.xy_cells);
    for (size_t ring = 0; ring < layout.counts.radial_cells; ++ring)
        for (size_t j = 0; j < n; ++j)
            cells.push_back({id(ring, j), id(ring + 1, j), id(ring + 1, j + 1), id(ring, j + 1)});
    Arr<Meshes::SemiStructuredXY_Z::BoundaryEdge> boundaries;
    boundaries.reserve(2 * n);
    for (size_t j = 0; j < n; ++j)
    {
        boundaries.push_back({id(0, j + 1), id(0, j), "rmin"});
        boundaries.push_back({id(layout.counts.radial_cells, j), id(layout.counts.radial_cells, j + 1), "rmax"});
    }

    Result result;
    result.mesh = std::make_shared<Meshes::SemiStructuredXY_Z>(nodes, cells, layout.z_edges, boundaries);
    // Use the mesh's own polygon metrics as the coupling reference area.
    long double area = 0;
    const auto first_height = layout.z_edges[1] - layout.z_edges[0];
    for (size_t xy = 0; xy < layout.counts.xy_cells; ++xy)
        area += result.mesh->cell_volume({static_cast<unsigned>(xy), 0}) / first_height;
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
    result.angular_cells = n;
    result.counts = layout.counts;
    return result;
}

} // namespace SimpleFluid
