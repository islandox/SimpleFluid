/**
 * @file BoundaryLayerMeshFactory.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief In-place boundary-layer mesh refinement implementation.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/BoundaryLayerMeshFactory.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace SimpleFluid
{
namespace
{

using Spec = BoundaryLayerMeshFactory::BoundaryLayerSpec;

/**
 * @brief Sum the cell widths in a geometric boundary-layer stack.
 * @param spec Layer specification, or null when no stack is requested.
 * @return Total stack thickness, or zero for a null specification.
 * @throws std::invalid_argument If the accumulated thickness is not finite.
 */
long double layer_thickness(const Spec* spec)
{
    if (spec == nullptr)
    {
        return 0.0L;
    }

    long double thickness = 0.0L;
    long double width = spec->first_cell_height;
    for (size_t layer = 0; layer < spec->count; ++layer)
    {
        thickness += width;
        width *= spec->growth_ratio;
        if (!std::isfinite(thickness) || !std::isfinite(width))
        {
            throw std::invalid_argument(
                "Boundary-layer stack thickness is not finite for "
                + spec->boundary_name + ".");
        }
    }
    return thickness;
}

} // namespace

BoundaryLayerMeshFactory::BoundaryLayerMeshFactory(
    Arr<BoundaryLayerSpec> layer_specs)
    : d_layer_specs(std::move(layer_specs))
{
    validate_specs(d_layer_specs);
}

void BoundaryLayerMeshFactory::validate_specs(
    const Arr<BoundaryLayerSpec>& layer_specs)
{
    std::unordered_set<std::string> unique_names;
    for (const auto& spec : layer_specs)
    {
        if (spec.boundary_name.empty())
        {
            throw std::invalid_argument(
                "Boundary-layer boundary name cannot be empty.");
        }
        if (!unique_names.insert(spec.boundary_name).second)
        {
            throw std::invalid_argument(
                "Boundary-layer boundary name is duplicated: "
                + spec.boundary_name + ".");
        }
        if (spec.count == 0)
        {
            throw std::invalid_argument(
                "Boundary-layer count must be positive for "
                + spec.boundary_name + ".");
        }
        if (!(spec.first_cell_height > 0.0)
            || !std::isfinite(spec.first_cell_height))
        {
            throw std::invalid_argument(
                "Boundary-layer first-cell height must be positive and finite for "
                + spec.boundary_name + ".");
        }
        if (spec.growth_ratio < 1.0
            || !std::isfinite(spec.growth_ratio))
        {
            throw std::invalid_argument(
                "Boundary-layer growth ratio must be finite and at least one for "
                + spec.boundary_name + ".");
        }
    }
}

BoundaryLayerMeshFactory::BoundaryLayerMeshFactory(
    SP<const Database> database)
    : d_layer_specs(read_specs(database))
{
}

/**
 * @brief Read parallel boundary-layer configuration arrays from a database.
 * @param database Configuration database.
 * @return Parsed boundary-layer specifications, or an empty array when none
 *         of the boundary-layer keys is present.
 * @throws std::invalid_argument If the database is null, configuration keys
 *         are incomplete, array sizes differ, or a count is not positive.
 */
Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec>
BoundaryLayerMeshFactory::read_specs(const SP<const Database>& database)
{
    if (!database)
    {
        throw std::invalid_argument(
            "BoundaryLayerMeshFactory requires a non-null database.");
    }

    constexpr std::array<const char*, 4> keys{
        "boundary_layer_boundary_names",
        "boundary_layer_counts",
        "boundary_layer_first_cell_heights",
        "boundary_layer_growth_ratios"};
    const auto has_any = std::ranges::any_of(
        keys, [&](const char* key) { return database->contains(key); });
    if (!has_any)
    {
        return {};
    }
    if (!std::ranges::all_of(
            keys, [&](const char* key) { return database->contains(key); }))
    {
        throw std::invalid_argument(
            "Boundary-layer mesh configuration requires names, counts, "
            "first-cell heights, and growth ratios.");
    }

    const auto& names =
        database->get<ArrString>("boundary_layer_boundary_names");
    const auto& counts =
        database->get<ArrInt>("boundary_layer_counts");
    const auto& first_heights =
        database->get<ArrReal>("boundary_layer_first_cell_heights");
    const auto& growth_ratios =
        database->get<ArrReal>("boundary_layer_growth_ratios");
    if (names.size() != counts.size()
        || names.size() != first_heights.size()
        || names.size() != growth_ratios.size())
    {
        throw std::invalid_argument(
            "Boundary-layer mesh configuration arrays must have matching sizes.");
    }

    Arr<BoundaryLayerSpec> specs;
    specs.reserve(names.size());
    for (size_t spec = 0; spec < names.size(); ++spec)
    {
        if (counts[spec] <= 0)
        {
            throw std::invalid_argument(
                "Boundary-layer count must be positive for "
                + names[spec] + ".");
        }
        specs.push_back({
            names[spec], static_cast<size_t>(counts[spec]),
            first_heights[spec], growth_ratios[spec]});
    }
    validate_specs(specs);
    return specs;
}

/**
 * @brief Find the layer specification attached to a boundary name.
 * @param boundary_name Boundary batch name to search for.
 * @return Matching specification, or null when the boundary has no stack.
 */
const BoundaryLayerMeshFactory::BoundaryLayerSpec*
BoundaryLayerMeshFactory::find_spec(
    const std::string& boundary_name) const noexcept
{
    const auto iter = std::find_if(
        d_layer_specs.begin(), d_layer_specs.end(),
        [&](const auto& spec)
        {
            return spec.boundary_name == boundary_name;
        });
    return iter == d_layer_specs.end() ? nullptr : &*iter;
}

/**
 * @brief Verify that every configured stack names an available boundary.
 * @param boundary_names Boundary names supported by the target mesh family.
 * @throws std::invalid_argument If a configured boundary is unavailable.
 */
void BoundaryLayerMeshFactory::validate_supported_boundaries(
    const ArrString& boundary_names) const
{
    for (const auto& spec : d_layer_specs)
    {
        if (std::find(boundary_names.begin(), boundary_names.end(),
                      spec.boundary_name) == boundary_names.end())
        {
            throw std::invalid_argument(
                "Boundary-layer boundary is not available on the input mesh: "
                + spec.boundary_name + ".");
        }
    }
}

/**
 * @brief Replace lower and upper coordinate intervals with graded layers.
 * @param input_edges Original strictly ordered coordinate edges.
 * @param lower Optional layer stack at the lower coordinate boundary.
 * @param upper Optional layer stack at the upper coordinate boundary.
 * @param coordinate_name Coordinate label used in diagnostics.
 * @return Refined, strictly increasing coordinate edges.
 * @throws std::invalid_argument If the input is invalid, the stacks overlap,
 *         or generated coordinates are not finite and distinct.
 */
ArrReal BoundaryLayerMeshFactory::refine_edges(
    const ArrReal& input_edges,
    const BoundaryLayerSpec* lower,
    const BoundaryLayerSpec* upper,
    const std::string& coordinate_name)
{
    if (lower == nullptr && upper == nullptr)
    {
        return input_edges;
    }
    if (input_edges.size() < 2)
    {
        throw std::invalid_argument(
            "Boundary-layer input coordinate requires at least two edges.");
    }

    const auto lower_bound = input_edges.front();
    const auto upper_bound = input_edges.back();
    const auto domain_width =
        static_cast<long double>(upper_bound) - lower_bound;
    const auto lower_thickness = layer_thickness(lower);
    const auto upper_thickness = layer_thickness(upper);
    if (!(domain_width > 0.0L)
        || lower_thickness + upper_thickness >= domain_width)
    {
        throw std::invalid_argument(
            "Boundary-layer stacks overlap on " + coordinate_name + ".");
    }

    const auto scale = std::max(
        {1.0, std::abs(lower_bound), std::abs(upper_bound),
         static_cast<real_t>(domain_width)});
    const auto tolerance = 128.0
                         * std::numeric_limits<real_t>::epsilon()
                         * scale;
    ArrReal result;
    result.reserve(
        input_edges.size()
        + (lower == nullptr ? 0 : lower->count)
        + (upper == nullptr ? 0 : upper->count));

    auto append_edge = [&](real_t edge)
    {
        if (!std::isfinite(edge)
            || (!result.empty() && edge <= result.back() + tolerance))
        {
            throw std::invalid_argument(
                "Boundary-layer edges are not distinct on "
                + coordinate_name + ".");
        }
        result.push_back(edge);
    };

    append_edge(lower_bound);
    if (lower != nullptr)
    {
        long double position = lower_bound;
        long double width = lower->first_cell_height;
        for (size_t layer = 0; layer < lower->count; ++layer)
        {
            position += width;
            append_edge(static_cast<real_t>(position));
            width *= lower->growth_ratio;
        }
    }

    const auto lower_interface =
        static_cast<real_t>(static_cast<long double>(lower_bound)
                          + lower_thickness);
    const auto upper_interface =
        static_cast<real_t>(static_cast<long double>(upper_bound)
                          - upper_thickness);
    for (size_t edge = 1; edge + 1 < input_edges.size(); ++edge)
    {
        if (input_edges[edge] > lower_interface + tolerance
            && input_edges[edge] < upper_interface - tolerance)
        {
            append_edge(input_edges[edge]);
        }
    }

    if (upper != nullptr)
    {
        append_edge(upper_interface);
        ArrReal widths;
        widths.reserve(upper->count);
        real_t width = upper->first_cell_height;
        for (size_t layer = 0; layer < upper->count; ++layer)
        {
            widths.push_back(width);
            width *= upper->growth_ratio;
        }

        long double position = upper_interface;
        for (auto iter = widths.rbegin(); iter != widths.rend(); ++iter)
        {
            position += *iter;
            append_edge(static_cast<real_t>(position));
        }
        result.back() = upper_bound;
    }
    else
    {
        append_edge(upper_bound);
    }

    return result;
}

void BoundaryLayerMeshFactory::build(
    Meshes::OrthogonalCartesian3D& mesh) const
{
    if (d_layer_specs.empty())
    {
        return;
    }
    const ArrString names{
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    validate_supported_boundaries(names);

    auto edges = mesh.cell_edges();
    edges[Meshes::OrthogonalCartesian3D::X] = refine_edges(
        edges[Meshes::OrthogonalCartesian3D::X],
        find_spec("xmin"), find_spec("xmax"), "Cartesian x coordinate");
    edges[Meshes::OrthogonalCartesian3D::Y] = refine_edges(
        edges[Meshes::OrthogonalCartesian3D::Y],
        find_spec("ymin"), find_spec("ymax"), "Cartesian y coordinate");
    edges[Meshes::OrthogonalCartesian3D::Z] = refine_edges(
        edges[Meshes::OrthogonalCartesian3D::Z],
        find_spec("zmin"), find_spec("zmax"), "Cartesian z coordinate");

    Meshes::OrthogonalCartesian3D refined(edges);
    mesh = std::move(refined);
}

void BoundaryLayerMeshFactory::build(
    Meshes::OrthogonalCylindrial3D& mesh) const
{
    if (d_layer_specs.empty())
    {
        return;
    }
    ArrString names{"rmin", "rmax", "zmin", "zmax"};
    if (!mesh.is_theta_periodic())
    {
        names.push_back("thetamin");
        names.push_back("thetamax");
    }
    validate_supported_boundaries(names);

    auto edges = mesh.cell_edges();
    edges[Meshes::OrthogonalCylindrial3D::R] = refine_edges(
        edges[Meshes::OrthogonalCylindrial3D::R],
        find_spec("rmin"), find_spec("rmax"), "cylindrical r coordinate");
    edges[Meshes::OrthogonalCylindrial3D::THETA] = refine_edges(
        edges[Meshes::OrthogonalCylindrial3D::THETA],
        find_spec("thetamin"), find_spec("thetamax"),
        "cylindrical theta coordinate");
    edges[Meshes::OrthogonalCylindrial3D::AXIAL] = refine_edges(
        edges[Meshes::OrthogonalCylindrial3D::AXIAL],
        find_spec("zmin"), find_spec("zmax"), "cylindrical z coordinate");

    Meshes::OrthogonalCylindrial3D refined(edges);
    mesh = std::move(refined);
}

void BoundaryLayerMeshFactory::build(
    Meshes::SemiStructuredXY_Z& mesh) const
{
    if (d_layer_specs.empty())
    {
        return;
    }
    ArrString boundary_names{"zmin", "zmax"};
    for (const auto batch_id : mesh.boundary_batch_ids())
    {
        const auto& name = mesh.boundary_batch_name(batch_id);
        if (std::find(boundary_names.begin(), boundary_names.end(), name)
            == boundary_names.end())
        {
            boundary_names.push_back(name);
        }
    }
    validate_supported_boundaries(boundary_names);
    for (const auto& spec : d_layer_specs)
    {
        if (spec.boundary_name != "zmin"
            && spec.boundary_name != "zmax")
        {
            throw std::invalid_argument(
                "SemiStructuredXY_Z boundary layers currently preserve XY "
                "topology and therefore support only zmin and zmax; requested "
                + spec.boundary_name + ".");
        }
    }

    const auto z_edges = refine_edges(
        mesh.z_edges(), find_spec("zmin"), find_spec("zmax"),
        "semi-structured z coordinate");

    Arr<Meshes::SemiStructuredXY_Z::BoundaryEdge> boundary_edges;
    for (const auto& side : mesh.topology().side_faces())
    {
        if (side.boundary_id
            == Meshes::SemiStructMeshTopo::invalid_boundary_id)
        {
            continue;
        }
        boundary_edges.push_back(
            {side.nodes[0], side.nodes[1],
             mesh.boundary_batch_name(side.boundary_id)});
    }

    Meshes::SemiStructuredXY_Z refined(
        mesh.xy_nodes(), mesh.xy_cell_nodes(), z_edges, boundary_edges);
    mesh = std::move(refined);
}

} // namespace SimpleFluid
