#include "NativeMeshGenerators.hh"
#include "geometry/mesh/FrontalDelaunay2D.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{
namespace
{
using Data = Meshes::UnstructuredMeshData;
using ID = Data::ID;
using CellType = MeshUtils::CellType;
using Spec = BoundaryLayerMeshFactory::BoundaryLayerSpec;

size_t positive_count_from_size(real_t length, real_t mesh_size)
{
    if (!std::isfinite(length) || length <= 0.0)
    {
        throw std::runtime_error("MeshFactory length scale must be finite and positive.");
    }
    if (!std::isfinite(mesh_size) || mesh_size <= 0.0)
    {
        throw std::runtime_error("MeshFactory mesh_size must be finite and positive.");
    }

    const auto requested_count = std::ceil(length / mesh_size);
    // size_t's exclusive upper bound is exact in binary floating point; casting
    // size_t::max() to real_t may round upward and admit an out-of-range cast.
    const auto exclusive_upper_bound = std::ldexp(real_t{1}, std::numeric_limits<size_t>::digits);
    if (!std::isfinite(requested_count) || requested_count >= exclusive_upper_bound)
    {
        throw std::overflow_error("MeshFactory requested cell count exceeds size_t.");
    }
    return std::max<size_t>(1, static_cast<size_t>(requested_count));
}

template<class Spec> real_t geometric_layer_thickness(const Spec* spec)
{
    if (spec == nullptr || spec->count == 0)
    {
        return 0.0;
    }

    real_t thickness = 0.0;
    real_t width = spec->first_cell_height;
    for (size_t layer = 0; layer < spec->count; ++layer)
    {
        thickness += width;
        width *= spec->growth_ratio;
    }

    return thickness;
}

/**
 * @brief Generate graded cell edges with optional boundary-layer refinement.
 *
 * Builds a non-uniform edge distribution for a mesh axis, inserting
 * geometrically graded layers at the lower and/or upper boundaries.
 *
 * @tparam Spec Type exposing count, first_cell_height, and growth_ratio.
 * @param lower Coordinate of the lower bound.
 * @param upper Coordinate of the upper bound.
 * @param base_cell_count Number of cells in the base (uniform) mesh.
 * @param lower_spec Boundary-layer specification at the lower end (may be null).
 * @param upper_spec Boundary-layer specification at the upper end (may be null).
 * @param axis_name Human-readable axis name for error messages.
 * @return Vector of edge coordinates (size = base_cell_count + layers + 1).
 * @throws std::runtime_error If layer counts overlap or thickness exceeds domain.
 */
template<class Spec>
ArrReal graded_edges(real_t lower, real_t upper, size_t base_cell_count, const Spec* lower_spec, const Spec* upper_spec,
    const std::string& axis_name)
{
    if (lower_spec == nullptr && upper_spec == nullptr)
    {
        return {};
    }

    const auto lower_count = lower_spec == nullptr ? 0 : lower_spec->count;
    const auto upper_count = upper_spec == nullptr ? 0 : upper_spec->count;
    if (lower_count + upper_count >= base_cell_count)
    {
        throw std::runtime_error("Boundary-layer counts overlap on " + axis_name + ".");
    }

    const auto length = upper - lower;
    const auto lower_thickness = geometric_layer_thickness(lower_spec);
    const auto upper_thickness = geometric_layer_thickness(upper_spec);
    const auto interior_length = length - lower_thickness - upper_thickness;
    if (interior_length <= 0.0)
    {
        throw std::runtime_error("Boundary-layer thicknesses overlap on " + axis_name + ".");
    }

    const auto interior_count = base_cell_count - lower_count - upper_count;
    ArrReal edges;
    edges.reserve(base_cell_count + 1);
    edges.push_back(lower);

    if (lower_spec != nullptr)
    {
        real_t width = lower_spec->first_cell_height;
        for (size_t layer = 0; layer < lower_spec->count; ++layer)
        {
            edges.push_back(edges.back() + width);
            width *= lower_spec->growth_ratio;
        }
    }

    const auto interior_width = interior_length / static_cast<real_t>(interior_count);
    for (size_t cell = 0; cell < interior_count; ++cell)
    {
        edges.push_back(edges.back() + interior_width);
    }

    if (upper_spec != nullptr)
    {
        ArrReal widths;
        widths.reserve(upper_spec->count);
        real_t width = upper_spec->first_cell_height;
        for (size_t layer = 0; layer < upper_spec->count; ++layer)
        {
            widths.push_back(width);
            width *= upper_spec->growth_ratio;
        }
        for (auto iter = widths.rbegin(); iter != widths.rend(); ++iter)
        {
            edges.push_back(edges.back() + *iter);
        }
    }

    edges.back() = upper;
    return edges;
}

/**
 * @brief Generate graded edges for a parametric sphere axis with boundary layers.
 *
 * Normalises the boundary-layer spec by radius and delegates to graded_edges
 * over the parametric range [-1, 1].
 *
 * @tparam Spec Type exposing count, first_cell_height, and growth_ratio.
 * @param base_cell_count Number of cells in the base (uniform) mesh.
 * @param radius Sphere radius.
 * @param spec Boundary-layer specification (may be null).
 * @return Vector of parametric edge coordinates, or empty if spec is null.
 */
template<class Spec> ArrReal symmetric_sphere_edges(size_t base_cell_count, real_t radius, const Spec* spec)
{
    if (spec == nullptr)
    {
        return {};
    }

    auto parametric_spec = *spec;
    parametric_spec.first_cell_height /= radius;
    return graded_edges(-1.0, 1.0, base_cell_count, &parametric_spec, &parametric_spec, "sphere parameter axis");
}

struct Generator
{
    const NativeMeshGeneratorOptions& o;
    std::vector<Data::Node> nodes;
    std::vector<Data::Cell> cells;
    std::vector<NativeBoundarySide> boundaries;

    const Spec* boundary_layer_spec(const std::string& name) const
    {
        for (const auto& spec : o.boundary_layer_specs)
            if (spec.boundary_name == name)
                return &spec;
        return nullptr;
    }
    void side(ID cell, unsigned ordinal, size_t boundary)
    {
        boundaries.push_back({cell, ordinal, o.domain_exterior_face_types.at(boundary)});
    }
    void hex(ID id, std::vector<ID> ids)
    {
        Data::Cell cell;
        cell.id = id;
        cell.type = CellType::HEXAHEDRON;
        cell.node_ids = std::move(ids);
        cells.push_back(std::move(cell));
    }
    void box();
    void cylinder();
    void annulus();
    void sphere();
};
void Generator::box()
{
    if (o.dimension != 3)
    {
        throw std::runtime_error("BOX MeshFactory currently constructs only 3D HEX_8 meshes.");
    }
    if (o.domain_exterior_face_types.size() < 6)
    {
        throw std::runtime_error("BOX MeshFactory requires six exterior face type names.");
    }

    auto validate_axis = [](const ArrReal& edges, const std::string& axis)
    {
        if (edges.size() < 2)
        {
            throw std::runtime_error("BOX MeshFactory axis " + axis + " must contain at least two cell edges.");
        }
        for (size_t i = 1; i < edges.size(); ++i)
        {
            if (edges[i] <= edges[i - 1])
            {
                throw std::runtime_error("BOX MeshFactory axis " + axis + " cell edges must be strictly increasing.");
            }
        }
    };

    validate_axis(o.box_cell_edges[X], "X");
    validate_axis(o.box_cell_edges[Y], "Y");
    validate_axis(o.box_cell_edges[Z], "Z");

    auto box_cell_edges = o.box_cell_edges;
    auto apply_axis_layers =
        [&](Dimension axis, const std::string& lower_name, const std::string& upper_name, const std::string& axis_name)
    {
        const auto& original = o.box_cell_edges[axis];
        auto edges = graded_edges(original.front(), original.back(), original.size() - 1,
            boundary_layer_spec(lower_name), boundary_layer_spec(upper_name), axis_name);
        if (!edges.empty())
        {
            box_cell_edges[axis] = std::move(edges);
            validate_axis(box_cell_edges[axis], axis_name);
        }
    };

    apply_axis_layers(X, o.domain_exterior_face_types[0], o.domain_exterior_face_types[1], "X");
    apply_axis_layers(Y, o.domain_exterior_face_types[2], o.domain_exterior_face_types[3], "Y");
    apply_axis_layers(Z, o.domain_exterior_face_types[4], o.domain_exterior_face_types[5], "Z");

    const size_t num_cells_x = box_cell_edges[X].size() - 1;
    const size_t num_cells_y = box_cell_edges[Y].size() - 1;
    const size_t num_cells_z = box_cell_edges[Z].size() - 1;

    auto node_id = [=](size_t i, size_t j, size_t k) -> ID
    { return 1 + i + (num_cells_x + 1) * (j + (num_cells_y + 1) * k); };
    for (size_t k = 0; k <= num_cells_z; ++k)
        for (size_t j = 0; j <= num_cells_y; ++j)
            for (size_t i = 0; i <= num_cells_x; ++i)
                nodes.push_back({node_id(i, j, k), {box_cell_edges[X][i], box_cell_edges[Y][j], box_cell_edges[Z][k]}});
    for (size_t k = 0; k < num_cells_z; ++k)
        for (size_t j = 0; j < num_cells_y; ++j)
            for (size_t i = 0; i < num_cells_x; ++i)
            {
                const ID id = 1 + i + num_cells_x * (j + num_cells_y * k);
                hex(id, {node_id(i, j, k), node_id(i + 1, j, k), node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                            node_id(i, j, k + 1), node_id(i + 1, j, k + 1), node_id(i + 1, j + 1, k + 1),
                            node_id(i, j + 1, k + 1)});
                if (i == 0)
                    side(id, 3, 0);
                if (i + 1 == num_cells_x)
                    side(id, 1, 1);
                if (j == 0)
                    side(id, 0, 2);
                if (j + 1 == num_cells_y)
                    side(id, 2, 3);
                if (k == 0)
                    side(id, 4, 4);
                if (k + 1 == num_cells_z)
                    side(id, 5, 5);
            }
}
void Generator::cylinder()
{
    if (o.dimension != 3)
    {
        throw std::runtime_error("CYLINDER MeshFactory currently constructs only 3D WEDGE_6 meshes.");
    }
    if (o.radius <= 0.0 || o.cylinder_height <= 0.0)
    {
        throw std::runtime_error("CYLINDER MeshFactory requires positive radius and height.");
    }
    if (!(o.cylinder_circumferential_mesh_size > 0.0) || !std::isfinite(o.cylinder_circumferential_mesh_size))
    {
        throw std::runtime_error("CYLINDER MeshFactory requires a finite positive "
                                 "cylinder_circumferential_mesh_size.");
    }
    if (o.domain_exterior_face_types.size() < 3)
    {
        throw std::runtime_error("CYLINDER MeshFactory requires boundary names {radial,zmin,zmax}.");
    }

    const auto base_radial_count = positive_count_from_size(o.radius, o.mesh_size);
    const auto base_height_count = positive_count_from_size(o.cylinder_height, o.mesh_size);
    ArrReal radial_edges = graded_edges(0.0, o.radius, base_radial_count, static_cast<const Spec*>(nullptr),
        boundary_layer_spec(o.domain_exterior_face_types[0]), "cylinder radius");
    if (radial_edges.empty())
    {
        radial_edges.reserve(base_radial_count + 1);
        for (size_t ring = 0; ring <= base_radial_count; ++ring)
        {
            radial_edges.push_back(o.radius * static_cast<real_t>(ring) / static_cast<real_t>(base_radial_count));
        }
    }

    ArrReal z_edges =
        graded_edges(0.0, o.cylinder_height, base_height_count, boundary_layer_spec(o.domain_exterior_face_types[1]),
            boundary_layer_spec(o.domain_exterior_face_types[2]), "cylinder height");
    if (z_edges.empty())
    {
        z_edges.reserve(base_height_count + 1);
        for (size_t layer = 0; layer <= base_height_count; ++layer)
        {
            z_edges.push_back(o.cylinder_height * static_cast<real_t>(layer) / static_cast<real_t>(base_height_count));
        }
    }

    const auto height_count = z_edges.size() - 1;

    const auto xy = Meshes::FrontalDelaunay2D::triangulate_disk(
        radial_edges, o.cylinder_circumferential_mesh_size, o.domain_exterior_face_types[0]);
    const auto n = xy.nodes.size();
    auto node_id = [=](size_t layer, size_t node) -> ID { return 1 + layer * n + node; };
    ArrBool radial(n, false);
    for (const auto& edge : xy.boundary_edges)
        radial[edge.node0] = radial[edge.node1] = true;
    for (size_t k = 0; k <= height_count; ++k)
        for (size_t i = 0; i < n; ++i)
            nodes.push_back({node_id(k, i), {xy.nodes[i].x, xy.nodes[i].y, z_edges[k]}});
    for (size_t k = 0; k < height_count; ++k)
        for (const auto& t : xy.triangles)
        {
            Data::Cell c;
            c.id = cells.size() + 1;
            c.type = CellType::TRIPRISM;
            c.node_ids = {node_id(k, t[0]), node_id(k, t[1]), node_id(k, t[2]), node_id(k + 1, t[0]),
                node_id(k + 1, t[1]), node_id(k + 1, t[2])};
            if (k == 0)
                side(c.id, 3, 1);
            if (k + 1 == height_count)
                side(c.id, 4, 2);
            const auto& cell_sides = native_mesh_side_node_ordinals(c.type);
            for (unsigned s = 0; s < 3; ++s)
            {
                bool all_surface = true;
                for (const auto ordinal : cell_sides[s])
                    all_surface = all_surface && radial[t[ordinal % 3]];
                if (all_surface)
                    side(c.id, s, 0);
            }
            cells.push_back(std::move(c));
        }
}
void Generator::annulus()
{
    if (o.dimension != 3)
    {
        throw std::runtime_error("ANNULUS MeshFactory currently constructs only 3D HEX_8 meshes.");
    }
    const auto& radial_edges = o.annulus_cell_edges[0];
    const auto& theta_edges = o.annulus_cell_edges[1];
    const auto& z_edges = o.annulus_cell_edges[2];
    auto validate_edges = [](const ArrReal& edges, const char* name)
    {
        if (edges.size() < 2)
            throw std::runtime_error(std::string("ANNULUS ") + name + " requires at least two edges.");
        for (size_t edge = 0; edge < edges.size(); ++edge)
        {
            if (!std::isfinite(edges[edge]) || (edge > 0 && edges[edge] <= edges[edge - 1]))
            {
                throw std::runtime_error(
                    std::string("ANNULUS ") + name + " edges must be finite and strictly increasing.");
            }
        }
    };
    validate_edges(radial_edges, "R");
    validate_edges(theta_edges, "Theta");
    validate_edges(z_edges, "Z");
    if (radial_edges.front() <= 0.0)
        throw std::runtime_error("ANNULUS inner radius must be positive.");

    constexpr real_t two_pi = 2.0 * std::numbers::pi_v<real_t>;
    const auto theta_span = theta_edges.back() - theta_edges.front();
    const auto angular_tolerance = 64.0 * std::numeric_limits<real_t>::epsilon() * two_pi;
    if (theta_span > two_pi + angular_tolerance)
    {
        throw std::runtime_error("ANNULUS Theta span cannot exceed 2 pi.");
    }
    const bool periodic = std::abs(theta_span - two_pi) <= angular_tolerance;
    const size_t required_boundaries = periodic ? 4 : 6;
    if (o.domain_exterior_face_types.size() < required_boundaries)
    {
        throw std::runtime_error(periodic ? "Periodic ANNULUS requires {rmin,rmax,zmin,zmax}."
                                          : "ANNULUS sector requires "
                                            "{rmin,rmax,thetamin,thetamax,zmin,zmax}.");
    }
    const size_t nr = radial_edges.size() - 1;
    const size_t nt = theta_edges.size() - 1;
    const size_t nz = z_edges.size() - 1;
    if (periodic && nt < 3)
        throw std::runtime_error("Periodic ANNULUS requires at least three theta cells.");

    const size_t theta_nodes = periodic ? nt : nt + 1;
    auto node_id = [=](size_t i, size_t j, size_t k) -> ID { return 1 + i + (nr + 1) * (j + theta_nodes * k); };
    for (size_t k = 0; k <= nz; ++k)
        for (size_t j = 0; j < theta_nodes; ++j)
            for (size_t i = 0; i <= nr; ++i)
                nodes.push_back({node_id(i, j, k), {radial_edges[i] * std::cos(theta_edges[j]),
                                                       radial_edges[i] * std::sin(theta_edges[j]), z_edges[k]}});
    for (size_t k = 0; k < nz; ++k)
        for (size_t j = 0; j < nt; ++j)
            for (size_t i = 0; i < nr; ++i)
            {
                const auto next = (j + 1) % theta_nodes;
                const ID id = 1 + i + nr * (j + nt * k);
                hex(id, {node_id(i, j, k), node_id(i + 1, j, k), node_id(i + 1, next, k), node_id(i, next, k),
                            node_id(i, j, k + 1), node_id(i + 1, j, k + 1), node_id(i + 1, next, k + 1),
                            node_id(i, next, k + 1)});
                if (i == 0)
                    side(id, 3, 0);
                if (i + 1 == nr)
                    side(id, 1, 1);
                if (!periodic && j == 0)
                    side(id, 0, 2);
                if (!periodic && j + 1 == nt)
                    side(id, 2, 3);
                if (k == 0)
                    side(id, 4, periodic ? 2 : 4);
                if (k + 1 == nz)
                    side(id, 5, periodic ? 3 : 5);
            }
}
void Generator::sphere()
{
    if (o.dimension != 3)
    {
        throw std::runtime_error("SPHERE MeshFactory currently constructs only 3D HEX_8 meshes.");
    }
    if (o.radius <= 0.0)
    {
        throw std::runtime_error("SPHERE MeshFactory requires a positive radius.");
    }
    if (o.domain_exterior_face_types.empty())
    {
        throw std::runtime_error("SPHERE MeshFactory requires boundary names {surface} "
                                 "or {lower_surface,upper_surface}.");
    }

    const auto cell_count = positive_count_from_size(2.0 * o.radius, o.mesh_size);
    const bool split_surface = o.domain_exterior_face_types.size() >= 2;
    const Spec* sphere_boundary_layer = nullptr;
    if (!o.boundary_layer_specs.empty())
    {
        if (!split_surface)
        {
            sphere_boundary_layer = boundary_layer_spec(o.domain_exterior_face_types[0]);
        }
        else
        {
            const auto* lower_spec = boundary_layer_spec(o.domain_exterior_face_types[0]);
            const auto* upper_spec = boundary_layer_spec(o.domain_exterior_face_types[1]);
            if ((lower_spec == nullptr) != (upper_spec == nullptr))
            {
                throw std::runtime_error("Split-surface sphere boundary layers require matching "
                                         "lower and upper layer specifications.");
            }
            if (lower_spec != nullptr && (lower_spec->count != upper_spec->count ||
                                             lower_spec->first_cell_height != upper_spec->first_cell_height ||
                                             lower_spec->growth_ratio != upper_spec->growth_ratio))
            {
                throw std::runtime_error("Split-surface sphere boundary layers must use identical "
                                         "layer count, first-cell height, and growth ratio.");
            }
            sphere_boundary_layer = lower_spec;
        }
    }

    ArrReal sphere_edges = symmetric_sphere_edges(cell_count, o.radius, sphere_boundary_layer);
    if (sphere_edges.empty())
    {
        sphere_edges.reserve(cell_count + 1);
        for (size_t index = 0; index <= cell_count; ++index)
        {
            sphere_edges.push_back(-1.0 + 2.0 * static_cast<real_t>(index) / static_cast<real_t>(cell_count));
        }
    }

    auto node_id = [=](size_t i, size_t j, size_t k) -> ID
    { return 1 + i + (cell_count + 1) * (j + (cell_count + 1) * k); };
    for (size_t k = 0; k <= cell_count; ++k)
        for (size_t j = 0; j <= cell_count; ++j)
            for (size_t i = 0; i <= cell_count; ++i)
            {
                const real_t u = sphere_edges[i], v = sphere_edges[j], w = sphere_edges[k];
                const auto norm = std::sqrt(u * u + v * v + w * w);
                const auto cube_radius = std::max({std::abs(u), std::abs(v), std::abs(w)});
                const auto scale = norm == 0.0 ? 0.0 : o.radius * cube_radius / norm;
                nodes.push_back({node_id(i, j, k), {u * scale, v * scale, w * scale}});
            }
    for (size_t k = 0; k < cell_count; ++k)
        for (size_t j = 0; j < cell_count; ++j)
            for (size_t i = 0; i < cell_count; ++i)
            {
                const ID id = 1 + i + cell_count * (j + cell_count * k);
                hex(id, {node_id(i, j, k), node_id(i + 1, j, k), node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                            node_id(i, j, k + 1), node_id(i + 1, j, k + 1), node_id(i + 1, j + 1, k + 1),
                            node_id(i, j + 1, k + 1)});
                // Preserve the original all-surface-node classifier, including
                // coarse grids whose interior side vertices all lie on the surface.
                const auto& cell = cells.back();
                const auto width = cell_count + 1;
                for (unsigned s = 0; s < 6; ++s)
                {
                    bool surface = true;
                    size_t layer_sum = 0;
                    for (const auto ordinal : native_mesh_side_node_ordinals(cell.type)[s])
                    {
                        auto n = cell.node_ids[ordinal] - 1;
                        const auto a = n % width, b = (n / width) % width, c = n / (width * width);
                        surface = surface &&
                                  (a == 0 || a == cell_count || b == 0 || b == cell_count || c == 0 || c == cell_count);
                        layer_sum += c;
                    }
                    if (surface)
                        side(id, s,
                            split_surface &&
                                    layer_sum > cell_count * native_mesh_side_node_ordinals(cell.type)[s].size() / 2
                                ? 1
                                : 0);
                }
            }
}
} // namespace

Data generate_native_mesh(const NativeMeshGeneratorOptions& options)
{
    BoundaryLayerMeshFactory::validate_specs(options.boundary_layer_specs);
    Generator g{options, {}, {}, {}};
    switch (options.domain_type)
    {
        case 0:
            g.box();
            break;
        case 1:
            g.cylinder();
            break;
        case 2:
            g.sphere();
            break;
        case 4:
            g.annulus();
            break;
        default:
            throw std::invalid_argument("Unsupported native mesh generator domain.");
    }
    auto result = build_native_mesh_data(
        std::move(g.nodes), std::move(g.cells), g.boundaries, options.domain_exterior_face_types);
    // MeshFactory preassigned IDs by first occurrence in the configured array.
    // Retain its full catalog, including configured names with no incident side.
    // The raw builder deliberately keeps STK's default first-face-encounter policy.
    std::map<std::string, int> configured_ids;
    std::unordered_map<int, std::string> configured_names;
    for (const auto& name : options.domain_exterior_face_types)
    {
        const auto [it, inserted] = configured_ids.emplace(name, static_cast<int>(configured_ids.size() + 1));
        if (inserted)
            configured_names.emplace(it->second, name);
    }
    for (auto& face : result.faces)
        if (face.boundary_id >= 0)
            face.boundary_id = configured_ids.at(result.boundary_names.at(face.boundary_id));
    result.boundary_names = std::move(configured_names);
    return result;
}
} // namespace SimpleFluid::detail
