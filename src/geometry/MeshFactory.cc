/**
 * @file MeshFactory.cc
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief MeshFactory implementation for BOX, CYLINDER, and SPHERE domain builders.
 * @version 0.1
 * @date 2026-05-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "MeshFactory.hh"
#include "STKMesh.hh"
#include "geometry/mesh/FrontalDelaunay2D.hh"

#include <Teuchos_CommHelpers.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

namespace
{

/**
 * @brief Internal tag used during structured mesh construction.
 *
 * Records the axial layer and exterior-surface flag for generated mesh nodes.
 */
struct FactoryNodeTag
{
    size_t layer = 0;
    bool surface = false;
};

/**
 * @brief Run a mesh-generation operation only on rank zero.
 *
 * Any root exception is converted to a message and broadcast so every rank
 * exits the collective construction path coherently.
 */
template <class Operation>
std::string run_on_root(const std::string& context, Operation&& operation)
{
    const auto comm = Tpetra::getDefaultComm();
    std::string error;
    if (comm->getRank() == 0)
    {
        try
        {
            std::forward<Operation>(operation)();
        }
        catch (const std::exception& exception)
        {
            error = context + ": " + exception.what();
        }
        catch (...)
        {
            error = context + ": unknown exception";
        }
    }

    int error_size =
        comm->getRank() == 0 ? static_cast<int>(error.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &error_size);
    std::string root_error(static_cast<size_t>(error_size), '\0');
    if (comm->getRank() == 0)
    {
        root_error = std::move(error);
    }
    if (error_size != 0)
    {
        Teuchos::broadcast(
            *comm, 0, error_size, root_error.data());
    }
    return root_error;
}

/**
 * @brief Collectively open and close STK modification while rank zero
 *        populates the global bulk geometry.
 */
template <class Operation>
void populate_bulk_on_root(stk::mesh::BulkData& bulk,
                           const std::string& context,
                           Operation&& operation)
{
    bulk.modification_begin();
    const auto error =
        run_on_root(context, std::forward<Operation>(operation));
    bulk.modification_end();
    if (!error.empty())
    {
        throw std::runtime_error(error);
    }
}

/**
 * @brief Compute the minimum positive cell count for a given length and mesh size.
 *
 * @param length Physical extent of the domain axis.
 * @param mesh_size Desired mesh element size.
 * @return Minimum number of cells (at least 1).
 * @throws std::runtime_error If length or mesh_size is non-positive.
 */
size_t positive_count_from_size(real_t length, real_t mesh_size)
{
    if (length <= 0.0)
    {
        throw std::runtime_error("MeshFactory length scale must be positive.");
    }
    if (mesh_size <= 0.0)
    {
        throw std::runtime_error("MeshFactory mesh_size must be positive.");
    }

    return std::max<size_t>(
        1, static_cast<size_t>(std::ceil(length / mesh_size)));
}

/**
 * @brief Declare an STK mesh part with I/O attributes.
 *
 * @param meta STK metadata to declare the part on.
 * @param name Name of the part (must be non-empty).
 * @param rank Entity rank of the part.
 * @return Pointer to the declared STK part.
 * @throws std::runtime_error If name is empty.
 */
stk::mesh::Part* declare_io_part(stk::mesh::MetaData& meta,
                                  const std::string& name,
                                  stk::mesh::EntityRank rank)
{
    if (name.empty())
    {
        throw std::runtime_error("MeshFactory boundary part name cannot be empty.");
    }

    auto& part = meta.declare_part(name, rank);
    stk::io::put_io_part_attribute(part);
    return &part;
}

/**
 * @brief Declare boundary-element sides for a structured-mesh element based on node tags.
 *
 * @tparam Classifier Callable type that receives side node tags and returns a Part*.
 * @param bulk STK bulk data.
 * @param elem STK entity handle of the element.
 * @param topo Element topology.
 * @param elem_node_ids Node IDs of the element.
 * @param node_tags Map from node ID to factory node tag.
 * @param classifier Callable that selects a boundary part from side node tags.
 */
template <class Classifier>
void declare_tagged_boundary_sides(
    stk::mesh::BulkData& bulk,
    stk::mesh::Entity elem,
    stk::topology topo,
    const stk::mesh::EntityIdVector& elem_node_ids,
    const std::unordered_map<stk::mesh::EntityId, FactoryNodeTag>& node_tags,
    Classifier classifier)
{
    for (unsigned side = 0; side < topo.num_sides(); ++side)
    {
        std::vector<unsigned> ordinals(topo.side_topology(side).num_nodes());
        topo.side_node_ordinals(side, ordinals.begin());

        std::vector<FactoryNodeTag> side_tags;
        side_tags.reserve(ordinals.size());
        for (const auto ordinal : ordinals)
        {
            side_tags.push_back(node_tags.at(elem_node_ids[ordinal]));
        }

        if (auto* part = classifier(side_tags); part != nullptr)
        {
            stk::mesh::PartVector parts{part};
            bulk.declare_element_side(elem, side, parts);
        }
    }
}

/**
 * @brief Compute the total thickness of a geometric boundary-layer stack.
 *
 * @tparam Spec Type exposing count, first_cell_height, and growth_ratio.
 * @param spec Pointer to the boundary-layer specification (may be null).
 * @return Total thickness of the layer stack, or zero if spec is null or count is zero.
 */
template <class Spec>
real_t geometric_layer_thickness(const Spec* spec)
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
template <class Spec>
ArrReal graded_edges(real_t lower,
                     real_t upper,
                     size_t base_cell_count,
                     const Spec* lower_spec,
                     const Spec* upper_spec,
                     const std::string& axis_name)
{
    if (lower_spec == nullptr && upper_spec == nullptr)
    {
        return {};
    }

    const auto lower_count =
        lower_spec == nullptr ? 0 : lower_spec->count;
    const auto upper_count =
        upper_spec == nullptr ? 0 : upper_spec->count;
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

    const auto interior_width =
        interior_length / static_cast<real_t>(interior_count);
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
template <class Spec>
ArrReal symmetric_sphere_edges(size_t base_cell_count,
                               real_t radius,
                               const Spec* spec)
{
    if (spec == nullptr)
    {
        return {};
    }

    auto parametric_spec = *spec;
    parametric_spec.first_cell_height /= radius;
    return graded_edges(-1.0, 1.0, base_cell_count,
                        &parametric_spec, &parametric_spec,
                        "sphere parameter axis");
}

} // namespace

/**
 * @brief Construct a MeshFactory from a configuration database.
 *
 * @param db Shared database containing mesh configuration entries.
 */
MeshFactory::MeshFactory(SP<const Database> db)
{
    d_dimension = db->get<int>("dimension");
    d_mesh_size = db->get<real_t>("mesh_size");

    d_domain_type = static_cast<DomainType>(db->get<int>("domain_type"));
    if (d_domain_type == DomainType::BOX)
    {
        d_box_cell_edges = {
            db->get<ArrReal>("X"),
            db->get<ArrReal>("Y"),
            db->get<ArrReal>("Z")
        };
    }
    else if (d_domain_type == DomainType::CYLINDER)
    {
        d_radius = db->get<real_t>("radius");
        d_cylinder_height = db->get<real_t>("height");
        d_cylinder_circumferential_mesh_size =
            db->contains("cylinder_circumferential_mesh_size")
                ? db->get<real_t>(
                      "cylinder_circumferential_mesh_size")
                : d_mesh_size;
    }
    else if (d_domain_type == DomainType::ANNULUS)
    {
        d_annulus_cell_edges = {
            db->get<ArrReal>("R"),
            db->get<ArrReal>("Theta"),
            db->get<ArrReal>("Z")
        };
    }
    else if (d_domain_type == DomainType::SPHERE)
    {
        d_radius = db->get<real_t>("radius");
    }
    else if (d_domain_type == DomainType::EXTERNAL)
    {
        d_external_mesh_file = db->get<std::string>("external_mesh_file");
    }

    d_domain_exterior_face_types = db->get<ArrString>("domain_exterior_face_types");

    d_boundary_layer_specs = BoundaryLayerMeshFactory::read_specs(db);
    if (!d_boundary_layer_specs.empty())
    {
        validate_boundary_layer_names();
    }
}

/**
 * @brief Look up the boundary-layer specification for a given boundary name.
 *
 * @param boundary_name Name of the mesh boundary.
 * @return Pointer to the matching BoundaryLayerSpec, or nullptr if not found.
 */
const MeshFactory::BoundaryLayerSpec* MeshFactory::boundary_layer_spec(
    const std::string& boundary_name) const
{
    const auto iter =
        std::find_if(d_boundary_layer_specs.begin(), d_boundary_layer_specs.end(),
                     [&](const BoundaryLayerSpec& spec)
                     { return spec.boundary_name == boundary_name; });
    return iter == d_boundary_layer_specs.end() ? nullptr : &*iter;
}

/**
 * @brief Validate that all boundary-layer names correspond to domain exterior face types.
 *
 * @throws std::runtime_error If a boundary-layer name is not found in the domain exterior face types.
 */
void MeshFactory::validate_boundary_layer_names() const
{
    for (const auto& spec : d_boundary_layer_specs)
    {
        const auto matches =
            std::find(d_domain_exterior_face_types.begin(),
                      d_domain_exterior_face_types.end(),
                      spec.boundary_name);
        if (matches == d_domain_exterior_face_types.end())
        {
            throw std::runtime_error(
                "Boundary-layer boundary name is not part of this domain: "
                + spec.boundary_name);
        }
    }
}

/**
 * @brief Build a mesh using the configured domain settings.
 *
 * @tparam Pack Tpetra type pack used for mesh storage and communication.
 * @return Shared pointer to the constructed mesh.
 * @throws std::runtime_error if the requested domain type is unsupported.
 */
template <TpetraTypePack Pack>
SP<Mesh<Pack>> MeshFactory::build()
{
    if (d_domain_type == DomainType::EXTERNAL)
    {
        auto mesh =
            std::make_shared<STKMesh<Pack>>(d_external_mesh_file);
        mesh->assemble();
        return mesh;
    }

    typename STKMesh<Pack>::Options options;
    int next_boundary_id = 1;
    for (const auto& boundary_name : d_domain_exterior_face_types)
    {
        if (!options.boundary_name_to_id.contains(boundary_name))
        {
            options.boundary_name_to_id.emplace(
                boundary_name, next_boundary_id++);
        }
    }
    auto mesh = std::make_shared<STKMesh<Pack>>(options);

    if (d_domain_type == DomainType::BOX)
    {
        build_box_mesh(mesh);
    }
    else if (d_domain_type == DomainType::CYLINDER)
    {
        build_cylinder_mesh(mesh);
    }
    else if (d_domain_type == DomainType::ANNULUS)
    {
        build_annulus_mesh(mesh);
    }
    else if (d_domain_type == DomainType::SPHERE)
    {
        build_sphere_mesh(mesh);
    }
    else
    {
        throw std::runtime_error("Unsupported domain type for MeshFactory::build");
    }
    return mesh;
}

template <TpetraTypePack Pack>
SP<MeshHandle<Pack>> MeshFactory::build_handle()
{
    return std::make_shared<MeshHandle<Pack>>(build<Pack>());
}

/**
 * @brief Build a structured hexahedral mesh for a BOX domain.
 *
 * @tparam Pack Tpetra type pack used for mesh storage and communication.
 * @param mesh Mesh instance to build into.
 * @throws std::runtime_error if the domain is not 3D or boundary metadata is invalid.
 */
template <TpetraTypePack Pack>
void MeshFactory::build_box_mesh(SP<STKMesh<Pack>>& mesh)
{
    if (d_dimension != 3)
    {
        throw std::runtime_error("BOX MeshFactory currently constructs only 3D HEX_8 meshes.");
    }
    if (d_domain_exterior_face_types.size() < 6)
    {
        throw std::runtime_error("BOX MeshFactory requires six exterior face type names.");
    }

    auto validate_axis = [](const ArrReal& edges, const std::string& axis)
    {
        if (edges.size() < 2)
        {
            throw std::runtime_error("BOX MeshFactory axis " + axis
                                    + " must contain at least two cell edges.");
        }
        for (size_t i = 1; i < edges.size(); ++i)
        {
            if (edges[i] <= edges[i - 1])
            {
                throw std::runtime_error("BOX MeshFactory axis " + axis
                                        + " cell edges must be strictly increasing.");
            }
        }
    };

    validate_axis(d_box_cell_edges[X], "X");
    validate_axis(d_box_cell_edges[Y], "Y");
    validate_axis(d_box_cell_edges[Z], "Z");

    auto box_cell_edges = d_box_cell_edges;
    auto apply_axis_layers = [&](Dimension axis,
                                 const std::string& lower_name,
                                 const std::string& upper_name,
                                 const std::string& axis_name)
    {
        const auto& original = d_box_cell_edges[axis];
        auto edges = graded_edges(original.front(), original.back(),
                                  original.size() - 1,
                                  boundary_layer_spec(lower_name),
                                  boundary_layer_spec(upper_name),
                                  axis_name);
        if (!edges.empty())
        {
            box_cell_edges[axis] = std::move(edges);
            validate_axis(box_cell_edges[axis], axis_name);
        }
    };

    apply_axis_layers(X, d_domain_exterior_face_types[0],
                      d_domain_exterior_face_types[1], "X");
    apply_axis_layers(Y, d_domain_exterior_face_types[2],
                      d_domain_exterior_face_types[3], "Y");
    apply_axis_layers(Z, d_domain_exterior_face_types[4],
                      d_domain_exterior_face_types[5], "Z");

    const size_t num_cells_x = box_cell_edges[X].size() - 1;
    const size_t num_cells_y = box_cell_edges[Y].size() - 1;
    const size_t num_cells_z = box_cell_edges[Z].size() - 1;

    auto meta = mesh->meta();
    auto bulk = mesh->bulk();

    auto& coord_field =
        meta->template declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coord_field, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coord_field);

    auto& hex_part = meta->declare_part_with_topology("box_hexes", stk::topology::HEX_8);
    stk::io::put_io_part_attribute(hex_part);

    std::unordered_map<std::string, stk::mesh::Part*> boundary_parts_by_name;
    auto declare_boundary_part = [&](const std::string& name) -> stk::mesh::Part*
    {
        if (name.empty())
        {
            throw std::runtime_error("BOX MeshFactory boundary part name cannot be empty.");
        }

        const auto iter = boundary_parts_by_name.find(name);
        if (iter != boundary_parts_by_name.end())
        {
            return iter->second;
        }

        auto& part = meta->declare_part(name, meta->side_rank());
        stk::io::put_io_part_attribute(part);
        boundary_parts_by_name.emplace(name, &part);
        return &part;
    };

    std::array<stk::mesh::Part*, 6> boundary_parts{};
    for (size_t i = 0; i < boundary_parts.size(); ++i)
    {
        boundary_parts[i] = declare_boundary_part(d_domain_exterior_face_types[i]);
    }

    auto node_id = [=](size_t i, size_t j, size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + (num_cells_x + 1) * (j + (num_cells_y + 1) * k));
    };

    auto element_id = [=](size_t i, size_t j, size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + num_cells_x * (j + num_cells_y * k));
    };

    auto declare_boundary_side = [&](stk::mesh::Entity elem,
                                        unsigned side_ordinal,
                                        stk::mesh::Part* part)
    {
        stk::mesh::PartVector parts{part};
        bulk->declare_element_side(elem, side_ordinal, parts);
    };

    populate_bulk_on_root(
        *bulk,
        "BOX MeshFactory failed to construct root geometry",
        [&]
        {
            for (size_t k = 0; k < num_cells_z; ++k)
            {
                for (size_t j = 0; j < num_cells_y; ++j)
                {
                    for (size_t i = 0; i < num_cells_x; ++i)
                    {
                        const stk::mesh::EntityIdVector hex_nodes{
                            node_id(i,     j,     k),
                            node_id(i + 1, j,     k),
                            node_id(i + 1, j + 1, k),
                            node_id(i,     j + 1, k),
                            node_id(i,     j,     k + 1),
                            node_id(i + 1, j,     k + 1),
                            node_id(i + 1, j + 1, k + 1),
                            node_id(i,     j + 1, k + 1)
                        };

                        const auto elem = stk::mesh::declare_element(
                            *bulk, hex_part,
                            element_id(i, j, k), hex_nodes);

                        if (i == 0)
                            declare_boundary_side(
                                elem, 3, boundary_parts[0]);
                        if (i + 1 == num_cells_x)
                            declare_boundary_side(
                                elem, 1, boundary_parts[1]);
                        if (j == 0)
                            declare_boundary_side(
                                elem, 0, boundary_parts[2]);
                        if (j + 1 == num_cells_y)
                            declare_boundary_side(
                                elem, 2, boundary_parts[3]);
                        if (k == 0)
                            declare_boundary_side(
                                elem, 4, boundary_parts[4]);
                        if (k + 1 == num_cells_z)
                            declare_boundary_side(
                                elem, 5, boundary_parts[5]);
                    }
                }
            }

            for (size_t k = 0; k <= num_cells_z; ++k)
            {
                for (size_t j = 0; j <= num_cells_y; ++j)
                {
                    for (size_t i = 0; i <= num_cells_x; ++i)
                    {
                        const auto node = bulk->get_entity(
                            stk::topology::NODE_RANK,
                            node_id(i, j, k));
                        double* coord =
                            stk::mesh::field_data(coord_field, node);
                        if (coord == nullptr)
                        {
                            throw std::runtime_error(
                                "failed to write node coordinates");
                        }

                        coord[0] = box_cell_edges[X][i];
                        coord[1] = box_cell_edges[Y][j];
                        coord[2] = box_cell_edges[Z][k];
                    }
                }
            }
        });
    mesh->assemble();
}

/**
 * @brief Build a frontal-Delaunay triangular-prism mesh for a cylinder.
 *
 * Boundary part order is {radial, zmin, zmax}.
 */
template <TpetraTypePack Pack>
void MeshFactory::build_cylinder_mesh(SP<STKMesh<Pack>>& mesh)
{
    if (d_dimension != 3)
    {
        throw std::runtime_error("CYLINDER MeshFactory currently constructs only 3D WEDGE_6 meshes.");
    }
    if (d_radius <= 0.0 || d_cylinder_height <= 0.0)
    {
        throw std::runtime_error("CYLINDER MeshFactory requires positive radius and height.");
    }
    if (!(d_cylinder_circumferential_mesh_size > 0.0)
        || !std::isfinite(d_cylinder_circumferential_mesh_size))
    {
        throw std::runtime_error(
            "CYLINDER MeshFactory requires a finite positive "
            "cylinder_circumferential_mesh_size.");
    }
    if (d_domain_exterior_face_types.size() < 3)
    {
        throw std::runtime_error("CYLINDER MeshFactory requires boundary names {radial,zmin,zmax}.");
    }

    const auto base_radial_count = positive_count_from_size(d_radius, d_mesh_size);
    const auto base_height_count = positive_count_from_size(d_cylinder_height, d_mesh_size);
    ArrReal radial_edges =
        graded_edges(0.0, d_radius, base_radial_count,
                     static_cast<const BoundaryLayerSpec*>(nullptr),
                     boundary_layer_spec(d_domain_exterior_face_types[0]),
                     "cylinder radius");
    if (radial_edges.empty())
    {
        radial_edges.reserve(base_radial_count + 1);
        for (size_t ring = 0; ring <= base_radial_count; ++ring)
        {
            radial_edges.push_back(
                d_radius * static_cast<real_t>(ring)
              / static_cast<real_t>(base_radial_count));
        }
    }

    ArrReal z_edges =
        graded_edges(0.0, d_cylinder_height, base_height_count,
                     boundary_layer_spec(d_domain_exterior_face_types[1]),
                     boundary_layer_spec(d_domain_exterior_face_types[2]),
                     "cylinder height");
    if (z_edges.empty())
    {
        z_edges.reserve(base_height_count + 1);
        for (size_t layer = 0; layer <= base_height_count; ++layer)
        {
            z_edges.push_back(
                d_cylinder_height * static_cast<real_t>(layer)
              / static_cast<real_t>(base_height_count));
        }
    }

    const auto height_count = z_edges.size() - 1;

    // Place nodes on circular fronts (including prescribed radial boundary
    // layers), then let the shared XY Delaunay kernel determine connectivity.
    // This is the dominant temporary geometry for the fissile-tank case, so
    // keep it exclusively on rank zero.
    std::optional<Meshes::FrontalDelaunay2D::Result> xy_mesh;
    const auto triangulation_error = run_on_root(
        "CYLINDER MeshFactory failed to triangulate root geometry",
        [&]
        {
            xy_mesh.emplace(
                Meshes::FrontalDelaunay2D::triangulate_disk(
                    radial_edges,
                    d_cylinder_circumferential_mesh_size,
                    d_domain_exterior_face_types[0]));
        });
    if (!triangulation_error.empty())
    {
        throw std::runtime_error(triangulation_error);
    }
    const size_t nodes_per_layer =
        xy_mesh.has_value() ? xy_mesh->nodes.size() : 0;

    auto meta = mesh->meta();
    auto bulk = mesh->bulk();

    auto& coord_field =
        meta->template declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coord_field, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coord_field);

    auto& wedge_part =
        meta->declare_part_with_topology("cylinder_wedges", stk::topology::WEDGE_6);
    stk::io::put_io_part_attribute(wedge_part);

    auto* radial_part = declare_io_part(*meta, d_domain_exterior_face_types[0],
                                        meta->side_rank());
    auto* zmin_part = declare_io_part(*meta, d_domain_exterior_face_types[1],
                                      meta->side_rank());
    auto* zmax_part = declare_io_part(*meta, d_domain_exterior_face_types[2],
                                      meta->side_rank());

    auto node_id_from_layer_index = [=](size_t layer,
                                        size_t node_index)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + layer * nodes_per_layer + node_index);
    };

    populate_bulk_on_root(
        *bulk,
        "CYLINDER MeshFactory failed to construct root geometry",
        [&]
        {
            const auto& root_xy_mesh = xy_mesh.value();
            std::unordered_map<stk::mesh::EntityId, FactoryNodeTag>
                node_tags;
            node_tags.reserve(
                (height_count + 1) * nodes_per_layer);

            ArrBool is_radial_boundary(nodes_per_layer, false);
            for (const auto& edge : root_xy_mesh.boundary_edges)
            {
                is_radial_boundary[edge.node0] = true;
                is_radial_boundary[edge.node1] = true;
            }

            for (size_t layer = 0; layer <= height_count; ++layer)
            {
                for (size_t node = 0;
                     node < nodes_per_layer;
                     ++node)
                {
                    node_tags.emplace(
                        node_id_from_layer_index(layer, node),
                        FactoryNodeTag{
                            layer, is_radial_boundary[node]});
                }
            }

            stk::mesh::EntityId next_element_id = 1;
            auto declare_wedge =
                [&](size_t layer,
                    const Meshes::FrontalDelaunay2D::Triangle&
                        bottom_nodes)
                {
                    const stk::mesh::EntityIdVector wedge_nodes{
                        node_id_from_layer_index(
                            layer, bottom_nodes[0]),
                        node_id_from_layer_index(
                            layer, bottom_nodes[1]),
                        node_id_from_layer_index(
                            layer, bottom_nodes[2]),
                        node_id_from_layer_index(
                            layer + 1, bottom_nodes[0]),
                        node_id_from_layer_index(
                            layer + 1, bottom_nodes[1]),
                        node_id_from_layer_index(
                            layer + 1, bottom_nodes[2])
                    };

                    const auto elem = stk::mesh::declare_element(
                        *bulk,
                        wedge_part,
                        next_element_id++,
                        wedge_nodes);

                    declare_tagged_boundary_sides(
                        *bulk,
                        elem,
                        stk::topology::WEDGE_6,
                        wedge_nodes,
                        node_tags,
                        [&](const std::vector<FactoryNodeTag>& tags)
                            -> stk::mesh::Part*
                        {
                            const auto all_layer =
                                [&](size_t layer_id)
                                {
                                    return std::all_of(
                                        tags.begin(),
                                        tags.end(),
                                        [=](const FactoryNodeTag& tag)
                                        {
                                            return tag.layer
                                                == layer_id;
                                        });
                                };
                            const auto all_surface = [&]()
                            {
                                return std::all_of(
                                    tags.begin(),
                                    tags.end(),
                                    [](const FactoryNodeTag& tag)
                                    {
                                        return tag.surface;
                                    });
                            };

                            if (all_layer(0)) return zmin_part;
                            if (all_layer(height_count))
                                return zmax_part;
                            if (all_surface()) return radial_part;
                            return nullptr;
                        });
                };

            for (size_t layer = 0;
                 layer < height_count;
                 ++layer)
            {
                for (const auto& triangle
                     : root_xy_mesh.triangles)
                {
                    declare_wedge(layer, triangle);
                }
            }

            for (size_t layer = 0;
                 layer <= height_count;
                 ++layer)
            {
                const auto z = z_edges[layer];
                for (size_t node_index = 0;
                     node_index < nodes_per_layer;
                     ++node_index)
                {
                    const auto node = bulk->get_entity(
                        stk::topology::NODE_RANK,
                        node_id_from_layer_index(
                            layer, node_index));
                    double* coord =
                        stk::mesh::field_data(coord_field, node);
                    if (coord == nullptr)
                    {
                        throw std::runtime_error(
                            "failed to write node coordinates");
                    }

                    coord[0] =
                        root_xy_mesh.nodes[node_index].x;
                    coord[1] =
                        root_xy_mesh.nodes[node_index].y;
                    coord[2] = z;
                }
            }
        });
    xy_mesh.reset();
    mesh->assemble();
}

/**
 * @brief Build a structured HEX_8 annulus or annular sector.
 *
 * A 2 pi span identifies the first and final angular edges and creates a
 * conforming periodic seam. Smaller spans retain physical theta faces.
 */
template <TpetraTypePack Pack>
void MeshFactory::build_annulus_mesh(SP<STKMesh<Pack>>& mesh)
{
    if (d_dimension != 3)
    {
        throw std::runtime_error(
            "ANNULUS MeshFactory currently constructs only 3D HEX_8 meshes.");
    }
    const auto& radial_edges = d_annulus_cell_edges[0];
    const auto& theta_edges = d_annulus_cell_edges[1];
    const auto& z_edges = d_annulus_cell_edges[2];
    auto validate_edges = [](const ArrReal& edges, const char* name)
    {
        if (edges.size() < 2)
            throw std::runtime_error(
                std::string("ANNULUS ") + name + " requires at least two edges.");
        for (size_t edge = 0; edge < edges.size(); ++edge)
        {
            if (!std::isfinite(edges[edge])
                || (edge > 0 && edges[edge] <= edges[edge - 1]))
            {
                throw std::runtime_error(
                    std::string("ANNULUS ") + name
                    + " edges must be finite and strictly increasing.");
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
    const auto angular_tolerance =
        64.0 * std::numeric_limits<real_t>::epsilon() * two_pi;
    if (theta_span > two_pi + angular_tolerance)
    {
        throw std::runtime_error("ANNULUS Theta span cannot exceed 2 pi.");
    }
    const bool periodic =
        std::abs(theta_span - two_pi) <= angular_tolerance;
    const size_t required_boundaries = periodic ? 4 : 6;
    if (d_domain_exterior_face_types.size() < required_boundaries)
    {
        throw std::runtime_error(
            periodic
                ? "Periodic ANNULUS requires {rmin,rmax,zmin,zmax}."
                : "ANNULUS sector requires "
                  "{rmin,rmax,thetamin,thetamax,zmin,zmax}.");
    }
    const size_t nr = radial_edges.size() - 1;
    const size_t nt = theta_edges.size() - 1;
    const size_t nz = z_edges.size() - 1;
    if (periodic && nt < 3)
        throw std::runtime_error(
            "Periodic ANNULUS requires at least three theta cells.");

    auto meta = mesh->meta();
    auto bulk = mesh->bulk();
    auto& coord_field =
        meta->template declare_field<double>(stk::topology::NODE_RANK,
                                             "coordinates");
    stk::mesh::put_field_on_mesh(
        coord_field, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coord_field);
    auto& hex_part =
        meta->declare_part_with_topology("annulus_hexes", stk::topology::HEX_8);
    stk::io::put_io_part_attribute(hex_part);
    std::array<stk::mesh::Part*, 6> boundary_parts{};
    for (size_t boundary = 0; boundary < required_boundaries; ++boundary)
    {
        boundary_parts[boundary] = declare_io_part(
            *meta, d_domain_exterior_face_types[boundary], meta->side_rank());
    }

    const size_t theta_nodes = periodic ? nt : nt + 1;
    auto node_id = [=](size_t i, size_t j, size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + (nr + 1) * (j + theta_nodes * k));
    };
    auto element_id = [=](size_t i, size_t j, size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + nr * (j + nt * k));
    };
    auto declare_side = [&](stk::mesh::Entity element,
                            unsigned ordinal,
                            stk::mesh::Part* part)
    {
        bulk->declare_element_side(
            element, ordinal, stk::mesh::PartVector{part});
    };

    populate_bulk_on_root(
        *bulk,
        "ANNULUS MeshFactory failed to construct root geometry",
        [&]
        {
            for (size_t k = 0; k < nz; ++k)
            {
                for (size_t j = 0; j < nt; ++j)
                {
                    const size_t next_j = (j + 1) % theta_nodes;
                    for (size_t i = 0; i < nr; ++i)
                    {
                        const stk::mesh::EntityIdVector nodes{
                            node_id(i, j, k),
                            node_id(i + 1, j, k),
                            node_id(i + 1, next_j, k),
                            node_id(i, next_j, k),
                            node_id(i, j, k + 1),
                            node_id(i + 1, j, k + 1),
                            node_id(i + 1, next_j, k + 1),
                            node_id(i, next_j, k + 1)};
                        const auto element =
                            stk::mesh::declare_element(
                                *bulk,
                                hex_part,
                                element_id(i, j, k),
                                nodes);
                        if (i == 0)
                            declare_side(
                                element, 3, boundary_parts[0]);
                        if (i + 1 == nr)
                            declare_side(
                                element, 1, boundary_parts[1]);
                        if (!periodic && j == 0)
                            declare_side(
                                element, 0, boundary_parts[2]);
                        if (!periodic && j + 1 == nt)
                            declare_side(
                                element, 2, boundary_parts[3]);
                        const size_t zmin_part =
                            periodic ? 2 : 4;
                        const size_t zmax_part =
                            periodic ? 3 : 5;
                        if (k == 0)
                            declare_side(
                                element,
                                4,
                                boundary_parts[zmin_part]);
                        if (k + 1 == nz)
                            declare_side(
                                element,
                                5,
                                boundary_parts[zmax_part]);
                    }
                }
            }

            for (size_t k = 0; k <= nz; ++k)
            {
                for (size_t j = 0; j < theta_nodes; ++j)
                {
                    const auto cosine = std::cos(theta_edges[j]);
                    const auto sine = std::sin(theta_edges[j]);
                    for (size_t i = 0; i <= nr; ++i)
                    {
                        const auto node = bulk->get_entity(
                            stk::topology::NODE_RANK,
                            node_id(i, j, k));
                        double* coordinates =
                            stk::mesh::field_data(
                                coord_field, node);
                        if (coordinates == nullptr)
                        {
                            throw std::runtime_error(
                                "failed to write node coordinates");
                        }
                        coordinates[0] =
                            radial_edges[i] * cosine;
                        coordinates[1] =
                            radial_edges[i] * sine;
                        coordinates[2] = z_edges[k];
                    }
                }
            }
        });
    mesh->assemble();
}

/**
 * @brief Build a hexahedral spherified-cube mesh for a sphere.
 *
 * Boundary part order is either {surface} or {lower_surface, upper_surface}.
 */
template <TpetraTypePack Pack>
void MeshFactory::build_sphere_mesh(SP<STKMesh<Pack>>& mesh)
{
    if (d_dimension != 3)
    {
        throw std::runtime_error("SPHERE MeshFactory currently constructs only 3D HEX_8 meshes.");
    }
    if (d_radius <= 0.0)
    {
        throw std::runtime_error("SPHERE MeshFactory requires a positive radius.");
    }
    if (d_domain_exterior_face_types.empty())
    {
        throw std::runtime_error(
            "SPHERE MeshFactory requires boundary names {surface} "
            "or {lower_surface,upper_surface}.");
    }

    const auto cell_count = positive_count_from_size(2.0 * d_radius, d_mesh_size);
    const bool split_surface = d_domain_exterior_face_types.size() >= 2;
    const BoundaryLayerSpec* sphere_boundary_layer = nullptr;
    if (!d_boundary_layer_specs.empty())
    {
        if (!split_surface)
        {
            sphere_boundary_layer =
                boundary_layer_spec(d_domain_exterior_face_types[0]);
        }
        else
        {
            const auto* lower_spec =
                boundary_layer_spec(d_domain_exterior_face_types[0]);
            const auto* upper_spec =
                boundary_layer_spec(d_domain_exterior_face_types[1]);
            if ((lower_spec == nullptr) != (upper_spec == nullptr))
            {
                throw std::runtime_error(
                    "Split-surface sphere boundary layers require matching "
                    "lower and upper layer specifications.");
            }
            if (lower_spec != nullptr
                && (lower_spec->count != upper_spec->count
                    || lower_spec->first_cell_height
                       != upper_spec->first_cell_height
                    || lower_spec->growth_ratio != upper_spec->growth_ratio))
            {
                throw std::runtime_error(
                    "Split-surface sphere boundary layers must use identical "
                    "layer count, first-cell height, and growth ratio.");
            }
            sphere_boundary_layer = lower_spec;
        }
    }

    ArrReal sphere_edges =
        symmetric_sphere_edges(cell_count, d_radius, sphere_boundary_layer);
    if (sphere_edges.empty())
    {
        sphere_edges.reserve(cell_count + 1);
        for (size_t index = 0; index <= cell_count; ++index)
        {
            sphere_edges.push_back(
                -1.0 + 2.0 * static_cast<real_t>(index)
                     / static_cast<real_t>(cell_count));
        }
    }

    auto meta = mesh->meta();
    auto bulk = mesh->bulk();

    auto& coord_field =
        meta->template declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coord_field, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coord_field);

    auto& hex_part = meta->declare_part_with_topology("sphere_hexes", stk::topology::HEX_8);
    stk::io::put_io_part_attribute(hex_part);

    auto* lower_surface_part = declare_io_part(*meta, d_domain_exterior_face_types[0],
                                               meta->side_rank());
    auto* upper_surface_part = split_surface
                             ? declare_io_part(*meta, d_domain_exterior_face_types[1],
                                               meta->side_rank())
                             : nullptr;

    auto node_id = [=](size_t i, size_t j, size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + (cell_count + 1) * (j + (cell_count + 1) * k));
    };

    populate_bulk_on_root(
        *bulk,
        "SPHERE MeshFactory failed to construct root geometry",
        [&]
        {
            std::unordered_map<
                stk::mesh::EntityId,
                FactoryNodeTag> node_tags;
            node_tags.reserve(
                (cell_count + 1)
                * (cell_count + 1)
                * (cell_count + 1));
            for (size_t k = 0; k <= cell_count; ++k)
            {
                for (size_t j = 0; j <= cell_count; ++j)
                {
                    for (size_t i = 0;
                         i <= cell_count;
                         ++i)
                    {
                        const bool surface =
                            i == 0 || i == cell_count
                            || j == 0 || j == cell_count
                            || k == 0 || k == cell_count;
                        node_tags.emplace(
                            node_id(i, j, k),
                            FactoryNodeTag{k, surface});
                    }
                }
            }

            for (size_t k = 0; k < cell_count; ++k)
            {
                for (size_t j = 0; j < cell_count; ++j)
                {
                    for (size_t i = 0; i < cell_count; ++i)
                    {
                        const stk::mesh::EntityIdVector
                            hex_nodes{
                                node_id(i,     j,     k),
                                node_id(i + 1, j,     k),
                                node_id(i + 1, j + 1, k),
                                node_id(i,     j + 1, k),
                                node_id(i,     j,     k + 1),
                                node_id(i + 1, j,     k + 1),
                                node_id(i + 1, j + 1, k + 1),
                                node_id(i,     j + 1, k + 1)
                            };

                        const auto elem =
                            stk::mesh::declare_element(
                                *bulk,
                                hex_part,
                                static_cast<
                                    stk::mesh::EntityId>(
                                    1 + i
                                    + cell_count
                                        * (j
                                           + cell_count
                                             * k)),
                                hex_nodes);

                        declare_tagged_boundary_sides(
                            *bulk,
                            elem,
                            stk::topology::HEX_8,
                            hex_nodes,
                            node_tags,
                            [=](
                                const std::vector<
                                    FactoryNodeTag>& tags)
                                -> stk::mesh::Part*
                            {
                                const auto all_surface =
                                    std::all_of(
                                        tags.begin(),
                                        tags.end(),
                                        [](
                                            const FactoryNodeTag&
                                                tag)
                                        {
                                            return tag.surface;
                                        });
                                if (!all_surface)
                                {
                                    return nullptr;
                                }
                                if (!split_surface)
                                {
                                    return lower_surface_part;
                                }

                                size_t layer_sum = 0;
                                for (const auto& tag : tags)
                                {
                                    layer_sum += tag.layer;
                                }
                                const auto lower_side =
                                    layer_sum
                                    <= (cell_count
                                        * tags.size())
                                       / 2;
                                return lower_side
                                    ? lower_surface_part
                                    : upper_surface_part;
                            });
                    }
                }
            }

            for (size_t k = 0; k <= cell_count; ++k)
            {
                for (size_t j = 0;
                     j <= cell_count;
                     ++j)
                {
                    for (size_t i = 0;
                         i <= cell_count;
                         ++i)
                    {
                        const real_t u = sphere_edges[i];
                        const real_t v = sphere_edges[j];
                        const real_t w = sphere_edges[k];
                        const auto norm =
                            std::sqrt(
                                u * u + v * v + w * w);
                        const auto cube_radius =
                            std::max({
                                std::abs(u),
                                std::abs(v),
                                std::abs(w)});
                        const auto scale =
                            norm == 0.0
                                ? 0.0
                                : d_radius
                                  * cube_radius / norm;

                        const auto node =
                            bulk->get_entity(
                                stk::topology::NODE_RANK,
                                node_id(i, j, k));
                        double* coord =
                            stk::mesh::field_data(
                                coord_field, node);
                        if (coord == nullptr)
                        {
                            throw std::runtime_error(
                                "failed to write node coordinates");
                        }

                        coord[0] = u * scale;
                        coord[1] = v * scale;
                        coord[2] = w * scale;
                    }
                }
            }
        });
    mesh->assemble();
}

template SP<Mesh<DefaultTpetraTypes>> MeshFactory::build<DefaultTpetraTypes>();
template SP<MeshHandle<DefaultTpetraTypes>>
MeshFactory::build_handle<DefaultTpetraTypes>();

} 
