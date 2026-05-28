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

#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace SimpleFluid
{

namespace
{

struct FactoryNodeTag
{
    std::size_t ring = 0;
    std::size_t layer = 0;
    bool surface = false;
};

std::size_t positive_count_from_size(real_t length, real_t mesh_size)
{
    if (length <= 0.0)
    {
        throw std::runtime_error("MeshFactory length scale must be positive.");
    }
    if (mesh_size <= 0.0)
    {
        throw std::runtime_error("MeshFactory mesh_size must be positive.");
    }

    return std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(length / mesh_size)));
}

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

template <class Spec>
real_t geometric_layer_thickness(const Spec* spec)
{
    if (spec == nullptr || spec->count == 0)
    {
        return 0.0;
    }

    real_t thickness = 0.0;
    real_t width = spec->first_cell_height;
    for (int layer = 0; layer < spec->count; ++layer)
    {
        thickness += width;
        width *= spec->growth_ratio;
    }

    return thickness;
}

template <class Spec>
ArrReal graded_edges(real_t lower,
                     real_t upper,
                     std::size_t base_cell_count,
                     const Spec* lower_spec,
                     const Spec* upper_spec,
                     const std::string& axis_name)
{
    if (lower_spec == nullptr && upper_spec == nullptr)
    {
        return {};
    }

    const auto lower_count = lower_spec == nullptr
                           ? 0
                           : static_cast<std::size_t>(lower_spec->count);
    const auto upper_count = upper_spec == nullptr
                           ? 0
                           : static_cast<std::size_t>(upper_spec->count);
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
        for (int layer = 0; layer < lower_spec->count; ++layer)
        {
            edges.push_back(edges.back() + width);
            width *= lower_spec->growth_ratio;
        }
    }

    const auto interior_width =
        interior_length / static_cast<real_t>(interior_count);
    for (std::size_t cell = 0; cell < interior_count; ++cell)
    {
        edges.push_back(edges.back() + interior_width);
    }

    if (upper_spec != nullptr)
    {
        ArrReal widths;
        widths.reserve(static_cast<std::size_t>(upper_spec->count));
        real_t width = upper_spec->first_cell_height;
        for (int layer = 0; layer < upper_spec->count; ++layer)
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

template <class Spec>
ArrReal symmetric_sphere_edges(std::size_t base_cell_count,
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

    const bool has_boundary_layer_keys =
        db->contains("boundary_layer_boundary_names")
     || db->contains("boundary_layer_counts")
     || db->contains("boundary_layer_first_cell_heights")
     || db->contains("boundary_layer_growth_ratios");
    if (has_boundary_layer_keys)
    {
        if (!db->contains("boundary_layer_boundary_names")
            || !db->contains("boundary_layer_counts")
            || !db->contains("boundary_layer_first_cell_heights")
            || !db->contains("boundary_layer_growth_ratios"))
        {
            throw std::runtime_error(
                "Boundary-layer mesh configuration requires names, counts, "
                "first-cell heights, and growth ratios.");
        }

        const auto& names = db->get<ArrString>("boundary_layer_boundary_names");
        const auto& counts = db->get<ArrInt>("boundary_layer_counts");
        const auto& first_heights =
            db->get<ArrReal>("boundary_layer_first_cell_heights");
        const auto& growth_ratios =
            db->get<ArrReal>("boundary_layer_growth_ratios");
        if (names.size() != counts.size()
            || names.size() != first_heights.size()
            || names.size() != growth_ratios.size())
        {
            throw std::runtime_error(
                "Boundary-layer mesh configuration arrays must have matching sizes.");
        }

        std::unordered_set<std::string> unique_names;
        d_boundary_layer_specs.reserve(names.size());
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            if (names[i].empty())
            {
                throw std::runtime_error("Boundary-layer boundary name cannot be empty.");
            }
            if (!unique_names.insert(names[i]).second)
            {
                throw std::runtime_error(
                    "Boundary-layer boundary name is duplicated: " + names[i]);
            }
            if (counts[i] <= 0)
            {
                throw std::runtime_error(
                    "Boundary-layer count must be positive for " + names[i] + ".");
            }
            if (first_heights[i] <= 0.0)
            {
                throw std::runtime_error(
                    "Boundary-layer first-cell height must be positive for "
                    + names[i] + ".");
            }
            if (growth_ratios[i] < 1.0)
            {
                throw std::runtime_error(
                    "Boundary-layer growth ratio must be at least one for "
                    + names[i] + ".");
            }

            d_boundary_layer_specs.push_back(
                BoundaryLayerSpec{names[i], counts[i], first_heights[i],
                                  growth_ratios[i]});
        }

        validate_boundary_layer_names();
    }
}

const MeshFactory::BoundaryLayerSpec* MeshFactory::boundary_layer_spec(
    const std::string& boundary_name) const
{
    const auto iter =
        std::find_if(d_boundary_layer_specs.begin(), d_boundary_layer_specs.end(),
                     [&](const BoundaryLayerSpec& spec)
                     { return spec.boundary_name == boundary_name; });
    return iter == d_boundary_layer_specs.end() ? nullptr : &*iter;
}

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
    auto mesh = std::make_shared<STKMesh<Pack>>();

    if (d_domain_type == DomainType::BOX)
    {
        build_box_mesh(mesh);
    }
    else if (d_domain_type == DomainType::CYLINDER)
    {
        build_cylinder_mesh(mesh);
    }
    else if (d_domain_type == DomainType::SPHERE)
    {
        build_sphere_mesh(mesh);
    }
    else if (d_domain_type == DomainType::EXTERNAL)
    {
        mesh = std::make_shared<STKMesh<Pack>>(d_external_mesh_file);
        mesh->assemble();
    }
    else
    {
        throw std::runtime_error("Unsupported domain type for MeshFactory::build");
    }
    return mesh;
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
        for (std::size_t i = 1; i < edges.size(); ++i)
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

    const std::size_t num_cells_x = box_cell_edges[X].size() - 1;
    const std::size_t num_cells_y = box_cell_edges[Y].size() - 1;
    const std::size_t num_cells_z = box_cell_edges[Z].size() - 1;

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
    for (std::size_t i = 0; i < boundary_parts.size(); ++i)
    {
        boundary_parts[i] = declare_boundary_part(d_domain_exterior_face_types[i]);
    }

    auto node_id = [=](std::size_t i, std::size_t j, std::size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + (num_cells_x + 1) * (j + (num_cells_y + 1) * k));
    };

    auto element_id = [=](std::size_t i, std::size_t j, std::size_t k)
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

    bulk->modification_begin();

    for (std::size_t k = 0; k < num_cells_z; ++k)
    {
        for (std::size_t j = 0; j < num_cells_y; ++j)
        {
            for (std::size_t i = 0; i < num_cells_x; ++i)
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
                    *bulk, hex_part, element_id(i, j, k), hex_nodes);

                if (i == 0)               declare_boundary_side(elem, 3, boundary_parts[0]);
                if (i + 1 == num_cells_x) declare_boundary_side(elem, 1, boundary_parts[1]);
                if (j == 0)               declare_boundary_side(elem, 0, boundary_parts[2]);
                if (j + 1 == num_cells_y) declare_boundary_side(elem, 2, boundary_parts[3]);
                if (k == 0)               declare_boundary_side(elem, 4, boundary_parts[4]);
                if (k + 1 == num_cells_z) declare_boundary_side(elem, 5, boundary_parts[5]);
            }
        }
    }

    for (std::size_t k = 0; k <= num_cells_z; ++k)
    {
        for (std::size_t j = 0; j <= num_cells_y; ++j)
        {
            for (std::size_t i = 0; i <= num_cells_x; ++i)
            {
                const auto node = bulk->get_entity(stk::topology::NODE_RANK,
                                                    node_id(i, j, k));
                double* coord = stk::mesh::field_data(coord_field, node);
                if (coord == nullptr)
                {
                    throw std::runtime_error("BOX MeshFactory failed to write node coordinates.");
                }

                coord[0] = box_cell_edges[X][i];
                coord[1] = box_cell_edges[Y][j];
                coord[2] = box_cell_edges[Z][k];
            }
        }
    }

    bulk->modification_end();
    mesh->assemble();
}

/**
 * @brief Build a wedge mesh for a cylindrical domain.
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
    if (d_domain_exterior_face_types.size() < 3)
    {
        throw std::runtime_error("CYLINDER MeshFactory requires boundary names {radial,zmin,zmax}.");
    }

    constexpr real_t pi = 3.141592653589793238462643383279502884;
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
        for (std::size_t ring = 0; ring <= base_radial_count; ++ring)
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
        for (std::size_t layer = 0; layer <= base_height_count; ++layer)
        {
            z_edges.push_back(
                d_cylinder_height * static_cast<real_t>(layer)
              / static_cast<real_t>(base_height_count));
        }
    }

    const auto radial_count = radial_edges.size() - 1;
    const auto height_count = z_edges.size() - 1;
    const auto angular_count = std::max<std::size_t>(
        8, static_cast<std::size_t>(std::ceil(2.0 * pi * d_radius / d_mesh_size)));
    const auto nodes_per_layer = 1 + radial_count * angular_count;

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

    auto node_id = [=](std::size_t layer,
                       std::size_t ring,
                       std::size_t sector) -> stk::mesh::EntityId
    {
        const auto layer_offset = layer * nodes_per_layer;
        if (ring == 0)
        {
            return static_cast<stk::mesh::EntityId>(1 + layer_offset);
        }

        return static_cast<stk::mesh::EntityId>(
            1 + layer_offset + 1 + (ring - 1) * angular_count
            + (sector % angular_count));
    };

    std::unordered_map<stk::mesh::EntityId, FactoryNodeTag> node_tags;
    node_tags.reserve((height_count + 1) * nodes_per_layer);

    bulk->modification_begin();

    stk::mesh::EntityId next_element_id = 1;
    auto declare_wedge = [&](std::size_t layer,
                             const std::array<stk::mesh::EntityId, 3>& bottom_nodes)
    {
        const stk::mesh::EntityIdVector wedge_nodes{
            bottom_nodes[0], bottom_nodes[1], bottom_nodes[2],
            bottom_nodes[0] + static_cast<stk::mesh::EntityId>(nodes_per_layer),
            bottom_nodes[1] + static_cast<stk::mesh::EntityId>(nodes_per_layer),
            bottom_nodes[2] + static_cast<stk::mesh::EntityId>(nodes_per_layer)
        };

        const auto elem =
            stk::mesh::declare_element(*bulk, wedge_part, next_element_id++, wedge_nodes);

        declare_tagged_boundary_sides(
            *bulk, elem, stk::topology::WEDGE_6, wedge_nodes, node_tags,
            [&](const std::vector<FactoryNodeTag>& tags) -> stk::mesh::Part*
            {
                const auto all_layer = [&](std::size_t layer_id)
                {
                    return std::all_of(tags.begin(), tags.end(),
                                       [=](const FactoryNodeTag& tag)
                                       { return tag.layer == layer_id; });
                };
                const auto all_ring = [&](std::size_t ring_id)
                {
                    return std::all_of(tags.begin(), tags.end(),
                                       [=](const FactoryNodeTag& tag)
                                       { return tag.ring == ring_id; });
                };

                if (all_layer(0)) return zmin_part;
                if (all_layer(height_count)) return zmax_part;
                if (all_ring(radial_count)) return radial_part;
                return nullptr;
            });
    };

    for (std::size_t layer = 0; layer <= height_count; ++layer)
    {
        node_tags.emplace(node_id(layer, 0, 0), FactoryNodeTag{0, layer, false});
        for (std::size_t ring = 1; ring <= radial_count; ++ring)
        {
            for (std::size_t sector = 0; sector < angular_count; ++sector)
            {
                node_tags.emplace(node_id(layer, ring, sector),
                                  FactoryNodeTag{ring, layer,
                                                 ring == radial_count});
            }
        }
    }

    for (std::size_t layer = 0; layer < height_count; ++layer)
    {
        for (std::size_t sector = 0; sector < angular_count; ++sector)
        {
            const auto next_sector = (sector + 1) % angular_count;

            declare_wedge(layer, {node_id(layer, 0, 0),
                                  node_id(layer, 1, sector),
                                  node_id(layer, 1, next_sector)});

            for (std::size_t ring = 1; ring < radial_count; ++ring)
            {
                declare_wedge(layer, {node_id(layer, ring, sector),
                                      node_id(layer, ring + 1, sector),
                                      node_id(layer, ring + 1, next_sector)});
                declare_wedge(layer, {node_id(layer, ring, sector),
                                      node_id(layer, ring + 1, next_sector),
                                      node_id(layer, ring, next_sector)});
            }
        }
    }

    for (std::size_t layer = 0; layer <= height_count; ++layer)
    {
        const auto z = z_edges[layer];
        for (std::size_t ring = 0; ring <= radial_count; ++ring)
        {
            const auto radius = radial_edges[ring];
            const auto sector_count = ring == 0 ? 1 : angular_count;
            for (std::size_t sector = 0; sector < sector_count; ++sector)
            {
                const auto theta = 2.0 * pi
                                 * static_cast<real_t>(sector)
                                 / static_cast<real_t>(angular_count);
                const auto node = bulk->get_entity(stk::topology::NODE_RANK,
                                                   node_id(layer, ring, sector));
                double* coord = stk::mesh::field_data(coord_field, node);
                if (coord == nullptr)
                {
                    throw std::runtime_error("CYLINDER MeshFactory failed to write node coordinates.");
                }

                coord[0] = radius * std::cos(theta);
                coord[1] = radius * std::sin(theta);
                coord[2] = z;
            }
        }
    }

    bulk->modification_end();
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
        for (std::size_t index = 0; index <= cell_count; ++index)
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

    auto node_id = [=](std::size_t i, std::size_t j, std::size_t k)
        -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + (cell_count + 1) * (j + (cell_count + 1) * k));
    };

    std::unordered_map<stk::mesh::EntityId, FactoryNodeTag> node_tags;
    node_tags.reserve((cell_count + 1) * (cell_count + 1) * (cell_count + 1));
    for (std::size_t k = 0; k <= cell_count; ++k)
    {
        for (std::size_t j = 0; j <= cell_count; ++j)
        {
            for (std::size_t i = 0; i <= cell_count; ++i)
            {
                const bool surface = i == 0 || i == cell_count
                                  || j == 0 || j == cell_count
                                  || k == 0 || k == cell_count;
                node_tags.emplace(node_id(i, j, k), FactoryNodeTag{0, k, surface});
            }
        }
    }

    bulk->modification_begin();

    for (std::size_t k = 0; k < cell_count; ++k)
    {
        for (std::size_t j = 0; j < cell_count; ++j)
        {
            for (std::size_t i = 0; i < cell_count; ++i)
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
                    static_cast<stk::mesh::EntityId>(
                        1 + i + cell_count * (j + cell_count * k)),
                    hex_nodes);

                declare_tagged_boundary_sides(
                    *bulk, elem, stk::topology::HEX_8, hex_nodes, node_tags,
                    [=](const std::vector<FactoryNodeTag>& tags) -> stk::mesh::Part*
                    {
                        const auto all_surface =
                            std::all_of(tags.begin(), tags.end(),
                                        [](const FactoryNodeTag& tag)
                                        { return tag.surface; });
                        if (!all_surface)
                        {
                            return nullptr;
                        }
                        if (!split_surface)
                        {
                            return lower_surface_part;
                        }

                        std::size_t layer_sum = 0;
                        for (const auto& tag : tags)
                        {
                            layer_sum += tag.layer;
                        }
                        const auto lower_side =
                            layer_sum <= (cell_count * tags.size()) / 2;
                        return lower_side ? lower_surface_part : upper_surface_part;
                    });
            }
        }
    }

    for (std::size_t k = 0; k <= cell_count; ++k)
    {
        for (std::size_t j = 0; j <= cell_count; ++j)
        {
            for (std::size_t i = 0; i <= cell_count; ++i)
            {
                const real_t u = sphere_edges[i];
                const real_t v = sphere_edges[j];
                const real_t w = sphere_edges[k];
                const auto norm = std::sqrt(u * u + v * v + w * w);
                const auto cube_radius = std::max({std::abs(u), std::abs(v), std::abs(w)});
                const auto scale = norm == 0.0 ? 0.0 : d_radius * cube_radius / norm;

                const auto node = bulk->get_entity(stk::topology::NODE_RANK,
                                                   node_id(i, j, k));
                double* coord = stk::mesh::field_data(coord_field, node);
                if (coord == nullptr)
                {
                    throw std::runtime_error("SPHERE MeshFactory failed to write node coordinates.");
                }

                coord[0] = u * scale;
                coord[1] = v * scale;
                coord[2] = w * scale;
            }
        }
    }

    bulk->modification_end();
    mesh->assemble();
}

template SP<Mesh<DefaultTpetraTypes>> MeshFactory::build<DefaultTpetraTypes>();

} 
