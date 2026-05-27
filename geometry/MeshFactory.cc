/**
 * @file MeshFactory.cc
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief 
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

#include <array>
#include <stdexcept>
#include <unordered_map>

namespace SimpleFluid
{
/**
 * @brief Construct a MeshFactory from a configuration database.
 *
 * @param db Shared database containing mesh configuration entries.
 */
MeshFactory::MeshFactory(SP<const Database>& db)
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
        return mesh;
    }
    else if (d_domain_type == DomainType::CYLINDER)
    {
        throw std::runtime_error("CYLINDER MeshFactory generation is not implemented.");
    }
    else if (d_domain_type == DomainType::SPHERE)
    {
        throw std::runtime_error("SPHERE MeshFactory generation is not implemented.");
    }
    else if (d_domain_type == DomainType::EXTERNAL)
    {
        mesh = std::make_shared<STKMesh<Pack>>(d_external_mesh_file);
        mesh->assemble();
        return mesh;
    }
    else
    {
        throw std::runtime_error("Unsupported domain type for MeshFactory::build");
    }
}

/**
 * @brief Build a structured hexahedral mesh for a BOX domain.
 *
 * @tparam Pack Tpetra type pack used for mesh storage and communication.
 * @param mesh Mesh instance to build into.
 * @throws std::runtime_error if the domain is not 3D or boundary metadata is invalid.
 */
template <TpetraTypePack Pack>
void MeshFactory::build_box_mesh(SP<Mesh<Pack>>& mesh)
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

    const std::size_t num_cells_x = d_box_cell_edges[X].size() - 1;
    const std::size_t num_cells_y = d_box_cell_edges[Y].size() - 1;
    const std::size_t num_cells_z = d_box_cell_edges[Z].size() - 1;

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

                coord[0] = d_box_cell_edges[X][i];
                coord[1] = d_box_cell_edges[Y][j];
                coord[2] = d_box_cell_edges[Z][k];
            }
        }
    }

    bulk->modification_end();
    mesh->assemble();
}

template SP<Mesh<DefaultTpetraTypes>> MeshFactory::build<DefaultTpetraTypes>();

} 
