/**
 * @file test_skewed_prism_mesh_helpers.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared test helper for structured skewed triangular-prism meshes.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "geometry/STKMesh.hh"

#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::test
{

namespace detail
{

template<TpetraTypePack Pack>
stk::mesh::Field<double>& declare_skewed_prism_coordinate_field(
    STKMesh<Pack>& mesh)
{
    auto meta = mesh.meta();
    auto& coord_field =
        meta->template declare_field<double>(
            stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coord_field, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coord_field);
    return coord_field;
}

inline vec3<> skewed_prism_coord(double x, double y, double z)
{
    return {x + 0.35 * y + 0.2 * z,
            y + 0.15 * z,
            z};
}

/** @brief Integer coordinates of a node in the source structured grid. */
struct LogicalNode
{
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
};

} // namespace detail

/**
 * @brief Build a skewed WEDGE_6 mesh by splitting each structured
 *        XY rectangle into two triangular prisms.
 *
 * The default 3 x 3 x 3 grid creates 54 prism cells, which is at least
 * 27 cells and contains both fully interior and exterior-adjacent cells.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
SP<Mesh<Pack>> make_skewed_prism_mesh(size_t n_x = 3,
                                      size_t n_y = 3,
                                      size_t n_z = 3)
{
    if (n_x < 3 || n_y < 3 || n_z < 3)
    {
        throw std::invalid_argument(
            "make_skewed_prism_mesh requires at least 3 cells per direction.");
    }

    auto mesh = std::make_shared<STKMesh<Pack>>();
    auto& coord_field =
        detail::declare_skewed_prism_coordinate_field(*mesh);
    auto meta = mesh->meta();
    auto bulk = mesh->bulk();

    auto& prism_part =
        meta->declare_part_with_topology("skewed_prisms",
                                         stk::topology::WEDGE_6);

    std::array<stk::mesh::Part*, 6> boundary_parts{};
    const std::array<const char*, 6> boundary_names{
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    for (size_t i = 0; i < boundary_parts.size(); ++i)
    {
        auto& part = meta->declare_part(boundary_names[i], meta->side_rank());
        boundary_parts[i] = &part;
    }

    auto node_id = [=](size_t i,
                       size_t j,
                       size_t k) -> stk::mesh::EntityId
    {
        return static_cast<stk::mesh::EntityId>(
            1 + i + (n_x + 1) * (j + (n_y + 1) * k));
    };

    auto logical_node = [=](stk::mesh::EntityId id)
        -> detail::LogicalNode
    {
        auto offset = static_cast<size_t>(id - 1);
        const auto i = offset % (n_x + 1);
        offset /= (n_x + 1);
        const auto j = offset % (n_y + 1);
        const auto k = offset / (n_y + 1);
        return {i, j, k};
    };

    bulk->modification_begin();

    stk::mesh::EntityId element_id = 1;
    auto declare_prism =
        [&](stk::mesh::EntityId n0,
            stk::mesh::EntityId n1,
            stk::mesh::EntityId n2,
            stk::mesh::EntityId n3,
            stk::mesh::EntityId n4,
            stk::mesh::EntityId n5)
    {
        const stk::mesh::EntityIdVector prism_nodes{n0, n1, n2, n3, n4, n5};
        const auto elem = stk::mesh::declare_element(
            *bulk, prism_part, element_id++, prism_nodes);

        const stk::topology topo(stk::topology::WEDGE_6);
        for (unsigned side = 0; side < topo.num_sides(); ++side)
        {
            std::vector<unsigned> ordinals(
                topo.side_topology(side).num_nodes());
            topo.side_node_ordinals(side, ordinals.begin());

            auto all_on_side =
                [&](auto predicate)
            {
                for (const auto ordinal : ordinals)
                {
                    if (!predicate(logical_node(prism_nodes[ordinal])))
                    {
                        return false;
                    }
                }
                return true;
            };

            stk::mesh::Part* boundary_part = nullptr;
            if (all_on_side([](const auto& node) { return node.k == 0; }))
            {
                boundary_part = boundary_parts[4];
            }
            else if (all_on_side(
                         [=](const auto& node) { return node.k == n_z; }))
            {
                boundary_part = boundary_parts[5];
            }
            else if (all_on_side(
                         [](const auto& node) { return node.i == 0; }))
            {
                boundary_part = boundary_parts[0];
            }
            else if (all_on_side(
                         [=](const auto& node) { return node.i == n_x; }))
            {
                boundary_part = boundary_parts[1];
            }
            else if (all_on_side(
                         [](const auto& node) { return node.j == 0; }))
            {
                boundary_part = boundary_parts[2];
            }
            else if (all_on_side(
                         [=](const auto& node) { return node.j == n_y; }))
            {
                boundary_part = boundary_parts[3];
            }

            if (boundary_part != nullptr)
            {
                stk::mesh::PartVector parts{boundary_part};
                bulk->declare_element_side(elem, side, parts);
            }
        }
    };

    for (size_t k = 0; k < n_z; ++k)
    {
        for (size_t j = 0; j < n_y; ++j)
        {
            for (size_t i = 0; i < n_x; ++i)
            {
                declare_prism(
                    node_id(i,     j,     k),
                    node_id(i + 1, j,     k),
                    node_id(i,     j + 1, k),
                    node_id(i,     j,     k + 1),
                    node_id(i + 1, j,     k + 1),
                    node_id(i,     j + 1, k + 1));
                declare_prism(
                    node_id(i + 1, j,     k),
                    node_id(i + 1, j + 1, k),
                    node_id(i,     j + 1, k),
                    node_id(i + 1, j,     k + 1),
                    node_id(i + 1, j + 1, k + 1),
                    node_id(i,     j + 1, k + 1));
            }
        }
    }

    for (size_t k = 0; k <= n_z; ++k)
    {
        for (size_t j = 0; j <= n_y; ++j)
        {
            for (size_t i = 0; i <= n_x; ++i)
            {
                const auto x = static_cast<double>(i)
                             / static_cast<double>(n_x);
                const auto y = static_cast<double>(j)
                             / static_cast<double>(n_y);
                const auto z = static_cast<double>(k)
                             / static_cast<double>(n_z);
                const auto node =
                    bulk->get_entity(stk::topology::NODE_RANK,
                                     node_id(i, j, k));
                auto* data = stk::mesh::field_data(coord_field, node);
                const auto coord = detail::skewed_prism_coord(x, y, z);
                data[0] = coord.x;
                data[1] = coord.y;
                data[2] = coord.z;
            }
        }
    }

    bulk->modification_end();
    mesh->assemble();
    return mesh;
}

} // namespace SimpleFluid::test
