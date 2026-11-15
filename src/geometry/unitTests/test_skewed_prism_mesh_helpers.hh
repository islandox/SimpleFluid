/** @file test_skewed_prism_mesh_helpers.hh
 * @brief Native skewed WEDGE_6 fixtures with source IDs and ordered side geometry.
 */
#pragma once

#include "geometry/MeshHandle.hh"
#include "geometry/details/NativeMeshAssembly.hh"
#include "parallel/CollectiveConstruction.hh"
#include "parallel/MeshPartitioner.hh"

#include <array>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::test
{
/** Build the established affine-skewed prism source directly as native records.
 * Cells/nodes retain the one-based source IDs and HEX/WEDGE side numbering used
 * by the independent frozen STK oracle. Distribution retains factory numbering.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
SP<MeshHandle<Pack>> make_skewed_prism_mesh(size_t n_x = 3, size_t n_y = 3, size_t n_z = 3)
{
    using Mesh = MeshHandle<Pack>;
    using Data = Meshes::UnstructuredMeshData;
    const auto comm = Tpetra::getDefaultComm();
    Data source;
    std::exception_ptr error;
    try
    {
        if (n_x < 3 || n_y < 3 || n_z < 3)
            throw std::invalid_argument("make_skewed_prism_mesh requires at least 3 cells per direction.");
        if (comm->getRank() == 0)
        {
            const auto node_id = [=](size_t i, size_t j, size_t k) -> Data::ID
            { return 1 + i + (n_x + 1) * (j + (n_y + 1) * k); };
            std::vector<Data::Node> nodes;
            for (size_t k = 0; k <= n_z; ++k)
                for (size_t j = 0; j <= n_y; ++j)
                    for (size_t i = 0; i <= n_x; ++i)
                    {
                        const auto x = static_cast<double>(i) / static_cast<double>(n_x);
                        const auto y = static_cast<double>(j) / static_cast<double>(n_y);
                        const auto z = static_cast<double>(k) / static_cast<double>(n_z);
                        nodes.push_back({node_id(i, j, k), {x + 0.35 * y + 0.2 * z, y + 0.15 * z, z}});
                    }
            std::vector<Data::Cell> cells;
            std::vector<SimpleFluid::detail::NativeBoundarySide> boundaries;
            const ArrString names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
            const auto logical_node = [=](Data::ID id)
            {
                const auto offset = static_cast<size_t>(id - 1);
                return std::array<size_t, 3>{
                    offset % (n_x + 1), offset / (n_x + 1) % (n_y + 1), offset / ((n_x + 1) * (n_y + 1))};
            };
            const auto append = [&](std::vector<Data::ID> ids)
            {
                Data::Cell cell;
                cell.id = cells.size() + 1;
                cell.type = MeshUtils::CellType::TRIPRISM;
                cell.node_ids = std::move(ids);
                const auto& sides = SimpleFluid::detail::native_mesh_side_node_ordinals(cell.type);
                for (unsigned side = 0; side < sides.size(); ++side)
                    for (size_t patch = 0; patch < names.size(); ++patch)
                    {
                        const auto axis = patch / 2;
                        const auto plane = patch % 2 ? std::array{n_x, n_y, n_z}[axis] : size_t{};
                        bool on_plane = true;
                        for (auto ordinal : sides[side])
                            on_plane = on_plane && logical_node(cell.node_ids[ordinal])[axis] == plane;
                        if (on_plane)
                        {
                            boundaries.push_back({cell.id, side, names[patch]});
                            break;
                        }
                    }
                cells.push_back(std::move(cell));
            };
            for (size_t k = 0; k < n_z; ++k)
                for (size_t j = 0; j < n_y; ++j)
                    for (size_t i = 0; i < n_x; ++i)
                    {
                        append({node_id(i, j, k), node_id(i + 1, j, k), node_id(i, j + 1, k), node_id(i, j, k + 1),
                            node_id(i + 1, j, k + 1), node_id(i, j + 1, k + 1)});
                        append({node_id(i + 1, j, k), node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                            node_id(i + 1, j, k + 1), node_id(i + 1, j + 1, k + 1), node_id(i, j + 1, k + 1)});
                    }
            source = SimpleFluid::detail::build_native_mesh_data(std::move(nodes), std::move(cells), boundaries, names);
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (error)
        std::rethrow_exception(error);
    if (any_failed)
        throw std::runtime_error("Native skewed prism construction failed on another rank.");
    auto partition = MeshPartitioner<Pack>::partition_native(std::move(source), comm, 1, true);
    return SimpleFluid::detail::collective_make_shared<Mesh>(
        *comm, std::move(partition.geometry), partition.indexer, comm);
}
} // namespace SimpleFluid::test
