/** @file BenchmarkMesh.hh
 * @brief Native sheared HEX geometry used by the diffusion benchmark.
 */
#pragma once
#include "geometry/details/NativeMeshGenerators.hh"

#include <array>
#include <map>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail::benchmark_mesh
{
inline Meshes::UnstructuredMeshData generate(const std::array<int, 3>& dimensions, double shear)
{
    using Data = Meshes::UnstructuredMeshData;
    auto unit_edges = [](int count)
    {
        if (count <= 0)
            throw std::invalid_argument("Benchmark mesh dimensions must be positive.");
        ArrReal edges(static_cast<size_t>(count) + 1);
        for (int i = 0; i <= count; ++i)
            edges[static_cast<size_t>(i)] = static_cast<double>(i) / count;
        return edges;
    };
    SimpleFluid::detail::NativeMeshGeneratorOptions options;
    options.box_cell_edges = {unit_edges(dimensions[0]), unit_edges(dimensions[1]), unit_edges(dimensions[2])};
    options.domain_exterior_face_types = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    auto data = SimpleFluid::detail::generate_native_mesh(options);
    std::vector<SimpleFluid::detail::NativeBoundarySide> boundaries;
    std::map<Data::ID, const Data::Face*> faces;
    for (const auto& face : data.faces)
        faces.emplace(face.id, &face);
    for (const auto& cell : data.cells)
        for (unsigned side = 0; side < cell.face_ids.size(); ++side)
        {
            const auto& face = *faces.at(cell.face_ids[side]);
            if (face.boundary_id >= 0)
                boundaries.push_back({cell.id, side, data.boundary_names.at(face.boundary_id)});
        }
    for (auto& node : data.nodes)
        node.coordinates.x += shear * node.coordinates.y;
    return SimpleFluid::detail::build_native_mesh_data(
        std::move(data.nodes), std::move(data.cells), boundaries, options.domain_exterior_face_types);
}
} // namespace SimpleFluid::detail::benchmark_mesh
