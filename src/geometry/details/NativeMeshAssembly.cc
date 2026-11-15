#include "NativeMeshAssembly.hh"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{
namespace
{
using Data = Meshes::UnstructuredMeshData;
using ID = Data::ID;
using CellType = MeshUtils::CellType;
} // namespace

const std::vector<std::vector<unsigned>>& native_mesh_side_node_ordinals(CellType type)
{
    // Exact source HEX_8/WEDGE_6 face orders retain the quad diagonal.
    static const std::vector<std::vector<unsigned>> hex{
        {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {0, 4, 7, 3}, {0, 3, 2, 1}, {4, 5, 6, 7}};
    static const std::vector<std::vector<unsigned>> wedge{
        {0, 1, 4, 3}, {1, 2, 5, 4}, {0, 3, 5, 2}, {0, 2, 1}, {3, 4, 5}};
    if (type == CellType::HEXAHEDRON)
        return hex;
    if (type == CellType::TRIPRISM)
        return wedge;
    throw std::invalid_argument("Native generator requires a HEX_8 or WEDGE_6 cell.");
}

Data build_native_mesh_data(std::vector<Data::Node> nodes, std::vector<Data::Cell> cells,
    const std::vector<NativeBoundarySide>& boundaries, const ArrString& boundary_names)
{
    Data result;
    result.nodes = std::move(nodes);
    result.cells = std::move(cells);
    std::map<ID, Data::Vec3> coordinates;
    std::map<std::pair<ID, unsigned>, std::string> labels;
    for (const auto& node : result.nodes)
        if (!coordinates.emplace(node.id, node.coordinates).second)
            throw std::invalid_argument("Native builder received duplicate node IDs.");
    for (const auto& boundary : boundaries)
        labels.emplace(std::pair{boundary.cell_id, boundary.side}, boundary.boundary_name);
    std::map<std::vector<ID>, size_t> face_index;
    std::vector<std::string> face_labels;
    std::map<ID, size_t> cell_index;
    for (size_t c = 0; c < result.cells.size(); ++c)
    {
        auto& cell = result.cells[c];
        if (!cell_index.emplace(cell.id, c).second)
            throw std::invalid_argument("Native builder received duplicate cell IDs.");
        std::vector<Data::Vec3> points;
        for (auto n : cell.node_ids)
            points.push_back(coordinates.at(n));
        if (cell.node_ids.size() != (cell.type == CellType::HEXAHEDRON ? 8 : 6))
            throw std::invalid_argument("Native builder cell node count does not match topology.");
        cell.volume =
            cell.type == CellType::HEXAHEDRON ? MeshUtils::hex_volume(points) : MeshUtils::wedge_volume(points);
        cell.center =
            cell.type == CellType::HEXAHEDRON ? MeshUtils::hex_centroid(points) : MeshUtils::wedge_centroid(points);
        cell.face_ids.clear();
        for (unsigned s = 0; s < native_mesh_side_node_ordinals(cell.type).size(); ++s)
        {
            if (cell.id > (Data::invalid_id - 6) / 10)
                throw std::overflow_error("Native builder side ID overflow.");
            const auto candidate = cell.id * 10 + s + 1;
            std::vector<ID> ids;
            for (auto ordinal : native_mesh_side_node_ordinals(cell.type)[s])
                ids.push_back(cell.node_ids[ordinal]);
            auto key = ids;
            std::sort(key.begin(), key.end());
            const auto [it, inserted] = face_index.emplace(key, result.faces.size());
            const auto label = labels.find({cell.id, s});
            if (inserted)
            {
                Data::Face f;
                f.id = candidate;
                f.node_ids = std::move(ids);
                f.owner = cell.id;
                result.faces.push_back(std::move(f));
                face_labels.push_back(label == labels.end() ? std::string{} : label->second);
            }
            else
            {
                auto& face = result.faces[it->second];
                if (face.neighbor != Data::invalid_id)
                    throw std::invalid_argument("Non-manifold native face.");
                face.neighbor = cell.id;
                face.id = std::min(face.id, candidate);
            }
        }
    }
    std::map<std::string, int> boundary_ids;
    auto boundary_id = [&](const std::string& name)
    {
        if (name.empty())
            throw std::invalid_argument("Native boundary name cannot be empty.");
        const auto [it, inserted] = boundary_ids.emplace(name, static_cast<int>(boundary_ids.size() + 1));
        if (inserted)
            result.boundary_names.emplace(it->second, name);
        return it->second;
    };
    for (size_t f = 0; f < result.faces.size(); ++f)
    {
        auto& face = result.faces[f];
        std::vector<Data::Vec3> points;
        for (auto n : face.node_ids)
            points.push_back(coordinates.at(n));
        face.center = MeshUtils::face_centroid(points);
        auto area_vector = MeshUtils::face_area_vector(points);
        face.area = area_vector.norm();
        if (!(face.area > 0))
            throw std::invalid_argument("Degenerate native face.");
        face.normal_from_owner = area_vector / face.area;
        if (face.normal_from_owner.dot(face.center - result.cells[cell_index.at(face.owner)].center) < 0)
            face.normal_from_owner = face.normal_from_owner * -1.0;
        if (face.neighbor == Data::invalid_id && !face_labels[f].empty())
            face.boundary_id = boundary_id(face_labels[f]);
    }
    // Revisit topology order after canonical shared-face IDs have settled.
    for (auto& cell : result.cells)
        for (const auto& ordinals : native_mesh_side_node_ordinals(cell.type))
        {
            std::vector<ID> key;
            for (auto ordinal : ordinals)
                key.push_back(cell.node_ids[ordinal]);
            std::sort(key.begin(), key.end());
            cell.face_ids.push_back(result.faces[face_index.at(key)].id);
        }
    for (const auto& name : boundary_names)
        boundary_id(name);
    return result;
}
} // namespace SimpleFluid::detail
