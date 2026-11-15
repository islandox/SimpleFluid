/** @file LegacyMeshSnapshot.tcc
 * @brief STK-free materialization of the explicit legacy compatibility snapshot.
 */
#pragma once
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "geometry/details/LegacyMeshSnapshot.hh"
#include "io/VTUWriter.hh"
#include "parallel/CollectiveConstruction.hh"
#include "parallel/MeshPartitioner.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace SimpleFluid::detail
{
namespace legacy_snapshot_detail
{
template<class Operation, class Comm> void collective_stage(const Comm& comm, Operation&& operation)
{
    std::exception_ptr error;
    try
    {
        std::forward<Operation>(operation)();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("Legacy mesh snapshot materialization failed on another rank.");
    }
}
} // namespace legacy_snapshot_detail

/** @brief Internal concrete Mesh; source-independent construction is one-shot. */
template<TpetraTypePack Pack> class LegacyMeshSnapshot final : public Mesh<Pack>
{
    using Base = Mesh<Pack>;
    using Data = Meshes::UnstructuredMeshData;
    using Base::check_connectivity;
    using Base::d_cells;
    using Base::d_faces;
    using Base::d_node_coords;
    using Base::d_owned_cell_global_ids;
    using typename Base::CellType;
    using typename Base::FaceType;
    using typename Base::global_ordinal_type;
    using typename Base::local_ordinal_type;
    using typename Base::ViewGO;
    using typename Base::ViewLO;

public:
    explicit LegacyMeshSnapshot(Data data)
    {
        const auto comm = Tpetra::getDefaultComm();
        legacy_snapshot_detail::collective_stage(*comm, [&] { load(data); });
        // The compatibility arrays now own all records. Release detached input
        // before partition packet exchange reaches its peak storage use.
        data = Data{};
        synchronize_boundary_catalog(*comm);
        const int has_cells = d_cells.empty() ? 0 : 1;
        int any_cells = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &has_cells, &any_cells);
        if (any_cells)
            MeshPartitioner<Pack>::partition(*this, comm);
        legacy_snapshot_detail::collective_stage(*comm,
            [&]
            {
                this->prefer_owned_face_owners();
                this->create_cell_face_distances();
                this->rebuild_boundary_face_batches();
                for (const auto& [id, name] : this->d_boundary_id_to_name)
                    this->d_boundary_id_to_face_batch[id].id = id;
                check_connectivity();
            });
        this->create_maps();
    }

    /** @brief Already finalized at construction; no retained input needs assembly. */
    void assemble() override {}
    void export_vtu(const std::string& filename) const override;

private:
    template<class Ordinal> static Ordinal checked_ordinal(auto value)
    {
        if (!std::in_range<Ordinal>(value))
            throw std::overflow_error("Legacy mesh snapshot entity ID or count overflow.");
        return static_cast<Ordinal>(value);
    }

    static size_t owned_count(size_t owned, size_t total)
    {
        if (owned == Data::all_owned)
            return total;
        if (owned > total)
            throw std::invalid_argument("Legacy mesh snapshot owned count exceeds local count.");
        return owned;
    }

    static bool finite(const typename Base::Vec3& value)
    {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    }

    void load(const Data& data)
    {
        this->d_spatial_dim = 3;
        checked_ordinal<local_ordinal_type>(data.cells.size());
        checked_ordinal<local_ordinal_type>(data.faces.size());
        checked_ordinal<local_ordinal_type>(data.nodes.size());
        const auto owned_cells = owned_count(data.num_owned_cells, data.cells.size());
        static_cast<void>(owned_count(data.num_owned_faces, data.faces.size()));
        static_cast<void>(owned_count(data.num_owned_nodes, data.nodes.size()));
        this->d_boundary_id_to_name = data.boundary_names;
        for (const auto& [id, name] : data.boundary_names)
        {
            if (id < 0)
                throw std::invalid_argument("Legacy mesh snapshot has an invalid boundary ID.");
            this->d_boundary_name_to_id.emplace(name, id);
        }
        d_node_coords.reserve(data.nodes.size());
        for (const auto& node : data.nodes)
        {
            const auto gid = checked_ordinal<global_ordinal_type>(node.id);
            if (!finite(node.coordinates) || node.id == Data::invalid_id ||
                !this->d_node_gid_to_lid.emplace(gid, checked_ordinal<local_ordinal_type>(d_node_coords.size())).second)
                throw std::invalid_argument("Legacy mesh snapshot has invalid or duplicate nodes.");
            d_node_coords.push_back(node.coordinates);
        }
        std::unordered_map<Data::ID, local_ordinal_type> face_lids;
        face_lids.reserve(data.faces.size());
        for (size_t f = 0; f < data.faces.size(); ++f)
        {
            if (data.faces[f].id == Data::invalid_id ||
                !face_lids.emplace(data.faces[f].id, checked_ordinal<local_ordinal_type>(f)).second)
                throw std::invalid_argument("Legacy mesh snapshot repeats a face ID.");
            this->d_owned_face_global_ids.push_back(checked_ordinal<global_ordinal_type>(data.faces[f].id));
        }
        d_cells.resize(data.cells.size());
        for (size_t c = 0; c < data.cells.size(); ++c)
        {
            const auto gid = checked_ordinal<global_ordinal_type>(data.cells[c].id);
            if (data.cells[c].id == Data::invalid_id ||
                !this->d_cell_gid_to_lid.emplace(gid, checked_ordinal<local_ordinal_type>(c)).second)
                throw std::invalid_argument("Legacy mesh snapshot repeats a cell ID.");
            if (c < owned_cells)
            {
                this->d_owned_cell_ids.push_back(checked_ordinal<local_ordinal_type>(c));
                d_owned_cell_global_ids.push_back(gid);
            }
            else
                this->d_ghost_cell_global_ids.push_back(gid);
        }
        std::vector<size_t> cell_face_offsets, cell_node_offsets, face_node_offsets;
        cell_face_offsets.reserve(data.cells.size());
        cell_node_offsets.reserve(data.cells.size());
        face_node_offsets.reserve(data.faces.size());
        for (size_t c = 0; c < data.cells.size(); ++c)
        {
            const auto& input = data.cells[c];
            auto& cell = d_cells[c];
            if ((input.type != CellType::HEXAHEDRON && input.type != CellType::TRIPRISM) ||
                input.node_ids.size() != (input.type == CellType::HEXAHEDRON ? 8U : 6U) ||
                input.face_ids.size() != (input.type == CellType::HEXAHEDRON ? 6U : 5U) || !finite(input.center) ||
                !std::isfinite(input.volume) || input.volume <= 0)
                throw std::invalid_argument("Legacy mesh snapshot cell type, counts, or metrics are invalid.");
            cell.owned = c < owned_cells;
            cell.type = input.type;
            cell.center = input.center;
            cell.volume = input.volume;
            cell_face_offsets.push_back(this->d_cell_owned_face_ids.size());
            for (const auto id : input.face_ids)
                this->d_cell_owned_face_ids.push_back(face_lids.at(id));
            cell_node_offsets.push_back(this->d_cell_owned_node_global_ids.size());
            for (const auto id : input.node_ids)
            {
                const auto gid = checked_ordinal<global_ordinal_type>(id);
                if (!this->d_node_gid_to_lid.contains(gid))
                    throw std::invalid_argument("Legacy mesh snapshot cell node is unavailable.");
                this->d_cell_owned_node_global_ids.push_back(gid);
            }
        }
        d_faces.resize(data.faces.size());
        for (size_t f = 0; f < data.faces.size(); ++f)
        {
            const auto& input = data.faces[f];
            auto& face = d_faces[f];
            if ((input.node_ids.size() != 3 && input.node_ids.size() != 4) || !finite(input.center) ||
                !finite(input.normal_from_owner) || !std::isfinite(input.area) || input.area <= 0 ||
                !std::isfinite(input.normal_from_owner.norm()) || input.normal_from_owner.norm() <= 0 ||
                input.owner == input.neighbor)
                throw std::invalid_argument("Legacy mesh snapshot face connectivity or metrics are invalid.");
            face.type = input.node_ids.size() == 3 ? FaceType::TRIANGLE : FaceType::QUAD;
            face.owner = this->d_cell_gid_to_lid.at(checked_ordinal<global_ordinal_type>(input.owner));
            face.neighbor = invalid_id<local_ordinal_type>();
            if (input.neighbor != Data::invalid_id)
            {
                const auto found = this->d_cell_gid_to_lid.find(checked_ordinal<global_ordinal_type>(input.neighbor));
                if (found != this->d_cell_gid_to_lid.end())
                    face.neighbor = found->second;
                else if (static_cast<size_t>(face.owner) < owned_cells)
                    throw std::invalid_argument("Legacy mesh snapshot owned cell lacks its adjacent cell.");
            }
            if (input.boundary_id < Base::invalid_boundary_id ||
                (input.boundary_id != Base::invalid_boundary_id &&
                    (input.neighbor != Data::invalid_id || !data.boundary_names.contains(input.boundary_id))))
                throw std::invalid_argument("Legacy mesh snapshot face has an invalid boundary assignment.");
            face.boundary_id = input.boundary_id;
            face.center = input.center;
            face.area = input.area;
            face.unit_normal_from_owner = input.normal_from_owner;
            face.unit_normal_from_neighbor = input.normal_from_owner * -1.0;
            face.owner_to_face_distance = (face.center - d_cells[face.owner].center).norm();
            if (face.neighbor != invalid_id<local_ordinal_type>())
            {
                face.neighbor_to_face_distance = (face.center - d_cells[face.neighbor].center).norm();
                face.cell_center_distance = (d_cells[face.neighbor].center - d_cells[face.owner].center).norm();
            }
            face_node_offsets.push_back(this->d_face_owned_node_global_ids.size());
            for (const auto id : input.node_ids)
            {
                const auto gid = checked_ordinal<global_ordinal_type>(id);
                if (!this->d_node_gid_to_lid.contains(gid))
                    throw std::invalid_argument("Legacy mesh snapshot face node is unavailable.");
                this->d_face_owned_node_global_ids.push_back(gid);
            }
        }
        // Set non-owning CSR slices only after every backing vector is final.
        for (size_t c = 0; c < d_cells.size(); ++c)
        {
            d_cells[c].faces =
                ViewLO(this->d_cell_owned_face_ids.data() + cell_face_offsets[c], data.cells[c].face_ids.size());
            d_cells[c].node_gids =
                ViewGO(this->d_cell_owned_node_global_ids.data() + cell_node_offsets[c], data.cells[c].node_ids.size());
        }
        for (size_t f = 0; f < d_faces.size(); ++f)
            d_faces[f].node_gids =
                ViewGO(this->d_face_owned_node_global_ids.data() + face_node_offsets[f], data.faces[f].node_ids.size());
        for (size_t c = 0; c < d_cells.size(); ++c)
        {
            std::unordered_set<local_ordinal_type> incident;
            for (const auto f : d_cells[c].faces)
            {
                if (!incident.insert(f).second || (d_faces[f].owner != c && d_faces[f].neighbor != c))
                    throw std::invalid_argument("Legacy mesh snapshot has invalid cell-face incidence.");
            }
        }
        for (size_t f = 0; f < d_faces.size(); ++f)
            for (const auto c : {d_faces[f].owner, d_faces[f].neighbor})
            {
                if (c == invalid_id<local_ordinal_type>())
                    continue;
                const auto& incidence = d_cells[c].faces;
                if (std::find(incidence.begin(), incidence.end(), f) == incidence.end())
                    throw std::invalid_argument("Legacy mesh snapshot face is absent from an adjacent cell.");
            }
    }

    void synchronize_boundary_catalog(const typename Pack::comm_type& comm)
    {
        int source = this->d_boundary_id_to_name.empty() ? comm.getSize() : comm.getRank();
        int root = comm.getSize();
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &source, &root);
        if (root == comm.getSize())
            return;
        int count = 0;
        legacy_snapshot_detail::collective_stage(comm,
            [&]
            {
                count = checked_ordinal<int>(this->d_boundary_id_to_name.size());
                for (const auto& [id, name] : this->d_boundary_id_to_name)
                    static_cast<void>(checked_ordinal<int>(name.size()));
            });
        Teuchos::broadcast(comm, root, 1, &count);
        std::vector<int> ids;
        std::unordered_map<int, std::string> catalog;
        legacy_snapshot_detail::collective_stage(comm,
            [&]
            {
                ids.reserve(static_cast<size_t>(count));
                catalog.reserve(static_cast<size_t>(count));
                if (comm.getRank() == root)
                    for (const auto& [id, name] : this->d_boundary_id_to_name)
                        ids.push_back(id);
                else
                    ids.resize(static_cast<size_t>(count));
            });
        Teuchos::broadcast(comm, root, count, ids.data());
        for (const auto id : ids)
        {
            int length = 0;
            legacy_snapshot_detail::collective_stage(comm,
                [&]
                {
                    if (comm.getRank() == root)
                        length = checked_ordinal<int>(this->d_boundary_id_to_name.at(id).size());
                });
            Teuchos::broadcast(comm, root, 1, &length);
            std::string name;
            legacy_snapshot_detail::collective_stage(comm,
                [&]
                {
                    if (comm.getRank() == root)
                        name = this->d_boundary_id_to_name.at(id);
                    else
                        name.resize(static_cast<size_t>(length));
                });
            Teuchos::broadcast(comm, root, length, name.data());
            legacy_snapshot_detail::collective_stage(comm, [&] { catalog.emplace(id, std::move(name)); });
        }
        legacy_snapshot_detail::collective_stage(comm,
            [&]
            {
                for (const auto& [id, name] : this->d_boundary_id_to_name)
                    if (!catalog.contains(id) || catalog.at(id) != name)
                        throw std::invalid_argument("Legacy mesh snapshot boundary catalogs differ across ranks.");
                std::unordered_map<std::string, int> reverse_catalog;
                reverse_catalog.reserve(catalog.size());
                for (const auto& [id, name] : catalog)
                    reverse_catalog.emplace(name, id);
                this->d_boundary_id_to_name = std::move(catalog);
                this->d_boundary_name_to_id = std::move(reverse_catalog);
            });
    }
};

template<TpetraTypePack Pack> void LegacyMeshSnapshot<Pack>::export_vtu(const std::string& filename) const
{
    check_connectivity();

    const int nranks = Tpetra::getDefaultComm()->getSize();
    const int myrank = Tpetra::getDefaultComm()->getRank();

    // Per-rank file names for parallel runs
    const std::string output_filename = [&]() -> std::string
    {
        if (nranks <= 1)
            return filename;
        const auto dot_pos = filename.rfind('.');
        if (dot_pos != std::string::npos)
        {
            return filename.substr(0, dot_pos) + "_rank" + std::to_string(myrank) + filename.substr(dot_pos);
        }
        return filename + "_rank" + std::to_string(myrank);
    }();

    std::unordered_map<global_ordinal_type, global_index_t> node_lid;
    node_lid.reserve(d_node_coords.size());
    VTUWriter::VectorData node_coords;
    node_coords.reserve(d_node_coords.size());
    VTUWriter::Int64Data cell_node_offsets;
    VTUWriter::Int64Data cell_node_ids;
    VTUWriter::UInt8Data vtu_cell_types;
    VTUWriter::Int64Data cell_gids;
    VTUWriter::IntData cell_types;
    VTUWriter::ScalarData cell_volumes;
    VTUWriter::VectorData cell_centroids;

    auto append_node = [&](global_ordinal_type node_gid) -> global_index_t
    {
        const auto iter = node_lid.find(node_gid);
        if (iter != node_lid.end())
        {
            return iter->second;
        }

        const auto lid = static_cast<global_index_t>(node_coords.size());
        node_lid.emplace(node_gid, lid);
        node_coords.push_back(Base::node_coord(node_gid));
        return lid;
    };

    // Export only owned cells in parallel; all locally visible cells in serial.
    const bool export_all = (nranks <= 1);
    const auto exported_cell_count = export_all ? d_cells.size() : d_owned_cell_global_ids.size();
    cell_node_offsets.reserve(exported_cell_count);
    vtu_cell_types.reserve(exported_cell_count);
    cell_gids.reserve(exported_cell_count);
    cell_types.reserve(exported_cell_count);
    cell_volumes.reserve(exported_cell_count);
    cell_centroids.reserve(exported_cell_count);

    for (size_t cell_index = 0; cell_index < d_cells.size(); ++cell_index)
    {
        if (!export_all && cell_index >= d_owned_cell_global_ids.size())
        {
            continue;
        }

        const auto cell_lid = detail::checked_size_to_ordinal<local_ordinal_type>(cell_index, "VTU cell local id");
        const auto& cell_info = d_cells[cell_index];
        for (const auto node_gid : cell_info.node_gids)
        {
            cell_node_ids.push_back(append_node(node_gid));
        }

        cell_node_offsets.push_back(static_cast<global_index_t>(cell_node_ids.size()));
        vtu_cell_types.push_back(static_cast<std::uint8_t>(MeshUtils::vtu_cell_type_code(cell_info.type)));
        cell_gids.push_back(static_cast<global_index_t>(Base::cell_global_id(cell_lid)));
        cell_types.push_back(static_cast<int>(cell_info.type));
        cell_volumes.push_back(cell_info.volume);
        cell_centroids.push_back(cell_info.center);
    }

    VTUWriter writer;
    writer.set_points(std::move(node_coords));
    writer.set_cells(std::move(cell_node_ids), std::move(cell_node_offsets), std::move(vtu_cell_types));
    writer.add_int64_cell_data("cell_gid", std::move(cell_gids));
    writer.add_int_cell_data("cell_type", std::move(cell_types));
    writer.add_scalar_cell_data("cell_volume", std::move(cell_volumes));
    writer.add_vector_cell_data("cell_centroid", std::move(cell_centroids));
    writer.write(output_filename);
}

template<TpetraTypePack Pack> SP<Mesh<Pack>> make_legacy_mesh_snapshot(Meshes::UnstructuredMeshData data)
{
    return collective_make_shared<LegacyMeshSnapshot<Pack>>(*Tpetra::getDefaultComm(), std::move(data));
}
} // namespace SimpleFluid::detail

SIMPLEFLUID_LEGACY_API_END
