/**
 * @file Mesh.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "Mesh.hh"

#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

namespace SimpleFluid
{

namespace detail
{

template <class Ordinal>
inline void check_crs_offsets(const std::vector<Ordinal>& offsets,
                              const std::vector<Ordinal>& ids,
                              std::size_t rows)
{
    CHECK(offsets.size() == rows + 1);
    CHECK(offsets.empty() || offsets.front() == 0);

    for (std::size_t i = 0; i + 1 < offsets.size(); ++i)
    {
        CHECK(offsets[i] <= offsets[i + 1]);
        CHECK(static_cast<std::size_t>(offsets[i + 1]) <= ids.size());
    }
}

} // namespace detail

template<TpetraTypePack Pack>
void Mesh<Pack>::assemble()
{
    check_connectivity();
    create_maps();
    create_device_views();
}

template<TpetraTypePack Pack>
void Mesh<Pack>::export_vtu(const std::string& filename) const
{
    check_connectivity();

    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Failed to open VTU output file: " + filename);
    }

    auto vtk_cell_type = [](CellType type, std::size_t node_count) -> int
    {
        switch (type)
        {
            case CellType::HEXAHEDRON:
                if (node_count != 8)
                {
                    throw std::runtime_error("VTU export expects 8 nodes for hexahedron cells.");
                }
                return 12;
            case CellType::TRIPRISM:
                if (node_count != 6)
                {
                    throw std::runtime_error("VTU export expects 6 nodes for triangular-prism cells.");
                }
                return 13;
        }

        throw std::runtime_error("VTU export encountered an unsupported cell type.");
    };

    out << std::setprecision(std::numeric_limits<real_t>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << d_num_nodes
        << "\" NumberOfCells=\"" << d_num_local_cells << "\">\n";

    out << "      <PointData/>\n";
    out << "      <CellData>\n";
    out << "        <DataArray type=\"Int64\" Name=\"cell_gid\" format=\"ascii\">\n";
    out << "          ";
    for (local_ordinal_type lid = 0; lid < d_num_local_cells; ++lid)
    {
        out << d_cell_gid[lid] << (lid + 1 == d_num_local_cells ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"cell_type\" format=\"ascii\">\n";
    out << "          ";
    for (local_ordinal_type lid = 0; lid < d_num_local_cells; ++lid)
    {
        out << d_cell_type[lid] << (lid + 1 == d_num_local_cells ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"cell_volume\" format=\"ascii\">\n";
    out << "          ";
    for (local_ordinal_type lid = 0; lid < d_num_local_cells; ++lid)
    {
        out << d_cell_volume[lid] << (lid + 1 == d_num_local_cells ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"cell_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& centroid : d_cell_centroid)
    {
        out << "          " << centroid.x << " " << centroid.y << " " << centroid.z << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& coord : d_node_coord)
    {
        out << "          " << coord.x << " " << coord.y << " " << coord.z << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (local_ordinal_type lid = 0; lid < d_num_local_cells; ++lid)
    {
        const auto begin = static_cast<std::size_t>(d_cell_node_offset[lid]);
        const auto end = static_cast<std::size_t>(d_cell_node_offset[static_cast<std::size_t>(lid) + 1]);

        out << "          ";
        for (std::size_t i = begin; i < end; ++i)
        {
            out << d_cell_node_ids[i] << (i + 1 == end ? "" : " ");
        }
        out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (local_ordinal_type lid = 0; lid < d_num_local_cells; ++lid)
    {
        const auto offset = d_cell_node_offset[static_cast<std::size_t>(lid) + 1];
        out << offset << (lid + 1 == d_num_local_cells ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (local_ordinal_type lid = 0; lid < d_num_local_cells; ++lid)
    {
        const auto begin = static_cast<std::size_t>(d_cell_node_offset[lid]);
        const auto end = static_cast<std::size_t>(d_cell_node_offset[static_cast<std::size_t>(lid) + 1]);
        const auto type = static_cast<CellType>(d_cell_type[lid]);
        out << vtk_cell_type(type, end - begin) << (lid + 1 == d_num_local_cells ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error("Failed while writing VTU output file: " + filename);
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::create_maps()
{
    CHECK(d_num_owned_cells <= d_num_local_cells);

    if (d_comm.is_null())
    {
        d_comm = Tpetra::getDefaultComm();
    }

    const auto invalid_global_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;

    d_owned_cell_map = Teuchos::rcp(new map_type(invalid_global_size,
                                                 d_cell_gid.data(),
                                                 d_num_owned_cells,
                                                 index_base,
                                                 d_comm));
    d_overlap_cell_map = Teuchos::rcp(new map_type(invalid_global_size,
                                                   d_cell_gid.data(),
                                                   d_num_local_cells,
                                                   index_base,
                                                   d_comm));

    d_num_global_cells = static_cast<global_ordinal_type>(d_owned_cell_map->getGlobalNumElements());
}

template<TpetraTypePack Pack>
void Mesh<Pack>::check_connectivity() const
{
    CHECK(static_cast<std::size_t>(d_num_owned_cells) <= static_cast<std::size_t>(d_num_local_cells));

    CHECK(d_cell_gid.size() == static_cast<std::size_t>(d_num_local_cells));
    CHECK(d_cell_type.size() == static_cast<std::size_t>(d_num_local_cells));
    CHECK(d_cell_volume.size() == static_cast<std::size_t>(d_num_local_cells));
    CHECK(d_cell_centroid.size() == static_cast<std::size_t>(d_num_local_cells));

    CHECK(d_face_owner.size() == static_cast<std::size_t>(d_num_faces));
    CHECK(d_face_neighbor.size() == static_cast<std::size_t>(d_num_faces));
    CHECK(d_face_type.size() == static_cast<std::size_t>(d_num_faces));
    CHECK(d_face_patch.size() == static_cast<std::size_t>(d_num_faces));
    CHECK(d_face_area.size() == static_cast<std::size_t>(d_num_faces));
    CHECK(d_face_area_vector.size() == static_cast<std::size_t>(d_num_faces));
    CHECK(d_face_centroid.size() == static_cast<std::size_t>(d_num_faces));

    CHECK(d_node_coord.size() == static_cast<std::size_t>(d_num_nodes));

    detail::check_crs_offsets(d_cell_face_offset, d_cell_face_ids, d_num_local_cells);
    detail::check_crs_offsets(d_cell_node_offset, d_cell_node_ids, d_num_local_cells);
    detail::check_crs_offsets(d_face_node_offset, d_face_node_ids, d_num_faces);

    for (std::size_t lid = 0; lid < d_cell_gid.size(); ++lid)
    {
        const auto iter = d_cell_gid_to_lid.find(d_cell_gid[lid]);
        CHECK(iter != d_cell_gid_to_lid.end());
        CHECK(iter->second == static_cast<local_ordinal_type>(lid));
    }

    const auto invalid_lid = detail::invalid_local_ordinal<local_ordinal_type>();
    for (std::size_t fid = 0; fid < d_face_owner.size(); ++fid)
    {
        CHECK(d_face_owner[fid] < d_num_local_cells);
        CHECK(d_face_neighbor[fid] == invalid_lid || d_face_neighbor[fid] < d_num_local_cells);
    }

    for (const auto face_id : d_cell_face_ids)
    {
        CHECK(face_id < d_num_faces);
    }

    for (const auto node_id : d_cell_node_ids)
    {
        CHECK(node_id < d_num_nodes);
    }

    for (const auto node_id : d_face_node_ids)
    {
        CHECK(node_id < d_num_nodes);
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::create_device_views()
{
    d_device_views.cell_gid = make_vector_view("cell_gid", d_cell_gid);
    d_device_views.cell_type = make_vector_view("cell_type", d_cell_type);

    d_device_views.cell_volume = make_vector_view("cell_volume", d_cell_volume);
    d_device_views.cell_centroid = make_vectorV3D_view("cell_centroid", d_cell_centroid);

    d_device_views.cell_face_offset = make_vector_view("cell_face_offset", d_cell_face_offset);
    d_device_views.cell_face_ids = make_vector_view("cell_face_ids", d_cell_face_ids);

    d_device_views.face_owner = make_vector_view("face_owner", d_face_owner);
    d_device_views.face_neighbor = make_vector_view("face_neighbor", d_face_neighbor);
    d_device_views.face_type = make_vector_view("face_type", d_face_type);
    d_device_views.face_patch = make_vector_view("face_patch", d_face_patch);

    d_device_views.face_area = make_vector_view("face_area", d_face_area);
    d_device_views.face_area_vector = make_vectorV3D_view("face_area_vector", d_face_area_vector);
    d_device_views.face_centroid = make_vectorV3D_view("face_centroid", d_face_centroid);

    d_device_views.node_coord = make_vectorV3D_view("node_coord", d_node_coord);

    d_device_views.cell_node_offset = make_vector_view("cell_node_offset", d_cell_node_offset);
    d_device_views.cell_node_ids = make_vector_view("cell_node_ids", d_cell_node_ids);

    d_device_views.face_node_offset = make_vector_view("face_node_offset", d_face_node_offset);
    d_device_views.face_node_ids = make_vector_view("face_node_ids", d_face_node_ids);
}

}
