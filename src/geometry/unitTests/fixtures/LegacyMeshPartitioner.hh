/** @brief Frozen test-only legacy partitioner and wire representation. */
#pragma once
#include "LegacyMesh.hh"
#include "parallel/MPI_interface.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <string>
#include <vector>


namespace SimpleFluid::test
{
namespace partition_detail {

/**
 * @brief Serialisable packet carrying the geometry and connectivity of a single cell.
 *
 * Used to redistribute cells across MPI ranks after partitioning.
 * `node_gids` and `node_coords` are parallel arrays. Face node IDs are sorted
 * to form stable keys, and `face_global_ids`, `face_boundary_ids`,
 * `face_neighbor_gids`, `face_neighbor_ranks`, and `face_geometry` are
 * parallel to those keys. Face centers, areas, normals, and distances retain
 * their source values and owner orientation.
 *
 * @tparam Pack Tpetra type pack providing GO, LO, map, and graph types.
 */
template<TpetraTypePack Pack>
struct CellPacket {
    using GO = typename Pack::global_ordinal_type;
    using Vec3 = MeshUtils::Vec3;
    using CellType = MeshUtils::CellType;

    struct FaceGeometry
    {
        Vec3 center{};
        double area = 0.0;
        Vec3 unit_normal_from_owner{};
        double owner_to_face_distance = 0.0;
        double neighbor_to_face_distance = 0.0;
        double cell_center_distance = 0.0;
        GO owner_gid = invalid_id<GO>();
    };

    GO gid{};
    CellType cell_type = CellType::INVALID;
    Vec3 center{};
    double volume = 0.0;
    std::vector<GO> node_gids;
    std::vector<Vec3> node_coords;
    std::vector<std::vector<GO>> face_node_keys;
    std::vector<GO> face_global_ids;
    std::vector<int> face_boundary_ids;
    std::vector<GO> face_neighbor_gids;
    std::vector<int> face_neighbor_ranks;
    std::vector<FaceGeometry> face_geometry;

    /**
     * @brief Validated, non-owning view of one packet body in an MPI buffer.
     *
     * Scalar fields are decoded with memcpy so the byte stream need not be
     * naturally aligned. Node and face callbacks expose one small record at a
     * time without allocating nested containers.
     */
    class View
    {
    public:
        struct Header
        {
            GO gid{};
            CellType cell_type = CellType::INVALID;
            Vec3 center{};
            double volume = 0.0;
            std::uint32_t node_count = 0;
        };

        struct Node
        {
            GO gid{};
            Vec3 coordinate{};
        };

        struct Face
        {
            GO global_id{};
            int boundary_id = LegacyMesh<Pack>::invalid_boundary_id;
            GO neighbor_gid = invalid_id<GO>();
            int neighbor_rank = -1;
            FaceGeometry geometry{};
            std::array<GO, 4> node_gids{};
            std::uint32_t node_count = 0;
        };

        View() = default;

        Header header() const
        {
            Cursor cursor(d_data, d_size);
            return read_header(cursor);
        }

        GO gid() const
        {
            Cursor cursor(d_data, d_size);
            return cursor.template read<GO>();
        }

        size_t serialized_size() const noexcept { return d_size; }
        const char* data() const noexcept { return d_data; }

        template<class Function>
        void for_each_node(Function&& function) const
        {
            Cursor cursor(d_data, d_size);
            const auto packet_header = read_header(cursor);
            for (std::uint32_t node = 0;
                 node < packet_header.node_count;
                 ++node)
            {
                Node decoded;
                decoded.gid = cursor.template read<GO>();
                decoded.coordinate.x =
                    cursor.template read<double>();
                decoded.coordinate.y =
                    cursor.template read<double>();
                decoded.coordinate.z =
                    cursor.template read<double>();
                function(decoded);
            }
        }

        template<class Function>
        void for_each_face(Function&& function) const
        {
            Cursor cursor(d_data, d_size);
            const auto packet_header = read_header(cursor);
            skip_nodes(cursor, packet_header.node_count);
            const auto face_count =
                cursor.template read<std::uint32_t>();
            for (std::uint32_t face = 0;
                 face < face_count;
                 ++face)
            {
                Face decoded;
                decoded.global_id = cursor.template read<GO>();
                decoded.boundary_id =
                    cursor.template read<std::int32_t>();
                decoded.neighbor_gid =
                    cursor.template read<GO>();
                decoded.neighbor_rank =
                    cursor.template read<std::int32_t>();
                decoded.geometry.center.x = cursor.template read<double>();
                decoded.geometry.center.y = cursor.template read<double>();
                decoded.geometry.center.z = cursor.template read<double>();
                decoded.geometry.area = cursor.template read<double>();
                decoded.geometry.unit_normal_from_owner.x =
                    cursor.template read<double>();
                decoded.geometry.unit_normal_from_owner.y =
                    cursor.template read<double>();
                decoded.geometry.unit_normal_from_owner.z =
                    cursor.template read<double>();
                decoded.geometry.owner_to_face_distance =
                    cursor.template read<double>();
                decoded.geometry.neighbor_to_face_distance =
                    cursor.template read<double>();
                decoded.geometry.cell_center_distance =
                    cursor.template read<double>();
                decoded.geometry.owner_gid = cursor.template read<GO>();
                decoded.node_count =
                    cursor.template read<std::uint32_t>();
                if (decoded.node_count != 3
                    && decoded.node_count != 4)
                {
                    throw std::runtime_error(
                        "CellPacket face has unsupported node count.");
                }
                for (std::uint32_t node = 0;
                     node < decoded.node_count;
                     ++node)
                {
                    decoded.node_gids[node] =
                        cursor.template read<GO>();
                }
                function(decoded);
            }
        }

        static View validate(const char* data, size_t size)
        {
            Cursor cursor(data, size);
            const auto packet_header = read_header(cursor);
            skip_nodes(cursor, packet_header.node_count);
            const auto face_count =
                cursor.template read<std::uint32_t>();
            for (std::uint32_t face = 0;
                 face < face_count;
                 ++face)
            {
                (void)cursor.template read<GO>();
                (void)cursor.template read<std::int32_t>();
                (void)cursor.template read<GO>();
                (void)cursor.template read<std::int32_t>();
                for (int metric = 0; metric < 10; ++metric)
                {
                    (void)cursor.template read<double>();
                }
                (void)cursor.template read<GO>();
                const auto node_count =
                    cursor.template read<std::uint32_t>();
                if (node_count != 3 && node_count != 4)
                {
                    throw std::runtime_error(
                        "CellPacket face has unsupported node count.");
                }
                for (std::uint32_t node = 0;
                     node < node_count;
                     ++node)
                {
                    (void)cursor.template read<GO>();
                }
            }
            return View(data, size - cursor.remaining());
        }

    private:
        class Cursor
        {
        public:
            Cursor(const char* data, size_t size)
                : d_current(data), d_end(data + size)
            {
            }

            template<class T>
            T read()
            {
                if (remaining() < sizeof(T))
                {
                    throw std::runtime_error(
                        "CellPacket buffer is truncated.");
                }
                T value;
                std::memcpy(&value, d_current, sizeof(T));
                d_current += sizeof(T);
                return value;
            }

            size_t remaining() const noexcept
            {
                return static_cast<size_t>(d_end - d_current);
            }

        private:
            const char* d_current;
            const char* d_end;
        };

        View(const char* data, size_t size)
            : d_data(data), d_size(size)
        {
        }

        static Header read_header(Cursor& cursor)
        {
            Header decoded;
            decoded.gid = cursor.template read<GO>();
            decoded.cell_type = static_cast<CellType>(
                cursor.template read<std::int32_t>());
            decoded.center.x = cursor.template read<double>();
            decoded.center.y = cursor.template read<double>();
            decoded.center.z = cursor.template read<double>();
            decoded.volume = cursor.template read<double>();
            decoded.node_count =
                cursor.template read<std::uint32_t>();
            return decoded;
        }

        static void skip_nodes(
            Cursor& cursor,
            std::uint32_t node_count)
        {
            for (std::uint32_t node = 0;
                 node < node_count;
                 ++node)
            {
                (void)cursor.template read<GO>();
                (void)cursor.template read<double>();
                (void)cursor.template read<double>();
                (void)cursor.template read<double>();
            }
        }

        const char* d_data = nullptr;
        size_t d_size = 0;
    };

    /**
     * @brief Parse packet boundaries without materializing packet contents.
     */
    static std::vector<View> view_packets(
        const char* data,
        size_t size)
    {
        if (size < sizeof(std::uint32_t))
        {
            throw std::runtime_error(
                "CellPacket buffer is truncated.");
        }
        std::uint32_t count = 0;
        std::memcpy(&count, data, sizeof(count));
        size_t offset = sizeof(count);
        std::vector<View> packets;
        packets.reserve(count);
        for (std::uint32_t packet = 0; packet < count; ++packet)
        {
            if (offset >= size)
            {
                throw std::runtime_error(
                    "CellPacket buffer is truncated.");
            }
            const auto view =
                View::validate(data + offset, size - offset);
            if (view.serialized_size() == 0)
            {
                throw std::runtime_error(
                    "CellPacket body is empty.");
            }
            offset += view.serialized_size();
            packets.push_back(view);
        }
        if (offset != size)
        {
            throw std::runtime_error(
                "CellPacket buffer contains trailing bytes.");
        }
        return packets;
    }

    /**
     * @brief Serialise a vector of cell packets into a flat byte buffer for MPI transmission.
     *
     * @param pkts Vector of cell packets to serialise.
     * @return Flat byte buffer suitable for MPI_Alltoallv.
     */
    static std::vector<char> serialize_packets(
        const std::vector<CellPacket>& pkts)
    {
        if (pkts.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "CellPacket count exceeds its wire representation.");
        }

        size_t bytes = sizeof(std::uint32_t);
        for (const auto& packet : pkts)
        {
            bytes += serialized_size(packet);
        }

        std::vector<char> buffer(bytes);
        Writer writer(buffer.data(), buffer.size());
        writer.write(static_cast<std::uint32_t>(pkts.size()));
        for (const auto& packet : pkts)
        {
            serialize_into(
                writer.current(), writer.remaining(), packet);
            writer.advance(serialized_size(packet));
        }
        return buffer;
    }

    /**
     * @brief Return the exact number of bytes occupied by one packet body.
     */
    static size_t serialized_size(const CellPacket& packet)
    {
        validate(packet);
        size_t bytes =
            sizeof(GO) + sizeof(std::int32_t)
          + 4 * sizeof(double) + sizeof(std::uint32_t)
          + packet.node_gids.size() * (sizeof(GO) + 3 * sizeof(double))
          + sizeof(std::uint32_t);
        for (const auto& nodes : packet.face_node_keys)
        {
            bytes +=
                sizeof(GO) + sizeof(std::int32_t) + sizeof(GO)
              + sizeof(std::int32_t) + 10 * sizeof(double) + sizeof(GO)
              + sizeof(std::uint32_t)
              + nodes.size() * sizeof(GO);
        }
        return bytes;
    }

    /**
     * @brief Serialise one already-materialized packet into caller storage.
     */
    static void serialize_into(
        char* data,
        size_t size,
        const CellPacket& packet)
    {
        const auto required = serialized_size(packet);
        if (size < required)
        {
            throw std::length_error(
                "CellPacket output buffer is too small.");
        }

        Writer writer(data, required);
        writer.write(packet.gid);
        writer.write(static_cast<std::int32_t>(packet.cell_type));
        writer.write(packet.center.x);
        writer.write(packet.center.y);
        writer.write(packet.center.z);
        writer.write(packet.volume);
        writer.write(static_cast<std::uint32_t>(
            packet.node_gids.size()));
        for (size_t node = 0; node < packet.node_gids.size(); ++node)
        {
            writer.write(packet.node_gids[node]);
            writer.write(packet.node_coords[node].x);
            writer.write(packet.node_coords[node].y);
            writer.write(packet.node_coords[node].z);
        }
        writer.write(static_cast<std::uint32_t>(
            packet.face_node_keys.size()));
        for (size_t face = 0;
             face < packet.face_node_keys.size();
             ++face)
        {
            writer.write(packet.face_global_ids[face]);
            writer.write(static_cast<std::int32_t>(
                packet.face_boundary_ids[face]));
            writer.write(packet.face_neighbor_gids[face]);
            writer.write(static_cast<std::int32_t>(
                packet.face_neighbor_ranks[face]));
            const auto& geometry = packet.face_geometry[face];
            writer.write(geometry.center.x);
            writer.write(geometry.center.y);
            writer.write(geometry.center.z);
            writer.write(geometry.area);
            writer.write(geometry.unit_normal_from_owner.x);
            writer.write(geometry.unit_normal_from_owner.y);
            writer.write(geometry.unit_normal_from_owner.z);
            writer.write(geometry.owner_to_face_distance);
            writer.write(geometry.neighbor_to_face_distance);
            writer.write(geometry.cell_center_distance);
            writer.write(geometry.owner_gid);
            const auto& nodes = packet.face_node_keys[face];
            writer.write(static_cast<std::uint32_t>(nodes.size()));
            for (const auto node_gid : nodes)
            {
                writer.write(node_gid);
            }
        }
    }

    /**
     * @brief Return the exact wire size of one cell read from legacy storage.
     */
    static size_t serialized_mesh_cell_size(
        const LegacyMesh<Pack>& mesh,
        typename Pack::local_ordinal_type cell_lid)
    {
        const auto& cell = mesh.cell(cell_lid);
        size_t bytes =
            sizeof(GO) + sizeof(std::int32_t)
          + 4 * sizeof(double) + sizeof(std::uint32_t)
          + cell.node_gids.size() * (sizeof(GO) + 3 * sizeof(double))
          + sizeof(std::uint32_t);
        for (const auto face_lid : cell.faces)
        {
            const auto& face = mesh.face(face_lid);
            bytes +=
                sizeof(GO) + sizeof(std::int32_t) + sizeof(GO)
              + sizeof(std::int32_t) + 10 * sizeof(double) + sizeof(GO)
              + sizeof(std::uint32_t)
              + face.node_gids.size() * sizeof(GO);
        }
        return bytes;
    }

    /**
     * @brief Serialise one legacy cell directly into its final MPI buffer.
     *
     * No intermediate CellPacket object or nested vectors are constructed.
     * `neighbor_rank` maps a valid adjacent cell GID to its partition rank.
     */
    template<class NeighborRank>
    static void serialize_mesh_cell(
        char* data,
        size_t size,
        const LegacyMesh<Pack>& mesh,
        typename Pack::local_ordinal_type cell_lid,
        GO gid,
        NeighborRank&& neighbor_rank)
    {
        const auto required =
            serialized_mesh_cell_size(mesh, cell_lid);
        if (size < required)
        {
            throw std::length_error(
                "CellPacket output buffer is too small.");
        }

        const auto& cell = mesh.cell(cell_lid);
        Writer writer(data, required);
        writer.write(gid);
        writer.write(static_cast<std::int32_t>(cell.type));
        writer.write(cell.center.x);
        writer.write(cell.center.y);
        writer.write(cell.center.z);
        writer.write(cell.volume);
        writer.write(static_cast<std::uint32_t>(
            cell.node_gids.size()));
        for (const auto node_gid : cell.node_gids)
        {
            const auto& coordinate = mesh.node_coord(node_gid);
            writer.write(node_gid);
            writer.write(coordinate.x);
            writer.write(coordinate.y);
            writer.write(coordinate.z);
        }

        writer.write(static_cast<std::uint32_t>(cell.faces.size()));
        for (const auto face_lid : cell.faces)
        {
            const auto& face = mesh.face(face_lid);
            auto adjacent_gid = invalid_id<GO>();
            if (face.owner == cell_lid
                && face.neighbor
                    != invalid_id<typename Pack::local_ordinal_type>())
            {
                adjacent_gid = mesh.cell_global_id(face.neighbor);
            }
            else if (face.neighbor == cell_lid)
            {
                adjacent_gid = mesh.cell_global_id(face.owner);
            }

            writer.write(mesh.face_global_id(face_lid));
            writer.write(static_cast<std::int32_t>(
                face.boundary_id));
            writer.write(adjacent_gid);
            writer.write(static_cast<std::int32_t>(
                adjacent_gid == invalid_id<GO>()
                    ? -1
                    : neighbor_rank(adjacent_gid)));
            writer.write(face.center.x);
            writer.write(face.center.y);
            writer.write(face.center.z);
            writer.write(face.area);
            writer.write(face.unit_normal_from_owner.x);
            writer.write(face.unit_normal_from_owner.y);
            writer.write(face.unit_normal_from_owner.z);
            writer.write(face.owner_to_face_distance);
            writer.write(face.neighbor_to_face_distance);
            writer.write(face.cell_center_distance);
            writer.write(mesh.cell_global_id(face.owner));
            writer.write(static_cast<std::uint32_t>(
                face.node_gids.size()));
            for (const auto node_gid : face.node_gids)
            {
                writer.write(node_gid);
            }
        }
    }

    /**
     * @brief Deserialise a flat byte buffer back into a vector of cell packets.
     *
     * @param data Pointer to the byte buffer received via MPI.
     * @param size Number of readable bytes at @p data.
     * @return Reconstructed vector of cell packets.
     */
    static std::vector<CellPacket> deserialize_packets(
        const char* data,
        size_t size)
    {
        std::vector<CellPacket> pkts;
        Reader reader(data, size);
        const auto count = reader.template read<std::uint32_t>();
        pkts.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            CellPacket pk;
            pk.gid = reader.template read<GO>();
            pk.cell_type = static_cast<CellType>(
                reader.template read<std::int32_t>());
            pk.center.x = reader.template read<double>();
            pk.center.y = reader.template read<double>();
            pk.center.z = reader.template read<double>();
            pk.volume = reader.template read<double>();
            const auto nn = reader.template read<std::uint32_t>();
            pk.node_gids.resize(nn);
            pk.node_coords.resize(nn);
            for (std::uint32_t node = 0; node < nn; ++node)
            {
                pk.node_gids[node] = reader.template read<GO>();
                pk.node_coords[node].x =
                    reader.template read<double>();
                pk.node_coords[node].y =
                    reader.template read<double>();
                pk.node_coords[node].z =
                    reader.template read<double>();
            }
            const auto nf = reader.template read<std::uint32_t>();
            pk.face_node_keys.resize(nf);
            pk.face_global_ids.resize(nf);
            pk.face_boundary_ids.resize(nf);
            pk.face_neighbor_gids.resize(nf);
            pk.face_neighbor_ranks.resize(nf);
            pk.face_geometry.resize(nf);
            for (std::uint32_t face = 0; face < nf; ++face)
            {
                pk.face_global_ids[face] =
                    reader.template read<GO>();
                pk.face_boundary_ids[face] =
                    reader.template read<std::int32_t>();
                pk.face_neighbor_gids[face] =
                    reader.template read<GO>();
                pk.face_neighbor_ranks[face] =
                    reader.template read<std::int32_t>();
                auto& geometry = pk.face_geometry[face];
                geometry.center.x = reader.template read<double>();
                geometry.center.y = reader.template read<double>();
                geometry.center.z = reader.template read<double>();
                geometry.area = reader.template read<double>();
                geometry.unit_normal_from_owner.x =
                    reader.template read<double>();
                geometry.unit_normal_from_owner.y =
                    reader.template read<double>();
                geometry.unit_normal_from_owner.z =
                    reader.template read<double>();
                geometry.owner_to_face_distance = reader.template read<double>();
                geometry.neighbor_to_face_distance = reader.template read<double>();
                geometry.cell_center_distance = reader.template read<double>();
                geometry.owner_gid = reader.template read<GO>();
                const auto node_count =
                    reader.template read<std::uint32_t>();
                auto& nodes = pk.face_node_keys[face];
                nodes.resize(node_count);
                for (auto& node_gid : nodes)
                {
                    node_gid = reader.template read<GO>();
                }
            }
            pkts.push_back(std::move(pk));
        }
        if (reader.remaining() != 0)
        {
            throw std::runtime_error(
                "CellPacket buffer contains trailing bytes.");
        }
        return pkts;
    }

private:
    class Writer
    {
    public:
        Writer(char* data, size_t size)
            : d_current(data), d_end(data + size)
        {
        }

        template<class T>
        void write(const T& value)
        {
            if (remaining() < sizeof(T))
            {
                throw std::length_error(
                    "CellPacket output buffer is too small.");
            }
            std::memcpy(d_current, &value, sizeof(T));
            d_current += sizeof(T);
        }

        char* current() const noexcept { return d_current; }
        size_t remaining() const noexcept
        {
            return static_cast<size_t>(d_end - d_current);
        }
        void advance(size_t bytes)
        {
            if (bytes > remaining())
            {
                throw std::length_error(
                    "CellPacket output buffer is too small.");
            }
            d_current += bytes;
        }

    private:
        char* d_current;
        char* d_end;
    };

    class Reader
    {
    public:
        Reader(const char* data, size_t size)
            : d_current(data), d_end(data + size)
        {
        }

        template<class T>
        T read()
        {
            if (remaining() < sizeof(T))
            {
                throw std::runtime_error(
                    "CellPacket buffer is truncated.");
            }
            T value;
            std::memcpy(&value, d_current, sizeof(T));
            d_current += sizeof(T);
            return value;
        }

        size_t remaining() const noexcept
        {
            return static_cast<size_t>(d_end - d_current);
        }

    private:
        const char* d_current;
        const char* d_end;
    };

    static void validate(const CellPacket& packet)
    {
        if (packet.node_gids.size() != packet.node_coords.size())
        {
            throw std::invalid_argument(
                "CellPacket node IDs and coordinates have different sizes.");
        }
        if (packet.face_node_keys.size()
                != packet.face_global_ids.size()
            || packet.face_node_keys.size()
                != packet.face_boundary_ids.size()
            || packet.face_node_keys.size()
                != packet.face_neighbor_gids.size()
            || packet.face_node_keys.size()
                != packet.face_neighbor_ranks.size()
            || packet.face_node_keys.size()
                != packet.face_geometry.size())
        {
            throw std::invalid_argument(
                "CellPacket face metadata arrays have different sizes.");
        }
        if (packet.node_gids.size()
                > std::numeric_limits<std::uint32_t>::max()
            || packet.face_node_keys.size()
                > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "CellPacket entity count exceeds its wire representation.");
        }
        for (const auto& nodes : packet.face_node_keys)
        {
            if (nodes.size()
                > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::overflow_error(
                    "CellPacket face-node count exceeds its wire "
                    "representation.");
            }
        }
    }

};

} // namespace partition_detail
template<TpetraTypePack Pack> class LegacyMeshPartitioner
{
public:
    using GO = typename Pack::global_ordinal_type;
    using LO = typename Pack::local_ordinal_type;
    using map_type = typename Pack::map_type;
    using graph_type = typename Pack::graph_type;
    using comm_type = typename Pack::comm_type;
    using Vec3 = MeshUtils::Vec3;
    using CellInfo = typename LegacyMesh<Pack>::CellInfo;
    using FaceInfo = typename LegacyMesh<Pack>::FaceInfo;
    using Packet = partition_detail::CellPacket<Pack>;
    using PacketView = typename Packet::View;
    static constexpr LO invalid_lid = invalid_id<LO>();
    static bool partition(LegacyMesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm);
private:
    struct PartitionGraph
    {
        std::vector<GO> row_gids;
        std::vector<std::vector<GO>> row_adjacency;
        std::vector<GO> column_gids;
        std::vector<double> row_weights;
        std::vector<std::vector<double>> row_edge_weights;
        double imbalance_tolerance = 1.1;
    };

    /**
     * @brief Cells this rank is responsible for reading from source geometry.
     *
     * A root-only source is represented by one rank number. More general
     * replicated or sparse inputs retain only the sorted GIDs selected on this
     * rank, never a global GID-to-source map on every rank.
     */
    struct SourceSelection
    {
        int single_source_rank = -1;
        std::vector<GO> local_source_gids;

        bool is_single_source() const noexcept
        {
            return single_source_rank >= 0;
        }

        bool provides(GO gid, int rank) const noexcept
        {
            return is_single_source()
                ? rank == single_source_rank
                : std::binary_search(
                      local_source_gids.begin(),
                      local_source_gids.end(),
                      gid);
        }
    };


    static PartitionGraph build_partition_graph(const LegacyMesh<Pack>& mesh,
        const SourceSelection& selection, const Teuchos::RCP<const comm_type>& comm);
    static std::unordered_map<GO, int> solve_partition_graph(const PartitionGraph& graph,
        const Teuchos::RCP<const comm_type>& comm, int gather_root = -1);
    static std::unordered_map<GO, int> compute_gid_to_rank_map(const LegacyMesh<Pack>& mesh,
        const SourceSelection& selection, const Teuchos::RCP<const comm_type>& comm);
    static SourceSelection compute_source_selection(const LegacyMesh<Pack>& mesh,
        const Teuchos::RCP<const comm_type>& comm);
    static void release_rebuildable_storage(LegacyMesh<Pack>& mesh);
    static void rebuild(LegacyMesh<Pack>& mesh, const std::vector<PacketView>& owned,
        const std::vector<PacketView>& ghost, const std::unordered_map<std::string, GO>& periodic_pairs);
    static my_mpi::Comm get_mpi_comm(const comm_type& tc) {
        auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&tc);
        if (mpi) return *(mpi->getRawMpiComm()); return MPI_COMM_NULL;
    }
    static MPI_Datatype mpi_go_type() { return my_mpi::type_trait<GO>(); }
};

}
