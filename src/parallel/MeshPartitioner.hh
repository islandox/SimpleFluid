/**
 * @file MeshPartitioner.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Zoltan2-based mesh domain decomposition for parallel runs.
 * @version 0.1
 * @date 2026-05-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "geometry/Mesh.hh"
#include "geometry/mesh/LocalGlobalIndexer.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "parallel/MPI_interface.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace SimpleFluid {

namespace partition_detail {

/**
 * @brief Serialisable packet carrying the geometry and connectivity of a single cell.
 *
 * Used to redistribute cells across MPI ranks after partitioning.
 * `node_gids` and `node_coords` are parallel arrays. Face node IDs are sorted
 * to form stable keys, and `face_global_ids`, `face_boundary_ids`,
 * `face_neighbor_gids`, and `face_neighbor_ranks` are parallel to those keys.
 *
 * @tparam Pack Tpetra type pack providing GO, LO, map, and graph types.
 */
template<TpetraTypePack Pack>
struct CellPacket {
    using GO = typename Pack::global_ordinal_type;
    using Vec3 = MeshUtils::Vec3;
    using CellType = MeshUtils::CellType;

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
            int boundary_id = Mesh<Pack>::invalid_boundary_id;
            GO neighbor_gid = invalid_id<GO>();
            int neighbor_rank = -1;
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
              + sizeof(std::int32_t) + sizeof(std::uint32_t)
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
        const Mesh<Pack>& mesh,
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
              + sizeof(std::int32_t) + sizeof(std::uint32_t)
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
        const Mesh<Pack>& mesh,
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
                != packet.face_neighbor_ranks.size())
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

/**
 * @brief Parallel mesh partitioner using Zoltan2/ParMETIS.
 *
 * Decomposes source mesh cells into per-rank subdomains via a 5-phase
 * algorithm: (1) Zoltan2 partition, (2) cell redistribution, (3) ghost
 * detection, (4) ghost exchange, (5) mesh rebuild.
 * Source cells may be replicated, distributed, or held only on rank zero.
 *
 * @tparam Pack Tpetra type pack providing GO, LO, map, graph, and comm types.
 */
template<TpetraTypePack Pack>
class MeshPartitioner {
public:
    using GO = typename Pack::global_ordinal_type;
    using LO = typename Pack::local_ordinal_type;
    using map_type = typename Pack::map_type;
    using graph_type = typename Pack::graph_type;
    using comm_type = typename Pack::comm_type;
    using Vec3 = MeshUtils::Vec3;
    using CellInfo = typename Mesh<Pack>::CellInfo;
    using FaceInfo = typename Mesh<Pack>::FaceInfo;
    using Packet = partition_detail::CellPacket<Pack>;
    using PacketView = typename Packet::View;
    using indexer_type =
        Meshes::UnstructuredMesh::local_global_indexer_t<LO, GO>;
    static constexpr LO invalid_lid = invalid_id<LO>();

    /**
     * @brief Local entity lists for a partitioned CRTP unstructured mesh.
     *
     * The input mesh is rebuilt to contain only entities visible on this
     * rank. The indexer maps its compact local IDs to IDs in the original
     * replicated mesh. Owned entities precede overlap entities, and
     * `cell_owner_ranks` is indexed by rebuilt local cell ordinal.
     */
    struct UnstructuredPartition
    {
        indexer_type indexer;
        std::vector<int> cell_owner_ranks;
    };

    /**
     * @brief Main entry point — partition source mesh cells across MPI ranks.
     *
     * Five-phase algorithm:
     * 1. **Zoltan2/ParMETIS partition** — build a CRS graph from cell
     *    adjacency and call Zoltan2 to assign each cell a destination rank.
     * 2. **Cell redistribution** — serialise owned cells into packets,
     *    exchange them via MPI_Alltoallv.
     * 3. **Ghost detection** — for each owned cell, identify adjacent
     *    off-rank cells that are not already owned.
     * 4. **Ghost exchange** — request ghost packets from the ranks that
     *    own them and validate non-owning views over the received bytes.
     * 5. **Mesh rebuild** — reconstruct the per-rank d_cells, d_faces,
     *    node tables, and face geometry from owned + ghost packets.
     *
     * @pre Every rank in @p comm calls this function collectively.
     * @param mesh Source mesh to partition (modified in place).
     * @param comm Teuchos MPI communicator.
     * @return true if partitioning occurred, false if single-rank or
     *         already partitioned.
     */
    static bool partition(Mesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Partition a replicated CRTP unstructured mesh.
     *
     * Cell geometry local IDs are used as graph global IDs. The input mesh
     * is replaced by rank-local geometry and the returned partition
     * contains its rebuilt local/global cell, face, and node indexer.
     *
     * @pre Every rank in @p comm calls this function collectively.
     * @param mesh Replicated geometry, rebuilt in place for the local rank.
     * @param comm Teuchos MPI communicator.
     * @return Global indexing and ownership metadata for the rebuilt mesh.
     */
    static UnstructuredPartition partition(
        Meshes::UnstructuredMesh& mesh,
        const Teuchos::RCP<const comm_type>& comm);

private:
    /**
     * @brief Distributed graph rows and referenced column cell IDs.
     *
     * Rows are owned by this rank; adjacency is parallel to `row_gids`, while
     * `column_gids` is the unique set of local and remote referenced cells.
     */
    struct PartitionGraph
    {
        std::vector<GO> row_gids;
        std::vector<std::vector<GO>> row_adjacency;
        std::vector<GO> column_gids;
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

    /**
     * @brief Owned-first ordering of source entity ordinals.
     *
     * Cells are divided into owned and ghost sets; faces and nodes are divided
     * into owned and overlap sets before the rank-local mesh is rebuilt.
     */
    struct LocalEntityOrder
    {
        std::vector<size_t> owned_cells;
        std::vector<size_t> ghost_cells;
        std::vector<size_t> owned_faces;
        std::vector<size_t> overlap_faces;
        std::vector<size_t> owned_nodes;
        std::vector<size_t> overlap_nodes;
    };

    static PartitionGraph build_partition_graph(
        const Mesh<Pack>& mesh,
        const SourceSelection& source_selection,
        const Teuchos::RCP<const comm_type>& comm);

    static PartitionGraph build_partition_graph(
        const Meshes::UnstructuredMesh& mesh,
        const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Collectively partition a distributed adjacency graph with Zoltan2.
     * @param gather_root Rank that receives the complete result, or -1 to
     *        replicate it on all ranks.
     * @return Destination rank indexed by global cell identifier on the
     *         requested recipients.
     */
    static std::unordered_map<GO, int> solve_partition_graph(
        const PartitionGraph& graph,
        const Teuchos::RCP<const comm_type>& comm,
        int gather_root = -1);

    template<class CellFaces, class OppositeCell, class FaceNodes, class CellNodes>
    static LocalEntityOrder reorder_local_entities(
        size_t cell_count,
        const std::vector<int>& cell_owner_ranks,
        int rank,
        CellFaces&& cell_faces,
        OppositeCell&& opposite_cell,
        FaceNodes&& face_nodes,
        CellNodes&& cell_nodes);

    /**
     * @brief Compute the mapping from cell global ID to destination MPI rank.
     *
     * Builds a Tpetra CRS graph from face-based cell adjacency, wraps it in
     * a Zoltan2 row-graph adapter, and solves the partitioning problem with
     * ParMETIS. The complete result exists only transiently on rank zero;
     * each source rank receives assignments for its cells and their direct
     * neighbors.
     *
     * @param mesh Source mesh whose owned cells are to be mapped.
     * @param source_selection Cells this rank is responsible for providing.
     * @param comm Teuchos MPI communicator.
     * @return Local map from required cell GIDs to destination rank.
     */
    static std::unordered_map<GO, int> compute_gid_to_rank_map(
        const Mesh<Pack>& mesh,
        const SourceSelection& source_selection,
        const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Select exactly one source rank for every available legacy cell.
     *
     * Root-only geometry is represented by a single integer. For replicated
     * or sparse input, rank zero selects sources and scatters only each rank's
     * selected GIDs.
     */
    static SourceSelection compute_source_selection(
        const Mesh<Pack>& mesh,
        const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Collectively assign every unstructured cell to an owner rank.
     * @return Owner rank indexed by source cell ordinal.
     */
    static std::vector<int> compute_unstructured_owner_ranks(
        const Meshes::UnstructuredMesh& mesh,
        const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Replace replicated geometry with this rank's owned-first subset.
     * @return Rebuilt local/global indexer and local cell owners.
     */
    static UnstructuredPartition rebuild(
        Meshes::UnstructuredMesh& mesh,
        std::vector<int> cell_owner_ranks,
        int rank);

    /**
     * @brief Release source-sized legacy storage before local reconstruction.
     */
    static void release_rebuildable_storage(Mesh<Pack>& mesh);

    /**
     * @brief Rebuild the mesh data structures from owned and ghost cell packets.
     *
     * Clears and repopulates d_cells, d_faces, node tables, face lookups,
     * cell-face adjacency, and face geometry (centroids, normals, areas,
     * distances).  Face deduplication uses sorted-node-GID keys.
     *
     * @param mesh Mesh instance to rebuild (modified in place).
     * @param owned_pkts Cell packets destined for this rank.
     * @param ghost_pkts Cell packets describing off-rank neighbours.
     * @param periodic_pairs Face-key → paired-GID map for periodic boundary faces.
     */
    static void rebuild(Mesh<Pack>& mesh, 
                        const std::vector<PacketView>& owned_pkts,
                        const std::vector<PacketView>& ghost_pkts,
                        const std::unordered_map<std::string, GO>& periodic_pairs);


    /**
     * @brief Extract the raw MPI_Comm from a Teuchos communicator wrapper.
     *
     * @param tc Teuchos communicator (must wrap an MpiComm<int>).
     * @return Raw MPI_Comm, or MPI_COMM_NULL if the cast fails.
     */
    static my_mpi::Comm get_mpi_comm(const comm_type& tc) {
        auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&tc);
        if (mpi) return *(mpi->getRawMpiComm()); return MPI_COMM_NULL;
    }
    /**
     * @brief Return the MPI datatype corresponding to the global ordinal type.
     *
     * @return Type-deduced MPI datatype via my_mpi::type_trait.
     */
    static MPI_Datatype mpi_go_type() { return my_mpi::type_trait<GO>(); }
};

} // namespace SimpleFluid
