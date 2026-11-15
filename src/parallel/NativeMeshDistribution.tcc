/** @file NativeMeshDistribution.tcc
 *  @brief Sparse native unstructured record distribution implementation.
 */
#pragma once

#include "parallel/NativeMeshDirectory.hh"

#include <climits>
#include <cmath>
#include <exception>
#include <functional>
#include <set>
#include <string>

namespace SimpleFluid::native_partition_detail
{
using Data = Meshes::UnstructuredMeshData;
using ID = Data::ID;
using Bytes = std::vector<char>;

// The wire contains only fixed-width scalar values and explicit vector lengths;
// native struct padding and container representations never cross the wire.
struct Writer
{
    Bytes bytes;
    template<class T> void scalar(T value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* first = reinterpret_cast<const char*>(&value);
        bytes.insert(bytes.end(), first, first + sizeof(T));
    }
    void ids(const std::vector<ID>& values)
    {
        scalar<std::uint64_t>(values.size());
        for (auto value : values)
            scalar(value);
    }
    void vector(const Data::Vec3& value)
    {
        for (int axis = 0; axis < 3; ++axis)
            scalar(value.component(axis));
    }
    void node(const Data::Node& value)
    {
        scalar(value.id);
        scalar(value.algebraic_id);
        vector(value.coordinates);
    }
    void face(const Data::Face& value)
    {
        scalar(value.id);
        scalar(value.algebraic_id);
        ids(value.node_ids);
        scalar(value.owner);
        scalar(value.neighbor);
        scalar<std::int32_t>(value.boundary_id);
        vector(value.center);
        vector(value.normal_from_owner);
        scalar(value.area);
    }
    void cell(const Data::Cell& value)
    {
        scalar(value.id);
        scalar(value.algebraic_id);
        scalar<std::int32_t>(static_cast<int>(value.type));
        ids(value.node_ids);
        ids(value.face_ids);
        vector(value.center);
        scalar(value.volume);
    }
    void blob(const Bytes& value)
    {
        scalar<std::uint64_t>(value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
};

struct Reader
{
    const Bytes& bytes;
    size_t offset = 0;
    template<class T> T scalar()
    {
        if (sizeof(T) > bytes.size() - offset)
            throw std::invalid_argument("Truncated native mesh record.");
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }
    std::vector<ID> ids()
    {
        const auto count = scalar<std::uint64_t>();
        if (count > (bytes.size() - offset) / sizeof(ID))
            throw std::invalid_argument("Invalid native ID count.");
        std::vector<ID> values;
        values.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i)
            values.push_back(scalar<ID>());
        return values;
    }
    Data::Vec3 vector()
    {
        Data::Vec3 value{};
        for (int axis = 0; axis < 3; ++axis)
            value.component(axis) = scalar<real_t>();
        return value;
    }
    Data::Node node()
    {
        Data::Node value;
        value.id = scalar<ID>();
        value.algebraic_id = scalar<ID>();
        value.coordinates = vector();
        return value;
    }
    Data::Face face()
    {
        Data::Face value;
        value.id = scalar<ID>();
        value.algebraic_id = scalar<ID>();
        value.node_ids = ids();
        value.owner = scalar<ID>();
        value.neighbor = scalar<ID>();
        value.boundary_id = scalar<std::int32_t>();
        value.center = vector();
        value.normal_from_owner = vector();
        value.area = scalar<real_t>();
        return value;
    }
    Data::Cell cell()
    {
        Data::Cell value;
        value.id = scalar<ID>();
        value.algebraic_id = scalar<ID>();
        value.type = static_cast<MeshUtils::CellType>(scalar<std::int32_t>());
        value.node_ids = ids();
        value.face_ids = ids();
        value.center = vector();
        value.volume = scalar<real_t>();
        return value;
    }
    Bytes blob()
    {
        const auto count = scalar<std::uint64_t>();
        if (count > bytes.size() - offset)
            throw std::invalid_argument("Invalid native mesh blob length.");
        Bytes value(bytes.begin() + offset, bytes.begin() + offset + count);
        offset += count;
        return value;
    }
    bool done() const { return offset == bytes.size(); }
};

inline Bytes packet(const Data& data)
{
    Writer out;
    out.cell(data.cells.at(0));
    out.template scalar<std::uint64_t>(data.faces.size());
    for (const auto& value : data.faces)
        out.face(value);
    out.template scalar<std::uint64_t>(data.nodes.size());
    for (const auto& value : data.nodes)
        out.node(value);
    return std::move(out.bytes);
}
inline Data unpack(const Bytes& bytes)
{
    Reader in{bytes};
    Data data;
    data.cells.push_back(in.cell());
    auto count = in.template scalar<std::uint64_t>();
    if (count > bytes.size())
        throw std::invalid_argument("Invalid native face count.");
    for (std::uint64_t i = 0; i < count; ++i)
        data.faces.push_back(in.face());
    count = in.template scalar<std::uint64_t>();
    if (count > bytes.size())
        throw std::invalid_argument("Invalid native node count.");
    for (std::uint64_t i = 0; i < count; ++i)
        data.nodes.push_back(in.node());
    if (!in.done())
        throw std::invalid_argument("Trailing native mesh bytes.");
    return data;
}

/** Reduce rank-local failures before entering the next collective phase. */
template<class Comm, class Function> void phase(const Comm& comm, const char* name, Function&& function)
{
    std::exception_ptr failure;
    try
    {
        function();
    }
    catch (...)
    {
        failure = std::current_exception();
    }
    int local = failure ? 1 : 0, failed = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &failed);
    if (failed)
    {
        // Allocate diagnostics only after every rank agrees to leave the
        // collective sequence. The failure handler itself remains nonallocating.
        if (failure)
        {
            try
            {
                std::rethrow_exception(failure);
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string("Native mesh distribution ") + name + ": " + error.what());
            }
            catch (...)
            {
                throw std::runtime_error(std::string("Native mesh distribution ") + name + ": unknown local failure");
            }
        }
        throw std::runtime_error(std::string("Native mesh distribution ") + name + ": failure on another rank");
    }
}

/** Bounded sparse exchange; only per-rank counts are replicated. */
template<class Comm> std::vector<Bytes> exchange(const std::vector<Writer>& outgoing, const Comm& comm, MPI_Comm mpi)
{
    const int ranks = comm.getSize();
    std::vector<int> sends, receives, send_offsets, receive_offsets;
    Bytes send, receive;
    phase(comm, "exchange preparation",
        [&]
        {
            sends.resize(ranks);
            receives.resize(ranks);
            send_offsets.resize(ranks);
            receive_offsets.resize(ranks);
            size_t total = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                const auto count = outgoing.at(rank).bytes.size();
                if (count > INT_MAX || total > INT_MAX - count)
                    throw std::overflow_error("Native MPI byte count overflow.");
                send_offsets[rank] = static_cast<int>(total);
                sends[rank] = static_cast<int>(count);
                total += count;
            }
            send.reserve(total);
            for (const auto& writer : outgoing)
                send.insert(send.end(), writer.bytes.begin(), writer.bytes.end());
        });
    if (ranks == 1)
        return {std::move(send)};
    MPI_Alltoall(sends.data(), 1, MPI_INT, receives.data(), 1, MPI_INT, mpi);
    phase(comm, "exchange allocation",
        [&]
        {
            size_t total = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                if (receives[rank] < 0 || total > static_cast<size_t>(INT_MAX - receives[rank]))
                    throw std::overflow_error("Native MPI receive count overflow.");
                receive_offsets[rank] = static_cast<int>(total);
                total += receives[rank];
            }
            receive.resize(total);
        });
    MPI_Alltoallv(send.data(), sends.data(), send_offsets.data(), MPI_BYTE, receive.data(), receives.data(),
        receive_offsets.data(), MPI_BYTE, mpi);
    std::vector<Bytes> incoming;
    phase(comm, "exchange decoding",
        [&]
        {
            incoming.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank)
                incoming[rank] = Bytes(
                    receive.begin() + receive_offsets[rank], receive.begin() + receive_offsets[rank] + receives[rank]);
        });
    return incoming;
}

inline size_t face_hash(std::vector<ID> nodes)
{
    std::sort(nodes.begin(), nodes.end());
    size_t hash = 0;
    for (auto node : nodes)
        hash ^= std::hash<ID>{}(node) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    return hash;
}
inline ID algebraic(ID source, ID explicit_id)
{
    return explicit_id == Data::invalid_id ? source : explicit_id;
}
} // namespace SimpleFluid::native_partition_detail

namespace SimpleFluid
{
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::partition_native(Meshes::UnstructuredMeshData source,
    const Teuchos::RCP<const comm_type>& comm, unsigned halo_depth, bool contiguous_algebraic_ids)
    -> NativeUnstructuredPartition
{
    namespace detail = native_partition_detail;
    using Data = detail::Data;
    using ID = detail::ID;
    using Writer = detail::Writer;
    using Reader = detail::Reader;
    using Bytes = detail::Bytes;
    if (comm.is_null())
        throw std::invalid_argument("Native mesh distribution requires a communicator.");
    const int rank = comm->getRank(), ranks = comm->getSize();
    const auto mpi = get_mpi_comm(*comm);
    const auto directory_rank = [ranks](ID id) { return detail::directory_rank(id, ranks); };
    const auto checked_gid = [](ID id) -> GO
    {
        if (id == Data::invalid_id || !std::in_range<GO>(id))
            throw std::overflow_error("Native mesh mapped entity ID exceeds global ordinal range.");
        return static_cast<GO>(id);
    };
    unsigned min_depth = 0, max_depth = 0;
    int contiguous_policy = contiguous_algebraic_ids ? 1 : 0, min_policy = 0, max_policy = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &contiguous_policy, &min_policy);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &contiguous_policy, &max_policy);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &halo_depth, &min_depth);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &halo_depth, &max_depth);
    std::vector<Writer> outgoing;
    std::unordered_map<ID, Bytes> directory_packets;
    std::unordered_map<int, std::string> catalog;
    detail::phase(*comm, "source validation",
        [&]
        {
            if (min_policy != max_policy)
                throw std::invalid_argument("Cell numbering policy must agree on every rank.");
            if (!halo_depth || min_depth != max_depth)
                throw std::invalid_argument("Halo depth must agree and be positive.");
            if (ranks > 1 && mpi == MPI_COMM_NULL)
                throw std::invalid_argument("Expected an MPI communicator.");
            const auto owned = source.num_owned_cells == Data::all_owned ? source.cells.size() : source.num_owned_cells;
            if (owned > source.cells.size())
                throw std::invalid_argument("Invalid owned source cell count.");
            if ((source.num_owned_faces != Data::all_owned && source.num_owned_faces > source.faces.size()) ||
                (source.num_owned_nodes != Data::all_owned && source.num_owned_nodes > source.nodes.size()))
                throw std::invalid_argument("Invalid owned source entity count.");
            const auto finite_vector = [](const auto& vector)
            { return std::isfinite(vector.x) && std::isfinite(vector.y) && std::isfinite(vector.z); };
            std::unordered_map<ID, const Data::Node*> nodes;
            std::unordered_map<ID, const Data::Face*> faces;
            for (const auto& node : source.nodes)
            {
                if (node.id == Data::invalid_id || !nodes.emplace(node.id, &node).second)
                    throw std::invalid_argument("Duplicate or invalid source node ID.");
            }
            for (const auto& face : source.faces)
            {
                if (face.id == Data::invalid_id || !faces.emplace(face.id, &face).second)
                    throw std::invalid_argument("Duplicate or invalid source face ID.");
            }
            std::unordered_set<ID> cells;
            outgoing.resize(ranks);
            for (size_t local = 0; local < owned; ++local)
            {
                const auto& cell = source.cells[local];
                if (!finite_vector(cell.center) || !std::isfinite(cell.volume) || cell.volume <= 0)
                    throw std::invalid_argument("Invalid source cell metrics.");
                checked_gid(cell.id);
                checked_gid(detail::algebraic(cell.id, cell.algebraic_id));
                if (!cells.insert(cell.id).second)
                    throw std::invalid_argument("Duplicate source cell ID.");
                if (cell.face_ids.empty() || cell.node_ids.empty())
                    throw std::invalid_argument("Empty source cell topology.");
                Data slice;
                slice.cells.push_back(cell);
                std::set<ID> needed_nodes(cell.node_ids.begin(), cell.node_ids.end());
                std::unordered_set<ID> seen_faces;
                for (auto face_id : cell.face_ids)
                {
                    if (!seen_faces.insert(face_id).second)
                        throw std::invalid_argument("Repeated source cell face.");
                    const auto& face = *faces.at(face_id);
                    checked_gid(face.id);
                    checked_gid(detail::algebraic(face.id, face.algebraic_id));
                    if (!finite_vector(face.center) || !finite_vector(face.normal_from_owner) ||
                        !std::isfinite(face.area) || face.area <= 0)
                        throw std::invalid_argument("Invalid source face metrics.");
                    if (face.owner != cell.id && face.neighbor != cell.id)
                        throw std::invalid_argument("Source cell/face incidence disagrees.");
                    if (face.owner == Data::invalid_id || face.owner == face.neighbor || face.node_ids.size() < 2)
                        throw std::invalid_argument("Invalid source face topology.");

                    if (face.boundary_id >= 0 && !source.boundary_names.contains(face.boundary_id))
                        throw std::invalid_argument("Source face boundary is absent from catalog.");
                    slice.faces.push_back(face);
                    needed_nodes.insert(face.node_ids.begin(), face.node_ids.end());
                }
                for (auto id : needed_nodes)
                {
                    const auto& node = *nodes.at(id);
                    checked_gid(node.id);
                    checked_gid(detail::algebraic(node.id, node.algebraic_id));
                    if (!finite_vector(node.coordinates))
                        throw std::invalid_argument("Nonfinite source node coordinates.");
                    slice.nodes.push_back(node);
                }
                outgoing[directory_rank(cell.id)].blob(detail::packet(slice));
            }
        });
    auto incoming = detail::exchange(outgoing, *comm, mpi);
    detail::phase(*comm, "unique cell directory",
        [&]
        {
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    auto packet = reader.blob();
                    const auto data = detail::unpack(packet);
                    if (!directory_packets.emplace(data.cells[0].id, std::move(packet)).second)
                        throw std::invalid_argument("Duplicate native source cell across ranks.");
                }
            }
            // Catalog is metadata, not geometry; every rank must retain it, even
            // ranks that own no cells. Conflicting names are rejected below.
            outgoing.assign(ranks, {});
            for (auto& writer : outgoing)
                for (const auto& [id, name] : source.boundary_names)
                {
                    if (id < 0)
                        throw std::invalid_argument("Invalid source boundary ID.");
                    writer.template scalar<std::int32_t>(id);
                    writer.blob(Bytes(name.begin(), name.end()));
                }
            source = {};
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    PartitionGraph graph;
    std::vector<std::uint64_t> directory_counts;
    std::uint64_t local_count = directory_packets.size();
    detail::phase(*comm, "directory count allocation", [&] { directory_counts.resize(ranks); });
    Teuchos::gatherAll(*comm, 1, &local_count, ranks, directory_counts.data());
    std::uint64_t total_cells = 0, directory_first = 0;
    std::unordered_map<ID, GO> directory_graph_ids;
    detail::phase(*comm, "catalog and graph IDs",
        [&]
        {
            for (int r = 0; r < ranks; ++r)
            {
                if (r < rank)
                    directory_first += directory_counts[r];
                if (directory_counts[r] > std::numeric_limits<std::uint64_t>::max() - total_cells)
                    throw std::overflow_error("Native global cell count overflow.");
                total_cells += directory_counts[r];
            }
            if (total_cells)
                checked_gid(total_cells - 1);
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const int id = reader.template scalar<std::int32_t>();
                    const auto text = reader.blob();
                    const std::string name(text.begin(), text.end());
                    auto [it, inserted] = catalog.emplace(id, name);
                    if (!inserted && it->second != name)
                        throw std::invalid_argument("Conflicting boundary catalog names.");
                }
            }
            std::vector<ID> ids;
            for (const auto& [id, bytes] : directory_packets)
                ids.push_back(id);
            std::sort(ids.begin(), ids.end());
            for (size_t i = 0; i < ids.size(); ++i)
                directory_graph_ids.emplace(ids[i], checked_gid(directory_first + i));
            // Query only referenced source IDs. Graph IDs are dense independent
            // ordinals, so sparse 64-bit source IDs never narrow to Tpetra GO.
            std::set<ID> references;
            for (auto id : ids)
            {
                const auto data = detail::unpack(directory_packets.at(id));
                for (const auto& face : data.faces)
                {
                    const auto other = face.owner == id ? face.neighbor : face.owner;
                    if (other != Data::invalid_id)
                        references.insert(other);
                }
            }
            outgoing.assign(ranks, {});
            for (auto id : references)
                outgoing[directory_rank(id)].scalar(id);
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    detail::phase(*comm, "adjacency directory",
        [&]
        {
            outgoing.assign(ranks, {});
            for (int requester = 0; requester < ranks; ++requester)
            {
                Reader reader{incoming[requester]};
                while (!reader.done())
                {
                    const auto id = reader.template scalar<ID>();
                    const auto found = directory_graph_ids.find(id);
                    if (found == directory_graph_ids.end())
                        throw std::invalid_argument("Missing source neighbor cell.");
                    outgoing[requester].scalar(id);
                    outgoing[requester].template scalar<ID>(found->second);
                }
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    const auto graph_rank = [&](ID graph_id)
    {
        // Count-balanced row distribution, including nonempty row sets on
        // every graph participant whenever global cell count permits it.
        const auto base = total_cells / ranks, remainder = total_cells % ranks;
        const auto large_end = (base + 1) * remainder;
        return graph_id < large_end ? static_cast<int>(graph_id / (base + 1))
                                    : static_cast<int>(remainder + (graph_id - large_end) / base);
    };
    detail::phase(*comm, "graph row routing",
        [&]
        {
            auto references = directory_graph_ids;
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const auto id = reader.template scalar<ID>();
                    references.emplace(id, checked_gid(reader.template scalar<ID>()));
                }
            }
            outgoing.assign(ranks, {});
            for (const auto& [id, bytes] : directory_packets)
            {
                const auto data = detail::unpack(bytes);
                const auto gid = directory_graph_ids.at(id);
                std::set<ID> neighbors;
                for (const auto& face : data.faces)
                {
                    const auto other = face.owner == id ? face.neighbor : face.owner;
                    if (other != Data::invalid_id)
                        neighbors.insert(static_cast<ID>(references.at(other)));
                }
                auto& writer = outgoing[graph_rank(static_cast<ID>(gid))];
                writer.template scalar<ID>(gid);
                writer.scalar(id);
                writer.ids(std::vector<ID>(neighbors.begin(), neighbors.end()));
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    std::unordered_map<GO, ID> source_by_graph_id;
    detail::phase(*comm, "graph row assembly",
        [&]
        {
            std::set<GO> columns;
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const auto gid = checked_gid(reader.template scalar<ID>());
                    const auto source_id = reader.template scalar<ID>();
                    auto neighbors = reader.ids();
                    graph.row_gids.push_back(gid);
                    columns.insert(gid);
                    source_by_graph_id.emplace(gid, source_id);
                    auto& adjacency = graph.row_adjacency.emplace_back();
                    for (auto neighbor : neighbors)
                    {
                        adjacency.push_back(checked_gid(neighbor));
                        columns.insert(checked_gid(neighbor));
                    }
                }
            }
            graph.column_gids.assign(columns.begin(), columns.end());
        });
    // Validate shared authority before entering ParMETIS: inconsistent
    // incidence must not reach a backend with mismatched graph assumptions.
    detail::phase(*comm, "shared entity routing",
        [&]
        {
            outgoing.assign(ranks, {});
            for (const auto& [id, bytes] : directory_packets)
            {
                const auto data = detail::unpack(bytes);
                const int owner = rank;
                const auto cell_ordinal = detail::algebraic(id, data.cells[0].algebraic_id);
                auto& cell_writer = outgoing[directory_rank(cell_ordinal)];
                cell_writer.template scalar<std::uint8_t>(3);
                cell_writer.scalar(cell_ordinal);
                cell_writer.scalar(id);
                // Shared face agreement is keyed by node set, so two different IDs
                // for one physical face are also diagnosed. Ordered node loops
                // and source metrics are part of the compared record.
                for (const auto& face : data.faces)
                {
                    auto& writer = outgoing[detail::face_hash(face.node_ids) % ranks];
                    writer.template scalar<std::uint8_t>(0);
                    writer.face(face);
                    writer.scalar(id);
                    const auto ordinal = detail::algebraic(face.id, face.algebraic_id);
                    auto& ordinal_writer = outgoing[directory_rank(ordinal)];
                    ordinal_writer.template scalar<std::uint8_t>(4);
                    ordinal_writer.scalar(ordinal);
                    ordinal_writer.scalar(face.id);
                    auto& identity_writer = outgoing[directory_rank(face.id)];
                    identity_writer.template scalar<std::uint8_t>(2);
                    identity_writer.face(face);
                }
                for (const auto& node : data.nodes)
                {
                    auto& writer = outgoing[directory_rank(node.id)];
                    writer.template scalar<std::uint8_t>(1);
                    writer.node(node);
                    writer.template scalar<std::int32_t>(owner);
                    const auto ordinal = detail::algebraic(node.id, node.algebraic_id);
                    auto& ordinal_writer = outgoing[directory_rank(ordinal)];
                    ordinal_writer.template scalar<std::uint8_t>(5);
                    ordinal_writer.scalar(ordinal);
                    ordinal_writer.scalar(node.id);
                }
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    std::unordered_map<ID, int> directory_node_owners;
    detail::phase(*comm, "shared entity agreement",
        [&]
        {
            std::map<std::vector<ID>, Bytes> face_records;
            std::map<std::vector<ID>, std::set<ID>> face_incidence;
            std::map<std::pair<std::uint8_t, ID>, ID> ordinal_records;
            std::unordered_map<ID, std::vector<ID>> face_keys;
            std::unordered_map<ID, Bytes> node_records;
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const auto kind = reader.template scalar<std::uint8_t>();
                    if (kind == 0 || kind == 2)
                    {
                        const auto face = reader.face();
                        auto key = face.node_ids;
                        std::sort(key.begin(), key.end());
                        auto canonical = face;
                        canonical.algebraic_id = detail::algebraic(face.id, face.algebraic_id);
                        Writer writer;
                        writer.face(canonical);
                        if (kind == 0)
                        {
                            face_incidence[key].insert(reader.template scalar<ID>());
                            const auto [it, inserted] = face_records.emplace(key, writer.bytes);
                            if (!inserted && it->second != writer.bytes)
                                throw std::invalid_argument("Inconsistent shared source face.");
                        }
                        else
                        {
                            const auto [known, new_id] = face_keys.emplace(face.id, key);
                            if (!new_id && known->second != key)
                                throw std::invalid_argument("Source face ID reused for different topology.");
                        }
                    }
                    else if (kind >= 3 && kind <= 5)
                    {
                        const auto ordinal = reader.template scalar<ID>(), source_id = reader.template scalar<ID>();
                        const auto [it, inserted] = ordinal_records.emplace(std::make_pair(kind, ordinal), source_id);
                        if (!inserted && it->second != source_id)
                            throw std::invalid_argument(
                                "Native algebraic ordinal reused for distinct source entities.");
                    }
                    else if (kind == 1)
                    {
                        const auto node = reader.node();
                        const int owner = reader.template scalar<std::int32_t>();
                        Writer writer;
                        auto canonical = node;
                        canonical.algebraic_id = detail::algebraic(node.id, node.algebraic_id);
                        writer.node(canonical);
                        const auto [it, inserted] = node_records.emplace(node.id, writer.bytes);
                        if (!inserted && it->second != writer.bytes)
                            throw std::invalid_argument("Inconsistent shared source node.");
                        const auto [where, first] = directory_node_owners.emplace(node.id, owner);
                        if (!first)
                            where->second = std::min(where->second, owner);
                    }
                }
            }
            for (const auto& [key, bytes] : face_records)
            {
                Reader reader{bytes};
                const auto face = reader.face();
                std::set<ID> expected{face.owner};
                if (face.neighbor != Data::invalid_id)
                    expected.insert(face.neighbor);
                if (face_incidence.at(key) != expected)
                    throw std::invalid_argument("Source face is absent from an incident source cell.");
            }
        });
    // ParMETIS requires nonempty graph participants. As in the composite
    // distributor, meshes smaller than the communicator use their balanced
    // graph-row placement directly, retaining valid empty destinations.
    std::unordered_map<GO, int> assignments;
    if (ranks > 1 && total_cells >= static_cast<ID>(ranks))
        assignments = solve_partition_graph(graph, comm, -2);
    else
        detail::phase(*comm, "small graph assignments",
            [&]
            {
                for (auto gid : graph.row_gids)
                    assignments.emplace(gid, rank);
            });
    std::unordered_map<ID, int> directory_owners;
    detail::phase(*comm, "assignment routing",
        [&]
        {
            outgoing.assign(ranks, {});
            for (const auto& [id, owner] : assignments)
            {
                if (owner < 0 || owner >= ranks)
                    throw std::runtime_error("Invalid native cell destination.");
                const auto source_id = source_by_graph_id.at(id);
                auto& writer = outgoing[directory_rank(source_id)];
                writer.scalar(source_id);
                writer.template scalar<std::int32_t>(owner);
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    detail::phase(*comm, "assignment validation",
        [&]
        {
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const auto id = reader.template scalar<ID>();
                    const auto owner = reader.template scalar<std::int32_t>();
                    if (!directory_packets.contains(id) || !directory_owners.emplace(id, owner).second)
                        throw std::runtime_error("Unknown or duplicate cell assignment.");
                }
            }
            if (directory_owners.size() != directory_packets.size())
                throw std::runtime_error("Incomplete cell assignments.");
            outgoing.assign(ranks, {});
            for (const auto& [id, bytes] : directory_packets)
            {
                const auto data = detail::unpack(bytes);
                for (const auto& node : data.nodes)
                {
                    auto& writer = outgoing[directory_rank(node.id)];
                    writer.scalar(node.id);
                    writer.template scalar<std::int32_t>(directory_owners.at(id));
                }
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    detail::phase(*comm, "node ownership agreement",
        [&]
        {
            directory_node_owners.clear();
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const auto id = reader.template scalar<ID>();
                    const auto owner = reader.template scalar<std::int32_t>();
                    const auto [where, first] = directory_node_owners.emplace(id, owner);
                    if (!first)
                        where->second = std::min(where->second, owner);
                }
            }
            outgoing.assign(ranks, {});
            for (const auto& [id, bytes] : directory_packets)
            {
                auto& writer = outgoing[directory_owners.at(id)];
                writer.template scalar<std::int32_t>(directory_owners.at(id));
                writer.blob(bytes);
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    std::map<ID, Data> local_packets;
    std::unordered_map<ID, int> local_owners;
    std::set<ID> frontier;
    const auto accept_packets = [&]
    {
        for (const auto& bytes : incoming)
        {
            Reader reader{bytes};
            while (!reader.done())
            {
                const int owner = reader.template scalar<std::int32_t>();
                auto data = detail::unpack(reader.blob());
                const auto id = data.cells[0].id;
                if (local_packets.emplace(id, std::move(data)).second)
                    frontier.insert(id);
                local_owners[id] = owner;
            }
        }
    };
    detail::phase(*comm, "owned cell reception", accept_packets);
    if (contiguous_algebraic_ids)
    {
        local_count = local_packets.size();
        Teuchos::gatherAll(*comm, 1, &local_count, ranks, directory_counts.data());
        detail::phase(*comm, "contiguous cell numbering",
            [&]
            {
                ID first = 0;
                for (int owner = 0; owner < rank; ++owner)
                    first += directory_counts[owner];
                outgoing.assign(ranks, {});
                for (auto& [id, data] : local_packets)
                {
                    data.cells[0].algebraic_id = first++;
                    auto& writer = outgoing[directory_rank(id)];
                    writer.scalar(id);
                    writer.scalar(data.cells[0].algebraic_id);
                }
            });
        incoming = detail::exchange(outgoing, *comm, mpi);
        detail::phase(*comm, "contiguous directory update",
            [&]
            {
                for (const auto& bytes : incoming)
                {
                    Reader reader{bytes};
                    while (!reader.done())
                    {
                        const auto id = reader.template scalar<ID>();
                        const auto ordinal = reader.template scalar<ID>();
                        auto data = detail::unpack(directory_packets.at(id));
                        data.cells[0].algebraic_id = ordinal;
                        directory_packets.at(id) = detail::packet(data);
                    }
                }
            });
    }
    for (unsigned layer = 0; layer < halo_depth; ++layer)
    {
        detail::phase(*comm, "halo requests",
            [&]
            {
                std::set<ID> requests;
                for (auto id : frontier)
                    for (const auto& face : local_packets.at(id).faces)
                    {
                        const auto other = face.owner == id ? face.neighbor : face.owner;
                        if (other != Data::invalid_id && !local_packets.contains(other))
                            requests.insert(other);
                    }
                frontier.clear();
                outgoing.assign(ranks, {});
                for (auto id : requests)
                    outgoing[directory_rank(id)].scalar(id);
            });
        incoming = detail::exchange(outgoing, *comm, mpi);
        detail::phase(*comm, "halo replies",
            [&]
            {
                outgoing.assign(ranks, {});
                for (int requester = 0; requester < ranks; ++requester)
                {
                    Reader reader{incoming[requester]};
                    auto& writer = outgoing[requester];
                    while (!reader.done())
                    {
                        const auto id = reader.template scalar<ID>();
                        writer.template scalar<std::int32_t>(directory_owners.at(id));
                        writer.blob(directory_packets.at(id));
                    }
                }
            });
        incoming = detail::exchange(outgoing, *comm, mpi);
        detail::phase(*comm, "halo reception", accept_packets);
    }
    detail::phase(*comm, "completed directory release",
        [&]
        {
            // All halo replies have been received and decoded. Keep only the
            // node-owner directory needed by the remaining sparse queries.
            decltype(directory_packets){}.swap(directory_packets);
            decltype(directory_owners){}.swap(directory_owners);
            decltype(directory_graph_ids){}.swap(directory_graph_ids);
            decltype(source_by_graph_id){}.swap(source_by_graph_id);
            decltype(assignments){}.swap(assignments);
            graph = {};
            incoming.clear();
            outgoing.clear();
        });
    std::set<ID> needed_nodes;
    detail::phase(*comm, "node ownership queries",
        [&]
        {
            for (const auto& [id, data] : local_packets)
                for (const auto& node : data.nodes)
                    needed_nodes.insert(node.id);
            outgoing.assign(ranks, {});
            for (auto id : needed_nodes)
                outgoing[directory_rank(id)].scalar(id);
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    detail::phase(*comm, "node ownership replies",
        [&]
        {
            outgoing.assign(ranks, {});
            for (int requester = 0; requester < ranks; ++requester)
            {
                Reader reader{incoming[requester]};
                while (!reader.done())
                {
                    auto id = reader.template scalar<ID>();
                    outgoing[requester].scalar(id);
                    outgoing[requester].template scalar<std::int32_t>(directory_node_owners.at(id));
                }
            }
        });
    incoming = detail::exchange(outgoing, *comm, mpi);
    NativeUnstructuredPartition result;
    detail::phase(*comm, "native publication",
        [&]
        {
            std::unordered_map<ID, int> node_owners;
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    const auto id = reader.template scalar<ID>();
                    node_owners.emplace(id, reader.template scalar<std::int32_t>());
                }
            }
            Data local;
            local.boundary_names = std::move(catalog);
            std::map<ID, Data::Face> faces;
            std::map<ID, Data::Node> nodes;
            std::set<ID> owned_faces;
            for (int owned_pass = 1; owned_pass >= 0; --owned_pass)
                for (const auto& [id, packet] : local_packets)
                {
                    if ((local_owners.at(id) == rank) != static_cast<bool>(owned_pass))
                        continue;
                    local.cells.push_back(packet.cells[0]);
                    result.cell_owner_ranks.push_back(local_owners.at(id));
                    for (const auto& face : packet.faces)
                    {
                        faces.emplace(face.id, face);
                        if (owned_pass)
                            owned_faces.insert(face.id);
                    }
                    for (const auto& node : packet.nodes)
                        nodes.emplace(node.id, node);
                }
            local.num_owned_cells = std::count(result.cell_owner_ranks.begin(), result.cell_owner_ranks.end(), rank);
            for (int owned_pass = 1; owned_pass >= 0; --owned_pass)
                for (const auto& [id, record] : faces)
                {
                    if (owned_faces.contains(id) != static_cast<bool>(owned_pass))
                        continue;
                    auto face = record;
                    if (!local_packets.contains(face.owner))
                    {
                        if (!local_packets.contains(face.neighbor))
                            throw std::runtime_error("Face has no local incident cell.");
                        std::swap(face.owner, face.neighbor);
                        for (int axis = 0; axis < 3; ++axis)
                            face.normal_from_owner.component(axis) = -face.normal_from_owner.component(axis);
                    }
                    local.faces.push_back(std::move(face));
                }
            local.num_owned_faces = owned_faces.size();
            for (int owned_pass = 1; owned_pass >= 0; --owned_pass)
                for (const auto& [id, node] : nodes)
                    if ((node_owners.at(id) == rank) == static_cast<bool>(owned_pass))
                        local.nodes.push_back(node);
            local.num_owned_nodes = std::count_if(local.nodes.begin(), local.nodes.end(),
                [&](const auto& node) { return node_owners.at(node.id) == rank; });
            auto geometry = std::make_shared<Meshes::UnstructuredMesh>(Meshes::UnstructuredMesh::from_data(local));
            std::vector<typename indexer_type::CellMapping> owned_cells, ghost_cells;
            std::vector<typename indexer_type::FaceMapping> owned_face_maps, overlap_faces;
            std::vector<typename indexer_type::NodeMapping> owned_nodes, overlap_nodes;
            for (size_t i = 0; i < local.cells.size(); ++i)
            {
                const auto& cell = local.cells[i];
                (i < local.num_owned_cells ? owned_cells : ghost_cells)
                    .push_back(
                        {geometry->cell_id(i), cell.id, checked_gid(detail::algebraic(cell.id, cell.algebraic_id))});
            }
            for (size_t i = 0; i < local.faces.size(); ++i)
            {
                const auto& face = local.faces[i];
                (i < local.num_owned_faces ? owned_face_maps : overlap_faces)
                    .push_back(
                        {geometry->face_id(i), face.id, checked_gid(detail::algebraic(face.id, face.algebraic_id))});
            }
            for (size_t i = 0; i < local.nodes.size(); ++i)
            {
                const auto& node = local.nodes[i];
                (i < local.num_owned_nodes ? owned_nodes : overlap_nodes)
                    .push_back(
                        {geometry->node_id(i), node.id, checked_gid(detail::algebraic(node.id, node.algebraic_id))});
            }
            result.geometry = std::move(geometry);
            result.indexer = indexer_type(std::move(owned_cells), std::move(ghost_cells), std::move(owned_face_maps),
                std::move(overlap_faces), std::move(owned_nodes), std::move(overlap_nodes));
        });
    return result;
}
} // namespace SimpleFluid
