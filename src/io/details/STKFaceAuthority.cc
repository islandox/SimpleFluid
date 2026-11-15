/** @file STKFaceAuthority.cc
 *  @brief Sparse whole-record face authority for transient mesh input.
 */
#include "io/details/STKFaceAuthority.hh"
#include "parallel/NativeMeshDirectory.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid::detail
{
namespace
{
using Data = Meshes::UnstructuredMeshData;
using ID = Data::ID;
using Bytes = std::vector<char>;
using Key = std::vector<ID>;

// No allocation is performed in the failure handler before agreement.
template<class Function> void phase(const Teuchos::Comm<int>& comm, const char* name, Function&& function)
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
    const int local = failure ? 1 : 0;
    int any = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &any);
    if (any)
    {
        if (failure)
            std::rethrow_exception(failure);
        throw std::runtime_error(std::string("STK input face authority failed on another rank: ") + name);
    }
}

struct Writer
{
    Bytes bytes;
    template<class T> void scalar(T value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* first = reinterpret_cast<const char*>(&value);
        bytes.insert(bytes.end(), first, first + sizeof(T));
    }
    void ids(const Key& values)
    {
        scalar<std::uint64_t>(values.size());
        for (auto value : values)
            scalar(value);
    }
    void vector(const Data::Vec3& value)
    {
        for (size_t axis = 0; axis < 3; ++axis)
            scalar(value.component(axis));
    }
    void text(const std::string& value)
    {
        scalar<std::uint64_t>(value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
    void face(const Data::Face& value)
    {
        scalar(value.id);
        ids(value.node_ids);
        scalar(value.owner);
        scalar(value.neighbor);
        scalar<std::int32_t>(value.boundary_id);
        vector(value.center);
        vector(value.normal_from_owner);
        scalar(value.area);
        scalar(value.algebraic_id);
    }
};

struct Reader
{
    const Bytes& bytes;
    size_t offset = 0;
    template<class T> T scalar()
    {
        if (sizeof(T) > bytes.size() - offset)
            throw std::invalid_argument("Truncated input face record.");
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }
    Key ids()
    {
        const auto count = scalar<std::uint64_t>();
        if (count > (bytes.size() - offset) / sizeof(ID))
            throw std::invalid_argument("Invalid input face ID count.");
        Key values;
        values.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i)
            values.push_back(scalar<ID>());
        return values;
    }
    Data::Vec3 vector()
    {
        Data::Vec3 value;
        for (size_t axis = 0; axis < 3; ++axis)
            value.component(axis) = scalar<real_t>();
        return value;
    }
    std::string text()
    {
        const auto count = scalar<std::uint64_t>();
        if (count > bytes.size() - offset)
            throw std::invalid_argument("Invalid input boundary name length.");
        std::string result(bytes.data() + offset, static_cast<size_t>(count));
        offset += count;
        return result;
    }
    Data::Face face()
    {
        Data::Face value;
        value.id = scalar<ID>();
        value.node_ids = ids();
        value.owner = scalar<ID>();
        value.neighbor = scalar<ID>();
        value.boundary_id = scalar<std::int32_t>();
        value.center = vector();
        value.normal_from_owner = vector();
        value.area = scalar<real_t>();
        value.algebraic_id = scalar<ID>();
        return value;
    }
    bool done() const noexcept { return offset == bytes.size(); }
};

std::vector<Bytes> exchange(const std::vector<Writer>& outgoing, const Teuchos::Comm<int>& comm, MPI_Comm mpi)
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
                if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    total > static_cast<size_t>(std::numeric_limits<int>::max()) - count)
                    throw std::overflow_error("Input face exchange exceeds MPI int count.");
                sends[rank] = static_cast<int>(count);
                send_offsets[rank] = static_cast<int>(total);
                total += count;
            }
            send.reserve(total);
            for (const auto& writer : outgoing)
                send.insert(send.end(), writer.bytes.begin(), writer.bytes.end());
        });
    MPI_Alltoall(sends.data(), 1, MPI_INT, receives.data(), 1, MPI_INT, mpi);
    phase(comm, "receive allocation",
        [&]
        {
            size_t total = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                if (receives[rank] < 0 || total > static_cast<size_t>(std::numeric_limits<int>::max() - receives[rank]))
                    throw std::overflow_error("Input face receive buffer exceeds MPI int count.");
                receive_offsets[rank] = static_cast<int>(total);
                total += receives[rank];
            }
            receive.resize(total);
        });
    MPI_Alltoallv(send.data(), sends.data(), send_offsets.data(), MPI_BYTE, receive.data(), receives.data(),
        receive_offsets.data(), MPI_BYTE, mpi);
    std::vector<Bytes> incoming;
    phase(comm, "receive decoding",
        [&]
        {
            incoming.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank)
                incoming[rank] = Bytes(
                    receive.begin() + receive_offsets[rank], receive.begin() + receive_offsets[rank] + receives[rank]);
        });
    return incoming;
}

Key node_set(Key nodes)
{
    std::sort(nodes.begin(), nodes.end());
    if (nodes.size() < 3 || std::adjacent_find(nodes.begin(), nodes.end()) != nodes.end())
        throw std::invalid_argument("Invalid source face node loop.");
    return nodes;
}

Key canonical_loop(const Key& nodes)
{
    Key best = nodes;
    for (int direction : {1, -1})
        for (size_t first = 0; first < nodes.size(); ++first)
        {
            Key candidate;
            candidate.reserve(nodes.size());
            for (size_t step = 0; step < nodes.size(); ++step)
            {
                const auto local =
                    direction == 1 ? (first + step) % nodes.size() : (first + nodes.size() - step) % nodes.size();
                candidate.push_back(nodes[local]);
            }
            if (candidate < best)
                best = std::move(candidate);
        }
    return best;
}

std::array<ID, 2> incidence(const Data::Face& face)
{
    if (face.owner == Data::invalid_id || face.owner == face.neighbor)
        throw std::invalid_argument("Invalid source face incidence.");
    return {std::min(face.owner, face.neighbor), std::max(face.owner, face.neighbor)};
}
ID algebraic(const Data::Face& face)
{
    return face.algebraic_id == Data::invalid_id ? face.id : face.algebraic_id;
}
bool finite(const Data::Vec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

// Match the downstream native wire agreement without changing source bits.
bool same_coordinates(const Data::Vec3& a, const Data::Vec3& b) noexcept
{
    return std::memcmp(&a.x, &b.x, sizeof(real_t)) == 0 && std::memcmp(&a.y, &b.y, sizeof(real_t)) == 0 &&
           std::memcmp(&a.z, &b.z, sizeof(real_t)) == 0;
}

struct Topology
{
    ID face_id;
    Key loop;
    std::array<ID, 2> cells;
    ID ordinal;
    int boundary;
    bool operator==(const Topology&) const = default;
};
Topology topology(const Data::Face& face)
{
    return {face.id, canonical_loop(face.node_ids), incidence(face), algebraic(face), face.boundary_id};
}

void write_topology(Writer& writer, const Data::Face& face)
{
    const auto value = topology(face);
    writer.scalar(value.face_id);
    writer.ids(value.loop);
    writer.scalar(value.cells[0]);
    writer.scalar(value.cells[1]);
    writer.scalar(value.ordinal);
    writer.scalar<std::int32_t>(value.boundary);
}
Topology read_topology(Reader& reader)
{
    Topology value;
    value.face_id = reader.scalar<ID>();
    value.loop = reader.ids();
    value.cells = {reader.scalar<ID>(), reader.scalar<ID>()};
    value.ordinal = reader.scalar<ID>();
    value.boundary = reader.scalar<std::int32_t>();
    return value;
}

struct Authority
{
    Data::Face face;
    ID priority;
    int producer;
    Topology topology;
};
} // namespace

void synchronize_stk_face_authority(Data& data, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    if (comm.is_null())
        throw std::invalid_argument("Input face authority requires a communicator.");
    if (comm->getSize() == 1)
        return;
    const int ranks = comm->getSize();
    MPI_Comm mpi = MPI_COMM_NULL;
    std::vector<Writer> outgoing;
    std::unordered_map<ID, size_t> local_faces;
    std::map<ID, ID> required;
    const auto destination = [ranks](ID id) { return native_partition_detail::directory_rank(id, ranks); };
    phase(*comm, "source preparation",
        [&]
        {
            const auto* wrapper = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.getRawPtr());
            if (!wrapper)
                throw std::invalid_argument("Input face authority requires an MPI communicator.");
            mpi = *wrapper->getRawMpiComm();
            if (mpi == MPI_COMM_NULL)
                throw std::invalid_argument("Input face authority communicator is null.");
            const auto owned = data.num_owned_cells == Data::all_owned ? data.cells.size() : data.num_owned_cells;
            if (owned > data.cells.size())
                throw std::invalid_argument("Invalid owned input cell count.");
            std::unordered_map<ID, const Data::Node*> nodes;
            for (const auto& node : data.nodes)
                if (!nodes.emplace(node.id, &node).second)
                    throw std::invalid_argument("Repeated local input node ID.");
            for (size_t local = 0; local < data.faces.size(); ++local)
                if (!local_faces.emplace(data.faces[local].id, local).second)
                    throw std::invalid_argument("Repeated local input face ID.");
            std::set<ID> owned_ids;
            for (size_t local = 0; local < owned; ++local)
            {
                const auto& cell = data.cells[local];
                if (cell.id == Data::invalid_id || !owned_ids.insert(cell.id).second)
                    throw std::invalid_argument("Invalid or repeated owned input cell ID.");
                std::set<ID> seen;
                for (auto id : cell.face_ids)
                {
                    if (!seen.insert(id).second)
                        throw std::invalid_argument("Repeated source cell face ID.");
                    const auto& face = data.faces.at(local_faces.at(id));
                    if (face.owner != cell.id && face.neighbor != cell.id)
                        throw std::invalid_argument("Owned input cell/face incidence disagrees.");
                    const auto [where, inserted] = required.emplace(id, cell.id);
                    if (!inserted)
                        where->second = std::min(where->second, cell.id);
                }
            }
            outgoing.resize(ranks);
            for (const auto& [id, name] : data.boundary_names)
            {
                if (id < 0)
                    throw std::invalid_argument("Invalid input boundary catalog ID.");
                auto& writer = outgoing[destination(static_cast<ID>(id))];
                writer.scalar<std::uint8_t>(1);
                writer.scalar<std::int32_t>(id);
                writer.text(name);
            }
            std::set<ID> sent_nodes;
            for (const auto& [id, priority] : required)
            {
                const auto& face = data.faces.at(local_faces.at(id));
                if (id == Data::invalid_id || !finite(face.center) || !finite(face.normal_from_owner) ||
                    !std::isfinite(face.area) || face.area <= 0 || face.boundary_id < -1)
                    throw std::invalid_argument("Invalid owned-incident input face record.");
                const auto key = node_set(face.node_ids);
                incidence(face);
                if ((face.boundary_id >= 0 && !data.boundary_names.contains(face.boundary_id)) ||
                    (face.boundary_id >= 0 && face.neighbor != Data::invalid_id))
                    throw std::invalid_argument("Invalid source face boundary label.");
                auto& writer = outgoing[destination(id)];
                writer.scalar<std::uint8_t>(0);
                writer.scalar(priority);
                writer.face(face);
                std::uint64_t key_hash = 0;
                for (auto node : key)
                    key_hash = native_partition_detail::directory_hash(key_hash ^ node);
                auto& physical = outgoing[key_hash % static_cast<std::uint64_t>(ranks)];
                physical.scalar<std::uint8_t>(3);
                physical.ids(key);
                write_topology(physical, face);
                for (auto node_id : key)
                {
                    const auto& node = *nodes.at(node_id);
                    if (node_id == Data::invalid_id || !finite(node.coordinates))
                        throw std::invalid_argument("Invalid source face node coordinates.");
                    if (sent_nodes.insert(node_id).second)
                    {
                        auto& node_writer = outgoing[destination(node_id)];
                        node_writer.scalar<std::uint8_t>(2);
                        node_writer.scalar(node_id);
                        node_writer.vector(node.coordinates);
                    }
                }
            }
        });
    auto incoming = exchange(outgoing, *comm, mpi);
    std::unordered_map<ID, Authority> directory;
    phase(*comm, "source agreement",
        [&]
        {
            std::unordered_map<ID, Data::Vec3> node_coordinates;
            std::unordered_map<int, std::string> boundary_names;
            std::map<Key, Topology> physical_faces;
            for (int producer = 0; producer < ranks; ++producer)
            {
                Reader reader{incoming[producer]};
                while (!reader.done())
                {
                    const auto kind = reader.scalar<std::uint8_t>();
                    if (kind == 0)
                    {
                        const auto priority = reader.scalar<ID>();
                        auto face = reader.face();
                        const auto shape = topology(face);
                        if (priority != face.owner && priority != face.neighbor)
                            throw std::invalid_argument("Input face candidate is not incident on its producing cell.");
                        const auto [where, inserted] =
                            directory.emplace(face.id, Authority{face, priority, producer, shape});
                        if (!inserted)
                        {
                            if (where->second.topology != shape)
                                throw std::invalid_argument(
                                    "Shared input face topology, ordinal or boundary label disagrees.");
                            if (std::pair{priority, producer} <
                                std::pair{where->second.priority, where->second.producer})
                                where->second = {std::move(face), priority, producer, shape};
                        }
                    }
                    else if (kind == 1)
                    {
                        const int id = reader.scalar<std::int32_t>();
                        auto name = reader.text();
                        const auto [where, inserted] = boundary_names.emplace(id, name);
                        if (!inserted && where->second != name)
                            throw std::invalid_argument("Input boundary catalogs disagree.");
                    }
                    else if (kind == 2)
                    {
                        const auto id = reader.scalar<ID>();
                        const auto coordinates = reader.vector();
                        const auto [where, inserted] = node_coordinates.emplace(id, coordinates);
                        if (!inserted && !same_coordinates(where->second, coordinates))
                            throw std::invalid_argument(
                                "Shared input source node coordinate representations disagree.");
                    }
                    else if (kind == 3)
                    {
                        auto key = reader.ids();
                        const auto shape = read_topology(reader);
                        const auto [where, inserted] = physical_faces.emplace(std::move(key), shape);
                        if (!inserted && where->second != shape)
                            throw std::invalid_argument(
                                "Physical input face identity or cyclic connectivity disagrees.");
                    }
                    else
                        throw std::invalid_argument("Unknown input face authority record.");
                }
            }
            outgoing.assign(ranks, {});
            for (const auto& [id, priority] : required)
                outgoing[destination(id)].scalar(id);
        });
    incoming = exchange(outgoing, *comm, mpi);
    phase(*comm, "authority replies",
        [&]
        {
            outgoing.assign(ranks, {});
            for (int requester = 0; requester < ranks; ++requester)
            {
                Reader reader{incoming[requester]};
                while (!reader.done())
                {
                    const auto id = reader.scalar<ID>();
                    const auto found = directory.find(id);
                    if (found == directory.end())
                        throw std::invalid_argument("Required input face authority is absent.");
                    outgoing[requester].face(found->second.face);
                }
            }
        });
    incoming = exchange(outgoing, *comm, mpi);
    std::vector<std::pair<size_t, Data::Face>> replacements;
    phase(*comm, "publication preparation",
        [&]
        {
            std::set<ID> resolved;
            replacements.reserve(required.size());
            for (const auto& bytes : incoming)
            {
                Reader reader{bytes};
                while (!reader.done())
                {
                    auto face = reader.face();
                    if (!required.contains(face.id) || !resolved.insert(face.id).second)
                        throw std::invalid_argument("Unexpected or duplicate input face authority reply.");
                    replacements.emplace_back(local_faces.at(face.id), std::move(face));
                }
            }
            if (resolved.size() != required.size())
                throw std::invalid_argument("Incomplete input face authority replies.");
            decltype(directory){}.swap(directory);
            incoming.clear();
            outgoing.clear();
        });
    static_assert(std::is_nothrow_move_assignable_v<Data::Face>);
    for (auto& [local, face] : replacements)
        data.faces[local] = std::move(face);
}
} // namespace SimpleFluid::detail
