/** @file GeneralVoFPhiClosure.cc
 * @brief Sparse persistent-identity forest closure with exact arithmetic gates.
 */
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using GO = DefaultTpetraTypes::global_ordinal_type;
using M = Meshes::ExactGeometryMeasure;
using Bytes = std::vector<char>;
class PhiProposalResource : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
struct Codec
{
    Bytes bytes;
    std::size_t cursor = 0, limit = 0, decode_limit = 0, decode_used = 0;
    explicit Codec(std::size_t budget) : limit(budget) {}
    Codec(const Bytes& input, std::size_t budget) : limit(input.size()), decode_limit(budget)
    {
        if (input.size() > decode_limit)
            throw PhiProposalResource("Phi decode copy exceeds its live packet arena.");
        bytes = input;
        if (bytes.capacity() > decode_limit)
            throw PhiProposalResource("Phi decode capacity exceeds its live packet arena.");
        decode_used = bytes.capacity();
    }
    void put(std::uint64_t v)
    {
        if (bytes.size() > limit || limit - bytes.size() < 8)
            throw PhiProposalResource("Phi closure framing exceeds its budget.");
        for (unsigned i = 0; i < 8; ++i)
            bytes.push_back(char(v >> (8 * i)));
    }
    std::uint64_t get()
    {
        if (cursor > bytes.size() || bytes.size() - cursor < 8)
            throw std::runtime_error("Truncated Phi closure packet.");
        std::uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i)
            v |= std::uint64_t(static_cast<unsigned char>(bytes[cursor++])) << (8 * i);
        return v;
    }
    void put(const Bytes& v)
    {
        if (bytes.size() > limit || limit - bytes.size() < 8 || v.size() > limit - bytes.size() - 8)
            throw PhiProposalResource("Phi closure payload exceeds its budget.");
        put(v.size());
        bytes.insert(bytes.end(), v.begin(), v.end());
    }
    Bytes blob()
    {
        const auto n = get();
        if (n > bytes.size() - cursor)
            throw std::runtime_error("Invalid Phi closure framing.");
        if (n > decode_limit - decode_used)
            throw PhiProposalResource("Phi decoded blobs exceed their live packet arena.");
        Bytes v(bytes.begin() + cursor, bytes.begin() + cursor + n);
        if (v.capacity() > decode_limit - decode_used)
            throw PhiProposalResource("Phi decoded blob capacity exceeds its live packet arena.");
        // Conservatively retain every blob's capacity until this Codec dies,
        // even when a scalar decoder has already released its temporary blob.
        decode_used += v.capacity();
        cursor += n;
        return v;
    }
    void put(const M& v, std::size_t b) { put(v.encode(b)); }
    M measure(std::size_t b) { return M::decode(blob(), b); }
    void done()
    {
        if (cursor != bytes.size())
            throw std::runtime_error("Trailing Phi closure bytes.");
    }
};
template<class Work> void collective(const Teuchos::Comm<int>& comm, Work work)
{
    std::exception_ptr error;
    int local = 0, global = 0;
    try
    {
        work();
    }
    catch (const GeneralVoFPhiInfeasible&)
    {
        error = std::current_exception();
        local = 1;
    }
    catch (const PhiProposalResource&)
    {
        error = std::current_exception();
        local = 2;
    }
    catch (...)
    {
        error = std::current_exception();
        local = 3;
    }
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &global);
    if (global == 1)
    {
        if (error)
            std::rethrow_exception(error);
        throw GeneralVoFPhiInfeasible("Bounded arithmetic closure infeasible on another rank.");
    }
    if (global == 2)
    {
        if (local == 2)
            std::rethrow_exception(error);
        throw PhiProposalResource("Bounded arithmetic closure resources exhausted on another rank.");
    }
    if (global == 3)
    {
        if (local == 3)
            std::rethrow_exception(error);
        throw std::runtime_error("General VoF Phi arithmetic closure invalid on another rank.");
    }
}
struct Edge
{
    GO face, other;
    int peer, authority, sign;
    Bytes identity;
    M volume, radius;
};
struct Node
{
    GO gid;
    Bytes identity, label;
    std::uint64_t depth = 0;
    std::vector<Edge> edges;
    GO parent_face = -1, parent = -1;
    M defect, radius, used;
};
std::size_t packet_capacity(const std::vector<OwnerPacket>& packets, std::size_t budget)
{
    if (packets.capacity() > budget / sizeof(OwnerPacket))
        throw PhiProposalResource("Phi packet metadata capacity exceeds its arena.");
    std::size_t used = packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
    {
        if (packet.bytes.capacity() > budget - used)
            throw PhiProposalResource("Phi packet payload capacities exceed their arena.");
        used += packet.bytes.capacity();
    }
    return used;
}
std::size_t phi_decode_budget(const std::vector<OwnerPacket>& inputs, const std::vector<OwnerPacket>& outputs,
    const Teuchos::Comm<int>& comm, std::size_t arena)
{
    std::size_t remaining = 0;
    collective(comm,
        [&]
        {
            const auto input = packet_capacity(inputs, arena);
            const auto output = packet_capacity(outputs, arena - input);
            remaining = arena - input - output;
        });
    return remaining;
}
OwnerExchange exchange_phi_packets(const std::vector<OwnerPacket>& packets,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm, std::size_t input_cap, std::size_t exchange_cap,
    std::size_t arena_cap, std::size_t* resident_input = nullptr)
{
    std::size_t resident = 0, payload = 0, wire = 0;
    collective(*comm,
        [&]
        {
            if (exchange_cap > arena_cap || packets.capacity() > (arena_cap - exchange_cap) / sizeof(OwnerPacket))
                throw PhiProposalResource("Phi caller packet metadata exceeds its arena share.");
            resident = packets.capacity() * sizeof(OwnerPacket);
            for (const auto& packet : packets)
            {
                if (packet.bytes.capacity() > arena_cap - exchange_cap - resident)
                    throw PhiProposalResource("Phi caller packet capacities exceed their arena share.");
                resident += packet.bytes.capacity();
                if (packet.bytes.size() > input_cap - payload)
                    throw PhiProposalResource("Phi outgoing payload exceeds its unchanged input cap.");
                payload += packet.bytes.size();
                if (packet.bytes.size() > std::numeric_limits<std::size_t>::max() - sizeof(std::uint64_t) ||
                    wire > std::numeric_limits<std::size_t>::max() - sizeof(std::uint64_t) - packet.bytes.size())
                    throw PhiProposalResource("Phi packet wire count overflow.");
                wire += sizeof(std::uint64_t) + packet.bytes.size();
            }
        });
    if (resident_input)
        *resident_input = resident;
    const auto context = [&](const std::exception& error)
    {
        return std::string(error.what()) + " Phi packet arena: payload=" + std::to_string(payload) +
               ", sent_wire=" + std::to_string(wire) +
               ", serial_received_wire=" + (comm->getSize() == 1 ? std::to_string(wire) : std::string("not-local")) +
               ", resident_input=" + std::to_string(resident) + ", input_cap=" + std::to_string(input_cap) +
               ", exchange_cap=" + std::to_string(exchange_cap) + ", arena_cap=" + std::to_string(arena_cap);
    };
    try
    {
        // The exchange cap includes its packed send, two receive copies and
        // explicit route/output metadata. Caller inputs are additional; their
        // audited capacities occupy the disjoint remainder of the same arena.
        return exchange_owner_packets(packets, comm, exchange_cap);
    }
    catch (const std::length_error& error)
    {
        throw PhiProposalResource(context(error));
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(context(error));
    }
}
M absolute(const M& v, std::size_t budget)
{
    return v.sign() < 0 ? vof_exact::leased_scale(v, -1, budget) : v;
}
} // namespace
static GeneralVoFPhiClosure close_general_vof_phi_impl(const SP<const MeshHandle<>>& mesh,
    const std::vector<GeneralVoFPhiCell>& cells, const std::vector<GeneralVoFPhiFace>& faces, std::size_t budget,
    bool bounded_rates)
{
    namespace E = vof_exact;
    const auto map = mesh->owned_cell_map();
    const auto comm = map->getComm();
    const int rank = comm->getRank();
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(budget);
    std::size_t edge_bound = 0, max_cell_key = 0, max_face_key = 0, scalar_budget = 0, packet_budget = 0;
    std::size_t packet_exchange_budget = 0, packet_arena_budget = 0;
    collective(*comm,
        [&]
        {
            storage.capacity(Domain::Input, cells.capacity(), sizeof(GeneralVoFPhiCell));
            storage.capacity(Domain::Input, faces.capacity(), sizeof(GeneralVoFPhiFace));
            for (const auto& c : cells)
            {
                storage.capacity(Domain::Input, c.identity.capacity(), sizeof(char));
                storage.measure(Domain::Input, c.arithmetic_radius);
                max_cell_key = std::max(max_cell_key, c.identity.size());
            }
            for (const auto& f : faces)
            {
                storage.capacity(Domain::Input, f.identity.capacity(), sizeof(char));
                storage.measure(Domain::Input, f.volume);
                storage.measure(Domain::Input, f.arithmetic_radius);
                max_face_key = std::max(max_face_key, f.identity.size());
            }
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            {
                const auto degree = mesh->faces(static_cast<DefaultTpetraTypes::local_ordinal_type>(c)).size();
                if (degree > std::numeric_limits<std::size_t>::max() - edge_bound)
                    throw std::runtime_error("Phi closure incidence overflow.");
                edge_bound += degree;
            }
        });
    std::uint64_t key_local[2] = {max_cell_key, max_face_key}, key_global[2]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 2, key_local, key_global);
    collective(*comm,
        [&]
        {
            const auto n = cells.size(), f = faces.size(), e = edge_bound;
            storage.capacity(Domain::Scratch, n, sizeof(Node) + 4 * sizeof(void*));
            storage.capacity(Domain::Scratch, n, 2 * std::size_t(key_global[0]));
            storage.capacity(Domain::Scratch, e, 2 * sizeof(Edge) + 2 * std::size_t(key_global[1]));
            storage.capacity(Domain::Scratch, 2 * f, 3 * sizeof(GO) + 2 * sizeof(int) + 12 * sizeof(void*));
            storage.capacity(Domain::Scratch, e, 3 * sizeof(GO) + 2 * sizeof(int) + 8 * sizeof(void*));
            storage.capacity(Domain::Scratch, n, 3 * (sizeof(std::pair<const GO, M>) + 4 * sizeof(void*)));
            storage.capacity(Domain::Result, 2 * f, sizeof(M));
            const auto packet_pool = storage.remaining() / 2;
            storage.capacity(Domain::Scratch, packet_pool, 1);
            packet_budget = packet_pool / 8;
            packet_exchange_budget = packet_pool / 2;
            packet_arena_budget = packet_pool;
            if (n > (std::numeric_limits<std::size_t>::max() - 48 - 2 * f - 2 * e) / 5)
                throw std::runtime_error("Phi closure scalar count overflow.");
            const auto result_slots = 2 * f, scratch_slots = 5 * n + 2 * e + 48;
            scalar_budget = storage.remaining() / (scratch_slots + result_slots);
            if (scalar_budget < 1024 || packet_budget < 1024)
                throw std::runtime_error("Phi closure cumulative resource reserve exhausted.");
            storage.capacity(Domain::Scratch, scratch_slots, scalar_budget);
            storage.capacity(Domain::Result, result_slots, scalar_budget);
        });
    std::map<GO, Node> nodes;
    std::map<GO, std::size_t> face_index;
    std::set<GO> endpoint_set;
    GeneralVoFPhiClosure result;
    collective(*comm,
        [&]
        {
            result.volume.reserve(faces.size());
            result.correction.reserve(faces.size());
            if (cells.size() != mesh->num_owned_cells())
                throw std::runtime_error("Incomplete closure cells.");
            for (const auto& c : cells)
            {
                if (map->getLocalElement(c.cell) < 0 || c.identity.empty() || c.arithmetic_radius.sign() < 0 ||
                    nodes.contains(c.cell))
                    throw std::runtime_error("Foreign closure cell.");
                Node n;
                n.gid = c.cell;
                n.identity = n.label = c.identity;
                n.radius = c.arithmetic_radius;
                nodes.emplace(c.cell, std::move(n));
            }
            for (std::size_t i = 0; i < faces.size(); ++i)
            {
                const auto& f = faces[i];
                const auto lid = mesh->owned_face_map()->getLocalElement(f.face);
                if (lid < 0 || f.first == f.second || f.identity.empty() || f.arithmetic_radius.sign() < 0 ||
                    !face_index.emplace(f.face, i).second)
                    throw std::runtime_error("Invalid closure face.");
                const auto lease = mesh->acquire_execution_view();
                if (mesh->is_geometry_exterior_face(lid) || mesh->cell_global_id(mesh->owner_cell(lid)) != f.first ||
                    mesh->cell_global_id(mesh->neighbor_cell(lid)) != f.second)
                    throw std::runtime_error("Wrong closure endpoint.");
                endpoint_set.insert(f.first);
                endpoint_set.insert(f.second);
                constexpr unsigned precision = 106;
                constexpr std::size_t precision_bytes = (precision + 7) / 8;
                const auto source_bytes = f.volume.storage_bytes();
                // Precision proposes a bounded representation; neither this
                // choice nor a failed proposal changes any acceptance radius.
                const bool can_propose = bounded_rates && f.arithmetic_radius.sign() &&
                                         source_bytes <= scalar_budget &&
                                         precision_bytes <= scalar_budget - source_bytes;
                auto proposed = f.volume;
                if (can_propose)
                {
                    try
                    {
                        proposed = f.volume.rounded_dyadic(precision, 16 * scalar_budget);
                    }
                    catch (const std::length_error& error)
                    {
                        throw PhiProposalResource(error.what());
                    }
                }
                auto rounding = bounded_rates && proposed.storage_identity() != f.volume.storage_identity()
                                    ? E::leased_add(proposed, E::leased_scale(f.volume, -1, scalar_budget), scalar_budget)
                                    : M{};
                if (absolute(rounding, scalar_budget).compare(f.arithmetic_radius) > 0)
                {
                    proposed = f.volume;
                    rounding = {};
                }
                result.volume.push_back(std::move(proposed));
                result.correction.push_back(std::move(rounding));
            }
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            {
                const auto lid = static_cast<DefaultTpetraTypes::local_ordinal_type>(f);
                if (!mesh->is_geometry_exterior_face(lid) && !face_index.contains(mesh->face_global_id(lid)))
                    throw std::runtime_error("Incomplete canonical closure faces.");
            }
        });
    Teuchos::Array<GO> endpoints(endpoint_set.begin(), endpoint_set.end());
    Teuchos::Array<int> owners(endpoints.size(), -1);
    (void) map->getRemoteIndexList(endpoints(), owners());
    std::map<GO, int> owner;
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(endpoints.size()); ++i)
            {
                if (owners[i] < 0)
                    throw std::runtime_error("Absent closure endpoint.");
                owner[endpoints[i]] = owners[i];
            }
        });
    std::vector<OwnerPacket> packets;
    std::size_t packet_bytes = 0;
    const auto enqueue = [&](int peer, Bytes bytes)
    {
        if (bytes.size() > packet_budget || packet_bytes > packet_budget - bytes.size() ||
            sizeof(OwnerPacket) > packet_budget - bytes.size() - packet_bytes)
            throw PhiProposalResource("Phi closure outgoing packets exceed their budget.");
        packet_bytes += bytes.size() + sizeof(OwnerPacket);
        packets.push_back({peer, std::move(bytes)});
    };
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < faces.size(); ++i)
            {
                const auto& f = faces[i];
                for (int sign : {1, -1})
                {
                    Codec c(packet_budget);
                    c.put(std::uint64_t(sign == 1 ? f.first : f.second));
                    c.put(std::uint64_t(sign == 1 ? f.second : f.first));
                    c.put(std::uint64_t(f.face));
                    c.put(std::uint64_t(owner.at(sign == 1 ? f.second : f.first)));
                    c.put(sign == 1 ? 1 : 0);
                    c.put(f.identity);
                    c.put(f.volume, scalar_budget);
                    c.put(f.arithmetic_radius, scalar_budget);
                    if (bounded_rates)
                    {
                        c.put(result.correction[i].sign() != 0);
                        if (result.correction[i].sign())
                            c.put(result.volume[i], scalar_budget);
                    }
                    enqueue(owner.at(sign == 1 ? f.first : f.second), std::move(c.bytes));
                }
            }
        });
    auto received = exchange_phi_packets(packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget);
    const auto receive_decode_budget = phi_decode_budget(packets, received.packets, *comm, packet_arena_budget);
    std::vector<GO> auth_faces;
    std::vector<int> auth_senders;
    collective(*comm,
        [&]
        {
            for (const auto& p : received.packets)
            {
                Codec c{p.bytes, receive_decode_budget};
                const auto gid = GO(c.get()), other = GO(c.get()), face = GO(c.get());
                const int peer = int(c.get()), sign = c.get() ? 1 : -1;
                const auto identity = c.blob();
                auto volume = c.measure(scalar_budget), radius = c.measure(scalar_budget);
                M rounding;
                if (bounded_rates)
                {
                    const auto changed = c.get();
                    if (changed > 1)
                        throw std::runtime_error("Invalid bounded rate proposal flag.");
                    if (changed)
                    {
                        const auto proposed = c.measure(scalar_budget);
                        rounding = absolute(E::leased_add(
                                                proposed, E::leased_scale(volume, -1, scalar_budget), scalar_budget),
                            scalar_budget);
                        // Reserve the exact discarded difference first. The
                        // forest may spend only the remaining ORIGINAL face
                        // capacity, while n.used below charges the discarded
                        // difference to the ORIGINAL incident-cell capacity.
                        radius = E::leased_add(radius, E::leased_scale(rounding, -1, scalar_budget), scalar_budget);
                        if (radius.sign() < 0)
                            throw std::runtime_error("Bounded rate exceeds its original face certificate.");
                        volume = proposed;
                    }
                }
                c.done();
                if (!nodes.contains(gid) || peer < 0 || peer >= comm->getSize())
                    throw std::runtime_error("Invalid incident closure edge.");
                auto& n = nodes.at(gid);
                if (std::any_of(n.edges.begin(), n.edges.end(), [&](const auto& e) { return e.face == face; }))
                    throw std::runtime_error("Duplicate incident closure edge.");
                n.edges.push_back({face, other, peer, p.peer, sign, identity, volume, radius});
                n.defect = E::leased_add(n.defect, E::leased_scale(volume, sign, scalar_budget), scalar_budget);
                if (bounded_rates)
                    n.used = E::leased_add(n.used, rounding, scalar_budget);
                auth_faces.push_back(face);
                auth_senders.push_back(p.peer);
            }
        });
    const std::set<GO> unique_auth_set(auth_faces.begin(), auth_faces.end());
    Teuchos::Array<GO> unique_auth(unique_auth_set.begin(), unique_auth_set.end());
    Teuchos::Array<int> authorities(unique_auth.size(), -1);
    (void) mesh->owned_face_map()->getRemoteIndexList(unique_auth(), authorities());
    std::map<GO, int> authority_by_face;
    for (std::size_t i = 0; i < std::size_t(unique_auth.size()); ++i)
        authority_by_face.emplace(unique_auth[i], authorities[i]);
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < auth_faces.size(); ++i)
                if (authority_by_face.at(auth_faces[i]) != auth_senders[i])
                    throw std::runtime_error("Unauthenticated closure face.");
        });
    received = {}; // Decoded/authenticated records now live in budgeted nodes.
    std::uint64_t local_count = nodes.size(), count = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_count, &count);
    for (std::uint64_t round = 0;; ++round)
    {
        packets.clear();
        packet_bytes = 0;
        collective(*comm,
            [&]
            {
                for (const auto& [gid, n] : nodes)
                    for (const auto& e : n.edges)
                    {
                        // Exactly immutable faces remain in divergence and final
                        // incidence, but cannot route an arithmetic correction.
                        if (e.radius.sign() == 0)
                            continue;
                        Codec c(packet_budget);
                        c.put(std::uint64_t(e.other));
                        c.put(std::uint64_t(gid));
                        c.put(std::uint64_t(e.face));
                        c.put(n.label);
                        c.put(n.depth);
                        enqueue(e.peer, std::move(c.bytes));
                    }
            });
        const auto updates =
            exchange_phi_packets(packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget);
        const auto update_decode_budget = phi_decode_budget(packets, updates.packets, *comm, packet_arena_budget);
        int changed = 0, global_changed = 0;
        collective(*comm,
            [&]
            {
                for (const auto& p : updates.packets)
                {
                    Codec c{p.bytes, update_decode_budget};
                    const GO gid = c.get(), other = c.get(), face = c.get();
                    const auto label = c.blob();
                    const auto depth = c.get();
                    c.done();
                    auto& n = nodes.at(gid);
                    const auto found = std::find_if(n.edges.begin(), n.edges.end(),
                        [&](const auto& e) { return e.face == face && e.other == other && e.peer == p.peer; });
                    if (found == n.edges.end() || found->radius.sign() <= 0 || label.empty() || depth >= count)
                        throw std::runtime_error("Unauthenticated forest label.");
                    bool better = label < n.label || (label == n.label && depth + 1 < n.depth);
                    if (label == n.label && depth + 1 == n.depth && n.parent_face >= 0)
                    {
                        const auto parent = std::find_if(
                            n.edges.begin(), n.edges.end(), [&](const auto& e) { return e.face == n.parent_face; });
                        better = found->identity < parent->identity;
                    }
                    if (better)
                    {
                        n.label = label;
                        n.depth = depth + 1;
                        n.parent_face = face;
                        n.parent = other;
                        changed = 1;
                    }
                }
            });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &changed, &global_changed);
        result.forest_rounds = std::size_t(round + 1);
        if (!global_changed)
            break;
        collective(*comm,
            [&]
            {
                if (round >= count)
                    throw std::runtime_error("Closure forest did not converge.");
            });
    }
    std::uint64_t local_depth = 0, depth = 0;
    for (const auto& [gid, n] : nodes)
        local_depth = std::max(local_depth, n.depth);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_depth, &depth);
    std::map<GO, M> local_correction;
    for (std::uint64_t level = depth; level > 0; --level)
    {
        packets.clear();
        packet_bytes = 0;
        collective(*comm,
            [&]
            {
                for (auto& [gid, n] : nodes)
                    if (n.depth == level)
                    {
                        const auto e = std::find_if(
                            n.edges.begin(), n.edges.end(), [&](const auto& e) { return e.face == n.parent_face; });
                        if (e == n.edges.end())
                            throw std::runtime_error("Missing forest parent.");
                        const auto delta = E::leased_scale(n.defect, -e->sign, scalar_budget);
                        if (absolute(delta, scalar_budget).compare(e->radius) > 0)
                            throw GeneralVoFPhiInfeasible("Face closure exceeds arithmetic radius.");
                        local_correction[e->face] = delta;
                        n.used = E::leased_add(n.used, absolute(delta, scalar_budget), scalar_budget);
                        Codec c(packet_budget);
                        c.put(std::uint64_t(e->other));
                        c.put(std::uint64_t(gid));
                        c.put(std::uint64_t(e->face));
                        c.put(n.defect, scalar_budget);
                        c.put(delta, scalar_budget);
                        enqueue(e->peer, std::move(c.bytes));
                    }
            });
        const auto sums =
            exchange_phi_packets(packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget);
        const auto sum_decode_budget = phi_decode_budget(packets, sums.packets, *comm, packet_arena_budget);
        collective(*comm,
            [&]
            {
                for (const auto& p : sums.packets)
                {
                    Codec c{p.bytes, sum_decode_budget};
                    const GO gid = c.get(), child = c.get(), face = c.get();
                    const auto sum = c.measure(scalar_budget), delta = c.measure(scalar_budget);
                    c.done();
                    auto& n = nodes.at(gid);
                    const auto e = std::find_if(n.edges.begin(), n.edges.end(),
                        [&](const auto& e) { return e.face == face && e.other == child && e.peer == p.peer; });
                    if (e == n.edges.end())
                        throw std::runtime_error("Wrong forest sum route.");
                    n.defect = E::leased_add(n.defect, sum, scalar_budget);
                    n.used = E::leased_add(n.used, absolute(delta, scalar_budget), scalar_budget);
                }
            });
    }
    collective(*comm,
        [&]
        {
            for (const auto& [gid, n] : nodes)
            {
                if ((!n.depth && n.defect.sign()) || n.used.compare(n.radius) > 0)
                    throw GeneralVoFPhiInfeasible("Incompatible closed component or cell arithmetic radius.");
            }
        });
    packets.clear();
    packet_bytes = 0;
    collective(*comm,
        [&]
        {
            for (const auto& [gid, n] : nodes)
                for (const auto& e : n.edges)
                {
                    const auto found = local_correction.find(e.face);
                    if (found != local_correction.end() && e.face == n.parent_face)
                    {
                        Codec c(packet_budget);
                        c.put(std::uint64_t(e.face));
                        c.put(std::uint64_t(gid));
                        c.put(found->second, scalar_budget);
                        enqueue(e.authority, std::move(c.bytes));
                    }
                }
        });
    auto corrections = exchange_phi_packets(packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget);
    const auto correction_decode_budget = phi_decode_budget(packets, corrections.packets, *comm, packet_arena_budget);
    collective(*comm,
        [&]
        {
            std::set<GO> seen;
            for (const auto& p : corrections.packets)
            {
                Codec c{p.bytes, correction_decode_budget};
                const GO face = c.get(), child = c.get();
                const auto delta = c.measure(scalar_budget);
                c.done();
                const auto i = face_index.at(face);
                const auto& f = faces[i];
                if (!seen.insert(face).second || (child != f.first && child != f.second) || owner.at(child) != p.peer ||
                    absolute(delta, scalar_budget).compare(f.arithmetic_radius) > 0)
                    throw std::runtime_error("Invalid canonical face correction.");
                result.correction[i] = E::leased_add(result.correction[i], delta, scalar_budget);
                result.volume[i] = E::leased_add(result.volume[i], delta, scalar_budget);
                if (absolute(result.correction[i], scalar_budget).compare(f.arithmetic_radius) > 0)
                    throw std::runtime_error("Canonical rate change exceeds its original face certificate.");
            }
        });
    corrections = {}; // Do not retain old receive payload across final exchange.
    // Reassemble canonical corrected controls independently of the forest
    // accumulation. Exact zero divergence is the acceptance predicate.
    packets.clear();
    packet_bytes = 0;
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < faces.size(); ++i)
                for (int sign : {1, -1})
                {
                    const auto endpoint = sign == 1 ? faces[i].first : faces[i].second;
                    Codec c(packet_budget);
                    c.put(std::uint64_t(endpoint));
                    c.put(std::uint64_t(faces[i].face));
                    c.put(E::leased_scale(result.volume[i], sign, scalar_budget), scalar_budget);
                    enqueue(owner.at(endpoint), std::move(c.bytes));
                }
        });
    const auto final_controls =
        exchange_phi_packets(packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget);
    const auto final_decode_budget = phi_decode_budget(packets, final_controls.packets, *comm, packet_arena_budget);
    collective(*comm,
        [&]
        {
            std::map<GO, M> divergence;
            std::map<GO, std::set<GO>> seen;
            for (const auto& p : final_controls.packets)
            {
                Codec c{p.bytes, final_decode_budget};
                const GO endpoint = c.get(), face = c.get();
                const auto value = c.measure(scalar_budget);
                c.done();
                const auto& n = nodes.at(endpoint);
                const auto e = std::find_if(n.edges.begin(), n.edges.end(),
                    [&](const auto& e) { return e.face == face && e.authority == p.peer; });
                if (e == n.edges.end() || !seen[endpoint].insert(face).second)
                    throw std::runtime_error("Invalid corrected Phi authority.");
                divergence[endpoint] = E::leased_add(divergence[endpoint], value, scalar_budget);
            }
            for (const auto& [gid, n] : nodes)
                if (seen[gid].size() != n.edges.size() || divergence[gid].sign())
                    throw std::runtime_error("Corrected Phi is not exactly source-free.");
        });
    return result;
}
GeneralVoFPhiClosure close_general_vof_phi(const SP<const MeshHandle<>>& mesh,
    const std::vector<GeneralVoFPhiCell>& cells, const std::vector<GeneralVoFPhiFace>& faces, std::size_t budget)
{
    return close_general_vof_phi_impl(mesh, cells, faces, budget, false);
}
GeneralVoFPhiClosure close_general_vof_bounded_rate(const SP<const MeshHandle<>>& mesh,
    const std::vector<GeneralVoFPhiCell>& cells, const std::vector<GeneralVoFPhiFace>& faces, std::size_t budget)
{
    try
    {
        return close_general_vof_phi_impl(mesh, cells, faces, budget, true);
    }
    catch (const GeneralVoFPhiInfeasible&)
    {
        return close_general_vof_phi_impl(mesh, cells, faces, budget, false);
    }
    catch (const PhiProposalResource&)
    {
        auto result = close_general_vof_phi_impl(mesh, cells, faces, budget, false);
        result.bounded_resource_fallback = true;
        return result;
    }
}
std::optional<GeneralVoFPhiClosure> try_close_general_vof_phi(const SP<const MeshHandle<>>& mesh,
    const std::vector<GeneralVoFPhiCell>& cells, const std::vector<GeneralVoFPhiFace>& faces, std::size_t budget)
{
    try
    {
        return close_general_vof_phi(mesh, cells, faces, budget);
    }
    catch (const GeneralVoFPhiInfeasible&)
    {
        return std::nullopt;
    }
}
GeneralVoFProjectionRateCertificate certify_general_vof_projection_rate(const SP<const MeshHandle<>>& mesh,
    const std::vector<GeneralVoFRoundedValue>& values, std::size_t budget, const std::vector<M>* primary_rate,
    const std::vector<M>* primary_radius)
{
    const auto map = mesh->owned_cell_map();
    const auto comm = map->getComm();
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(budget);
    std::size_t scalar_budget = 0, packet_budget = 0, packet_exchange_budget = 0, packet_arena_budget = 0;
    GeneralVoFProjectionRateCertificate result;
    collective(*comm,
        [&]
        {
            storage.capacity(Domain::Input, values.capacity(), sizeof(GeneralVoFRoundedValue));
            for (const auto& value : values)
            {
                storage.measure(Domain::Input, value.lower);
                storage.measure(Domain::Input, value.upper);
            }
            if (bool(primary_rate) != bool(primary_radius) ||
                (primary_rate && (primary_rate->size() != values.size() || primary_radius->size() != values.size())))
                throw std::runtime_error("Exact primary rate shape mismatch.");
            if (primary_rate)
                for (const auto* input : {primary_rate, primary_radius})
                {
                    storage.capacity(Domain::Input, input->capacity(), sizeof(M));
                    for (const auto& value : *input)
                        storage.measure(Domain::Input, value);
                }
            const auto n = mesh->num_owned_cells(), f = values.size();
            storage.capacity(Domain::Scratch, 2 * f, 3 * sizeof(GO) + 2 * sizeof(int) + 12 * sizeof(void*));
            storage.capacity(Domain::Scratch, n, 2 * (sizeof(std::pair<const GO, M>) + 4 * sizeof(void*)));
            const auto pool = storage.remaining() / 2;
            storage.capacity(Domain::Scratch, pool, 1);
            packet_budget = pool / 8;
            packet_exchange_budget = pool / 2;
            packet_arena_budget = pool;
            result.packet_input_cap = packet_budget;
            result.packet_exchange_cap = packet_exchange_budget;
            result.packet_arena_cap = packet_arena_budget;
            if (n > (std::numeric_limits<std::size_t>::max() - 32) / 2)
                throw std::runtime_error("Projection admission scalar overflow.");
            const auto slots = 4 * n + 2 * f + 48;
            storage.capacity(Domain::Result, 2 * n + 2 * f, sizeof(M));
            scalar_budget = storage.remaining() / slots;
            if (scalar_budget < 1024 || packet_budget < 1024)
                throw std::runtime_error("Projection admission cumulative reserve exhausted.");
            storage.capacity(Domain::Scratch, slots, scalar_budget);
            result.owned_rate.resize(f);
            result.owned_face_radius.resize(f);
            result.owned_cell_radius.resize(n);
            result.owned_net_rate.resize(n);
        });
    GeneralVoFProjectionArithmetic arithmetic(scalar_budget);
    std::set<GO> endpoint_set;
    collective(*comm,
        [&]
        {
            if (values.size() != mesh->num_owned_faces())
                throw std::runtime_error("Projection provenance shape.");
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t f = 0; f < values.size(); ++f)
            {
                const auto face = static_cast<DefaultTpetraTypes::local_ordinal_type>(f);
                if (mesh->is_geometry_exterior_face(face))
                {
                    if (values[f].value != 0 || arithmetic.radius(values[f]).sign() ||
                        (primary_rate && ((*primary_rate)[f].sign() || (*primary_radius)[f].sign())))
                        throw std::runtime_error("Nonzero closed wall provenance.");
                    continue;
                }
                endpoint_set.insert(mesh->cell_global_id(mesh->owner_cell(face)));
                endpoint_set.insert(mesh->cell_global_id(mesh->neighbor_cell(face)));
            }
        });
    Teuchos::Array<GO> ids(endpoint_set.begin(), endpoint_set.end());
    Teuchos::Array<int> ranks(ids.size(), -1);
    (void) map->getRemoteIndexList(ids(), ranks());
    std::map<GO, int> owners;
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(ids.size()); ++i)
            {
                if (ranks[i] < 0)
                    throw std::runtime_error("Absent projection endpoint.");
                owners.emplace(ids[i], ranks[i]);
            }
        });
    std::vector<OwnerPacket> packets;
    std::size_t used = 0;
    collective(*comm,
        [&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t f = 0; f < values.size(); ++f)
            {
                const auto face = static_cast<DefaultTpetraTypes::local_ordinal_type>(f);
                if (mesh->is_geometry_exterior_face(face))
                    continue;
                const auto flux =
                    primary_rate ? (*primary_rate)[f] : M::from_value(values[f].value).scaled(mesh->face_area(face));
                const auto radius = primary_radius ? (*primary_radius)[f]
                                                   : vof_exact::leased_scale(arithmetic.radius(values[f]),
                                                         mesh->face_area(face), scalar_budget);
                if (radius.sign() < 0)
                    throw std::runtime_error("Negative exact primary rate radius.");
                if (primary_rate)
                {
                    const auto view = flux.divided_by(M::from_value(mesh->face_area(face)), scalar_budget).enclosure();
                    if (!std::isfinite(static_cast<double>(view.value)) ||
                        static_cast<double>(view.value) != values[f].value)
                        throw std::runtime_error("Exact primary rate differs from its diagnostic view.");
                }
                result.owned_rate[f] = flux;
                result.owned_face_radius[f] = radius;
                for (int sign : {1, -1})
                {
                    const auto endpoint =
                        mesh->cell_global_id(sign == 1 ? mesh->owner_cell(face) : mesh->neighbor_cell(face));
                    Codec c(packet_budget);
                    c.put(std::uint64_t(endpoint));
                    c.put(std::uint64_t(mesh->face_global_id(face)));
                    c.put(vof_exact::leased_scale(flux, sign, scalar_budget), scalar_budget);
                    c.put(radius, scalar_budget);
                    if (c.bytes.size() > packet_budget || used > packet_budget - c.bytes.size() ||
                        sizeof(OwnerPacket) > packet_budget - used - c.bytes.size())
                        throw std::runtime_error("Projection admission packets exceed budget.");
                    used += sizeof(OwnerPacket) + c.bytes.size();
                    packets.push_back({owners.at(endpoint), std::move(c.bytes)});
                }
            }
        });
    const auto received = exchange_phi_packets(
        packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget, &result.packet_resident_input_bytes);
    const auto receive_decode_budget = phi_decode_budget(packets, received.packets, *comm, packet_arena_budget);
    result.packet_sent_bytes = received.stats.sent_bytes;
    result.packet_received_bytes = received.stats.received_bytes;
    result.packet_exchange_peak_bytes = received.stats.peak_scratch_bytes;
    result.packet_resident_output_bytes = packet_capacity(received.packets, packet_arena_budget);
    result.packet_decode_cap = receive_decode_budget;
    std::map<GO, M> divergence, radius;
    std::vector<GO> face_ids;
    std::vector<int> senders;
    collective(*comm,
        [&]
        {
            std::map<GO, std::set<GO>> seen;
            for (const auto& p : received.packets)
            {
                Codec c{p.bytes, receive_decode_budget};
                const GO endpoint = c.get(), face = c.get();
                const auto flux = c.measure(scalar_budget), error = c.measure(scalar_budget);
                c.done();
                result.packet_codec_peak_bytes = std::max(result.packet_codec_peak_bytes, c.decode_used);
                if (map->getLocalElement(endpoint) < 0 || error.sign() < 0 || !seen[endpoint].insert(face).second)
                    throw std::runtime_error("Invalid projection admission route.");
                divergence[endpoint] = vof_exact::leased_add(divergence[endpoint], flux, scalar_budget);
                radius[endpoint] = vof_exact::leased_add(radius[endpoint], error, scalar_budget);
                face_ids.push_back(face);
                senders.push_back(p.peer);
            }
        });
    const std::set<GO> unique_face_set(face_ids.begin(), face_ids.end());
    Teuchos::Array<GO> unique_faces(unique_face_set.begin(), unique_face_set.end());
    Teuchos::Array<int> authority(unique_faces.size(), -1);
    (void) mesh->owned_face_map()->getRemoteIndexList(unique_faces(), authority());
    std::map<GO, int> face_authority;
    for (std::size_t i = 0; i < std::size_t(unique_faces.size()); ++i)
        face_authority.emplace(unique_faces[i], authority[i]);
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < face_ids.size(); ++i)
                if (face_authority.at(face_ids[i]) != senders[i])
                    throw std::runtime_error("Wrong projection face authority.");
        });
    int accepted = 1, global = 0;
    collective(*comm,
        [&]
        {
            for (const auto& [cell, flux] : divergence)
                accepted = accepted && absolute(flux, scalar_budget).compare(radius[cell]) <= 0;
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &accepted, &global);
    result.arithmetic_admitted = global;
    collective(*comm,
        [&]
        {
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            {
                const auto gid = map->getGlobalElement(static_cast<DefaultTpetraTypes::local_ordinal_type>(c));
                result.owned_cell_radius[c] = radius[gid];
                result.owned_net_rate[c] = divergence[gid];
            }
        });
    return result;
}
bool general_vof_projection_arithmetic_admitted(
    const SP<const MeshHandle<>>& mesh, const std::vector<GeneralVoFRoundedValue>& values, std::size_t budget)
{
    return certify_general_vof_projection_rate(mesh, values, budget).arithmetic_admitted;
}
} // namespace SimpleFluid::detail
