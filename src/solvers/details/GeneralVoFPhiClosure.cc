/** @file Exact::GeneralVoFPhiClosure.cc
 * @brief Sparse persistent-identity forest closure with bounded floating-point arithmetic.
 */
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <bit>
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
} // namespace
namespace finite_phi
{
long double leased_add(long double a, long double b, std::size_t)
{
    const auto v = a + b;
    if (!std::isfinite(v))
        throw std::runtime_error("Finite Phi addition exceeds wide FP range.");
    return v;
}
long double leased_scale(long double a, double b, std::size_t)
{
    const auto v = a * b;
    if (!std::isfinite(v) || (a != 0 && b != 0 && v == 0))
        throw std::runtime_error("Finite Phi scaling exceeds wide FP range.");
    return v;
}
} // namespace finite_phi
namespace
{
struct FCodec : Codec
{
    using Codec::Codec;
    using Codec::put;
    void put(long double value, std::size_t)
    {
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite Phi packet value.");
        int exponent = 0;
        const auto mantissa = std::frexp(value, &exponent);
        const double high = static_cast<double>(mantissa);
        const double low = static_cast<double>(mantissa - high);
        put(std::uint64_t(std::int64_t(exponent)));
        put(std::bit_cast<std::uint64_t>(high));
        put(std::bit_cast<std::uint64_t>(low));
    }
    long double finite_measure(std::size_t)
    {
        const auto exponent = std::int64_t(get());
        const auto high = std::bit_cast<double>(get()), low = std::bit_cast<double>(get());
        if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max() ||
            !std::isfinite(high) || !std::isfinite(low))
            throw std::runtime_error("Invalid finite Phi scalar packet.");
        const auto value = std::ldexp(static_cast<long double>(high) + low, int(exponent));
        if (!std::isfinite(value) || ((high != 0 || low != 0) && value == 0))
            throw std::runtime_error("Finite Phi scalar packet exceeds wide FP range.");
        return value;
    }
};
struct FCell
{
    GO cell;
    Bytes identity;
    long double arithmetic_radius = 0;
};
struct FFace
{
    GO face, first, second;
    Bytes identity;
    long double volume = 0, arithmetic_radius = 0;
};
struct FEdge
{
    GO face, other;
    int peer, authority, sign;
    Bytes identity;
    long double volume = 0, radius = 0;
};
struct FNode
{
    GO gid;
    Bytes identity, label;
    std::uint64_t depth = 0;
    std::vector<FEdge> edges;
    GO parent_face = -1, parent = -1;
    long double defect = 0, radius = 0, used = 0, roundoff = 0;
};
} // namespace
static GeneralVoFFiniteRateClosure close_finite_phi_impl(const SP<const MeshHandle<>>& mesh,
    const std::vector<FCell>& cells, const std::vector<FFace>& faces, std::size_t budget)
{
    namespace E = finite_phi;
    const auto map = mesh->owned_cell_map();
    const auto comm = map->getComm();
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(budget);
    std::size_t edge_bound = 0, max_cell_key = 0, max_face_key = 0, scalar_budget = 0, packet_budget = 0;
    std::size_t packet_exchange_budget = 0, packet_arena_budget = 0;
    collective(*comm,
        [&]
        {
            storage.capacity(Domain::Input, cells.capacity(), sizeof(FCell));
            storage.capacity(Domain::Input, faces.capacity(), sizeof(FFace));
            for (const auto& c : cells)
            {
                storage.capacity(Domain::Input, c.identity.capacity(), sizeof(char));

                max_cell_key = std::max(max_cell_key, c.identity.size());
            }
            for (const auto& f : faces)
            {
                storage.capacity(Domain::Input, f.identity.capacity(), sizeof(char));

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
            storage.capacity(Domain::Scratch, n, sizeof(FNode) + 4 * sizeof(void*));
            storage.capacity(Domain::Scratch, n, 2 * std::size_t(key_global[0]));
            storage.capacity(Domain::Scratch, e, 2 * sizeof(FEdge) + 2 * std::size_t(key_global[1]));
            storage.capacity(Domain::Scratch, 2 * f, 3 * sizeof(GO) + 2 * sizeof(int) + 12 * sizeof(void*));
            storage.capacity(Domain::Scratch, e, 3 * sizeof(GO) + 2 * sizeof(int) + 8 * sizeof(void*));
            storage.capacity(Domain::Scratch, n, 3 * (sizeof(std::pair<const GO, long double>) + 4 * sizeof(void*)));
            storage.capacity(Domain::Result, 2 * f, sizeof(long double));
            const auto packet_pool = storage.remaining() / 2;
            storage.capacity(Domain::Scratch, packet_pool, 1);
            packet_budget = packet_pool / 8;
            packet_exchange_budget = packet_pool / 2;
            packet_arena_budget = packet_pool;
            if (n > (std::numeric_limits<std::size_t>::max() - 48 - 2 * f - 2 * e) / 5)
                throw std::runtime_error("Phi closure scalar count overflow.");
            const auto result_slots = 2 * f, scratch_slots = 5 * n + 2 * e + 48;
            scalar_budget = sizeof(long double);
            if (packet_budget < 1024)
                throw std::runtime_error("Phi closure cumulative resource reserve exhausted.");
            storage.capacity(Domain::Scratch, scratch_slots, scalar_budget);
            storage.capacity(Domain::Result, result_slots, scalar_budget);
        });
    std::map<GO, FNode> nodes;
    std::map<GO, std::size_t> face_index;
    std::set<GO> endpoint_set;
    GeneralVoFFiniteRateClosure result;
    collective(*comm,
        [&]
        {
            result.rate.reserve(faces.size());
            result.correction.reserve(faces.size());
            if (cells.size() != mesh->num_owned_cells())
                throw std::runtime_error("Incomplete closure cells.");
            for (const auto& c : cells)
            {
                if (map->getLocalElement(c.cell) < 0 || c.identity.empty() || c.arithmetic_radius < 0 ||
                    nodes.contains(c.cell))
                    throw std::runtime_error("Foreign closure cell.");
                FNode n;
                n.gid = c.cell;
                n.identity = n.label = c.identity;
                n.radius = c.arithmetic_radius;
                nodes.emplace(c.cell, std::move(n));
            }
            for (std::size_t i = 0; i < faces.size(); ++i)
            {
                const auto& f = faces[i];
                const auto lid = mesh->owned_face_map()->getLocalElement(f.face);
                if (lid < 0 || f.first == f.second || f.identity.empty() || f.arithmetic_radius < 0 ||
                    !face_index.emplace(f.face, i).second)
                    throw std::runtime_error("Invalid closure face.");
                const auto lease = mesh->acquire_execution_view();
                if (mesh->is_geometry_exterior_face(lid) || mesh->cell_global_id(mesh->owner_cell(lid)) != f.first ||
                    mesh->cell_global_id(mesh->neighbor_cell(lid)) != f.second)
                    throw std::runtime_error("Wrong closure endpoint.");
                endpoint_set.insert(f.first);
                endpoint_set.insert(f.second);
                const auto proposed = f.volume;
                const long double rounding = 0;
                result.rate.push_back(proposed);
                result.correction.push_back(rounding);
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
                    FCodec c(packet_budget);
                    c.put(std::uint64_t(sign == 1 ? f.first : f.second));
                    c.put(std::uint64_t(sign == 1 ? f.second : f.first));
                    c.put(std::uint64_t(f.face));
                    c.put(std::uint64_t(owner.at(sign == 1 ? f.second : f.first)));
                    c.put(sign == 1 ? 1 : 0);
                    c.put(f.identity);
                    c.put(f.volume, scalar_budget);
                    c.put(f.arithmetic_radius, scalar_budget);
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
                FCodec c{p.bytes, receive_decode_budget};
                const auto gid = GO(c.get()), other = GO(c.get()), face = GO(c.get());
                const int peer = int(c.get()), sign = c.get() ? 1 : -1;
                const auto identity = c.blob();
                auto volume = c.finite_measure(scalar_budget), radius = c.finite_measure(scalar_budget);
                c.done();
                if (!nodes.contains(gid) || peer < 0 || peer >= comm->getSize())
                    throw std::runtime_error("Invalid incident closure edge.");
                auto& n = nodes.at(gid);
                if (std::any_of(n.edges.begin(), n.edges.end(), [&](const auto& e) { return e.face == face; }))
                    throw std::runtime_error("Duplicate incident closure edge.");
                n.edges.push_back({face, other, peer, p.peer, sign, identity, volume, radius});
                n.roundoff += 8 * std::numeric_limits<long double>::epsilon() * (std::abs(n.defect) + std::abs(volume));
                n.defect = E::leased_add(n.defect, E::leased_scale(volume, sign, scalar_budget), scalar_budget);
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
                        // Immutable faces remain in divergence and final
                        // incidence, but cannot route an arithmetic correction.
                        if (e.radius == 0)
                            continue;
                        FCodec c(packet_budget);
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
                    FCodec c{p.bytes, update_decode_budget};
                    const GO gid = c.get(), other = c.get(), face = c.get();
                    const auto label = c.blob();
                    const auto depth = c.get();
                    c.done();
                    auto& n = nodes.at(gid);
                    const auto found = std::find_if(n.edges.begin(), n.edges.end(),
                        [&](const auto& e) { return e.face == face && e.other == other && e.peer == p.peer; });
                    if (found == n.edges.end() || found->radius <= 0 || label.empty() || depth >= count)
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
    std::map<GO, long double> local_correction;
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
                        if (std::abs(delta) > e->radius)
                            throw GeneralVoFPhiInfeasible("Face closure exceeds arithmetic radius.");
                        local_correction[e->face] = delta;
                        n.used = E::leased_add(n.used, std::abs(delta), scalar_budget);
                        FCodec c(packet_budget);
                        c.put(std::uint64_t(e->other));
                        c.put(std::uint64_t(gid));
                        c.put(std::uint64_t(e->face));
                        c.put(n.defect, scalar_budget);
                        c.put(n.roundoff, scalar_budget);
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
                    FCodec c{p.bytes, sum_decode_budget};
                    const GO gid = c.get(), child = c.get(), face = c.get();
                    const auto sum = c.finite_measure(scalar_budget), sum_error = c.finite_measure(scalar_budget),
                               delta = c.finite_measure(scalar_budget);
                    c.done();
                    auto& n = nodes.at(gid);
                    const auto e = std::find_if(n.edges.begin(), n.edges.end(),
                        [&](const auto& e) { return e.face == face && e.other == child && e.peer == p.peer; });
                    if (e == n.edges.end())
                        throw std::runtime_error("Wrong forest sum route.");
                    n.roundoff += sum_error + 8 * std::numeric_limits<long double>::epsilon() *
                                                  (std::abs(n.defect) + std::abs(sum));
                    n.defect = E::leased_add(n.defect, sum, scalar_budget);
                    n.used = E::leased_add(n.used, std::abs(delta), scalar_budget);
                }
            });
    }
    collective(*comm,
        [&]
        {
            for (const auto& [gid, n] : nodes)
            {
                if ((!n.depth && std::abs(n.defect) > n.roundoff) || n.used > n.radius)
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
                        FCodec c(packet_budget);
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
                FCodec c{p.bytes, correction_decode_budget};
                const GO face = c.get(), child = c.get();
                const auto delta = c.finite_measure(scalar_budget);
                c.done();
                const auto i = face_index.at(face);
                const auto& f = faces[i];
                if (!seen.insert(face).second || (child != f.first && child != f.second) || owner.at(child) != p.peer ||
                    std::abs(delta) > f.arithmetic_radius)
                    throw std::runtime_error("Invalid canonical face correction.");
                result.correction[i] = E::leased_add(result.correction[i], delta, scalar_budget);
                result.rate[i] = E::leased_add(result.rate[i], delta, scalar_budget);
                if (std::abs(result.correction[i]) > f.arithmetic_radius)
                    throw std::runtime_error("Canonical rate change exceeds its original face certificate.");
            }
        });
    corrections = {}; // Do not retain old receive payload across final exchange.
    // Reassemble canonical corrected controls independently of the forest.
    // Only the derived wide-FP accumulation error admits a nonzero residual.
    packets.clear();
    packet_bytes = 0;
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < faces.size(); ++i)
                for (int sign : {1, -1})
                {
                    const auto endpoint = sign == 1 ? faces[i].first : faces[i].second;
                    FCodec c(packet_budget);
                    c.put(std::uint64_t(endpoint));
                    c.put(std::uint64_t(faces[i].face));
                    c.put(E::leased_scale(result.rate[i], sign, scalar_budget), scalar_budget);
                    enqueue(owner.at(endpoint), std::move(c.bytes));
                }
        });
    const auto final_controls =
        exchange_phi_packets(packets, comm, packet_budget, packet_exchange_budget, packet_arena_budget);
    const auto final_decode_budget = phi_decode_budget(packets, final_controls.packets, *comm, packet_arena_budget);
    collective(*comm,
        [&]
        {
            std::map<GO, long double> divergence, uncertainty;
            std::map<GO, std::set<GO>> seen;
            for (const auto& p : final_controls.packets)
            {
                FCodec c{p.bytes, final_decode_budget};
                const GO endpoint = c.get(), face = c.get();
                const auto value = c.finite_measure(scalar_budget);
                c.done();
                const auto& n = nodes.at(endpoint);
                const auto e = std::find_if(n.edges.begin(), n.edges.end(),
                    [&](const auto& e) { return e.face == face && e.authority == p.peer; });
                if (e == n.edges.end() || !seen[endpoint].insert(face).second)
                    throw std::runtime_error("Invalid corrected Phi authority.");
                uncertainty[endpoint] += 16 * std::numeric_limits<long double>::epsilon() *
                                         (std::abs(divergence[endpoint]) + std::abs(value));
                divergence[endpoint] = E::leased_add(divergence[endpoint], value, scalar_budget);
            }
            for (const auto& [gid, n] : nodes)
                if (seen[gid].size() != n.edges.size() || std::abs(divergence[gid]) > uncertainty[gid])
                    throw std::runtime_error("Finite corrected Phi exceeds its operation-derived closure error.");
        });
    result.admitted_storage_bytes = budget - storage.remaining();
    return result;
}
GeneralVoFFiniteRateClosure close_general_vof_finite_rate(const SP<const MeshHandle<>>& mesh,
    std::span<const long double> rate, std::span<const long double> face_radius,
    std::span<const long double> cell_radius, std::size_t budget,
    const GeneralVoFFiniteProjectionClosureTopology* topology)
{
    std::vector<FCell> cells;
    std::vector<FFace> faces;
    const auto comm = mesh->owned_cell_map()->getComm();
    collective(*comm,
        [&]
        {
            if (rate.size() != mesh->num_owned_faces() || face_radius.size() != rate.size() ||
                cell_radius.size() != mesh->num_owned_cells() || std::numeric_limits<long double>::digits > 106 ||
                !std::numeric_limits<long double>::is_iec559)
                throw std::runtime_error("Finite Phi shapes or FP environment are unsupported.");
            const auto identity = [](GO gid)
            {
                Bytes bytes(8);
                for (unsigned i = 0; i < 8; ++i)
                    bytes[i] = char(std::uint64_t(gid) >> (56 - i * 8));
                return bytes;
            };
            for (std::size_t c = 0; c < cell_radius.size(); ++c)
            {
                if (!std::isfinite(cell_radius[c]) || cell_radius[c] < 0)
                    throw std::runtime_error("Invalid finite cell arithmetic capacity.");
                const auto gid = mesh->cell_global_id(DefaultTpetraTypes::local_ordinal_type(c));
                cells.push_back({gid, identity(gid), cell_radius[c]});
            }
            for (std::size_t f = 0; f < rate.size(); ++f)
            {
                const auto lid = DefaultTpetraTypes::local_ordinal_type(f);
                if (!std::isfinite(rate[f]) || !std::isfinite(face_radius[f]) || face_radius[f] < 0)
                    throw std::runtime_error("Invalid finite face arithmetic capacity.");
                if (mesh->is_geometry_exterior_face(lid))
                {
                    if (rate[f] != 0)
                        throw std::runtime_error("Finite Phi requires impermeable exterior faces.");
                    continue;
                }
                const auto gid = mesh->face_global_id(lid);
                faces.push_back({gid, mesh->cell_global_id(mesh->owner_cell(lid)),
                    mesh->cell_global_id(mesh->neighbor_cell(lid)), identity(gid), rate[f], face_radius[f]});
            }
            if (topology)
            {
                if (topology->cells.size() != cells.size() || topology->faces.size() != faces.size())
                    throw std::runtime_error("Finite Phi persistent topology shape mismatch.");
                for (std::size_t c = 0; c < cells.size(); ++c)
                {
                    if (topology->cells[c].cell != cells[c].cell || topology->cells[c].identity.empty())
                        throw std::runtime_error("Finite Phi foreign persistent cell identity.");
                    cells[c].identity = topology->cells[c].identity;
                }
                for (std::size_t f = 0; f < faces.size(); ++f)
                {
                    const auto& t = topology->faces[f];
                    if (t.face != faces[f].face || t.first != faces[f].first || t.second != faces[f].second ||
                        t.identity.empty())
                        throw std::runtime_error("Finite Phi foreign persistent face identity.");
                    faces[f].identity = t.identity;
                }
            }
        });
    std::size_t wrapper_storage = 0;
    collective(*comm,
        [&]
        {
            if (rate.size() > budget / (2 * sizeof(long double)) || cell_radius.size() > budget / sizeof(long double))
                throw std::length_error("Finite Phi output capacity exceeds its payload cap.");
            wrapper_storage = 2 * rate.size() * sizeof(long double);
            if (cell_radius.size() > (budget - wrapper_storage) / sizeof(long double))
                throw std::length_error("Finite Phi simultaneous output capacity exhausts its payload cap.");
            wrapper_storage += cell_radius.size() * sizeof(long double);

            if (wrapper_storage >= budget)
                throw std::length_error("Finite Phi simultaneous output capacity exhausts its payload cap.");
        });
    auto internal = close_finite_phi_impl(mesh, cells, faces, budget - wrapper_storage);
    GeneralVoFFiniteRateClosure result;
    result.rate.assign(rate.begin(), rate.end());
    result.correction.resize(rate.size());
    result.cell_radius.assign(cell_radius.begin(), cell_radius.end());
    result.forest_rounds = internal.forest_rounds;
    result.admitted_storage_bytes = internal.admitted_storage_bytes + wrapper_storage;
    for (std::size_t f = 0; f < faces.size(); ++f)
    {
        const auto lid = mesh->owned_face_map()->getLocalElement(faces[f].face);
        result.rate[lid] = internal.rate[f];
        result.correction[lid] = internal.correction[f];
    }
    return result;
}
#include "solvers/details/GeneralVoFExactPhiClosure.inc"
} // namespace SimpleFluid::detail
