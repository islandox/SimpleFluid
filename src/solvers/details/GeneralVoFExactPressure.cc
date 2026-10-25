/** @file GeneralVoFExactPressure.cc
 * @brief Bounded orthogonal pressure certificates and distributed exact CG.
 */
#include "solvers/details/GeneralVoFExactPressure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <bit>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
using GO = DefaultTpetraTypes::global_ordinal_type;
using LO = DefaultTpetraTypes::local_ordinal_type;
using Comm = Teuchos::Comm<int>;
using Bytes = std::vector<char>;
template<class Work> void collective(const Comm& comm, Work work)
{
    std::exception_ptr error;
    try
    {
        work();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = bool(error), global = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &global);
    if (global)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("Exact orthogonal pressure rejected on another rank.");
    }
}
void word(Bytes& out, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        out.push_back(char(value >> (8 * i)));
}
struct Reader
{
    std::span<const char> bytes;
    std::size_t position = 0;
    std::uint64_t word()
    {
        if (position > bytes.size() || bytes.size() - position < 8)
            throw std::runtime_error("Truncated exact pressure packet.");
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= std::uint64_t(static_cast<unsigned char>(bytes[position++])) << (8 * i);
        return value;
    }
    M measure(std::size_t limit)
    {
        auto n = word();
        if (n > bytes.size() - position || n > limit)
            throw std::runtime_error("Invalid exact pressure scalar framing.");
        auto value = M::decode(bytes.subspan(position, std::size_t(n)), limit);
        position += std::size_t(n);
        return value;
    }
    void done() const
    {
        if (position != bytes.size())
            throw std::runtime_error("Trailing exact pressure packet.");
    }
};
void append(Bytes& bytes, const M& value, std::size_t limit)
{
    const auto encoded = value.encode(limit);
    word(bytes, encoded.size());
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
}
struct Arena
{
    std::size_t limit, used = 0;
    void reserve(std::size_t size)
    {
        if (used > limit || size > limit - used)
            throw std::runtime_error("Exact pressure packet arena exhausted.");
        used += size;
    }
};
std::map<GO, int> owners_of(const DefaultTpetraTypes::map_type& map, const Teuchos::Array<GO>& original)
{
    const auto comm = map.getComm();
    Teuchos::Array<GO> unique;
    Teuchos::Array<int> ranks;
    collective(*comm,
        [&]
        {
            const std::set<GO> ids(original.begin(), original.end());
            unique.assign(ids.begin(), ids.end());
            ranks.resize(unique.size(), -1);
        });
    (void) map.getRemoteIndexList(unique(), ranks());
    std::map<GO, int> result;
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(unique.size()); ++i)
            {
                if (ranks[i] < 0)
                    throw std::runtime_error("Absent exact pressure directory owner.");
                result.emplace(unique[i], ranks[i]);
            }
            for (const auto gid : original)
                if (!result.contains(gid))
                    throw std::runtime_error("Incomplete exact pressure owner lookup.");
        });
    return result;
}
struct Edge
{
    LO row;
    GO other, face;
    M k;
};
struct Request
{
    int peer;
    std::uint64_t index;
    GO gid;
};
class Solve
{
public:
    SP<const MeshHandle<>> mesh;
    Teuchos::RCP<const Comm> comm;
    GeneralVoFExactStorage storage;
    std::size_t packet_budget = 0, cap = 0, aggregate_cap = 0, applications = 0;
    std::uint64_t epoch = 0;
    std::vector<Edge> edges;
    std::vector<GO> overlap_ids;
    std::vector<int> overlap_owners;
    std::vector<Request> imports;
    std::span<const GeneralVoFExactPressureFace> faces;
    Solve(SP<const MeshHandle<>> value, std::span<const GeneralVoFExactPressureFace> f, std::size_t budget)
        : mesh(std::move(value)), comm(mesh->owned_cell_map()->getComm()), storage(budget),
          epoch(mesh->geometry_epoch()), faces(f)
    {
    }
    M add(const M& a, const M& b) const
    {
        scalar(a);
        scalar(b);
        return retain_vector(vof_exact::leased_add(a, b, aggregate_cap));
    }
    M neg(const M& a) const
    {
        scalar(a);
        return retain_vector(vof_exact::leased_scale(a, -1, aggregate_cap));
    }
    M product(const M& a, const M& b) const { return retain_vector(aggregate_product(a, b)); }
    M retain_vector(M result) const
    {
        if (result.storage_bytes() > cap)
        {
            // The old aggregate and fresh copy coexist inside the existing
            // synchronous16A operation lease; the published vector still fitsK.
            if (aggregate_cap > std::numeric_limits<std::size_t>::max() / 16)
                throw std::runtime_error("Exact pressure compact operation lease overflows.");
            result = result.compacted(16 * aggregate_cap);
        }
        scalar(result);
        return result;
    }
    M aggregate_product(const M& a, const M& b) const
    {
        if (!a.sign() || !b.sign())
            return {};
        const auto inverse = vof_exact::leased_divide(M::from_value(1), b, aggregate_cap);
        return vof_exact::leased_divide(a, inverse, aggregate_cap);
    }
    void scalar(const M& value) const
    {
        const auto bytes = value.storage_bytes();
        if (bytes > cap)
            throw std::runtime_error("Exact pressure scalar capacity exhausted: stored_bytes=" + std::to_string(bytes) +
                                     ", limit=" + std::to_string(cap) + ".");
    }
    M global(const M& local)
    {
        std::vector<OwnerPacket> packets;
        collective(*comm,
            [&]
            {
                if (local.storage_bytes() > aggregate_cap)
                    throw std::runtime_error("Exact pressure aggregate capacity exhausted.");
                packets.push_back({0, local.encode(aggregate_cap)});
            });
        auto received = exchange_owner_packets(packets, comm, packet_budget);
        M result;
        Bytes encoded;
        collective(*comm,
            [&]
            {
                if (comm->getRank() == 0)
                {
                    std::set<int> senders;
                    for (const auto& packet : received.packets)
                    {
                        if (!senders.insert(packet.peer).second)
                            throw std::runtime_error("Duplicate exact pressure reduction sender.");
                        result = vof_exact::leased_add(result, M::decode(packet.bytes, aggregate_cap), aggregate_cap);
                    }
                    if (senders.size() != std::size_t(comm->getSize()))
                        throw std::runtime_error("Missing exact pressure reduction sender.");
                    encoded = result.encode(aggregate_cap);
                }
            });
        std::uint64_t count = encoded.size();
        Teuchos::broadcast(*comm, 0, 1, &count);
        collective(*comm,
            [&]
            {
                if (count > aggregate_cap || count > std::uint64_t(std::numeric_limits<int>::max()))
                    throw std::runtime_error("Exact pressure scalar broadcast exceeds budget.");
                encoded.resize(std::size_t(count));
            });
        Teuchos::broadcast(*comm, 0, int(count), encoded.data());
        collective(*comm,
            [&]
            {
                result = M::decode(encoded, aggregate_cap);
                if (result.storage_bytes() > aggregate_cap)
                    throw std::runtime_error("Exact pressure aggregate capacity exhausted.");
            });
        return result;
    }
    M dot(std::span<const M> a, std::span<const M> b)
    {
        M local;
        collective(*comm,
            [&]
            {
                if (a.size() != b.size())
                    throw std::runtime_error("Exact pressure inner-product shape mismatch.");
                for (std::size_t i = 0; i < a.size(); ++i)
                    local = vof_exact::leased_add(local, aggregate_product(a[i], b[i]), aggregate_cap);
            });
        return global(local);
    }
    std::vector<M> halo(std::span<const M> owned)
    {
        std::vector<OwnerPacket> packets;
        std::vector<M> result;
        collective(*comm,
            [&]
            {
                if (mesh->geometry_epoch() != epoch)
                    throw std::runtime_error("Exact pressure source geometry changed.");
                result.resize(overlap_ids.size());
                for (std::size_t i = 0; i < overlap_ids.size(); ++i)
                    if (overlap_owners[i] == comm->getRank())
                        result[i] = owned[mesh->owned_cell_map()->getLocalElement(overlap_ids[i])];
                Arena arena{packet_budget};
                for (const auto& request : imports)
                {
                    const auto row = mesh->owned_cell_map()->getLocalElement(request.gid);
                    if (row < 0)
                        throw std::runtime_error("Exact pressure halo request changed ownership.");
                    scalar(owned[row]);
                    arena.reserve(24 + owned[row].encode(cap).size() + sizeof(OwnerPacket));
                    Bytes bytes;
                    word(bytes, request.index);
                    word(bytes, std::bit_cast<std::uint64_t>(request.gid));
                    append(bytes, owned[row], cap);
                    packets.push_back({request.peer, std::move(bytes)});
                }
            });
        auto received = exchange_owner_packets(packets, comm, packet_budget);
        collective(*comm,
            [&]
            {
                std::set<std::size_t> seen;
                for (const auto& packet : received.packets)
                {
                    Reader reader{packet.bytes};
                    auto index = reader.word();
                    auto gid = std::bit_cast<GO>(reader.word());
                    if (index >= overlap_ids.size() || overlap_ids[index] != gid ||
                        overlap_owners[index] != packet.peer || packet.peer == comm->getRank() ||
                        !seen.insert(std::size_t(index)).second)
                        throw std::runtime_error("Foreign or duplicate exact pressure halo response.");
                    result[index] = reader.measure(cap);
                    reader.done();
                    scalar(result[index]);
                }
                for (std::size_t i = 0; i < overlap_ids.size(); ++i)
                    if (overlap_owners[i] != comm->getRank() && !seen.contains(i))
                        throw std::runtime_error("Missing exact pressure halo response.");
            });
        return result;
    }
    std::vector<M> apply(std::span<const M> owned)
    {
        const auto values = halo(owned);
        ++applications;
        std::vector<M> result;
        collective(*comm,
            [&]
            {
                result.resize(mesh->num_owned_cells());
                for (const auto& edge : edges)
                {
                    const auto other = mesh->overlap_cell_map()->getLocalElement(edge.other);
                    if (other < 0)
                        throw std::runtime_error("Exact pressure graph lacks halo endpoint.");
                    result[edge.row] = add(result[edge.row], product(edge.k, add(owned[edge.row], neg(values[other]))));
                }
            });
        return result;
    }
};
} // namespace
GeneralVoFExactPressureResult solve_general_vof_exact_pressure(const SP<const MeshHandle<>>& mesh,
    std::span<const GeneralVoFExactPressureFace> faces, std::span<const M> rhs, int remaining_iterations,
    std::size_t budget)
{
    if (!mesh)
        throw std::invalid_argument("Exact pressure needs an actual native mesh.");
    const auto comm = mesh->owned_cell_map()->getComm();
    std::unique_ptr<Solve> solve;
    Teuchos::Array<GO> endpoints, face_ids;
    collective(*comm,
        [&]
        {
            if (remaining_iterations < 0 || !budget || rhs.size() != mesh->num_owned_cells())
                throw std::runtime_error("Invalid exact pressure budget or RHS shape.");
            solve = std::make_unique<Solve>(mesh, faces, budget);
            auto& storage = solve->storage;
            storage.capacity(GeneralVoFStorageDomain::Input, rhs.size(), sizeof(M));
            for (const auto& value : rhs)
                storage.measure(GeneralVoFStorageDomain::Input, value);
            storage.capacity(GeneralVoFStorageDomain::Input, faces.size(), sizeof(GeneralVoFExactPressureFace));
            for (const auto& face : faces)
                storage.measure(GeneralVoFStorageDomain::Input, face.mobility);
            const auto n = mesh->num_owned_cells(), h = mesh->num_cells(), f = faces.size();
            std::size_t incidence_bound = 0;
            {
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t i = 0; i < n; ++i)
                {
                    const auto degree = mesh->faces(LO(i)).size();
                    if (degree > std::numeric_limits<std::size_t>::max() - incidence_bound)
                        throw std::runtime_error("Exact pressure incidence count overflow.");
                    incidence_bound += degree;
                }
            }
            const auto maximum = std::numeric_limits<std::size_t>::max();
            if (n > maximum / 64 || h > maximum / 64 || f > maximum / 64 || incidence_bound > maximum / 64)
                throw std::runtime_error("Exact pressure capacity count overflow.");
            storage.capacity(GeneralVoFStorageDomain::Scratch, h + 2 * f + incidence_bound,
                8 * sizeof(void*) + sizeof(Edge) + 4 * sizeof(GO));
            solve->edges.reserve(incidence_bound);
            solve->packet_budget = storage.remaining() / 16;
            // Packet input/output, transport scratch, import-route containers and
            // bounded decode sets share these conservatively reserved arenas.
            storage.capacity(GeneralVoFStorageDomain::Scratch, 8 * solve->packet_budget, 1);
            const auto slots = 12 * n + 6 * h + 6 * f + 2 * incidence_bound + 96;
            if (slots < n)
                throw std::runtime_error("Exact pressure scalar slot overflow.");
            // Global inner products may carry more limbs than any vector entry.
            // Keep vector slots at K and separately reserve32A, A=4K: at most
            // sixteen aggregate/pending values plus one synchronous16A operation.
            // Both reservations remain inside the original helper budget.
            constexpr std::size_t aggregate_slots = 32, aggregate_ratio = 4;
            if (slots > maximum - aggregate_slots * aggregate_ratio)
                throw std::runtime_error("Exact pressure aggregate slot overflow.");
            solve->cap = storage.remaining() / (slots + aggregate_slots * aggregate_ratio);
            solve->aggregate_cap = aggregate_ratio * solve->cap;
            if (solve->cap < 1024 || solve->packet_budget < 1024)
                throw std::runtime_error("Exact pressure cumulative reserve exhausted.");
            storage.capacity(GeneralVoFStorageDomain::Scratch, slots, solve->cap);
            storage.capacity(GeneralVoFStorageDomain::Scratch, aggregate_slots, solve->aggregate_cap);
            // At least16 of these dedicated slots form one scalar-operation lease;
            // the remainder cover live vectors, pending results and scalar reductions.
            std::set<GO> seen;
            const auto lease = mesh->acquire_execution_view();
            for (const auto& face : faces)
            {
                const auto local = mesh->owned_face_map()->getLocalElement(face.face);
                if (local < 0 || mesh->is_geometry_exterior_face(local) || face.first == face.second ||
                    face.mobility.sign() <= 0 || mesh->cell_global_id(mesh->owner_cell(local)) != face.first ||
                    mesh->cell_global_id(mesh->neighbor_cell(local)) != face.second || !seen.insert(face.face).second)
                    throw std::runtime_error("Invalid exact pressure canonical face.");
                solve->scalar(face.mobility);
                endpoints.push_back(face.first);
                endpoints.push_back(face.second);
            }
            for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
                if (!mesh->is_geometry_exterior_face(LO(i)) && !seen.contains(mesh->face_global_id(LO(i))))
                    throw std::runtime_error("Incomplete exact pressure canonical faces.");
            for (const auto& value : rhs)
                solve->scalar(value);
        });
    int minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &remaining_iterations, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &remaining_iterations, &maximum);
    collective(*comm,
        [&]
        {
            if (minimum != maximum)
                throw std::runtime_error("Exact pressure remaining iterations differ across ranks.");
        });
    const auto endpoint_owners = owners_of(*mesh->owned_cell_map(), endpoints);
    std::vector<OwnerPacket> packets;
    collective(*comm,
        [&]
        {
            Arena arena{solve->packet_budget};
            for (std::size_t i = 0; i < faces.size(); ++i)
                for (unsigned side = 0; side < 2; ++side)
                {
                    if (endpoint_owners.at(endpoints[2 * i + side]) < 0)
                        throw std::runtime_error("Missing exact pressure endpoint owner.");
                    arena.reserve(32 + faces[i].mobility.encode(solve->cap).size() + sizeof(OwnerPacket));
                    Bytes bytes;
                    word(bytes, std::bit_cast<std::uint64_t>(faces[i].face));
                    word(bytes, std::bit_cast<std::uint64_t>(side ? faces[i].second : faces[i].first));
                    word(bytes, std::bit_cast<std::uint64_t>(side ? faces[i].first : faces[i].second));
                    append(bytes, faces[i].mobility, solve->cap);
                    packets.push_back({endpoint_owners.at(endpoints[2 * i + side]), std::move(bytes)});
                }
        });
    auto incidence = exchange_owner_packets(packets, comm, solve->packet_budget);
    collective(*comm,
        [&]
        {
            for (const auto& packet : incidence.packets)
            {
                Reader reader{packet.bytes};
                face_ids.push_back(std::bit_cast<GO>(reader.word()));
            }
        });
    const auto face_owners = owners_of(*mesh->owned_face_map(), face_ids);
    collective(*comm,
        [&]
        {
            const auto lease = mesh->acquire_execution_view();
            std::set<std::pair<GO, GO>> seen;
            for (std::size_t i = 0; i < incidence.packets.size(); ++i)
            {
                const auto& packet = incidence.packets[i];
                Reader reader{packet.bytes};
                const auto face = std::bit_cast<GO>(reader.word()), gid = std::bit_cast<GO>(reader.word()),
                           other = std::bit_cast<GO>(reader.word());
                auto k = reader.measure(solve->cap);
                reader.done();
                const auto row = mesh->owned_cell_map()->getLocalElement(gid);
                if (row < 0 || face_owners.at(face_ids[i]) != packet.peer || !seen.emplace(face, gid).second ||
                    k.sign() <= 0 || mesh->overlap_cell_map()->getLocalElement(other) < 0)
                    throw std::runtime_error("Foreign exact pressure incidence.");
                if (solve->edges.size() == solve->edges.capacity())
                    throw std::runtime_error("Exact pressure incidence exceeds native degree bound.");
                const auto native_face = mesh->overlap_face_map()->getLocalElement(face);
                if (native_face < 0 || mesh->is_geometry_exterior_face(native_face))
                    throw std::runtime_error("Exact pressure incidence is not a local internal face.");
                const auto native_first = mesh->cell_global_id(mesh->owner_cell(native_face));
                const auto native_second = mesh->cell_global_id(mesh->neighbor_cell(native_face));
                if (!((native_first == gid && native_second == other) ||
                        (native_first == other && native_second == gid)))
                    throw std::runtime_error("Exact pressure incidence disagrees with native endpoints.");
                solve->scalar(k);
                solve->edges.push_back({row, other, face, std::move(k)});
            }
            for (std::size_t row = 0; row < mesh->num_owned_cells(); ++row)
                for (auto local : mesh->faces(LO(row)))
                    if (!mesh->is_geometry_exterior_face(local) &&
                        !seen.contains({mesh->face_global_id(local), mesh->cell_global_id(LO(row))}))
                        throw std::runtime_error("Missing exact pressure owned-cell incidence.");
            std::vector<OwnerPacket>().swap(incidence.packets);
            std::vector<OwnerPacket>().swap(packets);
            const auto ids = mesh->overlap_cell_map()->getLocalElementList();
            solve->overlap_ids.assign(ids.begin(), ids.end());
            solve->overlap_owners.resize(ids.size(), -1);
        });
    Teuchos::Array<GO> halo_ids;
    Teuchos::Array<int> halo_owners;
    collective(*comm,
        [&]
        {
            halo_ids.assign(solve->overlap_ids.begin(), solve->overlap_ids.end());
            halo_owners.resize(halo_ids.size(), -1);
        });
    (void) mesh->owned_cell_map()->getRemoteIndexList(halo_ids(), halo_owners());
    packets.clear();
    collective(*comm,
        [&]
        {
            Arena arena{solve->packet_budget};
            for (std::size_t i = 0; i < solve->overlap_ids.size(); ++i)
            {
                solve->overlap_owners[i] = halo_owners[i];
                if (halo_owners[i] < 0)
                    throw std::runtime_error("Absent exact pressure halo owner.");
                if (halo_owners[i] == comm->getRank())
                    continue;
                arena.reserve(16 + sizeof(OwnerPacket));
                Bytes bytes;
                word(bytes, i);
                word(bytes, std::bit_cast<std::uint64_t>(solve->overlap_ids[i]));
                packets.push_back({halo_owners[i], std::move(bytes)});
            }
        });
    auto requests = exchange_owner_packets(packets, comm, solve->packet_budget);
    collective(*comm,
        [&]
        {
            Arena import_metadata{2 * solve->packet_budget};
            if (requests.packets.size() > import_metadata.limit / sizeof(Request))
                throw std::runtime_error("Exact pressure imported route count exceeds metadata arena.");
            import_metadata.reserve(requests.packets.size() * sizeof(Request));
            solve->imports.reserve(requests.packets.size());
            std::set<std::pair<int, std::uint64_t>> seen;
            for (const auto& packet : requests.packets)
            {
                Reader reader{packet.bytes};
                auto index = reader.word();
                auto gid = std::bit_cast<GO>(reader.word());
                reader.done();
                import_metadata.reserve(8 * sizeof(void*));
                if (mesh->owned_cell_map()->getLocalElement(gid) < 0 || !seen.emplace(packet.peer, index).second)
                    throw std::runtime_error("Foreign exact pressure halo request.");
                solve->imports.push_back({packet.peer, index, gid});
            }
            std::vector<OwnerPacket>().swap(requests.packets);
            std::vector<OwnerPacket>().swap(packets);
        });
    GeneralVoFExactPressureResult result;
    std::vector<M> residual, direction;
    collective(*comm,
        [&]
        {
            result.owned_pressure.resize(rhs.size());
            residual.assign(rhs.begin(), rhs.end());
            direction = residual;
        });
    auto rr = solve->dot(residual, residual);
    while (rr.sign())
    {
        collective(*comm,
            [&]
            {
                if (result.iterations >= remaining_iterations)
                    throw std::runtime_error("Exact pressure shared iteration budget exhausted.");
            });
        const auto image = solve->apply(direction);
        const auto curvature = solve->dot(direction, image);
        M alpha;
        collective(*comm,
            [&]
            {
                if (curvature.sign() <= 0)
                    throw std::runtime_error("Exact pressure incompatible graph or CG breakdown.");
                alpha = vof_exact::leased_divide(rr, curvature, solve->aggregate_cap);
                for (std::size_t i = 0; i < rhs.size(); ++i)
                    result.owned_pressure[i] =
                        solve->add(result.owned_pressure[i], solve->product(alpha, direction[i]));
            });
        ++result.iterations;
        const auto applied = solve->apply(result.owned_pressure);
        collective(*comm,
            [&]
            {
                for (std::size_t i = 0; i < rhs.size(); ++i)
                    residual[i] = solve->add(rhs[i], solve->neg(applied[i]));
            });
        const auto next = solve->dot(residual, residual);
        if (!next.sign())
        {
            rr = next;
            break;
        }
        collective(*comm,
            [&]
            {
                const auto beta = vof_exact::leased_divide(next, rr, solve->aggregate_cap);
                for (std::size_t i = 0; i < rhs.size(); ++i)
                    direction[i] = solve->add(residual[i], solve->product(beta, direction[i]));
            });
        rr = next;
    }
    const auto check = solve->apply(result.owned_pressure);
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < rhs.size(); ++i)
                if (check[i].compare(rhs[i]))
                    throw std::runtime_error("Exact pressure recomputed residual is nonzero.");
        });
    result.overlap_pressure = solve->halo(result.owned_pressure);
    collective(*comm,
        [&]
        {
            result.face_rate_increment.reserve(faces.size());
            for (const auto& face : faces)
            {
                const auto a = mesh->overlap_cell_map()->getLocalElement(face.first),
                           b = mesh->overlap_cell_map()->getLocalElement(face.second);
                if (a < 0 || b < 0)
                    throw std::runtime_error("Exact pressure canonical face lacks endpoint halo.");
                result.face_rate_increment.push_back(solve->product(
                    face.mobility, solve->add(result.overlap_pressure[a], solve->neg(result.overlap_pressure[b]))));
            }
            if (mesh->geometry_epoch() != solve->epoch)
                throw std::runtime_error("Exact pressure source geometry changed before return.");
            result.admitted_storage_bytes = solve->storage.finish().total();
            result.operator_applications = solve->applications;
            GeneralVoFExactStorage retained(budget);
            for (const auto* values : {&result.owned_pressure, &result.overlap_pressure, &result.face_rate_increment})
            {
                retained.capacity(GeneralVoFStorageDomain::Result, values->capacity(), sizeof(M));
                for (const auto& value : *values)
                    retained.measure(GeneralVoFStorageDomain::Result, value);
            }
            result.storage_bytes = retained.finish().total();
        });
    return result;
}
} // namespace SimpleFluid::detail
