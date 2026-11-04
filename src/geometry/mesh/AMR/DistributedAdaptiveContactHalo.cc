/** @file DistributedAdaptiveContactHalo.cc
 * @brief Bounded owner queries over complete canonical point incidence.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/mesh/AMR/BudgetedOwnerExchange.hh"
#include "geometry/mesh/AMR/ClosedContactIndex.hh"
#include "geometry/mesh/AMR/ContactCoverIndex.hh"
#include "geometry/mesh/AMR/ContactPairSet.hh"
#include "geometry/mesh/AMR/ContactTracePool.hh"
#include "geometry/mesh/AMR/DeferredOwnerCharges.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "geometry/mesh/AMR/PagedRecords.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::uint64_t halo_magic = 0x53464348414c4f31ULL;
using Kind = ContactHaloMessageKind;
constexpr std::size_t header_bytes = 13 * 8;
constexpr std::size_t map_links = 4 * sizeof(void*);
struct Bounds
{
    AdaptiveContactBounds box;
    bool present = false;
};
void include(Bounds& target, const AdaptiveContactBounds& box)
{
    if (!target.present)
        target = {box, true};
    else
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            target.box.lower[axis] = std::min(target.box.lower[axis], box.lower[axis]);
            target.box.upper[axis] = std::max(target.box.upper[axis], box.upper[axis]);
        }
}
bool overlaps(const Bounds& first, const Bounds& second)
{
    return first.present && second.present && closed_contact_overlap(first.box, second.box);
}
void write_bounds(Writer& w, const Bounds& value)
{
    w.word(value.present);
    if (value.present)
        for (const auto* array : {&value.box.lower, &value.box.upper})
            for (const auto scalar : *array)
                for (const auto word : PLIC::scalar_packet(scalar))
                    w.word(word);
}
Bounds read_bounds(Reader& r)
{
    Bounds result;
    const auto present = r.word();
    if (present > 1)
        throw std::invalid_argument("Invalid spatial closure cover presence.");
    result.present = present;
    if (result.present)
    {
        for (auto* array : {&result.box.lower, &result.box.upper})
            for (auto& scalar : *array)
            {
                PLIC::ScalarPacketWords words;
                for (auto& word : words)
                    word = r.word();
                scalar = PLIC::scalar_from_packet(words);
                if (!std::isfinite(scalar))
                    throw std::invalid_argument("Nonfinite spatial closure cover.");
            }
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (result.box.upper[axis] < result.box.lower[axis])
                throw std::invalid_argument("Inverted spatial closure cover.");
    }
    return result;
}
AdaptiveBytes point_key(std::span<const char> domain, const ContactPointIdentity& point)
{
    Writer w;
    w.data.reserve(16 + domain.size() + PLIC::canonical_key_size(point));
    w.bytes(domain);
    w.bytes(PLIC::canonical_key(point));
    return std::move(w.data);
}
std::array<long double, 3> coordinates(const ContactPointIdentity& point)
{
    const auto values = PLIC::numerical_point<ArithmeticMode::Ordinary>(point);
    return {values[0].value, values[1].value, values[2].value};
}
bool contains(const AdaptiveContactBounds& box, const std::array<long double, 3>& point)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (point[axis] < box.lower[axis] || box.upper[axis] < point[axis])
            return false;
    return true;
}
void replace_word(AdaptiveBytes& bytes, std::size_t offset, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.at(offset + i) = char(value >> (8 * i));
}
} // namespace

struct DistributedAdaptiveContactHalo::Impl
{
    struct LocalOccurrence
    {
        std::size_t record, row, point;
    };
    struct PointGroup
    {
        const AdaptiveBytes* domain;
        std::array<long double, 3> coordinate;
        std::vector<LocalOccurrence> local;
        std::size_t cover_leaf = std::numeric_limits<std::size_t>::max();
    };
    struct Record
    {
        AdaptiveCellKey key;
        AdaptiveBytes domain;
        std::uint64_t revision;
        int owner = -1;
        std::shared_ptr<const ContactRecord> contact;
        std::size_t row = std::numeric_limits<std::size_t>::max(), body_bytes = 0, borrowed_bytes = 0, remote_refs = 0;
        std::optional<AdaptiveSplit> required;
    };
    struct RemoteOccurrence
    {
        std::size_t record;
        ContactPointIdentity physical;
    };
    struct DirectoryNode
    {
        Bounds left, right;
    };
    struct Frame
    {
        std::uint64_t query, page, total;
        bool final;
    };
    const AdaptiveProviderKernel& kernel;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    AdaptiveClosureOptions options;
    ContactHaloContext context;
    ContactHaloBeforeSend before_send;
    std::shared_ptr<Budget> arena;
    Budget& budget;
    DeferredOwnerCharges deferred_owners;
    IdentityBudget identities;
    std::optional<ContactTracePool> trace_pool;
    ContactHaloStatistics statistics;
    AdaptiveClosureStats diagnostics;
    std::vector<std::shared_ptr<const ContactIncidenceRow>> rows;
    std::vector<std::size_t> borrowed_rows;
    std::unique_ptr<DistributedAdaptiveIncidenceSession> incidence;
    struct PointKeyLess
    {
        using is_transparent = void;
        bool operator()(const AdaptiveBytes* a, const AdaptiveBytes* b) const { return *a < *b; }
        bool operator()(const AdaptiveBytes* a, const AdaptiveBytes& b) const { return *a < b; }
        bool operator()(const AdaptiveBytes& a, const AdaptiveBytes* b) const { return a < *b; }
    };
    std::set<const AdaptiveBytes*, PointKeyLess> dirty_points;
    ContactPairSet affected_pairs;
    std::set<std::size_t> possibly_unused_records;
    std::set<std::pair<int, std::size_t>> missing_records;
    std::map<std::pair<std::size_t, std::size_t>, ClosureRequirements> failing_pairs;
    std::vector<std::size_t> free_records, retired_records, free_rows;
    std::set<AdaptiveBytes> domains;
    std::map<AdaptiveBytes, PointGroup> points;
    PagedRecords<Record> records;
    struct RecordKeyLess
    {
        using is_transparent = void;
        bool operator()(const AdaptiveCellKey* a, const AdaptiveCellKey* b) const { return *a < *b; }
        bool operator()(const AdaptiveCellKey* a, const AdaptiveCellKey& b) const { return *a < b; }
        bool operator()(const AdaptiveCellKey& a, const AdaptiveCellKey* b) const { return a < *b; }
    };
    std::map<const AdaptiveCellKey*, std::size_t, RecordKeyLess> record_ids;
    std::map<AdaptiveBytes, std::vector<RemoteOccurrence>> remote_points;
    std::unique_ptr<ContactCoverIndex> index;
    std::vector<ClosedContactCover> covers;
    std::map<int, DirectoryNode> directory;
    std::map<int, std::vector<ClosedContactCover>> neighbors, previous_neighbors;
    Bounds local_cover;
    int ranks, rank, tree_width = 1;
    std::size_t page_reservation = 0, external_retained_bytes = 0, fork_overlap_bytes = 0;
    bool ready = false, evaluated = false;
    const char* diagnostic_phase = "construct";

    Impl(const AdaptiveProviderKernel& kernel_, Teuchos::RCP<const Teuchos::Comm<int>> comm_,
        AdaptiveClosureOptions options_, ContactHaloContext context_, ContactHaloBeforeSend before_send_)
        : kernel(kernel_), comm(std::move(comm_)), options(options_), context(context_),
          before_send(std::move(before_send_)), arena(std::make_shared<Budget>(Budget{options.byte_budget})),
          budget(*arena), deferred_owners(budget), trace_pool(std::in_place, budget), affected_pairs(budget),
          records(budget), ranks(comm.is_null() ? 0 : comm->getSize()), rank(comm.is_null() ? -1 : comm->getRank())
    {
        budget.add(sizeof(Impl) + sizeof(Budget) + 128 + sizeof(DistributedAdaptiveContactHalo));
    }
    void initialize_collective()
    {
        std::array<std::uint64_t, 10> semantics{context.epoch, context.topology, context.partition, context.round,
            context.transaction, options.byte_budget, options.contact_budget, options.tile_budget,
            std::bit_cast<std::uint64_t>(options.maximum_span_ratio), options.maximum_trace_jump};
        auto root = semantics;
        Teuchos::broadcast(*comm, 0, int(root.size()), root.data());
        collective(*comm,
            [&]
            {
                if (root != semantics || !options.byte_budget || !options.contact_budget || !options.tile_budget ||
                    !std::isfinite(options.maximum_span_ratio) || options.maximum_span_ratio < 1 ||
                    options.maximum_trace_jump != 1)
                    throw std::invalid_argument("Invalid or inconsistent spatial closure context/options.");
                while (tree_width < ranks)
                {
                    if (tree_width > std::numeric_limits<int>::max() / 2)
                        throw std::invalid_argument("Spatial directory rank extent overflows.");
                    tree_width *= 2;
                }
            });
    }
    ~Impl()
    {
        if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            std::fprintf(stderr,
                "ADAPTIVE_HALO rank=%d phase=%s ready=%d evaluated=%d owned=%zu remote=%zu points=%zu dirty_points=%zu "
                "pairs=%zu pair_bytes=%zu records_bytes=%zu index_bytes=%zu used=%zu peak=%zu cumulative=%zu\n",
                rank, diagnostic_phase, int(ready), int(evaluated), statistics.owned_records, statistics.remote_records,
                points.size(), dirty_points.size(), affected_pairs.size(), affected_pairs.storage_bytes(),
                records.storage_bytes(), index ? index->storage_bytes() : 0, budget.used, budget.peak,
                budget.cumulative);
        statistics.retained_bytes = budget.used;
        statistics.peak_bytes = budget.peak;
        statistics.cumulative_bytes = budget.cumulative;
        if (options.statistics)
        {
            auto& output = *options.statistics;
            output.cells = diagnostics.cells;
            output.contacts += diagnostics.contacts;
            output.geometry_records += diagnostics.geometry_records;
            output.trace_records += diagnostics.trace_records;
            output.transmitted_geometry_records += diagnostics.transmitted_geometry_records;
            output.decoded_geometry_records += diagnostics.decoded_geometry_records;
            output.unique_record_bytes += diagnostics.unique_record_bytes;
            output.replicated_record_bytes += diagnostics.replicated_record_bytes;
            output.decoded_geometry_bytes += diagnostics.decoded_geometry_bytes;
            output.cumulative_bytes += budget.cumulative;
            output.live_peak_bytes = std::max(output.live_peak_bytes, budget.peak);
            output.transient_geometry_peak_bytes =
                std::max(output.transient_geometry_peak_bytes, diagnostics.transient_geometry_peak_bytes);
            output.work += statistics.work;
        }
    }
    Writer writer(
        Kind kind, std::uint64_t query = 0, std::uint64_t page = 0, std::uint64_t total = 0, bool final = true)
    {
        Writer w;
        constexpr std::size_t initial = header_bytes + 8 + 6 * PLIC::ScalarPacketWords{}.size() * 8;
        if (page_reservation)
            throw std::invalid_argument("Overlapping spatial page reservations.");
        budget.add(initial);
        page_reservation = initial;
        w.data.reserve(initial);
        if (w.data.capacity() != initial)
            throw std::invalid_argument("Spatial header capacity differs from admission.");
        w.word(halo_magic);
        w.word(2);
        w.word(std::uint64_t(kind));
        w.word(context.epoch);
        w.word(context.topology);
        w.word(context.partition);
        w.word(context.round);
        w.word(context.transaction);
        w.word(rank);
        w.word(query);
        w.word(page);
        w.word(total);
        w.word(final);
        return w;
    }
    Frame read(Reader& r, Kind kind, int source)
    {
        if (r.word() != halo_magic || r.word() != 2 || r.word() != std::uint64_t(kind) || r.word() != context.epoch ||
            r.word() != context.topology || r.word() != context.partition || r.word() != context.round ||
            r.word() != context.transaction || r.word() != std::uint64_t(source))
            throw std::invalid_argument("Foreign or stale spatial closure page.");
        Frame frame{r.word(), r.word(), r.word(), false};
        const auto final = r.word();
        if (final > 1)
            throw std::invalid_argument("Invalid spatial closure completion flag.");
        frame.final = final;
        return frame;
    }
    bool any(bool local)
    {
        int value = local, result = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &value, &result);
        return result;
    }
    std::size_t credit()
    {
        std::uint64_t available = budget.limit - budget.used, common = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &available, &common);
        // Reserve simultaneous sender, receiver, framing, decode and transport
        // scratch. Pages are bounded before packet vectors are constructed.
        return std::min<std::size_t>(256 * 1024, common / 16);
    }
    OwnerExchange exchange(int destination, Writer&& w)
    {
        std::vector<OwnerPacket> outgoing;
        std::size_t outgoing_charge = 0;
        collective(*comm,
            [&]
            {
                if (destination >= 0)
                {
                    Reader header{w.data};
                    (void) header.word();
                    (void) header.word();
                    const auto kind = static_cast<Kind>(header.word());
                    outgoing_charge = sizeof(OwnerPacket) + w.data.capacity();
                    if (page_reservation)
                    {
                        if (page_reservation != w.data.capacity())
                            throw std::invalid_argument("Spatial page reservation disagreement.");
                        budget.add(sizeof(OwnerPacket));
                    }
                    else
                        budget.add(outgoing_charge);
                    outgoing.reserve(1);
                    outgoing.push_back({destination, std::move(w.data)});
                    if (before_send)
                    {
                        before_send(context, kind, outgoing);
                        std::size_t actual = outgoing.capacity() * sizeof(OwnerPacket);
                        for (const auto& packet : outgoing)
                        {
                            if (packet.bytes.capacity() > std::numeric_limits<std::size_t>::max() - actual)
                                throw std::invalid_argument("Spatial packet capacity overflows.");
                            actual += packet.bytes.capacity();
                        }
                        if (actual > outgoing_charge)
                            budget.add(actual - outgoing_charge);
                        else
                            budget.used -= outgoing_charge - actual;
                        outgoing_charge = actual;
                    }
                    ++statistics.pages;
                }
            });
        auto incoming = consume_budgeted_owner_packets(std::move(outgoing), comm, budget, outgoing_charge);
        page_reservation = 0;
        return incoming;
    }
    void retire(OwnerExchange& exchange) { retire_budgeted_owner_packets(exchange, budget); }
    void admit_writer(Writer& w, std::size_t capacity)
    {
        if (capacity > budget.limit - budget.used)
            throw std::invalid_argument("Spatial closure page exceeds available byte credit.");
        const auto previous = page_reservation;
        if (!previous)
            throw std::invalid_argument("Missing spatial page header reservation.");
        if (capacity < previous)
            throw std::invalid_argument("Spatial page credit cannot hold header.");
        budget.add(capacity);
        page_reservation = capacity;
        w.data.reserve(capacity);
        if (w.data.capacity() != capacity)
            throw std::invalid_argument("Spatial closure page capacity differs from admission.");
        budget.used -= previous;
    }
    std::size_t add_record(const AdaptiveCellKey& key, std::uint64_t revision, int owner, std::span<const char> domain)
    {
        if (const auto found = record_ids.find(key); found != record_ids.end())
        {
            auto& record = records[found->second];
            if (record.owner == owner && record.revision == revision && std::ranges::equal(record.domain, domain))
                return found->second;
            if (record.owner == rank || (owner != rank && record.remote_refs))
                throw std::invalid_argument("Conflicting spatial closure cell owner/revision/domain.");
            discard_record(found->second);
        }
        budget.add(sizeof(decltype(record_ids)::value_type) + map_links + key_capacity(key) + domain.size());
        std::size_t id;
        if (free_records.empty())
        {
            id = records.size();
            records.emplace_back();
        }
        else
        {
            id = free_records.back();
            free_records.pop_back();
        }
        auto& record = records[id];
        record.key = key;
        record.domain.assign(domain.begin(), domain.end());
        record.revision = revision;
        record.owner = owner;
        record_ids.emplace(&record.key, id);
        if (owner != rank)
        {
            budget.add(sizeof(decltype(missing_records)::value_type) + map_links);
            missing_records.emplace(owner, id);
        }
        return id;
    }
    void build_directory()
    {
        Bounds accumulated = local_cover;
        for (int stride = 1; stride < tree_width; stride *= 2)
        {
            int destination = -1;
            Writer outgoing;
            collective(*comm,
                [&]
                {
                    if (rank % (2 * stride) == stride)
                    {
                        destination = rank - stride;
                        outgoing = writer(Kind::Cover, stride);
                        write_bounds(outgoing, accumulated);
                    }
                });
            auto incoming = exchange(destination, std::move(outgoing));
            collective(*comm,
                [&]
                {
                    if (rank % (2 * stride) == 0)
                    {
                        Bounds right;
                        if (rank + stride < ranks)
                        {
                            if (incoming.packets.size() != 1 || incoming.packets.front().peer != rank + stride)
                                throw std::invalid_argument("Missing spatial directory child cover.");
                            Reader r{incoming.packets.front().bytes};
                            const auto frame = read(r, Kind::Cover, rank + stride);
                            if (frame.query != std::uint64_t(stride) || frame.page || frame.total || !frame.final)
                                throw std::invalid_argument("Invalid spatial directory child completion.");
                            right = read_bounds(r);
                            r.done();
                        }
                        else if (!incoming.packets.empty())
                            throw std::invalid_argument("Unexpected spatial directory child.");
                        budget.add(sizeof(decltype(directory)::value_type) + map_links);
                        directory.emplace(stride, DirectoryNode{accumulated, right});
                        if (right.present)
                            include(accumulated, right.box);
                    }
                    else if (!incoming.packets.empty())
                        throw std::invalid_argument("Foreign spatial directory cover delivery.");
                });
            retire(incoming);
        }
    }
    void discover_neighbors()
    {
        // One requester's metadata envelope traverses the distributed tree at
        // a time. Only overlapping owner fragments return to that requester.
        for (int requester = 0; requester < ranks; ++requester)
        {
            Writer outgoing;
            int destination = -1;
            collective(*comm,
                [&]
                {
                    if (rank == requester)
                    {
                        destination = 0;
                        outgoing = writer(Kind::DirectoryQuery, requester);
                        write_bounds(outgoing, local_cover);
                    }
                });
            auto incoming = exchange(destination, std::move(outgoing));
            Bounds query;
            bool active = false;
            collective(*comm,
                [&]
                {
                    if (rank == 0)
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != requester)
                            throw std::invalid_argument("Missing spatial directory root query.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::DirectoryQuery, requester);
                        if (frame.query != std::uint64_t(requester) || !frame.final)
                            throw std::invalid_argument("Invalid spatial directory root query.");
                        query = read_bounds(r);
                        r.done();
                        active = query.present;
                    }
                    else if (!incoming.packets.empty())
                        throw std::invalid_argument("Foreign spatial directory root query.");
                });
            retire(incoming);
            for (int stride = tree_width / 2; stride; stride /= 2)
            {
                outgoing = Writer{};
                destination = -1;
                bool left_active = false;
                collective(*comm,
                    [&]
                    {
                        if (rank % (2 * stride) == 0)
                        {
                            const auto& node = directory.at(stride);
                            left_active = active && overlaps(query, node.left);
                            if (rank + stride < ranks)
                            {
                                destination = rank + stride;
                                outgoing = writer(Kind::DirectoryQuery, requester, stride);
                                write_bounds(outgoing, active && overlaps(query, node.right) ? query : Bounds{});
                            }
                        }
                    });
                incoming = exchange(destination, std::move(outgoing));
                collective(*comm,
                    [&]
                    {
                        if (rank % (2 * stride) == 0)
                            active = left_active;
                        else if (rank % (2 * stride) == stride)
                        {
                            active = false;
                            if (incoming.packets.size() != 1 || incoming.packets.front().peer != rank - stride)
                                throw std::invalid_argument("Missing spatial directory branch terminal response.");
                            if (!incoming.packets.empty())
                            {
                                if (incoming.packets.size() != 1 || incoming.packets.front().peer != rank - stride)
                                    throw std::invalid_argument("Foreign spatial directory branch.");
                                Reader r{incoming.packets.front().bytes};
                                const auto frame = read(r, Kind::DirectoryQuery, rank - stride);
                                if (frame.query != std::uint64_t(requester) || frame.page != std::uint64_t(stride) ||
                                    !frame.final)
                                    throw std::invalid_argument("Invalid spatial directory branch completion.");
                                query = read_bounds(r);
                                r.done();
                                active = query.present;
                            }
                        }
                        else if (!incoming.packets.empty())
                            throw std::invalid_argument("Unexpected spatial directory branch query.");
                    });
                retire(incoming);
            }
            // Cyclic service bounds both fan-in and the aggregate receive page.
            for (int owner = 0; owner < ranks; ++owner)
            {
                std::size_t cursor = 0, page = 0;
                bool done = !(rank == owner && active && owner != requester);
                bool received_final = owner == requester;
                std::uint64_t expected_cover_page = 0;
                int subscribed = rank == owner && active && owner != requester;
                Teuchos::broadcast(*comm, owner, 1, &subscribed);
                while (any(!done))
                {
                    const auto cap = credit();
                    outgoing = Writer{};
                    destination = -1;
                    collective(*comm,
                        [&]
                        {
                            if (!done)
                            {
                                if (cap <= header_bytes + 8)
                                    throw std::invalid_argument("Insufficient spatial cover page credit.");
                                destination = requester;
                                outgoing = writer(Kind::LeafCovers, requester, page++);
                                admit_writer(outgoing, cap);
                                outgoing.word(0);
                                std::size_t count = 0;
                                while (cursor < covers.size())
                                {
                                    const auto& cover = covers[cursor];
                                    if (!closed_contact_overlap(query.box, cover.bounds))
                                    {
                                        ++cursor;
                                        continue;
                                    }
                                    const auto need =
                                        8 + cover.domain.size() + 8 + 6 * PLIC::ScalarPacketWords{}.size() * 8;
                                    if (need > cap - outgoing.data.size())
                                    {
                                        if (!count)
                                            throw std::invalid_argument("A spatial cover exceeds page credit.");
                                        break;
                                    }
                                    outgoing.bytes(cover.domain);
                                    write_bounds(outgoing, {cover.bounds, true});
                                    ++cursor;
                                    ++count;
                                }
                                done = cursor == covers.size();
                                replace_word(outgoing.data, header_bytes - 8, done);
                                replace_word(outgoing.data, header_bytes, count);
                            }
                        });
                    incoming = exchange(destination, std::move(outgoing));
                    collective(*comm,
                        [&]
                        {
                            if (rank == requester && !incoming.packets.empty())
                            {
                                if (received_final || incoming.packets.size() != 1 ||
                                    incoming.packets.front().peer != owner)
                                    throw std::invalid_argument("Duplicate or foreign owner-cover page.");
                                Reader r{incoming.packets.front().bytes};
                                const auto frame = read(r, Kind::LeafCovers, owner);
                                if (frame.page != expected_cover_page++)
                                    throw std::invalid_argument("Missing or duplicate owner-cover page.");
                                auto found = neighbors.find(owner);
                                if (found == neighbors.end())
                                {
                                    budget.add(sizeof(decltype(neighbors)::value_type) + map_links);
                                    found = neighbors.try_emplace(owner).first;
                                }
                                auto n = r.word();
                                if (frame.query != std::uint64_t(requester) || n > r.data.size() / 16)
                                    throw std::invalid_argument("Invalid owner-cover response count.");
                                for (std::uint64_t i = 0; i < n; ++i)
                                {
                                    const auto domain = r.bytes_view();
                                    const auto bounds = read_bounds(r);
                                    if (domain.empty() || !bounds.present || !overlaps(local_cover, bounds))
                                        throw std::invalid_argument("Foreign owner-cover response.");
                                    budget.add(domain.size());
                                    budget.append_capacity(found->second);
                                    found->second.push_back(
                                        {AdaptiveBytes(domain.begin(), domain.end()), bounds.box, owner, 0});
                                }
                                r.done();
                                received_final = frame.final;
                            }
                            else if (!incoming.packets.empty())
                                throw std::invalid_argument("Foreign owner-cover page delivery.");
                        });
                    retire(incoming);
                }
                collective(*comm,
                    [&]
                    {
                        if (rank == requester && subscribed && !received_final)
                            throw std::invalid_argument("Missing owner-cover terminal response.");
                    });
                // A non-overlapping leaf has no subscription and needs no point
                // queries. Completion of tree traversal is synchronous above.
            }
        }
    }
    bool candidate(int owner, const PointGroup& group) const
    {
        const auto found = neighbors.find(owner);
        if (found == neighbors.end())
            return false;
        for (const auto& cover : found->second)
            if (std::ranges::equal(*group.domain, cover.domain) && contains(cover.bounds, group.coordinate))
                return true;
        return false;
    }

    [[nodiscard]] std::size_t metadata_bytes() const;
    void install_row(IncidenceRowHandle, const AdaptiveLeaf&, bool row_precharged);
    void retire_row(std::size_t);
    void touch_point(const AdaptiveBytes&);
    void gather_pairs(const AdaptiveBytes&);
    void notify_dirty_points();
    void refresh_covers();
    void attach_record(std::size_t, ContactRecord&&, std::size_t);
    void discard_record(std::size_t);
    void remove_remote_points(const AdaptiveBytes&);
    void discover_points();
    void fetch_records();
    std::vector<AdaptiveMark> evaluate(bool retain_charge = false);
};

namespace
{
std::size_t row_storage(const ContactIncidenceRow& row)
{
    return sizeof(ContactIncidenceRow) + 128 + key_capacity(row.key) + row.domain.capacity() +
           row.points.storage_bytes();
}
} // namespace

std::size_t DistributedAdaptiveContactHalo::Impl::metadata_bytes() const
{
    std::size_t bytes =
        sizeof(Impl) + sizeof(Budget) + 128 + sizeof(DistributedAdaptiveContactHalo) +
        rows.capacity() * sizeof(IncidenceRowHandle) + borrowed_rows.capacity() * sizeof(std::size_t) +
        records.storage_bytes() +
        (free_rows.capacity() + free_records.capacity() + retired_records.capacity()) * sizeof(std::size_t) +
        identities.metadata_bytes() + deferred_owners.metadata_bytes();
    for (const auto& record : records)
        bytes += key_capacity(record.key) + record.domain.capacity();
    for (const auto& [key, id] : record_ids)
        bytes += sizeof(decltype(record_ids)::value_type) + map_links;
    for (const auto& domain : domains)
        bytes += sizeof(AdaptiveBytes) + map_links + domain.capacity();
    for (const auto& [key, point] : points)
        bytes += sizeof(decltype(points)::value_type) + map_links + key.capacity() +
                 point.local.capacity() * sizeof(LocalOccurrence);
    for (const auto& [key, remote] : remote_points)
        bytes += sizeof(decltype(remote_points)::value_type) + map_links + key.capacity() +
                 remote.capacity() * sizeof(RemoteOccurrence);
    bytes += covers.capacity() * sizeof(ClosedContactCover);
    for (const auto& cover : covers)
        bytes += cover.domain.capacity();
    for (const auto* map : {&neighbors, &previous_neighbors})
        for (const auto& [rank, values] : *map)
        {
            bytes +=
                sizeof(decltype(neighbors)::value_type) + map_links + values.capacity() * sizeof(ClosedContactCover);
            for (const auto& cover : values)
                bytes += cover.domain.capacity();
        }
    bytes += directory.size() * (sizeof(decltype(directory)::value_type) + map_links) +
             failing_pairs.size() * (sizeof(decltype(failing_pairs)::value_type) + map_links) +
             affected_pairs.storage_bytes() + possibly_unused_records.size() * (sizeof(std::size_t) + map_links) +
             missing_records.size() * (sizeof(decltype(missing_records)::value_type) + map_links);
    bytes += dirty_points.size() * (sizeof(const AdaptiveBytes*) + map_links);
    if (index)
        bytes += index->storage_bytes();
    if (incidence)
        bytes += incidence->metadata_bytes();
    if (trace_pool)
        bytes += trace_pool->metadata_bytes();
    return bytes;
}

void DistributedAdaptiveContactHalo::Impl::attach_record(std::size_t id, ContactRecord&& value, std::size_t bytes)
{
    const auto transferred = trace_pool->adopt(value);
    if (transferred > bytes)
        throw std::invalid_argument("Contact trace transfer exceeds record admission.");
    bytes -= transferred;
    budget.add(128);
    bytes += 128;
    auto keeper = arena;
    records[id].contact = std::shared_ptr<const ContactRecord>(new ContactRecord(std::move(value)),
        [keeper, bytes](const ContactRecord* record)
        {
            delete record;
            keeper->used -= bytes;
        });
    records[id].body_bytes = bytes;
}
void DistributedAdaptiveContactHalo::Impl::touch_point(const AdaptiveBytes& key)
{
    if (dirty_points.contains(key))
        return;
    const auto found = points.find(key);
    if (found == points.end())
        throw std::invalid_argument("Dirty contact point has no stable key owner.");
    budget.add(sizeof(const AdaptiveBytes*) + map_links);
    dirty_points.insert(&found->first);
    gather_pairs(found->first);
}
void DistributedAdaptiveContactHalo::Impl::gather_pairs(const AdaptiveBytes& key)
{
    const auto found = points.find(key);
    if (found == points.end())
        return;
    const auto& local = found->second.local;
    auto add = [&](std::size_t a, std::size_t b)
    {
        if (a == b)
            throw std::invalid_argument("Duplicate cell in complete normalized contact fan.");
        const auto ids = std::minmax(a, b);
        const std::pair<std::size_t, std::size_t> pair{ids.first, ids.second};
        (void) affected_pairs.insert(pair);
    };
    for (std::size_t i = 0; i < local.size(); ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
            add(local[i].record, local[j].record);
        if (const auto remote = remote_points.find(key); remote != remote_points.end())
            for (const auto& other : remote->second)
                add(local[i].record, other.record);
    }
}
void DistributedAdaptiveContactHalo::Impl::remove_remote_points(const AdaptiveBytes& key)
{
    const auto found = remote_points.find(key);
    if (found == remote_points.end())
        return;
    for (const auto& item : found->second)
    {
        auto& record = records[item.record];
        if (!record.remote_refs)
            throw std::invalid_argument("Remote incidence reference underflow.");
        if (!--record.remote_refs && !possibly_unused_records.contains(item.record))
        {
            budget.add(sizeof(std::size_t) + map_links);
            possibly_unused_records.insert(item.record);
        }
    }
    const auto bytes = sizeof(decltype(remote_points)::value_type) + map_links + found->first.capacity() +
                       found->second.capacity() * sizeof(RemoteOccurrence);
    remote_points.erase(found);
    budget.used -= bytes;
}
void DistributedAdaptiveContactHalo::Impl::discard_record(std::size_t id)
{
    auto& record = records[id];
    if (record.owner < 0)
        return;
    const auto found = record_ids.find(record.key);
    if (found != record_ids.end() && found->second == id)
    {
        const auto bytes = sizeof(decltype(record_ids)::value_type) + map_links;
        record_ids.erase(found);
        budget.used -= bytes;
    }
    // Tombstone slots are not reused until old pair obligations have retired.
    if (record.owner != rank && record.contact)
        --statistics.remote_records;
    if (missing_records.erase({record.owner, id}))
        budget.used -= sizeof(decltype(missing_records)::value_type) + map_links;
    record.owner = -1;
    budget.append_capacity(retired_records);
    retired_records.push_back(id);
}
void DistributedAdaptiveContactHalo::Impl::install_row(
    IncidenceRowHandle handle, const AdaptiveLeaf& leaf, bool row_precharged)
{
    const auto& row = *handle;
    if (row.domain.empty() || row.points.empty() || row.key != leaf.key || row.revision != leaf.revision ||
        row.domain != leaf.domain)
        throw std::invalid_argument("Spatial incidence row disagrees with its source owner.");
    std::size_t row_index;
    if (free_rows.empty())
    {
        budget.append_capacity(rows);
        budget.append_capacity(borrowed_rows);
        row_index = rows.size();
        rows.push_back({});
        borrowed_rows.push_back(0);
    }
    else
    {
        row_index = free_rows.back();
        free_rows.pop_back();
    }
    if (!row_precharged)
    {
        borrowed_rows[row_index] = row_storage(row);
        budget.add(borrowed_rows[row_index]);
    }
    rows[row_index] = std::move(handle);
    const auto id = add_record(row.key, row.revision, rank, row.domain);
    records[id].row = row_index;
    if (!records[id].contact)
    {
        const auto before = budget.used, identities_before = identities.admitted_bytes();
        auto record = prepare_contact(
            leaf, kernel, budget, identities, diagnostics.trace_records, diagnostics.transient_geometry_peak_bytes);
        record.identity = id;
        attach_record(id, std::move(record), budget.used - before - (identities.admitted_bytes() - identities_before));
        ++statistics.owned_records;
        ++diagnostics.geometry_records;
    }
    auto domain = domains.find(row.domain);
    if (domain == domains.end())
    {
        budget.add(sizeof(decltype(domains)::value_type) + map_links + row.domain.size());
        domain = domains.insert(row.domain).first;
    }
    for (std::size_t point_index = 0; point_index < row.points.size(); ++point_index)
    {
        const auto& point = row.points[point_index];
        const auto maximum = 16 + row.domain.size() + PLIC::canonical_key_size(point.normalized);
        if (maximum > (budget.limit - budget.used) / 4)
            throw std::invalid_argument("Spatial point key exceeds live byte budget.");
        budget.add(4 * maximum);
        auto key = point_key(row.domain, point.normalized);
        auto position = points.find(key);
        if (position == points.end())
        {
            const auto capacity = key.capacity();
            budget.add(sizeof(decltype(points)::value_type) + map_links);
            auto coordinate = coordinates(point.normalized);
            for (const auto scalar : coordinate)
                if (!std::isfinite(scalar))
                    throw std::invalid_argument("Nonfinite exact contact coordinate.");
            position = points.emplace(std::move(key), PointGroup{&*domain, coordinate, {}}).first;
            budget.used -= 4 * maximum - capacity;
        }
        else
        {
            AdaptiveBytes().swap(key);
            budget.used -= 4 * maximum;
        }
        touch_point(position->first);
        for (const auto& occurrence : position->second.local)
            if (occurrence.record == id)
                throw std::invalid_argument("Duplicate cell incidence at a normalized closure point.");
        auto& group = position->second;
        if (group.local.empty() && ready)
        {
            if (!index)
                index = std::make_unique<ContactCoverIndex>(budget, rank);
            group.cover_leaf = index->insert(&group.coordinate, group.domain);
        }
        budget.append_capacity(group.local);
        group.local.push_back({id, row_index, point_index});
        ++diagnostics.contacts;
    }
}
void DistributedAdaptiveContactHalo::Impl::retire_row(std::size_t row_index)
{
    const auto& row = *rows.at(row_index);
    const auto record = record_ids.find(row.key)->second;
    for (const auto& point : row.points)
    {
        const auto maximum = 16 + row.domain.size() + PLIC::canonical_key_size(point.normalized);
        if (maximum > (budget.limit - budget.used) / 4)
            throw std::invalid_argument("Spatial retirement key exceeds live budget.");
        budget.add(4 * maximum);
        {
            auto key = point_key(row.domain, point.normalized);
            touch_point(key);
            auto& group = points.at(key);
            std::erase_if(group.local, [&](const auto& item) { return item.row == row_index; });
            if (group.local.empty())
            {
                if (!index || group.cover_leaf == std::numeric_limits<std::size_t>::max())
                    throw std::invalid_argument("Missing contact point cover leaf.");
                index->erase(group.cover_leaf);
                group.cover_leaf = std::numeric_limits<std::size_t>::max();
            }
        }
        budget.used -= 4 * maximum;
    }
    records[record].row = std::numeric_limits<std::size_t>::max();
    const auto credit = borrowed_rows[row_index];
    if (credit && (rows[row_index].use_count() > 1 || deferred_owners.contains(rows[row_index])))
    {
        if (!deferred_owners.adopt(rows[row_index], credit))
            throw std::invalid_argument("Duplicate borrowed incidence-row admission.");
        borrowed_rows[row_index] = 0;
    }
    rows[row_index].reset();
    budget.used -= borrowed_rows[row_index];
    borrowed_rows[row_index] = 0;
    budget.append_capacity(free_rows);
    free_rows.push_back(row_index);
}
void DistributedAdaptiveContactHalo::Impl::refresh_covers()
{
    std::size_t bytes = covers.capacity() * sizeof(ClosedContactCover);
    for (const auto& cover : covers)
        bytes += cover.domain.capacity();
    std::vector<ClosedContactCover>().swap(covers);
    budget.used -= bytes;
    local_cover = {};
    if (!ready && !points.empty())
    {
        // Bootstrap visits the completed distinct point catalogue once. Later
        // updates maintain leaf handles at the changed point-group boundary.
        index = std::make_unique<ContactCoverIndex>(budget, rank);
        index->reserve(points.size());
        for (auto& [key, point] : points)
        {
            if (point.local.empty())
                throw std::invalid_argument("Empty bootstrap point group.");
            point.cover_leaf = index->insert(&point.coordinate, point.domain);
        }
    }
    if (index && !index->point_count())
        index.reset();
    if (!index)
        return;
    if (const auto bounds = index->aggregate_bounds())
        local_cover = {*bounds, true};
    covers = index->ownership_frontier_covers(8);
}

void DistributedAdaptiveContactHalo::Impl::notify_dirty_points()
{
    auto relevant = [&](int owner, const AdaptiveBytes& key)
    {
        const auto point = points.find(key);
        if (point == points.end())
            return false;
        if (candidate(owner, point->second))
            return true;
        const auto previous = previous_neighbors.find(owner);
        if (previous != previous_neighbors.end())
            for (const auto& cover : previous->second)
                if (std::ranges::equal(*point->second.domain, cover.domain) &&
                    contains(cover.bounds, point->second.coordinate))
                    return true;
        return false;
    };
    // Freeze the outgoing dirty set. Keys learned from peers are consumed by
    // this owner's queries, never recursively echoed in the same round.
    std::vector<const AdaptiveBytes*> outgoing_keys;
    collective(*comm,
        [&]
        {
            budget.add(dirty_points.size() * sizeof(const AdaptiveBytes*));
            outgoing_keys.reserve(dirty_points.size());
            for (const auto* key : dirty_points)
                outgoing_keys.push_back(key);
        });
    for (int offset = 1; offset < ranks; ++offset)
    {
        const int destination = (rank + offset) % ranks;
        const int source = (rank + ranks - offset) % ranks;
        std::size_t cursor = 0;
        std::uint64_t page = 0, expected_page = 0;
        bool sent_final = false, received_final = false;
        while (any(!sent_final))
        {
            const auto cap = credit();
            Writer outgoing;
            int peer = -1;
            collective(*comm,
                [&]
                {
                    if (sent_final)
                        return;
                    if (cap <= header_bytes + 8)
                        throw std::invalid_argument("Dirty contact page has no byte credit.");
                    peer = destination;
                    outgoing = writer(Kind::DirtyPoints, 0, page++);
                    admit_writer(outgoing, cap);
                    outgoing.word(0);
                    std::size_t count = 0;
                    while (cursor < outgoing_keys.size())
                    {
                        const auto& key = *outgoing_keys[cursor];
                        if (!relevant(destination, key))
                        {
                            ++cursor;
                            continue;
                        }
                        if (key.size() + 8 > cap - outgoing.data.size())
                        {
                            if (!count)
                                throw std::invalid_argument("Dirty point key exceeds page credit.");
                            break;
                        }
                        outgoing.bytes(key);
                        ++count;
                        ++cursor;
                    }
                    sent_final = cursor == outgoing_keys.size();
                    replace_word(outgoing.data, header_bytes, count);
                    replace_word(outgoing.data, header_bytes - 8, sent_final);
                });
            auto incoming = exchange(peer, std::move(outgoing));
            collective(*comm,
                [&]
                {
                    if (incoming.packets.empty())
                        return;
                    if (received_final || incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                        throw std::invalid_argument("Duplicate or foreign dirty point page.");
                    Reader r{incoming.packets.front().bytes};
                    const auto frame = read(r, Kind::DirtyPoints, source);
                    if (frame.query || frame.total || frame.page != expected_page++)
                        throw std::invalid_argument("Missing dirty point page.");
                    const auto count = r.word();
                    if (count > r.data.size() / 8)
                        throw std::invalid_argument("Invalid dirty point count.");
                    for (std::uint64_t i = 0; i < count; ++i)
                    {
                        const auto field = r.bytes_view();
                        budget.add(field.size());
                        {
                            const AdaptiveBytes key(field.begin(), field.end());
                            if (points.contains(key))
                                touch_point(key);
                        }
                        budget.used -= field.size();
                    }
                    r.done();
                    received_final = frame.final;
                });
            retire(incoming);
        }
        collective(*comm,
            [&]
            {
                if (!received_final)
                    throw std::invalid_argument("Missing dirty point terminal reply.");
            });
    }
    const auto bytes = outgoing_keys.capacity() * sizeof(const AdaptiveBytes*);
    std::vector<const AdaptiveBytes*>().swap(outgoing_keys);
    budget.used -= bytes;
}

void DistributedAdaptiveContactHalo::Impl::discover_points()
{
    using PointIterator = decltype(points)::const_iterator;
    struct Pending
    {
        PointIterator point;
        std::uint64_t total = 0, next = 0;
        bool started = false, done = false;
    };
    struct Query
    {
        std::uint64_t id;
        const PointGroup* group;
    };
    for (int offset = 1; offset < ranks; ++offset)
    {
        const int destination = (rank + offset) % ranks;
        const int source = (rank + ranks - offset) % ranks;
        auto cursor = dirty_points.cbegin();
        std::uint64_t next_query = 0, expected_query = 0, query_page = 0, expected_page = 0;
        while (true)
        {
            while (cursor != dirty_points.cend() &&
                   (points.at(**cursor).local.empty() || !candidate(destination, points.at(**cursor))))
                ++cursor;
            if (!any(cursor != dirty_points.cend()))
                break;
            const auto cap = credit();
            std::vector<Pending> pending;
            Writer outgoing;
            int peer = -1;
            const auto first_query = next_query;
            collective(*comm,
                [&]
                {
                    if (cursor != dirty_points.cend())
                    {
                        if (cap <= header_bytes + 8)
                            throw std::invalid_argument("Insufficient exact point query page credit.");
                        peer = destination;
                        outgoing = writer(Kind::Points, first_query, query_page++);
                        ++statistics.point_query_pages;
                        admit_writer(outgoing, cap);
                        outgoing.word(0);
                        std::size_t count = 0;
                        while (cursor != dirty_points.cend())
                        {
                            if (points.at(**cursor).local.empty() || !candidate(destination, points.at(**cursor)))
                            {
                                ++cursor;
                                continue;
                            }
                            const auto need = 16 + (*cursor)->size();
                            if (need > cap - outgoing.data.size())
                            {
                                if (!count)
                                    throw std::invalid_argument("An exact point key exceeds page credit.");
                                break;
                            }
                            outgoing.word(next_query++);
                            outgoing.bytes(**cursor);
                            budget.append_capacity(pending);
                            pending.push_back({points.find(**cursor), 0, 0, false, false});
                            ++statistics.exact_point_queries;
                            ++count;
                            ++cursor;
                        }
                        replace_word(outgoing.data, header_bytes, count);
                        statistics.query_bytes += outgoing.data.size();
                    }
                });
            auto incoming = exchange(peer, std::move(outgoing));
            std::vector<Query> queries;
            collective(*comm,
                [&]
                {
                    if (!incoming.packets.empty())
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                            throw std::invalid_argument("Foreign exact point query page.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::Points, source);
                        const auto n = r.word();
                        if (frame.query != expected_query || frame.page != expected_page++ || !frame.final || !n ||
                            n > r.data.size() / 16)
                            throw std::invalid_argument("Missing, duplicate or invalid exact point query page.");
                        for (std::uint64_t i = 0; i < n; ++i)
                        {
                            const auto id = r.word();
                            if (id != expected_query++)
                                throw std::invalid_argument("Nonsequential exact point query identity.");
                            const auto bytes = r.bytes_view();
                            budget.add(bytes.size());
                            AdaptiveBytes key(bytes.begin(), bytes.end());
                            if (key.capacity() > bytes.size())
                                budget.add(key.capacity() - bytes.size());
                            const auto found = points.find(key);
                            budget.append_capacity(queries);
                            queries.push_back({id, found == points.end() ? nullptr : &found->second});
                            const auto key_capacity = key.capacity();
                            AdaptiveBytes().swap(key);
                            budget.used -= key_capacity;
                        }
                        r.done();
                    }
                });
            // Queries are now pointers into immutable owner-local reverse rows;
            // encoded query pages can be retired before producing any fan page.
            retire(incoming);
            std::size_t query_cursor = 0, occurrence_cursor = 0;
            std::uint64_t response_page = 0, expected_response_page = 0;
            bool received_final = pending.empty();
            while (any(query_cursor < queries.size()))
            {
                const auto reply_cap = credit();
                outgoing = Writer{};
                peer = -1;
                collective(*comm,
                    [&]
                    {
                        if (query_cursor < queries.size())
                        {
                            if (reply_cap <= header_bytes + 8)
                                throw std::invalid_argument("Insufficient exact contact fan page credit.");
                            peer = source;
                            outgoing = writer(Kind::Fan, 0, response_page++);
                            admit_writer(outgoing, reply_cap);
                            outgoing.word(0);
                            std::size_t count = 0;
                            while (query_cursor < queries.size())
                            {
                                const auto& query = queries[query_cursor];
                                const auto total = query.group ? query.group->local.size() : 0;
                                std::size_t need = 32;
                                if (total)
                                {
                                    const auto& occurrence = query.group->local[occurrence_cursor];
                                    const auto& record = records[occurrence.record];
                                    need += 24 + key_wire_size(record.key) +
                                            PLIC::encoded_identity_size(
                                                rows[occurrence.row]->points[occurrence.point].physical);
                                }
                                if (need > reply_cap - outgoing.data.size())
                                {
                                    if (!count)
                                        throw std::invalid_argument("An exact contact fan item exceeds page credit.");
                                    break;
                                }
                                outgoing.word(query.id);
                                outgoing.word(total);
                                outgoing.word(occurrence_cursor);
                                outgoing.word(total ? 1 : 0);
                                if (total)
                                {
                                    const auto& occurrence = query.group->local[occurrence_cursor];
                                    const auto& record = records[occurrence.record];
                                    const auto& physical = rows[occurrence.row]->points[occurrence.point].physical;
                                    const auto key_size = key_wire_size(record.key);
                                    const auto point_size = PLIC::encoded_identity_size(physical);
                                    if (key_size > (budget.limit - budget.used) / 3 ||
                                        point_size > (budget.limit - budget.used - 3 * key_size) / 4)
                                        throw std::invalid_argument("Spatial fan encoding exceeds live byte budget.");
                                    const auto scratch = 3 * key_size + 4 * point_size;
                                    budget.add(scratch);
                                    {
                                        auto key = encode_adaptive_key(record.key);
                                        outgoing.bytes(key);
                                    }
                                    outgoing.word(record.revision);
                                    {
                                        auto point = PLIC::encode_identity(physical);
                                        outgoing.bytes(point);
                                    }
                                    budget.used -= scratch;
                                }
                                ++count;
                                if (!total || ++occurrence_cursor == total)
                                {
                                    occurrence_cursor = 0;
                                    ++query_cursor;
                                }
                            }
                            replace_word(outgoing.data, header_bytes, count);
                            replace_word(outgoing.data, header_bytes - 8, query_cursor == queries.size());
                            statistics.query_bytes += outgoing.data.size();
                        }
                    });
                incoming = exchange(peer, std::move(outgoing));
                collective(*comm,
                    [&]
                    {
                        if (!incoming.packets.empty())
                        {
                            if (received_final || incoming.packets.size() != 1 ||
                                incoming.packets.front().peer != destination)
                                throw std::invalid_argument("Foreign or duplicate exact contact fan page.");
                            Reader r{incoming.packets.front().bytes};
                            const auto frame = read(r, Kind::Fan, destination);
                            if (frame.query || frame.page != expected_response_page++)
                                throw std::invalid_argument("Missing exact contact fan page.");
                            const auto n = r.word();
                            if (!n || n > r.data.size() / 32)
                                throw std::invalid_argument("Invalid exact contact fan page count.");
                            for (std::uint64_t i = 0; i < n; ++i)
                            {
                                const auto id = r.word(), total = r.word(), index = r.word(), present = r.word();
                                if (id < first_query || id - first_query >= pending.size() || present > 1)
                                    throw std::invalid_argument("Foreign exact contact fan query.");
                                auto& request = pending[id - first_query];
                                if (request.done || index != request.next || (!total && present) ||
                                    (total && !present) || (request.started && total != request.total))
                                    throw std::invalid_argument("Duplicate or incomplete exact contact fan.");
                                request.started = true;
                                request.total = total;
                                if (!present)
                                {
                                    request.done = true;
                                    ++statistics.empty_point_replies;
                                    continue;
                                }
                                if (index >= total)
                                    throw std::invalid_argument("Exact contact fan offset exceeds completion count.");
                                const auto encoded_key = r.bytes_view();
                                auto key = decode_bounded_key(encoded_key, budget);
                                const auto revision = r.word();
                                const auto encoded_point = r.bytes_view();
                                auto physical = PLIC::decode_identity<ContactPointIdentity>(
                                    encoded_point, budget.limit - budget.used);
                                const auto record =
                                    add_record(key, revision, destination, *request.point->second.domain);
                                auto found = remote_points.find(request.point->first);
                                if (found == remote_points.end())
                                {
                                    budget.add(sizeof(decltype(remote_points)::value_type) + map_links +
                                               request.point->first.size());
                                    found = remote_points.try_emplace(request.point->first).first;
                                }
                                for (const auto& existing : found->second)
                                    if (existing.record == record)
                                        throw std::invalid_argument(
                                            "Duplicate cell at an exact normalized contact point.");
                                // A key cannot be simultaneously local and remote.
                                if (records[record].owner == rank)
                                    throw std::invalid_argument("Remote contact fan claims a local cell.");
                                budget.append_capacity(found->second);
                                found->second.push_back({record, std::move(physical)});
                                ++records[record].remote_refs;
                                ++statistics.matched_occurrences;
                                request.done = ++request.next == total;
                                const auto key_bytes = key_capacity(key);
                                key = {};
                                budget.used -= key_bytes;
                            }
                            r.done();
                            received_final = frame.final;
                        }
                    });
                retire(incoming);
            }
            collective(*comm,
                [&]
                {
                    if (!received_final)
                        throw std::invalid_argument("Missing final exact contact fan page.");
                    for (const auto& request : pending)
                        if (!request.done)
                            throw std::invalid_argument("Missing exact contact fan terminal response.");
                });
            const auto scratch = pending.capacity() * sizeof(Pending) + queries.capacity() * sizeof(Query);
            std::vector<Pending>().swap(pending);
            std::vector<Query>().swap(queries);
            budget.used -= scratch;
        }
    }
}

void DistributedAdaptiveContactHalo::Impl::fetch_records()
{
    struct Request
    {
        std::uint64_t query;
        std::size_t record;
    };
    for (int offset = 1; offset < ranks; ++offset)
    {
        const int destination = (rank + offset) % ranks;
        const int source = (rank + ranks - offset) % ranks;
        auto cursor = missing_records.lower_bound({destination, 0});
        const auto end = missing_records.lower_bound({destination + 1, 0});
        std::uint64_t next_query = 0, expected_query = 0, page = 0, expected_page = 0;
        std::set<std::size_t> served;
        while (true)
        {
            if (!any(cursor != end))
                break;
            const auto cap = credit();
            const auto first_query = next_query;
            std::vector<std::size_t> pending;
            Writer outgoing;
            int peer = -1;
            collective(*comm,
                [&]
                {
                    if (cursor != end)
                    {
                        if (cap <= header_bytes + 8)
                            throw std::invalid_argument("Insufficient contact record request page credit.");
                        peer = destination;
                        outgoing = writer(Kind::Records, first_query, page++);
                        admit_writer(outgoing, cap);
                        outgoing.word(0);
                        std::size_t count = 0;
                        while (cursor != end)
                        {
                            const auto id = cursor->second;
                            const auto& record = records[id];
                            if (record.owner != destination || record.contact)
                                throw std::invalid_argument("Stale missing-record subscription.");
                            const auto key_size = key_wire_size(record.key);
                            const auto need = 24 + key_size;
                            if (need > cap - outgoing.data.size())
                            {
                                if (!count)
                                    throw std::invalid_argument("A contact record request exceeds page credit.");
                                break;
                            }
                            if (key_size > (budget.limit - budget.used) / 3)
                                throw std::invalid_argument("Spatial request key exceeds live byte budget.");
                            budget.add(3 * key_size);
                            outgoing.word(next_query++);
                            {
                                const auto key = encode_adaptive_key(record.key);
                                outgoing.bytes(key);
                            }
                            budget.used -= 3 * key_size;
                            outgoing.word(record.revision);
                            ++statistics.record_fetches;
                            budget.append_capacity(pending);
                            pending.push_back(id);
                            ++cursor;
                            ++count;
                        }
                        replace_word(outgoing.data, header_bytes, count);
                        statistics.query_bytes += outgoing.data.size();
                    }
                });
            auto incoming = exchange(peer, std::move(outgoing));
            std::vector<Request> requests;
            collective(*comm,
                [&]
                {
                    if (!incoming.packets.empty())
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                            throw std::invalid_argument("Foreign contact record request page.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::Records, source);
                        const auto n = r.word();
                        if (frame.query != expected_query || frame.page != expected_page++ || !frame.final || !n ||
                            n > r.data.size() / 24)
                            throw std::invalid_argument("Missing or duplicate contact record request page.");
                        for (std::uint64_t i = 0; i < n; ++i)
                        {
                            const auto query = r.word();
                            if (query != expected_query++)
                                throw std::invalid_argument("Nonsequential contact record request.");
                            const auto encoded_key = r.bytes_view();
                            auto key = decode_bounded_key(encoded_key, budget);
                            const auto revision = r.word();
                            const auto found = record_ids.find(key);
                            if (found == record_ids.end() || records[found->second].owner != rank ||
                                records[found->second].revision != revision || !records[found->second].contact ||
                                served.contains(found->second))
                                throw std::invalid_argument("Foreign, stale or duplicate contact record request.");
                            budget.add(sizeof(std::size_t) + map_links);
                            served.insert(found->second);
                            budget.append_capacity(requests);
                            requests.push_back({query, found->second});
                            const auto key_bytes = key_capacity(key);
                            key = {};
                            budget.used -= key_bytes;
                        }
                        r.done();
                    }
                });
            retire(incoming);
            std::size_t request_cursor = 0, encoded_offset = 0, receive_cursor = 0;
            std::uint64_t response_page = 0, expected_response_page = 0;
            AdaptiveBytes encoded, assembly;
            std::size_t assembly_total = 0;
            bool received_final = pending.empty();
            while (any(request_cursor < requests.size()))
            {
                const auto reply_cap = credit();
                outgoing = Writer{};
                peer = -1;
                collective(*comm,
                    [&]
                    {
                        if (request_cursor < requests.size())
                        {
                            if (reply_cap <= header_bytes + 8 + 32)
                                throw std::invalid_argument("Insufficient contact record data page credit.");
                            peer = source;
                            outgoing = writer(Kind::RecordData, 0, response_page++);
                            ++statistics.record_pages;
                            admit_writer(outgoing, reply_cap);
                            outgoing.word(0);
                            std::size_t chunks = 0;
                            while (request_cursor < requests.size() && reply_cap - outgoing.data.size() > 32)
                            {
                                const auto& request = requests[request_cursor];
                                if (encoded.empty())
                                {
                                    encoded = encode_contact(*records[request.record].contact, budget);
                                    ++diagnostics.transmitted_geometry_records;
                                    diagnostics.replicated_record_bytes += encoded.size();
                                    statistics.record_bytes += encoded.size();
                                }
                                const auto count =
                                    std::min(encoded.size() - encoded_offset, reply_cap - outgoing.data.size() - 32);
                                outgoing.word(request.query);
                                outgoing.word(encoded.size());
                                outgoing.word(encoded_offset);
                                outgoing.bytes(std::span<const char>(encoded).subspan(encoded_offset, count));
                                ++chunks;
                                encoded_offset += count;
                                if (encoded_offset == encoded.size())
                                {
                                    const auto bytes = encoded.capacity();
                                    AdaptiveBytes().swap(encoded);
                                    budget.used -= bytes;
                                    encoded_offset = 0;
                                    ++request_cursor;
                                }
                            }
                            replace_word(outgoing.data, header_bytes, chunks);
                            replace_word(outgoing.data, header_bytes - 8, request_cursor == requests.size());
                        }
                    });
                incoming = exchange(peer, std::move(outgoing));
                collective(*comm,
                    [&]
                    {
                        if (!incoming.packets.empty())
                        {
                            if (received_final || incoming.packets.size() != 1 ||
                                incoming.packets.front().peer != destination)
                                throw std::invalid_argument("Foreign or duplicate contact record data page.");
                            Reader r{incoming.packets.front().bytes};
                            const auto frame = read(r, Kind::RecordData, destination);
                            if (frame.query || frame.page != expected_response_page++)
                                throw std::invalid_argument("Missing contact record data page.");
                            const auto n = r.word();
                            if (!n || n > r.data.size() / 32)
                                throw std::invalid_argument("Invalid contact record data chunk count.");
                            for (std::uint64_t i = 0; i < n; ++i)
                            {
                                const auto query = r.word(), total = r.word(), offset_ = r.word();
                                const auto bytes = r.bytes_view();
                                if (receive_cursor >= pending.size() || query != first_query + receive_cursor ||
                                    offset_ != assembly.size() || !total || bytes.empty() ||
                                    bytes.size() > total - std::min<std::uint64_t>(total, offset_) ||
                                    (!assembly.empty() && total != assembly_total))
                                    throw std::invalid_argument(
                                        "Duplicate, truncated or foreign contact record chunk.");
                                if (assembly.empty())
                                {
                                    budget.add(total);
                                    assembly.reserve(total);
                                    if (assembly.capacity() != total)
                                        throw std::invalid_argument(
                                            "Contact record assembly capacity differs from admission.");
                                    assembly_total = total;
                                }
                                assembly.insert(assembly.end(), bytes.begin(), bytes.end());
                                if (assembly.size() == assembly_total)
                                {
                                    auto& record = records[pending[receive_cursor]];
                                    if (record.contact || record.owner != destination)
                                        throw std::invalid_argument(
                                            "Duplicate or conflicting contact record completion.");
                                    const auto before = budget.used, identities_before = identities.admitted_bytes();
                                    auto decoded = decode_contact(assembly, budget, identities);
                                    if (decoded.key != record.key || decoded.domain != record.domain)
                                        throw std::invalid_argument(
                                            "Contact record reply key/domain differs from exact match.");
                                    decoded.identity = pending[receive_cursor];
                                    attach_record(pending[receive_cursor], std::move(decoded),
                                        budget.used - before - (identities.admitted_bytes() - identities_before));
                                    if (!missing_records.erase({destination, pending[receive_cursor]}))
                                        throw std::invalid_argument("Unrequested contact record completion.");
                                    budget.used -= sizeof(decltype(missing_records)::value_type) + map_links;
                                    ++statistics.remote_records;
                                    ++statistics.decoded_record_replies;
                                    ++diagnostics.decoded_geometry_records;
                                    diagnostics.decoded_geometry_bytes += sizeof(record.contact->spans);
                                    const auto capacity = assembly.capacity();
                                    AdaptiveBytes().swap(assembly);
                                    budget.used -= capacity;
                                    assembly_total = 0;
                                    ++receive_cursor;
                                }
                            }
                            r.done();
                            received_final = frame.final;
                        }
                    });
                retire(incoming);
            }
            collective(*comm,
                [&]
                {
                    if (!received_final || receive_cursor != pending.size() || !assembly.empty())
                        throw std::invalid_argument("Missing contact record terminal completion.");
                });
            const auto scratch = pending.capacity() * sizeof(std::size_t) + requests.capacity() * sizeof(Request);
            std::vector<std::size_t>().swap(pending);
            std::vector<Request>().swap(requests);
            budget.used -= scratch;
        }
        const auto served_bytes = served.size() * (sizeof(std::size_t) + map_links);
        served.clear();
        budget.used -= served_bytes;
    }
}

std::vector<AdaptiveMark> DistributedAdaptiveContactHalo::Impl::evaluate(bool retain_charge)
{
    if (!evaluated)
    {
        collective(*comm,
            [&]
            {
                diagnostic_phase = "pair_gather";
                while (!dirty_points.empty())
                {
                    const auto receipt = dirty_points.begin();
                    const auto* key = *receipt;
                    gather_pairs(*key);
                    const auto found = points.find(*key);
                    // Pair identities have been copied; no current owned row
                    // uses an empty-local group. Retire the receipt before its
                    // borrowed key owner, returning overlap before sorting.
                    dirty_points.erase(receipt);
                    budget.used -= sizeof(const AdaptiveBytes*) + map_links;
                    if (found != points.end() && found->second.local.empty())
                    {
                        remove_remote_points(found->first);
                        const auto bytes = sizeof(decltype(points)::value_type) + map_links + found->first.capacity() +
                                           found->second.local.capacity() * sizeof(LocalOccurrence);
                        points.erase(found);
                        budget.used -= bytes;
                    }
                }
                diagnostic_phase = "pair_sort";
                auto sorted_pairs = affected_pairs.take_sorted();
                statistics.unique_cell_pairs = sorted_pairs.size();
                diagnostic_phase = "pair_evaluate";
                struct PhysicalPair
                {
                    const AdaptiveBytes* key;
                    const ContactPointIdentity *first, *second;
                };
                auto erase_requirement = [&](const auto& ids)
                {
                    if (failing_pairs.erase(ids))
                        budget.used -= sizeof(decltype(failing_pairs)::value_type) + map_links;
                };
                auto charge_work = [&](std::size_t n)
                {
                    if (n > options.contact_budget - statistics.work)
                        throw std::invalid_argument("Spatial closure cumulative contact budget exhausted.");
                    statistics.work += n;
                };
                for (const auto& ids : sorted_pairs)
                {
                    auto& a = records[ids.first];
                    auto& b = records[ids.second];
                    if (a.owner < 0 || b.owner < 0 || (a.owner != rank && b.owner != rank))
                    {
                        erase_requirement(ids);
                        continue;
                    }
                    const auto local_id = a.owner == rank ? ids.first : ids.second;
                    const auto other_id = local_id == ids.first ? ids.second : ids.first;
                    const auto& local = records[local_id];
                    if (local.row >= rows.size() || !rows[local.row])
                        throw std::invalid_argument("Missing local spatial incidence row.");
                    std::vector<PhysicalPair> physical;
                    for (const auto& point : rows[local.row]->points)
                    {
                        const auto maximum = 16 + local.domain.size() + PLIC::canonical_key_size(point.normalized);
                        if (maximum > (budget.limit - budget.used) / 4)
                            throw std::invalid_argument("Contact pair key exceeds byte credit.");
                        budget.add(4 * maximum);
                        {
                            const auto key = point_key(local.domain, point.normalized);
                            const auto group = points.find(key);
                            const ContactPointIdentity* other = nullptr;
                            if (group != points.end())
                            {
                                for (const auto& item : group->second.local)
                                    if (item.record == other_id)
                                        other = &rows[item.row]->points[item.point].physical;
                                if (const auto remote = remote_points.find(key); remote != remote_points.end())
                                    for (const auto& item : remote->second)
                                        if (item.record == other_id)
                                            other = &item.physical;
                            }
                            if (other)
                            {
                                budget.append_capacity(physical);
                                physical.push_back(local_id == ids.first
                                                       ? PhysicalPair{&group->first, &point.physical, other}
                                                       : PhysicalPair{&group->first, other, &point.physical});
                            }
                        }
                        budget.used -= 4 * maximum;
                    }
                    std::ranges::sort(physical, [](const auto& x, const auto& y) { return *x.key < *y.key; });
                    if (physical.empty())
                    {
                        erase_requirement(ids);
                        continue;
                    }
                    if (!a.contact || !b.contact)
                        throw std::invalid_argument("Incomplete exact-match contact record cache.");
                    statistics.logical_contact_pairs += physical.size();
                    ClosureRequirements required;
                    charge_work(1);
                    ++statistics.span_pair_evaluations;
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        if (a.owner == rank && span_exceeds_ratio(a.contact->spans[axis], b.contact->spans[axis],
                                                   options.maximum_span_ratio))
                            join_closure_requirement(*a.contact, required.first, a.contact->factors[axis]);
                        if (b.owner == rank && span_exceeds_ratio(b.contact->spans[axis], a.contact->spans[axis],
                                                   options.maximum_span_ratio))
                            join_closure_requirement(*b.contact, required.second, b.contact->factors[axis]);
                    }
                    auto traces = [&](const PhysicalPair& point)
                    {
                        if (!b.contact->trace_faces().empty() &&
                            a.contact->trace_faces().size() >
                                (options.contact_budget - statistics.work) / b.contact->trace_faces().size())
                            throw std::invalid_argument("Spatial closure cumulative trace budget exhausted.");
                        const auto count = a.contact->trace_faces().size() * b.contact->trace_faces().size();
                        charge_work(count);
                        statistics.trace_pair_evaluations += count;
                        for (const auto& first : a.contact->trace_faces())
                            for (const auto& second : b.contact->trace_faces())
                            {
                                const auto request = trace_pair_requirements(
                                    first, second, *point.first, *point.second, options.maximum_trace_jump);
                                if (a.owner == rank && request.first)
                                    join_closure_requirement(*a.contact, required.first, std::uint64_t(*request.first));
                                if (b.owner == rank && request.second)
                                    join_closure_requirement(
                                        *b.contact, required.second, std::uint64_t(*request.second));
                            }
                    };
                    if (a.contact->traces_point_independent && b.contact->traces_point_independent)
                        traces(physical.front());
                    else
                        for (const auto& point : physical)
                            traces(point);
                    const auto scratch = physical.capacity() * sizeof(PhysicalPair);
                    std::vector<PhysicalPair>().swap(physical);
                    budget.used -= scratch;
                    if (required.first || required.second)
                    {
                        if (!failing_pairs.contains(ids))
                            budget.add(sizeof(decltype(failing_pairs)::value_type) + map_links);
                        failing_pairs.insert_or_assign(ids, required);
                    }
                    else
                        erase_requirement(ids);
                }
                sorted_pairs.clear();
                for (const auto id : possibly_unused_records)
                    if (records[id].owner >= 0 && records[id].owner != rank && !records[id].remote_refs)
                        discard_record(id);
                const auto unused_bytes = possibly_unused_records.size() * (sizeof(std::size_t) + map_links);
                possibly_unused_records.clear();
                budget.used -= unused_bytes;
                for (const auto id : retired_records)
                {
                    auto& record = records[id];
                    if (record.borrowed_bytes &&
                        (record.contact.use_count() > 1 || deferred_owners.contains(record.contact)))
                    {
                        if (!deferred_owners.adopt(record.contact, record.borrowed_bytes))
                            throw std::invalid_argument("Duplicate borrowed contact-record admission.");
                        record.borrowed_bytes = 0;
                    }
                    const auto bytes = key_capacity(record.key) + record.domain.capacity() + record.borrowed_bytes;
                    record = {};
                    record.owner = -1;
                    budget.used -= bytes;
                    budget.append_capacity(free_records);
                    free_records.push_back(id);
                }
                retired_records.clear();
                deferred_owners.reap();
                evaluated = true;
                diagnostic_phase = "ready";
            });
    }
    std::vector<AdaptiveMark> result;
    collective(*comm,
        [&]
        {
            std::map<std::size_t, std::optional<AdaptiveSplit>> required;
            for (const auto& [ids, requests] : failing_pairs)
            {
                auto join = [&](std::size_t id, const auto& request)
                {
                    if (!request)
                        return;
                    if (!required.contains(id))
                        budget.add(sizeof(decltype(required)::value_type) + map_links);
                    join_closure_requirement(*records[id].contact, required[id], std::uint64_t(*request));
                };
                join(ids.first, requests.first);
                join(ids.second, requests.second);
            }
            for (const auto& [id, request] : required)
            {
                budget.append_capacity(result);
                budget.add(key_capacity(records[id].key));
                result.push_back({records[id].key, *request});
            }
            const auto bytes = required.size() * (sizeof(decltype(required)::value_type) + map_links);
            required.clear();
            budget.used -= bytes;
            std::ranges::sort(result, {}, &AdaptiveMark::key);
        });
    std::size_t bytes = result.capacity() * sizeof(AdaptiveMark);
    for (const auto& mark : result)
        bytes += key_capacity(mark.key);
    if (!retain_charge)
        budget.used -= bytes;
    return result;
}

DistributedAdaptiveContactHalo::DistributedAdaptiveContactHalo(const AdaptiveProviderKernel& kernel,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveClosureOptions options, ContactHaloContext context,
    ContactHaloBeforeSend before_send)
{
    if (comm.is_null())
        throw std::invalid_argument("Spatial closure needs a communicator.");
    // No rank enters the constructor's first communication until every local
    // owner allocation has succeeded. Impl construction itself cannot communicate.
    collective(*comm,
        [&]
        {
            if (sizeof(Impl) > options.byte_budget)
                throw std::invalid_argument("Spatial closure session exceeds its byte budget.");
            impl_ = std::make_unique<Impl>(kernel, comm, options, context, std::move(before_send));
        });
    impl_->initialize_collective();
}
DistributedAdaptiveContactHalo::~DistributedAdaptiveContactHalo() = default;
DistributedAdaptiveContactHalo::DistributedAdaptiveContactHalo(DistributedAdaptiveContactHalo&&) noexcept = default;
DistributedAdaptiveContactHalo& DistributedAdaptiveContactHalo::operator=(
    DistributedAdaptiveContactHalo&&) noexcept = default;
Budget& DistributedAdaptiveContactHalo::allocation() noexcept
{
    impl_->deferred_owners.reap();
    return impl_->budget;
}
void DistributedAdaptiveContactHalo::for_each_incidence_row(
    const std::function<void(const ContactIncidenceRow&)>& visitor) const
{
    for (const auto& row : impl_->rows)
        if (row)
            visitor(*row);
}
ContactMarkLease::ContactMarkLease(std::vector<AdaptiveMark> marks, Budget& budget, std::size_t charged) noexcept
    : marks_(std::move(marks)), budget_(&budget), charged_(charged)
{
}
ContactMarkLease::~ContactMarkLease()
{
    reset();
}
ContactMarkLease::ContactMarkLease(ContactMarkLease&& other) noexcept
    : marks_(std::move(other.marks_)), budget_(std::exchange(other.budget_, nullptr)),
      charged_(std::exchange(other.charged_, 0))
{
}
ContactMarkLease& ContactMarkLease::operator=(ContactMarkLease&& other) noexcept
{
    if (this != &other)
    {
        reset();
        marks_ = std::move(other.marks_);
        budget_ = std::exchange(other.budget_, nullptr);
        charged_ = std::exchange(other.charged_, 0);
    }
    return *this;
}
void ContactMarkLease::reset() noexcept
{
    std::vector<AdaptiveMark>().swap(marks_);
    if (budget_)
        budget_->used -= charged_;
    budget_ = nullptr;
    charged_ = 0;
}

void DistributedAdaptiveContactHalo::bootstrap(
    std::span<const AdaptiveLeaf> owned, std::vector<ContactIncidenceRow> input)
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (state.ready || owned.size() != input.size())
                throw std::invalid_argument("Invalid or repeated spatial contact bootstrap.");
            state.budget.add(input.capacity() * sizeof(ContactIncidenceRow));
            std::map<AdaptiveCellKey, const AdaptiveLeaf*> source;
            std::size_t source_bytes = 0;
            for (const auto& leaf : owned)
            {
                const auto bytes = sizeof(decltype(source)::value_type) + map_links + key_capacity(leaf.key);
                state.budget.add(bytes);
                source_bytes += bytes;
                if (!source.emplace(leaf.key, &leaf).second)
                    throw std::invalid_argument("Duplicate spatial source key.");
            }
            for (auto& row : input)
            {
                const auto found = source.find(row.key);
                if (found == source.end())
                    throw std::invalid_argument("Missing spatial source row.");
                const auto bytes = row_storage(row);
                state.budget.add(bytes);
                auto keeper = state.arena;
                auto handle = std::shared_ptr<const ContactIncidenceRow>(new ContactIncidenceRow(std::move(row)),
                    [keeper, bytes](const ContactIncidenceRow* value)
                    {
                        delete value;
                        keeper->used -= bytes;
                    });
                state.install_row(std::move(handle), *found->second, true);
            }
            source.clear();
            state.budget.used -= source_bytes;
            const auto slots = input.capacity() * sizeof(ContactIncidenceRow);
            std::vector<ContactIncidenceRow>().swap(input);
            state.budget.used -= slots;
            state.statistics.bootstrap_owned_visits = owned.size();
            state.diagnostics.cells = owned.size();
        });
    collective(*state.comm, [&] { state.refresh_covers(); });
    state.diagnostic_phase = "directory";
    state.build_directory();
    state.discover_neighbors();
    state.diagnostic_phase = "point_fans";
    state.discover_points();
    state.diagnostic_phase = "record_fetch";
    state.fetch_records();
    state.ready = true;
    state.diagnostic_phase = "ready";
}

void DistributedAdaptiveContactHalo::bootstrap_canonical(
    std::span<const AdaptiveLeaf> owned, const ContactOwnedLeafLookup& lookup)
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (state.ready || state.incidence)
                throw std::invalid_argument("Repeated canonical contact bootstrap.");
            state.incidence = std::make_unique<DistributedAdaptiveIncidenceSession>(
                state.kernel, state.comm, state.options, state.context, state.budget, state.identities, state.arena);
        });
    state.diagnostic_phase = "canonical_incidence";
    auto rows = state.incidence->bootstrap(owned, lookup);
    state.diagnostic_phase = "local_rows";
    collective(*state.comm,
        [&]
        {
            for (auto& row : rows)
            {
                const auto& leaf = lookup(row->key);
                state.install_row(std::move(row), leaf, true);
            }
            const auto bytes = rows.capacity() * sizeof(IncidenceRowHandle);
            std::vector<IncidenceRowHandle>().swap(rows);
            state.budget.used -= bytes;
            state.statistics.bootstrap_owned_visits = owned.size();
            state.diagnostics.cells = owned.size();
        });
    collective(*state.comm, [&] { state.refresh_covers(); });
    state.diagnostic_phase = "directory";
    state.build_directory();
    state.discover_neighbors();
    state.diagnostic_phase = "point_fans";
    state.discover_points();
    state.diagnostic_phase = "record_fetch";
    state.fetch_records();
    state.ready = true;
    state.diagnostic_phase = "ready";
}

void DistributedAdaptiveContactHalo::apply_delta(std::span<const ContactRetiredLeaf> retired,
    std::span<const AdaptiveCellKey> inserted, ContactHaloContext next, const ContactOwnedLeafLookup& lookup)
{
    auto& state = *impl_;
    state.deferred_owners.reap();
    std::vector<IncidenceCellVersion> versions;
    collective(*state.comm,
        [&]
        {
            if (!state.ready || !state.incidence || !state.evaluated)
                throw std::invalid_argument("Incremental contact update needs a completed canonical session.");
            if (next == state.context)
                throw std::invalid_argument("Incremental contact context did not advance.");
            state.budget.add(retired.size() * sizeof(IncidenceCellVersion));
            versions.reserve(retired.size());
            for (const auto& item : retired)
            {
                const auto found = state.record_ids.find(item.key);
                if (found == state.record_ids.end())
                    throw std::invalid_argument("Retired contact cell is absent.");
                const auto& record = state.records[found->second];
                if (record.owner != state.rank || record.revision != item.revision || record.row >= state.rows.size())
                    throw std::invalid_argument("Retired contact owner or revision is stale.");
                state.budget.add(key_capacity(item.key));
                versions.push_back({item.key, item.revision, state.rows[record.row]->incidence_generation});
            }
        });
    auto delta = state.incidence->apply_delta(versions, inserted, next, lookup);
    collective(*state.comm,
        [&]
        {
            std::size_t bytes = versions.capacity() * sizeof(IncidenceCellVersion);
            for (const auto& item : versions)
                bytes += key_capacity(item.key);
            std::vector<IncidenceCellVersion>().swap(versions);
            state.budget.used -= bytes;
            state.context = next;
            state.evaluated = false;
            state.statistics.work = 0;
            for (const auto& item : delta.retired)
            {
                const auto found = state.record_ids.find(item.key);
                if (found == state.record_ids.end())
                    throw std::invalid_argument("Incidence retired an unknown cell.");
                const auto id = found->second;
                const auto& record = state.records[id];
                if (record.owner != state.rank || record.revision != item.leaf_revision ||
                    record.row >= state.rows.size() ||
                    state.rows[record.row]->incidence_generation != item.incidence_generation)
                    throw std::invalid_argument("Incidence retirement version is stale.");
                state.retire_row(record.row);
            }
            for (const auto& item : retired)
            {
                const auto found = state.record_ids.find(item.key);
                if (found == state.record_ids.end())
                    throw std::invalid_argument("Missing retired geometry record.");
                if (state.records[found->second].row < state.rows.size())
                    throw std::invalid_argument("Retired leaf still has a live incidence row.");
                state.discard_record(found->second);
                --state.statistics.owned_records;
            }
            for (auto& row : delta.changed)
            {
                if (const auto found = state.record_ids.find(row->key); found != state.record_ids.end())
                {
                    const auto& record = state.records[found->second];
                    if (record.owner == state.rank && record.row < state.rows.size())
                    {
                        if (record.revision != row->revision ||
                            state.rows[record.row]->incidence_generation >= row->incidence_generation)
                            throw std::invalid_argument("Changed incidence row did not advance its generation.");
                        state.retire_row(record.row);
                    }
                }
                const auto& leaf = lookup(row->key);
                state.install_row(std::move(row), leaf, true);
            }
            bytes = delta.retired.capacity() * sizeof(IncidenceCellVersion) +
                    delta.changed.capacity() * sizeof(IncidenceRowHandle);
            for (const auto& item : delta.retired)
                bytes += key_capacity(item.key);
            delta = {};
            state.budget.used -= bytes;
            state.statistics.delta_owned_visits += inserted.size();
            state.previous_neighbors.swap(state.neighbors);
            const auto directory_bytes =
                state.directory.size() * (sizeof(decltype(state.directory)::value_type) + map_links);
            state.directory.clear();
            state.budget.used -= directory_bytes;
        });
    state.initialize_collective();
    collective(*state.comm, [&] { state.refresh_covers(); });
    state.build_directory();
    state.discover_neighbors();
    state.notify_dirty_points();
    collective(*state.comm,
        [&]
        {
            for (const auto* key : state.dirty_points)
                state.remove_remote_points(*key);
            std::size_t bytes =
                state.previous_neighbors.size() * (sizeof(decltype(state.previous_neighbors)::value_type) + map_links);
            for (const auto& [owner, covers] : state.previous_neighbors)
            {
                bytes += covers.capacity() * sizeof(ClosedContactCover);
                for (const auto& cover : covers)
                    bytes += cover.domain.capacity();
            }
            state.previous_neighbors.clear();
            state.budget.used -= bytes;
        });
    state.discover_points();
    state.fetch_records();
}

std::unique_ptr<DistributedAdaptiveContactHalo> DistributedAdaptiveContactHalo::fork_candidate(
    ContactHaloContext context, std::size_t caller_live_bytes) const
{
    impl_->deferred_owners.reap();
    const auto& source = *impl_;
    std::unique_ptr<DistributedAdaptiveContactHalo> candidate;
    // Allocate the outer owner on every rank before its communicating ctor.
    collective(*source.comm,
        [&]
        {
            if (!source.ready || !source.evaluated || !source.incidence)
                throw std::invalid_argument("Contact candidate needs an evaluated canonical source.");
            const auto metadata = source.metadata_bytes();
            constexpr auto header = sizeof(Impl) + sizeof(Budget) + 128 + sizeof(DistributedAdaptiveContactHalo);
            if (caller_live_bytes > source.options.byte_budget ||
                metadata > source.options.byte_budget - caller_live_bytes ||
                header > source.options.byte_budget - caller_live_bytes - metadata)
                throw std::invalid_argument("Contact candidate metadata overlap exceeds byte budget.");
        });
    void* storage = nullptr;
    collective(*source.comm, [&] { storage = ::operator new(sizeof(DistributedAdaptiveContactHalo)); });
    try
    {
        candidate.reset(new (storage) DistributedAdaptiveContactHalo(
            source.kernel, source.comm, source.options, context, source.before_send));
    }
    catch (...)
    {
        ::operator delete(storage);
        throw;
    }
    auto& state = *candidate->impl_;
    collective(*source.comm,
        [&]
        {
            state.budget.add(caller_live_bytes);
            state.external_retained_bytes = caller_live_bytes;
            state.fork_overlap_bytes = source.metadata_bytes();
            state.budget.add(state.fork_overlap_bytes);
            state.trace_pool.emplace(*source.trace_pool, state.budget);
            auto copy_vector = [&](auto& target, const auto& from)
            {
                using Value = typename std::decay_t<decltype(target)>::value_type;
                state.budget.add(from.size() * sizeof(Value));
                target.reserve(from.size());
                target.assign(from.begin(), from.end());
            };
            copy_vector(state.rows, source.rows);
            state.budget.add(source.rows.size() * sizeof(std::size_t));
            state.borrowed_rows.resize(source.rows.size());
            for (std::size_t i = 0; i < source.rows.size(); ++i)
                if (source.rows[i])
                {
                    state.borrowed_rows[i] = row_storage(*source.rows[i]);
                    state.budget.add(state.borrowed_rows[i]);
                }
            copy_vector(state.free_rows, source.free_rows);
            copy_vector(state.free_records, source.free_records);
            state.records.reserve(source.records.size());
            for (const auto& record : source.records)
            {
                state.budget.add(key_capacity(record.key) + record.domain.size() + record.body_bytes);
                state.records.push_back(record);
                state.records.back().borrowed_bytes = record.body_bytes;
                if (record.contact)
                    for (const auto& trace : record.contact->trace_faces())
                        if (trace.face)
                            state.identities.retain(
                                state.budget, PLIC::planar_support<ArithmeticMode::Ordinary>(*trace.face));
            }
            for (const auto& [key, id] : source.record_ids)
            {
                state.budget.add(sizeof(decltype(state.record_ids)::value_type) + map_links);
                state.record_ids.emplace(&state.records[id].key, id);
            }
            for (const auto& domain : source.domains)
            {
                state.budget.add(sizeof(decltype(state.domains)::value_type) + map_links + domain.size());
                state.domains.insert(domain);
            }
            for (const auto& [key, point] : source.points)
            {
                state.budget.add(sizeof(decltype(state.points)::value_type) + map_links + key.size() +
                                 point.local.size() * sizeof(Impl::LocalOccurrence));
                const auto found = state.domains.find(*point.domain);
                if (found == state.domains.end())
                    throw std::invalid_argument("Missing forked contact domain.");
                state.points.emplace(key, Impl::PointGroup{&*found, point.coordinate, point.local, point.cover_leaf});
            }
            for (const auto& [key, remote] : source.remote_points)
            {
                state.budget.add(sizeof(decltype(state.remote_points)::value_type) + map_links + key.size() +
                                 remote.size() * sizeof(Impl::RemoteOccurrence));
                state.remote_points.emplace(key, remote);
            }
            for (const auto& [ids, request] : source.failing_pairs)
            {
                state.budget.add(sizeof(decltype(state.failing_pairs)::value_type) + map_links);
                state.failing_pairs.emplace(ids, request);
            }
            if (source.index)
            {
                state.index = std::make_unique<ContactCoverIndex>(*source.index, state.budget);
                for (auto& [key, point] : state.points)
                {
                    if (point.cover_leaf == std::numeric_limits<std::size_t>::max())
                        throw std::invalid_argument("Missing copied point cover leaf.");
                    state.index->rebind(point.cover_leaf, &point.coordinate, point.domain);
                }
            }
            state.budget.add(source.covers.size() * sizeof(ClosedContactCover));
            state.covers.reserve(source.covers.size());
            for (const auto& cover : source.covers)
            {
                state.budget.add(cover.domain.size());
                state.covers.push_back(cover);
            }
            for (const auto& [owner, covers] : source.neighbors)
            {
                state.budget.add(sizeof(decltype(state.neighbors)::value_type) + map_links +
                                 covers.size() * sizeof(ClosedContactCover));
                for (const auto& cover : covers)
                    state.budget.add(cover.domain.size());
                state.neighbors.emplace(owner, covers);
            }
            for (const auto& [id, node] : source.directory)
            {
                state.budget.add(sizeof(decltype(state.directory)::value_type) + map_links);
                state.directory.emplace(id, node);
            }
            state.local_cover = source.local_cover;
            state.statistics.owned_records = source.statistics.owned_records;
            state.statistics.remote_records = source.statistics.remote_records;
            state.diagnostics.cells = source.diagnostics.cells;
            state.ready = state.evaluated = true;
        });
    state.incidence = source.incidence->fork(state.budget, state.identities, state.arena);
    return candidate;
}
std::size_t DistributedAdaptiveContactHalo::cache_storage_bytes() const noexcept
{
    return impl_->budget.used - impl_->external_retained_bytes - impl_->fork_overlap_bytes;
}
void DistributedAdaptiveContactHalo::bind_external_retained_bytes(std::size_t bytes) noexcept
{
    impl_->budget.used -= impl_->fork_overlap_bytes;
    impl_->fork_overlap_bytes = 0;
    impl_->external_retained_bytes = bytes;
}

std::vector<AdaptiveMark> DistributedAdaptiveContactHalo::required_refinements()
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (!state.ready)
                throw std::invalid_argument("Spatial closure session has no complete bootstrap.");
        });
    return state.evaluate();
}
ContactMarkLease DistributedAdaptiveContactHalo::lease_required_refinements()
{
    auto& state = *impl_;
    collective(*state.comm,
        [&]
        {
            if (!state.ready)
                throw std::invalid_argument("Spatial closure session has no complete bootstrap.");
        });
    auto marks = state.evaluate(true);
    std::size_t bytes = marks.capacity() * sizeof(AdaptiveMark);
    for (const auto& mark : marks)
        bytes += key_capacity(mark.key);
    return ContactMarkLease(std::move(marks), state.budget, bytes);
}
const ContactHaloStatistics& DistributedAdaptiveContactHalo::statistics() const noexcept
{
    if (impl_->incidence)
    {
        const auto& incidence = impl_->incidence->statistics();
        impl_->statistics.bootstrap_owned_visits = incidence.bootstrap_owned_visits;
        impl_->statistics.delta_owned_visits = incidence.delta_owned_visits;
        impl_->statistics.full_owned_scans_after_bootstrap = incidence.full_owned_scans_after_bootstrap;
        impl_->statistics.dirty_supports = incidence.dirty_supports;
        impl_->statistics.dirty_support_members = incidence.dirty_support_members;
        impl_->statistics.dirty_lines = incidence.dirty_lines;
        impl_->statistics.dirty_subscriptions = incidence.dirty_subscriptions;
        impl_->statistics.recomputed_tiles = incidence.recomputed_tiles;
    }
    impl_->statistics.retained_bytes = impl_->budget.used;
    impl_->statistics.peak_bytes = impl_->budget.peak;
    impl_->statistics.cumulative_bytes = impl_->budget.cumulative;
    return impl_->statistics;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
