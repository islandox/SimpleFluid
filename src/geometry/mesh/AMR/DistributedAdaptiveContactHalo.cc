/** @file DistributedAdaptiveContactHalo.cc
 * @brief Bounded owner queries over complete canonical point incidence.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/mesh/AMR/ClosedContactIndex.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::uint64_t halo_magic = 0x53464348414c4f31ULL;
using Kind = ContactHaloMessageKind;
constexpr std::size_t header_bytes = 12 * 8;
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
        std::span<const char> domain;
        std::array<long double, 3> coordinate;
        std::vector<LocalOccurrence> local;
    };
    struct Record
    {
        AdaptiveCellKey key;
        AdaptiveBytes domain;
        std::uint64_t revision;
        int owner;
        std::optional<ContactRecord> contact;
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
    Budget budget;
    IdentityBudget identities;
    ContactHaloStatistics statistics;
    AdaptiveClosureStats diagnostics;
    std::vector<ContactIncidenceRow> rows;
    std::map<AdaptiveBytes, PointGroup> points;
    std::vector<Record> records;
    std::map<AdaptiveCellKey, std::size_t> record_ids;
    std::map<AdaptiveBytes, std::vector<RemoteOccurrence>> remote_points;
    std::unique_ptr<ClosedContactIndex> index;
    std::vector<ClosedContactCover> covers;
    std::map<int, DirectoryNode> directory;
    std::map<int, std::vector<ClosedContactCover>> neighbors;
    Bounds local_cover;
    int ranks, rank, tree_width = 1;
    std::size_t page_reservation = 0;
    bool ready = false, evaluated = false;

    Impl(const AdaptiveProviderKernel& kernel_, Teuchos::RCP<const Teuchos::Comm<int>> comm_,
        AdaptiveClosureOptions options_, ContactHaloContext context_, ContactHaloBeforeSend before_send_)
        : kernel(kernel_), comm(std::move(comm_)), options(options_), context(context_),
          before_send(std::move(before_send_)), budget{options.byte_budget},
          ranks(comm.is_null() ? 0 : comm->getSize()), rank(comm.is_null() ? -1 : comm->getRank())
    {
        budget.add(sizeof(Impl));
    }
    void initialize_collective()
    {
        std::array<std::uint64_t, 9> semantics{context.epoch, context.topology, context.partition, context.round,
            options.byte_budget, options.contact_budget, options.tile_budget,
            std::bit_cast<std::uint64_t>(options.maximum_span_ratio), options.maximum_trace_jump};
        auto root = semantics;
        Teuchos::broadcast(*comm, 0, int(root.size()), root.data());
        collective(*comm, [&]
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
    Writer writer(Kind kind, std::uint64_t query = 0, std::uint64_t page = 0,
        std::uint64_t total = 0, bool final = true)
    {
        Writer w;
        constexpr std::size_t initial = header_bytes + 8 + 6 * PLIC::ScalarPacketWords{}.size() * 8;
        if (page_reservation) throw std::invalid_argument("Overlapping spatial page reservations.");
        budget.add(initial);
        page_reservation = initial;
        w.data.reserve(initial);
        if (w.data.capacity() != initial) throw std::invalid_argument("Spatial header capacity differs from admission.");
        w.word(halo_magic);
        w.word(1);
        w.word(std::uint64_t(kind));
        w.word(context.epoch);
        w.word(context.topology);
        w.word(context.partition);
        w.word(context.round);
        w.word(rank);
        w.word(query);
        w.word(page);
        w.word(total);
        w.word(final);
        return w;
    }
    Frame read(Reader& r, Kind kind, int source)
    {
        if (r.word() != halo_magic || r.word() != 1 || r.word() != std::uint64_t(kind) ||
            r.word() != context.epoch || r.word() != context.topology || r.word() != context.partition ||
            r.word() != context.round || r.word() != std::uint64_t(source))
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
        collective(*comm, [&]
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
        const auto resident = budget.used;
        auto incoming = consume_owner_packets(std::move(outgoing), comm, budget.limit - budget.used);
        collective(*comm, [&]
            {
                if (incoming.stats.peak_scratch_bytes > budget.limit - resident)
                    throw std::invalid_argument("Spatial page transport exceeds live byte credit.");
                budget.peak = std::max(budget.peak, resident + incoming.stats.peak_scratch_bytes);
                budget.cumulative += incoming.stats.peak_scratch_bytes;
                budget.used -= outgoing_charge;
                page_reservation = 0;
                budget.add(incoming.packets.capacity() * sizeof(OwnerPacket));
                for (const auto& packet : incoming.packets)
                    budget.add(packet.bytes.capacity());
            });
        return incoming;
    }
    void retire(OwnerExchange& exchange)
    {
        std::size_t bytes = exchange.packets.capacity() * sizeof(OwnerPacket);
        for (const auto& packet : exchange.packets)
            bytes += packet.bytes.capacity();
        OwnerExchange empty;
        std::swap(exchange, empty);
        budget.used -= bytes;
    }
    void admit_writer(Writer& w, std::size_t capacity)
    {
        if (capacity > budget.limit - budget.used)
            throw std::invalid_argument("Spatial closure page exceeds available byte credit.");
        const auto previous = page_reservation;
        if (!previous) throw std::invalid_argument("Missing spatial page header reservation.");
        if (capacity < previous) throw std::invalid_argument("Spatial page credit cannot hold header.");
        budget.add(capacity);
        page_reservation = capacity;
        w.data.reserve(capacity);
        if (w.data.capacity() != capacity)
            throw std::invalid_argument("Spatial closure page capacity differs from admission.");
        budget.used -= previous;
    }
    std::size_t add_record(const AdaptiveCellKey& key, std::uint64_t revision, int owner,
        std::span<const char> domain)
    {
        if (const auto found = record_ids.find(key); found != record_ids.end())
        {
            const auto& record = records[found->second];
            if (record.owner != owner || record.revision != revision || !std::ranges::equal(record.domain, domain))
                throw std::invalid_argument("Conflicting spatial closure cell owner/revision/domain.");
            return found->second;
        }
        budget.add(sizeof(decltype(record_ids)::value_type) + map_links + 2 * key_capacity(key) + domain.size());
        budget.append_capacity(records);
        const auto id = records.size();
        records.push_back({key, AdaptiveBytes(domain.begin(), domain.end()), revision, owner, {}, {}});
        record_ids.emplace(key, id);
        return id;
    }
    void build_directory()
    {
        Bounds accumulated = local_cover;
        for (int stride = 1; stride < tree_width; stride *= 2)
        {
            int destination = -1;
            Writer outgoing;
            collective(*comm, [&]
                {
                    if (rank % (2 * stride) == stride)
                    {
                        destination = rank - stride;
                        outgoing = writer(Kind::Cover, stride);
                        write_bounds(outgoing, accumulated);
                    }
                });
            auto incoming = exchange(destination, std::move(outgoing));
            collective(*comm, [&]
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
            collective(*comm, [&]
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
            collective(*comm, [&]
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
                collective(*comm, [&]
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
                collective(*comm, [&]
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
                    collective(*comm, [&]
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
                                    const auto need = 8 + cover.domain.size() + 8 + 6 * PLIC::ScalarPacketWords{}.size() * 8;
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
                    collective(*comm, [&]
                        {
                            if (rank == requester && !incoming.packets.empty())
                            {
                                if (received_final || incoming.packets.size() != 1 || incoming.packets.front().peer != owner)
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
                                    found->second.push_back({AdaptiveBytes(domain.begin(), domain.end()), bounds.box, owner, 0});
                                }
                                r.done();
                                received_final = frame.final;
                            }
                            else if (!incoming.packets.empty())
                                throw std::invalid_argument("Foreign owner-cover page delivery.");
                        });
                    retire(incoming);
                }
                collective(*comm, [&]
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
            if (std::ranges::equal(group.domain, cover.domain) && contains(cover.bounds, group.coordinate))
                return true;
        return false;
    }

    void discover_points();
    void fetch_records();
    std::vector<AdaptiveMark> evaluate();
};

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
        auto cursor = points.cbegin();
        std::uint64_t next_query = 0, expected_query = 0, query_page = 0, expected_page = 0;
        while (true)
        {
            while (cursor != points.cend() && !candidate(destination, cursor->second))
                ++cursor;
            if (!any(cursor != points.cend()))
                break;
            const auto cap = credit();
            std::vector<Pending> pending;
            Writer outgoing;
            int peer = -1;
            const auto first_query = next_query;
            collective(*comm, [&]
                {
                    if (cursor != points.cend())
                    {
                        if (cap <= header_bytes + 8)
                            throw std::invalid_argument("Insufficient exact point query page credit.");
                        peer = destination;
                        outgoing = writer(Kind::Points, first_query, query_page++);
                        ++statistics.point_query_pages;
                        admit_writer(outgoing, cap);
                        outgoing.word(0);
                        std::size_t count = 0;
                        while (cursor != points.cend())
                        {
                            if (!candidate(destination, cursor->second))
                            {
                                ++cursor;
                                continue;
                            }
                            const auto need = 16 + cursor->first.size();
                            if (need > cap - outgoing.data.size())
                            {
                                if (!count)
                                    throw std::invalid_argument("An exact point key exceeds page credit.");
                                break;
                            }
                            outgoing.word(next_query++);
                            outgoing.bytes(cursor->first);
                            budget.append_capacity(pending);
                            pending.push_back({cursor, 0, 0, false, false});
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
            collective(*comm, [&]
                {
                    if (!incoming.packets.empty())
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                            throw std::invalid_argument("Foreign exact point query page.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::Points, source);
                        const auto n = r.word();
                        if (frame.query != expected_query || frame.page != expected_page++ || !frame.final ||
                            !n || n > r.data.size() / 16)
                            throw std::invalid_argument("Missing, duplicate or invalid exact point query page.");
                        for (std::uint64_t i = 0; i < n; ++i)
                        {
                            const auto id = r.word();
                            if (id != expected_query++)
                                throw std::invalid_argument("Nonsequential exact point query identity.");
                            const auto bytes = r.bytes_view();
                            budget.add(bytes.size());
                            AdaptiveBytes key(bytes.begin(), bytes.end());
                            if (key.capacity() > bytes.size()) budget.add(key.capacity() - bytes.size());
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
                collective(*comm, [&]
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
                                    need += 24 + key_wire_size(record.key) + PLIC::encoded_identity_size(rows[occurrence.row].points[occurrence.point].physical);
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
                                    const auto& physical = rows[occurrence.row].points[occurrence.point].physical;
                                    const auto key_size = key_wire_size(record.key);
                                    const auto point_size = PLIC::encoded_identity_size(physical);
                                    if (key_size > (budget.limit - budget.used) / 3 ||
                                        point_size > (budget.limit - budget.used - 3 * key_size) / 4)
                                        throw std::invalid_argument("Spatial fan encoding exceeds live byte budget.");
                                    const auto scratch = 3 * key_size + 4 * point_size;
                                    budget.add(scratch);
                                    { auto key = encode_adaptive_key(record.key); outgoing.bytes(key); }
                                    outgoing.word(record.revision);
                                    { auto point = PLIC::encode_identity(physical); outgoing.bytes(point); }
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
                collective(*comm, [&]
                    {
                        if (!incoming.packets.empty())
                        {
                            if (received_final || incoming.packets.size() != 1 || incoming.packets.front().peer != destination)
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
                                if (request.done || index != request.next || (!total && present) || (total && !present) ||
                                    (request.started && total != request.total))
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
                                auto physical = PLIC::decode_identity<ContactPointIdentity>(encoded_point, budget.limit - budget.used);
                                const auto record = add_record(key, revision, destination, request.point->second.domain);
                                auto found = remote_points.find(request.point->first);
                                if (found == remote_points.end())
                                {
                                    budget.add(sizeof(decltype(remote_points)::value_type) + map_links + request.point->first.size());
                                    found = remote_points.try_emplace(request.point->first).first;
                                }
                                for (const auto& existing : found->second)
                                    if (existing.record == record)
                                        throw std::invalid_argument("Duplicate cell at an exact normalized contact point.");
                                // A key cannot be simultaneously local and remote.
                                if (records[record].owner == rank)
                                    throw std::invalid_argument("Remote contact fan claims a local cell.");
                                budget.append_capacity(found->second);
                                found->second.push_back({record, std::move(physical)});
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
            collective(*comm, [&]
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
        std::size_t cursor = 0;
        std::uint64_t next_query = 0, expected_query = 0, page = 0, expected_page = 0;
        std::vector<unsigned char> served;
        collective(*comm, [&]
            {
                budget.add(records.size());
                served.resize(records.size(), 0);
                if (served.capacity() > records.size())
                    budget.add(served.capacity() - records.size());
            });
        while (true)
        {
            while (cursor < records.size() && (records[cursor].owner != destination || records[cursor].contact))
                ++cursor;
            if (!any(cursor < records.size()))
                break;
            const auto cap = credit();
            const auto first_query = next_query;
            std::vector<std::size_t> pending;
            Writer outgoing;
            int peer = -1;
            collective(*comm, [&]
                {
                    if (cursor < records.size())
                    {
                        if (cap <= header_bytes + 8)
                            throw std::invalid_argument("Insufficient contact record request page credit.");
                        peer = destination;
                        outgoing = writer(Kind::Records, first_query, page++);
                        admit_writer(outgoing, cap);
                        outgoing.word(0);
                        std::size_t count = 0;
                        while (cursor < records.size())
                        {
                            const auto& record = records[cursor];
                            if (record.owner != destination || record.contact)
                            {
                                ++cursor;
                                continue;
                            }
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
                            { const auto key = encode_adaptive_key(record.key); outgoing.bytes(key); }
                            budget.used -= 3 * key_size;
                            outgoing.word(record.revision);
                            ++statistics.record_fetches;
                            budget.append_capacity(pending);
                            pending.push_back(cursor++);
                            ++count;
                        }
                        replace_word(outgoing.data, header_bytes, count);
                        statistics.query_bytes += outgoing.data.size();
                    }
                });
            auto incoming = exchange(peer, std::move(outgoing));
            std::vector<Request> requests;
            collective(*comm, [&]
                {
                    if (!incoming.packets.empty())
                    {
                        if (incoming.packets.size() != 1 || incoming.packets.front().peer != source)
                            throw std::invalid_argument("Foreign contact record request page.");
                        Reader r{incoming.packets.front().bytes};
                        const auto frame = read(r, Kind::Records, source);
                        const auto n = r.word();
                        if (frame.query != expected_query || frame.page != expected_page++ || !frame.final ||
                            !n || n > r.data.size() / 24)
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
                                served[found->second])
                                throw std::invalid_argument("Foreign, stale or duplicate contact record request.");
                            served[found->second] = 1;
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
                collective(*comm, [&]
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
                                const auto count = std::min(encoded.size() - encoded_offset,
                                    reply_cap - outgoing.data.size() - 32);
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
                collective(*comm, [&]
                    {
                        if (!incoming.packets.empty())
                        {
                            if (received_final || incoming.packets.size() != 1 || incoming.packets.front().peer != destination)
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
                                    throw std::invalid_argument("Duplicate, truncated or foreign contact record chunk.");
                                if (assembly.empty())
                                {
                                    budget.add(total);
                                    assembly.reserve(total);
                                    if (assembly.capacity() != total)
                                        throw std::invalid_argument("Contact record assembly capacity differs from admission.");
                                    assembly_total = total;
                                }
                                assembly.insert(assembly.end(), bytes.begin(), bytes.end());
                                if (assembly.size() == assembly_total)
                                {
                                    auto& record = records[pending[receive_cursor]];
                                    if (record.contact || record.owner != destination)
                                        throw std::invalid_argument("Duplicate or conflicting contact record completion.");
                                    auto decoded = decode_contact(assembly, budget, identities);
                                    if (decoded.key != record.key || decoded.domain != record.domain)
                                        throw std::invalid_argument("Contact record reply key/domain differs from exact match.");
                                    decoded.identity = pending[receive_cursor];
                                    record.contact = std::move(decoded);
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
            collective(*comm, [&]
                {
                    if (!received_final || receive_cursor != pending.size() || !assembly.empty())
                        throw std::invalid_argument("Missing contact record terminal completion.");
                });
            const auto scratch = pending.capacity() * sizeof(std::size_t) + requests.capacity() * sizeof(Request);
            std::vector<std::size_t>().swap(pending);
            std::vector<Request>().swap(requests);
            budget.used -= scratch;
        }
        const auto served_bytes = served.capacity();
        std::vector<unsigned char>().swap(served);
        budget.used -= served_bytes;
    }
}

std::vector<AdaptiveMark> DistributedAdaptiveContactHalo::Impl::evaluate()
{
    if (!evaluated)
    {
        struct Occurrence
        {
            std::size_t record;
            const ContactPointIdentity* physical;
        };
        struct PhysicalPair
        {
            const ContactPointIdentity* first;
            const ContactPointIdentity* second;
        };
        struct Pair
        {
            std::vector<PhysicalPair> points;
        };
        // Record IDs are lossless interned identities: the catalogue binds each
        // ID to its full typed key, owner, revision and session context. No hash
        // or truncated key establishes pair identity.
        std::map<std::pair<std::size_t, std::size_t>, Pair> pairs;
        collective(*comm, [&]
            {
                for (const auto& [key, group] : points)
                {
                    std::vector<Occurrence> fan;
                    for (const auto& occurrence : group.local)
                    {
                        budget.append_capacity(fan);
                        fan.push_back({occurrence.record, &rows[occurrence.row].points[occurrence.point].physical});
                    }
                    if (const auto found = remote_points.find(key); found != remote_points.end())
                        for (const auto& occurrence : found->second)
                        {
                            budget.append_capacity(fan);
                            fan.push_back({occurrence.record, &occurrence.physical});
                        }
                    for (std::size_t first = 0; first < fan.size(); ++first)
                        for (std::size_t second = 0; second < first; ++second)
                        {
                            const auto& a = fan[first];
                            const auto& b = fan[second];
                            if (a.record == b.record)
                                throw std::invalid_argument("Duplicate cell in complete normalized contact fan.");
                            if (records[a.record].owner != rank && records[b.record].owner != rank)
                                continue;
                            if (statistics.logical_contact_pairs == std::numeric_limits<std::size_t>::max())
                                throw std::invalid_argument("Logical closed-contact pair count overflows.");
                            ++statistics.logical_contact_pairs;
                            const auto ids = std::minmax(a.record, b.record);
                            const std::pair<std::size_t, std::size_t> pair_key{ids.first, ids.second};
                            auto found = pairs.find(pair_key);
                            if (found == pairs.end())
                            {
                                budget.add(sizeof(decltype(pairs)::value_type) + map_links);
                                found = pairs.try_emplace(pair_key).first;
                            }
                            if (!records[a.record].contact || !records[b.record].contact)
                                throw std::invalid_argument("Incomplete exact-match contact record cache.");
                            if (!records[a.record].contact->traces_point_independent ||
                                !records[b.record].contact->traces_point_independent)
                            {
                                budget.append_capacity(found->second.points);
                                found->second.points.push_back(a.record < b.record ?
                                    PhysicalPair{a.physical, b.physical} : PhysicalPair{b.physical, a.physical});
                            }
                        }
                    const auto bytes = fan.capacity() * sizeof(Occurrence);
                    std::vector<Occurrence>().swap(fan);
                    budget.used -= bytes;
                }
                statistics.unique_cell_pairs = pairs.size();
                auto charge_work = [&](std::size_t count)
                {
                    if (count > options.contact_budget - statistics.work)
                        throw std::invalid_argument("Spatial closure cumulative comparison budget exhausted.");
                    statistics.work += count;
                };
                auto require = [&](Record& record, const std::optional<AdaptiveSplit>& split)
                {
                    if (record.owner == rank && split)
                        join_closure_requirement(*record.contact, record.required, std::uint64_t(*split));
                };
                // Numerical work starts only after the complete unique pair
                // table has been built. Repeated points never rerun span work.
                for (const auto& [ids, pair] : pairs)
                {
                    auto& a = records[ids.first];
                    auto& b = records[ids.second];
                    charge_work(1);
                    ++statistics.span_pair_evaluations;
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        if (a.owner == rank && span_exceeds_ratio(a.contact->spans[axis], b.contact->spans[axis], options.maximum_span_ratio))
                            join_closure_requirement(*a.contact, a.required, a.contact->factors[axis]);
                        if (b.owner == rank && span_exceeds_ratio(b.contact->spans[axis], a.contact->spans[axis], options.maximum_span_ratio))
                            join_closure_requirement(*b.contact, b.required, b.contact->factors[axis]);
                    }
                    auto traces = [&](const ContactPointIdentity& first_point, const ContactPointIdentity& second_point)
                    {
                        if (!b.contact->traces.empty() && a.contact->traces.size() >
                            (options.contact_budget - statistics.work) / b.contact->traces.size())
                            throw std::invalid_argument("Spatial closure cumulative trace budget exhausted.");
                        const auto count = a.contact->traces.size() * b.contact->traces.size();
                        charge_work(count);
                        statistics.trace_pair_evaluations += count;
                        for (const auto& first : a.contact->traces)
                            for (const auto& second : b.contact->traces)
                            {
                                const auto requirement = trace_pair_requirements(first, second, first_point,
                                    second_point, options.maximum_trace_jump);
                                require(a, requirement.first);
                                require(b, requirement.second);
                            }
                    };
                    if (a.contact->traces_point_independent && b.contact->traces_point_independent)
                    {
                        // These predicates never inspect their point arguments.
                        // Use a retained physical incidence, not a fabricated
                        // display point, to keep that contract explicit.
                        const auto& point = rows.front().points.front().physical;
                        traces(point, point);
                    }
                    else
                        for (const auto& physical : pair.points)
                            traces(*physical.first, *physical.second);
                }
                std::size_t bytes = pairs.size() * (sizeof(decltype(pairs)::value_type) + map_links);
                for (const auto& [key, pair] : pairs)
                    bytes += pair.points.capacity() * sizeof(PhysicalPair);
                pairs.clear();
                budget.used -= bytes;
                evaluated = true;
            });
    }
    std::vector<AdaptiveMark> result;
    collective(*comm, [&]
        {
            for (const auto& record : records)
                if (record.owner == rank && record.required)
                {
                    budget.append_capacity(result);
                    budget.add(key_capacity(record.key));
                    result.push_back({record.key, *record.required});
                }
        });
    // The returned vector transfers to the caller. Its construction overlap is
    // observed here, while subsequent calls do not retain prior result owners.
    std::size_t bytes = result.capacity() * sizeof(AdaptiveMark);
    for (const auto& mark : result)
        bytes += key_capacity(mark.key);
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
    collective(*comm, [&]
        {
            if (sizeof(Impl) > options.byte_budget)
                throw std::invalid_argument("Spatial closure session exceeds its byte budget.");
            impl_ = std::make_unique<Impl>(kernel, comm, options, context, std::move(before_send));
        });
    impl_->initialize_collective();
}
DistributedAdaptiveContactHalo::~DistributedAdaptiveContactHalo() = default;
DistributedAdaptiveContactHalo::DistributedAdaptiveContactHalo(DistributedAdaptiveContactHalo&&) noexcept = default;
DistributedAdaptiveContactHalo& DistributedAdaptiveContactHalo::operator=(DistributedAdaptiveContactHalo&&) noexcept = default;

void DistributedAdaptiveContactHalo::bootstrap(std::span<const AdaptiveLeaf> owned, std::vector<ContactIncidenceRow> rows)
{
    auto& state = *impl_;
    collective(*state.comm, [&]
        {
            if (state.ready || owned.size() != rows.size())
                throw std::invalid_argument("Invalid or repeated spatial contact bootstrap.");
            state.budget.add(rows.capacity() * sizeof(ContactIncidenceRow));
            for (const auto& row : rows)
            {
                state.budget.add(key_capacity(row.key) + row.domain.capacity() +
                    row.points.capacity() * sizeof(ContactPointIncidence));
                if (row.domain.empty() || row.points.empty())
                    throw std::invalid_argument("Incomplete spatial contact incidence row.");
            }
            state.rows = std::move(rows);
            state.diagnostics.cells = owned.size();
            std::map<AdaptiveCellKey, const AdaptiveLeaf*> source;
            for (const auto& leaf : owned)
            {
                state.budget.add(sizeof(decltype(source)::value_type) + map_links + key_capacity(leaf.key));
                if (!source.emplace(leaf.key, &leaf).second)
                    throw std::invalid_argument("Duplicate spatial source cell key.");
                if (leaf.geometry)
                    state.identities.retain(state.budget, leaf.geometry);
            }
            for (std::size_t row_index = 0; row_index < state.rows.size(); ++row_index)
            {
                const auto& row = state.rows[row_index];
                const auto found = source.find(row.key);
                if (found == source.end() || found->second->revision != row.revision || found->second->domain != row.domain)
                    throw std::invalid_argument("Spatial incidence row disagrees with its source owner.");
                const auto id = state.add_record(row.key, row.revision, state.rank, row.domain);
                auto record = prepare_contact(*found->second, state.kernel, state.budget, state.identities,
                    state.diagnostics.trace_records, state.diagnostics.transient_geometry_peak_bytes);
                record.identity = id;
                state.records[id].contact = std::move(record);
                ++state.statistics.owned_records;
                ++state.diagnostics.geometry_records;
                state.diagnostics.unique_record_bytes += sizeof(ContactRecord) + key_capacity(row.key) + row.domain.capacity();
                for (std::size_t point_index = 0; point_index < row.points.size(); ++point_index)
                {
                    const auto& point = row.points[point_index];
                    const auto maximum = 16 + row.domain.size() + PLIC::canonical_key_size(point.normalized);
                    if (maximum > (state.budget.limit - state.budget.used) / 4)
                        throw std::invalid_argument("Spatial point key exceeds live byte budget.");
                    state.budget.add(4 * maximum);
                    auto key = point_key(row.domain, point.normalized);
                    const auto capacity = key.capacity();
                    auto position = state.points.find(key);
                    if (position == state.points.end())
                    {
                        state.budget.add(sizeof(decltype(state.points)::value_type) + map_links);
                        auto coordinate = coordinates(point.normalized);
                        for (const auto scalar : coordinate)
                            if (!std::isfinite(scalar))
                                throw std::invalid_argument("Nonfinite exact contact coordinate.");
                        position = state.points.emplace(std::move(key), Impl::PointGroup{row.domain, coordinate, {}}).first;
                        state.budget.used -= 4 * maximum - capacity;
                    }
                    else
                    {
                        AdaptiveBytes().swap(key);
                        state.budget.used -= 4 * maximum;
                    }
                    for (const auto& occurrence : position->second.local)
                        if (occurrence.record == id)
                            throw std::invalid_argument("Duplicate cell incidence at a normalized closure point.");
                    state.budget.append_capacity(position->second.local);
                    position->second.local.push_back({id, row_index, point_index});
                    ++state.diagnostics.contacts;
                }
            }
            std::size_t source_bytes = source.size() * (sizeof(decltype(source)::value_type) + map_links);
            for (const auto& [key, leaf] : source)
                source_bytes += key_capacity(key);
            source.clear();
            state.budget.used -= source_bytes;

            std::size_t staging = sizeof(ClosedContactEntry) + sizeof(AdaptiveContactImage);
            for (const auto& row : state.rows)
                staging = std::max(staging, sizeof(ClosedContactEntry) + sizeof(AdaptiveContactImage) +
                    key_capacity(row.key) + row.domain.size());
            state.budget.add(staging);
            const auto resident = state.budget.used;
            state.index = std::make_unique<ClosedContactIndex>(state.budget.limit - resident);
            for (const auto& row : state.rows)
            {
                Bounds bounds;
                for (const auto& point : row.points)
                {
                    const auto coordinate = coordinates(point.normalized);
                    include(bounds, {coordinate, coordinate});
                }
                AdaptiveContactImage image;
                image.bounds = bounds.box;
                state.index->insert({row.key, row.domain, row.revision, state.rank, {image}});
                include(state.local_cover, bounds.box);
            }
            const auto index_statistics = state.index->statistics();
            state.budget.peak = std::max(state.budget.peak, resident + index_statistics.peak_bytes);
            state.budget.add(index_statistics.retained_bytes);
            const auto index_spent = index_statistics.admitted_bytes - index_statistics.retained_bytes;
            if (index_spent > std::numeric_limits<std::size_t>::max() - state.budget.cumulative)
                throw std::invalid_argument("Spatial index cumulative admission overflows.");
            state.budget.cumulative += index_spent;
            state.budget.used -= staging;
            state.covers = state.index->ownership_covers(8, state.budget.limit - state.budget.used);
            state.budget.peak = std::max(state.budget.peak, resident + state.index->statistics().peak_bytes);
            const auto before_cover_admission = state.budget.cumulative;
            state.budget.add(state.covers.capacity() * sizeof(ClosedContactCover));
            for (const auto& cover : state.covers)
                state.budget.add(cover.domain.capacity());
            const auto cover_bytes = state.budget.cumulative - before_cover_admission;
            const auto cover_spent = state.index->statistics().admitted_bytes - index_statistics.admitted_bytes;
            if (cover_spent < cover_bytes || cover_spent - cover_bytes >
                    std::numeric_limits<std::size_t>::max() - state.budget.cumulative)
                throw std::invalid_argument("Spatial cover cumulative admission overflows or disagrees.");
            state.budget.cumulative += cover_spent - cover_bytes;
        });
    state.build_directory();
    state.discover_neighbors();
    state.discover_points();
    state.fetch_records();
    state.ready = true;
}

std::vector<AdaptiveMark> DistributedAdaptiveContactHalo::required_refinements()
{
    auto& state = *impl_;
    collective(*state.comm, [&]
        {
            if (!state.ready)
                throw std::invalid_argument("Spatial closure session has no complete bootstrap.");
        });
    return state.evaluate();
}
const ContactHaloStatistics& DistributedAdaptiveContactHalo::statistics() const noexcept
{
    impl_->statistics.retained_bytes = impl_->budget.used;
    impl_->statistics.peak_bytes = impl_->budget.peak;
    impl_->statistics.cumulative_bytes = impl_->budget.cumulative;
    return impl_->statistics;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
