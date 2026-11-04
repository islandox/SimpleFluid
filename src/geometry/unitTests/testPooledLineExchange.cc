/** @file testPooledLineExchange.cc
 * @brief Literal closed-segment results and collective reply-completeness regressions.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/PooledLineExchange.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
using Access = P::details::GeometryIdentityAccess;
using Packet = SimpleFluid::OwnerPacket;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
constexpr std::size_t ample = 8 * 1024 * 1024, full_page = 64 * 1024;
constexpr std::uint64_t long_query = 10, short_query = 20, probe_query = 30;
using Wide = std::array<long double, 3>;
using Results = std::map<std::uint64_t, std::vector<Wide>>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

P::PointIdentity<ordinary> point(long double x)
{
    return Access::point(Wide{x, 0, 0});
}
AMR::AdaptiveBytes line_key()
{
    // The literal x axis is independent of the queried endpoint set.
    D::Writer key;
    const AMR::AdaptiveBytes domain{'d'};
    key.bytes(domain);
    key.bytes(P::canonical_key(Access::line({0, {1, 0, 0}, {0, 0, 0}})));
    return std::move(key.data);
}
AMR::AdaptiveBytes support_key(std::uint64_t query)
{
    AMR::AdaptiveBytes key{'s'};
    for (unsigned byte = 0; byte < 8; ++byte)
        key.push_back(char(query >> (8 * byte)));
    return key;
}
D::ContactHaloContext context(std::uint64_t round)
{
    return {7, 11, 13, round, 17};
}
int other_resident()
{
    return std::min(1, Tpetra::getDefaultComm()->getSize() - 1);
}
int global_sum(int local)
{
    int result = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}
struct LineState
{
    D::Budget budget{ample};
    D::IdentityBudget identities;
    D::IncidencePools pools;
    D::PooledLineRegistry registry;

    LineState() : pools(budget, identities), registry(budget, pools) {}
    explicit LineState(const LineState& source)
        : pools(source.pools, budget, identities), registry(source.registry, budget, pools)
    {
    }
};
struct Segment
{
    std::uint64_t query;
    int resident;
    long double first, last;
    bool remove = false;
    std::optional<std::uint64_t> support_query;
    std::uint64_t edge = 3;
};
Results exchange(LineState& state, std::initializer_list<Segment> segments, std::uint64_t round, bool bootstrap = false,
    std::size_t page = full_page, const D::PooledLineBeforeSend& hook = {})
{
    const auto comm = Tpetra::getDefaultComm();
    const auto line = line_key();
    std::vector<D::PooledLineEdit> edits;
    std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> declared;
    for (const auto& segment : segments)
        declared.emplace(segment.query, std::pair{segment.support_query.value_or(segment.query), segment.edge});
    D::collective(*comm,
        [&]
        {
            state.registry.begin_replay(bootstrap);
            for (const auto& segment : segments)
            {
                if (segment.resident != comm->getRank())
                    continue;
                const auto support = support_key(segment.support_query.value_or(segment.query));
                D::PooledLineSubscriber subscriber;
                subscriber.support = state.pools.supports.intern(support);
                subscriber.first = state.pools.points.intern(point(segment.first));
                subscriber.last = state.pools.points.intern(point(segment.last));
                subscriber.resident = segment.resident;
                subscriber.support_authority = D::authority(support, comm->getSize());
                subscriber.emission = segment.query;
                subscriber.image = 0;
                subscriber.edge = segment.edge;
                edits.push_back({state.pools.lines.intern(line), std::move(subscriber), segment.remove});
            }
        });
    Results result;
    std::size_t cursor = 0;
    D::exchange_pooled_line_edits(
        state.registry, state.pools,
        [&](D::PooledLineEdit& edit)
        {
            if (cursor == edits.size())
                return false;
            edit = edits[cursor++];
            return true;
        },
        0, context(round), comm, state.budget, page,
        [&](std::size_t received_line, const D::PooledLineSubscriber& subscriber)
        {
            EXPECT_EQ(state.pools.lines.get(received_line), line);
            const auto declared_query = declared.find(subscriber.emission);
            const auto support = declared_query == declared.end() ? subscriber.emission : declared_query->second.first;
            const auto edge = declared_query == declared.end() ? 3 : declared_query->second.second;
            EXPECT_EQ(state.pools.supports.get(subscriber.support), support_key(support));
            EXPECT_EQ(subscriber.resident, comm->getRank());
            EXPECT_EQ(subscriber.support_authority, D::authority(support_key(support), comm->getSize()));
            EXPECT_EQ(subscriber.image, 0u);
            EXPECT_EQ(subscriber.edge, edge);
            std::vector<Wide> coordinates;
            for (const auto slot : subscriber.result)
                coordinates.push_back(Access::point(state.pools.points.get(slot)));
            EXPECT_TRUE(result.emplace(subscriber.emission, std::move(coordinates)).second)
                << "A complete query must be published once.";
        },
        hook);
    D::collective(*comm, [&] { state.registry.finish_replay(0); });
    return result;
}
Results expected(std::initializer_list<std::pair<int, std::pair<std::uint64_t, std::vector<Wide>>>> values)
{
    Results result;
    for (const auto& [resident, query] : values)
        if (resident == Tpetra::getDefaultComm()->getRank())
            result.emplace(query);
    return result;
}
Results initial_results()
{
    return expected({{0, {long_query, {{0, 0, 0}, {2, 0, 0}}}}});
}
Results overlapping_results()
{
    // Closed-segment intersections are literal, sorted coordinate lists. Pool
    // slots, numerical queries and packet decoders do not derive this oracle.
    return expected({{0, {long_query, {{0, 0, 0}, {.5L, 0, 0}, {1.5L, 0, 0}, {2, 0, 0}}}},
        {other_resident(), {short_query, {{.5L, 0, 0}, {1.5L, 0, 0}}}}});
}
void bootstrap(LineState& source)
{
    const auto comm = Tpetra::getDefaultComm();
    D::collective(*comm,
        [&]
        {
            // Deliberately differ from coordinate order and between ranks.
            // Unsubscribed pooled endpoints must not enter the line query.
            const std::array<long double, 4> order = comm->getRank() % 2 ? std::array<long double, 4>{0, 2, .5L, 1.5L}
                                                                         : std::array<long double, 4>{2, 0, 1.5L, .5L};
            for (const auto x : order)
                (void) source.pools.points.intern(point(x));
        });
    EXPECT_EQ(exchange(source, {{long_query, 0, 0, 2}}, 0, true), initial_results());
}
struct Snapshot
{
    std::size_t used, cumulative, registry_bytes, processed, support_slots, line_slots;
    std::vector<Wide> points;
    bool operator==(const Snapshot&) const = default;
};
Snapshot snapshot(const LineState& state)
{
    Snapshot result{state.budget.used, state.budget.cumulative, state.registry.retained_bytes(),
        state.registry.processed_lines(), state.pools.supports.size(), state.pools.lines.size(), {}};
    for (std::size_t slot = 0; slot < state.pools.points.size(); ++slot)
        result.points.push_back(Access::point(state.pools.points.get(slot)));
    return result;
}
void verify_accepted_and_retry(const LineState& source, const Snapshot& before, std::size_t page = full_page,
    const D::PooledLineBeforeSend& hook = {})
{
    EXPECT_EQ(snapshot(source), before);
    {
        LineState probe(source);
        // If the failed candidate mutated the accepted line's endpoints or
        // cached long-query result, this probe would publish extra/different points.
        EXPECT_EQ(exchange(probe, {{probe_query, 0, 0, 2}}, 2), expected({{0, {probe_query, {{0, 0, 0}, {2, 0, 0}}}}}));
    }
    EXPECT_EQ(snapshot(source), before);
    {
        LineState retry(source);
        EXPECT_EQ(
            exchange(retry, {{short_query, other_resident(), .5L, 1.5L}}, 3, false, page, hook), overlapping_results());
    }
    EXPECT_EQ(snapshot(source), before);
}
std::size_t common_wire_bytes(std::uint64_t query)
{
    // Independently count the documented protocol fields: nine header words,
    // framed line, resident/authority, framed support, emission/image/edge.
    constexpr std::size_t word = sizeof(std::uint64_t);
    return 9 * word + word + line_key().size() + 2 * word + word + support_key(query).size() + 3 * word;
}
std::size_t point_field_bytes(long double x)
{
    // Ordinary canonical scalar encodings have value-dependent lengths.
    return sizeof(std::uint64_t) + P::encode_identity(point(x)).size();
}
std::size_t fragmented_page()
{
    constexpr std::size_t word = sizeof(std::uint64_t);
    std::size_t largest_point = 0;
    for (const auto x : {0.L, .5L, 1.5L, 2.L})
        largest_point = std::max(largest_point, point_field_bytes(x));
    std::size_t packet = 0;
    for (const auto& segment :
        std::array{Segment{long_query, 0, 0, 2}, Segment{short_query, 0, .5L, 1.5L}, Segment{probe_query, 0, 0, 2}})
    {
        const auto common = common_wire_bytes(segment.query);
        const auto edit = common + word + point_field_bytes(segment.first) + point_field_bytes(segment.last);
        const auto two_point_reply = common + 3 * word + 2 * largest_point;
        packet = std::max({packet, edit, two_point_reply});
    }
    return sizeof(Packet) + packet;
}
std::size_t complete_long_reply_bytes()
{
    std::size_t result = sizeof(Packet) + common_wire_bytes(long_query) + 3 * sizeof(std::uint64_t);
    for (const auto x : {0.L, .5L, 1.5L, 2.L})
        result += point_field_bytes(x);
    return result;
}
struct Reply
{
    std::uint64_t query, total, begin, count;
};
Reply reply(const Packet& packet)
{
    D::Reader reader{packet.bytes};
    for (unsigned word = 0; word < 9; ++word)
        (void) reader.word();
    (void) reader.bytes_view(); // line
    (void) reader.word();       // resident
    (void) reader.word();       // support authority
    (void) reader.bytes_view(); // support
    const auto query = reader.word();
    (void) reader.word(); // image
    (void) reader.word(); // edge
    return {query, reader.word(), reader.word(), reader.word()};
}
void replace_word(AMR::AdaptiveBytes& bytes, std::size_t word, std::uint64_t value)
{
    for (unsigned byte = 0; byte < 8; ++byte)
        bytes.at(word * 8 + byte) = char(value >> (8 * byte));
}
enum class Fault
{
    OmitWhole,
    DuplicateComplete,
    ForeignWave,
    StaleContext,
    TruncatePartial,
    OmitFinalPartial
};
std::string_view rejection_reason(Fault fault)
{
    switch (fault)
    {
        case Fault::OmitWhole:
            return "Incomplete changed line-query delivery";
        case Fault::DuplicateComplete:
            return "Repeated completed pooled line query";
        case Fault::ForeignWave:
        case Fault::StaleContext:
            return "Invalid pooled line context";
        case Fault::TruncatePartial:
            return "Invalid distributed adaptive message length";
        case Fault::OmitFinalPartial:
            return "Missing terminal pooled line reply chunk";
    }
    throw std::logic_error("Unknown test reply fault.");
}
void reject_fault(Fault fault)
{
    LineState source;
    bootstrap(source);
    const auto before = snapshot(source);
    const bool partial = fault == Fault::TruncatePartial || fault == Fault::OmitFinalPartial;
    const auto page = partial ? fragmented_page() : full_page;
    int injected = 0, rejected = 0, matching_reason = 0;
    {
        LineState candidate(source);
        const D::PooledLineBeforeSend hook = [&](unsigned layer, std::uint64_t wave, std::vector<Packet>& packets)
        {
            if (injected)
                return;
            for (std::size_t i = 0; i < packets.size(); ++i)
            {
                const auto item = reply(packets[i]);
                if (item.query != long_query ||
                    (fault == Fault::OmitFinalPartial && (!item.begin || item.begin + item.count != item.total)))
                    continue;
                EXPECT_EQ(layer, 0u);
                EXPECT_EQ(item.total, 4u);
                if (partial)
                    EXPECT_LT(item.count, item.total);
                else
                {
                    EXPECT_EQ(item.begin, 0u);
                    EXPECT_EQ(item.count, item.total);
                }
                switch (fault)
                {
                    case Fault::OmitWhole:
                    case Fault::OmitFinalPartial:
                        packets.erase(packets.begin() + i);
                        break;
                    case Fault::DuplicateComplete: {
                        auto duplicate = packets[i];
                        packets.push_back(std::move(duplicate));
                        break;
                    }
                    case Fault::ForeignWave:
                        replace_word(packets[i].bytes, 7, wave + 1);
                        break;
                    case Fault::StaleContext:
                        replace_word(packets[i].bytes, 4, 0);
                        break;
                    case Fault::TruncatePartial:
                        packets[i].bytes.pop_back();
                        break;
                }
                injected = 1;
                return;
            }
        };
        try
        {
            (void) exchange(candidate, {{short_query, other_resident(), .5L, 1.5L}}, 1, false, page, hook);
        }
        catch (const std::invalid_argument& error)
        {
            rejected = 1;
            matching_reason = std::string_view(error.what()).find(rejection_reason(fault)) != std::string_view::npos;
        }
        catch (const std::exception& error)
        {
            ADD_FAILURE() << "Unexpected exception type: " << error.what();
        }
    }
    int all_rejected = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    EXPECT_EQ(global_sum(injected), 1) << "Fault must mutate one authority's reply packet.";
    EXPECT_EQ(all_rejected, 1);
    EXPECT_GE(global_sum(matching_reason), 1) << rejection_reason(fault);
    verify_accepted_and_retry(source, before, page);
}

constexpr std::size_t bulk_queries = 512, cursor_credit = 256 * 1024;
constexpr std::uint64_t bulk_first_query = 1000;
struct CursorStorage
{
    D::Budget& budget;
    std::vector<char> bytes;
    std::size_t charge = 0;
    ~CursorStorage()
    {
        std::vector<char>().swap(bytes);
        budget.used -= charge;
    }
    void leave_credit(std::size_t free_bytes)
    {
        const auto available = budget.limit - budget.used;
        if (available <= free_bytes)
            throw std::logic_error("Cursor fixture has no retained allocation to create.");
        const auto requested = available - free_bytes;
        budget.add(requested);
        charge = requested;
        bytes.resize(requested, char(0x5a));
        if (bytes.capacity() != requested)
            throw std::logic_error("Cursor fixture capacity differs from its live charge.");
    }
};
struct BulkObservation
{
    std::size_t cursor_calls = 0, received = 0, retained_cursor_bytes = 0;
    std::size_t selected_page = 0, free_before_cursor = 0, free_after_cursor = 0;
    int reply_pages = 0;
};
BulkObservation exchange_after_cursor_growth(LineState& state, std::size_t remaining_credit)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto line = line_key(), support = support_key(long_query);
    D::PooledLineEdit prototype;
    D::collective(*comm,
        [&]
        {
            state.registry.begin_replay(false);
            prototype.line = state.pools.lines.intern(line);
            prototype.subscriber.support = state.pools.supports.intern(support);
            prototype.subscriber.first = state.pools.points.intern(point(0));
            prototype.subscriber.last = state.pools.points.intern(point(2));
            prototype.subscriber.resident = comm->getRank();
            prototype.subscriber.support_authority = D::authority(support, comm->getSize());
            prototype.subscriber.edge = 3;
        });
    BulkObservation result;
    // This caller-selected ceiling is fixed before the cursor constructs its
    // real retained buffer. It is deliberately not a live-credit algorithm.
    result.selected_page = 256 * 1024;
    result.free_before_cursor = state.budget.limit - state.budget.used;
    EXPECT_GT(result.free_before_cursor, result.selected_page);
    CursorStorage retained{state.budget};
    std::array<bool, bulk_queries> seen{};
    std::size_t cursor = comm->getRank();
    D::exchange_pooled_line_edits(
        state.registry, state.pools,
        [&](D::PooledLineEdit& edit)
        {
            ++result.cursor_calls;
            if (result.cursor_calls == 1)
            {
                retained.leave_credit(remaining_credit);
                result.retained_cursor_bytes = retained.bytes.capacity();
                result.free_after_cursor = state.budget.limit - state.budget.used;
            }
            if (cursor >= bulk_queries)
                return false;
            edit = prototype;
            edit.subscriber.emission = bulk_first_query + cursor;
            cursor += comm->getSize();
            return true;
        },
        0, context(4), comm, state.budget, result.selected_page,
        [&](std::size_t received_line, const D::PooledLineSubscriber& subscriber)
        {
            EXPECT_EQ(state.pools.lines.get(received_line), line);
            EXPECT_EQ(state.pools.supports.get(subscriber.support), support);
            EXPECT_EQ(subscriber.resident, comm->getRank());
            EXPECT_EQ(subscriber.support_authority, D::authority(support, comm->getSize()));
            EXPECT_EQ(subscriber.image, 0u);
            EXPECT_EQ(subscriber.edge, 3u);
            if (subscriber.emission < bulk_first_query || subscriber.emission - bulk_first_query >= bulk_queries)
            {
                ADD_FAILURE() << "Unrequested bulk query receipt: " << subscriber.emission;
                return;
            }
            const auto ordinal = std::size_t(subscriber.emission - bulk_first_query);
            EXPECT_EQ(ordinal % std::size_t(comm->getSize()), std::size_t(comm->getRank()));
            EXPECT_FALSE(seen[ordinal]);
            seen[ordinal] = true;
            ++result.received;
            EXPECT_EQ(subscriber.result.size(), 2u);
            if (subscriber.result.size() == 2)
            {
                EXPECT_EQ(Access::point(state.pools.points.get(subscriber.result[0])), (Wide{0, 0, 0}));
                EXPECT_EQ(Access::point(state.pools.points.get(subscriber.result[1])), (Wide{2, 0, 0}));
            }
        },
        [&](unsigned, std::uint64_t, std::vector<Packet>& packets) { result.reply_pages += !packets.empty(); });
    D::collective(*comm, [&] { state.registry.finish_replay(0); });
    std::size_t expected_local = 0;
    for (std::size_t ordinal = 0; ordinal < bulk_queries; ++ordinal)
    {
        const bool local = ordinal % std::size_t(comm->getSize()) == std::size_t(comm->getRank());
        EXPECT_EQ(seen[ordinal], local);
        expected_local += local;
    }
    EXPECT_EQ(result.cursor_calls, expected_local + 1);
    EXPECT_EQ(result.received, expected_local);
    EXPECT_EQ(result.free_after_cursor, remaining_credit);
    EXPECT_GT(result.retained_cursor_bytes, 0u);
    EXPECT_EQ(retained.bytes.front(), char(0x5a));
    EXPECT_EQ(retained.bytes.back(), char(0x5a));
    EXPECT_LE(state.budget.used, state.budget.limit);
    return result;
}

TEST(PooledLineExchange, CursorOwnedStorageRefreshesPageCreditBeforeEncodingAndFanIn)
{
    LineState source;
    bootstrap(source);
    const auto before = snapshot(source);
    for (unsigned attempt = 0; attempt < 2; ++attempt)
    {
        SCOPED_TRACE(attempt);
        LineState candidate(source);
        BulkObservation result;
        ASSERT_NO_THROW(result = exchange_after_cursor_growth(candidate, cursor_credit));
        EXPECT_EQ(global_sum(int(result.received)), int(bulk_queries));
        EXPECT_GT(global_sum(result.reply_pages), 1);
        RecordProperty("caller_page_ceiling", int(result.selected_page));
        RecordProperty("remaining_after_cursor", int(result.free_after_cursor));
        EXPECT_EQ(snapshot(source), before);
    }
    verify_accepted_and_retry(source, before);
}

TEST(PooledLineExchange, CursorAdmissionRejectionLeavesAcceptedRegistryAndFreshRetryIntact)
{
    LineState source;
    bootstrap(source);
    const auto before = snapshot(source);
    // Independently framed complete insertion: common fields, removal flag,
    // two endpoint byte fields and its packet slot. A byte less cannot stage it.
    const auto one_edit = sizeof(Packet) + common_wire_bytes(long_query) + sizeof(std::uint64_t) +
                          point_field_bytes(0) + point_field_bytes(2);
    int rejected = 0;
    {
        LineState candidate(source);
        try
        {
            (void) exchange_after_cursor_growth(candidate, one_edit - 1);
        }
        catch (const std::invalid_argument&)
        {
            rejected = 1;
        }
    }
    int all_rejected = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
    EXPECT_EQ(all_rejected, 1);
    EXPECT_EQ(snapshot(source), before);
    {
        LineState retry(source);
        BulkObservation result;
        ASSERT_NO_THROW(result = exchange_after_cursor_growth(retry, cursor_credit));
        EXPECT_EQ(global_sum(int(result.received)), int(bulk_queries));
    }
    verify_accepted_and_retry(source, before);
}

TEST(PooledLineExchange, LiteralOverlappingSegmentsKeepCoordinateOrderAcrossCompleteAndFragmentedReplies)
{
    LineState source;
    bootstrap(source);
    const auto before = snapshot(source);
    verify_accepted_and_retry(source, before);
    const auto page = fragmented_page();
    EXPECT_LT(page, complete_long_reply_bytes());
    RecordProperty("fragmented_page_bytes", int(page));
    RecordProperty("complete_long_reply_bytes", int(complete_long_reply_bytes()));
    int chunks = 0, first_chunks = 0, final_chunks = 0;
    std::uint64_t next_point = 0;
    const D::PooledLineBeforeSend observe = [&](unsigned layer, std::uint64_t, std::vector<Packet>& packets)
    {
        for (const auto& packet : packets)
        {
            const auto item = reply(packet);
            if (item.query != long_query)
                continue;
            EXPECT_EQ(layer, 0u);
            EXPECT_EQ(item.total, 4u);
            EXPECT_EQ(item.begin, next_point);
            EXPECT_GT(item.count, 0u);
            EXPECT_LT(item.count, item.total);
            next_point += item.count;
            ++chunks;
            first_chunks += item.begin == 0;
            final_chunks += item.begin + item.count == item.total;
        }
    };
    verify_accepted_and_retry(source, before, page, observe);
    EXPECT_GE(global_sum(chunks), 2);
    EXPECT_EQ(global_sum(first_chunks), 1);
    EXPECT_EQ(global_sum(final_chunks), 1);
    EXPECT_EQ(global_sum(int(next_point)), 4);
}

TEST(PooledLineExchange, DirtyLineWithZeroChangedQueriesCompletesWithEmptyOwners)
{
    LineState source;
    bootstrap(source);
    const auto before = snapshot(source);
    LineState candidate(source);
    const auto processed = candidate.registry.processed_lines();
    int nonempty_reply_pages = 0;
    const D::PooledLineBeforeSend hook = [&](unsigned, std::uint64_t, std::vector<Packet>& packets)
    { nonempty_reply_pages += !packets.empty(); };
    // Inserting and retiring the same endpoint-only subscription dirties the
    // line without changing any surviving result. Every resident input is on
    // rank zero, including in the registered two- and four-rank executions.
    EXPECT_TRUE(
        exchange(candidate, {{probe_query, 0, 0, 2}, {probe_query, 0, 0, 2, true}}, 1, false, full_page, hook).empty());
    EXPECT_EQ(global_sum(nonempty_reply_pages), 0);
    EXPECT_EQ(global_sum(int(candidate.registry.processed_lines() - processed)), 1);
    EXPECT_TRUE(exchange(candidate, {}, 2).empty());
    verify_accepted_and_retry(source, before);
}

TEST(PooledLineExchange, MissingWholeChangedReplyRejectsCollectivelyAndFreshForkRecovers)
{
    reject_fault(Fault::OmitWhole);
}

TEST(PooledLineExchange, RepeatedCompleteReplyRejectsCollectivelyAndFreshForkRecovers)
{
    reject_fault(Fault::DuplicateComplete);
}

TEST(PooledLineExchange, ForeignReplyWaveRejectsCollectivelyAndFreshForkRecovers)
{
    reject_fault(Fault::ForeignWave);
}

TEST(PooledLineExchange, StaleReplyContextRejectsCollectivelyAndFreshForkRecovers)
{
    reject_fault(Fault::StaleContext);
}

TEST(PooledLineExchange, TruncatedPartialReplyRejectsCollectivelyAndFreshForkRecovers)
{
    reject_fault(Fault::TruncatePartial);
}

TEST(PooledLineExchange, MissingFinalPartialReplyRejectsCollectivelyAndFreshForkRecovers)
{
    reject_fault(Fault::OmitFinalPartial);
}

TEST(PooledLineExchange, WideEmissionOrdinalsKeepFullIdentityThroughForkAndRetirement)
{
    constexpr auto wide = (std::uint64_t{1} << 32) + long_query;
    constexpr auto widest = std::numeric_limits<std::uint64_t>::max() - 3;
    LineState source;
    bootstrap(source);
    const auto before = snapshot(source);
    LineState changed(source);
    EXPECT_EQ(exchange(changed,
                  {{wide, other_resident(), .5L, 1.5L, false, long_query, wide},
                      {widest, other_resident(), .5L, 1.5L, false, long_query, widest}},
                  1),
        expected({{0, {long_query, {{0, 0, 0}, {.5L, 0, 0}, {1.5L, 0, 0}, {2, 0, 0}}}},
            {other_resident(), {wide, {{.5L, 0, 0}, {1.5L, 0, 0}}}},
            {other_resident(), {widest, {{.5L, 0, 0}, {1.5L, 0, 0}}}}}));
    const auto changed_before = snapshot(changed);
    {
        LineState fork(changed);
        EXPECT_TRUE(exchange(fork, {{wide, other_resident(), .5L, 1.5L, true, long_query, wide}}, 2).empty());
        EXPECT_EQ(
            exchange(fork, {{widest, other_resident(), .5L, 1.5L, true, long_query, widest}}, 3), initial_results());
        EXPECT_EQ(snapshot(changed), changed_before);
    }
    EXPECT_EQ(snapshot(source), before);
    LineState retry(source);
    EXPECT_EQ(exchange(retry, {{wide, other_resident(), .5L, 1.5L, false, long_query, wide}}, 4),
        expected({{0, {long_query, {{0, 0, 0}, {.5L, 0, 0}, {1.5L, 0, 0}, {2, 0, 0}}}},
            {other_resident(), {wide, {{.5L, 0, 0}, {1.5L, 0, 0}}}}}));
    EXPECT_EQ(snapshot(source), before);
}
} // namespace
