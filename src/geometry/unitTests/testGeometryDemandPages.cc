/** Persistent demand order and controlled owner-union component regressions. */
#include "geometry/PLIC/details/GeometryDemandOrder.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace
{
namespace G = SimpleFluid::PLIC::details::geometry_demand;
constexpr std::size_t ample = 64 * 1024 * 1024;
struct Counter
{
    std::size_t used = 0, limit = ample;
    static void retain(void* context, std::size_t bytes)
    {
        auto& value = *static_cast<Counter*>(context);
        if (bytes > value.limit - value.used)
            throw SimpleFluid::PLIC::GeometryError(
                SimpleFluid::PLIC::GeometryErrorCode::ResourceLimit, "Focused geometry demand admission exhausted.");
        value.used += bytes;
    }
    G::Admission admission() { return {this, retain}; }
};
struct OrderRow
{
    std::uint64_t generation = 0, position = 0, key = 0;
};
struct OrderLess
{
    bool operator()(const OrderRow& a, const OrderRow& b) const noexcept
    {
        return a.generation != b.generation ? a.generation > b.generation : a.position < b.position;
    }
};
struct KeyLess
{
    bool operator()(const OrderRow& a, const OrderRow& b) const noexcept { return a.key < b.key; }
};
using Order = G::OrderedPages<OrderRow, OrderLess>;
using Keys = G::OrderedPages<OrderRow, KeyLess>;
struct Selection
{
    Order::Ptr order;
    Keys::Ptr keys;
    std::uint64_t generation = 0;

    static Selection initial(std::size_t count, Counter& counter)
    {
        std::vector<OrderRow> input;
        for (std::size_t key = 0; key < count; ++key)
            input.push_back({0, key, key});
        return {Order(counter.admission()).build(input), Keys(counter.admission()).build(input), 0};
    }
    void prepend(std::span<const std::uint64_t> wanted, Counter& counter, G::PageWork& work)
    {
        ++generation;
        Order order_editor(counter.admission(), &work);
        Keys key_editor(counter.admission(), &work);
        for (std::size_t position = 0; position < wanted.size(); ++position)
        {
            const auto key = wanted[position];
            const auto* old = Keys::lower_bound(keys, [&](const auto& row) { return row.key < key; }, &work);
            const bool known = old && old->key == key;
            const auto accepted = known ? std::optional(*old) : std::nullopt;
            if (accepted)
                order = order_editor.edit(order, *accepted, nullptr, true);
            const OrderRow current{generation, position, key};
            order = order_editor.edit(order, current, &current, false);
            keys = key_editor.edit(keys, current, &current, known);
        }
    }
    std::size_t find(std::uint64_t key) const
    {
        const auto* row = Keys::lower_bound(keys, [&](const auto& value) { return value.key < key; });
        if (!row || row->key != key)
            throw std::invalid_argument("Unknown component fixture key.");
        return Order::rank(order, [&](const auto& value) { return OrderLess{}(value, *row); });
    }
};
using Payload = std::array<char, 256>;
auto physical_metadata(const Selection& selection)
{
    std::map<const void*, std::size_t> result;
    const auto append = [&](const auto& root)
    {
        using Tree = std::conditional_t<std::is_same_v<decltype(root), const Order::Ptr&>, Order, Keys>;
        Tree::visit_pages(root,
            [&](const auto& page)
            {
                using Ptr = typename Tree::Ptr;
                result.emplace(page.get(),
                    sizeof(*page) + page->rows.capacity() * sizeof(OrderRow) + page->children.capacity() * sizeof(Ptr));
            });
    };
    append(selection.order);
    append(selection.keys);
    return result;
}
struct ThrowingDefaultRow
{
    inline static std::size_t default_calls = 0;
    int key, payload;
    ThrowingDefaultRow()
    {
        ++default_calls;
        throw std::runtime_error("Generic Row default construction must not run.");
    }
    explicit ThrowingDefaultRow(int key, int payload) noexcept : key(key), payload(payload) {}
    ThrowingDefaultRow(const ThrowingDefaultRow& other) noexcept : key(other.key), payload(other.payload) {}
    ThrowingDefaultRow& operator=(const ThrowingDefaultRow& other) noexcept
    {
        key = other.key;
        payload = other.payload;
        return *this;
    }
};
struct ThrowingDefaultLess
{
    bool operator()(const ThrowingDefaultRow& first, const ThrowingDefaultRow& second) const noexcept
    {
        return first.key < second.key;
    }
};
static_assert(std::is_nothrow_copy_constructible_v<ThrowingDefaultRow>);
static_assert(std::is_nothrow_copy_assignable_v<ThrowingDefaultRow>);
static_assert(!G::stack_safe_default<ThrowingDefaultRow>);

} // namespace

TEST(GeometryDemandPages, SmallRequestedPrefixesKeepExactOrderAndBoundedCopiedMetadata)
{
    for (const auto width : {1u, 2u, 4u})
        for (const auto direction : {0u, 1u, 2u})
        {
            SCOPED_TRACE(width);
            SCOPED_TRACE(direction);
            std::array<std::size_t, 2> allocations{}, visits{}, payload{}, new_metadata{};
            for (std::size_t scale = 0; scale < 2; ++scale)
            {
                Counter initial;
                const auto count = std::size_t{4096} << scale;
                const auto accepted = Selection::initial(count, initial);
                const auto accepted_metadata = physical_metadata(accepted);
                auto current = accepted;
                std::vector<std::uint64_t> expected(count);
                std::iota(expected.begin(), expected.end(), 0);
                for (std::size_t batch = 0; batch < 40; ++batch)
                {
                    std::array<std::uint64_t, 4> request{};
                    for (std::size_t offset = 0; offset < width; ++offset)
                    {
                        const auto value = batch * 64 + offset;
                        const bool reverse = direction == 1 || (direction == 2 && batch % 2);
                        request[offset] = reverse ? 3000 - value : value;
                    }
                    Counter counter;
                    G::PageWork work;
                    current.prepend(std::span(request).first(width), counter, work);
                    EXPECT_EQ(counter.used, work.copied_bytes);
                    allocations[scale] += counter.used;
                    visits[scale] += work.visited_pages;
                    payload[scale] += work.copied_payload_bytes;
                    for (const auto key : std::span(request).first(width))
                        expected.erase(std::find(expected.begin(), expected.end(), key));
                    expected.insert(expected.begin(), request.begin(), request.begin() + width);
                    ASSERT_EQ(current.order->size, expected.size());
                    for (std::size_t slot = 0; slot < width; ++slot)
                    {
                        EXPECT_EQ(Order::at(current.order, slot).key, request[slot]);
                        EXPECT_EQ(current.find(request[slot]), slot);
                    }
                }
                for (std::size_t slot = 0; slot < expected.size(); ++slot)
                {
                    EXPECT_EQ(Order::at(current.order, slot).key, expected[slot]);
                    EXPECT_EQ(current.find(expected[slot]), slot);
                }
                EXPECT_EQ(Order::at(accepted.order, 0).key, 0u);
                EXPECT_EQ(Order::at(accepted.order, count - 1).key, count - 1);
                EXPECT_EQ(&Order::at(current.order, count - 1), &Order::at(accepted.order, count - 1));
                EXPECT_EQ(&Keys::at(current.keys, count - 1), &Keys::at(accepted.keys, count - 1));
                for (const auto& [owner, bytes] : physical_metadata(current))
                    if (!accepted_metadata.contains(owner))
                        new_metadata[scale] += bytes;
            }
            EXPECT_EQ(allocations[0], allocations[1]);
            EXPECT_EQ(payload[0], payload[1]);
            EXPECT_EQ(visits[0], visits[1]);
            EXPECT_EQ(new_metadata[0], new_metadata[1]);
        }
}

TEST(GeometryDemandPages, ExactAdmissionFailurePreservesTheAcceptedRootsAndSpentPrefix)
{
    Counter base;
    const auto accepted = Selection::initial(4096, base);
    const std::array<std::uint64_t, 4> request{4096, 63, 127, 4097};
    const auto replay = [&](Counter& counter)
    {
        auto candidate = accepted;
        G::PageWork work;
        candidate.prepend(request, counter, work);
        EXPECT_EQ(counter.used, work.copied_bytes);
        return candidate;
    };
    Counter measured;
    auto next = replay(measured);
    EXPECT_EQ(next.order->size, 4098u);
    Counter exact{0, measured.used};
    EXPECT_EQ(replay(exact).find(4097), 3u);
    Counter short_limit{0, measured.used - 1};
    EXPECT_THROW((void) replay(short_limit), SimpleFluid::PLIC::GeometryError);
    EXPECT_GT(short_limit.used, 0u);
    EXPECT_LE(short_limit.used, short_limit.limit);
    EXPECT_EQ(accepted.order->size, 4096u);
    EXPECT_EQ(accepted.find(63), 63u);
}

TEST(GeometryDemandPages, ThirdRootSharesPagesAndOutlivesEditorsWithoutRetainingOldRoots)
{
    Selection latest;
    std::weak_ptr<const Order::Node> first_root, second_root;
    {
        Counter counter;
        auto first = Selection::initial(4096, counter);
        first_root = first.order;
        latest = first;
        G::PageWork work;
        latest.prepend(std::array<std::uint64_t, 2>{80, 4000}, counter, work);
        second_root = latest.order;
        auto second = latest;
        latest.prepend(std::array<std::uint64_t, 2>{81, 3999}, counter, work);
        EXPECT_EQ(first.find(80), 80u);
        EXPECT_EQ(second.find(80), 0u);
        EXPECT_EQ(latest.find(81), 0u);
    }
    EXPECT_TRUE(first_root.expired());
    EXPECT_TRUE(second_root.expired());
    EXPECT_EQ(latest.find(81), 0u);
    EXPECT_EQ(latest.find(80), 2u);
    EXPECT_EQ(Order::at(latest.order, 4095).key, 4095u);
}

TEST(GeometryDemandPages, ControlledOwnerAliasesHaveOnePayloadAndIndependentRoleChannels)
{
    Counter counter;
    auto owner = std::make_shared<const Payload>();
    std::weak_ptr<const Payload> lifetime = owner;
    G::GeometryOwnerManifest empty;
    const auto first = empty.retained(0, 3, owner, sizeof(Payload), counter.admission());
    auto second = first.retained(0, 3, owner, sizeof(Payload), counter.admission());
    EXPECT_EQ(second.payload_bytes(), sizeof(Payload));
    EXPECT_EQ(second.references(0, 3, owner), 2u);
    EXPECT_EQ(second.owner_count(), 1u);
    second = second.retained(1, 3, owner, sizeof(Payload), counter.admission());
    EXPECT_EQ(second.payload_bytes(), 2 * sizeof(Payload));
    second = second.retired(0, 3, owner, counter.admission());
    EXPECT_EQ(second.references(0, 3, owner), 1u);
    auto third = second.retired(1, 3, owner, counter.admission());
    EXPECT_EQ(third.payload_bytes(), sizeof(Payload));
    EXPECT_GT(third.metadata_admission_bytes(), third.metadata_payload_bytes());

    const std::shared_ptr<const void> foreign(owner.get(), [](const void*) {});
    EXPECT_THROW(
        (void) third.retained(0, 3, foreign, sizeof(Payload), counter.admission()), SimpleFluid::PLIC::GeometryError);
    EXPECT_THROW(
        (void) third.retained(0, 3, owner, sizeof(Payload) - 1, counter.admission()), SimpleFluid::PLIC::GeometryError);
    EXPECT_TRUE(third.admits(0, 3, owner, sizeof(Payload)));
    owner.reset();
    EXPECT_FALSE(lifetime.expired());
}

TEST(GeometryDemandPages, ControlledOwnerReplacementHasStableExactAdmissionAndCachedTotals)
{
    Counter source_counter;
    G::GeometryOwnerManifest source;
    std::vector<std::shared_ptr<const Payload>> owners;
    for (std::size_t slot = 0; slot < 256; ++slot)
    {
        owners.push_back(std::make_shared<const Payload>());
        source = source.retained(0, 1, owners.back(), sizeof(Payload), source_counter.admission());
    }
    const auto update = [&](Counter& counter)
    {
        auto candidate = source.retained(0, 1, owners[17], sizeof(Payload), counter.admission());
        return candidate.retired(0, 1, owners[128], counter.admission());
    };
    Counter measured;
    const auto next = update(measured);
    Counter exact{0, measured.used};
    EXPECT_EQ(update(exact).payload_bytes(), 255 * sizeof(Payload));
    Counter short_limit{0, measured.used - 1};
    EXPECT_THROW((void) update(short_limit), SimpleFluid::PLIC::GeometryError);
    EXPECT_EQ(source.payload_bytes(), 256 * sizeof(Payload));
    EXPECT_EQ(source.references(0, 1, owners[17]), 1u);
    EXPECT_EQ(next.references(0, 1, owners[17]), 2u);
    std::size_t body = 0, count = 0;
    next.visit(
        [&](const auto& owner)
        {
            ++count;
            body += owner.bytes;
        });
    EXPECT_EQ(body, next.payload_bytes());
    EXPECT_EQ(count, next.owner_count());
    std::map<const void*, std::size_t> metadata;
    next.visit_metadata(
        [&](const void* owner, std::size_t bytes) { EXPECT_TRUE(metadata.emplace(owner, bytes).second); });
    std::size_t known_payload = 0;
    for (const auto& [owner, bytes] : metadata)
        known_payload += bytes;
    EXPECT_EQ(known_payload, next.metadata_payload_bytes());
}

TEST(GeometryDemandPages, CompressedSourcePagesPreserveRequestedAndUnrequestedOrderAcrossSparseStreams)
{
    for (const auto width : {1u, 2u, 4u})
        for (const auto direction : {0u, 1u, 2u})
        {
            SCOPED_TRACE(width);
            SCOPED_TRACE(direction);
            std::array<std::size_t, 2> allocated{}, new_payload{}, visits{}, source_rows{};
            for (std::size_t scale = 0; scale < 2; ++scale)
            {
                const auto count = std::size_t{8192} << scale;
                std::vector<std::shared_ptr<const Payload>> owners;
                std::vector<G::SourceSlot> physical;
                for (std::size_t slot = 0; slot < count; ++slot)
                {
                    if (!(slot % 64))
                        owners.push_back(std::make_shared<const Payload>());
                    physical.push_back({owners.back(), slot % 64});
                }
                Counter initial;
                const auto accepted = G::GeometryDemandOrder{}.prepended(physical, {}, initial.admission());
                EXPECT_LT(accepted.metadata_payload_bytes(), count * sizeof(G::OrderedSourceSlot));
                std::map<const void*, std::size_t> accepted_metadata;
                accepted.visit_metadata([&](const void* owner, std::size_t bytes)
                    { EXPECT_TRUE(accepted_metadata.emplace(owner, bytes).second); });
                auto current = accepted;
                std::vector<G::OrderStamp> stamps(count);
                for (std::size_t slot = 0; slot < count; ++slot)
                    stamps[slot] = accepted.at(slot).stamp;
                std::vector<std::uint64_t> expected(count);
                std::iota(expected.begin(), expected.end(), 0);
                for (std::size_t batch = 0; batch < 40; ++batch)
                {
                    std::array<G::SourceSlot, 4> prefix{};
                    std::array<G::OrderStamp, 4> retired{};
                    std::array<std::size_t, 4> keys{};
                    for (std::size_t offset = 0; offset < width; ++offset)
                    {
                        const auto value = batch * 64 + offset;
                        const bool reverse = direction == 1 || (direction == 2 && batch % 2);
                        keys[offset] = reverse ? 3000 - value : value;
                        prefix[offset] = physical[keys[offset]];
                        retired[offset] = stamps[keys[offset]];
                    }
                    Counter counter;
                    G::PageWork work;
                    current = current.prepended(
                        std::span(prefix).first(width), std::span(retired).first(width), counter.admission(), &work);
                    allocated[scale] += counter.used;
                    visits[scale] += work.visited_pages;
                    source_rows[scale] += work.visited_source_rows + work.copied_source_rows;
                    for (std::size_t offset = 0; offset < width; ++offset)
                    {
                        stamps[keys[offset]] = current.at(offset).stamp;
                        expected.erase(std::find(expected.begin(), expected.end(), keys[offset]));
                        EXPECT_EQ(current.at(offset).physical.source, prefix[offset].source);
                        EXPECT_EQ(current.at(offset).physical.row, prefix[offset].row);
                        EXPECT_EQ(current.rank(stamps[keys[offset]]), offset);
                    }
                    expected.insert(expected.begin(), keys.begin(), keys.begin() + width);
                }
                ASSERT_EQ(current.size(), expected.size());
                for (std::size_t slot = 0; slot < count; ++slot)
                {
                    const auto value = current.at(slot);
                    EXPECT_EQ(value.physical.source, physical[expected[slot]].source);
                    EXPECT_EQ(value.physical.row, physical[expected[slot]].row);
                    EXPECT_EQ(current.rank(stamps[expected[slot]]), slot);
                }
                std::size_t payload = 0;
                current.visit_metadata(
                    [&](const void* owner, std::size_t bytes)
                    {
                        payload += bytes;
                        if (!accepted_metadata.contains(owner))
                            new_payload[scale] += bytes;
                    });
                EXPECT_EQ(payload, current.metadata_payload_bytes());
                EXPECT_EQ(accepted.at(0).physical.source, physical[0].source);
                EXPECT_EQ(accepted.at(0).physical.row, 0u);
            }
            EXPECT_EQ(allocated[0], allocated[1]);
            EXPECT_EQ(new_payload[0], new_payload[1]);
            EXPECT_EQ(visits[0], visits[1]);
            EXPECT_EQ(source_rows[0], source_rows[1]);
        }
}

TEST(GeometryDemandPages, CompressedSourcePageAdmissionIsExactAndKeepsAcceptedStampsRetryable)
{
    const auto source = std::make_shared<const Payload>();
    std::vector<G::SourceSlot> input;
    for (std::size_t row = 0; row < 4096; ++row)
        input.push_back({source, row});
    Counter initial;
    const auto accepted = G::GeometryDemandOrder{}.prepended(input, {}, initial.admission());
    const std::array prefix{input[63], input[128], G::SourceSlot{source, 4096}, input[2047]};
    const std::array retired{accepted.at(63).stamp, accepted.at(128).stamp, accepted.at(2047).stamp};
    const auto apply = [&](Counter& counter)
    {
        const auto candidate = accepted.prepended(prefix, retired, counter.admission());
        EXPECT_EQ(candidate.size(), accepted.size() + 1);
        EXPECT_EQ(candidate.at(0).physical.row, 63u);
        EXPECT_EQ(candidate.at(2).physical.row, 4096u);
        return candidate;
    };
    Counter measured;
    const auto next = apply(measured);
    Counter exact{0, measured.used};
    EXPECT_EQ(apply(exact).size(), next.size());
    Counter short_limit{0, measured.used - 1};
    EXPECT_THROW((void) apply(short_limit), SimpleFluid::PLIC::GeometryError);
    EXPECT_GT(short_limit.used, 0u);
    EXPECT_EQ(accepted.rank(retired.front()), 63u);
    Counter retry;
    EXPECT_EQ(apply(retry).at(3).physical.row, 2047u);
    const auto invalid = G::OrderStamp{accepted.generation() + 1, 999};
    Counter invalid_input;
    EXPECT_THROW((void) accepted.prepended({}, std::span(&invalid, 1), invalid_input.admission()),
        SimpleFluid::PLIC::GeometryError);
    EXPECT_EQ(accepted.rank(retired.back()), 2047u);
}

TEST(GeometryDemandPages, SourcePointerArrayReboundPreservesEveryStampAndLogicalRank)
{
    auto old_page = std::make_shared<const Payload>();
    auto packed_page = std::make_shared<const Payload>(*old_page);
    std::vector<G::SourceSlot> input;
    for (std::size_t row = 0; row < 4096; ++row)
        input.push_back({old_page, row % 64});
    Counter initial;
    const auto accepted = G::GeometryDemandOrder{}.prepended(input, {}, initial.admission());
    std::array<G::OrderedSourceSlot, 64> updates{};
    for (std::size_t row = 0; row < updates.size(); ++row)
    {
        const auto slot = accepted.size() - updates.size() + row;
        updates[row] = {accepted.at(slot).stamp, {packed_page, row}};
    }
    const auto apply = [&](Counter& counter)
    {
        const auto candidate = accepted.rebound(updates, counter.admission());
        EXPECT_EQ(candidate.size(), accepted.size());
        EXPECT_EQ(candidate.generation(), accepted.generation());
        for (std::size_t row = 0; row < updates.size(); ++row)
        {
            const auto slot = candidate.size() - updates.size() + row;
            const auto value = candidate.at(slot);
            EXPECT_EQ(value.stamp, updates[row].stamp);
            EXPECT_EQ(candidate.rank(value.stamp), slot);
            EXPECT_EQ(value.physical.source, packed_page);
            EXPECT_EQ(value.physical.row, row);
        }
        return candidate;
    };
    Counter measured;
    const auto next = apply(measured);
    Counter exact{0, measured.used};
    EXPECT_EQ(apply(exact).metadata_payload_bytes(), next.metadata_payload_bytes());
    Counter short_limit{0, measured.used - 1};
    EXPECT_THROW((void) apply(short_limit), SimpleFluid::PLIC::GeometryError);
    EXPECT_EQ(accepted.at(accepted.size() - 1).physical.source, old_page);
    EXPECT_EQ(next.at(0).physical.source, old_page);
}

TEST(GeometryDemandPages, EmptyOrderKeepsItsGenerationAndRejectsRetiredStampsAfterNewDemand)
{
    const auto source = std::make_shared<const Payload>();
    Counter initial;
    const auto first =
        G::GeometryDemandOrder{}.prepended(std::array{G::SourceSlot{source, 1}}, {}, initial.admission());
    const auto stale = first.at(0).stamp;
    Counter clear;
    const auto empty = first.prepended({}, std::span(&stale, 1), clear.admission());
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.generation(), first.generation());
    std::size_t known = 0;
    empty.visit_metadata([&](const void*, std::size_t bytes) { known += bytes; });
    EXPECT_EQ(empty.metadata_payload_bytes(), known);
    EXPECT_GT(known, 0u);
    EXPECT_THROW((void) empty.rank(stale), SimpleFluid::PLIC::GeometryError);
    Counter appended;
    const auto next = empty.prepended(std::array{G::SourceSlot{source, 2}}, {}, appended.admission());
    EXPECT_GT(next.generation(), first.generation());
    EXPECT_NE(next.at(0).stamp, stale);
    EXPECT_EQ(next.rank(next.at(0).stamp), 0u);
    EXPECT_THROW((void) next.rank(stale), SimpleFluid::PLIC::GeometryError);
    Counter rejected;
    EXPECT_THROW(
        (void) next.prepended({}, std::span(&stale, 1), rejected.admission()), SimpleFluid::PLIC::GeometryError);
    const G::OrderedSourceSlot rebound{stale, {source, 3}};
    EXPECT_THROW((void) next.rebound(std::span(&rebound, 1), rejected.admission()), SimpleFluid::PLIC::GeometryError);
    EXPECT_EQ(first.at(0).physical.row, 1u);
    EXPECT_EQ(next.at(0).physical.row, 2u);
}

TEST(GeometryDemandPages, GenericThrowingDefaultRowKeepsItsNoDefaultPatchContract)
{
    using Pages = G::OrderedPages<ThrowingDefaultRow, ThrowingDefaultLess>;
    ThrowingDefaultRow::default_calls = 0;
    const std::array input{ThrowingDefaultRow{1, 10}, ThrowingDefaultRow{3, 30}};
    const std::array changes{ThrowingDefaultRow{3, 31}};
    Counter initial{37};
    const auto accepted = Pages(initial.admission()).build(input);
    const auto apply = [&](Counter& counter)
    { return Pages(counter.admission()).patched(accepted, changes, [](const auto&) { return true; }); };
    Counter measured{37};
    const auto updated = apply(measured);
    EXPECT_EQ(Pages::at(updated, 0).payload, 10);
    EXPECT_EQ(Pages::at(updated, 1).payload, 31);
    EXPECT_EQ(Pages::at(accepted, 1).payload, 30);
    Counter exact{37, measured.used};
    const auto completed = apply(exact);
    EXPECT_EQ(exact.used, measured.used);
    EXPECT_EQ(Pages::at(completed, 1).payload, 31);
    Counter short_limit{37, measured.used - 1};
    EXPECT_THROW((void) apply(short_limit), SimpleFluid::PLIC::GeometryError);
    EXPECT_GE(short_limit.used, 37u);
    EXPECT_LE(short_limit.used, short_limit.limit);
    EXPECT_EQ(Pages::at(accepted, 0).payload, 10);
    EXPECT_EQ(Pages::at(accepted, 1).payload, 30);
    EXPECT_EQ(ThrowingDefaultRow::default_calls, 0u);
}

TEST(GeometryDemandPages, BatchedEditsKeepNonzeroPredicatesAcrossByteBoundaries)
{
    struct Row
    {
        int key;
        int payload;
        bool operator==(const Row&) const = default;
    };
    const auto less = [](const Row& first, const Row& second) { return first.key < second.key; };
    using Pages = G::OrderedPages<Row, decltype(less)>;
    static_assert(std::is_trivially_default_constructible_v<Row>);
    static_assert(G::stack_safe_default<Row>);

    const std::array accepted_rows{Row{1, 10}, Row{2, 20}, Row{4, 40}};
    const std::array changes{Row{1, 8704}, Row{2, 0}, Row{3, 256}};
    Counter initial;
    const auto accepted = Pages(initial.admission()).build(accepted_rows);

    Counter edited;
    const auto updated =
        Pages(edited.admission()).patched(accepted, changes, [](const Row& row) { return row.payload; });

    const std::array expected{Row{1, 8704}, Row{3, 256}, Row{4, 40}};
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->size, expected.size());
    ASSERT_EQ(accepted->size, accepted_rows.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_EQ(Pages::at(updated, index), expected[index]);
        EXPECT_EQ(Pages::at(accepted, index), accepted_rows[index]);
    }
}

TEST(GeometryDemandPages, ClusteredBatchAdmitsPagePointersWithoutASecondRowInventory)
{
    constexpr std::size_t inserted = 4096;
    std::vector<OrderRow> rows, changes;
    for (std::size_t i = 0; i != 3 * G::leaf_capacity; ++i)
        rows.push_back({1, i, inserted + 1 + i});
    for (std::size_t i = 0; i != inserted; ++i)
        changes.push_back({1, i, i + 1});
    changes.push_back({0, 0, rows.front().key}); // Retire the former first row.
    Counter initial;
    const auto accepted = Keys(initial.admission()).build(rows);
    ASSERT_EQ(accepted->children.size(), 3u);
    Counter edited;
    G::PageWork work;
    std::size_t predicates = 0;
    const auto result = Keys(edited.admission(), &work).patched(accepted, changes,
        [&](const OrderRow& row) { ++predicates; return row.generation != 0; });
    ASSERT_TRUE(result);
    EXPECT_EQ(predicates, changes.size());
    EXPECT_EQ(result->size, inserted + rows.size() - 1);
    for (std::size_t i = 0; i != inserted; ++i)
        EXPECT_EQ(Keys::at(result, i).key, i + 1);
    for (std::size_t i = 1; i != rows.size(); ++i)
        EXPECT_EQ(Keys::at(result, inserted + i - 1).key, rows[i].key);

    std::size_t shared = 0;
    Keys::visit_pages(result, [&](const auto& page)
    {
        shared += page == accepted->children[1] || page == accepted->children[2];
        EXPECT_LE(page->occupancy(), page->leaf() ? G::leaf_capacity : G::branch_capacity);
        if (page != result)
            EXPECT_GE(page->occupancy(), page->leaf() ? G::leaf_capacity / 2 : G::branch_capacity / 2);
    });
    EXPECT_EQ(shared, 2u);
    ASSERT_GE(edited.used, work.copied_bytes);
    EXPECT_LT(edited.used - work.copied_bytes, inserted * sizeof(OrderRow) / 2);
}
