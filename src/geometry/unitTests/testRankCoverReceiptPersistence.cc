/** Real MPI owner-cover transport and detached exact receipt storage. */
#include "adaptive_storage_test_helpers.hh"
#include "geometry/mesh/AMR/ClosedContactIndex.hh"
#include "geometry/mesh/AMR/ContactRankCoverReceipts.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
using Snapshot = std::shared_ptr<const D::ContactRankCoverReceipts>;
constexpr std::size_t ample = 64 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Point = std::array<long double, 3>;
A::AdaptiveBytes domain(std::size_t component)
{
    A::AdaptiveBytes result{'d'};
    for (unsigned i = 0; i != sizeof(component); ++i)
        result.push_back(char(component >> (8 * i)));
    return result;
}
A::AdaptiveContactBounds box(long double x)
{
    return {{x, 0, 0}, {x + 1, 1, 1}};
}
Snapshot source(const std::shared_ptr<D::Budget>& budget, std::size_t cells)
{
    D::ContactRankCoverReceiptBuilder builder(budget);
    for (std::size_t i = 0; i != cells; ++i)
        builder.accept(1, domain(i), box(static_cast<long double>(i)));
    return builder.finish();
}
Snapshot changed(const std::shared_ptr<D::Budget>& arena, const Snapshot& previous, long double x)
{
    D::ContactRankCoverReceiptBuilder builder(arena, previous);
    previous->visit(
        [&](int owner, auto name, const auto& bounds, std::size_t count)
        {
            for (std::size_t i = 0; i != count; ++i)
                builder.accept(owner, name, std::ranges::equal(name, domain(0)) ? box(x) : bounds);
        });
    return builder.finish();
}
using Key = std::pair<std::uintptr_t, D::ContactPayloadRole>;
using Payload = std::map<Key, std::size_t>;
Payload payload(const Snapshot& snapshot)
{
    Payload result;
    snapshot->visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& row)
        {
            result.emplace(
                Key{reinterpret_cast<std::uintptr_t>(row.identity.address), row.identity.role}, row.payload_bytes);
        });
    return result;
}
std::size_t fresh_bytes(const Payload& before, const Payload& after)
{
    std::size_t result = 0;
    for (const auto& [key, bytes] : after)
        if (!before.contains(key))
            result += bytes;
    return result;
}
using ExactSelector =
    std::tuple<int, A::AdaptiveBytes, std::array<std::uint64_t, 6 * SimpleFluid::PLIC::ScalarPacketWords{}.size()>>;
using ExactFacts = std::map<ExactSelector, std::size_t>;
ExactFacts facts(const Snapshot& snapshot)
{
    ExactFacts result;
    snapshot->visit(
        [&](int owner, auto name, const auto& bounds, std::size_t count)
        {
            std::array<std::uint64_t, 6 * SimpleFluid::PLIC::ScalarPacketWords{}.size()> encoded;
            std::size_t at = 0;
            for (const auto* axis : {&bounds.lower, &bounds.upper})
                for (auto value : *axis)
                    for (auto word : SimpleFluid::PLIC::scalar_packet(value))
                        encoded[at++] = word;
            EXPECT_TRUE(result.emplace(ExactSelector{owner, A::AdaptiveBytes(name.begin(), name.end()), encoded}, count)
                    .second);
        });
    return result;
}
std::map<const void*, std::size_t> page_edges(const Snapshot& snapshot)
{
    std::map<const void*, std::size_t> result;
    snapshot->visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& row)
        {
            if (row.identity.role == D::ContactPayloadRole::RankCoverPage)
            {
                auto& edges = result[row.owner];
                if (row.kind == D::ContactPayloadKind::Buffer)
                    edges += row.live_count;
            }
        });
    return result;
}
struct ReuseProof
{
    std::size_t changed_facts = 0, edits = 0, protected_rows = 0, total_rows = 0, max_page_admission = 0;
};
ReuseProof expect_physical_reuse(const Snapshot& before, const Snapshot& after, std::size_t observed_owner_visits)
{
    const auto old_facts = facts(before), new_facts = facts(after);
    using Leaf = D::ContactRankCoverReceipts::BorrowedLeafPage;
    std::vector<Leaf> old_pages, new_pages;
    before->visit_borrowed_leaf_pages([&](const Leaf& page) { old_pages.push_back(page); });
    after->visit_borrowed_leaf_pages([&](const Leaf& page) { new_pages.push_back(page); });
    std::vector<ExactSelector> old_order;
    for (const auto& [key, count] : old_facts)
        old_order.push_back(key);
    std::vector<bool> dirty(old_pages.size(), false), excluded(old_pages.size(), false);
    const auto mark = [&](const ExactSelector& key)
    {
        if (old_order.empty())
            return;
        // Ordered insertion chooses the leaf containing the first old key
        // not before this selector; insertion after the end chooses the last.
        auto ordinal = std::size_t(std::lower_bound(old_order.begin(), old_order.end(), key) - old_order.begin());
        ordinal = std::min(ordinal, old_order.size() - 1);
        const auto page = std::partition_point(old_pages.begin(), old_pages.end(),
            [&](const Leaf& leaf) { return leaf.first_ordinal + leaf.count <= ordinal; });
        if (page == old_pages.end())
            throw std::invalid_argument("Borrowed receipt leaf intervals omit an accepted selector.");
        dirty[std::size_t(page - old_pages.begin())] = true;
    };
    ReuseProof result;
    result.total_rows = old_order.size();
    for (const auto* pages : {&old_pages, &new_pages})
        for (const auto& page : *pages)
        {
            if (!page.count)
                throw std::invalid_argument("Borrowed receipt leaf has no accepted rows.");
            // Actual leaf row storage supplies a conservative element width;
            // ordered pages admit at most64 rows or32 child owners plus their
            // object and the documented128-byte control allowance.
            const auto row_width = page.row_bytes / page.count;
            const auto max_buffer = std::max(D::persistent_ordered_detail::leaf_capacity * row_width,
                D::persistent_ordered_detail::branch_capacity * sizeof(std::shared_ptr<const void>));
            result.max_page_admission = std::max(result.max_page_admission, page.object_bytes + 128 + max_buffer);
        }
    for (const auto& [key, count] : old_facts)
    {
        const auto found = new_facts.find(key);
        if (found == new_facts.end() || found->second != count)
        {
            ++result.edits; // One erase or one existing multiplicity rewrite.
            result.changed_facts += found == new_facts.end() ? 1 : 2;
            mark(key);
        }
    }
    for (const auto& [key, count] : new_facts)
        if (!old_facts.contains(key))
        {
            result.edits += count; // accept() inserts, then rewrites each repeat.
            ++result.changed_facts;
            mark(key);
        }
    // Fewer than a minimum leaf's rows can be removed by this fixed edit, so
    // rebalancing needs only the neighboring accepted leaf. Do not enlarge
    // this halo silently if a future fixture changes the actual edit set.
    EXPECT_LT(result.edits, D::persistent_ordered_detail::leaf_capacity / 2);
    for (std::size_t i = 0; i != dirty.size(); ++i)
        if (dirty[i])
        {
            excluded[i] = true;
            if (i)
                excluded[i - 1] = true;
            if (i + 1 < dirty.size())
                excluded[i + 1] = true;
        }
    std::map<const void*, const Leaf*> current_pages;
    for (const auto& page : new_pages)
        current_pages.emplace(page.object, &page);
    std::size_t protected_pages = 0, expected_first = 0;
    for (std::size_t i = 0; i != old_pages.size(); ++i)
    {
        const auto& page = old_pages[i];
        EXPECT_EQ(page.first_ordinal, expected_first);
        expected_first += page.count;
        if (excluded[i])
            continue;
        ++protected_pages;
        result.protected_rows += page.count;
        EXPECT_FALSE(page.control.expired());
        const auto found = current_pages.find(page.object);
        if (found == current_pages.end())
        {
            ADD_FAILURE() << "Clean receipt leaf was copied: first=" << page.first_ordinal << " count=" << page.count;
            continue;
        }
        const auto& shared = *found->second;
        EXPECT_FALSE(page.control.owner_before(shared.control) || shared.control.owner_before(page.control));
        EXPECT_EQ(shared.object, page.object);
        EXPECT_EQ(shared.rows, page.rows);
        EXPECT_EQ(shared.object_bytes, page.object_bytes);
        EXPECT_EQ(shared.row_bytes, page.row_bytes);
        EXPECT_EQ(shared.count, page.count);
    }
    EXPECT_EQ(expected_first, old_facts.size());
    if (!old_order.empty())
    {
        EXPECT_GT(protected_pages, 0u);
        EXPECT_GE(2 * result.protected_rows, result.total_rows)
            << "protected=" << result.protected_rows << " total=" << result.total_rows << " edits=" << result.edits
            << " old_leaf_pages=" << old_pages.size();
    }
    // This independent graph census counts actual first/last events: two
    // snapshot headers, both root edges, then child/domain edges only from
    // pages that occur in exactly one physical root. Shared nodes are entered
    // through an exclusive parent's edge and stop without descending.
    const auto old_edges = page_edges(before), new_edges = page_edges(after);
    std::size_t expected_visits = 2 + std::size_t(!old_edges.empty()) + std::size_t(!new_edges.empty());
    for (const auto& [page, edges] : old_edges)
        if (!new_edges.contains(page))
            expected_visits += edges;
    for (const auto& [page, edges] : new_edges)
        if (!old_edges.contains(page))
            expected_visits += edges;
    EXPECT_EQ(observed_owner_visits, expected_visits);
    return result;
}
} // namespace

TEST(RankCoverReceiptStorage, FixedEditSharesDoubledDomainsAndDetachedThreeGenerations)
{
    for (const auto cells : {std::size_t(384), std::size_t(768)})
    {
        auto first_budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto first = source(first_budget, cells);
        D::ContactPhysicalResidency first_usage(first_budget);
        (void) first->observe(first_usage, true);
        const auto old_payload = payload(first);
        auto second_budget = std::make_shared<D::Budget>(D::Budget{ample});
        D::ContactPhysicalResidency second_usage(first_usage, second_budget);
        auto second = changed(second_budget, first, -2);
        const auto visits = second->observe(second_usage, true) + first->observe(second_usage, false);
        second_usage.finish_overlap();
        ASSERT_EQ(first->page_height(), 2u);
        ASSERT_EQ(second->page_height(), 2u);
        EXPECT_EQ(second->work().received_covers, cells);
        EXPECT_EQ(second->work().reconciled_selectors, cells);
        EXPECT_EQ(second->work().copied_domain_bytes, 0u);
        // One deletion and one insertion, at height two. Each ordered edit
        // creates at most two path nodes plus a rebalanced sibling pair per
        // level; first/last observation visits bounded page children/rows.
        const auto max_pages = 2 * (4 * second->page_height() + 1);
        EXPECT_LE(second->work().copied_pages, max_pages);
        EXPECT_LE(visits, 2 + max_pages * (1 + D::persistent_ordered_detail::leaf_capacity +
                                              D::persistent_ordered_detail::branch_capacity));
        EXPECT_LT(fresh_bytes(old_payload, payload(second)), old_payload.size() * 512);
        auto third_budget = std::make_shared<D::Budget>(D::Budget{ample});
        auto third = changed(third_budget, second, -4);
        const std::weak_ptr<const D::ContactRankCoverReceipts> first_control(first), second_control(second);
        first.reset();
        second.reset();
        EXPECT_TRUE(first_control.expired());
        EXPECT_TRUE(second_control.expired());
        EXPECT_TRUE(third->contains(1, domain(0), Point{-3.5L, 0, 0}));
        EXPECT_FALSE(third->contains(1, domain(0), Point{-1.5L, 0, 0}));
        EXPECT_TRUE(third->contains(1, domain(cells - 1), Point{static_cast<long double>(cells) - .5L, 0, 0}));
    }
}
TEST(RankCoverReceiptStorage, MultiplicityAndMutationExactCreditRollback)
{
    auto budget = std::make_shared<D::Budget>(D::Budget{ample});
    auto original = source(budget, 384);
    const auto census = payload(original);
    const auto source_baseline = budget->used;
    const auto run = [&](std::optional<std::size_t> allowance, bool rejected)
    {
        auto target = std::make_shared<D::Budget>(D::Budget{ample});
        std::size_t phase_peak = 0;
        bool entered_path_edit = false, published = false, exhausted_credit = false;
        {
            D::ContactPhysicalResidency residency(target);
            // Generic physical import is setup, outside the isolated cap gate.
            // It cannot consume the mutation's exact/one-short allowance.
            (void) original->observe(residency, true);
            D::ContactRankCoverReceiptBuilder duplicate(target, original);
            original->visit(
                [&](int owner, auto name, const auto& bounds, std::size_t count)
                {
                    for (std::size_t i = 0; i != count + std::size_t(std::ranges::equal(name, domain(1))); ++i)
                        duplicate.accept(owner, name, bounds);
                });
            const auto phase_baseline = target->used;
            const auto previous_peak = target->peak;
            target->peak = phase_baseline;
            if (allowance)
                target->limit = phase_baseline + *allowance;
            try
            {
                // All accepts matched old selectors and only filled admitted
                // numeric scratch. finish must now edit the changed multiplicity
                // row; no residency-index mutation occurs inside this phase.
                entered_path_edit = true;
                auto doubled = duplicate.finish();
                published = true;
                EXPECT_GT(doubled->work().copied_pages, 0u);
                EXPECT_EQ(doubled->work().copied_domain_bytes, 0u);
                bool found = false;
                doubled->visit(
                    [&](int, auto name, const auto&, std::size_t count)
                    {
                        if (std::ranges::equal(name, domain(1)))
                        {
                            found = true;
                            EXPECT_EQ(count, 2u);
                        }
                    });
                EXPECT_TRUE(found);
            }
            catch (const std::invalid_argument& error)
            {
                exhausted_credit =
                    std::string_view(error.what()).starts_with("Distributed adaptive payload budget exhausted:");
                if (!rejected)
                    ADD_FAILURE() << error.what();
            }
            phase_peak = target->peak - phase_baseline;
            target->limit = ample;
            target->peak = std::max(previous_peak, target->peak);
            EXPECT_TRUE(entered_path_edit);
            EXPECT_EQ(published, !rejected);
            EXPECT_EQ(exhausted_credit, rejected);
            // Failed private candidates are discarded; retryability is not an
            // advertised contract. The accepted physical graph never changed.
            EXPECT_EQ(payload(original), census);
            EXPECT_EQ(budget->used, source_baseline);
            bool accepted_once = false;
            original->visit(
                [&](int, auto name, const auto&, std::size_t count)
                {
                    if (std::ranges::equal(name, domain(1)))
                    {
                        accepted_once = true;
                        EXPECT_EQ(count, 1u);
                    }
                });
            EXPECT_TRUE(accepted_once);
        }
        EXPECT_EQ(target->used, 0u);
        EXPECT_EQ(payload(original), census);
        EXPECT_EQ(budget->used, source_baseline);
        return phase_peak;
    };
    const auto exact = run({}, false);
    ASSERT_GT(exact, 1u);
    EXPECT_EQ(run(exact, false), exact);
    (void) run(exact - 1, true);
}

namespace
{
struct Measurement
{
    std::size_t fresh, copied_pages, copied_bytes, owner_visits, changed_facts, height;
    std::size_t edits, protected_rows, total_rows, max_page_admission;
};
Measurement measure(std::size_t components, int fault = 0)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), ranks = comm->getSize();
    namespace S = SimpleFluid::test::adaptive_storage;
    const auto fixture = S::make_disconnected(S::Provider::Cartesian, components);
    const auto kernel = fixture.kernel;
    const auto first_region = fixture.changed.region;
    const auto first_domain = fixture.owned.front().domain;
    // Every component contains its complete native2x2x2 block, including actual
    // exterior-face contracts. A partial global grid would leave unpaired
    // interior supports; an empty partition is correctly rejected by geometry.
    EXPECT_EQ(fixture.owned.size(), 8 * components);
    std::map<A::AdaptiveBytes, std::size_t> inventory;
    std::array<A::AdaptiveContactBounds, 2> owner_bounds{};
    std::array<bool, 2> owner_present{};
    std::vector<A::AdaptiveLeaf> owned, refined;
    for (const auto& leaf : fixture.owned)
    {
        ++inventory[leaf.domain];
        const auto& ancestry = std::get<A::CartesianCellAncestry>(leaf.key.ancestry);
        const int owner = int(ancestry.index[0]);
        if (owner < 0 || owner >= 2)
            throw std::invalid_argument("Closed fixture has a nonbinary native owner index.");
        for (const auto& image : kernel->closed_contact_images(leaf))
        {
            auto& bounds = owner_bounds[owner];
            if (!owner_present[owner])
                bounds = image.bounds;
            else
                for (unsigned axis = 0; axis != 3; ++axis)
                {
                    bounds.lower[axis] = std::min(bounds.lower[axis], image.bounds.lower[axis]);
                    bounds.upper[axis] = std::max(bounds.upper[axis], image.bounds.upper[axis]);
                }
            owner_present[owner] = true;
        }
        if (rank == (ranks == 1 ? 0 : owner))
            owned.push_back(leaf);
        if (leaf.key.region == first_region)
        {
            auto split = kernel->split(leaf, A::AdaptiveSplit::Isotropic);
            EXPECT_EQ(split.children.size(), 8u);
            refined.insert(refined.end(), std::make_move_iterator(split.children.begin()),
                std::make_move_iterator(split.children.end()));
        }
    }
    EXPECT_EQ(inventory.size(), components);
    for (const auto& [name, count] : inventory)
        EXPECT_EQ(count, 8u);
    EXPECT_TRUE(owner_present[0] && owner_present[1]);
    EXPECT_TRUE(D::closed_contact_overlap(owner_bounds[0], owner_bounds[1]));
    EXPECT_EQ(refined.size(), 64u);
    EXPECT_EQ(owned.size(), ranks == 1 ? 8 * components : rank < 2 ? 4 * components : 0);
    EXPECT_EQ(owned.empty(), rank >= 2);
    auto foundation =
        std::make_unique<A::DistributedAdaptiveHierarchy>(owned, std::vector<A::AdaptiveFamily>{}, kernel, comm);
    const auto lookup = [&](const A::AdaptiveCellKey& key) -> const A::AdaptiveLeaf&
    {
        const auto found = std::ranges::find(owned, key, &A::AdaptiveLeaf::key);
        if (found == owned.end())
            throw std::invalid_argument("Missing rank-cover fixture leaf.");
        return *found;
    };
    auto defect = std::make_shared<int>(0);
    D::ContactHaloBeforeSend mutate = [defect, rank](const auto&, D::ContactHaloMessageKind kind, auto& packets)
    {
        if (!*defect || rank != 0 || kind != D::ContactHaloMessageKind::LeafCovers || packets.empty())
            return;
        if (*defect == 1)
            packets.front().bytes[6 * 8] ^= 1; // Stale current round.
        else if (*defect == 2)
            packets.clear(); // Missing subscribed owner completion.
        else if (*defect == 3)
            packets.front().bytes[8 * 8] ^= 1; // Foreign claimed owner.
        else
            packets.front().bytes[11 * 8] ^= 1; // Malformed reserved total.
    };
    auto context = D::ContactHaloContext{0, 0, 0, 0, 53};
    A::AdaptiveClosureOptions options;
    options.byte_budget = ample;
    auto first = std::make_unique<D::DistributedAdaptiveContactHalo>(*kernel, comm, options, context, mutate);
    first->bootstrap_canonical(owned, lookup, foundation->leaf_layout());
    {
        const auto marks = first->lease_required_refinements();
        EXPECT_TRUE(marks.marks().empty());
    }
    auto oldest = first->rank_cover_receipts();
    if (!oldest)
        throw std::invalid_argument("Missing production rank-cover receipt root.");
    const auto old_payload = payload(oldest);
    const auto baseline = first->statistics();
    if (ranks > 1 && rank < 2)
        EXPECT_GT(oldest->distinct_cover_count(), 8u);
    auto advance = [&](const auto& previous, std::uint64_t chosen_x)
    {
        // A fork starts at its supplied context. The following collective
        // delta, including a serial no-op ownership swap, advances that round.
        auto next = previous->fork_candidate(context);
        ++context.round;
        std::vector<D::ContactRetiredLeaf> retired;
        std::vector<A::AdaptiveCellKey> inserted;
        std::vector<A::AdaptiveLeaf> selected;
        for (const auto& child : refined)
        {
            const auto& ancestry = std::get<A::CartesianCellAncestry>(child.key.ancestry);
            const bool chosen = ancestry.index == std::array<std::uint64_t, 3>{chosen_x, 0, 0};
            if (rank == (ranks == 1 ? 0 : chosen ? 0 : 1))
                selected.push_back(child);
        }
        EXPECT_EQ(selected.size(), ranks == 1 ? 64u : rank == 0 ? 1u : rank == 1 ? 63u : 0u);
        for (const auto& leaf : owned)
            if (leaf.key.region == first_region &&
                std::ranges::find(selected, leaf.key, &A::AdaptiveLeaf::key) == selected.end())
                retired.push_back({leaf.key, leaf.revision});
        for (const auto& child : selected)
            if (std::ranges::find(owned, child.key, &A::AdaptiveLeaf::key) == owned.end())
                inserted.push_back(child.key);
        if (ranks == 1 && chosen_x == 3)
        {
            EXPECT_TRUE(retired.empty());
            EXPECT_TRUE(inserted.empty());
        }
        // First edit replaces exactly the first component's8 actual coarse
        // cells by64 native children. Next edit swaps the single rank0 child
        // from x=[0,.25] to x=[.75,1], preserving complete closed geometry.
        std::erase_if(owned, [&](const auto& leaf) { return leaf.key.region == first_region; });
        owned.insert(owned.end(), selected.begin(), selected.end());
        std::ranges::sort(owned, {}, &A::AdaptiveLeaf::key);
        next->apply_delta(retired, inserted, context, lookup);
        {
            const auto marks = next->lease_required_refinements();
            EXPECT_TRUE(marks.marks().empty());
        }
        return next;
    };
    *defect = fault;
    if (fault)
    {
        EXPECT_THROW((void) advance(first, 0), std::invalid_argument);
        EXPECT_EQ(payload(oldest), old_payload);
        EXPECT_EQ(first->statistics().rank_cover_pages_copied, baseline.rank_cover_pages_copied);
        return {};
    }
    auto second = advance(first, 0);
    auto middle = second->rank_cover_receipts();
    auto third = advance(second, 3);
    auto latest = third->rank_cover_receipts();
    // fork_candidate starts generation-local counters; bootstrap visits on
    // the accepted halo are not a baseline for this candidate's owner work.
    const auto current = second->statistics();
    const auto work = middle->work();
    const auto reuse = expect_physical_reuse(oldest, middle, current.rank_cover_owner_visits);
    Measurement result{fresh_bytes(old_payload, payload(middle)), work.copied_pages, work.copied_page_bytes,
        current.rank_cover_owner_visits, reuse.changed_facts, std::max(oldest->page_height(), middle->page_height()),
        reuse.edits, reuse.protected_rows, reuse.total_rows, reuse.max_page_admission};
    const std::weak_ptr<const D::ContactRankCoverReceipts> old_control(oldest), middle_control(middle);
    first.reset();
    second.reset();
    third.reset();
    foundation.reset();
    EXPECT_EQ(payload(oldest), old_payload);
    if (ranks > 1 && rank == 1)
    {
        EXPECT_TRUE(oldest->contains(0, first_domain, Point{.5L, 0, 0}));
        EXPECT_FALSE(middle->contains(0, first_domain, Point{.5L, 0, 0}));
        EXPECT_TRUE(middle->contains(0, first_domain, Point{.25L, 0, 0}));
        EXPECT_FALSE(latest->contains(0, first_domain, Point{.25L, 0, 0}));
        EXPECT_TRUE(latest->contains(0, first_domain, Point{1, 0, 0}));
    }
    oldest.reset();
    middle.reset();
    EXPECT_TRUE(old_control.expired());
    EXPECT_TRUE(middle_control.expired());
    return result;
}
} // namespace
TEST(RankCoverReceiptPersistence, ActualHeterogeneousOwnerBoundsShareDoubledUnchangedReceipts)
{
    const auto small = measure(48), large = measure(96);
    if (Tpetra::getDefaultComm()->getSize() > 1 && Tpetra::getDefaultComm()->getRank() < 2)
    {
        EXPECT_GT(small.changed_facts, 0u);
        EXPECT_GT(large.changed_facts, 0u);
        // A fixed first component is the only changed source. Tree rotations
        // may adjust adjacent homogeneous fragments; the selector symmetric
        // difference records those actual changes rather than assuming a union.
        const auto extra_branch_bytes =
            small.copied_pages * D::persistent_ordered_detail::branch_capacity * sizeof(std::shared_ptr<const void>);
        EXPECT_LE(large.fresh, small.fresh + extra_branch_bytes);
        for (const auto& measured : {small, large})
        {
            // Leaf buffers reserve their actual row count: fuller leaves can
            // increase cumulative bytes even when fewer pages are copied.
            // Bound each case by maximum admitted page size and edit/height
            // work. New selectors precede retirements, so this <32-row edit
            // can temporarily introduce at most one extra root level.
            const auto max_pages = measured.edits * (4 * (measured.height + 1) + 1);
            EXPECT_LE(measured.copied_pages, max_pages);
            EXPECT_LE(measured.copied_bytes, max_pages * measured.max_page_admission);
        }
        RecordProperty("small_protected_rank_cover_rows", std::to_string(small.protected_rows));
        RecordProperty("large_protected_rank_cover_rows", std::to_string(large.protected_rows));
        RecordProperty("small_total_rank_cover_rows", std::to_string(small.total_rows));
        RecordProperty("large_total_rank_cover_rows", std::to_string(large.total_rows));
        RecordProperty("small_new_rank_cover_bytes", std::to_string(small.fresh));
        RecordProperty("large_new_rank_cover_bytes", std::to_string(large.fresh));
        RecordProperty("small_cumulative_rank_cover_page_bytes", std::to_string(small.copied_bytes));
        RecordProperty("large_cumulative_rank_cover_page_bytes", std::to_string(large.copied_bytes));
    }
}
TEST(RankCoverReceiptPersistence, RejectsStaleMissingAndForeignCoverPublication)
{
    if (Tpetra::getDefaultComm()->getSize() == 1)
        GTEST_SKIP() << "Requires actual peer cover messages; covered by registered MPI2/4 tests.";
    for (int fault : {1, 2, 3, 4})
        (void) measure(12, fault);
}
