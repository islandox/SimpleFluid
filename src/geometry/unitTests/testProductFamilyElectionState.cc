/** @file testProductFamilyElectionState.cc
 * @brief Sparse factor contributors and immutable authority proof ownership.
 */
#include "geometry/mesh/AMR/AdaptiveFamilyLayout.hh"
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/ProductFamilyElectionState.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <set>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
using State = D::ProductFamilyElectionState;
using namespace D;
using AMR::AdaptiveBytes;
using AMR::AdaptiveOwner;
using SimpleFluid::OwnerPacket;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct ClosureScratch
{
    Budget& budget;
    std::size_t held = 0;
    ~ClosureScratch() { budget.used -= held; }
    void add(std::size_t bytes)
    {
        budget.add(bytes);
        held += bytes;
    }
    template<class T> void append(std::vector<T>& rows)
    {
        const auto before = budget.used;
        try
        {
            budget.append_capacity(rows);
        }
        catch (...)
        {
            held += budget.used - before;
            throw;
        }
        held += budget.used - before;
    }
};

#include "geometry/mesh/AMR/ProductFamilyFragmentPackets.ipp"

auto fake_proof(State& state, AMR::AdaptiveBytes bytes)
{
    return state.proof({bytes, {}});
}
constexpr std::size_t ample = 64 * 1024 * 1024;

auto arena(std::size_t limit = ample)
{
    return std::make_shared<D::Budget>(D::Budget{limit});
}
AMR::AdaptiveCellKey cell(std::uint64_t root, std::uint64_t leaf)
{
    return {3, AMR::ProductCellAncestry{{root, 1, {2, leaf}}, {root, {0, 1}}}};
}
AMR::AdaptiveBytes key(std::uint64_t root)
{
    return AMR::encode_adaptive_key(cell(root, 0));
}

TEST(ProductFamilyElectionState, FixedFactorEditSharesUnrelatedProofsWhenDirectoryDoubles)
{
    const auto measure = [](std::size_t count)
    {
        auto accepted_arena = arena();
        State accepted(accepted_arena);
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto family = key(i);
            accepted.change_member(family, cell(i, 0), 3, true);
            accepted.change_member(family, cell(i, 1), 9, true);
            const auto proof = fake_proof(accepted, AMR::AdaptiveBytes(111, 'f'));
            const std::array rows{accepted.proposal(0, cell(i, 0), 3, 9, proof)};
            accepted.set_proposals(family, rows);
        }
        EXPECT_EQ(accepted_arena->used, accepted.storage_bytes());
        const auto changed = key(0), clean = key(count - 1);
        const auto* clean_proposal = accepted.winner(clean).proposal;
        auto target_arena = arena();
        State target(target_arena, &accepted);
        EXPECT_EQ(target.work().pages, 0u);
        EXPECT_EQ(target_arena->used, accepted.storage_bytes());
        EXPECT_EQ(target.winner(clean).proposal, clean_proposal);
        target.change_member(changed, cell(0, 1), 9, false);
        const auto local = target.local(changed);
        EXPECT_TRUE(local);
        if (local)
        {
            EXPECT_EQ(local->anchor, cell(0, 0));
            EXPECT_EQ(local->anchor_revision, 3u);
            EXPECT_EQ(local->maximum, 3u);
        }
        const auto old_proof = target.winner(changed).proposal->proof;
        const std::array replacement{target.proposal(0, cell(0, 0), 3, 3, old_proof)};
        target.set_proposals(changed, replacement);
        EXPECT_EQ(target.winner(changed).maximum, 3u);
        EXPECT_EQ(target.winner(changed).proposal->proof, old_proof);
        EXPECT_EQ(target.winner(changed).proposal->anchor, accepted.winner(changed).proposal->anchor);
        EXPECT_EQ(target.dictionary_bytes(), accepted.dictionary_bytes());
        EXPECT_EQ(accepted.winner(changed).maximum, 9u);
        EXPECT_EQ(target.winner(clean).proposal, clean_proposal);
        EXPECT_EQ(target.work().contributor_edits, 1u);
        EXPECT_EQ(target.work().proposal_edits, 1u);
        EXPECT_GT(target.overlap_bytes(), 0u);
        target.finish_overlap();
        EXPECT_EQ(target_arena->used, target.storage_bytes());
        return target.work();
    };
    const auto small = measure(4096);
    const auto large = measure(8192);
    EXPECT_LT(large.pages, 2 * small.pages);
    EXPECT_LT(large.page_bytes, 2 * small.page_bytes);
}

TEST(ProductFamilyElectionState, MinimumUsesFullTypedAnchorAndMaximumTracksRetirement)
{
    auto memory = arena();
    State state(memory);
    const auto family = key(0);
    const auto a = cell(256, 2), b = cell(1, 2);
    ASSERT_LT(b, a);
    ASSERT_NE(AMR::encode_adaptive_key(a) < AMR::encode_adaptive_key(b), a < b);
    state.change_member(family, a, 41, true);
    state.change_member(family, b, 7, true);
    ASSERT_TRUE(state.local(family));
    EXPECT_EQ(state.local(family)->anchor, b);
    EXPECT_EQ(state.local(family)->maximum, 41u);
    const auto pa = fake_proof(state, {'a'}), pb = fake_proof(state, {'b'});
    const std::array proposals{state.proposal(0, a, 41, 41, pa), state.proposal(1, b, 7, 7, pb)};
    state.set_proposals(family, proposals);
    ASSERT_NE(state.winner(family).proposal, nullptr);
    EXPECT_EQ(state.winner(family).proposal->rank, 1);
    EXPECT_EQ(state.winner(family).maximum, 41u);
    auto next_memory = arena();
    State next(next_memory, &state);
    next.change_member(family, a, 41, false);
    next.set_proposals(family, std::span(proposals).last(1));
    EXPECT_EQ(next.local(family)->maximum, 7u);
    EXPECT_EQ(next.winner(family).maximum, 7u);
    EXPECT_EQ(next.winner(family).proposal, proposals[1].get());
    EXPECT_EQ(state.winner(family).maximum, 41u);
}

TEST(ProductFamilyElectionState, OneByteShortRollsBackTheCandidateWithAcceptedProofIntact)
{
    const auto family = key(0);
    auto original_arena = arena();
    State source(original_arena);
    source.change_member(family, cell(0, 0), 4, true);
    source.change_member(family, cell(0, 1), 9, true);
    const auto witness = fake_proof(source, AMR::AdaptiveBytes(353, 'p'));
    const std::array rows{source.proposal(0, cell(0, 0), 4, 9, witness)};
    source.set_proposals(family, rows);
    const auto edit = [&](const auto& memory)
    {
        State candidate(memory, &source);
        candidate.change_member(family, cell(0, 1), 9, false);
        candidate.change_member(family, cell(0, 2), 17, true);
        const auto proof = fake_proof(candidate, AMR::AdaptiveBytes(739, 'n'));
        const std::array replacement{candidate.proposal(0, cell(0, 0), 4, 17, proof)};
        candidate.set_proposals(family, replacement);
        EXPECT_EQ(candidate.local(family)->maximum, 17u);
        EXPECT_EQ(candidate.winner(family).maximum, 17u);
    };
    auto measured = arena();
    edit(measured);
    EXPECT_EQ(measured->used, 0u);
    ASSERT_GT(measured->peak, source.storage_bytes());
    auto tight = arena(measured->peak - 1);
    EXPECT_THROW(edit(tight), std::invalid_argument);
    EXPECT_EQ(tight->used, 0u);
    EXPECT_EQ(source.local(family)->maximum, 9u);
    EXPECT_EQ(source.winner(family).proposal, rows.front().get());
    tight->limit = measured->peak;
    EXPECT_NO_THROW(edit(tight));
    EXPECT_EQ(tight->used, 0u);
}

struct ElectedSequenceMetrics
{
    std::size_t body_bytes = 0, maximum_index_pages = 0, maximum_family_pages = 0;
};

ElectedSequenceMetrics check_elected_sequence(
    const AMR::AdaptiveBootstrap& detached, std::span<const AdaptiveFamilyProofView> starting)
{
    using Layout = AMR::AdaptiveFamilyLayout;
    using LayoutStorage = AdaptiveFamilyStorage;
    std::vector<AMR::AdaptiveFamily> initial;
    for (const auto proof : starting)
        initial.push_back(decode_family_proof_fragment(proof, ample));
    // The hierarchy's bootstrap intern likewise creates the exact elected
    // byte controls independently of the authority proof owners.
    initial = AMR::decode_adaptive_batch(AMR::encode_adaptive_batch({}, initial, ample), ample).families;
    auto initial_memory = arena();
    auto dictionary = std::make_unique<State>(initial_memory);
    {
        ProductElectedPartDelta seed(initial_memory);
        for (const auto& family : initial)
            seed.record(family.parent, true);
        seed.apply(*dictionary);
    }
    auto families = std::make_shared<const Layout>(LayoutStorage::build(initial, initial_memory, 0));
    std::vector<AMR::AdaptiveFamily>().swap(initial);
    auto historical = families;
    std::weak_ptr<const Layout> historical_root = historical;
    const auto initial_size = dictionary->elected_dictionary_size();
    const auto* root = (*families->begin()).parent.factor_recipe->root.get();
    const auto last_parent = std::ranges::max_element(detached.owned, {},
        [](const auto& leaf) { return std::get<AMR::ProductCellAncestry>(leaf.key.ancestry).axial.root; });
    const auto last_key = AMR::adaptive_family_key(last_parent->key, AMR::AdaptiveSplit::Axial);
    const auto axis = *families->at(last_key).parent.factor_recipe->axis;
    std::weak_ptr<const AMR::AdaptiveBytes> retired_axis = families->at(last_key).parent.factor_recipe->axis;
    EXPECT_NE(root, detached.owned.front().factor_recipe->root.get());

    struct Edit
    {
        std::unique_ptr<State> dictionary;
        std::shared_ptr<const Layout> families;
        LayoutStorage::Work work;
    };
    const auto change = [&](const AMR::AdaptiveBytes& key, const AMR::AdaptiveFamily* replacement,
                            const std::shared_ptr<Budget>& memory, std::uint64_t epoch)
    {
        auto next_dictionary = std::make_unique<State>(memory, dictionary.get());
        const auto before = families->at(key);
        LayoutStorage::Work work;
        const std::array edits{LayoutStorage::Edit{key, before.revision, replacement}};
        auto next = [&]
        {
            ProductElectedPartDelta parts(memory);
            auto value = LayoutStorage::edited(*families, edits, memory, epoch, work,
                [&](const AMR::AdaptiveLeaf* removed, const AMR::AdaptiveLeaf* inserted)
                {
                    if (removed)
                        parts.record(*removed, false);
                    if (inserted)
                        parts.record(*inserted, true);
                });
            parts.apply(*next_dictionary);
            return value;
        }();
        EXPECT_EQ(memory->used, next_dictionary->storage_bytes() + LayoutStorage::storage_bytes(next) +
                                    next_dictionary->overlap_bytes() + LayoutStorage::overlap_bytes(next));
        // Keep the immutable view private until both owners are sealed.
        next_dictionary->finish_overlap();
        LayoutStorage::finish_overlap(next);
        EXPECT_EQ(memory->used, next_dictionary->storage_bytes() + LayoutStorage::storage_bytes(next));
        return Edit{std::move(next_dictionary), std::make_shared<const Layout>(std::move(next)), work};
    };

    ElectedSequenceMetrics metrics;
    std::uint64_t epoch = 0;
    for (const auto& parent : detached.owned)
    {
        const auto split = detached.kernel->split(parent, AMR::AdaptiveSplit::Axial);
        const auto key = AMR::adaptive_family_key(parent.key, AMR::AdaptiveSplit::Axial);
        const auto old = families->at(key);
        const auto previous_anchor = AMR::decode_adaptive_key(old.anchor);
        const auto changed_anchor =
            std::ranges::find_if(split.children, [&](const auto& row) { return row.key != previous_anchor; });
        if (changed_anchor == split.children.end())
        {
            ADD_FAILURE() << "Missing alternate actual Product anchor";
            return {};
        }
        const auto witness = detached.kernel->ancestry_family(*changed_anchor, {parent.key, AMR::AdaptiveSplit::Axial});
        auto fragment = encode_family_proof_fragment(witness, ample);
        for (auto& part : fragment.parts)
            if (auto actual = dictionary->elected_part(*part))
                part = std::move(actual);
        auto replacement = decode_family_proof_fragment(fragment.view(), ample);
        EXPECT_NE(replacement.anchor, old.anchor);
        EXPECT_EQ(replacement.parent.factor_recipe->root.get(), root);
        auto next = change(key, &replacement, arena(), ++epoch);
        metrics.maximum_index_pages = std::max(metrics.maximum_index_pages, next.dictionary->work().pages);
        metrics.maximum_family_pages = std::max(metrics.maximum_family_pages, next.work.copied_pages);
        dictionary = std::move(next.dictionary);
        families = std::move(next.families);
    }
    EXPECT_EQ(dictionary->elected_dictionary_size(), initial_size);
    EXPECT_EQ(metrics.maximum_index_pages, 0u);

    // Drop one real family and its unique axis part. Test the exact
    // combined private index/root peak before publishing the accepted edit.
    auto measured = arena();
    {
        const auto probe = change(last_key, nullptr, measured, epoch + 1);
        EXPECT_FALSE(probe.dictionary->elected_part(axis));
    }
    EXPECT_EQ(measured->used, 0u);
    auto short_memory = arena(measured->peak - 1);
    const auto* accepted_root = families.get();
    EXPECT_THROW((void) change(last_key, nullptr, short_memory, epoch + 1), std::invalid_argument);
    EXPECT_EQ(short_memory->used, 0u);
    EXPECT_EQ(families.get(), accepted_root);
    EXPECT_TRUE(families->contains(last_key));
    EXPECT_TRUE(dictionary->elected_part(axis));
    short_memory->limit = measured->peak;
    auto removed = change(last_key, nullptr, short_memory, epoch + 1);
    dictionary = std::move(removed.dictionary);
    families = std::move(removed.families);
    EXPECT_FALSE(dictionary->elected_part(axis));
    EXPECT_FALSE(families->contains(last_key));
    EXPECT_FALSE(retired_axis.expired());
    historical.reset();
    EXPECT_TRUE(historical_root.expired());
    EXPECT_TRUE(retired_axis.expired());

    AdaptiveLeafBodyDelta census(arena());
    std::set<const void*> root_owners;
    std::set<AdaptiveFamilyProofPart, ProductProofPartLess> actual_parts;
    for (const auto family : *families)
    {
        root_owners.insert(family.parent.factor_recipe->root.get());
        visit_elected_product_parts(family.parent,
            [&](const auto& part)
            {
                if (!part)
                    return;
                const auto [seen, inserted] = actual_parts.insert(part);
                if (!inserted)
                {
                    EXPECT_EQ(seen->get(), part.get());
                    EXPECT_FALSE(seen->owner_before(part));
                    EXPECT_FALSE(part.owner_before(*seen));
                }
            });
        census.record(std::span(&family.parent, 1), true);
    }
    EXPECT_EQ(dictionary->elected_dictionary_size(), actual_parts.size());
    EXPECT_EQ(root_owners, (std::set<const void*>{root}));
    metrics.body_bytes = census.apply({}, AdaptiveLeafBodyRole::Family).usage.payload_bytes();
    return metrics;
}

TEST(ProductFamilyElectionState, RealAxialRootsShareRetainedAndTransportDictionaryParts)
{
    struct Metrics
    {
        std::size_t retained, dictionary, wire, workspace, edited_pages;
        ElectedSequenceMetrics elected;
    };
    const auto measure = [](std::size_t intervals) -> Metrics
    {
        AMR::ExtrudedAMRGrid grid;
        grid.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}};
        grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}};
        grid.z_edges.clear();
        std::vector<std::uint64_t> selected;
        for (std::size_t i = 0; i <= intervals; ++i)
        {
            grid.z_edges.push_back(double(i) / double(intervals));
            if (i < intervals)
                selected.push_back(2 * i);
        }
        // Include one real adjacent base parent at the first interval. Both
        // interval counts then exercise the same B+ tree height, while the
        // final interval still has exactly one elected axis owner.
        selected.push_back(1);
        std::sort(selected.begin(), selected.end());
        AMR::ExtrudedAMR product(grid);
        auto detached = product.detach_owned(selected, {'d'});
        auto memory = arena();
        State state(memory);
        struct Saved
        {
            AMR::AdaptiveBytes family, anchor;
            AMR::AdaptiveCellKey extra;
            std::shared_ptr<const State::Proof> proof;
        };
        std::vector<Saved> saved;
        const auto root = detached.owned.front().factor_recipe->root;
        const void* physical_root = nullptr;
        for (const auto& parent : detached.owned)
        {
            const auto split = detached.kernel->split(parent, AMR::AdaptiveSplit::Axial);
            const auto anchor_it = std::ranges::min_element(split.children, {}, &AMR::AdaptiveLeaf::key);
            const auto& anchor = *anchor_it;
            const auto& extra = anchor_it == split.children.begin() ? split.children.back() : split.children.front();
            const auto family = detached.kernel->ancestry_family(anchor, {parent.key, AMR::AdaptiveSplit::Axial});
            EXPECT_EQ(family.parent.factor_recipe->root, root);
            const auto fragment = encode_family_proof_fragment(family, ample);
            auto proof = state.proof(fragment.view());
            const auto roundtrip = decode_family_proof_fragment(proof->view(), ample);
            EXPECT_EQ(AMR::encode_adaptive_family(roundtrip), AMR::encode_adaptive_family(family));
            const auto* owner = roundtrip.parent.factor_recipe->root.get();
            if (!physical_root)
                physical_root = owner;
            EXPECT_EQ(owner, physical_root);
            state.change_member(family.key, anchor.key, anchor.revision, true);
            state.change_member(family.key, extra.key, anchor.revision + 2, true);
            const std::array proposals{state.proposal(0, anchor.key, anchor.revision, anchor.revision + 2, proof)};
            state.set_proposals(family.key, proposals);
            saved.push_back({family.key, family.anchor, extra.key, std::move(proof)});
        }
        EXPECT_EQ(saved.size(), intervals + 1);
        EXPECT_EQ(memory->used, state.storage_bytes());
        std::vector<ProductProofFrame> frames;
        for (const auto& row : saved)
            frames.push_back({1, row.family, row.anchor, 1, 1, row.proof->view()});
        Budget wire_memory{ample};
        ClosureScratch wire{wire_memory};
        std::vector<OwnerPacket> packets;
        const ProductProofContext context{3, 4, 5};
        encode_product_frames(frames, 1, context, wire, packets);
        EXPECT_EQ(packets.size(), 1u);
        Reader reader{packets.front().bytes};
        EXPECT_EQ(reader.word(), product_fragment_magic);
        EXPECT_EQ(reader.word(), 1u);
        EXPECT_EQ(reader.word(), context.topology);
        EXPECT_EQ(reader.word(), context.partition);
        EXPECT_EQ(reader.word(), context.transaction);
        const auto count = reader.word();
        std::size_t root_entries = 0;
        for (std::size_t i = 0; i < count; ++i)
            root_entries += std::ranges::equal(reader.bytes_view(), *root);
        EXPECT_EQ(root_entries, 1u);
        EXPECT_EQ(reader.word(), intervals + 1);
        const auto peak = wire_memory.peak;
        ProductWaveDictionary received;
        std::size_t decoded = 0;
        EXPECT_THROW(decode_product_frames(packets.front(), 1, {3, 4, 4}, received, wire,
                         [](const auto&) { ADD_FAILURE() << "Stale Product frame emitted a receipt"; }),
            std::invalid_argument);
        decode_product_frames(packets.front(), 1, context, received, wire,
            [&](const ProductProofFrame& row)
            {
                const auto family = decode_family_proof_fragment(row.proof, ample);
                EXPECT_TRUE(std::ranges::equal(family.key, row.family));
                EXPECT_TRUE(std::ranges::equal(family.anchor, row.anchor));
                ++decoded;
            });
        EXPECT_EQ(decoded, intervals + 1);

        auto candidate_memory = arena();
        State candidate(candidate_memory, &state);
        const auto& changed = saved.front();
        const auto old = state.winner(changed.family);
        candidate.change_member(changed.family, changed.extra, old.maximum, false);
        const auto local = candidate.local(changed.family);
        EXPECT_TRUE(local);
        if (!local)
            return {};
        EXPECT_EQ(local->anchor, old.proposal->anchor);
        EXPECT_LT(local->maximum, old.maximum);
        const std::array replacement{
            candidate.proposal(0, local->anchor, local->anchor_revision, local->maximum, old.proposal->proof)};
        candidate.set_proposals(changed.family, replacement);
        EXPECT_EQ(candidate.winner(changed.family).proposal->proof, old.proposal->proof);
        EXPECT_EQ(candidate.dictionary_bytes(), state.dictionary_bytes());
        EXPECT_EQ(candidate.winner(saved.back().family).proposal, state.winner(saved.back().family).proposal);
        candidate.finish_overlap();
        EXPECT_EQ(candidate_memory->used, candidate.storage_bytes());
        std::vector<AdaptiveFamilyProofView> starting;
        for (const auto& row : saved)
            starting.push_back(row.proof->view());
        return {state.storage_bytes(), state.dictionary_bytes(), packets.front().bytes.size(),
            std::max(peak, wire_memory.peak), candidate.work().pages, check_elected_sequence(detached, starting)};
    };
    const auto small = measure(64), large = measure(128);
    EXPECT_LT(large.retained, 3 * small.retained);
    EXPECT_LT(large.dictionary, 3 * small.dictionary);
    EXPECT_LT(large.wire, 3 * small.wire);
    EXPECT_LT(large.workspace, 3 * small.workspace);
    EXPECT_LT(large.edited_pages, 2 * small.edited_pages);
    EXPECT_GT(small.elected.body_bytes, 0u);
    EXPECT_LT(large.elected.body_bytes, 3 * small.elected.body_bytes);
    EXPECT_LT(large.elected.maximum_family_pages, 2 * small.elected.maximum_family_pages);
}

TEST(ProductFamilyElectionState, DuplicateForeignAndStaleReceiptsReject)
{
    const auto family = key(2), anchor = AMR::encode_adaptive_key(cell(1, 0));
    constexpr int ranks = 4;
    const auto shard = authority(family, ranks);
    const ProductProofFrame receipt{1, family, anchor, 2, 7, {}};
    EXPECT_NO_THROW(validate_product_proposal_route(receipt, shard, ranks, false));
    EXPECT_THROW(validate_product_proposal_route(receipt, shard, ranks, true), std::invalid_argument);
    EXPECT_THROW(validate_product_proposal_route(receipt, (shard + 1) % ranks, ranks, false), std::invalid_argument);
    auto foreign = receipt;
    foreign.peer = ranks;
    EXPECT_THROW(validate_product_proposal_route(foreign, shard, ranks, false), std::invalid_argument);
    const AMR::AdaptiveOwner actual{anchor, 1, 2};
    EXPECT_NO_THROW(validate_product_anchor_receipt(anchor, 1, 2, actual));
    EXPECT_THROW(validate_product_anchor_receipt(anchor, 2, 2, actual), std::invalid_argument);
    EXPECT_THROW(validate_product_anchor_receipt(anchor, 1, 1, actual), std::invalid_argument);
    EXPECT_THROW(validate_product_anchor_receipt(family, 1, 2, actual), std::invalid_argument);
}

TEST(ProductFamilyElectionState, MalformedDictionarySelectorAndTruncatedFrameRejectBeforeReceipt)
{
    const auto family = key(2), anchor = AMR::encode_adaptive_key(cell(1, 0));
    const AdaptiveBytes record{'p'};
    const std::array parts{std::make_shared<const AdaptiveBytes>(AdaptiveBytes{'d'})};
    std::array frames{ProductProofFrame{0, family, anchor, 2, 7, {record, parts}}};
    const ProductProofContext context{3, 4, 5};
    Budget memory{ample};
    ClosureScratch scratch{memory};
    std::vector<OwnerPacket> packets;
    encode_product_frames(frames, 1, context, scratch, packets);
    ASSERT_EQ(packets.size(), 1u);
    ASSERT_GE(packets.front().bytes.size(), sizeof(std::uint64_t));
    // One dictionary value gives the only selector ID 0. Its final encoded
    // u64 is changed to 1, independently known to be outside that dictionary.
    auto foreign_selector = packets.front();
    std::fill(foreign_selector.bytes.end() - sizeof(std::uint64_t), foreign_selector.bytes.end(), 0);
    foreign_selector.bytes[foreign_selector.bytes.size() - sizeof(std::uint64_t)] = 1;
    auto truncated = packets.front();
    truncated.bytes.pop_back();
    ProductWaveDictionary received;
    std::size_t receipts = 0;
    const auto receive = [&](const ProductProofFrame&) { ++receipts; };
    EXPECT_THROW(
        decode_product_frames(foreign_selector, 1, context, received, scratch, receive), std::invalid_argument);
    EXPECT_EQ(receipts, 0u);
    EXPECT_THROW(decode_product_frames(truncated, 1, context, received, scratch, receive), std::invalid_argument);
    EXPECT_EQ(receipts, 0u);
    EXPECT_NO_THROW(decode_product_frames(packets.front(), 1, context, received, scratch, receive));
    EXPECT_EQ(receipts, 1u);
}

TEST(ProductFamilyElectionState, CollectiveMalformedFrameRejectsWithoutChangingAcceptedState)
{
    AMR::ExtrudedAMRGrid grid;
    grid.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    grid.base_cells = {{0, 1, 2, 3}};
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR product(grid);
    const std::array<std::uint64_t, 1> selected{0};
    const auto detached = product.detach_owned(selected, {'f'});
    ASSERT_EQ(detached.owned.size(), 1u);
    const auto family = detached.kernel->split(detached.owned.front(), AMR::AdaptiveSplit::Axial).family;
    const auto fragment = encode_family_proof_fragment(family, ample);
    ASSERT_FALSE(fragment.parts.empty());
    const auto anchor = AMR::decode_adaptive_key(family.anchor);
    const auto revision = family.parent.revision;
    auto accepted_memory = arena();
    State accepted(accepted_memory);
    accepted.change_member(family.key, anchor, revision, true);
    const auto proof = accepted.proof(fragment.view());
    const std::array proposals{accepted.proposal(0, anchor, revision, revision, proof)};
    accepted.set_proposals(family.key, proposals);
    const auto* accepted_proposal = accepted.winner(family.key).proposal;
    const auto accepted_bytes = accepted_memory->used;
    const auto exact = AMR::encode_adaptive_family(family);
    const auto comm = Tpetra::getDefaultComm();
    const bool malformed_rank = comm->getRank() == comm->getSize() - 1;
    const ProductProofContext context{3, 4, 5};

    // The selector failure precedes receipt delivery. Trailing bytes instead
    // fail after every rank has decoded a valid private receipt. Both must
    // propagate the last rank's rejection to all ranks before publication.
    for (const bool trailing : {false, true})
    {
        Budget memory{ample};
        {
            ClosureScratch scratch{memory};
            std::array frames{ProductProofFrame{0, family.key, family.anchor, revision, revision, fragment.view()}};
            std::vector<OwnerPacket> packets;
            encode_product_frames(frames, 1, context, scratch, packets);
            ASSERT_EQ(packets.size(), 1u);
            auto malformed = packets.front();
            if (malformed_rank)
            {
                if (trailing)
                    malformed.bytes.push_back(0);
                else
                    // The real Product proof has several parts; UINT64_MAX
                    // is outside its dictionary regardless of their ordering.
                    std::fill(malformed.bytes.end() - sizeof(std::uint64_t), malformed.bytes.end(), char(0xff));
            }
            ProductWaveDictionary received;
            std::optional<AMR::AdaptiveFamily> candidate;
            std::size_t receipts = 0;
            const auto receive = [&](const ProductProofFrame& row)
            {
                candidate.emplace(decode_family_proof_fragment(row.proof, ample));
                EXPECT_EQ(AMR::encode_adaptive_family(*candidate), exact);
                ++receipts;
            };
            EXPECT_THROW(
                collective(*comm, [&] { decode_product_frames(malformed, 1, context, received, scratch, receive); }),
                std::invalid_argument);
            EXPECT_EQ(receipts, std::size_t(trailing || !malformed_rank));
            candidate.reset();
            EXPECT_EQ(accepted.winner(family.key).proposal, accepted_proposal);
            EXPECT_EQ(accepted.winner(family.key).maximum, revision);
            ASSERT_TRUE(accepted.local(family.key));
            EXPECT_EQ(accepted.local(family.key)->anchor, anchor);
            EXPECT_EQ(accepted_proposal->proof, proof);
            EXPECT_EQ(accepted_memory->used, accepted_bytes);
            receipts = 0;
            EXPECT_NO_THROW(collective(
                *comm, [&] { decode_product_frames(packets.front(), 1, context, received, scratch, receive); }));
            EXPECT_EQ(receipts, 1u);
        }
        EXPECT_EQ(memory.used, 0u);
    }
}

} // namespace
