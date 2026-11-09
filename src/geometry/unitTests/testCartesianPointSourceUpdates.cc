#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using RowMap = std::map<AdaptiveCellKey, IncidenceRowHandle>;
using LeafMap = std::map<AdaptiveCellKey, AdaptiveLeaf>;
using FullPointPairs = std::vector<std::pair<AdaptiveBytes, AdaptiveBytes>>;
FullPointPairs keys(const ContactPointArray& points)
{
    FullPointPairs result;
    for (const auto value : points)
        result.emplace_back(PLIC::canonical_key(value.physical), PLIC::canonical_key(value.normalized));
    return result;
}
void consume(std::vector<IncidenceRowDelivery>& output, RowMap& rows, Budget& budget)
{
    for (const auto& delivery : output)
        rows[delivery.leaf->key] = delivery.row;
    const auto bytes = output.capacity() * sizeof(IncidenceRowDelivery);
    std::vector<IncidenceRowDelivery>().swap(output);
    budget.used -= bytes;
}
void consume(IncidenceDelta& output, RowMap& rows, Budget& budget)
{
    for (const auto& old : output.retired)
        rows.erase(old.key);
    for (const auto& delivery : output.changed)
        rows[delivery.leaf->key] = delivery.row;
    auto bytes = output.retired.capacity() * sizeof(IncidenceCellVersion) +
                 output.changed.capacity() * sizeof(IncidenceRowDelivery);
    for (const auto& old : output.retired)
        bytes += key_capacity(old.key);
    output = {};
    budget.used -= bytes;
}
class ExplicitKernel final : public AdaptiveProviderKernel
{
    const AdaptiveProviderKernel& inner_;

public:
    explicit ExplicitKernel(const AdaptiveProviderKernel& value) : inner_(value) {}
    AdaptiveBytes descriptor() const override { return inner_.descriptor(); }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        return inner_.geometry(leaf);
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override { return inner_.surfaces(leaf); }
    AdaptiveSplitResult split(const AdaptiveLeaf& leaf, AdaptiveSplit kind) const override
    {
        return inner_.split(leaf, kind);
    }
    AdaptiveTrace traces(const AdaptiveLeaf& leaf) const override { return inner_.traces(leaf); }
    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf& leaf) const override
    {
        return inner_.contact_capabilities(leaf);
    }
    std::vector<AdaptiveContactImage> closed_contact_images(const AdaptiveLeaf& leaf) const override
    {
        return inner_.closed_contact_images(leaf);
    }
    ContactSourcePoint contact_point(const AdaptiveLeaf& leaf, const ContactSourcePoint& point) const override
    {
        return inner_.contact_point(leaf, point);
    }
};
TEST(CartesianPointSourceUpdates, SparseRefinementsShareFlatValueBlocksWithoutHistoricalProofPins)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    auto config = grid();
    const auto kernel = cartesian_adaptive_kernel(config);
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    LeafMap current;
    const auto original = leaves(config);
    for (const auto& leaf : original)
        current.emplace(leaf.key, leaf);
    auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf& { return current.at(key); };
    std::shared_ptr<const ContactSourcePointSnapshot> latest;
    std::weak_ptr<const ContactSourcePointSnapshot> old_proof;
    std::vector<std::pair<IncidenceRowHandle, FullPointPairs>> escaped;
    struct StoredCapture
    {
        std::shared_ptr<const ContactSourcePointView> view;
        CapturedSourcePointReference reference;
        ContactSourcePoint point;
    };
    std::vector<StoredCapture> stored_captures;
    std::size_t external_identity_admission = 0;
    std::vector<std::weak_ptr<const PLIC::details::GeometryIdentityBacking>> plane_owners;
    {
        IdentityBudget identities;
        DistributedAdaptiveIncidenceSession session(*kernel, comm, {}, {}, *budget, identities, budget);
        RowMap rows;
        auto output = session.bootstrap(original, lookup);
        for (const auto& delivery : output)
            escaped.emplace_back(delivery.row, keys(delivery.row->points));
        consume(output, rows, *budget);
        old_proof = session.point_pool().source();
        ASSERT_TRUE(session.point_pool().source());
        ASSERT_EQ(session.point_pool().source()->storage_block_count(), 1);
        ASSERT_TRUE(session.point_pool().source()->component_indices({0}));
        ContactSourcePointSnapshot::LookupWork bootstrap_read;
        (void) session.point_pool().source()->value({0}, &bootstrap_read);
        EXPECT_EQ(bootstrap_read.directory_comparisons, 1u);
        constexpr std::array<std::array<std::uint64_t, 3>, 4> targets{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}};
        constexpr std::array<std::size_t, 4> appended{19, 14, 14, 10};
        std::size_t expected_points = 27;
        for (std::size_t step = 0; step != targets.size(); ++step)
        {
            const auto previous = session.point_pool().source();
            IncidencePointDirectory<> shadow(
                session.point_pool(), *budget, budget, session.point_pool().source_arena());
            const AdaptiveCellKey parent{config.region_id, CartesianCellAncestry{0, targets[step]}};
            const auto before = rows.at(parent);
            const auto split = kernel->split(current.at(parent), AdaptiveSplit::Isotropic);
            const std::array retired{IncidenceCellVersion{parent, before->revision, before->incidence_generation}};
            current.erase(parent);
            std::vector<AdaptiveCellKey> inserted;
            for (const auto& child : split.children)
            {
                current.emplace(child.key, child);
                inserted.push_back(child.key);
            }
            ContactHaloContext context;
            context.round = step + 1;
            auto change = session.apply_delta(retired, inserted, context, lookup);
            consume(change, rows, *budget);
            expected_points += appended[step];
            latest = session.point_pool().source();
            ASSERT_TRUE(latest);
            EXPECT_EQ(latest->size(), expected_points);
            EXPECT_EQ(latest->storage_block_count(), step + 2);
            EXPECT_EQ(latest->new_storage_records(), appended[step]);
            EXPECT_EQ(latest->verified_prefix_records(), 0u);
            EXPECT_GT(latest->index_metadata_bytes(), 0u);
            EXPECT_GT(latest->index_build_work().pages.copied_pages, 0u);
            EXPECT_GT(latest->index_build_admitted_bytes(), 0u);
            EXPECT_EQ(latest->slot_namespace(), previous->slot_namespace());
            for (std::size_t i = 0; i < previous->size(); ++i)
            {
                EXPECT_EQ(latest->value({i}), previous->value({i}));
                EXPECT_EQ(latest->component_indices({i}), previous->component_indices({i}));
            }
            for (std::size_t i = previous->size(); i < latest->size(); ++i)
                ASSERT_EQ(shadow.intern(latest->value({i})), i);
            auto capture = ContactSourcePointView::create(previous, *budget, budget, shadow.source_arena());
            const auto stored = capture->capture(shadow, previous->size());
            capture->freeze();
            ASSERT_TRUE(capture->stored_location(stored));
            stored_captures.push_back({capture, stored, latest->value({previous->size()})});
            const auto used_before_rejected = budget->used;
            const auto spent = budget->cumulative;
            const std::shared_ptr<const ContactSourcePointSnapshot> foreign_control(latest.get(), [](const auto*) {});
            EXPECT_FALSE(shadow.publish_source(foreign_control));
            EXPECT_FALSE(shadow.publish_source(previous));
            EXPECT_EQ(budget->used, used_before_rejected);
            EXPECT_EQ(budget->cumulative, spent);
            if (step == 0)
            {
                std::size_t required = 0;
                {
                    IncidencePointDirectory<> probe(shadow, *budget, budget, shadow.source_arena());
                    const auto live = budget->used;
                    budget->peak = live;
                    ASSERT_TRUE(probe.publish_source(latest));
                    required = budget->peak - live;
                }
                ASSERT_GT(required, 0u);
                const auto live = budget->used;
                const auto limit = budget->limit;
                budget->limit = live + required - 1;
                EXPECT_THROW((void) shadow.publish_source(latest), std::invalid_argument);
                EXPECT_EQ(budget->used, live);
                EXPECT_EQ(shadow.source(), previous);
                EXPECT_EQ(capture->get(stored), latest->value({previous->size()}));
                budget->limit = live + required;
                budget->peak = live;
                const bool published = shadow.publish_source(latest);
                EXPECT_EQ(budget->peak, budget->limit);
                budget->limit = limit;
                ASSERT_TRUE(published);
            }
            else
                ASSERT_TRUE(shadow.publish_source(latest));
            EXPECT_EQ(shadow.physical_pool().size(), 0u);
            for (const auto& saved : stored_captures)
                EXPECT_EQ(saved.view->get(saved.reference), saved.point);
            EXPECT_EQ(session.point_pool().physical_pool().size(), 0);
            EXPECT_EQ(session.statistics().source_membership_cells, current.size());
            EXPECT_EQ(session.statistics().retained_explicit_line_records, 0);
            EXPECT_EQ(session.statistics().retained_stride_line_records, 0);
            const auto& saved = session.statistics();
            EXPECT_GT(saved.retained_saved_layers, 0);
            EXPECT_EQ(saved.retained_explicit_layer_input_slots, 0);
            EXPECT_EQ(saved.retained_explicit_layer_output_slots, 0);
            EXPECT_EQ(saved.retained_explicit_layer_edge_result_slots, 0);
            EXPECT_EQ(saved.retained_layer_endpoint_identity_slots, 0);
            EXPECT_GT(saved.retained_certified_layer_input_slots + saved.retained_certified_layer_output_slots, 0);
            EXPECT_GT(saved.retained_certified_layer_descriptor_bytes, 0);
            std::set<const void*> seen;
            std::size_t actual_selector_records = 0;
            latest->visit_borrowed_payload(
                [&](const ContactPayloadAllocation& item)
                {
                    if (item.identity.role == ContactPayloadRole::SourcePointSelectors &&
                        seen.insert(item.identity.address).second)
                        actual_selector_records += item.live_count;
                });
            EXPECT_EQ(actual_selector_records, expected_points); // 84, not27+46+60+74+84.
            std::size_t old_lookup_buffers = 0, scalar_indexes = 0, point_indexes = 0;
            latest->visit_borrowed_payload(
                [&](const ContactPayloadAllocation& item)
                {
                    old_lookup_buffers += item.identity.role == ContactPayloadRole::SourcePointLookup;
                    scalar_indexes += item.identity.role == ContactPayloadRole::SourcePointScalarIndex;
                    point_indexes += item.identity.role == ContactPayloadRole::SourcePointIdentityIndex;
                });
            EXPECT_EQ(old_lookup_buffers, 0u);
            EXPECT_GT(scalar_indexes, 0u);
            EXPECT_GT(point_indexes, 0u);
            for (const auto& [row, expected] : escaped)
                EXPECT_EQ(keys(row->points), expected);
        }
        rows.clear();
        external_identity_admission = identities.admitted_bytes();
        const auto& ledger = identities.borrowed_geometry_ledger();
        RecordProperty("external_identity_owner_count", ledger.owner_count());
        RecordProperty("external_identity_payload_admission", ledger.payload_bytes());
        RecordProperty("external_identity_bookkeeping_admission", ledger.bookkeeping_bytes());
        EXPECT_EQ(external_identity_admission, ledger.payload_bytes() + ledger.bookkeeping_bytes());
        session.for_each_borrowed_plane_identity([&](const auto& plane)
            { plane_owners.emplace_back(PLIC::details::GeometryIdentityAccess::backing(plane)); });
    }
    // The external identity ledger owns its pins for the caller's admission
    // scope. Its destructor has now destroyed those owners; return that scope's
    // credit separately from row/source self-releasing owners.
    EXPECT_TRUE(std::ranges::all_of(plane_owners, [](const auto& owner) { return owner.expired(); }));
    const auto monotone_spent = budget->cumulative;
    budget->used -= external_identity_admission;
    EXPECT_EQ(budget->cumulative, monotone_spent);
    RecordProperty("external_identity_admission_released", external_identity_admission);
    // Current and escaped historical roots share value storage. Visit each
    // root twice and deduplicate actual allocation identity, not admission bytes.
    std::set<std::pair<const void*, ContactPayloadRole>> seen_payload;
    std::size_t selector_records = 0, first_unique = 0;
    const auto collect = [&](const ContactPayloadAllocation& item)
    {
        if (seen_payload.emplace(item.identity.address, item.identity.role).second &&
            item.identity.role == ContactPayloadRole::SourcePointSelectors)
            selector_records += item.live_count;
    };
    latest->visit_borrowed_payload(collect);
    for (const auto& escaped_row : escaped)
        escaped_row.first->points.visit_borrowed_payload(collect);
    first_unique = seen_payload.size();
    latest->visit_borrowed_payload(collect);
    for (const auto& escaped_row : escaped)
        escaped_row.first->points.visit_borrowed_payload(collect);
    EXPECT_EQ(seen_payload.size(), first_unique);
    EXPECT_EQ(selector_records, 84);
    EXPECT_FALSE(old_proof.expired()); // Explicit escaping original row bindings still own the old proof.
    for (const auto& [row, expected] : escaped)
        EXPECT_EQ(keys(row->points), expected);
    escaped.clear();
    EXPECT_FALSE(old_proof.expired()); // A stored suffix also pins its captured prefix.
    for (const auto& saved : stored_captures)
        EXPECT_EQ(saved.view->get(saved.reference), saved.point);
    stored_captures.clear();
    EXPECT_TRUE(old_proof.expired()); // Latest shared data blocks do not pin its historical snapshot/manifest.
    ASSERT_TRUE(latest);
    EXPECT_EQ(latest->size(), 84);
    for (std::size_t i = 0; i != latest->size(); ++i)
        EXPECT_TRUE(latest->find_components(*latest->component_indices({i})));
    latest.reset();
    old_proof.reset();
    EXPECT_EQ(budget->used, 0);
}
TEST(CartesianPointSourceUpdates, LargerAcceptedPrefixDoesNotCopyOrReindexPointValues)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    struct Work
    {
        std::size_t old_points = 0, new_points = 0, new_axes = 0, directory_bytes = 0, admitted_bytes = 0;
    };
    const auto refine = [](std::uint64_t x_cells, Work& work)
    {
        auto config = grid();
        config.block_cells[0] = x_cells;
        config.extent[0] = double(x_cells) / 2;
        const auto kernel = cartesian_adaptive_kernel(config);
        auto budget = std::make_shared<Budget>(Budget{ample});
        auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
        IdentityBudget identities;
        DistributedAdaptiveIncidenceSession session(*kernel, comm, {}, {}, *budget, identities, budget);
        std::vector<AdaptiveLeaf> original;
        LeafMap current;
        for (std::uint64_t z = 0; z < 2; ++z)
            for (std::uint64_t y = 0; y < 2; ++y)
                for (std::uint64_t x = 0; x < x_cells; ++x)
                {
                    auto leaf =
                        cartesian_adaptive_leaf(config, {config.region_id, CartesianCellAncestry{0, {x, y, z}}}, {'d'});
                    current.emplace(leaf.key, leaf);
                    original.push_back(std::move(leaf));
                }
        const auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf& { return current.at(key); };
        RowMap rows;
        auto initial = session.bootstrap(original, lookup);
        consume(initial, rows, *budget);
        const auto accepted = session.point_pool().source();
        ASSERT_TRUE(accepted);
        using BufferKey = std::pair<ContactPayloadRole, const void*>;
        using Inventory = std::map<BufferKey, std::size_t>;
        const auto value_buffers = [](const ContactSourcePointSnapshot& snapshot)
        {
            Inventory result;
            snapshot.visit_borrowed_payload(
                [&](const ContactPayloadAllocation& item)
                {
                    const auto role = item.identity.role;
                    if (role == ContactPayloadRole::SourcePointComponents ||
                        role == ContactPayloadRole::SourcePointSelectors ||
                        role == ContactPayloadRole::SourcePointLookup ||
                        role == ContactPayloadRole::SourcePointExceptions)
                        result.emplace(BufferKey{role, item.identity.address}, item.payload_bytes);
                });
            return result;
        };
        const auto accepted_buffers = value_buffers(*accepted);
        const auto parent = original.front().key;
        const auto split = kernel->split(current.at(parent), AdaptiveSplit::Isotropic);
        const std::array retired{
            IncidenceCellVersion{parent, rows.at(parent)->revision, rows.at(parent)->incidence_generation}};
        current.erase(parent);
        std::vector<AdaptiveCellKey> inserted;
        for (const auto& child : split.children)
        {
            current.emplace(child.key, child);
            inserted.push_back(child.key);
        }
        ContactHaloContext context;
        context.round = 1;
        auto update = session.apply_delta(retired, inserted, context, lookup);
        consume(update, rows, *budget);
        const auto recaptured = session.point_pool().source();
        ASSERT_TRUE(recaptured);
        ASSERT_NE(recaptured, accepted);
        EXPECT_EQ(recaptured->verified_prefix_records(), 0u);
        const auto new_buffers = value_buffers(*recaptured);
        for (const auto& [key, bytes] : accepted_buffers)
        {
            ASSERT_TRUE(new_buffers.contains(key));
            EXPECT_EQ(new_buffers.at(key), bytes);
        }
        work = {accepted->size(), recaptured->new_storage_records(), recaptured->indexed_axis_values(),
            recaptured->copied_directory_bytes(), recaptured->value_build_admitted_bytes()};
    };
    Work small, large;
    ASSERT_NO_FATAL_FAILURE(refine(2, small));
    ASSERT_NO_FATAL_FAILURE(refine(8, large));
    EXPECT_GE(large.old_points, 2 * small.old_points);
    EXPECT_GT(small.new_points, 0u);
    EXPECT_EQ(large.new_points, small.new_points);
    EXPECT_GT(small.new_axes, 0u);
    EXPECT_EQ(large.new_axes, small.new_axes);
    EXPECT_GT(small.directory_bytes, 0u);
    EXPECT_EQ(large.directory_bytes, small.directory_bytes);
    EXPECT_GT(small.admitted_bytes, 0u);
    EXPECT_EQ(large.admitted_bytes, small.admitted_bytes);
}

TEST(CartesianPointSourceUpdates, SameControlTranslatedAndNonbinaryRefinementMatchesExplicitAndFresh)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    for (bool nonbinary : {false, true})
    {
        auto config = grid();
        config.origin = nonbinary ? std::array<double, 3>{.1, -.3, 2.7} : std::array<double, 3>{.25, -.5, 1.5};
        config.extent = nonbinary ? std::array<double, 3>{.7, 1.1, 2.3} : std::array<double, 3>{2, 4, 8};
        const auto kernel = cartesian_adaptive_kernel(config);
        ExplicitKernel explicit_kernel(*kernel);
        auto native_budget = std::make_shared<Budget>(Budget{ample}),
             explicit_budget = std::make_shared<Budget>(Budget{ample});
        auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
        IdentityBudget native_ids, explicit_ids;
        DistributedAdaptiveIncidenceSession native(*kernel, comm, {}, {}, *native_budget, native_ids, native_budget);
        DistributedAdaptiveIncidenceSession baseline(
            explicit_kernel, comm, {}, {}, *explicit_budget, explicit_ids, explicit_budget);
        const auto original = leaves(config);
        LeafMap current;
        for (const auto& leaf : original)
            current.emplace(leaf.key, leaf);
        auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf& { return current.at(key); };
        RowMap actual, expected;
        auto a = native.bootstrap(original, lookup), b = baseline.bootstrap(original, lookup);
        consume(a, actual, *native_budget);
        consume(b, expected, *explicit_budget);
        const auto parent = original.front().key;
        const auto split = kernel->split(current.at(parent), AdaptiveSplit::Isotropic);
        const std::array old_a{
            IncidenceCellVersion{parent, actual.at(parent)->revision, actual.at(parent)->incidence_generation}};
        const std::array old_b{
            IncidenceCellVersion{parent, expected.at(parent)->revision, expected.at(parent)->incidence_generation}};
        current.erase(parent);
        std::vector<AdaptiveCellKey> inserted;
        for (const auto& child : split.children)
        {
            current.emplace(child.key, child);
            inserted.push_back(child.key);
        }
        ContactHaloContext context;
        context.round = 1;
        std::string native_rejection, baseline_rejection;
        try
        {
            auto delta = native.apply_delta(old_a, inserted, context, lookup);
            consume(delta, actual, *native_budget);
        }
        catch (const std::exception& failure)
        {
            native_rejection = failure.what();
        }
        try
        {
            auto delta = baseline.apply_delta(old_b, inserted, context, lookup);
            consume(delta, expected, *explicit_budget);
        }
        catch (const std::exception& failure)
        {
            baseline_rejection = failure.what();
        }
        ASSERT_EQ(native_rejection, baseline_rejection);
        if (!nonbinary)
            ASSERT_TRUE(baseline_rejection.empty());
        if (!baseline_rejection.empty())
        {
            RecordProperty("preserved_nonbinary_baseline_rejection", baseline_rejection);
            continue;
        }
        ASSERT_EQ(actual.size(), expected.size());
        for (const auto& [key, row] : actual)
        {
            ASSERT_TRUE(expected.contains(key));
            EXPECT_EQ(row->revision, expected.at(key)->revision);
            EXPECT_EQ(row->incidence_generation, expected.at(key)->incidence_generation);
            EXPECT_EQ(keys(row->points), keys(expected.at(key)->points));
        }
        auto fresh_budget = std::make_shared<Budget>(Budget{ample});
        IdentityBudget fresh_ids;
        DistributedAdaptiveIncidenceSession fresh(*kernel, comm, {}, context, *fresh_budget, fresh_ids, fresh_budget);
        std::vector<AdaptiveLeaf> all;
        for (const auto& [key, leaf] : current)
            all.push_back(leaf);
        auto fresh_rows = fresh.bootstrap(all, lookup);
        for (const auto& delivery : fresh_rows)
            EXPECT_EQ(keys(delivery.row->points), keys(actual.at(delivery.leaf->key)->points));
        const auto bytes = fresh_rows.capacity() * sizeof(IncidenceRowDelivery);
        {
            decltype(fresh_rows) destroyed;
            destroyed.swap(fresh_rows);
        }
        fresh_budget->used -= bytes;
        std::vector<IncidenceCellVersion> retire_a, retire_b;
        for (const auto& child : split.children)
        {
            retire_a.push_back({child.key, actual.at(child.key)->revision, actual.at(child.key)->incidence_generation});
            retire_b.push_back(
                {child.key, expected.at(child.key)->revision, expected.at(child.key)->incidence_generation});
            current.erase(child.key);
        }
        current.emplace(parent, original.front());
        const std::array restored{parent};
        ++context.round;
        native_rejection.clear();
        baseline_rejection.clear();
        try
        {
            auto delta = native.apply_delta(retire_a, restored, context, lookup);
            consume(delta, actual, *native_budget);
        }
        catch (const std::exception& failure)
        {
            native_rejection = failure.what();
        }
        try
        {
            auto delta = baseline.apply_delta(retire_b, restored, context, lookup);
            consume(delta, expected, *explicit_budget);
        }
        catch (const std::exception& failure)
        {
            baseline_rejection = failure.what();
        }
        ASSERT_EQ(native_rejection, baseline_rejection);
        if (!nonbinary)
            ASSERT_TRUE(baseline_rejection.empty());
        if (!baseline_rejection.empty())
        {
            RecordProperty("preserved_nonbinary_coarsen_rejection", baseline_rejection);
            continue;
        }
        ASSERT_EQ(actual.size(), expected.size());
        for (const auto& [key, row] : actual)
        {
            ASSERT_TRUE(expected.contains(key));
            EXPECT_EQ(row->revision, expected.at(key)->revision);
            EXPECT_EQ(row->incidence_generation, expected.at(key)->incidence_generation);
            EXPECT_EQ(keys(row->points), keys(expected.at(key)->points));
        }
        EXPECT_EQ(native.statistics().source_membership_cells, original.size());
        EXPECT_EQ(native.statistics().explicit_coordinate_points, 0);
        auto restored_budget = std::make_shared<Budget>(Budget{ample});
        IdentityBudget restored_ids;
        DistributedAdaptiveIncidenceSession fresh_restored(
            *kernel, comm, {}, context, *restored_budget, restored_ids, restored_budget);
        auto restored_rows = fresh_restored.bootstrap(original, lookup);
        for (const auto& delivery : restored_rows)
            EXPECT_EQ(keys(delivery.row->points), keys(actual.at(delivery.leaf->key)->points));
        const auto restored_bytes = restored_rows.capacity() * sizeof(IncidenceRowDelivery);
        {
            decltype(restored_rows) destroyed;
            destroyed.swap(restored_rows);
        }
        restored_budget->used -= restored_bytes;
    }
}
TEST(CartesianPointSourceUpdates, RegularExplicitRegularRestorationKeepsNativeRetirementControls)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    // B is separated from A by two whole cells. Its revision-only edit exposes
    // sticky native provenance; its refinement also publishes new point values
    // while A still has genuine explicit support contributors.
    for (const bool refine_distant : {false, true})
    {
        SCOPED_TRACE(refine_distant);
        auto config = grid();
        config.block_cells = {4, 2, 2};
        const auto kernel = cartesian_adaptive_kernel(config);
        ExplicitKernel explicit_kernel(*kernel);
        std::vector<AdaptiveLeaf> original;
        for (std::uint64_t z = 0; z != 2; ++z)
            for (std::uint64_t y = 0; y != 2; ++y)
                for (std::uint64_t x = 0; x != 4; ++x)
                    original.push_back(cartesian_adaptive_leaf(
                        config, {config.region_id, CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
        ASSERT_EQ(original.size(), 16);
        LeafMap current;
        for (const auto& leaf : original)
            current.emplace(leaf.key, leaf);
        const auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf& { return current.at(key); };
        auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
        auto actual_budget = std::make_shared<Budget>(Budget{ample}),
             baseline_budget = std::make_shared<Budget>(Budget{ample});
        IdentityBudget actual_ids, baseline_ids;
        DistributedAdaptiveIncidenceSession actual(*kernel, comm, {}, {}, *actual_budget, actual_ids, actual_budget);
        DistributedAdaptiveIncidenceSession baseline(
            explicit_kernel, comm, {}, {}, *baseline_budget, baseline_ids, baseline_budget);
        RowMap actual_rows, baseline_rows;
        auto a = actual.bootstrap(original, lookup), b = baseline.bootstrap(original, lookup);
        consume(a, actual_rows, *actual_budget);
        consume(b, baseline_rows, *baseline_budget);
        ContactHaloContext context;
        const auto compare = [&]
        {
            ASSERT_EQ(actual_rows.size(), current.size());
            ASSERT_EQ(actual_rows.size(), baseline_rows.size());
            for (const auto& [cell, row] : actual_rows)
            {
                ASSERT_TRUE(baseline_rows.contains(cell));
                EXPECT_EQ(row->revision, baseline_rows.at(cell)->revision);
                EXPECT_EQ(row->incidence_generation, baseline_rows.at(cell)->incidence_generation);
                EXPECT_EQ(keys(row->points), keys(baseline_rows.at(cell)->points));
            }
            auto fresh_budget = std::make_shared<Budget>(Budget{ample});
            IdentityBudget fresh_ids;
            DistributedAdaptiveIncidenceSession fresh(
                *kernel, comm, {}, context, *fresh_budget, fresh_ids, fresh_budget);
            std::vector<AdaptiveLeaf> active;
            for (const auto& [cell, leaf] : current)
                active.push_back(leaf);
            RowMap fresh_rows;
            auto output = fresh.bootstrap(active, lookup);
            consume(output, fresh_rows, *fresh_budget);
            ASSERT_EQ(fresh_rows.size(), actual_rows.size());
            for (const auto& [cell, row] : fresh_rows)
            {
                EXPECT_EQ(row->revision, actual_rows.at(cell)->revision);
                EXPECT_EQ(keys(row->points), keys(actual_rows.at(cell)->points));
            }
        };
        compare();
        EXPECT_EQ(actual.statistics().source_membership_cells, 16);
        const AdaptiveCellKey near{config.region_id, CartesianCellAncestry{0, {0, 0, 0}}};
        const AdaptiveCellKey distant{config.region_id, CartesianCellAncestry{0, {3, 0, 0}}};
        const auto change = [&](const AdaptiveCellKey& retired, std::span<const AdaptiveCellKey> inserted)
        {
            const std::array old_a{IncidenceCellVersion{
                retired, actual_rows.at(retired)->revision, actual_rows.at(retired)->incidence_generation}};
            const std::array old_b{IncidenceCellVersion{
                retired, baseline_rows.at(retired)->revision, baseline_rows.at(retired)->incidence_generation}};
            ++context.round;
            auto changed_a = actual.apply_delta(old_a, inserted, context, lookup);
            auto changed_b = baseline.apply_delta(old_b, inserted, context, lookup);
            consume(changed_a, actual_rows, *actual_budget);
            consume(changed_b, baseline_rows, *baseline_budget);
            compare();
        };
        auto& explicit_a = current.at(near);
        ++explicit_a.revision;
        explicit_a.geometry = kernel->geometry(explicit_a); // Supplied geometry deliberately opts A out of Source.
        const std::array a_key{near};
        change(near, a_key);
        EXPECT_EQ(actual.statistics().source_membership_cells, 15);
        std::vector<AdaptiveCellKey> b_keys;
        if (refine_distant)
        {
            const auto split = kernel->split(current.at(distant), AdaptiveSplit::Isotropic);
            current.erase(distant);
            for (const auto& child : split.children)
            {
                current.emplace(child.key, child);
                b_keys.push_back(child.key);
            }
        }
        else
        {
            ++current.at(distant).revision;
            b_keys.push_back(distant);
        }
        change(distant, b_keys);
        EXPECT_EQ(current.size(), refine_distant ? 23u : 16u);
        EXPECT_EQ(actual.statistics().source_membership_cells, current.size() - 1);
        auto& restored_a = current.at(near);
        ++restored_a.revision;
        restored_a.geometry.reset();
        change(near, a_key);
        const auto& statistics = actual.statistics();
        EXPECT_EQ(statistics.source_membership_cells, current.size());
        EXPECT_EQ(statistics.explicit_coordinate_points, 0);
        EXPECT_EQ(statistics.explicit_tile_ring_slots, 0);
        EXPECT_EQ(statistics.explicit_line_records, 0);
        EXPECT_EQ(statistics.stride_line_records, 0);
        EXPECT_EQ(statistics.retained_explicit_line_records, 0);
        EXPECT_EQ(statistics.retained_stride_line_records, 0);
        for (const auto& [cell, row] : actual_rows)
        {
            const auto view = row->points.source_page_view();
            ASSERT_TRUE(view);
            EXPECT_EQ(view->page_count(), 0); // Current B rows must release physical suffix pages too.
        }
    }
}
} // namespace
