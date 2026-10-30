/** @file testGeometryProvenanceBorrow.cc
 * @brief Independent finite provenance identity, lifetime and pair-budget gates.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace PLIC = SimpleFluid::PLIC;
namespace detail = PLIC::details;
using SimpleFluid::ArithmeticMode;
using Access = PLIC::DynamicGeometryAccess;
using Identity = std::shared_ptr<const PLIC::DynamicPointProvenance>;
constexpr std::array finite_modes{ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification};

template<class Owner>
concept CanBorrowSource = requires(Owner&& owner) { Access::borrowed_original_source(std::forward<Owner>(owner)); };
static_assert(CanBorrowSource<Identity&>);
static_assert(CanBorrowSource<const Identity&>);
static_assert(!CanBorrowSource<Identity>);
static_assert(!CanBorrowSource<const Identity>);
static_assert(
    std::is_same_v<decltype(Access::borrowed_original_source(std::declval<const Identity&>())), const Identity&>);
static_assert(std::is_same_v<decltype(Access::original_source(std::declval<const Identity&>())), Identity>);

PLIC::GeometryLimits accounting_limits(std::size_t pairs)
{
    PLIC::GeometryLimits result;
    // These tests inspect provenance metadata, not interval arithmetic. Track
    // traversal pairs with Ordinary Work even for Bounded-labeled records, so
    // unsupported bounded numerical environments retain all metadata coverage.
    result.arithmetic_mode = ArithmeticMode::Ordinary;
    result.maximum_pair_tests = pairs;
    result.maximum_operations = 1024;
    result.byte_budget = 1024 * 1024;
    return result;
}

Identity native(
    ArithmeticMode mode, std::array<long double, 3> values = {1, 2, 4}, std::array<long double, 3> errors = {})
{
    PLIC::DynamicPoint point(values, errors, mode);
    Access::seal_represented_source(point);
    return Access::identity(point);
}

Identity exported(const Identity& root, std::size_t edges)
{
    Identity owner = root;
    for (std::size_t i = 0; i < edges; ++i)
    {
        PLIC::DynamicPoint point(root->value, root->absolute_error, root->arithmetic_mode);
        Access::seal_export(point, owner, {});
        owner = Access::identity(point);
    }
    return owner;
}

// Deliberately construct malformed interior metadata rather than asking the
// sealing API to reject/propagate it before the traversal under test sees it.
Identity link(const Identity& source, ArithmeticMode mode, bool numerical_only = false)
{
    auto record = std::make_shared<PLIC::DynamicPointProvenance>();
    record->value = source->value;
    record->absolute_error = source->absolute_error;
    record->arithmetic_mode = mode;
    record->numerical_only = numerical_only;
    record->source_origin = source;
    return record;
}

std::vector<long> owner_counts(const Identity& head)
{
    std::vector<long> result;
    for (const auto* owner = &head; *owner; owner = &(*owner)->source_origin)
        result.push_back(owner->use_count());
    return result;
}

// Frozen pre-change ownership traversal, used only for a matched metadata trace.
// The independent assertions below also require exactly one charge per accepted
// source_origin edge and the known cap+1 rejection boundary.
Identity legacy_owning_source(Identity point)
{
    while (point && point->source_origin)
    {
        if (point->numerical_only || point->source_origin->arithmetic_mode != point->arithmetic_mode)
            return {};
        if (detail::active_work)
            detail::active_work->pair();
        point = point->source_origin;
    }
    return point && !point->numerical_only ? point : nullptr;
}

class GeometryProvenanceBorrow : public ::testing::Test
{
    detail::Work* saved_work_ = detail::active_work;
    PLIC::GeometryStatistics saved_statistics_ = detail::statistics;
    std::fenv_t saved_environment_{};

protected:
    void SetUp() override
    {
        ASSERT_EQ(std::fegetenv(&saved_environment_), 0);
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
        detail::active_work = nullptr;
        PLIC::reset_geometry_statistics();
    }
    void TearDown() override
    {
        detail::active_work = saved_work_;
        detail::statistics = saved_statistics_;
        (void) std::fesetenv(&saved_environment_);
    }
};

TEST_F(GeometryProvenanceBorrow, BorrowedRootPinsNoAdditionalOwnerAlongDeepExports)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto root = native(mode);
        const auto head = exported(root, 1024);
        const auto before = owner_counts(head);
        const auto root_owners = root.use_count();
        detail::Operation operation(accounting_limits(1024));
        const auto& borrowed = Access::borrowed_original_source(head);
        ASSERT_TRUE(borrowed);
        EXPECT_EQ(borrowed.get(), root.get());
        EXPECT_EQ(root.use_count(), root_owners);
        EXPECT_EQ(owner_counts(head), before);
        EXPECT_EQ(operation.work.pairs, 1024u);
        EXPECT_EQ(operation.work.operations, 0u);
        EXPECT_EQ(operation.work.bytes, 0u);
        EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 0u);
        // The reference is an existing source_origin slot, not a copied owner.
        const auto* slot = &head;
        while ((*slot)->source_origin)
            slot = &(*slot)->source_origin;
        EXPECT_EQ(&borrowed, slot);
    }
}

TEST_F(GeometryProvenanceBorrow, OwningApiRetainsRootAfterExportChainDies)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        Identity retained;
        std::weak_ptr<const PLIC::DynamicPointProvenance> root_observer, head_observer;
        {
            const auto root = native(mode);
            const auto head = exported(root, 11);
            root_observer = root;
            head_observer = head;
            {
                detail::Operation operation(accounting_limits(11));
                retained = Access::original_source(head);
                ASSERT_EQ(retained.get(), root.get());
                EXPECT_EQ(operation.work.pairs, 11u);
            }
            {
                // Owning queries may safely accept a temporary weak lock:
                // the result copies its root before that input owner expires.
                detail::Operation operation(accounting_limits(11));
                const auto temporary_input_result = Access::original_source(head_observer.lock());
                EXPECT_EQ(temporary_input_result.get(), root.get());
                EXPECT_EQ(operation.work.pairs, 11u);
            }
        }
        EXPECT_TRUE(head_observer.expired());
        ASSERT_FALSE(root_observer.expired());
        EXPECT_EQ(retained.use_count(), 1);
        EXPECT_EQ(retained->value, (std::array<long double, 3>{1, 2, 4}));
        retained.reset();
        EXPECT_TRUE(root_observer.expired());
    }
}

TEST_F(GeometryProvenanceBorrow, InteriorNumericalAndModeRejectionsPreserveEarlierEdgeCharges)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto other_mode =
            mode == ArithmeticMode::Ordinary ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
        const auto root = native(mode);
        const std::array malformed{link(link(root, mode, true), mode), link(link(link(root, other_mode), mode), mode)};
        for (const auto& head : malformed)
        {
            const auto before = owner_counts(head);
            detail::Operation operation(accounting_limits(8));
            EXPECT_FALSE(Access::borrowed_original_source(head));
            EXPECT_EQ(operation.work.pairs, 1u);
            EXPECT_EQ(operation.work.operations, 0u);
            EXPECT_EQ(owner_counts(head), before);
            EXPECT_FALSE(Access::same_represented_source(head, root));
            EXPECT_EQ(operation.work.pairs, 2u);
            EXPECT_EQ(PLIC::geometry_statistics().rejections, 0u);
        }
    }
}

TEST_F(GeometryProvenanceBorrow, NativeCoordinateIdentityDoesNotPromoteGeneratedOrUncertainPoints)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto a = native(mode), b = native(mode);
        ASSERT_NE(a.get(), b.get());
        EXPECT_TRUE(Access::same_represented_source(a, b));
        const auto first_chain = exported(a, 3), second_chain = exported(b, 4);
        {
            detail::Operation operation(accounting_limits(7));
            EXPECT_TRUE(Access::same_represented_source(first_chain, second_chain));
            EXPECT_EQ(operation.work.pairs, 7u);
        }
        PLIC::DynamicPoint generated_a({1, 2, 4}, {}, mode), generated_b({1, 2, 4}, {}, mode);
        Access::seal_generated(generated_a, {});
        Access::seal_generated(generated_b, {});
        const auto ga = Access::identity(generated_a), gb = Access::identity(generated_b);
        EXPECT_FALSE(Access::same_represented_source(ga, gb));
        EXPECT_FALSE(Access::same_represented_source(a, ga));
        EXPECT_TRUE(Access::same_represented_source(ga, ga));
        const auto uncertain_a = native(mode, {1, 2, 4}, {.125L, 0, 0});
        const auto uncertain_b = native(mode, {1, 2, 4}, {.125L, 0, 0});
        EXPECT_FALSE(Access::same_represented_source(uncertain_a, uncertain_b));
        EXPECT_FALSE(Access::same_represented_source(a, uncertain_a));
        EXPECT_TRUE(Access::same_represented_source(uncertain_a, uncertain_a));
        const auto displaced = native(mode, {1, 2, 5});
        EXPECT_FALSE(Access::same_represented_source(a, displaced));
        PLIC::DynamicPoint numerical({1, 2, 4}, {}, mode);
        Access::seal_numerical(numerical);
        const auto numeric_identity = Access::identity(numerical);
        EXPECT_FALSE(Access::same_represented_source(numeric_identity, numeric_identity));
        EXPECT_FALSE(Access::borrowed_original_source(numeric_identity));
    }
}

TEST_F(GeometryProvenanceBorrow, DifferentHeadModesShortCircuitBeforeEitherChainIsWalked)
{
    const auto a = exported(native(ArithmeticMode::Ordinary), 3);
    const auto b = exported(native(ArithmeticMode::BoundedCertification), 5);
    detail::Operation operation(accounting_limits(1));
    EXPECT_FALSE(Access::same_represented_source(a, b));
    EXPECT_EQ(operation.work.pairs, 0u);
    EXPECT_EQ(operation.work.operations, 0u);
}

TEST_F(GeometryProvenanceBorrow, OnePairPerEdgeAndCapPlusOneFailurePreserveOwners)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto root = native(mode);
        const auto head = exported(root, 17);
        const auto before = owner_counts(head);
        {
            detail::Operation operation(accounting_limits(17));
            EXPECT_EQ(Access::borrowed_original_source(head).get(), root.get());
            EXPECT_EQ(operation.work.pairs, 17u);
        }
        PLIC::reset_geometry_statistics();
        {
            detail::Operation operation(accounting_limits(16));
            try
            {
                (void) Access::borrowed_original_source(head);
                ADD_FAILURE() << "The seventeenth edge must exhaust the sixteen-pair cap.";
            }
            catch (const PLIC::GeometryError& error)
            {
                EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
                EXPECT_FALSE(error.recoverable_trial());
            }
            EXPECT_EQ(operation.work.pairs, 17u);
            EXPECT_EQ(operation.work.operations, 0u);
            EXPECT_EQ(PLIC::geometry_statistics().rejections, 1u);
        }
        EXPECT_EQ(detail::active_work, nullptr);
        EXPECT_EQ(owner_counts(head), before);
    }
}

TEST_F(GeometryProvenanceBorrow, BorrowedOwningAndLegacyWalksHaveIdenticalIndependentPairTrace)
{
    for (const auto mode : finite_modes)
        for (const std::size_t edges : {0u, 1u, 17u, 1024u})
        {
            SCOPED_TRACE(static_cast<int>(mode));
            SCOPED_TRACE(edges);
            const auto root = native(mode);
            const auto head = exported(root, edges);
            std::array<std::size_t, 3> pair_trace{};
            for (unsigned path = 0; path < pair_trace.size(); ++path)
            {
                PLIC::reset_geometry_statistics();
                detail::Operation operation(accounting_limits(std::max<std::size_t>(edges, 1)));
                if (path == 0)
                {
                    const auto owning = legacy_owning_source(head);
                    EXPECT_EQ(owning.get(), root.get());
                }
                else if (path == 1)
                {
                    const auto owning = Access::original_source(head);
                    EXPECT_EQ(owning.get(), root.get());
                }
                else
                {
                    const auto& borrowed = Access::borrowed_original_source(head);
                    EXPECT_EQ(borrowed.get(), root.get());
                }
                pair_trace[path] = operation.work.pairs;
                EXPECT_EQ(pair_trace[path], edges);
                EXPECT_EQ(operation.work.operations, 0u);
                EXPECT_EQ(operation.work.bytes, 0u);
                EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 0u);
                EXPECT_EQ(PLIC::geometry_statistics().rejections, 0u);
            }
            EXPECT_EQ(pair_trace, (std::array<std::size_t, 3>{edges, edges, edges}));
            if (mode == ArithmeticMode::Ordinary && edges == 17)
            {
                RecordProperty("legacy_pairs", std::to_string(pair_trace[0]));
                RecordProperty("owning_pairs", std::to_string(pair_trace[1]));
                RecordProperty("borrowed_pairs", std::to_string(pair_trace[2]));
            }
        }
}

TEST_F(GeometryProvenanceBorrow, NamedWeakLockPinsAncestryAndExpiredWeakRemainsEmpty)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        std::weak_ptr<const PLIC::DynamicPointProvenance> weak_head, weak_root;
        {
            auto root = native(mode);
            auto head = exported(root, 9);
            weak_head = head;
            weak_root = root;
            const auto owner = weak_head.lock();
            head.reset();
            root.reset();
            ASSERT_EQ(owner.use_count(), 1);
            detail::Operation operation(accounting_limits(9));
            const auto& borrowed = Access::borrowed_original_source(owner);
            ASSERT_TRUE(borrowed);
            EXPECT_EQ(borrowed.use_count(), 1);
            EXPECT_EQ(owner.use_count(), 1);
            EXPECT_FALSE(weak_root.expired());
            EXPECT_EQ(operation.work.pairs, 9u);
        }
        EXPECT_TRUE(weak_head.expired());
        EXPECT_TRUE(weak_root.expired());
        const auto expired_owner = weak_head.lock();
        EXPECT_FALSE(expired_owner);
        EXPECT_FALSE(Access::borrowed_original_source(expired_owner));
        EXPECT_FALSE(Access::original_source(expired_owner));
    }
}

TEST_F(GeometryProvenanceBorrow, ScopeGuardsReleaseLockedOwnersOnSuccessAndBudgetException)
{
    for (const auto mode : finite_modes)
        for (const bool fail : {false, true})
        {
            SCOPED_TRACE(static_cast<int>(mode));
            SCOPED_TRACE(fail);
            std::weak_ptr<const PLIC::DynamicPointProvenance> weak_head, weak_root;
            bool caught = false;
            try
            {
                detail::Operation operation(accounting_limits(fail ? 3 : 4));
                ASSERT_EQ(detail::active_work, &operation.work);
                auto root = native(mode);
                auto head = exported(root, 4);
                weak_root = root;
                weak_head = head;
                const auto owner = weak_head.lock();
                head.reset();
                root.reset();
                const auto& borrowed = Access::borrowed_original_source(owner);
                ASSERT_TRUE(borrowed);
                EXPECT_FALSE(fail);
                EXPECT_EQ(operation.work.pairs, 4u);
            }
            catch (const PLIC::GeometryError& error)
            {
                caught = true;
                EXPECT_TRUE(fail);
                EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
            }
            EXPECT_EQ(caught, fail);
            EXPECT_EQ(detail::active_work, nullptr);
            EXPECT_TRUE(weak_head.expired());
            EXPECT_TRUE(weak_root.expired());
        }
}
TEST_F(GeometryProvenanceBorrow, ExportSharesOnlyExactlyMatchingImmutableSupports)
{
    const PLIC::PlaneAuthority first{{2, 7}}, second{{9, 11}};
    const std::vector<PLIC::PlaneAuthority> expected{first, second};
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto endpoint = native(mode, {0, 0, 0});
        PLIC::DynamicPoint source({1, 2, 4}, {.125L, 0, 0}, mode);
        Access::seal_generated(source, {{endpoint, endpoint}}, expected);
        const auto root = Access::identity(source);
        PLIC::DynamicPoint copy(source.value, source.absolute_error, mode);
        Access::seal_export(copy, root, {second, first, second});
        const auto exported = Access::identity(copy);
        ASSERT_TRUE(root->support_payload);
        ASSERT_TRUE(exported);
        EXPECT_NE(exported, root);
        EXPECT_FALSE(Access::same_source(source, copy));
        EXPECT_EQ(exported->support_payload, root->support_payload);
        EXPECT_EQ(&source.supports(), &copy.supports());
        EXPECT_EQ(copy.supports(), expected);
        EXPECT_EQ(exported->source_origin, root);
        EXPECT_EQ(Access::original_source(exported), root);
        EXPECT_TRUE(exported->constructed);
        EXPECT_FALSE(exported->represented_native_source);
        EXPECT_FALSE(exported->numerical_only);
        EXPECT_TRUE(exported->source_edges.empty());
        ASSERT_EQ(root->source_edges.size(), 1u);
        EXPECT_EQ(root->source_edges[0][0], endpoint);
        EXPECT_EQ(exported->value, source.value);
        EXPECT_EQ(exported->absolute_error, source.absolute_error);
        EXPECT_EQ(exported->arithmetic_mode, mode);

        PLIC::DynamicPoint independent(source.value, source.absolute_error, mode);
        Access::seal(independent, expected);
        EXPECT_NE(Access::identity(independent)->support_payload, root->support_payload);
        auto different = second;
        ++different.key.back();
        PLIC::DynamicPoint changed(source.value, source.absolute_error, mode);
        Access::seal_export(changed, root, {first, different});
        EXPECT_NE(Access::identity(changed)->support_payload, root->support_payload);
        EXPECT_NE(changed.supports(), expected);

        for (std::size_t field = 0; field < 3; ++field)
        {
            auto mutated = copy;
            if (field == 0)
                mutated.value[0] += 1;
            else if (field == 1)
                mutated.absolute_error[0] += 1;
            else
                mutated.arithmetic_mode = mode == ArithmeticMode::Ordinary ? ArithmeticMode::BoundedCertification
                                                                          : ArithmeticMode::Ordinary;
            EXPECT_TRUE(mutated.supports().empty());
            EXPECT_FALSE(Access::identity(mutated));
        }
    }
}

TEST_F(GeometryProvenanceBorrow, SupportPayloadLivesThroughExportsAndCountsOnceAcrossAncestry)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        std::weak_ptr<const PLIC::DynamicPointProvenance> weak_source;
        std::shared_ptr<const PLIC::DynamicPointSupportPayload> retained_payload;
        {
            PLIC::DynamicPoint exported;
            {
                PLIC::DynamicPoint source({1, 2, 4}, {.125L, 0, 0}, mode);
                Access::seal(source, {{{2, 7, 13}}});
                const auto root = Access::identity(source);
                weak_source = root;
                exported = PLIC::DynamicPoint(source.value, source.absolute_error, mode);
                Access::seal_export(exported, root, source.supports());
                retained_payload = root->support_payload;
            }
            EXPECT_FALSE(weak_source.expired());
            ASSERT_EQ(exported.supports().size(), 1u);
            EXPECT_EQ(exported.supports()[0].key, (std::vector<std::uint64_t>{2, 7, 13}));
            auto expected = 2 * sizeof(PLIC::DynamicPointProvenance) + sizeof(PLIC::DynamicPointSupportPayload) +
                            retained_payload->authorities.capacity() * sizeof(PLIC::PlaneAuthority);
            for (const auto& authority : retained_payload->authorities)
                expected += authority.key.capacity() * sizeof(std::uint64_t);
            EXPECT_EQ(Access::identity_bytes(Access::identity(exported)), expected);
        }
        EXPECT_TRUE(weak_source.expired());
        EXPECT_EQ(retained_payload->authorities[0].key, (std::vector<std::uint64_t>{2, 7, 13}));
        const std::weak_ptr<const PLIC::DynamicPointSupportPayload> weak_payload = retained_payload;
        retained_payload.reset();
        EXPECT_TRUE(weak_payload.expired());
    }
}

TEST_F(GeometryProvenanceBorrow, SharedSupportsDoNotTurnIndependentUncertainPointsIntoIdenticalPoints)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "BoundedCertification requires a supported floating-point environment.";
    constexpr auto mode = ArithmeticMode::BoundedCertification;
    PLIC::DynamicPoint first({0, 0, 0}, {.125L, .125L, .125L}, mode);
    Access::seal(first, {{{2, 7}}});
    PLIC::DynamicPoint second(first.value, first.absolute_error, mode);
    Access::seal_export(second, Access::identity(first), first.supports());
    ASSERT_EQ(Access::identity(first)->support_payload, Access::identity(second)->support_payload);
    EXPECT_FALSE(Access::same_source(first, second));
    EXPECT_FALSE(PLIC::points_identical(PLIC::Point<mode>(first), PLIC::Point<mode>(second)));
    PLIC::DynamicPoint independent(first.value, first.absolute_error, mode);
    Access::seal(independent, first.supports());
    EXPECT_FALSE(Access::same_source(first, independent));
    EXPECT_FALSE(PLIC::points_identical(PLIC::Point<mode>(first), PLIC::Point<mode>(independent)));
    for (const auto& candidate : {second, independent})
    {
        const std::array<PLIC::Point<mode>, 4> tetrahedron{PLIC::Point<mode>(first), PLIC::Point<mode>(candidate),
            PLIC::Point<mode>({1, 0, 0}), PLIC::Point<mode>({0, 1, 0})};
        const auto determinant = PLIC::orientation(tetrahedron);
        EXPECT_LT(determinant.lower(), 0);
        EXPECT_GT(determinant.upper(), 0);
    }
}
TEST_F(GeometryProvenanceBorrow, SupportPayloadReuseFitsAWorkCapThatRejectsANewOwner)
{
    for (const auto mode : finite_modes)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        PLIC::DynamicPoint source({1, 2, 4}, {}, mode);
        Access::seal(source, {{{2, 7}}});
        const auto root = Access::identity(source);
        auto limits = accounting_limits(1024);
        limits.byte_budget = sizeof(PLIC::DynamicPointSupportPayload) - 1;
        detail::Operation operation(limits);
        PLIC::DynamicPoint reused(source.value, source.absolute_error, mode);
        ASSERT_NO_THROW(Access::seal_export(reused, root, source.supports()));
        ASSERT_TRUE(Access::identity(reused));
        EXPECT_EQ(Access::identity(reused)->support_payload, root->support_payload);
        PLIC::DynamicPoint empty(source.value, source.absolute_error, mode);
        EXPECT_NO_THROW(Access::seal(empty, {}));
        PLIC::DynamicPoint fresh(source.value, source.absolute_error, mode);
        try
        {
            Access::seal_export(fresh, root, {{{2, 8}}});
            FAIL() << "A distinct nonempty support payload must respect the construction byte cap.";
        }
        catch (const PLIC::GeometryError& error)
        {
            EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
        }
        EXPECT_FALSE(Access::identity(fresh));
    }
}
TEST_F(GeometryProvenanceBorrow, FreshSupportPayloadTrimsOnlySpareOuterCapacity)
{
    PLIC::PlaneAuthority first{{2, 7}}, second{{9, 11, 13}};
    first.key.reserve(17);
    second.key.reserve(23);
    const auto first_capacity = first.key.capacity(), second_capacity = second.key.capacity();
    std::vector<PLIC::PlaneAuthority> supports;
    supports.reserve(31);
    supports.push_back(std::move(second));
    supports.push_back(std::move(first));
    PLIC::DynamicPoint point({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    Access::seal(point, std::move(supports));
    const auto payload = Access::identity(point)->support_payload;
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->authorities.capacity(), 2u);
    EXPECT_EQ(payload->authorities[0].key.capacity(), first_capacity);
    EXPECT_EQ(payload->authorities[1].key.capacity(), second_capacity);
    EXPECT_EQ(payload->authorities[0].key, (std::vector<std::uint64_t>{2, 7}));
    EXPECT_EQ(payload->authorities[1].key, (std::vector<std::uint64_t>{9, 11, 13}));

    std::vector<PLIC::PlaneAuthority> oversized;
    oversized.reserve(8);
    oversized.push_back({{2, 7}});
    auto limits = accounting_limits(1024);
    limits.byte_budget = sizeof(PLIC::DynamicPointSupportPayload) + sizeof(PLIC::PlaneAuthority) - 1;
    detail::Operation operation(limits);
    PLIC::DynamicPoint rejected({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    EXPECT_THROW(Access::seal(rejected, std::move(oversized)), PLIC::GeometryError);
    EXPECT_FALSE(Access::identity(rejected));
}

TEST_F(GeometryProvenanceBorrow, VertexAuthoritiesFreezeOnlyExactSequencesAndCopiesDetachIndependently)
{
    const PLIC::PlaneAuthority first{{2, 7}}, second{{9, 11, 13}};
    PLIC::DynamicPoint point({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    Access::seal(point, {first, second});
    const auto root = Access::identity(point);
    detail::StoredVertexAuthorities holder(std::vector<PLIC::PlaneAuthority>{first, second});
    holder.share_if_equal(root->support_payload);
    ASSERT_EQ(holder.shared_payload(), root->support_payload);
    const auto sibling = holder;
    const auto source_values = point.supports();
    EXPECT_EQ(&holder.values(), &point.supports());
    static_assert(std::is_const_v<std::remove_reference_t<decltype(*holder.begin())>>);
    detail::StoredVertexAuthorities reordered(std::vector<PLIC::PlaneAuthority>{second, first});
    reordered.share_if_equal(root->support_payload);
    EXPECT_TRUE(reordered.owned_values());
    EXPECT_EQ(reordered.values(), (std::vector<PLIC::PlaneAuthority>{second, first}));
    detail::StoredVertexAuthorities different(std::vector<PLIC::PlaneAuthority>{first, {{9, 11, 14}}});
    different.share_if_equal(root->support_payload);
    EXPECT_TRUE(different.owned_values());

    holder.append({{17, 19}});
    ASSERT_TRUE(holder.owned_values());
    EXPECT_EQ(holder.values(), (std::vector<PLIC::PlaneAuthority>{first, second, {{17, 19}}}));
    EXPECT_EQ(sibling.shared_payload(), root->support_payload);
    EXPECT_EQ(sibling.values(), source_values);
    EXPECT_EQ(point.supports(), source_values);
    EXPECT_EQ(Access::identity(point), root);
    auto cleared = sibling;
    cleared.clear();
    EXPECT_TRUE(cleared.empty());
    EXPECT_TRUE(cleared.owned_values());
    EXPECT_EQ(sibling.values(), source_values);
}

TEST_F(GeometryProvenanceBorrow, VertexAuthorityDetachPrechargesAllBuffersAndFailsAtomically)
{
    PLIC::DynamicPoint point({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    Access::seal(point, {{{2, 7}}, {{9, 11, 13}}});
    const auto payload = Access::identity(point)->support_payload;
    detail::StoredVertexAuthorities holder(point.supports());
    holder.share_if_equal(payload);
    // Two existing keys contain five words; appending the first adds two more.
    const auto detach_bytes = 2 * sizeof(PLIC::PlaneAuthority) + 5 * sizeof(std::uint64_t);
    const auto append_bytes = 3 * sizeof(PLIC::PlaneAuthority) + 7 * sizeof(std::uint64_t);
    for (const bool append : {false, true})
    {
        auto draft = holder;
        auto limits = accounting_limits(1024);
        limits.byte_budget = (append ? append_bytes : detach_bytes) - 1;
        {
            detail::Operation operation(limits);
            if (append)
                EXPECT_THROW(draft.append(draft.values()[0]), PLIC::GeometryError);
            else
                EXPECT_THROW(draft.detach(), PLIC::GeometryError);
        }
        EXPECT_EQ(draft.shared_payload(), payload);
        EXPECT_EQ(draft.values(), point.supports());
        limits.byte_budget = append ? append_bytes : detach_bytes;
        {
            detail::Operation operation(limits);
            if (append)
                EXPECT_NO_THROW(draft.append(draft.values()[0]));
            else
                EXPECT_NO_THROW(draft.detach());
        }
        ASSERT_TRUE(draft.owned_values());
        EXPECT_EQ(draft.size(), append ? 3u : 2u);
        EXPECT_EQ(draft.values()[0].key, (std::vector<std::uint64_t>{2, 7}));
        if (append)
            EXPECT_EQ(draft.values()[2].key, draft.values()[0].key);
        EXPECT_EQ(holder.shared_payload(), payload);
    }
}

TEST_F(GeometryProvenanceBorrow, AliasedVertexAuthorityAppendPinsItsLastImmutableOwner)
{
    detail::StoredVertexAuthorities holder;
    std::weak_ptr<const PLIC::DynamicPointSupportPayload> weak_payload;
    {
        PLIC::DynamicPoint point({1, 2, 4}, {}, ArithmeticMode::Ordinary);
        Access::seal(point, {{{2, 7}}});
        const auto payload = Access::identity(point)->support_payload;
        weak_payload = payload;
        holder = point.supports();
        holder.share_if_equal(payload);
    }
    EXPECT_FALSE(weak_payload.expired());
    holder.append(holder.values()[0]);
    EXPECT_TRUE(weak_payload.expired());
    EXPECT_EQ(holder.values(), (std::vector<PLIC::PlaneAuthority>{{{2, 7}}, {{2, 7}}}));
}
TEST_F(GeometryProvenanceBorrow, MovedFromSharedVertexAuthoritiesRetainTheirAppendCap)
{
    PLIC::DynamicPoint point({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    Access::seal(point, {{{2, 7}}});
    detail::StoredVertexAuthorities holder(point.supports());
    holder.share_if_equal(Access::identity(point)->support_payload);
    const auto moved = std::move(holder);
    ASSERT_TRUE(holder.empty());
    EXPECT_FALSE(holder.shared_payload());
    ASSERT_FALSE(holder.owned_values());
    const PLIC::PlaneAuthority authority{{11, 13}};
    const auto required = sizeof(PLIC::PlaneAuthority) + 2 * sizeof(std::uint64_t);
    for (const auto cap : {std::size_t(1), required - 1})
    {
        auto limits = accounting_limits(1024);
        limits.byte_budget = cap;
        detail::Operation operation(limits);
        EXPECT_THROW(holder.append(authority), PLIC::GeometryError);
        EXPECT_TRUE(holder.empty());
        EXPECT_FALSE(holder.owned_values());
        EXPECT_FALSE(holder.shared_payload());
    }
    auto limits = accounting_limits(1024);
    limits.byte_budget = required;
    {
        detail::Operation operation(limits);
        EXPECT_NO_THROW(holder.append(authority));
    }
    EXPECT_EQ(holder.values(), (std::vector<PLIC::PlaneAuthority>{authority}));
    EXPECT_EQ(moved.values(), point.supports());
    EXPECT_EQ(moved.shared_payload(), Access::identity(point)->support_payload);
}
} // namespace
