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
} // namespace
