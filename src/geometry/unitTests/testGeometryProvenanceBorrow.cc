/** Independent finite snapshot identity, lifetime and admission gates. */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace PLIC = SimpleFluid::PLIC;
namespace detail = PLIC::details;
using SimpleFluid::ArithmeticMode;
using Access = PLIC::DynamicGeometryAccess;
using Identity = PLIC::PointProvenanceRef;
using View = PLIC::PointProvenanceView;
using Observer = std::weak_ptr<const PLIC::PointSnapshotContextBase>;
constexpr std::array finite_modes{ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification};
template<class Owner>
concept CanBorrowSource = requires(Owner&& owner) { Access::borrowed_original_source(std::forward<Owner>(owner)); };
static_assert(CanBorrowSource<Identity&> && CanBorrowSource<const Identity&> && CanBorrowSource<View>);
static_assert(!CanBorrowSource<Identity> && !CanBorrowSource<const Identity>);
static_assert(std::is_same_v<decltype(Access::borrowed_original_source(std::declval<const Identity&>())), View>);
static_assert(std::is_same_v<decltype(Access::original_source(std::declval<const Identity&>())), Identity>);
static_assert(sizeof(PLIC::LocalAuthorityRange<ArithmeticMode::Ordinary>) == 2 * sizeof(std::uint32_t));

PLIC::GeometryLimits accounting_limits(std::size_t pairs)
{
    PLIC::GeometryLimits limits;
    // Metadata coverage does not require Bounded numerical arithmetic.
    limits.arithmetic_mode = ArithmeticMode::Ordinary;
    limits.maximum_pair_tests = pairs;
    limits.maximum_operations = 1024;
    limits.byte_budget = 1024 * 1024;
    return limits;
}
Identity native(
    ArithmeticMode mode, std::array<long double, 3> value = {1, 2, 4}, std::array<long double, 3> error = {})
{
    PLIC::DynamicPoint point(value, error, mode);
    Access::seal_represented_source(point);
    return Access::identity(point);
}
Identity exported(const Identity& root, std::size_t edges)
{
    Identity owner = root;
    const auto facts = root.view();
    for (std::size_t i = 0; i < edges; ++i)
    {
        PLIC::DynamicPoint point(facts.value(), facts.absolute_error(), facts.arithmetic_mode());
        Access::seal_export(point, owner, {});
        owner = Access::identity(point);
    }
    return owner;
}
std::vector<long> owner_counts(const Identity& head)
{
    std::vector<long> counts;
    // Never construct an owning handle to observe a borrowed walk.
    for (auto point = head.view(); point; point = point.source_origin())
        counts.push_back(point.context()->weak_from_this().use_count());
    return counts;
}
Identity legacy_owning_source(Identity point)
{
    while (point && point.view().source_origin())
    {
        const auto source = point.view().source_origin();
        if (point.view().numerical_only() || source.arithmetic_mode() != point.view().arithmetic_mode())
            return {};
        if (detail::active_work)
            detail::active_work->pair();
        point = source.owning();
    }
    return point && !point.view().numerical_only() ? point : Identity{};
}
std::vector<PLIC::PlaneAuthority> keys(PLIC::PointAuthorityRange range)
{
    return {range.begin(), range.end()};
}
template<class Function> void typed_modes(Function&& function)
{
    function.template operator()<ArithmeticMode::Ordinary>();
    function.template operator()<ArithmeticMode::BoundedCertification>();
}
class GeometryProvenanceBorrow : public ::testing::Test
{
    detail::Work* saved_work_ = detail::active_work;
    PLIC::GeometryStatistics statistics_ = detail::statistics;
    std::fenv_t environment_{};

protected:
    void SetUp() override
    {
        ASSERT_EQ(std::fegetenv(&environment_), 0);
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
        detail::active_work = nullptr;
        PLIC::reset_geometry_statistics();
    }
    void TearDown() override
    {
        detail::active_work = saved_work_;
        detail::statistics = statistics_;
        (void) std::fesetenv(&environment_);
    }
};
TEST_F(GeometryProvenanceBorrow, BorrowedRootPinsNoAdditionalOwnerAlongDeepExports)
{
    for (const auto mode : finite_modes)
    {
        const auto root = native(mode), head = exported(root, 1024);
        const auto before = owner_counts(head);
        detail::Operation operation(accounting_limits(1024));
        const auto borrowed = Access::borrowed_original_source(head);
        EXPECT_EQ(borrowed, root.view());
        EXPECT_EQ(owner_counts(head), before);
        EXPECT_EQ(operation.work.pairs, 1024u);
        EXPECT_EQ(operation.work.operations, 0u);
        EXPECT_EQ(operation.work.bytes, 0u);
        EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 0u);
        EXPECT_EQ(borrowed.context_identity(), root.context_identity());
        EXPECT_EQ(borrowed.local_ref(), root.local_ref());
    }
}
TEST_F(GeometryProvenanceBorrow, BatchedAliasRowsKeepTheSamePairBudgetWithoutPerRowOwnerPins)
{
    typed_modes(
        [&]<ArithmeticMode Mode>()
        {
            for (const std::size_t edges : {17u, 1024u})
            {
                const auto root = native(Mode);
                const auto head = [&]
                {
                    PLIC::PointSnapshotBuilder<Mode> builder;
                    auto current = root.view();
                    for (std::size_t i = 0; i < edges; ++i)
                        current = builder.export_alias(current.value(), current.absolute_error(), current, {});
                    const auto ref = builder.import(current);
                    const auto context = builder.freeze();
                    EXPECT_EQ(context->export_aliases.size(), edges);
                    EXPECT_EQ(context->dependencies.size(), 1u);
                    return PLIC::bind_point(context, ref);
                }();
                const auto before = owner_counts(head);
                for (const unsigned path : {0u, 1u, 2u})
                {
                    detail::Operation operation(accounting_limits(edges));
                    if (path == 0)
                        EXPECT_EQ(legacy_owning_source(head), root);
                    else if (path == 1)
                        EXPECT_EQ(Access::original_source(head), root);
                    else
                        EXPECT_EQ(Access::borrowed_original_source(head), root.view());
                    EXPECT_EQ(operation.work.pairs, edges);
                    EXPECT_EQ(operation.work.operations, 0u);
                    EXPECT_EQ(operation.work.bytes, 0u);
                }
                {
                    detail::Operation operation(accounting_limits(edges - 1));
                    EXPECT_THROW((void) Access::borrowed_original_source(head), PLIC::GeometryError);
                    EXPECT_EQ(operation.work.pairs, edges);
                }
                EXPECT_EQ(owner_counts(head), before);
            }
        });
}
TEST_F(GeometryProvenanceBorrow, OwningApiRetainsRootAfterExportChainDies)
{
    for (const auto mode : finite_modes)
    {
        Identity retained;
        Observer root_observer, head_observer;
        {
            const auto root = native(mode), head = exported(root, 11);
            root_observer = root.context_pin();
            head_observer = head.context_pin();
            const auto slot = head.local_ref();
            detail::Operation operation(accounting_limits(22));
            retained = Access::original_source(head);
            EXPECT_EQ(retained, root);
            const auto from_lock = Access::original_source(PLIC::bind_point(head_observer.lock(), slot));
            EXPECT_EQ(from_lock, root);
            EXPECT_EQ(operation.work.pairs, 22u);
        }
        EXPECT_TRUE(head_observer.expired());
        EXPECT_FALSE(root_observer.expired());
        EXPECT_EQ(retained.context_pin().use_count(), 1);
        EXPECT_EQ(retained.view().value(), (std::array<long double, 3>{1, 2, 4}));
        retained.reset();
        EXPECT_TRUE(root_observer.expired());
    }
}
TEST_F(GeometryProvenanceBorrow, InvalidInteriorTransitionsAreRejectedBeforeSnapshotPublication)
{
    typed_modes(
        [&]<ArithmeticMode Mode>()
        {
            constexpr auto other =
                Mode == ArithmeticMode::Ordinary ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
            const auto root = native(Mode), head = exported(root, 1);
            const auto before = owner_counts(head);
            detail::Operation operation(accounting_limits(8));
            EXPECT_EQ(Access::borrowed_original_source(head), root.view());
            PLIC::PointSnapshotBuilder<other> foreign;
            EXPECT_THROW((void) foreign.import(root), PLIC::GeometryError);
            EXPECT_THROW((void) foreign.freeze(), PLIC::GeometryError);
            EXPECT_TRUE(foreign.context()->dependencies.empty());
            // A rejected transition preserves all already accepted edge charges.
            EXPECT_EQ(operation.work.pairs, 1u);
            EXPECT_EQ(owner_counts(head), before);
            PLIC::PointSnapshotBuilder<Mode> numerical;
            const auto point = numerical.numerical({1, 2, 4}, {});
            const auto numerical_context = numerical.freeze();
            PLIC::PointSnapshotBuilder<Mode> propagation;
            const auto copied = propagation.export_alias(point.value(), {}, point, {});
            const auto copied_context = propagation.freeze();
            EXPECT_TRUE(copied.numerical_only());
            EXPECT_FALSE(copied.source_origin());
            EXPECT_TRUE(copied.supports().empty());
            EXPECT_FALSE(Access::borrowed_original_source(copied));
            EXPECT_FALSE(Access::same_represented_source(copied, root.view()));
            EXPECT_EQ(operation.work.pairs, 1u);
            PLIC::PointSnapshotBuilder<Mode> promote;
            EXPECT_THROW((void) promote.native_source(point.value(), {}, {}, point), PLIC::GeometryError);
            EXPECT_THROW((void) promote.freeze(), PLIC::GeometryError);
        });
}
TEST_F(GeometryProvenanceBorrow, NativeCoordinateIdentityDoesNotPromoteGeneratedOrUncertainPoints)
{
    for (const auto mode : finite_modes)
    {
        const auto a = native(mode), b = native(mode);
        EXPECT_NE(a, b);
        EXPECT_TRUE(Access::same_represented_source(a, b));
        const auto first = exported(a, 3), second = exported(b, 4);
        {
            detail::Operation op(accounting_limits(7));
            EXPECT_TRUE(Access::same_represented_source(first, second));
            EXPECT_EQ(op.work.pairs, 7u);
        }
        PLIC::DynamicPoint ga({1, 2, 4}, {}, mode), gb({1, 2, 4}, {}, mode);
        Access::seal_generated(ga, {});
        Access::seal_generated(gb, {});
        EXPECT_FALSE(Access::same_represented_source(Access::identity(ga), Access::identity(gb)));
        EXPECT_FALSE(Access::same_represented_source(a, Access::identity(ga)));
        EXPECT_TRUE(Access::same_represented_source(Access::identity(ga), Access::identity(ga)));
        const auto uncertain_a = native(mode, {1, 2, 4}, {.125L, 0, 0});
        const auto uncertain_b = native(mode, {1, 2, 4}, {.125L, 0, 0});
        EXPECT_FALSE(Access::same_represented_source(uncertain_a, uncertain_b));
        EXPECT_FALSE(Access::same_represented_source(a, uncertain_a));
        EXPECT_TRUE(Access::same_represented_source(uncertain_a, uncertain_a));
        EXPECT_FALSE(Access::same_represented_source(a, native(mode, {1, 2, 5})));
        PLIC::DynamicPoint numeric({1, 2, 4}, {}, mode);
        Access::seal_numerical(numeric);
        const auto n = Access::identity(numeric);
        EXPECT_FALSE(Access::same_represented_source(n, n));
        EXPECT_FALSE(Access::borrowed_original_source(n));
    }
}
TEST_F(GeometryProvenanceBorrow, DifferentHeadModesShortCircuitBeforeEitherChainIsWalked)
{
    const auto a = exported(native(ArithmeticMode::Ordinary), 3);
    const auto b = exported(native(ArithmeticMode::BoundedCertification), 5);
    detail::Operation op(accounting_limits(1));
    EXPECT_FALSE(Access::same_represented_source(a, b));
    EXPECT_EQ(op.work.pairs, 0u);
    EXPECT_EQ(op.work.operations, 0u);
}
TEST_F(GeometryProvenanceBorrow, OnePairPerEdgeAndCapPlusOneFailurePreserveOwners)
{
    for (const auto mode : finite_modes)
    {
        const auto root = native(mode), head = exported(root, 17);
        const auto before = owner_counts(head);
        {
            detail::Operation op(accounting_limits(17));
            EXPECT_EQ(Access::borrowed_original_source(head), root.view());
            EXPECT_EQ(op.work.pairs, 17u);
        }
        PLIC::reset_geometry_statistics();
        {
            detail::Operation op(accounting_limits(16));
            try
            {
                (void) Access::borrowed_original_source(head);
                FAIL() << "Edge 17 must exceed a 16-pair cap.";
            }
            catch (const PLIC::GeometryError& e)
            {
                EXPECT_EQ(e.code(), PLIC::GeometryErrorCode::ResourceLimit);
                EXPECT_FALSE(e.recoverable_trial());
            }
            EXPECT_EQ(op.work.pairs, 17u);
            EXPECT_EQ(op.work.operations, 0u);
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
            const auto root = native(mode), head = exported(root, edges);
            std::array<std::size_t, 3> trace{};
            for (unsigned path = 0; path < 3; ++path)
            {
                PLIC::reset_geometry_statistics();
                detail::Operation op(accounting_limits(std::max<std::size_t>(edges, 1)));
                if (path == 0)
                    EXPECT_EQ(legacy_owning_source(head), root);
                else if (path == 1)
                    EXPECT_EQ(Access::original_source(head), root);
                else
                    EXPECT_EQ(Access::borrowed_original_source(head), root.view());
                trace[path] = op.work.pairs;
                EXPECT_EQ(op.work.operations, 0u);
                EXPECT_EQ(op.work.bytes, 0u);
                EXPECT_EQ(PLIC::geometry_statistics().floating_operations, 0u);
                EXPECT_EQ(PLIC::geometry_statistics().rejections, 0u);
            }
            EXPECT_EQ(trace, (std::array<std::size_t, 3>{edges, edges, edges}));
        }
}
TEST_F(GeometryProvenanceBorrow, NamedWeakLockPinsAncestryAndExpiredWeakRemainsEmpty)
{
    for (const auto mode : finite_modes)
    {
        Observer weak_head, weak_root;
        PLIC::DynamicStoredPointRef slot;
        {
            auto root = native(mode), head = exported(root, 9);
            weak_head = head.context_pin();
            weak_root = root.context_pin();
            slot = head.local_ref();
            const auto owner = PLIC::bind_point(weak_head.lock(), slot);
            head.reset();
            root.reset();
            EXPECT_EQ(owner.context_pin().use_count(), 1);
            detail::Operation op(accounting_limits(9));
            const auto borrowed = Access::borrowed_original_source(owner);
            EXPECT_TRUE(borrowed);
            EXPECT_EQ(owner.context_pin().use_count(), 1);
            EXPECT_EQ(borrowed.context()->weak_from_this().use_count(), 1);
            EXPECT_FALSE(weak_root.expired());
            EXPECT_EQ(op.work.pairs, 9u);
        }
        EXPECT_TRUE(weak_head.expired());
        EXPECT_TRUE(weak_root.expired());
        const auto empty = PLIC::bind_point(weak_head.lock(), slot);
        EXPECT_FALSE(empty);
        EXPECT_FALSE(Access::borrowed_original_source(empty));
        EXPECT_FALSE(Access::original_source(empty));
    }
}
TEST_F(GeometryProvenanceBorrow, ScopeGuardsReleaseLockedOwnersOnSuccessAndBudgetException)
{
    for (const auto mode : finite_modes)
        for (bool fail : {false, true})
        {
            Observer wh, wr;
            bool caught = false;
            try
            {
                auto root = native(mode), head = exported(root, 4);
                wr = root.context_pin();
                wh = head.context_pin();
                const auto owner = PLIC::bind_point(wh.lock(), head.local_ref());
                head.reset();
                root.reset();
                detail::Operation op(accounting_limits(fail ? 3 : 4));
                EXPECT_EQ(detail::active_work, &op.work);
                const auto borrowed = Access::borrowed_original_source(owner);
                EXPECT_TRUE(borrowed);
                EXPECT_FALSE(fail);
                EXPECT_EQ(op.work.pairs, 4u);
            }
            catch (const PLIC::GeometryError& e)
            {
                caught = true;
                EXPECT_TRUE(fail);
                EXPECT_EQ(e.code(), PLIC::GeometryErrorCode::ResourceLimit);
            }
            EXPECT_EQ(caught, fail);
            EXPECT_EQ(detail::active_work, nullptr);
            EXPECT_TRUE(wh.expired());
            EXPECT_TRUE(wr.expired());
        }
}
TEST_F(GeometryProvenanceBorrow, ExportSharesOnlyExactlyMatchingImmutableSupports)
{
    const auto first = PLIC::PlaneAuthority::opaque({2, 7}), second = PLIC::PlaneAuthority::opaque({9, 11});
    const std::vector<PLIC::PlaneAuthority> expected{first, second};
    for (auto mode : finite_modes)
    {
        const auto endpoint = native(mode, {0, 0, 0});
        PLIC::DynamicPoint source({1, 2, 4}, {.125L, 0, 0}, mode);
        Access::seal_generated(source, {{endpoint.view(), endpoint.view()}}, expected);
        const auto root = Access::identity(source);
        PLIC::DynamicPoint copy(source.value, source.absolute_error, mode);
        Access::seal_export(copy, root, {second, first, second});
        const auto exported = Access::identity(copy);
        EXPECT_NE(exported, root);
        EXPECT_FALSE(Access::same_source(source, copy));
        EXPECT_EQ(copy.supports(), expected);
        EXPECT_EQ(exported.view().source_origin(), root.view());
        EXPECT_EQ(Access::original_source(exported), root);
        EXPECT_TRUE(exported.view().constructed());
        EXPECT_FALSE(exported.view().represented_native_source());
        EXPECT_FALSE(exported.view().numerical_only());
        EXPECT_TRUE(exported.view().source_edges().empty());
        ASSERT_EQ(root.view().source_edges().size(), 1u);
        EXPECT_EQ(root.view().source_edges()[0][0], endpoint.view());
        for (std::size_t i = 0; i < 2; ++i)
            EXPECT_EQ(copy.supports()[i].opaque_storage_owner(), source.supports()[i].opaque_storage_owner());
        PLIC::DynamicPoint independent(source.value, source.absolute_error, mode);
        Access::seal(independent, expected);
        EXPECT_FALSE(Access::same_source(source, independent));
        PLIC::DynamicPoint unsealed(copy.value, copy.absolute_error, mode);
        EXPECT_TRUE(unsealed.supports().empty());
        EXPECT_FALSE(Access::identity(unsealed));
        EXPECT_FALSE(Access::same_source(unsealed, unsealed));
        EXPECT_FALSE(Access::same_source(unsealed, copy));
        const auto valid_copy = copy;
        EXPECT_EQ(Access::identity(valid_copy), exported);
        EXPECT_TRUE(Access::same_source(copy, valid_copy));
        PLIC::DynamicPoint changed(source.value, source.absolute_error, mode);
        Access::seal_export(changed, root, {first, PLIC::PlaneAuthority::opaque({9, 12})});
        EXPECT_NE(changed.supports(), expected);
        for (std::size_t field = 0; field < 3; ++field)
        {
            auto mutated = copy;
            if (field == 0)
                mutated.value[0] += 1;
            else if (field == 1)
                mutated.absolute_error[0] += 1;
            else
                mutated.arithmetic_mode =
                    mode == ArithmeticMode::Ordinary ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
            EXPECT_TRUE(mutated.supports().empty());
            EXPECT_FALSE(Access::identity(mutated));
            EXPECT_FALSE(Access::same_source(mutated, copy));
            EXPECT_FALSE(Access::same_source(copy, mutated));
            EXPECT_FALSE(Access::same_source(mutated, mutated));
            EXPECT_EQ(Access::storage_bytes(mutated, false), Access::storage_bytes(copy, false));
            EXPECT_EQ(Access::storage_bytes(mutated), Access::storage_bytes(copy));
            EXPECT_GT(Access::storage_bytes(mutated, false), sizeof(PLIC::DynamicPoint));
            mutated.value = copy.value;
            mutated.absolute_error = copy.absolute_error;
            mutated.arithmetic_mode = copy.arithmetic_mode;
            EXPECT_EQ(mutated.supports(), copy.supports());
            EXPECT_EQ(Access::identity(mutated), exported);
            EXPECT_TRUE(Access::same_source(mutated, copy));
            EXPECT_TRUE(Access::same_source(copy, mutated));
            EXPECT_TRUE(Access::same_source(mutated, mutated));
        }
    }
}
TEST_F(GeometryProvenanceBorrow, SupportPayloadLivesThroughExportsAndCountsOnceAcrossAncestry)
{
    for (auto mode : finite_modes)
    {
        Observer parent;
        std::weak_ptr<const void> words;
        Identity retained;
        {
            PLIC::DynamicPoint source({1, 2, 4}, {.125L, 0, 0}, mode);
            std::vector<std::uint64_t> values{2, 7, 13};
            values.reserve(17);
            const auto bytes = sizeof(values) + values.capacity() * sizeof(std::uint64_t);
            auto key = PLIC::PlaneAuthority::opaque(std::move(values));
            words = key.opaque_storage_owner();
            Access::seal(source, {key});
            const auto root = Access::identity(source);
            parent = root.context_pin();
            PLIC::DynamicPoint output(source.value, source.absolute_error, mode);
            Access::seal_export(output, root, keys(source.supports()));
            retained = Access::identity(output);
            EXPECT_EQ(Access::identity_bytes(retained) - Access::identity_bytes(retained, false), bytes);
        }
        EXPECT_FALSE(parent.expired());
        EXPECT_FALSE(words.expired());
        EXPECT_EQ(retained.view().supports()[0].words(), (std::vector<std::uint64_t>{2, 7, 13}));
        retained.reset();
        EXPECT_TRUE(parent.expired());
        EXPECT_TRUE(words.expired());
    }
}
TEST_F(GeometryProvenanceBorrow, SharedSupportsDoNotTurnIndependentUncertainPointsIntoIdenticalPoints)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires supported FP environment.";
    constexpr auto mode = ArithmeticMode::BoundedCertification;
    PLIC::DynamicPoint first({0, 0, 0}, {.125L, .125L, .125L}, mode);
    Access::seal(first, {PLIC::PlaneAuthority::opaque({2, 7})});
    PLIC::DynamicPoint second(first.value, first.absolute_error, mode);
    Access::seal_export(second, Access::identity(first), keys(first.supports()));
    PLIC::DynamicPoint independent(first.value, first.absolute_error, mode);
    Access::seal(independent, keys(first.supports()));
    for (const auto& candidate : {second, independent})
    {
        EXPECT_FALSE(Access::same_source(first, candidate));
        EXPECT_FALSE(PLIC::points_identical(PLIC::Point<mode>(first), PLIC::Point<mode>(candidate)));
        const std::array<PLIC::Point<mode>, 4> tetrahedron{PLIC::Point<mode>(first), PLIC::Point<mode>(candidate),
            PLIC::Point<mode>({1, 0, 0}), PLIC::Point<mode>({0, 1, 0})};
        const auto determinant = PLIC::orientation(tetrahedron);
        EXPECT_LT(determinant.lower(), 0);
        EXPECT_GT(determinant.upper(), 0);
    }
}
TEST_F(GeometryProvenanceBorrow, RegisteredAuthorityReuseFitsAWorkCapThatRejectsANewOwner)
{
    typed_modes(
        [&]<ArithmeticMode Mode>()
        {
            detail::Operation operation(accounting_limits(1024));
            PLIC::PointSnapshotBuilder<Mode> builder;
            const auto a = PLIC::PlaneAuthority::opaque({2, 7}), b = PLIC::PlaneAuthority::opaque({2, 8});
            const auto issued = builder.register_authority(a);
            const auto admitted = operation.work.bytes;
            operation.work.limits.byte_budget = admitted + 1;
            EXPECT_EQ(builder.register_authority(a), issued);
            EXPECT_EQ(operation.work.bytes, admitted);
            EXPECT_TRUE(builder.context()->incidence.empty());
            EXPECT_THROW((void) builder.register_authority(b), PLIC::GeometryError);
            EXPECT_THROW((void) builder.freeze(), PLIC::GeometryError);
            EXPECT_TRUE(builder.context()->incidence.empty());
        });
}
TEST_F(GeometryProvenanceBorrow, FreshAuthorityTablesTrimSpareOuterCapacityAndRetainWordCapacities)
{
    std::vector<std::uint64_t> a{2, 7}, b{9, 11, 13};
    a.reserve(17);
    b.reserve(23);
    const auto ca = a.capacity(), cb = b.capacity();
    auto first = PLIC::PlaneAuthority::opaque(std::move(a)), second = PLIC::PlaneAuthority::opaque(std::move(b));
    const auto oa = first.opaque_storage_owner(), ob = second.opaque_storage_owner();
    std::vector<PLIC::PlaneAuthority> supports;
    supports.reserve(31);
    supports.push_back(second);
    supports.push_back(first);
    PLIC::DynamicPoint point({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    Access::seal(point, std::move(supports));
    const auto ctx = std::static_pointer_cast<const PLIC::PointSnapshotContext<ArithmeticMode::Ordinary>>(
        Access::identity(point).context_pin());
    EXPECT_EQ(ctx->authority_table.capacity(), 2u);
    EXPECT_EQ(ctx->incidence.capacity(), 2u);
    EXPECT_EQ(point.supports()[0].opaque_storage_owner(), oa);
    EXPECT_EQ(point.supports()[1].opaque_storage_owner(), ob);
    EXPECT_EQ(
        point.supports()[0].opaque_payload_bytes(), sizeof(std::vector<std::uint64_t>) + ca * sizeof(std::uint64_t));
    EXPECT_EQ(
        point.supports()[1].opaque_payload_bytes(), sizeof(std::vector<std::uint64_t>) + cb * sizeof(std::uint64_t));
    auto limits = accounting_limits(1024);
    limits.byte_budget = 1;
    detail::Operation op(limits);
    PLIC::DynamicPoint rejected({1, 2, 4}, {}, ArithmeticMode::Ordinary);
    EXPECT_THROW(Access::seal(rejected, {first}), PLIC::GeometryError);
    EXPECT_FALSE(Access::identity(rejected));
}
TEST_F(GeometryProvenanceBorrow, CsrAuthorityCopiesPreserveExactSequencesAndAppendIndependently)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const auto a = PLIC::PlaneAuthority::opaque({2, 7}), b = PLIC::PlaneAuthority::opaque({9, 11, 13}),
               c = PLIC::PlaneAuthority::opaque({17, 19});
    const std::array original{a, b}, reverse{b, a};
    const std::array extra{c};
    PLIC::PointSnapshotBuilder<mode> builder;
    const auto holder = builder.authority_range(Access::admit_incidence<mode>(original));
    const auto sibling = holder;
    const auto reordered = builder.authority_range(Access::admit_incidence<mode>(reverse));
    const auto appended = builder.append_authority(holder, Access::admit_incidence<mode>(extra));
    const auto ctx = builder.freeze();
    EXPECT_EQ(PLIC::resolve_authorities(ctx, holder), (std::vector<PLIC::PlaneAuthority>{a, b}));
    EXPECT_EQ(PLIC::resolve_authorities(ctx, sibling), PLIC::resolve_authorities(ctx, holder));
    EXPECT_EQ(PLIC::resolve_authorities(ctx, reordered), (std::vector<PLIC::PlaneAuthority>{b, a}));
    EXPECT_EQ(PLIC::resolve_authorities(ctx, appended), (std::vector<PLIC::PlaneAuthority>{a, b, c}));
    EXPECT_TRUE(PLIC::resolve_authorities(ctx, PLIC::LocalAuthorityRange<mode>{}).empty());
}
TEST_F(GeometryProvenanceBorrow, CsrCopyAndAliasedAppendPrechargeGrowthAndRejectAtomically)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const std::array keys{PLIC::PlaneAuthority::opaque({2, 7}), PLIC::PlaneAuthority::opaque({9, 11, 13})};
    for (bool append : {false, true})
    {
        const auto run = [&](std::size_t cap, bool expect_failure)
        {
            PLIC::PointSnapshotBuilder<mode> builder;
            const auto original = builder.authority_range(Access::admit_incidence<mode>(keys));
            detail::Operation operation(accounting_limits(1024));
            operation.work.limits.byte_budget = cap;
            PLIC::LocalAuthorityRange<mode> result;
            const auto mutate = [&]
            {
                if (append)
                    result = builder.append_authority(original,
                        Access::admit_incidence<mode>(PLIC::resolve_authorities(builder.context(), original)));
                else
                    result = builder.import_authority_range(PLIC::resolve_authorities(builder.context(), original));
            };
            if (expect_failure)
            {
                EXPECT_THROW(mutate(), PLIC::GeometryError);
                EXPECT_THROW((void) builder.freeze(), PLIC::GeometryError);
            }
            else
            {
                EXPECT_NO_THROW(mutate());
                const auto ctx = builder.freeze();
                EXPECT_EQ(result.size(), append ? 4u : 2u);
            }
            EXPECT_EQ(PLIC::resolve_authorities(builder.context(), original),
                (std::vector<PLIC::PlaneAuthority>{keys[0], keys[1]}));
            return operation.work.bytes;
        };
        const auto required = run(1024 * 1024, false);
        ASSERT_GT(required, 0u);
        EXPECT_LT(run(required - 1, true), required);
        EXPECT_EQ(run(required, false), required);
    }
}
TEST_F(GeometryProvenanceBorrow, AliasedCsrAppendKeepsWordOwnerWithoutPinningTheOldSnapshot)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    Observer source;
    std::weak_ptr<const void> words;
    auto ctx = [&]
    {
        PLIC::PointSnapshotBuilder<mode> builder;
        PLIC::LocalAuthorityRange<mode> original;
        {
            PLIC::DynamicPoint point({1, 2, 4}, {}, mode);
            auto key = PLIC::PlaneAuthority::opaque({2, 7});
            words = key.opaque_storage_owner();
            Access::seal(point, {key});
            source = Access::identity(point).context_pin();
            original = builder.import_authority_range(point.supports());
        }
        EXPECT_TRUE(source.expired());
        EXPECT_FALSE(words.expired());
        const auto appended = builder.append_authority(
            original, Access::admit_incidence<mode>(PLIC::resolve_authorities(builder.context(), original)));
        auto frozen = builder.freeze();
        EXPECT_EQ(appended.size(), 2u);
        EXPECT_EQ(PLIC::resolve_authorities(frozen, appended),
            (std::vector<PLIC::PlaneAuthority>{
                PLIC::PlaneAuthority::opaque({2, 7}), PLIC::PlaneAuthority::opaque({2, 7})}));
        return frozen;
    }();
    EXPECT_FALSE(words.expired());
    ctx.reset();
    EXPECT_TRUE(words.expired());
}
TEST_F(GeometryProvenanceBorrow, MovedFromOwningTokenStaysEmptyWhileFreshCsrAdmissionKeepsItsCap)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    auto holder = native(mode);
    const auto moved = std::move(holder);
    EXPECT_FALSE(holder);
    EXPECT_EQ(holder, Identity{});
    EXPECT_TRUE(moved);
    const std::array keys{PLIC::PlaneAuthority::opaque({11, 13})};
    const auto run = [&](std::size_t cap, bool failure)
    {
        PLIC::PointSnapshotBuilder<mode> builder;
        detail::Operation operation(accounting_limits(1024));
        operation.work.limits.byte_budget = cap;
        if (failure)
        {
            EXPECT_THROW((void) builder.authority_range(Access::admit_incidence<mode>(keys)), PLIC::GeometryError);
            EXPECT_THROW((void) builder.freeze(), PLIC::GeometryError);
        }
        else
        {
            const auto range = builder.authority_range(Access::admit_incidence<mode>(keys));
            const auto ctx = builder.freeze();
            EXPECT_EQ(range.size(), 1u);
        }
        EXPECT_FALSE(holder);
        EXPECT_TRUE(moved);
        return operation.work.bytes;
    };
    const auto required = run(1024 * 1024, false);
    ASSERT_GT(required, 1u);
    EXPECT_LT(run(1, true), required);
    EXPECT_LT(run(required - 1, true), required);
    EXPECT_EQ(run(required, false), required);
}
} // namespace
