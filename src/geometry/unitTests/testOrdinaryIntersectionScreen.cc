/** @file testOrdinaryIntersectionScreen.cc
 * @brief Independent original-kernel guard and first-failure prefix oracle.
 */
#include "geometry/PLIC/details/OrdinaryIntersection.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
namespace detail = SimpleFluid::PLIC::details;
constexpr auto mode = ArithmeticMode::Ordinary;
using Operand = detail::OrdinaryIntersectionOperand;
using Access = detail::OrdinaryIntersectionAccess;
GeometryLimits limits()
{
    GeometryLimits result;
    result.arithmetic_mode = mode;
    return result;
}
std::shared_ptr<const CellGeometryCatalog<mode>> catalogue(GeometryPoint lower, GeometryPoint upper)
{
    CartesianSourceDescription description;
    for (std::size_t axis = 0; axis < 3; ++axis)
        description.axes[axis] = {lower[axis], upper[axis]};
    CartesianSourceCellSlot cell;
    cell.upper = {1, 1, 1};
    description.cells.push_back(cell);
    const auto context = CartesianCellSource::make(std::move(description), limits());
    return CellGeometryCatalog<mode>::cartesian(context, limits());
}
CellGeometryView<mode> source(GeometryPoint lower, GeometryPoint upper)
{
    return {catalogue(lower, upper), 0};
}
struct Outcome
{
    std::optional<GeometryErrorCode> error;
    bool untyped_invalid_input = false;
    std::string message;
    ConvexGeometryPtr<mode> result;
    std::size_t operations = 0, pairs = 0, bytes = 0;
    std::uint64_t intersections = 0, floating_operations = 0;
};
Outcome run(Operand& first, Operand& second, GeometryLimits cap, bool prechecked)
{
    Outcome result;
    reset_geometry_statistics();
    detail::Operation guard(cap);
    try
    {
        if (prechecked)
        {
            auto receipt = Access::screen(first, second);
            if (!receipt.separated())
                result.result = receipt.finish();
        }
        else
            result.result = first.geometry()->intersect(*second.geometry());
    }
    catch (const GeometryError& error)
    {
        result.error = error.code();
        result.message = error.what();
    }
    catch (const std::invalid_argument& error)
    {
        result.untyped_invalid_input = true;
        result.message = error.what();
    }
    result.operations = guard.work.operations;
    result.pairs = guard.work.pairs;
    result.bytes = guard.work.bytes;
    result.intersections = geometry_statistics().intersections;
    result.floating_operations = geometry_statistics().floating_operations;
    return result;
}
void same_prefix(const Outcome& actual, const Outcome& old)
{
    EXPECT_EQ(actual.error, old.error);
    EXPECT_EQ(actual.untyped_invalid_input, old.untyped_invalid_input);
    EXPECT_EQ(actual.message, old.message);
    EXPECT_EQ(actual.operations, old.operations);
    EXPECT_EQ(actual.pairs, old.pairs);
    EXPECT_EQ(actual.bytes, old.bytes);
    EXPECT_EQ(actual.intersections, old.intersections);
    EXPECT_EQ(actual.floating_operations, old.floating_operations);
    EXPECT_EQ(bool(actual.result), bool(old.result));
    if (actual.result && old.result)
        EXPECT_TRUE(actual.result->equivalent(*old.result));
}
TEST(OrdinaryIntersectionScreen, ScalarPairAndFirstFailurePrefixesMatchOriginalKernel)
{
    struct Restore
    {
        std::fenv_t environment{};
        Restore() { (void) std::fegetenv(&environment); }
        ~Restore() { (void) std::fesetenv(&environment); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    // Independent physical oracles: y-contact has no interior, translated unit
    // boxes overlap by .5^3, and the contained half box has volume .5^3.
    const std::array<GeometryPoint, 3> lower{{{0, 1, 0}, {.5, .5, .5}, {.25, .25, .25}}};
    const std::array<GeometryPoint, 3> upper{{{1, 2, 1}, {1.5, 1.5, 1.5}, {.75, .75, .75}}};
    for (std::size_t scenario = 0; scenario < lower.size(); ++scenario)
    {
        SCOPED_TRACE(scenario);
        auto first = Operand::native(source({0, 0, 0}, {1, 1, 1}));
        auto second = Operand::native(source(lower[scenario], upper[scenario]));
        // Preparation is deliberately outside the guard oracle: both paths
        // compare exactly the same actual immutable operands and caps, while
        // exercising a receipt issued from their authenticated native views.
        CellGeometryOperationWorkspace preparation(limits());
        first.prepare(preparation);
        second.prepare(preparation);
        ASSERT_TRUE(first.prepared_frame_matches());
        ASSERT_TRUE(second.prepared_frame_matches());
        const auto full = run(first, second, limits(), false);
        ASSERT_FALSE(full.error) << full.message;
        if (scenario == 0)
            EXPECT_FALSE(full.result);
        else
        {
            ASSERT_TRUE(full.result);
            EXPECT_EQ(full.result->volume().value, .125L);
        }
        same_prefix(run(first, second, limits(), true), full);
        for (const auto scalar : {std::size_t(1), full.operations > 1 ? full.operations - 1 : std::size_t(1),
                 full.operations, full.operations + 1})
        {
            auto cap = limits();
            cap.maximum_operations = std::max<std::size_t>(1, scalar);
            SCOPED_TRACE(::testing::Message() << "scalar cap=" << cap.maximum_operations);
            const auto old = run(first, second, cap, false);
            same_prefix(run(first, second, cap, true), old);
        }
        for (const auto pairs : {std::size_t(1), full.pairs > 1 ? full.pairs - 1 : std::size_t(1),
                 std::max<std::size_t>(1, full.pairs), full.pairs + 1})
        {
            auto cap = limits();
            cap.maximum_pair_tests = pairs;
            SCOPED_TRACE(::testing::Message() << "pair cap=" << pairs);
            const auto old = run(first, second, cap, false);
            same_prefix(run(first, second, cap, true), old);
        }
    }
}
TEST(OrdinaryIntersectionScreen, ExactOwnerBindingAndSingleUseCannotBeBypassed)
{
    auto first = Operand::existing(ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    auto second = Operand::existing(ConvexCellGeometry<mode>::box({.5, .5, .5}, {1.5, 1.5, 1.5}));
    detail::Operation guard(limits());
    auto receipt = Access::screen(first, second);
    ASSERT_FALSE(receipt.separated());
    // Even equal-shape/equal-frame/equal-bound owners are distinct operands.
    first = Operand::existing(ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    EXPECT_THROW((void) receipt.finish(), GeometryError);
    auto valid = Access::screen(first, second);
    EXPECT_TRUE(valid.finish());
    EXPECT_THROW((void) valid.finish(), GeometryError);
}
struct RestoreEnvironment
{
    std::fenv_t environment{};
    RestoreEnvironment() { (void) std::fegetenv(&environment); }
    ~RestoreEnvironment() { (void) std::fesetenv(&environment); }
};
TEST(OrdinaryIntersectionScreen, WrongParentModeMatchesOriginalFirstErrorBeforeSeparatedScreen)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    auto first = Operand::existing(ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    auto second = Operand::existing(ConvexCellGeometry<mode>::box({0, 1, 0}, {1, 2, 1}));
    auto bounded = limits();
    bounded.arithmetic_mode = ArithmeticMode::BoundedCertification;
    const auto old = run(first, second, bounded, false);
    ASSERT_TRUE(old.untyped_invalid_input);
    same_prefix(run(first, second, bounded, true), old);
    EXPECT_EQ(old.operations, 0u);
    EXPECT_EQ(old.pairs, 0u);
    EXPECT_EQ(old.intersections, 0u);
}
TEST(OrdinaryIntersectionScreen, SeparatedDecisionAndFinishRejectChangedControlsAndPoisonReuse)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    auto first = Operand::existing(ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    auto second = Operand::existing(ConvexCellGeometry<mode>::box({0, 1, 0}, {1, 2, 1}));
    detail::Operation guard(limits());
    auto decision = Access::screen(first, second);
    auto finish = Access::screen(first, second);
    const auto operations = guard.work.operations, pairs = guard.work.pairs;
    const auto intersections = geometry_statistics().intersections;
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    EXPECT_THROW((void) decision.separated(), GeometryError);
    EXPECT_THROW((void) finish.finish(), GeometryError);
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    EXPECT_THROW((void) decision.separated(), GeometryError);
    EXPECT_THROW((void) finish.finish(), GeometryError);
    EXPECT_EQ(guard.work.operations, operations);
    EXPECT_EQ(guard.work.pairs, pairs);
    EXPECT_EQ(geometry_statistics().intersections, intersections);
    auto fresh = Access::screen(first, second);
    EXPECT_TRUE(fresh.separated());
    EXPECT_THROW((void) fresh.separated(), GeometryError);
    EXPECT_THROW((void) fresh.finish(), GeometryError);
}
TEST(OrdinaryIntersectionScreen, SeparatedDecisionAndFinishRejectDifferentWork)
{
    RestoreEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    auto first = Operand::existing(ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    auto second = Operand::existing(ConvexCellGeometry<mode>::box({0, 1, 0}, {1, 2, 1}));
    detail::Operation guard(limits());
    auto decision = Access::screen(first, second);
    auto finish = Access::screen(first, second);
    detail::Work other(limits());
    struct RestoreWork
    {
        detail::Work* previous = detail::active_work;
        ~RestoreWork() { detail::active_work = previous; }
    } restore_work;
    detail::active_work = &other;
    EXPECT_THROW((void) decision.separated(), GeometryError);
    EXPECT_THROW((void) finish.finish(), GeometryError);
    EXPECT_EQ(other.operations, 0u);
    EXPECT_EQ(other.pairs, 0u);
}
TEST(OrdinaryIntersectionScreen, CounterRegressionInvalidatesSeparatedDecisionAndFinish)
{
    auto first = Operand::existing(ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    auto second = Operand::existing(ConvexCellGeometry<mode>::box({0, 1, 0}, {1, 2, 1}));
    detail::Operation guard(limits());
    for (const auto field : {0, 1, 2})
    {
        guard.work.charge(1, 1);
        guard.work.pair();
        auto decision = Access::screen(first, second);
        auto finish = Access::screen(first, second);
        const auto operations = guard.work.operations, pairs = guard.work.pairs, bytes = guard.work.bytes;
        if (field == 0)
            guard.work.operations = 0;
        if (field == 1)
            guard.work.pairs = 0;
        if (field == 2)
            guard.work.bytes = 0;
        EXPECT_THROW((void) decision.separated(), GeometryError);
        EXPECT_THROW((void) finish.finish(), GeometryError);
        // Restore only deliberately corrupted test bookkeeping; production
        // receipt validation never refunds or mutates executed work counters.
        guard.work.operations = operations;
        guard.work.pairs = pairs;
        guard.work.bytes = bytes;
    }
}
TEST(OrdinaryIntersectionScreen, BorrowedSnapshotsAddNoExistingOwnerOrNativeCataloguePins)
{
    auto geometry = ConvexCellGeometry<mode>::box({0, 0, 0}, {1, 1, 1});
    auto contact = ConvexCellGeometry<mode>::box({0, 1, 0}, {1, 2, 1});
    auto first = Operand::existing(geometry), second = Operand::existing(contact);
    const auto owners = geometry.use_count(), contacts = contact.use_count();
    auto a = catalogue({0, 0, 0}, {1, 1, 1}), b = catalogue({0, 1, 0}, {1, 2, 1});
    auto native_first = Operand::native({a, 0}), native_second = Operand::native({b, 0});
    const auto a_pins = a.use_count(), b_pins = b.use_count();
    detail::Operation guard(limits());
    auto owning_receipt = Access::screen(first, second);
    auto native_receipt = Access::screen(native_first, native_second);
    EXPECT_EQ(geometry.use_count(), owners);
    EXPECT_EQ(contact.use_count(), contacts);
    EXPECT_EQ(a.use_count(), a_pins);
    EXPECT_EQ(b.use_count(), b_pins);
    EXPECT_TRUE(owning_receipt.separated());
    EXPECT_FALSE(native_receipt.finish());
    EXPECT_EQ(geometry.use_count(), owners);
    EXPECT_EQ(contact.use_count(), contacts);
    EXPECT_EQ(a.use_count(), a_pins);
    EXPECT_EQ(b.use_count(), b_pins);
}
} // namespace
