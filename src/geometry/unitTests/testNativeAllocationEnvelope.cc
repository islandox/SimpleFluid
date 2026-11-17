#include "geometry/details/NativeAllocationEnvelope.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <gtest/gtest.h>

namespace
{
namespace G = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
static_assert(!std::is_copy_constructible_v<G::NativeAllocationEnvelope>);
static_assert(!std::is_move_constructible_v<G::NativeAllocationEnvelope>);
struct Child
{
    int value = 42;
};
struct KernelLike
{
    explicit KernelLike(std::shared_ptr<Child> value) : child(std::move(value)) {}
    std::shared_ptr<Child> child;
};
TEST(NativeAllocationEnvelope, FixedWeakExtentSurvivesObjectAndDynamicChildWithoutOwningSourceController)
{
    struct Controller
    {
        std::shared_ptr<D::Budget> counter;
        std::shared_ptr<Child> child;
    };
    auto counter = std::make_shared<D::Budget>(D::Budget{65536});
    G::NativeConstructionAdmission admission(counter);
    const auto administration = counter->used;
    auto controller = std::make_shared<Controller>();
    controller->counter = counter;
    controller->child = admission.shared<Child>();
    std::weak_ptr<Controller> source_control = controller;
    std::weak_ptr<Child> dynamic_child = controller->child;
    auto tracked = G::NativeAllocationEnvelopeAccess::make<KernelLike>(admission, controller->child);
    std::weak_ptr<KernelLike> target = tracked.owner;
    const auto fixed = tracked.envelope->fixed_allocation_bytes();
    EXPECT_TRUE(tracked.envelope->matches(tracked.owner));
    EXPECT_TRUE(tracked.envelope->object_alive());
    controller.reset();
    EXPECT_TRUE(source_control.expired());
    EXPECT_FALSE(dynamic_child.expired());
    tracked.owner.reset();
    EXPECT_TRUE(target.expired());
    EXPECT_TRUE(dynamic_child.expired());
    dynamic_child.reset(); // retire the test's separate weak child-control observation before the allocation census
    EXPECT_FALSE(tracked.envelope->object_alive());
    EXPECT_EQ(tracked.envelope->fixed_allocation_bytes(), fixed);
    EXPECT_EQ(counter->used, administration + fixed);
    std::size_t extents = 0, bytes = 0;
    tracked.envelope->visit_allocations(
        [&](const G::NativeAllocationExtent& extent)
        {
            EXPECT_NE(extent.address, nullptr);
            bytes += extent.bytes;
            ++extents;
        });
    EXPECT_EQ(extents, 4u);
    EXPECT_EQ(bytes, fixed);
    target.reset();
    tracked.envelope.reset();
    EXPECT_EQ(counter->used, administration);
}
TEST(NativeAllocationEnvelope, SourceAndOperationScopeExpireWhileTheChargedWeakEnvelopeSurvives)
{
    struct Controller
    {
        std::shared_ptr<D::Budget> counter;
        std::shared_ptr<Child> child;
    };
    std::weak_ptr<D::Budget> counter_observation;
    std::weak_ptr<Controller> controller_observation;
    std::weak_ptr<KernelLike> kernel_observation;
    std::shared_ptr<const G::NativeAllocationEnvelope> envelope;
    {
        auto source = std::make_shared<Controller>();
        source->counter = std::make_shared<D::Budget>(D::Budget{65536});
        counter_observation = source->counter;
        controller_observation = source;
        G::NativeConstructionAdmission admission(source->counter);
        source->child = admission.shared<Child>();
        auto tracked = G::NativeAllocationEnvelopeAccess::make<KernelLike>(admission, source->child);
        kernel_observation = tracked.owner;
        envelope = tracked.envelope;
    }
    EXPECT_TRUE(controller_observation.expired());
    EXPECT_TRUE(kernel_observation.expired());
    EXPECT_FALSE(envelope->object_alive());
    EXPECT_FALSE(counter_observation.expired());
    kernel_observation.reset();
    envelope.reset();
    EXPECT_TRUE(counter_observation.expired());
}

TEST(NativeAllocationEnvelope, ForeignAndAliasedControlsDoNotMatchTheCapturedTarget)
{
    auto counter = std::make_shared<D::Budget>(D::Budget{65536});
    G::NativeConstructionAdmission admission(counter);
    auto tracked = G::NativeAllocationEnvelopeAccess::make<Child>(admission);
    auto other = G::NativeAllocationEnvelopeAccess::make<Child>(admission);
    EXPECT_FALSE(tracked.envelope->matches(other.owner));
    auto foreign = std::shared_ptr<Child>(tracked.owner.get(), [](auto*) {});
    ASSERT_EQ(foreign.get(), tracked.owner.get());
    EXPECT_FALSE(tracked.envelope->shares_control(foreign));
    auto foreign_envelope =
        std::shared_ptr<const G::NativeAllocationEnvelope>(tracked.envelope.get(), [](const auto*) {});
    EXPECT_FALSE(tracked.envelope->authentic_owner(foreign_envelope));
    EXPECT_TRUE(tracked.envelope->authentic_owner(tracked.envelope));
    auto alias = std::shared_ptr<const void>(tracked.owner, &tracked.owner->value);
    EXPECT_TRUE(tracked.envelope->shares_control(alias));
}
TEST(NativeAllocationEnvelope, EveryDeniedFixedAllocationRollsBackToTheAcceptedCounterPrefix)
{
    auto probe_counter = std::make_shared<D::Budget>(D::Budget{65536});
    G::NativeConstructionAdmission probe(probe_counter);
    const auto prefix = probe_counter->used;
    auto measured = G::NativeAllocationEnvelopeAccess::make<Child>(probe);
    const auto fixed = measured.envelope->fixed_allocation_bytes();
    EXPECT_EQ(probe_counter->used - prefix, fixed);
    measured.owner.reset();
    measured.envelope.reset();
    EXPECT_EQ(probe_counter->used, prefix);
    auto counter = std::make_shared<D::Budget>(D::Budget{65536});
    G::NativeConstructionAdmission admission(counter);
    const auto accepted = counter->used;
    counter->limit = accepted + fixed - 1;
    EXPECT_THROW((void) G::NativeAllocationEnvelopeAccess::make<Child>(admission), std::invalid_argument);
    EXPECT_EQ(counter->used, accepted);
    counter->limit = accepted + fixed;
    auto retry = G::NativeAllocationEnvelopeAccess::make<Child>(admission);
    EXPECT_EQ(counter->used, accepted + fixed);
    EXPECT_TRUE(retry.envelope->matches(retry.owner));
}
} // namespace
