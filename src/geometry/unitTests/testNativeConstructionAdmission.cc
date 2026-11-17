#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/NativeConstructionAdmission.hh"
#include <cstdint>
#include <gtest/gtest.h>
#include <thread>

namespace
{
using namespace SimpleFluid::Meshes;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
struct alignas(128) Aligned
{
    std::array<std::byte, 128> payload;
};
struct Throwing
{
    Throwing() { throw std::runtime_error("expected constructor rejection"); }
};
TEST(NativeConstructionAdmission, ActualControlAndAlignedAllocationsRetireAfterLastOwner)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    const auto prefix = arena->used;
    auto owner = admission.shared<Aligned>();
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(owner.get()) % alignof(Aligned), 0u);
    EXPECT_GT(arena->used - prefix, sizeof(Aligned)); // actual rebound shared control included
    const auto retained = arena->used;
    auto alias = owner;
    owner.reset();
    EXPECT_EQ(arena->used, retained);
    alias.reset();
    EXPECT_EQ(arena->used, prefix);
    auto separate = admission.unique<Aligned>();
    EXPECT_EQ(arena->used - prefix, sizeof(Aligned));
    separate.reset();
    EXPECT_EQ(arena->used, prefix);
}
TEST(NativeConstructionAdmission, ConstructorFailureReleasesOnlyItsOwnAllocation)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    auto accepted = admission.shared<std::uint64_t>(42);
    const auto before = arena->used;
    const auto cumulative = arena->cumulative;
    EXPECT_THROW((void) admission.shared<Throwing>(), std::runtime_error);
    EXPECT_EQ(arena->used, before);
    EXPECT_GT(arena->cumulative, cumulative);
    EXPECT_GT(arena->peak, before);
    EXPECT_THROW((void) admission.unique<Throwing>(), std::runtime_error);
    EXPECT_EQ(arena->used, before);
    EXPECT_EQ(*accepted, 42u);
}
TEST(NativeConstructionAdmission, VectorReallocationBudgetsSimultaneousOldAndNewBuffers)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    const auto prefix = arena->used;
    {
        NativeConstructorVector<std::uint64_t> values(admission.allocator<std::uint64_t>());
        values.reserve(4);
        const auto before = arena->used;
        const auto old_capacity = values.capacity();
        values.reserve(8);
        EXPECT_EQ(arena->used - prefix, values.capacity() * sizeof(std::uint64_t));
        EXPECT_GE(arena->peak, before + values.capacity() * sizeof(std::uint64_t));
        EXPECT_GE(old_capacity, 4u);
    }
    EXPECT_EQ(arena->used, prefix);
}
TEST(NativeConstructionAdmission, OneByteShortRejectsBeforeAllocatorAndRetryPreservesAcceptedPrefix)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    const auto domain_prefix = arena->used;
    auto accepted = admission.shared<std::uint64_t>(42);
    const auto prefix = arena->used;
    constexpr std::size_t bytes = 17 * sizeof(Aligned);
    arena->limit = prefix + bytes - 1;
    auto memory = admission.allocator<Aligned>();
    EXPECT_THROW((void) memory.allocate(17), std::invalid_argument);
    EXPECT_EQ(arena->used, prefix);
    EXPECT_EQ(*accepted, 42u);
    arena->limit = prefix + bytes;
    auto* output = memory.allocate(17);
    EXPECT_EQ(arena->used, prefix + bytes);
    memory.deallocate(output, 17);
    EXPECT_EQ(arena->used, prefix);
    accepted.reset();
    EXPECT_EQ(arena->used, domain_prefix);
}
TEST(NativeConstructionAdmission, DetachedCounterFollowsActualControlWithoutRetainingSourceController)
{
    struct SourceController
    {
        std::shared_ptr<D::Budget> counter = std::make_shared<D::Budget>(D::Budget{65536});
        std::shared_ptr<char> identity = std::make_shared<char>('s');
    };
    std::shared_ptr<Aligned> output;
    std::weak_ptr<D::Budget> counter;
    std::weak_ptr<char> source_identity;
    std::weak_ptr<SourceController> source_control;
    {
        auto source = std::make_shared<SourceController>();
        source_control = source;
        counter = source->counter;
        source_identity = source->identity;
        NativeConstructionAdmission admission(source->counter);
        output = admission.shared<Aligned>();
    }
    EXPECT_TRUE(source_control.expired());
    EXPECT_TRUE(source_identity.expired());
    EXPECT_FALSE(counter.expired());
    std::weak_ptr<Aligned> physical_control = output;
    output.reset();
    EXPECT_TRUE(physical_control.expired());
    // An observing weak pointer still keeps the actual shared control allocation
    // and its detached allocation counter alive until that control is released.
    EXPECT_FALSE(counter.expired());
    physical_control.reset();
    EXPECT_TRUE(counter.expired());
}

TEST(NativeConstructionAdmission, ACounterCannotBeReboundToIndependentRacingStates)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    const auto before = arena->used;
    EXPECT_THROW((void) NativeConstructionAdmission(arena), std::invalid_argument);
    EXPECT_EQ(arena->used, before);
    auto shared_domain = admission;
    auto value = shared_domain.shared<unsigned>(7);
    EXPECT_EQ(*value, 7u);
}
TEST(NativeConstructionAdmission, IndependentOwnersRetireConcurrentlyAgainstOneDetachedCounter)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    const auto prefix = arena->used;
    std::vector<std::shared_ptr<unsigned>> first, second;
    for (unsigned index = 0; index != 256; ++index)
        (index % 2 ? first : second).push_back(admission.shared<unsigned>(index));
    std::thread a([values = std::move(first)]() mutable { values.clear(); });
    std::thread b([values = std::move(second)]() mutable { values.clear(); });
    a.join();
    b.join();
    EXPECT_EQ(arena->used, prefix);
}

TEST(NativeConstructionAdmission, MapAndHashRebindingKeepsNodesAndBucketsInTheSameArena)
{
    auto arena = std::make_shared<D::Budget>(D::Budget{65536});
    NativeConstructionAdmission admission(arena);
    const auto prefix = arena->used;
    using Pair = std::pair<const unsigned, std::uint64_t>;
    {
        NativeConstructorMap<unsigned, std::uint64_t> ordered(std::less<unsigned>{}, admission.allocator<Pair>());
        NativeConstructorHash<unsigned, std::uint64_t> hashed(
            0, std::hash<unsigned>{}, std::equal_to<unsigned>{}, admission.allocator<Pair>());
        ordered.emplace(1, 7);
        const auto ordered_bytes = arena->used;
        EXPECT_GT(ordered_bytes - prefix, sizeof(Pair));
        hashed.emplace(1, 9);
        EXPECT_GT(arena->used, ordered_bytes + sizeof(Pair));
        EXPECT_EQ(ordered.at(1), 7u);
        EXPECT_EQ(hashed.at(1), 9u);
    }
    EXPECT_EQ(arena->used, prefix);
}
} // namespace
