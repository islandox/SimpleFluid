#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "native_constructor_allocation_operators.inc"
#include <gtest/gtest.h>
#include <array>
namespace
{
namespace G = SimpleFluid::geometry_detail;
struct Counter
{
    std::size_t limit = 0, used = 0, peak = 0, cumulative = 0;
    G::NativeCounterAllocationEnvelope native_allocation;
    void add(std::size_t bytes)
    {
        if (used > limit || bytes > limit - used)
            throw std::invalid_argument("test codec cap");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
};
struct Domain
{
    G::NativeOwnedCounter<Counter> counter;
    G::NativeConstructionAdmission admission;
    explicit Domain(std::size_t limit = 4 * 1024 * 1024)
        : counter(G::NativeConstructionCounterAccess::create<Counter>(limit)), admission(counter)
    {
    }
};
TEST(NativePublicationCodecs, SeparateVectorControlKeepsOnlyRawObjectMemoryThroughWeakLifetime)
{
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    auto tracked = G::NativeObjectOwnerAccess<std::vector<char>>::make(domain.admission);
    const auto storage = G::NativeObjectOwnerAccess<std::vector<char>>::storage(tracked.owner);
    storage.owner->reserve(*tracked.owner, 137);
    tracked.owner->resize(137, 's');
    std::weak_ptr<const std::vector<char>> weak = tracked.owner;
    tracked.owner.reset();
    const auto fixed = tracked.envelope->fixed_allocation_bytes();
    const auto remaining = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(storage.owner->exclusive_bytes(), 0u);
    EXPECT_EQ(remaining, trace.all_live);
    EXPECT_GE(remaining, fixed);
    tracked.envelope.reset();
    weak.reset();
    // An independent storage observer remains, but no freed payload is charged.
    EXPECT_EQ(storage.owner->exclusive_bytes(), 0u);
}
TEST(NativePublicationCodecs, SeparateControlEveryHeapPrefixAndConstructorFailureRetireExactly)
{
    const auto operation = [](const G::NativeConstructionAdmission& admission)
    {
        auto owner = G::NativeObjectOwnerAccess<std::vector<char>>::make(admission);
        const auto& storage = G::NativeObjectOwnerAccess<std::vector<char>>::storage(owner.owner);
        storage.owner->reserve(*owner.owner, 97);
        owner.owner->resize(97, 'v');
        return owner;
    };
    Domain measured;
    native_constructor_test::Capture capture(measured.counter.owner()->used);
    {
        const auto value = operation(measured.admission);
    }
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    for (std::size_t ordinal = 0; ordinal < trace.phase[0].allocations; ++ordinal)
    {
        Domain domain;
        const auto accepted = domain.admission.shared<unsigned>(211);
        const auto prefix = domain.counter.owner()->used;
        bool failed = false;
        {
            native_constructor_test::Failure failure(ordinal);
            try
            {
                const auto value = operation(domain.admission);
            }
            catch (const std::bad_alloc&)
            {
                failed = true;
            }
        }
        EXPECT_TRUE(failed) << ordinal;
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
        EXPECT_EQ(*accepted, 211u);
        {
            const auto retry = operation(domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix);
    }
    struct Throwing
    {
        Throwing() { throw std::bad_alloc(); }
    };
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    EXPECT_THROW((void) G::NativeObjectOwnerAccess<Throwing>::make_without_buffers(domain.admission), std::bad_alloc);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativePublicationCodecs, SeparateOveralignedObjectBlockSurvivesUntilLastWeakControl)
{
    struct alignas(256) Aligned
    {
        std::array<std::byte, 256> values{};
    };
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    auto tracked = G::NativeObjectOwnerAccess<Aligned>::make_without_buffers(domain.admission);
    const auto address = reinterpret_cast<std::uintptr_t>(tracked.owner.get());
    std::weak_ptr<const Aligned> weak = tracked.owner;
    tracked.owner.reset();
    const auto live = domain.counter.owner()->used - prefix;
    const auto fixed = tracked.envelope->fixed_allocation_bytes();
    const auto trace = capture.finish();
    EXPECT_EQ(address % alignof(Aligned), 0u);
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_EQ(live, trace.all_live);
    EXPECT_EQ(live, fixed);
    EXPECT_TRUE(weak.expired());
    tracked.envelope.reset();
    weak.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
} // namespace
