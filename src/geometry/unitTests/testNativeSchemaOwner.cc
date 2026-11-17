#include "geometry/details/NativeSchemaOwner.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <gtest/gtest.h>

namespace
{
namespace G = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
TEST(NativeSchemaOwner, CopyIsDetachedAndItsActualBufferAndFixedAdministrationAreAdmitted)
{
    auto arena = G::NativeConstructionCounterAccess::create<D::Budget>(65536);
    G::NativeConstructionAdmission admission(arena);
    const auto prefix = arena.owner()->used;
    const std::array<char, 19> input{'s', 'c', 'h', 'e', 'm', 'a'};
    auto schema = G::NativeSchemaOwnerAccess::copy(input, admission);
    EXPECT_TRUE(schema->matches(input));
    EXPECT_NE(schema->span().data(), input.data());
    EXPECT_EQ(arena.owner()->used - prefix, schema->allocation()->fixed_allocation_bytes() + schema->capacity_bytes());
    const auto output = schema->descriptor();
    EXPECT_TRUE(std::ranges::equal(output, input));
    EXPECT_NE(output.data(), schema->span().data());
    schema.reset();
    EXPECT_EQ(arena.owner()->used, prefix);
}
TEST(NativeSchemaOwner, AdoptionKeepsTheExistingBufferAndDoesNotRetainItsFormerController)
{
    struct Controller
    {
        std::vector<char> schema;
    };
    auto controller = std::make_shared<Controller>();
    controller->schema.assign(2048, 's');
    const auto* original = controller->schema.data();
    const auto capacity = controller->schema.capacity();
    std::weak_ptr<Controller> source = controller;
    auto arena = G::NativeConstructionCounterAccess::create<D::Budget>(65536);
    G::NativeConstructionAdmission admission(arena);
    const auto prefix = arena.owner()->used;
    auto schema = G::NativeSchemaOwnerAccess::adopt(std::move(controller->schema), admission);
    controller.reset();
    EXPECT_TRUE(source.expired());
    EXPECT_EQ(schema->span().data(), original);
    EXPECT_EQ(schema->capacity_bytes(), capacity);
    EXPECT_EQ(arena.owner()->used - prefix, schema->allocation()->fixed_allocation_bytes() + capacity);
    EXPECT_EQ(schema->span().size(), 2048u);
    schema.reset();
    EXPECT_EQ(arena.owner()->used, prefix);
}
TEST(NativeSchemaOwner, ExactPeakAndOneByteShortPreserveTheAcceptedPrefix)
{
    const std::array<char, 129> input{'s'};
    auto probe_arena = G::NativeConstructionCounterAccess::create<D::Budget>(65536);
    G::NativeConstructionAdmission probe(probe_arena);
    const auto probe_prefix = probe_arena.owner()->used;
    auto measured = G::NativeSchemaOwnerAccess::copy(input, probe);
    const auto peak = probe_arena.owner()->peak - probe_prefix;
    measured.reset();
    EXPECT_EQ(probe_arena.owner()->used, probe_prefix);
    auto arena = G::NativeConstructionCounterAccess::create<D::Budget>(65536);
    G::NativeConstructionAdmission admission(arena);
    const auto prefix = arena.owner()->used;
    arena.owner()->limit = prefix + peak - 1;
    EXPECT_THROW((void) G::NativeSchemaOwnerAccess::copy(input, admission), std::invalid_argument);
    EXPECT_EQ(arena.owner()->used, prefix);
    arena.owner()->limit = prefix + peak;
    auto retry = G::NativeSchemaOwnerAccess::copy(input, admission);
    EXPECT_TRUE(retry->matches(input));
}
TEST(NativeSchemaOwner, EmptyRootSchemaIsAnExplicitValidMetadataValue)
{
    auto arena = G::NativeConstructionCounterAccess::create<D::Budget>(65536);
    G::NativeConstructionAdmission admission(arena);
    auto schema = G::NativeSchemaOwnerAccess::copy({}, admission);
    EXPECT_TRUE(schema->span().empty());
    EXPECT_EQ(schema->capacity_bytes(), 0u);
    EXPECT_TRUE(schema->matches({}));
}
} // namespace
