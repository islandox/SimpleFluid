#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/PLIC/details/NativeConstructionWireAccess.hh"
#include "geometry/mesh/NativeFragmentDecodePlan.hh"
#include "geometry/unitTests/pipeline_allocation_operators.inc"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<const NativeFragmentDescriptor> cartesian()
{
    NativeFragmentCache cache;
    return extract_native_fragment(cartesian_region("preflight", Vec3D<ArrReal>{{{0, .5, 1}, {0, 1}, {0, 1}}}), cache);
}
std::shared_ptr<const NativeFragmentDescriptor> product()
{
    auto topology = std::make_shared<const ExtrudedTopology>(4, Arr<Arr<unsigned>>{{0, 1, 2, 3}}, 2);
    const auto construction = PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>::make(topology,
        {{{0, 0, 0}, {}}, {{1, 0, 0}, {}}, {{1, 1, 0}, {}}, {{0, 1, 0}, {}}, {{0, 0, 0}, {0, 1, 2, 3}}}, {0, 1, 2, 3},
        {{0, {}}, {1, {}}, {0, {0, 1}}}, {0, 2, 1});
    NativeFragmentCache cache;
    return extract_native_fragment(
        ExtrudedRegion("preflight", topology, std::make_shared<const ExtrudedGeometry>(topology, construction)), cache);
}

TEST(NativeFragmentDecodePlan, CartesianCountsHaveIndependentTensorExpectationsAndAllocateNothing)
{
    const auto recipe = cartesian();
    pipeline_trace::begin();
    const auto plan = NativeFragmentDecodePlan::scan(recipe);
    pipeline_trace::recording.store(false);
    const auto trace = pipeline_trace::summary();
    ASSERT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(trace.phase[static_cast<unsigned>(pipeline_trace::Phase::Candidate)].allocations, 0u);
    EXPECT_EQ(plan.counts().cells, 2u);
    EXPECT_EQ(plan.counts().nodes, 12u);
    EXPECT_EQ(plan.counts().name_bytes, 9u);
    EXPECT_TRUE(plan.matches(recipe));
    auto foreign = std::make_shared<const NativeFragmentDescriptor>(*recipe);
    EXPECT_FALSE(plan.matches(foreign));
}
TEST(NativeFragmentDecodePlan, ProductCountsFollowExplicitBaseAndAxisDagsWithoutGeometryConstruction)
{
    const auto recipe = product();
    pipeline_trace::begin();
    const auto plan = NativeFragmentDecodePlan::scan(recipe);
    pipeline_trace::recording.store(false);
    const auto trace = pipeline_trace::summary();
    ASSERT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(trace.phase[static_cast<unsigned>(pipeline_trace::Phase::Candidate)].allocations, 0u);
    EXPECT_EQ(plan.counts().nodes, 4u);
    EXPECT_EQ(plan.counts().cells, 1u);
    EXPECT_EQ(plan.counts().loop_incidence, 4u);
    EXPECT_EQ(plan.counts().maximum_loop, 4u);
    EXPECT_EQ(plan.counts().boundary_edges, 4u);
    EXPECT_EQ(plan.counts().point_recipes, 5u);
    EXPECT_EQ(plan.counts().point_parents, 4u);
    EXPECT_EQ(plan.counts().active_points, 4u);
    EXPECT_EQ(plan.counts().axial_recipes, 3u);
    EXPECT_EQ(plan.counts().axial_parents, 2u);
    EXPECT_EQ(plan.counts().selected_axial, 3u);
    EXPECT_EQ(plan.counts().first_layer, 0u);
    EXPECT_EQ(plan.counts().layers, 2u);
    EXPECT_TRUE(plan.matches(recipe));
}
TEST(NativeFragmentDecodePlan, TruncatedAndForwardParentPacketsReject)
{
    const auto source = product();
    auto recipe = std::make_shared<NativeFragmentDescriptor>(*source);
    auto truncated = *recipe->parts[3];
    truncated.pop_back();
    recipe->parts[3] = std::make_shared<const NativeFragmentBytes>(std::move(truncated));
    EXPECT_THROW((void) NativeFragmentDecodePlan::scan(recipe), std::exception);
    recipe->parts[3] = source->parts[3];
    auto base = *source->parts[1];
    // Base tag/version, topology length+bytes, then point recipe count precede
    // the first root. Its first recipe cannot name any earlier parent.
    std::uint64_t topology_length = 0;
    for (unsigned index = 0; index != 8; ++index)
        topology_length |= std::uint64_t(static_cast<unsigned char>(base[2 + index])) << (8 * index);
    std::size_t cursor = 2 + 8 + static_cast<std::size_t>(topology_length) + 8;
    for (unsigned axis = 0; axis != 3; ++axis)
        (void) PLIC::details::NativeConstructionWireAccess::ordinary_scalar(base, cursor);
    ASSERT_LT(cursor, base.size());
    base[cursor] = 1;
    recipe->parts[1] = std::make_shared<const NativeFragmentBytes>(std::move(base));
    EXPECT_THROW((void) NativeFragmentDecodePlan::scan(recipe), std::exception);
}
TEST(NativeFragmentDecodePlan, ReboundPartsAndKindCannotUseTheOriginalPlan)
{
    const auto source = cartesian();
    auto recipe = std::make_shared<NativeFragmentDescriptor>(*source);
    const auto plan = NativeFragmentDecodePlan::scan(recipe);
    ASSERT_TRUE(plan.matches(recipe));
    recipe->parts[1] = std::make_shared<const NativeFragmentBytes>(*recipe->parts[1]);
    EXPECT_FALSE(plan.matches(recipe));
    recipe->parts[1] = source->parts[1];
    ASSERT_TRUE(plan.matches(recipe));
    recipe->kind = NativeFragmentDescriptor::Kind::ConstructedProduct;
    EXPECT_FALSE(plan.matches(recipe));
}
} // namespace
