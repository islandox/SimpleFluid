#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "geometry/unitTests/pipeline_allocation_operators.inc"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Construction = PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(NativeProductViewAllocations, SharedViewAllocatesOnlyItsImplAndOwningWrapperBeyondTopologyEncoding)
{
    auto topology = std::make_shared<const ExtrudedTopology>(4, Arr<Arr<unsigned>>{{0, 1, 2, 3}}, 1);
    const auto source = Construction::make(topology,
        {{{0, 0, 0}, {}}, {{1, 0, 0}, {}}, {{1, 1, 0}, {}}, {{0, 1, 0}, {}}}, {0, 1, 2, 3}, {{0, {}}, {1, {}}}, {0, 1});
    pipeline_trace::begin();
    auto packet = topology->encode_base();
    pipeline_trace::recording.store(false);
    const auto encoding = pipeline_trace::summary();
    ASSERT_FALSE(pipeline_trace::overflow.load());
    ASSERT_FALSE(packet.empty());
    pipeline_trace::begin();
    auto view = Construction::shared_view(topology, source, source, 0, 1);
    pipeline_trace::recording.store(false);
    const auto publication = pipeline_trace::summary();
    ASSERT_FALSE(pipeline_trace::overflow.load());
    const auto phase = static_cast<unsigned>(pipeline_trace::Phase::Candidate);
    // The current standard ownership expressions allocate one combined Impl
    // control/object plus one wrapper object and one wrapper shared control.
    // Source base, selection and coordinate values are retained existing owners.
    EXPECT_EQ(publication.phase[phase].allocations, encoding.phase[phase].allocations + 3);
    EXPECT_EQ(view->base_storage_identity(), source->base_storage_identity());
    EXPECT_EQ(view->axis_storage_identity(), source->axis_storage_identity());
    EXPECT_EQ(view->axis_values_storage_identity(), source->axis_values_storage_identity());
    EXPECT_DOUBLE_EQ(view->cell_volume(0), 1);
    const auto center = view->cell_centroid(0);
    EXPECT_DOUBLE_EQ(center.x, .5);
    EXPECT_DOUBLE_EQ(center.y, .5);
    EXPECT_DOUBLE_EQ(center.z, .5);
    EXPECT_TRUE(source->atomic_cell_geometry(0)->equivalent(*view->atomic_cell_geometry(0)));
}
} // namespace
