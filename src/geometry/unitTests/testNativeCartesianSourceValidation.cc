/** Independent raw-C++ and original-numerics checks for first Cartesian proof. */
#include "geometry/PLIC/details/NativeBoxSourceAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "pipeline_allocation_operators.inc"
#include "utils/testing_environment.hh"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <utility>

namespace
{
using namespace SimpleFluid;
namespace N = geometry_detail;
namespace D = Meshes::AMR::adaptive_detail;
using Box = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
using Access = PLIC::details::NativeBoxSourceAccess;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(NativeCartesianSourceValidation, CompleteFirstProofScratchBoundsRawRequestedCppAndRetiresAfterFaces)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const auto counter = N::NativeConstructionCounterAccess::create<D::Budget>(16 * 1024 * 1024);
    const N::NativeConstructionAdmission admission(counter);
    const auto baseline = admission.counters();
    struct Verification
    {
        const N::NativeConstructionAdmission& admission;
        std::size_t faces = 0;
    } verification{admission};
    pipeline_trace::begin();
    Access::verify({-.7, .125, -2.3}, {1.1, 2.75, 3.9}, admission, &verification,
        [](void* raw, const Box& box)
        {
            auto& v = *static_cast<Verification*>(raw);
            for (std::size_t face = 0; face < box.num_faces(); ++face)
            {
                auto original = PLIC::details::NativePlanarOwnerAccess::cell_face(box, face, v.admission);
                auto copied = PLIC::details::NativePlanarOwnerAccess::make_wide(original->points(), v.admission);
                copied = PLIC::details::NativePlanarOwnerAccess::oriented_like(copied,
                    {original->area_vector().x, original->area_vector().y, original->area_vector().z}, v.admission);
                if (!original->contains(*copied) || !copied->contains(*original) ||
                    original->oppositely_oriented(*copied))
                    throw std::invalid_argument("Independent original face/copy comparison failed.");
                ++v.faces;
            }
        });
    pipeline_trace::recording.store(false);
    const auto raw = pipeline_trace::summary();
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(verification.faces, 6u);
    EXPECT_EQ(raw.all_live, 0u);
    EXPECT_LE(raw.all_peak, admission.counters().peak - baseline.used);
    EXPECT_EQ(admission.counters().used, baseline.used);
    EXPECT_EQ(admission.statistics().reserved_live, 0u);
    EXPECT_EQ(admission.statistics().profiled_live, 0u);
    EXPECT_EQ(admission.statistics().profiled_cumulative_upper, Access::scratch_bound());
}
TEST(NativeCartesianSourceValidation, ExactProfileSucceedsAndOneByteShortPrecedesOriginalConstruction)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const auto counter = N::NativeConstructionCounterAccess::create<D::Budget>(16 * 1024 * 1024);
    const N::NativeConstructionAdmission admission(counter);
    const auto baseline = admission.counters();
    const auto bound = Access::scratch_bound();
    counter.owner()->limit = baseline.used + bound - 1;
    bool visited = false;
    const auto callback = [](void* raw, const Box&) { *static_cast<bool*>(raw) = true; };
    const auto before = PLIC::geometry_statistics();
    EXPECT_THROW(Access::verify({0, 0, 0}, {1, 2, 3}, admission, &visited, callback), std::invalid_argument);
    EXPECT_FALSE(visited);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, before.constructions);
    EXPECT_EQ(admission.counters().used, baseline.used);
    counter.owner()->limit = baseline.used + bound;
    EXPECT_NO_THROW(Access::verify({0, 0, 0}, {1, 2, 3}, admission, &visited, callback));
    EXPECT_TRUE(visited);
    EXPECT_EQ(admission.counters().used, baseline.used);
}
TEST(NativeCartesianSourceValidation, OriginalBoxAcceptanceAndRejectionRemainTheOracle)
{
    ASSERT_TRUE(N::native_constructor_capacity_profile_supported());
    const std::array<std::pair<PLIC::GeometryPoint, PLIC::GeometryPoint>, 5> cases{
        {{{-.7, .125, -2.3}, {1.1, 2.75, 3.9}}, {{1e16, -8, .1}, {1e16 + 8, -2, .9}}, {{0, 0, 0}, {0, 1, 1}},
            {{0, 0, 0}, {1, -1, 1}}, {{0, 0, 0}, {std::numeric_limits<double>::infinity(), 1, 1}}}};
    for (const auto& [lower, upper] : cases)
    {
        const auto counter = N::NativeConstructionCounterAccess::create<D::Budget>(16 * 1024 * 1024);
        const N::NativeConstructionAdmission admission(counter);
        bool original_accepts = false, admitted_accepts = false;
        try
        {
            original_accepts = bool(Box::box(lower, upper));
        }
        catch (const std::invalid_argument&)
        {
        }
        try
        {
            Access::verify(lower, upper, admission, nullptr, [](void*, const Box&) {});
            admitted_accepts = true;
        }
        catch (const std::invalid_argument&)
        {
        }
        EXPECT_EQ(admitted_accepts, original_accepts);
        EXPECT_EQ(admission.statistics().reserved_live, 0u);
    }
}
} // namespace
