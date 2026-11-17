#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/NativeIdentityOwnerAccess.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <gtest/gtest.h>

namespace
{
namespace P = SimpleFluid::PLIC;
namespace G = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
constexpr std::size_t cap = 16 * 1024 * 1024;
TEST(NativeIdentityProjection, TightAdmittedIdentityBudgetIncludesEachConcreteBacking)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    using Access = P::details::GeometryIdentityAccess;
    using OwnerAccess = P::details::NativeIdentityOwnerAccess;
    const G::NativeConstructionAdmission admission(G::NativeConstructionCounterAccess::create<D::Budget>(cap));
    const auto axis = Access::support_axis_admitted(1, .5L, admission);
    auto translated = axis;
    for (unsigned layer = 0; layer != 3; ++layer)
        translated = Access::translated_support_admitted(translated, {1.L, .25L, -2.L}, admission);
    for (const auto& sample : {axis, translated})
    {
        const auto key = P::canonical_key(sample);
        std::size_t depth = 0, logical_bytes = 0;
        for (auto owner = Access::backing(sample); owner; owner = owner->parent)
        {
            const auto* typed = OwnerAccess::backing(owner);
            ASSERT_TRUE(typed);
            std::size_t payload = 0;
            typed->storage().owner->visit_allocations([&](const auto& allocation)
                {
                    if (allocation.address == owner->native.get())
                        payload = allocation.bytes;
                });
            ASSERT_GT(payload, 0u);
            logical_bytes += sizeof(P::details::NativeGeometryIdentityBacking) + payload;
            ++depth;
        }
        // The complete backing path and the outer translation point stack
        // coexist at the final hash. These inputs have no other payload buffers.
        logical_bytes += depth * sizeof(const P::details::GeometryIdentityBacking*) +
            (depth - 1) * sizeof(P::details::OrdinaryIdentityPoint);
        const auto before = admission.statistics().actual_cumulative;
        EXPECT_THROW((void) Access::decode_support_admitted(key, logical_bytes - 1, admission), std::invalid_argument);
        EXPECT_EQ(admission.statistics().actual_cumulative, before);
        EXPECT_EQ(Access::decode_support_admitted(key, logical_bytes, admission), sample);
    }
}
} // namespace
