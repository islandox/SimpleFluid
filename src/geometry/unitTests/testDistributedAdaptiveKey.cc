/** @file testDistributedAdaptiveKey.cc
 * @brief Generic provenance encoding is separate from provider admission.
 */
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid::Meshes;
TEST(DistributedAdaptiveKey, GenericTokensAndDepthRoundTripWithoutProviderLimits)
{
    constexpr auto high = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> path(129, high);
    path[0] = 99;
    const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> keys = {
        {high, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{high, path}},
        {high, ::SimpleFluid::Meshes::AMR::ProductCellAncestry{{high, high, path}, {high, path}}},
        {high, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{std::numeric_limits<unsigned>::max(),
                   {high, high - 1, high - 2}}},
        {high, ::SimpleFluid::Meshes::AMR::StaticCellAncestry{high, high - 1}}};
    for (const auto& key : keys)
        EXPECT_EQ(
            ::SimpleFluid::Meshes::AMR::decode_adaptive_key(::SimpleFluid::Meshes::AMR::encode_adaptive_key(key)), key);
    auto truncated = ::SimpleFluid::Meshes::AMR::encode_adaptive_key(keys[0]);
    truncated.pop_back();
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::decode_adaptive_key(truncated), std::invalid_argument);
    auto invalid_count = ::SimpleFluid::Meshes::AMR::encode_adaptive_key(keys[0]);
    // Header, region, variant and convex root precede the path count.
    for (std::size_t i = 48; i < 56; ++i)
        invalid_count[i] = char(255);
    EXPECT_THROW((void) ::SimpleFluid::Meshes::AMR::decode_adaptive_key(invalid_count), std::invalid_argument);
}
TEST(DistributedAdaptiveKey, PolyhedralKernelRejectsNonOctantOrTooDeepActualAncestry)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveLeaf leaf;
    leaf.key = {0, ::SimpleFluid::Meshes::AMR::ConvexCellAncestry{0, {}}};
    leaf.domain = {'d'};
    leaf.geometry =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    for (std::size_t f = 0; f < leaf.geometry->num_faces(); ++f)
        leaf.surfaces.push_back(
            {::SimpleFluid::Meshes::AMR::AdaptiveBytes{char(f + 1)}, leaf.geometry->face_geometry(f), 1, true, {}});
    auto kernel = ::SimpleFluid::Meshes::AMR::convex_adaptive_kernel(3, 0);
    EXPECT_NO_THROW(kernel->validate(leaf));
    auto invalid = leaf;
    std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(invalid.key.ancestry).path = {99};
    EXPECT_EQ(
        ::SimpleFluid::Meshes::AMR::decode_adaptive_key(::SimpleFluid::Meshes::AMR::encode_adaptive_key(invalid.key)),
        invalid.key);
    EXPECT_THROW(kernel->validate(invalid), std::invalid_argument);
    EXPECT_THROW(
        (void) kernel->split(invalid, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic), std::invalid_argument);
    std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(invalid.key.ancestry).path.assign(4, 0);
    EXPECT_THROW(kernel->validate(invalid), std::invalid_argument);
}
} // namespace
