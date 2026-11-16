#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/ContactPointArray.hh"
#include "geometry/mesh/AMR/IncidencePools.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <type_traits>
#include <vector>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
static_assert(D::IdentityPool<D::ContactPointIdentity>::uses_hashing);
static_assert(D::IdentityPool<SimpleFluid::Meshes::AMR::AdaptiveBytes>::uses_hashing);
static_assert(D::IdentityPool<SimpleFluid::Meshes::AMR::AdaptiveCellKey>::uses_hashing);
static_assert(!D::IdentityPool<D::IncidencePools::Plane>::uses_hashing);
static_assert(!D::IdentityPool<D::IncidencePools::Normal>::uses_hashing);
static_assert(!D::IdentityPool<D::IncidencePools::Image>::uses_hashing);
static_assert(std::is_same_v<decltype(D::IncidencePools::points), D::IncidencePointDirectory<>>);
static_assert(std::is_same_v<D::IncidencePointDirectory<>::Store, D::IdentityPool<D::ContactPointIdentity>>);
static_assert(std::is_empty_v<D::IdentityPool<D::ContactPointIdentity>::HashPolicy>);
static_assert(std::is_same_v<decltype(D::ContactPointArray::Reference::normalized), D::ContactPointIdentity>);
static_assert(std::is_same_v<decltype(D::ContactPointArray::Reference::physical), D::ContactPointIdentity>);
D::ContactPointIdentity point(long double x, long double y = 0, long double z = 0)
{
    return P::details::GeometryIdentityAccess::point({x, y, z});
}
std::vector<std::pair<D::ContactPointIdentity, D::ContactPointIdentity>> values(const D::ContactPointArray& array)
{
    std::vector<std::pair<D::ContactPointIdentity, D::ContactPointIdentity>> result;
    for (const auto& value : array)
        result.emplace_back(value.normalized, value.physical);
    return result;
}
const void* buffer_identity(const D::ContactPointArray& array, D::ContactPayloadRole role)
{
    const void* identity = nullptr;
    array.visit_borrowed_payload(
        [&](const D::ContactPayloadAllocation& allocation)
        {
            if (allocation.identity.role == role)
                identity = allocation.identity.address;
        });
    return identity;
}
TEST(ContactPayloadCensus, CategoriesRemainDistinctAtSameOrdinalAndAddress)
{
    const std::array<D::ContactPayloadRole, 3> roles{D::BasicContactPayloadRole::PointArrayObject,
        D::ParallelSupportPayloadRole::GroupObject, D::GeometryContactPayloadRole::SourcePointSnapshotObject};
    for (const auto role : roles)
        EXPECT_EQ(role.ordinal(), 0u);
    EXPECT_EQ(roles[0].category(), D::ContactPayloadCategory::Basic);
    EXPECT_EQ(roles[1].category(), D::ContactPayloadCategory::ParallelSupport);
    EXPECT_EQ(roles[2].category(), D::ContactPayloadCategory::Geometry);
    EXPECT_LT(roles[0], roles[1]);
    EXPECT_LT(roles[1], roles[2]);
    EXPECT_EQ((std::set<D::ContactPayloadRole>(roles.begin(), roles.end()).size()), 3u);
    const int allocation = 0;
    const D::ContactPayloadIdentity basic{&allocation, roles[0]}, parallel{&allocation, roles[1]},
        geometry{&allocation, roles[2]};
    EXPECT_NE(basic, parallel);
    EXPECT_NE(basic, geometry);
    EXPECT_NE(parallel, geometry);
    EXPECT_EQ(basic, (D::ContactPayloadIdentity{&allocation, D::BasicContactPayloadRole::PointArrayObject}));
}
TEST(ContactPointArray, EmptyAndIdentityRowsHaveNoUnusedNormalizedArray)
{
    D::ContactPointArray array;
    EXPECT_EQ(array.size(), 0u);
    EXPECT_EQ(array.capacity(), 0u);
    EXPECT_EQ(array.storage_bytes(), 0u);
    EXPECT_EQ(array.begin(), array.end());
    EXPECT_THROW((void) array.at(0), std::out_of_range);
    array.reserve(8);
    for (unsigned i = 0; i < 8; ++i)
    {
        const auto value = point(i, i + 1, i + 2);
        array.push_back({value, value});
        EXPECT_EQ(array[i].physical, value);
        EXPECT_FALSE(buffer_identity(array, D::BasicContactPayloadRole::PointArrayNormalized));
        EXPECT_EQ(P::encode_identity(array[i].normalized), P::encode_identity(value));
    }
    EXPECT_TRUE(array.shares_normalized_storage());
    EXPECT_EQ(array.storage_bytes(), 8 * sizeof(D::ContactPointIdentity));
    RecordProperty("identity_row_buffer_bytes", std::to_string(array.storage_bytes()));
    RecordProperty("two_value_row_buffer_bytes", std::to_string(8 * sizeof(D::ContactPointIncidence)));
}
TEST(ContactPointArray, FirstDifferentNormalizationPreservesEveryEarlierAndLaterValue)
{
    D::ContactPointArray array;
    array.reserve(4);
    const std::array<D::ContactPointIncidence, 4> input{
        {{point(0), point(0)}, {point(1), point(1)}, {point(0, 1), point(32, 1)}, {point(3), point(3)}}};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        array.push_back(input[i]);
        EXPECT_EQ(array.shares_normalized_storage(), i < 2);
        for (std::size_t j = 0; j <= i; ++j)
        {
            EXPECT_EQ(array[j].normalized, input[j].normalized);
            EXPECT_EQ(array[j].physical, input[j].physical);
            EXPECT_EQ(P::encode_identity(array[j].normalized), P::encode_identity(input[j].normalized));
            EXPECT_EQ(P::encode_identity(array[j].physical), P::encode_identity(input[j].physical));
        }
    }
    EXPECT_EQ(array.storage_bytes(), 4 * sizeof(D::ContactPointIncidence));
    ASSERT_TRUE(buffer_identity(array, D::BasicContactPayloadRole::PointArrayNormalized));
    EXPECT_NE(buffer_identity(array, D::BasicContactPayloadRole::PointArrayNormalized),
        buffer_identity(array, D::BasicContactPayloadRole::PointArrayPhysical));
}
TEST(ContactPointArray, CopiesAndMovesOwnCompleteArraysAfterSourceDestruction)
{
    D::ContactPointArray copied;
    std::vector<std::pair<D::ContactPointIdentity, D::ContactPointIdentity>> expected;
    {
        D::ContactPointArray source{{point(0), point(0)}, {point(1), point(33)}, {point(2), point(2)}};
        expected = values(source);
        copied = source;
        EXPECT_NE(buffer_identity(copied, D::BasicContactPayloadRole::PointArrayPhysical),
            buffer_identity(source, D::BasicContactPayloadRole::PointArrayPhysical));
        EXPECT_NE(buffer_identity(copied, D::BasicContactPayloadRole::PointArrayNormalized),
            buffer_identity(source, D::BasicContactPayloadRole::PointArrayNormalized));
        source.push_back({point(4), point(36)});
        EXPECT_EQ(values(copied), expected);
    }
    EXPECT_EQ(values(copied), expected);
    const auto* physical = buffer_identity(copied, D::BasicContactPayloadRole::PointArrayPhysical);
    const auto* normalized = buffer_identity(copied, D::BasicContactPayloadRole::PointArrayNormalized);
    D::ContactPointArray moved(std::move(copied));
    EXPECT_EQ(values(moved), expected);
    EXPECT_EQ(buffer_identity(moved, D::BasicContactPayloadRole::PointArrayPhysical), physical);
    EXPECT_EQ(buffer_identity(moved, D::BasicContactPayloadRole::PointArrayNormalized), normalized);
    moved.reserve(17);
    EXPECT_EQ(values(moved), expected);
    moved.push_back({point(5), point(37)});
    EXPECT_EQ(moved.size(), 4u);
    EXPECT_EQ(moved[3].normalized, point(5));
    EXPECT_EQ(moved[3].physical, point(37));
}
TEST(ContactPointArray, FullPrecisionAndCanonicalZeroDoNotChangePacketBytes)
{
    const auto x = std::nextafter(1.L, 2.L);
    const auto precise = point(x, -0.L, 0.L);
    const auto equal = point(x, 0.L, -0.L);
    ASSERT_EQ(precise, equal);
    ASSERT_EQ(P::encode_identity(precise), P::encode_identity(equal));
    D::ContactPointArray array{{precise, equal}};
    EXPECT_TRUE(array.shares_normalized_storage());
    EXPECT_EQ(P::encode_identity(array[0].normalized), P::encode_identity(precise));
    EXPECT_EQ(P::encode_identity(array[0].physical), P::encode_identity(equal));
    const auto different = point(1, 0, 0);
    ASSERT_NE(precise, different);
    array.push_back({different, precise});
    EXPECT_FALSE(array.shares_normalized_storage());
    EXPECT_EQ(P::encode_identity(array[1].normalized), P::encode_identity(different));
    EXPECT_EQ(P::encode_identity(array[1].physical), P::encode_identity(precise));
}

TEST(ContactPointArray, PointLookupPreservesEnvironmentAndFullValueIdentity)
{
    struct Environment
    {
        std::fenv_t saved;
        Environment() { std::fegetenv(&saved); }
        ~Environment() { std::fesetenv(&saved); }
    } environment;
    const auto a = point(1.1L, 2.3L, 4.7L);
    const auto b = point(std::nextafter(1.1L, 2.L), 2.3L, 4.7L);
    ASSERT_NE(a, b);
    D::Budget budget{1024 * 1024};
    D::IdentityBudget identities;
    D::IncidencePools pools(budget, identities);
    std::fesetround(FE_UPWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
    const auto first = pools.points.intern(a), second = pools.points.intern(b);
    EXPECT_NE(first, second);
    EXPECT_EQ(pools.points.intern(a), first);
    EXPECT_EQ(pools.points.find(b), second);
    EXPECT_EQ(pools.points.get(first), a);
    EXPECT_EQ(pools.points.get(second), b);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
}
} // namespace
