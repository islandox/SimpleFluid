#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/ContactPointArray.hh"
#include "geometry/mesh/AMR/IncidencePointDirectory.hh"
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
namespace
{
using namespace cartesian_incidence_test;
struct ConstantPointHash
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const ContactSourcePoint&) const noexcept { return 7; }
};
TEST(IncidencePointDirectory, ValueReadsAndCollisionLookupKeepFullPointIdentity)
{
    Budget budget{ample};
    {
        IncidencePointDirectory<ConstantPointHash> values(budget);
        EXPECT_EQ(values.size(), 0);
        for (unsigned i = 0; i != 257; ++i)
            EXPECT_EQ(values.intern(point(i, i % 3, i % 7)), i);
        for (unsigned i = 0; i != 257; ++i)
        {
            const auto saved = values.get(i);
            EXPECT_EQ(saved, point(i, i % 3, i % 7));
            EXPECT_EQ(values.find(saved), i);
        }
        EXPECT_FALSE(values.find(point(257)));
        EXPECT_EQ(values.intern(point(128, 2, 2)), 128); // 128%3==2, 128%7==2.
        EXPECT_EQ(values.source_size(), 0);
    }
    EXPECT_EQ(budget.used, 0);
}
TEST(IncidencePointDirectory, ForkPreservesPartialPhysicalPagesAndEscapingCapturedRows)
{
    auto origin = std::make_shared<Budget>(Budget{ample});
    auto candidate = std::make_shared<Budget>(Budget{ample});
    std::shared_ptr<const ContactPointPageView> escaped;
    CapturedPointReference first{}, seventeenth{};
    {
        IncidencePointDirectory<> values(*origin, origin);
        for (unsigned i = 0; i != 17; ++i)
            EXPECT_EQ(values.intern(point(i)), i);
        auto view = values.page_view();
        first = view->capture(values.physical_pool(), 0);
        seventeenth = view->capture(values.physical_pool(), 16);
        view->freeze();
        escaped = view;
        EXPECT_EQ(values.page_count(), 2);
        {
            IncidencePointDirectory<> fork(values, *candidate, candidate);
            EXPECT_EQ(fork.intern(point(17)), 17);
            EXPECT_EQ(fork.page_count(), 3); // A shared partial tail is immutable.
            EXPECT_EQ(values.size(), 17);
            EXPECT_EQ(view->get(first), point(0));
            EXPECT_EQ(view->get(seventeenth), point(16));
        }
        EXPECT_EQ(candidate->used, 0);
        Budget rejected{0};
        EXPECT_THROW((IncidencePointDirectory<>(values, rejected)), std::exception);
        EXPECT_EQ(rejected.used, 0);
        EXPECT_EQ(view->get(seventeenth), point(16));
    }
    EXPECT_GT(origin->used, 0);
    EXPECT_EQ(escaped->get(first), point(0));
    EXPECT_EQ(escaped->get(seventeenth), point(16));
    escaped.reset();
    EXPECT_EQ(origin->used, 0);
}
TEST(IncidencePointDirectory, CanonicalComponentsReassembleExactIssuedIdentityWithoutReevaluation)
{
    using Access = PLIC::details::GeometryIdentityAccess;
    RestoreEnvironment restore;
    const auto original = point(0.1L, -0.7L, 2.3L);
    const std::array components{Access::point_component(original, 0), Access::point_component(original, 1),
        Access::point_component(original, 2)};
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
    const auto restored = Access::point_components(components);
    EXPECT_EQ(restored, original);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
}
} // namespace
