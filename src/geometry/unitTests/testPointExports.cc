/** @file testPointExports.cc
 * @brief Explicit finite point publication accounting, identity and lifetime contracts.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <algorithm>
#include <array>
#include <barrier>
#include <cfenv>
#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <thread>
#include <type_traits>

namespace
{
namespace P = SimpleFluid::PLIC;
using SimpleFluid::ArithmeticMode;
using Access = P::DynamicGeometryAccess;
constexpr std::size_t export_budget = 64 * 1024 * 1024;
template<ArithmeticMode Mode> using Geometry = P::ConvexCellGeometry<Mode>;
template<ArithmeticMode Mode> using GeometryPtr = P::ConvexGeometryPtr<Mode>;

template<class Handle>
concept HasLvaluePoints = requires(const Handle& handle) { handle.points(); };
template<class Handle>
concept HasRvaluePoints = requires(const Handle&& handle) { std::move(handle).points(); };
static_assert(HasLvaluePoints<P::DynamicPointExport> && !HasRvaluePoints<P::DynamicPointExport>);
static_assert(HasLvaluePoints<P::PointExport<ArithmeticMode::Ordinary>> &&
              !HasRvaluePoints<P::PointExport<ArithmeticMode::Ordinary>>);
static_assert(HasLvaluePoints<P::PointExport<ArithmeticMode::BoundedCertification>> &&
              !HasRvaluePoints<P::PointExport<ArithmeticMode::BoundedCertification>>);
static_assert(
    std::is_same_v<decltype(std::declval<const P::DynamicPointExport&>().points()), std::span<const P::DynamicPoint>>);
static_assert(std::is_same_v<decltype(std::declval<const P::PointExport<ArithmeticMode::Ordinary>&>().points()),
    std::span<const P::Point<ArithmeticMode::Ordinary>>>);
static_assert(noexcept(std::declval<const P::DynamicConvexGeometry&>().vertex_count()));

struct CacheObservation
{
    std::weak_ptr<const P::DynamicPointExportPayload> dynamic;
    std::weak_ptr<const P::PointExportPayload<ArithmeticMode::Ordinary>> ordinary;
    std::weak_ptr<const P::PointExportPayload<ArithmeticMode::BoundedCertification>> bounded;
    std::vector<std::weak_ptr<const P::DynamicPointProvenance>> vertices;
};
CacheObservation observe(const P::DynamicConvexGeometry& geometry)
{
    const auto& cache = Access::implementation(geometry)->point_export_cache;
    if (!cache)
        return {};
    const std::lock_guard lock(cache->mutex);
    return {cache->dynamic_root, cache->ordinary_root, cache->bounded_root, cache->vertex_provenance};
}
template<ArithmeticMode Mode> CacheObservation observe(const Geometry<Mode>& geometry)
{
    return observe(*geometry.dynamic_geometry());
}
template<class T> bool has_owner(const std::weak_ptr<T>& weak)
{
    const std::weak_ptr<T> empty;
    return weak.owner_before(empty) || empty.owner_before(weak);
}
void expect_cold(const CacheObservation& cache)
{
    EXPECT_FALSE(has_owner(cache.dynamic));
    EXPECT_FALSE(has_owner(cache.ordinary));
    EXPECT_FALSE(has_owner(cache.bounded));
    for (const auto& vertex : cache.vertices)
        EXPECT_FALSE(has_owner(vertex));
}
template<ArithmeticMode Mode> void same_points(std::span<const P::Point<Mode>> a, std::span<const P::Point<Mode>> b)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_EQ(a[i].value, b[i].value);
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            EXPECT_EQ(a[i].absolute_error, b[i].absolute_error);
        EXPECT_EQ(a[i].supports(), b[i].supports());
    }
}

template<ArithmeticMode Mode> void query_contract()
{
    const auto box = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    expect_cold(observe(*box));
    P::GeometryAllocationLedger geometry_ledger;
    std::size_t geometry_bytes = 0;
    geometry_ledger.retain(box, geometry_bytes, export_budget);
    const auto retained_bytes = geometry_ledger.payload_bytes();
    const auto retained_owners = geometry_ledger.owner_count();
    const auto context = P::GeometryPacketContext{{7, 11, 13, 17}};
    P::ConstructionRegistry<Mode> registry(context);
    registry.register_native(3, 0, box);
    const auto numerical = box->encode(context), authenticated = box->encode(registry);
    EXPECT_EQ(box->vertex_count(), 8u);
    EXPECT_EQ(box->dynamic_geometry()->vertex_count(), 8u);
    EXPECT_EQ(box->volume().value, 1);
    const auto moments = box->moments({0, 0, 0});
    for (const auto& first : moments.first_moment)
        EXPECT_EQ(first.value, .5L);
    EXPECT_EQ(box->centroid(), (P::GeometryPoint{.5, .5, .5}));
    const auto support = box->support({1, 0, 0}, {0, 0, 0});
    EXPECT_EQ(support.lower.value, 0);
    EXPECT_EQ(support.upper.value, 1);
    EXPECT_EQ(box->local_points().size(), 8u);
    const auto decoded = Geometry<Mode>::decode(authenticated, registry);
    EXPECT_EQ(decoded->volume().value, 1);
    expect_cold(observe(*decoded));
    expect_cold(observe(*box));
    P::GeometryAllocationLedger publication_ledger;
    std::size_t publication_bytes = 0;
    const auto publication = box->export_points(publication_ledger, publication_bytes, export_budget);
    ASSERT_EQ(publication.points().size(), 8u);
    EXPECT_GT(publication_bytes, 0u);
    EXPECT_EQ(box->encode(context), numerical);
    EXPECT_EQ(box->encode(registry), authenticated);
    geometry_ledger.retain(box, geometry_bytes, export_budget);
    EXPECT_EQ(geometry_ledger.payload_bytes(), retained_bytes);
    EXPECT_EQ(geometry_ledger.owner_count(), retained_owners);
}
TEST(PointExportsTest, OrdinaryQueriesAndPacketsLeaveGlobalPublicationCold)
{
    query_contract<ArithmeticMode::Ordinary>();
}
TEST(PointExportsTest, BoundedQueriesAndPacketsLeaveGlobalPublicationCold)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    query_contract<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode, bool Typed> void admission_contract()
{
    const auto measured = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    std::size_t cost = 0;
    {
        P::GeometryAllocationLedger ledger;
        const auto publication = [&]
        {
            if constexpr (Typed)
                return measured->export_points(ledger, cost, export_budget);
            else
                return measured->dynamic_geometry()->export_points(ledger, cost, export_budget);
        }();
        ASSERT_EQ(publication.points().size(), 8u);
        const auto charged = cost;
        const auto repeated = [&]
        {
            if constexpr (Typed)
                return measured->export_points(ledger, cost, export_budget);
            else
                return measured->dynamic_geometry()->export_points(ledger, cost, export_budget);
        }();
        EXPECT_EQ(repeated.points().data(), publication.points().data());
        EXPECT_EQ(cost, charged);
    }
    ASSERT_GT(cost, 1u);
    const auto exact = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger exact_ledger;
    std::size_t exact_bytes = 0;
    if constexpr (Typed)
        EXPECT_NO_THROW((void) exact->export_points(exact_ledger, exact_bytes, cost));
    else
        EXPECT_NO_THROW((void) exact->dynamic_geometry()->export_points(exact_ledger, exact_bytes, cost));
    EXPECT_EQ(exact_bytes, cost);
    const auto rejected = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger failed_ledger;
    std::size_t failed_bytes = 0;
    if constexpr (Typed)
    {
        EXPECT_THROW((void) rejected->export_points(failed_ledger, failed_bytes, cost - 1), P::GeometryError);
        expect_cold(observe(*rejected));
        EXPECT_THROW((void) rejected->export_points(failed_ledger, failed_bytes, export_budget), P::GeometryError);
    }
    else
    {
        EXPECT_THROW((void) rejected->dynamic_geometry()->export_points(failed_ledger, failed_bytes, cost - 1),
            P::GeometryError);
        expect_cold(observe(*rejected));
        EXPECT_THROW((void) rejected->dynamic_geometry()->export_points(failed_ledger, failed_bytes, export_budget),
            P::GeometryError);
    }
    expect_cold(observe(*rejected));
    if constexpr (Typed)
    {
        const auto geometry = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        const auto dynamic = geometry->dynamic_geometry()->export_points(ledger, used, export_budget);
        const auto dynamic_cost = used;
        const auto typed = geometry->export_points(ledger, used, export_budget);
        EXPECT_GE(used - dynamic_cost, sizeof(P::PointExportPayload<Mode>) + 8 * sizeof(P::Point<Mode>));
        ASSERT_EQ(dynamic.points().size(), typed.points().size());
        for (std::size_t i = 0; i < typed.points().size(); ++i)
            EXPECT_EQ(Access::identity(dynamic.points()[i]), Access::identity(typed.points()[i].dynamic()));
    }
}
TEST(PointExportsTest, OrdinaryDynamicAndTypedAdmissionHasExactCapsAndPoisonedFailure)
{
    admission_contract<ArithmeticMode::Ordinary, false>();
    admission_contract<ArithmeticMode::Ordinary, true>();
}
TEST(PointExportsTest, BoundedDynamicAndTypedAdmissionHasExactCapsAndPoisonedFailure)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    admission_contract<ArithmeticMode::BoundedCertification, false>();
    admission_contract<ArithmeticMode::BoundedCertification, true>();
}

template<ArithmeticMode Mode> void concurrent_contract()
{
    const auto box = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    const auto alias = Geometry<Mode>::from_dynamic(box->dynamic_geometry());
    std::array<P::GeometryAllocationLedger, 2> ledgers;
    std::array<std::size_t, 2> used{};
    std::array<std::optional<P::PointExport<Mode>>, 2> publications;
    std::array<std::exception_ptr, 2> errors{};
    std::array<bool, 2> environments{};
    std::barrier start(2);
    const auto worker = [&](std::size_t index)
    {
        try
        {
            const auto rounding = std::fesetround(FE_TONEAREST);
            environments[index] = rounding == 0 && P::certification_environment();
            start.arrive_and_wait();
            if constexpr (Mode == ArithmeticMode::BoundedCertification)
                if (!environments[index])
                    throw std::runtime_error("Worker lacks bounded FP environment.");
            publications[index] = (index == 0 ? box : alias)->export_points(ledgers[index], used[index], export_budget);
        }
        catch (...)
        {
            errors[index] = std::current_exception();
        }
    };
    std::thread first(worker, 0), second(worker, 1);
    first.join();
    second.join();
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_FALSE(errors[i]);
        ASSERT_TRUE(publications[i]);
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            EXPECT_TRUE(environments[i]);
        EXPECT_GT(used[i], 0u);
    }
    EXPECT_EQ(used[0], used[1]);
    EXPECT_EQ(publications[0]->points().data(), publications[1]->points().data());
    for (std::size_t i = 0; i < box->vertex_count(); ++i)
        EXPECT_EQ(Access::identity(publications[0]->points()[i].dynamic()),
            Access::identity(publications[1]->points()[i].dynamic()));
}
TEST(PointExportsTest, OrdinaryAliasesAndConcurrentFirstExportsSharePublication)
{
    concurrent_contract<ArithmeticMode::Ordinary>();
}
TEST(PointExportsTest, BoundedAliasesAndConcurrentFirstExportsSharePublication)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    concurrent_contract<ArithmeticMode::BoundedCertification>();
}

TEST(PointExportsTest, CopiedUncertainPointsSurvivePayloadEvictionAndReuseProvenance)
{
    constexpr auto mode = ArithmeticMode::BoundedCertification;
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    const auto box = Geometry<mode>::box({0, 0, 0}, {1, 1, 1});
    const auto cut = box->clip({{3, 0, 0}, {0, 0, 0}, 1}, false).inside;
    ASSERT_TRUE(cut);
    std::optional<P::Point<mode>> survivor;
    std::size_t survivor_index = 0;
    CacheObservation first;
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        const auto publication = cut->export_points(ledger, used, export_budget);
        const auto points = publication.points();
        const auto uncertain = std::find_if(points.begin(), points.end(),
            [](const auto& point) { return point.absolute_error[0] > 0 && point.value[0] > 0; });
        ASSERT_NE(uncertain, points.end());
        survivor_index = static_cast<std::size_t>(uncertain - points.begin());
        survivor = *uncertain;
        ASSERT_GT(survivor->absolute_error[0], 0);
        first = observe(*cut);
        ASSERT_EQ(first.vertices.size(), cut->vertex_count());
    }
    EXPECT_TRUE(first.dynamic.expired());
    EXPECT_TRUE(first.bounded.expired());
    for (std::size_t i = 0; i < first.vertices.size(); ++i)
        EXPECT_EQ(first.vertices[i].expired(), i != survivor_index);
    const auto live_identity = Access::identity(survivor->dynamic());
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto regenerated = cut->export_points(ledger, used, export_budget);
    const auto& point = regenerated.points()[survivor_index];
    EXPECT_EQ(Access::identity(point.dynamic()), live_identity);
    EXPECT_EQ(point.value, survivor->value);
    EXPECT_EQ(point.absolute_error, survivor->absolute_error);
    EXPECT_EQ(point.supports(), survivor->supports());
    EXPECT_TRUE(P::points_identical<mode>(point, *survivor));
}

template<ArithmeticMode Mode> void ledger_lifetime_contract()
{
    const auto box = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    CacheObservation cache;
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        {
            const auto publication = box->export_points(ledger, used, export_budget);
            cache = observe(*box);
        }
        EXPECT_FALSE(cache.dynamic.expired());
        if constexpr (Mode == ArithmeticMode::Ordinary)
            EXPECT_FALSE(cache.ordinary.expired());
        else
            EXPECT_FALSE(cache.bounded.expired());
    }
    EXPECT_TRUE(cache.dynamic.expired());
    EXPECT_TRUE(cache.ordinary.expired());
    EXPECT_TRUE(cache.bounded.expired());
    for (const auto& vertex : cache.vertices)
        EXPECT_TRUE(vertex.expired());
}
TEST(PointExportsTest, OrdinaryLedgerPinsPublicationUntilTransactionDies)
{
    ledger_lifetime_contract<ArithmeticMode::Ordinary>();
}
TEST(PointExportsTest, BoundedLedgerPinsPublicationUntilTransactionDies)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    ledger_lifetime_contract<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void transformations_contract()
{
    const auto box = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto original = box->export_points(ledger, used, export_budget);
    const auto rebased = box->clip({{1, 0, 0}, {4, 0, 0}, 0}, false).inside;
    ASSERT_TRUE(rebased);
    const auto& before = Access::implementation(*box->dynamic_geometry());
    const auto& after = Access::implementation(*rebased->dynamic_geometry());
    EXPECT_TRUE(before->frame.origin != after->frame.origin || before->frame.exponent != after->frame.exponent);
    EXPECT_EQ(before->point_export_cache, after->point_export_cache);
    const auto reexported = rebased->export_points(ledger, used, export_budget);
    EXPECT_EQ(original.points().data(), reexported.points().data());
    const std::array donors{box};
    const auto pieces = P::ConvexPartition<Mode>(donors).partition(*box);
    ASSERT_EQ(pieces.size(), 1u);
    const auto& certified = Access::implementation(*pieces[0].geometry->dynamic_geometry());
    ASSERT_TRUE(certified->partition_certificate);
    EXPECT_EQ(before->point_export_cache, certified->point_export_cache);
    const auto certificate_export = pieces[0].geometry->export_points(ledger, used, export_budget);
    EXPECT_EQ(original.points().data(), certificate_export.points().data());
    const auto shifted = box->translated({2, 0, 0});
    EXPECT_NE(before->point_export_cache, Access::implementation(*shifted->dynamic_geometry())->point_export_cache);
    const auto shifted_export = shifted->export_points(ledger, used, export_budget);
    ASSERT_EQ(shifted_export.points().size(), original.points().size());
    EXPECT_NE(shifted_export.points().data(), original.points().data());
    for (std::size_t i = 0; i < original.points().size(); ++i)
    {
        EXPECT_EQ(shifted_export.points()[i].value[0], original.points()[i].value[0] + 2);
        EXPECT_EQ(shifted_export.points()[i].value[1], original.points()[i].value[1]);
        EXPECT_EQ(shifted_export.points()[i].value[2], original.points()[i].value[2]);
        EXPECT_NE(
            Access::identity(shifted_export.points()[i].dynamic()), Access::identity(original.points()[i].dynamic()));
        EXPECT_FALSE(P::points_identical<Mode>(shifted_export.points()[i], original.points()[i]));
    }
}
TEST(PointExportsTest, OrdinaryRebaseAndCertificateCopiesShareButTranslationDetaches)
{
    transformations_contract<ArithmeticMode::Ordinary>();
}
TEST(PointExportsTest, BoundedRebaseAndCertificateCopiesShareButTranslationDetaches)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    transformations_contract<ArithmeticMode::BoundedCertification>();
}

struct RestoreEnvironment
{
    std::fenv_t saved{};
    ~RestoreEnvironment() { (void) std::fesetenv(&saved); }
};

TEST(PointExportsTest, OrdinaryColdWarmAndEvictedExportsUseConstructionRoundingAndRestoreCaller)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    RestoreEnvironment environment;
    ASSERT_EQ(std::fegetenv(&environment.saved), 0);
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    // Adding this half-ulp displacement to 1 rounds to 1 under ties-to-even,
    // while upward rounding would produce the next representable value.
    const auto half_ulp = std::ldexp(1.L, -std::numeric_limits<long double>::digits);
    std::array<P::Point<mode>, 8> vertices;
    for (std::size_t i = 0; i < vertices.size(); ++i)
        vertices[i] = P::Point<mode>({i & 1 ? .5L : half_ulp, i & 2 ? 1.L : 0.L, i & 4 ? 1.L : 0.L});
    const std::vector<std::vector<std::size_t>> faces{
        {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    const auto reference = Geometry<mode>::make_local({1, 0, 0}, vertices, faces);
    const auto delayed = Geometry<mode>::make_local({1, 0, 0}, vertices, faces);
    P::GeometryAllocationLedger reference_ledger;
    std::size_t reference_bytes = 0;
    const auto eager_compatible = reference->export_points(reference_ledger, reference_bytes, export_budget);
    for (std::size_t i = 0; i < vertices.size(); ++i)
        EXPECT_EQ(eager_compatible.points()[i].value[0], i & 1 ? 1.5L : 1.L);
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    CacheObservation evicted;
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        const auto cold = delayed->export_points(ledger, used, export_budget);
        EXPECT_EQ(std::fegetround(), FE_UPWARD);
        same_points<mode>(eager_compatible.points(), cold.points());
        const auto warm = delayed->export_points(ledger, used, export_budget);
        EXPECT_EQ(std::fegetround(), FE_UPWARD);
        EXPECT_EQ(cold.points().data(), warm.points().data());
        evicted = observe(*delayed);
    }
    ASSERT_TRUE(evicted.dynamic.expired());
    ASSERT_TRUE(evicted.ordinary.expired());
    P::GeometryAllocationLedger regenerated_ledger;
    std::size_t regenerated_bytes = 0;
    const auto regenerated = delayed->export_points(regenerated_ledger, regenerated_bytes, export_budget);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    same_points<mode>(eager_compatible.points(), regenerated.points());
}

TEST(PointExportsTest, BoundedWarmExportRejectsUnsupportedCallerRounding)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    RestoreEnvironment environment;
    ASSERT_EQ(std::fegetenv(&environment.saved), 0);
    const auto box = Geometry<ArithmeticMode::BoundedCertification>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto warm = box->export_points(ledger, used, export_budget);
    ASSERT_EQ(warm.points().size(), 8u);
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    P::GeometryAllocationLedger unsupported_ledger;
    std::size_t unsupported_bytes = 0;
    EXPECT_THROW((void) box->export_points(unsupported_ledger, unsupported_bytes, export_budget), std::invalid_argument);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
}

TEST(PointExportsTest, LargeAnchorPureRebasePreservesNativeFaceCoordinatesAndBounds)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const auto box = Geometry<mode>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto original = box->export_points(ledger, used, export_budget);
    const auto original_points = original.points();
    ASSERT_EQ(original_points.size(), 8u);
    const auto extent = std::minmax_element(original_points.begin(), original_points.end(),
        [](const auto& a, const auto& b) { return a.value[0] < b.value[0]; });
    EXPECT_EQ(extent.first->value[0], 0);
    EXPECT_EQ(extent.second->value[0], 1);
    std::size_t upper_face = box->faces().size();
    for (std::size_t face = 0; face < box->faces().size(); ++face)
        if (std::ranges::all_of(
                box->faces()[face], [&](std::size_t vertex) { return original_points[vertex].value[0] == 1; }))
            upper_face = face;
    ASSERT_LT(upper_face, box->faces().size());
    const auto original_face = box->face_geometry(upper_face);
    const auto clipped = box->clip({{1, 0, 0}, {0x1p65, 0, 0}, 0}, false);
    ASSERT_TRUE(clipped.inside);
    ASSERT_FALSE(clipped.outside);
    const auto& rebased = clipped.inside;
    const auto stored =
        P::details::checked_geometry_storage<mode>(Access::implementation(*rebased->dynamic_geometry()));
    ASSERT_FALSE(stored->vertices.empty());
    // At this anchor, finite current-frame x values cannot retain the unit span.
    // Original physical coordinates remain the public/native reference.
    for (const auto& vertex : stored->vertices)
        EXPECT_EQ(vertex.coordinates[0].value, stored->vertices.front().coordinates[0].value);
    EXPECT_TRUE(original_face->cell_bounds_overlap(*rebased));
    const auto rebased_face = rebased->face_geometry(upper_face);
    EXPECT_EQ(rebased_face->points(), original_face->points());
    EXPECT_TRUE(rebased_face->cell_bounds_overlap(*box));
    const auto reexported = rebased->export_points(ledger, used, export_budget);
    EXPECT_EQ(original_points.data(), reexported.points().data());
    same_points<mode>(original_points, reexported.points());
    for (std::size_t face = 0; face < box->faces().size(); ++face)
        EXPECT_EQ(rebased->face_geometry(face)->points(), box->face_geometry(face)->points());
}

TEST(PointExportsTest, ColdAndWarmThinRebasesKeepCoordinatesErrorsSupportsAndPackets)
{
    constexpr auto mode = ArithmeticMode::BoundedCertification;
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded geometry requires its supported FP environment.";
    const P::GeometryPoint lower{16, -8, 4}, upper{17, -7, 5};
    const auto cold = Geometry<mode>::box(lower, upper), warm = Geometry<mode>::box(lower, upper);
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto warm_export = warm->export_points(ledger, used, export_budget);
    const P::GeometryPacketContext context{{19, 23, 29, 31}};
    P::ConstructionRegistry<mode> cold_registry(context), warm_registry(context);
    cold_registry.register_native(1, 0, cold);
    warm_registry.register_native(1, 0, warm);
    const double thickness = 1e-6;
    for (const bool reverse : {false, true})
    {
        const P::GeometryHalfspace plane{
            {reverse ? -1. : 1., 0, 0}, {upper[0], lower[1], lower[2]}, reverse ? thickness : -thickness};
        cold_registry.register_plane(reverse ? 3 : 2, plane);
        warm_registry.register_plane(reverse ? 3 : 2, plane);
        const auto cold_split = cold->clip(plane, false), warm_split = warm->clip(plane, false);
        const auto a = reverse ? cold_split.inside : cold_split.outside;
        const auto b = reverse ? warm_split.inside : warm_split.outside;
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        const auto expected = static_cast<long double>(thickness);
        for (const auto& thin : {a, b})
        {
            const auto volume = thin->volume();
            EXPECT_LE(volume.lower(), expected);
            EXPECT_GE(volume.upper(), expected);
            EXPECT_LE(std::abs(volume.value - expected) + volume.absolute_error, expected * 8e-15L);
            EXPECT_EQ(thin->origin(), cold->origin());
        }
        const auto a_export = a->export_points(ledger, used, export_budget);
        const auto b_export = b->export_points(ledger, used, export_budget);
        same_points<mode>(a_export.points(), b_export.points());
        EXPECT_EQ(a->encode(context), b->encode(context));
        const auto packet = a->encode(cold_registry);
        EXPECT_EQ(packet, b->encode(warm_registry));
        const auto replay = Geometry<mode>::decode(packet, cold_registry);
        const auto replay_export = replay->export_points(ledger, used, export_budget);
        same_points<mode>(a_export.points(), replay_export.points());
        EXPECT_EQ(replay->encode(cold_registry), packet);
    }
}
} // namespace
