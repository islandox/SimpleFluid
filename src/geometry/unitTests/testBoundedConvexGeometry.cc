/** @file testBoundedConvexGeometry.cc
 * @brief Independent analytic/reference tests of bounded finite geometry.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <type_traits>

namespace
{
using namespace SimpleFluid::Meshes;
using SimpleFluid::ArithmeticMode;
using Faces = std::vector<std::vector<std::size_t>>;
SimpleFluid::PLIC::GeometryLimits geometry_limits(ArithmeticMode mode)
{
    SimpleFluid::PLIC::GeometryLimits limits;
    limits.arithmetic_mode = mode;
    return limits;
}
void encloses(const SimpleFluid::PLIC::DynamicGeometryValue& value, long double oracle)
{
    EXPECT_LE(value.lower(), oracle);
    EXPECT_GE(value.upper(), oracle);
    EXPECT_GE(value.absolute_error, 0);
}
SimpleFluid::PLIC::DynamicConvexGeometryPtr cube()
{
    return SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1});
}

template<class Function> void expect_geometry_error(Function&& operation, const std::string& expected)
{
    try
    {
        operation();
        FAIL() << "Expected geometry rejection: " << expected;
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_EQ(error.what(), expected);
    }
}

TEST(BoundedConvexGeometryTest, DiagnosticsIdentifyModeBeforeAndDuringOperations)
{
    namespace detail = SimpleFluid::PLIC::details;
    ASSERT_EQ(detail::active_work, nullptr);
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const std::string prefix =
            mode == ArithmeticMode::Ordinary ? "Ordinary FP geometry: " : "Bounded FP geometry: ";
        auto limits = geometry_limits(mode);
        limits.maximum_vertices = std::numeric_limits<std::size_t>::max();
        expect_geometry_error([&] { detail::Work work(limits); }, prefix + "derived scalar-work guard overflow.");
        EXPECT_EQ(detail::active_work, nullptr);

        if (mode == ArithmeticMode::BoundedCertification && !SimpleFluid::PLIC::certification_environment())
            continue;
        limits = geometry_limits(mode);
        limits.byte_budget = 0;
        expect_geometry_error([&] { detail::Operation scope(limits); }, prefix + "zero resource cap.");
        EXPECT_EQ(detail::active_work, nullptr);

        limits.byte_budget = 1;
        expect_geometry_error([&]
            { (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits); },
            prefix + "construction storage cap exhausted.");
        EXPECT_EQ(detail::active_work, nullptr);
    }
    expect_geometry_error(
        [] { detail::reject("no active geometry operation."); }, "Finite FP geometry: no active geometry operation.");
}

TEST(BoundedConvexGeometryTest, OrdinaryDiagnosticsRetainModeOutsideOperationScope)
{
    namespace detail = SimpleFluid::PLIC::details;
    ASSERT_EQ(detail::active_work, nullptr);
    const auto box =
        SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, geometry_limits(ArithmeticMode::Ordinary));
    const auto volume = box->volume();
    expect_geometry_error(
        [&] { (void) volume.lower(); }, "Ordinary FP geometry: ordinary values do not carry certified bounds.");
    using Scalar = detail::ordinary_kernel::Scalar;
    expect_geometry_error([] { (void) Scalar(std::numeric_limits<long double>::infinity()); },
        "Ordinary FP geometry: nonfinite arithmetic result.");
    EXPECT_EQ(detail::active_work, nullptr);
}

TEST(BoundedConvexGeometryTest, OrdinaryNonpositiveVolumeDiagnosticReportsComputedValue)
{
    namespace detail = SimpleFluid::PLIC::details;
    namespace ordinary = detail::ordinary_kernel;
    // Exercise finalization directly: earlier import validation can reject
    // these collapsed/inverted shells before the volume gate is reached.
    for (const long double height : {0.L, -1.L, 1.L})
    {
        ordinary::StoredGeometry poly;
        poly.vertices.resize(4);
        poly.vertices[1].coordinates[0] = ordinary::StorageScalar(1);
        poly.vertices[2].coordinates[1] = ordinary::StorageScalar(1);
        poly.vertices[3].coordinates[2] = ordinary::StorageScalar(height);
        poly.faces = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
        detail::Operation scope(poly.limits);
        if (height > 0)
        {
            EXPECT_NO_THROW(ordinary::finish<ArithmeticMode::Ordinary>(poly));
            EXPECT_GT(poly.volume.value, 0);
            continue;
        }
        try
        {
            ordinary::finish<ArithmeticMode::Ordinary>(poly);
            FAIL() << "Nonpositive scalar volume must still reject.";
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            const std::string prefix = "Ordinary FP geometry: computed signed volume is not positive (volume=";
            ASSERT_TRUE(message.starts_with(prefix)) << message;
            EXPECT_EQ(std::stold(message.substr(prefix.size())), poly.volume.value);
            EXPECT_LE(poly.volume.value, 0);
        }
    }
}

template<ArithmeticMode Mode> void fixed_policy_scalar_contract()
{
    namespace arithmetic = SimpleFluid::PLIC::details;
    using Policy = arithmetic::ArithmeticPolicy<Mode>;
    using Number = std::conditional_t<Mode == ArithmeticMode::Ordinary,
        SimpleFluid::detail::ordinary_fp::Scalar<Policy>, SimpleFluid::detail::directed_fp::Interval<Policy>>;
    using Storage = arithmetic::StoredValue<Mode>;
    auto limits = geometry_limits(Mode);
    limits.maximum_operations = 1;
    arithmetic::Operation scope(limits);
    Storage stored(2.75L);
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
    {
        stored = Storage(2, 4);
        stored.estimate = 2.75L;
    }
    const Number rebound(stored);
    const Storage roundtrip(rebound);
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        EXPECT_EQ(rebound.value, stored.value);
        EXPECT_EQ(roundtrip.value, stored.value);
    }
    else
    {
        EXPECT_EQ(rebound.lo(), stored.lo());
        EXPECT_EQ(rebound.hi(), stored.hi());
        EXPECT_EQ(rebound.estimate, stored.estimate);
        EXPECT_EQ(roundtrip.lo(), stored.lo());
        EXPECT_EQ(roundtrip.hi(), stored.hi());
        EXPECT_EQ(roundtrip.estimate, stored.estimate);
    }
    EXPECT_EQ(scope.work.operations, 0u);
    const auto sum = Number(1) + Number(2);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        EXPECT_EQ(sum.value, 3);
    else
    {
        EXPECT_EQ(sum.lo(), 3);
        EXPECT_EQ(sum.hi(), 3);
        EXPECT_EQ(sum.estimate, 3);
    }
    EXPECT_EQ(scope.work.operations, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().interval_operations,
        Mode == ArithmeticMode::BoundedCertification ? 1u : 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().floating_operations,
        Mode == ArithmeticMode::BoundedCertification ? 2u : 1u);
    EXPECT_THROW((void) (sum + Number(1)), std::invalid_argument);
    EXPECT_EQ(scope.work.operations, 2u);
}

TEST(BoundedConvexGeometryTest, FixedArithmeticPoliciesPreserveTypedStorageAndScalarCaps)
{
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    SimpleFluid::PLIC::reset_geometry_statistics();
    fixed_policy_scalar_contract<ArithmeticMode::Ordinary>();
    SimpleFluid::PLIC::reset_geometry_statistics();
    fixed_policy_scalar_contract<ArithmeticMode::BoundedCertification>();

    using Ordinary = SimpleFluid::PLIC::details::ArithmeticPolicy<ArithmeticMode::Ordinary>;
    using Certified = SimpleFluid::PLIC::details::ArithmeticPolicy<ArithmeticMode::BoundedCertification>;
    const auto subnormal = std::numeric_limits<long double>::denorm_min();
    ASSERT_GT(subnormal, 0);
    EXPECT_NO_THROW(Ordinary::finite(subnormal));
    EXPECT_THROW(Certified::finite(subnormal), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, OrdinaryStorageRetainsOneScalarAcrossGeometryAndPlanarRecords)
{
    namespace arithmetic = SimpleFluid::PLIC::details;
    constexpr auto ordinary_mode = ArithmeticMode::Ordinary;
    constexpr auto certified_mode = ArithmeticMode::BoundedCertification;
    using Ordinary = arithmetic::StoredValue<ordinary_mode>;
    using Certified = arithmetic::StoredValue<certified_mode>;
    static_assert(sizeof(Ordinary) == sizeof(long double));
    static_assert(sizeof(Certified) == 3 * sizeof(long double));
    static_assert(!std::is_constructible_v<Ordinary, Certified>);
    static_assert(!std::is_constructible_v<Certified, Ordinary>);
    static_assert(sizeof(arithmetic::StoredPoint<ordinary_mode>) == 3 * sizeof(long double));
    static_assert(sizeof(arithmetic::StoredPoint<certified_mode>) == 9 * sizeof(long double));
    static_assert(sizeof(arithmetic::StoredVertex<ordinary_mode>) < sizeof(arithmetic::StoredVertex<certified_mode>));
    static_assert(sizeof(arithmetic::StoredPlane<ordinary_mode>) < sizeof(arithmetic::StoredPlane<certified_mode>));
    static_assert(
        sizeof(arithmetic::StoredIntegrals<ordinary_mode>) < sizeof(arithmetic::StoredIntegrals<certified_mode>));
    static_assert(
        sizeof(arithmetic::StoredPlanarVertex<ordinary_mode>) < sizeof(arithmetic::StoredPlanarVertex<certified_mode>));
    static_assert(
        sizeof(arithmetic::GeometryStorage<ordinary_mode>) < sizeof(arithmetic::GeometryStorage<certified_mode>));
    static_assert(!std::is_constructible_v<Ordinary, long double, long double>);
    const auto ordinary_storage = std::make_shared<arithmetic::GeometryStorage<ordinary_mode>>();
    const auto certified_storage = std::make_shared<arithmetic::GeometryStorage<certified_mode>>();
    const std::shared_ptr<const SimpleFluid::PLIC::DynamicConvexGeometry::Impl> ordinary_base = ordinary_storage;
    const std::shared_ptr<const SimpleFluid::PLIC::DynamicConvexGeometry::Impl> certified_base = certified_storage;
    EXPECT_EQ(arithmetic::checked_geometry_storage<ordinary_mode>(ordinary_base).get(), ordinary_storage.get());
    EXPECT_EQ(arithmetic::checked_geometry_storage<certified_mode>(certified_base).get(), certified_storage.get());
    EXPECT_THROW((void) arithmetic::checked_geometry_storage<certified_mode>(ordinary_base), std::invalid_argument);
    EXPECT_THROW((void) arithmetic::checked_geometry_storage<ordinary_mode>(certified_base), std::invalid_argument);

    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    const auto ordinary =
        SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, geometry_limits(ordinary_mode));
    const auto certified = cube();
    EXPECT_LT(ordinary->storage_bytes(), certified->storage_bytes());
    EXPECT_EQ(ordinary->volume().value, 1);
    encloses(certified->volume(), 1);
    const SimpleFluid::PLIC::GeometryPacketContext context{{127, 131, 137, 139}};
    const std::array registry{ordinary};
    const auto restored = SimpleFluid::PLIC::DynamicConvexGeometry::decode(
        ordinary->encode(context), context, registry, geometry_limits(ordinary_mode));
    EXPECT_EQ(restored->volume().value, ordinary->volume().value);
    EXPECT_EQ(restored->arithmetic_mode(), ordinary_mode);
    const auto moments = restored->moments();
    for (const auto& component : moments.first_moment)
        EXPECT_EQ(component.value, .5L);
}

TEST(BoundedConvexGeometryTest, FixedModeValueQueriesMatchRuntimeResultsAndCounters)
{
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    const SimpleFluid::PLIC::DynamicGeometryValue certified{1.25L, .01L, ArithmeticMode::BoundedCertification};
    const SimpleFluid::PLIC::DynamicGeometryValue ordinary{1.25L, 3, ArithmeticMode::Ordinary};
    const auto compare = [](auto fixed_query, auto runtime_query)
    {
        SimpleFluid::PLIC::reset_geometry_statistics();
        const auto expected = runtime_query();
        const auto reference = SimpleFluid::PLIC::geometry_statistics();
        SimpleFluid::PLIC::reset_geometry_statistics();
        EXPECT_EQ(fixed_query(), expected);
        const auto observed = SimpleFluid::PLIC::geometry_statistics();
        EXPECT_EQ(observed.floating_operations, reference.floating_operations);
        EXPECT_EQ(observed.interval_operations, reference.interval_operations);
        EXPECT_EQ(observed.wider_routes, reference.wider_routes);
        EXPECT_EQ(observed.rejections, reference.rejections);
        EXPECT_EQ(observed.ledger_operations, reference.ledger_operations);
    };
    compare([&] { return certified.lower<ArithmeticMode::BoundedCertification>(); }, [&] { return certified.lower(); });
    compare([&] { return certified.upper<ArithmeticMode::BoundedCertification>(); }, [&] { return certified.upper(); });
    compare([&] { return certified.as_double<ArithmeticMode::BoundedCertification>(); },
        [&] { return certified.as_double(); });
    compare([&] { return ordinary.as_double<ArithmeticMode::Ordinary>(); }, [&] { return ordinary.as_double(); });

    // Fixed Ordinary queries retain the public contract: they do not turn an
    // untracked nominal value into a certificate, including a zero radius.
    EXPECT_THROW((void) ordinary.lower<ArithmeticMode::Ordinary>(), std::invalid_argument);
    EXPECT_THROW((void) ordinary.upper<ArithmeticMode::Ordinary>(), std::invalid_argument);
    EXPECT_THROW((void) ordinary.lower(), std::invalid_argument);
    EXPECT_THROW((void) ordinary.upper(), std::invalid_argument);
}

template<ArithmeticMode Mode> void fixed_value_publication_range_contract()
{
    const auto double_subnormal = static_cast<long double>(std::numeric_limits<double>::denorm_min());
    const SimpleFluid::PLIC::DynamicGeometryValue representable{double_subnormal, 0, Mode};
    EXPECT_EQ(representable.template as_double<Mode>(), representable.as_double());
    EXPECT_EQ(representable.template as_double<Mode>(), std::numeric_limits<double>::denorm_min());
    const std::array rejected{std::numeric_limits<long double>::infinity(), double_subnormal / 2,
        static_cast<long double>(std::numeric_limits<double>::max()) * 2,
        std::numeric_limits<long double>::denorm_min()};
    for (const auto value : rejected)
    {
        const SimpleFluid::PLIC::DynamicGeometryValue input{value, 0, Mode};
        EXPECT_THROW((void) input.template as_double<Mode>(), std::invalid_argument);
        EXPECT_THROW((void) input.as_double(), std::invalid_argument);
    }
}

TEST(BoundedConvexGeometryTest, FixedModeValueQueriesRejectForeignTagsAndPreserveRangeGates)
{
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    const SimpleFluid::PLIC::DynamicGeometryValue certified{1, 0, ArithmeticMode::BoundedCertification};
    const SimpleFluid::PLIC::DynamicGeometryValue ordinary{1, 0, ArithmeticMode::Ordinary};
    EXPECT_THROW((void) certified.lower<ArithmeticMode::Ordinary>(), std::invalid_argument);
    EXPECT_THROW((void) certified.upper<ArithmeticMode::Ordinary>(), std::invalid_argument);
    EXPECT_THROW((void) certified.as_double<ArithmeticMode::Ordinary>(), std::invalid_argument);
    EXPECT_THROW((void) ordinary.lower<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
    EXPECT_THROW((void) ordinary.upper<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
    EXPECT_THROW((void) ordinary.as_double<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
    for (const auto tag : {ArithmeticMode::Exact, static_cast<ArithmeticMode>(99)})
    {
        const SimpleFluid::PLIC::DynamicGeometryValue foreign{1, 0, tag};
        EXPECT_THROW((void) foreign.lower<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
        EXPECT_THROW((void) foreign.upper<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
        EXPECT_THROW((void) foreign.as_double<ArithmeticMode::Ordinary>(), std::invalid_argument);
        EXPECT_THROW((void) foreign.as_double<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
    }
    fixed_value_publication_range_contract<ArithmeticMode::Ordinary>();
    fixed_value_publication_range_contract<ArithmeticMode::BoundedCertification>();
    const SimpleFluid::PLIC::DynamicGeometryValue subnormal{
        std::numeric_limits<long double>::denorm_min(), 0, ArithmeticMode::BoundedCertification};
    EXPECT_THROW((void) subnormal.lower<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
    EXPECT_THROW((void) subnormal.upper<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
    EXPECT_THROW((void) subnormal.lower(), std::invalid_argument);
    EXPECT_THROW((void) subnormal.upper(), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, OrdinaryGeometryUsesNominalValuesWithoutClaimingBounds)
{
    const auto ordinary = geometry_limits(ArithmeticMode::Ordinary);
    const auto certified = geometry_limits(ArithmeticMode::BoundedCertification);
    const std::array<SimpleFluid::PLIC::DynamicPoint, 4> input{SimpleFluid::PLIC::DynamicPoint({0, 0, 0}),
        SimpleFluid::PLIC::DynamicPoint({1, 0, 0}, {2, 0, 0}), SimpleFluid::PLIC::DynamicPoint({0, 1, 0}),
        SimpleFluid::PLIC::DynamicPoint({0, 0, 1})};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron(input, certified), std::invalid_argument);
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto tetra = SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron(input, ordinary);
    ASSERT_TRUE(tetra);
    EXPECT_EQ(tetra->arithmetic_mode(), ArithmeticMode::Ordinary);
    const auto volume = tetra->volume();
    EXPECT_EQ(volume.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_NEAR(volume.value, 1.L / 6, 1e-18L);
    EXPECT_THROW((void) volume.lower(), std::invalid_argument);
    EXPECT_THROW((void) volume.upper(), std::invalid_argument);
    EXPECT_THROW((void) volume.relative_error_at_most(volume.value, 1e-12), std::invalid_argument);
    EXPECT_DOUBLE_EQ(volume.as_double(), 1. / 6);
    const auto center = tetra->centroid();
    for (const auto coordinate : center)
        EXPECT_DOUBLE_EQ(coordinate, .25);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().interval_operations, 0u);
    for (const auto& point : tetra->points())
        EXPECT_EQ(point.arithmetic_mode, ArithmeticMode::Ordinary);

    const std::array<SimpleFluid::PLIC::DynamicPoint, 4> uncertain_y{
        SimpleFluid::PLIC::DynamicPoint({.975, .5, 0}, {0, std::ldexp(1.L, -30), 0}),
        SimpleFluid::PLIC::DynamicPoint({1.025, .5, 0}), SimpleFluid::PLIC::DynamicPoint({.975, .75, 0}),
        SimpleFluid::PLIC::DynamicPoint({.975, .5, .25})};
    const Faces faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const auto bounded = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(uncertain_y, faces, certified);
    EXPECT_THROW((void) bounded->clip({{0, 1, 0}, {}, .5}, false), std::invalid_argument);
    const auto nominal = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(uncertain_y, faces, ordinary);
    const auto contact = nominal->clip({{0, 1, 0}, {}, .5}, false);
    EXPECT_FALSE(contact.inside);
    ASSERT_TRUE(contact.outside);
    EXPECT_EQ(contact.outside->arithmetic_mode(), ArithmeticMode::Ordinary);
}

TEST(BoundedConvexGeometryTest, OrdinaryGeometryRetainsFiniteDomainTopologyAndResourceChecks)
{
    auto ordinary = geometry_limits(ArithmeticMode::Ordinary);
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({1, 0, 0}, {0, 1, 1}, ordinary), std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::box(
                     {0, 0, 0}, {std::numeric_limits<double>::infinity(), 1, 1}, ordinary),
        std::invalid_argument);
    const auto box = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, ordinary);
    EXPECT_THROW((void) box->clip({{0, 0, 0}, {}, 0}), std::invalid_argument);
    auto cap = ordinary;
    cap.maximum_operations = 1;
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, cap), std::invalid_argument);
    cap = ordinary;
    cap.byte_budget = 1;
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, cap), std::invalid_argument);
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> flat{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::make(
                     flat, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}, ordinary),
        std::invalid_argument);
    struct RoundingRestore
    {
        int previous = std::fegetround();
        ~RoundingRestore() { std::fesetround(previous); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    EXPECT_NO_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, ordinary));
    EXPECT_THROW((void) cube(), std::invalid_argument);
}

struct OrdinarySharedCornerFixture
{
    SimpleFluid::PLIC::DynamicConvexGeometryPtr parent, cutter;
    SimpleFluid::PLIC::DynamicPoint source_corner;
};

OrdinarySharedCornerFixture ordinary_shared_corner_fixture(long double downward_shift = 0)
{
    const auto limits = geometry_limits(ArithmeticMode::Ordinary);
    const std::array<SimpleFluid::PLIC::GeometryPoint, 8> vertices{
        {{0, 0, 0}, {.25, 0, 0}, {.35, .5, 0}, {.1, .5, 0}, {0, 0, .5}, {.25, 0, .5}, {.35, .5, .5}, {.1, .5, .5}}};
    const Faces faces{{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    // Only the native cell is imported. All swept points, tetrahedra and cuts
    // below use Ordinary FP; these are captured inputs, not rational oracles.
    const auto native =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::make(vertices, faces);
    const auto root = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*native, limits);
    std::vector<SimpleFluid::PLIC::DynamicPoint> top;
    SimpleFluid::PLIC::DynamicPoint edge;
    for (const auto& point : root->points())
        if (point.value[2] == .5L)
        {
            top.push_back(point);
            if (point.value[0] == 0 && point.value[1] == 0)
                edge = point;
        }
    auto corner = SimpleFluid::PLIC::dynamic_point_average(top);
    if (downward_shift != 0)
        corner = SimpleFluid::PLIC::dynamic_point_displaced(
            corner, SimpleFluid::PLIC::DynamicPoint({0, 0, -downward_shift}, {}, ArithmeticMode::Ordinary));
    const auto source_point = [](std::array<long double, 3> value)
    {
        const std::array input{SimpleFluid::PLIC::DynamicPoint(value, {}, ArithmeticMode::Ordinary)};
        return SimpleFluid::PLIC::dynamic_point_average(input);
    };
    const auto b = source_point({0, 0, 0.499997040787199163156L});
    const auto c = source_point({0.249935364847273739278L, 0, 0.499999597319108794318L});
    const auto d = source_point({0.174991507443973570943L, 0.250042432475806645484L, 0.499994860273139090859L});
    const auto first = SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron({corner, d, c, b}, limits);
    const auto cutter = SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron({corner, edge, b, c}, limits);
    auto parent = root;
    for (const auto face : {3u, 2u, 1u})
    {
        const auto split = parent->clip_by_face(*first, face, false);
        parent = face == 1 ? split.outside : split.inside;
        if (!parent)
            throw std::runtime_error("Captured shared-corner fixture lost its positive parent.");
    }
    return {parent, cutter, corner};
}

TEST(BoundedConvexGeometryTest, OrdinaryCapturedCrossingsRetainAuthenticatedSourceCorner)
{
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Source-rank filter requires its supported FP environment.";
    const auto fixture = ordinary_shared_corner_fixture();
    const auto& parent = fixture.parent;
    const auto& corner = fixture.source_corner;
    // Three retained support equations reconstruct the original wall point.
    // The old crossing formula displaced it by a few wider-FP increments and
    // manufactured a coplanar tetrahedron on the following grazing cut.
    const auto restored = std::find_if(parent->points().begin(), parent->points().end(),
        [&](const auto& point) { return point.value == corner.value; });
    ASSERT_NE(restored, parent->points().end());
    EXPECT_EQ(restored->supports(), corner.supports());
    ASSERT_GT(parent->volume().value, 0);
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto split = parent->clip_by_face(*fixture.cutter, 1, false);
    ASSERT_TRUE(split.inside);
    EXPECT_FALSE(split.outside);
    EXPECT_EQ(split.inside->storage_identity(), parent->storage_identity());
    EXPECT_EQ(split.inside->volume().value, parent->volume().value);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().interval_operations, 0u);
}

TEST(BoundedConvexGeometryTest, OrdinaryNearbyDistinctSourceRetainsPositiveGrazingFragment)
{
    const auto fixture = ordinary_shared_corner_fixture(0x1p-40L);
    EXPECT_TRUE(fixture.source_corner.supports().empty());
    const auto split = fixture.parent->clip_by_face(*fixture.cutter, 1, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const auto before = fixture.parent->volume().value;
    const auto inside = split.inside->volume().value, outside = split.outside->volume().value;
    EXPECT_GT(outside, 0);
    EXPECT_LT(outside, before * 1e-12L);
    EXPECT_NEAR(inside + outside, before, before * 8e-15L);
}

TEST(BoundedConvexGeometryTest, OrdinaryCapturedOppositeSourcePlanesPreserveTheWholeParent)
{
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Source-plane filter requires its supported FP environment.";
    const auto fixture = ordinary_shared_corner_fixture();
    const auto limits = geometry_limits(ArithmeticMode::Ordinary);
    const auto source_point = [](std::array<long double, 3> value)
    {
        const std::array input{SimpleFluid::PLIC::DynamicPoint(value, {}, ArithmeticMode::Ordinary)};
        return SimpleFluid::PLIC::dynamic_point_average(input);
    };
    const auto d = source_point({0.174991507443973570943L, 0.250042432475806645484L, 0.499994860273139090859L});
    const auto f = source_point({0.350007862053653909896L, 0.500055715528393436662L, 0.500000065164287879681L});
    const auto g = source_point({0.100022802874966634604L, 0.500114014374833145272L, 0.499999705241252620898L});
    const auto& a = fixture.source_corner;
    const auto& e = fixture.cutter->points()[1];
    const auto first = SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron({a, d, f, g}, limits);
    const auto opposite = SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron({a, e, f, g}, limits);
    const auto parent = fixture.parent->clip_by_face(*first, 2, false).outside;
    ASSERT_TRUE(parent);
    ASSERT_GT(parent->volume().value, 0);
    // Both original triangles contain precisely a/f/g. Opposite winding must
    // preserve zero incidence rather than turn rounded residuals into a sliver.
    const auto split = parent->clip_by_face(*opposite, 2, false);
    ASSERT_TRUE(split.inside);
    EXPECT_FALSE(split.outside);
    EXPECT_EQ(split.inside->storage_identity(), parent->storage_identity());
    EXPECT_EQ(split.inside->volume().value, parent->volume().value);
}

TEST(BoundedConvexGeometryTest, OrdinarySourcePlaneIdentityRejectsLookAlikesAndUnprovedTriangles)
{
    namespace arithmetic = SimpleFluid::PLIC::details;
    namespace ordinary = arithmetic::ordinary_kernel;
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Source-plane filter requires its supported FP environment.";
    const auto token = [](std::array<long double, 3> value)
    {
        const std::array input{SimpleFluid::PLIC::DynamicPoint(value, {}, ArithmeticMode::Ordinary)};
        return SimpleFluid::PLIC::DynamicGeometryAccess::identity(SimpleFluid::PLIC::dynamic_point_average(input));
    };
    const std::array points{token({0, 0, 0}), token({1, 0, 0}), token({0, 1, 0})};
    ordinary::Plane a, b;
    a.normal[2] = ordinary::StorageScalar(1);
    b.normal[2] = ordinary::StorageScalar(-1);
    a.normal_error = b.normal_error = 0;
    a.source_vertices = {points[0], points[1], points[2]};
    b.source_vertices = {points[2], points[1], points[0]};
    const auto limits = geometry_limits(ArithmeticMode::Ordinary);
    {
        arithmetic::Operation scope(limits);
        EXPECT_TRUE(ordinary::same_source_plane(a, b));
        const auto look_alike = token({0, 1, 0});
        b.source_vertices[0] = look_alike;
        EXPECT_FALSE(ordinary::same_source_plane(a, b));
        b.source_vertices[0] = points[2];
        b.source_vertices[1] = points[2];
        EXPECT_FALSE(ordinary::same_source_plane(a, b));
        b.source_vertices[1] = points[1];
        b.normal_error.reset();
        EXPECT_FALSE(ordinary::same_source_plane(a, b));
        b.normal_error = 1;
        EXPECT_FALSE(ordinary::same_source_plane(a, b));
        b.normal_error = 0;
        b.source_vertices.push_back(points[0]);
        EXPECT_FALSE(ordinary::same_source_plane(a, b));
        b.source_vertices.pop_back();
        {
            const auto expired = token({0, 1, 0});
            b.source_vertices[0] = expired;
        }
        EXPECT_FALSE(ordinary::same_source_plane(a, b));
        b.source_vertices[0] = points[2];
    }
    auto capped = limits;
    capped.maximum_pair_tests = 1;
    arithmetic::Operation limited(capped);
    EXPECT_THROW((void) ordinary::same_source_plane(a, b), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, OrdinarySourceRankIncludesConstructionErrorAndRejectsMissingWitness)
{
    namespace arithmetic = SimpleFluid::PLIC::details;
    namespace ordinary = arithmetic::ordinary_kernel;
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Source-rank filter requires its supported FP environment.";
    arithmetic::Operation scope(geometry_limits(ArithmeticMode::Ordinary));
    const auto plane = [](std::array<long double, 3> normal, long double error)
    {
        ordinary::Plane result;
        for (std::size_t axis = 0; axis < 3; ++axis)
            result.normal[axis] = ordinary::StorageScalar(normal[axis]);
        result.normal_error = error;
        return result;
    };
    const auto a = plane({1, 1, 0}, 0), c = plane({0, 0, 1}, 0);
    const auto ulp = std::nextafter(1.L, 2.L) - 1;
    // The original first two normals may be identical even though their
    // separately rounded representations have a nonzero determinant.
    auto b = plane({1, 1 + ulp, 0}, ulp);
    EXPECT_FALSE(ordinary::independent_source_normals(a, b, c));
    b = plane({1, 1 + 0x1p-30L, 0}, ulp);
    EXPECT_TRUE(ordinary::independent_source_normals(a, b, c));
    b.normal_error.reset();
    EXPECT_FALSE(ordinary::independent_source_normals(a, b, c));
    b.normal_error = std::numeric_limits<long double>::infinity();
    EXPECT_FALSE(ordinary::independent_source_normals(a, b, c));
}

TEST(BoundedConvexGeometryTest, OrdinarySourceRecoveryRejectsMissingAmbiguousAndCappedEvidence)
{
    namespace arithmetic = SimpleFluid::PLIC::details;
    namespace ordinary = arithmetic::ordinary_kernel;
    if (!SimpleFluid::PLIC::certification_environment())
        GTEST_SKIP() << "Source-rank filter requires its supported FP environment.";
    const auto limits = geometry_limits(ArithmeticMode::Ordinary);
    const std::array input{SimpleFluid::PLIC::DynamicPoint({.175L, .25L, .5L}, {}, ArithmeticMode::Ordinary)};
    const auto source = SimpleFluid::PLIC::dynamic_point_average(input);
    const auto token = SimpleFluid::PLIC::DynamicGeometryAccess::identity(source);
    const auto look_alike =
        SimpleFluid::PLIC::DynamicGeometryAccess::identity(SimpleFluid::PLIC::dynamic_point_average(input));
    ASSERT_NE(token, look_alike);
    ordinary::Vertex vertex;
    std::array<std::shared_ptr<ordinary::Plane>, 3> planes;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        auto p = std::make_shared<ordinary::Plane>();
        p->normal[axis] = ordinary::StorageScalar(1);
        p->normal_error = 0;
        p->source_vertices = {token};
        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
            p->anchor[coordinate] = ordinary::StorageScalar(source.value[coordinate]);
        vertex.coordinates[axis] = ordinary::StorageScalar(std::nextafter(source.value[axis], 0.L));
        vertex.incident.push_back(p);
        planes[axis] = std::move(p);
    }
    const auto original = vertex;
    {
        arithmetic::Operation scope(limits);
        planes[2]->source_vertices.clear();
        EXPECT_FALSE(ordinary::recover_source_identity(vertex, {}));
        planes[2]->source_vertices = {token};
        for (auto& p : planes)
            p->source_vertices.push_back(look_alike);
        EXPECT_FALSE(ordinary::recover_source_identity(vertex, {}));
        for (auto& p : planes)
            p->source_vertices = {token};
    }
    auto capped = limits;
    capped.maximum_pair_tests = 1;
    {
        arithmetic::Operation limited(capped);
        EXPECT_THROW((void) ordinary::recover_source_identity(vertex, {}), std::invalid_argument);
    }
    EXPECT_FALSE(vertex.source_identity);
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_EQ(vertex.coordinates[axis].value, original.coordinates[axis].value);
    EXPECT_TRUE(vertex.authorities.empty());
}

TEST(BoundedConvexGeometryTest, OrdinaryPointHelpersKeepExplicitModesAndRejectMixedInputs)
{
    const SimpleFluid::PLIC::DynamicPoint a({1, 2, 3}, {4, 4, 4}, ArithmeticMode::Ordinary);
    const SimpleFluid::PLIC::DynamicPoint delta({.5, 0, 0}, {}, ArithmeticMode::Ordinary);
    const auto moved = SimpleFluid::PLIC::dynamic_point_displaced(a, delta);
    EXPECT_EQ(moved.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_EQ(moved.value[0], 1.5L);
    const std::array points{a, moved};
    const auto average = SimpleFluid::PLIC::dynamic_point_average(points);
    EXPECT_EQ(average.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_EQ(average.value[0], 1.25L);
    const auto mode = geometry_limits(ArithmeticMode::Ordinary);
    const auto traced = SimpleFluid::PLIC::dynamic_backtrace_constrained(a, {.5, 0, 0}, .25, {}, mode);
    EXPECT_EQ(traced.arithmetic_mode, ArithmeticMode::Ordinary);
    EXPECT_EQ(traced.value[0], .875L);
    const auto wall = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, mode);
    const std::array constraints{SimpleFluid::PLIC::DynamicFaceConstraint{wall, 0}};
    const SimpleFluid::PLIC::DynamicPoint on_wall({0, .5, .5}, {}, ArithmeticMode::Ordinary);
    const auto tangent = SimpleFluid::PLIC::dynamic_backtrace_constrained(on_wall, {1, 2, 3}, .125, constraints, mode);
    EXPECT_EQ(tangent.value[0], 0);
    EXPECT_EQ(tangent.value[1], .25L);
    EXPECT_EQ(tangent.value[2], .125L);
    const SimpleFluid::PLIC::DynamicPoint certified({.5, 0, 0});
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_point_displaced(a, certified), std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_point_interpolate(a, certified, .5L), std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_points_identical(a, certified), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, OrdinarySpatialRangesKeepTheirExtentAcrossFrames)
{
    const auto mode = geometry_limits(ArithmeticMode::Ordinary);
    const auto left = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, mode);
    const auto overlap = SimpleFluid::PLIC::DynamicConvexGeometry::box({.75, 0, 0}, {1.75, 1, 1}, mode);
    EXPECT_FALSE(left->bounds_prove_disjoint_interiors(*overlap));
    EXPECT_FALSE(overlap->bounds_prove_disjoint_interiors(*left));
    const auto rebased = left->clip({{1, 0, 0}, {16, 0, 0}, 0}, false).inside;
    ASSERT_TRUE(rebased);
    EXPECT_FALSE(rebased->bounds_prove_disjoint_interiors(*overlap));
    const auto intersection = rebased->intersect(*overlap);
    ASSERT_TRUE(intersection);
    EXPECT_DOUBLE_EQ(intersection->volume().as_double(), .25);
    const auto contact = SimpleFluid::PLIC::DynamicConvexGeometry::box({1, 0, 0}, {2, 1, 1}, mode);
    EXPECT_TRUE(left->bounds_prove_disjoint_interiors(*contact));
    EXPECT_TRUE(contact->bounds_prove_disjoint_interiors(*rebased));
}

TEST(BoundedConvexGeometryTest, GeometryModesArePinnedAcrossOperationsPartitionsAndPackets)
{
    const auto ordinary = geometry_limits(ArithmeticMode::Ordinary);
    const auto bounded = geometry_limits(ArithmeticMode::BoundedCertification);
    const auto a = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, ordinary);
    const auto b = cube();
    EXPECT_THROW((void) a->intersect(*b), std::invalid_argument);
    EXPECT_THROW((void) a->bounds_prove_disjoint_interiors(*b), std::invalid_argument);
    EXPECT_THROW((void) a->clip_by_face(*b, 0), std::invalid_argument);
    const std::array wrong{b};
    EXPECT_THROW(a->certify_partition(wrong, 1e-12), std::invalid_argument);
    const std::array mixed{a, b};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexPartition(mixed, ordinary), std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(a->points(), a->faces(), bounded),
        std::invalid_argument);
    const SimpleFluid::PLIC::GeometryPacketContext context{{101, 103, 107, 109}};
    const auto packet = a->encode(context);
    const std::array registry{a};
    const auto decoded = SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, registry, ordinary);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->arithmetic_mode(), ArithmeticMode::Ordinary);
    EXPECT_DOUBLE_EQ(decoded->volume().as_double(), 1);
    const std::array donors{SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1}, ordinary),
        SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1}, ordinary)};
    const auto partition = SimpleFluid::PLIC::DynamicConvexPartition(donors, ordinary).partition(*a);
    ASSERT_EQ(partition.size(), 2u);
    std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> cover;
    for (const auto& piece : partition)
    {
        EXPECT_EQ(piece.geometry->arithmetic_mode(), ArithmeticMode::Ordinary);
        EXPECT_DOUBLE_EQ(piece.geometry->volume().as_double(), .5);
        cover.push_back(piece.geometry);
    }
    EXPECT_NO_THROW(a->certify_partition(cover, 0));
    const std::array certified_registry{b};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, certified_registry, bounded),
        std::invalid_argument);
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(b->encode(context), context, registry, ordinary),
        std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, mixed, ordinary),
        std::invalid_argument);
    const auto exact = geometry_limits(ArithmeticMode::Exact);
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, exact), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, AnalyticCubeAndObliqueTetrahedronEnclosures)
{
    ASSERT_TRUE(SimpleFluid::PLIC::certification_environment());
    const auto parent = cube();
    encloses(parent->volume(), 1);
    EXPECT_EQ(parent->volume().absolute_error, 0);
    const auto m = parent->moments();
    for (const auto& moment : m.first_moment)
        encloses(moment, .5L);
    const auto cut = parent->clip({{1, 1, 1}, {}, 1});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    encloses(cut.inside->volume(), 1.L / 6);
    encloses(cut.outside->volume(), 5.L / 6);
    const auto small = cut.inside->moments();
    for (const auto& moment : small.first_moment)
        encloses(moment, 1.L / 24);
    encloses(cut.section.area, std::sqrt(3.L) / 2);
    const std::array pieces{cut.inside, cut.outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
    EXPECT_TRUE(cut.inside->volume().relative_error_at_most(1.L / 6, 8e-15));
}

TEST(BoundedConvexGeometryTest, TranslationAndPowerOfTwoScaleDoNotChangeLocalAccuracy)
{
    for (const int exponent : {-100, 0, 100})
    {
        const double scale = std::ldexp(1., exponent), anchor = 16 * scale;
        const auto cell = SimpleFluid::PLIC::DynamicConvexGeometry::box(
            {anchor, -anchor, anchor}, {anchor + scale, -anchor + scale, anchor + scale});
        const auto volume = std::ldexp(1.L, 3 * exponent);
        encloses(cell->volume(), volume);
        const auto cut = cell->clip({{1, 1, 1}, {anchor, -anchor, anchor}, scale});
        ASSERT_TRUE(cut.inside);
        encloses(cut.inside->volume(), volume / 6);
        for (const auto& moment : cut.inside->moments().first_moment)
            encloses(moment, std::ldexp(1.L / 24, 4 * exponent));
        EXPECT_TRUE(cut.inside->volume().relative_error_at_most(volume / 6, 8e-15));
        for (const auto& span : cell->axis_spans())
            encloses(span, scale);
    }
}

TEST(BoundedConvexGeometryTest, RationalSourceExportRetainsIndependentOracleInsideBounds)
{
    const auto exact =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1})
            ->clip({{10, 0, 0}, {}, 1})
            .inside;
    ASSERT_TRUE(exact);
    const auto finite = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*exact);
    encloses(finite->volume(), .1L);
    const auto cut = finite->clip({{0, 1, 1}, {}, .5});
    ASSERT_TRUE(cut.inside);
    encloses(cut.inside->volume(), .0125L);
    const auto oracle = exact->clip({{0, 1, 1}, {}, .5}).inside->moments();
    const auto measured = cut.inside->moments();
    EXPECT_LE(std::abs(measured.volume.value - oracle.volume.value),
        measured.volume.absolute_error + oracle.volume.absolute_error);
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_LE(std::abs(measured.first_moment[axis].value - oracle.first_moment[axis].value),
            measured.first_moment[axis].absolute_error + oracle.first_moment[axis].absolute_error);
}

TEST(BoundedConvexGeometryTest, ComplementaryAndReversedCutsRetainStructuralEvidence)
{
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto parent = cube();
    const SimpleFluid::PLIC::GeometryHalfspace plane{{3, 2, 1}, {}, 2.25};
    const auto cut = parent->clip(plane);
    const auto reverse = parent->clip({{-3, -2, -1}, {}, -2.25});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(reverse.outside);
    EXPECT_LE(std::abs(cut.inside->volume().value - reverse.outside->volume().value),
        cut.inside->volume().absolute_error + reverse.outside->volume().absolute_error);
    const auto nested = cut.inside->clip({{0, 1, 0}, {}, .3});
    const std::array pieces{nested.inside, nested.outside, cut.outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
    const auto same = cut.inside->clip(plane);
    EXPECT_TRUE(same.inside);
    EXPECT_FALSE(same.outside);
    const auto opposite = cut.inside->clip({{-3, -2, -1}, {}, -2.25});
    EXPECT_FALSE(opposite.inside);
    EXPECT_TRUE(opposite.outside);
    const auto counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_GT(counters.structural_partition_reuses, 0U);
    EXPECT_EQ(counters.exact_predicates, 0U);
    EXPECT_EQ(counters.rational_constructions, 0U);
    EXPECT_EQ(counters.rational_measures, 0U);
    EXPECT_EQ(counters.rational_ledgers, 0U);
}

TEST(BoundedConvexGeometryTest, TangenciesPreserveFaceEdgeAndPointDimension)
{
    const auto parent = cube();
    const auto face = parent->clip({{1, 0, 0}, {}, 0});
    EXPECT_FALSE(face.inside);
    EXPECT_TRUE(face.outside);
    encloses(face.section.area, 1);
    EXPECT_EQ(face.section.vertices.size(), 4U);
    const auto edge = parent->clip({{1, 1, 0}, {}, 0});
    EXPECT_FALSE(edge.inside);
    EXPECT_EQ(edge.section.vertices.size(), 2U);
    EXPECT_EQ(edge.section.area.value, 0);
    const auto vertex = parent->clip({{1, 1, 1}, {}, 0});
    EXPECT_FALSE(vertex.inside);
    EXPECT_EQ(vertex.section.vertices.size(), 1U);
    EXPECT_EQ(vertex.section.area.value, 0);
}

TEST(BoundedConvexGeometryTest, IndependentContainedPiecesNeedNoNetVolumeAssumption)
{
    const auto parent = cube();
    const auto box = [](double lo, double hi)
    { return SimpleFluid::PLIC::DynamicConvexGeometry::box({lo, 0, 0}, {hi, 1, 1}); };
    const std::array good{box(0, .5), box(.5, 1)};
    EXPECT_NO_THROW(parent->certify_partition(good, 8e-15));
    const std::array gap{box(0, .4), box(.6, 1)};
    EXPECT_THROW(parent->certify_partition(gap, 8e-15), std::invalid_argument);
    // The .2 overlap and .2 hole balance exactly in the bulk sum.
    const std::array balanced{box(0, .4), box(.2, .6), box(.8, 1)};
    EXPECT_THROW(parent->certify_partition(balanced, 8e-15), std::invalid_argument);
    const std::array outside{box(-.1, .4), box(.5, 1)};
    EXPECT_THROW(parent->certify_partition(outside, 8e-15), std::invalid_argument);
    const std::array duplicate{good[0], good[0]};
    EXPECT_THROW(parent->certify_partition(duplicate, 8e-15), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, ComplementaryRemainderSharesPrecheckPairCapAndRejectsWithoutMutation)
{
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const Faces faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    // The geometrically identical parent has a fresh identity, so containment
    // still needs 4*(6+4) vertex tests. Structural/ancestry work adds seven
    // pairs and the piece-pair attempt adds one, exhausting this cap before
    // the first remainder clip. Touching x bounds avoid intersection clipping.
    limits.maximum_pair_tests = 48;
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, faces, limits);
    const auto source = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, faces);
    const auto left = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1});
    const auto right = SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1});
    const std::array pieces{source->intersect(*left), source->intersect(*right)};
    ASSERT_TRUE(pieces[0]);
    ASSERT_TRUE(pieces[1]);
    ASSERT_EQ(pieces[0]->points().size(), 6u);
    ASSERT_EQ(pieces[1]->points().size(), 4u);
    ASSERT_EQ(pieces[0]->faces().size(), 5u);
    ASSERT_EQ(pieces[1]->faces().size(), 4u);
    const std::array before{parent->volume(), pieces[0]->volume(), pieces[1]->volume()};
    ASSERT_GT(before[0].absolute_error, 0);

    SimpleFluid::PLIC::reset_geometry_statistics();
    // The same prechecks fit the cap when independent measures suffice.
    EXPECT_NO_THROW(parent->certify_partition(pieces, 1));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 0u);
    try
    {
        parent->certify_partition(pieces, 0);
        FAIL() << "The shared pair cap must reject the complementary remainder proof.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_STREQ(error.what(), "Bounded FP geometry: pair-test cap exhausted.");
    }
    const auto default_parent = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, faces);
    EXPECT_NO_THROW(default_parent->certify_partition(pieces, 0));
    const std::array after{parent->volume(), pieces[0]->volume(), pieces[1]->volume()};
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_EQ(after[i].value, before[i].value);
        EXPECT_EQ(after[i].absolute_error, before[i].absolute_error);
    }
    const auto counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(counters.source_imports, 0u);
    EXPECT_EQ(counters.exact_predicates, 0u);
    EXPECT_EQ(counters.rational_constructions, 0u);
    EXPECT_EQ(counters.rational_measures, 0u);
    EXPECT_EQ(counters.rational_ledgers, 0u);
}

TEST(BoundedConvexGeometryTest, IntersectionUsesActualObliqueShell)
{
    const auto tetra = cube()->clip({{1, 1, 1}, {}, 1}).inside;
    const auto inner = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1});
    const auto overlap = tetra->intersect(*inner);
    ASSERT_TRUE(overlap);
    encloses(overlap->volume(), 7.L / 48);
    const auto outside = SimpleFluid::PLIC::DynamicConvexGeometry::box({2, 0, 0}, {3, 1, 1});
    EXPECT_FALSE(tetra->intersect(*outside));
}

TEST(BoundedConvexGeometryTest, CertifiedBoundsRejectDisjointTetrahedronBeforeAmbiguousPlaneSigns)
{
    for (const int exponent : {-20, 0, 20})
        for (const bool translated : {false, true})
        {
            const double scale = std::ldexp(1., exponent);
            const double shift = translated ? std::ldexp(scale, 40) : 0;
            const SimpleFluid::PLIC::GeometryPoint origin{shift, -shift, shift};
            std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
            vertices[0].value = {.975 * scale, .5 * scale, 0};
            vertices[0].absolute_error[1] = std::ldexp(scale, -30);
            vertices[1].value = {1.025 * scale, .5 * scale, 0};
            vertices[2].value = {.975 * scale, .75 * scale, 0};
            vertices[3].value = {.975 * scale, .5 * scale, .25 * scale};
            const auto tetra = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
                origin, vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
            const auto donor =
                SimpleFluid::PLIC::DynamicConvexGeometry::box({origin[0] - scale, origin[1] - scale, origin[2] - scale},
                    {origin[0] + .85 * scale, origin[1] + .5 * scale, origin[2] + scale});
            SimpleFluid::PLIC::reset_geometry_statistics();
            EXPECT_TRUE(tetra->bounds_prove_disjoint_interiors(*donor));
            EXPECT_TRUE(donor->bounds_prove_disjoint_interiors(*tetra));
            EXPECT_FALSE(tetra->intersect(*donor));
            EXPECT_FALSE(donor->intersect(*tetra));
            const auto counters = SimpleFluid::PLIC::geometry_statistics();
            EXPECT_EQ(counters.intersections, 2u);
            EXPECT_EQ(counters.support_distances, 0u);
            EXPECT_EQ(counters.volume_evaluations, 0u);
            EXPECT_EQ(counters.constructions, 0u);
            EXPECT_EQ(counters.source_imports, 0u);
            EXPECT_EQ(counters.rational_constructions, 0u);
            EXPECT_EQ(counters.rational_measures, 0u);
            EXPECT_EQ(counters.rational_ledgers, 0u);
        }
}

TEST(BoundedConvexGeometryTest, CertifiedBoundsDistinguishOneUlpGapContactAndOverlapAcrossFrames)
{
    for (const int exponent : {-20, 0, 20})
        for (const bool translated : {false, true})
        {
            const double scale = std::ldexp(1., exponent);
            const double shift = translated ? std::ldexp(scale, 40) : 0;
            const SimpleFluid::PLIC::GeometryPoint origin{shift, -shift, shift};
            const double contact = origin[0] + scale;
            const auto left =
                SimpleFluid::PLIC::DynamicConvexGeometry::box(origin, {contact, origin[1] + scale, origin[2] + scale});
            // A noncutting plane still rebases the private numerical frame.
            // The cached bounds must follow that rebase without moving the box.
            const auto rebased = left->clip(
                {{1, 0, 0}, {origin[0] + 16 * scale, origin[1], origin[2]}, 0}, false).inside;
            ASSERT_TRUE(rebased);
            const std::array identities{left->intersect(*rebased), rebased->intersect(*left)};
            for (const auto& identity : identities)
            {
                ASSERT_TRUE(identity);
                encloses(identity->volume(), static_cast<long double>(scale) * scale * scale);
            }
            for (const double start : {std::nextafter(contact, std::numeric_limits<double>::infinity()), contact,
                     std::nextafter(contact, -std::numeric_limits<double>::infinity())})
            {
                const auto right = SimpleFluid::PLIC::DynamicConvexGeometry::box(
                    {start, origin[1], origin[2]}, {origin[0] + 2 * scale, origin[1] + scale, origin[2] + scale});
                EXPECT_EQ(left->bounds_prove_disjoint_interiors(*right), start >= contact);
                EXPECT_EQ(right->bounds_prove_disjoint_interiors(*left), start >= contact);
                EXPECT_EQ(rebased->bounds_prove_disjoint_interiors(*right), start >= contact);
                EXPECT_EQ(right->bounds_prove_disjoint_interiors(*rebased), start >= contact);
                const std::array intersections{left->intersect(*right), right->intersect(*left)};
                for (const auto& intersection : intersections)
                    if (start >= contact)
                        EXPECT_FALSE(intersection);
                    else
                    {
                        ASSERT_TRUE(intersection);
                        EXPECT_GT(intersection->volume().lower(), 0);
                        const auto width = static_cast<long double>(contact) - static_cast<long double>(start);
                        encloses(intersection->volume(), width * scale * scale);
                    }
            }
        }
}

TEST(BoundedConvexGeometryTest, BoundsCertificateDeclinesAnUncertainNominalGap)
{
    const long double error = std::ldexp(1.L, -30);
    const std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices{
        SimpleFluid::PLIC::DynamicPoint({1, 0, 0}, {error, 0, 0}), SimpleFluid::PLIC::DynamicPoint({1, 1, 0}),
        SimpleFluid::PLIC::DynamicPoint({1, 0, 1}), SimpleFluid::PLIC::DynamicPoint({1.25, 0, 0})};
    const auto tetra =
        SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto donor = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1 - std::ldexp(1., -31), 1, 1});
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_FALSE(tetra->bounds_prove_disjoint_interiors(*donor));
    EXPECT_FALSE(donor->bounds_prove_disjoint_interiors(*tetra));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().volume_evaluations, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().constructions, 0u);
}

TEST(BoundedConvexGeometryTest, ConstructedDonorPartitionPublishesOneMeasuredTaggedPiecePerDonor)
{
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const auto parent =
        SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto before = parent->volume();
    const std::array donors{SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1}),
        SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1})};
    const SimpleFluid::PLIC::DynamicConvexPartition partition(donors);
    EXPECT_GT(partition.storage_bytes(), 0u);
    const auto result = partition.partition(*parent);
    ASSERT_EQ(result.size(), 2u);
    const std::array expected{7.L / 48, 1.L / 48};
    std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> pieces;
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i].donor, i);
        ASSERT_TRUE(result[i].geometry);
        const auto measured = result[i].geometry->volume();
        EXPECT_GT(measured.lower(), 0);
        encloses(measured, expected[i]);
        EXPECT_NO_THROW((void) result[i].geometry->moments());
        pieces.push_back(result[i].geometry);
    }
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(pieces, 0));
    const auto counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(counters.structural_partition_reuses, 1u);
    EXPECT_EQ(counters.support_distances, 0u);
    EXPECT_EQ(counters.intersections, 0u);
    EXPECT_EQ(counters.constructions, 0u);
    EXPECT_EQ(counters.volume_evaluations, 0u);
    EXPECT_EQ(counters.source_imports, 0u);
    EXPECT_EQ(counters.rational_constructions, 0u);
    EXPECT_EQ(counters.rational_measures, 0u);
    EXPECT_EQ(counters.rational_ledgers, 0u);
    EXPECT_EQ(parent->volume().value, before.value);
    EXPECT_EQ(parent->volume().absolute_error, before.absolute_error);
}

TEST(BoundedConvexGeometryTest, ConstructedDonorPartitionRejectsMissingOverlappingDuplicateAndCappedWork)
{
    const auto parent = cube();
    const auto left = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1});
    const auto right = SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1});
    const std::array missing{left};
    const SimpleFluid::PLIC::DynamicConvexPartition incomplete(missing);
    EXPECT_THROW((void) incomplete.partition(*parent), std::invalid_argument);
    const std::array overlap{SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.75, 1, 1}), right};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexPartition(overlap), std::invalid_argument);
    const std::array duplicate{left, left};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexPartition(duplicate), std::invalid_argument);
    const std::array null_donor{SimpleFluid::PLIC::DynamicConvexGeometryPtr{}};
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexPartition(null_donor), std::invalid_argument);
    const std::array good{left, right};
    const SimpleFluid::PLIC::DynamicConvexPartition partition(good);
    EXPECT_THROW((void) partition.partition(*parent, 0), std::invalid_argument);
    EXPECT_THROW((void) partition.partition(*parent, 1), std::invalid_argument);
    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    limits.maximum_pair_tests = 1;
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexPartition(good, limits), std::invalid_argument);
    EXPECT_NO_THROW((void) partition.partition(*parent));
}

TEST(BoundedConvexGeometryTest, ConstructedPartitionHonorsTighterRootChildAndAllocationCaps)
{
    const auto box = cube();
    const auto before = box->volume();
    const std::array alias{box};
    for (const bool vertex_cap : {false, true})
    {
        auto limits = SimpleFluid::PLIC::GeometryLimits{};
        limits.maximum_vertices = vertex_cap ? 7 : 8;
        limits.maximum_faces = vertex_cap ? 6 : 5;
        const SimpleFluid::PLIC::DynamicConvexPartition partition(alias, limits);
        EXPECT_THROW((void) partition.partition(*box), std::invalid_argument);
    }
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const auto tetra =
        SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const std::array donors{SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1}),
        SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1})};
    for (const bool vertex_cap : {false, true})
    {
        auto limits = SimpleFluid::PLIC::GeometryLimits{};
        limits.maximum_vertices = vertex_cap ? 5 : 8;
        limits.maximum_faces = vertex_cap ? 6 : 4;
        const SimpleFluid::PLIC::DynamicConvexPartition partition(donors, limits);
        EXPECT_THROW((void) partition.partition(*tetra), std::invalid_argument);
    }
    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    limits.byte_budget = 128;
    const SimpleFluid::PLIC::DynamicConvexPartition allocation_capped(alias, limits);
    try
    {
        (void) allocation_capped.partition(*box);
        FAIL() << "Alias and no-cut paths must account their actual proof allocations.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_STREQ(error.what(), "Bounded FP geometry: construction storage cap exhausted.");
    }
    EXPECT_EQ(box->volume().value, before.value);
    EXPECT_EQ(box->volume().absolute_error, before.absolute_error);
}

TEST(BoundedConvexGeometryTest, ConstructedPartitionCertificateDoesNotTransferToForeignOrDecodedGeometry)
{
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const Faces faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, faces);
    const std::array donors{SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1}),
        SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1})};
    const auto result = SimpleFluid::PLIC::DynamicConvexPartition(donors).partition(*parent);
    std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> pieces, shifted, decoded;
    const SimpleFluid::PLIC::GeometryPacketContext context{{9, 8, 7, 6}};
    const std::array registry{parent};
    for (const auto& piece : result)
    {
        pieces.push_back(piece.geometry);
        shifted.push_back(piece.geometry->translated({0, 0, 0}));
        decoded.push_back(
            SimpleFluid::PLIC::DynamicConvexGeometry::decode(piece.geometry->encode(context), context, registry));
    }
    const auto foreign = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, faces);
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(foreign->certify_partition(pieces, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 0u);
    EXPECT_GT(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    for (const auto* unlinked : {&shifted, &decoded})
    {
        SimpleFluid::PLIC::reset_geometry_statistics();
        EXPECT_NO_THROW(parent->certify_partition(*unlinked, 0));
        EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 0u);
        EXPECT_GT(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    }
    const auto moved_parent = parent->translated({2, 0, 0});
    EXPECT_THROW(moved_parent->certify_partition(pieces, 0), std::invalid_argument);
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(pieces, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
}

TEST(BoundedConvexGeometryTest, ConstructedBatchCertificateRequiresEveryUnmodifiedMemberExactlyOnce)
{
    const auto parent = cube();
    const std::array donors{SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {.5, 1, 1}),
        SimpleFluid::PLIC::DynamicConvexGeometry::box({.5, 0, 0}, {1, 1, 1})};
    const SimpleFluid::PLIC::DynamicConvexPartition partition(donors);
    const auto first = partition.partition(*parent);
    const auto second = partition.partition(*parent);
    ASSERT_EQ(first.size(), 2u);
    ASSERT_EQ(second.size(), 2u);
    const auto& a = first[0].geometry;
    const auto& b = first[1].geometry;
    const std::array missing{a};
    const std::array duplicate{a, a};
    const std::array added{a, b, b};
    const std::array translated{a, b->translated({.25, 0, 0})};
    const std::array reclipped{a, b->clip({{1, 0, 0}, {}, .75}, false).inside};
    ASSERT_TRUE(reclipped[1]);
    EXPECT_THROW(parent->certify_partition(missing, 0), std::invalid_argument);
    EXPECT_THROW(parent->certify_partition(duplicate, 0), std::invalid_argument);
    EXPECT_THROW(parent->certify_partition(added, 0), std::invalid_argument);
    EXPECT_THROW(parent->certify_partition(translated, 0), std::invalid_argument);
    EXPECT_THROW(parent->certify_partition(reclipped, 0), std::invalid_argument);

    // Independent batches describe the same physical cover but cannot supply
    // each other's exact membership certificate; ordinary checking remains.
    const std::array mixed{a, second[1].geometry};
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(mixed, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 0u);
    const std::array unchanged{a->clip({{1, 0, 0}, {16, 0, 0}, 0}, false).inside,
        b->clip({{1, 0, 0}, {}, 2}, false).inside};
    ASSERT_TRUE(unchanged[0]);
    ASSERT_TRUE(unchanged[1]);
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(unchanged, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().intersections, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().constructions, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().volume_evaluations, 0u);
}

TEST(BoundedConvexGeometryTest, SharedAndIndependentClipSupportsProveDisjointInteriorsBeforeConstruction)
{
    const SimpleFluid::PLIC::GeometryHalfspace plane{{3, 2, 1}, {}, 2.25};
    const auto split = cube()->clip(plane, false);
    const auto independent_same = cube()->clip(plane, false);
    const auto independent_reverse = cube()->clip({{-3, -2, -1}, {}, -2.25}, false);
    ASSERT_TRUE(split.inside);
    const std::array opposites{split.outside, independent_same.outside, independent_reverse.inside};
    for (const auto& opposite : opposites)
    {
        ASSERT_TRUE(opposite);
        SimpleFluid::PLIC::reset_geometry_statistics();
        EXPECT_FALSE(split.inside->intersect(*opposite));
        EXPECT_FALSE(opposite->intersect(*split.inside));
        const auto counters = SimpleFluid::PLIC::geometry_statistics();
        EXPECT_EQ(counters.support_distances, 0u);
        EXPECT_EQ(counters.volume_evaluations, 0u);
        EXPECT_EQ(counters.constructions, 0u);
        EXPECT_EQ(counters.source_imports, 0u);
        EXPECT_EQ(counters.exact_predicates, 0u);
        EXPECT_EQ(counters.rational_constructions, 0u);
        EXPECT_EQ(counters.rational_measures, 0u);
        EXPECT_EQ(counters.rational_ledgers, 0u);
    }
    // Equal input planes with the same halfspace orientation still overlap.
    EXPECT_TRUE(split.inside->intersect(*independent_same.inside));
    EXPECT_TRUE(split.inside->intersect(*independent_reverse.outside));

    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    // The five-edge setup cap charges one visit per inherited edge, map entry
    // and winding check. Its 15 visits fit; the separating-face scan does not.
    limits.maximum_pair_tests = 15;
    const auto capped = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits)->clip(plane, false);
    ASSERT_TRUE(capped.inside);
    ASSERT_TRUE(capped.outside);
    try
    {
        (void) capped.inside->intersect(*independent_same.outside);
        FAIL() << "The separating-support scan must honor the shared pair cap.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_STREQ(error.what(), "Bounded FP geometry: pair-test cap exhausted.");
    }
}

TEST(BoundedConvexGeometryTest, ImportedNativeSupportKeysProveSeparationAndTranslationClearsThem)
{
    // Native boxes provide only immutable source authority. No source clipping
    // or rational reference measures participate in this bounded intersection.
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto neighbor_source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({1, 0, 0}, {2, 1, 1});
    const auto left = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    const auto right = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*neighbor_source);
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_FALSE(left->intersect(*right));
    EXPECT_FALSE(right->intersect(*left));
    const auto separated = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(separated.support_distances, 0u);
    EXPECT_EQ(separated.volume_evaluations, 0u);
    EXPECT_EQ(separated.constructions, 0u);
    const auto shifted = right->translated({-.5, 0, 0});
    const auto overlap = left->intersect(*shifted);
    ASSERT_TRUE(overlap);
    encloses(overlap->volume(), .5L);
    const auto counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(counters.source_imports, 0u);
    EXPECT_EQ(counters.exact_predicates, 0u);
    EXPECT_EQ(counters.rational_constructions, 0u);
    EXPECT_EQ(counters.rational_measures, 0u);
    EXPECT_EQ(counters.rational_ledgers, 0u);
}

TEST(BoundedConvexGeometryTest, IntersectionMeasuresOnePieceWhileComplementaryClipCertifiesBoth)
{
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    vertices[1].value = {1, 0, 0};
    vertices[2].value = {0, 1, 0};
    vertices[3].value = {0, 0, 1};
    const double thickness = std::ldexp(1., -20);
    vertices[2].absolute_error[1] = thickness / 4;
    const auto parent =
        SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto before = parent->volume();
    const double cut = 1 - thickness;
    const auto cutter = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, cut, 1});
    // Anchored integration certifies this tiny tip as well as the bulk. The
    // complementary API measures both; intersection measures its inside only.
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto split = parent->clip({{0, 1, 0}, {}, cut}, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    EXPECT_GT(split.inside->volume().lower(), 0);
    EXPECT_GT(split.outside->volume().lower(), 0);
    const auto expected_tip = static_cast<long double>(thickness) * thickness * thickness / 6;
    encloses(split.outside->volume(), expected_tip);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().volume_evaluations, 2u);
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto intersection = parent->intersect(*cutter);
    ASSERT_TRUE(intersection);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().volume_evaluations, 1u);
    EXPECT_GT(intersection->volume().lower(), 0);
    encloses(intersection->volume(), (1.L - thickness * thickness * thickness) / 6);
    const auto tip = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, cut, 0}, {1, 1 + thickness, 1});
    const auto tip_intersection = parent->intersect(*tip);
    ASSERT_TRUE(tip_intersection);
    EXPECT_GT(tip_intersection->volume().lower(), 0);
    encloses(tip_intersection->volume(), expected_tip);
    EXPECT_EQ(parent->volume().value, before.value);
    EXPECT_EQ(parent->volume().absolute_error, before.absolute_error);
    const auto counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(counters.source_imports, 0u);
    EXPECT_EQ(counters.exact_predicates, 0u);
    EXPECT_EQ(counters.rational_constructions, 0u);
    EXPECT_EQ(counters.rational_measures, 0u);
    EXPECT_EQ(counters.rational_ledgers, 0u);
}

TEST(BoundedConvexGeometryTest, IncidentInputPlanesContractAnalyticEdgeAcrossOrdersAndFrames)
{
    for (const int exponent : {-20, 0, 20})
        for (const bool reverse : {false, true})
        {
            const double scale = std::ldexp(1., exponent);
            const SimpleFluid::PLIC::GeometryPoint origin{16 * scale, -8 * scale, 4 * scale};
            std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
            vertices[1].value = {2 * scale, 0, 0};
            vertices[2].value = {0, 2 * scale, 0};
            vertices[3].value = {0, 0, 2 * scale};
            vertices[1].absolute_error[0] = std::ldexp(scale, -30);
            vertices[2].absolute_error[1] = std::ldexp(scale, -30);
            const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
                origin, vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
            const auto before = parent->volume();
            const std::array<SimpleFluid::PLIC::GeometryHalfspace, 2> planes{
                SimpleFluid::PLIC::GeometryHalfspace{{1, -.2, 0}, origin, .25 * scale},
                SimpleFluid::PLIC::GeometryHalfspace{{0, 1, 0}, origin, .5 * scale}};
            const auto first = parent->clip(planes[reverse ? 1 : 0], false).inside;
            ASSERT_TRUE(first);
            const auto result = first->clip(planes[reverse ? 0 : 1], false).inside;
            ASSERT_TRUE(result);
            // The original binary64 coefficient .2 is the plane authority.
            const long double expected_x = (.25L + static_cast<long double>(.2) * .5L) * scale;
            bool found = false;
            for (const auto& vertex : result->local_points())
                if (vertex.value[2] == 0 && vertex.value[1] == .5L * scale && vertex.value[0] > 0)
                {
                    const SimpleFluid::PLIC::DynamicGeometryValue x{vertex.value[0], vertex.absolute_error[0]};
                    encloses(x, expected_x);
                    EXPECT_LE(x.absolute_error, 8 * std::numeric_limits<long double>::epsilon() * scale);
                    EXPECT_EQ(vertex.absolute_error[1], 0);
                    EXPECT_EQ(vertex.absolute_error[2], 0);
                    found = true;
                }
            EXPECT_TRUE(found);
            EXPECT_EQ(parent->volume().value, before.value);
            EXPECT_EQ(parent->volume().absolute_error, before.absolute_error);
        }
}

TEST(BoundedConvexGeometryTest, IncidentPlaneContractionRetainsDifferentAdmissibleSourceIntersections)
{
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    vertices[1].value = {1, 0, 0};
    vertices[2].value = {0, 1.125L, 0};
    vertices[2].absolute_error[1] = .125L;
    vertices[3].value = {0, 0, 1};
    const auto parent =
        SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto before = parent->volume();
    const auto cutter = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, .625, 1});
    const auto result = parent->intersect(*cutter);
    ASSERT_TRUE(result);
    bool found = false;
    for (const auto& vertex : result->points())
        if (vertex.value[2] == 0 && vertex.value[1] == .625L && vertex.value[0] > 0)
        {
            // y-height 1 and 1.25 are both admitted source states. Their edge
            // intersections at y=.625 have exactly representable x=.375/.5.
            const SimpleFluid::PLIC::DynamicGeometryValue x{vertex.value[0], vertex.absolute_error[0]};
            encloses(x, .375L);
            encloses(x, .5L);
            EXPECT_GT(x.absolute_error, 0);
            found = true;
        }
    EXPECT_TRUE(found);
    EXPECT_EQ(parent->volume().value, before.value);
    EXPECT_EQ(parent->volume().absolute_error, before.absolute_error);
}

TEST(BoundedConvexGeometryTest, RetainedEquationResolvesNearParallelResidualAcrossSignsAndFrames)
{
    const double slope = std::nextafter(.25, std::numeric_limits<double>::infinity());
    for (const int exponent : {-20, 0, 20})
    {
        const double scale = std::ldexp(1., exponent);
        const SimpleFluid::PLIC::GeometryPoint origin{16 * scale, -8 * scale, 4 * scale};
        const SimpleFluid::PLIC::GeometryPoint anchor{origin[0] + .375 * scale, origin[1] + .5 * scale, origin[2]};
        std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
        vertices[0].value = {0, .75 * scale, 0};
        vertices[1].value = {scale, .75 * scale, 0};
        vertices[2].value = {0, 1.25 * scale, 0};
        vertices[2].absolute_error[1] = std::ldexp(scale, -30);
        vertices[3].value = {0, .75 * scale, scale};
        const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
            origin, vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
        const auto old = parent->clip({{1, -.25, 0}, anchor, 0}, false).inside;
        ASSERT_TRUE(old);
        const auto before = old->volume();
        const auto independent = old->support({1, -slope, 0}, anchor).upper;
        EXPECT_LT(independent.lower(), 0);
        EXPECT_GT(independent.upper(), 0);
        // On the retained old plane, the new residual is exactly
        // -nextafter_delta*(y-anchor_y), strictly negative for every vertex.
        const auto inside = old->clip({{1, -slope, 0}, anchor, 0}, false);
        ASSERT_TRUE(inside.inside);
        EXPECT_FALSE(inside.outside);
        const auto reversed = old->clip({{-1, slope, 0}, anchor, 0}, false);
        EXPECT_FALSE(reversed.inside);
        ASSERT_TRUE(reversed.outside);
        EXPECT_EQ(inside.inside->storage_identity(), old->storage_identity());
        EXPECT_EQ(reversed.outside->storage_identity(), old->storage_identity());
        EXPECT_EQ(old->volume().value, before.value);
        EXPECT_EQ(old->volume().absolute_error, before.absolute_error);
    }
}

TEST(BoundedConvexGeometryTest, RetainedEquationStillRejectsAnHonestlyAmbiguousNearParallelResidual)
{
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    vertices[0].value = {0, .5, 0};
    vertices[1].value = {1, .5, 0};
    vertices[2].value = {0, 1.25, 0};
    vertices[3].value = {0, .5, 1};
    for (const auto i : {0u, 1u, 3u})
        vertices[i].absolute_error[1] = std::ldexp(1.L, -30);
    const SimpleFluid::PLIC::GeometryPoint anchor{.375, .5, 0};
    const auto parent =
        SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto old = parent->clip({{1, -.25, 0}, anchor, 0}, false).inside;
    ASSERT_TRUE(old);
    const auto before = old->volume();
    const double slope = std::nextafter(.25, std::numeric_limits<double>::infinity());
    // Admitted source states place old-plane vertices on either side of y=.5.
    try
    {
        (void) old->clip({{1, -slope, 0}, anchor, 0}, false);
        FAIL() << "Genuine source uncertainty must not become a decided plane sign.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_STREQ(error.what(),
            "Bounded FP geometry: support/topology sign unresolved from original coordinate enclosures.");
    }
    EXPECT_EQ(old->volume().value, before.value);
    EXPECT_EQ(old->volume().absolute_error, before.absolute_error);
}

TEST(BoundedConvexGeometryTest, PartitionProofRetainsTemporaryWedgesWithoutMeasuringThem)
{
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    vertices[1].value = {1, 0, 0};
    vertices[2].value = {0, 1, 0};
    vertices[2].absolute_error[1] = std::ldexp(1.L, -50);
    vertices[3].value = {0, 0, 1};
    const auto parent =
        SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const SimpleFluid::PLIC::GeometryPoint anchor{.375, .5, 0};
    const SimpleFluid::PLIC::GeometryHalfspace p{{1, -.25, 0}, anchor, 0};
    const SimpleFluid::PLIC::GeometryHalfspace q{
        {1, -std::nextafter(.25, std::numeric_limits<double>::infinity()), 0}, anchor, 0};
    const SimpleFluid::PLIC::GeometryHalfspace row{{0, 1, 0}, {}, .5};
    const auto old = parent->clip(row, false).inside->clip(p, false);
    const auto next = parent->clip(q, false);
    const std::array pieces{
        old.outside, next.inside->clip(row, false).outside, old.inside, next.outside->clip(row, false).outside};
    std::vector<SimpleFluid::PLIC::DynamicGeometryValue> before{parent->volume()};
    for (const auto& piece : pieces)
    {
        ASSERT_TRUE(piece);
        ASSERT_GT(piece->volume().lower(), 0);
        before.push_back(piece->volume());
    }
    // This infinitesimal temporary wedge occurs while subtracting the upper Q
    // donor from the lower-left P remainder. Public clips certify both pieces;
    // private coverage witnesses retain them without requesting transfer measures.
    const auto measured = old.inside->clip(q, false);
    ASSERT_TRUE(measured.inside);
    ASSERT_TRUE(measured.outside);
    EXPECT_GT(measured.inside->volume().lower(), 0);
    EXPECT_GT(measured.outside->volume().lower(), 0);
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(pieces, 1));
    const auto precheck_measures = SimpleFluid::PLIC::geometry_statistics().volume_evaluations;
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(pieces, 0));
    const auto proof = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(proof.structural_partition_reuses, 0u);
    EXPECT_GT(proof.constructions, 0u);
    EXPECT_EQ(proof.volume_evaluations, precheck_measures);
    const std::span gap(pieces.data(), pieces.size() - 1);
    try
    {
        parent->certify_partition(gap, 0);
        FAIL() << "Every uncovered remainder, including temporary wedges, must be retained.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_STREQ(error.what(), "Bounded FP geometry: partition closure exceeds its physical budget and has a "
                                  "nonempty geometric remainder.");
    }
    // A single physical donor may own several temporary proof fragments. Only
    // its complete parent intersection is published and numerically measured.
    const auto physical = SimpleFluid::PLIC::DynamicConvexPartition(pieces).partition(*parent);
    ASSERT_EQ(physical.size(), pieces.size());
    std::array<bool, 4> seen{};
    std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> certified;
    for (const auto& piece : physical)
    {
        ASSERT_LT(piece.donor, pieces.size());
        EXPECT_FALSE(seen[piece.donor]);
        seen[piece.donor] = true;
        ASSERT_TRUE(piece.geometry);
        const auto measured = piece.geometry->volume();
        EXPECT_GT(measured.lower(), 0);
        const auto original = before[piece.donor + 1];
        EXPECT_LE(std::abs(measured.value - original.value), measured.absolute_error + original.absolute_error);
        certified.push_back(piece.geometry);
    }
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(certified, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().intersections, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().constructions, 0u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().volume_evaluations, 0u);
    EXPECT_EQ(parent->volume().value, before[0].value);
    EXPECT_EQ(parent->volume().absolute_error, before[0].absolute_error);
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
        EXPECT_EQ(pieces[i]->volume().value, before[i + 1].value);
        EXPECT_EQ(pieces[i]->volume().absolute_error, before[i + 1].absolute_error);
    }
}

TEST(BoundedConvexGeometryTest, MomentsAreOptionalAndSourceWorkIsSeparated)
{
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    SimpleFluid::PLIC::reset_geometry_statistics();
    const auto finite = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    (void) finite->volume();
    auto counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(counters.source_imports, 1U);
    EXPECT_EQ(counters.moment_evaluations, 0U);
    (void) finite->moments();
    (void) finite->centroid();
    const auto shifted_moments = finite->moments({.25, .25, .25});
    for (const auto& moment : shifted_moments.first_moment)
        encloses(moment, .25L);
    counters = SimpleFluid::PLIC::geometry_statistics();
    EXPECT_EQ(counters.moment_evaluations, 1U);
    const auto translated = finite->translated({2, 0, 0});
    (void) translated->moments();
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().moment_evaluations, 2U);
    EXPECT_EQ(counters.rational_measures, 0U);
}

TEST(BoundedConvexGeometryTest, ProvenZeroIsDifferentFromUncertainOrientation)
{
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    vertices[1].value = {1, 0, 0};
    vertices[2].value = {0, 1, 0};
    vertices[3].value = {0, 0, 1};
    encloses(SimpleFluid::PLIC::dynamic_orientation(vertices), 1);
    ASSERT_TRUE(SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron(vertices));
    vertices[3] = vertices[0];
    EXPECT_EQ(SimpleFluid::PLIC::dynamic_orientation(vertices).absolute_error, 0);
    EXPECT_FALSE(SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron(vertices));
    vertices[3].absolute_error[2] = 1e-4L;
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron(vertices), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, TinyRepresentableVolumesRemainPositiveAndPublicationRejectsUnderflow)
{
    const auto cell = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1e-200, 1e-200, 1e-200});
    EXPECT_GT(cell->volume().lower(), 0);
    EXPECT_THROW((void) cell->volume().as_double(), std::invalid_argument);
    const auto cut = cell->clip({{1, 0, 0}, {}, 5e-201});
    ASSERT_TRUE(cut.inside);
    EXPECT_GT(cut.inside->volume().lower(), 0);
}

TEST(BoundedConvexGeometryTest, RangeBudgetAndUnsupportedEnvironmentReject)
{
    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    limits.maximum_operations = 1;
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits), std::invalid_argument);
    limits = {};
    limits.byte_budget = 1;
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits), std::invalid_argument);
    EXPECT_THROW((void) cube()->clip({{0, 0, 0}, {}, 0}), std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::box(
                     {0, 0, 0}, {std::numeric_limits<double>::infinity(), 1, 1}),
        std::invalid_argument);
    const int original = std::fegetround();
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    EXPECT_FALSE(SimpleFluid::PLIC::certification_environment());
    EXPECT_THROW((void) cube(), std::invalid_argument);
    ASSERT_EQ(std::fesetround(original), 0);
}

TEST(BoundedConvexGeometryTest, MalformedAndUnresolvedPublicShellsCannotMintPlaneAuthority)
{
    const std::array<SimpleFluid::PLIC::GeometryPoint, 4> tetra{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::make(tetra, {{0, 1, 2}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}),
        std::invalid_argument);
    auto points = cube()->points();
    points[6].absolute_error[0] = 1e-3L;
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(points, cube()->faces()), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, AuthenticatedSupportProvenanceSurvivesAffineWorkAndRejectsMutation)
{
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto finite = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    const auto ids = source->face_geometry(0)->vertex_identities();
    std::vector<SimpleFluid::PLIC::DynamicPoint> points;
    for (const auto& id : ids)
        points.push_back(SimpleFluid::PLIC::dynamic_source_point(*source, id));
    ASSERT_FALSE(points.front().supports().empty());
    for (const auto& p : finite->local_points())
        EXPECT_TRUE(p.supports().empty());
    const auto center = SimpleFluid::PLIC::dynamic_point_average(points);
    EXPECT_FALSE(center.supports().empty());
    const std::array planar{center, points[0], points[1], points[2]};
    EXPECT_EQ(SimpleFluid::PLIC::dynamic_orientation(planar).absolute_error, 0);
    EXPECT_EQ(SimpleFluid::PLIC::dynamic_orientation(planar).value, 0);
    auto changed = center;
    changed.value[0] += .1L;
    EXPECT_TRUE(changed.supports().empty());
    EXPECT_FALSE(SimpleFluid::PLIC::dynamic_points_identical(center, changed));
    const std::array altered{changed, points[0], points[1], points[2]};
    EXPECT_GT(std::abs(SimpleFluid::PLIC::dynamic_orientation(altered).value), 0);
    const std::array constraints{SimpleFluid::PLIC::DynamicFaceConstraint{finite, 0}};
    const auto departure = SimpleFluid::PLIC::dynamic_backtrace_constrained(points[0], {1, 2, 3}, .01, constraints);
    EXPECT_EQ(departure.value[0], points[0].value[0]);
    EXPECT_FALSE(departure.supports().empty());
    const std::array wrong{SimpleFluid::PLIC::DynamicFaceConstraint{finite, 1}};
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_backtrace_constrained(points[0], {1, 2, 3}, .01, wrong),
        std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, NearlyParallelRankTwoWallsRetainExactTransverseCoordinates)
{
    const SimpleFluid::PLIC::GeometryPoint a{.125, -.0250000000000000222045, 0};
    const SimpleFluid::PLIC::GeometryPoint b{.25, -.0499999999999999333866, 0};
    for (const int exponent : {-20, 0, 20})
        for (const bool reverse : {false, true})
        {
            const double scale = std::ldexp(1., exponent);
            const SimpleFluid::PLIC::GeometryPoint anchor{17.1 * scale, -7.5 * scale, 4.5 * scale};
            SimpleFluid::PLIC::DynamicPoint source;
            for (std::size_t axis = 0; axis < 3; ++axis)
                source.value[axis] = anchor[axis];
            const auto box =
                SimpleFluid::PLIC::DynamicConvexGeometry::box({anchor[0] - scale, anchor[1] - scale, anchor[2] - scale},
                    {anchor[0] + scale, anchor[1] + scale, anchor[2] + scale});
            const auto first = box->clip({a, anchor, 0}, false).inside;
            const auto second = box->clip({b, anchor, 0}, false).inside;
            ASSERT_TRUE(first);
            ASSERT_TRUE(second);
            std::array walls{SimpleFluid::PLIC::DynamicFaceConstraint{first, first->faces().size() - 1},
                SimpleFluid::PLIC::DynamicFaceConstraint{second, second->faces().size() - 1}};
            if (reverse)
                std::reverse(walls.begin(), walls.end());
            const std::array repeated{walls[0], walls[1], walls[0], walls[1]};
            const SimpleFluid::PLIC::GeometryPoint velocity{
                .0054681010783616067 * scale, .017501103753775994 * scale, -3.6235787746928473e-5 * scale};
            const double timestep = .002;
            const auto departure = SimpleFluid::PLIC::dynamic_backtrace_constrained(source, velocity, timestep, walls);
            const auto duplicate =
                SimpleFluid::PLIC::dynamic_backtrace_constrained(source, velocity, timestep, repeated);
            ASSERT_EQ(departure.value[0], source.value[0]);
            ASSERT_EQ(departure.value[1], source.value[1]);
            ASSERT_EQ(departure.absolute_error[0], 0);
            ASSERT_EQ(departure.absolute_error[1], 0);
            EXPECT_EQ(duplicate.value, departure.value);
            EXPECT_EQ(duplicate.absolute_error, departure.absolute_error);
            encloses({departure.value[2], departure.absolute_error[2]},
                source.value[2] - static_cast<long double>(velocity[2]) * timestep);

            auto x = source, y = source;
            x.value[0] += .25L * scale;
            y.value[1] += .25L * scale;
            const auto swept = SimpleFluid::PLIC::DynamicConvexGeometry::tetrahedron({source, departure, x, y});
            ASSERT_TRUE(swept);
            // Exact wall tangency leaves a face on this donor boundary; no
            // coordinate tolerance may manufacture a positive outside piece.
            const auto contact = swept->clip({{0, -1, 0}, anchor, 0}, false);
            EXPECT_TRUE(contact.inside);
            EXPECT_FALSE(contact.outside);

            const auto third = box->clip({{0, 0, 1}, anchor, 0}, false).inside;
            ASSERT_TRUE(third);
            const std::array corner{
                walls[0], walls[1], SimpleFluid::PLIC::DynamicFaceConstraint{third, third->faces().size() - 1}};
            const auto stationary =
                SimpleFluid::PLIC::dynamic_backtrace_constrained(source, velocity, timestep, corner);
            EXPECT_EQ(stationary.value, source.value);
            EXPECT_EQ(stationary.absolute_error, source.absolute_error);
        }
}

TEST(BoundedConvexGeometryTest, ConstrainedBacktraceAuthenticatesTheOffsetPlaneBeforeZeroOrNonzeroMotion)
{
    const auto wall = cube()->clip({{1, 0, 0}, {}, .25}, false).inside;
    ASSERT_TRUE(wall);
    const std::array constraints{SimpleFluid::PLIC::DynamicFaceConstraint{wall, wall->faces().size() - 1}};
    SimpleFluid::PLIC::DynamicPoint correct, wrong;
    correct.value = {.25L, .5L, .5L};
    wrong.value = {0, .5L, .5L};
    const auto departure =
        SimpleFluid::PLIC::dynamic_backtrace_constrained(correct, {.5, .25, .125}, .125, constraints);
    EXPECT_EQ(departure.value, (std::array<long double, 3>{.25L, .46875L, .484375L}));
    EXPECT_EQ(departure.absolute_error, (std::array<long double, 3>{0, 0, 0}));
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_backtrace_constrained(wrong, {.5, .25, .125}, .125, constraints),
        std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_backtrace_constrained(wrong, {0, 0, 0}, .125, constraints),
        std::invalid_argument);
    const auto stationary = SimpleFluid::PLIC::dynamic_backtrace_constrained(correct, {0, 0, 0}, .125, constraints);
    EXPECT_EQ(stationary.value, correct.value);
    EXPECT_EQ(stationary.absolute_error, correct.absolute_error);
}

TEST(BoundedConvexGeometryTest, NumericalPacketsRequireMatchingPinnedSourceRegistry)
{
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto finite = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    const auto cut = finite->clip({{3, 2, 1}, {}, 2.25}, false);
    ASSERT_TRUE(cut.inside);
    const SimpleFluid::PLIC::GeometryPacketContext context{{17, 2, 3, 11}};
    const auto packet = cut.inside->encode(context);
    const std::array registry{finite};
    const auto decoded = SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, registry);
    EXPECT_LE(std::abs(decoded->volume().value - cut.inside->volume().value),
        decoded->volume().absolute_error + cut.inside->volume().absolute_error);
    auto stale = context;
    ++stale.publication[3];
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, stale, registry), std::invalid_argument);
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, {}), std::invalid_argument);
    const auto foreign_source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({2, 2, 2}, {3, 3, 3});
    const std::array foreign{SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*foreign_source)};
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, foreign), std::invalid_argument);
    auto damaged = packet;
    damaged.pop_back();
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(damaged, context, registry), std::invalid_argument);
    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    limits.byte_budget = packet.size() - 1;
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(packet, context, registry, limits),
        std::invalid_argument);
    auto trailing = packet;
    trailing.push_back(0);
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::decode(trailing, context, registry), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, NonbinaryEdgeParameterRetainsExactRepresentableIntersection)
{
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {3, 1, 1});
    const auto cut = parent->clip({{1, 0, 0}, {}, 1}, false);
    ASSERT_TRUE(cut.inside);
    for (const auto& p : cut.inside->points())
        EXPECT_EQ(p.absolute_error[0], 0);
    const auto oblique = cut.inside->clip({{1, 0, -1}, {}, 0}, false);
    ASSERT_TRUE(oblique.inside);
    encloses(oblique.inside->volume(), .5L);
}

TEST(BoundedConvexGeometryTest, TangentSectionComposesAtomicFacesAndRetainsCollinearEdges)
{
    std::vector<SimpleFluid::PLIC::GeometryPoint> vertices;
    for (unsigned i = 0; i < 8; ++i)
        vertices.push_back({double(i & 1), double((i >> 1) & 1), double((i >> 2) & 1)});
    const Faces tiled{{0, 4, 6}, {0, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    const auto atomic = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, tiled);
    const auto contact = atomic->clip({{1, 0, 0}, {}, 0});
    EXPECT_FALSE(contact.inside);
    encloses(contact.section.area, 1);
    EXPECT_EQ(contact.section.vertices.size(), 4U);
    vertices.push_back({0, 0, .5});
    const Faces collinear{{0, 8, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4, 8}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    const auto propagated = SimpleFluid::PLIC::DynamicConvexGeometry::make(vertices, collinear);
    const auto edge = propagated->clip({{1, 1, 0}, {}, 0});
    EXPECT_FALSE(edge.inside);
    EXPECT_EQ(edge.section.vertices.size(), 3U);
    EXPECT_EQ(edge.section.area.value, 0);
    EXPECT_EQ(edge.section.area.absolute_error, 0);
    const auto split = propagated->clip({{1, -1, 0}, {}, 0});
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    encloses(split.inside->volume(), .5L);
    encloses(split.outside->volume(), .5L);
    encloses(split.section.area, std::sqrt(2.L));
    const std::array pieces{split.inside, split.outside};
    EXPECT_NO_THROW(propagated->certify_partition(pieces, 8e-15));
}

TEST(BoundedConvexGeometryTest, RawShellCannotDoubleCountDisconnectedCoincidentComponents)
{
    std::vector<SimpleFluid::PLIC::GeometryPoint> vertices;
    for (unsigned i = 0; i < 8; ++i)
        vertices.push_back({double(i & 1), double((i >> 1) & 1), double((i >> 2) & 1)});
    const Faces cube_faces{{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    auto unused = vertices;
    unused.push_back({.5, .5, .5});
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::make(unused, cube_faces), std::invalid_argument);
    auto duplicated = vertices;
    duplicated.insert(duplicated.end(), vertices.begin(), vertices.end());
    auto faces = cube_faces;
    for (auto face : cube_faces)
    {
        for (auto& index : face)
            index += 8;
        faces.push_back(std::move(face));
    }
    EXPECT_THROW((void) SimpleFluid::PLIC::DynamicConvexGeometry::make(duplicated, faces), std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, DecodedNonbinaryComplementsRetainOriginalPlaneIncidence)
{
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    const auto split = parent->clip({{3, 0, 0}, {}, 1}, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const SimpleFluid::PLIC::GeometryPacketContext context{{9, 8, 7, 6}};
    const std::array registry{parent};
    const auto inside =
        SimpleFluid::PLIC::DynamicConvexGeometry::decode(split.inside->encode(context), context, registry);
    const auto outside =
        SimpleFluid::PLIC::DynamicConvexGeometry::decode(split.outside->encode(context), context, registry);
    EXPECT_FALSE(inside->intersect(*outside));
    const std::array pieces{inside, outside};
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
}

TEST(BoundedConvexGeometryTest, HangingSourcePointCarriesTheCompleteIncidentFanSupports)
{
    const auto coarse =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto fine =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {.5, .5, .5});
    std::vector<char> point_identity;
    for (const auto& identity : fine->face_geometry(4)->vertex_identities())
    {
        const auto p = SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Ordinary>(identity);
        if (p == SimpleFluid::PLIC::GeometryPoint{.5, .5, 0})
            point_identity = identity;
    }
    ASSERT_FALSE(point_identity.empty());
    const auto coarse_only = SimpleFluid::PLIC::dynamic_source_point(*coarse, point_identity);
    const std::array sources{coarse, fine};
    const auto complete = SimpleFluid::PLIC::dynamic_source_point(sources, point_identity);
    EXPECT_GT(complete.supports().size(), coarse_only.supports().size());
    EXPECT_EQ(complete.value, coarse_only.value);
    EXPECT_EQ(complete.absolute_error, coarse_only.absolute_error);
    EXPECT_THROW((void) SimpleFluid::PLIC::dynamic_source_point(
                     std::span<const SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>>{},
                     point_identity),
        std::invalid_argument);
}

TEST(BoundedConvexGeometryTest, StructuralProofRejectsAncestorOverlapAndChargesItsWork)
{
    const auto parent = cube();
    const auto split = parent->clip({{1, 0, 0}, {}, .5}, false);
    const auto nested = split.inside->clip({{0, 1, 0}, {}, .5}, false);
    const std::array overlap{split.inside, nested.inside, nested.outside, split.outside};
    EXPECT_THROW(parent->certify_partition(overlap, 8e-15), std::invalid_argument);
    auto limits = SimpleFluid::PLIC::GeometryLimits{};
    // Each four-edge cuboid cut uses 12 directed-boundary visits. An eight-leaf
    // complete complement tree needs 15 proof-node visits, exceeding that cap.
    limits.maximum_pair_tests = 12;
    const auto limited = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1}, limits);
    std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> complete{limited};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        SimpleFluid::PLIC::GeometryPoint normal{};
        normal[axis] = 1;
        std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> children;
        for (const auto& piece : complete)
        {
            const auto cut = piece->clip({normal, {}, .5}, false);
            ASSERT_TRUE(cut.inside);
            ASSERT_TRUE(cut.outside);
            children.push_back(cut.inside);
            children.push_back(cut.outside);
        }
        complete = std::move(children);
    }
    ASSERT_EQ(complete.size(), 8u);
    const auto before = limited->volume();
    SimpleFluid::PLIC::reset_geometry_statistics();
    try
    {
        limited->certify_partition(complete, 0);
        FAIL() << "The complete structural proof must honor its shared pair cap.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_STREQ(error.what(), "Bounded FP geometry: pair-test cap exhausted.");
    }
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    EXPECT_EQ(limited->volume().value, before.value);
    EXPECT_EQ(limited->volume().absolute_error, before.absolute_error);
    for (const auto& piece : complete)
        encloses(piece->volume(), .125L);
}

TEST(BoundedConvexGeometryTest, RepeatedAuthenticatedNonbinaryWallSupportDoesNotCreateANewRank)
{
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1})
            ->clip({{3, 2, 0}, {}, 1})
            .inside;
    ASSERT_TRUE(source);
    std::size_t cut_face = source->num_faces();
    for (std::size_t f = 0; f < source->num_faces(); ++f)
    {
        const auto normal = source->face_geometry(f)->area_vector();
        if (normal.x > 0 && normal.y > 0)
            cut_face = f;
    }
    ASSERT_LT(cut_face, source->num_faces());
    const auto finite = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    const auto identity = source->face_geometry(cut_face)->vertex_identities().front();
    const auto point = SimpleFluid::PLIC::dynamic_source_point(*source, identity);
    const std::array single{SimpleFluid::PLIC::DynamicFaceConstraint{finite, cut_face}};
    const std::array repeated{single[0], single[0], single[0], single[0]};
    const auto expected = SimpleFluid::PLIC::dynamic_backtrace_constrained(point, {.4, .7, .2}, .03, single);
    const auto actual = SimpleFluid::PLIC::dynamic_backtrace_constrained(point, {.4, .7, .2}, .03, repeated);
    EXPECT_EQ(actual.value, expected.value);
    EXPECT_EQ(actual.absolute_error, expected.absolute_error);
    EXPECT_EQ(actual.supports(), expected.supports());
    const auto stationary = SimpleFluid::PLIC::dynamic_backtrace_constrained(point, {0, 0, 0}, .03, repeated);
    EXPECT_TRUE(SimpleFluid::PLIC::dynamic_points_identical(point, stationary));
    EXPECT_EQ(stationary.supports(), point.supports());
    EXPECT_EQ(stationary.value, point.value);
    EXPECT_EQ(stationary.absolute_error, point.absolute_error);
}

TEST(BoundedConvexGeometryTest, AllAuthenticatedCornerWallsRemainSatisfiedAfterProvenRankThree)
{
    const std::array<SimpleFluid::PLIC::GeometryPoint, 6> vertices{
        {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    const Faces faces{{4, 0, 2}, {4, 2, 1}, {4, 1, 3}, {4, 3, 0}, {5, 2, 0}, {5, 1, 2}, {5, 3, 1}, {5, 0, 3}};
    const auto source =
        SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::make(vertices, faces);
    const auto finite = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
    std::vector<char> identity;
    for (const auto& point : source->face_geometry(0)->vertex_identities())
        if (SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Ordinary>(point) ==
            SimpleFluid::PLIC::GeometryPoint{0, 0, 1})
            identity = point;
    ASSERT_FALSE(identity.empty());
    std::vector<SimpleFluid::PLIC::DynamicFaceConstraint> walls;
    for (std::size_t f = 0; f < source->num_faces(); ++f)
        if (source->face_geometry(f)->contains_point(identity))
            walls.push_back({finite, f});
    ASSERT_EQ(walls.size(), 4u);
    const auto point = SimpleFluid::PLIC::dynamic_source_point(*source, identity);
    const auto departure = SimpleFluid::PLIC::dynamic_backtrace_constrained(point, {.1, .2, .3}, .01, walls);
    EXPECT_TRUE(SimpleFluid::PLIC::dynamic_points_identical(point, departure));
    EXPECT_EQ(departure.value, point.value);
    EXPECT_EQ(departure.absolute_error, point.absolute_error);
}

TEST(BoundedConvexGeometryTest, FarSupportSlabRetainsTheUnchangedMinorityRelativeBudget)
{
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {1, 1, 1});
    const double thickness = 1e-6;
    for (const bool reverse : {false, true})
    {
        const SimpleFluid::PLIC::GeometryHalfspace plane{
            {reverse ? -1. : 1., 0, 0}, {1, 0, 0}, reverse ? thickness : -thickness};
        const auto split = parent->clip(plane, false);
        const auto thin = reverse ? split.inside : split.outside;
        const auto bulk = reverse ? split.outside : split.inside;
        ASSERT_TRUE(thin);
        ASSERT_TRUE(bulk);
        const auto volume = thin->volume();
        encloses(volume, static_cast<long double>(thickness));
        const auto complete_error =
            std::abs(volume.value - static_cast<long double>(thickness)) + volume.absolute_error;
        EXPECT_LE(complete_error, static_cast<long double>(thickness) * 8e-15L);
        EXPECT_TRUE(volume.relative_error_at_most(thickness, 8e-15));
        const std::array pieces{split.inside, split.outside};
        EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
        EXPECT_EQ(thin->origin(), parent->origin());
    }
}

TEST(BoundedConvexGeometryTest, RebasedThinComplementsKeepPhysicalReferenceAndPacketAccuracy)
{
    const SimpleFluid::PLIC::GeometryPacketContext context{{4, 3, 2, 1}};
    for (const int exponent : {-20, 0, 20})
    {
        const double scale = std::ldexp(1., exponent);
        const SimpleFluid::PLIC::GeometryPoint lower{16 * scale, -8 * scale, 4 * scale};
        const SimpleFluid::PLIC::GeometryPoint upper{17 * scale, -7 * scale, 5 * scale};
        const auto source =
            SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box(lower, upper);
        const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::from_geometry(*source);
        const std::array registry{parent};
        const double thickness = 1e-6 * scale;
        const auto oracle = static_cast<long double>(thickness) * scale * scale;
        for (const bool reverse : {false, true})
        {
            const SimpleFluid::PLIC::GeometryHalfspace plane{
                {reverse ? -1. : 1., 0, 0}, {upper[0], lower[1], lower[2]}, reverse ? thickness : -thickness};
            const auto split = parent->clip(plane, false);
            const auto thin = reverse ? split.inside : split.outside;
            ASSERT_TRUE(thin);
            EXPECT_EQ(thin->origin(), parent->origin());
            const auto check = [&](const SimpleFluid::PLIC::DynamicConvexGeometryPtr& geometry)
            {
                const auto volume = geometry->volume();
                encloses(volume, oracle);
                EXPECT_LE(std::abs(volume.value - oracle) + volume.absolute_error, oracle * 8e-15L);
                EXPECT_TRUE(volume.relative_error_at_most(oracle, 8e-15));
                EXPECT_EQ(geometry->moments().origin, parent->origin());
                const auto moment = geometry->moments(lower);
                encloses(moment.first_moment[0], oracle * (scale - static_cast<long double>(thickness) / 2));
                encloses(moment.first_moment[1], oracle * scale / 2);
                encloses(moment.first_moment[2], oracle * scale / 2);
                const auto local = geometry->local_points();
                ASSERT_EQ(local.size(), geometry->points().size());
                for (std::size_t i = 0; i < local.size(); ++i)
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        const auto global = geometry->points()[i];
                        const auto difference = global.value[axis] - geometry->origin()[axis];
                        EXPECT_LE(std::abs(local[i].value[axis] - difference),
                            local[i].absolute_error[axis] + global.absolute_error[axis]);
                    }
            };
            check(thin);
            const auto decoded =
                SimpleFluid::PLIC::DynamicConvexGeometry::decode(thin->encode(context), context, registry);
            check(decoded);
            const auto repeated = decoded->clip(plane, false);
            EXPECT_EQ(static_cast<bool>(repeated.inside), reverse);
            EXPECT_EQ(static_cast<bool>(repeated.outside), !reverse);
            const std::array pieces{split.inside, split.outside};
            EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
        }
    }
}

TEST(BoundedConvexGeometryTest, CrossFrameSourceFaceRetainsThinFragmentCoordinates)
{
    const auto parent = cube();
    const double thickness = 1e-6;
    const auto slab = parent->clip({{1, 0, 0}, {1, 0, 0}, -thickness}, false).outside;
    ASSERT_TRUE(slab);
    const double cut_x = 1 - 5e-7;
    const auto cutter = SimpleFluid::PLIC::DynamicConvexGeometry::box({0, 0, 0}, {cut_x, 1, 1});
    std::size_t face = cutter->faces().size();
    for (std::size_t f = 0; f < cutter->faces().size(); ++f)
        if (std::all_of(cutter->faces()[f].begin(), cutter->faces()[f].end(),
                [&](auto i) { return cutter->points()[i].value[0] == cut_x; }))
            face = f;
    ASSERT_LT(face, cutter->faces().size());
    const auto split = slab->clip_by_face(*cutter, face, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const auto right = 1.L - static_cast<long double>(cut_x);
    const std::array expected{static_cast<long double>(thickness) - right, right};
    const std::array pieces{split.inside, split.outside};
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
        const auto volume = pieces[i]->volume();
        encloses(volume, expected[i]);
        EXPECT_LE(std::abs(volume.value - expected[i]) + volume.absolute_error, expected[i] * 8e-15L);
    }
    EXPECT_NO_THROW(slab->certify_partition(pieces, 8e-15));
}

TEST(BoundedConvexGeometryTest, CapturedThinTetrahedronRetainsPositiveAnchoredVolumeAndMoments)
{
    const long double captured[4][3][2]{
        {{0L, 0L}, {0L, 0L}, {0L, 0L}},
        {{-9.88731491447778268987e-05L, -9.88731491447768511167e-05L},
            {6.54208209137720794572e-05L, 6.54208209137724047179e-05L},
            {-1.68758301665101508038e-07L, -1.68758301664993087821e-07L}},
        {{-0.100129270305452642247L, -0.100129270305452642192L}, {-0.5L, -0.5L},
            {0.499999194638217581913L, 0.499999194638217582022L}},
        {{0.100015724107307711133L, 0.100015724107307711242L},
            {0.500111431056786749833L, 0.50011143105678675005L},
            {0.500000130328575757735L, 0.500000130328575757952L}}};
    const SimpleFluid::PLIC::GeometryPoint origin{.3, .25, .25};
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex)
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto lower = captured[vertex][axis][0], upper = captured[vertex][axis][1];
            const auto middle = std::midpoint(lower, upper);
            const auto radius = std::max(middle - lower, upper - middle);
            vertices[vertex].value[axis] = std::ldexp(middle, -1);
            vertices[vertex].absolute_error[axis] = radius == 0 ? 0 :
                std::nextafter(std::ldexp(radius, -1), std::numeric_limits<long double>::infinity());
        }
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        origin, vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto before = parent->volume();
    const auto above = parent->clip({{0, 0, -1}, origin, -.25}, false).inside;
    ASSERT_TRUE(above);
    const auto complementary = above->clip({{0, 1, 0}, origin, .25}, false);
    ASSERT_TRUE(complementary.inside);
    ASSERT_TRUE(complementary.outside);
    ASSERT_EQ(complementary.outside->points().size(), 6u);
    EXPECT_GT(complementary.outside->volume().lower(), 0);
    const std::array cover{complementary.inside, complementary.outside};
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(above->certify_partition(cover, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    const auto donor = SimpleFluid::PLIC::DynamicConvexGeometry::box({-1, -1, -1}, {1, .5, 1});
    const auto result = above->intersect(*donor);
    ASSERT_TRUE(result);
    ASSERT_EQ(result->points().size(), 4u);
    const SimpleFluid::PLIC::DynamicPoint* low = nullptr;
    const SimpleFluid::PLIC::DynamicPoint* high = nullptr;
    std::vector<const SimpleFluid::PLIC::DynamicPoint*> edge;
    std::array<long double, 3> centroid{};
    for (const auto& point : result->points())
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
            centroid[axis] += point.value[axis] / 4;
        if (point.value[1] < .5L)
            low = &point;
        else if (point.value[2] > .5L)
            high = &point;
        else
            edge.push_back(&point);
    }
    ASSERT_TRUE(low);
    ASSERT_TRUE(high);
    ASSERT_EQ(edge.size(), 2u);
    EXPECT_EQ(low->value[2], .5L);
    EXPECT_EQ(low->absolute_error[2], 0);
    EXPECT_EQ(high->value[1], .5L);
    EXPECT_EQ(high->absolute_error[1], 0);
    for (const auto* point : edge)
        for (const auto axis : {1u, 2u})
        {
            EXPECT_EQ(point->value[axis], .5L);
            EXPECT_EQ(point->absolute_error[axis], 0);
        }
    // Three vertices have y=.5; three have z=.5. The determinant factors
    // into the x-edge length and the independent y/z heights.
    const auto expected = std::abs(edge[0]->value[0] - edge[1]->value[0]) *
                          (.5L - low->value[1]) * (high->value[2] - .5L) / 6;
    ASSERT_GT(expected, 0);
    const auto volume = result->volume();
    EXPECT_GT(volume.lower(), 0);
    EXPECT_LT(volume.upper(), 1e-18L);
    encloses(volume, expected);
    const auto moments = result->moments(origin);
    for (std::size_t axis = 0; axis < 3; ++axis)
        encloses(moments.first_moment[axis], expected * (centroid[axis] - origin[axis]));
    const auto whole = above->moments(origin);
    const auto inside = complementary.inside->moments(origin);
    const auto outside = complementary.outside->moments(origin);
    const auto conserved = [](const SimpleFluid::PLIC::DynamicGeometryValue& total,
                               const SimpleFluid::PLIC::DynamicGeometryValue& first,
                               const SimpleFluid::PLIC::DynamicGeometryValue& second)
    {
        EXPECT_LE(std::nextafter(first.lower() + second.lower(), -std::numeric_limits<long double>::infinity()),
            total.upper());
        EXPECT_GE(std::nextafter(first.upper() + second.upper(), std::numeric_limits<long double>::infinity()),
            total.lower());
    };
    conserved(whole.volume, inside.volume, outside.volume);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        EXPECT_GT(outside.first_moment[axis].lower(), 0);
        conserved(whole.first_moment[axis], inside.first_moment[axis], outside.first_moment[axis]);
    }
    EXPECT_EQ(parent->volume().value, before.value);
    EXPECT_EQ(parent->volume().absolute_error, before.absolute_error);
}

TEST(BoundedConvexGeometryTest, CapturedSourceFaceFactorsAStrictNonzeroNearCoplanarResidual)
{
    const long double captured[4][3][2]{
        {{1.38777878078144567553e-17L, 1.38777878078144567553e-17L}, {0L, 0L}, {0L, 0L}},
        {{-.349999999999999977796L, -.349999999999999977796L}, {-0.5L, -0.5L}, {0L, 0L}},
        {{-.349999999999999977796L, -.349999999999999977796L}, {-0.5L, -0.5L},
            {-5.91842560168116945882e-06L, -5.9184256016810610386e-06L}},
        {{.149870729694547357753L, .149870729694547357808L}, {-0.5L, -0.5L},
            {-8.05361782418086580987e-07L, -8.0536178241797816077e-07L}}};
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex)
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto lower = captured[vertex][axis][0], upper = captured[vertex][axis][1];
            const auto middle = std::midpoint(lower, upper);
            const auto radius = std::max(middle - lower, upper - middle);
            vertices[vertex].value[axis] = std::ldexp(middle, -1);
            vertices[vertex].absolute_error[axis] = radius == 0 ? 0 :
                std::nextafter(std::ldexp(radius, -1), std::numeric_limits<long double>::infinity());
        }
    const auto source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        {.175, .25, .5}, vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const std::array<SimpleFluid::PLIC::GeometryPoint, 8> corners{
        {{0, 0, 0}, {.25, 0, 0}, {.35, .5, 0}, {.1, .5, 0}, {0, 0, .5}, {.25, 0, .5}, {.35, .5, .5}, {.1, .5, .5}}};
    const auto donor = SimpleFluid::PLIC::DynamicConvexGeometry::make(
        corners, {{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}});
    const auto source_before = source->volume(), donor_before = donor->volume();
    // The preceding swept tetrahedron cuts out the actual six-vertex overlay
    // atom. Its lower (.35,.5,z) corner retains uncertainty in z, although the
    // next source plane has an identically zero z coefficient.
    const long double departed[3][2]{
        {-1.6985112052913864172e-05L, -1.69851120529137015417e-05L},
        {8.48649516132128503912e-05L, 8.48649516132129588114e-05L},
        {-1.02794537218217521372e-05L, -1.02794537218207763553e-05L}};
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> old_vertices{vertices[0], {}, vertices[3], vertices[2]};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto lower = departed[axis][0], upper = departed[axis][1];
        const auto middle = std::midpoint(lower, upper);
        old_vertices[1].value[axis] = std::ldexp(middle, -1);
        old_vertices[1].absolute_error[axis] =
            std::nextafter(std::ldexp(std::max(middle - lower, upper - middle), -1),
                std::numeric_limits<long double>::infinity());
    }
    const auto old_source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        {.175, .25, .5}, old_vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto atom = donor->clip_by_face(*old_source, 0).outside;
    ASSERT_TRUE(atom);
    ASSERT_EQ(atom->points().size(), 6u);
    const auto is_lower_corner = [](const SimpleFluid::PLIC::DynamicPoint& point)
    { return point.value[0] == .35 && point.value[1] == .5 && point.value[2] < .5 && point.absolute_error[2] > 0; };
    ASSERT_TRUE(std::any_of(atom->points().begin(), atom->points().end(), is_lower_corner));
    const auto atom_before = atom->volume();
    // Face 0 contains (0,0,.5), (0,0,.5+h), and (sx,.25,.5).
    // At q=(.35,.5,.5), its residual is h*(.35-2*sx)/4>0:
    // h<0 and .35-2*sx=-2^-56 exactly. It is not a contact.
    const auto sx = static_cast<long double>(.175) + std::ldexp(1.L, -57);
    EXPECT_EQ(static_cast<long double>(.35) - 2 * sx, -std::ldexp(1.L, -56));
    SimpleFluid::PLIC::DynamicConvexGeometryPtr descendant;
    ASSERT_NO_THROW(descendant = source->clip({{0, 0, -1}, {0, 0, .5}, std::ldexp(1., -22)}, false).inside);
    ASSERT_TRUE(descendant);
    ASSERT_EQ(descendant->faces()[0].size(), 4u);
    for (const auto& support : {source, descendant})
        for (const auto& target : {donor, atom})
        {
            SCOPED_TRACE(::testing::Message() << "atom=" << (target == atom)
                                             << " descendant=" << (support == descendant));
            SimpleFluid::PLIC::DynamicClipResult result;
            ASSERT_NO_THROW(result = target->clip_by_face(*support, 0));
            ASSERT_TRUE(result.inside);
            ASSERT_TRUE(result.outside);
            EXPECT_GT(result.inside->volume().lower(), 0);
            EXPECT_GT(result.outside->volume().lower(), 0);
            const auto is_corner = [](const SimpleFluid::PLIC::DynamicPoint& point)
            {
                return point.value == std::array<long double, 3>{.35, .5, .5} &&
                       point.absolute_error == std::array<long double, 3>{};
            };
            EXPECT_FALSE(std::any_of(result.inside->points().begin(), result.inside->points().end(), is_corner));
            EXPECT_TRUE(std::any_of(result.outside->points().begin(), result.outside->points().end(), is_corner));
            if (target == atom)
            {
                EXPECT_FALSE(std::any_of(
                    result.inside->points().begin(), result.inside->points().end(), is_lower_corner));
                EXPECT_TRUE(std::any_of(
                    result.outside->points().begin(), result.outside->points().end(), is_lower_corner));
            }
            const auto repeated = result.inside->clip_by_face(*support, 0);
            ASSERT_TRUE(repeated.inside);
            EXPECT_FALSE(repeated.outside);
            EXPECT_EQ(repeated.inside->volume().value, result.inside->volume().value);
            EXPECT_EQ(repeated.inside->volume().absolute_error, result.inside->volume().absolute_error);
            auto remaining = result.inside;
            for (std::size_t face = 1; face < 4; ++face)
            {
                SCOPED_TRACE(::testing::Message() << "face=" << face);
                const auto before = remaining->volume();
                if (target == atom && face == 2)
                {
                    // Independent relative imports do not certify that their
                    // overlapping uncertain departed vertices are identical.
                    EXPECT_THROW((void) remaining->clip_by_face(*source, face), std::invalid_argument);
                    EXPECT_EQ(remaining->volume().value, before.value);
                    EXPECT_EQ(remaining->volume().absolute_error, before.absolute_error);
                    break;
                }
                SimpleFluid::PLIC::DynamicClipResult next;
                ASSERT_NO_THROW(next = remaining->clip_by_face(*source, face));
                ASSERT_TRUE(next.inside);
                EXPECT_GT(next.inside->volume().lower(), 0);
                if (face == 1)
                {
                    // The common z=.5/face0 line extends beyond the source
                    // edge, whose endpoints have y=0 and y=.25. Incidence on
                    // that infinite line proves face1 contact at y near .5.
                    const auto beyond_edge = [](const SimpleFluid::PLIC::DynamicPoint& point)
                    { return point.value[1] > .4 && point.value[2] == .5 && point.absolute_error[2] == 0; };
                    EXPECT_TRUE(std::any_of(
                        next.inside->points().begin(), next.inside->points().end(), beyond_edge));
                }
                SimpleFluid::PLIC::DynamicClipResult repeat;
                ASSERT_NO_THROW(repeat = next.inside->clip_by_face(*source, face));
                ASSERT_TRUE(repeat.inside);
                EXPECT_FALSE(repeat.outside);
                EXPECT_EQ(repeat.inside->volume().value, next.inside->volume().value);
                EXPECT_EQ(repeat.inside->volume().absolute_error, next.inside->volume().absolute_error);
                EXPECT_EQ(remaining->volume().value, before.value);
                EXPECT_EQ(remaining->volume().absolute_error, before.absolute_error);
                remaining = next.inside;
            }
        }
    // Global imports can retain the identity of the same sealed input point.
    // Construct each point once and share those exact immutable copies across
    // the old and new swept tetrahedra; no coordinate uncertainty is removed.
    const SimpleFluid::PLIC::DynamicPoint origin({.175, .25, .5});
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> global;
    for (std::size_t i = 0; i < global.size(); ++i)
        global[i] = SimpleFluid::PLIC::dynamic_point_displaced(vertices[i], origin);
    const std::array old_global{
        global[0], SimpleFluid::PLIC::dynamic_point_displaced(old_vertices[1], origin), global[3], global[2]};
    const Faces tetra_faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const auto shared_source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(global, tetra_faces);
    const auto shared_old = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(old_global, tetra_faces);
    const auto shared_source_before = shared_source->volume(), shared_old_before = shared_old->volume();
    const auto shifted = shared_source->translated({0, 0, 1e-6});
    // Translation clears both the moved vertex IDs and the moved planes'
    // issued IDs. Otherwise these formerly shared face vertices would create
    // a false contact section across the now strictly separated planes.
    for (const auto& separated :
        {shared_old->clip_by_face(*shifted, 2, true), shifted->clip_by_face(*shared_old, 2, true)})
    {
        EXPECT_FALSE(separated.inside);
        ASSERT_TRUE(separated.outside);
        EXPECT_TRUE(separated.section.vertices.empty());
    }
    const auto shared_atom = donor->clip_by_face(*shared_old, 0).outside;
    ASSERT_TRUE(shared_atom);
    ASSERT_EQ(shared_atom->points().size(), 6u);
    const auto shared_atom_before = shared_atom->volume();
    // An even permutation moves the old face0 to face1 with the same normal.
    // Its three shared input points certify the whole supporting plane, even
    // for the atom's constructed lower corner without an input-point ID.
    const std::array permuted{old_global[0], old_global[2], old_global[3], old_global[1]};
    const auto same_plane = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(permuted, tetra_faces);
    const auto same = shared_atom->clip_by_face(*same_plane, 1);
    EXPECT_FALSE(same.inside);
    ASSERT_TRUE(same.outside);
    EXPECT_EQ(same.outside->volume().value, shared_atom_before.value);
    EXPECT_EQ(same.outside->volume().absolute_error, shared_atom_before.absolute_error);
    const long double next_departed[3][2]{
        {.350015724107307711133L, .350015724107307711242L},
        {.500111431056786749833L, .50011143105678675005L},
        {1.30328575757734758944e-07L, 1.30328575757951599379e-07L}};
    SimpleFluid::PLIC::DynamicPoint next_local;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto lower = next_departed[axis][0], upper = next_departed[axis][1];
        const auto middle = std::midpoint(lower, upper);
        next_local.value[axis] = std::ldexp(middle, -1);
        next_local.absolute_error[axis] =
            std::nextafter(std::ldexp(std::max(middle - lower, upper - middle), -1),
                std::numeric_limits<long double>::infinity());
    }
    // The next swept tetrahedron has the same three face points in reverse
    // order. Zero incidence transfers regardless of the halfspace orientation.
    const std::array next_global{
        old_global[0], old_global[1], SimpleFluid::PLIC::dynamic_point_displaced(next_local, origin), old_global[2]};
    const auto reversed_plane = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(next_global, tetra_faces);
    const auto reversed = shared_atom->clip_by_face(*reversed_plane, 1);
    ASSERT_TRUE(reversed.inside);
    EXPECT_FALSE(reversed.outside);
    EXPECT_EQ(reversed.inside->volume().value, shared_atom_before.value);
    EXPECT_EQ(reversed.inside->volume().absolute_error, shared_atom_before.absolute_error);
    EXPECT_TRUE(std::any_of(
        reversed.inside->points().begin(), reversed.inside->points().end(), is_lower_corner));
    const auto repeated_plane = reversed.inside->clip_by_face(*reversed_plane, 1);
    ASSERT_TRUE(repeated_plane.inside);
    EXPECT_FALSE(repeated_plane.outside);
    EXPECT_EQ(shared_atom->volume().value, shared_atom_before.value);
    EXPECT_EQ(shared_atom->volume().absolute_error, shared_atom_before.absolute_error);
    // P and Q share center/departed-corner identities, but their other two
    // native corners have a real binary64 quad defect. At p, Q(p) factors as
    // -h*((.25+.1)-2*sx)/4<0, with h>0 and the bracket exactly 2^-56.
    const std::array p_input{SimpleFluid::PLIC::DynamicPoint({.25, 0, .5})},
        q_input{SimpleFluid::PLIC::DynamicPoint({.1, .5, .5})};
    const auto p = SimpleFluid::PLIC::dynamic_point_average(p_input),
               q = SimpleFluid::PLIC::dynamic_point_average(q_input);
    EXPECT_EQ((p.value[0] + q.value[0]) - 2 * sx, std::ldexp(1.L, -56));
    const auto below =
        SimpleFluid::PLIC::dynamic_point_displaced(global[0], SimpleFluid::PLIC::DynamicPoint({0, 0, -.01}));
    const auto& common_departed = next_global[2];
    const auto negative_residual = SimpleFluid::PLIC::dynamic_orientation({global[0], common_departed, q, p});
    EXPECT_LT(negative_residual.upper(), 0);
    const std::array p_vertices{p, global[0], common_departed, below};
    const long double q_departed[3][2]{
        {-.149954394250066746372L, -.149954394250066746344L},
        {.500228028749666101568L, .500228028749666101785L},
        {-5.89517494759938653209e-07L, -5.89517494759830232992e-07L}};
    SimpleFluid::PLIC::DynamicPoint q_local;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto lower = q_departed[axis][0], upper = q_departed[axis][1];
        const auto middle = std::midpoint(lower, upper);
        q_local.value[axis] = std::ldexp(middle, -1);
        q_local.absolute_error[axis] =
            std::nextafter(std::ldexp(std::max(middle - lower, upper - middle), -1),
                std::numeric_limits<long double>::infinity());
    }
    const std::array q_vertices{
        global[0], q, common_departed, SimpleFluid::PLIC::dynamic_point_displaced(q_local, origin)};
    const auto p_source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(p_vertices, tetra_faces);
    const auto q_source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(q_vertices, tetra_faces);
    SimpleFluid::PLIC::DynamicConvexGeometryPtr sliver;
    ASSERT_NO_THROW(sliver = shared_atom->clip_by_face(*p_source, 0).outside);
    ASSERT_TRUE(sliver);
    EXPECT_GT(sliver->volume().lower(), 0);
    const auto sliver_before = sliver->volume();
    SimpleFluid::PLIC::DynamicClipResult q_cut;
    ASSERT_NO_THROW(q_cut = sliver->clip_by_face(*q_source, 0));
    ASSERT_TRUE(q_cut.inside);
    ASSERT_TRUE(q_cut.outside);
    const auto tiny = q_cut.inside->volume();
    EXPECT_EQ(tiny.lower(), 0);
    EXPECT_GT(tiny.upper(), 0);
    EXPECT_GT(tiny.absolute_error, 0);
    EXPECT_EQ(tiny.value, tiny.absolute_error);
    EXPECT_GT(q_cut.outside->volume().lower(), 0);
    const auto tiny_moments = q_cut.inside->moments({0, 0, 0});
    EXPECT_EQ(tiny_moments.volume.value, tiny.value);
    EXPECT_EQ(tiny_moments.volume.absolute_error, tiny.absolute_error);
    for (const auto& first : tiny_moments.first_moment)
    {
        EXPECT_GE(first.lower(), 0);
        EXPECT_GE(first.upper(), 0);
    }
    const std::array tiny_cover{q_cut.inside, q_cut.outside};
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(sliver->certify_partition(tiny_cover, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    EXPECT_THROW((void) q_cut.inside->centroid(), std::invalid_argument);
    const SimpleFluid::PLIC::GeometryPacketContext tiny_context{{61, 67, 71, 73}};
    EXPECT_THROW((void) q_cut.inside->encode(tiny_context), std::invalid_argument);
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(q_cut.inside->points(), q_cut.inside->faces()),
        std::invalid_argument);
    const auto negative_p_vertex = [](const SimpleFluid::PLIC::DynamicPoint& point)
    { return point.value[1] == 0 && point.value[2] < .5 && point.value[2] > .4999999L; };
    EXPECT_TRUE(std::any_of(q_cut.inside->points().begin(), q_cut.inside->points().end(), negative_p_vertex));
    EXPECT_EQ(sliver->volume().value, sliver_before.value);
    EXPECT_EQ(sliver->volume().absolute_error, sliver_before.absolute_error);
    const auto q_source_before = q_source->volume();
    SimpleFluid::PLIC::DynamicClipResult next_cap;
    ASSERT_NO_THROW(next_cap = q_cut.inside->clip_by_face(*q_source, 1));
    ASSERT_TRUE(next_cap.inside);
    ASSERT_TRUE(next_cap.outside);
    const auto closed_oriented = [](const SimpleFluid::PLIC::DynamicConvexGeometryPtr& child)
    {
        std::map<std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>> edges;
        std::set<std::size_t> used;
        for (const auto& face : child->faces())
        {
            ASSERT_GE(face.size(), 3u);
            for (std::size_t i = 0; i < face.size(); ++i)
            {
                const auto a = face[i], b = face[(i + 1) % face.size()];
                ASSERT_LT(a, child->points().size());
                ASSERT_LT(b, child->points().size());
                EXPECT_NE(a, b);
                used.insert(a);
                auto& entry = edges[{std::min(a, b), std::max(a, b)}];
                ++entry.first;
                entry.second += a < b ? 1 : -1;
            }
        }
        EXPECT_EQ(used.size(), child->points().size());
        EXPECT_EQ(child->points().size() + child->faces().size(), edges.size() + 2);
        for (const auto& [edge, incidence] : edges)
        {
            EXPECT_EQ(incidence.first, 2u);
            EXPECT_EQ(incidence.second, 0);
        }
        EXPECT_GE(child->volume().lower(), 0);
        EXPECT_GT(child->volume().upper(), 0);
    };
    ASSERT_NO_FATAL_FAILURE(closed_oriented(next_cap.inside));
    ASSERT_NO_FATAL_FAILURE(closed_oriented(next_cap.outside));
    using Coordinate = std::pair<std::array<long double, 3>, std::array<long double, 3>>;
    const auto cap_edges = [](const SimpleFluid::PLIC::DynamicConvexGeometryPtr& child, bool reverse)
    {
        std::multiset<std::pair<Coordinate, Coordinate>> edges;
        const auto& cap = child->faces().back();
        for (std::size_t i = 0; i < cap.size(); ++i)
        {
            const auto& a = child->points()[cap[i]];
            const auto& b = child->points()[cap[(i + 1) % cap.size()]];
            const Coordinate first{a.value, a.absolute_error}, second{b.value, b.absolute_error};
            edges.emplace(reverse ? second : first, reverse ? first : second);
        }
        return edges;
    };
    EXPECT_EQ(cap_edges(next_cap.inside, false), cap_edges(next_cap.outside, true));
    const std::array next_cover{next_cap.inside, next_cap.outside};
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(q_cut.inside->certify_partition(next_cover, 0));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 1u);
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().support_distances, 0u);
    EXPECT_EQ(q_cut.inside->volume().value, tiny.value);
    EXPECT_EQ(q_cut.inside->volume().absolute_error, tiny.absolute_error);
    EXPECT_EQ(q_source->volume().value, q_source_before.value);
    EXPECT_EQ(q_source->volume().absolute_error, q_source_before.absolute_error);
    // A different retained P contains center, current p and departed p.
    // Its intersection with the roof extends beyond the center-to-p segment
    // to the left wall near q. Q is strictly positive at that constructed
    // crossing because its signed line parameter is negative and Q(p)<0.
    const std::array line_vertices{global[0], p, global[3], below};
    const auto line_source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(line_vertices, tetra_faces);
    const auto line_source_before = line_source->volume();
    const auto roof = donor->clip_by_face(*line_source, 0).outside;
    ASSERT_TRUE(roof);
    const auto near_q = std::find_if(roof->points().begin(), roof->points().end(),
        [](const SimpleFluid::PLIC::DynamicPoint& point)
        {
            return point.value[0] < .15 && point.value[1] > .4 && point.value[2] == .5 &&
                   point.absolute_error[0] > 0 && point.absolute_error[2] == 0;
        });
    ASSERT_NE(near_q, roof->points().end());
    const auto crossing = *near_q;
    EXPECT_GT(crossing.value[1] - crossing.absolute_error[1],
        std::max(global[0].value[1] + global[0].absolute_error[1], p.value[1] + p.absolute_error[1]));
    const auto roof_before = roof->volume();
    SimpleFluid::PLIC::DynamicClipResult line_cut;
    ASSERT_NO_THROW(line_cut = roof->clip_by_face(*q_source, 0));
    ASSERT_TRUE(line_cut.inside);
    ASSERT_TRUE(line_cut.outside);
    const auto copied_crossing = [&](const SimpleFluid::PLIC::DynamicPoint& point)
    { return point.value == crossing.value && point.absolute_error == crossing.absolute_error; };
    EXPECT_FALSE(std::any_of(line_cut.inside->points().begin(), line_cut.inside->points().end(), copied_crossing));
    EXPECT_TRUE(std::any_of(line_cut.outside->points().begin(), line_cut.outside->points().end(), copied_crossing));
    ASSERT_NO_FATAL_FAILURE(closed_oriented(line_cut.inside));
    ASSERT_NO_FATAL_FAILURE(closed_oriented(line_cut.outside));
    const std::array line_cover{line_cut.inside, line_cut.outside};
    EXPECT_NO_THROW(roof->certify_partition(line_cover, 0));
    EXPECT_EQ(roof->volume().value, roof_before.value);
    EXPECT_EQ(roof->volume().absolute_error, roof_before.absolute_error);
    EXPECT_EQ(line_source->volume().value, line_source_before.value);
    EXPECT_EQ(line_source->volume().absolute_error, line_source_before.absolute_error);
    EXPECT_EQ(q_source->volume().value, q_source_before.value);
    EXPECT_EQ(q_source->volume().absolute_error, q_source_before.absolute_error);
    // After Q face0, this new near-q crossing lies on P, Q0 and the left
    // wall. Its z enclosure straddles the roof, so coordinates alone cannot
    // certify the transverse coefficient on Q0's center/q/departed basis.
    const auto p_at_q = SimpleFluid::PLIC::dynamic_orientation({global[0], global[3], p, q});
    EXPECT_LT(p_at_q.upper(), 0);
    const auto on_q0 = std::find_if(line_cut.inside->points().begin(), line_cut.inside->points().end(),
        [](const SimpleFluid::PLIC::DynamicPoint& point)
        { return point.value[0] < .15 && point.value[1] > .4 && point.absolute_error[2] > 0; });
    ASSERT_NE(on_q0, line_cut.inside->points().end());
    const auto q0_crossing = *on_q0;
    EXPECT_LE(q0_crossing.value[2] - q0_crossing.absolute_error[2], .5L);
    EXPECT_GE(q0_crossing.value[2] + q0_crossing.absolute_error[2], .5L);
    const auto before_second_cut = line_cut.inside->volume();
    SimpleFluid::PLIC::DynamicClipResult affine_cut;
    ASSERT_NO_THROW(affine_cut = line_cut.inside->clip_by_face(*q_source, 1));
    ASSERT_TRUE(affine_cut.inside);
    ASSERT_TRUE(affine_cut.outside);
    const auto copied_q0_crossing = [&](const SimpleFluid::PLIC::DynamicPoint& point)
    { return point.value == q0_crossing.value && point.absolute_error == q0_crossing.absolute_error; };
    EXPECT_FALSE(std::any_of(
        affine_cut.inside->points().begin(), affine_cut.inside->points().end(), copied_q0_crossing));
    EXPECT_TRUE(std::any_of(
        affine_cut.outside->points().begin(), affine_cut.outside->points().end(), copied_q0_crossing));
    ASSERT_NO_FATAL_FAILURE(closed_oriented(affine_cut.inside));
    ASSERT_NO_FATAL_FAILURE(closed_oriented(affine_cut.outside));
    const std::array affine_cover{affine_cut.inside, affine_cut.outside};
    EXPECT_NO_THROW(line_cut.inside->certify_partition(affine_cover, 0));
    EXPECT_EQ(line_cut.inside->volume().value, before_second_cut.value);
    EXPECT_EQ(line_cut.inside->volume().absolute_error, before_second_cut.absolute_error);
    EXPECT_EQ(q_source->volume().value, q_source_before.value);
    EXPECT_EQ(q_source->volume().absolute_error, q_source_before.absolute_error);
    auto remaining = shared_atom;
    for (std::size_t face = 0; face < 4; ++face)
    {
        SCOPED_TRACE(::testing::Message() << "shared global face=" << face);
        const auto before = remaining->volume();
        if (face == 2)
        {
            auto changed = global;
            changed[3].absolute_error[0] =
                std::nextafter(changed[3].absolute_error[0], std::numeric_limits<long double>::infinity());
            const auto changed_source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded(changed, tetra_faces);
            EXPECT_THROW((void) remaining->clip_by_face(*changed_source, face), std::invalid_argument);
            const auto relative_source =
                SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local({0, 0, 0}, global, tetra_faces);
            EXPECT_THROW((void) remaining->clip_by_face(*relative_source, face), std::invalid_argument);
        }
        SimpleFluid::PLIC::DynamicClipResult next;
        ASSERT_NO_THROW(next = remaining->clip_by_face(*shared_source, face));
        if (face == 2)
        {
            // The common departed point completes the boundary-only contact;
            // a certified empty intersection ends the remaining face sweep.
            EXPECT_FALSE(next.inside);
            ASSERT_TRUE(next.outside);
            EXPECT_GT(next.outside->volume().lower(), 0);
            EXPECT_EQ(remaining->volume().value, before.value);
            EXPECT_EQ(remaining->volume().absolute_error, before.absolute_error);
            break;
        }
        ASSERT_TRUE(next.inside);
        EXPECT_GT(next.inside->volume().lower(), 0);
        const SimpleFluid::PLIC::GeometryPacketContext context{{17, 23, 41, 59}};
        const std::array registry{shared_source, shared_old, donor};
        const auto decoded =
            SimpleFluid::PLIC::DynamicConvexGeometry::decode(next.inside->encode(context), context, registry);
        ASSERT_TRUE(decoded);
        EXPECT_EQ(decoded->points().size(), next.inside->points().size());
        EXPECT_EQ(decoded->faces().size(), next.inside->faces().size());
        EXPECT_LE(decoded->volume().lower(), next.inside->volume().upper());
        EXPECT_GE(decoded->volume().upper(), next.inside->volume().lower());
        // Packets omit local point IDs. Explicit retained face incidence must
        // still validate the shell and each repeated boundary clip.
        for (std::size_t boundary = 0; boundary < decoded->faces().size(); ++boundary)
        {
            const auto repeated_boundary = decoded->clip_by_face(*decoded, boundary);
            ASSERT_TRUE(repeated_boundary.inside);
            EXPECT_FALSE(repeated_boundary.outside);
        }
        const auto repeat = next.inside->clip_by_face(*shared_source, face);
        ASSERT_TRUE(repeat.inside);
        EXPECT_FALSE(repeat.outside);
        EXPECT_EQ(repeat.inside->volume().value, next.inside->volume().value);
        EXPECT_EQ(repeat.inside->volume().absolute_error, next.inside->volume().absolute_error);
        EXPECT_EQ(remaining->volume().value, before.value);
        EXPECT_EQ(remaining->volume().absolute_error, before.absolute_error);
        remaining = next.inside;
    }
    EXPECT_EQ(shared_source->volume().value, shared_source_before.value);
    EXPECT_EQ(shared_source->volume().absolute_error, shared_source_before.absolute_error);
    EXPECT_EQ(shared_old->volume().value, shared_old_before.value);
    EXPECT_EQ(shared_old->volume().absolute_error, shared_old_before.absolute_error);
    EXPECT_EQ(source->volume().value, source_before.value);
    EXPECT_EQ(source->volume().absolute_error, source_before.absolute_error);
    EXPECT_EQ(donor->volume().value, donor_before.value);
    EXPECT_EQ(donor->volume().absolute_error, donor_before.absolute_error);
    EXPECT_EQ(atom->volume().value, atom_before.value);
    EXPECT_EQ(atom->volume().absolute_error, atom_before.absolute_error);
}

TEST(BoundedConvexGeometryTest, CapturedOverlayCrossingRetainsOriginalThreeSupportPointIdentity)
{
    // Two swept faces reconstruct the original source vertex on the donor's
    // z=.5 wall. Its coordinate intervals alone cannot classify face 2.
    const long double captured[4][3][2]{
        {{1.38777878078144567553e-17L, 1.38777878078144567553e-17L}, {0L, 0L}, {0L, 0L}},
        {{-1.6985112052913864172e-05L, -1.69851120529137015417e-05L},
            {8.48649516132128503912e-05L, 8.48649516132129588114e-05L},
            {-1.02794537218217521372e-05L, -1.02794537218207763553e-05L}},
        {{.149870729694547357753L, .149870729694547357808L}, {-0.5L, -0.5L},
            {-8.05361782418086580987e-07L, -8.0536178241797816077e-07L}},
        {{-.349999999999999977796L, -.349999999999999977796L}, {-0.5L, -0.5L},
            {-5.91842560168116945882e-06L, -5.9184256016810610386e-06L}}};
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> vertices;
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex)
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto lower = captured[vertex][axis][0], upper = captured[vertex][axis][1];
            const auto middle = std::midpoint(lower, upper);
            const auto radius = std::max(middle - lower, upper - middle);
            vertices[vertex].value[axis] = std::ldexp(middle, -1);
            vertices[vertex].absolute_error[axis] = radius == 0 ? 0 :
                std::nextafter(std::ldexp(radius, -1), std::numeric_limits<long double>::infinity());
        }
    const auto source = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        {.175, .25, .5}, vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const std::array<SimpleFluid::PLIC::GeometryPoint, 8> corners{
        {{0, 0, 0}, {.25, 0, 0}, {.35, .5, 0}, {.1, .5, 0}, {0, 0, .5}, {.25, 0, .5}, {.35, .5, .5}, {.1, .5, .5}}};
    const auto donor = SimpleFluid::PLIC::DynamicConvexGeometry::make(
        corners, {{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}});
    const auto source_before = source->volume(), donor_before = donor->volume();
    const auto first = donor->clip_by_face(*source, 0).inside;
    ASSERT_TRUE(first);
    const auto second = first->clip_by_face(*source, 1).inside;
    ASSERT_TRUE(second);
    const auto before = second->volume();
    const auto third = second->clip_by_face(*source, 2);
    ASSERT_TRUE(third.inside);
    EXPECT_GT(third.inside->volume().lower(), 0);
    const auto repeated = third.inside->clip_by_face(*source, 2);
    ASSERT_TRUE(repeated.inside);
    EXPECT_FALSE(repeated.outside);
    EXPECT_EQ(repeated.inside->volume().value, third.inside->volume().value);
    EXPECT_EQ(repeated.inside->volume().absolute_error, third.inside->volume().absolute_error);
    // This displacement is smaller than the coordinate enclosure, but moves
    // the source vertex off the exact donor wall. Proximity cannot transfer
    // its incidence to the unchanged donor crossing.
    const auto shifted = source->translated({0, 0, 1e-20});
    try
    {
        (void) second->clip_by_face(*shifted, 2);
        FAIL() << "A displaced source vertex cannot certify the donor crossing's incidence.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("support/topology sign unresolved"), std::string::npos);
    }
    EXPECT_EQ(second->volume().value, before.value);
    EXPECT_EQ(second->volume().absolute_error, before.absolute_error);
    EXPECT_EQ(source->volume().value, source_before.value);
    EXPECT_EQ(source->volume().absolute_error, source_before.absolute_error);
    EXPECT_EQ(donor->volume().value, donor_before.value);
    EXPECT_EQ(donor->volume().absolute_error, donor_before.absolute_error);
}

TEST(BoundedConvexGeometryTest, CapturedSweptTetrahedronIdentityAndComplementsHaveZeroClosure)
{
    // The first skew coarse/fine transport sweep supplied the same immutable
    // tetrahedron as its sole cover. These are its captured local intervals,
    // independent of source-plane authority or the full solver initialization.
    using Coordinate = std::array<long double, 3>; // lower, upper, central value
    const std::array<std::array<Coordinate, 3>, 4> captured{
        {{{{1.38777878078144567553e-17L, 1.38777878078144567553e-17L, 1.38777878078144567553e-17L}, {0L, 0L, 0L},
             {0L, 0L, 0L}}},
            {{{-1.6985112052913864172e-05L, -1.69851120529137015417e-05L, -1.69851120529137828569e-05L},
                {8.48649516132127961811e-05L, 8.48649516132130130215e-05L, 8.48649516132129046013e-05L},
                {-1.02794537218217521372e-05L, -1.02794537218207763553e-05L, -1.02794537218212642463e-05L}}},
            {{{0.149870729694547357753L, 0.149870729694547357808L, 0.14987072969454735778L},
                {-0.500000000000000000054L, -0.499999999999999999946L, -0.5L},
                {-8.05361782418086580987e-07L, -8.0536178241797816077e-07L, -8.05361782418032370878e-07L}}},
            {{{-0.349999999999999977823L, -0.349999999999999977768L, -0.349999999999999977796L},
                {-0.500000000000000000054L, -0.499999999999999999946L, -0.5L},
                {-5.91842560168116945882e-06L, -5.9184256016810610386e-06L, -5.91842560168111524871e-06L}}}}};
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> points;
    for (std::size_t v = 0; v < points.size(); ++v)
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto& x = captured[v][axis];
            points[v].value[axis] = std::ldexp(x[2], -1);
            const auto radius = std::ldexp(std::max(x[2] - x[0], x[1] - x[2]), -1);
            points[v].absolute_error[axis] =
                radius == 0 ? 0 : std::nextafter(radius, std::numeric_limits<long double>::infinity());
        }
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        {.175, .25, .5}, points, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto volume = parent->volume();
    ASSERT_GT(volume.absolute_error, volume.value * 8e-15L);
    const std::array same{parent};
    EXPECT_NO_THROW(parent->certify_partition(same, 0));
    EXPECT_NO_THROW(parent->certify_partition(same, 8e-15));
    const auto split = parent->clip({{0, 1, 0}, {.175, .25, .5}, -.125}, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const std::array complement{split.inside, split.outside};
    EXPECT_NO_THROW(parent->certify_partition(complement, 0));
    // A coverage proof does not narrow any measured transfer's enclosure.
    EXPECT_EQ(parent->volume().absolute_error, volume.absolute_error);
    EXPECT_FALSE(volume.relative_error_at_most(volume.value, 8e-15));
}

TEST(BoundedConvexGeometryTest, CapturedThreeDonorCoverNeedsGeometricCompletenessRatherThanIndependentMeasures)
{
    // Actual next face-sweep failure: independently intersected donors meet at
    // z=.5 and y=.5. Their measure errors are correlated through this tetrahedron.
    using Coordinate = std::array<long double, 3>;
    const std::array<std::array<Coordinate, 3>, 4> captured{
        {{{{1.38777878078144567553e-17L, 1.38777878078144567553e-17L, 1.38777878078144567553e-17L}, {0L, 0L, 0L},
             {0L, 0L, 0L}}},
            {{{-1.6985112052913864172e-05L, -1.69851120529137015417e-05L, -1.69851120529137828569e-05L},
                {8.48649516132127961811e-05L, 8.48649516132130130215e-05L, 8.48649516132129046013e-05L},
                {-1.02794537218217521372e-05L, -1.02794537218207763553e-05L, -1.02794537218212642463e-05L}}},
            {{{0.350015724107307711133L, 0.350015724107307711242L, 0.350015724107307711187L},
                {0.500111431056786749833L, 0.50011143105678675005L, 0.500111431056786749942L},
                {1.30328575757734758944e-07L, 1.30328575757951599379e-07L, 1.30328575757843179161e-07L}}},
            {{{0.149870729694547357753L, 0.149870729694547357808L, 0.14987072969454735778L},
                {-0.500000000000000000054L, -0.499999999999999999946L, -0.5L},
                {-8.05361782418086580987e-07L, -8.0536178241797816077e-07L, -8.05361782418032370878e-07L}}}}};
    std::array<SimpleFluid::PLIC::DynamicPoint, 4> points;
    for (std::size_t v = 0; v < points.size(); ++v)
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto& x = captured[v][axis];
            points[v].value[axis] = std::ldexp(x[2], -1);
            const auto radius = std::ldexp(std::max(x[2] - x[0], x[1] - x[2]), -1);
            points[v].absolute_error[axis] =
                radius == 0 ? 0 : std::nextafter(radius, std::numeric_limits<long double>::infinity());
        }
    const auto parent = SimpleFluid::PLIC::DynamicConvexGeometry::make_bounded_local(
        {.175, .25, .5}, points, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto donor_box = SimpleFluid::PLIC::DynamicConvexGeometry::box({-1, -1, -1}, {1, 1, 1});
    const auto z_split = donor_box->clip({{0, 0, 1}, {}, .5}, false);
    const auto y_split = z_split.outside->clip({{0, 1, 0}, {}, .5}, false);
    const std::array donors{z_split.inside, y_split.inside, y_split.outside};
    std::vector<SimpleFluid::PLIC::DynamicConvexGeometryPtr> pieces;
    for (const auto& donor : donors)
    {
        const auto intersection = parent->intersect(*donor);
        ASSERT_TRUE(intersection);
        pieces.push_back(intersection);
    }
    ASSERT_EQ(pieces.size(), 3u);
    EXPECT_GT(pieces.back()->volume().lower(), 0);
    EXPECT_LT(pieces.back()->volume().upper(), 1e-17L);
    const auto measured = parent->volume();
    SimpleFluid::PLIC::reset_geometry_statistics();
    EXPECT_NO_THROW(parent->certify_partition(pieces, 8e-15));
    EXPECT_EQ(SimpleFluid::PLIC::geometry_statistics().structural_partition_reuses, 0u);
    EXPECT_NO_THROW(parent->certify_partition(pieces, 0));
    std::reverse(pieces.begin(), pieces.end());
    EXPECT_NO_THROW(parent->certify_partition(pieces, 0));
    std::reverse(pieces.begin(), pieces.end());
    EXPECT_EQ(parent->volume().absolute_error, measured.absolute_error);
    auto overlap = pieces;
    overlap.push_back(pieces.front());
    EXPECT_THROW(parent->certify_partition(overlap, 8e-15), std::invalid_argument);
    auto gap = pieces;
    gap.pop_back();
    EXPECT_THROW(parent->certify_partition(gap, 8e-15), std::invalid_argument);
}

} // namespace
