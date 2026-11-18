#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/NativeIdentityOwnerAccess.hh"
#include "geometry/PLIC/details/ordinary/ExtrudedConstructionInternal.hh"
#include "geometry/mesh/details/NativeProviderOwnerAccess.hh"
#include "geometry/mesh/details/NativeTopologyAccess.hh"
#include "pipeline_allocation_trace.hh"
#include <bit>
#include <gtest/gtest.h>
#include <optional>

#include "native_constructor_allocation_operators.inc"
#include "native_constructor_allocation_test.hh"

namespace
{
namespace P = SimpleFluid::PLIC;
namespace G = SimpleFluid::geometry_detail;
namespace M = SimpleFluid::Meshes;
namespace A = M::details;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Construction = P::ExtrudedConstruction<ordinary>;
using Access = P::details::ExtrudedConstructionAccess;
using Vec3 = SimpleFluid::MeshUtils::Vec3;
using SimpleFluid::Arr;
struct Counter
{
    size_t limit = 0, used = 0, peak = 0, cumulative = 0;
    G::NativeCounterAllocationEnvelope native_allocation;
    size_t attempts = 0, fail_at = 0;
    void add(size_t bytes)
    {
        ++attempts;
        if ((fail_at && attempts == fail_at) || used > limit || bytes > limit - used)
            throw std::invalid_argument("test pre-allocation rejection");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
};
struct Domain
{
    G::NativeOwnedCounter<Counter> counter;
    G::NativeConstructionAdmission admission;
    explicit Domain(size_t cap = 16 * 1024 * 1024)
        : counter(G::NativeConstructionCounterAccess::create<Counter>(cap)), admission(counter)
    {
    }
};
struct Input
{
    Arr<Vec3> nodes;
    double area;
    Arr<Arr<unsigned>> loops;
    std::vector<P::PointRecipe> points;
    std::vector<size_t> active;
    std::vector<P::ScalarRecipe> values{{0, {}}, {1, {}}, {2, {}}};
    std::vector<size_t> selection{0, 1, 2};
    Input(Arr<Vec3> polygon, double analytic_area) : nodes(std::move(polygon)), area(analytic_area), loops(1)
    {
        for (unsigned i = 0; i < nodes.size(); ++i)
        {
            loops[0].push_back(i);
            active.push_back(i);
            points.push_back({{nodes[i].x, nodes[i].y, 0}, {}});
        }
    }
    auto topology(const G::NativeConstructionAdmission& admission, unsigned layers = 2) const
    {
        return A::NativeTopologyAccess::topology(
            A::NativeTopologyAccess::make(static_cast<unsigned>(nodes.size()), loops, layers, {}, admission), {},
            admission);
    }
    auto make(const G::NativeConstructionAdmission& admission) const
    {
        return Access::make(topology(admission), points, active, values, selection, admission);
    }
};
std::vector<Input> fixtures()
{
    // Exact rational coordinates, no axis-aligned side: every side reaches the
    // maximum Source-support branch. Areas are independently hand-integrated.
    return {Input({{0, 0, 0}, {3, 1, 0}, {1, 4, 0}}, 5.5), Input({{0, 0, 0}, {3, 1, 0}, {2, 4, 0}, {-1, 2, 0}}, 9),
        Input({{0, 0, 0}, {2, 1, 0}, {3, 3, 0}, {1, 4, 0}, {-1, 3, 0}, {-2, 1, 0}}, 12)};
}
void same_vec(Vec3 a, Vec3 b)
{
    EXPECT_EQ(std::bit_cast<uint64_t>(a.x), std::bit_cast<uint64_t>(b.x));
    EXPECT_EQ(std::bit_cast<uint64_t>(a.y), std::bit_cast<uint64_t>(b.y));
    EXPECT_EQ(std::bit_cast<uint64_t>(a.z), std::bit_cast<uint64_t>(b.z));
}
TEST(AdmittedProductConstruction, PrismSourceInventoryContainsEveryActualRequest)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    auto samples = fixtures();
    // Non-axis sides and a nine-vertex cap exercise the next growth capacity.
    samples.emplace_back(Arr<Vec3>{{0, -5, 0}, {3, -4, 0}, {5, -2, 0}, {6, 1, 0}, {5, 4, 0},
                             {2, 6, 0}, {-2, 5, 0}, {-5, 2, 0}, {-4, -2, 0}},
        84);
    for (const auto& input : samples)
    {
        const auto bound = P::ConvexCellGeometry<ordinary>::unit_prism_construction_capacity_bound(input.nodes.size());
        pipeline_trace::begin();
        {
            P::ordinary_extrusion_detail::Product product;
            product.base.reserve(input.nodes.size());
            for (const auto point : input.nodes)
                product.base.push_back({point.x, point.y, 0});
            const auto geometry = P::ordinary_extrusion_detail::product_poly(product);
        }
        pipeline_trace::recording.store(false);
        const auto trace = pipeline_trace::summary();
        EXPECT_FALSE(pipeline_trace::overflow.load());
        EXPECT_EQ(trace.all_live, 0u);
        EXPECT_EQ(trace.phase[0].bytes, bound) << "q=" << input.nodes.size();
    }
}
TEST(AdmittedProductConstruction, FourSideAxisPrismInventoryAndOriginalBoxKeepTheSamePhysicalCell)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    const auto bound = P::ConvexCellGeometry<ordinary>::unit_prism_construction_capacity_bound(4);
    pipeline_trace::begin();
    {
        P::ordinary_extrusion_detail::Product product;
        product.base.reserve(4);
        product.base = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        const auto geometry = P::ordinary_extrusion_detail::product_poly(product);
    }
    pipeline_trace::recording.store(false);
    const auto prism = pipeline_trace::summary();
    pipeline_trace::begin();
    {
        const auto box = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    }
    pipeline_trace::recording.store(false);
    const auto box = pipeline_trace::summary();
    // Only the four side support branches differ from the helper's maximum.
    const auto branch_difference = 4 * (P::details::ordinary_native::source_support_exact_allocation_bound(4) -
                                           P::details::ordinary_native::source_support_exact_allocation_bound(4, true));
    EXPECT_EQ(prism.phase[0].bytes + branch_difference, bound);
    EXPECT_GT(box.phase[0].bytes, 0u);
    EXPECT_EQ(box.all_live, 0u);
    EXPECT_GT(prism.phase[0].allocations, 0u);
    EXPECT_EQ(prism.all_live, 0u);
    // Allocation counts can coincide for two different constructor paths. The
    // unit-prism inventory above is independent; no count advantage is assumed.
    P::ordinary_extrusion_detail::Product product;
    product.base = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    const auto actual = P::ordinary_extrusion_detail::product_poly(product);
    const auto expected = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    EXPECT_TRUE(actual->equivalent(*expected));
    EXPECT_EQ(actual->volume().value, 1);
    EXPECT_EQ(actual->centroid(), (P::GeometryPoint{.5, .5, .5}));
}
TEST(AdmittedProductConstruction, EveryActualAllocationIsAlreadyCoveredAndAllLegacyMetricsMatch)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    for (const auto& input : fixtures())
    {
        Domain domain;
        const auto prefix = domain.counter.owner()->used;
        native_constructor_test::admitted_used = &domain.counter.owner()->used;
        native_constructor_test::admitted_prefix = prefix;
        native_constructor_test::uncovered = 0;
        pipeline_trace::begin();
        auto admitted = input.make(domain.admission);
        pipeline_trace::recording.store(false);
        native_constructor_test::admitted_used = nullptr;
        EXPECT_EQ(native_constructor_test::uncovered, 0u);
        const auto legacy = Construction::make(
            input.topology(domain.admission), input.points, input.active, input.values, input.selection);
        EXPECT_EQ(admitted->encode(), legacy->encode());
        for (size_t c = 0; c < 2; ++c)
        {
            EXPECT_DOUBLE_EQ(admitted->cell_volume(c), input.area);
            EXPECT_EQ(
                std::bit_cast<uint64_t>(admitted->cell_volume(c)), std::bit_cast<uint64_t>(legacy->cell_volume(c)));
            same_vec(admitted->cell_centroid(c), legacy->cell_centroid(c));
        }
        const auto faces = input.topology(domain.admission)->layout().faces;
        for (size_t f = 0; f < faces; ++f)
        {
            EXPECT_EQ(std::bit_cast<uint64_t>(admitted->face_area(f)), std::bit_cast<uint64_t>(legacy->face_area(f)));
            same_vec(admitted->face_centroid(f), legacy->face_centroid(f));
            same_vec(admitted->face_area_vector(f), legacy->face_area_vector(f));
        }
    }
}
TEST(AdmittedProductConstruction, SeparatePartDecodeAndTagEightFullDecodePreserveAllBytes)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto input = fixtures().back();
    auto source = input.make(domain.admission);
    const auto base = source->encode_base(), values = source->encode_axis_values(),
               selection = source->encode_axis_selection();
    const auto view = source->encode_view(), full = source->encode();
    auto decoded =
        Access::decode_parts(input.topology(domain.admission), base, values, selection, view, domain.admission);
    EXPECT_EQ(decoded->encode(), full);
    auto static_decoded = Access::decode(input.topology(domain.admission), full, domain.admission);
    EXPECT_EQ(static_decoded->encode(), full);
    auto shared = Access::decode_parts(
        input.topology(domain.admission), base, values, selection, view, domain.admission, source, source, source);
    EXPECT_EQ(shared->base_storage_identity(), source->base_storage_identity());
    EXPECT_EQ(shared->axis_storage_identity(), source->axis_storage_identity());
    EXPECT_EQ(shared->axis_values_storage_identity(), source->axis_values_storage_identity());
}
TEST(AdmittedProductConstruction, EveryUnderlyingHeapFailureReleasesOnlyThisConstructorAndRetryWorks)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    const auto input = fixtures().front();
    Domain measured;
    pipeline_trace::begin();
    {
        const auto source = input.make(measured.admission);
    }
    pipeline_trace::recording.store(false);
    const auto count = pipeline_trace::summary().phase[0].allocations;
    ASSERT_GT(count, 100u);
    for (size_t ordinal = 0; ordinal < count; ++ordinal)
    {
        Domain domain;
        auto accepted = domain.admission.shared<unsigned>(73);
        const auto prefix = domain.counter.owner()->used;
        {
            native_constructor_test::Failure fail(ordinal);
            EXPECT_THROW({ const auto source = input.make(domain.admission); }, std::bad_alloc) << ordinal;
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
        EXPECT_EQ(*accepted, 73u);
        {
            const auto retry = input.make(domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
    }
}
TEST(AdmittedProductConstruction, InvalidForwardDagRejectsBeforeNativeStorageAllocation)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    auto input = fixtures().front();
    const auto topology = input.topology(domain.admission);
    input.points.front().parents = {0};
    const auto prefix = domain.counter.owner()->used;
    const auto attempts = domain.counter.owner()->attempts;
    EXPECT_THROW(Access::make(topology, input.points, input.active, input.values, input.selection, domain.admission),
        std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    EXPECT_EQ(domain.counter.owner()->attempts, attempts);
}
TEST(AdmittedProductConstruction, LongParentsAndDuplicateRecipeSignaturesPreserveSupportAncestry)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const Arr<Arr<unsigned>> cells{{0, 1, 2, 3}};
    const auto topology = A::NativeTopologyAccess::topology(
        A::NativeTopologyAccess::make(4, cells, 2, {}, domain.admission), {}, domain.admission);
    std::vector<P::PointRecipe> points{{{-2, -2, 0}, {}}, {{4, -2, 0}, {}}, {{4, 4, 0}, {}}, {{-2, 4, 0}, {}}};
    std::vector<size_t> many;
    for (unsigned i = 0; i < 32; ++i)
        many.push_back(i % 2);
    points.push_back({{}, many});
    points.push_back({{}, {1, 2}});
    points.push_back({{}, {2, 3}});
    points.push_back({{}, {3, 0}});
    points.push_back(points[0]);
    points.push_back(points[1]);
    std::vector<size_t> duplicate_parents;
    for (const auto parent : many)
        duplicate_parents.push_back(parent + 8);
    points.push_back({{}, duplicate_parents});
    const std::vector<size_t> active{10, 5, 6, 7};
    const std::vector<P::ScalarRecipe> values{{0, {}}, {2, {}}, {0, many}};
    const std::vector<size_t> selected{0, 2, 1};
    const auto admitted = Access::make(topology, points, active, values, selected, domain.admission);
    const auto legacy = Construction::make(topology, points, active, values, selected);
    EXPECT_EQ(admitted->encode(), legacy->encode());
    EXPECT_DOUBLE_EQ(admitted->cell_volume(0), 18);
    for (size_t f = 0; f < topology->layout().faces; ++f)
    {
        const auto a = admitted->certified_face_geometry(f), b = legacy->certified_face_geometry(f);
        EXPECT_EQ(P::canonical_key(P::planar_support<ordinary>(*a)), P::canonical_key(P::planar_support<ordinary>(*b)));
    }
    const auto side = topology->indexer().face_ordinal({0, 0, M::SemiStructuredIndexer::SIDE});
    const auto support = P::planar_support<ordinary>(*admitted->certified_face_geometry(side));
    const auto& backing = P::details::GeometryIdentityAccess::backing(support);
    const auto* native = P::details::NativeIdentityOwnerAccess::backing(backing);
    ASSERT_TRUE(native);
    EXPECT_EQ(backing->depth, 1u);
    EXPECT_FALSE(backing->parent);
    EXPECT_GT(native->storage().owner->exclusive_bytes(), 0u);
}
TEST(AdmittedProductConstruction, PartialMissBuildsOnlyChangedFactorsAndWarmViewAllocatesEightFixedBlocks)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto input = fixtures()[1];
    const auto source = input.make(domain.admission);
    const auto topology = input.topology(domain.admission);
    const auto base = source->encode_base(), values = source->encode_axis_values();
    const auto selection = source->encode_axis_selection(), view = source->encode_view();
    const auto before = domain.counter.owner()->attempts;
    const auto warm =
        Access::decode_parts(topology, base, values, selection, view, domain.admission, source, source, source);
    EXPECT_EQ(domain.counter.owner()->attempts - before, 8u); // Two canonical fixed factories; no factor factory.
    EXPECT_EQ(warm->base_storage_identity(), source->base_storage_identity());
    auto changed_values = input.values;
    changed_values.push_back({3, {}});
    const auto new_values = Construction::make(topology, input.points, input.active, changed_values, input.selection);
    const auto changed_packet = new_values->encode_axis_values();
    const auto constructions = P::geometry_statistics().constructions;
    const auto changed =
        Access::decode_parts(topology, base, changed_packet, selection, view, domain.admission, source);
    EXPECT_EQ(P::geometry_statistics().constructions, constructions);
    EXPECT_EQ(changed->base_storage_identity(), source->base_storage_identity());
    EXPECT_NE(changed->axis_values_storage_identity(), source->axis_values_storage_identity());
    EXPECT_EQ(changed->encode(), new_values->encode());
    auto changed_points = input.points;
    for (auto& point : changed_points)
        point.root[0] += 0.25;
    const auto new_base = Construction::make(topology, changed_points, input.active, input.values, input.selection);
    const auto changed_base = new_base->encode_base();
    const auto before_base = P::geometry_statistics().constructions;
    const auto changed_geometry =
        Access::decode_parts(topology, changed_base, values, selection, view, domain.admission, {}, source, source);
    EXPECT_EQ(P::geometry_statistics().constructions, before_base + 1);
    EXPECT_EQ(changed_geometry->axis_values_storage_identity(), source->axis_values_storage_identity());
    EXPECT_EQ(changed_geometry->axis_storage_identity(), source->axis_storage_identity());
    EXPECT_EQ(changed_geometry->encode(), new_base->encode());
}
TEST(AdmittedProductConstruction, ExactModeledPeakAndOneByteShortAndTypedFailurePrefixes)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    const auto input = fixtures().front();
    Domain measured;
    const auto attempts = measured.counter.owner()->attempts;
    {
        const auto source = input.make(measured.admission);
    }
    const auto count = measured.counter.owner()->attempts - attempts;
    const auto peak = measured.counter.owner()->peak;
    Domain exact(peak);
    {
        const auto source = input.make(exact.admission);
    }
    Domain short_domain(peak - 1);
    const auto short_prefix = short_domain.counter.owner()->used;
    EXPECT_THROW({ const auto source = input.make(short_domain.admission); }, std::invalid_argument);
    EXPECT_EQ(short_domain.counter.owner()->used, short_prefix);
    short_domain.counter.owner()->limit = peak;
    {
        const auto source = input.make(short_domain.admission);
    }
    EXPECT_EQ(short_domain.counter.owner()->used, short_prefix);
    for (size_t ordinal = 1; ordinal <= count; ++ordinal)
    {
        Domain domain;
        const auto prefix = domain.counter.owner()->used;
        domain.counter.owner()->fail_at = domain.counter.owner()->attempts + ordinal;
        EXPECT_THROW({ const auto source = input.make(domain.admission); }, std::invalid_argument) << ordinal;
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
        domain.counter.owner()->fail_at = 0;
        {
            const auto retry = input.make(domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
    }
}
TEST(AdmittedProductConstruction, SourceSupportOwnsExactDetachedStorageAndWeakControlExpiry)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    const std::vector<P::details::OrdinaryIdentityPoint> polygon{{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}};
    std::optional<P::PlaneSupportIdentity<ordinary>> saved;
    std::weak_ptr<const G::NativeAllocationState> state;
    std::weak_ptr<const P::details::GeometryIdentityBacking> backing;
    std::weak_ptr<char> source_controller;
    {
        Domain domain;
        auto controller = std::make_shared<char>('x');
        source_controller = controller;
        saved.emplace(P::details::GeometryIdentityAccess::support_source_admitted(polygon, domain.admission));
        const auto legacy = P::details::GeometryIdentityAccess::support_source(polygon);
        EXPECT_EQ(P::canonical_key(*saved), P::canonical_key(legacy));
        state = domain.admission.state();
        backing = P::details::GeometryIdentityAccess::backing(*saved);
        EXPECT_TRUE(P::details::NativeIdentityOwnerAccess::admitted(backing.lock()));
    }
    EXPECT_TRUE(source_controller.expired());
    EXPECT_FALSE(state.expired());
    EXPECT_FALSE(backing.expired());
    saved.reset();
    EXPECT_TRUE(backing.expired());
    // Actual combined weak control still owns only its detached Record/State.
    EXPECT_FALSE(state.expired());
    backing.reset();
    EXPECT_TRUE(state.expired());
}
TEST(AdmittedProductConstruction, ExistingNumericalWorkRejectsBeforeConstructorAllocations)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto input = fixtures().front();
    const auto topology = input.topology(domain.admission);
    const auto source = input.make(domain.admission);
    const auto full = source->encode(), base = source->encode_base(), values = source->encode_axis_values();
    const auto selection = source->encode_axis_selection(), view = source->encode_view();
    P::GeometryLimits limits;
    limits.arithmetic_mode = ordinary;
    P::details::Operation operation(limits);
    // Legacy nesting continues to work. Its ledger retains these owners after
    // the temporary geometry dies, despite completing its pending drain.
    {
        const auto legacy = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    }
    const auto owners = operation.work.allocations.owner_count();
    ASSERT_GT(owners, 0u);
    const auto prefix = domain.counter.owner()->used, attempts = domain.counter.owner()->attempts;
    EXPECT_THROW((void) P::ConvexCellGeometry<ordinary>::unit_prism_construction_capacity_bound(3), std::invalid_argument);
    EXPECT_THROW(Access::make(topology, input.points, input.active, input.values, input.selection, domain.admission),
        std::invalid_argument);
    EXPECT_THROW(Access::decode(topology, full, domain.admission), std::invalid_argument);
    EXPECT_THROW(
        Access::decode_parts(topology, base, values, selection, view, domain.admission), std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    EXPECT_EQ(domain.counter.owner()->attempts, attempts);
    EXPECT_EQ(operation.work.allocations.owner_count(), owners);
    // A validated complete factor hit constructs no temporary numerical prism.
    {
        const auto warm =
            Access::decode_parts(topology, base, values, selection, view, domain.admission, source, source, source);
    }
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    EXPECT_EQ(operation.work.allocations.owner_count(), owners);
}
TEST(AdmittedProductConstruction, BorrowedIdentityProjectionRejectsForeignAddressAndEnvelopeControl)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto support = P::details::GeometryIdentityAccess::support_axis_admitted(0, 1, domain.admission);
    const auto& owner = P::details::GeometryIdentityAccess::backing(support);
    EXPECT_TRUE(P::details::NativeIdentityOwnerAccess::admitted(owner));
    EXPECT_TRUE(P::details::NativeIdentityOwnerAccess::admitted(*owner));
    const auto foreign_control =
        std::shared_ptr<const P::details::GeometryIdentityBacking>(owner.get(), [](const auto*) {});
    EXPECT_THROW(P::details::NativeIdentityOwnerAccess::admitted(foreign_control), std::invalid_argument);
    const auto other_support = P::details::GeometryIdentityAccess::support_axis_admitted(1, 1, domain.admission);
    const auto& other = P::details::GeometryIdentityAccess::backing(other_support);
    ASSERT_NE(owner.get(), other.get());
    EXPECT_TRUE(P::details::NativeIdentityOwnerAccess::admitted(*other));
    const auto foreign_address = std::shared_ptr<const P::details::GeometryIdentityBacking>(owner, other.get());
    EXPECT_THROW(P::details::NativeIdentityOwnerAccess::admitted(foreign_address), std::invalid_argument);
    const auto legacy_support = P::details::GeometryIdentityAccess::support_axis(0, 1);
    const auto& legacy = P::details::GeometryIdentityAccess::backing(legacy_support);
    EXPECT_FALSE(P::details::NativeIdentityOwnerAccess::admitted(*legacy));
    EXPECT_LT(legacy->allocation_bytes(), owner->allocation_bytes());
}
} // namespace
