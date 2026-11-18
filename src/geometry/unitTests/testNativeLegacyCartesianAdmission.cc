#include "geometry/mesh/NativeFragmentRecipe.hh"
#include "geometry/mesh/details/NativeLegacyCartesianAccess.hh"
#include "native_constructor_allocation_operators.inc"
#include "native_constructor_allocation_test.hh"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <optional>

namespace
{
namespace M = SimpleFluid::Meshes;
namespace G = SimpleFluid::geometry_detail;
using Access = M::details::NativeLegacyCartesianAccess;
using Mesh = M::OrthogonalCartesian3D;
using SimpleFluid::ArrReal;
using SimpleFluid::Vec3D;
struct Counter
{
    size_t limit = 0, used = 0, peak = 0, cumulative = 0;
    G::NativeCounterAllocationEnvelope native_allocation;
    size_t attempts = 0, fail_at = 0;
    void add(size_t bytes)
    {
        ++attempts;
        if ((fail_at && attempts == fail_at) || used > limit || bytes > limit - used)
            throw std::invalid_argument("test native admission rejection");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
};
struct Domain
{
    G::NativeOwnedCounter<Counter> counter;
    G::NativeConstructionAdmission admission;
    explicit Domain(size_t cap = 8 * 1024 * 1024)
        : counter(G::NativeConstructionCounterAccess::create<Counter>(cap)), admission(counter)
    {
    }
};
Access::Axes view(const Vec3D<ArrReal>& axes)
{
    return {axes[0], axes[1], axes[2]};
}
void bits(double a, double b)
{
    EXPECT_EQ(std::bit_cast<uint64_t>(a), std::bit_cast<uint64_t>(b));
}
void vector_bits(Mesh::Vec3 a, Mesh::Vec3 b)
{
    bits(a.x, b.x);
    bits(a.y, b.y);
    bits(a.z, b.z);
}
void all_metrics(const Mesh& reference, const Mesh& candidate, bool labels = true)
{
    ASSERT_EQ(reference.num_cells(), candidate.num_cells());
    ASSERT_EQ(reference.num_faces(), candidate.num_faces());
    ASSERT_EQ(reference.num_nodes(), candidate.num_nodes());
    for (size_t axis = 0; axis < 3; ++axis)
    {
        ASSERT_EQ(reference.cell_edges()[axis].size(), candidate.cell_edges()[axis].size());
        for (size_t i = 0; i < reference.cell_edges()[axis].size(); ++i)
            bits(reference.cell_edges()[axis][i], candidate.cell_edges()[axis][i]);
    }
    for (size_t c = 0; c < reference.num_cells(); ++c)
    {
        const auto id = reference.cell_id(c);
        bits(reference.cell_volume(id), candidate.cell_volume(id));
        vector_bits(reference.cell_centroid(id), candidate.cell_centroid(id));
        EXPECT_EQ(reference.cell_faces(id), candidate.cell_faces(id));
        EXPECT_EQ(reference.topology().neighbor_cells(id), candidate.topology().neighbor_cells(id));
    }
    for (size_t f = 0; f < reference.num_faces(); ++f)
    {
        const auto id = reference.face_id(f);
        bits(reference.face_area(id), candidate.face_area(id));
        vector_bits(reference.face_centroid(id), candidate.face_centroid(id));
        vector_bits(reference.face_normal(id), candidate.face_normal(id));
        vector_bits(reference.face_area_vector(id), candidate.face_area_vector(id));
        EXPECT_EQ(reference.owner_cell(id), candidate.owner_cell(id));
        EXPECT_EQ(reference.neighbor_cell(id), candidate.neighbor_cell(id));
        EXPECT_EQ(reference.boundary_id(id), candidate.boundary_id(id));
        const auto owner = reference.owner_cell(id);
        bits(reference.cell_to_face_distance(id, owner), candidate.cell_to_face_distance(id, owner));
        bits(reference.face_cell_center_distance(id), candidate.face_cell_center_distance(id));
        if (reference.neighbor_cell(id) != Mesh::invalid_cell_id())
            vector_bits(reference.cell_center_vector(id, owner), candidate.cell_center_vector(id, owner));
    }
    for (size_t n = 0; n < reference.num_nodes(); ++n)
        vector_bits(reference.node_coordinates(reference.node_id(n)), candidate.node_coordinates(candidate.node_id(n)));
    for (const auto id : reference.boundary_batch_ids())
    {
        if (labels) EXPECT_EQ(reference.boundary_batch_name(id), candidate.boundary_batch_name(id));
        EXPECT_EQ(std::ranges::distance(reference.boundary_face_batch(id)),
            std::ranges::distance(candidate.boundary_face_batch(id)));
    }
}
std::shared_ptr<const M::NativeFragmentDescriptor> descriptor(const Access::Region& region)
{
    M::NativeFragmentCache cache;
    return M::extract_native_fragment(M::MultiRegionMesh::Region(region), cache);
}
TEST(NativeLegacyCartesianAdmission, OriginalMetricsAndCanonicalTagOneBytesIncludingPeriodicAndShiftedAxes)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    const std::array<Vec3D<ArrReal>, 3> inputs{{{{{-0.0, 2, 5}, {-1, 2, 4}, {3, 5}}},
        {{{9007199254740992., 9007199254740994., 9007199254740998.}, {-1e-5, 0, 3e-5}, {1, 2, 5}}},
        {{{-2, 0, 3}, {4, 5, 9}, {-1, 1, 2}}}}};
    for (size_t test = 0; test < inputs.size(); ++test)
    {
        const Vec3D<bool> periodic{test == 2, false, test == 2};
        auto original = std::make_shared<Mesh>(inputs[test], periodic);
        const auto source = M::native_region(std::string(87, 's'), original);
        const auto wire = descriptor(source);
        ASSERT_EQ(wire->parts.size(), 1u);
        Domain domain;
        const auto made = Access::make(view(inputs[test]), Access::default_names, periodic, domain.admission);
        const auto copied = Access::detached_copy(*original, domain.admission);
        const auto decoded = Access::decode(*wire->parts[0], domain.admission);
        all_metrics(*original, *made);
        all_metrics(*original, *copied);
        all_metrics(*original, decoded.region.geometry().native());
        const auto echoed = descriptor(decoded.region);
        EXPECT_EQ(*echoed->parts[0], *wire->parts[0]);
        EXPECT_EQ(echoed->view, wire->view);
        EXPECT_EQ(copied->geometry_epoch(), 0u);
    }
    Mesh analytic(inputs[0]);
    EXPECT_DOUBLE_EQ(analytic.cell_volume({0, 0, 0}), 12);
    EXPECT_EQ(analytic.cell_centroid({0, 0, 0}), (Mesh::Vec3{1, .5, 4}));
}
TEST(NativeLegacyCartesianAdmission, UsesOriginalMidpointExpressionAndCopiesStoredMetrics)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    // The legacy sum/divide overflows here, while std::midpoint would remain
    // finite. This detects substituting modern rectilinear arithmetic.
    const Vec3D<ArrReal> axes{{{1e308, 1.1e308}, {0, 1e-150}, {0, 1e-150}}};
    Mesh original(axes);
    ASSERT_TRUE(std::isinf(original.cell_centroid({0, 0, 0}).x));
    Domain domain;
    const auto made = Access::make(view(axes), Access::default_names, {}, domain.admission);
    const auto copied = Access::detached_copy(original, domain.admission);
    all_metrics(original, *made);
    all_metrics(original, *copied);
}
TEST(NativeLegacyCartesianAdmission, DetachesMutableConstWrappedGeometryWithTopologyAliasesAndFreshEpoch)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    const Vec3D<ArrReal> axes{{{-0.0, 2, 5}, {-1, 2, 4}, {3, 5}}};
    const std::array<std::string, 6> names{std::string(91, 'a'), "right", "", std::string(103, 'b'), "bottom", "top"};
    Access::Names name_views;
    for (size_t i = 0; i < 6; ++i)
        name_views[i] = names[i];
    Domain destination;
    std::optional<M::details::NativeLegacyCartesianRegion> detached;
    std::weak_ptr<Mesh> original_mesh;
    std::weak_ptr<char> controller;
    std::weak_ptr<const Access::Provider> old_provider;
    {
        Domain source_domain;
        auto token = std::make_shared<char>('c');
        controller = token;
        auto geometry = std::make_shared<Mesh>(axes);
        *geometry = Mesh(axes);
        ASSERT_GT(geometry->geometry_epoch(), 0u);
        original_mesh = geometry;
        const auto topology_native = Access::make(view(axes), name_views, {}, source_domain.admission);
        auto topology = Access::wrap(topology_native, source_domain.admission);
        auto physical = std::make_shared<const Access::Provider>(geometry);
        old_provider = physical;
        const Access::Region region(std::string(97, 'r'), topology, physical);
        detached.emplace(Access::detach_region(region, destination.admission));
        all_metrics(*geometry, detached->region.geometry().native(), false);
        EXPECT_EQ(detached->region.geometry().geometry_epoch(), 0u);
        for (size_t i = 0; i < 6; ++i)
            EXPECT_EQ(detached->region.topology().boundary_batch_name(i), names[i]);
        // No original read/motion lease survives the synchronous copy.
        *geometry = Mesh({{{0, 1, 2}, {0, 1, 2}, {0, 1}}});
        EXPECT_NO_THROW(detached->region.validate_static());
    }
    EXPECT_TRUE(original_mesh.expired());
    EXPECT_TRUE(old_provider.expired());
    EXPECT_TRUE(controller.expired());
    EXPECT_DOUBLE_EQ(detached->region.geometry().cell_volume(0), 12);
    const auto wire = descriptor(detached->region);
    const auto decoded = Access::decode(*wire->parts[0], destination.admission);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_EQ(decoded.region.topology().boundary_batch_name(i), names[i]);
}
struct Census
{
    std::map<std::shared_ptr<const void>, size_t, std::owner_less<std::shared_ptr<const void>>> seen;
    G::NativeConstructorOwnerSink sink()
    {
        return {this, [](void* raw, const std::shared_ptr<const void>& owner, G::NativeConstructorOwnerKind, size_t)
            { return ++static_cast<Census*>(raw)->seen[owner] == 1; },
            [](void*, const auto& envelope) { EXPECT_TRUE(envelope->authentic_owner(envelope)); },
            [](void*, const auto& state) { EXPECT_TRUE(bool(state)); }};
    }
};
TEST(NativeLegacyCartesianAdmission, ActualProviderFirstLastEdgesAndForeignControlsReject)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    const Vec3D<ArrReal> axes{{{0, 1, 2}, {0, 1}, {0, 1}}};
    Domain domain;
    const auto native = Access::make(view(axes), Access::default_names, {}, domain.admission);
    const auto provider = Access::wrap(native, domain.admission);
    const Access::Region region("cartesian", provider, provider);
    Census census;
    Access::visit_region(region, census.sink());
    EXPECT_EQ(census.seen[provider], 2u);
    EXPECT_EQ(census.seen[native], 1u);
    const auto fake = std::shared_ptr<const Mesh>(native.get(), [](const Mesh*) {});
    EXPECT_THROW(Access::visit(fake, census.sink()), std::invalid_argument);
    const auto fake_provider = std::shared_ptr<const Access::Provider>(provider.get(), [](const auto*) {});
    EXPECT_THROW(Access::visit_provider(fake_provider, census.sink()), std::invalid_argument);
    const auto value_copy = std::make_shared<const Mesh>(*native);
    EXPECT_THROW(Access::visit(value_copy, census.sink()), std::invalid_argument);
}
template<class Operation> void independent_allocations(const Operation& operation)
{
    Domain measured;
    auto accepted = measured.admission.shared<unsigned>(37);
    const auto prefix = measured.counter.owner()->used, initial = measured.counter.owner()->attempts;
    native_constructor_test::Capture capture(measured.counter.owner()->used);
    using Result = decltype(operation(measured.admission));
    std::optional<Result> result;
    result.emplace(operation(measured.admission));
    const auto used = measured.counter.owner()->used, typed = measured.counter.owner()->attempts - initial;
    const auto trace = capture.finish();
    const auto heap = trace.phase[0].allocations;
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(used - prefix, trace.all_live);
    EXPECT_GT(heap, 30u);
    result.reset();
    EXPECT_EQ(measured.counter.owner()->used, prefix);
    EXPECT_EQ(*accepted, 37u);
    for (size_t i = 0; i < heap; ++i)
    {
        Domain domain;
        auto old = domain.admission.shared<unsigned>(37);
        const auto before = domain.counter.owner()->used;
        bool rejected = false;
        {
            native_constructor_test::Failure fail(i);
            try
            {
                const auto candidate = operation(domain.admission);
            }
            catch (const std::bad_alloc&)
            {
                rejected = true;
            }
        }
        EXPECT_TRUE(rejected) << i;
        EXPECT_EQ(domain.counter.owner()->used, before) << i;
        EXPECT_EQ(*old, 37u);
        {
            const auto retry = operation(domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, before);
    }
    for (size_t i = 1; i <= typed; ++i)
    {
        Domain domain;
        auto old = domain.admission.shared<unsigned>(37);
        const auto before = domain.counter.owner()->used;
        domain.counter.owner()->fail_at = domain.counter.owner()->attempts + i;
        bool rejected = false;
        try
        {
            const auto candidate = operation(domain.admission);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        domain.counter.owner()->fail_at = 0;
        EXPECT_TRUE(rejected) << i;
        EXPECT_EQ(domain.counter.owner()->used, before) << i;
        EXPECT_EQ(*old, 37u);
        {
            const auto retry = operation(domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, before);
    }
}
TEST(NativeLegacyCartesianAdmission, IndependentCppTraceEveryHeapAndAdmissionFailurePrefix)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    const Vec3D<ArrReal> axes{{{-0.0, 2, 5}, {-1, 2, 4}, {3, 5}}};
    const std::array<std::string, 6> names{
        std::string(93, 'a'), "right", "down", std::string(107, 'b'), "lower", "upper"};
    Access::Names borrowed;
    for (size_t i = 0; i < 6; ++i)
        borrowed[i] = names[i];
    const std::string region_name(121, 'r');
    Domain input_domain;
    const auto source = Access::make_region(region_name, view(axes), borrowed, {}, input_domain.admission);
    const auto wire = descriptor(source.region);
    ASSERT_EQ(wire->parts.size(), 1u);
    // Inputs/long strings/descriptor bytes are all outside capture.
    independent_allocations(
        [&](const auto& admission) { return Access::make_region(region_name, view(axes), borrowed, {}, admission); });
    independent_allocations([&](const auto& admission) { return Access::detach_region(source.region, admission); });
    independent_allocations([&](const auto& admission) { return Access::decode(*wire->parts[0], admission); });
}
TEST(NativeLegacyCartesianAdmission, ExactModeledPeakOneByteShortRetryAndPreallocationPacketRejection)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    const Vec3D<ArrReal> axes{{{0, 1, 2}, {0, 1}, {0, 1}}};
    Domain measured;
    {
        const auto result = Access::make_region("cartesian", view(axes), Access::default_names, {}, measured.admission);
    }
    const auto peak = measured.counter.owner()->peak;
    Domain exact(peak);
    {
        const auto result = Access::make_region("cartesian", view(axes), Access::default_names, {}, exact.admission);
    }
    Domain short_domain(peak - 1);
    const auto prefix = short_domain.counter.owner()->used;
    EXPECT_THROW(Access::make_region("cartesian", view(axes), Access::default_names, {}, short_domain.admission),
        std::invalid_argument);
    EXPECT_EQ(short_domain.counter.owner()->used, prefix);
    short_domain.counter.owner()->limit = peak;
    {
        const auto retry =
            Access::make_region("cartesian", view(axes), Access::default_names, {}, short_domain.admission);
    }
    EXPECT_EQ(short_domain.counter.owner()->used, prefix);
    Domain input;
    const auto valid = Access::make_region("cartesian", view(axes), Access::default_names, {}, input.admission);
    auto malformed = *descriptor(valid.region)->parts[0];
    malformed.push_back('x');
    const auto attempts = short_domain.counter.owner()->attempts;
    EXPECT_THROW(Access::decode(malformed, short_domain.admission), std::invalid_argument);
    EXPECT_EQ(short_domain.counter.owner()->attempts, attempts);
    EXPECT_EQ(short_domain.counter.owner()->used, prefix);
}
} // namespace
