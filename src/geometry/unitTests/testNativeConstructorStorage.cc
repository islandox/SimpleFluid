#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include "geometry/mesh/details/NativeProviderOwnerAccess.hh"
#include "geometry/mesh/details/NativeSemiStructuredAccess.hh"
#include "geometry/mesh/details/NativeTopologyAccess.hh"
#include <bit>
#include <gtest/gtest.h>
#include <map>
#include <unordered_map>

#include "native_constructor_allocation_operators.inc"
#include "native_constructor_allocation_test.hh"

namespace
{
namespace G = SimpleFluid::geometry_detail;
namespace M = SimpleFluid::Meshes;
namespace A = SimpleFluid::Meshes::details;
using SimpleFluid::Arr;
using SimpleFluid::real_t;
using SimpleFluid::MeshUtils::Vec3;
struct Counter
{
    size_t limit = 0, used = 0, peak = 0, cumulative = 0;
    G::NativeCounterAllocationEnvelope native_allocation;
    size_t attempts = 0, fail_at = 0;
    void add(size_t bytes)
    {
        ++attempts;
        if ((fail_at && attempts == fail_at) || used > limit || bytes > limit - used)
            throw std::invalid_argument("test admission rejected before allocation");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
};
struct Domain
{
    G::NativeOwnedCounter<Counter> counter;
    G::NativeConstructionAdmission admission;
    explicit Domain(size_t limit = 4 * 1024 * 1024)
        : counter(G::NativeConstructionCounterAccess::create<Counter>(limit)), admission(counter)
    {
    }
};
const Arr<Vec3> nodes{{0, 0, 0}, {3, 0, 0}, {3, 2, 0}, {0, 2, 0}};
const Arr<Arr<unsigned>> loops{{0, 1, 3}, {1, 2, 3}};
const Arr<real_t> z{0, 0.25, 2};
Arr<M::SemiStructMeshTopo::BoundaryEdge> labels()
{
    return {{0, 1, std::string(93, 'a')}, {1, 2, "right"}, {2, 3, "upper"}, {3, 0, "left"}};
}
auto make_topology(const G::NativeConstructionAdmission& admission)
{
    return A::NativeTopologyAccess::topology(
        A::NativeTopologyAccess::make(4, loops, 2, labels(), admission), {}, admission);
}
void equal_bits(double a, double b)
{
    EXPECT_EQ(std::bit_cast<uint64_t>(a), std::bit_cast<uint64_t>(b));
}
void equal_vec(Vec3 a, Vec3 b)
{
    equal_bits(a.x, b.x);
    equal_bits(a.y, b.y);
    equal_bits(a.z, b.z);
}
TEST(NativeConstructorStorage, ConcreteBuffersNodesBucketsAndLongStringsAreExactlyObserved)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    {
        auto storage = G::native_constructor_storage(domain.admission);
        const auto control_prefix = domain.counter.owner()->used;
        std::vector<unsigned> values;
        storage.owner->reserve(values, 37);
        EXPECT_EQ(values.capacity(), 37u);
        values.assign(37, 42);
        std::string small, large;
        storage.owner->string(small, "abc");
        storage.owner->string(large, std::string(113, 'x'));
        using Pair = std::pair<const unsigned, unsigned>;
        std::unordered_map<unsigned, unsigned, std::hash<unsigned>, std::equal_to<unsigned>,
            G::NativeStorageAllocator<Pair>>
            map(0, std::hash<unsigned>{}, std::equal_to<unsigned>{}, G::NativeStorageAllocator<Pair>(storage.owner));
        for (unsigned i = 0; i < 19; ++i)
            map.emplace(i, i + 1);
        size_t observed = 0, allocations = 0;
        storage.owner->visit_allocations(
            [&](G::NativeAllocationExtent extent)
            {
                ASSERT_NE(extent.address, nullptr);
                EXPECT_GT(extent.bytes, 0u);
                observed += extent.bytes;
                ++allocations;
            });
        EXPECT_GT(allocations, map.size() + 2); // Includes buckets and separate vector/string receipts.
        EXPECT_EQ(observed, storage.owner->exclusive_bytes());
        EXPECT_EQ(domain.counter.owner()->used - control_prefix, observed);
        EXPECT_EQ(large, std::string(113, 'x'));
    }
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativeConstructorStorage, AdmittedAndLegacyTopologyPreserveIntegerConnectivityAndLabels)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const M::SemiStructMeshTopo legacy(4, loops, 2, labels());
    const auto admitted = A::NativeTopologyAccess::make(4, loops, 2, labels(), domain.admission);
    ASSERT_EQ(admitted.side_faces().size(), 5u);
    for (size_t i = 0; i < legacy.side_faces().size(); ++i)
    {
        const auto& a = legacy.side_face(i);
        const auto& b = admitted.side_face(i);
        EXPECT_EQ(a.nodes, b.nodes);
        EXPECT_EQ(a.owner, b.owner);
        EXPECT_EQ(a.neighbor, b.neighbor);
        EXPECT_EQ(a.boundary_id, b.boundary_id);
    }
    for (unsigned c = 0; c < loops.size(); ++c)
        EXPECT_EQ(legacy.cell_side_faces(c), admitted.cell_side_faces(c));
    for (const int id : legacy.boundary_batch_ids())
    {
        EXPECT_EQ(legacy.boundary_batch_name(id), admitted.boundary_batch_name(id));
        const auto a = legacy.boundary_face_batch(id).face_lids;
        const auto b = admitted.boundary_face_batch(id).face_lids;
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i)
            EXPECT_EQ(a[i], b[i]);
    }
}
TEST(NativeConstructorStorage, StaticRemoteAndDetachedMetricsMatchEveryLegacyOutputBit)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    auto source = std::make_shared<M::SemiStructuredXY_Z>(nodes, loops, z, labels());
    auto decoded = A::NativeSemiStructuredAccess::make(nodes, loops, z, labels(), domain.admission);
    auto detached = A::NativeSemiStructuredAccess::detached_copy(*source, domain.admission);
    ASSERT_NE(source.get(), detached.get());
    EXPECT_EQ(detached->geometry_epoch(), 0u);
    for (size_t c = 0; c < source->num_cells(); ++c)
    {
        const auto id = source->cell_id(c);
        equal_bits(source->cell_volume(id), decoded->cell_volume(id));
        equal_bits(source->cell_volume(id), detached->cell_volume(id));
        equal_vec(source->cell_centroid(id), decoded->cell_centroid(id));
        equal_vec(source->cell_centroid(id), detached->cell_centroid(id));
        EXPECT_DOUBLE_EQ(decoded->cell_volume(id), c < 2 ? 0.75 : 5.25); // Independent triangle area * dz.
    }
    for (size_t f = 0; f < source->num_faces(); ++f)
    {
        const auto id = source->face_id(f);
        equal_bits(source->face_area(id), decoded->face_area(id));
        equal_bits(source->face_area(id), detached->face_area(id));
        equal_vec(source->face_centroid(id), decoded->face_centroid(id));
        equal_vec(source->face_centroid(id), detached->face_centroid(id));
        equal_vec(source->face_normal(id), decoded->face_normal(id));
        equal_vec(source->face_normal(id), detached->face_normal(id));
        EXPECT_EQ(source->owner_cell(id), detached->owner_cell(id));
        EXPECT_EQ(source->neighbor_cell(id), detached->neighbor_cell(id));
    }
    std::weak_ptr<M::SemiStructuredXY_Z> source_control = source;
    source.reset();
    EXPECT_TRUE(source_control.expired());
    EXPECT_DOUBLE_EQ(detached->cell_volume(detached->cell_id(0)), 0.75);
}
TEST(NativeConstructorStorage, PlainExtrudedGeometryRetainsItsOwnLegacyMetricExpressions)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    auto topology = make_topology(domain.admission);
    const M::ExtrudedGeometry legacy(topology, nodes, z);
    auto admitted = A::NativeTopologyAccess::geometry(topology, nodes, z, domain.admission);
    for (size_t c = 0; c < topology->layout().cells; ++c)
    {
        equal_bits(legacy.cell_volume(c), admitted->cell_volume(c));
        equal_vec(legacy.cell_centroid(c), admitted->cell_centroid(c));
    }
    for (size_t f = 0; f < topology->layout().faces; ++f)
    {
        equal_bits(legacy.face_area(f), admitted->face_area(f));
        equal_vec(legacy.face_centroid(f), admitted->face_centroid(f));
        equal_vec(legacy.face_area_vector(f), admitted->face_area_vector(f));
    }
    const auto prefix = domain.counter.owner()->cumulative;
    EXPECT_EQ(admitted->z_edges(), z);
    EXPECT_EQ(admitted->z_edges(), z);
    EXPECT_EQ(domain.counter.owner()->cumulative, prefix);
}
TEST(NativeConstructorStorage, TopologyDecodePreservesCanonicalBytesAndRejectsTrailingData)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto source = make_topology(domain.admission);
    auto bytes = source->encode_base();
    const auto copy = A::NativeTopologyAccess::decode_base(bytes, 2, domain.admission);
    EXPECT_EQ(copy->encode_base(), bytes);
    EXPECT_TRUE(A::NativeTopologyAccess::matches_base(*copy, bytes));
    bytes.push_back('x');
    EXPECT_THROW(A::NativeTopologyAccess::decode_base(bytes, 2, domain.admission), std::invalid_argument);
}
TEST(NativeConstructorStorage, InvalidConnectivityAndLabelsReleaseEveryConstructorPrefix)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    EXPECT_THROW(A::NativeTopologyAccess::make(4, {{0, 1, 1}}, 2, {}, domain.admission), std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    EXPECT_THROW(A::NativeTopologyAccess::make(4, loops, 2, {{0, 1, ""}}, domain.admission), std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    EXPECT_THROW(
        A::NativeTopologyAccess::make(4, loops, 2, {{1, 3, "interior"}}, domain.admission), std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
struct Census
{
    struct Entry
    {
        size_t references = 0, bytes = 0;
        G::NativeConstructorOwnerKind kind;
    };
    std::map<std::shared_ptr<const void>, Entry, std::owner_less<std::shared_ptr<const void>>> owners;
    std::map<const void*, size_t> allocations;
    std::map<std::shared_ptr<const G::NativeAllocationEnvelope>, size_t,
        std::owner_less<std::shared_ptr<const G::NativeAllocationEnvelope>>>
        envelopes;
    size_t state_edges = 0;
    G::NativeConstructorOwnerSink sink()
    {
        return {this,
            [](void* raw, const std::shared_ptr<const void>& owner, G::NativeConstructorOwnerKind kind, size_t bytes)
            {
                auto& self = *static_cast<Census*>(raw);
                auto [entry, fresh] = self.owners.try_emplace(owner, Entry{0, bytes, kind});
                EXPECT_EQ(entry->second.bytes, bytes);
                EXPECT_EQ(entry->second.kind, kind);
                ++entry->second.references;
                return fresh;
            },
            [](void* raw, const std::shared_ptr<const G::NativeAllocationEnvelope>& envelope)
            {
                auto& self = *static_cast<Census*>(raw);
                if (++self.envelopes[envelope] != 1)
                    return;
                envelope->visit_allocations([&](G::NativeAllocationExtent extent)
                    { EXPECT_TRUE(self.allocations.emplace(extent.address, extent.bytes).second); });
            },
            [](void* raw, const std::shared_ptr<const G::NativeAllocationState>& state)
            {
                EXPECT_NE(state, nullptr);
                ++static_cast<Census*>(raw)->state_edges;
            }};
    }
    size_t bytes() const
    {
        size_t total = 0;
        for (const auto& [owner, entry] : owners)
            total += entry.bytes;
        for (const auto& [address, bytes] : allocations)
            total += bytes;
        return total;
    }
};
TEST(NativeConstructorStorage, SharedOwnerDiamondCountsEveryPhysicalAllocationOnce)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    auto first = make_topology(domain.admission);
    auto second = A::NativeTopologyAccess::with_layers(*first, 2, {}, domain.admission);
    auto third = A::NativeTopologyAccess::with_layers(*first, 2, {}, domain.admission);
    EXPECT_EQ(first->base_storage_identity(), second->base_storage_identity());
    EXPECT_EQ(first->base_storage_identity(), third->base_storage_identity());
    auto geometry_a = A::NativeTopologyAccess::geometry(first, nodes, z, domain.admission);
    auto geometry_b = A::NativeTopologyAccess::geometry(second, nodes, z, domain.admission);
    auto geometry_c = A::NativeTopologyAccess::geometry(third, nodes, z, domain.admission);
    Census census;
    auto sink = census.sink();
    for (const auto& topology : {first, second, third})
        A::NativeProviderOwnerAccess::topology(topology, sink);
    for (const auto& geometry : {geometry_a, geometry_b, geometry_c})
        A::NativeProviderOwnerAccess::geometry(geometry, sink, nullptr);
    EXPECT_EQ(census.bytes(), domain.counter.owner()->used - prefix);
    size_t bases = 0;
    for (const auto& [owner, entry] : census.owners)
        if (entry.kind == G::NativeConstructorOwnerKind::TopologyBase)
        {
            ++bases;
            EXPECT_EQ(entry.references, 3u);
        }
    EXPECT_EQ(bases, 1u);
    EXPECT_GT(census.state_edges, 0u);
    // The manifest/census holds actual controls. Removing user roots alone
    // cannot destroy a physical object still named by a physical-owner record.
    std::weak_ptr<const M::ExtrudedGeometry> weak = geometry_a;
    geometry_a.reset();
    EXPECT_FALSE(weak.expired());
    census = {};
    EXPECT_TRUE(weak.expired());
}
TEST(NativeConstructorStorage, EveryTypedAllocationFailurePreservesAcceptedOwnersAndRetry)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain census;
    const auto start = census.counter.owner()->attempts;
    {
        const auto topology = make_topology(census.admission);
        const auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, census.admission);
    }
    const auto allocations = census.counter.owner()->attempts - start;
    ASSERT_GT(allocations, 30u); // Controls, envelopes, metadata, loops, hash nodes/buckets, long labels, metrics.
    for (size_t allocation = 1; allocation <= allocations; ++allocation)
    {
        Domain domain;
        auto accepted = domain.admission.shared<uint64_t>(0x123456789ULL);
        const auto prefix = domain.counter.owner()->used;
        domain.counter.owner()->fail_at = domain.counter.owner()->attempts + allocation;
        EXPECT_THROW(
            {
                const auto topology = make_topology(domain.admission);
                const auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, domain.admission);
            },
            std::invalid_argument)
            << "allocation " << allocation;
        EXPECT_EQ(domain.counter.owner()->used, prefix) << "allocation " << allocation;
        EXPECT_EQ(*accepted, 0x123456789ULL);
        domain.counter.owner()->fail_at = 0;
        {
            const auto topology = make_topology(domain.admission);
            const auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, domain.admission);
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix);
    }
}
TEST(NativeConstructorStorage, ExactModeledPeakSucceedsOneByteShortRejectsAndRetries)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain measured;
    {
        auto topology = make_topology(measured.admission);
        auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, measured.admission);
    }
    const auto exact_peak = measured.counter.owner()->peak;
    Domain exact(exact_peak);
    {
        auto topology = make_topology(exact.admission);
        auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, exact.admission);
    }
    Domain short_domain(exact_peak - 1);
    const auto prefix = short_domain.counter.owner()->used;
    EXPECT_THROW(
        {
            auto topology = make_topology(short_domain.admission);
            auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, short_domain.admission);
        },
        std::invalid_argument);
    EXPECT_EQ(short_domain.counter.owner()->used, prefix);
    short_domain.counter.owner()->limit = exact_peak;
    {
        auto topology = make_topology(short_domain.admission);
        auto geometry = A::NativeTopologyAccess::geometry(topology, nodes, z, short_domain.admission);
    }
    EXPECT_EQ(short_domain.counter.owner()->used, prefix);
}
TEST(NativeConstructorStorage, ProductSharedViewsKeepActualBaseAxisValuesAndSealCompatibility)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    namespace P = SimpleFluid::PLIC;
    using Ordinary = P::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Ordinary>;
    const std::vector<std::pair<Arr<Vec3>, double>> polygons{{{{0, 0, 0}, {2, 0, 0}, {0, 2, 0}}, 2},
        {{{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}}, 4},
        {{{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {2, 3, 0}, {0, 3, 0}, {-1, 1, 0}}, 9}};
    for (const auto& [polygon, area] : polygons)
    {
        Domain domain;
        Arr<Arr<unsigned>> cell(1);
        std::vector<P::PointRecipe> roots;
        std::vector<size_t> active;
        for (unsigned i = 0; i < polygon.size(); ++i)
        {
            cell[0].push_back(i);
            active.push_back(i);
            roots.push_back({{polygon[i].x, polygon[i].y, 0}, {}});
        }
        auto original_topology = A::NativeTopologyAccess::topology(
            A::NativeTopologyAccess::make(static_cast<unsigned>(polygon.size()), cell, 2, {}, domain.admission), {},
            domain.admission);
        auto source = Ordinary::make(original_topology, roots, active, {{0, {}}, {1, {}}, {2, {}}}, {0, 1, 2});
        auto view_topology = A::NativeTopologyAccess::with_layers(*original_topology, 1, {}, domain.admission);
        auto first =
            P::details::ExtrudedConstructionAccess::shared_view(view_topology, source, source, 0, 1, domain.admission);
        auto second =
            P::details::ExtrudedConstructionAccess::shared_view(view_topology, source, source, 1, 1, domain.admission);
        const std::array<size_t, 2> selected{0, 2};
        auto third = P::details::ExtrudedConstructionAccess::shared_axis_selection(
            view_topology, source, source, selected, domain.admission);
        EXPECT_EQ(first->base_storage_identity(), source->base_storage_identity());
        EXPECT_EQ(second->axis_storage_identity(), source->axis_storage_identity());
        EXPECT_EQ(third->axis_values_storage_identity(), source->axis_values_storage_identity());
        EXPECT_NE(third->axis_storage_identity(), source->axis_storage_identity());
        EXPECT_DOUBLE_EQ(first->cell_volume(0), area);
        EXPECT_DOUBLE_EQ(second->cell_volume(0), area);
        EXPECT_DOUBLE_EQ(third->cell_volume(0), 2 * area);
        auto geometry = A::NativeTopologyAccess::geometry(view_topology, third, domain.admission);
        const auto cumulative = domain.counter.owner()->cumulative;
        EXPECT_EQ(geometry->z_edges(), (Arr<real_t>{0, 2}));
        EXPECT_EQ(geometry->z_edges(), (Arr<real_t>{0, 2}));
        EXPECT_EQ(domain.counter.owner()->cumulative, cumulative);
        std::weak_ptr<const Ordinary> source_wrapper = source;
        source.reset();
        EXPECT_TRUE(source_wrapper.expired());
        EXPECT_DOUBLE_EQ(first->cell_volume(0), area);
        EXPECT_DOUBLE_EQ(geometry->cell_volume(0), 2 * area);
    }
}
TEST(NativeConstructorStorage, UnsupportedProfileRejectsBeforeConcreteBufferAllocation)
{
    if (G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Qualified profile active";
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    EXPECT_THROW(make_topology(domain.admission), std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}

/** Inputs and source owners are prepared before Capture/Failure. The operator
 * interposer sees requested C++ extents independently of storage receipts. */
template<class Setup, class Make> void independent_constructor_case(const Setup& setup, const Make& make)
{
    Domain measured;
    auto context = setup(measured.admission);
    auto accepted = measured.admission.shared<unsigned>(73);
    const auto prefix = measured.counter.owner()->used;
    const auto initial_attempts = measured.counter.owner()->attempts;
    native_constructor_test::Capture capture(measured.counter.owner()->used);
    auto result = make(measured.admission, context);
    const auto used = measured.counter.owner()->used;
    const auto typed = measured.counter.owner()->attempts - initial_attempts;
    const auto trace = capture.finish();
    const auto heap = trace.phase[0].allocations;
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(used - prefix, trace.all_live); // Raw global new/delete, not receipt enumeration.
    EXPECT_GT(heap, 8u);
    EXPECT_GT(typed, 8u);
    result.reset();
    EXPECT_EQ(measured.counter.owner()->used, prefix);
    EXPECT_EQ(*accepted, 73u);
    for (size_t allocation = 0; allocation < heap; ++allocation)
    {
        Domain domain;
        auto input = setup(domain.admission);
        auto old = domain.admission.shared<unsigned>(73);
        const auto old_used = domain.counter.owner()->used;
        bool rejected = false;
        {
            native_constructor_test::Failure failure(allocation);
            try
            {
                const auto candidate = make(domain.admission, input);
            }
            catch (const std::bad_alloc&)
            {
                rejected = true;
            }
        }
        EXPECT_TRUE(rejected) << "heap allocation " << allocation;
        EXPECT_EQ(domain.counter.owner()->used, old_used) << "heap allocation " << allocation;
        EXPECT_EQ(*old, 73u);
        {
            const auto retry = make(domain.admission, input);
        }
        EXPECT_EQ(domain.counter.owner()->used, old_used);
    }
    for (size_t allocation = 1; allocation <= typed; ++allocation)
    {
        Domain domain;
        auto input = setup(domain.admission);
        auto old = domain.admission.shared<unsigned>(73);
        const auto old_used = domain.counter.owner()->used;
        domain.counter.owner()->fail_at = domain.counter.owner()->attempts + allocation;
        bool rejected = false;
        try
        {
            const auto candidate = make(domain.admission, input);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        domain.counter.owner()->fail_at = 0;
        EXPECT_TRUE(rejected) << "admission allocation " << allocation;
        EXPECT_EQ(domain.counter.owner()->used, old_used) << "admission allocation " << allocation;
        EXPECT_EQ(*old, 73u);
        {
            const auto retry = make(domain.admission, input);
        }
        EXPECT_EQ(domain.counter.owner()->used, old_used);
    }
}
TEST(NativeConstructorStorage, IndependentRawTraceCoversLongLabelsAliasesAndLayerViews)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    // Every source string and alias view exists before the capture begins.
    const auto boundaries = labels();
    const std::string first(141, 'u'), second(149, 'v'), changed(157, 'w');
    const std::array<A::NativeTopologyAccess::AliasView, 2> aliases{
        {{boundaries[0].batch_name, first}, {boundaries[1].batch_name, second}}};
    const std::array<A::NativeTopologyAccess::AliasView, 1> overrides{{{boundaries[0].batch_name, changed}}};
    const auto make = [&](const G::NativeConstructionAdmission& admission, unsigned)
    {
        return A::NativeTopologyAccess::topology(
            A::NativeTopologyAccess::make(4, loops, 2, boundaries, admission), aliases, admission);
    };
    independent_constructor_case([](const auto&) { return 0u; }, make);
    independent_constructor_case([&](const auto& admission) { return make(admission, 0); },
        [&](const auto& admission, const auto& source)
        { return A::NativeTopologyAccess::with_layers(*source, 3, overrides, admission); });
}
struct RawHashOwner
{
    using Pair = std::pair<const unsigned, std::string>;
    G::NativeConstructorStorageOwner storage;
    std::unordered_map<unsigned, std::string, std::hash<unsigned>, std::equal_to<unsigned>,
        G::NativeStorageAllocator<Pair>>
        values;
    explicit RawHashOwner(const G::NativeConstructionAdmission& admission)
        : storage(G::native_constructor_storage(admission)),
          values(0, std::hash<unsigned>{}, std::equal_to<unsigned>{}, G::NativeStorageAllocator<Pair>(storage.owner))
    {
    }
};
TEST(NativeConstructorStorage, IndependentRawTraceCoversActualHashNodesBucketsAndStringBuffers)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    const std::string source(113, 'n');
    independent_constructor_case([](const auto&) { return 0u; },
        [&](const auto& admission, unsigned)
        {
            auto result = admission.template shared<RawHashOwner>(admission);
            for (unsigned i = 0; i < 23; ++i)
            {
                std::string value;
                result->storage.owner->string(value, i % 4 ? std::string_view(source) : std::string_view("sso"));
                result->values.emplace(i, std::move(value));
            }
            result->values.rehash(127); // Exercises simultaneous old/new bucket arrays.
            return result;
        });
}
TEST(NativeConstructorStorage, IndependentRawTraceCoversPlainExtrusionAndSemiStructuredMakeDetach)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    const auto boundaries = labels();
    independent_constructor_case(
        [&](const auto& admission)
        {
            return A::NativeTopologyAccess::topology(
                A::NativeTopologyAccess::make(4, loops, 2, boundaries, admission), {}, admission);
        },
        [&](const auto& admission, const auto& topology)
        { return A::NativeTopologyAccess::geometry(topology, nodes, z, admission); });
    independent_constructor_case([](const auto&) { return 0u; }, [&](const auto& admission, unsigned)
        { return A::NativeSemiStructuredAccess::make(nodes, loops, z, boundaries, admission); });
    const M::SemiStructuredXY_Z source(nodes, loops, z, boundaries);
    independent_constructor_case([](const auto&) { return 0u; }, [&](const auto& admission, unsigned)
        { return A::NativeSemiStructuredAccess::detached_copy(source, admission); });
}
TEST(NativeConstructorStorage, CopiedAndMovedConstructionWrappersRequireAuthenticatedRewrap)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    namespace P = SimpleFluid::PLIC;
    using Ordinary = P::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Ordinary>;
    Domain domain;
    const auto topology = make_topology(domain.admission);
    std::vector<P::PointRecipe> roots;
    std::vector<size_t> active;
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        roots.push_back({{nodes[i].x, nodes[i].y, 0}, {}});
        active.push_back(i);
    }
    const auto legacy = Ordinary::make(topology, roots, active, {{0, {}}, {1, {}}, {2, {}}}, {0, 1, 2});
    const auto admitted =
        P::details::ExtrudedConstructionAccess::shared_view(topology, legacy, legacy, 0, 2, domain.admission);
    auto copy = std::make_shared<Ordinary>(*admitted);
    auto moved = std::make_shared<Ordinary>(std::move(*copy));
    EXPECT_EQ(moved->encode(), admitted->encode());
    Census census;
    auto sink = census.sink();
    const auto identity = +[](void*, const P::PlaneSupportIdentity<SimpleFluid::ArithmeticMode::Ordinary>&) {};
    const auto topology_visit = +[](void*, const std::shared_ptr<const M::ExtrudedTopology>&) {};
    EXPECT_THROW(P::details::ExtrudedConstructionOwnerAccess::visit(moved, sink, identity, topology_visit),
        std::invalid_argument);
    EXPECT_TRUE(census.owners.empty());
    auto copied_again = std::make_shared<Ordinary>(*admitted);
    EXPECT_THROW(P::details::ExtrudedConstructionOwnerAccess::visit(copied_again, sink, identity, topology_visit),
        std::invalid_argument);
    const auto rewrapped =
        P::details::ExtrudedConstructionAccess::shared_view(topology, moved, moved, 0, 2, domain.admission);
    EXPECT_NO_THROW(P::details::ExtrudedConstructionOwnerAccess::visit(rewrapped, sink, identity, topology_visit));
    // A foreign shared control around the same legacy object also fails its
    // private factory-issued exact-self weak-control check.
    const auto foreign = std::shared_ptr<const Ordinary>(legacy.get(), [](const Ordinary*) {});
    EXPECT_THROW(P::details::ExtrudedConstructionOwnerAccess::visit(foreign, sink, identity, topology_visit),
        std::invalid_argument);
}
TEST(NativeConstructorStorage, ForeignEnvelopeOrTargetControlsRejectBeforeProjection)
{
    if (!G::native_constructor_capacity_profile_supported())
        GTEST_SKIP() << "Unqualified STL profile";
    Domain domain;
    const auto storage = G::native_constructor_storage(domain.admission);
    Census census;
    const auto sink = census.sink();
    const auto foreign_envelope =
        std::shared_ptr<const G::NativeAllocationEnvelope>(storage.envelope.get(), [](const auto*) {});
    EXPECT_THROW(sink.fixed(foreign_envelope), std::invalid_argument);
    EXPECT_THROW(
        G::NativeConstructorOwnerSink::require_envelope(foreign_envelope, storage.owner), std::invalid_argument);
    const auto foreign_owner = std::shared_ptr<G::NativeConstructorStorage>(storage.owner.get(), [](auto*) {});
    EXPECT_THROW(sink.storage_reference(foreign_owner, storage.envelope), std::invalid_argument);
    EXPECT_TRUE(census.owners.empty());
    EXPECT_TRUE(census.envelopes.empty());
}
} // namespace
