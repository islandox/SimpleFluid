/** Accepted-source geometry residency uses the sealed exact current root union. */
#include "adaptive_storage_test_helpers.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <optional>
#include <string_view>
#include <tuple>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
namespace F = SimpleFluid::test::adaptive_storage;
using SimpleFluid::ArithmeticMode;
using Hierarchy = AMR::DistributedAdaptiveHierarchy;
using Access = D::GeometrySnapshotAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t ample = 64 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::vector<const AMR::AdaptiveLeaf*> rows(const Hierarchy& source)
{
    std::vector<const AMR::AdaptiveLeaf*> selected;
    for (const auto& leaf : source.owned_leaves())
        selected.push_back(&leaf);
    return selected;
}
std::shared_ptr<const AMR::AdaptiveGeometrySnapshot> acquire(const Hierarchy& source,
    std::span<const AMR::AdaptiveLeaf* const> selected, AMR::GeometrySnapshotPurpose purpose,
    ArithmeticMode mode = ordinary)
{
    std::size_t used = 0;
    return source.geometry_snapshot(selected, mode, purpose, used, ample);
}
auto make_source(F::Provider provider, std::size_t components)
{
    auto fixture = F::make_disconnected(provider, components);
    return std::make_shared<const Hierarchy>(std::move(fixture.owned), std::vector<AMR::AdaptiveFamily>{},
        std::move(fixture.kernel), Tpetra::getDefaultComm());
}
/** Independent full graph and administrative-credit census. It includes
 * snapshot headers and stored index controls without consulting the cache. */
struct UnionReconstruction
{
    std::size_t scalar = 0, payload = 0;
};
using PhysicalKey = std::pair<std::uint8_t, const void*>;
struct PhysicalOwner
{
    std::shared_ptr<const void> owner;
    std::size_t bytes;
};
using PhysicalOwners = std::map<PhysicalKey, PhysicalOwner>;
PhysicalOwners walked_owners(const Hierarchy& source)
{
    PhysicalOwners result;
    source.visit_geometry_snapshots(
        [&](const auto& snapshot)
        {
            P::GeometryAllocationLedger ordinary, bounded;
            std::size_t used = 0;
            snapshot.retain(ordinary, bounded, used, ample);
            const P::details::geometry_demand::Admission allocation{
                &used, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
            for (unsigned channel = 0; channel < 2; ++channel)
                P::details::GeometryOwnerImportAccess::export_manifest(
                    channel ? bounded : ordinary, allocation, nullptr, channel)
                    .visit(
                        [&](const auto& row)
                        {
                            auto [at, added] = result.emplace(
                                PhysicalKey{row.key.kind, row.owner.get()}, PhysicalOwner{row.owner, row.bytes});
                            if (!added)
                            {
                                EXPECT_EQ(at->second.bytes, row.bytes);
                                EXPECT_FALSE(row.owner.owner_before(at->second.owner));
                                EXPECT_FALSE(at->second.owner.owner_before(row.owner));
                            }
                        });
        });
    return result;
}
UnionReconstruction fresh_union(const Hierarchy& source)
{
    const auto expected = walked_owners(source);
    PhysicalOwners actual;
    struct Metadata
    {
        std::shared_ptr<const void> owner;
        std::size_t admitted;
    };
    std::map<const void*, Metadata> administration;
    Access::visit_cache_owner_union(source,
        [&](const auto& row)
        {
            auto [at, added] =
                actual.emplace(PhysicalKey{row.key.kind, row.owner.get()}, PhysicalOwner{row.owner, row.bytes});
            if (!added)
            {
                EXPECT_EQ(at->second.bytes, row.bytes);
                EXPECT_FALSE(row.owner.owner_before(at->second.owner));
                EXPECT_FALSE(at->second.owner.owner_before(row.owner));
            }
        },
        {&administration, [](void* raw, const std::shared_ptr<const void>& owner, std::size_t, std::size_t admitted)
            {
                auto& rows = *static_cast<decltype(administration)*>(raw);
                EXPECT_TRUE(rows.emplace(owner.get(), Metadata{owner, admitted}).second);
                return true;
            }});
    EXPECT_EQ(actual.size(), expected.size());
    std::size_t payload = 0;
    for (const auto& [key, row] : expected)
    {
        const auto found = actual.find(key);
        EXPECT_NE(found, actual.end());
        if (found != actual.end())
        {
            EXPECT_EQ(found->second.bytes, row.bytes);
            EXPECT_FALSE(found->second.owner.owner_before(row.owner));
            EXPECT_FALSE(row.owner.owner_before(found->second.owner));
        }
        payload += row.bytes;
    }
    std::map<const void*, Metadata> controls;
    source.visit_geometry_snapshots(
        [&](const auto& snapshot)
        {
            Access::visit_owner_manifest_metadata(snapshot,
                {&controls,
                    [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
                    {
                        auto& rows = *static_cast<decltype(controls)*>(raw);
                        auto [at, added] = rows.emplace(owner.get(), Metadata{owner, admitted - payload});
                        if (!added)
                            EXPECT_EQ(at->second.admitted, admitted - payload);
                        return true;
                    }});
        });
    std::size_t admin = 0;
    for (const auto& [owner, row] : administration)
        admin += row.admitted;
    for (const auto& [owner, row] : controls)
        admin += row.admitted;
    return {payload + admin, payload};
}
/** Full reference for the native cache. Start from the actual bound snapshot,
 * source and output root bindings, without using the maintained union rows.
 * Mutable index pages are inspected only for their own storage extents. */
struct NativeCacheReference
{
    using Charge = P::details::NativePaidExtentCharge;
    using Observation = P::details::NativeWeakOwnerObservation;
    using Manifest = P::details::geometry_demand::GeometryOwnerManifest;
    using MetadataSink = P::details::geometry_demand::MetadataSink;
    static std::uint8_t external() { return P::details::GeometryOwnerGraphEditor::external_kind(); }

    PhysicalOwners owners, paid;
    std::map<const void*, std::size_t> root_control_allowance;
    std::shared_ptr<const void> arena;

    static bool include(PhysicalOwners& rows, std::uint8_t kind,
        const std::shared_ptr<const void>& owner, std::size_t bytes)
    {
        auto [at, inserted] = rows.emplace(PhysicalKey{kind, owner.get()}, PhysicalOwner{owner, bytes});
        EXPECT_EQ(at->second.bytes, bytes);
        EXPECT_FALSE(at->second.owner.owner_before(owner));
        EXPECT_FALSE(owner.owner_before(at->second.owner));
        return inserted;
    }
    void charge(const Charge& claim)
    {
        if (!claim.valid() || claim.source_only())
            return;
        const auto origin = claim.origin();
        if (origin.get() == arena.get() && !origin.owner_before(arena) && !arena.owner_before(origin))
            include(paid, claim.kind(), claim.owner(), claim.bytes());
    }
    void state(const std::shared_ptr<const SimpleFluid::geometry_detail::NativeAllocationState>& value)
    {
        include(owners, external(), value, value->allocation.bytes);
        include(owners, external(), value->counter_owner,
            value->counter_allocations.object.bytes + value->counter_allocations.control.bytes);
        if (value->mirror.owner)
            include(owners, external(), value->mirror.owner, value->mirror.owner_proof.bytes());
    }
    void fixed(const std::shared_ptr<const SimpleFluid::geometry_detail::NativeAllocationEnvelope>& allocation)
    {
        std::size_t bytes = 0;
        allocation->visit_allocations([&](const auto& extent) { bytes += extent.bytes; });
        include(owners, external(), allocation, bytes);
        state(allocation->state());
    }
    void weak(const Observation& observation)
    {
        if (const auto* envelope = observation.native_envelope())
            fixed(*envelope);
        if (const auto* native = observation.native_state())
            state(*native);
        if (const auto* scalar = observation.detached_scalar())
        {
            include(owners, external(), scalar->owner, scalar->bytes);
            if (scalar->keeper)
                include(owners, external(), scalar->keeper, scalar->keeper_bytes);
        }
        if (const auto* contact = observation.contact_control_group())
        {
            include(owners, external(), contact->owner, contact->bytes);
            include(owners, external(), contact->keeper, contact->keeper_bytes);
        }
    }
    void financial(const Charge& value)
    {
        weak(value.physical_observation());
        weak(value.financial_observation());
        value.inspect_financial({this,
            [](void* raw, auto kind, const auto& owner, auto bytes)
            { include(static_cast<NativeCacheReference*>(raw)->owners, kind, owner, bytes); },
            [](void* raw, const auto& allocation) { static_cast<NativeCacheReference*>(raw)->fixed(allocation); },
            [](void* raw, const auto& storage)
            {
                auto& audit = *static_cast<NativeCacheReference*>(raw);
                audit.fixed(storage.envelope);
                include(audit.owners, external(), storage.owner, storage.owner->exclusive_bytes());
            },
            [](void* raw, const auto& allocation) { static_cast<NativeCacheReference*>(raw)->state(allocation); },
            nullptr,
            [](void* raw, const auto& observation) { static_cast<NativeCacheReference*>(raw)->weak(observation); }});
    }
    void root(const P::details::GeometryOwnerRootBinding& binding)
    {
        if (!binding)
            return;
        // Both required native root types account their header through its
        // native fixed allocation and a zero-byte semantic owner record.
        include(owners, external(), binding.header(), 0);
        const MetadataSink metadata{this,
            [](void* raw, const auto& owner, auto payload, auto admitted)
            {
                auto& audit = *static_cast<NativeCacheReference*>(raw);
                auto [at, inserted] = audit.root_control_allowance.emplace(owner.get(), admitted - payload);
                EXPECT_EQ(at->second, admitted - payload);
                include(audit.owners, external(), owner, payload);
                return inserted;
            },
            [](void* raw, const auto& owner, auto bytes)
            { return include(static_cast<NativeCacheReference*>(raw)->owners, external(), owner, bytes); },
            [](void* raw, const Charge& value) { static_cast<NativeCacheReference*>(raw)->charge(value); },
            [](void* raw, const Charge& value) { static_cast<NativeCacheReference*>(raw)->financial(value); }};
        const Manifest::RecordSink records{this, [](void* raw, const auto& row)
        { include(static_cast<NativeCacheReference*>(raw)->owners, row.key.kind, row.owner, row.bytes); }};
        if (const auto* receipts = binding.paid_receipts())
        {
            receipts->visit({this, [](void* raw, const Charge& value, auto)
            { static_cast<NativeCacheReference*>(raw)->charge(value); }});
            receipts->visit_administration(metadata);
        }
        for (const auto& payload : binding.payload().owners)
            payload.visit_dependencies(metadata, records);
        if (const auto* group = binding.source_group(); group && group->origin().get() == arena.get() &&
            !group->origin().owner_before(arena) && !arena.owner_before(group->origin()))
            group->visit_delta({}, {this, [](void* raw, auto kind, const auto& owner, auto bytes, bool retain)
            {
                EXPECT_TRUE(retain);
                include(static_cast<NativeCacheReference*>(raw)->paid, kind, owner, bytes);
            }});
    }
};
void expect_native_seal(const Hierarchy& source)
{
    using Charge = NativeCacheReference::Charge;
    NativeCacheReference expected;
    expected.owners = walked_owners(source);
    expected.arena = Access::contact_cache(source).allocation_arena();
    source.visit_geometry_snapshots([&](const auto& snapshot)
    {
        Access::visit_owner_manifest_metadata(snapshot, {&expected.root_control_allowance,
            [](void* raw, const auto& owner, auto payload, auto admitted)
            {
                auto& controls = *static_cast<decltype(expected.root_control_allowance)*>(raw);
                const auto [at, inserted] = controls.emplace(owner.get(), admitted - payload);
                EXPECT_EQ(at->second, admitted - payload);
                return inserted;
            }});
    });
    Access::visit_native_cache_bindings(source, [&](const auto& root) { expected.root(root); },
        [&](const auto& charge) { expected.charge(charge); });
    PhysicalOwners actual;
    struct Administration
    {
        NativeCacheReference& expected;
        PhysicalOwners owners;
    } administration{expected, {}};
    Access::visit_cache_owner_union(source,
        [&](const auto& row) { NativeCacheReference::include(actual, row.key.kind, row.owner, row.bytes); },
        {&administration,
            [](void* raw, const auto& owner, auto, auto admitted)
            { return NativeCacheReference::include(static_cast<Administration*>(raw)->owners,
                NativeCacheReference::external(), owner, admitted); },
            [](void* raw, const auto& owner, auto bytes)
            { return NativeCacheReference::include(static_cast<Administration*>(raw)->owners,
                NativeCacheReference::external(), owner, bytes); },
            [](void* raw, const Charge& value) { static_cast<Administration*>(raw)->expected.charge(value); },
            [](void*, const Charge&) {}});
    ASSERT_EQ(actual.size(), expected.owners.size());
    std::size_t resident = 0;
    for (const auto& [key, row] : expected.owners)
    {
        const auto found = actual.find(key);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.bytes, row.bytes);
        EXPECT_FALSE(found->second.owner.owner_before(row.owner));
        EXPECT_FALSE(row.owner.owner_before(found->second.owner));
        resident += row.bytes;
    }
    for (const auto& [address, bytes] : expected.root_control_allowance)
        resident += bytes;
    for (const auto& [key, row] : administration.owners)
        if (!expected.owners.contains(key))
            resident += row.bytes;
    std::size_t paid = 0;
    for (const auto& [key, row] : expected.paid)
        paid += row.bytes;
    const auto seal = Access::cache_admission_statistics(source);
    EXPECT_EQ(seal.resident_bytes, resident);
    EXPECT_EQ(seal.lease_bytes + paid, resident);
}
std::size_t walked_payload(const Hierarchy& source)
{
    const auto rows = walked_owners(source);
    std::size_t bytes = 0;
    for (const auto& [key, row] : rows)
        bytes += row.bytes;
    return bytes;
}
std::size_t shared_owner_count(const Hierarchy& source)
{
    using Key = std::tuple<unsigned, std::uint8_t, const void*>;
    std::map<Key, std::size_t> multiplicity;
    source.visit_geometry_snapshots(
        [&](const auto& snapshot)
        {
            P::GeometryAllocationLedger ordinary_owners, bounded_owners;
            std::size_t bytes = 0;
            snapshot.retain(ordinary_owners, bounded_owners, bytes, ample);
            const auto record = [&](unsigned channel, const P::GeometryAllocationLedger& ledger)
            {
                P::details::GeometryAllocationOwnerAccess::visit(ledger,
                    [&](std::uint8_t kind, const void* owner, std::size_t) { ++multiplicity[{channel, kind, owner}]; });
            };
            record(0, ordinary_owners);
            record(1, bounded_owners);
        });
    return std::count_if(multiplicity.begin(), multiplicity.end(), [](const auto& item) { return item.second > 1; });
}
void expect_exact_seal(const Hierarchy& source)
{
    const auto before = Access::cache_admission_statistics(source);
    const auto reconstructed = fresh_union(source);
    EXPECT_EQ(before.resident_bytes, reconstructed.scalar);
    EXPECT_EQ(before.lease_bytes, before.resident_bytes);
    // The maintained manifests and the snapshot graph must name the same exact
    // owner payload; only the admission bookkeeping of the two shapes differs.
    EXPECT_EQ(reconstructed.payload, walked_payload(source));
    const auto after = Access::cache_admission_statistics(source);
    EXPECT_EQ(after.union_censuses, before.union_censuses);
    EXPECT_EQ(after.census_root_visits, before.census_root_visits);
}
} // namespace

TEST(AdaptiveGeometryCacheAdmission, WarmProviderSourcesAdmitTheExactUnionWithoutVisitingSnapshotRoots)
{
    for (const auto provider : F::providers)
        for (const auto components : {2u, 4u})
        {
            SCOPED_TRACE(F::name(provider));
            SCOPED_TRACE(components);
            auto source = make_source(provider, components);
            const auto selected = rows(*source);
            ASSERT_GT(selected.size(), 1u);
            const auto prefix = std::span(selected).first(selected.size() / 2);
            auto first = acquire(*source, prefix, AMR::GeometrySnapshotPurpose::State);
            const auto raw = acquire(*source, prefix, AMR::GeometrySnapshotPurpose::OriginalSources);
            expect_exact_seal(*source);
            EXPECT_GT(shared_owner_count(*source), 0u);
            const auto before = Access::cache_admission_statistics(*source);
            const auto required = before.cache_header_bytes + before.resident_bytes;
            D::Budget exact{required};
            Access::admit_cache_residency(*source, exact);
            EXPECT_EQ(exact.used, required);
            EXPECT_EQ(exact.peak, required);
            D::Budget short_limit{required - 1};
            EXPECT_THROW(Access::admit_cache_residency(*source, short_limit), std::invalid_argument);
            EXPECT_EQ(short_limit.used, before.cache_header_bytes);
            const auto after = Access::cache_admission_statistics(*source);
            EXPECT_EQ(after.admission_queries, before.admission_queries + 2);
            EXPECT_EQ(after.union_censuses, before.union_censuses);
            EXPECT_EQ(after.census_root_visits, before.census_root_visits);
            EXPECT_EQ(after.resident_bytes, before.resident_bytes);

            // Replacement stages only the changed immutable root frontier.
            const auto replacement = acquire(*source, selected, AMR::GeometrySnapshotPurpose::State);
            EXPECT_NE(replacement, first);
            const auto changed = Access::cache_admission_statistics(*source);
            EXPECT_EQ(changed.union_censuses, after.union_censuses);
            EXPECT_EQ(changed.census_root_visits, after.census_root_visits);
            EXPECT_GT(changed.union_updates, after.union_updates);
            expect_exact_seal(*source);
            D::Budget after_replacement{ample};
            Access::admit_cache_residency(*source, after_replacement);
            EXPECT_EQ(after_replacement.used, changed.cache_header_bytes + changed.resident_bytes);
            const auto admitted = Access::cache_admission_statistics(*source);
            EXPECT_EQ(admitted.census_root_visits, changed.census_root_visits);
            EXPECT_TRUE(first->matches_layout(source->leaf_layout()));
            EXPECT_TRUE(raw->matches_layout(source->leaf_layout()));
            EXPECT_EQ(first->ordinary_catalog()->size(), prefix.size());

            std::weak_ptr<const AMR::AdaptiveLeafLayout> layout = source->leaf_layout();
            source.reset();
            EXPECT_TRUE(layout.expired());
            EXPECT_EQ(first->native_metrics().size(), prefix.size());
            EXPECT_GT(first->native_metrics()[0].volume, 0);
            EXPECT_GT(replacement->native_metrics()[selected.size() - 1].volume, 0);
        }
}

TEST(AdaptiveGeometryCacheAdmission, AllArithmeticRootsAndChangedControlsKeepAnExactSeal)
{
    if (!AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires inspectable construction controls.";
    struct Environment
    {
        std::fenv_t saved{};
        int rounding = std::fegetround();
        Environment() { (void) std::fegetenv(&saved); }
        ~Environment() { (void) std::fesetenv(&saved); }
    } environment;
    const auto source = make_source(F::Provider::Cartesian, 2);
    const auto selected = rows(*source);
    std::shared_ptr<const AMR::AdaptiveGeometrySnapshot> original;
    for (const auto mode : {ordinary, ArithmeticMode::BoundedCertification, ArithmeticMode::Exact})
        for (const auto purpose : {AMR::GeometrySnapshotPurpose::State, AMR::GeometrySnapshotPurpose::OriginalSources})
        {
            auto root = acquire(*source, selected, purpose, mode);
            if (mode == ordinary && purpose == AMR::GeometrySnapshotPurpose::State)
                original = root;
            expect_exact_seal(*source);
        }
    const auto native = original->native_metrics()[0];
    const auto before = Access::cache_admission_statistics(*source);
    auto stale = *selected.front();
    ++stale.revision;
    EXPECT_THROW(
        (void) acquire(*source, std::array{&stale}, AMR::GeometrySnapshotPurpose::State), std::invalid_argument);
    EXPECT_EQ(Access::cache_admission_statistics(*source).resident_bytes, before.resident_bytes);
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    const auto changed = acquire(*source, std::array{selected.front()}, AMR::GeometrySnapshotPurpose::State);
    EXPECT_NE(changed, original);
    EXPECT_EQ(changed->construction_controls().rounding, FE_UPWARD);
    EXPECT_EQ(changed->ordinary_catalog()->size(), 1u);
    expect_exact_seal(*source);
    ASSERT_EQ(std::fesetround(environment.rounding), 0);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(original->native_metrics()[0].volume),
        std::bit_cast<std::uint64_t>(native.volume));
}

TEST(AdaptiveGeometryCacheAdmission, AdaptationAndNativeBindingReuseTheAcceptedSeal)
{
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    const AMR::CartesianAMR mesh(grid);
    std::vector<std::uint64_t> indices(mesh.geometry()->num_cells());
    std::iota(indices.begin(), indices.end(), 0);
    auto detached = mesh.detach_owned(indices, {'w', 'a', 'r', 'm'});
    const auto source = std::make_shared<const Hierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), Tpetra::getDefaultComm());
    const auto selected = rows(*source);
    const auto geometry = acquire(*source, selected, AMR::GeometrySnapshotPurpose::State);
    const AMR::AdaptiveMark mark{selected.front()->key, AMR::AdaptiveSplit::Isotropic};
    const auto before_adaptation = Access::cache_admission_statistics(*source);
    const auto plan = source->plan_adaptation(std::array{mark});
    const auto after_adaptation = Access::cache_admission_statistics(*source);
    EXPECT_GT(after_adaptation.admission_queries, before_adaptation.admission_queries);
    EXPECT_EQ(after_adaptation.census_root_visits, before_adaptation.census_root_visits);
    EXPECT_FALSE(geometry->matches_layout(plan.candidate()->leaf_layout()));
    expect_native_seal(*source);
    auto native = AMR::publish_adaptive_native(*source, AMR::native_publication_schema(*source));
    const auto before_binding = Access::cache_admission_statistics(*source);
    const auto bound = source->with_partition(native.partition(), native.mesh()->owned_cell_map());
    const auto after_binding = Access::cache_admission_statistics(*source);
    EXPECT_EQ(after_binding.census_root_visits, before_binding.census_root_visits);
    EXPECT_GT(after_binding.admission_queries, before_binding.admission_queries);
    const auto bound_stats = Access::cache_admission_statistics(*bound);
    EXPECT_EQ(bound_stats.union_censuses, 0u);
    EXPECT_EQ(bound_stats.census_root_visits, 0u);
    EXPECT_EQ(acquire(*bound, selected, AMR::GeometrySnapshotPurpose::State), geometry);
    auto previous = AMR::publish_adaptive_native(*bound, AMR::native_publication_schema(*bound));
    auto current = AMR::publish_adaptive_native(*bound, AMR::native_publication_schema(*bound));
    ASSERT_NE(previous.partition(), current.partition());
    expect_native_seal(*bound);
}

TEST(AdaptiveGeometryCacheAdmission, WarmHitInstallsTheMaintainedExactOwnerSetIntoAFreshLedger)
{
    for (const auto provider : {F::Provider::Cartesian, F::Provider::Static})
    {
        SCOPED_TRACE(F::name(provider));
        auto source = make_source(provider, 4);
        const auto selected = rows(*source);
        ASSERT_GT(selected.size(), 1u);
        const auto first = acquire(*source, selected, AMR::GeometrySnapshotPurpose::State);

        // A fresh supplied ledger takes the warm fast path: it installs the
        // maintained immutable owner set and charges that exact cached union.
        P::GeometryAllocationLedger ordinary_owners, bounded_owners;
        std::size_t used = 0;
        const auto hit = source->geometry_snapshot(
            selected, ordinary, AMR::GeometrySnapshotPurpose::State, used, ample, &ordinary_owners, &bounded_owners);
        ASSERT_EQ(hit, first);
        EXPECT_GT(ordinary_owners.owner_count(), 0u);
        EXPECT_GT(used, 0u);

        // The installed set is exactly the root's own census, not a subset.
        P::GeometryAllocationLedger census_ordinary, census_bounded;
        std::size_t census_used = 0;
        hit->retain(census_ordinary, census_bounded, census_used, ample);
        const auto pair = Access::owner_pair(*hit);
        // Imported own-index storage is bookkeeping; the payload set contains
        // the complete stored body plus the explicitly admitted snapshot header.
        EXPECT_EQ(ordinary_owners.payload_bytes(), pair.owners[0].payload_bytes() + sizeof(*hit));
        EXPECT_EQ(ordinary_owners.owner_count(), pair.owners[0].owner_count() + 1);
        EXPECT_LE(used, census_used);

        // Imported owners are recognized by later retention of the same root:
        // no additional owner, no additional cumulative admission.
        const auto owners_before = ordinary_owners.owner_count();
        const auto used_before = used;
        P::details::GeometryAllocationOwnerAccess::reset_work_statistics();
        EXPECT_EQ(source->geometry_snapshot(selected, ordinary, AMR::GeometrySnapshotPurpose::State, used, ample,
                      &ordinary_owners, &bounded_owners),
            hit);
        EXPECT_EQ(used, used_before);
        const auto qualified = P::details::GeometryAllocationOwnerAccess::work_statistics();
        EXPECT_EQ(qualified.import_fallbacks, 0u);
        EXPECT_EQ(qualified.metadata_credit_seed_rows, 0u);
        const auto bookkeeping_before = ordinary_owners.bookkeeping_bytes() + bounded_owners.bookkeeping_bytes();
        hit->retain(ordinary_owners, bounded_owners, used, ample);
        EXPECT_EQ(ordinary_owners.owner_count(), owners_before);
        EXPECT_EQ(used - used_before,
            ordinary_owners.bookkeeping_bytes() + bounded_owners.bookkeeping_bytes() - bookkeeping_before);
        const auto retained = used;
        hit->retain(ordinary_owners, bounded_owners, used, ample);
        EXPECT_EQ(used, retained);
        expect_exact_seal(*source);
    }
}

TEST(AdaptiveGeometryCacheAdmission, RejectedDemandConstructionKeepsAcceptedRootsAndSeals)
{
    if (!AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires inspectable construction controls.";
    const auto source = make_source(F::Provider::Cartesian, 4);
    const auto selected = rows(*source);
    ASSERT_GT(selected.size(), 2u);
    const auto prefix = std::span(selected).first(selected.size() / 2);
    const auto accepted = acquire(*source, prefix, AMR::GeometrySnapshotPurpose::State);
    expect_exact_seal(*source);
    const auto before = Access::cache_admission_statistics(*source);

    // A twin hierarchy of the same shape discovers what the same demand
    // extension costs; the real one is then given an insufficient cap.
    const auto twin = make_source(F::Provider::Cartesian, 4);
    const auto twin_rows = rows(*twin);
    ASSERT_EQ(twin_rows.size(), selected.size());
    std::size_t twin_used = 0;
    (void) twin->geometry_snapshot(
        std::span(twin_rows).first(prefix.size()), ordinary, AMR::GeometrySnapshotPurpose::State, twin_used, ample);
    twin_used = 0;
    (void) twin->geometry_snapshot(twin_rows, ordinary, AMR::GeometrySnapshotPurpose::State, twin_used, ample);
    ASSERT_GT(twin_used, 0u);

    std::size_t used = 0;
    EXPECT_THROW(
        (void) source->geometry_snapshot(selected, ordinary, AMR::GeometrySnapshotPurpose::State, used, twin_used / 2),
        std::exception);
    // The genuinely spent prefix stays accounted for after the rejection.
    EXPECT_GT(used, 0u);
    // Accepted roots and their exact residency seal are unchanged.
    const auto after = Access::cache_admission_statistics(*source);
    EXPECT_EQ(after.resident_bytes, before.resident_bytes);
    EXPECT_EQ(after.lease_bytes, before.lease_bytes);
    expect_exact_seal(*source);
    D::Budget exact{after.cache_header_bytes + after.resident_bytes};
    Access::admit_cache_residency(*source, exact);
    EXPECT_EQ(exact.used, after.cache_header_bytes + after.resident_bytes);
    D::Budget short_limit{after.cache_header_bytes + after.resident_bytes - 1};
    EXPECT_THROW(Access::admit_cache_residency(*source, short_limit), std::invalid_argument);

    // The accepted root is still the answer to its own all-known request, and a
    // later ample demand still publishes a complete extension.
    const auto hit = acquire(*source, prefix, AMR::GeometrySnapshotPurpose::State);
    EXPECT_EQ(hit, accepted);
    const auto extended = acquire(*source, selected, AMR::GeometrySnapshotPurpose::State);
    EXPECT_EQ(extended->ordinary_catalog()->size(), selected.size());
    expect_exact_seal(*source);
}

TEST(AdaptiveGeometryCacheAdmission, LateLeaseRejectionPreservesManifestSealAndAcceptedRoot)
{
    const auto source = make_source(F::Provider::Static, 4);
    const auto selected = rows(*source);
    ASSERT_GT(selected.size(), 2u);
    const auto prefix = std::span(selected).first(selected.size() / 2);
    const auto accepted = acquire(*source, prefix, AMR::GeometrySnapshotPurpose::State);
    const auto before = Access::cache_admission_statistics(*source);
    const auto payload_before = walked_payload(*source);
    {
        // Exhaust only the backing arena. Construction/capture use their ample
        // finite budgets and finish, then final publication lease growth fails.
        auto blocker = Access::contact_cache(*source).external_residency_lease();
        blocker.resize(blocker.available());
        std::size_t used = 0;
        P::GeometryAllocationLedger operation_ordinary, operation_bounded;
        EXPECT_THROW((void) source->geometry_snapshot(selected, ordinary, AMR::GeometrySnapshotPurpose::State, used,
                         ample, &operation_ordinary, &operation_bounded),
            std::exception);
        EXPECT_GT(used, 0u);
        const auto after_failure = used;
        const auto owner = std::make_shared<const int>(1);
        EXPECT_THROW(operation_ordinary.retain_external_owner(owner, sizeof(int), used, ample), P::GeometryError);
        EXPECT_THROW(operation_bounded.retain_external_owner(owner, sizeof(int), used, ample), P::GeometryError);
        EXPECT_EQ(used, after_failure);
        const auto after = Access::cache_admission_statistics(*source);
        EXPECT_EQ(after.union_censuses, before.union_censuses);
        EXPECT_GT(after.union_updates, before.union_updates);
        EXPECT_EQ(after.resident_bytes, before.resident_bytes);
        EXPECT_EQ(after.lease_bytes, before.lease_bytes);
        EXPECT_EQ(walked_payload(*source), payload_before);
        expect_exact_seal(*source);
        EXPECT_EQ(acquire(*source, prefix, AMR::GeometrySnapshotPurpose::State), accepted);
    }
    const auto extended = acquire(*source, selected, AMR::GeometrySnapshotPurpose::State);
    EXPECT_EQ(extended->ordinary_catalog()->size(), selected.size());
    expect_exact_seal(*source);
}

TEST(AdaptiveGeometryCacheAdmission, StoredSnapshotManifestSharesItsIndexWithoutCaptureAllocation)
{
    const auto source = make_source(F::Provider::Static, 2);
    const auto snapshot = acquire(*source, rows(*source), AMR::GeometrySnapshotPurpose::State);
    D::Budget no_allocation{0};
    const auto first = Access::owner_manifest(*snapshot, no_allocation);
    const auto second = Access::owner_manifest(*snapshot, no_allocation);
    EXPECT_GT(first.owner_count(), 0u);
    EXPECT_EQ(no_allocation.used, 0u);
    EXPECT_EQ(no_allocation.peak, 0u);
    const auto root = [](const auto& manifest)
    {
        std::shared_ptr<const void> result;
        manifest.visit_dependencies(
            {&result,
                [](void* raw, const std::shared_ptr<const void>& owner, std::size_t, std::size_t)
                {
                    *static_cast<std::shared_ptr<const void>*>(raw) = owner;
                    return false;
                }},
            {nullptr, [](void*, const auto&) {}});
        return result;
    };
    const auto first_root = root(first), second_root = root(second);
    EXPECT_EQ(first_root.get(), second_root.get());
    EXPECT_FALSE(first_root.owner_before(second_root));
    EXPECT_FALSE(second_root.owner_before(first_root));
    expect_exact_seal(*source);
}

TEST(AdaptiveGeometryCacheAdmission, OptionalTwoBlockLookaheadReservesUnionAndRetirementWithOtherRoots)
{
    if (Tpetra::getDefaultComm()->getSize() != 1 || !AMR::geometry_construction_controls().supported)
        GTEST_SKIP() << "Requires one rank and inspectable construction controls.";
    struct Result
    {
        bool accepted = false;
        std::size_t prepared = 0;
    };
    const auto attempt = [&](std::size_t headroom)
    {
        AMR::AMRGrid grid;
        grid.block_cells = {8, 4, 4};
        AMR::CartesianAMR native(grid);
        std::vector<std::uint64_t> ids(native.geometry()->num_cells());
        std::iota(ids.begin(), ids.end(), 0);
        auto input = native.detach_owned(ids, {'b', 'o', 'u', 'n', 'd', 'a', 'r', 'y'});
        auto source = std::make_shared<const Hierarchy>(
            std::move(input.owned), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
        const AMR::AdaptiveLeaf *initial = nullptr, *requested = nullptr, *unrelated = nullptr;
        for (const auto& leaf : source->owned_leaves())
        {
            if (leaf.native_cell == 0)
                initial = &leaf;
            if (leaf.native_cell == 60)
                requested = &leaf;
            if (leaf.native_cell == 127)
                unrelated = &leaf;
        }
        EXPECT_TRUE(initial && requested && unrelated);
        if (!initial || !requested || !unrelated)
            return Result{};
        const auto accepted = acquire(*source, std::array{initial}, AMR::GeometrySnapshotPurpose::State);
        const auto raw = acquire(*source, std::array{unrelated}, AMR::GeometrySnapshotPurpose::OriginalSources);
        const auto bounded = acquire(
            *source, std::array{unrelated}, AMR::GeometrySnapshotPurpose::State, ArithmeticMode::BoundedCertification);
        const auto before = Access::cache_admission_statistics(*source);
        auto blocker = Access::contact_cache(*source).external_residency_lease();
        const auto available = blocker.available();
        blocker.resize(available - std::min(available, headroom));
        P::GeometryAllocationLedger zero, one;
        std::size_t used = 37;
        P::details::GeometryAllocationOwnerAccess::reset_work_statistics();
        std::shared_ptr<const AMR::AdaptiveGeometrySnapshot> candidate;
        try
        {
            candidate = source->geometry_snapshot(
                std::array{requested}, ordinary, AMR::GeometrySnapshotPurpose::State, used, ample, &zero, &one);
        }
        catch (const std::invalid_argument& error)
        {
            // Only the constrained backing-arena rejection is a budget outcome.
            // Geometry/identity/control failures remain test failures.
            if (!std::string_view(error.what()).starts_with("Distributed adaptive payload budget exhausted:"))
                throw;
            const auto after = Access::cache_admission_statistics(*source);
            EXPECT_EQ(after.resident_bytes, before.resident_bytes);
            EXPECT_EQ(after.lease_bytes, before.lease_bytes);
            const auto spent = used;
            const auto probe = std::make_shared<const int>(1);
            EXPECT_THROW(zero.retain_external_owner(probe, sizeof(int), used, ample), P::GeometryError);
            EXPECT_THROW(one.retain_external_owner(probe, sizeof(int), used, ample), P::GeometryError);
            EXPECT_EQ(used, spent);
            EXPECT_EQ(acquire(*source, std::array{initial}, AMR::GeometrySnapshotPurpose::State), accepted);
            expect_exact_seal(*source);
            return Result{};
        }
        const auto work = P::details::GeometryAllocationOwnerAccess::work_statistics();
        EXPECT_EQ(work.import_fallbacks, 0u);
        EXPECT_EQ(work.metadata_credit_seed_rows, 0u);
        EXPECT_EQ(candidate->ordinary_catalog()->size(), 2u);
        const auto selected = candidate->find(*requested);
        EXPECT_TRUE(selected.has_value());
        if (selected)
            EXPECT_DOUBLE_EQ(candidate->native_metrics().at(*selected).volume, 1.0 / 128.0);
        std::size_t prepared = 0;
        candidate->visit_prepared_cartesian_sources(
            [&](const auto& page)
            {
                EXPECT_EQ(page.size(), 64u);
                ++prepared;
            });
        EXPECT_LE(prepared, 2u);
        // Their exact header controls must still be part of the maintained
        // root union after this independent Ordinary replacement.
        std::array<bool, 2> found{};
        const std::array<std::shared_ptr<const AMR::AdaptiveGeometrySnapshot>, 2> preserved{raw, bounded};
        Access::visit_cache_owner_union(*source,
            [&](const auto& row)
            {
                for (std::size_t i = 0; i < preserved.size(); ++i)
                    if (row.owner.get() == preserved[i].get())
                    {
                        EXPECT_FALSE(row.owner.owner_before(preserved[i]));
                        EXPECT_FALSE(preserved[i].owner_before(row.owner));
                        found[i] = true;
                    }
            },
            {nullptr, [](void*, const auto&, std::size_t, std::size_t) { return true; }});
        EXPECT_TRUE(found[0]);
        EXPECT_TRUE(found[1]);
        expect_exact_seal(*source);
        return Result{true, prepared};
    };
    std::optional<std::size_t> demand_only, one_page, two_pages;
    for (std::size_t headroom = 64 * 1024; headroom <= 32 * 1024 * 1024; headroom *= 2)
    {
        SCOPED_TRACE(headroom);
        const auto result = attempt(headroom);
        // Once required publication fits, optional work must never turn a
        // larger finite arena into a rejection of that same required demand.
        if (demand_only)
            EXPECT_TRUE(result.accepted);
        if (!result.accepted)
            continue;
        if (result.prepared == 0)
            demand_only = headroom;
        else if (result.prepared == 1)
            one_page = headroom;
        else
        {
            two_pages = headroom;
            break;
        }
    }
    ASSERT_TRUE(demand_only);
    ASSERT_TRUE(two_pages);
    if (!one_page)
    {
        auto lower = *demand_only, upper = *two_pages;
        while (upper - lower > 1)
        {
            const auto middle = lower + (upper - lower) / 2;
            SCOPED_TRACE(middle);
            const auto result = attempt(middle);
            ASSERT_TRUE(result.accepted);
            if (result.prepared == 1)
            {
                one_page = middle;
                break;
            }
            if (result.prepared == 0)
                lower = middle;
            else
                upper = middle;
        }
    }
    EXPECT_TRUE(one_page);
    if (demand_only)
        RecordProperty("DemandOnlyArenaHeadroom", std::to_string(*demand_only));
    if (one_page)
        RecordProperty("SinglePreparedBlockHeadroom", std::to_string(*one_page));
    if (two_pages)
        RecordProperty("TwoPreparedBlocksHeadroom", std::to_string(*two_pages));
}
