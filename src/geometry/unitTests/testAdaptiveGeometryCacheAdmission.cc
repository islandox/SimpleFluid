/** Accepted-source geometry residency uses the sealed exact current root union. */
#include "adaptive_storage_test_helpers.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
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
/** Exact union rebuilt from each cached root's own immutable owner manifests in
 * the cache's slot/channel order. Empty channels are not imported, exactly like
 * the cache. Independent of the cache's maintained set. */
struct UnionReconstruction
{
    std::size_t scalar = 0, payload = 0;
};
UnionReconstruction fresh_union(const Hierarchy& source)
{
    P::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t ordinary_used = 0, bounded_used = 0;
    source.visit_geometry_snapshots(
        [&](const auto& snapshot)
        {
            for (const bool bounded : {false, true})
            {
                D::Budget budget{ample};
                const auto manifest = D::GeometrySnapshotAccess::owner_manifest(snapshot, budget, bounded);
                if (!manifest.owner_count())
                    continue;
                auto& ledger = bounded ? bounded_owners : ordinary_owners;
                auto& used = bounded ? bounded_used : ordinary_used;
                P::details::GeometryOwnerImportAccess::import(ledger, manifest, used, ample);
            }
        });
    UnionReconstruction result;
    result.payload = ordinary_owners.payload_bytes() + bounded_owners.payload_bytes();
    result.scalar = result.payload + ordinary_owners.bookkeeping_bytes() + bounded_owners.bookkeeping_bytes();
    return result;
}
/** The same exact union reached by walking the snapshot dependency graph. */
std::size_t walked_payload(const Hierarchy& source)
{
    P::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t bytes = 0;
    source.visit_geometry_snapshots(
        [&](const auto& snapshot) { snapshot.retain(ordinary_owners, bounded_owners, bytes, ample); });
    return ordinary_owners.payload_bytes() + bounded_owners.payload_bytes();
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

            // Replacement remains a full exact union census in this stage.
            // The subsequent accepted-source query reuses its resulting seal.
            const auto replacement = acquire(*source, selected, AMR::GeometrySnapshotPurpose::State);
            EXPECT_NE(replacement, first);
            const auto changed = Access::cache_admission_statistics(*source);
            EXPECT_EQ(changed.union_censuses, after.union_censuses + 1);
            EXPECT_GT(changed.census_root_visits, after.census_root_visits);
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
    expect_exact_seal(*source);
    auto native = AMR::assemble_adaptive_native(*source, source->configuration_descriptor());
    const auto before_binding = Access::cache_admission_statistics(*source);
    const auto bound = source->with_partition(native.partition, native.mesh->owned_cell_map());
    const auto after_binding = Access::cache_admission_statistics(*source);
    EXPECT_EQ(after_binding.census_root_visits, before_binding.census_root_visits);
    EXPECT_GT(after_binding.admission_queries, before_binding.admission_queries);
    const auto bound_stats = Access::cache_admission_statistics(*bound);
    EXPECT_EQ(bound_stats.resident_bytes, before_binding.resident_bytes);
    EXPECT_EQ(bound_stats.union_censuses, 0u);
    EXPECT_EQ(bound_stats.census_root_visits, 0u);
    expect_exact_seal(*bound);
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
        EXPECT_EQ(ordinary_owners.payload_bytes(), census_ordinary.payload_bytes());
        EXPECT_EQ(ordinary_owners.owner_count(), census_ordinary.owner_count());
        EXPECT_LE(used, census_used);

        // Imported owners are recognized by later retention of the same root:
        // no additional owner, no additional cumulative admission.
        const auto owners_before = ordinary_owners.owner_count();
        const auto used_before = used;
        hit->retain(ordinary_owners, bounded_owners, used, ample);
        EXPECT_EQ(ordinary_owners.owner_count(), owners_before);
        EXPECT_EQ(used, used_before);
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
