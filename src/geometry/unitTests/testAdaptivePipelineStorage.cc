/** @file testAdaptivePipelineStorage.cc
 * @brief Whole-pipeline baseline observations and trace invariants.
 */
/** Whole retained C++ allocations across actual hierarchy transactions.
 * Explicit physical-owner census supplements allocations outside operator new.
 * Native publication is measured independently while its scope is unresolved. */
#include "adaptive_storage_test_helpers.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "pipeline_allocation_operators.inc"
#include "pipeline_allocation_trace.hh"
#include "pipeline_payload_roles.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace
{
namespace A = SimpleFluid::Meshes::AMR;
namespace D = A::adaptive_detail;
namespace F = SimpleFluid::test::adaptive_storage;
using State = std::shared_ptr<const A::DistributedAdaptiveHierarchy>;
using Access = D::GeometrySnapshotAccess;
using OwnerAccess = SimpleFluid::PLIC::details::GeometryAllocationOwnerAccess;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
inline std::array<std::size_t, static_cast<unsigned>(D::PipelineWork::Count)> pipeline_work{};
constexpr std::size_t ample = 256 * 1024 * 1024;
constexpr std::array phase_names{"candidate", "transport", "directory", "native_publication", "native_cell_routes"};
constexpr std::array work_names{"active_leaf_callback_visits", "leased_directory_wire_keys", "numeric_output_rows",
    "cell_geometry_requests", "surface_batch_requests", "exact_cell_shell_calls", "native_materialization_requests",
    "native_face_certificate_requests", "ordered_pages_created", "ordered_page_admitted_bytes", "leaf_pages_created",
    "leaf_row_iterator_dereferences", "wire_sent_packets", "wire_received_packets", "wire_sent_payload_bytes",
    "wire_received_payload_bytes", "wire_leaf_encode_visits", "wire_family_encode_visits", "wire_leaf_decode_visits",
    "wire_family_decode_visits", "wire_product_frame_encode_visits", "wire_product_frame_decode_visits",
    "native_compatibility_publications", "native_exact_partition_bindings", "native_compatibility_partition_bindings",
    "native_source_proof_hits", "native_source_proofs_issued", "native_incidence_proof_hits",
    "native_incidence_proofs_issued", "native_remote_source_hits", "native_remote_source_decodes",
    "native_remote_incidence_hits", "native_remote_incidence_decodes"};
constexpr std::array owner_phase_names{"first_publication", "repeat_publication", "adaptation", "migration",
    "edited_publication", "source_warm", "adapted_warm", "migrated_warm"};
static_assert(work_names.size() == static_cast<unsigned>(D::PipelineWork::Count));
static_assert(phase_names.size() == pipeline_trace::phases);

/** Persist named per-rank measurements in captured serial/MPI stdout. Reports
 * allocate only after operation capture; a diagnostic pause also protects the
 * prepared-page companion's report from contaminating its trace. */
void record_property(const std::string& key, const std::string& value)
{
    const pipeline_trace::Pause pause;
    testing::Test::RecordProperty(key, value);
    std::fprintf(stdout, "AMR_PIPELINE_BASELINE rank=%d ranks=%d key=%s value=%s\n",
        Tpetra::getDefaultComm()->getRank(), Tpetra::getDefaultComm()->getSize(), key.c_str(), value.c_str());
    std::fflush(stdout);
}

struct TraceLifetime
{
    TraceLifetime() { pipeline_trace::begin(); }
    ~TraceLifetime() { pipeline_trace::recording.store(false); }
};
struct ObserverScope
{
    D::PipelineObserver previous;
    ObserverScope()
        : previous(D::set_pipeline_observer({nullptr, [](void*, D::PipelinePhase value) noexcept
              { pipeline_trace::phase = static_cast<pipeline_trace::Phase>(value); },
              [](void*, D::PipelineWork value, std::size_t count) noexcept
              { pipeline_work[static_cast<unsigned>(value)] += count; }}))
    {
    }
    ~ObserverScope() { (void) D::set_pipeline_observer(previous); }
};
struct Graph
{
    std::size_t owners = 0, traced_owners = 0, untraced_owner_bytes = 0;
    std::set<const void*> identities;
    std::map<std::uint64_t, std::size_t> allocations;
    std::map<std::uint64_t, std::size_t> scalar_numeric_allocations;
    std::array<std::size_t, static_cast<unsigned>(D::ContactPayloadRole::Count)> role_bytes{}, role_entries{},
        role_extent_bytes{};
};
Graph graph(const State& state, unsigned membership)
{
    pipeline_trace::Pause pause;
    pipeline_trace::prepare_census();
    Graph result;
    Access::visit_pipeline_graph(*state,
        [&](const D::PipelineGraphOwner& owner)
        {
            // Only exact standalone scalar-buffer extents qualify. An interior
            // numeric address never exempts its enclosing nonnumeric allocation.
            if (owner.kind == D::PipelineGraphOwner::Kind::NumericOutput)
                if (const auto* allocation = pipeline_trace::mark(owner.identity, membership);
                    allocation && allocation->address == owner.identity && allocation->bytes == owner.bytes)
                    result.scalar_numeric_allocations.emplace(allocation->serial, allocation->bytes);
            if (!result.identities.insert(owner.identity).second)
                return;
            ++result.owners;
            if (const auto* allocation = pipeline_trace::mark(owner.identity, membership))
            {
                ++result.traced_owners;
                result.allocations.emplace(allocation->serial, allocation->bytes);
            }
            else
                result.untraced_owner_bytes += owner.bytes;
        });
    std::set<std::pair<const void*, D::ContactPayloadRole>> payloads;
    std::map<std::uint64_t, std::pair<unsigned, std::size_t>> role_extents;
    D::contact_cache(*state).for_each_borrowed_contact_payload(
        [&](const D::ContactPayloadAllocation& value)
        {
            if (payloads.emplace(value.identity.address, value.identity.role).second)
            {
                const auto role = static_cast<unsigned>(value.identity.role);
                result.role_bytes[role] += value.payload_bytes;
                ++result.role_entries[role];
                if (const auto* allocation = pipeline_trace::mark(value.identity.address, membership))
                {
                    auto [entry, inserted] =
                        role_extents.emplace(allocation->serial, std::pair{role, allocation->bytes});
                    if (!inserted)
                        entry->second.first = std::min(entry->second.first, role);
                }
            }
        });
    // Each real allocation contributes to exactly one role, choosing the
    // lowest enum role when subobjects/aliases have several roles.
    for (const auto& [serial, value] : role_extents)
        result.role_extent_bytes[value.first] += value.second;
    return result;
}
std::vector<A::AdaptiveBytes> encoded(const State& source)
{
    std::vector<A::AdaptiveBytes> rows;
    for (const auto& leaf : source->owned_leaves())
        rows.push_back(A::encode_adaptive_leaf(leaf));
    for (const auto family : source->owned_families())
        rows.push_back(A::encode_adaptive_family(family.materialize()));
    return rows;
}
void warm(const State& source)
{
    std::vector<const A::AdaptiveLeaf*> rows;
    for (const auto& leaf : source->owned_leaves())
        rows.push_back(&leaf);
    std::size_t used = 0;
    (void) source->geometry_snapshot(
        rows, SimpleFluid::ArithmeticMode::Ordinary, A::GeometrySnapshotPurpose::State, used, ample);
}
struct Input
{
    std::vector<A::AdaptiveLeaf> leaves;
    std::vector<A::AdaptiveFamily> families;
    std::vector<A::AdaptiveMark> refine;
    std::shared_ptr<const A::AdaptiveProviderKernel> kernel;
    A::AdaptiveBytes changed_family;
};
Input refined(F::Provider provider, std::size_t components)
{
    auto fixture = F::make_disconnected(provider, components);
    if (provider == F::Provider::Static)
    {
        // Static participants remain static. Real Cartesian companions provide
        // saved-family edits while the static participants grow independently.
        auto dispatch = std::make_shared<F::details::DispatchKernel>();
        dispatch->routes = std::dynamic_pointer_cast<const F::details::DispatchKernel>(fixture.kernel)->routes;
        for (std::size_t i = 0; i != components; ++i)
        {
            auto companion = F::details::component(F::Provider::Cartesian, 512 + i);
            for (auto& row : companion.owned)
            {
                dispatch->routes.emplace(row.key.region, companion.kernel);
                fixture.owned.push_back(std::move(row));
            }
        }
        dispatch->seal_routes();
        fixture.kernel = dispatch;
    }
    Input result;
    result.kernel = fixture.kernel;
    const auto comm = Tpetra::getDefaultComm();
    const int active = std::min(comm->getSize(), 2);
    // Request one row per physical region through the actual transaction API.
    // Its provider refinement group supplies all block/factor companions needed
    // for a later valid inverse; a manual single-cell split cannot do that.
    std::set<std::uint64_t> split_regions;
    for (auto& row : fixture.owned)
    {
        const auto component = (row.key.region - 1000) / 4;
        const auto owner = int(component % active);
        if (owner != comm->getRank())
            continue;
        if (row.provider != A::AdaptiveProviderKind::Static && split_regions.insert(row.key.region).second)
        {
            const auto mode = row.provider == A::AdaptiveProviderKind::Product ? A::AdaptiveSplit::Axial
                                                                               : A::AdaptiveSplit::Isotropic;
            result.refine.push_back({row.key, mode});
            if (result.changed_family.empty())
                result.changed_family = A::adaptive_family_key(row.key, mode);
        }
        result.leaves.push_back(std::move(row));
    }
    std::sort(result.leaves.begin(), result.leaves.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    return result;
}
struct Measurement
{
    pipeline_trace::Summary adapt, migrate, native, initial_native, repeat_native;
    std::array<pipeline_trace::Summary, 3> warm_live_generations{};
    std::array<pipeline_trace::Summary, 3> warm_operations{};
    std::array<SimpleFluid::PLIC::GeometryStatistics, 3> warm_geometry{};
    std::array<D::GeometryCacheAdmissionStatistics, 3> warm_cache_before{}, warm_cache_after{};
    SimpleFluid::PLIC::GeometryStatistics adapt_geometry{}, migrate_geometry{}, native_geometry{}, initial_geometry{},
        repeat_geometry{};
    decltype(pipeline_work) adapt_work{}, migrate_work{}, native_work{}, initial_work{}, repeat_work{};
    std::array<std::size_t, 3> scalar_numeric_bytes{}, scalar_numeric_allocations{};
    std::array<decltype(pipeline_work), 3> warm_work{};
    D::ContactHaloStatistics contact{};
    D::ContactStorageBreakdown breakdown{};
    decltype(Graph::role_bytes) role_bytes{}, role_entries{}, role_extent_bytes{};
    std::array<std::size_t, 8> graph_attributed_live_extent_bytes{};
    std::size_t raw_unattributed_live_allocations = 0, raw_unattributed_live_bytes = 0;
    std::size_t live_unclassified_cpp_bytes = 0;
    A::AdaptiveBytes migrated_key;
    std::array<SimpleFluid::PLIC::details::GeometryAllocationWork, 8> owner_work_events{};
    std::size_t graph_owners = 0, shared_owners = 0, family_count = 0, history_count = 0, saved_history_count = 0;
    std::size_t owner_work = 0, cover_work = 0, cache_censuses = 0, untraced_owner_bytes = 0;
};

Measurement run(F::Provider provider, std::size_t components)
{
    const auto comm = Tpetra::getDefaultComm();
    const TraceLifetime trace;
    const ObserverScope observer;
    Measurement result;
    const auto measured_warm = [&](const State& state, std::size_t generation)
    {
        pipeline_trace::next_epoch();
        pipeline_work = {};
        Access::reset_pipeline_geometry_statistics();
        OwnerAccess::reset_work_statistics();
        result.warm_cache_before[generation] = Access::cache_admission_statistics(*state);
        warm(state);
        result.warm_operations[generation] = pipeline_trace::summary();
        result.warm_geometry[generation] = Access::pipeline_geometry_statistics();
        result.warm_work[generation] = pipeline_work;
        result.owner_work_events[5 + generation] = OwnerAccess::work_statistics();
        result.warm_cache_after[generation] = Access::cache_admission_statistics(*state);
    };
    {
        auto input = refined(provider, components);
        A::DistributedAdaptiveOptions options;
        options.byte_budget = ample;
        auto source = std::make_shared<const A::DistributedAdaptiveHierarchy>(
            std::move(input.leaves), std::move(input.families), std::move(input.kernel), comm, options);
        source = source->apply(source->plan_adaptation(input.refine));
        measured_warm(source, 0);
        // Native publication is first measured on the snapshot-warmed source,
        // then repeated before any edit. Returned physical outputs are released.
        pipeline_trace::next_epoch();
        pipeline_work = {};
        Access::reset_pipeline_geometry_statistics();
        OwnerAccess::reset_work_statistics();
        {
            const D::PipelinePhaseScope native_phase(D::PipelinePhase::NativePublication);
            auto publication = A::assemble_adaptive_native(*source, source->configuration_descriptor());
            auto bound = source->with_partition(publication.partition, publication.mesh->owned_cell_map());
            EXPECT_TRUE(bound->geometry_unchanged());
            result.initial_native = pipeline_trace::summary();
            result.initial_work = pipeline_work;
            result.initial_geometry = Access::pipeline_geometry_statistics();
            result.owner_work_events[0] = OwnerAccess::work_statistics();
            EXPECT_NE(publication.mesh, nullptr);
            source = std::move(bound);
        }
        pipeline_trace::next_epoch();
        pipeline_work = {};
        Access::reset_pipeline_geometry_statistics();
        OwnerAccess::reset_work_statistics();
        {
            const D::PipelinePhaseScope native_phase(D::PipelinePhase::NativePublication);
            auto publication = A::assemble_adaptive_native(*source, source->configuration_descriptor());
            auto bound = source->with_partition(publication.partition, publication.mesh->owned_cell_map());
            EXPECT_TRUE(bound->geometry_unchanged());
            result.repeat_native = pipeline_trace::summary();
            result.repeat_work = pipeline_work;
            result.repeat_geometry = Access::pipeline_geometry_statistics();
            result.owner_work_events[1] = OwnerAccess::work_statistics();
            EXPECT_NE(publication.mesh, nullptr);
            source = std::move(bound);
        }
        EXPECT_NE(source->native_partition(), nullptr);
        const auto source_graph = graph(source, 1);
        const auto numeric_output = [&](const Graph& value, std::size_t generation)
        {
            result.scalar_numeric_allocations[generation] = value.scalar_numeric_allocations.size();
            for (const auto& [serial, bytes] : value.scalar_numeric_allocations)
                result.scalar_numeric_bytes[generation] += bytes;
        };
        numeric_output(source_graph, 0);
        result.warm_live_generations[0] = pipeline_trace::summary();
        std::vector<A::AdaptiveBytes> before;
        {
            pipeline_trace::Pause pause;
            before = encoded(source);
            result.family_count = source->owned_families().size();
            // Generic providers retain ordinary Layer history. SavedLayer is
            // specific to the trusted Cartesian incidence-source protocol.
            result.history_count =
                source_graph.role_entries[static_cast<unsigned>(D::ContactPayloadRole::IncidenceLayerObject)];
            result.saved_history_count = D::contact_cache(*source).statistics().retained_saved_layers;
        }
        EXPECT_TRUE(comm->getRank() >= 2 || result.family_count > 0);
        // Binding can prune completed incidence histories. Report its actual
        // retained layers, including zero, alongside the real saved families.
        // The Mixed fixture's real product/polyhedron join requires retained
        // incidence work; disjoint aligned boxes can have no such layers.
        if (provider == F::Provider::Mixed && comm->getRank() < 2)
            EXPECT_GT(result.history_count, 0u);
        const auto cache_before = Access::cache_admission_statistics(*source);
        // One physical component has the same requested family at N and 2N.
        std::vector<A::AdaptiveBytes> coarsen;
        if (comm->getRank() == 0)
            coarsen.push_back(input.changed_family);
        pipeline_work = {};
        pipeline_trace::next_epoch();
        Access::reset_pipeline_geometry_statistics();
        OwnerAccess::reset_work_statistics();
        auto second = source->apply(source->plan_adaptation({}, coarsen));
        result.adapt = pipeline_trace::summary();
        result.adapt_work = pipeline_work;
        result.adapt_geometry = Access::pipeline_geometry_statistics();
        result.owner_work_events[2] = OwnerAccess::work_statistics();
        EXPECT_LT(second->global_cells(), source->global_cells());
        measured_warm(second, 1);
        const auto second_graph = graph(second, 2);
        numeric_output(second_graph, 1);
        result.warm_live_generations[1] = pipeline_trace::summary();
        result.role_bytes = second_graph.role_bytes;
        result.role_entries = second_graph.role_entries;
        result.role_extent_bytes = second_graph.role_extent_bytes;
        {
            pipeline_trace::Pause pause;
            result.breakdown = D::contact_cache(*second).contact_storage_breakdown();
        }
        std::size_t classified = 0;
        for (const auto& [serial, bytes] : second_graph.allocations)
            classified += bytes;
        result.live_unclassified_cpp_bytes = pipeline_trace::summary().all_live - classified;
        result.graph_owners = second_graph.owners;
        result.untraced_owner_bytes = second_graph.untraced_owner_bytes;
        for (const auto* identity : second_graph.identities)
            result.shared_owners += source_graph.identities.contains(identity);
        const auto& statistics = D::contact_cache(*second).statistics();
        result.contact = statistics;
        result.owner_work = statistics.contact_owner_admission_work;
        result.cover_work = statistics.rank_cover_owner_visits;
        result.cache_censuses =
            Access::cache_admission_statistics(*source).union_censuses - cache_before.union_censuses;
        {
            pipeline_trace::Pause pause;
            EXPECT_EQ(encoded(source), before);
            if (comm->getRank() < 2)
                EXPECT_GT(result.shared_owners, 0u);
        }

        // Real ownership change for one retained leaf, including Static rows.
        std::vector<int> destinations(second->owned_leaves().size(), comm->getRank());
        if (comm->getSize() > 1 && comm->getRank() == 0)
        {
            EXPECT_FALSE(destinations.empty());
            // Component zero is physically identical at both fixture sizes.
            auto ordinal = std::size_t(0);
            if (provider == F::Provider::Static)
                for (std::size_t i = 0; i != destinations.size(); ++i)
                    if (second->owned_leaves().at_ordinal(i).provider == A::AdaptiveProviderKind::Static)
                    {
                        ordinal = i;
                        break;
                    }
            {
                pipeline_trace::Pause pause;
                result.migrated_key = A::encode_adaptive_key(second->owned_leaves().at_ordinal(ordinal).key);
            }
            destinations[ordinal] = 1;
        }
        pipeline_work = {};
        pipeline_trace::next_epoch();
        Access::reset_pipeline_geometry_statistics();
        OwnerAccess::reset_work_statistics();
        auto third = [&]
        {
            auto migration = second->plan_migration(destinations);
            if (comm->getSize() > 1 && comm->getRank() == 0)
                EXPECT_GT(migration.native_migration().remote_sent_packets, 0u);
            return second->apply(migration);
        }();
        result.migrate = pipeline_trace::summary();
        result.migrate_work = pipeline_work;
        result.migrate_geometry = Access::pipeline_geometry_statistics();
        result.owner_work_events[3] = OwnerAccess::work_statistics();
        EXPECT_EQ(third->global_cells(), second->global_cells());
        if (comm->getSize() > 1)
        {
            const pipeline_trace::Pause pause;
            std::vector<A::AdaptiveCellKey> moved;
            if (comm->getRank() == 0)
                moved.push_back(A::decode_adaptive_key(result.migrated_key));
            const auto owners = third->lookup_cells(moved);
            if (comm->getRank() == 0)
            {
                EXPECT_EQ(owners.size(), 1u);
                if (owners.size() == 1)
                    EXPECT_EQ(owners.front().owner, 1);
            }
        }
        measured_warm(third, 2);
        const auto third_graph = graph(third, 4);
        numeric_output(third_graph, 2);
        result.warm_live_generations[2] = pipeline_trace::summary();
        const auto union_before_release = pipeline_trace::physical_unions();
        result.graph_attributed_live_extent_bytes = union_before_release;
        const auto unattributed = pipeline_trace::unattributed();
        result.raw_unattributed_live_allocations = unattributed.first;
        result.raw_unattributed_live_bytes = unattributed.second;
        // Graph attribution is a lower bound: generic provider internals,
        // separate control blocks and aggregate PLIC child buffers may remain
        // unclassified. All captured extents stay in the raw operation totals.

        EXPECT_GT(std::accumulate(union_before_release.begin(), union_before_release.end(), std::size_t(0)), 0u);
        const std::weak_ptr<const A::DistributedAdaptiveHierarchy> weak_source = source, weak_second = second;
        const std::weak_ptr<const A::AdaptiveLeafLayout> weak_source_layout = source->leaf_layout();
        source.reset();
        second.reset();
        EXPECT_TRUE(weak_source.expired());
        EXPECT_TRUE(weak_second.expired());
        EXPECT_TRUE(weak_source_layout.expired());
        // Third keeps physical pages it needs, without predecessor controllers.
        std::uint64_t local_cells = third->owned_leaves().size(), global_cells = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
        EXPECT_EQ(third->global_cells(), global_cells);
        if (comm->getSize() == 4 && comm->getRank() >= 2)
            EXPECT_TRUE(third->owned_leaves().empty());

        pipeline_work = {};
        pipeline_trace::next_epoch();
        {
            // Complete physical publication includes owned/ghost reconstruction,
            // scalar maps and binding. Every allocation remains in the trace.
            Access::reset_pipeline_geometry_statistics();
            OwnerAccess::reset_work_statistics();
            const D::PipelinePhaseScope native_phase(D::PipelinePhase::NativePublication);
            auto publication = A::assemble_adaptive_native(*third, third->configuration_descriptor());
            auto bound = third->with_partition(publication.partition, publication.mesh->owned_cell_map());
            EXPECT_TRUE(bound->geometry_unchanged());
            result.native = pipeline_trace::summary();
            result.native_work = pipeline_work;
            result.native_geometry = Access::pipeline_geometry_statistics();
            result.owner_work_events[4] = OwnerAccess::work_statistics();
            EXPECT_NE(publication.mesh, nullptr);
        }
        EXPECT_FALSE(pipeline_trace::overflow.load());
    }
    return result;
}
void report(std::string prefix, const pipeline_trace::Summary& value)
{
    const pipeline_trace::Pause pause;
    record_property(prefix + "_retained_requested_cpp_bytes", std::to_string(value.retained_bytes));
    record_property(prefix + "_retained_requested_cpp_allocations", std::to_string(value.retained_allocations));
    record_property(prefix + "_all_live_requested_cpp_bytes", std::to_string(value.all_live));
    record_property(prefix + "_all_live_requested_cpp_allocations", std::to_string(value.live_allocations));
    record_property(prefix + "_unattributed_live_requested_cpp_bytes", std::to_string(value.unattributed_live_bytes));
    record_property(
        prefix + "_unattributed_live_requested_cpp_allocations", std::to_string(value.unattributed_live_allocations));
    record_property(prefix + "_retained_stream_phase_bytes", std::to_string(value.retained_streaming_bytes));
    record_property(prefix + "_all_live_requested_cpp_peak", std::to_string(value.all_peak));
    record_property(prefix + "_incremental_live_requested_cpp_peak", std::to_string(value.incremental_peak));
    for (std::size_t i = 0; i != pipeline_trace::phases; ++i)
    {
        record_property(prefix + "_phase_" + phase_names[i] + "_requests", std::to_string(value.phase[i].allocations));
        record_property(prefix + "_phase_" + phase_names[i] + "_bytes", std::to_string(value.phase[i].bytes));
    }
}
} // namespace

TEST(AdaptivePipelineStorage, WholePipelineBaselineReportsExistingWork)
{
    record_property("plic_mesh_domain_matches_calling_image",
        Access::pipeline_geometry_statistics_domain() == SimpleFluid::PLIC::geometry_statistics_domain() ? "1" : "0");
    record_property("graph_attribution_scope", "partial_lower_bound_all_unmatched_extents_remain_in_raw_totals");
    record_property("scaling_gate_status", "baseline_observation_only_future_stage6_unqualified");
    for (const auto provider : F::providers)
    {
        SCOPED_TRACE(F::name(provider));
        // Cartesian blocks require even dimensions >=2 and refine all eight
        // root cells. Four/eight components still give 128/256 owned rows
        // per active rank before the fixed coarsening, spanning leaf pages.
        const auto components = provider == F::Provider::Cartesian || provider == F::Provider::Static ? 4u : 16u;
        record_property(std::string(F::name(provider)) + "_small_components", std::to_string(components));
        record_property(std::string(F::name(provider)) + "_large_components", std::to_string(2 * components));
        const auto small = run(provider, components);
        const auto large = run(provider, 2 * components);
        EXPECT_EQ(small.migrated_key, large.migrated_key);
        report(std::string(F::name(provider)) + "_small_adapt", small.adapt);
        report(std::string(F::name(provider)) + "_large_adapt", large.adapt);
        report(std::string(F::name(provider)) + "_small_migrate", small.migrate);
        report(std::string(F::name(provider)) + "_large_migrate", large.migrate);
        report(std::string(F::name(provider)) + "_small_native", small.native);
        report(std::string(F::name(provider)) + "_large_native", large.native);
        if (Tpetra::getDefaultComm()->getRank() < 2)
            EXPECT_GT(large.shared_owners, small.shared_owners);
        record_property(std::string(F::name(provider)) + "_large_untraced_reported_owner_payload",
            std::to_string(large.untraced_owner_bytes));
        const auto work_report = [&](const std::string& size, const Measurement& measured)
        {
            const auto prefix = std::string(F::name(provider)) + "_" + size;
            for (std::size_t generation = 0; generation != 3; ++generation)
            {
                report(prefix + "_warm_live_generation_" + std::to_string(generation + 1),
                    measured.warm_live_generations[generation]);
                record_property(prefix + "_warm_live_generation_" + std::to_string(generation + 1) +
                                    "_standalone_scalar_numeric_buffer_bytes",
                    std::to_string(measured.scalar_numeric_bytes[generation]));
                record_property(prefix + "_warm_live_generation_" + std::to_string(generation + 1) +
                                    "_standalone_scalar_numeric_buffer_allocations",
                    std::to_string(measured.scalar_numeric_allocations[generation]));
                const auto warm_prefix = prefix + "_warm_operation_" + std::to_string(generation + 1);
                report(warm_prefix, measured.warm_operations[generation]);
                const auto& before = measured.warm_cache_before[generation];
                const auto& after = measured.warm_cache_after[generation];
                const auto cache_property = [&](const char* name, std::size_t value)
                { record_property(warm_prefix + name, std::to_string(value)); };
                cache_property("_cache_admission_queries", after.admission_queries - before.admission_queries);
                cache_property("_cache_union_censuses", after.union_censuses - before.union_censuses);
                cache_property("_logical_manifest_owner_sum", after.census_root_visits - before.census_root_visits);
                cache_property("_catalog_copied_metadata_bytes",
                    after.catalog_copied_metadata_bytes - before.catalog_copied_metadata_bytes);
                cache_property("_catalog_copied_pages", after.catalog_copied_pages - before.catalog_copied_pages);
                cache_property("_catalog_visited_pages", after.catalog_visited_pages - before.catalog_visited_pages);
                cache_property("_cache_resident_bytes", after.resident_bytes);
                cache_property("_cache_lease_bytes", after.lease_bytes);
                for (std::size_t i = 0; i != pipeline_work.size(); ++i)
                    record_property(
                        warm_prefix + "_work_" + work_names[i], std::to_string(measured.warm_work[generation][i]));
            }
            for (std::size_t i = 0; i != pipeline_work.size(); ++i)
            {
                record_property(prefix + "_adapt_work_" + work_names[i], std::to_string(measured.adapt_work[i]));
                record_property(prefix + "_migration_work_" + work_names[i], std::to_string(measured.migrate_work[i]));
                record_property(prefix + "_publication_work_" + work_names[i], std::to_string(measured.native_work[i]));
                record_property(
                    prefix + "_first_publication_work_" + work_names[i], std::to_string(measured.initial_work[i]));
                record_property(
                    prefix + "_repeat_publication_work_" + work_names[i], std::to_string(measured.repeat_work[i]));
            }
            const auto geometry_report = [&](const char* phase, const auto& stats)
            {
                record_property(prefix + phase + "_plic_constructions", std::to_string(stats.constructions));
                record_property(prefix + phase + "_plic_source_imports", std::to_string(stats.source_imports));
                record_property(prefix + phase + "_plic_volume_evaluations", std::to_string(stats.volume_evaluations));
            };
            geometry_report("_adapt", measured.adapt_geometry);
            geometry_report("_migration", measured.migrate_geometry);
            geometry_report("_publication", measured.native_geometry);
            geometry_report("_first_publication", measured.initial_geometry);
            geometry_report("_repeat_publication", measured.repeat_geometry);
            for (std::size_t generation = 0; generation != 3; ++generation)
                geometry_report(
                    ("_warm_operation_" + std::to_string(generation + 1)).c_str(), measured.warm_geometry[generation]);
            report(prefix + "_first_publication", measured.initial_native);
            report(prefix + "_repeat_publication", measured.repeat_native);
            for (std::size_t phase = 0; phase != measured.owner_work_events.size(); ++phase)
            {
                const auto& events = measured.owner_work_events[phase];
                const auto named = [&](const char* name, std::size_t value)
                {
                    record_property(
                        prefix + "_geometry_owner_phase_" + owner_phase_names[phase] + name, std::to_string(value));
                };
                named("_push_attempts", events.pushes);
                named("_admission_attempts", events.admissions);
                named("_pending_owner_visits", events.pending_visits);
                named("_import_calls", events.import_calls);
                named("_actual_import_rows", events.import_rows);
                named("_export_calls", events.export_calls);
                named("_actual_export_rows", events.export_rows);
                named("_actual_manifest_visit_rows", events.manifest_visit_rows);
                named("_metadata_credit_queries", events.metadata_credit_queries);
                named("_metadata_credit_seed_rows", events.metadata_credit_seed_rows);
                named("_metadata_credit_created", events.metadata_credit_created);
                named("_paired_payload_credits", events.paired_payload_credits);
                named("_import_fallbacks", events.import_fallbacks);
                named("_import_delta_rows", events.import_delta_rows);
                named("_import_delta_pages", events.import_delta_pages);
                named("_certified_import_hits", events.certified_import_hits);
                named("_manifest_lookup_pages", events.manifest_lookup_pages);
                named("_index_control_lookup_pages", events.index_control_lookup_pages);
                named("_control_table_enumeration_rows", events.control_table_enumeration_rows);
                named("_reservation_path_pages", events.reservation_path_pages);
            }
            const auto& c = measured.contact;
            const auto property = [&](const char* label, std::size_t value)
            { record_property(prefix + label, std::to_string(value)); };
            property("_record_pages_copied", c.contact_record_pages_copied);
            property("_row_pages_copied", c.contact_row_pages_copied);
            property("_work_pages_copied", c.contact_work_pages_copied);
            property("_index_pages_copied", c.contact_index_pages_copied);
            property("_cover_pages_copied", c.contact_cover_pages_copied);
            property("_rank_cover_pages_copied", c.rank_cover_pages_copied);
            property("_catalog_pages_copied", c.contact_catalog_pages_copied);
            property("_trace_pages_copied", c.contact_trace_pages_copied);
            property("_query_pages_copied", c.contact_query_pages_copied);
            property("_rank_cover_required_wire_visits", c.rank_cover_wire_visits);
            property("_full_owned_scans", c.full_owned_scans_after_bootstrap);
            property("_source_cache_union_censuses_during_adaptation", measured.cache_censuses);
            property("_saved_family_count", measured.family_count);
            property("_retained_incidence_history_layers", measured.history_count);
            property("_retained_saved_incidence_layers", measured.saved_history_count);
            property("_raw_live_cpp_bytes_outside_current_graph", measured.live_unclassified_cpp_bytes);
            property("_raw_unattributed_live_allocations", measured.raw_unattributed_live_allocations);
            property("_raw_unattributed_live_bytes", measured.raw_unattributed_live_bytes);
            for (std::size_t mask = 1; mask != 8; ++mask)
                record_property(prefix + "_partial_graph_membership_" + std::to_string(mask) + "_cpp_bytes",
                    std::to_string(measured.graph_attributed_live_extent_bytes[mask]));
            const auto& b = measured.breakdown;
            property("_group_metadata", b.group_metadata);
            property("_owner_index_metadata", b.owner_index_metadata);
            property("_domain_metadata", b.domain_metadata);
            property("_domain_shared_body", b.domain_shared_body);
            property("_fact_metadata", b.fact_metadata);
            property("_fact_view_metadata", b.fact_view_metadata);
            property("_point_credit_metadata", b.point_credit_metadata);
            property("_point_shared_body", b.point_shared_body);
            property("_binding_metadata", b.binding_metadata);
            property("_fan_owner_metadata", b.fan_owner_metadata);
            property("_fan_shared_body", b.fan_shared_body);
            // Every enum value is emitted, including zero observed entries.
            // Role payloads are a diagnostic census, not the allocation union.
            for (std::size_t role = 0; role != measured.role_bytes.size(); ++role)
            {
                record_property(prefix + "_payload_role_" +
                                    D::contact_payload_role_name(static_cast<D::ContactPayloadRole>(role)) + "_bytes",
                    std::to_string(measured.role_bytes[role]));
                record_property(prefix + "_payload_role_" +
                                    D::contact_payload_role_name(static_cast<D::ContactPayloadRole>(role)) + "_entries",
                    std::to_string(measured.role_entries[role]));
                record_property(prefix + "_payload_role_" +
                                    D::contact_payload_role_name(static_cast<D::ContactPayloadRole>(role)) +
                                    "_unique_traced_extent_bytes",
                    std::to_string(measured.role_extent_bytes[role]));
            }
        };
        work_report("small", small);
        work_report("large", large);
        // Stage 1 reports current scaling. Future stages own bounded-work gates.
        record_property(std::string(F::name(provider)) + "_small_owner_visits", std::to_string(small.owner_work));
        record_property(std::string(F::name(provider)) + "_large_owner_visits", std::to_string(large.owner_work));
        record_property(std::string(F::name(provider)) + "_small_cover_visits", std::to_string(small.cover_work));
        record_property(std::string(F::name(provider)) + "_large_cover_visits", std::to_string(large.cover_work));
    }
}

TEST(AdaptivePipelineStorage, WarmAcceptedCacheAdmissionExactLimitAndOneByteShort)
{
    for (const auto provider : F::providers)
    {
        auto input = refined(provider, 4);
        auto source = std::make_shared<const A::DistributedAdaptiveHierarchy>(
            std::move(input.leaves), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
        source = source->apply(source->plan_adaptation(input.refine));
        warm(source);
        D::Budget observed{ample};
        Access::admit_cache_residency(*source, observed);
        ASSERT_GT(observed.used, 0u);
        D::Budget exact{observed.used};
        EXPECT_NO_THROW(Access::admit_cache_residency(*source, exact));
        EXPECT_EQ(exact.used, observed.used);
        D::Budget short_budget{observed.used - 1};
        EXPECT_THROW(Access::admit_cache_residency(*source, short_budget), std::invalid_argument);
        EXPECT_EQ(short_budget.used, Access::cache_admission_statistics(*source).cache_header_bytes);
    }
}

TEST(AdaptivePipelineStorage, RequestedCppTraceCountsAlignedBuffersAndExceptionPhaseRestore)
{
    const TraceLifetime trace;
    const ObserverScope observer;
    {
        const D::PipelinePhaseScope directory(D::PipelinePhase::Directory);
        auto* aligned = ::operator new(129, std::align_val_t(64));
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned) % 64, 0u);
        const auto allocated = pipeline_trace::summary();
        const auto channel = static_cast<unsigned>(pipeline_trace::Phase::Directory);
        EXPECT_EQ(allocated.phase[channel].allocations, 1u);
        EXPECT_EQ(allocated.phase[channel].bytes, 129u);
        EXPECT_EQ(allocated.phase[channel].live, 129u);
        ::operator delete(aligned, std::align_val_t(64));
        EXPECT_EQ(pipeline_trace::summary().phase[channel].live, 0u);
        try
        {
            const D::PipelinePhaseScope transport(D::PipelinePhase::Transport);
            throw 1;
        }
        catch (int)
        {
            EXPECT_EQ(pipeline_trace::phase, pipeline_trace::Phase::Directory);
        }
    }
    EXPECT_EQ(pipeline_trace::phase, pipeline_trace::Phase::Candidate);
    EXPECT_FALSE(pipeline_trace::overflow.load());
}

TEST(AdaptivePipelineStorage, TraceSerialsDeduplicateAliasesAndRejectTableExhaustion)
{
    const TraceLifetime trace;
    alignas(64) std::array<std::byte, 256> storage{};
    pipeline_trace::begin(8);
    pipeline_trace::allocated(storage.data(), 64);
    pipeline_trace::prepare_census();
    const auto* first = pipeline_trace::mark(storage.data(), 1);
    ASSERT_NE(first, nullptr);
    const auto serial = first->serial;
    EXPECT_EQ(pipeline_trace::mark(storage.data() + 16, 2)->serial, serial);
    EXPECT_EQ(pipeline_trace::physical_unions()[3], 64u);
    pipeline_trace::freed(storage.data());
    pipeline_trace::allocated(storage.data(), 64);
    pipeline_trace::prepare_census();
    const auto* reused = pipeline_trace::mark(storage.data(), 4);
    ASSERT_NE(reused, nullptr);
    EXPECT_NE(reused->serial, serial);
    EXPECT_EQ(reused->membership, 4u);
    for (unsigned i = 1; i != 9; ++i)
        pipeline_trace::allocated(storage.data() + 16 * i, 1);
    EXPECT_TRUE(pipeline_trace::overflow.load());
    // A fresh independent run must clear both the exhaustion flag and peak base.
    pipeline_trace::begin();
    const auto clean = pipeline_trace::summary();
    EXPECT_EQ(clean.all_peak, 0u);
    EXPECT_EQ(clean.incremental_peak, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
}

TEST(AdaptivePipelineStorage, EveryAllocationOverloadRecordsRequestedExtent)
{
    const TraceLifetime trace;
    void* scalar = ::operator new(17, std::nothrow);
    void* array = ::operator new[](19, std::nothrow);
    void* aligned = ::operator new(23, std::align_val_t(64), std::nothrow);
    void* aligned_array = ::operator new[](29, std::align_val_t(128), std::nothrow);
    const auto measured = pipeline_trace::summary();
    EXPECT_EQ(measured.phase[0].allocations, 4u);
    EXPECT_EQ(measured.phase[0].bytes, 88u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_array) % 128, 0u);
    ::operator delete(scalar, std::size_t(17));
    ::operator delete[](array, std::size_t(19));
    ::operator delete(aligned, std::size_t(23), std::align_val_t(64));
    ::operator delete[](aligned_array, std::size_t(29), std::align_val_t(128));
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}

TEST(AdaptivePipelineStorage, TrustedCartesianSourceWarmCacheSpansPreparedPages)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The prepared-page companion is serial; distributed provider transactions run separately.";
    for (const auto cells : {256u, 512u})
    {
        const TraceLifetime trace;
        const ObserverScope observer;
        A::AMRGrid grid;
        grid.root_blocks = {cells / 8, 1, 1};
        grid.block_cells = {2, 2, 2};
        const A::CartesianAMR mesh(grid);
        std::vector<std::uint64_t> indices(mesh.geometry()->num_cells());
        std::iota(indices.begin(), indices.end(), 0);
        auto input = mesh.detach_owned(indices, {'p', 'i', 'p', 'e'});
        auto source = std::make_shared<const A::DistributedAdaptiveHierarchy>(
            std::move(input.owned), std::move(input.families), std::move(input.kernel), Tpetra::getDefaultComm());
        std::vector<const A::AdaptiveLeaf*> selected;
        for (const auto& leaf : source->owned_leaves())
            selected.push_back(&leaf);
        // Prepared pages are optional look-ahead for streaming State requests.
        // A single all-cell demand intentionally constructs no prepared pages.
        for (std::size_t first = 0; first < selected.size(); first += 4)
        {
            std::size_t stream_used = 0;
            (void) source->geometry_snapshot(
                std::span(selected).subspan(first, std::min<std::size_t>(4, selected.size() - first)),
                SimpleFluid::ArithmeticMode::Ordinary, A::GeometrySnapshotPurpose::State, stream_used, ample);
        }
        std::size_t used = 0;
        auto snapshot = source->geometry_snapshot(
            selected, SimpleFluid::ArithmeticMode::Ordinary, A::GeometrySnapshotPurpose::State, used, ample);
        std::size_t prepared = 0;
        snapshot->visit_prepared_cartesian_sources([&](const auto& page) { prepared += page.size(); });
        EXPECT_EQ(prepared, cells);
        const auto owners = graph(source, 1);
        EXPECT_GT(owners.traced_owners, 0u);
        pipeline_trace::next_epoch();
        warm(source);
        report("trusted_cartesian_" + std::to_string(cells) + "_warm", pipeline_trace::summary());
        record_property("trusted_cartesian_" + std::to_string(cells) + "_prepared_cells", std::to_string(prepared));
        EXPECT_FALSE(pipeline_trace::overflow.load());
    }
}

TEST(AdaptivePipelineStorage, AllocationOwnerWorkCountsActualEnumerationAndPreAdmissionFailure)
{
    namespace P = SimpleFluid::PLIC;
    using Import = P::details::GeometryOwnerImportAccess;
    using Row = P::details::geometry_demand::OwnerRecord;
    const auto first = std::make_shared<const char>('a');
    const auto second = std::make_shared<const char>('b');
    const auto third = std::make_shared<const char>('c');
    const auto admission = [](D::Budget& budget)
    {
        return P::details::geometry_demand::Admission{
            &budget, [](void* context, std::size_t bytes) { static_cast<D::Budget*>(context)->add(bytes); }};
    };
    P::GeometryAllocationLedger source;
    std::size_t used = 0;
    source.retain_external_owner(first, 1, used, ample);
    source.retain_external_owner(second, 1, used, ample);
    ASSERT_EQ(source.owner_count(), 2u);
    D::Budget export_budget{ample};
    OwnerAccess::reset_work_statistics();
    const auto manifest = Import::export_manifest(source, admission(export_budget));
    EXPECT_EQ(OwnerAccess::work_statistics().export_rows, 2u);
    P::GeometryAllocationLedger fresh;
    std::size_t fresh_used = 0;
    OwnerAccess::reset_work_statistics();
    EXPECT_EQ(Import::import(fresh, manifest, fresh_used, ample), 0u);
    EXPECT_EQ(OwnerAccess::work_statistics().import_calls, 1u);
    EXPECT_EQ(OwnerAccess::work_statistics().import_rows, 0u);
    EXPECT_EQ(OwnerAccess::work_statistics().manifest_visit_rows, 0u);
    P::GeometryAllocationLedger populated;
    std::size_t populated_used = 0;
    populated.retain_external_owner(third, 1, populated_used, ample);
    OwnerAccess::reset_work_statistics();
    EXPECT_EQ(Import::import(populated, manifest, populated_used, ample), 2u);
    EXPECT_EQ(OwnerAccess::work_statistics().import_rows, 2u);
    EXPECT_EQ(OwnerAccess::work_statistics().manifest_visit_rows, 2u);
    EXPECT_EQ(populated.owner_count(), 3u);
    D::Budget merged_budget{ample};
    OwnerAccess::reset_work_statistics();
    const auto merged = Import::export_manifest(populated, admission(merged_budget));
    EXPECT_EQ(merged.owner_count(), 3u);
    EXPECT_EQ(OwnerAccess::work_statistics().export_rows, 3u);
    D::Budget short_budget{3 * sizeof(Row) - 1};
    OwnerAccess::reset_work_statistics();
    EXPECT_THROW((void) Import::export_manifest(populated, admission(short_budget)), std::invalid_argument);
    EXPECT_EQ(OwnerAccess::work_statistics().export_calls, 1u);
    EXPECT_EQ(OwnerAccess::work_statistics().export_rows, 0u);
}

TEST(AdaptivePipelineStorage, SharedAliasesMarkOneRealControlAllocation)
{
    struct Object
    {
        std::array<char, 64> payload{};
    };
    const TraceLifetime trace;
    auto owner = std::make_shared<Object>();
    std::shared_ptr<char> alias(owner, &owner->payload[17]);
    pipeline_trace::prepare_census();
    const auto* object = pipeline_trace::mark(owner.get(), 1);
    ASSERT_NE(object, nullptr);
    const auto serial = object->serial;
    const auto extent = object->bytes;
    const auto* member = pipeline_trace::mark(alias.get(), 2);
    ASSERT_NE(member, nullptr);
    EXPECT_EQ(member->serial, serial);
    EXPECT_EQ(pipeline_trace::physical_unions()[3], extent);
    owner.reset();
    EXPECT_EQ(pipeline_trace::summary().all_live, extent);
    alias.reset();
    EXPECT_EQ(pipeline_trace::summary().all_live, 0u);
}
