/** @file testAdaptivePipelineScaling.cc
 * @brief Allocation-domain reporting and independent factory-qualified AMR work gates.
 */
#include "adaptive_pipeline_scaling_fixtures.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/details/AllocationDomain.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationNumbering.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include "native_constructor_allocation_operators.inc"
#include "native_publication_test_helpers.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <vector>

namespace
{
using namespace SimpleFluid;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
namespace F = test::pipeline_scaling;
using State = std::shared_ptr<const A::DistributedAdaptiveHierarchy>;
using Access = D::GeometrySnapshotAccess;
using OwnerWork = PLIC::details::GeometryAllocationOwnerAccess;
namespace Receipts = PLIC::details::geometry_demand;
using Work = D::PipelineWork;
using Placement = std::map<A::AdaptiveCellKey, int>;
using WorkCounts = std::array<std::size_t, static_cast<unsigned>(Work::Count)>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t pipeline_budget = 512 * 1024 * 1024;
using AllocationDomain = geometry_detail::AllocationDomain;
constexpr std::array domain_names{"unattributed", "leaf_pages", "family_storage", "contact_halo", "geometry_snapshot",
    "owner_union", "native_publication", "external_third_party"};
static_assert(domain_names.size() == static_cast<std::size_t>(AllocationDomain::Count));

void report_domains(const pipeline_trace::Summary& memory, const Teuchos::Comm<int>& comm, std::string_view phase)
{
    const pipeline_trace::Pause pause;
    std::size_t live = 0;
    for (std::size_t i = 0; i < domain_names.size(); ++i)
    {
        const auto& domain = memory.domain[i];
        live += domain.live;
        std::fprintf(stdout,
            "AMR_PIPELINE_DOMAIN rank=%d ranks=%d phase=%.*s domain=%s live=%zu peak=%zu "
            "requested=%zu allocations=%zu deallocations=%zu\n",
            comm.getRank(), comm.getSize(), int(phase.size()), phase.data(), domain_names[i], domain.live, domain.peak,
            domain.bytes, domain.allocations, domain.deallocations);
    }
    EXPECT_EQ(live, memory.all_live);
    EXPECT_EQ(
        memory.domain[static_cast<std::size_t>(AllocationDomain::Unattributed)].live, memory.unattributed_live_bytes);
    EXPECT_EQ(memory.retained_allocations + memory.shared_allocations, memory.live_allocations);
    EXPECT_EQ(memory.retained_bytes + memory.shared_bytes, memory.all_live);
}

/** Factory products and expectation tables are fixture allocations. Their
 * bootstrap and all subsequent pipeline operations are measured separately. */
template<class Make> auto make_fixture(Make&& make)
{
    const pipeline_trace::Pause pause;
    return make();
}

// Exclude the fixture's shared envelope while leaving hierarchy construction traced.
template<class T> struct FixtureHierarchyAllocator
{
    using value_type = T;

    FixtureHierarchyAllocator() = default;
    template<class U> FixtureHierarchyAllocator(const FixtureHierarchyAllocator<U>&) noexcept {}

    T* allocate(std::size_t count)
    {
        const pipeline_trace::Pause pause;
        return std::allocator<T>{}.allocate(count);
    }
    void deallocate(T* address, std::size_t count) noexcept { std::allocator<T>{}.deallocate(address, count); }
    template<class U> bool operator==(const FixtureHierarchyAllocator<U>&) const noexcept { return true; }
};

State make_fixture_hierarchy(A::AdaptiveBootstrap&& input, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    return std::allocate_shared<const A::DistributedAdaptiveHierarchy>(
        FixtureHierarchyAllocator<A::DistributedAdaptiveHierarchy>{}, std::move(input.owned), std::move(input.families),
        std::move(input.kernel), comm, A::DistributedAdaptiveOptions{.byte_budget = pipeline_budget});
}

struct Operation
{
    WorkCounts work{};
    PLIC::GeometryStatistics geometry{};
    PLIC::details::GeometryAllocationWork owners{};
    pipeline_trace::Summary memory{};

    std::size_t count(Work event) const { return work[static_cast<unsigned>(event)]; }
};

class Trace
{
public:
    explicit Trace(const Teuchos::Comm<int>& comm) : comm_(comm)
    {
        pipeline_trace::begin();
        previous_ = D::set_pipeline_observer({this, [](void*, D::PipelinePhase phase) noexcept
            { pipeline_trace::phase = static_cast<pipeline_trace::Phase>(phase); },
            [](void* context, Work event, std::size_t size) noexcept
            { static_cast<Trace*>(context)->counts_[static_cast<unsigned>(event)] += size; },
            [](void* context, D::PipelineBudgetRejection rejection) noexcept
            { static_cast<Trace*>(context)->budget_rejection_ = rejection; },
            [](void* context, D::PipelineNativeOwnerStage stage, bool entering,
                D::PipelineNativeOwnerAdmission sample) noexcept
            {
                auto& boundary = static_cast<Trace*>(context)->native_owners_[static_cast<unsigned>(stage)];
                (entering ? boundary.before : boundary.after) = OwnerSample{sample, Receipts::receipt_work()};
            }});
    }

    ~Trace()
    {
        (void) D::set_pipeline_observer(previous_);
        pipeline_trace::recording.store(false);
    }

    template<class Run> auto measure(std::string_view phase, Run&& run, Operation* saved = nullptr)
    {
        pipeline_trace::next_epoch();
        counts_ = {};
        budget_rejection_.reset();
        native_owners_ = {};
        Access::reset_pipeline_geometry_statistics();
        OwnerWork::reset_work_statistics();
        Receipts::reset_receipt_work();
        const auto report = [&]
        {
            const Operation result{counts_, Access::pipeline_geometry_statistics(), OwnerWork::work_statistics(),
                pipeline_trace::summary()};
            const auto& memory = result.memory;
            const auto requested = std::accumulate(memory.phase.begin(), memory.phase.end(), std::size_t{},
                [](std::size_t sum, const auto& phase) { return sum + phase.bytes; });
            const pipeline_trace::Pause pause;
            std::fprintf(stdout,
                "AMR_PIPELINE_SCALING rank=%d ranks=%d phase=%.*s retained=%zu live=%zu peak=%zu cumulative=%zu "
                "leaf_pages=%zu owner_import_rows=%zu owner_export_rows=%zu plic_constructions=%llu "
                "cell_route_bytes=%zu source_proofs_issued=%zu incidence_proofs_issued=%zu "
                "leaf_visits=%zu ordered_pages=%zu wire_sent=%zu wire_received=%zu\n",
                comm_.getRank(), comm_.getSize(), int(phase.size()), phase.data(), memory.retained_bytes,
                memory.all_live, memory.all_peak, requested, result.count(Work::LeafPageCreated),
                result.owners.import_rows, result.owners.export_rows,
                static_cast<unsigned long long>(result.geometry.constructions),
                memory.phase[static_cast<unsigned>(pipeline_trace::Phase::NativeCellRoutes)].bytes,
                result.count(Work::NativeSourceProofIssue), result.count(Work::NativeIncidenceProofIssue),
                result.count(Work::LeafRowDereference), result.count(Work::OrderedPageCreated),
                result.count(Work::WireSentBytes), result.count(Work::WireReceivedBytes));
            report_domains(memory, comm_, phase);
            constexpr std::array names{"output", "source", "union"};
            for (std::size_t slot = 0; slot < names.size(); ++slot)
            {
                const auto& owner = native_owners_[slot];
                if (!owner.before || !owner.after)
                    continue;
                const auto& a = owner.before->admission;
                const auto& b = owner.after->admission;
                const auto& first = owner.before->receipts;
                const auto& last = owner.after->receipts;
                std::fprintf(stdout,
                    "AMR_NATIVE_OWNERS rank=%d ranks=%d phase=%.*s stage=%s native_live=%zu/%zu "
                    "native_peak=%zu/%zu native_cumulative=%zu/%zu native_limit=%zu "
                    "index_spent=%zu/%zu index_limit=%zu\n",
                    comm_.getRank(), comm_.getSize(), int(phase.size()), phase.data(), names[slot], a.live, b.live,
                    a.peak, b.peak, a.cumulative, b.cumulative, b.limit, a.index_spent, b.index_spent, b.index_limit);
                std::fprintf(stdout,
                    "AMR_RECEIPT_STORAGE rank=%d ranks=%d phase=%.*s stage=%s fact_blocks=%zu proof_blocks=%zu "
                    "leaf_rows=%zu leaf_proofs=%zu administration_index=%zu\n",
                    comm_.getRank(), comm_.getSize(), int(phase.size()), phase.data(), names[slot],
                    last.fact_blocks - first.fact_blocks, last.proof_blocks - first.proof_blocks,
                    last.leaf_rows - first.leaf_rows, last.leaf_proofs - first.leaf_proofs,
                    last.administration_index - first.administration_index);
                EXPECT_LE(b.live, b.limit);
                EXPECT_LE(b.index_spent, b.index_limit);
                EXPECT_GE(b.cumulative, a.cumulative);
                EXPECT_GE(b.index_spent, a.index_spent);
            }
            if (budget_rejection_)
            {
                const auto& rejected = *budget_rejection_;
                std::fprintf(stdout,
                    "AMR_PIPELINE_BUDGET rank=%d phase=%.*s used=%zu request=%zu limit=%zu cumulative_spent=%zu\n",
                    comm_.getRank(), int(phase.size()), phase.data(), rejected.used, rejected.request, rejected.limit,
                    rejected.cumulative_spent);
            }
            std::fflush(stdout);
            EXPECT_FALSE(pipeline_trace::overflow.load());
            if (saved)
                *saved = result;
        };
        auto result = [&]
        {
            try
            {
                return run();
            }
            catch (...)
            {
                report();
                throw;
            }
        }();
        report();
        return result;
    }

    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;

private:
    struct OwnerSample
    {
        D::PipelineNativeOwnerAdmission admission;
        Receipts::ReceiptWork receipts;
    };
    struct OwnerBoundary
    {
        std::optional<OwnerSample> before, after;
    };
    const Teuchos::Comm<int>& comm_;
    WorkCounts counts_{};
    std::array<OwnerBoundary, 3> native_owners_{};
    std::optional<D::PipelineBudgetRejection> budget_rejection_;
    D::PipelineObserver previous_{};
};

struct BoundPublication
{
    State hierarchy;
    A::AdaptiveNativePublicationView view;
};

BoundPublication publish(Trace& trace, const State& state, bool warm_repeat = false)
{
    const auto before = Access::cache_admission_statistics(*state);
    Operation publication, binding;
    auto view = trace.measure(
        "publication",
        [&]
        {
            const D::PipelinePhaseScope phase(D::PipelinePhase::NativePublication);
            return A::publish_adaptive_native(*state, A::native_publication_schema(*state));
        },
        &publication);
    const auto after = Access::cache_admission_statistics(*state);
    auto bound = trace.measure(
        "binding", [&] { return state->with_partition(view.partition(), view.mesh()->owned_cell_map()); }, &binding);
    const pipeline_trace::Pause pause;
    EXPECT_EQ(publication.count(Work::NativeCompatibilityPublication), 0u);
    EXPECT_EQ(binding.count(Work::NativeExactPartitionBinding), 1u);
    EXPECT_EQ(binding.count(Work::NativeCompatibilityPartitionBinding), 0u);
    EXPECT_EQ(binding.geometry.constructions, 0u);
    EXPECT_EQ(publication.owners.import_fallbacks, 0u);
    EXPECT_EQ(binding.owners.import_fallbacks, 0u);
    EXPECT_EQ(after.union_updates, before.union_updates + 1);
    EXPECT_EQ(after.union_censuses, before.union_censuses);
    std::fprintf(stdout,
        "AMR_PIPELINE_OWNER_UNION rank=%d edges=%zu first=%zu last=%zu copied_pages=%zu "
        "copied_bytes=%zu lookup_pages=%zu\n",
        state->communicator()->getRank(), after.union_edge_visits - before.union_edge_visits,
        after.union_first_transitions - before.union_first_transitions,
        after.union_last_transitions - before.union_last_transitions,
        after.union_index_copied_pages - before.union_index_copied_pages,
        after.union_index_copied_bytes - before.union_index_copied_bytes,
        after.union_lookup_pages - before.union_lookup_pages);
    EXPECT_EQ(bound->native_partition(), view.partition());
    EXPECT_FALSE(view.key_registry_storage().owns_key_maps);
    const auto cells = view.local_keys().size();
    const auto full = view.partition()->plan().cell_ids().size();
    const auto routes = publication.memory.phase[static_cast<unsigned>(pipeline_trace::Phase::NativeCellRoutes)];
    EXPECT_EQ(routes.bytes, cells * sizeof(Meshes::SparseNativeCell) + full * sizeof(Meshes::NativeSparseAliasRoute));
    EXPECT_EQ(routes.allocations, std::size_t(cells != 0) + std::size_t(full != 0));
    if (warm_repeat)
    {
        EXPECT_EQ(publication.geometry.constructions, 0u);
        EXPECT_EQ(publication.geometry.source_imports, 0u);
        EXPECT_EQ(publication.count(Work::NativeSourceProofIssue), 0u);
        EXPECT_EQ(publication.count(Work::NativeRemoteSourceDecode), 0u);
        EXPECT_EQ(publication.count(Work::OrderedPageCreated), 0u);
    }
    return {std::move(bound), std::move(view)};
}

void warm_geometry(Trace& trace, const State& state)
{
    std::vector<const A::AdaptiveLeaf*> leaves;
    {
        const pipeline_trace::Pause pause;
        leaves.reserve(state->owned_leaves().size());
        for (const auto& leaf : state->owned_leaves())
            leaves.push_back(&leaf);
    }
    const auto snapshot = trace.measure("state_geometry",
        [&]
        {
            std::size_t used = 0;
            return state->geometry_snapshot(
                leaves, ArithmeticMode::Ordinary, A::GeometrySnapshotPurpose::State, used, pipeline_budget);
        });
    const pipeline_trace::Pause pause;
    EXPECT_TRUE(snapshot);
}

A::AdaptiveCellKey cartesian_key(unsigned level, std::uint64_t x, std::uint64_t y = 0, std::uint64_t z = 0)
{
    return {0, A::CartesianCellAncestry{level, {x, y, z}}};
}

void cartesian_cells(Placement& cells, unsigned level, std::size_t block, int owner, bool insert)
{
    const std::size_t width = 2u << level;
    for (std::size_t z = 0; z < width; ++z)
        for (std::size_t y = 0; y < width; ++y)
            for (std::size_t x = 0; x < width; ++x)
            {
                const auto key = cartesian_key(level, block * width + x, y, z);
                if (insert)
                    cells.emplace(key, owner);
                else
                    cells.erase(key);
            }
}

State cartesian_source(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, const A::AMRGrid& grid)
{
    auto input = make_fixture(
        [&]
        {
            const A::CartesianAMR grid_source(grid);
            const int active = std::min(comm->getSize(), 2);
            std::vector<std::uint64_t> ids;
            for (std::size_t id = 0; id < grid_source.geometry()->num_cells(); ++id)
                if (int((id / 8) % active) == comm->getRank())
                    ids.push_back(id);
            return grid_source.detach_owned(ids, {'s', 'c', 'a', 'l', 'e'});
        });
    return make_fixture_hierarchy(std::move(input), comm);
}

struct LeafTree
{
    struct LeafPage
    {
        std::weak_ptr<const void> control;
        std::span<const A::AdaptiveLeaf> rows;
    };
    std::vector<LeafPage> leaves;
    std::size_t depth = 0;
};

LeafTree inspect_leaf_tree(const A::DistributedAdaptiveHierarchy& state)
{
    const pipeline_trace::Pause pause;
    LeafTree result;
    std::map<const void*, std::size_t> depth;
    Access::visit_pipeline_leaf_pages(state,
        [&](const auto& page)
        {
            const auto page_depth = page.parent ? depth.at(page.parent) + 1 : 1;
            depth.emplace(page.address, page_depth);
            result.depth = std::max(result.depth, page_depth);
            if (!page.child_count)
            {
                ASSERT_FALSE(page.rows.empty());
                EXPECT_EQ(page.maximum, page.rows.back().key);
                EXPECT_LE(page.capacity, D::AdaptiveLeafPages::leaf_capacity);
                result.leaves.push_back({page.control, page.rows});
            }
        });
    return result;
}

/** These seeds have one branch over well-filled leaves. Derive the
 * edited leaves and split count independently from old key intervals. */
class LeafPageOracle
{
public:
    LeafPageOracle(
        const A::DistributedAdaptiveHierarchy& source, const Placement& old_cells, const Placement& new_cells)
    {
        const pipeline_trace::Pause pause;
        const auto before = inspect_leaf_tree(source);
        if (source.owned_leaves().empty())
            return;
        EXPECT_EQ(before.depth, 2u);
        EXPECT_GE(before.leaves.size(), 2u);
        const int rank = source.communicator()->getRank();
        std::vector<int> changes(before.leaves.size());
        std::vector<bool> dirty(before.leaves.size());
        const auto assign = [&](const A::AdaptiveCellKey& key, int delta)
        {
            auto page = std::lower_bound(before.leaves.begin(), before.leaves.end(), key,
                [](const auto& leaf, const auto& bound) { return leaf.rows.back().key < bound; });
            if (page == before.leaves.end())
                --page;
            const auto index = page - before.leaves.begin();
            changes[index] += delta;
            dirty[index] = true;
        };
        for (const auto& [key, owner] : old_cells)
            if (owner == rank && (!new_cells.contains(key) || new_cells.at(key) != rank))
                assign(key, -1);
        for (const auto& [key, owner] : new_cells)
            if (owner == rank && (!old_cells.contains(key) || old_cells.at(key) != rank))
                assign(key, 1);
        constexpr auto capacity = D::AdaptiveLeafPages::leaf_capacity;
        for (std::size_t i = 0; i < before.leaves.size(); ++i)
        {
            if (!dirty[i])
            {
                unchanged_.push_back(before.leaves[i]);
                continue;
            }
            const auto rows = int(before.leaves[i].rows.size()) + changes[i];
            EXPECT_GE(rows, int(capacity / 2)) << "The fixed fixture must avoid rebalancing adjacent leaves.";
            if (rows > 0)
                new_leaves_ += (rows + capacity - 1) / capacity;
        }
        EXPECT_FALSE(unchanged_.empty()) << "No unaffected leaf page was exercised.";
        EXPECT_LE(new_leaves_ + unchanged_.size(), D::AdaptiveLeafPages::branch_capacity);
    }

    void check(const A::DistributedAdaptiveHierarchy& target, const Operation& operation) const
    {
        const pipeline_trace::Pause pause;
        const auto after = inspect_leaf_tree(target);
        for (const auto& page : unchanged_)
        {
            const auto same = std::find_if(after.leaves.begin(), after.leaves.end(),
                [&](const auto& current)
                {
                    return current.rows.data() == page.rows.data() && current.rows.size() == page.rows.size() &&
                           !current.control.owner_before(page.control) && !page.control.owner_before(current.control);
                });
            EXPECT_NE(same, after.leaves.end());
        }
        EXPECT_EQ(after.leaves.size(), new_leaves_ + unchanged_.size());
        EXPECT_LE(operation.count(Work::LeafPageCreated), new_leaves_ + (new_leaves_ != 0));
    }

private:
    std::vector<LeafTree::LeafPage> unchanged_;
    std::size_t new_leaves_ = 0;
};

void check_cartesian(const BoundPublication& result, const Placement& expected, std::size_t volume)
{
    const pipeline_trace::Pause pause;
    const auto& state = *result.hierarchy;
    const auto& comm = *state.communicator();
    const auto& mesh = *result.view.partition()->geometry();
    const auto owned = result.view.owned_gids();
    std::size_t local_count = 0;
    double local_volume = 0;
    for (const auto& [key, owner] : expected)
    {
        EXPECT_EQ(state.owned_leaves().contains(key), owner == comm.getRank());
        if (owner != comm.getRank())
            continue;
        ++local_count;
        const auto at = owned.find(key);
        if (at == owned.end())
        {
            ADD_FAILURE() << "Independent Cartesian key is absent from the native result.";
            continue;
        }
        const auto gid = at->second;
        const auto& cartesian = std::get<A::CartesianCellAncestry>(key.ancestry);
        const auto dx = std::ldexp(1.0, -int(cartesian.level));
        EXPECT_EQ(mesh.cell_volume(gid), dx * dx * dx);
        const auto center = mesh.cell_centroid(gid);
        EXPECT_EQ(center.x, dx * (cartesian.index[0] + .5));
        EXPECT_EQ(center.y, dx * (cartesian.index[1] + .5));
        EXPECT_EQ(center.z, dx * (cartesian.index[2] + .5));
        local_volume += mesh.cell_volume(gid);
    }
    EXPECT_EQ(owned.size(), local_count);
    EXPECT_EQ(state.global_cells(), expected.size());
    double global_volume = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    EXPECT_EQ(global_volume, volume);
}

std::vector<A::AdaptiveMark> local_marks(const std::vector<A::AdaptiveMark>& marks, const F::Cells& cells, int rank)
{
    std::vector<A::AdaptiveMark> result;
    for (const auto& mark : marks)
        if (cells.at(mark.key).owner == rank)
            result.push_back(mark);
    return result;
}

Placement placement(const F::Cells& cells)
{
    const pipeline_trace::Pause pause;
    Placement result;
    for (const auto& [key, cell] : cells)
        result.emplace(key, cell.owner);
    return result;
}

std::vector<A::AdaptiveCellKey> unaffected_sources(const F::Cells& before, const F::Cells& after, int rank)
{
    std::vector<const F::Box*> changed;
    for (const auto& [key, cell] : before)
        if (!after.contains(key))
            changed.push_back(&cell.box);
    const auto contacts = [](const F::Box& a, const F::Box& b)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (a.lower[axis] + a.width[axis] < b.lower[axis] || b.lower[axis] + b.width[axis] < a.lower[axis])
                return false;
        return true;
    };
    std::vector<A::AdaptiveCellKey> unchanged;
    for (const auto& [key, cell] : before)
        if (cell.owner == rank && after.contains(key) &&
            std::none_of(
                changed.begin(), changed.end(), [&](const auto* region) { return contacts(cell.box, *region); }))
            unchanged.push_back(key);
    return unchanged;
}

void check_source_reuse(const State& first, const State& second, std::span<const A::AdaptiveCellKey> unchanged)
{
    const pipeline_trace::Pause pause;
    const auto a = D::NativePublicationSession::capture(*first);
    const auto b = D::NativePublicationSession::capture(*second);
    ASSERT_TRUE(a.cache && b.cache && a.cache->indexes && b.cache->indexes);
    if (!first->owned_leaves().empty())
        EXPECT_FALSE(unchanged.empty());
    for (const auto& key : unchanged)
    {
        const auto old = a.cache->indexes->find(key), current = b.cache->indexes->find(key);
        ASSERT_TRUE(old && current);
        EXPECT_EQ(current->source, old->source);
        EXPECT_FALSE(current->source.owner_before(old->source));
        EXPECT_FALSE(old->source.owner_before(current->source));
    }
}

State setup_fixture(Trace& trace, F::Fixture& fixture, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    return trace.measure("bootstrap", [&] { return make_fixture_hierarchy(std::move(fixture.source), comm); });
}

void check_boxes(const BoundPublication& result, const F::Cells& expected)
{
    const pipeline_trace::Pause pause;
    const auto& state = *result.hierarchy;
    const auto& comm = *state.communicator();
    const auto& plan = result.view.partition()->plan();
    const auto& mesh = *result.view.partition()->geometry();
    const auto owned = result.view.owned_gids();
    const auto keys = result.view.local_keys();
    std::size_t local_count = 0;
    double local_volume = 0, expected_volume = 0;
    for (const auto& [key, cell] : expected)
    {
        expected_volume += cell.box.volume();
        const bool local = cell.owner == comm.getRank();
        EXPECT_EQ(state.owned_leaves().contains(key), local);
        if (!local)
            continue;
        ++local_count;
        const auto location = owned.find(key);
        if (location == owned.end())
        {
            ADD_FAILURE() << "Missing independently enumerated owner-local cell key.";
            continue;
        }
        const auto gid = location->second;
        EXPECT_EQ(mesh.cell_volume(gid), cell.box.volume());
        EXPECT_EQ(
            mesh.cell_centroid(gid), (MeshUtils::Vec3{cell.box.center(0), cell.box.center(1), cell.box.center(2)}));
        local_volume += mesh.cell_volume(gid);
    }
    EXPECT_EQ(owned.size(), local_count);
    EXPECT_EQ(state.global_cells(), expected.size());
    double global_volume = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    EXPECT_EQ(global_volume, expected_volume);

    const int remote = !plan.ghost_cells().empty();
    int global_remote = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &remote, &global_remote);
    EXPECT_EQ(global_remote, std::min(2, comm.getSize()) == 2 ? 2 : 0);
    if (comm.getRank() >= 2)
        EXPECT_TRUE(plan.cell_ids().empty());

    D::collective(comm,
        [&]
        {
            for (const auto gid : plan.owned_faces())
            {
                const auto first_id = mesh.owner_cell(gid);
                const auto second_id = mesh.neighbor_cell(gid);
                const auto& first = expected.at(keys.at(first_id)).box;
                const F::Box* second = second_id == Meshes::MultiRegionMesh::invalid_cell_id()
                                           ? nullptr
                                           : &expected.at(keys.at(second_id)).box;
                const auto area = mesh.face_area_vector(gid);
                const std::array vector{area.x, area.y, area.z};
                const auto axis = std::find_if(vector.begin(), vector.end(), [](double entry) { return entry != 0; });
                if (axis == vector.end())
                {
                    ADD_FAILURE() << "Canonical box face has zero area vector.";
                    continue;
                }
                EXPECT_EQ(std::count_if(vector.begin(), vector.end(), [](double entry) { return entry != 0; }), 1);
                const auto normal = std::size_t(axis - vector.begin());
                const auto plane = first.lower[normal] + (*axis > 0 ? first.width[normal] : 0);
                const auto centroid = mesh.face_centroid(gid);
                const std::array actual_center{centroid.x, centroid.y, centroid.z};
                EXPECT_EQ(actual_center[normal], plane);
                if (second)
                    EXPECT_EQ(plane, second->lower[normal] + (*axis < 0 ? second->width[normal] : 0));
                double expected_area = 1;
                for (std::size_t tangential = 0; tangential < 3; ++tangential)
                    if (tangential != normal)
                    {
                        const auto low = second ? std::max(first.lower[tangential], second->lower[tangential])
                                                : first.lower[tangential];
                        const auto high = second ? std::min(first.lower[tangential] + first.width[tangential],
                                                       second->lower[tangential] + second->width[tangential])
                                                 : first.lower[tangential] + first.width[tangential];
                        EXPECT_GT(high, low);
                        expected_area *= high - low;
                        EXPECT_EQ(actual_center[tangential], .5 * (low + high));
                    }
                EXPECT_EQ(mesh.face_area(gid), expected_area);
            }
        });
}

State move_leaf(Trace& trace, const State& source, const A::AdaptiveCellKey& key, int destination)
{
    const auto& comm = *source->communicator();
    std::vector<int> destinations;
    {
        const pipeline_trace::Pause pause;
        for (const auto& leaf : source->owned_leaves())
            destinations.push_back(leaf.key == key ? destination : comm.getRank());
    }
    const auto plan = trace.measure("migration_plan", [&] { return source->plan_migration(destinations); });
    {
        const pipeline_trace::Pause pause;
        const auto& wire = plan.native_migration();
        EXPECT_EQ(wire.remote_sent_packets > 0, destination != 0 && comm.getRank() == 0);
        EXPECT_EQ(wire.remote_received_payload_bytes > 0, destination != 0 && comm.getRank() == destination);
        if (wire.remote_sent_packets)
            EXPECT_GT(plan.actual_migration_application_bytes(), wire.remote_sent_payload_bytes);
    }
    return trace.measure("migration_apply", [&] { return source->apply(plan); });
}

/** Generation lifetimes and shared pages are independent of allocation domains. */
void release_generations(BoundPublication& first, BoundPublication& second, const BoundPublication& third,
    std::span<const AllocationDomain> required_domains = {})
{
    const pipeline_trace::Pause pause;
    ASSERT_TRUE(first.hierarchy && second.hierarchy && third.hierarchy);
    EXPECT_NE(first.hierarchy, second.hierarchy);
    EXPECT_NE(second.hierarchy, third.hierarchy);
    const std::array states{first.hierarchy.get(), second.hierarchy.get(), third.hierarchy.get()};
    for (const auto* state : states)
    {
        EXPECT_NE(state->native_partition(), nullptr);
        for (const auto& page : inspect_leaf_tree(*state).leaves)
            EXPECT_FALSE(page.control.expired());
    }
    const auto before = pipeline_trace::summary();
    const auto& comm = *third.hierarchy->communicator();
    report_domains(before, comm, "retained_ABC");
    std::fprintf(stdout,
        "AMR_PIPELINE_GENERATIONS rank=%d ranks=%d A_cells=%llu B_cells=%llu C_cells=%llu "
        "live_allocations=%zu live_bytes=%zu unattributed_allocations=%zu unattributed_bytes=%zu "
        "current_map_profile=%zu\n",
        comm.getRank(), comm.getSize(), static_cast<unsigned long long>(first.hierarchy->global_cells()),
        static_cast<unsigned long long>(second.hierarchy->global_cells()),
        static_cast<unsigned long long>(third.hierarchy->global_cells()), before.live_allocations, before.all_live,
        before.unattributed_live_allocations, before.unattributed_live_bytes,
        third.view.storage().required_maps_estimate);
    std::weak_ptr<const A::DistributedAdaptiveHierarchy> old = first.hierarchy, middle = second.hierarchy;
    const std::weak_ptr<const A::AdaptiveLeafLayout> old_layout = first.hierarchy->leaf_layout();
    const auto current_pages = inspect_leaf_tree(*third.hierarchy);
    first = {};
    second = {};
    EXPECT_TRUE(old.expired());
    EXPECT_TRUE(middle.expired());
    EXPECT_TRUE(old_layout.expired());
    // Weak controls can remain until the probes retire; they must not pin bodies.
    old.reset();
    middle.reset();
    for (const auto& page : current_pages.leaves)
        EXPECT_FALSE(page.control.expired());
    const auto after = pipeline_trace::summary();
    EXPECT_LE(after.all_live, before.all_live);
    report_domains(after, comm, "retained_C");
    for (const auto domain : required_domains)
    {
        const auto index = static_cast<std::size_t>(domain);
        EXPECT_GT(before.domain[index].live, 0u) << domain_names[index] << " with A/B/C live";
        EXPECT_GT(after.domain[index].live, 0u) << domain_names[index] << " with only C live";
    }
}

std::uint64_t box_trial(std::size_t size, F::Fixture (*factory)(std::size_t, int, int))
{
    SCOPED_TRACE(size);
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), active = std::min(comm->getSize(), 2);
    Trace trace(*comm);
    auto fixture = make_fixture([&] { return factory(size, rank, active); });
    auto source = setup_fixture(trace, fixture, comm);
    F::Cells expected, edited;
    std::vector<A::AdaptiveCellKey> unchanged;
    std::vector<A::AdaptiveMark> setup, marks;
    std::vector<A::AdaptiveBytes> inverse;
    {
        const pipeline_trace::Pause pause;
        setup = local_marks(fixture.setup, fixture.roots, rank);
        // Expected tables and edit requests stay outside allocation capture.
        expected = fixture.roots;
        for (const auto& mark : fixture.setup)
            F::split(expected, mark);
        marks = local_marks(fixture.edit, expected, rank);
        edited = expected;
        for (const auto& mark : fixture.edit)
        {
            F::split(edited, mark);
            if (rank == 0)
                inverse.push_back(A::adaptive_family_key(mark.key, mark.split));
        }
        unchanged = unaffected_sources(expected, edited, rank);
        fixture.roots.clear();
        std::vector<A::AdaptiveMark>().swap(fixture.setup);
        std::vector<A::AdaptiveMark>().swap(fixture.edit);
    }
    source = trace.measure("setup_families", [&] { return source->apply(source->plan_adaptation(setup)); });
    warm_geometry(trace, source);
    auto first = publish(trace, source);
    auto a = publish(trace, first.hierarchy, true);
    first = {};
    source.reset();
    check_boxes(a, expected);
    Operation edited_work;
    {
        const LeafPageOracle pages(*a.hierarchy, placement(expected), placement(edited));
        source = trace.measure(
            "refinement", [&] { return a.hierarchy->apply(a.hierarchy->plan_adaptation(marks)); }, &edited_work);
        pages.check(*source, edited_work);
    }
    warm_geometry(trace, source);
    auto b = publish(trace, source);
    source.reset();
    check_boxes(b, edited);
    check_source_reuse(a.hierarchy, b.hierarchy, unchanged);
    source = move_leaf(trace, b.hierarchy, fixture.move, active - 1);
    edited.at(fixture.move).owner = active - 1;
    warm_geometry(trace, source);
    auto c = publish(trace, source);
    source.reset();
    c = publish(trace, c.hierarchy, true);
    check_boxes(c, edited);
    {
        const pipeline_trace::Pause pause;
        const auto current = c.hierarchy->lookup_cells(std::span(&fixture.move, 1));
        EXPECT_EQ(current.at(0).owner, active - 1);
        if (const auto it = expected.find(fixture.move); it != expected.end())
            it->second.owner = active - 1;
    }
    source = trace.measure("coarsening", [&] { return c.hierarchy->apply(c.hierarchy->plan_adaptation({}, inverse)); });
    check_boxes(publish(trace, source), expected);
    source.reset();
    release_generations(a, b, c);
    std::uint64_t constructions = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &edited_work.geometry.constructions, &constructions);
    return constructions;
}

void cartesian_trial(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, std::size_t blocks)
{
    SCOPED_TRACE(blocks);
    const auto rank = comm->getRank();
    const auto active = std::min(comm->getSize(), 2);
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.root_blocks = {blocks, 1, 1};
    grid.extent = {double(2 * blocks), 2.0, 2.0};
    Trace trace(*comm);
    auto root = trace.measure("bootstrap", [&] { return cartesian_source(comm, grid); });
    std::vector<A::AdaptiveMark> setup;
    Placement expected;
    {
        const pipeline_trace::Pause pause;
        for (std::size_t block = 0; block < blocks; ++block)
            cartesian_cells(expected, (block == 8 || block == 9) ? 1 : 0, block, int(block % active), true);
        for (std::uint64_t block : {8, 9})
            if (int(block % active) == rank)
                setup.push_back({cartesian_key(0, 2 * block), A::AdaptiveSplit::Isotropic});
    }
    auto source = trace.measure("setup_families", [&] { return root->apply(root->plan_adaptation(setup)); });
    root.reset();
    warm_geometry(trace, source);
    auto first = publish(trace, source);
    auto a = publish(trace, first.hierarchy, true);
    first = {};
    source.reset();
    const auto volume = 8 * blocks;
    check_cartesian(a, expected, volume);

    Placement edited;
    std::vector<A::AdaptiveCellKey> unchanged;
    {
        const pipeline_trace::Pause pause;
        edited = expected;
        cartesian_cells(edited, 0, 2, 0, false);
        cartesian_cells(edited, 1, 2, 0, true);
        for (const auto& [key, owner] : expected)
        {
            const auto& cartesian = std::get<A::CartesianCellAncestry>(key.ancestry);
            const auto x = std::ldexp(double(cartesian.index[0]), -int(cartesian.level));
            const auto width = std::ldexp(1.0, -int(cartesian.level));
            if (owner == rank && (x + width < 4.0 || x > 6.0))
                unchanged.push_back(key);
        }
    }
    const A::AdaptiveMark mark{cartesian_key(0, 4), A::AdaptiveSplit::Isotropic};
    const auto marks = rank == 0 ? std::span(&mark, 1) : std::span<const A::AdaptiveMark>{};
    Operation refinement;
    const LeafPageOracle pages(*a.hierarchy, expected, edited);
    auto refined = trace.measure(
        "refinement", [&] { return a.hierarchy->apply(a.hierarchy->plan_adaptation(marks)); }, &refinement);
    pages.check(*refined, refinement);
    warm_geometry(trace, refined);
    auto b = publish(trace, refined);
    refined.reset();
    check_source_reuse(a.hierarchy, b.hierarchy, unchanged);
    const auto moving = cartesian_key(1, 9, 1, 1);
    check_cartesian(b, edited, volume);
    auto migrated = move_leaf(trace, b.hierarchy, moving, active - 1);
    warm_geometry(trace, migrated);
    auto c = publish(trace, migrated);
    migrated.reset();
    {
        const pipeline_trace::Pause pause;
        edited.at(moving) = active - 1;
        const auto location = c.hierarchy->lookup_cells(std::span(&moving, 1));
        EXPECT_EQ(location.at(0).owner, active - 1);
    }
    c = publish(trace, c.hierarchy, true);
    check_cartesian(c, edited, volume);

    std::vector<A::AdaptiveBytes> inverse;
    {
        const pipeline_trace::Pause pause;
        if (rank == 0)
            inverse.push_back(A::adaptive_family_key(mark.key, A::AdaptiveSplit::Isotropic));
    }
    {
        const auto restored =
            trace.measure("coarsening", [&] { return c.hierarchy->apply(c.hierarchy->plan_adaptation({}, inverse)); });
        check_cartesian(publish(trace, restored), expected, volume);
    }
    release_generations(a, b, c);
}

TEST(AdaptivePipelineScaling, CartesianFixedEdit)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    ASSERT_TRUE(comm->getSize() == 1 || comm->getSize() == 2 || comm->getSize() == 4);
    for (std::size_t blocks : {32, 64})
        cartesian_trial(comm, blocks);
}

TEST(AdaptivePipelineScaling, ProductFixedEdit)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto small = box_trial(16, F::product);
    const auto large = box_trial(32, F::product);
    // The added patches have no closed contact with the edited Product
    // column. Their presence cannot require more physical constructions.
    EXPECT_LE(large, small);
}

TEST(AdaptivePipelineScaling, PolyhedralFixedEdit)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    for (std::size_t pairs : {16, 32})
        box_trial(pairs, F::polyhedral);
}

TEST(AdaptivePipelineScaling, StaticCompanionFixedEdit)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    for (std::size_t groups : {4, 8})
        box_trial(groups, F::static_companion);
}

TEST(AdaptivePipelineScaling, MixedFixedEdit)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    for (std::size_t distant : {16, 32})
        box_trial(distant, F::mixed);
}

TEST(AdaptivePipelineScaling, RetainedAllocationDomainsAcrossThreePublications)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    A::AMRGrid grid;
    // One global block, owned entirely by rank zero. Other ranks deliberately
    // stay empty in MPI2/MPI4; the grid is never replicated per rank.
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {2, 2, 2};
    Trace trace(*comm);
    auto source = trace.measure("bootstrap", [&] { return cartesian_source(comm, grid); });
    warm_geometry(trace, source);
    auto first = publish(trace, source);
    source.reset();
    const A::AdaptiveMark mark{cartesian_key(0, 0), A::AdaptiveSplit::Isotropic};
    const auto marks = comm->getRank() == 0 ? std::span(&mark, 1) : std::span<const A::AdaptiveMark>{};
    auto refined = trace.measure("refinement",
        [&] { return first.hierarchy->apply(first.hierarchy->plan_adaptation(marks)); });
    warm_geometry(trace, refined);
    auto second = publish(trace, refined);
    refined.reset();
    const auto third = publish(trace, second.hierarchy, true);
    const pipeline_trace::Pause pause;
    constexpr std::size_t cells_per_block = 2 * 2 * 2;
    const bool owns_block = comm->getRank() == 0;
    EXPECT_EQ(first.hierarchy->owned_leaves().size(), owns_block ? cells_per_block : 0u);
    EXPECT_EQ(first.hierarchy->global_cells(), cells_per_block);
    // The Cartesian mark refines its block into eight child blocks.
    EXPECT_EQ(third.hierarchy->global_cells(), 8 * cells_per_block);
    EXPECT_EQ(third.hierarchy->owned_leaves().size(), owns_block ? 8 * cells_per_block : 0u);
    // The refined owner keeps leaf pages and saved families, a contact snapshot,
    // a geometry cache with its owner index, and the native publication payload.
    constexpr std::array required_domains{AllocationDomain::LeafPages, AllocationDomain::FamilyStorage,
        AllocationDomain::ContactHalo, AllocationDomain::GeometrySnapshot, AllocationDomain::OwnerUnion,
        AllocationDomain::NativePublication};
    release_generations(first, second, third,
        owns_block ? std::span<const AllocationDomain>{required_domains} : std::span<const AllocationDomain>{});
}

struct IncidenceHistory
{
    std::set<const void*> layers, histories;
    std::size_t line_rows = 0, longest_replay = 0;
};

IncidenceHistory incidence_history(const A::DistributedAdaptiveHierarchy& state)
{
    const pipeline_trace::Pause pause;
    using Role = D::ContactPayloadRole;
    IncidenceHistory history;
    Access::contact_cache(state).for_each_borrowed_contact_payload(
        [&](const auto& owner)
        {
            switch (owner.identity.role)
            {
                case Role::IncidenceLayerObject:
                    history.layers.insert(owner.identity.address);
                    break;
                case Role::IncidenceLayerHistoryObject:
                    history.histories.insert(owner.identity.address);
                    break;
                case Role::IncidenceLayerHistoryEntries:
                    history.longest_replay = std::max(history.longest_replay, owner.live_count);
                    break;
                case Role::LineHistoryRows:
                    history.line_rows += owner.live_count;
                    break;
                default:
                    break;
            }
        });
    return history;
}

std::vector<PLIC::GeometryPoint> contact_points(
    const A::DistributedAdaptiveHierarchy& state, const A::AdaptiveCellKey& key)
{
    const pipeline_trace::Pause pause;
    std::set<PLIC::GeometryPoint> points;
    D::subface_snapshot(state)->visit_pieces(
        [&](const D::CanonicalSubfaceView& piece)
        {
            const bool first = piece.first().key() == key;
            const auto second = piece.second();
            if (!first && (!second || second->key() != key))
                return;
            const auto image = piece.second_image();
            const auto& shift = PLIC::details::GeometryIdentityAccess::displacement(image);
            for (std::size_t i = 0; i < piece.point_count(); ++i)
            {
                auto point = PLIC::point_display<ArithmeticMode::Ordinary>(piece.point(i));
                if (!first)
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        point[axis] -= shift[axis];
                points.insert(point);
            }
        });
    return {points.begin(), points.end()};
}

TEST(AdaptivePipelineScaling, JoinedHistoryPreservesDistantWitnesses)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), active = std::min(comm->getSize(), 2);
    Trace trace(*comm);
    auto fixture = make_fixture([&] { return F::mixed(4, rank, active); });
    auto state = setup_fixture(trace, fixture, comm);
    F::Cells cells;
    {
        const pipeline_trace::Pause pause;
        cells = fixture.roots;
        fixture.roots.clear();
    }
    auto product = F::product_key(7, 0, 0);
    auto convex = F::convex_key(9, 0);
    std::array<State, 3> retained;
    std::vector<A::AdaptiveMark> deepest_split;
    for (unsigned depth = 1; depth <= 6; ++depth)
    {
        const auto context = make_fixture([&] { return testing::ScopedTrace(__FILE__, __LINE__, depth); });
        std::vector<A::AdaptiveMark> local;
        {
            const pipeline_trace::Pause pause;
            deepest_split = {{product, A::AdaptiveSplit::Isotropic}, {convex, A::AdaptiveSplit::Isotropic}};
            local = local_marks(deepest_split, cells, rank);
            for (const auto& mark : deepest_split)
                F::split(cells, mark);
            // After the first split vertex zero of every selected quad stays
            // on the same physical corner of the east interface.
            product = F::product_child(product, depth == 1 ? 1 : 0, 0, A::AdaptiveSplit::Isotropic);
            convex = F::convex_child(convex, 0);
        }
        state = trace.measure("deepen_join", [&] { return state->apply(state->plan_adaptation(local)); });
        EXPECT_EQ(state->global_cells(), cells.size());
        if (depth >= 4)
            retained[depth - 4] = state;
    }
    check_boxes(publish(trace, state), cells);
    const auto original = incidence_history(*state);
    const auto original_points = contact_points(*state, rank == 0 ? product : convex);
    const int has_history = !original.layers.empty() && !original.histories.empty() && original.longest_replay > 0;
    int any_history = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &has_history, &any_history);
    EXPECT_NE(any_history, 0);
    if (rank < active)
    {
        EXPECT_FALSE(original_points.empty());
        EXPECT_NE(std::find(original_points.begin(), original_points.end(), PLIC::GeometryPoint{1, 0, 0}),
            original_points.end());
    }

    const auto distant = F::convex_key(9, 5);
    std::vector<A::AdaptiveMark> refine;
    std::vector<A::AdaptiveBytes> inverse;
    {
        const pipeline_trace::Pause pause;
        if (rank == 0)
        {
            refine.push_back({distant, A::AdaptiveSplit::Isotropic});
            inverse.push_back(A::adaptive_family_key(distant, A::AdaptiveSplit::Isotropic));
        }
    }
    for (unsigned step = 0; step < 16; ++step)
    {
        const auto context = make_fixture([&] { return testing::ScopedTrace(__FILE__, __LINE__, step); });
        state = trace.measure("distant_edit", [&]
            { return state->apply(step % 2 ? state->plan_adaptation({}, inverse) : state->plan_adaptation(refine)); });
        const pipeline_trace::Pause pause;
        const auto current = incidence_history(*state);
        EXPECT_TRUE(std::includes(current.layers.begin(), current.layers.end(), original.layers.begin(),
            original.layers.end(), std::less<const void*>{}));
        EXPECT_TRUE(std::includes(current.histories.begin(), current.histories.end(), original.histories.begin(),
            original.histories.end(), std::less<const void*>{}));
        EXPECT_GE(current.line_rows, original.line_rows);
        EXPECT_EQ(contact_points(*state, rank == 0 ? product : convex), original_points);
    }
    const auto moving = F::product_child(deepest_split.front().key, 2, 1, A::AdaptiveSplit::Isotropic);
    state = move_leaf(trace, state, moving, active - 1);
    const auto owner = state->lookup_cells(std::span(&moving, 1));
    EXPECT_EQ(owner.at(0).owner, active - 1);
    {
        const pipeline_trace::Pause pause;
        inverse.clear();
        if (rank == 0)
            for (const auto& mark : deepest_split)
                inverse.push_back(A::adaptive_family_key(mark.key, mark.split));
    }
    state = trace.measure("inverse_join", [&] { return state->apply(state->plan_adaptation({}, inverse)); });
    EXPECT_EQ(state->global_cells(), retained[1]->global_cells());
    // The active inverse restores the same owners that marked the parents at
    // depth six. Determine them from the actual local keys and replay the marks.
    std::vector<A::AdaptiveMark> recreate;
    {
        const pipeline_trace::Pause pause;
        for (const auto& mark : deepest_split)
            if (state->owned_leaves().contains(mark.key))
                recreate.push_back(mark);
    }
    state = trace.measure("recreate_join", [&] { return state->apply(state->plan_adaptation(recreate)); });
    check_boxes(publish(trace, state), cells);
}

TEST(AdaptivePipelineScaling, PeriodicCartesianSavedLayers)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    Trace trace(*comm);
    A::AMRGrid grid;
    grid.root_blocks = {2, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 4, 2};
    grid.periodic[0] = true;
    const auto source = trace.measure("bootstrap", [&] { return cartesian_source(comm, grid); });
    std::vector<A::AdaptiveMark> marks;
    {
        const pipeline_trace::Pause pause;
        for (const auto& key : {cartesian_key(0, 0), cartesian_key(0, 2)})
            if (source->owned_leaves().contains(key))
                marks.push_back({key, A::AdaptiveSplit::Isotropic});
    }
    const auto refined =
        trace.measure("refine_periodic_pair", [&] { return source->apply(source->plan_adaptation(marks)); });
    EXPECT_EQ(refined->global_cells(), 144u);
    const auto& work = Access::contact_cache(*refined).statistics();
    const std::array<std::uint64_t, 3> local{
        work.retained_saved_layers, work.source_result_selectors, work.retained_certified_layer_output_slots};
    std::array<std::uint64_t, 3> global{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    EXPECT_GT(global[0], 0u) << "Retained certified layer owners";
    EXPECT_GT(global[2], 0u) << "Retained certified layer output";
    const auto history = incidence_history(*refined);
    const auto history_labels = std::array<std::uint64_t, 2>{history.layers.size(), history.longest_replay};
    std::array<std::uint64_t, 2> replay{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(history_labels.size()), history_labels.data(), replay.data());
    EXPECT_GT(replay[0], 0u);
    EXPECT_GT(replay[1], 0u);
    {
        const pipeline_trace::Pause pause;
        ::testing::Test::RecordProperty("source_result_selectors", std::to_string(global[1]));
        ::testing::Test::RecordProperty("maximum_retained_replay_entries", std::to_string(replay[1]));
    }
    const auto bound = publish(trace, refined);
    const auto repeat = publish(trace, bound.hierarchy, true);
    int has_enriched_seam = 0;
    Access::contact_cache(*refined).for_each_borrowed_contact_binding(
        [&](const D::ContactHaloBorrowedBinding& binding)
        {
            if (!binding.row || binding.retired)
                return;
            const auto& row = *binding.row;
            for (const double seam : {0., 4.})
                if (std::count_if(row.points.begin(), row.points.end(), [seam](const auto& point)
                        { return PLIC::point_display<ArithmeticMode::Ordinary>(point.physical)[0] == seam; }) > 4)
                {
                    has_enriched_seam = 1;
                    EXPECT_EQ(contact_points(*refined, row.key), contact_points(*repeat.hierarchy, row.key));
                }
        });
    int any_enriched_seam = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &has_enriched_seam, &any_enriched_seam);
    EXPECT_NE(any_enriched_seam, 0);
}

TEST(AdaptivePipelineScaling, SmallFullPublicationOracle)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    for (const bool periodic : {false, true})
    {
        Trace trace(*comm);
        A::AMRGrid grid;
        grid.root_blocks = {2, 1, 1};
        grid.block_cells = {2, 2, 2};
        grid.extent = {4.0, 2.0, 2.0};
        grid.periodic[0] = periodic;
        const auto source = trace.measure("bootstrap", [&] { return cartesian_source(comm, grid); });
        const auto first = publish(trace, source);
        const auto repeat = publish(trace, first.hierarchy, true);
        const pipeline_trace::Pause pause;
        const auto oracle = A::assemble_adaptive_native(*source, source->configuration_descriptor());
        test::native_publication::complete_parity(first.view, oracle);
        test::native_publication::complete_parity(repeat.view, oracle);
    }
    const int active = std::min(comm->getSize(), 2);
    using Factory = F::Fixture (*)(std::size_t, int, int);
    for (const auto& [factory, size] : std::array<std::pair<Factory, std::size_t>, 4>{
             {{F::product, 1}, {F::polyhedral, 1}, {F::static_companion, 2}, {F::mixed, 0}}})
    {
        Trace trace(*comm);
        auto fixture = make_fixture([&] { return factory(size, comm->getRank(), active); });
        const auto source = setup_fixture(trace, fixture, comm);
        const auto qualified = publish(trace, source);
        const pipeline_trace::Pause pause;
        const auto oracle = A::assemble_adaptive_native(*source, source->configuration_descriptor());
        test::native_publication::complete_parity(
            qualified.view, oracle, test::native_publication::ExplicitOrdinals::Repacked);
    }
}

TEST(AdaptivePipelineScaling, FundedAllocatorRejectsExhaustedAccountAndUpstreamBadAlloc)
{
    using Allocator = geometry_detail::DomainAllocator<std::uint64_t, AllocationDomain::NativePublication>;
    constexpr std::size_t old_bytes = 8 * sizeof(std::uint64_t), new_bytes = 16 * sizeof(std::uint64_t);
    for (const bool inject_upstream_failure : {false, true})
    {
        // The first account is one byte short of the real old/new overlap.
        // The second funds it exactly and fails in the upstream allocator.
        const auto limit = old_bytes + new_bytes - std::size_t(!inject_upstream_failure);
        geometry_detail::AllocationAccount account{limit};
        Trace trace(*Tpetra::getDefaultComm());
        const auto baseline = pipeline_trace::summary();
        {
            std::vector<std::uint64_t, Allocator> values{Allocator(account)};
            values.reserve(8);
            for (std::uint64_t i = 0; i < 8; ++i)
                values.push_back(3 * i + 1);
            const auto before = pipeline_trace::summary();
            const auto funded = account.statistics();
            const auto* original_data = values.data();
            bool rejected = false;
            {
                std::optional<native_constructor_test::Failure> failure;
                if (inject_upstream_failure)
                    failure.emplace(0);
                try
                {
                    values.reserve(16);
                }
                catch (const std::bad_alloc&)
                {
                    rejected = true;
                }
            }
            const auto failed = pipeline_trace::summary();
            const auto after_failure = account.statistics();
            const auto domain = static_cast<std::size_t>(AllocationDomain::NativePublication);
            {
                const pipeline_trace::Pause pause;
                EXPECT_TRUE(rejected);
                EXPECT_EQ(values.size(), 8u);
                EXPECT_EQ(values.capacity(), 8u);
                EXPECT_EQ(values.data(), original_data);
                for (std::size_t i = 0; i < values.size(); ++i)
                    EXPECT_EQ(values[i], 3 * i + 1);
                EXPECT_EQ(after_failure.live, old_bytes);
                EXPECT_EQ(after_failure.live, funded.live);
                EXPECT_EQ(after_failure.peak, funded.peak);
                EXPECT_EQ(after_failure.allocations, funded.allocations);
                EXPECT_EQ(after_failure.deallocations, funded.deallocations);
                EXPECT_EQ(failed.domain[domain].allocations, before.domain[domain].allocations);
                EXPECT_EQ(failed.domain[domain].bytes, before.domain[domain].bytes);
                EXPECT_EQ(failed.domain[domain].live, before.domain[domain].live);
                EXPECT_EQ(failed.domain[domain].peak, before.domain[domain].peak);
                EXPECT_EQ(failed.domain[domain].deallocations, before.domain[domain].deallocations);
                EXPECT_EQ(failed.all_live, before.all_live);
                EXPECT_EQ(failed.all_peak, before.all_peak);
                EXPECT_EQ(failed.live_allocations, before.live_allocations);
                EXPECT_EQ(failed.retained_bytes, before.retained_bytes);
                EXPECT_EQ(failed.unattributed_live_bytes, before.unattributed_live_bytes);
            }
            // Exception unwinding must restore the unscoped allocation domain.
            auto* plain = ::operator new(31);
            const auto escaped = pipeline_trace::summary();
            ::operator delete(plain);
            {
                const pipeline_trace::Pause pause;
                EXPECT_EQ(escaped.unattributed_live_allocations, 1u);
                EXPECT_EQ(escaped.unattributed_live_bytes, 31u);
                EXPECT_EQ(escaped.domain[domain].live, old_bytes);
            }
            // A smaller growth fits the exhausted account; the funded case retries
            // the original request. Both must have released their pending charge.
            const std::size_t recovery_capacity = inject_upstream_failure ? 16 : 15;
            const auto recovery_bytes = recovery_capacity * sizeof(std::uint64_t);
            values.reserve(recovery_capacity);
            const auto recovered = account.statistics();
            const auto captured = pipeline_trace::summary();
            const pipeline_trace::Pause pause;
            EXPECT_EQ(recovered.live, recovery_bytes);
            EXPECT_EQ(recovered.peak, old_bytes + recovery_bytes);
            EXPECT_EQ(recovered.allocations, 2u);
            EXPECT_EQ(recovered.deallocations, 1u);
            EXPECT_EQ(captured.domain[domain].live, recovery_bytes);
            EXPECT_EQ(captured.domain[domain].bytes, old_bytes + recovery_bytes);
            EXPECT_EQ(captured.domain[domain].allocations, 2u);
            EXPECT_EQ(captured.domain[domain].deallocations, 1u);
            EXPECT_EQ(captured.unattributed_live_bytes, 0u);
            for (std::size_t i = 0; i < values.size(); ++i)
                EXPECT_EQ(values[i], 3 * i + 1);
        }
        const auto released = pipeline_trace::summary();
        const auto retired = account.statistics();
        const pipeline_trace::Pause pause;
        EXPECT_EQ(retired.live, 0u);
        EXPECT_EQ(retired.allocations, 2u);
        EXPECT_EQ(retired.deallocations, 2u);
        EXPECT_EQ(released.all_live, baseline.all_live);
        EXPECT_EQ(released.live_allocations, baseline.live_allocations);
        EXPECT_EQ(released.unattributed_live_bytes, baseline.unattributed_live_bytes);
        EXPECT_FALSE(pipeline_trace::overflow.load());
    }
}

TEST(AdaptivePipelineScaling, FundedSharedControlRetiresWhenConstructionThrowsBadAlloc)
{
    struct Rejected
    {
        Rejected() { throw std::bad_alloc(); }
    };
    using Allocator = geometry_detail::DomainAllocator<Rejected, AllocationDomain::OwnerUnion>;
    geometry_detail::AllocationAccount account{pipeline_budget};
    Trace trace(*Tpetra::getDefaultComm());
    bool rejected = false;
    try
    {
        (void) std::allocate_shared<Rejected>(Allocator(account));
    }
    catch (const std::bad_alloc&)
    {
        rejected = true;
    }
    const auto captured = pipeline_trace::summary();
    const auto retired = account.statistics();
    const pipeline_trace::Pause pause;
    const auto& domain = captured.domain[static_cast<std::size_t>(AllocationDomain::OwnerUnion)];
    EXPECT_TRUE(rejected);
    EXPECT_GT(retired.peak, sizeof(Rejected));
    EXPECT_EQ(retired.live, 0u);
    EXPECT_EQ(retired.allocations, 1u);
    EXPECT_EQ(retired.deallocations, 1u);
    EXPECT_EQ(domain.bytes, retired.peak);
    EXPECT_EQ(domain.live, 0u);
    EXPECT_EQ(domain.allocations, 1u);
    EXPECT_EQ(domain.deallocations, 1u);
    EXPECT_EQ(captured.all_live, 0u);
    EXPECT_EQ(captured.unattributed_live_bytes, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
}

TEST(AdaptivePipelineScaling, CollectiveRejectionPreservesAcceptedGraphs)
{
    ASSERT_NO_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile());
    const auto comm = Tpetra::getDefaultComm();
    Trace trace(*comm);
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    const auto seed = trace.measure("bootstrap", [&] { return cartesian_source(comm, grid); });
    warm_geometry(trace, seed);
    const auto initial = publish(trace, seed);
    const auto accepted = publish(trace, initial.hierarchy);
    const auto& state = *initial.hierarchy;
    const auto schema = A::native_publication_schema(state);
    const auto& arena = *Access::contact_cache(state).allocation_arena();
    auto& limit = Access::contact_cache(state).allocation_arena()->limit;
    const auto original_limit = limit;
    const auto used = arena.used;
    const auto resident = Access::cache_admission_statistics(state);
    const auto input = D::NativePublicationSession::capture(state);
    const auto receipt = input.cache->receipt;
    const auto indexes = input.cache->indexes;
    const auto revision = input.cache->revision;
    using Root = PLIC::details::GeometryOwnerRootBinding;
    const auto roots = [&]
    {
        std::array<Root, 2> result;
        std::size_t next = 0;
        Access::visit_native_cache_bindings(
            state, [&](const auto& root) { result.at(next++) = root; }, [](const auto&) {});
        return result;
    };
    const auto before = roots();
    const auto snapshots = [&]
    {
        const pipeline_trace::Pause pause;
        std::vector<const A::AdaptiveGeometrySnapshot*> result;
        state.visit_geometry_snapshots([&](const auto& snapshot) { result.push_back(&snapshot); });
        return result;
    };
    const auto cached = snapshots();
    const auto check_accepted = [&]
    {
        const pipeline_trace::Pause pause;
        const auto after = D::NativePublicationSession::capture(state);
        EXPECT_EQ(after.observation, input.observation);
        EXPECT_EQ(after.layout, input.layout);
        EXPECT_EQ(after.source_coverage, input.source_coverage);
        EXPECT_EQ(after.cache->indexes, indexes);
        EXPECT_EQ(after.cache->receipt, receipt);
        EXPECT_EQ(after.cache->revision, revision);
        EXPECT_EQ(after.cache->commit_token, nullptr);
        EXPECT_EQ(snapshots(), cached);
        EXPECT_EQ(arena.used, used);
        EXPECT_EQ(Access::cache_admission_statistics(state).lease_bytes, resident.lease_bytes);
        EXPECT_EQ(state.native_partition(), initial.view.partition());
        EXPECT_TRUE(D::NativePublicationSession::receipt_matches(
            after, accepted.view.partition(), accepted.view.mesh()->owned_cell_map()));
        const auto current = roots();
        for (std::size_t i = 0; i < current.size(); ++i)
        {
            EXPECT_EQ(current[i].header(), before[i].header());
            EXPECT_FALSE(current[i].header().owner_before(before[i].header()));
            EXPECT_FALSE(before[i].header().owner_before(current[i].header()));
            for (std::size_t channel = 0; channel < 2; ++channel)
                EXPECT_TRUE(current[i].payload().owners[channel].same_root(before[i].payload().owners[channel]));
        }
    };

    // The last owner is empty in MPI4. Reject before allocating the first
    // mirrored native counter, with its existing source bytes still spent.
    D::collective(*comm,
        [&]
        {
            if (original_limit - used < sizeof(D::Budget))
                throw std::invalid_argument("Rollback fixture has insufficient counter headroom.");
        });
    if (comm->getRank() == comm->getSize() - 1)
        limit = used + sizeof(D::Budget) - 1;
    EXPECT_THROW((void) A::publish_adaptive_native(state, schema), std::invalid_argument);
    limit = original_limit;
    check_accepted();

    auto wrong_map = accepted.view.mesh()->owned_cell_map();
    if (comm->getRank() == 0)
        wrong_map = Teuchos::null;
    EXPECT_THROW((void) state.with_partition(accepted.view.partition(), wrong_map), std::invalid_argument);
    check_accepted();
    (void) publish(trace, initial.hierarchy);
}
} // namespace
