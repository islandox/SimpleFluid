/** Private test/probe adapter. All accesses borrow quiescent storage. */
#pragma once
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
namespace retention = PLIC::details::retention;
enum class RetentionGroup : std::uint32_t
{
    Accepted,
    Candidate,
    PreparedSource,
    PreparedCandidate,
    CopiedPoints,
    CopiedTrajectories,
    ActiveWork,
    ExternalFixture
};
/** This seam must never return a shared_ptr or invoke a lazy geometry query. */
struct RetentionAccess
{
    static const GeneralVoF::Data* data(const GeneralVoF& s) noexcept { return s.data_.get(); }
    static const GeneralVoF::Data* source(const GeneralVoF::Trial& t) noexcept { return t.source_.get(); }
    static const GeneralVoF::Data* candidate(const GeneralVoF::Trial& t) noexcept { return t.candidate_.get(); }
    static const auto& trajectories(const GeneralVoFFiniteTrajectories& t) noexcept { return t.storage_; }
    static const GeneralVoF::Data* source(const GeneralVoFPreparedTransport& p) noexcept
    {
        return static_cast<const GeneralVoF::Data*>(p.authenticated_source_.get());
    }
    static const GeneralVoF::Data* candidate(const GeneralVoFPreparedTransport& p) noexcept
    {
        return static_cast<const GeneralVoF::Data*>(p.authenticated_candidate_.get());
    }
    static bool authentic_public_pins(const GeneralVoFPreparedTransport& p) noexcept
    {
        const auto* s = source(p);
        if (p.source_identity.get() != s || p.candidate_identity.get() != candidate(p) ||
            p.balance_work.get() != p.authenticated_balance_work_.get())
            return false;
        if (!s)
            return !p.mesh_identity && !p.topology_identity && !p.balance_work;
        if (p.mesh_identity.get() != s->hierarchy.get() || p.topology_identity.get() != s->topology.get())
            return false;
        return !p.balance_work || (p.balance_work->source_identity.get() == s &&
                                     p.balance_work->mesh_identity.get() == s->hierarchy.get() &&
                                     p.balance_work->topology_identity.get() == s->topology.get());
    }
};
/** Bounded caller scratch is separate from construction admission budgets. */
class RetentionRoots
{
public:
    explicit RetentionRoots(retention::ProbeScratch& scratch) : roots(&scratch), ledgers(&scratch) {}
    bool complete_manifest = true;
    std::pmr::vector<retention::BorrowedRoot> roots;
    std::pmr::vector<const PLIC::GeometryAllocationLedger*> ledgers;
    template<class T> void add(const T& value, RetentionGroup group)
    {
        roots.push_back(retention::root(value, static_cast<std::uint32_t>(group)));
    }
    template<class T> void pointer(const std::shared_ptr<const T>& value, RetentionGroup group)
    {
        if (value)
            add(*value, group);
    }
    void leaf(const AMR::AdaptiveLeaf& value, RetentionGroup group)
    {
        pointer(value.geometry, group);
        for (const auto& surface : value.surfaces)
        {
            pointer(surface.geometry, group);
            pointer(surface.native_whole_geometry, group);
            add(surface.authority_image, group);
        }
    }
    void geometry_snapshot(const AMR::AdaptiveGeometrySnapshot& value, RetentionGroup group)
    {
        value.visit_external_owners(
            [&](const void* owner, std::size_t bytes)
            {
                roots.push_back(retention::external_root(owner, bytes, static_cast<std::uint32_t>(group)));
            });
        pointer(value.ordinary_catalog(), group);
        pointer(value.finite_ordinary_catalog(), group);
        pointer(value.bounded_catalog(), group);
        value.visit_authoritative_surfaces(
            [&](const AMR::GeometrySnapshotSurface& binding)
            {
                pointer(binding.surface.geometry, group);
                pointer(binding.surface.native_whole_geometry, group);
                add(binding.surface.authority_image, group);
            });
    }
    void hierarchy(const GeneralVoF::Hierarchy* h, RetentionGroup group)
    {
        if (!h)
            return;
        h->visit_geometry_snapshots([&](const AMR::AdaptiveGeometrySnapshot& value)
            { geometry_snapshot(value, group); });
        for (const auto& value : h->owned_leaves())
            leaf(value, group);
        for (const auto& [key, value] : h->families())
            leaf(value.parent, group);
    }
    void tile(const AMR::AdaptiveTile& value, RetentionGroup group)
    {
        pointer(value.geometry, group);
        add(value.second_image_identity, group);
    }
    void state(const GeneralVoF::Data* value, RetentionGroup group)
    {
        if (!value)
            return;
        if (value->options.arithmetic_mode == ArithmeticMode::Exact)
            complete_manifest = false;
        if (value->topology)
        {
            if (value->topology->geometry_snapshot)
                geometry_snapshot(*value->topology->geometry_snapshot, group);
            pointer(value->topology->cell_geometry, group);
            pointer(value->topology->ordinary_cell_geometry, group);
            pointer(value->topology->bounded_cell_geometry, group);
        }
        hierarchy(value->hierarchy.get(), group);
        if (value->supplied_hierarchy.get() != value->hierarchy.get())
            hierarchy(value->supplied_hierarchy.get(), group);
        if (!value->topology)
            return;
        const auto& t = *value->topology;
        std::visit([&](const auto& points) { add(points, group); }, t.finite_nodes.snapshot);
        for (const auto& v : t.tiles)
            tile(v, group);
        for (const auto& v : t.extra_surface_tiles)
            tile(v, group);
        for (const auto& v : t.draft.tiles)
            tile(v, group);
        for (const auto& v : t.nodes)
            add(v.coordinate_identity, group);
        for (const auto& v : t.draft.nodes)
            add(v.coordinate_identity, group);
        for (const auto& v : t.draft.owned)
            leaf(v, group);
        for (const auto& v : t.draft.ghosts)
            leaf(v, group);
        for (const auto& [node, walls] : t.node_walls)
            for (const auto& wall : walls)
            {
                pointer(wall.geometry, group);
                add(wall.finite_normal, group);
            }
        if (const auto* cache = t.spatial_cache.get())
        {
            pointer(cache->geometry_catalog, group);
            for (const auto& f : cache->frontiers)
            {
                pointer(f.geometry, group);
                if (f.support_identity)
                    add(*f.support_identity, group);
                if (f.original_boundary)
                    add(*f.original_boundary, group);
            }
        }
    }
    void state(const GeneralVoF& value, RetentionGroup group = RetentionGroup::Accepted)
    {
        state(RetentionAccess::data(value), group);
    }
    void snapshot(const geometry::swept_volume::OrdinaryTrajectorySnapshot* snapshot, RetentionGroup group)
    {
        if (!snapshot)
            return;
        for (const auto& e : snapshot->entries())
        {
            add(e.current, group);
            add(e.departure, group);
        }
    }
    void trajectories(const GeneralVoFFiniteTrajectories& value, RetentionGroup group)
    {
        const auto& storage = RetentionAccess::trajectories(value);
        if (const auto* p = std::get_if<std::shared_ptr<const GeneralVoFFiniteTrajectories::Snapshot>>(&storage))
            snapshot(p->get(), group);
        else
            for (const auto& e : std::get<std::vector<GeneralVoFFiniteNodeTrajectory>>(storage))
            {
                add(e.point, group);
                add(e.departure, group);
            }
    }
    void trial(const GeneralVoF::Trial& value)
    {
        state(RetentionAccess::source(value), RetentionGroup::Accepted);
        state(RetentionAccess::candidate(value), RetentionGroup::Candidate);
        trajectories(value.finite_node_trajectories(), RetentionGroup::Candidate);
        if (value.prepared_transport())
            prepared(*value.prepared_transport());
    }
    void prepared(const GeneralVoFPreparedTransport& value)
    {
        complete_manifest &= RetentionAccess::authentic_public_pins(value);
        state(RetentionAccess::source(value), RetentionGroup::PreparedSource);
        state(RetentionAccess::candidate(value), RetentionGroup::PreparedCandidate);
        trajectories(value.node_trajectories, RetentionGroup::PreparedCandidate);
    }
    void work(const geometry::swept_volume::Work& value)
    {
        ledgers.push_back(&value.allocations);
        for (const auto& snapshot_owner : value.trajectory_owners)
            snapshot(snapshot_owner.get(), RetentionGroup::ActiveWork);
    }
};
/** Covered PLIC storage roots: accepted/candidate cell shells, explicit hierarchy
 * and family shells, topology/draft tiles and walls, canonical point snapshot,
 * cached frontier shells, prepared source/candidate and copied trajectories,
 * and caller-supplied Work ledgers. Native recipes, mesh/partition fields,
 * trajectory owner/entry buffers, allocator/control blocks and MPI allocations
 * are outside this PLIC census.
 * No hierarchy.geometry(), snapshot.point(), trajectory.at(), candidate(),
 * serialization, weak locks or numerical geometry calls are permitted here. */
} // namespace SimpleFluid::VoF::general_vof_detail
