/** @file AdaptiveGeometrySnapshotAccess.hh
 * @brief Private mesh acquisition from an already authenticated native publication draft.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include <cstddef>
#include <memory>
#include <span>
namespace SimpleFluid::PLIC::details
{
struct OwnerManifestPair;
class GeometryOwnerRootBinding;
} // namespace SimpleFluid::PLIC::details
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveContactHalo;
struct Budget;
struct AdaptiveLeafPageObservation;
struct PipelineGraphOwner
{
    enum class Kind
    {
        Object,
        Buffer,
        Geometry,
        AdmissionIdentity,
        NumericOutput
    };
    const void* identity;
    std::size_t bytes;
    Kind kind;
};
using PipelineGraphVisit = std::function<void(const PipelineGraphOwner&)>;
/** Admission diagnostics read scalars and six exact weak root controls. A root
 * census counter advances only when replacement actually traverses snapshots.
 * Independent explicit payload censuses remain available on the hierarchy. */
struct GeometryCacheAdmissionStatistics
{
    std::size_t cache_header_bytes = 0, resident_bytes = 0, lease_bytes = 0;
    std::size_t admission_queries = 0, union_censuses = 0, census_root_visits = 0;
    /** Cumulative persistent-catalog construction work over the cache lifetime. */
    std::size_t catalog_copied_metadata_bytes = 0, catalog_copied_pages = 0, catalog_visited_pages = 0;
    std::size_t catalog_copied_source_rows = 0, catalog_visited_source_rows = 0;
    std::size_t union_updates = 0, union_edge_visits = 0, union_dependency_entries = 0;
    std::size_t union_first_transitions = 0, union_last_transitions = 0;
    std::size_t union_index_copied_pages = 0, union_index_copied_bytes = 0, union_index_visited_pages = 0;
    std::size_t union_lookup_pages = 0;
    std::size_t union_delta_import_rows = 0;
};
struct GeometrySnapshotOwnerStatistics
{
    std::size_t dependency_entries = 0, edge_visits = 0, first_transitions = 0, last_transitions = 0;
    std::size_t index_copied_pages = 0, index_copied_bytes = 0, index_visited_pages = 0;
    std::size_t imported_entries = 0;
    std::size_t index_lookup_pages = 0;
    std::size_t stored_metadata_payload_bytes = 0, stored_metadata_admission_bytes = 0;
};
/** Narrow internal acquisition from an authenticated hierarchy. */
class GeometrySnapshotAccess
{
public:
    [[nodiscard]] static PLIC::details::OwnerManifestPair owner_pair(const AdaptiveGeometrySnapshot&);
    [[nodiscard]] static PLIC::details::GeometryOwnerRootBinding owner_binding(
        const std::shared_ptr<const AdaptiveGeometrySnapshot>&);

    /** Independent quiescent pointer-slot census for native Cartesian Raw roots.
     * Rejects other graph families; records every incoming edge before pruning
     * a physical owner already expanded by this diagnostic. No geometry work. */
    static void visit_cartesian_payload_edges(const AdaptiveGeometrySnapshot&,
        const std::function<void(const void* parent, const void* owner, std::size_t bytes)>&,
        const std::function<void(const void* binding_page, const void* buffer, std::size_t bytes)>& = {},
        const std::function<void(const void* binding_page, const AdaptiveCellKey& maximum_key)>& = {});
    [[nodiscard]] static GeometrySnapshotOwnerStatistics owner_statistics(const AdaptiveGeometrySnapshot&);
    /** Immutable binding-data control, independent of snapshot and hierarchy lifetime. */
    [[nodiscard]] static std::shared_ptr<const void> binding_owner(const AdaptiveGeometrySnapshot&);
    static void visit_owner_manifest_metadata(
        const AdaptiveGeometrySnapshot&, PLIC::details::geometry_demand::MetadataSink);
    /** Only native-publication consumers may call this seam, after the owner
     * fetch and authority phases have produced the complete draft. Supplied row
     * addresses must belong to that draft. This is not provider-input admission. */
    [[nodiscard]] static std::shared_ptr<const AdaptiveGeometrySnapshot> native_draft(
        const DistributedAdaptiveHierarchy&, const AdaptiveNativeDraft&, std::span<const AdaptiveLeaf* const>,
        const HierarchyObservation&, ArithmeticMode, GeometrySnapshotPurpose, std::size_t& used, std::size_t limit,
        PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&);
    /** Borrowed diagnostic access, valid only while the hierarchy is retained. */
    [[nodiscard]] static const DistributedAdaptiveContactHalo& contact_cache(const DistributedAdaptiveHierarchy&);
    [[nodiscard]] static GeometryCacheAdmissionStatistics cache_admission_statistics(
        const DistributedAdaptiveHierarchy&);
    /** Partial borrowed accepted-owner attribution for offline diagnostics.
     * Aggregate geometry payload identities do not enumerate every child buffer;
     * generic provider internals and separate shared controls may be unmatched.
     * Visitors acquire no pins. Raw allocation tracing remains authoritative. */
    static void visit_pipeline_graph(const DistributedAdaptiveHierarchy&, const PipelineGraphVisit&);
    /** Visit the immutable leaf tree while the accepted layout is retained. */
    static void visit_pipeline_leaf_pages(const DistributedAdaptiveHierarchy&,
        const std::function<void(const AdaptiveLeafPageObservation&)>&);
    /** Observe the Mesh library image's actual PLIC domain without double counting callers. */
    [[nodiscard]] static PLIC::GeometryStatistics pipeline_geometry_statistics();
    [[nodiscard]] static const void* pipeline_geometry_statistics_domain();
    static void reset_pipeline_geometry_statistics();
    /** Same source-cache admission used by adapt, migrate and native binding.
     * Separate caller Budget retains its existing monotonic spent prefix. */
    static void admit_cache_residency(const DistributedAdaptiveHierarchy&, Budget&);
    /** Explicit offline diagnostic; qualified admission never calls this census. */
    static void visit_cache_owner_union(const DistributedAdaptiveHierarchy&,
        const std::function<void(const PLIC::details::geometry_demand::OwnerRecord&)>&,
        PLIC::details::geometry_demand::MetadataSink);
    /** Quiescent access to the two actual native root slots and native domain
     * payment facts for an independent cold completeness check. */
    static void visit_native_cache_bindings(const DistributedAdaptiveHierarchy&,
        const std::function<void(const PLIC::details::GeometryOwnerRootBinding&)>&,
        const std::function<void(const PLIC::details::NativePaidExtentCharge&)>&);

    static void visit_binding_sources(const AdaptiveGeometrySnapshot&, const std::function<void(const void*)>&);
    /** Borrow the stored payload-only owner set in constant work; excludes the
     * snapshot header and both owner-manifest index graphs. `budget` is untouched. */
    [[nodiscard]] static geometry_snapshot_detail::PreparedOwnerManifest owner_manifest(
        const AdaptiveGeometrySnapshot&, Budget&, bool bounded = false);
    /** Borrow both stored payload channels by value. No census, new index or
     * admission is performed; payload_prepaid remains a compatibility argument. */
    static void owner_manifests(const AdaptiveGeometrySnapshot&, Budget&,
        geometry_snapshot_detail::PreparedOwnerManifest& ordinary,
        geometry_snapshot_detail::PreparedOwnerManifest& bounded, bool payload_prepaid = false);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
