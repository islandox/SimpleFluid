/** @file AdaptiveGeometrySnapshotAccess.hh
 * @brief Private mesh acquisition from an already authenticated native publication draft.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include <cstddef>
#include <memory>
#include <span>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveContactHalo;
struct Budget;
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
};
/** Narrow internal acquisition from an authenticated hierarchy. */
class GeometrySnapshotAccess
{
public:
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
    /** Observe the Mesh library image's actual PLIC domain without double counting callers. */
    [[nodiscard]] static PLIC::GeometryStatistics pipeline_geometry_statistics();
    [[nodiscard]] static const void* pipeline_geometry_statistics_domain();
    static void reset_pipeline_geometry_statistics();
    /** Same source-cache admission used by adapt, migrate and native binding.
     * Separate caller Budget retains its existing monotonic spent prefix. */
    static void admit_cache_residency(const DistributedAdaptiveHierarchy&, Budget&);
    static void visit_binding_sources(const AdaptiveGeometrySnapshot&, const std::function<void(const void*)>&);
    /** Exact immutable owner set of a published root, charged to `budget`. */
    [[nodiscard]] static geometry_snapshot_detail::PreparedOwnerManifest owner_manifest(
        const AdaptiveGeometrySnapshot&, Budget&, bool bounded = false);
    /** Finite capture of both channels. `payload_prepaid` requires prior full
     * graph admission in the same hierarchy overlap; new bookkeeping and index
     * construction are always charged to the remaining budget. */
    static void owner_manifests(const AdaptiveGeometrySnapshot&, Budget&,
        geometry_snapshot_detail::PreparedOwnerManifest& ordinary,
        geometry_snapshot_detail::PreparedOwnerManifest& bounded, bool payload_prepaid = false);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
