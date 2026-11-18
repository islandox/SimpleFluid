/** @file NativeSparseAccess.hh
 * @brief Private admitted producer of source-cell fragments and sparse shards.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativeSparseAssembler;
class NativeSourceSubsetCoverage;
struct NativeSourceCellFragment
{
    MultiRegionMesh::Region region;
    geometry_detail::NativeConstructorStorageOwner metadata;
};
/** No public SparseNativeMeshInput flags or proof registration exist. This
 * private producer consumes Session-created source/incidence seals only. */
class SIMPLEFLUID_LOCAL NativeSparseAccess final
{
public:
    using Admission = geometry_detail::NativeConstructionAdmission;
    [[nodiscard]] static NativeSourceCellFragment source_cell_fragment(
        const AdaptiveLeaf&, const AdaptiveCellMetrics&, const Admission&,
        std::shared_ptr<const NativeSourceSubsetCoverage> = {});
    [[nodiscard]] static PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> source_provider_geometry(
        const std::shared_ptr<const ExplicitRegionProvider>&);
    /** Only the private source-cell form supplies an original geometry. */
    [[nodiscard]] static PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> source_provider_geometry(
        const ExplicitRegionProvider&);
    [[nodiscard]] static const std::shared_ptr<const NativeSourceSubsetCoverage>& source_provider_coverage(
        const std::shared_ptr<const ExplicitRegionProvider>&);
    [[nodiscard]] static bool visit_source_provider(
        const std::shared_ptr<const ExplicitRegionProvider>&, const geometry_detail::NativeConstructorOwnerSink&);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
