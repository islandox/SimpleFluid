/** @file NativeSparseAccess.hh
 * @brief Private admitted producer of source-cell fragments and sparse shards.
 */
#pragma once
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/NativeSparseFragments.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include <optional>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativeSparseAssembler;
class NativePublicationNumbering;
struct NativeSparseSealedCell
{
    std::uint64_t gid;
    std::size_t region;
    NativeSparseSeal seal;
};
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
    static void visit_geometry(const std::shared_ptr<const MultiRegionMesh>&,
        const geometry_detail::NativeConstructorOwnerSink&,
        PLIC::details::GeometryOwnerGraphEditor&, bool, PLIC::details::NativeGeometryAccounting*);
    [[nodiscard]] static bool visit_source_provider(
        const std::shared_ptr<const ExplicitRegionProvider>&, const geometry_detail::NativeConstructorOwnerSink&);
    /** Only the actual admitted sparse mesh and index can name an issuance. */
    [[nodiscard]] static std::optional<NativeSparseProof> output_provenance(
        const std::shared_ptr<const MultiRegionMesh>&);

private:
    friend class NativeSparseAssembler;
    [[nodiscard]] static std::shared_ptr<MultiRegionMesh> make(NativePublicationSession&,
        std::shared_ptr<const NativePublicationNumbering>, std::shared_ptr<const NativeSparseFragments>,
        std::shared_ptr<const NativeSparseNumericIndex>, std::span<const NativeSparseSealedCell>,
        std::uint64_t global_faces, std::uint64_t global_nodes);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
