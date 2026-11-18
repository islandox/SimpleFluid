/** @file NativeFragmentAccess.hh
 * @brief Private exact-control provenance for immutable detached native fragments.
 */
#pragma once
#include "geometry/PLIC/details/ExtrudedConstructionOwnerAccess.hh"
#include "geometry/details/NativeConstructorStorage.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include "geometry/mesh/NativeFragmentRecipe.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativePublicationSession;
class NativeSourceSubsetCoverage;
}
namespace SimpleFluid::Meshes
{
class NativeFragmentAccess;
/** A detached provider value. It has no source mesh/controller/cache capture. */
class SIMPLEFLUID_LOCAL NativeFragmentRecord final
{
    class ConstructionKey
    {
        friend class NativeFragmentAccess;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
    };

public:
    NativeFragmentRecord(
        ConstructionKey, MultiRegionMesh::Region region, geometry_detail::NativeConstructorStorageOwner metadata)
        : metadata_(std::move(metadata)), region_(std::move(region))
    {
    }
    [[nodiscard]] const MultiRegionMesh::Region& region() const noexcept { return region_; }
    [[nodiscard]] const auto& allocation() const noexcept { return allocation_; }
    [[nodiscard]] const auto& metadata() const noexcept { return metadata_; }

private:
    friend class NativeFragmentAccess;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    MultiRegionMesh::Region region_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
class SIMPLEFLUID_LOCAL NativeFragmentOrigin final
{
    class ConstructionKey
    {
        friend class NativeFragmentAccess;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
    };

public:
    explicit NativeFragmentOrigin(ConstructionKey) {}

private:
    friend class NativeFragmentAccess;
    std::weak_ptr<const NativeFragmentDescriptor> descriptor_;
    std::shared_ptr<const NativeFragmentRecord> fragment_;
    AMR::GeometryConstructionControls controls_;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> descriptor_allocation_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
/** Internal producer entry points. This establishes immutable provenance only;
 * a source/native equivalence capability still requires the session verifier. */
class SIMPLEFLUID_LOCAL NativeFragmentAccess
{
public:
    using Admission = geometry_detail::NativeConstructionAdmission;
    [[nodiscard]] static std::shared_ptr<const NativeFragmentDescriptor> extract(
        const MultiRegionMesh::Region&, NativeFragmentCache&, const Admission&);
    [[nodiscard]] static std::shared_ptr<const NativeFragmentRecord> find(
        const std::shared_ptr<const NativeFragmentDescriptor>&, AMR::GeometryConstructionControls);
    [[nodiscard]] static std::shared_ptr<const NativeFragmentRecord> materialize(
        const std::shared_ptr<const NativeFragmentDescriptor>&, const Admission&);
    /** Reuse immutable Product constructor components from the actual prior
     * admitted source. The successor still needs independent route authority. */
    [[nodiscard]] static std::shared_ptr<const NativeFragmentRecord> materialize_product_view(
        const std::shared_ptr<const NativeFragmentDescriptor>&, const NativeFragmentDescriptor& prior_descriptor,
        const NativeFragmentRecord& prior_fragment, const Admission&);
    /** Owner-local normalization only: complete parts/view bytes must agree.
     * Public decoder output cannot manufacture a matching source origin. */
    [[nodiscard]] static std::shared_ptr<const NativeFragmentDescriptor> rebind(
        const std::shared_ptr<const NativeFragmentDescriptor>& source,
        const std::shared_ptr<const NativeFragmentDescriptor>& normalized, const Admission&);
    using GeometryVisit = void (*)(void*, const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
    using CoverageVisit = void (*)(void*, const std::shared_ptr<const AMR::adaptive_detail::NativeSourceSubsetCoverage>&);
    /** Physical projection only. The returned flag authenticates descriptor
     * control; copied descriptors still project their real opaque Origin edge
     * but receive no source-equivalence authority. */
    [[nodiscard]] static bool origin_bound(const std::shared_ptr<const NativeFragmentDescriptor>&) noexcept;
    static void visit_descriptor(const std::shared_ptr<const NativeFragmentDescriptor>&,
        const geometry_detail::NativeConstructorOwnerSink&,
        PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit, GeometryVisit = nullptr, CoverageVisit = nullptr);
    static void visit_origin(const std::shared_ptr<const NativeFragmentDescriptor>&,
        const geometry_detail::NativeConstructorOwnerSink&,
        PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit, GeometryVisit = nullptr, CoverageVisit = nullptr);
    static void visit(const std::shared_ptr<const NativeFragmentRecord>&,
        const geometry_detail::NativeConstructorOwnerSink&,
        PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit, GeometryVisit = nullptr, CoverageVisit = nullptr);

private:
    friend class AMR::adaptive_detail::NativePublicationSession;
    static std::shared_ptr<const NativeFragmentRecord> materialize(
        const std::shared_ptr<const NativeFragmentDescriptor>&, const Admission&, const NativeFragmentDescriptor*,
        const NativeFragmentRecord*);
    static std::shared_ptr<const NativeFragmentDescriptor> bind(const std::shared_ptr<const NativeFragmentDescriptor>&,
        std::shared_ptr<const NativeFragmentRecord>, const Admission&);
    static std::shared_ptr<const NativeFragmentRecord> freeze(
        MultiRegionMesh::Region, geometry_detail::NativeConstructorStorageOwner, const Admission&);
};
} // namespace SimpleFluid::Meshes
