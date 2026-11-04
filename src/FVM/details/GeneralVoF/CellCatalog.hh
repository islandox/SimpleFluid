/** @file CellCatalog.hh
 * @brief Compact accepted geometry catalog construction and authenticated native facts.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
{
SIMPLEFLUID_FVM_LOCAL std::size_t checked_product(std::size_t, std::size_t);
SIMPLEFLUID_FVM_LOCAL void charge_items(wire::Budget&, std::size_t count, std::size_t width);
SIMPLEFLUID_FVM_LOCAL void reserve_leaf_references(
    std::vector<const AMR::AdaptiveLeaf*>&, std::size_t count, wire::Budget&);
SIMPLEFLUID_FVM_LOCAL void reserve_indices(std::vector<std::size_t>&, std::size_t count, wire::Budget&);
SIMPLEFLUID_FVM_LOCAL bool eligible(const GeneralVoF::Hierarchy&, const AMR::AdaptiveLeaf&, ArithmeticMode);
SIMPLEFLUID_FVM_LOCAL bool same_native_recipe(const AMR::AdaptiveLeaf&, const AMR::AdaptiveLeaf&);
SIMPLEFLUID_FVM_LOCAL bool same_native_representation(
    const PLIC::CellGeometryView<ArithmeticMode::Ordinary>&, const PLIC::CellGeometryView<ArithmeticMode::Ordinary>&);
/** Construction-only borrowed axis reads, decoded once per actual recipe.
 * Its implementation never materializes a wide block or a convex cell. */
class SIMPLEFLUID_FVM_LOCAL NativeMetrics
{
public:
    NativeMetrics(const GeneralVoF::Hierarchy&, ArithmeticMode, wire::Budget&);
    ~NativeMetrics();
    NativeMetrics(const NativeMetrics&) = delete;
    NativeMetrics& operator=(const NativeMetrics&) = delete;
    AMR::AdaptiveCellMetrics operator()(const AMR::AdaptiveLeaf&);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
enum class CatalogPurpose
{
    State,
    OriginalSources
};
/** Named numeric controls only; exception status flags are not construction
 * controls. Unsupported platforms conservatively decline binding reuse. */
struct CatalogConstructionControls
{
    int rounding = -1;
    std::uint64_t x87_control = 0, simd_control = 0;
    bool supported = false;
    bool operator==(const CatalogConstructionControls&) const = default;
};
SIMPLEFLUID_FVM_LOCAL CatalogConstructionControls construction_controls();
/** Stack-scoped proof for initial/full catalogs in one constructor only. This
 * does not authorize publication and must never be stored in accepted Data. */
struct CatalogReuseAuthority
{
    const void* operation = nullptr;
    const void* original_catalog = nullptr;
    CatalogConstructionControls controls;
    bool stable = false;
};
struct CatalogBuild
{
    std::shared_ptr<const PLIC::CellGeometryCatalog<ArithmeticMode::Ordinary>> ordinary;
    // Present only when Ordinary fallback roots require the old separate import.
    std::shared_ptr<const PLIC::CellGeometryCatalog<ArithmeticMode::Ordinary>> finite_ordinary;
    std::shared_ptr<const PLIC::CellGeometryCatalog<ArithmeticMode::BoundedCertification>> bounded;
    std::vector<AMR::AdaptiveCellMetrics> native_metrics;
    CatalogReuseAuthority reuse_authority;
    template<ArithmeticMode Mode>
    PLIC::CellGeometryView<Mode> view(std::size_t slot) const
        requires(Mode != ArithmeticMode::Exact)
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return {finite_ordinary ? finite_ordinary : ordinary, slot};
        else
            return {bounded, slot};
    }
};
/** Owned rows first in their required order, then retained full support.
 * Publication authority remains with the validated hierarchy/native binding.
 * OriginalSources omits selected imports and native metric queries/storage.
 * An optional admitted_originals ledger joins already-admitted draft geometry
 * and support roots to the same cumulative operation scope. */
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL CatalogBuild build(const GeneralVoF::Hierarchy&, std::span<const AMR::AdaptiveLeaf* const>,
    wire::Budget&, const GeneralVoF::Data* reuse = nullptr, CatalogPurpose purpose = CatalogPurpose::State,
    PLIC::GeometryAllocationLedger* admitted_originals = nullptr,
    const CatalogReuseAuthority* reuse_authority = nullptr, const void* construction_operation = nullptr);
extern template CatalogBuild build<ArithmeticMode::Ordinary>(const GeneralVoF::Hierarchy&,
    std::span<const AMR::AdaptiveLeaf* const>, wire::Budget&, const GeneralVoF::Data*, CatalogPurpose,
    PLIC::GeometryAllocationLedger*, const CatalogReuseAuthority*, const void*);
extern template CatalogBuild build<ArithmeticMode::BoundedCertification>(const GeneralVoF::Hierarchy&,
    std::span<const AMR::AdaptiveLeaf* const>, wire::Budget&, const GeneralVoF::Data*, CatalogPurpose,
    PLIC::GeometryAllocationLedger*, const CatalogReuseAuthority*, const void*);
extern template CatalogBuild build<ArithmeticMode::Exact>(const GeneralVoF::Hierarchy&,
    std::span<const AMR::AdaptiveLeaf* const>, wire::Budget&, const GeneralVoF::Data*, CatalogPurpose,
    PLIC::GeometryAllocationLedger*, const CatalogReuseAuthority*, const void*);
} // namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
