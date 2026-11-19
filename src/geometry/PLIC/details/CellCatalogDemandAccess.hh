/** @file CellCatalogDemandAccess.hh
 * @brief Private persistent catalog capture and demand extension.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/GeometryDemandOrder.hh"
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include <optional>

namespace SimpleFluid::PLIC::details
{
struct CellCatalogRebindBound
{
    std::size_t bytes = 0, owners = 0;
};
struct CellCatalogDemandWork
{
    std::size_t cumulative_metadata_bytes = 0;
    /** Decoded physical selector slots, including flat requested slots and
     * persistent input lookups. Scalar column reads and geometry evaluation are
     * outside this count; diagnostic visitors do not acquire a work sink. */
    geometry_demand::PageWork pages;
};
class CellCatalogDemandAccess
{
public:
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> initialize(
        std::shared_ptr<const CellGeometryCatalog<M>>, GeometryLimits = {}, CellCatalogDemandWork* = nullptr);
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> extended(
        std::shared_ptr<const CellGeometryCatalog<M>> accepted, std::shared_ptr<const CellGeometryCatalog<M>> requested,
        std::span<const std::optional<std::size_t>> previous_slots, GeometryLimits = {},
        CellCatalogDemandWork* = nullptr);
    /** Replace the physical Cartesian source of existing demand ordinals, for
     * example when an optional prepared page now covers those cells. Retires the
     * previous source holds and retains the new sources in one persistent edit. */
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> rebound(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::span<const std::size_t> ordinals,
        std::span<const geometry_demand::SourceSlot> physical, GeometryLimits = {});
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> rebound(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::span<const std::size_t>,
        std::span<const geometry_demand::SourceSlot>, GeometryLimits, CellCatalogDemandWork*);
    /** Synchronous construction receipts. Every local allocation and scratch
     * charge reaches the borrowed outer allowance before allocation. Created
     * source callbacks name only newly allocated explicit shell pages. */
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> initialize_precharged(
        std::shared_ptr<const CellGeometryCatalog<M>>, GeometryLimits, CellCatalogDemandWork*,
        geometry_demand::Admission);
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> extended_precharged(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::shared_ptr<const CellGeometryCatalog<M>>,
        std::span<const std::optional<std::size_t>>, GeometryLimits, CellCatalogDemandWork*, geometry_demand::Admission,
        geometry_demand::SourceSink created = {nullptr, nullptr});
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> rebound_precharged(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::span<const std::size_t>,
        std::span<const geometry_demand::SourceSlot>, GeometryLimits, geometry_demand::Admission);
    /** Construction and final new-owner envelope for AMR batches of at most four. */
    template<ArithmeticMode M>
    [[nodiscard]] static CellCatalogRebindBound rebound_bound(const CellGeometryCatalog<M>&, std::size_t count);
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> rebound_precharged(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::span<const std::size_t>,
        std::span<const geometry_demand::SourceSlot>, GeometryLimits, geometry_demand::Admission,
        CellCatalogDemandWork*);
    template<ArithmeticMode M>
    [[nodiscard]] static geometry_demand::OrderStamp stamp(const CellGeometryCatalog<M>&, std::size_t);
    template<ArithmeticMode M> [[nodiscard]] static bool persistent(const CellGeometryCatalog<M>&) noexcept;
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> exported(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::span<const std::size_t>, bool permutation,
        GeometryLimits = {});
    /** Diagnostic selectors with exact controls, rows and backing buffers. */
    template<ArithmeticMode M>
    static void visit_selector_pages(const CellGeometryCatalog<M>&, const geometry_demand::SelectorPageVisit&);
    template<ArithmeticMode M>
    static void visit_borrowed_dependencies(
        const CellGeometryCatalog<M>&, geometry_demand::BorrowedMetadataSink, geometry_demand::BorrowedSourceSink);
    template<ArithmeticMode M>
    static void visit_dependencies(
        const CellGeometryCatalog<M>&, geometry_demand::MetadataSink, geometry_demand::SourceSink);
private:
    template<ArithmeticMode M> static std::shared_ptr<CellGeometryCatalog<M>> make_catalog();
};
} // namespace SimpleFluid::PLIC::details
