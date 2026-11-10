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
struct CellCatalogDemandWork
{
    std::size_t cumulative_metadata_bytes = 0;
    geometry_demand::PageWork pages;
};
class CellCatalogDemandAccess
{
public:
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> initialize(
        std::shared_ptr<const CellGeometryCatalog<M>>, GeometryLimits = {});
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> extended(
        std::shared_ptr<const CellGeometryCatalog<M>> accepted, std::shared_ptr<const CellGeometryCatalog<M>> requested,
        std::span<const std::optional<std::size_t>> previous_slots, GeometryLimits = {},
        CellCatalogDemandWork* = nullptr);
    template<ArithmeticMode M>
    [[nodiscard]] static geometry_demand::OrderStamp stamp(const CellGeometryCatalog<M>&, std::size_t);
    template<ArithmeticMode M> [[nodiscard]] static bool persistent(const CellGeometryCatalog<M>&) noexcept;
    template<ArithmeticMode M>
    [[nodiscard]] static std::shared_ptr<const CellGeometryCatalog<M>> exported(
        std::shared_ptr<const CellGeometryCatalog<M>>, std::span<const std::size_t>, bool permutation,
        GeometryLimits = {});
    template<ArithmeticMode M>
    static void visit_borrowed_dependencies(
        const CellGeometryCatalog<M>&, geometry_demand::BorrowedMetadataSink, geometry_demand::BorrowedSourceSink);
    template<ArithmeticMode M>
    static void visit_dependencies(
        const CellGeometryCatalog<M>&, geometry_demand::MetadataSink, geometry_demand::SourceSink);
};
} // namespace SimpleFluid::PLIC::details
