/** @file NativeCellGeometrySourceAccess.hh
 * @brief Private trusted adoption of authenticated native Cartesian source ancestry.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include <utility>
namespace SimpleFluid::PLIC::details
{
/** State bootstrap caches moments immediately; raw source-only queries retain
 * the original box's first-query lazy moment environment. */
enum class NativeCellGeometryMoments
{
    PrimedState,
    DeferredRaw
};
/** Plain original stored-plane coefficients; no geometry/catalog owner. */
struct NativeOriginalFace
{
    std::array<long double, 3> normal{}, anchor{};
    GeometryPoint frame_origin{};
    int frame_exponent = 0, orientation = 1;
    long double offset = 0;
};
/** Exact original normalized-frame bounds, without coordinate roundtrips. */
struct NativeFramedBounds
{
    GeometryPoint frame_origin{};
    int frame_exponent = 0;
    std::array<std::array<long double, 2>, 3> bounds{};
};
/** A persistent native fact can pin its source directly, independent of catalog order. */
struct NativeCartesianCellReference
{
    std::shared_ptr<const CartesianCellSource> source;
    std::size_t cell_slot = 0;
};
/** The native adapter establishes recipe/key/revision/owner and original support
 * ancestry before adoption. Public geometry construction cannot attach keys. */
class NativeCellGeometrySourceAccess
{
public:
    template<ArithmeticMode M>
    [[nodiscard]] static ConvexGeometryPtr<M> existing_geometry(const CellGeometryView<M>& view)
    {
        view.require_valid();
        if (view.is_procedural())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Procedural view has no existing geometry owner.");
        return view.fallback();
    }
    template<ArithmeticMode M>
    [[nodiscard]] static typename CellGeometryCatalog<M>::ExplicitEntry explicit_entry(const CellGeometryView<M>& view)
    {
        view.require_valid();
        return view.catalog_->explicit_entry(view.slot_);
    }
    [[nodiscard]] static NativeCartesianCellReference source_cell(
        const CellGeometryView<ArithmeticMode::Ordinary>& view)
    {
        view.require_valid();
        if (!view.is_procedural())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Materialized view has no procedural source slot.");
        return {view.native_source(), view.source_slot()};
    }
    [[nodiscard]] static NativeOriginalFace original_face(
        const CellGeometryView<ArithmeticMode::Ordinary>& view, std::size_t face)
    {
        const auto ref = source_cell(view);
        return ref.source->original_face(ref.cell_slot, face);
    }
    [[nodiscard]] static NativeFramedBounds framed_bounds(const CellGeometryView<ArithmeticMode::Ordinary>& view)
    {
        const auto ref = source_cell(view);
        return ref.source->framed_bounds(ref.cell_slot);
    }
    /** Sufficient State-only compatibility proof. A false result requests the
     * unchanged existing original/imported fallback, never a numerical rejection. */
    [[nodiscard]] static bool state_compatible(GeometryPoint lower, GeometryPoint upper, GeometryLimits limits = {});
    /** Conservative admission for a bounded page with canonical box supports. */
    [[nodiscard]] static std::size_t regular_page_byte_bound(std::size_t cells, std::size_t axis_points);
    [[nodiscard]] static std::pair<GeometryPoint, GeometryPoint> raw_bounds(
        const NativeCartesianCellReference&);
    [[nodiscard]] static GeometryLimits source_limits(const NativeCartesianCellReference& ref)
    {
        return ref.source->query_limits();
    }
    [[nodiscard]] static std::span<const PlaneSupportIdentity<ArithmeticMode::Ordinary>> retained_supports(
        const CartesianCellSource& source)
    {
        return source.retained_supports();
    }
    [[nodiscard]] static std::array<std::size_t, 3> exclusive_bytes(const CartesianCellSource& source)
    {
        return source.retention_bytes();
    }
    [[nodiscard]] static std::array<std::size_t, 3> base_bytes(const CartesianCellSource& source)
    {
        return source.base_retention_bytes();
    }
    [[nodiscard]] static std::shared_ptr<const void> base_owner(const CartesianCellSource& source)
    {
        return source.base_owner();
    }
    static void visit_local_coordinates(const NativeCartesianCellReference& ref, void* context,
        void (*visitor)(void*, std::size_t, const PointCoordinates<ArithmeticMode::Ordinary>&))
    {
        ref.source->visit_local_coordinates(ref.cell_slot, context, visitor);
    }
    [[nodiscard]] static std::shared_ptr<const CartesianCellSource> make(CartesianSourceDescription description,
        GeometryLimits limits = {}, NativeCellGeometryMoments moments = NativeCellGeometryMoments::PrimedState)
    {
        return CartesianCellSource::make_authenticated(std::move(description), limits, moments);
    }
};
} // namespace SimpleFluid::PLIC::details
