/** @file CellGeometryView.ipp
 * @brief Small mode-dependent catalog and owning view forwarding definitions.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <algorithm>
namespace SimpleFluid::PLIC
{
template<ArithmeticMode M>
CellGeometryView<M>::CellGeometryView(std::shared_ptr<const CellGeometryCatalog<M>> catalog, std::size_t slot)
    : catalog_(std::move(catalog)), slot_(slot)
{
    if (!catalog_ || slot_ >= catalog_->size())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry view slot.");
}
template<ArithmeticMode M>
CellGeometryView<M> CellGeometryView<M>::from_existing(ConvexGeometryPtr<M> geometry, GeometryLimits limits)
{
    if (!geometry)
        return {};
    return {CellGeometryCatalog<M>::single_existing(std::move(geometry), limits), 0};
}
template<ArithmeticMode M> void CellGeometryView<M>::require_valid() const
{
    if (!catalog_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Empty cell geometry view.");
}
template<ArithmeticMode M> bool CellGeometryView<M>::is_procedural() const
{
    require_valid();
    return catalog_->procedural_slot(slot_);
}
template<ArithmeticMode M> const ConvexGeometryPtr<M>& CellGeometryView<M>::fallback() const
{
    require_valid();
    return catalog_->shell_geometry(slot_);
}
template<ArithmeticMode M> const typename CellGeometryCatalog<M>::SourcePtr& CellGeometryView<M>::native_source() const
{
    require_valid();
    return catalog_->source_owner(slot_);
}
template<ArithmeticMode M> std::size_t CellGeometryView<M>::source_slot() const
{
    require_valid();
    return catalog_->physical_row(slot_);
}
template<ArithmeticMode M> GeometryValue<M> CellGeometryView<M>::volume() const
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->volume(source_slot());
    return fallback()->volume();
}
template<ArithmeticMode M> GeometryMoments<M> CellGeometryView<M>::moments() const
{
    if constexpr (M == ArithmeticMode::Exact)
        return fallback()->moments();
    else
        return moments(origin());
}
template<ArithmeticMode M>
GeometryMoments<M> CellGeometryView<M>::moments(GeometryPoint origin) const
    requires(M != ArithmeticMode::Exact)
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->moments(source_slot(), origin);
    return fallback()->moments(origin);
}
template<ArithmeticMode M>
GeometryPoint CellGeometryView<M>::origin() const
    requires(M != ArithmeticMode::Exact)
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->origin(source_slot());
    return fallback()->origin();
}
template<ArithmeticMode M> GeometryPoint CellGeometryView<M>::centroid() const
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->centroid(source_slot());
    return fallback()->centroid();
}
template<ArithmeticMode M> std::array<GeometryValue<M>, 3> CellGeometryView<M>::axis_spans() const
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->axis_spans(source_slot());
    return fallback()->axis_spans();
}
template<ArithmeticMode M> std::size_t CellGeometryView<M>::num_faces() const
{
    if (is_procedural())
        return 6;
    return fallback()->num_faces();
}
template<ArithmeticMode M>
std::size_t CellGeometryView<M>::vertex_count() const
    requires(M != ArithmeticMode::Exact)
{
    if (is_procedural())
        return 8;
    return fallback()->vertex_count();
}
template<ArithmeticMode M>
std::array<long double, 3> CellGeometryView<M>::corner(std::size_t index) const
    requires(M == ArithmeticMode::Ordinary)
{
    if (is_procedural())
        return native_source()->corner(source_slot(), index);
    return details::cell_view_corner(*fallback(), index);
}
template<ArithmeticMode M>
CellGeometryFace<M> CellGeometryView<M>::ordered_face(std::size_t face) const
    requires(M != ArithmeticMode::Exact)
{
    if (face >= num_faces())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry face.");
    return {*this, face};
}
template<ArithmeticMode M>
CellGeometryFaces<M> CellGeometryView<M>::faces() const
    requires(M != ArithmeticMode::Exact)
{
    require_valid();
    return CellGeometryFaces<M>(*this);
}
template<ArithmeticMode M> std::size_t CellGeometryFace<M>::size() const
{
    return view_.is_procedural() ? 4 : view_.fallback()->faces().at(face_).size();
}
template<ArithmeticMode M> std::size_t CellGeometryFace<M>::at(std::size_t i) const
{
    if (i >= size())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry face vertex.");
    if (view_.is_procedural())
        return details::cell_view_box_face_vertex(face_, i);
    return view_.fallback()->faces().at(face_)[i];
}
template<ArithmeticMode M>
CellGeometryFaceMetrics CellGeometryFace<M>::metrics() const
    requires(M == ArithmeticMode::Ordinary)
{
    return {area(), centroid(), area_vector(), orientation()};
}
template<ArithmeticMode M>
GeometryValue<M> CellGeometryFace<M>::area() const
    requires(M == ArithmeticMode::Ordinary)
{
    if (view_.is_procedural())
        return view_.native_source()->face_area(view_.source_slot(), face_);
    return details::cell_view_face_area(*view_.fallback(), face_);
}
template<ArithmeticMode M>
GeometryPoint CellGeometryFace<M>::centroid() const
    requires(M == ArithmeticMode::Ordinary)
{
    if (view_.is_procedural())
        return view_.native_source()->face_centroid(view_.source_slot(), face_);
    return details::cell_view_face_centroid(*view_.fallback(), face_);
}
template<ArithmeticMode M>
GeometryPoint CellGeometryFace<M>::area_vector() const
    requires(M == ArithmeticMode::Ordinary)
{
    if (view_.is_procedural())
        return view_.native_source()->face_area_vector(view_.source_slot(), face_);
    return details::cell_view_face_area_vector(*view_.fallback(), face_);
}
template<ArithmeticMode M>
int CellGeometryFace<M>::orientation() const
    requires(M == ArithmeticMode::Ordinary)
{
    if (view_.is_procedural())
        return view_.native_source()->face_orientation(view_.source_slot(), face_);
    return details::cell_view_face_orientation(*view_.fallback(), face_);
}
template<ArithmeticMode M>
std::array<std::array<long double, 2>, 3> CellGeometryView<M>::closed_contact_bounds() const
    requires(M == ArithmeticMode::Ordinary)
{
    if (is_procedural())
        return native_source()->closed_contact_bounds(source_slot());
    return fallback()->closed_contact_bounds();
}
template<ArithmeticMode M>
Support<M> CellGeometryView<M>::support(GeometryPoint normal, GeometryPoint anchor) const
    requires(M != ArithmeticMode::Exact)
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->support(source_slot(), normal, anchor);
    return fallback()->support(normal, anchor);
}
template<ArithmeticMode M>
PlaneSupportIdentity<ArithmeticMode::Ordinary> CellGeometryView<M>::source_support(std::size_t face) const
    requires(M == ArithmeticMode::Ordinary)
{
    if (is_procedural())
        return native_source()->source_support(source_slot(), face);
    return details::cell_view_source_support(*fallback(), face);
}
template<ArithmeticMode M>
bool CellGeometryView<M>::face_contains(std::size_t face, const PlanarFace<ArithmeticMode::Ordinary>& other) const
    requires(M == ArithmeticMode::Ordinary)
{
    if (is_procedural())
        return native_source()->face_contains(source_slot(), face, other);
    return details::cell_view_face_contains(*fallback(), face, other);
}
template<ArithmeticMode M>
bool CellGeometryView<M>::contains_source_point(
    std::size_t face, const PointIdentity<ArithmeticMode::Ordinary>& point) const
    requires(M == ArithmeticMode::Ordinary)
{
    if (is_procedural())
        return native_source()->contains_source_point(source_slot(), face, point);
    return details::cell_view_contains_source_point(*fallback(), face, point);
}
template<ArithmeticMode M>
ConvexGeometryPtr<M> CellGeometryView<M>::materialize(CellGeometryOperationWorkspace& work) const
    requires(M != ArithmeticMode::Exact)
{
    if constexpr (M == ArithmeticMode::Ordinary)
        if (is_procedural())
            return native_source()->materialize(source_slot(), work);
    work.retain_temporary(fallback());
    return fallback();
}
template<ArithmeticMode M>
void CellGeometryView<M>::validate_source_authority(GeometryLimits limits) const
    requires(M == ArithmeticMode::Ordinary)
{
    if (!is_procedural())
        details::cell_view_validate_source_authority(*fallback(), limits);
}
template<ArithmeticMode M>
CellGeometryRange<M>::CellGeometryRange(std::shared_ptr<const CellGeometryCatalog<M>> catalog)
    : catalog_(std::move(catalog)), count_(catalog_ ? catalog_->size() : 0)
{
}
template<ArithmeticMode M>
CellGeometryRange<M>::CellGeometryRange(
    std::shared_ptr<const CellGeometryCatalog<M>> catalog, std::size_t begin, std::size_t count)
    : catalog_(std::move(catalog)), first_(begin), count_(count)
{
    const auto available = catalog_ ? catalog_->size() : 0;
    if (begin > available || count > available - begin)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry catalog slice.");
}
template<ArithmeticMode M> CellGeometryView<M> CellGeometryRange<M>::at(std::size_t index) const
{
    if (index >= size())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry range index.");
    return {catalog_, first_ + index};
}
} // namespace SimpleFluid::PLIC
