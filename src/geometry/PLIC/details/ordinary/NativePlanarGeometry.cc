/** @file NativePlanarGeometry.cc
 * @brief Ordinary planar ownership, checked native support and shared span queries.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/details/NativePlanarCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/PLIC/details/NativeGeometryByteWriter.hh"
#include "geometry/PLIC/details/GeometryPacketHeader.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/CanonicalExportCoordinates.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensusAccess.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/NativePlanarQueries.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace SimpleFluid::PLIC
{
namespace on = details::ordinary_native;
using IdentityAccess = details::GeometryIdentityAccess;
using OrdinaryPoint = PointIdentity<ArithmeticMode::Ordinary>;
using OrdinaryLine = LineIdentity<ArithmeticMode::Ordinary>;
using OrdinarySupport = PlaneSupportIdentity<ArithmeticMode::Ordinary>;
using OrdinaryDisplacement = DisplacementIdentity<ArithmeticMode::Ordinary>;
using OrdinaryFace = PlanarFace<ArithmeticMode::Ordinary>;
using OrdinaryFacePtr = PlanarFacePtr<ArithmeticMode::Ordinary>;
struct OrdinaryFace::Impl
{
    std::vector<on::WidePoint> polygon;
    on::WidePoint normal{};
    std::optional<OrdinarySupport> support;
    bool retained_support = false;
};
bool details::NativePlanarOwnerAccess::admitted(const Pointer& face)
{
    return face && face->impl_ && geometry_detail::NativeObjectOwnerAccess<Face>::admitted(face) &&
           geometry_detail::NativeObjectOwnerAccess<Face::Impl>::admitted(face->impl_);
}
const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&
details::NativePlanarOwnerAccess::wrapper_envelope(const Pointer& face)
{
    if (!admitted(face))
        throw std::invalid_argument("Foreign native planar wrapper/control.");
    return geometry_detail::NativeObjectOwnerAccess<Face>::envelope(face);
}
std::shared_ptr<const void> details::NativePlanarOwnerAccess::payload_owner(const Pointer& face)
{
    if (!admitted(face))
        throw std::invalid_argument("Foreign native planar payload/control.");
    return face->impl_;
}
const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>&
details::NativePlanarOwnerAccess::payload_envelope(const Pointer& face)
{
    if (!admitted(face))
        throw std::invalid_argument("Foreign native planar payload envelope.");
    return geometry_detail::NativeObjectOwnerAccess<Face::Impl>::envelope(face->impl_);
}
const geometry_detail::NativeConstructorStorageOwner& details::NativePlanarOwnerAccess::storage(const Pointer& face)
{
    if (!admitted(face))
        throw std::invalid_argument("Foreign native planar storage receipt.");
    return geometry_detail::NativeObjectOwnerAccess<Face::Impl>::storage(face->impl_);
}
std::shared_ptr<const void> details::RetentionAccess::planar_payload_owner(const OrdinaryFace& face) noexcept
{
    return face.impl_;
}
std::size_t details::RetentionAccess::planar_metadata_bytes() noexcept
{
    return sizeof(OrdinaryFace::Impl);
}
const PlaneSupportIdentity<ArithmeticMode::Ordinary>* details::RetentionAccess::planar_support_borrowed(
    const OrdinaryFace& face) noexcept
{
    return face.impl_->support ? &*face.impl_->support : nullptr;
}
void GeometryAllocationLedger::retain_planar_face(const OrdinaryFacePtr& face, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!face)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null Ordinary planar face admission.");
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, ArithmeticMode::Ordinary);
        else
            impl_->mode = ArithmeticMode::Ordinary;
        const bool native = details::NativePlanarOwnerAccess::admitted(face);
        if (native)
        {
            impl_->native_typed_receipt(allocation_detail::Kind::PlanarWrapper, face,
                details::NativePlanarOwnerAccess::wrapper_envelope(face), {}, used, limit);
            impl_->native_typed_receipt(allocation_detail::Kind::PlanarPayload, face->impl_,
                details::NativePlanarOwnerAccess::payload_envelope(face),
                details::NativePlanarOwnerAccess::storage(face), used, limit);
        }
        impl_->admit({allocation_detail::Kind::PlanarWrapper, face}, native ? 0 : sizeof(OrdinaryFace), used, limit);
        impl_->admit({allocation_detail::Kind::PlanarPayload, face->impl_},
            native ? 0 : face->exclusive_storage_bytes(), used, limit);
        if (native)
        {
            impl_->native_object_children(details::NativePlanarOwnerAccess::wrapper_envelope(face), {}, used, limit);
            impl_->native_object_children(details::NativePlanarOwnerAccess::payload_envelope(face),
                details::NativePlanarOwnerAccess::storage(face), used, limit);
        }
        if (face->impl_->support)
            impl_->retain_identity(*face->impl_->support, used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}
struct EdgePointIndex<ArithmeticMode::Ordinary>::Impl
{
    std::map<on::WidePoint, OrdinaryPoint> points;
};
struct OrdinaryEdgeQueryScratch::Impl
{
    struct Hit
    {
        long double projection;
        std::size_t ordinal;
        const OrdinaryPoint* identity;
    };
    std::unique_ptr<Hit[]> hits;
    std::shared_ptr<const EdgePointIndex<ArithmeticMode::Ordinary>::Impl> index;
    std::size_t capacity = 0, size = 0;
};
namespace
{
using Polygon = std::vector<on::WidePoint>;
namespace predicates = details::ordinary_predicates;
constexpr std::size_t vertex_cap = 4096;
namespace planar_queries = details::ordinary_planar_queries;
using planar_queries::axis_of;
using planar_queries::center;
using planar_queries::defining_plane;
using planar_queries::DefiningPlane;
using planar_queries::norm;
using planar_queries::on_plane;
using planar_queries::point_inside;
void validate(const Polygon& p, const geometry_detail::NativeConstructionAdmission* admission = nullptr)
{
    if (p.size() > vertex_cap)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary native planar vertex cap exhausted.");
    if (p.size() < 3)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary native planar polygon has too few vertices.");
    const auto storage = admission ? geometry_detail::native_constructor_storage(*admission)
                                   : geometry_detail::NativeConstructorStorageOwner{};
    std::set<on::WidePoint, std::less<on::WidePoint>, geometry_detail::NativeStorageAllocator<on::WidePoint>> seen(
        std::less<on::WidePoint>{}, geometry_detail::NativeStorageAllocator<on::WidePoint>(storage.owner));
    for (const auto& v : p)
    {
        for (auto x : v)
            on::finite(x);
        if (!seen.insert(v).second)
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Ordinary native planar vertex is repeated.");
    }
    const auto plane = defining_plane(p);
    if (!(norm(on::polygon_vector(p)) > 0))
        throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, "Ordinary native face area is unresolved.");
    for (const auto& v : p)
        if (!on_plane(p, plane, v))
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Ordinary native face is nonplanar.");
    for (std::size_t e = 0; e < p.size(); ++e)
    {
        for (const auto& v : p)
            if (plane.winding * predicates::require_sign(predicates::projected_orientation(
                                    p[e], p[(e + 1) % p.size()], v, plane.projection)) <
                0)
                throw GeometryError(GeometryErrorCode::InvalidTopology, "Ordinary native face is nonconvex.");
    }
}
bool coplanar(std::span<const on::WidePoint> a, std::span<const on::WidePoint> b)
{
    const auto plane = defining_plane(a);
    return std::all_of(b.begin(), b.end(), [&](const auto& p) { return on_plane(a, plane, p); });
}
template<class Vector>
Vector clip_loop_storage(
    const Vector& source, const on::WidePoint& a, const on::WidePoint& b, const on::WidePoint& n, bool inside)
{
    Vector result(source.get_allocator());
    if (source.empty())
        return result;
    const auto edge = on::sub(b, a);
    auto distance = [&](const auto& p)
    {
        const auto axis = axis_of(n);
        const auto d = on::cross(edge, on::sub(p, a))[axis] * (n[axis] < 0 ? -1.L : 1.L);
        return inside ? d : -d;
    };
    auto append = [&](const auto& p)
    {
        if (result.empty() || result.back() != p)
            result.push_back(p);
    };
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const auto& p = source[i];
        const auto& q = source[(i + 1) % source.size()];
        const auto dp = distance(p), dq = distance(q);
        if (dp >= 0)
            append(p);
        if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0))
        {
            // Identical endpoint order gives neighboring polygons identical
            // intersections.
            const auto& lo = p < q ? p : q;
            const auto& hi = p < q ? q : p;
            const auto dl = distance(lo), dh = distance(hi);
            const auto denominator = dl - dh;
            if (denominator == 0)
                on::reject("unresolved native edge intersection.");
            append(on::add(lo, on::mul(on::sub(hi, lo), dl / denominator)));
        }
    }
    if (result.size() > 1 && result.front() == result.back())
        result.pop_back();
    if (result.size() < 3 || norm(on::polygon_vector(result)) == 0)
        result.clear();
    return result;
}
Polygon clip_loop(
    const Polygon& source, const on::WidePoint& a, const on::WidePoint& b, const on::WidePoint& n, bool inside)
{
    return clip_loop_storage(source, a, b, n, inside);
}
Polygon intersect_loop(Polygon a, const Polygon& b, bool retained_support = false)
{
    if (!retained_support && !coplanar(a, b))
        on::reject("native planar intersection has distinct supports.");
    const auto n = on::polygon_vector(b);
    for (std::size_t i = 0; i < b.size() && !a.empty(); ++i)
        a = clip_loop(a, b[i], b[(i + 1) % b.size()], n, true);
    return a;
}
::SimpleFluid::MeshUtils::Vec3 vector(const on::WidePoint& p)
{
    const auto d = on::display(p);
    return {d[0], d[1], d[2]};
}
OrdinaryLine line_identity(on::WidePoint a, on::WidePoint b)
{
    auto d = on::sub(b, a);
    const auto axis = axis_of(d);
    if (d[axis] == 0)
        on::reject("zero native edge.");
    d = on::mul(d, 1 / d[axis]);
    a = on::sub(a, on::mul(d, a[axis]));
    return IdentityAccess::line({axis, d, a});
}
} // namespace
OrdinaryFacePtr OrdinaryFace::make(std::span<const GeometryPoint> vertices)
{
    if (vertices.size() > vertex_cap)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary native planar vertex cap exhausted.");
    Polygon points;
    points.reserve(vertices.size());
    for (const auto& p : vertices)
        points.push_back(on::wide(p));
    return make_wide(points);
}
OrdinaryFacePtr OrdinaryFace::make_wide(std::span<const WidePoint> vertices)
{
    if (vertices.size() > vertex_cap)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary native planar vertex cap exhausted.");
    auto impl = std::make_shared<Impl>();
    impl->polygon.assign(vertices.begin(), vertices.end());
    validate(impl->polygon);
    impl->normal = on::polygon_vector(impl->polygon);
    impl->support = on::make_source_support(impl->polygon);
    return OrdinaryFacePtr(new OrdinaryFace(std::move(impl)));
}
OrdinaryFacePtr OrdinaryFace::from_retained_support(
    std::span<const WidePoint> vertices, WidePoint normal, OrdinarySupport support)
{
    planar_queries::validate_retained_loop(vertices, normal);
    auto impl = std::make_shared<Impl>();
    impl->polygon.assign(vertices.begin(), vertices.end());
    impl->normal = normal;
    impl->support = std::move(support);
    impl->retained_support = true;
    return OrdinaryFacePtr(new OrdinaryFace(std::move(impl)));
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::retained(
    std::span<const Face::WidePoint> vertices, Face::WidePoint normal,
    PlaneSupportIdentity<ArithmeticMode::Ordinary> support,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    planar_queries::validate_retained_loop(vertices, normal);
    auto payload = geometry_detail::NativeObjectOwnerAccess<Face::Impl>::make(admission);
    geometry_detail::NativeObjectOwnerAccess<Face::Impl>::storage(payload.owner)
        .owner->copy(payload.owner->polygon, vertices);
    payload.owner->normal = normal;
    payload.owner->support = std::move(support);
    payload.owner->retained_support = true;
    auto wrapper = geometry_detail::NativeObjectOwnerAccess<Face>::construct_without_buffers(
        admission, [&](Face* target) { ::new (static_cast<void*>(target)) Face(payload.owner); });
    return std::move(wrapper.owner);
}

namespace
{
OrdinarySupport native_source_support(
    std::span<const on::WidePoint> points, const geometry_detail::NativeConstructionAdmission& admission)
{
    // Preserve make_source_support's axis test, collinear removal and sorted
    // source-point order while admitting its concrete polygon scratch.
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (std::all_of(
                points.begin(), points.end(), [&](const auto& point) { return point[axis] == points.front()[axis]; }))
            return IdentityAccess::support_axis_admitted(axis, points.front()[axis], admission);
    const auto storage = geometry_detail::native_constructor_storage(admission);
    Polygon polygon;
    storage.owner->copy(polygon, points);
    bool changed = true;
    while (changed && polygon.size() > 3)
    {
        changed = false;
        for (std::size_t i = 0; i < polygon.size(); ++i)
            if (on::cross(on::sub(polygon[i], polygon[(i + polygon.size() - 1) % polygon.size()]),
                    on::sub(polygon[(i + 1) % polygon.size()], polygon[i])) == on::WidePoint{})
            {
                polygon.erase(polygon.begin() + i);
                changed = true;
                break;
            }
    }
    std::sort(polygon.begin(), polygon.end());
    return IdentityAccess::support_source_admitted(polygon, admission);
}
} // namespace
PlaneSupportIdentity<ArithmeticMode::Ordinary> details::NativePlanarOwnerAccess::source_support(
    std::span<const Face::WidePoint> points, const geometry_detail::NativeConstructionAdmission& admission)
{
    return native_source_support(points, admission);
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::make_wide(
    std::span<const Face::WidePoint> points, const geometry_detail::NativeConstructionAdmission& admission)
{
    if (points.size() > vertex_cap)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary native planar vertex cap exhausted.");
    // The original validation and normal arithmetic run before support issuance.
    // Its uniqueness tree uses the actual native node allocator.
    const auto scratch = geometry_detail::native_constructor_storage(admission);
    Polygon polygon;
    scratch.owner->copy(polygon, points);
    validate(polygon, &admission);
    const auto normal = on::polygon_vector(polygon);
    auto support = native_source_support(polygon, admission);
    auto result = retained(polygon, normal, std::move(support), admission);
    std::const_pointer_cast<Face::Impl>(result->impl_)->retained_support = false;
    return result;
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::make(
    std::span<const GeometryPoint> points, const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto storage = geometry_detail::native_constructor_storage(admission);
    Polygon polygon;
    storage.owner->reserve(polygon, points.size());
    for (const auto& point : points)
        polygon.push_back(on::wide(point));
    return make_wide(polygon, admission);
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::translated(const Face& face,
    const OrdinaryDisplacement& displacement, const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto shift = IdentityAccess::displacement(displacement);
    auto support = IdentityAccess::translated_support_admitted(*face.impl_->support, shift, admission);
    auto result = retained(face.points(), face.impl_->normal, std::move(support), admission);
    auto payload = std::const_pointer_cast<Face::Impl>(result->impl_);
    for (auto& point : payload->polygon)
        point = on::add(point, shift);
    planar_queries::validate_retained_loop(payload->polygon, payload->normal);
    return result;
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::reversed(
    const Face& face, const geometry_detail::NativeConstructionAdmission& admission)
{
    // Construct exactly one final polygon; no independently allocated reverse temporary.
    auto result = retained(face.points(), face.impl_->normal, *face.impl_->support, admission);
    auto payload = std::const_pointer_cast<Face::Impl>(result->impl_);
    std::reverse(payload->polygon.begin(), payload->polygon.end());
    return result;
}
OrdinaryIncidenceFrame capture_incidence_frame(const OrdinaryFace& face)
{
    return {*face.impl_->support, face.impl_->normal};
}
OrdinaryFacePtr restore_incidence_loop(const OrdinaryIncidenceFrame& frame, std::span<const OrdinaryPoint> points)
{
    std::vector<OrdinaryFace::WidePoint> coordinates;
    coordinates.reserve(points.size());
    for (const auto& point : points)
        coordinates.push_back(IdentityAccess::point(point));
    return OrdinaryFace::from_retained_support(coordinates, frame.normal, frame.support);
}
OrdinaryFacePtr OrdinaryFace::from_native_face(const MultiRegionMesh& mesh, std::uint64_t face)
{
    const auto lease = mesh.acquire_execution_view();
    if (mesh.sparse_native())
        return mesh.certified_sparse_face_geometry(face);
    const auto native = mesh.native_face(face);
    const auto& region = mesh.regions().at(native.region);
    auto result = std::visit(
        [&](const auto& r) -> OrdinaryFacePtr
        {
            if constexpr (requires { r.geometry().certified_face_geometry(native.face); })
                return r.geometry().certified_face_geometry(native.face);
            else
                return {};
        },
        region);
    if (result)
    {
        const auto owner = std::visit([&](const auto& r) { return r.topology().owner_cell(native.face); }, region);
        if (mesh.composite_cell(native.region, owner) != mesh.owner_cell(face))
            result = result->reversed();
        return result;
    }
    if (mesh.region_layout(native.region).family == Meshes::RegionLayout::Family::Cylindrical)
        on::reject("analytic curved faces are not finite planar node hulls.");
    std::vector<GeometryPoint> p;
    for (auto id : mesh.face_nodes(face))
    {
        const auto q = mesh.node_coordinates(id);
        p.push_back({q.x, q.y, q.z});
    }
    const auto n = mesh.face_area_vector(face);
    return make(p)->oriented_like({n.x, n.y, n.z});
}

const std::vector<OrdinaryFace::WidePoint>& OrdinaryFace::points() const
{
    return impl_->polygon;
}
GeometryValue<ArithmeticMode::Ordinary> OrdinaryFace::area() const
{
    return planar_queries::area(impl_->polygon);
}
::SimpleFluid::MeshUtils::Vec3 OrdinaryFace::centroid() const
{
    const auto result = planar_queries::centroid(impl_->polygon);
    return {result[0], result[1], result[2]};
}
::SimpleFluid::MeshUtils::Vec3 OrdinaryFace::area_vector() const
{
    const auto result = planar_queries::area_vector(impl_->polygon);
    return {result[0], result[1], result[2]};
}
GeometryMeasure<ArithmeticMode::Ordinary> OrdinaryFace::coplanar_area_ratio(
    const OrdinaryFace& denominator, std::size_t budget) const
{
    if (storage_bytes() > budget || denominator.storage_bytes() > budget - storage_bytes())
        on::reject("native face measure byte cap exhausted.");
    if (*impl_->support != *denominator.impl_->support && !coplanar(impl_->polygon, denominator.impl_->polygon))
        on::reject("native face area ratio has distinct supports.");
    const auto a = on::polygon_vector(impl_->polygon), b = on::polygon_vector(denominator.impl_->polygon);
    const auto axis = axis_of(b);
    return GeometryMeasure<ArithmeticMode::Ordinary>::from_nominal(a[axis] / b[axis]);
}
GeometryMeasure<ArithmeticMode::Ordinary> OrdinaryFace::area_vector_dot(
    GeometryPoint velocity, std::size_t budget) const
{
    if (storage_bytes() > budget)
        on::reject("native face measure byte cap exhausted.");
    return GeometryMeasure<ArithmeticMode::Ordinary>::from_nominal(
        on::dot(on::polygon_vector(impl_->polygon), on::wide(velocity)));
}
OrdinaryFacePtr OrdinaryFace::reversed() const
{
    auto p = impl_->polygon;
    std::reverse(p.begin(), p.end());
    return from_retained_support(p, impl_->normal, *impl_->support);
}
OrdinaryFacePtr OrdinaryFace::translated(GeometryPoint shift) const
{
    auto p = impl_->polygon;
    const auto delta = on::wide(shift);
    for (auto& v : p)
        v = on::add(v, delta);
    return from_retained_support(p, impl_->normal, on::translate_support(*impl_->support, delta));
}
OrdinaryFacePtr OrdinaryFace::translated(const OrdinaryDisplacement& packet) const
{
    auto p = impl_->polygon;
    const auto shift = IdentityAccess::displacement(packet);
    for (auto& v : p)
        v = on::add(v, shift);
    return from_retained_support(p, impl_->normal, on::translate_support(*impl_->support, shift));
}
OrdinaryFacePtr OrdinaryFace::oriented_like(GeometryPoint direction) const
{
    const auto sign = on::dot(on::polygon_vector(impl_->polygon), on::wide(direction));
    if (sign == 0)
        on::reject("native face orientation is unresolved.");
    return sign < 0 ? reversed() : OrdinaryFacePtr(new OrdinaryFace(impl_));
}
OrdinaryFacePtr OrdinaryFace::intersection(const OrdinaryFace& other) const
{
    const bool common = *impl_->support == *other.impl_->support;
    auto p = intersect_loop(impl_->polygon, other.impl_->polygon, common);
    return p.empty() ? OrdinaryFacePtr{} : from_retained_support(p, impl_->normal, *impl_->support);
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::oriented_like(
    const Pointer& face, GeometryPoint direction, const geometry_detail::NativeConstructionAdmission& admission)
{
    if (!face)
        throw std::invalid_argument("Null native planar orientation input.");
    const auto sign = on::dot(on::polygon_vector(face->points()), on::wide(direction));
    if (sign == 0)
        on::reject("native face orientation is unresolved.");
    return sign < 0 ? reversed(*face, admission) : face;
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::intersection(
    const Face& first, const Face& second, const geometry_detail::NativeConstructionAdmission& admission)
{
    const bool common = *first.impl_->support == *second.impl_->support;
    if (!common && !coplanar(first.points(), second.points()))
        on::reject("native planar intersection has distinct supports.");
    geometry_detail::NativeGeometryScratchVector<on::WidePoint> polygon(
        geometry_detail::NativeGeometryScratchAllocator<on::WidePoint>(admission.state()));
    polygon.assign(first.points().begin(), first.points().end());
    const auto normal = on::polygon_vector(second.points());
    for (std::size_t i = 0; i < second.points().size() && !polygon.empty(); ++i)
        polygon = clip_loop_storage(
            polygon, second.points()[i], second.points()[(i + 1) % second.points().size()], normal, true);
    return polygon.empty() ? Pointer{} : retained(polygon, first.impl_->normal, *first.impl_->support, admission);
}
std::vector<OrdinaryPoint> OrdinaryFace::vertex_identities() const
{
    std::vector<OrdinaryPoint> result;
    for (const auto& p : impl_->polygon)
        result.push_back(IdentityAccess::point(p));
    return result;
}
std::vector<OrdinaryLine> OrdinaryFace::edge_support_identities() const
{
    std::vector<OrdinaryLine> result;
    const auto& p = impl_->polygon;
    for (std::size_t i = 0; i < p.size(); ++i)
        result.push_back(line_identity(p[i], p[(i + 1) % p.size()]));
    return result;
}
OrdinaryFacePtr OrdinaryFace::with_edge_vertices(std::span<const OrdinaryPoint> points) const
{
    Polygon result;
    for (std::size_t i = 0; i < impl_->polygon.size(); ++i)
    {
        const auto& a = impl_->polygon[i];
        const auto& b = impl_->polygon[(i + 1) % impl_->polygon.size()];
        const auto d = on::sub(b, a);
        const auto length = on::dot(d, d);
        std::map<long double, on::WidePoint> interior;
        for (const auto& packet : points)
        {
            const auto p = IdentityAccess::point(packet), v = on::sub(p, a);
            const auto t = on::dot(v, d);
            if (on::cross(v, d) == on::WidePoint{} && t > 0 && t < length)
                interior.emplace(t, p);
        }
        result.push_back(a);
        for (const auto& [t, p] : interior)
            result.push_back(p);
    }
    return from_retained_support(result, impl_->normal, *impl_->support);
}
bool OrdinaryFace::equivalent(const OrdinaryFace& other) const
{
    return contains(other) && other.contains(*this);
}
bool OrdinaryFace::contains_point(const OrdinaryPoint& packet) const
{
    const auto p = IdentityAccess::point(packet);
    if (std::find(impl_->polygon.begin(), impl_->polygon.end(), p) != impl_->polygon.end())
        return true;
    return point_inside(impl_->polygon, p);
}
bool OrdinaryFace::contains(const OrdinaryFace& other) const
{
    return planar_queries::contains(impl_->polygon, other.impl_->polygon, *impl_->support == *other.impl_->support);
}
bool OrdinaryFace::oppositely_oriented(const OrdinaryFace& other) const
{
    return on::dot(on::polygon_vector(impl_->polygon), on::polygon_vector(other.impl_->polygon)) < 0;
}
bool OrdinaryFace::cell_bounds_overlap(const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell) const
{
    const details::OrdinaryExportCoordinates coordinates(cell);
    for (std::size_t a = 0; a < 3; ++a)
    {
        auto lo = impl_->polygon[0][a], hi = lo;
        for (const auto& q : impl_->polygon)
        {
            lo = std::min(lo, q[a]);
            hi = std::max(hi, q[a]);
        }
        // Numeric rebases can round their working coordinates differently.
        // Preserve the canonical export-source bounds without exported points.
        auto clo = coordinates.at(0)[a], chi = clo;
        for (std::size_t vertex = 1; vertex < coordinates.size(); ++vertex)
        {
            const auto value = coordinates.at(vertex)[a];
            clo = std::min(clo, value);
            chi = std::max(chi, value);
        }
        if (hi < clo || chi < lo)
            return false;
    }
    return true;
}
ConvexGeometryPtr<ArithmeticMode::Ordinary> OrdinaryFace::cell_shell(std::span<const OrdinaryFacePtr> faces)
{
    if (faces.size() < 4 || faces.size() > vertex_cap)
        on::reject("native cell face cap exhausted.");
    std::map<on::WidePoint, std::size_t> ids;
    std::vector<Point<ArithmeticMode::Ordinary>> points;
    std::vector<std::vector<std::size_t>> loops;
    for (const auto& face : faces)
    {
        if (!face)
            on::reject("null native cell face.");
        std::vector<std::size_t> loop;
        for (const auto& p : face->points())
        {
            auto [it, inserted] = ids.emplace(p, points.size());
            if (inserted)
            {
                if (points.size() >= vertex_cap)
                    on::reject("native cell vertex cap exhausted.");
                points.emplace_back(p);
            }
            loop.push_back(it->second);
        }
        loops.push_back(std::move(loop));
    }
    std::size_t pair_tests = 0;
    for (auto& face : loops)
    {
        std::vector<std::size_t> propagated;
        for (std::size_t edge = 0; edge < face.size(); ++edge)
        {
            const auto first = face[edge], second = face[(edge + 1) % face.size()];
            const auto a = points[first].value, b = points[second].value, delta = on::sub(b, a);
            const auto axis = axis_of(delta);
            std::map<long double, std::size_t> interior;
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                if (++pair_tests > 262144)
                    on::reject("native cell edge-propagation pair cap exhausted.");
                if (i == first || i == second)
                    continue;
                const auto p = points[i].value;
                if (p[axis] > std::min(a[axis], b[axis]) && p[axis] < std::max(a[axis], b[axis]) &&
                    on::cross(delta, on::sub(p, a)) == on::WidePoint{})
                    interior.emplace(delta[axis] > 0 ? p[axis] : -p[axis], i);
            }
            propagated.push_back(first);
            for (const auto& [coordinate, i] : interior)
                propagated.push_back(i);
        }
        face = std::move(propagated);
    }
    std::vector<OrdinarySupport> supports;
    for (const auto& face : faces)
        supports.push_back(*face->impl_->support);
    return ConvexCellGeometry<ArithmeticMode::Ordinary>::make_points(points, loops)->with_native_supports(supports);
}
void OrdinaryFace::certify_partition(std::span<const OrdinaryFacePtr> tiles) const
{
    certify_partition_impl(tiles, nullptr);
}
void details::NativePlanarOwnerAccess::certify_partition(
    const Face& face, std::span<const Pointer> tiles, const geometry_detail::NativeConstructionAdmission& admission)
{
    face.certify_partition_impl(tiles, &admission);
}
void OrdinaryFace::certify_partition_impl(
    std::span<const OrdinaryFacePtr> tiles, const geometry_detail::NativeConstructionAdmission* admission) const
{
    constexpr std::size_t byte_budget = 64 * 1024 * 1024, pair_cap = 262144, tile_cap = 65536;
    constexpr long double physical_budget = 8e-15L;
    if (tiles.empty() || tiles.size() > tile_cap)
        on::reject("native planar partition tile cap exhausted.");
    std::size_t bytes = 0, pairs = 0;
    GeometryAllocationLedger allocations;
    const auto charge = [&](std::size_t count, std::size_t size)
    {
        if (size && count > (byte_budget - bytes) / size)
            on::reject("native planar partition byte cap exhausted.");
        bytes += count * size;
    };
    const auto pair = [&]
    {
        if (++pairs > pair_cap)
            on::reject("native planar partition pair-test cap exhausted.");
    };
    charge(impl_->polygon.size(), sizeof(on::WidePoint) + 4 * sizeof(void*));
    const auto state = admission ? admission->state() : std::shared_ptr<const geometry_detail::NativeAllocationState>{};
    std::set<on::WidePoint, std::less<on::WidePoint>, geometry_detail::NativeGeometryScratchAllocator<on::WidePoint>>
        vertices(impl_->polygon.begin(), impl_->polygon.end(), std::less<on::WidePoint>{},
            geometry_detail::NativeGeometryScratchAllocator<on::WidePoint>(state));
    for (std::size_t i = 0; i < tiles.size(); ++i)
    {
        if (!tiles[i] || !contains(*tiles[i]))
            on::reject("native planar partition leaves its source.");
        charge(tiles[i]->points().size(), sizeof(on::WidePoint) + 4 * sizeof(void*));
        vertices.insert(tiles[i]->points().begin(), tiles[i]->points().end());
        for (std::size_t j = 0; j < i; ++j)
        {
            pair();
            if (admission ? details::NativePlanarOwnerAccess::intersection(*tiles[i], *tiles[j], *admission)
                          : tiles[i]->intersection(*tiles[j]))
                on::reject("native planar partition interiors overlap.");
        }
    }
    // Compare oriented boundaries after splitting at every admitted represented
    // vertex. No area tolerance can authorize a missing sliver or duplicate edge.
    using Edge = std::pair<on::WidePoint, on::WidePoint>;
    std::map<Edge, std::int64_t, std::less<Edge>,
        geometry_detail::NativeGeometryScratchAllocator<std::pair<const Edge, std::int64_t>>>
        chain(std::less<Edge>{},
            geometry_detail::NativeGeometryScratchAllocator<std::pair<const Edge, std::int64_t>>(state));
    const auto source_normal = on::polygon_vector(impl_->polygon);
    const auto add_boundary = [&](const Polygon& polygon, int contribution)
    {
        const auto normal = on::polygon_vector(polygon);
        if (on::dot(normal, source_normal) < 0)
            contribution = -contribution;
        for (std::size_t e = 0; e < polygon.size(); ++e)
        {
            const auto a = polygon[e], b = polygon[(e + 1) % polygon.size()], delta = on::sub(b, a);
            const auto axis = axis_of(delta);
            charge(vertices.size(), sizeof(on::WidePoint));
            geometry_detail::NativeGeometryScratchVector<on::WidePoint> points{
                geometry_detail::NativeGeometryScratchAllocator<on::WidePoint>(state)};
            for (const auto& point : vertices)
            {
                pair();
                if (point[axis] >= std::min(a[axis], b[axis]) && point[axis] <= std::max(a[axis], b[axis]) &&
                    on::cross(delta, on::sub(point, a)) == on::WidePoint{})
                    points.push_back(point);
            }
            std::sort(points.begin(), points.end(), [axis, &delta](const auto& p, const auto& q)
                { return delta[axis] > 0 ? p[axis] < q[axis] : p[axis] > q[axis]; });
            if (points.size() < 2 || points.front() != a || points.back() != b)
                on::reject("native planar edge incidence is unresolved.");
            for (std::size_t i = 1; i < points.size(); ++i)
            {
                const auto& p = points[i - 1];
                const auto& q = points[i];
                const Edge edge = p < q ? Edge{p, q} : Edge{q, p};
                if (!chain.contains(edge))
                    charge(1, sizeof(Edge) + sizeof(std::int64_t) + 4 * sizeof(void*));
                chain[edge] += p < q ? contribution : -contribution;
            }
        }
    };
    add_boundary(impl_->polygon, -1);
    for (const auto& tile : tiles)
        add_boundary(tile->points(), 1);
    for (const auto& [edge, coefficient] : chain)
        if (coefficient != 0)
            on::reject("native planar partition boundary does not close.");

    struct Sum
    {
        long double value = 0, correction = 0;
        void add(long double term)
        {
            const auto next = value + term;
            correction += std::abs(value) >= std::abs(term) ? (value - next) + term : (term - next) + value;
            value = next;
        }
        long double result() const { return value + correction; }
    };
    const auto origin = impl_->polygon.front();
    const auto axis = axis_of(source_normal);
    const auto metric = norm(source_normal) / std::abs(source_normal[axis]);
    const auto integrate = [&](const Polygon& polygon)
    {
        std::array<Sum, 4> result;
        const auto sign = on::polygon_vector(polygon)[axis] < 0 ? -1.L : 1.L;
        for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
        {
            const auto area = sign *
                              on::cross(on::sub(polygon[i], polygon[0]), on::sub(polygon[i + 1], polygon[0]))[axis] *
                              metric / 2;
            result[0].add(area);
            const auto relative = on::add(
                on::add(on::sub(polygon[0], origin), on::sub(polygon[i], origin)), on::sub(polygon[i + 1], origin));
            for (std::size_t a = 0; a < 3; ++a)
                result[a + 1].add(area * (relative[a] / 3));
        }
        return std::array<long double, 4>{
            result[0].result(), result[1].result(), result[2].result(), result[3].result()};
    };
    const auto expected = integrate(impl_->polygon);
    std::array<Sum, 4> actual;
    for (const auto& tile : tiles)
    {
        const auto measures = integrate(tile->points());
        for (std::size_t a = 0; a < 4; ++a)
            actual[a].add(measures[a]);
    }
    if (std::abs(actual[0].result() - expected[0]) > physical_budget * expected[0])
        on::reject("native planar partition does not conserve area.");
    for (std::size_t a = 0; a < 3; ++a)
    {
        long double extent = 0;
        for (const auto& p : impl_->polygon)
            extent = std::max(extent, std::abs(p[a] - origin[a]));
        if (std::abs(actual[a + 1].result() - expected[a + 1]) > physical_budget * expected[0] * extent)
            on::reject("native planar partition does not conserve its local first moment.");
    }
}

namespace
{
template<class Writer>
void emit_face_key(Writer& w, const OrdinaryFace& face, const on::WidePoint& normal, bool incidence,
    const geometry_detail::NativeConstructionAdmission* admission = nullptr)
{
    // Each concrete polygon is the only dynamic scratch in this traversal.
    // Its receipt precedes the vector, hence retires after the real buffer.
    const auto scratch = admission ? geometry_detail::native_constructor_storage(*admission)
                                   : geometry_detail::NativeConstructorStorageOwner{};
    Polygon polygon;
    if (!incidence)
    {
        if (scratch.owner)
            scratch.owner->copy(polygon, std::span(face.points()));
        else
            polygon = face.points();
    }
    if (!incidence)
    {
        bool changed = true;
        while (changed && polygon.size() > 3)
        {
            changed = false;
            for (std::size_t i = 0; i < polygon.size(); ++i)
            {
                const auto previous = (i + polygon.size() - 1) % polygon.size();
                const auto next = (i + 1) % polygon.size();
                if (on::cross(on::sub(polygon[i], polygon[previous]), on::sub(polygon[next], polygon[i])) ==
                    on::WidePoint{})
                {
                    polygon.erase(polygon.begin() + i);
                    changed = true;
                    break;
                }
            }
        }
        std::rotate(polygon.begin(), std::min_element(polygon.begin(), polygon.end()), polygon.end());
    }
    const auto& points = incidence ? face.points() : polygon;
    const auto support = planar_support<ArithmeticMode::Ordinary>(face);
    w.tag(GeometryTag::OrdinaryPlanarFace);
    w.word(canonical_key_size(support));
    w.identity(support);
    // The retained normal is not a derived area vector; keep its exact codec
    // representation, including for incidence packets.
    w.point(normal);
    w.word(points.size());
    for (const auto& p : points)
        w.point(p);
}
std::size_t face_packet_size(std::size_t body)
{
    if (geometry_packet_profile(ArithmeticMode::Ordinary) == GeometryNumericProfile::Unsupported ||
        body > geometry_packet_default_budget - geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet exceeds its kind or byte budget.");
    return on::encoding_add(geometry_packet_header_bytes, body);
}
} // namespace

template<> std::vector<char> canonical_face_key<ArithmeticMode::Ordinary>(const OrdinaryFace& face)
{
    on::Writer w;
    emit_face_key(w, face, face.impl_->normal, false);
    return std::move(w.bytes);
}
std::vector<char> OrdinaryFace::encode(std::size_t byte_budget) const
{
    return encode_geometry_packet(
        GeometryPacketKind::PlanarFace, ArithmeticMode::Ordinary, canonical_face_key(*this), byte_budget);
}
std::vector<char> OrdinaryFace::encode_incidence(std::size_t byte_budget) const
{
    on::Writer w;
    emit_face_key(w, *this, impl_->normal, true);
    return encode_geometry_packet(
        GeometryPacketKind::PlanarFaceIncidence, ArithmeticMode::Ordinary, w.bytes, byte_budget);
}
std::vector<char> details::NativePlanarCodecAccess::canonical(
    const OrdinaryFace& face, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!storage.owner)
        throw std::invalid_argument("Native canonical face key needs its concrete storage owner.");
    const auto& admission = storage.owner->admission();
    const auto frame = capture_incidence_frame(face);
    on::CountingWriter count;
    emit_face_key(count, face, frame.normal, false, &admission);
    std::vector<char> result;
    storage.owner->reserve(result, count.bytes.size());
    on::BasicWriter<details::NativeGeometryByteWriter> writer{{&result, &admission}};
    emit_face_key(writer, face, frame.normal, false, &admission);
    if (result.size() != count.bytes.size())
        throw std::invalid_argument("Native canonical face length differs.");
    return result;
}
std::size_t details::NativePlanarCodecAccess::size(
    const OrdinaryFace& face, bool incidence, const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto frame = capture_incidence_frame(face);
    on::CountingWriter count;
    emit_face_key(count, face, frame.normal, incidence, &admission);
    return face_packet_size(count.bytes.size());
}
void details::NativePlanarCodecAccess::append(std::vector<char>& bytes, const OrdinaryFace& face, bool incidence,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto extent = size(face, incidence, admission);
    if (extent > bytes.capacity() - bytes.size())
        throw std::invalid_argument("Native planar destination was not preadmitted.");
    const auto initial = bytes.size();
    const auto frame = capture_incidence_frame(face);
    const auto header = details::geometry_packet_header(
        incidence ? GeometryPacketKind::PlanarFaceIncidence : GeometryPacketKind::PlanarFace, ArithmeticMode::Ordinary,
        extent - geometry_packet_header_bytes, geometry_packet_default_budget);
    bytes.insert(bytes.end(), header.begin(), header.end());
    on::BasicWriter<details::NativeGeometryByteWriter> writer{{&bytes, &admission}};
    emit_face_key(writer, face, frame.normal, incidence, &admission);
    if (bytes.size() - initial != extent)
        throw std::invalid_argument("Native planar packet length differs from its exact count.");
}
std::vector<char> details::NativePlanarCodecAccess::encode(
    const OrdinaryFace& face, bool incidence, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!storage.owner)
        throw std::invalid_argument("Native planar encoding needs its concrete storage owner.");
    const auto& admission = storage.owner->admission();
    std::vector<char> bytes;
    storage.owner->reserve(bytes, size(face, incidence, admission));
    append(bytes, face, incidence, admission);
    return bytes;
}
std::size_t OrdinaryFace::encoding_scratch_bytes() const
{
    on::require_encoding_capacity_profile();
    return on::encoding_product(impl_->polygon.size(), sizeof(WidePoint));
}
std::size_t OrdinaryFace::encoded_size(std::size_t scratch_budget) const
{
    if (encoding_scratch_bytes() > scratch_budget)
        throw std::invalid_argument("Planar encoding scratch exceeds its byte budget.");
    on::CountingWriter w;
    emit_face_key(w, *this, impl_->normal, false);
    return face_packet_size(w.bytes.size());
}
std::size_t OrdinaryFace::encoded_incidence_size() const
{
    on::require_encoding_capacity_profile();
    on::CountingWriter w;
    emit_face_key(w, *this, impl_->normal, true);
    return face_packet_size(w.bytes.size());
}
std::size_t on::fallback_face_allocation_bound(std::size_t points)
{
    // Sum all explicitly owned buffers, even where their lifetimes only partly
    // overlap: face_geometry loop, support source, retained polygon, unique set.
    const auto polygon = encoding_product(points, sizeof(WidePoint));
    const auto unique = encoding_product(points, sizeof(WidePoint) + 4 * sizeof(void*));
    return encoding_add(encoding_add(encoding_add(polygon, source_support_allocation_bound(points)),
                            sizeof(OrdinaryFace::Impl) + sizeof(OrdinaryFace)),
        encoding_add(polygon, unique));
}
std::size_t OrdinaryFace::construction_capacity_bound(std::size_t vertices)
{
    on::require_encoding_capacity_profile();
    return on::fallback_face_allocation_bound(vertices);
}

std::size_t OrdinaryFace::exclusive_storage_bytes() const
{
    return sizeof(Impl) + impl_->polygon.capacity() * sizeof(WidePoint);
}
std::size_t OrdinaryFace::storage_bytes() const
{
    auto bytes = exclusive_storage_bytes();
    visit_identity_allocations(*this,
        [&](const GeometryIdentityAllocation& allocation)
        {
            bytes += allocation.payload_bytes;
            return true;
        });
    return bytes;
}
OrdinaryFacePtr OrdinaryFace::decode(std::span<const char> bytes, std::size_t byte_budget)
{
    const auto envelope = decode_geometry_packet(bytes, ArithmeticMode::Ordinary, byte_budget);
    if (envelope.kind != GeometryPacketKind::PlanarFace && envelope.kind != GeometryPacketKind::PlanarFaceIncidence)
        throw std::invalid_argument("Foreign planar face packet kind.");
    on::Reader r{envelope.body, 0, true};
    if (r.tag() != GeometryTag::OrdinaryPlanarFace)
        on::reject("foreign native planar packet mode.");
    const auto encoded_support = r.packet();
    auto shape = r;
    (void) shape.point();
    const auto points = shape.count(vertex_cap);
    if (points > (shape.bytes.size() - shape.cursor) / 48)
        on::reject("planar point count exceeds the remaining encoded coordinates.");
    std::size_t native_used = sizeof(OrdinaryFace) + sizeof(Impl);
    if (native_used > byte_budget || points > (byte_budget - native_used) / sizeof(WidePoint))
        on::reject("planar polygon exceeds the native record budget.");
    native_used += points * sizeof(WidePoint);
    if (encoded_support.size() > byte_budget - native_used)
        on::reject("planar support exceeds the remaining native record budget.");
    auto support = IdentityAccess::decode_key<OrdinarySupport>(
        details::checked_identity_key(GeometryIdentityKind::PlaneSupport, ArithmeticMode::Ordinary, encoded_support),
        byte_budget - native_used);
    GeometryAllocationLedger allocations;
    allocations.retain(support, native_used, byte_budget);
    const auto normal = r.point();
    const auto n = r.count(vertex_cap);
    if (n > (r.bytes.size() - r.cursor) / 48)
        on::reject("planar point count exceeds the remaining encoded coordinates.");
    Polygon p;
    p.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        p.push_back(r.point());
    r.finish();
    auto result = from_retained_support(p, normal, std::move(support));
    const auto actual = sizeof(OrdinaryFace) + result->exclusive_storage_bytes();
    const auto reserved = sizeof(OrdinaryFace) + sizeof(Impl) + points * sizeof(WidePoint);
    if (actual > reserved && actual - reserved > byte_budget - native_used)
        on::reject("planar polygon capacity exceeds the native record budget.");
    const auto canonical = envelope.kind == GeometryPacketKind::PlanarFace ? result->encode(byte_budget)
                                                                           : result->encode_incidence(byte_budget);
    if (canonical.size() != bytes.size() || !std::equal(canonical.begin(), canonical.end(), bytes.begin()))
        throw std::invalid_argument("Noncanonical planar face packet.");
    return result;
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::decode(
    std::span<const char> bytes, std::size_t byte_budget, const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto envelope = decode_geometry_packet(bytes, ArithmeticMode::Ordinary, byte_budget);
    if (envelope.kind != GeometryPacketKind::PlanarFace && envelope.kind != GeometryPacketKind::PlanarFaceIncidence)
        throw std::invalid_argument("Foreign planar face packet kind.");
    on::Reader reader{envelope.body, 0, true};
    if (reader.tag() != GeometryTag::OrdinaryPlanarFace)
        on::reject("foreign native planar packet mode.");
    const auto encoded_support = reader.packet();
    auto shape = reader;
    (void) shape.point();
    const auto points = shape.count(vertex_cap);
    if (points > (shape.bytes.size() - shape.cursor) / 48)
        on::reject("planar point count exceeds the remaining encoded coordinates.");
    std::size_t native_used = sizeof(OrdinaryFace) + sizeof(Face::Impl);
    if (native_used > byte_budget || points > (byte_budget - native_used) / sizeof(Face::WidePoint))
        on::reject("planar polygon exceeds the native record budget.");
    native_used += points * sizeof(Face::WidePoint);
    if (encoded_support.size() > byte_budget - native_used)
        on::reject("planar support exceeds the remaining native record budget.");
    auto support = IdentityAccess::decode_support_admitted(encoded_support, byte_budget - native_used, admission);
    // Keep the legacy numerical ledger's logical payload/metadata gate on a
    // separate monotone cursor; its actual administration uses the same domain.
    GeometryAllocationLedger allocations;
    details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(allocations, admission, native_used, byte_budget);
    allocations.retain(support, native_used, byte_budget);
    const auto normal = reader.point();
    const auto count = reader.count(vertex_cap);
    if (count > (reader.bytes.size() - reader.cursor) / 48)
        on::reject("planar point count exceeds the remaining encoded coordinates.");
    const auto scratch = geometry_detail::native_constructor_storage(admission);
    Polygon polygon;
    scratch.owner->reserve(polygon, count);
    for (std::size_t i = 0; i < count; ++i)
        polygon.push_back(reader.point());
    reader.finish();
    auto result = retained(polygon, normal, std::move(support), admission);
    const auto actual = sizeof(OrdinaryFace) + result->exclusive_storage_bytes();
    const auto reserved = sizeof(OrdinaryFace) + sizeof(Face::Impl) + points * sizeof(Face::WidePoint);
    if (actual > reserved && actual - reserved > byte_budget - native_used)
        on::reject("planar polygon capacity exceeds the native record budget.");
    const auto canonical = details::NativePlanarCodecAccess::encode(
        *result, envelope.kind == GeometryPacketKind::PlanarFaceIncidence, scratch);
    if (canonical.size() != bytes.size() || !std::equal(canonical.begin(), canonical.end(), bytes.begin()))
        throw std::invalid_argument("Noncanonical planar face packet.");
    return result;
}
template<> OrdinarySupport planar_support<ArithmeticMode::Ordinary>(const OrdinaryFace& face)
{
    return *face.impl_->support;
}
template<> int planar_orientation<ArithmeticMode::Ordinary>(const OrdinaryFace& face)
{
    return planar_queries::orientation(face.points());
}
template<>
std::vector<OrdinaryFacePtr> subtract_planar_faces<ArithmeticMode::Ordinary>(
    const OrdinaryFace& source, std::span<const OrdinaryFacePtr> removed)
{
    std::vector<Polygon> remains{source.points()};
    for (const auto& face : removed)
    {
        if (!face || (*source.impl_->support != *face->impl_->support && !coplanar(source.points(), face->points())))
            on::reject("native face subtraction has foreign support.");
        std::vector<Polygon> next;
        const auto& clip = face->points();
        const auto n = on::polygon_vector(clip);
        for (auto poly : remains)
        {
            for (std::size_t e = 0; e < clip.size() && !poly.empty(); ++e)
            {
                auto outside = clip_loop(poly, clip[e], clip[(e + 1) % clip.size()], n, false);
                if (!outside.empty())
                    next.push_back(std::move(outside));
                poly = clip_loop(poly, clip[e], clip[(e + 1) % clip.size()], n, true);
            }
        }
        remains = std::move(next);
    }
    std::vector<OrdinaryFacePtr> result;
    for (const auto& p : remains)
        result.push_back(OrdinaryFace::from_retained_support(p, source.impl_->normal, *source.impl_->support));
    return result;
}
geometry_detail::NativeGeometryScratchVector<OrdinaryFacePtr> details::NativePlanarOwnerAccess::subtract(
    const Face& source, std::span<const Pointer> removed, const geometry_detail::NativeConstructionAdmission& admission)
{
    using geometry_detail::NativeGeometryScratchAllocator;
    using geometry_detail::NativeGeometryScratchVector;
    using Loop = NativeGeometryScratchVector<on::WidePoint>;
    const auto& state = admission.state();
    NativeGeometryScratchVector<Loop> remains{NativeGeometryScratchAllocator<Loop>(state)};
    remains.emplace_back(source.points().begin(), source.points().end(), NativeGeometryScratchAllocator<on::WidePoint>(state));
    for (const auto& face : removed)
    {
        if (!face || (*source.impl_->support != *face->impl_->support && !coplanar(source.points(), face->points())))
            on::reject("native face subtraction has foreign support.");
        NativeGeometryScratchVector<Loop> next{NativeGeometryScratchAllocator<Loop>(state)};
        const auto& clip = face->points();
        const auto normal = on::polygon_vector(clip);
        for (auto& poly : remains)
            for (std::size_t edge = 0; edge < clip.size() && !poly.empty(); ++edge)
            {
                const auto& from = clip[edge];
                const auto& to = clip[(edge + 1) % clip.size()];
                auto outside = clip_loop_storage(poly, from, to, normal, false);
                if (!outside.empty())
                    next.push_back(std::move(outside));
                poly = clip_loop_storage(poly, from, to, normal, true);
            }
        remains = std::move(next);
    }
    NativeGeometryScratchVector<Pointer> result{NativeGeometryScratchAllocator<Pointer>(state)};
    result.reserve(remains.size());
    for (const auto& loop : remains)
        result.push_back(retained(loop, source.impl_->normal, *source.impl_->support, admission));
    return result;
}
EdgePointIndex<ArithmeticMode::Ordinary>::EdgePointIndex(std::span<const OrdinaryPoint> points)
{
    auto impl = std::make_shared<Impl>();
    for (const auto& p : points)
        impl->points.emplace(IdentityAccess::point(p), p);
    impl_ = std::move(impl);
}
OrdinaryEdgeQueryScratch::OrdinaryEdgeQueryScratch() = default;
OrdinaryEdgeQueryScratch::~OrdinaryEdgeQueryScratch() = default;
OrdinaryEdgeQueryScratch::OrdinaryEdgeQueryScratch(OrdinaryEdgeQueryScratch&&) noexcept = default;
OrdinaryEdgeQueryScratch& OrdinaryEdgeQueryScratch::operator=(OrdinaryEdgeQueryScratch&&) noexcept = default;
std::size_t OrdinaryEdgeQueryScratch::reserve_required_peak(std::size_t count) const
{
    if (count <= capacity())
        return storage_bytes();
    const auto old = storage_bytes();
    const auto bookkeeping = impl_ ? 0 : sizeof(Impl);
    if (count > (std::numeric_limits<std::size_t>::max() - old - bookkeeping) / sizeof(Impl::Hit))
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary edge-query capacity overflow.");
    return old + bookkeeping + count * sizeof(Impl::Hit);
}
void OrdinaryEdgeQueryScratch::reserve(std::size_t count, std::size_t byte_cap)
{
    if (reserve_required_peak(count) > byte_cap)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary edge-query byte cap exhausted.");
    if (count <= capacity())
        return;
    // Admission includes the old allocation until replacement has succeeded.
    auto hits = std::make_unique<Impl::Hit[]>(count);
    if (!impl_)
    {
        auto next = std::make_unique<Impl>();
        next->hits = std::move(hits);
        next->capacity = count;
        impl_ = std::move(next);
    }
    else
    {
        std::copy_n(impl_->hits.get(), impl_->size, hits.get());
        impl_->hits = std::move(hits);
        impl_->capacity = count;
    }
    ++growths_;
}
std::size_t OrdinaryEdgeQueryScratch::capacity() const
{
    return impl_ ? impl_->capacity : 0;
}
std::size_t OrdinaryEdgeQueryScratch::storage_bytes() const
{
    return impl_ ? sizeof(Impl) + impl_->capacity * sizeof(Impl::Hit) : 0;
}
std::size_t OrdinaryEdgeQueryScratch::size() const
{
    return impl_ ? impl_->size : 0;
}
const OrdinaryPoint& OrdinaryEdgeQueryScratch::point(std::size_t index) const
{
    if (index >= size())
        throw std::out_of_range("Ordinary edge-query point index.");
    return *impl_->hits[index].identity;
}
void OrdinaryEdgeQueryScratch::clear() noexcept
{
    if (impl_)
    {
        impl_->size = 0;
        impl_->index.reset();
    }
}
void OrdinaryEdgeQueryScratch::release() noexcept
{
    impl_.reset();
}
std::size_t OrdinaryEdgeQueryScratch::growths() const
{
    return growths_;
}
std::size_t OrdinaryEdgeQueryScratch::queries() const
{
    return queries_;
}
std::size_t OrdinaryEdgeQueryScratch::scanned_points() const
{
    return scanned_points_;
}
void EdgePointIndex<ArithmeticMode::Ordinary>::closed_segment(
    const OrdinaryPoint& first, const OrdinaryPoint& second, OrdinaryEdgeQueryScratch& scratch) const
{
    scratch.clear();
    ++scratch.queries_;
    try
    {
        const auto a = IdentityAccess::point(first), b = IdentityAccess::point(second), d = on::sub(b, a);
        const auto length = on::dot(d, d);
        if (length == 0)
            on::reject("zero finite source edge.");
        if (scratch.capacity() < impl_->points.size())
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary edge-query capacity exhausted.");
        if (impl_->points.empty())
            return;
        auto& work = *scratch.impl_;
        work.index = impl_;
        std::size_t ordinal = 0;
        for (const auto& [p, id] : impl_->points)
        {
            ++scratch.scanned_points_;
            const auto v = on::sub(p, a);
            const auto t = on::dot(v, d);
            if (on::cross(v, d) == on::WidePoint{} && t >= 0 && t <= length)
                work.hits[work.size++] = {t, ordinal, &id};
            ++ordinal;
        }
        std::sort(work.hits.get(), work.hits.get() + work.size, [](const auto& a, const auto& b)
            { return a.projection < b.projection || (a.projection == b.projection && a.ordinal < b.ordinal); });
        std::size_t retained = 0;
        for (std::size_t i = 0; i < work.size; ++i)
            if (retained == 0 || work.hits[i].projection != work.hits[retained - 1].projection)
                work.hits[retained++] = work.hits[i];
        work.size = retained;
    }
    catch (...)
    {
        scratch.clear();
        throw;
    }
}
std::vector<OrdinaryPoint> EdgePointIndex<ArithmeticMode::Ordinary>::closed_segment(
    const OrdinaryPoint& first, const OrdinaryPoint& second) const
{
    const auto a = IdentityAccess::point(first), b = IdentityAccess::point(second), d = on::sub(b, a);
    const auto length = on::dot(d, d);
    if (length == 0)
        on::reject("zero finite source edge.");
    std::map<long double, OrdinaryPoint> result;
    for (const auto& [p, id] : impl_->points)
    {
        const auto v = on::sub(p, a);
        const auto t = on::dot(v, d);
        if (on::cross(v, d) == on::WidePoint{} && t >= 0 && t <= length)
            result.emplace(t, id);
    }
    std::vector<OrdinaryPoint> output;
    for (auto& [t, id] : result)
        output.push_back(std::move(id));
    return output;
}
std::size_t EdgePointIndex<ArithmeticMode::Ordinary>::storage_bytes() const
{
    std::size_t size = sizeof(Impl);
    for (const auto& [p, id] : impl_->points)
        size += sizeof(p) + sizeof(id) + id.payload_bytes() + 4 * sizeof(void*);
    return size;
}
template<> OrdinaryDisplacement displacement<ArithmeticMode::Ordinary>(GeometryPoint from, GeometryPoint to)
{
    return IdentityAccess::displacement(on::sub(on::wide(to), on::wide(from)));
}
template<>
OrdinaryDisplacement displacement_difference<ArithmeticMode::Ordinary>(
    const OrdinaryDisplacement& a, const OrdinaryDisplacement& b)
{
    return IdentityAccess::displacement(on::sub(IdentityAccess::displacement(a), IdentityAccess::displacement(b)));
}
template<> GeometryPoint displacement_display<ArithmeticMode::Ordinary>(const OrdinaryDisplacement& p)
{
    return on::display(IdentityAccess::displacement(p));
}
template<>
OrdinaryPoint point_translated<ArithmeticMode::Ordinary>(const OrdinaryPoint& p, const OrdinaryDisplacement& d)
{
    return IdentityAccess::point(on::add(IdentityAccess::point(p), IdentityAccess::displacement(d)));
}
template<> GeometryPoint point_display<ArithmeticMode::Ordinary>(const OrdinaryPoint& p)
{
    return on::display(IdentityAccess::point(p));
}
template<>
std::array<GeometryValue<ArithmeticMode::Ordinary>, 3> numerical_point<ArithmeticMode::Ordinary>(const OrdinaryPoint& p)
{
    const auto v = IdentityAccess::point(p);
    return {{{v[0]}, {v[1]}, {v[2]}}};
}
template<>
GeometryMeasure<ArithmeticMode::Ordinary> point_projection<ArithmeticMode::Ordinary>(
    const OrdinaryPoint& p, GeometryPoint n, GeometryPoint anchor)
{
    return GeometryMeasure<ArithmeticMode::Ordinary>::from_nominal(
        on::dot(on::wide(n), on::sub(IdentityAccess::point(p), on::wide(anchor))));
}
template<>
OrdinaryPoint periodic_point<ArithmeticMode::Ordinary>(
    const OrdinaryPoint& identity, GeometryPoint lower, GeometryPoint upper, std::array<bool, 3> periodic)
{
    auto p = IdentityAccess::point(identity);
    for (std::size_t a = 0; a < 3; ++a)
        if (periodic[a])
        {
            const auto width = static_cast<long double>(upper[a]) - lower[a];
            if (!(width > 0))
                on::reject("invalid finite periodic span.");
            p[a] -= std::floor((p[a] - lower[a]) / width) * width;
            if (!(p[a] >= lower[a] && p[a] < upper[a]))
                on::reject("finite periodic image outside canonical interval.");
        }
    return IdentityAccess::point(p);
}
namespace details::ordinary_native
{
void certify_cell_partition(const ConvexCellGeometry<ArithmeticMode::Ordinary>& parent,
    std::span<const std::shared_ptr<const ConvexCellGeometry<ArithmeticMode::Ordinary>>> pieces, double budget)
{
    if (!std::isfinite(budget) || budget < 0 || pieces.empty() || pieces.size() > 65536)
        reject("invalid finite cell partition request.");
    constexpr std::size_t byte_cap = 64 * 1024 * 1024, pair_cap = 262144;
    std::size_t bytes = 0, pairs = 0;
    using Profile = details::ordinary_kernel::NativeSplitCppProfile;
    std::optional<geometry_detail::NativeConstructionAdmission> admission;
    if (Profile::active())
        admission.emplace(Profile::admission());
    GeometryAllocationLedger allocations;
    if (admission)
        details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(allocations, *admission, bytes, byte_cap);
    const auto charge = [&](std::size_t n, std::size_t size)
    {
        if (size && n > (byte_cap - bytes) / size)
            reject("native cell partition byte cap exhausted.");
        bytes += n * size;
    };
    using FacePtr = PlanarFacePtr<ArithmeticMode::Ordinary>;
    using Edge = std::pair<WidePoint, WidePoint>;
    std::map<OrdinarySupport, std::vector<std::pair<FacePtr, int>>> groups;
    const auto add_cell = [&](const auto& cell, int sign)
    {
        for (std::size_t f = 0; f < cell.num_faces(); ++f)
        {
            auto face = admission ? details::NativePlanarOwnerAccess::cell_face(cell, f, *admission)
                                  : cell.face_geometry(f);
            auto support = planar_support<ArithmeticMode::Ordinary>(*face);
            charge(1, sizeof(std::pair<FacePtr, int>) + 4 * sizeof(void*));
            allocations.retain(support, bytes, byte_cap);
            auto group = groups.find(support);
            if (group == groups.end())
            {
                // Support identity records construction provenance. Independently
                // built parent/child faces can prove the same represented plane
                // with different source recipes. Reuse the exact predicate used
                // by planar partition/subtraction; no metric tolerance merges
                // planes, and every represented vertex test consumes the cap.
                for (auto candidate = groups.begin(); candidate != groups.end(); ++candidate)
                {
                    if (face->points().size() > pair_cap - pairs)
                        reject("native cell partition pair-test cap exhausted.");
                    pairs += face->points().size();
                    if (coplanar(candidate->second.front().first->points(), face->points()))
                    {
                        group = candidate;
                        break;
                    }
                }
                if (group == groups.end())
                {
                    Profile::node<decltype(groups)::value_type>();
                    group = groups.try_emplace(std::move(support)).first;
                }
            }
            Profile::next(group->second);
            group->second.emplace_back(std::move(face), sign);
        }
    };
    add_cell(parent, -1);
    for (const auto& piece : pieces)
    {
        if (!piece || !(piece->volume().value > 0))
            reject("nonpositive native partition piece.");
        add_cell(*piece, 1);
    }
    // Every admitted cell is an outward convex closed shell. Equality of the
    // oriented boundary chains therefore proves equality of their indicator
    // sums: overlaps and gaps cannot cancel merely by equal scalar volumes.
    for (const auto& [support, faces] : groups)
    {
        std::set<WidePoint> vertices;
        for (const auto& [face, sign] : faces)
        {
            charge(face->points().size(), sizeof(WidePoint) + 4 * sizeof(void*));
            if (Profile::active())
                for (const auto& point : face->points())
                {
                    if (!vertices.contains(point))
                        Profile::node<WidePoint>();
                    vertices.insert(point);
                }
            else
                vertices.insert(face->points().begin(), face->points().end());
        }
        std::map<Edge, std::int64_t> chain;
        for (const auto& [face, sign] : faces)
        {
            const auto& polygon = face->points();
            for (std::size_t e = 0; e < polygon.size(); ++e)
            {
                const auto a = polygon[e], b = polygon[(e + 1) % polygon.size()], delta = sub(b, a);
                std::size_t axis = 0;
                for (std::size_t d = 1; d < 3; ++d)
                    if (std::abs(delta[d]) > std::abs(delta[axis]))
                        axis = d;
                charge(vertices.size(), sizeof(WidePoint));
                std::vector<WidePoint> edge_points;
                if (Profile::active())
                    Profile::reserve(edge_points, vertices.size());
                for (const auto& p : vertices)
                {
                    if (++pairs > pair_cap)
                        reject("native cell partition pair-test cap exhausted.");
                    if (p[axis] >= std::min(a[axis], b[axis]) && p[axis] <= std::max(a[axis], b[axis]) &&
                        cross(delta, sub(p, a)) == WidePoint{})
                        edge_points.push_back(p);
                }
                std::sort(edge_points.begin(), edge_points.end(), [axis, &delta](const auto& p, const auto& q)
                    { return delta[axis] > 0 ? p[axis] < q[axis] : p[axis] > q[axis]; });
                if (edge_points.size() < 2 || edge_points.front() != a || edge_points.back() != b)
                    reject("native cell partition edge incidence is unresolved.");
                for (std::size_t i = 1; i < edge_points.size(); ++i)
                {
                    const auto& p = edge_points[i - 1];
                    const auto& q = edge_points[i];
                    const Edge edge = p < q ? Edge{p, q} : Edge{q, p};
                    if (!chain.contains(edge))
                    {
                        charge(1, sizeof(Edge) + sizeof(std::int64_t) + 4 * sizeof(void*));
                        Profile::node<decltype(chain)::value_type>();
                    }
                    chain[edge] += p < q ? sign : -sign;
                }
            }
        }
        for (const auto& [edge, coefficient] : chain)
            if (coefficient)
                reject("native cell partition has an unmatched oriented boundary.");
    }
    const auto expected = parent.moments(parent.origin());
    std::array<long double, 4> sums{}, corrections{};
    const auto accumulate = [&](std::size_t axis, long double value)
    {
        const auto next = sums[axis] + value;
        corrections[axis] +=
            std::abs(sums[axis]) >= std::abs(value) ? (sums[axis] - next) + value : (value - next) + sums[axis];
        sums[axis] = next;
    };
    for (const auto& piece : pieces)
    {
        const auto value = piece->moments(expected.origin);
        accumulate(0, value.volume.value);
        for (std::size_t axis = 0; axis < 3; ++axis)
            accumulate(axis + 1, value.first_moment[axis].value);
    }
    if (std::abs(sums[0] + corrections[0] - expected.volume.value) > budget * expected.volume.value)
        reject("native cell partition does not conserve volume.");
    const auto spans = parent.axis_spans();
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (std::abs(sums[axis + 1] + corrections[axis + 1] - expected.first_moment[axis].value) >
            budget * expected.volume.value * spans[axis].value)
            reject("native cell partition does not conserve its local first moment.");
}
} // namespace details::ordinary_native
} // namespace SimpleFluid::PLIC
