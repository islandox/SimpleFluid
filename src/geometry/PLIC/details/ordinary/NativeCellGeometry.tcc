/** @file NativeCellGeometry.tcc
 * @brief Native finite cell import, refinement, and source-coordinate packets.
 */
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/LocalCoordinates.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/PLIC/details/NativeBoxSourceAccess.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/PLIC/details/NativeGeometryByteWriter.hh"
#include "geometry/PLIC/details/GeometryPacketHeader.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/CanonicalExportCoordinates.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid::PLIC
{
namespace native_cell_detail
{
namespace on = details::ordinary_native;
NativeConvexShell canonical_shell(
    std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& faces)
{
    if (vertices.size() < 4 || vertices.size() > 4096 || faces.size() < 4 || faces.size() > 4096)
        on::reject("invalid native cell shell size.");
    NativeConvexShell result;
    result.vertices.assign(vertices.begin(), vertices.end());
    for (const auto& face : faces)
    {
        if (face.size() < 3)
            on::reject("invalid native face loop.");
        std::vector<std::size_t> loop;
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto a = face[i], b = face[(i + 1) % face.size()];
            if (a >= vertices.size() || b >= vertices.size() || a == b)
                on::reject("invalid native face vertex.");
            const auto p = on::wide(vertices[a]), d = on::sub(on::wide(vertices[b]), p);
            const auto length = on::dot(d, d);
            std::map<long double, std::size_t> interior;
            for (std::size_t j = 0; j < vertices.size(); ++j)
                if (j != a && j != b)
                {
                    const auto q = on::sub(on::wide(vertices[j]), p);
                    const auto along = on::dot(d, q);
                    if (on::cross(d, q) == on::WidePoint{} && along > 0 && along < length)
                        interior.emplace(along, j);
                }
            loop.push_back(a);
            for (const auto& [t, j] : interior)
                loop.push_back(j);
        }
        result.outward_faces.push_back(std::move(loop));
    }
    return result;
}
} // namespace native_cell_detail
template<ArithmeticMode Mode>
NativeConvexShell ConvexCellGeometry<Mode>::native_cell_shell(const MultiRegionMesh& mesh, std::uint64_t cell)
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    const auto execution = mesh.acquire_execution_view();
    if (mesh.sparse_native() || mesh.has_construction())
        on::reject("retained native geometry requires its cell interface, not display-node shell export.");
    if (mesh.region_layout(mesh.native_cell(cell).first).family == Meshes::RegionLayout::Family::Cylindrical)
        on::reject("analytic cylindrical cells are not planar native shells.");
    for (const auto& interface : mesh.interfaces())
        std::visit(
            [](const auto& join)
            {
                if constexpr (requires { join.periodic_translation; })
                    if (join.periodic_translation)
                        on::reject("native shell extraction requires nontranslated joins.");
            },
            interface);
    NativeConvexShell shell;
    std::map<GeometryPoint, std::size_t> ids;
    for (const auto face_id : mesh.cell_faces(cell))
    {
        if (mesh.region_layout(mesh.native_face(face_id).region).family == Meshes::RegionLayout::Family::Cylindrical)
            on::reject("analytic cylindrical faces cannot be replaced by planar node loops.");
        std::vector<std::size_t> face;
        std::vector<on::WidePoint> polygon;
        for (const auto node : mesh.face_nodes(face_id))
        {
            const auto native = mesh.node_coordinates(node);
            const GeometryPoint p{native.x, native.y, native.z};
            const auto [it, inserted] = ids.emplace(p, shell.vertices.size());
            if (inserted)
                shell.vertices.push_back(p);
            face.push_back(it->second);
            polygon.push_back(on::wide(p));
        }
        if (face.size() < 3)
            on::reject("native cell has an invalid face loop.");
        const auto outward = mesh.face_area_vector_outward(face_id, cell);
        const auto orientation = on::dot(on::polygon_vector(polygon), on::wide({outward.x, outward.y, outward.z}));
        if (orientation == 0)
            on::reject("native face orientation is unresolved.");
        if (orientation < 0)
            std::reverse(face.begin(), face.end());
        shell.outward_faces.push_back(std::move(face));
    }
    return native_cell_detail::canonical_shell(shell.vertices, shell.outward_faces);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::from_native_cell(const MultiRegionMesh& mesh, std::uint64_t cell)
    requires(Mode == ArithmeticMode::Ordinary)
{
    (void) mesh.geometry_epoch();
    if (mesh.sparse_native())
        return mesh.certified_sparse_provider_cell_geometry(cell);
    if (!mesh.interfaces().empty())
        return from_native_cell_atomic(mesh, cell);
    const auto [region, local] = mesh.native_cell(cell);
    auto native = std::visit(
        [&](const auto& r) -> ConvexGeometryPtr<Mode>
        {
            if constexpr (requires { r.geometry().certified_cell_geometry(local); })
                return r.geometry().certified_cell_geometry(local);
            else
                return {};
        },
        mesh.regions().at(region));
    if (native)
        return native;
    const auto shell = native_cell_shell(mesh, cell);
    return make(shell.vertices, shell.outward_faces);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::from_native_cell_atomic(
    const MultiRegionMesh& mesh, std::uint64_t cell)
    requires(Mode == ArithmeticMode::Ordinary)
{
    (void) mesh.geometry_epoch();
    if (mesh.sparse_native())
        return mesh.certified_sparse_cell_geometry(cell);
    for (const auto& interface : mesh.interfaces())
        std::visit(
            [](const auto& join)
            {
                if constexpr (requires { join.periodic_translation; })
                    if (join.periodic_translation)
                        details::ordinary_native::reject("periodic finite atomic cell images are unsupported.");
            },
            interface);
    if (!mesh.interfaces().empty())
    {
        std::vector<PlanarFacePtr<Mode>> faces;
        for (auto face : mesh.cell_faces(cell))
        {
            auto geometry = PlanarFace<Mode>::from_native_face(mesh, face);
            if (mesh.owner_cell(face) != cell)
                geometry = geometry->reversed();
            faces.push_back(std::move(geometry));
        }
        return PlanarFace<Mode>::cell_shell(faces);
    }
    const auto [region, local] = mesh.native_cell(cell);
    auto source = std::visit(
        [&](const auto& r) -> ConvexGeometryPtr<Mode>
        {
            if constexpr (requires { r.geometry().atomic_cell_geometry(local); })
                return r.geometry().atomic_cell_geometry(local);
            else
                return {};
        },
        mesh.regions().at(region));
    if (source)
        return source;
    const auto shell = native_cell_shell(mesh, cell);
    return make(shell.vertices, shell.outward_faces);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::from_native_shell(
    std::span<const GeometryPoint> vertices, const std::vector<std::vector<std::size_t>>& faces)
    requires(Mode == ArithmeticMode::Ordinary)
{
    const auto shell = native_cell_detail::canonical_shell(vertices, faces);
    return make(shell.vertices, shell.outward_faces);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::extruded_polygon(
    std::span<const std::array<double, 2>> polygon, double lo, double hi)
    requires(Mode == ArithmeticMode::Ordinary)
{
    if (polygon.size() < 3 || polygon.size() > 2048 || !std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo))
        details::ordinary_native::reject("invalid finite extruded polygon.");
    std::vector<GeometryPoint> points;
    std::vector<std::vector<std::size_t>> faces;
    for (const auto& p : polygon)
        points.push_back({p[0], p[1], lo});
    for (const auto& p : polygon)
        points.push_back({p[0], p[1], hi});
    const auto n = polygon.size();
    std::vector<std::size_t> bottom, top;
    for (std::size_t i = 0; i < n; ++i)
    {
        bottom.push_back(n - 1 - i);
        top.push_back(n + i);
    }
    faces.push_back(std::move(bottom));
    faces.push_back(std::move(top));
    for (std::size_t i = 0; i < n; ++i)
        faces.push_back({i, (i + 1) % n, (i + 1) % n + n, i + n});
    return make(points, faces);
}
template<ArithmeticMode Mode>
NumericalConvexShell<Mode> ConvexCellGeometry<Mode>::numerical_shell() const
    requires(Mode == ArithmeticMode::Ordinary)
{
    NumericalConvexShell<Mode> shell;
    shell.origin = origin();
    shell.outward_faces.reserve(faces().size());
    for (const auto face : faces())
        shell.outward_faces.emplace_back(face.begin(), face.end());
    for (const auto& p : local_points())
        shell.relative_vertices.push_back(
            {GeometryValue<Mode>{p.value[0]}, GeometryValue<Mode>{p.value[1]}, GeometryValue<Mode>{p.value[2]}});
    for (std::size_t f = 0; f < num_faces(); ++f)
        shell.face_support_identities.push_back(planar_support<Mode>(*face_geometry(f)));
    return shell;
}
template<ArithmeticMode Mode>
bool ConvexCellGeometry<Mode>::same_native_representation(const ConvexCellGeometry& other) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    const auto left = details::checked_geometry_storage<Mode>(geometry_->impl_);
    const auto right = details::checked_geometry_storage<Mode>(other.geometry_->impl_);
    const auto source_root = [](const auto& geometry)
    {
        return !geometry.numerical_only && !geometry.partition_proof_only &&
               (!geometry.construction || (geometry.construction->kind() == details::GeometryConstruction::Kind::Source &&
                   geometry.construction->source().native())) &&
               !geometry.partition_certificate && !geometry.export_source && !geometry.native_export_source &&
               !geometry.export_seal &&
               geometry.ancestry && !geometry.ancestry->parent && !geometry.ancestry->split &&
               !geometry.ancestry->outside && geometry.ancestry->depth == 0;
    };
    // Signed zero is observable in exported numerical values. Comparing scalar
    // fields avoids padding bytes in wider-FP storage and allocates no packets.
    const auto number_equal = [](auto a, auto b) { return a == b && (a != 0 || std::signbit(a) == std::signbit(b)); };
    const auto point_equal = [&](const auto& a, const auto& b)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!number_equal(a[axis], b[axis]))
                return false;
        return true;
    };
    const auto stored_point_equal = [&](const auto& a, const auto& b)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!number_equal(a[axis].value, b[axis].value))
                return false;
        return true;
    };
    const auto frame_equal = [&](const auto& a, const auto& b)
    { return a.exponent == b.exponent && point_equal(a.origin, b.origin); };
    if (!source_root(*left) || !source_root(*right) || !frame_equal(left->frame, right->frame) ||
        !point_equal(left->reference_origin, right->reference_origin) ||
        !number_equal(left->volume.value, right->volume.value) || left->faces != right->faces ||
        left->vertices.size() != right->vertices.size() || left->surfaces.size() != right->surfaces.size())
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!number_equal(left->bounds[axis].lo, right->bounds[axis].lo) ||
            !number_equal(left->bounds[axis].hi, right->bounds[axis].hi))
            return false;
    for (std::size_t index = 0; index < left->vertices.size(); ++index)
    {
        const auto& a = left->vertices[index];
        const auto& b = right->vertices[index];
        const auto original = [](const auto& p)
        {
            return p && p.arithmetic_mode() == ArithmeticMode::Ordinary && p.represented_native_source() &&
                   !p.constructed() && !p.numerical_only() && !p.source_origin() && p.source_edges().empty();
        };
        const auto a_source = resolve_point(left->point_context, a.source_identity);
        const auto b_source = resolve_point(right->point_context, b.source_identity);
        if (a.numerical_only || b.numerical_only || !stored_point_equal(a.coordinates, b.coordinates) ||
            !original(a_source) || !original(b_source) || !point_equal(a_source.value(), b_source.value()) ||
            !point_equal(a_source.absolute_error(), b_source.absolute_error()) ||
            a.incident.size() != b.incident.size())
            return false;
        // Incidence order contributes to subsequent native queries. Match its
        // face indices, never pointer addresses or merely equal coordinates.
        for (std::size_t incident = 0; incident < a.incident.size(); ++incident)
        {
            std::size_t face = 0;
            while (face < left->surfaces.size() && left->surfaces[face].plane != a.incident[incident])
                ++face;
            if (face == left->surfaces.size() || right->surfaces[face].plane != b.incident[incident])
                return false;
        }
    }
    for (std::size_t face = 0; face < left->surfaces.size(); ++face)
    {
        const auto& a = *left->surfaces[face].plane;
        const auto& b = *right->surfaces[face].plane;
        if (left->surfaces[face].orientation != right->surfaces[face].orientation || !frame_equal(a.frame, b.frame) ||
            !stored_point_equal(a.normal, b.normal) || !stored_point_equal(a.anchor, b.anchor) ||
            !number_equal(a.offset.value, b.offset.value) || a.input || b.input || !a.native_support ||
            !b.native_support || *a.native_support != *b.native_support ||
            a.normal_error.has_value() != b.normal_error.has_value() ||
            (a.normal_error && !number_equal(*a.normal_error, *b.normal_error)))
            return false;
    }
    // Each publication retains its construction-time FP environment. Equal
    // framed scalars alone do not establish equal exported world coordinates.
    const details::OrdinaryExportCoordinates left_export(*this), right_export(other);
    for (std::size_t vertex = 0; vertex < left->vertices.size(); ++vertex)
        if (!point_equal(left_export.at(vertex), right_export.at(vertex)))
            return false;
    // A previously prepared moment cache may have been evaluated under a
    // different caller environment. Compare its public numerical result too;
    // the existing inline cache storage requires no additional allocation.
    const auto left_moments = moments(), right_moments = other.moments();
    if (!number_equal(left_moments.volume.value, right_moments.volume.value))
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!number_equal(left_moments.first_moment[axis].value, right_moments.first_moment[axis].value))
            return false;
    return true;
}
template<ArithmeticMode Mode>
PlanarFacePtr<Mode> ConvexCellGeometry<Mode>::face_geometry(std::size_t face) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    const details::OrdinaryExportCoordinates vertices(*this);
    std::vector<std::array<long double, 3>> loop;
    loop.reserve(faces().at(face).size());
    for (auto i : faces().at(face))
        loop.push_back(vertices.at(i));
    const auto stored = details::checked_geometry_storage<Mode>(geometry_->impl_);
    const auto& surface = stored->surfaces.at(face);
    std::array<long double, 3> normal;
    for (std::size_t axis = 0; axis < 3; ++axis)
        normal[axis] = surface.orientation * surface.plane->normal[axis].value;
    auto support = surface.plane->native_support;
    if (!support)
        support = details::ordinary_native::make_source_support(loop);
    return PlanarFace<Mode>::from_retained_support(loop, normal, std::move(*support));
}
details::NativeOriginalCellMetrics details::NativeCellGeometrySourceAccess::original_convex_metrics(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell)
{
    namespace ordinary = details::ordinary_kernel;
    constexpr auto mode = ArithmeticMode::Ordinary;
    const auto stored = checked_geometry_storage<mode>(DynamicGeometryAccess::implementation(*cell.dynamic_geometry()));
    const details::Operation scope(stored->limits);
    if (stored->partition_proof_only)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Private partition witness has no source metrics.");
    const auto integrals = stored->moment_cache->ready.load(std::memory_order_acquire)
        ? *stored->moment_cache->integrals : ordinary::integrate<mode>(*stored, true);
    const auto moments = ordinary::represented_moments(integrals, stored->frame, stored->reference_origin);
    return {GeometryValue<mode>{moments.volume.value}.as_double(),
        ordinary::represented_centroid(moments, stored->volume, stored->reference_origin)};
}
void details::NativeCellGeometrySourceAccess::visit_original_convex_face_points(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell, void* context,
    void (*visit)(void*, const PointIdentity<ArithmeticMode::Ordinary>&))
{
    const NativeOrdinaryExportCoordinates coordinates(cell);
    for (const auto& face : cell.faces())
        for (const auto vertex : face)
            visit(context, GeometryIdentityAccess::point(coordinates.at(vertex)));
}
details::NativePlanarOwnerAccess::Pointer details::NativePlanarOwnerAccess::cell_face(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell, std::size_t face,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    const NativeOrdinaryExportCoordinates coordinates(cell);
    geometry_detail::NativeConstructorVector<Face::WidePoint> loop(admission.allocator<Face::WidePoint>());
    const auto row = cell.faces().at(face);
    loop.reserve(row.size());
    for (const auto vertex : row)
        loop.push_back(coordinates.at(vertex));
    const auto stored = checked_geometry_storage<ArithmeticMode::Ordinary>(
        DynamicGeometryAccess::implementation(*cell.dynamic_geometry()));
    const auto& surface = stored->surfaces.at(face);
    Face::WidePoint normal;
    for (std::size_t axis = 0; axis < 3; ++axis)
        normal[axis] = surface.orientation * surface.plane->normal[axis].value;
    auto support = surface.plane->native_support;
    if (!support)
        support = source_support(loop, admission);
    return retained(loop, normal, std::move(*support), admission);
}
template<ArithmeticMode Mode> bool ConvexCellGeometry<Mode>::quality_above(double minimum) const
{
    if (!std::isfinite(minimum) || minimum < 0)
        details::reject(Mode, "invalid finite scaled-volume quality bound.");
    const auto span = axis_spans();
    long double largest = 0;
    for (const auto& v : span)
        largest = std::max(largest, v.value);
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return volume().value > static_cast<long double>(minimum) * largest * largest * largest;
    else
        return volume().lower() > static_cast<long double>(minimum) * largest * largest * largest;
}
template<ArithmeticMode Mode>
bool ConvexCellGeometry<Mode>::axis_span_exceeds(const ConvexCellGeometry& other, std::size_t axis, double ratio) const
{
    if (axis >= 3 || !std::isfinite(ratio) || ratio < 1)
        details::reject(Mode, "invalid finite physical span comparison.");
    const auto a = axis_spans()[axis], b = other.axis_spans()[axis];
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return a.value > ratio * b.value;
    else
        return a.lower() > ratio * b.upper();
}
template<ArithmeticMode Mode> bool ConvexCellGeometry<Mode>::equivalent(const ConvexCellGeometry& other) const
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        try
        {
            const std::array candidate{from_dynamic(other.geometry_)};
            details::ordinary_native::certify_cell_partition(*this, candidate, 8e-15);
            return true;
        }
        catch (const std::invalid_argument&)
        {
            return false;
        }
    }
    else
    {
        if (origin() != other.origin() || faces() != other.faces())
            return false;
        const auto a = local_points(), b = other.local_points();
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].value != b[i].value || a[i].absolute_error != b[i].absolute_error)
                return false;
        return true;
    }
}

template<ArithmeticMode Mode>
std::size_t ConvexCellGeometry<Mode>::box_construction_capacity_bound()
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    on::require_encoding_capacity_profile();
    using namespace details;
    using Vertex = StoredVertex<Mode>;
    using Plane = StoredPlane<Mode>;
    using Surface = StoredSurface<Mode>;
    using PlanePtr = StoredPlanePtr<Mode>;
    std::size_t bytes = 0;
    auto add = [&](std::size_t count, std::size_t size)
    { bytes = on::encoding_add(bytes, on::encoding_product(count, size)); };
    // Sum complete growth histories, not only final capacities. For 8, 6 and
    // 4 insertions the respective 1+2+4+8 / 1+2+4 bounds are 15 / 7.
    // box initializer-list inner vectors, outer-vector copies, import copies.
    add(3 * 6 * 4, sizeof(std::size_t));
    add(2 * 6, sizeof(Face));
    // make's input conversion and one native-fact snapshot batch. Complete
    // table growth histories include overlap with the previous buffer.
    add(15, sizeof(DynamicPoint));
    add(15, sizeof(PointProvenanceView));
    add(1, sizeof(PointSnapshotContext<Mode>) + 2 * sizeof(void*));
    add(1, sizeof(NativePointFactBlock<Mode>) + 2 * sizeof(void*));
    add(1, sizeof(std::shared_ptr<const NativePointFactBlock<Mode>>));
    add(15, sizeof(PointCoordinates<Mode>) + sizeof(NativePointRefRow<Mode>));
    // import's framed input, immutable graph, and per-face scratch histories.
    add(15, sizeof(StoredPoint<Mode>));
    add(1, sizeof(GeometryStorage<Mode>) + sizeof(Ancestry) + sizeof(StoredMomentCache<Mode>));
    add(15, sizeof(Vertex));
    add(15, sizeof(Surface));
    add(8 * 7, sizeof(PlanePtr));
    add(6, sizeof(Plane));
    add(6 * 4, sizeof(StoredPointRef<Mode>));
    add(6 * 7, sizeof(DynamicPoint));
    add(6 * 7, sizeof(on::WidePoint));
    add(6, on::source_support_allocation_bound(4));
    add(12, sizeof(std::pair<const std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>>) + 4 * sizeof(void*));
    add(6 * 4, sizeof(std::size_t) + 4 * sizeof(void*));
    // validate_connected_boundary's graph, visitation and BFS growth histories.
    add(8 + 6, sizeof(char));
    add(6, sizeof(std::vector<std::size_t>));
    add(6 * 7 + 15, sizeof(std::size_t));
    add(24, sizeof(std::pair<const std::pair<std::size_t, std::size_t>, std::size_t>) + 4 * sizeof(void*));
    // prepare_finish cache and plane set; original source supports are empty.
    // Snapshot/fact ownership is admitted once for the batch.
    add(1, sizeof(PointExportCache));
    add(8, sizeof(StoredPointRef<Mode>));
    add(24, sizeof(PlanePtr) + 4 * sizeof(void*));
    add(1, identity_ledger_bookkeeping_bound(8));
    // Snapshot owner traversal uses a small explicit pending stack as well as
    // the identity ledger's ordered owner nodes.
    add(15, 2 * sizeof(void*) + sizeof(std::shared_ptr<const void>));
    // Final freeze retains the draft through all admissions and allocations.
    add(1, sizeof(GeometryDraft<Mode>) + sizeof(FrozenTopology<Mode>) + sizeof(GeometryStorage<Mode>));
    add(8, sizeof(FrozenVertex<Mode>));
    add(6, sizeof(PlanePtr));
    add(7 + 24 + 9 + 24 + 6, sizeof(std::uint32_t));
    add(6, sizeof(int));
    add(6, sizeof(std::pair<const StoredPlane<Mode>* const, std::size_t>) + 4 * sizeof(void*));
    add(1, sizeof(DynamicConvexGeometry) + sizeof(ConvexCellGeometry));
    return bytes;
}

std::size_t details::NativeBoxSourceAccess::scratch_bound()
{
    constexpr auto M = ArithmeticMode::Ordinary;
    namespace on = details::ordinary_native;
    using namespace details;
    geometry_detail::NativeConstructorStorage::require_profile();
    if (active_work)
        on::reject("native box source proof requires a fresh numerical operation.");
#if defined(__GLIBCXX__)
    constexpr std::size_t V = 8, F = 6, I = 24, E = 12;
    std::size_t bytes = 0;
    const auto add = [&](std::size_t count, std::size_t width)
    { bytes = on::encoding_add(bytes, on::encoding_product(count, width)); };
    const auto growth = [](std::size_t count)
    {
        std::size_t total = 0, capacity = 1;
        while (count)
        {
            total = on::encoding_add(total, capacity);
            if (capacity >= count)
                break;
            capacity = on::encoding_product(capacity, 2);
        }
        return total;
    };
    const auto combined = [&]<class T>(std::size_t count)
    { add(count, sizeof(std::_Sp_counted_ptr_inplace<T, std::allocator<void>, __gnu_cxx::__default_lock_policy>)); };
    const auto split = [&]<class T>()
    {
        add(1, sizeof(T));
        add(1, sizeof(std::_Sp_counted_ptr<T*, __gnu_cxx::__default_lock_policy>));
    };
    // fixed_box_face_rows and GeometryKernel::make's two growing input tables.
    add(F, sizeof(Face));
    add(I, sizeof(std::size_t));
    add(growth(V), sizeof(DynamicPoint) + sizeof(PointProvenanceView));
    combined.template operator()<PointSnapshotContext<M>>(1);
    combined.template operator()<NativePointFactBlock<M>>(1);
    add(1, sizeof(std::shared_ptr<const NativePointFactBlock<M>>));
    add(growth(V), sizeof(PointCoordinates<M>) + sizeof(NativePointRefRow<M>));
    // Original error-free points have no witnesses, generated ancestry,
    // foreign context, authority, alias, recipe or dependency tables.
    combined.template operator()<GeometryDraft<M>>(1);
    combined.template operator()<Ancestry>(1);
    add(growth(V), sizeof(StoredPoint<M>) + sizeof(StoredVertex<M>));
    add(F, sizeof(Face));
    add(I, sizeof(std::size_t));
    using Edge = std::pair<const std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>>;
    add(E, sizeof(std::_Rb_tree_node<Edge>));
    add(I, sizeof(std::_Rb_tree_node<std::size_t>));
    combined.template operator()<StoredPlane<M>>(F);
    add(F * growth(4), sizeof(on::WidePoint));
    add(F, on::source_support_exact_allocation_bound(4, true));
    add(V * growth(3), sizeof(StoredPlanePtr<M>));
    add(I, sizeof(StoredPointRef<M>));
    add(growth(F), sizeof(StoredSurface<M>));
    // Complete connected-boundary edge/adjacency/BFS scratch.
    add(V + F, sizeof(char));
    add(F, sizeof(std::vector<std::size_t>));
    using FirstFace = std::pair<const std::pair<std::size_t, std::size_t>, std::size_t>;
    add(E, sizeof(std::_Rb_tree_node<FirstFace>));
    add(F * growth(4) + growth(F), sizeof(std::size_t));
    combined.template operator()<StoredMomentCache<M>>(1);
    combined.template operator()<PointExportCache>(1);
    add(V, sizeof(DynamicStoredPointRef));
    add(F, sizeof(std::_Rb_tree_node<StoredPlanePtr<M>>));
    // One outer operation holds the Context/FactBlock and six supports until
    // all callbacks finish. Its pending stack never exceeds one original row.
    add(1, sizeof(GeometryAllocationLedger::Impl));
    using LedgerEntry = typename decltype(GeometryAllocationLedger::Impl::owners)::value_type;
    add(F + 2, sizeof(std::_Rb_tree_node<LedgerEntry>));
    add(1, sizeof(allocation_detail::Item));
    using Slot = std::pair<const StoredPlane<M>* const, std::size_t>;
    add(F, sizeof(std::_Rb_tree_node<Slot>));
    combined.template operator()<FrozenTopology<M>>(1);
    add(F, sizeof(StoredPlanePtr<M>));
    add(F + 1 + I + V + 1 + I + F, sizeof(std::uint32_t));
    add(F, sizeof(int));
    add(V, sizeof(FrozenVertex<M>));
    combined.template operator()<GeometryStorage<M>>(1);
    split.template operator()<DynamicConvexGeometry>();
    split.template operator()<ConvexCellGeometry<M>>();
    return bytes;
#else
    on::reject("unsupported native box source allocation profile.");
#endif
}
void details::NativeBoxSourceAccess::verify(GeometryPoint lower, GeometryPoint upper,
    const geometry_detail::NativeConstructionAdmission& admission, void* context,
    void (*callback)(void*, const ConvexCellGeometry<ArithmeticMode::Ordinary>&))
{
    if (!callback)
        throw std::invalid_argument("Native box source proof needs its synchronous verifier.");
    auto phase = admission.reserve_profile({scratch_bound()});
    {
        // Preserve the original predicates, support ancestry, coordinate
        // publication and representability checks; no alternative box algebra.
        Operation operation(mode_limits(ArithmeticMode::Ordinary));
        const auto canonical = ConvexCellGeometry<ArithmeticMode::Ordinary>::box(lower, upper);
        callback(context, *canonical);
    }
    phase.finish_scratch();
}

/** This is deliberately a separate inventory from the retained-payload box
 * helper above. It includes allocation controls and qualified node layouts. */
template<ArithmeticMode Mode>
std::size_t ConvexCellGeometry<Mode>::unit_prism_construction_capacity_bound(std::size_t q)
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    on::require_encoding_capacity_profile();
    // A pre-existing numerical ledger pins newly constructed snapshot/fact/
    // support owners after the temporary prism dies. This reservation cannot
    // retire those retained bytes, so only a fresh numerical scope qualifies.
    // Public legacy constructor nesting retains its original behavior.
    if (details::active_work)
        on::reject("unit-prism admission requires a fresh numerical operation.");
    const GeometryLimits limits;
    if (q < 3 || q > limits.maximum_vertices / 2 || limits.maximum_faces < 2 || q > limits.maximum_faces - 2)
        on::reject("unit prism exceeds existing native geometry vertex/face caps.");
#if defined(__GLIBCXX__)
    using namespace details;
    const auto V = on::encoding_product(2, q), F = on::encoding_add(q, 2);
    const auto E = on::encoding_product(3, q), I = on::encoding_product(6, q);
    std::size_t bytes = 0;
    const auto add = [&](std::size_t count, std::size_t size)
    { bytes = on::encoding_add(bytes, on::encoding_product(count, size)); };
    const auto growth = [&](std::size_t count)
    {
        std::size_t total = 0, capacity = 1;
        while (count)
        {
            total = on::encoding_add(total, capacity);
            if (capacity >= count) break;
            capacity = on::encoding_product(capacity, 2);
        }
        return total;
    };
    const auto combined = [&]<class T>(std::size_t count)
    {
        add(count, sizeof(std::_Sp_counted_ptr_inplace<T, std::allocator<void>, __gnu_cxx::__default_lock_policy>));
    };
    const auto split = [&]<class T>()
    {
        add(1, sizeof(T));
        add(1, sizeof(std::_Sp_counted_ptr<T*, __gnu_cxx::__default_lock_policy>));
    };
    // The admitted caller reserves Product.base and product_geometry's exact
    // vertex/face/cap-loop buffers. Side initializer-list rows each allocate 4.
    add(q, sizeof(on::WidePoint));
    add(V, sizeof(Point<Mode>));
    add(F, sizeof(Face));
    add(on::encoding_add(on::encoding_product(2, q), on::encoding_product(4, q)), sizeof(std::size_t));
    // ConvexCellGeometry::make_points -> dynamic_points -> make_bounded.
    add(V, sizeof(DynamicPoint));
    add(growth(V), sizeof(DynamicPoint));
    add(V, sizeof(PointProvenanceView));
    combined.template operator()<PointSnapshotContext<Mode>>(1);
    combined.template operator()<NativePointFactBlock<Mode>>(1);
    add(1, sizeof(std::shared_ptr<const NativePointFactBlock<Mode>>));
    add(growth(V), sizeof(PointCoordinates<Mode>));
    add(growth(V), sizeof(NativePointRefRow<Mode>));
    // No authorities, witnesses, foreign dependencies, generated recipes or
    // publication aliases are reachable for these original error-free points.
    // ordinary_kernel::import; faces copies allocate exact outer/inner sizes.
    combined.template operator()<GeometryDraft<Mode>>(1);
    combined.template operator()<Ancestry>(1);
    add(growth(V), sizeof(StoredPoint<Mode>));
    add(growth(V), sizeof(StoredVertex<Mode>));
    add(F, sizeof(Face)); add(I, sizeof(std::size_t));
    using EdgeEntry = std::pair<const std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>>;
    add(E, sizeof(std::_Rb_tree_node<EdgeEntry>));
    add(I, sizeof(std::_Rb_tree_node<std::size_t>)); // Per-face distinct sets, including cap rows.
    combined.template operator()<StoredPlane<Mode>>(F);
    const auto face_growth = on::encoding_add(on::encoding_product(2, growth(q)),
        on::encoding_product(q, growth(4)));
    add(face_growth, sizeof(on::WidePoint)); // native_points in every face.
    add(2, on::source_support_exact_allocation_bound(q, true)); // Both caps are known Z axes.
    add(q, on::source_support_exact_allocation_bound(4)); // Vertical side: exact max of Axis/Source branches.
    add(on::encoding_product(V, growth(3)), sizeof(StoredPlanePtr<Mode>));
    add(I, sizeof(StoredPointRef<Mode>));
    add(growth(F), sizeof(StoredSurface<Mode>));
    // Every nonincident convexity query collects the original points of its
    // face: four for a side and q for a cap. The original prism has distinct
    // points and no keyed authorities. The intersection of any two planes'
    // source-vertex lists at the queried vertex is one edge containing that
    // vertex. A nonincident face excludes it, so infer_edges' result stays empty.
    const auto side_queries = on::encoding_product(q, V - 4);
    const auto cap_queries = on::encoding_product(2, V - q);
    add(on::encoding_product(side_queries, growth(4)), sizeof(PointProvenanceView));
    add(on::encoding_product(cap_queries, growth(q)), sizeof(PointProvenanceView));
    // validate_connected_boundary. The inspected std::map::emplace's usable-key
    // fast path rejects a repeated edge BEFORE allocating a node, so E nodes.
    add(V, sizeof(char)); add(F, sizeof(char));
    add(F, sizeof(std::vector<std::size_t>));
    using FirstFaceEntry = std::pair<const std::pair<std::size_t, std::size_t>, std::size_t>;
    add(E, sizeof(std::_Rb_tree_node<FirstFaceEntry>));
    add(face_growth, sizeof(std::size_t)); // Face adjacency degree equals its loop size.
    add(growth(F), sizeof(std::size_t)); // Connected-boundary BFS.
    // prepare_finish; integrate uses fixed-stack compensated predicates only.
    combined.template operator()<StoredMomentCache<Mode>>(1);
    combined.template operator()<PointExportCache>(1);
    add(V, sizeof(DynamicStoredPointRef));
    add(F, sizeof(std::_Rb_tree_node<StoredPlanePtr<Mode>>));
    // The active ledger admits one Context, one NativeFactBlock and one
    // depth-one support backing per plane. Its pending stack needs one slot.
    add(1, sizeof(GeometryAllocationLedger::Impl));
    using LedgerEntry = typename decltype(GeometryAllocationLedger::Impl::owners)::value_type;
    add(on::encoding_add(F, 2), sizeof(std::_Rb_tree_node<LedgerEntry>));
    add(1, sizeof(allocation_detail::Item));
    // freeze_geometry retains all draft buffers while allocating frozen ones.
    using SlotEntry = std::pair<const StoredPlane<Mode>* const, std::size_t>;
    add(F, sizeof(std::_Rb_tree_node<SlotEntry>));
    combined.template operator()<FrozenTopology<Mode>>(1);
    add(F, sizeof(StoredPlanePtr<Mode>));
    add(on::encoding_add(on::encoding_add(F + 1, I),
        on::encoding_add(on::encoding_add(V + 1, I), F)), sizeof(std::uint32_t));
    add(F, sizeof(int));
    add(V, sizeof(FrozenVertex<Mode>));
    combined.template operator()<GeometryStorage<Mode>>(1);
    split.template operator()<DynamicConvexGeometry>();
    split.template operator()<ConvexCellGeometry<Mode>>();
    return bytes;
#else
    on::reject("unsupported unit-prism allocation layout.");
#endif
}

namespace native_cell_detail
{
template<class Writer>
void emit_cell_key(Writer& w, const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell, bool admitted = false)
{
    constexpr auto Mode = ArithmeticMode::Ordinary;
    namespace on = details::ordinary_native;
    w.tag(GeometryTag::OrdinaryCell);
    w.point(on::wide(cell.origin()));
    if (admitted)
    {
        w.word(cell.vertex_count());
        details::visit_local_coordinates<Mode>(
            cell, [&](std::size_t, const PointCoordinates<Mode>& point) { w.point(point.value); });
    }
    else
    {
        const auto vertices = cell.local_points();
        w.word(vertices.size());
        for (const auto& p : vertices)
            w.point(p.value);
    }
    w.word(cell.faces().size());
    const auto stored =
        details::checked_geometry_storage<Mode>(DynamicGeometryAccess::implementation(*cell.dynamic_geometry()));
    for (std::size_t index = 0; index < cell.faces().size(); ++index)
    {
        const auto& f = cell.faces()[index];
        w.word(f.size());
        for (auto i : f)
            w.word(i);
        auto support = stored->surfaces.at(index).plane->native_support;
        if (!support)
        {
            if (admitted)
                throw std::invalid_argument("Native cell codec needs its retained source support.");
            support = planar_support<Mode>(*cell.face_geometry(index));
        }
        w.word(canonical_key_size(*support));
        w.identity(*support);
    }
}
} // namespace native_cell_detail

std::vector<char> canonical_cell_key(const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell)
{
    details::ordinary_native::Writer w;
    native_cell_detail::emit_cell_key(w, cell);
    return std::move(w.bytes);
}
template<ArithmeticMode Mode>
std::vector<char> ConvexCellGeometry<Mode>::encode_certificate(std::size_t byte_budget) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    return encode_geometry_packet(GeometryPacketKind::CellCertificate, Mode, canonical_cell_key(*this), byte_budget);
}
std::size_t details::NativeCellCodecAccess::size(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell, const geometry_detail::NativeConstructionAdmission&)
{
    details::ordinary_native::CountingWriter count;
    native_cell_detail::emit_cell_key(count, cell, true);
    (void) details::geometry_packet_header(GeometryPacketKind::CellCertificate, ArithmeticMode::Ordinary,
        count.bytes.size(), geometry_packet_default_budget);
    return details::ordinary_native::encoding_add(geometry_packet_header_bytes, count.bytes.size());
}
void details::NativeCellCodecAccess::append(std::vector<char>& bytes,
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto extent = size(cell, admission);
    if (extent > bytes.capacity() - bytes.size())
        throw std::invalid_argument("Native cell destination was not preadmitted.");
    const auto initial = bytes.size();
    const auto header = details::geometry_packet_header(GeometryPacketKind::CellCertificate, ArithmeticMode::Ordinary,
        extent - geometry_packet_header_bytes, geometry_packet_default_budget);
    bytes.insert(bytes.end(), header.begin(), header.end());
    details::ordinary_native::BasicWriter<details::NativeGeometryByteWriter> writer{{&bytes, &admission}};
    native_cell_detail::emit_cell_key(writer, cell, true);
    if (bytes.size() - initial != extent)
        throw std::invalid_argument("Native cell certificate length differs from its exact count.");
}
std::vector<char> details::NativeCellCodecAccess::encode(const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell,
    const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!storage.owner)
        throw std::invalid_argument("Native cell encoding needs its concrete storage owner.");
    const auto& admission = storage.owner->admission();
    std::vector<char> bytes;
    storage.owner->reserve(bytes, size(cell, admission));
    append(bytes, cell, admission);
    return bytes;
}
namespace native_cell_detail
{
struct DecodeCppPlan
{
    std::size_t vertices = 0, faces = 0, incidence = 0, support_nodes = 0, support_depth = 0;
    std::size_t support_retained = 0, support_scratch = 0, bytes = 0;
};
DecodeCppPlan decode_cpp_plan(std::span<const char> packet, std::size_t byte_budget)
{
    constexpr auto M = ArithmeticMode::Ordinary;
    const auto envelope = decode_geometry_packet(packet, M, byte_budget);
    if (envelope.kind != GeometryPacketKind::CellCertificate)
        on::reject("foreign cell certificate kind.");
    on::Reader reader{envelope.body, 0, true};
    if (reader.tag() != GeometryTag::OrdinaryCell)
        on::reject("foreign finite cell packet mode.");
    (void) reader.point();
    DecodeCppPlan plan;
    plan.vertices = reader.count(4096);
    for (std::size_t i = 0; i < plan.vertices; ++i)
        (void) reader.point();
    plan.faces = reader.count(4096);
    for (std::size_t f = 0; f < plan.faces; ++f)
    {
        const auto count = reader.count(plan.vertices);
        plan.incidence = on::encoding_add(plan.incidence, count);
        for (std::size_t i = 0; i < count; ++i)
            if (reader.word() >= plan.vertices)
                on::reject("invalid finite cell packet vertex.");
        const auto support = details::ordinary_support_import_cpp_profile(reader.packet());
        plan.support_nodes = on::encoding_add(plan.support_nodes, support.nodes);
        plan.support_depth = std::max(plan.support_depth, support.depth);
        plan.support_retained = on::encoding_add(plan.support_retained, support.retained);
        plan.support_scratch = std::max(plan.support_scratch, support.scratch);
    }
    reader.finish();
    geometry_detail::NativeConstructorStorage::require_profile();
#if defined(__GLIBCXX__)
    using namespace details;
    const auto V = plan.vertices, F = plan.faces, I = plan.incidence;
    const auto add = [&](std::size_t count, std::size_t width)
    { plan.bytes = on::encoding_add(plan.bytes, on::encoding_product(count, width)); };
    const auto growth = [&](std::size_t count)
    {
        std::size_t total = 0, capacity = 1;
        while (count)
        {
            total = on::encoding_add(total, capacity);
            if (capacity >= count)
                break;
            capacity = on::encoding_product(capacity, 2);
        }
        return total;
    };
    const auto combined = [&]<class T>(std::size_t count)
    { add(count, sizeof(std::_Sp_counted_ptr_inplace<T, std::allocator<void>, __gnu_cxx::__default_lock_policy>)); };
    const auto split = [&]<class T>(std::size_t count)
    {
        add(count, sizeof(T));
        add(count, sizeof(std::_Sp_counted_ptr<T*, __gnu_cxx::__default_lock_policy>));
    };
    // Original decode rows and make_local's typed/dynamic conversion and framed input.
    add(V, sizeof(Point<M>));
    add(F, sizeof(Face));
    add(I, sizeof(std::size_t));
    add(F, sizeof(PlaneSupportIdentity<M>));
    add(on::encoding_add(V, growth(V)), sizeof(DynamicPoint));
    add(V, sizeof(PointProvenanceView));
    combined.template operator()<PointSnapshotContext<M>>(1);
    combined.template operator()<NativePointFactBlock<M>>(1);
    add(1, sizeof(std::shared_ptr<const NativePointFactBlock<M>>));
    add(growth(V), sizeof(PointCoordinates<M>) + sizeof(NativePointRefRow<M>));
    combined.template operator()<GeometryDraft<M>>(1);
    combined.template operator()<Ancestry>(1);
    add(growth(V), sizeof(StoredPoint<M>) + sizeof(StoredVertex<M>));
    add(F, sizeof(Face));
    add(I, sizeof(std::size_t));
    // Sum of per-row vector growth histories is below four times total incidence.
    add(on::encoding_product(4, I), sizeof(StoredPlanePtr<M>));
    add(on::encoding_product(2, I), sizeof(StoredPointRef<M>));
    add(growth(F), sizeof(StoredSurface<M>));
    combined.template operator()<StoredPlane<M>>(on::encoding_product(2, F));
    using Edge = std::pair<const std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>>;
    add(I, sizeof(std::_Rb_tree_node<Edge>));
    add(I, sizeof(std::_Rb_tree_node<std::size_t>));
    add(on::encoding_product(4, I), sizeof(on::WidePoint));
    // Each initially constructed plane has one original source support. This
    // exact per-face inventory includes its unique payload, backing/control,
    // original-point buffer and depth-one canonical pointer scratch.
    on::Reader face_reader{envelope.body, 0, true};
    (void) face_reader.tag();
    (void) face_reader.point();
    const auto points = face_reader.count(4096);
    for (std::size_t i = 0; i < points; ++i)
        (void) face_reader.point();
    (void) face_reader.count(4096);
    for (std::size_t f = 0; f < F; ++f)
    {
        const auto degree = face_reader.count(V);
        for (std::size_t i = 0; i < degree; ++i)
            (void) face_reader.word();
        (void) face_reader.packet();
        add(1, on::source_support_exact_allocation_bound(degree));
    }
    // Connected-boundary vertices, face adjacency, edge map and BFS work.
    add(on::encoding_add(V, F), sizeof(char));
    add(F, sizeof(std::vector<std::size_t>));
    using FirstFace = std::pair<const std::pair<std::size_t, std::size_t>, std::size_t>;
    add(I, sizeof(std::_Rb_tree_node<FirstFace>));
    add(on::encoding_add(on::encoding_product(4, I), growth(F)), sizeof(std::size_t));
    combined.template operator()<StoredMomentCache<M>>(1);
    combined.template operator()<PointExportCache>(1);
    add(V, sizeof(DynamicStoredPointRef));
    add(F, sizeof(std::_Rb_tree_node<StoredPlanePtr<M>>));
    // Initial freeze plus with_native_supports' exact immutable metadata clone.
    using Slot = std::pair<const StoredPlane<M>* const, std::size_t>;
    add(F, sizeof(std::_Rb_tree_node<Slot>));
    combined.template operator()<FrozenTopology<M>>(2);
    add(on::encoding_product(2, F), sizeof(StoredPlanePtr<M>));
    const auto indices = on::encoding_add(
        on::encoding_add(V, on::encoding_product(2, F)), on::encoding_add(2, on::encoding_product(2, I)));
    const auto width = compact_topology({V, F, F, I, I}) ? sizeof(std::uint32_t) : sizeof(std::uint64_t);
    add(on::encoding_product(2, indices), width);
    add(on::encoding_product(2, F), sizeof(int));
    add(on::encoding_product(2, V), sizeof(FrozenVertex<M>));
    combined.template operator()<GeometryStorage<M>>(2);
    split.template operator()<DynamicConvexGeometry>(2);
    split.template operator()<ConvexCellGeometry<M>>(2);
    // The one active numerical ledger pins parsed supports and both immutable
    // construction generations. Fresh source points have no dependency,
    // authority, alias, witness, split, recipe or partition-certificate owners.
    const auto owner_rows = on::encoding_add(plan.support_nodes, on::encoding_add(on::encoding_product(3, F), 13));
    add(1, sizeof(GeometryAllocationLedger::Impl));
    using LedgerEntry = typename decltype(GeometryAllocationLedger::Impl::owners)::value_type;
    add(owner_rows, sizeof(std::_Rb_tree_node<LedgerEntry>));
    add(growth(owner_rows), sizeof(allocation_detail::Item));
    add(1, plan.support_retained);
    add(1, plan.support_scratch);
    // Final canonical byte validation owns a growth writer and the two local
    // coordinate conversion vectors, plus the largest identity ancestry stack.
    add(on::encoding_product(4, envelope.body.size()), sizeof(char));
    add(V, sizeof(DynamicPoint) + sizeof(Point<M>));
    add(plan.support_depth, sizeof(const GeometryIdentityBacking*));
    return plan;
#else
    throw std::invalid_argument("Unsupported native cell decoding C++ allocation profile.");
#endif
}
} // namespace native_cell_detail
ConvexGeometryPtr<ArithmeticMode::Ordinary> details::NativeCellCodecAccess::decode(std::span<const char> packet,
    std::size_t byte_budget, const geometry_detail::NativeConstructionAdmission& admission)
{
    constexpr auto M = ArithmeticMode::Ordinary;
    using namespace details;
    namespace O = geometry_detail;
    using Cell = ConvexCellGeometry<M>;
    if (details::active_work)
        throw std::invalid_argument("Native certificate admission requires a fresh numerical operation.");
    const auto plan = native_cell_detail::decode_cpp_plan(packet, byte_budget);
    // Keep every original verifier allocation inside this finite closed phase.
    // None of its opaque owners is allowed to escape the deep admitted copy.
    auto phase = admission.reserve_profile({plan.bytes});
    ConvexGeometryPtr<M> result;
    {
        const auto verified = Cell::decode_certificate(packet, byte_budget);
        const auto old =
            checked_geometry_storage<M>(DynamicGeometryAccess::implementation(*verified->dynamic_geometry()));
        if (!old->point_context || !old->point_export_cache || !old->moment_cache || !old->ancestry ||
            old->construction || old->export_source || old->native_export_source || old->partition_certificate ||
            old->ancestry->parent || old->ancestry->split || old->vertices.borrowed())
            throw std::invalid_argument("Native certificate decoder produced a foreign source closure.");
        const auto& context = *old->point_context;
        if (!context.dependencies.empty() || !context.generated_recipes.empty() || !context.export_aliases.empty() ||
            !context.generated_points.empty() || !context.export_points.empty() || !context.numerical_points.empty() ||
            !context.witnesses.empty() || !context.authority_table.empty() || !context.incidence.empty() ||
            !context.edges.empty())
            throw std::invalid_argument("Native certificate decoder has non-original point dependencies.");
        auto ancestry = O::NativeObjectOwnerAccess<Ancestry>::make_without_buffers(admission);
        ancestry.owner->outside = old->ancestry->outside;
        ancestry.owner->depth = old->ancestry->depth;
        auto point_context = O::NativeObjectOwnerAccess<PointSnapshotContext<M>>::make(admission);
        auto& point_storage = O::NativeObjectOwnerAccess<PointSnapshotContext<M>>::storage(point_context.owner);
        point_storage.owner->reserve(point_context.owner->native_fact_blocks, context.native_fact_blocks.size());
        for (const auto& block : context.native_fact_blocks)
        {
            if (!block || !block->witnesses.empty())
                throw std::invalid_argument("Native certificate source facts contain foreign error witnesses.");
            auto next = O::NativeObjectOwnerAccess<NativePointFactBlock<M>>::make(admission);
            O::NativeObjectOwnerAccess<NativePointFactBlock<M>>::storage(next.owner)
                .owner->copy(next.owner->points, std::span(block->points));
            point_context.owner->native_fact_blocks.push_back(std::move(next.owner));
        }
        point_storage.owner->copy(point_context.owner->native_refs, std::span(context.native_refs));
        point_context.owner->frozen = context.frozen;
        auto cache = O::NativeObjectOwnerAccess<PointExportCache>::make(admission);
        cache.owner->environment = old->point_export_cache->environment;
        O::NativeObjectOwnerAccess<PointExportCache>::storage(cache.owner)
            .owner->copy(cache.owner->vertex_provenance,
                std::span<const DynamicStoredPointRef>(old->point_export_cache->vertex_provenance));
        cache.owner->provenance_root = point_context.owner;
        auto moments = O::NativeObjectOwnerAccess<StoredMomentCache<M>>::make_without_buffers(admission);
        if (old->moment_cache->integrals)
            std::call_once(moments.owner->prepared, [&] { moments.owner->integrals = old->moment_cache->integrals; });
        moments.owner->ready.store(old->moment_cache->ready.load(std::memory_order_acquire), std::memory_order_release);
        auto topology = O::NativeObjectOwnerAccess<FrozenTopology<M>>::make(admission);
        const auto& topology_storage = O::NativeObjectOwnerAccess<FrozenTopology<M>>::storage(topology.owner);
        topology_storage.owner->reserve(topology.owner->planes, old->topology->planes.size());
        // Decode original support packets directly into their independently
        // admitted immutable DAG owners, retaining the original face/slot map.
        O::NativeConstructorVector<PlaneSupportIdentity<M>> supports(admission.allocator<PlaneSupportIdentity<M>>());
        supports.reserve(plan.faces);
        const auto envelope = decode_geometry_packet(packet, M, byte_budget);
        ordinary_native::Reader reader{envelope.body, 0, true};
        (void) reader.tag();
        (void) reader.point();
        const auto n = reader.count(4096);
        for (std::size_t i = 0; i < n; ++i)
            (void) reader.point();
        (void) reader.count(4096);
        for (std::size_t f = 0; f < plan.faces; ++f)
        {
            const auto count = reader.count(n);
            for (std::size_t i = 0; i < count; ++i)
                (void) reader.word();
            supports.push_back(
                GeometryIdentityAccess::decode_support_admitted(reader.packet(), byte_budget, admission));
        }
        for (std::size_t slot = 0; slot < old->topology->planes.size(); ++slot)
        {
            const auto& source = *old->topology->planes[slot];
            if (source.point_context != old->point_context.get() || !source.authorities.empty())
                throw std::invalid_argument("Native certificate plane has a foreign source context.");
            auto plane = O::NativeObjectOwnerAccess<StoredPlane<M>>::make(admission);
            plane.owner->frame = source.frame;
            plane.owner->normal = source.normal;
            plane.owner->anchor = source.anchor;
            plane.owner->offset = source.offset;
            plane.owner->input = source.input;
            plane.owner->normal_error = source.normal_error;
            plane.owner->point_context = point_context.owner.get();
            O::NativeObjectOwnerAccess<StoredPlane<M>>::storage(plane.owner)
                .owner->copy(plane.owner->source_vertices, std::span(source.source_vertices));
            std::optional<std::size_t> face;
            std::visit(
                [&](const auto& arrays)
                {
                    for (std::size_t f = 0; f < arrays.surface_planes.size(); ++f)
                        if (arrays.surface_planes[f] == slot)
                        {
                            face = f;
                            break;
                        }
                },
                old->topology->arrays);
            if (!face)
                throw std::invalid_argument("Native certificate plane lacks its original face support.");
            plane.owner->native_support = supports[*face];
            topology.owner->planes.push_back(std::move(plane.owner));
        }
        std::visit(
            [&](const auto& source)
            {
                using Arrays = std::remove_cvref_t<decltype(source)>;
                auto& copy = topology.owner->arrays.template emplace<Arrays>();
                topology_storage.owner->copy(copy.face_offsets, std::span(source.face_offsets));
                topology_storage.owner->copy(copy.face_vertices, std::span(source.face_vertices));
                topology_storage.owner->copy(copy.incident_offsets, std::span(source.incident_offsets));
                topology_storage.owner->copy(copy.incident_planes, std::span(source.incident_planes));
                topology_storage.owner->copy(copy.surface_planes, std::span(source.surface_planes));
                topology_storage.owner->copy(copy.surface_orientations, std::span(source.surface_orientations));
            },
            old->topology->arrays);
        auto geometry = O::NativeObjectOwnerAccess<GeometryStorage<M>>::make(
            admission, std::shared_ptr<const Ancestry>(ancestry.owner));
        const auto& geometry_storage = O::NativeObjectOwnerAccess<GeometryStorage<M>>::storage(geometry.owner);
        std::vector<FrozenVertex<M>> vertices;
        geometry_storage.owner->reserve(vertices, old->vertices.size());
        for (std::size_t i = 0; i < old->vertices.size(); ++i)
        {
            auto vertex = old->vertices[i];
            vertex.incident = topology.owner->incidence(i);
            vertices.push_back(std::move(vertex));
        }
        geometry.owner->vertices = FrozenVertexBuffer<M>(std::move(vertices));
        geometry.owner->frame = old->frame;
        geometry.owner->reference_origin = old->reference_origin;
        geometry.owner->limits = old->limits;
        geometry.owner->bounds = old->bounds;
        geometry.owner->partition_proof_only = old->partition_proof_only;
        geometry.owner->numerical_only = old->numerical_only;
        geometry.owner->export_seal = old->export_seal;
        geometry.owner->topology = topology.owner;
        geometry.owner->faces = topology.owner->faces();
        geometry.owner->surfaces = SurfaceRange<M>(topology.owner.get());
        geometry.owner->point_context = point_context.owner;
        geometry.owner->volume = old->volume;
        geometry.owner->moment_cache = moments.owner;
        geometry.owner->point_export_cache = cache.owner;
        auto dynamic = O::NativeObjectOwnerAccess<DynamicConvexGeometry>::construct_without_buffers(admission,
            [&](DynamicConvexGeometry* target)
            { ::new (static_cast<void*>(target)) DynamicConvexGeometry(geometry.owner); });
        auto typed = O::NativeObjectOwnerAccess<Cell>::construct_without_buffers(
            admission, [&](Cell* target) { ::new (static_cast<void*>(target)) Cell(dynamic.owner); });
        result = std::move(typed.owner);
    } // Every default decoder object dies before its closed physical phase retires.
    phase.finish_scratch();
    return result;
}
template<ArithmeticMode Mode>
std::size_t ConvexCellGeometry<Mode>::certificate_encoding_scratch_bytes() const
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    on::require_encoding_capacity_profile();
    const auto storage = details::checked_geometry_storage<Mode>(DynamicGeometryAccess::implementation(*geometry_));
    std::size_t fallback = 0;
    for (std::size_t index = 0; index < faces().size(); ++index)
        if (!storage->surfaces.at(index).plane->native_support)
            fallback = std::max(fallback, on::fallback_face_allocation_bound(faces()[index].size()));
    // local_points converts a reserved DynamicPoint vector to an independently
    // reserved typed vector before releasing the former. Count both throughout
    // the traversal, together with the largest original face fallback.
    return on::encoding_add(on::encoding_product(vertex_count(), sizeof(DynamicPoint) + sizeof(Point<Mode>)), fallback);
}
template<ArithmeticMode Mode>
std::size_t ConvexCellGeometry<Mode>::encoded_certificate_size(std::size_t scratch_budget) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    if (certificate_encoding_scratch_bytes() > scratch_budget)
        throw std::invalid_argument("Cell certificate encoding scratch exceeds its byte budget.");
    on::CountingWriter w;
    native_cell_detail::emit_cell_key(w, *this);
    if (geometry_packet_profile(Mode) == GeometryNumericProfile::Unsupported ||
        w.bytes.size() > geometry_packet_default_budget - geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet exceeds its kind or byte budget.");
    return on::encoding_add(geometry_packet_header_bytes, w.bytes.size());
}

template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::decode_certificate(
    std::span<const char> bytes, std::size_t byte_budget)
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    const auto envelope = decode_geometry_packet(bytes, Mode, byte_budget);
    if (envelope.kind != GeometryPacketKind::CellCertificate)
        on::reject("foreign cell certificate kind.");
    on::Reader r{envelope.body, 0, true};
    if (r.tag() != GeometryTag::OrdinaryCell)
        on::reject("foreign finite cell packet mode.");
    const auto origin = on::display(r.point());
    GeometryLimits record_limits;
    record_limits.arithmetic_mode = Mode;
    record_limits.byte_budget = byte_budget;
    details::Operation record_operation(record_limits);
    auto& work = *details::active_work;
    if (work.bytes > work.limits.byte_budget)
        on::reject("cell construction budget is already exhausted.");
    const auto allowance = std::min(byte_budget, work.limits.byte_budget - work.bytes);
    const auto record_cap = work.bytes + allowance;
    struct RestoreBudget
    {
        details::Work& work;
        std::size_t previous;
        ~RestoreBudget() { work.limits.byte_budget = previous; }
    } restore{work, work.limits.byte_budget};
    work.limits.byte_budget = record_cap;
    auto& native_used = work.bytes;
    auto& allocations = work.allocations;
    const auto charge = [&](std::size_t count, std::size_t size)
    {
        if (native_used > record_cap || (size && count > (record_cap - native_used) / size))
            on::reject("cell record exceeds its native decoding budget.");
        native_used += count * size;
    };
    charge(1, sizeof(ConvexCellGeometry<Mode>) + sizeof(DynamicConvexGeometry));
    const auto n = r.count(4096);
    if (n > (r.bytes.size() - r.cursor) / 48)
        on::reject("cell point count exceeds the remaining encoded coordinates.");
    charge(n, sizeof(Point<Mode>));
    std::vector<Point<Mode>> p;
    p.reserve(n);
    if (p.capacity() > n)
        charge(p.capacity() - n, sizeof(Point<Mode>));
    for (std::size_t i = 0; i < n; ++i)
        p.emplace_back(r.point());
    const auto nf = r.count(4096);
    constexpr auto minimum_face_bytes = (1 + 3 + 1 + 1 + 2) * sizeof(std::uint64_t) + sizeof(GeometryTag);
    // Count, triangle indices, blob length, axis-support tag/index and zero scalar.
    if (nf > (r.bytes.size() - r.cursor) / minimum_face_bytes)
        on::reject("cell face count exceeds the remaining encoded face records.");
    charge(nf, sizeof(std::vector<std::size_t>) + sizeof(PlaneSupportIdentity<ArithmeticMode::Ordinary>));
    std::vector<std::vector<std::size_t>> faces;
    faces.reserve(nf);
    std::vector<PlaneSupportIdentity<ArithmeticMode::Ordinary>> supports;
    supports.reserve(nf);
    std::size_t index_count = 0;
    if (faces.capacity() > nf)
        charge(faces.capacity() - nf, sizeof(std::vector<std::size_t>));
    if (supports.capacity() > nf)
        charge(supports.capacity() - nf, sizeof(PlaneSupportIdentity<ArithmeticMode::Ordinary>));
    for (std::size_t f = 0; f < nf; ++f)
    {
        const auto count = r.count(n);
        if (count > (r.bytes.size() - r.cursor) / 8)
            on::reject("cell face index count exceeds the remaining encoded indices.");
        charge(count, sizeof(std::size_t));
        index_count += count;
        std::vector<std::size_t> face;
        face.reserve(count);
        if (face.capacity() > count)
            charge(face.capacity() - count, sizeof(std::size_t));
        for (std::size_t j = 0; j < count; ++j)
        {
            const auto i = r.word();
            if (i >= n)
                on::reject("invalid finite cell packet vertex.");
            face.push_back(i);
        }
        faces.push_back(std::move(face));
        const auto support = r.packet();
        if (support.empty())
            on::reject("finite cell packet lost its source support.");
        if (support.size() > record_cap - native_used)
            on::reject("cell support exceeds the remaining native record budget.");
        auto identity = details::GeometryIdentityAccess::decode_key<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(
            details::checked_identity_key(GeometryIdentityKind::PlaneSupport, Mode, support), record_cap - native_used);
        allocations.retain(identity, native_used, record_cap);
        supports.push_back(std::move(identity));
    }
    r.finish();
    // These arrays/planes are unavoidably new during reconstruction. Check
    // this lower bound before make_local allocates any output graph. The
    // shared Work/ledger then enforces the remaining cumulative requests.
    std::size_t reconstruction = 0;
    const auto require = [&](std::size_t count, std::size_t size)
    {
        const auto remaining = record_cap - native_used;
        if (reconstruction > remaining || (size && count > (remaining - reconstruction) / size))
            on::reject("cell reconstruction exceeds the remaining native record budget.");
        reconstruction += count * size;
    };
    require(1, sizeof(details::GeometryDraft<Mode>) + sizeof(details::GeometryStorage<Mode>) +
                   sizeof(details::FrozenTopology<Mode>));
    require(n, sizeof(details::FrozenVertex<Mode>));
    require(nf, sizeof(details::StoredPlanePtr<Mode>) + sizeof(int) +
                    sizeof(std::pair<const details::StoredPlane<Mode>* const, std::size_t>) + 4 * sizeof(void*));
    // Native shells have one incidence per face vertex; use the actual selected width.
    const auto index_bytes = details::compact_topology({n, nf, nf, index_count, index_count}) ? sizeof(std::uint32_t)
                                                                                              : sizeof(std::uint64_t);
    require(details::topology_sum(details::topology_sum(n, nf), 2), index_bytes);
    require(index_count, 2 * index_bytes);
    require(nf, index_bytes);
    require(n, sizeof(details::StoredVertex<Mode>) + sizeof(DynamicPoint));
    require(nf,
        sizeof(details::StoredPlane<Mode>) + sizeof(details::StoredSurface<Mode>) + sizeof(std::vector<std::size_t>));
    require(index_count, sizeof(std::size_t));
    auto construction_limits = work.limits;
    construction_limits.byte_budget = allowance;
    auto result = make_local(origin, p, faces, construction_limits)->with_native_supports(supports);
    allocations.retain(result, native_used, record_cap);
    const auto canonical = canonical_cell_key(*result);
    if (canonical.size() != envelope.body.size() ||
        !std::equal(canonical.begin(), canonical.end(), envelope.body.begin()))
        on::reject("noncanonical cell certificate body.");
    return result;
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::with_native_supports(
    std::span<const PlaneSupportIdentity<ArithmeticMode::Ordinary>> supports) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    const auto old = details::checked_geometry_storage<Mode>(geometry_->impl_);
    if (supports.size() != old->surfaces.size())
        details::reject(Mode, "native support count differs from cell faces.");
    details::Operation scope(old->limits);
    details::active_work->retain_geometry<Mode>(geometry_);
    auto copy = details::clone_geometry_metadata<Mode>(*old);
    auto topology = details::clone_topology<Mode>(*old->topology);
    // Every face keeps its existing slot; no identity-map allocation or owner merge is needed.
    for (std::size_t f = 0; f < old->surfaces.size(); ++f)
    {
        const auto slot = std::visit(
            [&](const auto& arrays) { return static_cast<std::size_t>(arrays.surface_planes[f]); }, topology->arrays);
        details::active_work->retain_identity(supports[f]);
        auto plane = details::clone_plane<Mode>(*old->surfaces[f].plane);
        plane->native_support = supports[f];
        topology->planes[slot] = std::move(plane);
    }
    details::bind_topology<Mode>(*copy, std::move(topology));
    details::active_work->charge(1, sizeof(DynamicConvexGeometry));
    auto result = from_dynamic(DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(copy))));
    DynamicGeometryAccess::retain_prepaid_native_binding<Mode>(result, details::active_work->allocations,
        details::active_work->bytes, details::active_work->limits.byte_budget);
    return result;
}

template<ArithmeticMode Mode>
std::vector<ConvexOctantPiece<Mode>> ConvexCellGeometry<Mode>::split_aabb_midpoints(double minimum) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    if (!std::isfinite(minimum) || minimum < 0)
        details::reject(Mode, "invalid finite split quality bound.");
    const auto p = local_points();
    std::array<long double, 3> lo = p.front().value, hi = lo;
    for (const auto& v : p)
        for (std::size_t a = 0; a < 3; ++a)
        {
            lo[a] = std::min(lo[a], v.value[a]);
            hi[a] = std::max(hi[a], v.value[a]);
        }
    GeometryPoint mid{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        const auto nominal = std::midpoint(lo[a], hi[a]);
        mid[a] = GeometryValue<Mode>{nominal}.as_double();
        if (!(mid[a] > lo[a] && mid[a] < hi[a]))
            details::reject(Mode, "finite split midpoint is not interior.");
    }
    const auto root = details::checked_geometry_storage<Mode>(geometry_->impl_);
    std::vector<ConvexOctantPiece<Mode>> output;
    std::vector<ConvexGeometryPtr<Mode>> pieces;
    for (unsigned octant = 0; octant < 8; ++octant)
    {
        auto child = from_dynamic(geometry_);
        for (std::size_t a = 0; a < 3 && child; ++a)
        {
            GeometryHalfspace plane;
            plane.anchor = origin();
            plane.normal[a] = ((octant >> a) & 1) ? -1 : 1;
            plane.offset = plane.normal[a] * mid[a];
            child = child->clip(plane, false).inside;
        }
        if (!child)
            continue;
        if (!child->quality_above(minimum))
            details::reject(Mode, "positive finite child violates scaled-volume quality.");
        const auto stored = details::checked_geometry_storage<Mode>(child->geometry_->impl_);
        std::vector<ConvexFaceOrigin> origins;
        for (const auto& surface : stored->surfaces)
        {
            bool found = false;
            for (std::size_t f = 0; f < root->surfaces.size(); ++f)
                if (surface.plane == root->surfaces[f].plane)
                {
                    origins.push_back({ConvexFaceOrigin::Kind::ParentFace, f});
                    found = true;
                    break;
                }
            if (found)
                continue;
            if (!surface.plane->input)
                details::reject(Mode, "finite split lost face ancestry.");
            const auto& cut = *surface.plane->input;
            for (std::size_t a = 0; a < 3; ++a)
                if (cut.normal[a] != 0)
                {
                    origins.push_back({ConvexFaceOrigin::Kind::CutPlane, a});
                    found = true;
                    break;
                }
            if (!found)
                details::reject(Mode, "finite split lost cut axis.");
        }
        pieces.push_back(child);
        output.push_back({octant, std::move(child), std::move(origins)});
    }
    certify_partition(pieces);
    return output;
}
} // namespace SimpleFluid::PLIC
