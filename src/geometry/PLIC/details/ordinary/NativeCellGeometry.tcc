/** @file NativeCellGeometry.tcc
 * @brief Native finite cell import, refinement, and source-coordinate packets.
 */
#include "geometry/PLIC/PlanarGeometry.hh"
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
        return mesh.certified_sparse_cell_geometry(cell);
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
    shell.outward_faces = faces();
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
        return !geometry.numerical_only && !geometry.partition_proof_only && !geometry.construction &&
               !geometry.partition_certificate && !geometry.export_source && !geometry.export_seal &&
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
    add(1, sizeof(DynamicConvexGeometry) + sizeof(ConvexCellGeometry));
    return bytes;
}

namespace native_cell_detail
{
template<class Writer> void emit_cell_key(Writer& w, const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell)
{
    constexpr auto Mode = ArithmeticMode::Ordinary;
    namespace on = details::ordinary_native;
    w.tag(GeometryTag::OrdinaryCell);
    w.point(on::wide(cell.origin()));
    const auto vertices = cell.local_points();
    w.word(vertices.size());
    for (const auto& p : vertices)
        w.point(p.value);
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
            support = planar_support<Mode>(*cell.face_geometry(index));
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
    require(1, sizeof(details::GeometryStorage<Mode>));
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
    auto copy = std::make_shared<details::GeometryStorage<Mode>>(*old);
    std::map<details::StoredPlanePtr<Mode>, details::StoredPlanePtr<Mode>> replacement;
    for (std::size_t f = 0; f < copy->surfaces.size(); ++f)
    {
        const auto source = copy->surfaces[f].plane;
        auto plane = std::make_shared<details::StoredPlane<Mode>>(*source);
        details::active_work->retain_identity(supports[f]);
        plane->native_support = supports[f];
        replacement[source] = plane;
        copy->surfaces[f].plane = std::move(plane);
    }
    for (auto& vertex : copy->vertices)
        for (auto& plane : vertex.incident)
            if (replacement.contains(plane))
                plane = replacement.at(plane);
    auto result = from_dynamic(DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(copy))));
    details::active_work->allocations.retain(
        result, details::active_work->bytes, details::active_work->limits.byte_budget);
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
