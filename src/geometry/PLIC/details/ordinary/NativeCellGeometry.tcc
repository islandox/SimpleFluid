/** @file NativeCellGeometry.tcc
 * @brief Native finite cell import, refinement, and source-coordinate packets.
 * Included after the common typed definitions by the compiled construction provider.
 */
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
PlanarFacePtr<Mode> ConvexCellGeometry<Mode>::face_geometry(std::size_t face) const
    requires(Mode == ArithmeticMode::Ordinary)
{
    const auto vertices = points();
    std::vector<std::array<long double, 3>> loop;
    for (auto i : faces().at(face))
        loop.push_back(vertices.at(i).value);
    const auto stored = details::checked_geometry_storage<Mode>(geometry_->impl_);
    const auto& surface = stored->surfaces.at(face);
    std::array<long double, 3> normal;
    for (std::size_t axis = 0; axis < 3; ++axis)
        normal[axis] = surface.orientation * surface.plane->normal[axis].value;
    auto support = surface.plane->native_support;
    if (!support)
        support = details::ordinary_native::source_support_packet(loop);
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
std::vector<char> ConvexCellGeometry<Mode>::encode_certificate() const
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    on::Writer w;
    w.word(0x5346465043454c32ULL);
    w.point(on::wide(origin()));
    const auto vertices = local_points();
    w.word(vertices.size());
    for (const auto& p : vertices)
        w.point(p.value);
    w.word(faces().size());
    const auto stored = details::checked_geometry_storage<Mode>(geometry_->impl_);
    for (std::size_t index = 0; index < faces().size(); ++index)
    {
        const auto& f = faces()[index];
        w.word(f.size());
        for (auto i : f)
            w.word(i);
        auto support = stored->surfaces.at(index).plane->native_support;
        if (!support)
            support = planar_support<Mode>(*face_geometry(index));
        w.packet(details::GeometryIdentityAccess::key(*support));
    }
    return std::move(w.bytes);
}
template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::decode_certificate(std::span<const char> bytes)
    requires(Mode == ArithmeticMode::Ordinary)
{
    namespace on = details::ordinary_native;
    on::Reader r{bytes};
    if (r.word() != 0x5346465043454c32ULL)
        on::reject("foreign finite cell packet mode.");
    const auto origin = on::display(r.point());
    const auto n = r.count(4096);
    std::vector<Point<Mode>> p;
    p.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        p.emplace_back(r.point());
    const auto nf = r.count(4096);
    std::vector<std::vector<std::size_t>> faces;
    faces.reserve(nf);
    std::vector<PlaneSupportIdentity<ArithmeticMode::Ordinary>> supports;
    for (std::size_t f = 0; f < nf; ++f)
    {
        const auto count = r.count(n);
        std::vector<std::size_t> face;
        face.reserve(count);
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
        supports.push_back(decode_identity<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(support));
    }
    r.finish();
    return make_local(origin, p, faces)->with_native_supports(supports);
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
