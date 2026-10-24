#include "geometry/details/ConvexCellGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
ExactPlanarFacePtr ConvexCellGeometry::face_geometry(std::size_t face) const
{
    auto result = std::make_shared<ExactPlanarFace::Impl>();
    for (auto vertex : impl_->poly.faces.at(face))
        result->polygon.push_back(impl_->poly.vertices.at(vertex));
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(result)));
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
void validate_planar_polygon(const Polygon& polygon)
{
    if (polygon.size() < 3)
        throw std::invalid_argument("Exact planar face needs at least three vertices.");
    const auto normal = polygon_vector(polygon);
    if (zero(normal))
        throw std::invalid_argument("Exact planar face is degenerate.");
    std::set<P> unique;
    for (const auto& p : polygon)
        if (!unique.insert(p).second || dot(normal, sub(p, polygon[0])) != 0)
            throw std::invalid_argument("Exact planar face repeats vertices or is nonplanar.");
    for (std::size_t e = 0; e < polygon.size(); ++e)
    {
        const auto a = polygon[e], d = sub(polygon[(e + 1) % polygon.size()], a);
        const Q length = dot(d, d);
        for (const auto& p : polygon)
        {
            const auto q = sub(p, a);
            if (dot(normal, cross(d, q)) < 0)
                throw std::invalid_argument("Exact planar face is nonconvex/nonsimple.");
            const Q along = dot(q, d);
            if (p != a && p != polygon[(e + 1) % polygon.size()] && zero(cross(d, q)) && along > 0 && along < length)
                throw std::invalid_argument("Exact planar face has overlapping collinear edges.");
        }
    }
    (void) square_root(dot(normal, normal)).as_double();
    (void) rounded_point(polygon_center(polygon));
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
ExactPlanarFacePtr ExactPlanarFace::make(std::span<const GeometryPoint> vertices)
{
    auto impl = std::make_shared<Impl>();
    for (const auto& p : vertices)
        impl->polygon.push_back(point(p));
    validate_planar_polygon(impl->polygon);
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(impl)));
}

CertifiedGeometryValue ExactPlanarFace::area() const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    const auto n = polygon_vector(impl_->polygon);
    return square_root(dot(n, n));
}

MeshUtils::Vec3 ExactPlanarFace::centroid() const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    return rounded_point(polygon_center(impl_->polygon));
}

MeshUtils::Vec3 ExactPlanarFace::area_vector() const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.face_measure_evaluations;
    return rounded_point(polygon_vector(impl_->polygon));
}

ExactPlanarFacePtr ExactPlanarFace::reversed() const
{
    auto impl = std::make_shared<Impl>(*impl_);
    std::reverse(impl->polygon.begin(), impl->polygon.end());
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(impl)));
}

ExactPlanarFacePtr ExactPlanarFace::translated(GeometryPoint displacement) const
{
    const auto shift = point(displacement);
    auto impl = std::make_shared<Impl>(*impl_);
    for (auto& p : impl->polygon)
        p = add(p, shift);
    validate_planar_polygon(impl->polygon);
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(impl)));
}

ExactPlanarFacePtr ExactPlanarFace::oriented_like(GeometryPoint vector) const
{
    const auto sign = dot(polygon_vector(impl_->polygon), point(vector));
    if (sign == 0)
        throw std::invalid_argument("Exact planar orientation is unresolved.");
    if (sign < 0)
        return reversed();
    return ExactPlanarFacePtr(new ExactPlanarFace(impl_));
}

bool ExactPlanarFace::oppositely_oriented(const ExactPlanarFace& other) const
{
    return dot(polygon_vector(impl_->polygon), polygon_vector(other.impl_->polygon)) < 0;
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace exact_geometry_detail
{
P authority_point(std::span<const char> packet)
{
    WireReader r{packet};
    if (r.word() != 0x5346504f494e5431ULL)
        throw std::invalid_argument("Invalid exact authority point.");
    P p;
    for (auto& q : p)
        q = r.rational();
    r.finish();
    return p;
}

P displacement_point(std::span<const char> packet)
{
    if (packet.empty())
        return {};
    WireReader r{packet};
    if (r.word() != 0x5346444953503631ULL)
        throw std::invalid_argument("Invalid exact displacement certificate.");
    P p;
    for (auto& q : p)
        q = r.rational();
    r.finish();
    return p;
}

std::vector<char> displacement_packet(const P& p)
{
    if (zero(p))
        return {};
    WireWriter w;
    w.word(0x5346444953503631ULL);
    for (const auto& q : p)
        w.rational(q);
    return std::move(w.data);
}

std::vector<char> point_packet(const P& p)
{
    WireWriter w;
    w.word(0x5346504f494e5431ULL);
    for (const auto& q : p)
        w.rational(q);
    return std::move(w.data);
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
std::vector<char> exact_displacement(GeometryPoint from, GeometryPoint to)
{
    for (unsigned a = 0; a < 3; ++a)
        if (!std::isfinite(from[a]) || !std::isfinite(to[a]))
            throw std::invalid_argument("Exact displacement needs finite represented endpoints.");
    return displacement_packet(sub(point(to), point(from)));
}

std::vector<char> exact_displacement_difference(std::span<const char> first, std::span<const char> second)
{
    return displacement_packet(sub(displacement_point(first), displacement_point(second)));
}

GeometryPoint exact_displacement_display(std::span<const char> packet)
{
    const auto p = rounded_point(displacement_point(packet));
    return {p.x, p.y, p.z};
}

std::vector<char> exact_point_translated(std::span<const char> identity, std::span<const char> displacement)
{
    return point_packet(add(authority_point(identity), displacement_point(displacement)));
}

std::vector<char> exact_periodic_point(
    std::span<const char> identity, GeometryPoint lower, GeometryPoint upper, std::array<bool, 3> periodic)
{
    auto p = authority_point(identity);
    for (unsigned a = 0; a < 3; ++a)
        if (periodic[a])
        {
            if (!std::isfinite(lower[a]) || !std::isfinite(upper[a]) || lower[a] >= upper[a])
                throw std::invalid_argument("Invalid periodic root interval.");
            if (p[a] == exact(upper[a]))
                p[a] = exact(lower[a]);
        }
    return point_packet(p);
}

ExactPlanarFacePtr ExactPlanarFace::translated_exact(std::span<const char> displacement) const
{
    auto translated = std::make_shared<Impl>();
    const auto shift = displacement_point(displacement);
    for (const auto& p : impl_->polygon)
        translated->polygon.push_back(add(p, shift));
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(translated)));
}

GeometryPoint exact_point_display(std::span<const char> identity)
{
    const auto point = rounded_point(authority_point(identity));
    return {point.x, point.y, point.z};
}

std::array<CertifiedGeometryValue, 3> numerical_point(std::span<const char> identity)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.source_coordinate_exports;
    const auto point = authority_point(identity);
    std::array<CertifiedGeometryValue, 3> result;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result[axis] = measured(point[axis], true);
    return result;
}

ExactEdgePointIndex::ExactEdgePointIndex(std::span<const std::vector<char>> points)
{
    auto index = std::make_shared<Impl>();
    std::set<P> unique;
    for (const auto& point : points)
        unique.insert(authority_point(point));
    if (unique.size() < 2)
        throw std::invalid_argument("Exact edge index needs distinct segment endpoints.");
    index->origin = *unique.begin();
    index->direction = sub(*unique.rbegin(), index->origin);
    while (index->direction[index->axis] == 0)
        ++index->axis;
    for (const auto& packet : points)
    {
        const auto p = authority_point(packet);
        if (!zero(cross(sub(p, index->origin), index->direction)))
            throw std::invalid_argument("Edge authority point lies off its exact line.");
        index->points.emplace(p[index->axis], packet);
    }
    impl_ = std::move(index);
}

std::vector<std::vector<char>> ExactEdgePointIndex::closed_segment(
    std::span<const char> a, std::span<const char> b) const
{
    const auto first = authority_point(a), second = authority_point(b);
    if (first == second || !zero(cross(sub(first, impl_->origin), impl_->direction)) ||
        !zero(cross(sub(second, impl_->origin), impl_->direction)))
        throw std::invalid_argument("Invalid authority segment query.");
    const Q lo = std::min(first[impl_->axis], second[impl_->axis]),
            hi = std::max(first[impl_->axis], second[impl_->axis]);
    std::vector<std::vector<char>> result;
    for (auto it = impl_->points.lower_bound(lo); it != impl_->points.end() && it->first <= hi; ++it)
        result.push_back(it->second);
    return result;
}

std::size_t ExactEdgePointIndex::storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + sizeof(Impl);
    for (const auto& [key, value] : impl_->points)
        bytes += sizeof(key) + sizeof(value) + value.capacity() + 3 * sizeof(void*);
    return bytes;
}

ExactPlanarFacePtr ExactPlanarFace::intersection(const ExactPlanarFace& other) const
{
    const auto normal = polygon_vector(impl_->polygon);
    for (const auto& point : other.impl_->polygon)
        if (dot(normal, sub(point, impl_->polygon[0])) != 0)
            throw std::invalid_argument("Authority faces are not coplanar.");
    auto clipper = other.impl_->polygon;
    if (dot(polygon_vector(clipper), normal) < 0)
        std::reverse(clipper.begin(), clipper.end());
    auto polygon = polygon_intersection(impl_->polygon, clipper, normal);
    if (polygon.empty())
        return {};
    auto result = std::make_shared<Impl>();
    result->polygon = std::move(polygon);
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(result)));
}

std::vector<std::vector<char>> ExactPlanarFace::vertex_identities() const
{
    std::vector<std::vector<char>> result;
    for (const auto& p : impl_->polygon)
    {
        WireWriter w;
        w.word(0x5346504f494e5431ULL);
        for (const auto& q : p)
            w.rational(q);
        result.push_back(std::move(w.data));
    }
    return result;
}

std::vector<std::vector<char>> ExactPlanarFace::edge_support_identities() const
{
    std::vector<std::vector<char>> result;
    for (std::size_t i = 0; i < impl_->polygon.size(); ++i)
    {
        const auto& p = impl_->polygon[i];
        auto d = sub(impl_->polygon[(i + 1) % impl_->polygon.size()], p);
        std::size_t axis = 0;
        while (d[axis] == 0)
            ++axis;
        const Q factor = d[axis];
        for (auto& q : d)
            q /= factor;
        const auto moment = cross(p, d);
        WireWriter w;
        w.word(0x53464c494e453631ULL);
        for (const auto& q : d)
            w.rational(q);
        for (const auto& q : moment)
            w.rational(q);
        result.push_back(std::move(w.data));
    }
    return result;
}

ExactPlanarFacePtr ExactPlanarFace::with_edge_vertices(std::span<const std::vector<char>> points) const
{
    std::set<P> candidates;
    for (const auto& packet : points)
    {
        WireReader r{packet};
        if (r.word() != 0x5346504f494e5431ULL)
            throw std::invalid_argument("Invalid exact authority point.");
        P p;
        for (auto& q : p)
            q = r.rational();
        r.finish();
        candidates.insert(p);
    }
    auto result = std::make_shared<Impl>();
    for (std::size_t i = 0; i < impl_->polygon.size(); ++i)
    {
        const auto& a = impl_->polygon[i];
        const auto& b = impl_->polygon[(i + 1) % impl_->polygon.size()];
        const auto d = sub(b, a);
        const Q length = dot(d, d);
        std::map<Q, P> interior;
        for (const auto& p : candidates)
        {
            const auto offset = sub(p, a);
            const Q along = dot(offset, d);
            if (zero(cross(offset, d)) && along > 0 && along < length)
                interior.emplace(along, p);
        }
        result->polygon.push_back(a);
        for (const auto& [distance, p] : interior)
            result->polygon.push_back(p);
    }
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(result)));
}

std::vector<char> exact_planar_support(const ExactPlanarFace& face)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.source_support_exports;
    auto normal = polygon_vector(face.impl_->polygon);
    std::size_t axis = 0;
    while (normal[axis] == 0)
        ++axis;
    const Q divisor = normal[axis];
    for (auto& q : normal)
        q /= divisor;
    WireWriter w;
    w.word(0x53464a504c414e36ULL);
    for (const auto& q : normal)
        w.rational(q);
    w.rational(dot(normal, face.impl_->polygon[0]));
    return std::move(w.data);
}

int exact_planar_orientation(const ExactPlanarFace& face)
{
    const auto normal = polygon_vector(face.impl_->polygon);
    for (const auto& q : normal)
        if (q != 0)
            return q > 0 ? 1 : -1;
    throw std::invalid_argument("Degenerate mixed root support.");
}

std::vector<ExactPlanarFacePtr> subtract_planar_faces(
    const ExactPlanarFace& source, std::span<const ExactPlanarFacePtr> removed)
{
    const auto normal = polygon_vector(source.impl_->polygon);
    auto half = [](const Polygon& polygon, const Plane& plane)
    {
        Polygon output;
        auto append = [&](const P& p)
        {
            if (output.empty() || output.back() != p)
                output.push_back(p);
        };
        for (std::size_t i = 0; i < polygon.size(); ++i)
        {
            const auto& a = polygon[i];
            const auto& b = polygon[(i + 1) % polygon.size()];
            const Q da = distance(plane, a), db = distance(plane, b);
            if (da <= 0)
                append(a);
            if ((da < 0 && db > 0) || (da > 0 && db < 0))
                append(add(a, mul(sub(b, a), Q(da / (da - db)))));
        }
        if (output.size() > 1 && output.front() == output.back())
            output.pop_back();
        return minimal_polygon(std::move(output));
    };
    std::vector<Polygon> pieces{source.impl_->polygon};
    for (const auto& cutter : removed)
    {
        if (!cutter || !source.contains(*cutter))
            throw std::invalid_argument("Mixed selected root face is not wholly contained.");
        auto boundary = cutter->impl_->polygon;
        if (dot(polygon_vector(boundary), normal) < 0)
            std::reverse(boundary.begin(), boundary.end());
        std::vector<Polygon> next;
        for (auto piece : pieces)
        {
            for (std::size_t e = 0; e < boundary.size() && !piece.empty(); ++e)
            {
                const Plane inside{cross(sub(boundary[(e + 1) % boundary.size()], boundary[e]), normal), boundary[e]};
                Plane outside = inside;
                outside.normal = mul(outside.normal, Q(-1));
                auto remainder = half(piece, outside);
                if (!remainder.empty())
                    next.push_back(std::move(remainder));
                piece = half(piece, inside);
            }
        }
        pieces = std::move(next);
    }
    std::vector<ExactPlanarFacePtr> result, coverage(removed.begin(), removed.end());
    for (auto& polygon : pieces)
    {
        auto impl = std::make_shared<ExactPlanarFace::Impl>();
        impl->polygon = std::move(polygon);
        auto face = ExactPlanarFacePtr(new ExactPlanarFace(std::move(impl)));
        coverage.push_back(face);
        result.push_back(std::move(face));
    }
    source.certify_partition(coverage);
    return result;
}

bool ExactPlanarFace::contains_exact_point(std::span<const char> identity) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.source_membership_predicates;
    const auto p = authority_point(identity), normal = polygon_vector(impl_->polygon);
    if (dot(normal, sub(p, impl_->polygon[0])) != 0)
        return false;
    for (std::size_t e = 0; e < impl_->polygon.size(); ++e)
        if (dot(normal, cross(sub(impl_->polygon[(e + 1) % impl_->polygon.size()], impl_->polygon[e]),
                            sub(p, impl_->polygon[e]))) < 0)
            return false;
    return true;
}

bool ExactPlanarFace::cell_bounds_overlap(const ConvexCellGeometry& cell) const
{
    if (cell.impl_->poly.vertices.empty())
        return false;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        Q flo = impl_->polygon[0][axis], fhi = flo;
        Q clo = cell.impl_->poly.vertices[0][axis], chi = clo;
        for (const auto& p : impl_->polygon)
        {
            flo = std::min(flo, p[axis]);
            fhi = std::max(fhi, p[axis]);
        }
        for (const auto& p : cell.impl_->poly.vertices)
        {
            clo = std::min(clo, p[axis]);
            chi = std::max(chi, p[axis]);
        }
        if (fhi < clo || chi < flo)
            return false;
    }
    return true;
}

bool ExactPlanarFace::contains(const ExactPlanarFace& other) const
{
    const auto n = polygon_vector(impl_->polygon);
    for (const auto& p : other.impl_->polygon)
    {
        if (dot(n, sub(p, impl_->polygon[0])) != 0)
            return false;
        for (std::size_t e = 0; e < impl_->polygon.size(); ++e)
            if (dot(n, cross(sub(impl_->polygon[(e + 1) % impl_->polygon.size()], impl_->polygon[e]),
                           sub(p, impl_->polygon[e]))) < 0)
                return false;
    }
    return true;
}

bool ExactPlanarFace::equivalent(const ExactPlanarFace& other) const
{
    return contains(other) && other.contains(*this);
}

void ExactPlanarFace::certify_partition(std::span<const ExactPlanarFacePtr> tiles) const
{
    const auto normal = polygon_vector(impl_->polygon);
    std::size_t axis = 0;
    while (normal[axis] == 0)
        ++axis;
    const auto origin = impl_->polygon[0];
    const auto expected = polygon_moments(impl_->polygon, axis, origin);
    FaceMoment sum{};
    std::vector<Polygon> pieces;
    for (const auto& tile : tiles)
    {
        if (!tile || !contains(*tile))
            throw std::invalid_argument("Whole native fine face is not contained in its exact coarse face.");
        auto polygon = tile->impl_->polygon;
        if (dot(polygon_vector(polygon), normal) < 0)
            std::reverse(polygon.begin(), polygon.end());
        for (const auto& previous : pieces)
            if (!polygon_intersection(polygon, previous, normal).empty())
                throw std::invalid_argument("Exact nested face tiles overlap.");
        const auto m = polygon_moments(polygon, axis, origin);
        sum.area += m.area;
        sum.first = add(sum.first, m.first);
        pieces.push_back(std::move(polygon));
    }
    if (sum.area != expected.area || sum.first != expected.first)
        throw std::invalid_argument("Exact nested face tiles have a represented coverage gap.");
}

std::vector<char> ExactPlanarFace::encode() const
{
    WireWriter w;
    w.word(0x5346464143454331ULL);
    w.word(1);
    auto polygon = minimal_polygon(impl_->polygon);
    w.word(polygon.size());
    for (const auto& p : polygon)
        for (const auto& q : p)
            w.rational(q);
    return std::move(w.data);
}

std::size_t ExactPlanarFace::storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + sizeof(Impl) + impl_->polygon.capacity() * sizeof(P);
    for (const auto& p : impl_->polygon)
        for (const auto& q : p)
            bytes += (q.backend().num().capacity() + q.backend().denom().capacity()) *
                     sizeof(boost::multiprecision::limb_type);
    return bytes;
}

std::vector<char> ExactPlanarFace::encode_incidence() const
{
    WireWriter w;
    w.word(0x5346464143454331ULL);
    w.word(1);
    w.word(impl_->polygon.size());
    for (const auto& p : impl_->polygon)
        for (const auto& q : p)
            w.rational(q);
    return std::move(w.data);
}

ExactPlanarFacePtr ExactPlanarFace::decode(std::span<const char> packet)
{
    WireReader r{packet};
    if (r.word() != 0x5346464143454331ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact face certificate.");
    auto impl = std::make_shared<Impl>();
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        P p;
        for (auto& q : p)
            q = r.rational();
        impl->polygon.push_back(std::move(p));
    }
    r.finish();
    validate_planar_polygon(impl->polygon);
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(impl)));
}

ConvexGeometryPtr ExactPlanarFace::cell_shell(std::span<const ExactPlanarFacePtr> faces)
{
    auto geometry = std::make_shared<ConvexCellGeometry::Impl>();
    std::map<P, std::size_t> vertices;
    for (const auto& face : faces)
    {
        if (!face)
            throw std::invalid_argument("Exact cell shell needs all authoritative face certificates.");
        Face loop;
        for (const auto& p : face->impl_->polygon)
        {
            const auto [it, inserted] = vertices.emplace(p, geometry->poly.vertices.size());
            if (inserted)
                geometry->poly.vertices.push_back(p);
            loop.push_back(it->second);
        }
        geometry->poly.faces.push_back(std::move(loop));
    }
    // Canonical native fine faces can introduce points into neighboring coarse
    // edges. Propagate those exact identities before checking the closed shell.
    for (auto& face : geometry->poly.faces)
    {
        Face propagated;
        for (std::size_t e = 0; e < face.size(); ++e)
        {
            const auto a = face[e], b = face[(e + 1) % face.size()];
            const auto d = sub(geometry->poly.vertices[b], geometry->poly.vertices[a]);
            const Q length = dot(d, d);
            std::vector<std::pair<Q, std::size_t>> interior;
            for (std::size_t n = 0; n < geometry->poly.vertices.size(); ++n)
            {
                const auto q = sub(geometry->poly.vertices[n], geometry->poly.vertices[a]);
                const Q t = dot(q, d);
                if (zero(cross(q, d)) && t > 0 && t < length)
                    interior.emplace_back(t, n);
            }
            std::sort(interior.begin(), interior.end());
            propagated.push_back(a);
            for (const auto& [t, n] : interior)
                propagated.push_back(n);
        }
        face = std::move(propagated);
    }
    validate(geometry->poly);
    const auto origin = rounded_point(geometry->poly.vertices[0]);
    geometry->origin = {origin.x, origin.y, origin.z};
    geometry->product = product_description(geometry->poly);
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(geometry)));
}

ExactPlanarFacePtr ExactExtrudedConstruction::certified_face_geometry(std::size_t face) const
{
    auto result = std::make_shared<ExactPlanarFace::Impl>();
    for (auto node : impl_->topology->face_nodes(face))
    {
        auto n = impl_->topology->indexer().node_id(node);
        auto p = impl_->base->points[impl_->base->active.at(n.ij)];
        p[2] = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + n.k)];
        result->polygon.push_back(std::move(p));
    }
    const auto owner = impl_->topology->owner_cell(face);
    const auto id = impl_->topology->indexer().cell_id(owner);
    auto center = impl_->base->centers.at(id.ij);
    center[2] = (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)] +
                    impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) /
                2;
    if (dot(polygon_vector(result->polygon), sub(result->polygon[0], center)) < 0)
        std::reverse(result->polygon.begin(), result->polygon.end());
    validate_planar_polygon(result->polygon);
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(result)));
}

ExactPlanarFacePtr ExactPolyhedralComplex::face_geometry(std::uint64_t face) const
{
    auto result = std::make_shared<ExactPlanarFace::Impl>();
    for (auto node : impl_->faces.at(face).nodes)
        result->polygon.push_back(impl_->nodes.at(node).point);
    validate_planar_polygon(result->polygon);
    return ExactPlanarFacePtr(new ExactPlanarFace(std::move(result)));
}

ExactPlanarFacePtr ExactPlanarFace::from_native_face(const MultiRegionMesh& mesh, std::uint64_t face)
{
    if (mesh.sparse_native())
        return mesh.certified_sparse_face_geometry(face);
    (void) mesh.geometry_epoch();
    const auto native = mesh.native_face(face);
    const auto& region = mesh.regions().at(native.region);
    auto result = std::visit(
        [&](const auto& r) -> ExactPlanarFacePtr
        {
            if constexpr (requires { r.geometry().certified_face_geometry(native.face); })
                return r.geometry().certified_face_geometry(native.face);
            else
                return {};
        },
        region);
    if (!result)
    {
        if (mesh.region_layout(native.region).family == RegionLayout::Family::Cylindrical)
            throw std::invalid_argument("Analytic curved faces require a provider-specific exact certificate.");
        const bool constructed = std::visit(
            [](const auto& r)
            {
                if constexpr (requires { r.geometry().has_exact_construction(); })
                    return r.geometry().has_exact_construction();
                else
                    return false;
            },
            region);
        if (constructed)
            throw std::invalid_argument("Constructed face cannot fall back to its display-node polygon.");
        std::vector<GeometryPoint> vertices;
        for (auto node : mesh.face_nodes(face))
        {
            auto p = mesh.node_coordinates(node);
            vertices.push_back({p.x, p.y, p.z});
        }
        result = make(vertices);
        auto v = mesh.face_area_vector(face);
        return result->oriented_like({v.x, v.y, v.z});
    }
    const auto native_owner = std::visit([&](const auto& r) { return r.topology().owner_cell(native.face); }, region);
    if (mesh.composite_cell(native.region, native_owner) != mesh.owner_cell(face))
        result = result->reversed();
    return result;
}
} // namespace SimpleFluid::Meshes
