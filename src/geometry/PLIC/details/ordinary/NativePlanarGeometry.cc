#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <map>
#include <set>

namespace SimpleFluid::PLIC
{
namespace on = details::ordinary_native;
using OrdinaryFace = PlanarFace<ArithmeticMode::Ordinary>;
using OrdinaryFacePtr = PlanarFacePtr<ArithmeticMode::Ordinary>;
struct OrdinaryFace::Impl
{
    std::vector<on::WidePoint> polygon;
    on::WidePoint normal{};
    std::vector<char> support;
    bool retained_support = false;
};
struct EdgePointIndex<ArithmeticMode::Ordinary>::Impl
{
    std::map<on::WidePoint, std::vector<char>> points;
};
namespace
{
using Polygon = std::vector<on::WidePoint>;
namespace predicates = details::ordinary_predicates;
constexpr std::size_t vertex_cap = 4096;
long double norm(const on::WidePoint& p)
{
    const auto v = std::hypot(p[0], p[1], p[2]);
    on::finite(v);
    return v;
}
std::size_t axis_of(const on::WidePoint& n)
{
    std::size_t axis = 0;
    for (std::size_t i = 1; i < 3; ++i)
        if (std::abs(n[i]) > std::abs(n[axis]))
            axis = i;
    return axis;
}
struct DefiningPlane
{
    std::array<std::size_t, 3> vertices;
    unsigned projection;
    int winding;
};
DefiningPlane defining_plane(const Polygon& p)
{
    bool unresolved = false;
    for (std::size_t i = 1; i + 1 < p.size(); ++i)
    {
        unsigned projection = 0;
        predicates::Result best;
        for (unsigned a = 0; a < 3; ++a)
        {
            const auto area = predicates::projected_orientation(p[0], p[i], p[i + 1], a);
            unresolved = unresolved || area.sign == predicates::Sign::Unresolved;
            if ((area.sign == predicates::Sign::Positive || area.sign == predicates::Sign::Negative) &&
                std::abs(area.value) > std::abs(best.value))
            {
                projection = a;
                best = area;
            }
        }
        if (best.sign == predicates::Sign::Positive || best.sign == predicates::Sign::Negative)
            return {{0, i, i + 1}, projection, predicates::require_sign(best)};
    }
    throw GeometryError(unresolved ? GeometryErrorCode::UnresolvedArithmetic : GeometryErrorCode::InvalidTopology,
        "Ordinary native face has no resolved noncollinear defining triangle.");
}
bool on_plane(const Polygon& p, const DefiningPlane& plane, const on::WidePoint& v)
{
    // Incidence of the defining triangle is structural, independent of the
    // rounding in a separately computed normal. No proximity test is used.
    for (const auto index : plane.vertices)
        if (v == p[index])
            return true;
    const auto& t = plane.vertices;
    return predicates::require_sign(predicates::orientation({p[t[0]], p[t[1]], p[t[2]], v})) == 0;
}
void validate(const Polygon& p)
{
    if (p.size() > vertex_cap)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary native planar vertex cap exhausted.");
    if (p.size() < 3)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary native planar polygon has too few vertices.");
    std::set<on::WidePoint> seen;
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
bool coplanar(const Polygon& a, const Polygon& b)
{
    const auto plane = defining_plane(a);
    return std::all_of(b.begin(), b.end(), [&](const auto& p) { return on_plane(a, plane, p); });
}
bool point_inside(const Polygon& poly, const on::WidePoint& p, bool check_plane = true)
{
    // A represented boundary vertex has structural incidence, including for
    // retained native supports. Keep this query shortcut out of on_plane:
    // polygon validation must still establish planarity of every input vertex.
    if (std::find(poly.begin(), poly.end(), p) != poly.end())
        return true;
    const auto plane = defining_plane(poly);
    if (check_plane && !on_plane(poly, plane, p))
        return false;
    for (std::size_t i = 0; i < poly.size(); ++i)
        if (plane.winding * predicates::require_sign(predicates::projected_orientation(
                                poly[i], poly[(i + 1) % poly.size()], p, plane.projection)) <
            0)
            return false;
    return true;
}
Polygon clip_loop(
    const Polygon& source, const on::WidePoint& a, const on::WidePoint& b, const on::WidePoint& n, bool inside)
{
    Polygon result;
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
            // Identical endpoint order gives neighboring polygons identical intersections.
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
Polygon intersect_loop(Polygon a, const Polygon& b, bool retained_support = false)
{
    if (!retained_support && !coplanar(a, b))
        on::reject("native planar intersection has distinct supports.");
    const auto n = on::polygon_vector(b);
    for (std::size_t i = 0; i < b.size() && !a.empty(); ++i)
        a = clip_loop(a, b[i], b[(i + 1) % b.size()], n, true);
    return a;
}
on::WidePoint center(const Polygon& p)
{
    const auto n = on::polygon_vector(p);
    const auto axis = axis_of(n);
    on::WidePoint weighted{};
    long double total = 0;
    for (std::size_t i = 1; i + 1 < p.size(); ++i)
    {
        const auto v = on::cross(on::sub(p[i], p[0]), on::sub(p[i + 1], p[0]));
        total += v[axis];
        weighted = on::add(weighted, on::mul(on::add(on::add(p[0], p[i]), p[i + 1]), v[axis] / 3));
    }
    if (total == 0)
        on::reject("native polygon centroid has zero area.");
    return on::mul(weighted, 1 / total);
}
::SimpleFluid::MeshUtils::Vec3 vector(const on::WidePoint& p)
{
    const auto d = on::display(p);
    return {d[0], d[1], d[2]};
}
std::vector<char> line_packet(on::WidePoint a, on::WidePoint b)
{
    auto d = on::sub(b, a);
    const auto axis = axis_of(d);
    if (d[axis] == 0)
        on::reject("zero native edge.");
    d = on::mul(d, 1 / d[axis]);
    a = on::sub(a, on::mul(d, a[axis]));
    on::Writer w;
    w.word(0x534646504c494e31ULL);
    w.word(axis);
    w.point(d);
    w.point(a);
    return std::move(w.bytes);
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
    impl->support = on::source_support_packet(impl->polygon);
    return OrdinaryFacePtr(new OrdinaryFace(std::move(impl)));
}
OrdinaryFacePtr OrdinaryFace::from_retained_support(
    std::span<const WidePoint> vertices, WidePoint normal, std::vector<char> support)
{
    if (vertices.size() < 3 || vertices.size() > vertex_cap || support.empty() || !(norm(normal) > 0))
        on::reject("invalid retained native face support.");
    auto impl = std::make_shared<Impl>();
    impl->polygon.assign(vertices.begin(), vertices.end());
    impl->normal = normal;
    impl->support = std::move(support);
    impl->retained_support = true;
    std::set<WidePoint> unique;
    for (const auto& p : impl->polygon)
    {
        for (auto value : p)
            on::finite(value);
        if (!unique.insert(p).second)
            on::reject("repeated retained native face vertex.");
    }
    const auto projected = on::polygon_vector(impl->polygon);
    const auto axis = axis_of(normal);
    if (projected[axis] == 0)
        on::reject("retained native face has zero projected area.");
    const auto orientation = projected[axis] < 0 ? -1.L : 1.L;
    for (std::size_t edge = 0; edge < impl->polygon.size(); ++edge)
        for (const auto& p : impl->polygon)
            if (orientation * on::cross(on::sub(impl->polygon[(edge + 1) % impl->polygon.size()], impl->polygon[edge]),
                                  on::sub(p, impl->polygon[edge]))[axis] <
                0)
                on::reject("nonconvex retained native face loop.");
    return OrdinaryFacePtr(new OrdinaryFace(std::move(impl)));
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
    return {norm(on::polygon_vector(impl_->polygon))};
}
::SimpleFluid::MeshUtils::Vec3 OrdinaryFace::centroid() const
{
    return vector(center(impl_->polygon));
}
::SimpleFluid::MeshUtils::Vec3 OrdinaryFace::area_vector() const
{
    return vector(on::polygon_vector(impl_->polygon));
}
GeometryMeasure<ArithmeticMode::Ordinary> OrdinaryFace::coplanar_area_ratio(
    const OrdinaryFace& denominator, std::size_t budget) const
{
    if (storage_bytes() > budget || denominator.storage_bytes() > budget - storage_bytes())
        on::reject("native face measure byte cap exhausted.");
    if (impl_->support != denominator.impl_->support && !coplanar(impl_->polygon, denominator.impl_->polygon))
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
    return from_retained_support(p, impl_->normal, impl_->support);
}
OrdinaryFacePtr OrdinaryFace::translated(GeometryPoint shift) const
{
    auto p = impl_->polygon;
    const auto delta = on::wide(shift);
    for (auto& v : p)
        v = on::add(v, delta);
    return from_retained_support(p, impl_->normal, on::translated_support_packet(impl_->support, delta));
}
OrdinaryFacePtr OrdinaryFace::translated_packet(std::span<const char> packet) const
{
    auto p = impl_->polygon;
    const auto shift = on::read_displacement(packet);
    for (auto& v : p)
        v = on::add(v, shift);
    return from_retained_support(p, impl_->normal, on::translated_support_packet(impl_->support, shift));
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
    const bool common = impl_->support == other.impl_->support;
    auto p = intersect_loop(impl_->polygon, other.impl_->polygon, common);
    return p.empty() ? OrdinaryFacePtr{} : from_retained_support(p, impl_->normal, impl_->support);
}
std::vector<std::vector<char>> OrdinaryFace::vertex_identities() const
{
    std::vector<std::vector<char>> result;
    for (const auto& p : impl_->polygon)
        result.push_back(on::point_packet(p));
    return result;
}
std::vector<std::vector<char>> OrdinaryFace::edge_support_identities() const
{
    std::vector<std::vector<char>> result;
    const auto& p = impl_->polygon;
    for (std::size_t i = 0; i < p.size(); ++i)
        result.push_back(line_packet(p[i], p[(i + 1) % p.size()]));
    return result;
}
OrdinaryFacePtr OrdinaryFace::with_edge_vertices(std::span<const std::vector<char>> points) const
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
            const auto p = on::read_point(packet), v = on::sub(p, a);
            const auto t = on::dot(v, d);
            if (on::cross(v, d) == on::WidePoint{} && t > 0 && t < length)
                interior.emplace(t, p);
        }
        result.push_back(a);
        for (const auto& [t, p] : interior)
            result.push_back(p);
    }
    return from_retained_support(result, impl_->normal, impl_->support);
}
bool OrdinaryFace::equivalent(const OrdinaryFace& other) const
{
    return contains(other) && other.contains(*this);
}
bool OrdinaryFace::contains_point(std::span<const char> packet) const
{
    const auto p = on::read_point(packet);
    if (std::find(impl_->polygon.begin(), impl_->polygon.end(), p) != impl_->polygon.end())
        return true;
    return point_inside(impl_->polygon, p);
}
bool OrdinaryFace::contains(const OrdinaryFace& other) const
{
    return std::all_of(other.impl_->polygon.begin(), other.impl_->polygon.end(),
        [&](const auto& p) { return point_inside(impl_->polygon, p, impl_->support != other.impl_->support); });
}
bool OrdinaryFace::oppositely_oriented(const OrdinaryFace& other) const
{
    return on::dot(on::polygon_vector(impl_->polygon), on::polygon_vector(other.impl_->polygon)) < 0;
}
bool OrdinaryFace::cell_bounds_overlap(const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell) const
{
    const auto p = cell.points();
    for (std::size_t a = 0; a < 3; ++a)
    {
        auto lo = impl_->polygon[0][a], hi = lo;
        for (const auto& q : impl_->polygon)
        {
            lo = std::min(lo, q[a]);
            hi = std::max(hi, q[a]);
        }
        auto clo = p[0].value[a], chi = clo;
        for (const auto& q : p)
        {
            clo = std::min(clo, q.value[a]);
            chi = std::max(chi, q.value[a]);
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
    std::vector<std::vector<char>> supports;
    for (const auto& face : faces)
        supports.push_back(face->impl_->support);
    return ConvexCellGeometry<ArithmeticMode::Ordinary>::make_points(points, loops)->with_native_supports(supports);
}
void OrdinaryFace::certify_partition(std::span<const OrdinaryFacePtr> tiles) const
{
    constexpr std::size_t byte_budget = 64 * 1024 * 1024, pair_cap = 262144, tile_cap = 65536;
    constexpr long double physical_budget = 8e-15L;
    if (tiles.empty() || tiles.size() > tile_cap)
        on::reject("native planar partition tile cap exhausted.");
    std::size_t bytes = 0, pairs = 0;
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
    std::set<on::WidePoint> vertices(impl_->polygon.begin(), impl_->polygon.end());
    for (std::size_t i = 0; i < tiles.size(); ++i)
    {
        if (!tiles[i] || !contains(*tiles[i]))
            on::reject("native planar partition leaves its source.");
        charge(tiles[i]->points().size(), sizeof(on::WidePoint) + 4 * sizeof(void*));
        vertices.insert(tiles[i]->points().begin(), tiles[i]->points().end());
        for (std::size_t j = 0; j < i; ++j)
        {
            pair();
            if (tiles[i]->intersection(*tiles[j]))
                on::reject("native planar partition interiors overlap.");
        }
    }
    // Compare oriented boundaries after splitting at every admitted represented
    // vertex. No area tolerance can authorize a missing sliver or duplicate edge.
    using Edge = std::pair<on::WidePoint, on::WidePoint>;
    std::map<Edge, std::int64_t> chain;
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
            std::vector<on::WidePoint> points;
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

std::vector<char> OrdinaryFace::encode() const
{
    auto polygon = impl_->polygon;
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
    on::Writer w;
    w.word(0x5346465046414332ULL);
    w.packet(impl_->support);
    w.point(impl_->normal);
    w.word(polygon.size());
    for (const auto& p : polygon)
        w.point(p);
    return std::move(w.bytes);
}
std::vector<char> OrdinaryFace::encode_incidence() const
{
    on::Writer w;
    w.word(0x5346465046414332ULL);
    w.packet(impl_->support);
    w.point(impl_->normal);
    w.word(impl_->polygon.size());
    for (const auto& p : impl_->polygon)
        w.point(p);
    return std::move(w.bytes);
}

std::size_t OrdinaryFace::storage_bytes() const
{
    return sizeof(Impl) + impl_->polygon.capacity() * sizeof(WidePoint) + impl_->support.capacity();
}
OrdinaryFacePtr OrdinaryFace::decode(std::span<const char> bytes)
{
    on::Reader r{bytes};
    if (r.word() != 0x5346465046414332ULL)
        on::reject("foreign native planar packet mode.");
    const auto encoded_support = r.packet();
    std::vector<char> support(encoded_support.begin(), encoded_support.end());
    const auto normal = r.point();
    const auto n = r.count(vertex_cap);
    Polygon p;
    p.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        p.push_back(r.point());
    r.finish();
    return from_retained_support(p, normal, std::move(support));
}
template<> std::vector<char> planar_support<ArithmeticMode::Ordinary>(const OrdinaryFace& face)
{
    return face.impl_->support;
}
template<> int planar_orientation<ArithmeticMode::Ordinary>(const OrdinaryFace& face)
{
    const auto n = on::polygon_vector(face.points());
    return n[axis_of(n)] < 0 ? -1 : 1;
}
template<>
std::vector<OrdinaryFacePtr> subtract_planar_faces<ArithmeticMode::Ordinary>(
    const OrdinaryFace& source, std::span<const OrdinaryFacePtr> removed)
{
    std::vector<Polygon> remains{source.points()};
    for (const auto& face : removed)
    {
        if (!face || (source.impl_->support != face->impl_->support && !coplanar(source.points(), face->points())))
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
        result.push_back(OrdinaryFace::from_retained_support(p, source.impl_->normal, source.impl_->support));
    return result;
}
EdgePointIndex<ArithmeticMode::Ordinary>::EdgePointIndex(std::span<const std::vector<char>> points)
{
    auto impl = std::make_shared<Impl>();
    for (const auto& p : points)
        impl->points.emplace(on::read_point(p), p);
    impl_ = std::move(impl);
}
std::vector<std::vector<char>> EdgePointIndex<ArithmeticMode::Ordinary>::closed_segment(
    std::span<const char> first, std::span<const char> second) const
{
    const auto a = on::read_point(first), b = on::read_point(second), d = on::sub(b, a);
    const auto length = on::dot(d, d);
    if (length == 0)
        on::reject("zero finite source edge.");
    std::map<long double, std::vector<char>> result;
    for (const auto& [p, id] : impl_->points)
    {
        const auto v = on::sub(p, a);
        const auto t = on::dot(v, d);
        if (on::cross(v, d) == on::WidePoint{} && t >= 0 && t <= length)
            result.emplace(t, id);
    }
    std::vector<std::vector<char>> output;
    for (auto& [t, id] : result)
        output.push_back(std::move(id));
    return output;
}
std::size_t EdgePointIndex<ArithmeticMode::Ordinary>::storage_bytes() const
{
    std::size_t size = sizeof(Impl);
    for (const auto& [p, id] : impl_->points)
        size += sizeof(p) + sizeof(id) + id.capacity() + 4 * sizeof(void*);
    return size;
}
template<> std::vector<char> displacement<ArithmeticMode::Ordinary>(GeometryPoint from, GeometryPoint to)
{
    return on::displacement_packet(on::sub(on::wide(to), on::wide(from)));
}
template<>
std::vector<char> displacement_difference<ArithmeticMode::Ordinary>(std::span<const char> a, std::span<const char> b)
{
    return on::displacement_packet(on::sub(on::read_displacement(a), on::read_displacement(b)));
}
template<> GeometryPoint displacement_display<ArithmeticMode::Ordinary>(std::span<const char> p)
{
    return on::display(on::read_displacement(p));
}
template<>
std::vector<char> point_translated<ArithmeticMode::Ordinary>(std::span<const char> p, std::span<const char> d)
{
    return on::point_packet(on::add(on::read_point(p), on::read_displacement(d)));
}
template<> GeometryPoint point_display<ArithmeticMode::Ordinary>(std::span<const char> p)
{
    return on::display(on::read_point(p));
}
template<>
std::array<GeometryValue<ArithmeticMode::Ordinary>, 3> numerical_point<ArithmeticMode::Ordinary>(
    std::span<const char> p)
{
    const auto v = on::read_point(p);
    return {{{v[0]}, {v[1]}, {v[2]}}};
}
template<>
GeometryMeasure<ArithmeticMode::Ordinary> point_projection<ArithmeticMode::Ordinary>(
    std::span<const char> p, GeometryPoint n, GeometryPoint anchor)
{
    return GeometryMeasure<ArithmeticMode::Ordinary>::from_nominal(
        on::dot(on::wide(n), on::sub(on::read_point(p), on::wide(anchor))));
}
template<>
std::vector<char> periodic_point<ArithmeticMode::Ordinary>(
    std::span<const char> identity, GeometryPoint lower, GeometryPoint upper, std::array<bool, 3> periodic)
{
    auto p = on::read_point(identity);
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
    return on::point_packet(p);
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
    const auto charge = [&](std::size_t n, std::size_t size)
    {
        if (size && n > (byte_cap - bytes) / size)
            reject("native cell partition byte cap exhausted.");
        bytes += n * size;
    };
    using FacePtr = PlanarFacePtr<ArithmeticMode::Ordinary>;
    using Edge = std::pair<WidePoint, WidePoint>;
    std::map<std::vector<char>, std::vector<std::pair<FacePtr, int>>> groups;
    const auto add_cell = [&](const auto& cell, int sign)
    {
        for (std::size_t f = 0; f < cell.num_faces(); ++f)
        {
            auto face = cell.face_geometry(f);
            auto support = planar_support<ArithmeticMode::Ordinary>(*face);
            charge(1, sizeof(std::pair<FacePtr, int>) + 4 * sizeof(void*));
            charge(support.size(), 1);
            groups[std::move(support)].emplace_back(std::move(face), sign);
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
                        charge(1, sizeof(Edge) + sizeof(std::int64_t) + 4 * sizeof(void*));
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
