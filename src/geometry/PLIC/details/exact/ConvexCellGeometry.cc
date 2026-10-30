#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/details/exact/ConvexCellGeometryInternal.hh"

namespace SimpleFluid::PLIC
{
namespace exact_geometry_detail
{
constinit thread_local bool count_exact_geometry = false;

constinit thread_local BackendStatistics<ArithmeticMode::Exact> exact_geometry_counts;

void sample_exact_measure_result(const Q& value)
{
    if (!count_exact_geometry)
        return;
    ++exact_geometry_counts.measure_values;
    I numerator = boost::multiprecision::numerator(value);
    if (numerator < 0)
        numerator = -numerator;
    const I denominator = boost::multiprecision::denominator(value);
    const auto numerator_bits = numerator == 0 ? 0 : boost::multiprecision::msb(numerator) + 1;
    const auto denominator_bits = boost::multiprecision::msb(denominator) + 1;
    exact_geometry_counts.maximum_result_numerator_bits =
        std::max(exact_geometry_counts.maximum_result_numerator_bits, std::uint64_t(numerator_bits));
    exact_geometry_counts.maximum_result_denominator_bits =
        std::max(exact_geometry_counts.maximum_result_denominator_bits, std::uint64_t(denominator_bits));
}

P point(GeometryPoint p)
{
    return {exact(p[0]), exact(p[1]), exact(p[2])};
}

P add(const P& a, const P& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

P sub(const P& a, const P& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

P mul(const P& a, const Q& b)
{
    return {a[0] * b, a[1] * b, a[2] * b};
}

Q dot(const P& a, const P& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

P cross(const P& a, const P& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

bool zero(const P& p)
{
    return p[0] == 0 && p[1] == 0 && p[2] == 0;
}

Q distance(const Plane& plane, const P& p)
{
    // An absent coordinate contributes exactly zero, even when its represented
    // rational displacement is expensive to construct. Retain the same signed
    // distance (not a rescaled predicate) for edge intersection parameters.
    Q result = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (plane.normal[axis] != 0)
            result += plane.normal[axis] * (p[axis] - plane.anchor[axis]);
    if (plane.offset != 0)
        result -= plane.offset;
    return result;
}

GeometryValue<ArithmeticMode::Exact> square_root(const Q& squared)
{
    if (squared < 0)
        throw std::logic_error("Negative squared geometry measure.");
    if (squared == 0)
        return {};
    // Scaling the rational before sqrt avoids underflow of the squared area.
    const auto numerator = boost::multiprecision::numerator(squared);
    const auto denominator = boost::multiprecision::denominator(squared);
    const auto shift = static_cast<int>(boost::multiprecision::msb(numerator)) -
                       static_cast<int>(boost::multiprecision::msb(denominator));
    const int half_shift = shift / 2;
    const Q scale = half_shift >= 0 ? Q(I{1} << (2 * half_shift)) : Q(1, I{1} << (-2 * half_shift));
    const auto v = std::ldexp(std::sqrt(Q(squared / scale).convert_to<long double>()), half_shift);
    if (!std::isfinite(v) || !(v > 0))
        throw std::runtime_error("Section area is not representable.");
    auto lo = v, hi = v;
    for (unsigned step = 0; step < 16; ++step)
    {
        const Q l = exact_wide(lo), h = exact_wide(hi);
        if (l * l <= squared && h * h >= squared)
            return {v, std::nextafter(std::max(v - lo, hi - v), std::numeric_limits<long double>::infinity()), true};
        lo = std::nextafter(lo, 0.0L);
        hi = std::nextafter(hi, std::numeric_limits<long double>::infinity());
    }
    throw std::runtime_error("Section area square-root bound is inconclusive.");
}

P area_vector(const Poly& poly, const Face& face)
{
    P result{};
    const auto& origin = poly.vertices[face[0]];
    for (std::size_t i = 1; i + 1 < face.size(); ++i)
        result = add(result, cross(sub(poly.vertices[face[i]], origin), sub(poly.vertices[face[i + 1]], origin)));
    return mul(result, Q(1, 2));
}

Moments exact_moments(const Poly& poly, const P& origin)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.moment_evaluations;
    Moments result{};
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            // A triangle incident on the moment origin has zero determinant
            // and zero first moment. Equality is exact; collinear triangles
            // not incident on that origin are handled by the determinant below.
            if (poly.vertices[face[0]] == origin || poly.vertices[face[i]] == origin ||
                poly.vertices[face[i + 1]] == origin)
                continue;
            const P a = sub(poly.vertices[face[0]], origin), b = sub(poly.vertices[face[i]], origin),
                    c = sub(poly.vertices[face[i + 1]], origin);
            const Q volume = dot(a, cross(b, c)) / 6;
            if (volume == 0)
                continue;
            result.volume += volume;
            result.first = add(result.first, mul(add(add(a, b), c), Q(volume / 4)));
        }
    return result;
}

Q exact_volume(const Poly& poly, const P& origin)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.volume_evaluations;
    // Volume-only consumers use the identical oriented fan determinants and
    // exact accumulation. Full moment admission and cover proofs still call
    // exact_moments, including their independent first-moment certificates.
    Q result = 0;
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            if (poly.vertices[face[0]] == origin || poly.vertices[face[i]] == origin ||
                poly.vertices[face[i + 1]] == origin)
                continue;
            const P a = sub(poly.vertices[face[0]], origin), b = sub(poly.vertices[face[i]], origin),
                    c = sub(poly.vertices[face[i + 1]], origin);
            result += dot(a, cross(b, c)) / 6;
        }
    return result;
}

std::vector<Plane> planes(const Poly& poly)
{
    std::vector<Plane> result;
    for (const auto& face : poly.faces)
        result.push_back({area_vector(poly, face), poly.vertices[face[0]]});
    return result;
}

// Native/public convex outputs retain numeric admission. Only opaque exact
// transport fragments defer conversion until their measures are aggregated.
void validate(const Poly& poly, bool measure_outputs)
{
    if (poly.vertices.size() < 4 || poly.faces.size() < 4)
        throw std::invalid_argument("Convex shell needs at least four vertices and faces.");
    std::set<P> vertices;
    for (const auto& p : poly.vertices)
        if (!vertices.insert(p).second)
            throw std::invalid_argument("Convex shell has duplicate vertices.");
    std::map<std::pair<std::size_t, std::size_t>, unsigned> directed;
    std::vector<unsigned> used(poly.vertices.size());
    for (const auto& face : poly.faces)
    {
        if (face.size() < 3)
            throw std::invalid_argument("Convex face has fewer than three vertices.");
        std::set<std::size_t> unique;
        for (const auto index : face)
        {
            if (index >= poly.vertices.size() || !unique.insert(index).second)
                throw std::invalid_argument("Convex face indices are invalid or repeated.");
            ++used[index];
        }
        const auto normal = area_vector(poly, face);
        if (zero(normal))
            throw std::invalid_argument("Convex shell has a zero-area face.");
        const Plane plane{normal, poly.vertices[face[0]]};
        for (const auto index : face)
            if (distance(plane, poly.vertices[index]) != 0)
                throw std::invalid_argument("Convex shell has a nonplanar face.");
        for (const auto& p : poly.vertices)
            if (distance(plane, p) > 0)
                throw std::invalid_argument("Shell is nonconvex or incorrectly oriented.");
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto a = face[i], b = face[(i + 1) % face.size()];
            if (++directed[{a, b}] != 1)
                throw std::invalid_argument("Shell has duplicated directed incidence.");
            const P edge = sub(poly.vertices[b], poly.vertices[a]);
            for (const auto j : face)
            {
                if (dot(normal, cross(edge, sub(poly.vertices[j], poly.vertices[a]))) < 0)
                    throw std::invalid_argument("Convex face is nonsimple or nonconvex.");
                const auto displacement = sub(poly.vertices[j], poly.vertices[a]);
                if (j != a && j != b && zero(cross(edge, displacement)) && dot(displacement, edge) > 0 &&
                    dot(displacement, edge) < dot(edge, edge))
                    throw std::invalid_argument("Convex face has overlapping collinear edges.");
            }
        }
    }
    for (const auto& [edge, count] : directed)
        if (!directed.contains({edge.second, edge.first}))
            throw std::invalid_argument("Convex shell is not a closed oriented manifold.");
    if (std::find(used.begin(), used.end(), 0) != used.end())
        throw std::invalid_argument("Convex shell has unused vertices.");
    const auto m = exact_moments(poly, poly.vertices[0]);
    if (!(m.volume > 0))
        throw std::invalid_argument("Convex shell has nonpositive volume.");
    if (measure_outputs)
    {
        (void) measured(m.volume);
        for (const auto& first : m.first)
            (void) measured(first, true);
    }
}

Cut clipped(const Poly& input, const Plane& plane, bool measure_outputs)
{
    UncachedPlaneDistances distances{input, plane};
    return clipped_impl(
        input, plane, measure_outputs, distances, [](const Poly& output, bool measure) { validate(output, measure); });
}

void clipped_incidence(const Poly& poly, const Plane& plane)
{
    if (poly.vertices.size() < 4 || poly.faces.size() < 4)
        throw std::runtime_error("Certified convex cut lost full-dimensional incidence.");
    std::vector<unsigned> used(poly.vertices.size());
    std::map<std::pair<std::size_t, std::size_t>, unsigned> edges;
    for (const auto& face : poly.faces)
    {
        if (face.size() < 3)
            throw std::runtime_error("Certified convex cut has a short face.");
        std::set<std::size_t> distinct;
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto a = face[i], b = face[(i + 1) % face.size()];
            if (a >= poly.vertices.size() || b >= poly.vertices.size() || !distinct.insert(a).second ||
                ++edges[{a, b}] != 1)
                throw std::runtime_error("Certified convex cut has invalid oriented incidence.");
            ++used[a];
        }
    }
    for (const auto& [edge, count] : edges)
        if (!edges.contains({edge.second, edge.first}))
            throw std::runtime_error("Certified convex cut is not an oriented closed shell.");
    if (std::find(used.begin(), used.end(), 0) != used.end() ||
        dot(area_vector(poly, poly.faces.back()), plane.normal) <= 0)
        throw std::runtime_error("Certified convex cut has unused vertices or reversed cap.");
}

std::optional<CertifiedPoly> certified_clip(CertifiedPolyRef input, const Plane& plane, ClipScratch& scratch)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.clip_constructions;
    if (zero(plane.normal))
        throw std::invalid_argument("Certified clipping plane has a zero normal.");
    // Preconditions are a fully admitted, full-dimensional, oriented convex
    // shell and an exact halfspace. Sutherland--Hodgman preserves each convex
    // planar face's order/planarity; every new point is an exact edge/plane
    // intersection. Unique rational vertices, positive-area retained faces,
    // the reversed cut-edge single-cycle cap and compaction preserve the shell.
    // Both strict vertex signs imply a positive-volume intersection. Thus the
    // global all-vertices convexity and first-moment admission need not recur.
    // Check the derived oriented incidence independently; final partition
    // volume/first-moment and disjointness proofs remain mandatory downstream.
    try
    {
        CachedPlaneDistances distances(input.get(), plane, scratch);
        auto cut = clipped_impl(
            input.get(), plane, false, distances, [&](const Poly& output, bool) { clipped_incidence(output, plane); });
        if (!cut.poly)
            return {};
        return CertifiedPoly(std::move(*cut.poly));
    }
    catch (const ClipCacheExhausted&)
    {
        // Caching is optional. Its lease is fully released before the single
        // constant-workspace retry. Geometry/incidence/bad-allocation errors
        // are deliberately NOT caught, and every physical proof is unchanged.
        if (scratch.fallbacks != std::numeric_limits<std::size_t>::max())
            ++scratch.fallbacks;
        return certified_clip_uncached(input, plane);
    }
}

std::optional<CertifiedPoly> certified_clip_uncached(CertifiedPolyRef input, const Plane& plane)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.clip_constructions;
    if (zero(plane.normal))
        throw std::invalid_argument("Certified clipping plane has a zero normal.");
    // Identical convexity preservation proof, with the pre-existing constant
    // distance workspace for phase queries that have no caller scratch budget.
    UncachedPlaneDistances distances{input.get(), plane};
    auto cut = clipped_impl(
        input.get(), plane, false, distances, [&](const Poly& output, bool) { clipped_incidence(output, plane); });
    if (!cut.poly)
        return {};
    return CertifiedPoly(std::move(*cut.poly));
}

std::optional<CertifiedPoly> certified_intersection(CertifiedPolyRef a, CertifiedPolyRef b, ClipScratch& scratch)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.intersection_constructions;
    std::optional<CertifiedPoly> result{CertifiedPoly(a.get())};
    for (const auto& plane : planes(b.get()))
    {
        result = certified_clip(result->ref(), plane, scratch);
        if (!result)
            break;
    }
    return result;
}

std::optional<CertifiedPoly> certified_intersection(
    CertifiedPolyRef a, std::span<const Plane> clipper_planes, ClipScratch& scratch)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.intersection_constructions;
    std::optional<CertifiedPoly> result{CertifiedPoly(a.get())};
    for (const auto& plane : clipper_planes)
    {
        result = certified_clip(result->ref(), plane, scratch);
        if (!result)
            break;
    }
    return result;
}

std::optional<Poly> intersection(const Poly& a, const Poly& b, bool measure_outputs)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.intersection_constructions;
    std::optional<Poly> result = a;
    for (const auto& plane : planes(b))
    {
        result = clipped(*result, plane, measure_outputs).poly;
        if (!result)
            break;
    }
    return result;
}

bool contained(const Poly& a, const Poly& b)
{
    for (const auto& plane : planes(b))
        for (const auto& p : a.vertices)
            if (distance(plane, p) > 0)
                return false;
    return true;
}

std::optional<Product> product_description(const Poly& poly)
{
    Q lo = poly.vertices[0][2], hi = lo;
    for (const auto& p : poly.vertices)
    {
        lo = std::min(lo, p[2]);
        hi = std::max(hi, p[2]);
    }
    if (!(hi > lo))
        return {};
    for (const auto& p : poly.vertices)
        if (p[2] != lo && p[2] != hi)
            return {};
    std::vector<P> base;
    for (const auto& face : poly.faces)
        if (std::all_of(face.begin(), face.end(), [&](auto i) { return poly.vertices[i][2] == lo; }))
        {
            // Multiple coplanar atomic caps need the generic shell path.
            if (!base.empty())
                return {};
            for (auto it = face.rbegin(); it != face.rend(); ++it)
                base.push_back({poly.vertices[*it][0], poly.vertices[*it][1], 0});
        }
    if (base.size() * 2 != poly.vertices.size())
        return {};
    for (const auto& p : base)
        if (std::find(poly.vertices.begin(), poly.vertices.end(), P{p[0], p[1], hi}) == poly.vertices.end())
            return {};
    return Product{std::move(base), lo, hi};
}

Moments product_moments(const Product& product, const P& origin)
{
    if (count_exact_geometry)
        ++exact_geometry_counts.moment_evaluations;
    Moments result{};
    Q area = 0, first_x = 0, first_y = 0;
    for (std::size_t i = 0; i < product.base.size(); ++i)
    {
        const auto a = sub(product.base[i], origin), b = sub(product.base[(i + 1) % product.base.size()], origin);
        const Q cross2 = a[0] * b[1] - a[1] * b[0];
        area += cross2 / 2;
        first_x += (a[0] + b[0]) * cross2 / 6;
        first_y += (a[1] + b[1]) * cross2 / 6;
    }
    const Q height = product.upper - product.lower;
    result.volume = area * height;
    result.first = {
        first_x * height, first_y * height, result.volume * ((product.lower + product.upper) / 2 - origin[2])};
    return result;
}

Poly product_poly(const Product& product)
{
    Poly result;
    const auto n = product.base.size();
    for (const auto& p : product.base)
        result.vertices.push_back({p[0], p[1], product.lower});
    for (const auto& p : product.base)
        result.vertices.push_back({p[0], p[1], product.upper});
    Face bottom, top;
    for (std::size_t i = 0; i < n; ++i)
    {
        bottom.push_back(n - 1 - i);
        top.push_back(n + i);
    }
    result.faces = {std::move(bottom), std::move(top)};
    for (std::size_t i = 0; i < n; ++i)
        result.faces.push_back({i, (i + 1) % n, (i + 1) % n + n, i + n});
    validate(result);
    return result;
}

std::optional<Product> product_intersection(const Product& a, const Product& b)
{
    Product result{a.base, std::max(a.lower, b.lower), std::min(a.upper, b.upper)};
    if (!(result.upper > result.lower))
        return {};
    for (std::size_t edge = 0; edge < b.base.size(); ++edge)
    {
        const auto& anchor = b.base[edge];
        const P delta = sub(b.base[(edge + 1) % b.base.size()], anchor);
        const Plane plane{{delta[1], -delta[0], 0}, anchor};
        std::vector<P> clipped_base;
        auto append = [&](const P& p)
        {
            if (clipped_base.empty() || clipped_base.back() != p)
                clipped_base.push_back(p);
        };
        for (std::size_t i = 0; i < result.base.size(); ++i)
        {
            const auto& x = result.base[i];
            const auto& y = result.base[(i + 1) % result.base.size()];
            const Q dx = distance(plane, x), dy = distance(plane, y);
            if (dx <= 0)
                append(x);
            if ((dx < 0 && dy > 0) || (dx > 0 && dy < 0))
                append(add(x, mul(sub(y, x), Q(dx / (dx - dy)))));
        }
        if (clipped_base.size() > 1 && clipped_base.front() == clipped_base.back())
            clipped_base.pop_back();
        result.base = std::move(clipped_base);
        if (result.base.size() < 3)
            return {};
    }
    const auto m = product_moments(result, P{});
    if (m.volume == 0)
        return {};
    if (m.volume < 0)
        throw std::logic_error("Product clipping reversed polygon orientation.");
    return result;
}

NativeConvexShell canonical_native_shell(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& outward_faces)
{
    Poly shell;
    std::vector<GeometryPoint> displayed;
    std::map<P, std::size_t> ids;
    std::vector<std::size_t> remap;
    for (const auto& p : vertices)
    {
        const auto exact_point = point(p);
        const auto [it, inserted] = ids.emplace(exact_point, shell.vertices.size());
        if (inserted)
        {
            shell.vertices.push_back(exact_point);
            displayed.push_back(p);
        }
        remap.push_back(it->second);
    }
    for (const auto& face : outward_faces)
    {
        Face mapped;
        for (const auto index : face)
        {
            if (index >= remap.size())
                throw std::invalid_argument("Native shell face has an invalid vertex index.");
            mapped.push_back(remap[index]);
        }
        shell.faces.push_back(std::move(mapped));
    }
    // Fine-face vertices must also subdivide every incident coarse edge.
    for (auto& face : shell.faces)
    {
        Face propagated;
        for (std::size_t edge = 0; edge < face.size(); ++edge)
        {
            const auto a = face[edge], b = face[(edge + 1) % face.size()];
            const auto delta = sub(shell.vertices[b], shell.vertices[a]);
            const Q length2 = dot(delta, delta);
            if (length2 == 0)
                throw std::invalid_argument("Native cell has a zero edge.");
            std::vector<std::pair<Q, std::size_t>> interior;
            for (std::size_t j = 0; j < shell.vertices.size(); ++j)
            {
                const auto offset = sub(shell.vertices[j], shell.vertices[a]);
                const Q along = dot(offset, delta);
                if (zero(cross(delta, offset)) && along > 0 && along < length2)
                    interior.emplace_back(along, j);
            }
            std::sort(interior.begin(), interior.end());
            propagated.push_back(a);
            for (const auto& [position, j] : interior)
                propagated.push_back(j);
        }
        face = std::move(propagated);
    }
    validate(shell);
    return {std::move(displayed), std::move(shell.faces)};
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{
template<> void enable_backend_statistics<ArithmeticMode::Exact>(bool enabled)
{
    count_exact_geometry = enabled;
}

template<> void reset_backend_statistics<ArithmeticMode::Exact>()
{
    exact_geometry_counts = {};
}

template<> BackendStatistics<ArithmeticMode::Exact> backend_statistics<ArithmeticMode::Exact>()
{
    return exact_geometry_counts;
}

NumericalConvexShell<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::numerical_shell() const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.source_coordinate_exports;
    NumericalConvexShell<ArithmeticMode::Exact> result;
    result.origin = impl_->origin;
    result.outward_faces = impl_->poly.faces;
    result.relative_vertices.reserve(impl_->poly.vertices.size());
    const auto origin = point(result.origin);
    for (const auto& vertex : impl_->poly.vertices)
    {
        std::array<GeometryValue<ArithmeticMode::Exact>, 3> enclosed;
        for (std::size_t axis = 0; axis < 3; ++axis)
            enclosed[axis] = measured(vertex[axis] - origin[axis], true);
        result.relative_vertices.push_back(enclosed);
    }
    result.face_support_identities.reserve(result.outward_faces.size());
    for (std::size_t face = 0; face < result.outward_faces.size(); ++face)
        result.face_support_identities.push_back(planar_support<ArithmeticMode::Exact>(*face_geometry(face)));
    return result;
}

std::vector<char> canonical_cell_key(const ConvexCellGeometry<ArithmeticMode::Exact>& cell)
{
    WireWriter w;
    w.word(0x5346455841435431ULL);
    w.word(1);
    for (auto v : cell.impl_->origin)
        w.real(v);
    w.word(cell.impl_->poly.vertices.size());
    const auto origin = point(cell.impl_->origin);
    for (const auto& p : cell.impl_->poly.vertices)
        for (const auto& q : sub(p, origin))
            w.rational(q);
    w.word(cell.impl_->poly.faces.size());
    for (const auto& face : cell.impl_->poly.faces)
    {
        w.word(face.size());
        for (auto id : face)
            w.word(id);
    }
    return std::move(w.data);
}

std::vector<char> ConvexCellGeometry<ArithmeticMode::Exact>::encode_certificate(std::size_t byte_budget) const
{
    return encode_geometry_packet(
        GeometryPacketKind::CellCertificate, ArithmeticMode::Exact, canonical_cell_key(*this), byte_budget);
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::decode_certificate(
    std::span<const char> packet, std::size_t byte_budget)
{
    const auto envelope = decode_geometry_packet(packet, ArithmeticMode::Exact, byte_budget);
    if (envelope.kind != GeometryPacketKind::CellCertificate)
        throw std::invalid_argument("Foreign exact cell certificate kind.");
    WireReader r{envelope.body, 0, true};
    if (r.word() != 0x5346455841435431ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact geometry certificate version/tag.");
    auto impl = std::make_shared<Impl>();
    for (auto& v : impl->origin)
        v = r.real();
    const auto origin = point(impl->origin);
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        P p;
        for (auto& q : p)
            q = r.rational();
        impl->poly.vertices.push_back(add(p, origin));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        Face face;
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            face.push_back(r.word());
        impl->poly.faces.push_back(std::move(face));
    }
    r.finish();
    validate(impl->poly);
    impl->product = product_description(impl->poly);
    auto result =
        ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
    const auto canonical = canonical_cell_key(*result);
    if (canonical.size() != envelope.body.size() ||
        !std::equal(canonical.begin(), canonical.end(), envelope.body.begin()))
        throw std::invalid_argument("Noncanonical exact cell certificate body.");
    return result;
}

ConvexCellGeometry<ArithmeticMode::Exact>::ConvexCellGeometry(std::shared_ptr<const Impl> implementation)
    : impl_(std::move(implementation))
{
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces)
{
    auto impl = std::make_shared<Impl>();
    for (const auto& p : vertices)
        impl->poly.vertices.push_back(point(p));
    impl->poly.faces = faces;
    validate(impl->poly);
    impl->origin = vertices[0];
    impl->product = product_description(impl->poly);
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::box(
    GeometryPoint lower, GeometryPoint upper)
{
    for (std::size_t a = 0; a < 3; ++a)
        if (!(upper[a] > lower[a]))
            throw std::invalid_argument("Convex box must have positive represented spans.");
    std::array<GeometryPoint, 8> v;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t a = 0; a < 3; ++a)
            v[i][a] = (i >> a) & 1 ? upper[a] : lower[a];
    return make(v, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}});
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::extruded_polygon(
    std::span<const std::array<double, 2>> polygon, double lower_z, double upper_z)
{
    Product product;
    product.lower = exact(lower_z);
    product.upper = exact(upper_z);
    if (!(product.upper > product.lower) || polygon.size() < 3)
        throw std::invalid_argument("Extruded polygon needs a positive represented Z interval and convex CCW base.");
    for (const auto& p : polygon)
        product.base.push_back({exact(p[0]), exact(p[1]), 0});
    auto impl = std::make_shared<Impl>();
    impl->poly = product_poly(product);
    impl->product = std::move(product);
    impl->origin = {polygon[0][0], polygon[0][1], lower_z};
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
}

NativeConvexShell ConvexCellGeometry<ArithmeticMode::Exact>::native_cell_shell(
    const MultiRegionMesh& mesh, std::uint64_t cell)
{
    return ConvexCellGeometry<ArithmeticMode::Ordinary>::native_cell_shell(mesh, cell);
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::from_native_cell(
    const MultiRegionMesh& mesh, std::uint64_t cell)
{
    return from_geometry(*ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell(mesh, cell));
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::from_native_shell(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& outward_faces)
{
    const auto shell = canonical_native_shell(vertices, outward_faces);
    return make(shell.vertices, shell.outward_faces);
}

GeometryValue<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::volume() const
{
    return measured(exact_volume(impl_->poly, point(impl_->origin)));
}

GeometryMoments<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::moments() const
{
    const auto exact = impl_->product ? product_moments(*impl_->product, point(impl_->origin))
                                      : exact_moments(impl_->poly, point(impl_->origin));
    GeometryMoments<ArithmeticMode::Exact> result;
    result.origin = impl_->origin;
    result.volume = measured(exact.volume);
    for (std::size_t a = 0; a < 3; ++a)
        result.first_moment[a] = measured(exact.first[a], true);
    return result;
}

GeometryMoments<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::moments_binary64() const
{
    const auto exact = impl_->product ? product_moments(*impl_->product, point(impl_->origin))
                                      : exact_moments(impl_->poly, point(impl_->origin));
    GeometryMoments<ArithmeticMode::Exact> result;
    result.origin = impl_->origin;
    result.volume = measured<double>(exact.volume);
    for (std::size_t a = 0; a < 3; ++a)
        result.first_moment[a] = measured<double>(exact.first[a], true);
    return result;
}

std::vector<ConvexOctantPiece<ArithmeticMode::Exact>> ConvexCellGeometry<ArithmeticMode::Exact>::split_aabb_midpoints(
    double minimum_scaled_volume) const
{
    if (!std::isfinite(minimum_scaled_volume) || minimum_scaled_volume < 0)
        throw std::invalid_argument("Convex split quality bound must be finite and nonnegative.");
    P lower = impl_->poly.vertices[0], upper = lower;
    for (const auto& p : impl_->poly.vertices)
        for (std::size_t a = 0; a < 3; ++a)
        {
            lower[a] = std::min(lower[a], p[a]);
            upper[a] = std::max(upper[a], p[a]);
        }
    P mid = mul(add(lower, upper), Q(1, 2));
    std::vector<ConvexOctantPiece<ArithmeticMode::Exact>> result;
    std::vector<ConvexGeometryPtr<ArithmeticMode::Exact>> certificates;
    for (unsigned octant = 0; octant < 8; ++octant)
    {
        std::optional<Poly> poly = impl_->poly;
        std::vector<ConvexFaceOrigin> origins;
        for (std::size_t f = 0; f < poly->faces.size(); ++f)
            origins.push_back({ConvexFaceOrigin::Kind::ParentFace, f});
        for (std::size_t axis = 0; axis < 3 && poly; ++axis)
        {
            P normal{};
            const bool upper_half = (octant >> axis) & 1;
            normal[axis] = upper_half ? -1 : 1;
            auto cut = clipped(*poly, {normal, P{}, upper_half ? -mid[axis] : mid[axis]});
            if (!cut.poly)
            {
                poly.reset();
                break;
            }
            std::vector<ConvexFaceOrigin> next;
            for (auto source : cut.face_sources)
                next.push_back(source == std::numeric_limits<std::size_t>::max()
                                   ? ConvexFaceOrigin{ConvexFaceOrigin::Kind::CutPlane, axis}
                                   : origins.at(source));
            origins = std::move(next);
            poly = std::move(cut.poly);
        }
        if (!poly)
            continue;
        const auto moments = exact_moments(*poly, point(impl_->origin));
        (void) measured(moments.volume).as_double();
        P lo = poly->vertices[0], hi = lo;
        for (const auto& p : poly->vertices)
            for (std::size_t a = 0; a < 3; ++a)
            {
                lo[a] = std::min(lo[a], p[a]);
                hi[a] = std::max(hi[a], p[a]);
            }
        Q largest = 0;
        for (std::size_t a = 0; a < 3; ++a)
            largest = std::max(largest, Q(hi[a] - lo[a]));
        if (!(moments.volume > exact(minimum_scaled_volume) * largest * largest * largest))
            throw std::invalid_argument("Positive convex split child violates exact scaled-volume quality bound.");
        auto implementation = std::make_shared<Impl>();
        implementation->poly = std::move(*poly);
        implementation->origin = impl_->origin;
        implementation->product = product_description(implementation->poly);
        auto geometry = ConvexGeometryPtr<ArithmeticMode::Exact>(
            new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(implementation)));
        certificates.push_back(geometry);
        result.push_back({octant, std::move(geometry), std::move(origins)});
    }
    certify_partition(certificates);
    return result;
}

bool ConvexCellGeometry<ArithmeticMode::Exact>::quality_above(double minimum_scaled_volume) const
{
    if (!std::isfinite(minimum_scaled_volume) || minimum_scaled_volume < 0)
        throw std::invalid_argument("Convex quality bound must be finite and nonnegative.");
    P lower = impl_->poly.vertices[0], upper = lower;
    for (const auto& p : impl_->poly.vertices)
        for (std::size_t a = 0; a < 3; ++a)
        {
            lower[a] = std::min(lower[a], p[a]);
            upper[a] = std::max(upper[a], p[a]);
        }
    Q largest = 0;
    for (std::size_t a = 0; a < 3; ++a)
        largest = std::max(largest, Q(upper[a] - lower[a]));
    const auto volume = exact_volume(impl_->poly, point(impl_->origin));
    return volume > exact(minimum_scaled_volume) * largest * largest * largest;
}

std::array<GeometryValue<ArithmeticMode::Exact>, 3> ConvexCellGeometry<ArithmeticMode::Exact>::axis_spans() const
{
    P lo = impl_->poly.vertices[0], hi = lo;
    for (const auto& p : impl_->poly.vertices)
        for (std::size_t a = 0; a < 3; ++a)
        {
            lo[a] = std::min(lo[a], p[a]);
            hi[a] = std::max(hi[a], p[a]);
        }
    std::array<GeometryValue<ArithmeticMode::Exact>, 3> result;
    for (std::size_t a = 0; a < 3; ++a)
        result[a] = measured(Q(hi[a] - lo[a]));
    return result;
}

bool ConvexCellGeometry<ArithmeticMode::Exact>::axis_span_exceeds(
    const ConvexCellGeometry<ArithmeticMode::Exact>& other, std::size_t axis, double ratio) const
{
    if (axis >= 3 || !std::isfinite(ratio) || ratio < 1)
        throw std::invalid_argument("Physical span comparison needs a valid axis and finite ratio>=1.");
    auto span = [&](const Poly& poly)
    {
        Q lo = poly.vertices[0][axis], hi = lo;
        for (const auto& p : poly.vertices)
        {
            lo = std::min(lo, p[axis]);
            hi = std::max(hi, p[axis]);
        }
        return Q(hi - lo);
    };
    return span(impl_->poly) > exact(ratio) * span(other.impl_->poly);
}

std::size_t ConvexCellGeometry<ArithmeticMode::Exact>::num_faces() const noexcept
{
    return impl_->poly.faces.size();
}

ClipResult<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::clip(
    const GeometryHalfspace& halfspace) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.clip_constructions;
    const Plane plane{point(halfspace.normal), point(halfspace.anchor), exact(halfspace.offset)};
    if (zero(plane.normal))
        throw std::invalid_argument("Clipping halfspace needs a nonzero finite normal.");
    auto inside = clipped(impl_->poly, plane);
    auto outside = clipped(impl_->poly, {mul(plane.normal, -1), plane.anchor, -plane.offset});
    ClipResult<ArithmeticMode::Exact> result;
    auto retain = [&](std::optional<Poly>& poly) -> ConvexGeometryPtr<ArithmeticMode::Exact>
    {
        if (!poly)
            return {};
        auto impl = std::make_shared<Impl>();
        impl->poly = std::move(*poly);
        impl->origin = impl_->origin;
        impl->product = product_description(impl->poly);
        return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
    };
    if (inside.poly && !inside.cap.empty())
    {
        for (const auto index : inside.cap)
        {
            std::array<long double, 3> p;
            for (std::size_t a = 0; a < 3; ++a)
                p[a] = measured(inside.poly->vertices[index][a]).value;
            result.section.vertices.push_back(p);
        }
        const auto area = area_vector(*inside.poly, inside.cap);
        result.section.area = square_root(dot(area, area));
    }
    else
    {
        // A supporting plane may touch a point, an edge, or a tiled face.
        // None retains a positive volume on the opposite side, but its section
        // is still a geometric object independent of physical phase policy.
        P total_area{};
        for (const auto& p : impl_->poly.vertices)
            if (distance(plane, p) == 0)
            {
                std::array<long double, 3> display;
                for (std::size_t a = 0; a < 3; ++a)
                    display[a] = measured(p[a]).value;
                result.section.vertices.push_back(display);
            }
        for (const auto& face : impl_->poly.faces)
            if (std::all_of(
                    face.begin(), face.end(), [&](auto i) { return distance(plane, impl_->poly.vertices[i]) == 0; }))
                total_area = add(total_area, area_vector(impl_->poly, face));
        result.section.area = square_root(dot(total_area, total_area));
    }
    result.inside = retain(inside.poly);
    result.outside = retain(outside.poly);
    result.inside_apertures = std::move(inside.apertures);
    result.outside_apertures = std::move(outside.apertures);
    return result;
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::intersect(
    const ConvexCellGeometry<ArithmeticMode::Exact>& other) const
{
    if (impl_->product && other.impl_->product)
    {
        auto product = product_intersection(*impl_->product, *other.impl_->product);
        if (!product)
            return {};
        auto impl = std::make_shared<Impl>();
        impl->poly = product_poly(*product);
        impl->origin = impl_->origin;
        impl->product = std::move(product);
        return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
    }
    auto poly = intersection(impl_->poly, other.impl_->poly);
    if (!poly)
        return {};
    auto impl = std::make_shared<Impl>();
    impl->poly = std::move(*poly);
    impl->origin = impl_->origin;
    impl->product = product_description(impl->poly);
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(impl)));
}

void ConvexCellGeometry<ArithmeticMode::Exact>::certify_partition(
    std::span<const ConvexGeometryPtr<ArithmeticMode::Exact>> pieces) const
{
    if (count_exact_geometry)
        ++exact_geometry_counts.cover_certifications;
    const P origin = point(impl_->origin);
    const auto expected = exact_moments(impl_->poly, origin);
    Moments sum{};
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
        if (!pieces[i] || !contained(pieces[i]->impl_->poly, impl_->poly))
            throw std::invalid_argument("Coverage piece is empty or outside the represented cell.");
        const auto actual = exact_moments(pieces[i]->impl_->poly, origin);
        sum.volume += actual.volume;
        sum.first = add(sum.first, actual.first);
        for (std::size_t j = 0; j < i; ++j)
            if (intersection(pieces[i]->impl_->poly, pieces[j]->impl_->poly))
                throw std::invalid_argument("Coverage pieces have positive overlapping volume.");
    }
    if (sum.volume != expected.volume || sum.first != expected.first)
        throw std::invalid_argument("Coverage has a real represented volume or first-moment gap.");
}

bool ConvexCellGeometry<ArithmeticMode::Exact>::equivalent(const ConvexCellGeometry<ArithmeticMode::Exact>& other) const
{
    return contained(impl_->poly, other.impl_->poly) && contained(other.impl_->poly, impl_->poly);
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
namespace exact_geometry_detail
{
void ClipScratch::add(std::size_t bytes)
{
    if (bytes > limit - used)
        throw ClipCacheExhausted("Optional clipping cache exceeds live lease: used=" + std::to_string(used) +
                                 ", request=" + std::to_string(bytes) + ", limit=" + std::to_string(limit));
    used += bytes;
    peak = std::max(peak, used);
}

void ClipScratch::scalar(const Q& q)
{
    for (auto capacity : {q.backend().num().capacity(), q.backend().denom().capacity()})
    {
        if (capacity > (limit - used) / sizeof(boost::multiprecision::limb_type))
            throw ClipCacheExhausted("Optional clipping scalar cache exceeds live lease: used=" + std::to_string(used) +
                                     ", limb_count=" + std::to_string(capacity) + ", limit=" + std::to_string(limit));
        add(capacity * sizeof(boost::multiprecision::limb_type));
    }
}

CachedPlaneDistances::CachedPlaneDistances(const Poly& input, const Plane& halfspace, ClipScratch& budget)
    : lease(budget), scratch(budget), poly(input), plane(halfspace)
{
    scratch.add(sizeof(filter));
    if (poly.vertices.size() > (scratch.limit - scratch.used) / sizeof(Entry))
        throw ClipCacheExhausted("Optional clipping distance cache exceeds its live count lease.");
    scratch.add(poly.vertices.size() * sizeof(Entry));
    input_values.resize(poly.vertices.size());
    if (input_values.capacity() > poly.vertices.size())
        scratch.add((input_values.capacity() - poly.vertices.size()) * sizeof(Entry));
    filter = detail::certified_filter_plane(plane.normal, plane.anchor, plane.offset);
}

const Q& CachedPlaneDistances::input(std::size_t i)
{
    auto& entry = input_values[i];
    if (!entry.exact)
    {
        auto value = distance(plane, poly.vertices[i]);
        scratch.scalar(value);
        entry.value = std::move(value);
        entry.exact = true;
        entry.sign = entry.value < 0 ? -1 : entry.value > 0 ? 1 : 0;
    }
    return entry.value;
}

int CachedPlaneDistances::sign(std::size_t i)
{
    auto& entry = input_values[i];
    if (entry.sign == 2)
    {
        if (const auto certified = detail::try_certified_support_sign(filter, poly.vertices[i]))
            entry.sign = *certified;
        else
            (void) input(i);
    }
    return entry.sign;
}

void CachedPlaneDistances::append(int sign)
{
    scratch.add(4 * sizeof(int)); // old/new geometric growth capacities
    output_signs.push_back(sign);
}

void WireWriter::word(std::uint64_t value)
{
    for (unsigned b = 0; b < 8; ++b)
        data.push_back(static_cast<char>(value >> (8 * b)));
}

void WireWriter::real(double value)
{
    (void) exact(value);
    word(std::bit_cast<std::uint64_t>(value));
}

void WireWriter::integer(I value)
{
    word(value < 0);
    if (value < 0)
        value = -value;
    std::vector<unsigned char> bytes;
    export_bits(value, std::back_inserter(bytes), 8, false);
    word(bytes.size());
    for (auto byte : bytes)
        data.push_back(static_cast<char>(byte));
}

void WireWriter::rational(const Q& value)
{
    integer(boost::multiprecision::numerator(value));
    integer(boost::multiprecision::denominator(value));
}

std::uint64_t WireReader::word()
{
    if (data.size() - position < 8)
        throw std::invalid_argument("Truncated exact geometry certificate.");
    std::uint64_t v = 0;
    for (unsigned b = 0; b < 8; ++b)
        v |= std::uint64_t(static_cast<unsigned char>(data[position++])) << (8 * b);
    return v;
}

std::size_t WireReader::count()
{
    auto n = word();
    if (n > (data.size() - position) / 8 + 1)
        throw std::invalid_argument("Invalid exact geometry count.");
    return static_cast<std::size_t>(n);
}

double WireReader::real()
{
    auto v = std::bit_cast<double>(word());
    (void) exact(v);
    return v;
}

I WireReader::integer()
{
    const auto start = position;
    auto sign = word(), n = word();
    if (sign > 1 || n > data.size() - position)
        throw std::invalid_argument("Invalid exact geometry integer.");
    std::vector<unsigned char> bytes;
    bytes.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        bytes.push_back(static_cast<unsigned char>(data[position++]));
    I value;
    import_bits(value, bytes.begin(), bytes.end(), 8, false);
    if (sign)
        value = -value;
    if (canonical_encoding)
    {
        WireWriter canonical;
        canonical.integer(value);
        const auto encoded = data.subspan(start, position - start);
        if (canonical.data.size() != encoded.size() ||
            !std::equal(canonical.data.begin(), canonical.data.end(), encoded.begin()))
            throw std::invalid_argument("Noncanonical exact geometry integer encoding.");
    }
    return value;
}

Q WireReader::rational()
{
    const auto start = position;
    I n = integer(), d = integer();
    if (d <= 0)
        throw std::invalid_argument("Exact geometry denominator must be positive.");
    Q value(n, d);
    if (canonical_encoding)
    {
        WireWriter canonical;
        canonical.rational(value);
        const auto encoded = data.subspan(start, position - start);
        if (canonical.data.size() != encoded.size() ||
            !std::equal(canonical.data.begin(), canonical.data.end(), encoded.begin()))
            throw std::invalid_argument("Noncanonical exact geometry rational encoding.");
    }
    return value;
}

void WireReader::finish()
{
    if (position != data.size())
        throw std::invalid_argument("Trailing exact geometry packet data.");
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source)
{
    auto impl = std::make_shared<Impl>();
    const auto origin = point(source.origin());
    for (const auto& vertex : source.local_points())
    {
        P p;
        for (std::size_t axis = 0; axis < 3; ++axis)
            p[axis] = origin[axis] + exact_wide(vertex.value[axis]);
        impl->poly.vertices.push_back(std::move(p));
    }
    impl->poly.faces = source.faces();
    validate(impl->poly);
    impl->origin = source.origin();
    impl->product = product_description(impl->poly);
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry(std::move(impl)));
}
} // namespace SimpleFluid::PLIC
