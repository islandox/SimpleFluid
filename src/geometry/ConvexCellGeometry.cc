/** @file ConvexCellGeometry.cc
 * @brief Exact represented-input convex clipping and moment certificates.
 */
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/ExactPlanarFace.hh"
#include "geometry/ExactPolyhedralComplex.hh"
#include "geometry/mesh/ExactExtrudedConstruction.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <algorithm>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
namespace
{
using Q = boost::multiprecision::cpp_rational;
using I = boost::multiprecision::cpp_int;
using P = std::array<Q, 3>;
using Face = std::vector<std::size_t>;
struct Poly
{
    std::vector<P> vertices;
    std::vector<Face> faces;
};
struct Plane
{
    P normal, anchor;
    Q offset = 0;
};
struct Product
{
    std::vector<P> base;
    Q lower, upper;
};

Q exact(double x)
{
    if (!std::isfinite(x))
        throw std::invalid_argument("Convex geometry requires finite coordinates.");
    const auto bits = std::bit_cast<std::uint64_t>(x);
    const auto exponent = (bits >> 52) & 2047;
    I mantissa = bits & ((std::uint64_t{1} << 52) - 1);
    if (exponent)
        mantissa += I{1} << 52;
    if (bits >> 63)
        mantissa = -mantissa;
    const int shift = exponent ? static_cast<int>(exponent) - 1023 - 52 : -1074;
    return shift >= 0 ? Q(mantissa << shift) : Q(mantissa, I{1} << -shift);
}

Q exact_wide(long double x)
{
    if (!std::isfinite(x))
        throw std::runtime_error("Unrepresentable geometry measure.");
    if (x == 0)
        return 0;
    int exponent;
    auto fraction = std::frexp(std::abs(x), &exponent);
    I mantissa = 0;
    for (int bit = 0; bit < std::numeric_limits<long double>::digits; ++bit)
    {
        fraction *= 2;
        mantissa <<= 1;
        if (fraction >= 1)
        {
            ++mantissa;
            fraction -= 1;
        }
    }
    if (x < 0)
        mantissa = -mantissa;
    const int shift = exponent - std::numeric_limits<long double>::digits;
    return shift >= 0 ? Q(mantissa << shift) : Q(mantissa, I{1} << -shift);
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
    return dot(plane.normal, sub(p, plane.anchor)) - plane.offset;
}

template<class Storage = long double>
CertifiedGeometryValue measured(const Q& value, bool allow_underflow_interval = false)
{
    const long double v = value.convert_to<Storage>();
    if (!std::isfinite(v) || (v == 0 && value != 0 && !allow_underflow_interval))
        throw std::runtime_error("Positive/nonzero exact geometry is not representable in admitted measure storage.");
    Q error = value - exact_wide(v);
    if (error < 0)
        error = -error;
    if (error == 0)
        return {v, 0, value != 0};
    // Outward rounding in Storage, including a subnormal enclosure when the
    // exact first moment lies below that format's minimum positive value.
    const long double bound = std::nextafter(error.convert_to<Storage>(), std::numeric_limits<Storage>::infinity());
    if (!std::isfinite(bound) || exact_wide(bound) < error)
        throw std::runtime_error("Geometry conversion error could not be certified.");
    return {v, bound, value != 0};
}

CertifiedGeometryValue square_root(const Q& squared)
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

struct ExactMoments
{
    Q volume;
    P first;
};
ExactMoments exact_moments(const Poly& poly, const P& origin)
{
    ExactMoments result{};
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const P a = sub(poly.vertices[face[0]], origin), b = sub(poly.vertices[face[i]], origin),
                    c = sub(poly.vertices[face[i + 1]], origin);
            const Q volume = dot(a, cross(b, c)) / 6;
            result.volume += volume;
            result.first = add(result.first, mul(add(add(a, b), c), Q(volume / 4)));
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

void validate(const Poly& poly)
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
    (void) measured(m.volume);
    for (const auto& first : m.first)
        (void) measured(first, true);
}

struct Cut
{
    std::optional<Poly> poly;
    Face cap;
    std::vector<CertifiedGeometryValue> apertures;
    std::vector<std::size_t> face_sources;
};
Cut clipped(const Poly& input, const Plane& plane)
{
    Cut result;
    result.apertures.resize(input.faces.size());
    bool negative = false, positive = false;
    for (const auto& p : input.vertices)
    {
        const auto d = distance(plane, p);
        negative = negative || d < 0;
        positive = positive || d > 0;
    }
    if (!negative && positive)
    {
        // Zero 3D measure does not erase a represented face contact.
        for (std::size_t f = 0; f < input.faces.size(); ++f)
            if (std::all_of(input.faces[f].begin(), input.faces[f].end(),
                    [&](auto i) { return distance(plane, input.vertices[i]) == 0; }))
            {
                const auto area = area_vector(input, input.faces[f]);
                result.apertures[f] = square_root(dot(area, area));
            }
        return result;
    }
    if (!positive)
    {
        result.poly = input;
        for (std::size_t f = 0; f < input.faces.size(); ++f)
            result.face_sources.push_back(f);
        for (std::size_t f = 0; f < input.faces.size(); ++f)
        {
            const auto area = area_vector(input, input.faces[f]);
            result.apertures[f] = square_root(dot(area, area));
        }
        return result;
    }
    Poly output;
    std::map<P, std::size_t> indices;
    auto index = [&](const P& p)
    {
        const auto [it, inserted] = indices.emplace(p, output.vertices.size());
        if (inserted)
            output.vertices.push_back(p);
        return it->second;
    };
    std::map<std::size_t, std::size_t> cap_next;
    for (std::size_t f = 0; f < input.faces.size(); ++f)
    {
        Face face;
        auto append = [&](const P& p)
        {
            const auto n = index(p);
            if (face.empty() || face.back() != n)
                face.push_back(n);
        };
        const auto& original = input.faces[f];
        for (std::size_t i = 0; i < original.size(); ++i)
        {
            const auto& a = input.vertices[original[i]];
            const auto& b = input.vertices[original[(i + 1) % original.size()]];
            const Q da = distance(plane, a), db = distance(plane, b);
            if (da <= 0)
                append(a);
            if ((da < 0 && db > 0) || (da > 0 && db < 0))
                append(add(a, mul(sub(b, a), Q(da / (da - db)))));
        }
        if (face.size() > 1 && face.front() == face.back())
            face.pop_back();
        if (face.size() < 3 || zero(area_vector(output, face)))
            continue;
        const auto area = area_vector(output, face);
        result.apertures[f] = square_root(dot(area, area));
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto a = face[i], b = face[(i + 1) % face.size()];
            if (distance(plane, output.vertices[a]) == 0 && distance(plane, output.vertices[b]) == 0)
                if (!cap_next.emplace(b, a).second)
                    throw std::runtime_error("Clipped cap has unresolved/nonmanifold edge incidence.");
        }
        output.faces.push_back(std::move(face));
        result.face_sources.push_back(f);
    }
    if (cap_next.size() < 3)
        throw std::runtime_error("Positive convex cut has an unresolved cap.");
    auto next = cap_next.begin()->first;
    const auto first = next;
    do
    {
        if (result.cap.size() >= cap_next.size() || !cap_next.contains(next))
            throw std::runtime_error("Clipped cap is not one closed cycle.");
        result.cap.push_back(next);
        next = cap_next.at(next);
    } while (next != first);
    if (result.cap.size() != cap_next.size())
        throw std::runtime_error("Clipped cap has disconnected cycles.");
    output.faces.push_back(result.cap);
    result.face_sources.push_back(std::numeric_limits<std::size_t>::max());
    // Face tangencies can introduce vertices unused by every positive-area face.
    Poly compact;
    std::map<std::size_t, std::size_t> compact_ids;
    for (const auto& face : output.faces)
    {
        Face mapped;
        for (const auto old : face)
        {
            const auto [it, inserted] = compact_ids.emplace(old, compact.vertices.size());
            if (inserted)
                compact.vertices.push_back(output.vertices[old]);
            mapped.push_back(it->second);
        }
        compact.faces.push_back(std::move(mapped));
    }
    result.cap = compact.faces.back();
    validate(compact);
    result.poly = std::move(compact);
    return result;
}

std::optional<Poly> intersection(const Poly& a, const Poly& b)
{
    std::optional<Poly> result = a;
    for (const auto& plane : planes(b))
    {
        result = clipped(*result, plane).poly;
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

ExactMoments product_moments(const Product& product, const P& origin)
{
    ExactMoments result{};
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
struct WireWriter
{
    std::vector<char> data;
    void word(std::uint64_t value)
    {
        for (unsigned b = 0; b < 8; ++b)
            data.push_back(static_cast<char>(value >> (8 * b)));
    }
    void real(double value)
    {
        (void) exact(value);
        word(std::bit_cast<std::uint64_t>(value));
    }
    void integer(I value)
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
    void rational(const Q& value)
    {
        integer(boost::multiprecision::numerator(value));
        integer(boost::multiprecision::denominator(value));
    }
};
struct WireReader
{
    std::span<const char> data;
    std::size_t position = 0;
    std::uint64_t word()
    {
        if (data.size() - position < 8)
            throw std::invalid_argument("Truncated exact geometry certificate.");
        std::uint64_t v = 0;
        for (unsigned b = 0; b < 8; ++b)
            v |= std::uint64_t(static_cast<unsigned char>(data[position++])) << (8 * b);
        return v;
    }
    std::size_t count()
    {
        auto n = word();
        if (n > (data.size() - position) / 8 + 1)
            throw std::invalid_argument("Invalid exact geometry count.");
        return static_cast<std::size_t>(n);
    }
    double real()
    {
        auto v = std::bit_cast<double>(word());
        (void) exact(v);
        return v;
    }
    I integer()
    {
        auto sign = word(), n = word();
        if (sign > 1 || n > data.size() - position)
            throw std::invalid_argument("Invalid exact geometry integer.");
        std::vector<unsigned char> bytes;
        bytes.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            bytes.push_back(static_cast<unsigned char>(data[position++]));
        I value;
        import_bits(value, bytes.begin(), bytes.end(), 8, false);
        return sign ? -value : value;
    }
    Q rational()
    {
        I n = integer(), d = integer();
        if (d <= 0)
            throw std::invalid_argument("Exact geometry denominator must be positive.");
        return Q(n, d);
    }
    void finish()
    {
        if (position != data.size())
            throw std::invalid_argument("Trailing exact geometry packet data.");
    }
};

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

} // namespace

struct ConvexCellGeometry::Impl
{
    Poly poly;
    GeometryPoint origin;
    std::optional<Product> product;
};

std::vector<char> ConvexCellGeometry::encode_exact_certificate() const
{
    WireWriter w;
    w.word(0x5346455841435431ULL);
    w.word(1);
    for (auto v : impl_->origin)
        w.real(v);
    w.word(impl_->poly.vertices.size());
    const auto origin = point(impl_->origin);
    for (const auto& p : impl_->poly.vertices)
        for (const auto& q : sub(p, origin))
            w.rational(q);
    w.word(impl_->poly.faces.size());
    for (const auto& face : impl_->poly.faces)
    {
        w.word(face.size());
        for (auto id : face)
            w.word(id);
    }
    return std::move(w.data);
}

ConvexGeometryPtr ConvexCellGeometry::decode_exact_certificate(std::span<const char> packet)
{
    WireReader r{packet};
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
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
}

double CertifiedGeometryValue::as_double() const
{
    const double result = static_cast<double>(value);
    if (!std::isfinite(result) || (result == 0 && (value != 0 || exact_nonzero)))
        throw std::runtime_error("Geometry measure cannot be represented in binary64 storage.");
    return result;
}

long double CertifiedGeometryValue::binary64_absolute_error() const
{
    const auto rounded = as_double();
    const auto error = std::abs(static_cast<long double>(rounded) - value) + absolute_error;
    return error == 0 ? 0 : std::nextafter(error, std::numeric_limits<long double>::infinity());
}

ConvexCellGeometry::ConvexCellGeometry(std::shared_ptr<const Impl> implementation) : impl_(std::move(implementation)) {}

ConvexGeometryPtr ConvexCellGeometry::make(std::span<const GeometryPoint> vertices, const std::vector<Face>& faces)
{
    auto impl = std::make_shared<Impl>();
    for (const auto& p : vertices)
        impl->poly.vertices.push_back(point(p));
    impl->poly.faces = faces;
    validate(impl->poly);
    impl->origin = vertices[0];
    impl->product = product_description(impl->poly);
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
}

ConvexGeometryPtr ConvexCellGeometry::box(GeometryPoint lower, GeometryPoint upper)
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

ConvexGeometryPtr ConvexCellGeometry::extruded_polygon(
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
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
}

NativeConvexShell ConvexCellGeometry::native_cell_shell(const MultiRegionMesh& mesh, std::uint64_t cell)
{
    if (mesh.has_exact_construction())
        throw std::invalid_argument(
            "Constructed native geometry requires lossless exact cell certificates, not display-node shells.");
    const auto execution = mesh.acquire_execution_view();
    if (mesh.region_layout(mesh.native_cell(cell).first).family == RegionLayout::Family::Cylindrical)
        throw std::invalid_argument("Analytic cylindrical cells are not planar-faced convex shell providers.");
    for (const auto& interface : mesh.interfaces())
        std::visit(
            [](const auto& join)
            {
                if constexpr (requires { join.periodic_translation; })
                    if (join.periodic_translation)
                        throw std::invalid_argument("Native convex shell extraction requires nontranslated joins.");
            },
            interface);
    Poly shell;
    std::vector<GeometryPoint> displayed;
    std::map<P, std::size_t> ids;
    for (const auto face_id : mesh.cell_faces(cell))
    {
        if (mesh.region_layout(mesh.native_face(face_id).region).family == RegionLayout::Family::Cylindrical)
            throw std::invalid_argument(
                "Analytic cylindrical interface faces cannot be replaced by planar node hulls.");
        Face face;
        for (const auto node : mesh.face_nodes(face_id))
        {
            const auto native = mesh.node_coordinates(node);
            const GeometryPoint display{native.x, native.y, native.z};
            const auto exact_point = point(display);
            const auto [it, inserted] = ids.emplace(exact_point, shell.vertices.size());
            if (inserted)
            {
                shell.vertices.push_back(exact_point);
                displayed.push_back(display);
            }
            face.push_back(it->second);
        }
        if (face.size() < 3)
            throw std::invalid_argument("Native cell has an invalid face loop.");
        const auto native_normal = mesh.face_area_vector_outward(face_id, cell);
        const Q orientation = dot(area_vector(shell, face), point({native_normal.x, native_normal.y, native_normal.z}));
        if (orientation == 0)
            throw std::invalid_argument("Native face orientation is unresolved.");
        if (orientation < 0)
            std::reverse(face.begin(), face.end());
        shell.faces.push_back(std::move(face));
    }
    return canonical_native_shell(displayed, shell.faces);
}

ConvexGeometryPtr ConvexCellGeometry::from_native_cell(const MultiRegionMesh& mesh, std::uint64_t cell)
{
    (void) mesh.geometry_epoch();
    if (mesh.has_exact_construction() && !mesh.interfaces().empty() && !mesh.has_qualified_exact_interfaces())
        throw std::invalid_argument(
            "Joined constructed providers require separately qualified exact interface geometry.");
    if (!mesh.interfaces().empty() && mesh.has_qualified_exact_interfaces())
        return from_native_cell_atomic(mesh, cell);
    const auto [region, local] = mesh.native_cell(cell);
    auto exact_geometry = std::visit(
        [&](const auto& r) -> ConvexGeometryPtr
        {
            if constexpr (requires { r.geometry().certified_cell_geometry(local); })
                return r.geometry().certified_cell_geometry(local);
            else
                return {};
        },
        mesh.regions().at(region));
    if (exact_geometry)
        return exact_geometry;
    const auto shell = native_cell_shell(mesh, cell);
    return make(shell.vertices, shell.outward_faces);
}

ConvexGeometryPtr ConvexCellGeometry::from_native_shell(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& outward_faces)
{
    const auto shell = canonical_native_shell(vertices, outward_faces);
    return make(shell.vertices, shell.outward_faces);
}

GeometryMoments ConvexCellGeometry::moments() const
{
    const auto exact = impl_->product ? product_moments(*impl_->product, point(impl_->origin))
                                      : exact_moments(impl_->poly, point(impl_->origin));
    GeometryMoments result;
    result.origin = impl_->origin;
    result.volume = measured(exact.volume);
    for (std::size_t a = 0; a < 3; ++a)
        result.first_moment[a] = measured(exact.first[a], true);
    return result;
}

GeometryMoments ConvexCellGeometry::moments_binary64() const
{
    const auto exact = impl_->product ? product_moments(*impl_->product, point(impl_->origin))
                                      : exact_moments(impl_->poly, point(impl_->origin));
    GeometryMoments result;
    result.origin = impl_->origin;
    result.volume = measured<double>(exact.volume);
    for (std::size_t a = 0; a < 3; ++a)
        result.first_moment[a] = measured<double>(exact.first[a], true);
    return result;
}

std::vector<ConvexOctantPiece> ConvexCellGeometry::split_aabb_midpoints(double minimum_scaled_volume) const
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
    std::vector<ConvexOctantPiece> result;
    std::vector<ConvexGeometryPtr> certificates;
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
        auto geometry = ConvexGeometryPtr(new ConvexCellGeometry(std::move(implementation)));
        certificates.push_back(geometry);
        result.push_back({octant, std::move(geometry), std::move(origins)});
    }
    certify_partition(certificates);
    return result;
}
bool ConvexCellGeometry::quality_above(double minimum_scaled_volume) const
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
    const auto volume = exact_moments(impl_->poly, point(impl_->origin)).volume;
    return volume > exact(minimum_scaled_volume) * largest * largest * largest;
}
std::array<CertifiedGeometryValue, 3> ConvexCellGeometry::axis_spans() const
{
    P lo = impl_->poly.vertices[0], hi = lo;
    for (const auto& p : impl_->poly.vertices)
        for (std::size_t a = 0; a < 3; ++a)
        {
            lo[a] = std::min(lo[a], p[a]);
            hi[a] = std::max(hi[a], p[a]);
        }
    std::array<CertifiedGeometryValue, 3> result;
    for (std::size_t a = 0; a < 3; ++a)
        result[a] = measured(Q(hi[a] - lo[a]));
    return result;
}
bool ConvexCellGeometry::axis_span_exceeds(const ConvexCellGeometry& other, std::size_t axis, double ratio) const
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

std::size_t ConvexCellGeometry::num_faces() const noexcept
{
    return impl_->poly.faces.size();
}

ConvexClipResult ConvexCellGeometry::clip(const GeometryHalfspace& halfspace) const
{
    const Plane plane{point(halfspace.normal), point(halfspace.anchor), exact(halfspace.offset)};
    if (zero(plane.normal))
        throw std::invalid_argument("Clipping halfspace needs a nonzero finite normal.");
    auto inside = clipped(impl_->poly, plane);
    auto outside = clipped(impl_->poly, {mul(plane.normal, -1), plane.anchor, -plane.offset});
    ConvexClipResult result;
    auto retain = [&](std::optional<Poly>& poly) -> ConvexGeometryPtr
    {
        if (!poly)
            return {};
        auto impl = std::make_shared<Impl>();
        impl->poly = std::move(*poly);
        impl->origin = impl_->origin;
        impl->product = product_description(impl->poly);
        return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
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

ConvexGeometryPtr ConvexCellGeometry::intersect(const ConvexCellGeometry& other) const
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
        return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
    }
    auto poly = intersection(impl_->poly, other.impl_->poly);
    if (!poly)
        return {};
    auto impl = std::make_shared<Impl>();
    impl->poly = std::move(*poly);
    impl->origin = impl_->origin;
    impl->product = product_description(impl->poly);
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(impl)));
}

void ConvexCellGeometry::certify_partition(std::span<const ConvexGeometryPtr> pieces) const
{
    const P origin = point(impl_->origin);
    const auto expected = exact_moments(impl_->poly, origin);
    ExactMoments sum{};
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

bool ConvexCellGeometry::equivalent(const ConvexCellGeometry& other) const
{
    return contained(impl_->poly, other.impl_->poly) && contained(other.impl_->poly, impl_->poly);
}
struct ExactExtrudedBaseData
{
    std::vector<ExactPointRecipe> recipes;
    std::vector<std::size_t> active;
    std::vector<P> points, centers, deltas, edge_centers;
    std::vector<Q> areas;
    std::vector<MeshUtils::Vec3> display;
    std::vector<char> topology;
};
struct ExactExtrudedAxisValues
{
    std::vector<ExactScalarRecipe> zrecipes;
    std::vector<Q> z;
};
struct ExactExtrudedAxisData
{
    std::shared_ptr<ExactExtrudedAxisValues> values = std::make_shared<ExactExtrudedAxisValues>();
    std::vector<std::size_t> zactive;
    std::vector<double> display;
};
struct ExactExtrudedConstruction::Impl
{
    std::shared_ptr<const ExtrudedTopology> topology;
    std::shared_ptr<ExactExtrudedBaseData> base = std::make_shared<ExactExtrudedBaseData>();
    std::shared_ptr<ExactExtrudedAxisData> axis = std::make_shared<ExactExtrudedAxisData>();
    std::size_t begin = 0, layers = 0;
};

namespace
{
std::vector<char> topology_descriptor(const ExtrudedTopology& topology)
{
    WireWriter w;
    const auto& indexer = topology.indexer();
    w.word(indexer.num_nodes_per_layer);
    w.word(indexer.num_layers);
    w.word(indexer.num_cells_per_layer);
    for (std::size_t c = 0; c < indexer.num_cells_per_layer; ++c)
    {
        auto nodes = topology.base_cell_nodes(c);
        w.word(nodes.size());
        for (auto n : nodes)
            w.word(n);
    }
    const auto ids = topology.boundary_batch_ids();
    w.word(ids.size());
    for (auto id : ids)
    {
        const auto& name = topology.boundary_batch_name(id);
        w.word(name.size());
        for (unsigned char ch : name)
            w.word(ch);
    }
    for (const auto& e : topology.native_topology().side_faces())
    {
        w.word(e.nodes[0]);
        w.word(e.nodes[1]);
        w.word(e.owner);
        w.word(e.neighbor);
        w.word(e.boundary_id);
    }
    return std::move(w.data);
}
MeshUtils::Vec3 rounded_point(const P& p)
{
    return {measured(p[0]).as_double(), measured(p[1]).as_double(), measured(p[2]).as_double()};
}
} // namespace

std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::make(
    std::shared_ptr<const ExtrudedTopology> topology, std::vector<ExactPointRecipe> recipes,
    std::vector<std::size_t> active, std::vector<ExactScalarRecipe> zrecipes, std::vector<std::size_t> zactive)
{
    if (!topology || active.size() != topology->indexer().num_nodes_per_layer ||
        zactive.size() != std::size_t(topology->indexer().num_layers) + 1)
        throw std::invalid_argument("Exact extrusion construction and retained topology differ.");
    auto impl = std::make_shared<Impl>();
    impl->topology = std::move(topology);
    impl->base->topology = impl->topology->encode_base();
    impl->base->recipes = std::move(recipes);
    impl->axis->values->zrecipes = std::move(zrecipes);
    impl->base->active = std::move(active);
    impl->axis->zactive = std::move(zactive);
    for (std::size_t i = 0; i < impl->base->recipes.size(); ++i)
    {
        const auto& recipe = impl->base->recipes[i];
        P p{};
        (void) point(recipe.root);
        if (recipe.parents.empty())
            p = point(recipe.root);
        else
        {
            for (auto parent : recipe.parents)
            {
                if (parent >= i)
                    throw std::invalid_argument("Exact point construction is not an acyclic ancestry DAG.");
                p = add(p, impl->base->points[parent]);
            }
            p = mul(p, Q(1, recipe.parents.size()));
        }
        if (p[2] != 0)
            throw std::invalid_argument("Exact extrusion base must lie in XY.");
        impl->base->points.push_back(std::move(p));
    }
    std::set<std::size_t> unique;
    std::set<P> unique_points;
    std::set<GeometryPoint> display;
    for (auto id : impl->base->active)
    {
        if (id >= impl->base->points.size() || !unique.insert(id).second ||
            !unique_points.insert(impl->base->points[id]).second)
            throw std::invalid_argument("Exact extrusion has invalid/duplicate active points.");
        const auto p = rounded_point(impl->base->points[id]);
        if (!display.insert({p.x, p.y, p.z}).second)
            throw std::invalid_argument("Exact extrusion display vertices are indistinguishable.");
    }
    for (std::size_t i = 0; i < impl->axis->values->zrecipes.size(); ++i)
    {
        const auto& recipe = impl->axis->values->zrecipes[i];
        Q z = 0;
        (void) exact(recipe.root);
        if (recipe.parents.empty())
            z = exact(recipe.root);
        else
        {
            for (auto parent : recipe.parents)
            {
                if (parent >= i)
                    throw std::invalid_argument("Exact axial construction is not acyclic.");
                z += impl->axis->values->z[parent];
            }
            z /= recipe.parents.size();
        }
        impl->axis->values->z.push_back(z);
    }
    Q previous = 0;
    double previous_display = 0;
    for (std::size_t k = 0; k < impl->axis->zactive.size(); ++k)
    {
        const auto id = impl->axis->zactive[k];
        if (id >= impl->axis->values->z.size())
            throw std::invalid_argument("Exact axial point index is invalid.");
        const auto value = measured(impl->axis->values->z[id]).as_double();
        if (k && (!(impl->axis->values->z[id] > previous) || !(value > previous_display)))
            throw std::invalid_argument("Exact axial/display edges must be distinct and increasing.");
        previous = impl->axis->values->z[id];
        previous_display = value;
        if (k)
        {
            const auto lo = measured(impl->axis->values->z[impl->axis->zactive[k - 1]]).as_double();
            const auto center =
                measured(Q((impl->axis->values->z[id] + impl->axis->values->z[impl->axis->zactive[k - 1]]) / 2))
                    .as_double();
            if (!(center > lo && center < value))
                throw std::invalid_argument("Exact extrusion has unrepresentable signed axial metric distances.");
        }
    }
    for (std::size_t c = 0; c < impl->topology->indexer().num_cells_per_layer; ++c)
    {
        Product product;
        product.lower = 0;
        product.upper = 1;
        for (auto n : impl->topology->base_cell_nodes(c))
            product.base.push_back(impl->base->points[impl->base->active[n]]);
        for (std::size_t i = 0; i < product.base.size(); ++i)
            if (cross(sub(product.base[(i + 1) % product.base.size()], product.base[i]),
                    sub(product.base[(i + 2) % product.base.size()], product.base[(i + 1) % product.base.size()]))[2] <=
                0)
                throw std::invalid_argument("Exact extrusion base must be strictly convex and counterclockwise.");
        (void) product_poly(product);
        std::vector<P> display_polygon;
        for (const auto& p : product.base)
        {
            auto v = rounded_point(p);
            display_polygon.push_back(point({v.x, v.y, 0}));
        }
        for (std::size_t i = 0; i < display_polygon.size(); ++i)
        {
            const auto a = display_polygon[i], b = display_polygon[(i + 1) % display_polygon.size()],
                       c = display_polygon[(i + 2) % display_polygon.size()];
            if (cross(sub(b, a), sub(c, b))[2] <= 0)
                throw std::invalid_argument(
                    "Exact extrusion display polygon is not a distinct strictly convex native loop.");
            for (const auto& p : display_polygon)
                if (cross(sub(b, a), sub(p, a))[2] < 0)
                    throw std::invalid_argument("Exact extrusion display polygon is nonsimple.");
        }
        const auto m = product_moments(product, P{});
        impl->base->areas.push_back(m.volume);
        impl->base->centers.push_back({m.first[0] / m.volume, m.first[1] / m.volume, 0});
    }
    for (const auto& edge : impl->topology->native_topology().side_faces())
    {
        const auto a = impl->base->points[impl->base->active[edge.nodes[0]]],
                   b = impl->base->points[impl->base->active[edge.nodes[1]]];
        impl->base->deltas.push_back(sub(b, a));
        impl->base->edge_centers.push_back(mul(add(a, b), Q(1, 2)));
    }
    for (std::size_t e = 0; e < impl->base->deltas.size(); ++e)
    {
        const auto& edge = impl->topology->native_topology().side_face(e);
        const auto face = rounded_point(impl->base->edge_centers[e]);
        const auto owner = rounded_point(impl->base->centers.at(edge.owner));
        const auto& d = impl->base->deltas[e];
        const auto normal = rounded_point({d[1], -d[0], 0});
        const auto owner_distance = (face - owner).dot(normal);
        if (!std::isfinite(owner_distance) || !(owner_distance > 0))
            throw std::invalid_argument("Exact extrusion has unrepresentable signed base metric distances.");
        if (edge.neighbor != SemiStructMeshTopo::invalid_ordinal)
        {
            const auto neighbor = rounded_point(impl->base->centers.at(edge.neighbor));
            const auto neighbor_distance = (face - neighbor).dot(normal);
            if (!std::isfinite(neighbor_distance) || !(neighbor_distance < 0))
                throw std::invalid_argument("Exact extrusion neighbor metric distance is unresolved.");
        }
    }
    impl->layers = impl->axis->zactive.size() - 1;
    for (auto id : impl->base->active)
        impl->base->display.push_back(rounded_point(impl->base->points[id]));
    for (auto id : impl->axis->zactive)
        impl->axis->display.push_back(measured(impl->axis->values->z[id]).as_double());
    return std::shared_ptr<const ExactExtrudedConstruction>(new ExactExtrudedConstruction(std::move(impl)));
}

bool ExactExtrudedConstruction::matches_topology(const std::shared_ptr<const ExtrudedTopology>& topology) const noexcept
{
    return impl_->topology == topology;
}
std::vector<MeshUtils::Vec3> ExactExtrudedConstruction::display_points() const
{
    return impl_->base->display;
}
std::vector<double> ExactExtrudedConstruction::display_z() const
{
    const auto span = z_view();
    return {span.begin(), span.end()};
}
const std::vector<MeshUtils::Vec3>& ExactExtrudedConstruction::display_points_ref() const noexcept
{
    return impl_->base->display;
}
std::span<const double> ExactExtrudedConstruction::z_view() const noexcept
{
    return std::span<const double>(impl_->axis->display).subspan(impl_->begin, impl_->layers + 1);
}
std::size_t ExactExtrudedConstruction::first_layer() const noexcept
{
    return impl_->begin;
}
double ExactExtrudedConstruction::cell_volume(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    return measured(
        Q(impl_->base->areas.at(id.ij) * (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                                             impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)])))
        .as_double();
}
MeshUtils::Vec3 ExactExtrudedConstruction::cell_centroid(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    auto p = impl_->base->centers.at(id.ij);
    p[2] = (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)] +
               impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) /
           2;
    return rounded_point(p);
}
double ExactExtrudedConstruction::face_area(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return measured(impl_->base->areas.at(id.ij)).as_double();
    const Q height = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                     impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    return square_root(Q(dot(impl_->base->deltas.at(id.ij), impl_->base->deltas.at(id.ij)) * height * height))
        .as_double();
}
MeshUtils::Vec3 ExactExtrudedConstruction::face_centroid(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    auto p = id.orientation == SemiStructuredIndexer::AXIAL ? impl_->base->centers.at(id.ij)
                                                            : impl_->base->edge_centers.at(id.ij);
    p[2] = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    if (id.orientation != SemiStructuredIndexer::AXIAL)
        p[2] = (p[2] + impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) / 2;
    return rounded_point(p);
}
MeshUtils::Vec3 ExactExtrudedConstruction::face_area_vector(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return {0, 0, (id.k == 0 ? -1 : 1) * measured(impl_->base->areas.at(id.ij)).as_double()};
    const auto& d = impl_->base->deltas.at(id.ij);
    const Q h = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)] -
                impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    return rounded_point({d[1] * h, -d[0] * h, 0});
}
ConvexGeometryPtr ExactExtrudedConstruction::certified_cell_geometry(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    Product product;
    for (auto n : impl_->topology->base_cell_nodes(id.ij))
        product.base.push_back(impl_->base->points[impl_->base->active[n]]);
    product.lower = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)];
    product.upper = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)];
    auto geometry = std::make_shared<ConvexCellGeometry::Impl>();
    geometry->poly = product_poly(product);
    geometry->product = std::move(product);
    const auto first = rounded_point(geometry->poly.vertices[0]);
    geometry->origin = {first.x, first.y, first.z};
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(geometry)));
}
std::vector<char> ExactExtrudedConstruction::encode() const
{
    WireWriter w;
    w.word(0x5346455854524331ULL);
    w.word(1);
    const auto signature = topology_descriptor(*impl_->topology);
    w.word(signature.size());
    for (auto byte : signature)
        w.data.push_back(byte);
    w.word(impl_->base->recipes.size());
    for (const auto& p : impl_->base->recipes)
    {
        for (auto x : p.root)
            w.real(x);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(impl_->base->active.size());
    for (auto id : impl_->base->active)
        w.word(id);
    w.word(impl_->axis->values->zrecipes.size());
    for (const auto& p : impl_->axis->values->zrecipes)
    {
        w.real(p.root);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(impl_->layers + 1);
    for (std::size_t k = 0; k <= impl_->layers; ++k)
        w.word(impl_->axis->zactive.at(impl_->begin + k));
    return std::move(w.data);
}
std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::decode(
    std::shared_ptr<const ExtrudedTopology> topology, std::span<const char> packet)
{
    if (!topology)
        throw std::invalid_argument("Exact extrusion decode needs retained topology.");
    WireReader r{packet};
    if (r.word() != 0x5346455854524331ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact extrusion packet.");
    auto n = r.word();
    if (n > r.data.size() - r.position)
        throw std::invalid_argument("Invalid exact extrusion topology descriptor.");
    const auto expected = topology_descriptor(*topology);
    if (n != expected.size() || !std::equal(expected.begin(), expected.end(), r.data.begin() + r.position))
        throw std::invalid_argument("Exact extrusion packet belongs to different topology/order.");
    r.position += n;
    std::vector<ExactPointRecipe> points;
    std::vector<ExactScalarRecipe> z;
    std::vector<std::size_t> active, zactive;
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ExactPointRecipe p;
        for (auto& x : p.root)
            x = r.real();
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(r.word());
        points.push_back(std::move(p));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        active.push_back(r.word());
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ExactScalarRecipe p;
        p.root = r.real();
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(r.word());
        z.push_back(std::move(p));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        zactive.push_back(r.word());
    r.finish();
    return make(std::move(topology), std::move(points), std::move(active), std::move(z), std::move(zactive));
}
std::size_t ExactExtrudedConstruction::storage_bytes() const
{
    return base_storage_bytes() + axis_storage_bytes() + view_storage_bytes();
}
double ExactExtrudedConstruction::scaled_base_area(std::size_t base_cell) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Q largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->base->points[impl_->base->active[loop[i]]],
            impl_->base->points[impl_->base->active[loop[(i + 1) % loop.size()]]]);
        const Q length = dot(d, d);
        largest = std::max(largest, length);
    }
    return measured(Q(impl_->base->areas.at(base_cell) / largest)).as_double();
}
bool ExactExtrudedConstruction::base_quality_above(std::size_t base_cell, double minimum) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Q largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->base->points[impl_->base->active[loop[i]]],
            impl_->base->points[impl_->base->active[loop[(i + 1) % loop.size()]]]);
        const Q length = dot(d, d);
        largest = std::max(largest, length);
    }
    return impl_->base->areas.at(base_cell) > exact(minimum) * largest;
}
void ExactExtrudedConstruction::certify_base_embedding() const
{
    struct Candidate
    {
        std::size_t cell;
        Q xmin, xmax, ymin, ymax;
    };
    std::vector<Candidate> candidates;
    auto polygon = [&](std::size_t cell)
    {
        Product product;
        product.lower = 0;
        product.upper = 1;
        for (auto n : impl_->topology->base_cell_nodes(cell))
            product.base.push_back(impl_->base->points[impl_->base->active[n]]);
        return product;
    };
    for (std::size_t c = 0; c < impl_->base->areas.size(); ++c)
    {
        const auto p = polygon(c);
        Candidate candidate{c, p.base[0][0], p.base[0][0], p.base[0][1], p.base[0][1]};
        for (const auto& v : p.base)
        {
            candidate.xmin = std::min(candidate.xmin, v[0]);
            candidate.xmax = std::max(candidate.xmax, v[0]);
            candidate.ymin = std::min(candidate.ymin, v[1]);
            candidate.ymax = std::max(candidate.ymax, v[1]);
        }
        candidates.push_back(std::move(candidate));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.xmin < b.xmin; });
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        const auto& a = candidates[i];
        std::erase_if(active, [&](auto j) { return candidates[j].xmax <= a.xmin; });
        for (auto j : active)
        {
            const auto& b = candidates[j];
            if (std::max(a.ymin, b.ymin) < std::min(a.ymax, b.ymax) &&
                product_intersection(polygon(a.cell), polygon(b.cell)))
                throw std::invalid_argument("Extruded root base cells have positive overlapping interiors.");
        }
        active.push_back(i);
    }
    // A vertex strictly inside another native whole edge is an unresolved
    // T-junction/overlapping seam, even when all cell interiors are disjoint.
    std::vector<std::size_t> vertices = impl_->base->active;
    std::sort(vertices.begin(), vertices.end(),
        [&](auto a, auto b) { return impl_->base->points[a][0] < impl_->base->points[b][0]; });
    for (const auto& edge : impl_->topology->native_topology().side_faces())
    {
        const auto a = impl_->base->points[impl_->base->active[edge.nodes[0]]],
                   b = impl_->base->points[impl_->base->active[edge.nodes[1]]], delta = sub(b, a);
        const Q length = dot(delta, delta);
        const Q xmin = std::min(a[0], b[0]), xmax = std::max(a[0], b[0]), ymin = std::min(a[1], b[1]),
                ymax = std::max(a[1], b[1]);
        auto it = std::lower_bound(vertices.begin(), vertices.end(), xmin,
            [&](auto id, const Q& x) { return impl_->base->points[id][0] < x; });
        for (; it != vertices.end() && impl_->base->points[*it][0] <= xmax; ++it)
        {
            const auto& p = impl_->base->points[*it];
            if (p[1] < ymin || p[1] > ymax)
                continue;
            const auto offset = sub(p, a);
            const Q along = dot(offset, delta);
            if (zero(cross(delta, offset)) && along > 0 && along < length)
                throw std::invalid_argument("Extruded root embedding has an unresolved whole-edge T-junction.");
        }
    }
}

namespace
{
using Polygon = std::vector<P>;
P polygon_vector(const Polygon& polygon)
{
    P area{};
    if (polygon.size() < 3)
        return area;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
        area = add(area, cross(sub(polygon[i], polygon[0]), sub(polygon[i + 1], polygon[0])));
    return mul(area, Q(1, 2));
}
Polygon minimal_polygon(Polygon polygon)
{
    bool changed = true;
    while (changed && polygon.size() >= 3)
    {
        changed = false;
        for (std::size_t i = 0; i < polygon.size(); ++i)
            if (zero(cross(sub(polygon[i], polygon[(i + polygon.size() - 1) % polygon.size()]),
                    sub(polygon[(i + 1) % polygon.size()], polygon[i]))))
            {
                polygon.erase(polygon.begin() + i);
                changed = true;
                break;
            }
    }
    if (polygon.size() < 3 || zero(polygon_vector(polygon)))
        return {};
    auto first = std::min_element(polygon.begin(), polygon.end());
    std::rotate(polygon.begin(), first, polygon.end());
    return polygon;
}
Polygon polygon_intersection(Polygon polygon, const Polygon& clipper, const P& normal)
{
    for (std::size_t e = 0; e < clipper.size() && !polygon.empty(); ++e)
    {
        const Plane plane{cross(sub(clipper[(e + 1) % clipper.size()], clipper[e]), normal), clipper[e]};
        Polygon result;
        auto append = [&](const P& p)
        {
            if (result.empty() || result.back() != p)
                result.push_back(p);
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
        if (result.size() > 1 && result.front() == result.back())
            result.pop_back();
        polygon = std::move(result);
    }
    return minimal_polygon(std::move(polygon));
}
struct FaceMoment
{
    Q area;
    P first;
};
FaceMoment polygon_moments(const Polygon& polygon, std::size_t axis, const P& origin)
{
    FaceMoment result{};
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        const auto a = sub(polygon[0], origin), b = sub(polygon[i], origin), c = sub(polygon[i + 1], origin);
        const Q area = cross(sub(b, a), sub(c, a))[axis] / 2;
        result.area += area;
        result.first = add(result.first, mul(add(add(a, b), c), Q(area / 3)));
    }
    return result;
}
P polygon_center(const Polygon& polygon)
{
    const auto normal = polygon_vector(polygon);
    std::size_t axis = 0;
    while (axis < 3 && normal[axis] == 0)
        ++axis;
    if (axis == 3)
        throw std::invalid_argument("Exact face has zero area.");
    const auto m = polygon_moments(polygon, axis, polygon[0]);
    return add(polygon[0], mul(m.first, Q(1 / m.area)));
}
void surface_write(WireWriter& w, const PolyhedralSurfaceUse& surface)
{
    w.word(static_cast<std::uint64_t>(surface.key.kind));
    w.word(surface.key.region);
    w.word(surface.key.creator.root);
    w.word(surface.key.creator.path.size());
    for (auto v : surface.key.creator.path)
        w.word(v);
    w.word(surface.key.ordinal);
    w.word(surface.orientation == 1 ? 1 : 0);
    w.word(surface.exterior);
    w.word(surface.boundary.size());
    for (unsigned char ch : surface.boundary)
        w.data.push_back(static_cast<char>(ch));
}
PolyhedralSurfaceUse surface_read(WireReader& r)
{
    PolyhedralSurfaceUse surface;
    auto kind = r.word();
    if (kind > 1)
        throw std::invalid_argument("Unknown exact surface ancestry kind.");
    surface.key.kind = static_cast<PolyhedralSurfaceKey::Kind>(kind);
    surface.key.region = r.word();
    surface.key.creator.root = r.word();
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        surface.key.creator.path.push_back(r.word());
    surface.key.ordinal = r.word();
    auto orientation = r.word(), exterior = r.word();
    if (orientation > 1 || exterior > 1)
        throw std::invalid_argument("Invalid exact surface incidence.");
    surface.orientation = orientation ? 1 : -1;
    surface.exterior = exterior;
    n = r.word();
    if (n > r.data.size() - r.position)
        throw std::invalid_argument("Invalid exact boundary string.");
    surface.boundary.assign(r.data.begin() + r.position, r.data.begin() + r.position + n);
    r.position += n;
    return surface;
}
void cell_key_write(WireWriter& w, const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& key)
{
    const auto* ancestry = std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&key.ancestry);
    if (!ancestry)
        throw std::invalid_argument("Exact polyhedral complex needs convex-cell ancestry.");
    w.word(key.region);
    w.word(ancestry->root);
    w.word(ancestry->path.size());
    for (auto p : ancestry->path)
        w.word(p);
}
::SimpleFluid::Meshes::AMR::AdaptiveCellKey cell_key_read(WireReader& r)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey key;
    key.region = r.word();
    ::SimpleFluid::Meshes::AMR::ConvexCellAncestry ancestry;
    ancestry.root = r.word();
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        ancestry.path.push_back(r.word());
    key.ancestry = std::move(ancestry);
    return key;
}
using ComplexID = std::uint64_t;
constexpr ComplexID absent_cell = std::numeric_limits<ComplexID>::max();
struct ComplexCellRow
{
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey key;
    ConvexGeometryPtr geometry;
    double volume = 0;
    MeshUtils::Vec3 center{};
    std::vector<ComplexID> faces, nodes;
};
struct ComplexFaceRow
{
    PolyhedralSurfaceUse surface;
    ComplexID owner = absent_cell, neighbor = absent_cell;
    std::vector<ComplexID> nodes;
    double area = 0;
    MeshUtils::Vec3 vector{}, center{};
};
struct ComplexNodeRow
{
    std::uint64_t region = 0;
    P point;
};
} // namespace
struct ExactPolyhedralComplex::Impl
{
    ComplexID global_cells = 0, global_faces = 0, global_nodes = 0;
    std::map<ComplexID, ComplexCellRow> cells;
    std::map<ComplexID, ComplexFaceRow> faces;
    std::map<ComplexID, ComplexNodeRow> nodes;
    std::vector<ComplexID> cell_ids, resident_ids, face_ids, node_ids;
};

namespace
{
void complex_validate(ExactPolyhedralComplex::Impl& impl)
{
    impl.cell_ids.clear();
    impl.resident_ids.clear();
    impl.face_ids.clear();
    impl.node_ids.clear();
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> cell_keys;
    for (auto& [id, cell] : impl.cells)
    {
        if (id >= impl.global_cells ||
            !std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&cell.key.ancestry) ||
            !std::isfinite(cell.volume) || !(cell.volume > 0) || !std::isfinite(cell.center.x) ||
            !std::isfinite(cell.center.y) || !std::isfinite(cell.center.z))
            throw std::invalid_argument("Invalid exact complex cell metric/catalog record.");
        if (!cell_keys.insert(cell.key).second)
            throw std::invalid_argument("Exact complex cell ancestry is duplicated.");
        impl.cell_ids.push_back(id);
        if (cell.geometry)
            impl.resident_ids.push_back(id);
        else if (!cell.faces.empty() || !cell.nodes.empty())
            throw std::invalid_argument("Metric-only exact cell cannot retain topology.");
    }
    std::set<ComplexID> used_nodes, used_faces, used_cells;
    for (auto& [id, face] : impl.faces)
    {
        if (id >= impl.global_faces || !impl.cells.contains(face.owner) ||
            (face.neighbor != absent_cell && (!impl.cells.contains(face.neighbor) || face.owner >= face.neighbor)) ||
            face.surface.exterior != (face.neighbor == absent_cell) ||
            (face.surface.orientation != 1 && face.surface.orientation != -1))
            throw std::invalid_argument("Invalid exact complex face incidence/catalog.");
        if (!face.surface.exterior && !face.surface.boundary.empty())
            throw std::invalid_argument("Exact complex physical boundary identity is inconsistent.");
        used_cells.insert(face.owner);
        if (face.neighbor != absent_cell)
            used_cells.insert(face.neighbor);
        Polygon polygon;
        std::set<ComplexID> unique;
        for (auto node : face.nodes)
        {
            if (!impl.nodes.contains(node) || !unique.insert(node).second ||
                impl.nodes.at(node).region != face.surface.key.region)
                throw std::invalid_argument("Invalid exact face vertex identity.");
            polygon.push_back(impl.nodes.at(node).point);
            used_nodes.insert(node);
        }
        if (polygon.size() < 3)
            throw std::invalid_argument("Exact face has too few vertices.");
        const auto area = polygon_vector(polygon);
        if (zero(area))
            throw std::invalid_argument("Exact face is degenerate.");
        const Plane plane{area, polygon[0]};
        for (const auto& p : polygon)
            if (distance(plane, p) != 0)
                throw std::invalid_argument("Exact complex face is nonplanar.");
        for (std::size_t e = 0; e < polygon.size(); ++e)
            for (const auto& p : polygon)
                if (dot(area, cross(sub(polygon[(e + 1) % polygon.size()], polygon[e]), sub(p, polygon[e]))) < 0)
                    throw std::invalid_argument("Exact face is nonsimple or nonconvex.");
        face.area = square_root(dot(area, area)).as_double();
        face.vector = rounded_point(area);
        face.center = rounded_point(polygon_center(polygon));
        const auto owner_distance = (face.center - impl.cells.at(face.owner).center).dot(face.vector);
        if (!std::isfinite(owner_distance) || !(owner_distance > 0))
            throw std::invalid_argument("Exact face has an unresolved signed native owner distance.");
        if (face.neighbor != absent_cell)
        {
            const auto distance = (face.center - impl.cells.at(face.neighbor).center).dot(face.vector);
            if (!std::isfinite(distance) || !(distance < 0))
                throw std::invalid_argument("Exact face has an unresolved signed native neighbor distance.");
        }
        impl.face_ids.push_back(id);
    }
    for (auto& [id, cell] : impl.cells)
        if (cell.geometry)
        {
            Poly shell;
            std::map<ComplexID, std::size_t> vertices;
            std::set<ComplexID> face_ids, nodes;
            for (auto face_id : cell.faces)
            {
                if (!impl.faces.contains(face_id) || !face_ids.insert(face_id).second)
                    throw std::invalid_argument("Exact resident cell has missing/repeated face.");
                const auto& face = impl.faces.at(face_id);
                if (face.owner != id && face.neighbor != id)
                    throw std::invalid_argument("Exact resident cell has nonincident face.");
                used_faces.insert(face_id);
                Face loop;
                for (auto node : face.nodes)
                {
                    const auto [it, inserted] = vertices.emplace(node, shell.vertices.size());
                    if (inserted)
                        shell.vertices.push_back(impl.nodes.at(node).point);
                    loop.push_back(it->second);
                    nodes.insert(node);
                }
                if (face.owner != id)
                    std::reverse(loop.begin(), loop.end());
                shell.faces.push_back(std::move(loop));
            }
            validate(shell);
            // Access to the authoritative shell is checked by the member caller;
            // closure and moments are certified here without narrowing vertices.
            std::vector<ComplexID> expected_nodes(nodes.begin(), nodes.end());
            if (cell.nodes != expected_nodes)
                throw std::invalid_argument("Exact cell node catalog disagrees with its incident faces.");
        }
    for (const auto& [face_id, face] : impl.faces)
        for (const auto incident : {face.owner, face.neighbor})
            if (incident != absent_cell && impl.cells.at(incident).geometry &&
                std::find(impl.cells.at(incident).faces.begin(), impl.cells.at(incident).faces.end(), face_id) ==
                    impl.cells.at(incident).faces.end())
                throw std::invalid_argument("Exact face is missing reciprocal membership in its full incident cell.");
    if (used_faces.size() != impl.faces.size() || used_nodes.size() != impl.nodes.size() ||
        used_cells.size() != impl.cells.size())
        throw std::invalid_argument("Exact sparse complex contains orphan cell/face/node records.");
    std::set<std::pair<std::uint64_t, GeometryPoint>> display_points;
    for (const auto& [id, node] : impl.nodes)
    {
        if (id >= impl.global_nodes)
            throw std::invalid_argument("Exact complex node ID exceeds global catalog.");
        const auto display = rounded_point(node.point);
        if (!display_points.insert({node.region, {display.x, display.y, display.z}}).second)
            throw std::invalid_argument("Distinct exact explicit vertices collide in native display coordinates.");
        impl.node_ids.push_back(id);
    }
}
} // namespace

std::shared_ptr<const ExactPolyhedralComplex> ExactPolyhedralComplex::make(std::vector<PolyhedralComplexCell> input)
{
    if (input.empty())
        throw std::invalid_argument("Exact complex root needs at least one actual convex cell.");
    auto impl = std::make_shared<Impl>();
    impl->global_cells = input.size();
    struct Incident
    {
        ComplexID cell;
        Polygon polygon;
        PolyhedralSurfaceUse use;
    };
    std::map<PolyhedralSurfaceKey, std::vector<Incident>> surfaces;
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> keys;
    std::map<std::uint64_t, std::set<P>> candidate_vertices;
    struct Box
    {
        ComplexID cell;
        P lo, hi;
    };
    std::vector<Box> boxes;
    for (const auto& cell : input)
    {
        if (!cell.geometry || cell.gid >= input.size() || impl->cells.contains(cell.gid) ||
            !keys.insert(cell.key).second || cell.surfaces.size() != cell.geometry->impl_->poly.faces.size() ||
            !std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&cell.key.ancestry))
            throw std::invalid_argument("Exact complex cell binding/topology is invalid.");
        const auto& poly = cell.geometry->impl_->poly;
        const auto m = exact_moments(poly, point(cell.geometry->impl_->origin));
        const auto center = add(point(cell.geometry->impl_->origin), mul(m.first, Q(1 / m.volume)));
        impl->cells.emplace(cell.gid,
            ComplexCellRow{cell.key, cell.geometry, measured(m.volume).as_double(), rounded_point(center), {}, {}});
        Box box{cell.gid, poly.vertices[0], poly.vertices[0]};
        for (const auto& p : poly.vertices)
        {
            candidate_vertices[cell.key.region].insert(p);
            for (std::size_t a = 0; a < 3; ++a)
            {
                box.lo[a] = std::min(box.lo[a], p[a]);
                box.hi[a] = std::max(box.hi[a], p[a]);
            }
        }
        boxes.push_back(std::move(box));
        for (std::size_t f = 0; f < poly.faces.size(); ++f)
        {
            const auto& use = cell.surfaces[f];
            if (use.key.region != cell.key.region || (use.orientation != 1 && use.orientation != -1))
                throw std::invalid_argument("Surface lineage/region/orientation is invalid.");
            Polygon polygon;
            for (auto index : poly.faces[f])
                polygon.push_back(poly.vertices[index]);
            if (use.orientation < 0)
                std::reverse(polygon.begin(), polygon.end());
            surfaces[use.key].push_back({cell.gid, minimal_polygon(std::move(polygon)), use});
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) { return a.lo[0] < b.lo[0]; });
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& a = boxes[i];
        std::erase_if(active, [&](auto j) { return boxes[j].hi[0] <= a.lo[0]; });
        for (auto j : active)
        {
            const auto& b = boxes[j];
            if (std::max(a.lo[1], b.lo[1]) < std::min(a.hi[1], b.hi[1]) &&
                std::max(a.lo[2], b.lo[2]) < std::min(a.hi[2], b.hi[2]) &&
                intersection(
                    impl->cells.at(a.cell).geometry->impl_->poly, impl->cells.at(b.cell).geometry->impl_->poly))
                throw std::invalid_argument("Exact complex root/leaf interiors overlap.");
        }
        active.push_back(i);
    }
    struct Tile
    {
        PolyhedralSurfaceUse use;
        ComplexID owner, neighbor;
        Polygon canonical;
    };
    std::vector<Tile> tiles;
    for (const auto& [key, entries] : surfaces)
    {
        const auto normal = polygon_vector(entries[0].polygon);
        const auto anchor = entries[0].polygon[0];
        const bool exterior = entries[0].use.exterior;
        const auto boundary = entries[0].use.boundary;
        std::vector<std::size_t> plus, minus;
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const auto& entry = entries[i];
            if (entry.polygon.empty() || dot(polygon_vector(entry.polygon), normal) <= 0 ||
                entry.use.exterior != exterior || entry.use.boundary != boundary)
                throw std::invalid_argument("Exact ancestral surface has inconsistent normal/patch membership.");
            for (const auto& p : entry.polygon)
                if (dot(normal, sub(p, anchor)) != 0)
                    throw std::invalid_argument("One exact surface ancestry refers to different represented planes.");
            (entry.use.orientation > 0 ? plus : minus).push_back(i);
        }
        auto certify_side = [&](const auto& side)
        {
            for (std::size_t i = 0; i < side.size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    if (!polygon_intersection(entries[side[i]].polygon, entries[side[j]].polygon, normal).empty())
                        throw std::invalid_argument("Same-side ancestral face patches overlap.");
        };
        certify_side(plus);
        certify_side(minus);
        if (exterior)
        {
            if (!plus.empty() && !minus.empty())
                throw std::invalid_argument("Physical boundary ancestry has both cell-side orientations.");
            for (const auto& e : entries)
                tiles.push_back({e.use, e.cell, absent_cell, e.polygon});
        }
        else
        {
            if (plus.empty() || minus.empty())
                throw std::invalid_argument("Interior ancestral surface has a missing incident side.");
            std::vector<std::vector<Polygon>> coverage(entries.size());
            for (auto p : plus)
                for (auto n : minus)
                {
                    auto polygon = polygon_intersection(entries[p].polygon, entries[n].polygon, normal);
                    if (polygon.empty())
                        continue;
                    if (entries[p].cell == entries[n].cell)
                        throw std::invalid_argument("A convex cell cannot be its own interface neighbor.");
                    coverage[p].push_back(polygon);
                    coverage[n].push_back(polygon);
                    const auto owner = std::min(entries[p].cell, entries[n].cell),
                               neighbor = std::max(entries[p].cell, entries[n].cell);
                    auto use = entries[p].cell == owner ? entries[p].use : entries[n].use;
                    tiles.push_back({std::move(use), owner, neighbor, std::move(polygon)});
                }
            std::size_t axis = 0;
            while (normal[axis] == 0)
                ++axis;
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                const auto origin = entries[i].polygon[0];
                const auto expected = polygon_moments(entries[i].polygon, axis, origin);
                FaceMoment sum{};
                for (const auto& piece : coverage[i])
                {
                    auto m = polygon_moments(piece, axis, origin);
                    sum.area += m.area;
                    sum.first = add(sum.first, m.first);
                }
                if (sum.area != expected.area || sum.first != expected.first)
                    throw std::invalid_argument("Exact shared-face overlay has a real represented gap.");
            }
        }
    }
    std::sort(tiles.begin(), tiles.end(),
        [](const auto& a, const auto& b)
        {
            if (a.use.key != b.use.key)
                return a.use.key < b.use.key;
            return a.canonical < b.canonical;
        });
    for (const auto& tile : tiles)
        for (const auto& p : tile.canonical)
            candidate_vertices[tile.use.key.region].insert(p);
    std::vector<Polygon> polygons;
    for (const auto& tile : tiles)
    {
        Polygon polygon;
        for (std::size_t e = 0; e < tile.canonical.size(); ++e)
        {
            const auto& a = tile.canonical[e];
            const auto& b = tile.canonical[(e + 1) % tile.canonical.size()];
            const auto d = sub(b, a);
            const Q length = dot(d, d);
            std::vector<std::pair<Q, P>> interior;
            for (const auto& p : candidate_vertices[tile.use.key.region])
            {
                const auto offset = sub(p, a);
                const Q along = dot(offset, d);
                if (zero(cross(offset, d)) && along > 0 && along < length)
                    interior.emplace_back(along, p);
            }
            std::sort(interior.begin(), interior.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            polygon.push_back(a);
            for (const auto& [along, p] : interior)
                polygon.push_back(p);
        }
        if (tile.use.orientation < 0)
            std::reverse(polygon.begin(), polygon.end());
        polygons.push_back(std::move(polygon));
    }
    std::map<std::pair<std::uint64_t, P>, ComplexID> node_ids;
    for (std::size_t f = 0; f < tiles.size(); ++f)
        for (const auto& p : polygons[f])
            node_ids.emplace(std::pair{tiles[f].use.key.region, p}, 0);
    ComplexID node_id = 0;
    for (auto& [key, id] : node_ids)
    {
        id = node_id;
        impl->nodes.emplace(node_id++, ComplexNodeRow{key.first, key.second});
    }
    impl->global_nodes = node_id;
    for (std::size_t f = 0; f < tiles.size(); ++f)
    {
        const auto& tile = tiles[f];
        ComplexFaceRow face;
        face.surface = tile.use;
        face.owner = tile.owner;
        face.neighbor = tile.neighbor;
        for (const auto& p : polygons[f])
            face.nodes.push_back(node_ids.at({tile.use.key.region, p}));
        impl->faces.emplace(f, std::move(face));
        impl->cells.at(tile.owner).faces.push_back(f);
        if (tile.neighbor != absent_cell)
            impl->cells.at(tile.neighbor).faces.push_back(f);
    }
    impl->global_faces = tiles.size();
    for (auto& [id, cell] : impl->cells)
    {
        std::set<ComplexID> nodes;
        for (auto face : cell.faces)
            for (auto node : impl->faces.at(face).nodes)
                nodes.insert(node);
        cell.nodes.assign(nodes.begin(), nodes.end());
    }
    complex_validate(*impl);
    for (const auto& [id, cell] : impl->cells)
    {
        Poly reconstructed;
        std::map<ComplexID, std::size_t> vertices;
        for (auto face_id : cell.faces)
        {
            const auto& face = impl->faces.at(face_id);
            Face loop;
            for (auto n : face.nodes)
            {
                auto [it, inserted] = vertices.emplace(n, reconstructed.vertices.size());
                if (inserted)
                    reconstructed.vertices.push_back(impl->nodes.at(n).point);
                loop.push_back(it->second);
            }
            if (face.owner != id)
                std::reverse(loop.begin(), loop.end());
            reconstructed.faces.push_back(std::move(loop));
        }
        if (!contained(reconstructed, cell.geometry->impl_->poly) ||
            !contained(cell.geometry->impl_->poly, reconstructed))
            throw std::invalid_argument("Atomic face closure changed the canonical convex cell shell.");
    }
    return std::shared_ptr<const ExactPolyhedralComplex>(new ExactPolyhedralComplex(std::move(impl)));
}
std::uint64_t ExactPolyhedralComplex::global_cells() const noexcept
{
    return impl_->global_cells;
}
std::uint64_t ExactPolyhedralComplex::global_faces() const noexcept
{
    return impl_->global_faces;
}
std::uint64_t ExactPolyhedralComplex::global_nodes() const noexcept
{
    return impl_->global_nodes;
}
const std::vector<ComplexID>& ExactPolyhedralComplex::cell_ids() const noexcept
{
    return impl_->cell_ids;
}
const std::vector<ComplexID>& ExactPolyhedralComplex::resident_cell_ids() const noexcept
{
    return impl_->resident_ids;
}
const std::vector<ComplexID>& ExactPolyhedralComplex::face_ids() const noexcept
{
    return impl_->face_ids;
}
const std::vector<ComplexID>& ExactPolyhedralComplex::node_ids() const noexcept
{
    return impl_->node_ids;
}
bool ExactPolyhedralComplex::has_full_cell(ComplexID id) const noexcept
{
    return impl_->cells.contains(id) && bool(impl_->cells.at(id).geometry);
}
const std::vector<ComplexID>& ExactPolyhedralComplex::cell_faces(ComplexID id) const
{
    return impl_->cells.at(id).faces;
}
const std::vector<ComplexID>& ExactPolyhedralComplex::cell_nodes(ComplexID id) const
{
    return impl_->cells.at(id).nodes;
}
const std::vector<ComplexID>& ExactPolyhedralComplex::face_nodes(ComplexID id) const
{
    return impl_->faces.at(id).nodes;
}
ComplexID ExactPolyhedralComplex::owner_cell(ComplexID id) const
{
    return impl_->faces.at(id).owner;
}
ComplexID ExactPolyhedralComplex::neighbor_cell(ComplexID id) const
{
    return impl_->faces.at(id).neighbor;
}
const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& ExactPolyhedralComplex::cell_key(ComplexID id) const
{
    return impl_->cells.at(id).key;
}
const PolyhedralSurfaceUse& ExactPolyhedralComplex::face_surface(ComplexID id) const
{
    return impl_->faces.at(id).surface;
}
MeshUtils::Vec3 ExactPolyhedralComplex::node_coordinates(ComplexID id) const
{
    return rounded_point(impl_->nodes.at(id).point);
}
double ExactPolyhedralComplex::cell_volume(ComplexID id) const
{
    return impl_->cells.at(id).volume;
}
MeshUtils::Vec3 ExactPolyhedralComplex::cell_centroid(ComplexID id) const
{
    return impl_->cells.at(id).center;
}
double ExactPolyhedralComplex::face_area(ComplexID id) const
{
    return impl_->faces.at(id).area;
}
MeshUtils::Vec3 ExactPolyhedralComplex::face_centroid(ComplexID id) const
{
    return impl_->faces.at(id).center;
}
MeshUtils::Vec3 ExactPolyhedralComplex::face_area_vector(ComplexID id) const
{
    return impl_->faces.at(id).vector;
}
ConvexGeometryPtr ExactPolyhedralComplex::cell_geometry(ComplexID id) const
{
    const auto& geometry = impl_->cells.at(id).geometry;
    if (!geometry)
        throw std::invalid_argument("Metric-only exact complex neighbor has no resident cell certificate.");
    return geometry;
}
std::vector<char> ExactPolyhedralComplex::face_identity(ComplexID id) const
{
    const auto& face = impl_->faces.at(id);
    WireWriter w;
    auto surface = face.surface;
    surface.orientation = 1;
    surface_write(w, surface);
    Polygon polygon;
    for (auto node : face.nodes)
        polygon.push_back(impl_->nodes.at(node).point);
    if (face.surface.orientation < 0)
        std::reverse(polygon.begin(), polygon.end());
    polygon = minimal_polygon(std::move(polygon));
    w.word(polygon.size());
    for (const auto& p : polygon)
        for (const auto& q : p)
            w.rational(q);
    return std::move(w.data);
}
std::vector<char> ExactPolyhedralComplex::encode() const
{
    return encode(impl_->resident_ids);
}
std::vector<char> ExactPolyhedralComplex::encode(std::span<const ComplexID> visible) const
{
    std::set<ComplexID> full, cells, faces, nodes;
    for (auto id : visible)
    {
        if (!has_full_cell(id) || !full.insert(id).second)
            throw std::invalid_argument("Exact complex sparse selection is missing/duplicate/nonresident.");
        cells.insert(id);
        for (auto f : impl_->cells.at(id).faces)
            faces.insert(f);
    }
    for (auto f : faces)
    {
        const auto& face = impl_->faces.at(f);
        cells.insert(face.owner);
        if (face.neighbor != absent_cell)
            cells.insert(face.neighbor);
        for (auto n : face.nodes)
            nodes.insert(n);
    }
    WireWriter w;
    w.word(0x5346504f4c594331ULL);
    w.word(1);
    w.word(impl_->global_cells);
    w.word(impl_->global_faces);
    w.word(impl_->global_nodes);
    w.word(cells.size());
    for (auto id : cells)
    {
        const auto& cell = impl_->cells.at(id);
        w.word(id);
        cell_key_write(w, cell.key);
        w.real(cell.volume);
        w.real(cell.center.x);
        w.real(cell.center.y);
        w.real(cell.center.z);
        w.word(full.contains(id));
        if (full.contains(id))
        {
            const auto certificate = cell.geometry->encode_exact_certificate();
            w.word(certificate.size());
            w.data.insert(w.data.end(), certificate.begin(), certificate.end());
            w.word(cell.faces.size());
            for (auto f : cell.faces)
                w.word(f);
            w.word(cell.nodes.size());
            for (auto n : cell.nodes)
                w.word(n);
        }
    }
    w.word(faces.size());
    for (auto id : faces)
    {
        const auto& face = impl_->faces.at(id);
        w.word(id);
        surface_write(w, face.surface);
        w.word(face.owner);
        w.word(face.neighbor);
        w.word(face.nodes.size());
        for (auto n : face.nodes)
            w.word(n);
    }
    w.word(nodes.size());
    for (auto id : nodes)
    {
        const auto& node = impl_->nodes.at(id);
        w.word(id);
        w.word(node.region);
        for (const auto& q : node.point)
            w.rational(q);
    }
    return std::move(w.data);
}
std::shared_ptr<const ExactPolyhedralComplex> ExactPolyhedralComplex::decode(std::span<const char> packet)
{
    WireReader r{packet};
    if (r.word() != 0x5346504f4c594331ULL || r.word() != 1)
        throw std::invalid_argument("Unknown exact complex packet version/tag.");
    auto impl = std::make_shared<Impl>();
    impl->global_cells = r.word();
    impl->global_faces = r.word();
    impl->global_nodes = r.word();
    if (!impl->global_cells || !impl->global_faces || !impl->global_nodes)
        throw std::invalid_argument("Exact complex global catalog is empty.");
    auto count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        auto id = r.word();
        ComplexCellRow cell;
        cell.key = cell_key_read(r);
        cell.volume = r.real();
        cell.center = {r.real(), r.real(), r.real()};
        auto full = r.word();
        if (full > 1)
            throw std::invalid_argument("Invalid exact complex residency flag.");
        if (full)
        {
            auto n = r.word();
            if (n > r.data.size() - r.position)
                throw std::invalid_argument("Invalid exact cell certificate length.");
            cell.geometry = ConvexCellGeometry::decode_exact_certificate(r.data.subspan(r.position, n));
            r.position += n;
            n = r.count();
            for (std::size_t j = 0; j < n; ++j)
                cell.faces.push_back(r.word());
            n = r.count();
            for (std::size_t j = 0; j < n; ++j)
                cell.nodes.push_back(r.word());
            const auto m = exact_moments(cell.geometry->impl_->poly, point(cell.geometry->impl_->origin));
            const auto center = rounded_point(add(point(cell.geometry->impl_->origin), mul(m.first, Q(1 / m.volume))));
            if (cell.volume != measured(m.volume).as_double() || cell.center.x != center.x ||
                cell.center.y != center.y || cell.center.z != center.z)
                throw std::invalid_argument("Exact cell packet metric/certificate mismatch.");
        }
        if (!impl->cells.emplace(id, std::move(cell)).second)
            throw std::invalid_argument("Repeated exact complex cell ID.");
    }
    count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        auto id = r.word();
        ComplexFaceRow face;
        face.surface = surface_read(r);
        face.owner = r.word();
        face.neighbor = r.word();
        auto n = r.count();
        for (std::size_t j = 0; j < n; ++j)
            face.nodes.push_back(r.word());
        if (!impl->faces.emplace(id, std::move(face)).second)
            throw std::invalid_argument("Repeated exact complex face ID.");
    }
    count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        auto id = r.word();
        ComplexNodeRow node;
        node.region = r.word();
        for (auto& q : node.point)
            q = r.rational();
        if (!impl->nodes.emplace(id, std::move(node)).second)
            throw std::invalid_argument("Repeated exact complex node ID.");
    }
    r.finish();
    complex_validate(*impl);
    for (const auto& [id, cell] : impl->cells)
        if (cell.geometry)
        {
            Poly poly;
            std::map<ComplexID, std::size_t> ids;
            for (auto f : cell.faces)
            {
                const auto& face = impl->faces.at(f);
                Face loop;
                for (auto n : face.nodes)
                {
                    const auto [it, inserted] = ids.emplace(n, poly.vertices.size());
                    if (inserted)
                        poly.vertices.push_back(impl->nodes.at(n).point);
                    loop.push_back(it->second);
                }
                if (face.owner != id)
                    std::reverse(loop.begin(), loop.end());
                poly.faces.push_back(std::move(loop));
            }
            if (!contained(poly, cell.geometry->impl_->poly) || !contained(cell.geometry->impl_->poly, poly))
                throw std::invalid_argument("Exact sparse atomic topology changed its canonical cell geometry.");
        }
    return std::shared_ptr<const ExactPolyhedralComplex>(new ExactPolyhedralComplex(std::move(impl)));
}
std::size_t ExactPolyhedralComplex::storage_bytes() const
{
    std::size_t bytes = sizeof(ExactPolyhedralComplex) + sizeof(Impl);
    std::set<const ConvexCellGeometry*> certificates;
    for (const auto& [id, cell] : impl_->cells)
    {
        bytes += sizeof(decltype(impl_->cells)::value_type) + 4 * sizeof(void*) +
                 (cell.faces.capacity() + cell.nodes.capacity()) * sizeof(ComplexID);
        if (cell.geometry && certificates.insert(cell.geometry.get()).second)
            bytes += cell.geometry->storage_bytes();
        bytes += std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(cell.key.ancestry).path.capacity() *
                 sizeof(ComplexID);
    }
    for (const auto& [id, face] : impl_->faces)
        bytes += sizeof(decltype(impl_->faces)::value_type) + 4 * sizeof(void*) +
                 face.nodes.capacity() * sizeof(ComplexID) + face.surface.boundary.capacity() +
                 face.surface.key.creator.path.capacity() * sizeof(ComplexID);
    for (const auto& [id, node] : impl_->nodes)
    {
        bytes += sizeof(decltype(impl_->nodes)::value_type) + 4 * sizeof(void*);
        for (const auto& q : node.point)
            bytes += (q.backend().num().capacity() + q.backend().denom().capacity()) *
                     sizeof(boost::multiprecision::limb_type);
    }
    bytes += (impl_->cell_ids.capacity() + impl_->resident_ids.capacity() + impl_->face_ids.capacity() +
                 impl_->node_ids.capacity()) *
             sizeof(ComplexID);
    return bytes;
}

std::size_t ConvexCellGeometry::storage_bytes() const
{
    std::size_t bytes = sizeof(ConvexCellGeometry) + sizeof(Impl) + impl_->poly.vertices.capacity() * sizeof(P) +
                        impl_->poly.faces.capacity() * sizeof(Face);
    auto add_q = [&](const Q& q)
    {
        bytes +=
            (q.backend().num().capacity() + q.backend().denom().capacity()) * sizeof(boost::multiprecision::limb_type);
    };
    for (const auto& p : impl_->poly.vertices)
        for (const auto& q : p)
            add_q(q);
    for (const auto& face : impl_->poly.faces)
        bytes += face.capacity() * sizeof(std::size_t);
    if (impl_->product)
    {
        bytes += impl_->product->base.capacity() * sizeof(P);
        for (const auto& p : impl_->product->base)
            for (const auto& q : p)
                add_q(q);
        add_q(impl_->product->lower);
        add_q(impl_->product->upper);
    }
    return bytes;
}
ConvexGeometryPtr ConvexCellGeometry::from_native_cell_atomic(const MultiRegionMesh& mesh, std::uint64_t cell)
{
    (void) mesh.geometry_epoch();
    if (mesh.has_exact_construction() && !mesh.interfaces().empty() && !mesh.has_qualified_exact_interfaces())
        throw std::invalid_argument("Joined constructed atomic geometry needs qualified exact interfaces.");
    if (!mesh.interfaces().empty() && mesh.has_qualified_exact_interfaces())
    {
        for (const auto& interface : mesh.interfaces())
            std::visit(
                [](const auto& join)
                {
                    if constexpr (requires { join.periodic_translation; })
                        if (join.periodic_translation)
                            throw std::invalid_argument("Periodic exact atomic cell images remain unsupported.");
                },
                interface);
        std::vector<ExactPlanarFacePtr> faces;
        for (auto face : mesh.cell_faces(cell))
        {
            auto certificate = ExactPlanarFace::from_native_face(mesh, face);
            if (mesh.owner_cell(face) != cell)
                certificate = certificate->reversed();
            faces.push_back(std::move(certificate));
        }
        return ExactPlanarFace::cell_shell(faces);
    }
    const auto [region, local] = mesh.native_cell(cell);
    auto geometry = std::visit(
        [&](const auto& r) -> ConvexGeometryPtr
        {
            if constexpr (requires { r.geometry().atomic_cell_geometry(local); })
                return r.geometry().atomic_cell_geometry(local);
            else
                return {};
        },
        mesh.regions().at(region));
    if (geometry)
        return geometry;
    if (mesh.has_exact_construction())
        throw std::invalid_argument("Constructed provider lacks an authoritative ordered atomic shell.");
    return from_native_cell(mesh, cell);
}
ConvexGeometryPtr ExactExtrudedConstruction::atomic_cell_geometry(std::size_t cell) const
{
    const auto id = impl_->topology->indexer().cell_id(cell);
    P center = impl_->base->centers.at(id.ij);
    center[2] = (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)] +
                    impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) /
                2;
    auto geometry = std::make_shared<ConvexCellGeometry::Impl>();
    std::map<std::uint64_t, std::size_t> ids;
    for (auto face : impl_->topology->cell_faces(cell))
    {
        Face loop;
        for (auto node : impl_->topology->face_nodes(face))
        {
            auto [it, inserted] = ids.emplace(node, geometry->poly.vertices.size());
            if (inserted)
            {
                const auto n = impl_->topology->indexer().node_id(node);
                auto p = impl_->base->points[impl_->base->active.at(n.ij)];
                p[2] = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + n.k)];
                geometry->poly.vertices.push_back(std::move(p));
            }
            loop.push_back(it->second);
        }
        const auto area = area_vector(geometry->poly, loop);
        const auto direction = dot(area, sub(geometry->poly.vertices[loop[0]], center));
        if (direction == 0)
            throw std::invalid_argument("Exact product face has unresolved interior visibility.");
        if (direction < 0)
            std::reverse(loop.begin(), loop.end());
        geometry->poly.faces.push_back(std::move(loop));
    }
    validate(geometry->poly);
    const auto origin = rounded_point(geometry->poly.vertices[0]);
    geometry->origin = {origin.x, origin.y, origin.z};
    geometry->product = product_description(geometry->poly);
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(geometry)));
}
ConvexGeometryPtr ExactPolyhedralComplex::atomic_cell_geometry(std::uint64_t cell_id) const
{
    const auto& cell = impl_->cells.at(cell_id);
    if (!cell.geometry)
        throw std::invalid_argument("Metric-only neighbor has no exact atomic shell.");
    auto geometry = std::make_shared<ConvexCellGeometry::Impl>();
    std::map<std::uint64_t, std::size_t> ids;
    for (auto face_id : cell.faces)
    {
        const auto& face = impl_->faces.at(face_id);
        Face loop;
        for (auto node : face.nodes)
        {
            auto [it, inserted] = ids.emplace(node, geometry->poly.vertices.size());
            if (inserted)
                geometry->poly.vertices.push_back(impl_->nodes.at(node).point);
            loop.push_back(it->second);
        }
        if (face.owner != cell_id)
            std::reverse(loop.begin(), loop.end());
        geometry->poly.faces.push_back(std::move(loop));
    }
    validate(geometry->poly);
    geometry->origin = cell.geometry->impl_->origin;
    geometry->product = product_description(geometry->poly);
    return ConvexGeometryPtr(new ConvexCellGeometry(std::move(geometry)));
}

std::vector<char> ExtrudedTopology::encode_base() const
{
    WireWriter w;
    w.word(0x5346544f50424131ULL);
    w.word(1);
    w.word(indexer().num_nodes_per_layer);
    w.word(indexer().num_cells_per_layer);
    for (std::size_t c = 0; c < indexer().num_cells_per_layer; ++c)
    {
        auto nodes = base_cell_nodes(c);
        w.word(nodes.size());
        for (auto n : nodes)
            w.word(n);
    }
    std::vector<SemiStructMeshTopo::BoundaryEdge> boundaries;
    for (const auto& e : d_topology.side_faces())
        if (e.neighbor == SemiStructMeshTopo::invalid_ordinal)
            boundaries.push_back({e.nodes[0], e.nodes[1], d_topology.boundary_batch_name(e.boundary_id)});
    w.word(boundaries.size());
    for (const auto& b : boundaries)
    {
        w.word(b.node0);
        w.word(b.node1);
        w.word(b.batch_name.size());
        w.data.insert(w.data.end(), b.batch_name.begin(), b.batch_name.end());
    }
    return std::move(w.data);
}
std::shared_ptr<const ExtrudedTopology> ExtrudedTopology::decode_base(std::span<const char> packet, unsigned layers)
{
    WireReader r{packet};
    if (r.word() != 0x5346544f50424131ULL || r.word() != 1)
        throw std::invalid_argument("Unknown shared base topology packet.");
    auto nodes = r.word(), cells = r.count();
    if (nodes > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Shared base node extent overflow.");
    Arr<Arr<unsigned>> loops;
    for (std::size_t c = 0; c < cells; ++c)
    {
        Arr<unsigned> loop;
        auto n = r.count();
        for (std::size_t i = 0; i < n; ++i)
        {
            auto id = r.word();
            if (id >= nodes)
                throw std::invalid_argument("Invalid shared base node ordinal.");
            loop.push_back(id);
        }
        loops.push_back(std::move(loop));
    }
    Arr<SemiStructMeshTopo::BoundaryEdge> boundaries;
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        auto a = r.word(), b = r.word(), length = r.word();
        if (a >= nodes || b >= nodes || length > r.data.size() - r.position)
            throw std::invalid_argument("Invalid shared base boundary record.");
        std::string name(r.data.begin() + r.position, r.data.begin() + r.position + length);
        r.position += length;
        boundaries.push_back({static_cast<unsigned>(a), static_cast<unsigned>(b), std::move(name)});
    }
    r.finish();
    return std::make_shared<const ExtrudedTopology>(nodes, loops, layers, boundaries);
}
namespace
{
void write_point_recipes(WireWriter& w, const ExactExtrudedBaseData& base)
{
    w.word(base.recipes.size());
    for (const auto& p : base.recipes)
    {
        for (auto x : p.root)
            w.real(x);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(base.active.size());
    for (auto id : base.active)
        w.word(id);
}
void read_point_recipes(WireReader& r, std::vector<ExactPointRecipe>& points, std::vector<std::size_t>& active)
{
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ExactPointRecipe p;
        for (auto& x : p.root)
            x = r.real();
        auto m = r.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(r.word());
        points.push_back(std::move(p));
    }
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        active.push_back(r.word());
}
void read_axis_parts(
    std::span<const char> packet, std::vector<ExactScalarRecipe>& roots, std::vector<std::size_t>& active)
{
    WireReader r{packet};
    if (r.word() != 0x5346415849534331ULL || r.word() != 1)
        throw std::invalid_argument("Unknown shared axis packet.");
    auto values = r.word();
    if (values > r.data.size() - r.position)
        throw std::invalid_argument("Invalid axis values length.");
    WireReader v{r.data.subspan(r.position, values)};
    r.position += values;
    if (v.word() != 0x5346415856414c31ULL || v.word() != 1)
        throw std::invalid_argument("Unknown axis value DAG packet.");
    auto n = v.count();
    for (std::size_t i = 0; i < n; ++i)
    {
        ExactScalarRecipe p;
        p.root = v.real();
        auto m = v.count();
        for (std::size_t j = 0; j < m; ++j)
            p.parents.push_back(v.word());
        roots.push_back(std::move(p));
    }
    v.finish();
    n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        active.push_back(r.word());
    r.finish();
}
} // namespace
std::vector<char> ExactExtrudedConstruction::encode_base() const
{
    WireWriter w;
    w.word(0x534647454f424131ULL);
    w.word(1);
    w.word(impl_->base->topology.size());
    w.data.insert(w.data.end(), impl_->base->topology.begin(), impl_->base->topology.end());
    write_point_recipes(w, *impl_->base);
    return std::move(w.data);
}
std::vector<char> ExactExtrudedConstruction::encode_axis_values() const
{
    WireWriter w;
    w.word(0x5346415856414c31ULL);
    w.word(1);
    w.word(impl_->axis->values->zrecipes.size());
    for (const auto& p : impl_->axis->values->zrecipes)
    {
        w.real(p.root);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    return std::move(w.data);
}
std::vector<char> ExactExtrudedConstruction::encode_axis_selection() const
{
    WireWriter w;
    w.word(impl_->axis->zactive.size());
    for (auto id : impl_->axis->zactive)
        w.word(id);
    return std::move(w.data);
}
std::vector<char> ExactExtrudedConstruction::combine_axis_parts(
    std::span<const char> values, std::span<const char> selection)
{
    WireWriter w;
    w.word(0x5346415849534331ULL);
    w.word(1);
    w.word(values.size());
    w.data.insert(w.data.end(), values.begin(), values.end());
    w.data.insert(w.data.end(), selection.begin(), selection.end());
    return std::move(w.data);
}
std::vector<char> ExactExtrudedConstruction::encode_axis() const
{
    return combine_axis_parts(encode_axis_values(), encode_axis_selection());
}
std::vector<char> ExactExtrudedConstruction::encode_view() const
{
    WireWriter w;
    w.word(0x5346505256494531ULL);
    w.word(1);
    w.word(impl_->begin);
    w.word(impl_->layers);
    return std::move(w.data);
}
std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::shared_view(
    std::shared_ptr<const ExtrudedTopology> topology, std::shared_ptr<const ExactExtrudedConstruction> base,
    std::shared_ptr<const ExactExtrudedConstruction> axis, std::size_t begin, std::size_t layers)
{
    if (!topology || !base || !axis || !layers || topology->indexer().num_layers != layers ||
        begin >= axis->impl_->axis->zactive.size() || layers >= axis->impl_->axis->zactive.size() - begin ||
        topology->encode_base() != base->impl_->base->topology)
        throw std::invalid_argument("Shared product view has foreign base topology or invalid axial extent.");
    auto impl = std::make_shared<Impl>();
    impl->topology = std::move(topology);
    impl->base = base->impl_->base;
    impl->axis = axis->impl_->axis;
    impl->begin = begin;
    impl->layers = layers;
    auto result = std::shared_ptr<const ExactExtrudedConstruction>(new ExactExtrudedConstruction(std::move(impl)));
    // Base visibility and axis representability were certified independently;
    // paired native volume/face outputs still require representable products.
    Q minimum = 0, maximum = 0;
    for (std::size_t k = 0; k < layers; ++k)
    {
        const Q width = result->impl_->axis->values->z[result->impl_->axis->zactive[begin + k + 1]] -
                        result->impl_->axis->values->z[result->impl_->axis->zactive[begin + k]];
        if (k == 0 || width < minimum)
            minimum = width;
        if (width > maximum)
            maximum = width;
    }
    for (std::size_t c = 0; c < result->impl_->base->areas.size(); ++c)
    {
        (void) measured(Q(result->impl_->base->areas[c] * minimum)).as_double();
        (void) measured(Q(result->impl_->base->areas[c] * maximum)).as_double();
    }
    for (const auto& edge : result->impl_->base->deltas)
    {
        const Q length = dot(edge, edge);
        (void) square_root(Q(length * minimum * minimum)).as_double();
        (void) square_root(Q(length * maximum * maximum)).as_double();
    }
    return result;
}
std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::shared_axis_selection(
    std::shared_ptr<const ExtrudedTopology> topology, std::shared_ptr<const ExactExtrudedConstruction> base,
    std::shared_ptr<const ExactExtrudedConstruction> coordinate_anchor, std::vector<std::size_t> active)
{
    if (!topology || !base || !coordinate_anchor || active.size() != std::size_t(topology->indexer().num_layers) + 1 ||
        topology->encode_base() != base->impl_->base->topology)
        throw std::invalid_argument("Shared axis selection belongs to a different base or layer layout.");
    auto impl = std::make_shared<Impl>();
    impl->topology = std::move(topology);
    impl->base = base->impl_->base;
    impl->axis = std::make_shared<ExactExtrudedAxisData>();
    impl->axis->values = coordinate_anchor->impl_->axis->values;
    impl->axis->zactive = std::move(active);
    impl->layers = impl->axis->zactive.size() - 1;
    for (std::size_t k = 0; k < impl->axis->zactive.size(); ++k)
    {
        const auto id = impl->axis->zactive[k];
        if (id >= impl->axis->values->z.size())
            throw std::invalid_argument("Shared axis selection has an unknown exact point.");
        const auto value = measured(impl->axis->values->z[id]).as_double();
        if (k)
        {
            const auto previous = impl->axis->zactive[k - 1];
            const auto lo = impl->axis->display.back();
            const auto center =
                measured(Q((impl->axis->values->z[id] + impl->axis->values->z[previous]) / 2)).as_double();
            if (!(value > lo && center > lo && center < value))
                throw std::invalid_argument("Shared axis selection has unrepresentable native widths/centers.");
        }
        impl->axis->display.push_back(value);
    }
    auto selection = std::shared_ptr<const ExactExtrudedConstruction>(new ExactExtrudedConstruction(std::move(impl)));
    return shared_view(selection->impl_->topology, selection, selection, 0, selection->impl_->layers);
}
std::shared_ptr<const ExactExtrudedConstruction> ExactExtrudedConstruction::decode_shared(
    std::shared_ptr<const ExtrudedTopology> topology, std::span<const char> base_packet,
    std::span<const char> axis_packet, std::span<const char> view_packet,
    std::shared_ptr<const ExactExtrudedConstruction> base_anchor,
    std::shared_ptr<const ExactExtrudedConstruction> axis_anchor)
{
    if (!topology)
        throw std::invalid_argument("Shared product decoder requires retained topology.");
    WireReader view{view_packet};
    if (view.word() != 0x5346505256494531ULL || view.word() != 1)
        throw std::invalid_argument("Unknown product run view packet.");
    const auto begin = view.word(), layers = view.word();
    view.finish();
    if (base_anchor && base_anchor->encode_base() != std::vector<char>(base_packet.begin(), base_packet.end()))
        throw std::invalid_argument("Shared product base reference does not match exact descriptor.");
    std::vector<ExactScalarRecipe> roots;
    std::vector<std::size_t> active;
    read_axis_parts(axis_packet, roots, active);
    if (active.size() < 2 || active.size() > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Shared axis selection extent is invalid.");
    WireReader base{base_packet};
    if (base.word() != 0x534647454f424131ULL || base.word() != 1)
        throw std::invalid_argument("Unknown exact product base packet.");
    auto bytes = base.word();
    if (bytes > base.data.size() - base.position)
        throw std::invalid_argument("Invalid shared base topology binding.");
    auto expected = topology->encode_base();
    if (bytes != expected.size() || !std::equal(expected.begin(), expected.end(), base.data.begin() + base.position))
        throw std::invalid_argument("Exact shared product base topology mismatch.");
    base.position += bytes;
    std::vector<ExactPointRecipe> points;
    std::vector<std::size_t> point_ids;
    read_point_recipes(base, points, point_ids);
    base.finish();
    std::shared_ptr<const ExactExtrudedConstruction> source;
    if (!base_anchor || !axis_anchor)
    {
        auto full_topology = topology->with_layers(active.size() - 1);
        source = make(full_topology, std::move(points), std::move(point_ids), roots, active);
    }
    if (!base_anchor)
        base_anchor = source;
    if (axis_anchor)
    {
        const auto expected_values = [&]
        {
            WireWriter w;
            w.word(0x5346415856414c31ULL);
            w.word(1);
            w.word(roots.size());
            for (const auto& p : roots)
            {
                w.real(p.root);
                w.word(p.parents.size());
                for (auto id : p.parents)
                    w.word(id);
            }
            return w.data;
        }();
        if (axis_anchor->encode_axis_values() != expected_values)
            throw std::invalid_argument("Shared product axis reference has different exact coordinate constructions.");
        if (axis_anchor->encode_axis_selection() !=
            [&]
            {
                WireWriter w;
                w.word(active.size());
                for (auto id : active)
                    w.word(id);
                return w.data;
            }())
        {
            // Form the layout before moving its selection. Function argument
            // evaluation must not observe active after another argument moved it.
            auto selection_topology = topology->with_layers(active.size() - 1);
            axis_anchor =
                shared_axis_selection(std::move(selection_topology), base_anchor, axis_anchor, std::move(active));
        }
    }
    else
        axis_anchor = source;
    return shared_view(std::move(topology), std::move(base_anchor), std::move(axis_anchor), begin, layers);
}
const void* ExactExtrudedConstruction::base_storage_identity() const noexcept
{
    return impl_->base.get();
}
const void* ExactExtrudedConstruction::axis_storage_identity() const noexcept
{
    return impl_->axis.get();
}
const void* ExactExtrudedConstruction::axis_values_storage_identity() const noexcept
{
    return impl_->axis->values.get();
}
std::size_t ExactExtrudedConstruction::view_storage_bytes() const noexcept
{
    return sizeof(ExactExtrudedConstruction) + sizeof(Impl);
}
std::size_t ExactExtrudedConstruction::base_storage_bytes() const
{
    const auto& b = *impl_->base;
    std::size_t bytes = sizeof(b) + b.recipes.capacity() * sizeof(ExactPointRecipe) +
                        b.active.capacity() * sizeof(std::size_t) + b.display.capacity() * sizeof(MeshUtils::Vec3) +
                        b.topology.capacity();
    for (const auto& r : b.recipes)
        bytes += r.parents.capacity() * sizeof(std::size_t);
    auto scalar = [](const Q& q)
    {
        return (q.backend().num().capacity() + q.backend().denom().capacity()) *
               sizeof(boost::multiprecision::limb_type);
    };
    for (const auto* values : {&b.points, &b.centers, &b.deltas, &b.edge_centers})
    {
        bytes += values->capacity() * sizeof(P);
        for (const auto& p : *values)
            for (const auto& q : p)
                bytes += scalar(q);
    }
    bytes += b.areas.capacity() * sizeof(Q);
    for (const auto& q : b.areas)
        bytes += scalar(q);
    return bytes;
}
std::size_t ExactExtrudedConstruction::axis_values_storage_bytes() const
{
    const auto& a = *impl_->axis->values;
    std::size_t bytes = sizeof(a) + a.zrecipes.capacity() * sizeof(ExactScalarRecipe) + a.z.capacity() * sizeof(Q);
    for (const auto& r : a.zrecipes)
        bytes += r.parents.capacity() * sizeof(std::size_t);
    for (const auto& q : a.z)
        bytes +=
            (q.backend().num().capacity() + q.backend().denom().capacity()) * sizeof(boost::multiprecision::limb_type);
    return bytes;
}
std::size_t ExactExtrudedConstruction::axis_storage_bytes() const
{
    return sizeof(ExactExtrudedAxisData) + impl_->axis->zactive.capacity() * sizeof(std::size_t) +
           impl_->axis->display.capacity() * sizeof(double) + axis_values_storage_bytes();
}

struct ExactPlanarFace::Impl
{
    Polygon polygon;
};
namespace
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
} // namespace
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
    const auto n = polygon_vector(impl_->polygon);
    return square_root(dot(n, n));
}
MeshUtils::Vec3 ExactPlanarFace::centroid() const
{
    return rounded_point(polygon_center(impl_->polygon));
}
MeshUtils::Vec3 ExactPlanarFace::area_vector() const
{
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
