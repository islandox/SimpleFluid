/** @file ConvexCellGeometry.cc
 * @brief Exact represented-input convex clipping and moment certificates.
 */
#include "geometry/ConvexCellGeometry.hh"
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
    if (mesh.has_exact_construction() && !mesh.interfaces().empty())
        throw std::invalid_argument(
            "Joined constructed providers require separately qualified exact interface geometry.");
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
struct ExactExtrudedConstruction::Impl
{
    std::shared_ptr<const ExtrudedTopology> topology;
    std::vector<ExactPointRecipe> recipes;
    std::vector<ExactScalarRecipe> zrecipes;
    std::vector<std::size_t> active, zactive;
    std::vector<P> points, centers, deltas, edge_centers;
    std::vector<Q> z, areas;
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
    impl->recipes = std::move(recipes);
    impl->zrecipes = std::move(zrecipes);
    impl->active = std::move(active);
    impl->zactive = std::move(zactive);
    for (std::size_t i = 0; i < impl->recipes.size(); ++i)
    {
        const auto& recipe = impl->recipes[i];
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
                p = add(p, impl->points[parent]);
            }
            p = mul(p, Q(1, recipe.parents.size()));
        }
        if (p[2] != 0)
            throw std::invalid_argument("Exact extrusion base must lie in XY.");
        impl->points.push_back(std::move(p));
    }
    std::set<std::size_t> unique;
    std::set<P> unique_points;
    std::set<GeometryPoint> display;
    for (auto id : impl->active)
    {
        if (id >= impl->points.size() || !unique.insert(id).second || !unique_points.insert(impl->points[id]).second)
            throw std::invalid_argument("Exact extrusion has invalid/duplicate active points.");
        const auto p = rounded_point(impl->points[id]);
        if (!display.insert({p.x, p.y, p.z}).second)
            throw std::invalid_argument("Exact extrusion display vertices are indistinguishable.");
    }
    for (std::size_t i = 0; i < impl->zrecipes.size(); ++i)
    {
        const auto& recipe = impl->zrecipes[i];
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
                z += impl->z[parent];
            }
            z /= recipe.parents.size();
        }
        impl->z.push_back(z);
    }
    Q previous = 0;
    double previous_display = 0;
    for (std::size_t k = 0; k < impl->zactive.size(); ++k)
    {
        const auto id = impl->zactive[k];
        if (id >= impl->z.size())
            throw std::invalid_argument("Exact axial point index is invalid.");
        const auto value = measured(impl->z[id]).as_double();
        if (k && (!(impl->z[id] > previous) || !(value > previous_display)))
            throw std::invalid_argument("Exact axial/display edges must be distinct and increasing.");
        previous = impl->z[id];
        previous_display = value;
        if (k)
        {
            const auto lo = measured(impl->z[impl->zactive[k - 1]]).as_double();
            const auto center = measured(Q((impl->z[id] + impl->z[impl->zactive[k - 1]]) / 2)).as_double();
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
            product.base.push_back(impl->points[impl->active[n]]);
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
        impl->areas.push_back(m.volume);
        impl->centers.push_back({m.first[0] / m.volume, m.first[1] / m.volume, 0});
    }
    for (const auto& edge : impl->topology->native_topology().side_faces())
    {
        const auto a = impl->points[impl->active[edge.nodes[0]]], b = impl->points[impl->active[edge.nodes[1]]];
        impl->deltas.push_back(sub(b, a));
        impl->edge_centers.push_back(mul(add(a, b), Q(1, 2)));
    }
    for (std::size_t e = 0; e < impl->deltas.size(); ++e)
    {
        const auto& edge = impl->topology->native_topology().side_face(e);
        const auto face = rounded_point(impl->edge_centers[e]);
        const auto owner = rounded_point(impl->centers.at(edge.owner));
        const auto& d = impl->deltas[e];
        const auto normal = rounded_point({d[1], -d[0], 0});
        if (!((face - owner).dot(normal) > 0))
            throw std::invalid_argument("Exact extrusion has unrepresentable signed base metric distances.");
        if (edge.neighbor != SemiStructMeshTopo::invalid_ordinal)
        {
            const auto neighbor = rounded_point(impl->centers.at(edge.neighbor));
            if (!((face - neighbor).dot(normal) < 0))
                throw std::invalid_argument("Exact extrusion neighbor metric distance is unresolved.");
        }
    }
    return std::shared_ptr<const ExactExtrudedConstruction>(new ExactExtrudedConstruction(std::move(impl)));
}

bool ExactExtrudedConstruction::matches_topology(const std::shared_ptr<const ExtrudedTopology>& topology) const noexcept
{
    return impl_->topology == topology;
}
std::vector<MeshUtils::Vec3> ExactExtrudedConstruction::display_points() const
{
    std::vector<MeshUtils::Vec3> result;
    for (auto id : impl_->active)
        result.push_back(rounded_point(impl_->points[id]));
    return result;
}
std::vector<double> ExactExtrudedConstruction::display_z() const
{
    std::vector<double> result;
    for (auto id : impl_->zactive)
        result.push_back(measured(impl_->z[id]).as_double());
    return result;
}
double ExactExtrudedConstruction::cell_volume(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    return measured(
        Q(impl_->areas.at(id.ij) * (impl_->z[impl_->zactive.at(id.k + 1)] - impl_->z[impl_->zactive.at(id.k)])))
        .as_double();
}
MeshUtils::Vec3 ExactExtrudedConstruction::cell_centroid(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    auto p = impl_->centers.at(id.ij);
    p[2] = (impl_->z[impl_->zactive.at(id.k)] + impl_->z[impl_->zactive.at(id.k + 1)]) / 2;
    return rounded_point(p);
}
double ExactExtrudedConstruction::face_area(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return measured(impl_->areas.at(id.ij)).as_double();
    const Q height = impl_->z[impl_->zactive.at(id.k + 1)] - impl_->z[impl_->zactive.at(id.k)];
    return square_root(Q(dot(impl_->deltas.at(id.ij), impl_->deltas.at(id.ij)) * height * height)).as_double();
}
MeshUtils::Vec3 ExactExtrudedConstruction::face_centroid(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    auto p = id.orientation == SemiStructuredIndexer::AXIAL ? impl_->centers.at(id.ij) : impl_->edge_centers.at(id.ij);
    p[2] = impl_->z[impl_->zactive.at(id.k)];
    if (id.orientation != SemiStructuredIndexer::AXIAL)
        p[2] = (p[2] + impl_->z[impl_->zactive.at(id.k + 1)]) / 2;
    return rounded_point(p);
}
MeshUtils::Vec3 ExactExtrudedConstruction::face_area_vector(std::size_t face) const
{
    auto id = impl_->topology->indexer().face_id(face);
    if (id.orientation == SemiStructuredIndexer::AXIAL)
        return {0, 0, (id.k == 0 ? -1 : 1) * measured(impl_->areas.at(id.ij)).as_double()};
    const auto& d = impl_->deltas.at(id.ij);
    const Q h = impl_->z[impl_->zactive.at(id.k + 1)] - impl_->z[impl_->zactive.at(id.k)];
    return rounded_point({d[1] * h, -d[0] * h, 0});
}
ConvexGeometryPtr ExactExtrudedConstruction::certified_cell_geometry(std::size_t cell) const
{
    auto id = impl_->topology->indexer().cell_id(cell);
    Product product;
    for (auto n : impl_->topology->base_cell_nodes(id.ij))
        product.base.push_back(impl_->points[impl_->active[n]]);
    product.lower = impl_->z[impl_->zactive.at(id.k)];
    product.upper = impl_->z[impl_->zactive.at(id.k + 1)];
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
    w.word(impl_->recipes.size());
    for (const auto& p : impl_->recipes)
    {
        for (auto x : p.root)
            w.real(x);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(impl_->active.size());
    for (auto id : impl_->active)
        w.word(id);
    w.word(impl_->zrecipes.size());
    for (const auto& p : impl_->zrecipes)
    {
        w.real(p.root);
        w.word(p.parents.size());
        for (auto id : p.parents)
            w.word(id);
    }
    w.word(impl_->zactive.size());
    for (auto id : impl_->zactive)
        w.word(id);
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
    std::size_t bytes = sizeof(Impl) + impl_->recipes.capacity() * sizeof(ExactPointRecipe) +
                        impl_->zrecipes.capacity() * sizeof(ExactScalarRecipe) +
                        (impl_->active.capacity() + impl_->zactive.capacity()) * sizeof(std::size_t);
    for (const auto& r : impl_->recipes)
        bytes += r.parents.capacity() * sizeof(std::size_t);
    for (const auto& r : impl_->zrecipes)
        bytes += r.parents.capacity() * sizeof(std::size_t);
    auto scalar = [&](const Q& q)
    {
        return sizeof(Q) + (q.backend().num().capacity() + q.backend().denom().capacity()) *
                               sizeof(boost::multiprecision::limb_type);
    };
    for (const auto* points : {&impl_->points, &impl_->centers, &impl_->deltas, &impl_->edge_centers})
    {
        bytes += points->capacity() * sizeof(P);
        for (const auto& p : *points)
            for (const auto& q : p)
                bytes += scalar(q);
    }
    for (const auto* values : {&impl_->z, &impl_->areas})
    {
        bytes += values->capacity() * sizeof(Q);
        for (const auto& q : *values)
            bytes += scalar(q);
    }
    return bytes; // conservative allocated-storage bound, including inline limb capacity
}
double ExactExtrudedConstruction::scaled_base_area(std::size_t base_cell) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Q largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->points[impl_->active[loop[i]]], impl_->points[impl_->active[loop[(i + 1) % loop.size()]]]);
        const Q length = dot(d, d);
        largest = std::max(largest, length);
    }
    return measured(Q(impl_->areas.at(base_cell) / largest)).as_double();
}
bool ExactExtrudedConstruction::base_quality_above(std::size_t base_cell, double minimum) const
{
    const auto loop = impl_->topology->base_cell_nodes(base_cell);
    Q largest = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
    {
        auto d = sub(impl_->points[impl_->active[loop[i]]], impl_->points[impl_->active[loop[(i + 1) % loop.size()]]]);
        const Q length = dot(d, d);
        largest = std::max(largest, length);
    }
    return impl_->areas.at(base_cell) > exact(minimum) * largest;
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
            product.base.push_back(impl_->points[impl_->active[n]]);
        return product;
    };
    for (std::size_t c = 0; c < impl_->areas.size(); ++c)
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
    std::vector<std::size_t> vertices = impl_->active;
    std::sort(
        vertices.begin(), vertices.end(), [&](auto a, auto b) { return impl_->points[a][0] < impl_->points[b][0]; });
    for (const auto& edge : impl_->topology->native_topology().side_faces())
    {
        const auto a = impl_->points[impl_->active[edge.nodes[0]]], b = impl_->points[impl_->active[edge.nodes[1]]],
                   delta = sub(b, a);
        const Q length = dot(delta, delta);
        const Q xmin = std::min(a[0], b[0]), xmax = std::max(a[0], b[0]), ymin = std::min(a[1], b[1]),
                ymax = std::max(a[1], b[1]);
        auto it = std::lower_bound(
            vertices.begin(), vertices.end(), xmin, [&](auto id, const Q& x) { return impl_->points[id][0] < x; });
        for (; it != vertices.end() && impl_->points[*it][0] <= xmax; ++it)
        {
            const auto& p = impl_->points[*it];
            if (p[1] < ymin || p[1] > ymax)
                continue;
            const auto offset = sub(p, a);
            const Q along = dot(offset, delta);
            if (zero(cross(delta, offset)) && along > 0 && along < length)
                throw std::invalid_argument("Extruded root embedding has an unresolved whole-edge T-junction.");
        }
    }
}

} // namespace SimpleFluid::Meshes
