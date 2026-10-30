/** @file NativeGeometrySupport.hh
 * @brief Finite source coordinates, incidence identities, and checked wire encoding.
 */
#pragma once
#include "geometry/PLIC/CertifiedGeometryProperties.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::PLIC::details::ordinary_native
{
using WidePoint = std::array<long double, 3>;
[[noreturn]] inline void reject(const char* reason)
{
    throw std::invalid_argument(std::string("Ordinary FP geometry: ") + reason);
}
inline void finite(long double v)
{
    if (!std::isfinite(v))
        reject("nonfinite native geometry coordinate.");
}
inline WidePoint add(WidePoint a, const WidePoint& b)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        a[i] += b[i];
        finite(a[i]);
    }
    return a;
}
inline WidePoint sub(WidePoint a, const WidePoint& b)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        a[i] -= b[i];
        finite(a[i]);
    }
    return a;
}
inline WidePoint mul(WidePoint a, long double b)
{
    for (auto& v : a)
    {
        v *= b;
        finite(v);
    }
    return a;
}
inline long double dot(const WidePoint& a, const WidePoint& b)
{
    const auto v = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    finite(v);
    return v;
}
inline WidePoint cross(const WidePoint& a, const WidePoint& b)
{
    WidePoint r{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    for (auto v : r)
        finite(v);
    return r;
}
inline WidePoint wide(GeometryPoint p)
{
    return {p[0], p[1], p[2]};
}
inline GeometryPoint display(WidePoint p)
{
    return {GeometryValue<ArithmeticMode::Ordinary>{p[0]}.as_double(),
        GeometryValue<ArithmeticMode::Ordinary>{p[1]}.as_double(),
        GeometryValue<ArithmeticMode::Ordinary>{p[2]}.as_double()};
}
struct Writer
{
    std::vector<char> bytes;
    void word(std::uint64_t value)
    {
        for (unsigned i = 0; i < 8; ++i)
            bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
    }
    void real(long double value)
    {
        finite(value);
        if (value == 0)
        {
            word(0);
            word(0);
            return;
        }
        int exponent = 0;
        auto mantissa = std::frexp(std::abs(value), &exponent);
        word((std::uint64_t(std::uint32_t(exponent)) << 1) | (std::signbit(value) ? 1 : 0));
        word(std::numeric_limits<long double>::digits);
        for (int written = 0; written < std::numeric_limits<long double>::digits; written += 32)
        {
            mantissa = std::ldexp(mantissa, 32);
            const auto limb = static_cast<std::uint32_t>(mantissa);
            word(limb);
            mantissa -= limb;
        }
    }
    void point(const WidePoint& p)
    {
        for (auto v : p)
            real(v);
    }
    void packet(std::span<const char> b)
    {
        word(b.size());
        bytes.insert(bytes.end(), b.begin(), b.end());
    }
};
struct Reader
{
    std::span<const char> bytes;
    std::size_t cursor = 0;
    std::uint64_t word()
    {
        if (bytes.size() - cursor < 8)
            reject("truncated finite geometry packet.");
        std::uint64_t r = 0;
        for (unsigned i = 0; i < 8; ++i)
            r |= std::uint64_t(static_cast<unsigned char>(bytes[cursor++])) << (8 * i);
        return r;
    }
    std::size_t count(std::size_t maximum = 65536)
    {
        const auto n = word();
        if (n > maximum || n > bytes.size() - cursor)
            reject("excessive finite geometry packet count.");
        return static_cast<std::size_t>(n);
    }
    long double real()
    {
        const auto descriptor = word(), bits = word();
        if (bits == 0)
        {
            if (descriptor)
                reject("noncanonical finite zero.");
            return 0;
        }
        if (bits != std::numeric_limits<long double>::digits || descriptor >> 33)
            reject("unsupported finite coordinate representation.");
        long double mantissa = 0;
        for (std::size_t read = 0; read < bits; read += 32)
        {
            const auto limb = word();
            if (limb > std::numeric_limits<std::uint32_t>::max())
                reject("invalid finite coordinate significand.");
            mantissa += std::ldexp(static_cast<long double>(limb), -static_cast<int>(read + 32));
        }
        if (!(mantissa >= .5L && mantissa < 1))
            reject("noncanonical finite coordinate significand.");
        const auto exponent = std::bit_cast<std::int32_t>(std::uint32_t(descriptor >> 1));
        auto v = std::ldexp(mantissa, exponent);
        finite(v);
        if (v == 0)
            reject("finite coordinate packet underflows.");
        return descriptor & 1 ? -v : v;
    }
    WidePoint point() { return {real(), real(), real()}; }
    std::span<const char> packet()
    {
        const auto n = count(bytes.size());
        auto result = bytes.subspan(cursor, n);
        cursor += n;
        return result;
    }
    void finish()
    {
        if (cursor != bytes.size())
            reject("trailing finite geometry packet bytes.");
    }
};
inline constexpr std::uint64_t point_tag = 0x53464650504e5431ULL;
inline constexpr std::uint64_t displacement_tag = 0x5346465044535031ULL;
inline std::vector<char> point_packet(const WidePoint& p)
{
    Writer w;
    w.word(point_tag);
    w.point(p);
    return std::move(w.bytes);
}
inline WidePoint read_point(std::span<const char> packet)
{
    Reader r{packet};
    if (r.word() != point_tag)
        reject("foreign point identity arithmetic mode.");
    const auto p = r.point();
    r.finish();
    return p;
}
inline std::vector<char> displacement_packet(const WidePoint& p)
{
    if (p == WidePoint{})
        return {};
    Writer w;
    w.word(displacement_tag);
    w.point(p);
    return std::move(w.bytes);
}
inline WidePoint read_displacement(std::span<const char> packet)
{
    if (packet.empty())
        return {};
    Reader r{packet};
    if (r.word() != displacement_tag)
        reject("foreign displacement arithmetic mode.");
    const auto p = r.point();
    r.finish();
    return p;
}
inline WidePoint polygon_vector(std::span<const WidePoint> points)
{
    if (points.size() < 3)
        return {};
    WidePoint sum{};
    for (std::size_t i = 1; i + 1 < points.size(); ++i)
        sum = add(sum, cross(sub(points[i], points[0]), sub(points[i + 1], points[0])));
    return mul(sum, .5L);
}
inline PlaneSupportIdentity<ArithmeticMode::Ordinary> plane_packet(
    const WidePoint& normal, const WidePoint& anchor, long double offset = 0)
{
    std::size_t axis = 0;
    for (std::size_t i = 1; i < 3; ++i)
        if (std::abs(normal[i]) > std::abs(normal[axis]))
            axis = i;
    if (normal[axis] == 0)
        reject("zero native support normal.");
    WidePoint canonical{};
    for (std::size_t i = 0; i < 3; ++i)
        canonical[i] = normal[i] / normal[axis];
    const auto location = (dot(normal, anchor) + offset) / normal[axis];
    finite(location);
    Writer w;
    w.word(0x5346465053505431ULL);
    w.word(axis);
    w.point(canonical);
    w.real(location);
    return details::GeometryIdentityAccess::make<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(std::move(w.bytes));
}
} // namespace SimpleFluid::PLIC::details::ordinary_native

namespace SimpleFluid::PLIC::details::ordinary_native
{
inline PlaneSupportIdentity<ArithmeticMode::Ordinary> source_support_packet(std::span<const WidePoint> source)
{
    if (source.size() < 3)
        reject("native support needs a face source.");
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (std::all_of(source.begin(), source.end(), [&](const auto& p) { return p[axis] == source.front()[axis]; }))
        {
            Writer w;
            w.word(0x53464e4158495331ULL);
            w.word(axis);
            w.real(source.front()[axis]);
            return details::GeometryIdentityAccess::make<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(
                std::move(w.bytes));
        }
    std::vector<WidePoint> polygon(source.begin(), source.end());
    bool changed = true;
    while (changed && polygon.size() > 3)
    {
        changed = false;
        for (std::size_t i = 0; i < polygon.size(); ++i)
            if (cross(sub(polygon[i], polygon[(i + polygon.size() - 1) % polygon.size()]),
                    sub(polygon[(i + 1) % polygon.size()], polygon[i])) == WidePoint{})
            {
                polygon.erase(polygon.begin() + i);
                changed = true;
                break;
            }
    }
    std::sort(polygon.begin(), polygon.end());
    Writer w;
    w.word(0x53464e534f555231ULL);
    w.word(polygon.size());
    for (const auto& p : polygon)
        w.point(p);
    return details::GeometryIdentityAccess::make<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(std::move(w.bytes));
}
inline PlaneSupportIdentity<ArithmeticMode::Ordinary> input_support_packet(
    GeometryPoint normal, GeometryPoint anchor, double offset)
{
    std::size_t nonzero = 0, axis = 0;
    for (std::size_t i = 0; i < 3; ++i)
        if (normal[i] != 0)
        {
            ++nonzero;
            axis = i;
        }
    if (nonzero == 1)
    {
        const long double location =
            static_cast<long double>(anchor[axis]) + static_cast<long double>(offset) / normal[axis];
        if ((location - anchor[axis]) * normal[axis] == offset)
        {
            Writer w;
            w.word(0x53464e4158495331ULL);
            w.word(axis);
            w.real(location);
            return details::GeometryIdentityAccess::make<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(
                std::move(w.bytes));
        }
    }
    int direction = 1;
    for (auto n : normal)
        if (n != 0)
        {
            direction = n < 0 ? -1 : 1;
            break;
        }
    Writer w;
    w.word(0x53464e494e505431ULL);
    for (auto n : normal)
        w.real(static_cast<long double>(direction) * n);
    w.point(wide(anchor));
    w.real(static_cast<long double>(direction) * offset);
    return details::GeometryIdentityAccess::make<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(std::move(w.bytes));
}
} // namespace SimpleFluid::PLIC::details::ordinary_native

namespace SimpleFluid::PLIC
{
template<ArithmeticMode> class ConvexCellGeometry;
namespace details::ordinary_native
{
void certify_cell_partition(const ConvexCellGeometry<ArithmeticMode::Ordinary>&,
    std::span<const std::shared_ptr<const ConvexCellGeometry<ArithmeticMode::Ordinary>>>, double relative_budget);
}
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC::details::ordinary_native
{
inline PlaneSupportIdentity<ArithmeticMode::Ordinary> translated_support_packet(
    const PlaneSupportIdentity<ArithmeticMode::Ordinary>& support, const WidePoint& shift)
{
    if (shift == WidePoint{})
        return support;
    Writer writer;
    writer.word(0x53464e5452414e31ULL);
    writer.packet(details::GeometryIdentityAccess::key(support));
    writer.point(shift);
    return details::GeometryIdentityAccess::make<PlaneSupportIdentity<ArithmeticMode::Ordinary>>(
        std::move(writer.bytes), support);
}
} // namespace SimpleFluid::PLIC::details::ordinary_native
