#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include <bit>
#include <cmath>
#include <limits>
namespace SimpleFluid::PLIC
{
namespace scalar_detail = details::ordinary_kernel;
ScalarPacketWords scalar_packet(long double x)
{
    scalar_detail::finite(x);
    ScalarPacketWords words{};
    words[0] = std::signbit(x) && x != 0 ? 1 : 0;
    int exponent = 0;
    auto fraction = std::frexp(std::abs(x), &exponent);
    words[1] = static_cast<std::uint64_t>(static_cast<std::int64_t>(exponent));
    constexpr unsigned chunks = (std::numeric_limits<long double>::digits + 31) / 32;
    for (unsigned i = 0; i < chunks; ++i)
    {
        fraction = std::scalbn(fraction, 32);
        const auto chunk = static_cast<std::uint64_t>(fraction);
        words[2 + i] = chunk;
        fraction -= static_cast<long double>(chunk);
    }
    if (fraction != 0)
        scalar_detail::reject("wider-FP significand cannot be serialized losslessly.");
    return words;
}
long double scalar_from_packet(std::span<const std::uint64_t> words)
{
    if (words.size() != ScalarPacketWords{}.size())
        scalar_detail::reject("invalid numerical packet scalar word count.");
    const auto negative = words[0];
    const auto exponent = std::bit_cast<std::int64_t>(words[1]);
    const auto minimum_exponent =
        std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits;
    if (negative > 1 || exponent < minimum_exponent || exponent > std::numeric_limits<long double>::max_exponent)
        scalar_detail::reject("invalid numerical packet scalar exponent/sign.");
    constexpr unsigned chunks = (std::numeric_limits<long double>::digits + 31) / 32;
    long double significand = 0;
    for (unsigned i = 0; i < chunks; ++i)
    {
        const auto chunk = words[2 + i];
        if (chunk > std::numeric_limits<std::uint32_t>::max())
            scalar_detail::reject("invalid numerical packet significand.");
        significand = std::scalbn(significand, 32) + static_cast<long double>(chunk);
    }
    const auto value = std::scalbn(significand, static_cast<int>(exponent) - 32 * chunks);
    scalar_detail::finite(value);
    if (significand != 0 && value == 0)
        scalar_detail::reject("numerical packet loses nonzero coordinates.");
    return negative ? -value : value;
}
} // namespace SimpleFluid::PLIC
