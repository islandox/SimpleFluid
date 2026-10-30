#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryCodecTags.hh"
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace SimpleFluid::PLIC
{
namespace
{
using Measure = GeometryMeasure<ArithmeticMode::Ordinary>;
void admit_bytes(std::size_t bytes)
{
    if (bytes < sizeof(Measure))
        throw std::length_error("Ordinary FP geometry: measure storage cap exhausted.");
}
constexpr std::size_t measure_header_bytes = 2;
constexpr std::uint8_t measure_version = 1;
} // namespace

GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::from_nominal(long double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Ordinary FP geometry: nonfinite retained measure.");
    return GeometryMeasure(value);
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::volume(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& geometry)
{
    return from_nominal(geometry.volume().value);
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::operator+(
    const GeometryMeasure& other) const
{
    return from_nominal(value_ + other.value_);
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::operator-(
    const GeometryMeasure& other) const
{
    return from_nominal(value_ - other.value_);
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::scaled(double factor) const
{
    const auto result = from_nominal(value_ * factor);
    if (result.value_ == 0 && value_ != 0 && factor != 0)
        throw std::invalid_argument("Ordinary FP geometry: scaled nonzero measure underflows.");
    return result;
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::divided_by(
    const GeometryMeasure& denominator, std::size_t budget) const
{
    admit_bytes(budget);
    if (denominator.value_ == 0)
        throw std::invalid_argument("Ordinary FP geometry: zero measure divisor.");
    const auto result = from_nominal(value_ / denominator.value_);
    if (result.value_ == 0 && value_ != 0)
        throw std::invalid_argument("Ordinary FP geometry: nonzero measure quotient underflows.");
    return result;
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::squared(std::size_t budget) const
{
    admit_bytes(budget);
    const auto result = from_nominal(value_ * value_);
    if (result.value_ == 0 && value_ != 0)
        throw std::invalid_argument("Ordinary FP geometry: nonzero squared measure underflows.");
    return result;
}
bool GeometryMeasure<ArithmeticMode::Ordinary>::relative_error_at_most(
    const GeometryMeasure& reference, double tolerance) const
{
    if (!std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument("Ordinary FP geometry: invalid relative measure tolerance.");
    if (reference.value_ == 0)
        return value_ == 0;
    const auto error = from_nominal(value_ - reference.value_);
    const auto allowance = from_nominal(std::abs(reference.value_) * tolerance);
    return std::abs(error.value_) <= allowance.value_;
}
double GeometryMeasure<ArithmeticMode::Ordinary>::log_ratio(const GeometryMeasure& denominator) const
{
    if (!(value_ > 0) || !(denominator.value_ > 0))
        throw std::invalid_argument("Ordinary FP geometry: logarithmic ratio requires positive measures.");
    return static_cast<double>(std::log(value_) - std::log(denominator.value_));
}
std::vector<char> GeometryMeasure<ArithmeticMode::Ordinary>::encode(std::size_t budget) const
{
    std::array<char, 128> text;
    const auto result = std::to_chars(text.data(), text.data() + text.size(), value_, std::chars_format::scientific,
        std::numeric_limits<long double>::max_digits10);
    if (result.ec != std::errc{})
        throw std::invalid_argument("Ordinary FP geometry: retained measure encoding failed.");
    const auto count = static_cast<std::size_t>(result.ptr - text.data());
    if (budget < measure_header_bytes || count > budget - measure_header_bytes)
        throw std::length_error("Ordinary FP geometry: retained measure packet cap exhausted.");
    std::vector<char> packet;
    packet.reserve(measure_header_bytes + count);
    packet.push_back(static_cast<char>(GeometryTag::OrdinaryMeasure));
    packet.push_back(static_cast<char>(measure_version));
    packet.insert(packet.end(), text.data(), result.ptr);
    return packet;
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::decode(
    std::span<const char> packet, std::size_t budget)
{
    if (packet.size() > budget || packet.size() <= measure_header_bytes ||
        static_cast<GeometryTag>(static_cast<std::uint8_t>(packet[0])) != GeometryTag::OrdinaryMeasure ||
        static_cast<std::uint8_t>(packet[1]) != measure_version)
        throw std::invalid_argument("Ordinary FP geometry: invalid retained measure packet.");
    long double value;
    const auto result = std::from_chars(
        packet.data() + measure_header_bytes, packet.data() + packet.size(), value, std::chars_format::scientific);
    if (result.ec != std::errc{} || result.ptr != packet.data() + packet.size())
        throw std::invalid_argument("Ordinary FP geometry: malformed retained measure.");
    return from_nominal(value);
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::compacted(std::size_t budget) const
{
    admit_bytes(budget);
    return *this;
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::rounded_dyadic(
    unsigned bits, std::size_t budget) const
{
    admit_bytes(budget);
    if (!bits)
        throw std::invalid_argument("Ordinary FP geometry: zero dyadic precision.");
    if (!value_ || bits >= std::numeric_limits<long double>::digits)
        return *this;
    int exponent;
    const auto fraction = std::frexp(value_, &exponent);
    const auto rounded = std::ldexp(std::trunc(std::ldexp(fraction, bits)), exponent - static_cast<int>(bits));
    if (rounded == 0)
        throw std::invalid_argument("Ordinary FP geometry: nonzero rounded measure underflows.");
    return from_nominal(rounded);
}
} // namespace SimpleFluid::PLIC
