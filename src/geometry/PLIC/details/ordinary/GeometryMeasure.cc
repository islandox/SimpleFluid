#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryCodecTags.hh"
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#if defined(_LIBCPP_VERSION)
#include <cfenv>
#include <istream>
#include <locale>
#include <ostream>
#include <streambuf>
#endif
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
#if defined(_LIBCPP_VERSION)
class MeasureConversionEnvironment
{
public:
    MeasureConversionEnvironment()
    {
        if (std::feholdexcept(&saved_) != 0)
            throw std::invalid_argument("Ordinary FP geometry: cannot preserve measure conversion environment.");
        if (std::fesetround(FE_TONEAREST) != 0)
        {
            std::fesetenv(&saved_);
            throw std::invalid_argument("Ordinary FP geometry: cannot select nearest measure conversion.");
        }
    }
    ~MeasureConversionEnvironment() { std::fesetenv(&saved_); }
    MeasureConversionEnvironment(const MeasureConversionEnvironment&) = delete;
    MeasureConversionEnvironment& operator=(const MeasureConversionEnvironment&) = delete;

private:
    std::fenv_t saved_;
};
// libc++ has no long-double from_chars, and its to_chars overload can narrow
// to double. Classic-locale streams retain the full nominal representation;
// their text buffers borrow bounded caller storage rather than growing strings.
class MeasureOutputBuffer : public std::streambuf
{
public:
    explicit MeasureOutputBuffer(std::span<char> text) { setp(text.data(), text.data() + text.size()); }
    [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(pptr() - pbase()); }
};
class MeasureInputBuffer : public std::streambuf
{
public:
    explicit MeasureInputBuffer(std::span<const char> text)
    {
        // The get area is read-only during extraction; no put area is installed.
        auto* begin = const_cast<char*>(text.data());
        setg(begin, begin, begin + text.size());
    }
    [[nodiscard]] bool consumed() const { return gptr() == egptr(); }
};
bool scientific_significand_nonzero(std::span<const char> text)
{
    const auto malformed = [] { throw std::invalid_argument("Ordinary FP geometry: malformed retained measure."); };
    const auto digit = [](char value) { return value >= '0' && value <= '9'; };
    std::size_t at = 0, digits = 0;
    bool nonzero = false;
    if (at < text.size() && text[at] == '-')
        ++at;
    const auto scan_digits = [&]
    {
        while (at < text.size() && digit(text[at]))
        {
            nonzero = nonzero || text[at] != '0';
            ++at;
            ++digits;
        }
    };
    scan_digits();
    if (at < text.size() && text[at] == '.')
    {
        ++at;
        scan_digits();
    }
    if (!digits || at == text.size() || (text[at] != 'e' && text[at] != 'E'))
        malformed();
    ++at;
    if (at < text.size() && (text[at] == '+' || text[at] == '-'))
        ++at;
    const auto exponent_begin = at;
    while (at < text.size() && digit(text[at]))
        ++at;
    if (at == exponent_begin || at != text.size())
        malformed();
    return nonzero;
}
#endif
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
#if defined(_LIBCPP_VERSION)
    const MeasureConversionEnvironment environment;
    MeasureOutputBuffer buffer(text);
    std::ostream output(&buffer);
    output.imbue(std::locale::classic());
    output.setf(std::ios_base::scientific, std::ios_base::floatfield);
    output.precision(std::numeric_limits<long double>::max_digits10);
    output << value_;
    if (!output)
        throw std::invalid_argument("Ordinary FP geometry: retained measure encoding failed.");
    const auto count = buffer.size();
#else
    const auto result = std::to_chars(text.data(), text.data() + text.size(), value_, std::chars_format::scientific,
        std::numeric_limits<long double>::max_digits10);
    if (result.ec != std::errc{})
        throw std::invalid_argument("Ordinary FP geometry: retained measure encoding failed.");
    const auto count = static_cast<std::size_t>(result.ptr - text.data());
#endif
    if (budget < measure_header_bytes || count > budget - measure_header_bytes)
        throw std::length_error("Ordinary FP geometry: retained measure packet cap exhausted.");
    std::vector<char> packet;
    packet.reserve(measure_header_bytes + count);
    packet.push_back(static_cast<char>(GeometryTag::OrdinaryMeasure));
    packet.push_back(static_cast<char>(measure_version));
    packet.insert(packet.end(), text.data(), text.data() + count);
    return packet;
}
GeometryMeasure<ArithmeticMode::Ordinary> GeometryMeasure<ArithmeticMode::Ordinary>::decode(
    std::span<const char> packet, std::size_t budget)
{
    if (packet.size() > budget || packet.size() <= measure_header_bytes ||
        static_cast<GeometryTag>(static_cast<std::uint8_t>(packet[0])) != GeometryTag::OrdinaryMeasure ||
        static_cast<std::uint8_t>(packet[1]) != measure_version)
        throw std::invalid_argument("Ordinary FP geometry: invalid retained measure packet.");
    long double value = 0;
#if defined(_LIBCPP_VERSION)
    const auto text = packet.subspan(measure_header_bytes);
    const bool nonzero = scientific_significand_nonzero(text);
    const MeasureConversionEnvironment environment;
    MeasureInputBuffer buffer(text);
    std::istream input(&buffer);
    input.imbue(std::locale::classic());
    input >> std::noskipws >> value;
    // libc++ sets failbit on ERANGE even for a correctly rounded, nonzero
    // subnormal. Grammar, complete consumption and range still have to hold.
    if (input.bad() || !buffer.consumed() || !std::isfinite(value) ||
        (input.fail() && std::fpclassify(value) != FP_SUBNORMAL) || (nonzero && value == 0))
        throw std::invalid_argument("Ordinary FP geometry: malformed retained measure.");
#else
    const auto result = std::from_chars(
        packet.data() + measure_header_bytes, packet.data() + packet.size(), value, std::chars_format::scientific);
    if (result.ec != std::errc{} || result.ptr != packet.data() + packet.size())
        throw std::invalid_argument("Ordinary FP geometry: malformed retained measure.");
#endif
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
