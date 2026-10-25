#include "geometry/PLIC/CertifiedGeometryProperties.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::PLIC
{
namespace
{
long double narrowing_error(long double value, long double error, double rounded)
{
    const auto total = std::abs(static_cast<long double>(rounded) - value) + error;
    return total == 0 ? 0 : std::nextafter(total, std::numeric_limits<long double>::infinity());
}
} // namespace

double CertifiedGeometryValue<ArithmeticMode::Ordinary>::as_double() const
{
    return DynamicGeometryValue{value, 0, arithmetic_mode}.as_double();
}

long double CertifiedGeometryValue<ArithmeticMode::Ordinary>::binary64_absolute_error() const
{
    return narrowing_error(value, 0, as_double());
}

long double CertifiedGeometryValue<ArithmeticMode::BoundedCertification>::lower() const
{
    return DynamicGeometryValue{value, absolute_error, arithmetic_mode}.lower();
}

long double CertifiedGeometryValue<ArithmeticMode::BoundedCertification>::upper() const
{
    return DynamicGeometryValue{value, absolute_error, arithmetic_mode}.upper();
}

double CertifiedGeometryValue<ArithmeticMode::BoundedCertification>::as_double() const
{
    return DynamicGeometryValue{value, absolute_error, arithmetic_mode}.as_double();
}

bool CertifiedGeometryValue<ArithmeticMode::BoundedCertification>::relative_error_at_most(
    long double target, double tolerance) const
{
    return DynamicGeometryValue{value, absolute_error, arithmetic_mode}.relative_error_at_most(target, tolerance);
}

long double CertifiedGeometryValue<ArithmeticMode::BoundedCertification>::binary64_absolute_error() const
{
    return narrowing_error(value, absolute_error, as_double());
}

double CertifiedGeometryValue<ArithmeticMode::Exact>::as_double() const
{
    const double result = static_cast<double>(value);
    if (!std::isfinite(result) || (result == 0 && (value != 0 || exact_nonzero)))
        throw std::runtime_error("Geometry measure cannot be represented in binary64 storage.");
    return result;
}

long double CertifiedGeometryValue<ArithmeticMode::Exact>::binary64_absolute_error() const
{
    return narrowing_error(value, absolute_error, as_double());
}
} // namespace SimpleFluid::PLIC
