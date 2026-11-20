/** @file QbmmReportNumber.hh
 * @brief Narrow a scaled QBMM diagnostic without changing kernel acceptance.
 */
#pragma once

#include "utils/export_helper.hh"
#include <cmath>
#include <limits>

namespace SimpleFluid::Reacting::qbmm_detail
{
SIMPLEFLUID_EQUATIONS_LOCAL inline double diagnostic_double(long double value, int exponent = 0)
{
    constexpr auto unavailable = std::numeric_limits<double>::quiet_NaN();
    if (value == 0)
        return static_cast<double>(value);
    const auto scaled = exponent == 0 ? value : std::scalbn(value, exponent);
    if (!std::isfinite(scaled) || std::abs(scaled) > std::numeric_limits<double>::max())
        return unavailable;
    const auto rounded = static_cast<double>(scaled);
    return rounded == 0 ? unavailable : rounded;
}
} // namespace SimpleFluid::Reacting::qbmm_detail
