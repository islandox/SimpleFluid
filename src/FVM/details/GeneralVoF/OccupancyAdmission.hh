#pragma once
#include "utils/CompensatedSum.hh"
#include <cmath>

namespace SimpleFluid::VoF
{
/** Fixed physical occupancy requirement for algebraic transport. This is
 * independent of PLIC target-volume accuracy and accumulated arithmetic
 * provenance; neither a fit option nor an error history changes this limit. */
inline constexpr long double algebraic_physical_occupancy_relative_limit = 1e-13L;

struct AlgebraicOccupancyAdmission
{
    long double native_prediction = 0, native_published = 0, native_limit = 0, geometric_limit = 0;
    double rounded_native_published = 0, rounded_native_limit = 0;
    bool geometry_admitted = false, native_admitted = false, publication_admitted = false;
    [[nodiscard]] bool admitted() const { return geometry_admitted && native_admitted && publication_admitted; }
};

/** The wide geometry and native binary64 publication are independent physical
 * references. Preserve both, including the actual binary64 sum used by public
 * consumers; roundoff provenance cannot enlarge either physical limit. */
inline AlgebraicOccupancyAdmission algebraic_occupancy_admission(double gas, double carrier, double native_volume,
    long double geometric_volume, long double predicted, long double published, long double uncertainty)
{
    AlgebraicOccupancyAdmission result;
    ::SimpleFluid::detail::CompensatedSum<long double> native_prediction, native_published;
    native_prediction += predicted;
    native_prediction += geometric_volume;
    native_prediction += -native_volume;
    native_published += gas;
    native_published += carrier;
    native_published += -native_volume;
    result.native_prediction = native_prediction.value();
    result.native_published = native_published.value();
    const double rounded_sum = gas + carrier;
    result.rounded_native_published = rounded_sum - native_volume;
    result.geometric_limit = algebraic_physical_occupancy_relative_limit * geometric_volume;
    result.native_limit = algebraic_physical_occupancy_relative_limit * native_volume;
    result.rounded_native_limit = static_cast<double>(algebraic_physical_occupancy_relative_limit) * native_volume;
    result.geometry_admitted =
        std::abs(predicted) + uncertainty <= result.geometric_limit && std::abs(published) <= result.geometric_limit;
    result.native_admitted = std::abs(result.native_prediction) + uncertainty <= result.native_limit &&
                             std::abs(result.native_published) <= result.native_limit;
    result.publication_admitted = std::abs(result.rounded_native_published) <= result.rounded_native_limit;
    return result;
}
} // namespace SimpleFluid::VoF
