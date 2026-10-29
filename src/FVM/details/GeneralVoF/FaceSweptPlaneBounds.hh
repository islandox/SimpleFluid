/** @file FaceSweptPlaneBounds.hh
 * @brief Outward offset support for the Ordinary translating-plane face witness.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
namespace SimpleFluid::VoF::general_vof_detail
{
/** Enclose the initial offset, the exact sum of the represented initial offset
 * and displacement, and the binary64 translated endpoint used by area queries.
 * Adjacent long-double values bound the addition's rounding loss. This interval
 * is geometric support for strict side predicates, never a clipping tolerance.
 */
[[nodiscard]] inline std::optional<std::array<long double, 2>> swept_face_offset_bounds(
    double initial, long double displacement)
{
    if (!std::isfinite(initial) || !std::isfinite(displacement))
        return {};
    const auto translated = static_cast<long double>(initial) + displacement;
    const auto represented = static_cast<double>(translated);
    if (!std::isfinite(translated) || !std::isfinite(represented))
        return {};
    const auto lower = std::nextafter(translated, -std::numeric_limits<long double>::infinity());
    const auto upper = std::nextafter(translated, std::numeric_limits<long double>::infinity());
    if (!std::isfinite(lower) || !std::isfinite(upper))
        return {};
    return std::array<long double, 2>{
        std::min({static_cast<long double>(initial), static_cast<long double>(represented), lower}),
        std::max({static_cast<long double>(initial), static_cast<long double>(represented), upper})};
}
} // namespace SimpleFluid::VoF::general_vof_detail
