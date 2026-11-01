/** @file PlaneOrientation.hh
 * @brief Private orientation provenance and checked booleans shared by interface wire formats.
 */
#pragma once
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include <cmath>

namespace SimpleFluid::VoF::plane_wire_detail
{
template<class Writer> void boolean(Writer& writer, bool value)
{
    writer.word(value);
}
template<class Reader> bool boolean(Reader& reader)
{
    const auto value = reader.word();
    if (value > 1)
        throw std::invalid_argument("Invalid PLIC interface boolean.");
    return bool(value);
}
inline PLIC::PolyhedralOrientationMethod orientation_method(std::uint64_t value)
{
    if (value > std::uint64_t(PLIC::PolyhedralOrientationMethod::AngularVolumeFit))
        throw std::invalid_argument("Invalid PLIC interface orientation method.");
    return PLIC::PolyhedralOrientationMethod(value);
}
template<class Writer> void orientation(Writer& writer, PLIC::PolyhedralOrientationMethod method)
{
    const auto value = std::uint64_t(method);
    (void) orientation_method(value);
    writer.word(value);
}
template<class Reader> PLIC::PolyhedralOrientationMethod orientation(Reader& reader)
{
    return orientation_method(reader.word());
}
inline PLIC::PlicModelPolicy model_policy(std::uint64_t value)
{
    if (value > std::uint64_t(PLIC::PlicModelPolicy::ConservativeApproximation))
        throw std::invalid_argument("Invalid PLIC interface model policy.");
    return PLIC::PlicModelPolicy(value);
}
inline PLIC::PlicModelQuality model_quality(std::uint64_t value)
{
    if (value > std::uint64_t(PLIC::PlicModelQuality::AboveThreshold))
        throw std::invalid_argument("Invalid PLIC interface model quality.");
    return PLIC::PlicModelQuality(value);
}
/** Context-free provenance checks shared by both plane codecs. The caller's
 * immutable threshold and policy are checked separately on state admission. */
inline void model_metadata(const PLIC::PolyhedralInterface& plane)
{
    (void) orientation_method(std::uint64_t(plane.orientation_method));
    (void) model_policy(std::uint64_t(plane.model_policy));
    (void) model_quality(std::uint64_t(plane.model_quality));
    if ((plane.model_policy == PLIC::PlicModelPolicy::ConservativeApproximation &&
            (!plane.finite_geometry ||
                plane.finite_target_volume_difference.arithmetic_mode != ArithmeticMode::Ordinary)) ||
        (plane.pure_gas && plane.pure_carrier) ||
        ((plane.pure_gas || plane.pure_carrier) && plane.stencil_volume_residual_measured) ||
        !std::isfinite(plane.stencil_volume_residual) || plane.stencil_volume_residual < 0 ||
        plane.stencil_volume_residual_measured != (plane.model_quality != PLIC::PlicModelQuality::NotMeasured) ||
        (plane.model_quality == PLIC::PlicModelQuality::AboveThreshold &&
            (plane.pure_gas || plane.pure_carrier ||
                (plane.orientation_method != PLIC::PolyhedralOrientationMethod::Gradient &&
                    (plane.orientation_method != PLIC::PolyhedralOrientationMethod::AngularVolumeFit ||
                        !plane.fit_converged)))) ||
        (plane.fit_converged && plane.orientation_method != PLIC::PolyhedralOrientationMethod::AngularVolumeFit) ||
        (plane.model_policy == PLIC::PlicModelPolicy::RequireModelAgreement &&
            plane.model_quality == PLIC::PlicModelQuality::AboveThreshold))
        throw std::invalid_argument("PLIC interface model provenance is inconsistent.");
}
} // namespace SimpleFluid::VoF::plane_wire_detail
