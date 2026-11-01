/** @file Wire.cc
 * @brief Shared scalar, interface-plane and mode-checked phase record codecs.
 */
#include "FVM/details/GeneralVoF/Wire.hh"
#include "FVM/details/GeneralVoF/PlaneOrientation.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include <bit>
#include <cmath>
#include <limits>
namespace SimpleFluid::VoF::general_vof_detail
{
void real(wire::Writer& w, double value)
{
    w.word(std::bit_cast<std::uint64_t>(value));
}
double real(wire::Reader& r)
{
    return std::bit_cast<double>(r.word());
}
void finite_real(wire::Writer& w, long double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Finite VoF wire value is nonfinite.");
    int exponent = 0;
    const auto mantissa = std::frexp(value, &exponent);
    const auto high = static_cast<double>(mantissa);
    const auto low = static_cast<double>(mantissa - high);
    if (static_cast<long double>(high) + low != mantissa)
        throw std::invalid_argument("Finite VoF wire significand exceeds its fixed representation.");
    real(w, high);
    real(w, low);
    w.word(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
}
long double finite_real(wire::Reader& r)
{
    const auto high = real(r), low = real(r);
    const auto exponent = std::bit_cast<std::int64_t>(r.word());
    if (!std::isfinite(high) || !std::isfinite(low) || exponent < std::numeric_limits<int>::min() ||
        exponent > std::numeric_limits<int>::max())
        throw std::invalid_argument("Finite VoF wire scalar is malformed.");
    const auto result = std::scalbn(static_cast<long double>(high) + low, int(exponent));
    if (!std::isfinite(result) || ((high != 0 || low != 0) && result == 0))
        throw std::invalid_argument("Finite VoF wire scalar exceeds qualified range.");
    return result;
}
void finite_value(wire::Writer& w, PLIC::DynamicGeometryValue v)
{
    finite_detail::validate(v);
    w.word(std::uint64_t(v.arithmetic_mode));
    finite_real(w, v.value);
    finite_real(w, v.absolute_error);
}
PLIC::DynamicGeometryValue finite_value(wire::Reader& r, std::optional<ArithmeticMode> expected)
{
    const auto mode = r.word();
    if (mode > std::uint64_t(ArithmeticMode::BoundedCertification))
        throw std::invalid_argument("Finite VoF wire arithmetic mode is invalid.");
    PLIC::DynamicGeometryValue v{finite_real(r), finite_real(r), ArithmeticMode(mode)};
    finite_detail::validate(v);
    if ((v.arithmetic_mode == ArithmeticMode::Ordinary && v.absolute_error != 0) ||
        (expected && v.arithmetic_mode != *expected))
        throw std::invalid_argument("Finite VoF wire scalar arithmetic mode differs from its state.");
    return v;
}
void validate_plane_metadata(const PLIC::PolyhedralInterface& p,
    const PLIC::PolyhedralReconstructionOptions& options, ArithmeticMode mode)
{
    plane_wire_detail::model_metadata(p);
    if (p.model_policy != options.model_policy ||
        (mode != ArithmeticMode::Ordinary && p.model_policy != PLIC::PlicModelPolicy::RequireModelAgreement))
        throw std::invalid_argument("PLIC interface model policy differs from its immutable state.");
    if (!std::isfinite(p.stencil_volume_residual) || p.stencil_volume_residual < 0 ||
        (p.stencil_volume_residual_measured &&
            p.model_quality != (p.stencil_volume_residual > options.maximum_model_residual
                                   ? PLIC::PlicModelQuality::AboveThreshold
                                   : PLIC::PlicModelQuality::WithinThreshold)))
        throw std::invalid_argument("PLIC interface model quality differs from its measured residual.");
}
void plane(wire::Writer& w, const PLIC::PolyhedralInterface& p)
{
    plane_wire_detail::model_metadata(p);
    w.word(1); // Model policy and measured quality are part of the plane protocol.
    w.word(std::uint64_t(p.model_policy));
    w.word(std::uint64_t(p.model_quality));
    for (auto v : p.gas.normal)
        real(w, v);
    for (auto v : p.gas.anchor)
        real(w, v);
    real(w, p.gas.offset);
    w.word(p.pure_gas);
    w.word(p.pure_carrier);
    real(w, p.stencil_volume_residual);
    plane_wire_detail::boolean(w, p.stencil_volume_residual_measured);
    real(w, p.stencil_condition);
    real(w, p.orientation_step);
    w.word(p.fit_iterations);
    plane_wire_detail::orientation(w, p.orientation_method);
    w.word(p.fit_converged);
    finite_real(w, p.target_volume_numerical_error_estimate);
    w.word(p.finite_geometry);
    if (p.finite_geometry)
        finite_value(w, p.finite_target_volume_difference);
}
PLIC::PolyhedralInterface plane(wire::Reader& r, std::optional<ArithmeticMode> mode)
{
    PLIC::PolyhedralInterface p;
    if (r.word() != 1)
        throw std::invalid_argument("Unknown General VoF interface plane protocol.");
    p.model_policy = plane_wire_detail::model_policy(r.word());
    p.model_quality = plane_wire_detail::model_quality(r.word());
    if (mode && *mode != ArithmeticMode::Ordinary &&
        p.model_policy != PLIC::PlicModelPolicy::RequireModelAgreement)
        throw std::invalid_argument("General VoF interface model policy has a foreign arithmetic mode.");
    for (auto& v : p.gas.normal)
        v = real(r);
    for (auto& v : p.gas.anchor)
        v = real(r);
    p.gas.offset = real(r);
    p.pure_gas = plane_wire_detail::boolean(r);
    p.pure_carrier = plane_wire_detail::boolean(r);
    p.stencil_volume_residual = real(r);
    p.stencil_volume_residual_measured = plane_wire_detail::boolean(r);
    p.stencil_condition = real(r);
    p.orientation_step = real(r);
    p.fit_iterations = r.word();
    p.orientation_method = plane_wire_detail::orientation(r);
    p.fit_converged = plane_wire_detail::boolean(r);
    p.target_volume_numerical_error_estimate = finite_real(r);
    p.finite_geometry = plane_wire_detail::boolean(r);
    if (p.finite_geometry)
        p.finite_target_volume_difference = finite_value(r, mode);
    plane_wire_detail::model_metadata(p);
    return p;
}
void finite_phases(wire::Writer& w, const PLIC::DynamicPhaseMeasures& p)
{
    finite_value(w, p.bulk);
    finite_value(w, p.gas);
    finite_value(w, p.carrier);
}
PLIC::DynamicPhaseMeasures finite_phases(wire::Reader& r, ArithmeticMode mode)
{
    return {finite_value(r, mode), finite_value(r, mode), finite_value(r, mode)};
}
} // namespace SimpleFluid::VoF::general_vof_detail
