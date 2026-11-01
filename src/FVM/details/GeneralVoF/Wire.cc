/** @file Wire.cc
 * @brief Shared scalar, interface-plane and mode-checked phase record codecs.
 */
#include "FVM/details/GeneralVoF/Wire.hh"
#include "FVM/details/GeneralVoF/PlaneOrientation.hh"
#include "FVM/details/ProtocolTags.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/VoFWireProtocol.hh"
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
void validate_plane_metadata(const PLIC::DynamicPolyhedralReconstructionResult& result,
    const PLIC::PolyhedralReconstructionOptions& options, ArithmeticMode mode)
{
    if (PLIC::reconstruction_mode(result) != mode)
        throw std::invalid_argument("PLIC interface arithmetic mode differs from its immutable state.");
    std::visit(
        [&](const auto& p)
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
        },
        result);
}
void plane(wire::Writer& w, const PLIC::DynamicPolyhedralReconstructionResult& result)
{
    std::visit(
        [&](const auto& p)
        {
            plane_wire_detail::model_metadata(p);
            packet_codec::write_header(w, packet_codec::Family::InterfacePlane, 2, packet_codec::Record::Value);
            protocol_tags::write(w, p.model_policy, PLIC::PlicModelPolicy::ConservativeApproximation);
            protocol_tags::write(w, p.model_quality, PLIC::PlicModelQuality::AboveThreshold);
            for (auto v : p.geometry.gas.normal)
                real(w, v);
            for (auto v : p.geometry.gas.anchor)
                real(w, v);
            real(w, p.geometry.gas.offset);
            w.word(p.geometry.pure_gas);
            w.word(p.geometry.pure_carrier);
            real(w, p.stencil_volume_residual);
            plane_wire_detail::boolean(w, p.stencil_volume_residual_measured);
            real(w, p.stencil_condition);
            real(w, p.orientation_step);
            w.word(p.fit_iterations);
            plane_wire_detail::orientation(w, p.orientation_method);
            w.word(p.fit_converged);
            finite_real(w, p.target_volume_numerical_error_estimate);
            constexpr auto Mode = decltype(p.target_volume_difference)::arithmetic_mode;
            w.word(Mode != ArithmeticMode::Exact);
            if constexpr (Mode != ArithmeticMode::Exact)
            {
                long double error = 0;
                if constexpr (Mode == ArithmeticMode::BoundedCertification)
                    error = p.target_volume_difference.absolute_error;
                finite_value(w, {p.target_volume_difference.value, error, Mode});
            }
        },
        result);
}
PLIC::DynamicPolyhedralReconstructionResult plane(wire::Reader& r, std::optional<ArithmeticMode> mode)
{
    PLIC::PolyhedralReconstructionResult<ArithmeticMode::Exact> p;
    packet_codec::read_header(r, packet_codec::Family::InterfacePlane, 2, packet_codec::Record::Value);
    p.model_policy = protocol_tags::read(r, PLIC::PlicModelPolicy::ConservativeApproximation);
    p.model_quality = protocol_tags::read(r, PLIC::PlicModelQuality::AboveThreshold);
    if (mode && *mode != ArithmeticMode::Ordinary &&
        p.model_policy != PLIC::PlicModelPolicy::RequireModelAgreement)
        throw std::invalid_argument("General VoF interface model policy has a foreign arithmetic mode.");
    for (auto& v : p.geometry.gas.normal)
        v = real(r);
    for (auto& v : p.geometry.gas.anchor)
        v = real(r);
    p.geometry.gas.offset = real(r);
    p.geometry.pure_gas = plane_wire_detail::boolean(r);
    p.geometry.pure_carrier = plane_wire_detail::boolean(r);
    p.stencil_volume_residual = real(r);
    p.stencil_volume_residual_measured = plane_wire_detail::boolean(r);
    p.stencil_condition = real(r);
    p.orientation_step = real(r);
    p.fit_iterations = r.word();
    p.orientation_method = plane_wire_detail::orientation(r);
    p.fit_converged = plane_wire_detail::boolean(r);
    p.target_volume_numerical_error_estimate = finite_real(r);
    const auto finite_geometry = plane_wire_detail::boolean(r);
    if (finite_geometry)
    {
        const auto difference = finite_value(r, mode);
        if (difference.arithmetic_mode == ArithmeticMode::Ordinary)
        {
            PLIC::PolyhedralReconstructionResult<ArithmeticMode::Ordinary> typed;
            PLIC::copy_reconstruction_metadata(typed, p);
            typed.target_volume_difference = {difference.value};
            plane_wire_detail::model_metadata(typed);
            return PLIC::to_dynamic(std::move(typed));
        }
        PLIC::PolyhedralReconstructionResult<ArithmeticMode::BoundedCertification> typed;
        PLIC::copy_reconstruction_metadata(typed, p);
        typed.target_volume_difference = {difference.value, difference.absolute_error};
        plane_wire_detail::model_metadata(typed);
        return PLIC::to_dynamic(std::move(typed));
    }
    plane_wire_detail::model_metadata(p);
    return PLIC::to_dynamic(std::move(p));
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
