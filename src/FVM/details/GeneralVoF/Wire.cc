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
namespace
{
using ReconstructionHeader = ReconstructionPacketHeader;
constexpr auto reconstruction_header_fields = []
{
    std::array<bool, sizeof(ReconstructionHeader)> fields{};
    const auto mark = [&](std::size_t offset, std::size_t size)
    {
        for (std::size_t i = offset; i < offset + size; ++i)
            fields[i] = true;
    };
    mark(offsetof(ReconstructionHeader, family), sizeof(ReconstructionHeader::family));
    mark(offsetof(ReconstructionHeader, version), sizeof(ReconstructionHeader::version));
    mark(offsetof(ReconstructionHeader, kind), sizeof(ReconstructionHeader::kind));
    mark(offsetof(ReconstructionHeader, mode), sizeof(ReconstructionHeader::mode));
    mark(offsetof(ReconstructionHeader, profile), sizeof(ReconstructionHeader::profile));
    mark(offsetof(ReconstructionHeader, body_length), sizeof(ReconstructionHeader::body_length));
    return fields;
}();
void reconstruction_profile_check()
{
    if (reconstruction_numeric_profile() == PLIC::GeometryNumericProfile::Unsupported)
        throw std::invalid_argument("Unsupported PLIC reconstruction native scalar profile.");
}
void reconstruction_header(wire::Writer& writer, ArithmeticMode mode)
{
    reconstruction_profile_check();
    const auto body_length = static_cast<std::uint64_t>(reconstruction_body_wire_bytes(mode));
    // Member assignments never determine padding: initialize the serialized
    // storage itself, then overwrite only the known member positions.
    std::array<uint8_t, sizeof(ReconstructionHeader)> bytes{};
    bytes[offsetof(ReconstructionHeader, family)] = static_cast<uint8_t>(packet_codec::Family::InterfacePlane);
    bytes[offsetof(ReconstructionHeader, version)] = reconstruction_packet_version;
    bytes[offsetof(ReconstructionHeader, kind)] = static_cast<uint8_t>(packet_codec::Record::Value);
    bytes[offsetof(ReconstructionHeader, mode)] = static_cast<uint8_t>(mode);
    bytes[offsetof(ReconstructionHeader, profile)] = static_cast<uint8_t>(reconstruction_numeric_profile());
    for (unsigned i = 0; i < sizeof(std::uint64_t); ++i)
        bytes[offsetof(ReconstructionHeader, body_length) + i] = static_cast<uint8_t>(body_length >> (8 * i));
    for (const auto byte : bytes)
        writer.byte(byte);
}
ArithmeticMode reconstruction_header(wire::Reader& reader, std::optional<ArithmeticMode> expected)
{
    reconstruction_profile_check();
    if (reader.offset > reader.data.size() || reader.data.size() - reader.offset < sizeof(ReconstructionHeader))
        throw std::invalid_argument("Truncated PLIC reconstruction header.");
    std::array<uint8_t, sizeof(ReconstructionHeader)> bytes{};
    for (auto& byte : bytes)
        byte = reader.byte();
    if (bytes[offsetof(ReconstructionHeader, family)] != static_cast<uint8_t>(packet_codec::Family::InterfacePlane) ||
        bytes[offsetof(ReconstructionHeader, version)] != reconstruction_packet_version ||
        bytes[offsetof(ReconstructionHeader, kind)] != static_cast<uint8_t>(packet_codec::Record::Value))
        throw std::invalid_argument("Unknown PLIC reconstruction family, version or kind.");
    const auto mode_byte = bytes[offsetof(ReconstructionHeader, mode)];
    if (mode_byte > static_cast<uint8_t>(ArithmeticMode::Exact) ||
        (expected && mode_byte != static_cast<uint8_t>(*expected)) ||
        bytes[offsetof(ReconstructionHeader, profile)] != static_cast<uint8_t>(reconstruction_numeric_profile()))
        throw std::invalid_argument("Foreign PLIC reconstruction arithmetic mode or native scalar profile.");
    for (std::size_t i = 0; i < bytes.size(); ++i)
        if (!reconstruction_header_fields[i] && bytes[i])
            throw std::invalid_argument("Nonzero PLIC reconstruction header padding.");
    std::uint64_t length = 0;
    for (unsigned i = 0; i < sizeof(std::uint64_t); ++i)
        length |= std::uint64_t(bytes[offsetof(ReconstructionHeader, body_length) + i]) << (8 * i);
    const auto mode = static_cast<ArithmeticMode>(mode_byte);
    if (length != reconstruction_body_wire_bytes(mode) || length > reader.data.size() - reader.offset)
        throw std::invalid_argument("Truncated or incorrectly sized PLIC reconstruction body.");
    return mode;
}
PLIC::ScalarPacketWords reconstruction_scalar_packet(long double value)
{
    auto words = PLIC::scalar_packet(value);
    // ScalarPacket's geometric normalization removes the sign of zero. A
    // reconstruction diagnostic preserves the supplied finite value and signed zero.
    words[0] = std::signbit(value) ? 1 : 0;
    return words;
}
void reconstruction_scalar(wire::Writer& writer, long double value)
{
    for (const auto word : reconstruction_scalar_packet(value))
        writer.word(word);
}
long double reconstruction_scalar(wire::Reader& reader)
{
    PLIC::ScalarPacketWords words;
    for (auto& word : words)
        word = reader.word();
    const auto value = PLIC::scalar_from_packet(words);
    if (reconstruction_scalar_packet(value) != words)
        throw std::invalid_argument("Noncanonical PLIC reconstruction native scalar.");
    return value;
}
void reconstruction_boolean(wire::Writer& writer, bool value)
{
    writer.byte(static_cast<uint8_t>(value));
}
bool reconstruction_boolean(wire::Reader& reader)
{
    const auto value = reader.byte();
    if (value > 1)
        throw std::invalid_argument("Invalid PLIC reconstruction boolean.");
    return value != 0;
}
template<class Count> void reconstruction_count(wire::Writer& writer, Count value)
{
    static_assert(std::is_unsigned_v<Count>);
    if constexpr (sizeof(Count) > sizeof(std::uint64_t))
        if (value > std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("PLIC reconstruction count exceeds wire width.");
    writer.word(static_cast<std::uint64_t>(value));
}
template<class Count> Count reconstruction_count(wire::Reader& reader)
{
    static_assert(std::is_unsigned_v<Count>);
    const auto value = reader.word();
    if constexpr (sizeof(Count) < sizeof(std::uint64_t))
        if (value > std::numeric_limits<Count>::max())
            throw std::invalid_argument("PLIC reconstruction count exceeds its destination width.");
    return static_cast<Count>(value);
}
void reconstruction_resolution(wire::Writer& writer, const PLIC::PolyhedralResolutionAssessment& assessment)
{
    using Reason = PLIC::PolyhedralResolutionAssessment::Reason;
    const auto reason = static_cast<std::underlying_type_t<Reason>>(assessment.reason);
    if (reason < 0 || reason > static_cast<std::underlying_type_t<Reason>>(Reason::IllConditionedAngularSupport))
        throw std::invalid_argument("Unknown PLIC reconstruction resolution reason.");
    writer.byte(static_cast<uint8_t>(reason));
    for (const auto value : {assessment.local_cell_span, assessment.requested_cell_span,
             assessment.gas_support_fraction, assessment.carrier_support_fraction, assessment.stencil_condition,
             assessment.directional_strength, assessment.directional_uncertainty, assessment.angular_support_condition})
        real(writer, value);
    reconstruction_count(writer, assessment.angular_support_cells);
    reconstruction_boolean(writer, assessment.angular_fit_eligible);
    reconstruction_count(writer, assessment.stencil_cells);
}
PLIC::PolyhedralResolutionAssessment reconstruction_resolution(wire::Reader& reader)
{
    using Reason = PLIC::PolyhedralResolutionAssessment::Reason;
    const auto reason = reader.byte();
    if (reason > static_cast<uint8_t>(Reason::IllConditionedAngularSupport))
        throw std::invalid_argument("Unknown PLIC reconstruction resolution reason.");
    PLIC::PolyhedralResolutionAssessment assessment;
    assessment.reason = static_cast<Reason>(reason);
    for (auto* value : {&assessment.local_cell_span, &assessment.requested_cell_span, &assessment.gas_support_fraction,
             &assessment.carrier_support_fraction, &assessment.stencil_condition, &assessment.directional_strength,
             &assessment.directional_uncertainty, &assessment.angular_support_condition})
        *value = real(reader);
    assessment.angular_support_cells = reconstruction_count<std::size_t>(reader);
    assessment.angular_fit_eligible = reconstruction_boolean(reader);
    assessment.stencil_cells = reconstruction_count<std::size_t>(reader);
    return assessment;
}
template<ArithmeticMode Mode>
void reconstruction_body(wire::Writer& writer, const PLIC::PolyhedralReconstructionResult<Mode>& result)
{
    if constexpr (Mode != ArithmeticMode::Ordinary)
        if (result.target_volume_difference.absolute_error < 0)
            throw std::invalid_argument("Negative PLIC reconstruction residual radius.");
    for (const auto value : result.geometry.gas.normal)
        real(writer, value);
    for (const auto value : result.geometry.gas.anchor)
        real(writer, value);
    real(writer, result.geometry.gas.offset);
    reconstruction_boolean(writer, result.geometry.pure_gas);
    reconstruction_boolean(writer, result.geometry.pure_carrier);
    real(writer, result.target_volume_residual);
    real(writer, result.stencil_volume_residual);
    reconstruction_boolean(writer, result.stencil_volume_residual_measured);
    reconstruction_boolean(writer, result.fit_converged);
    protocol_tags::write(writer, result.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
    protocol_tags::write(writer, result.model_policy, PLIC::PlicModelPolicy::ConservativeApproximation);
    protocol_tags::write(writer, result.model_quality, PLIC::PlicModelQuality::AboveThreshold);
    reconstruction_scalar(writer, result.target_volume_numerical_error_estimate);
    real(writer, result.stencil_condition);
    real(writer, result.orientation_step);
    reconstruction_count(writer, result.fit_iterations);
    reconstruction_resolution(writer, result.resolution);
    reconstruction_scalar(writer, result.target_volume_difference.value);
    if constexpr (Mode != ArithmeticMode::Ordinary)
        reconstruction_scalar(writer, result.target_volume_difference.absolute_error);
    if constexpr (Mode == ArithmeticMode::Exact)
        reconstruction_boolean(writer, result.target_volume_difference.exact_nonzero);
}
template<ArithmeticMode Mode> PLIC::DynamicPolyhedralReconstructionResult reconstruction_body(wire::Reader& reader)
{
    PLIC::PolyhedralReconstructionResult<Mode> result;
    for (auto& value : result.geometry.gas.normal)
        value = real(reader);
    for (auto& value : result.geometry.gas.anchor)
        value = real(reader);
    result.geometry.gas.offset = real(reader);
    result.geometry.pure_gas = reconstruction_boolean(reader);
    result.geometry.pure_carrier = reconstruction_boolean(reader);
    result.target_volume_residual = real(reader);
    result.stencil_volume_residual = real(reader);
    result.stencil_volume_residual_measured = reconstruction_boolean(reader);
    result.fit_converged = reconstruction_boolean(reader);
    result.orientation_method = protocol_tags::read(reader, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
    result.model_policy = protocol_tags::read(reader, PLIC::PlicModelPolicy::ConservativeApproximation);
    result.model_quality = protocol_tags::read(reader, PLIC::PlicModelQuality::AboveThreshold);
    result.target_volume_numerical_error_estimate = reconstruction_scalar(reader);
    result.stencil_condition = real(reader);
    result.orientation_step = real(reader);
    result.fit_iterations = reconstruction_count<unsigned>(reader);
    result.resolution = reconstruction_resolution(reader);
    result.target_volume_difference.value = reconstruction_scalar(reader);
    if constexpr (Mode != ArithmeticMode::Ordinary)
    {
        result.target_volume_difference.absolute_error = reconstruction_scalar(reader);
        if (result.target_volume_difference.absolute_error < 0)
            throw std::invalid_argument("Negative PLIC reconstruction residual radius.");
    }
    if constexpr (Mode == ArithmeticMode::Exact)
        result.target_volume_difference.exact_nonzero = reconstruction_boolean(reader);
    plane_wire_detail::model_metadata(result);
    return PLIC::to_dynamic(std::move(result));
}
} // namespace
void plane(wire::Writer& writer, const PLIC::DynamicPolyhedralReconstructionResult& result)
{
    std::visit(
        [&](const auto& typed)
        {
            constexpr auto mode = decltype(typed.target_volume_difference)::arithmetic_mode;
            plane_wire_detail::model_metadata(typed);
            reconstruction_header(writer, mode);
            reconstruction_body(writer, typed);
        },
        result);
}
PLIC::DynamicPolyhedralReconstructionResult plane(wire::Reader& reader, std::optional<ArithmeticMode> expected)
{
    const auto mode = reconstruction_header(reader, expected);
    const auto body_bytes = reconstruction_body_wire_bytes(mode);
    wire::Reader body{reader.data.subspan(reader.offset, body_bytes)};
    auto result = [&]() -> PLIC::DynamicPolyhedralReconstructionResult
    {
        switch (mode)
        {
            case ArithmeticMode::Ordinary:
                return reconstruction_body<ArithmeticMode::Ordinary>(body);
            case ArithmeticMode::BoundedCertification:
                return reconstruction_body<ArithmeticMode::BoundedCertification>(body);
            case ArithmeticMode::Exact:
                return reconstruction_body<ArithmeticMode::Exact>(body);
        }
        throw std::invalid_argument("Unknown PLIC reconstruction arithmetic mode.");
    }();
    body.done();
    reader.offset += body_bytes;
    return result;
}
PLIC::DynamicPolyhedralReconstructionResult plane_for_exact_transfer(wire::Reader& reader)
{
    auto result = plane(reader);
    std::visit(
        [](const auto& typed)
        {
            for (const auto value : typed.geometry.gas.normal)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite exact transfer plane metadata.");
            for (const auto value : typed.geometry.gas.anchor)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite exact transfer plane metadata.");
            for (const auto value : {typed.geometry.gas.offset, typed.target_volume_residual, typed.stencil_condition,
                     typed.orientation_step})
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite exact transfer plane metadata.");
        },
        result);
    return result;
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
