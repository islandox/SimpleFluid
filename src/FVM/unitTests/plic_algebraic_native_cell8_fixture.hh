/** @file plic_algebraic_native_cell8_fixture.hh
 * @brief Actual native candidate input at the post-projection cell-8 rejection.
 *
 * Captured 2026-10-30 from GF stability .2, second attempted step after
 * accepted time .087217791393195965. The five rows retain the exact ordered
 * shell, both fitting labels, observed-mixed flags, frame and native supports.
 * Source: algebraic_vof_continuation_20261030/native-cell8/input-002.json.
 * Represented source packets are restored; historical plane pointers/ancestry
 * are not reproduced. Ordinary measure-only queries use this numerical shell.
 */
#pragma once
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace captured_native_cell8
{
using namespace SimpleFluid;
using WidePoint = std::array<long double, 3>;
using Cell = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
using Sample = PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary>;
struct Row
{
    std::uint64_t id;
    PLIC::GeometryPoint origin;
    int frame_exponent;
    std::array<WidePoint, 8> stored_vertices;
    double gas, carrier;
    bool observed_mixed;
    long double stored_volume;
    std::array<unsigned, 6> support_indices;
};
inline const std::vector<std::vector<std::size_t>> faces{
    {3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
inline constexpr std::array<std::uint64_t, 4> stencil_ids{4, 9, 12, 20};
inline constexpr std::array<Row, 5> rows{{
    {8, {0x1.8000000000000p-1, 0x0.0p+0, 0x0.0p+0}, -1,
        {{
            {0x0p+0L, 0x0p+0L, 0x0p+0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p+0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p+0L},
            {-0x1p-2L, 0x0p+0L, 0x0p+0L},
            {0x0p+0L, 0x0p+0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p+0L, 0x1p-1L},
        }},
        0x1.6c369dd457270p-14, 0x1.fa4f2588aea36p-8, true, 0x1p-7L, {0, 1, 2, 3, 4, 5}},
    {4, {0x1.0000000000000p-1, 0x0.0p+0, 0x0.0p+0}, -1,
        {{
            {0x0p+0L, 0x0p+0L, 0x0p+0L},
            {0x1p-2L, 0x0p+0L, 0x0p+0L},
            {0xb333333333333p-53L, 0x1p-1L, 0x0p+0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p+0L},
            {0x0p+0L, 0x0p+0L, 0x1p-1L},
            {0x1p-2L, 0x0p+0L, 0x1p-1L},
            {0xb333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
        }},
        0x1.76cdcfbba34bbp-8, 0x1.12646088b968ap-9, true, 0x1p-7L, {0, 1, 5, 4, 3, 6}},
    {9, {0x1.8000000000000p-1, 0x0.0p+0, 0x1.0000000000000p-2}, -1,
        {{
            {0x0p+0L, 0x0p+0L, 0x0p+0L},
            {0x3333333333333p-53L, 0x1p-1L, 0x0p+0L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x0p+0L},
            {-0x1p-2L, 0x0p+0L, 0x0p+0L},
            {0x0p+0L, 0x0p+0L, 0x1p-1L},
            {0x3333333333333p-53L, 0x1p-1L, 0x1p-1L},
            {-0x4cccccccccccdp-53L, 0x1p-1L, 0x1p-1L},
            {-0x1p-2L, 0x0p+0L, 0x1p-1L},
        }},
        0x1.2fae9097ed1eep-33, 0x1.ffffff6828b7bp-8, true, 0x1p-7L, {1, 7, 2, 3, 4, 5}},
    {12, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, -1,
        {{
            {0x0p+0L, 0x0p+0L, 0x0p+0L},
            {-0x1p-2L, 0x0p+0L, 0x0p+0L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x0p+0L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x0p+0L},
            {0x0p+0L, 0x0p+0L, 0x1p-1L},
            {-0x1p-2L, 0x0p+0L, 0x1p-1L},
            {-0xb333333333333p-53L, -0x1p-1L, 0x1p-1L},
            {-0x3333333333333p-53L, -0x1p-1L, 0x1p-1L},
        }},
        0x0.0p+0, 0x1.0000000000000p-7, false, 0x1p-7L, {0, 1, 8, 9, 3, 2}},
    {20, {0x1.8000000000000p-1, 0x0.0p+0, 0x0.0p+0}, 0,
        {{
            {0x0p+0L, 0x0p+0L, 0x0p+0L},
            {0x1p-3L, 0x0p+0L, 0x0p+0L},
            {0x16666666666667p-55L, 0x1p-2L, 0x0p+0L},
            {0x3333333333333p-54L, 0x1p-2L, 0x0p+0L},
            {0x0p+0L, 0x0p+0L, 0x1p-1L},
            {0x1p-3L, 0x0p+0L, 0x1p-1L},
            {0x16666666666667p-55L, 0x1p-2L, 0x1p-1L},
            {0x3333333333333p-54L, 0x1p-2L, 0x1p-1L},
        }},
        0x0.0p+0, 0x1.0000000000000p-6, false, 0x20000000000001p-59L, {0, 7, 5, 10, 3, 2}},
}};
// Each compact key retains its exact byte count and zero-padded little-endian storage words.
inline const std::vector<std::vector<std::uint64_t>> support_keys{
    {0x19ULL, 0x210ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0xc0000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x2000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL,
        0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0x19ULL, 0x110ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL, 0x0ULL,
        0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0xc0000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL, 0x200000000ULL, 0x4000000000ULL, 0x8ccccccc00000000ULL,
        0xccccd00000000000ULL, 0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x4100000000ULL, 0x20000000000ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x190000000000ULL, 0x2010000000000ULL, 0x1000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
};
// The ordinary certificate uses portable numeric limbs, never raw ABI padding.
struct CertificateWriter
{
    std::vector<char> bytes;
    void word(std::uint64_t value)
    {
        for (unsigned i = 0; i < 8; ++i)
            bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
    }
    void support(const std::vector<std::uint64_t>& key)
    {
        const auto count = key.front();
        word(count);
        for (std::uint64_t i = 0; i < count; ++i)
            bytes.push_back(static_cast<char>((key[1 + i / 8] >> (8 * (i % 8))) & 255));
    }
    std::vector<char> certificate() const
    {
        using Header = PLIC::GeometryPacketHeader;
        std::vector<char> packet(sizeof(Header), 0);
        packet[offsetof(Header, family)] = static_cast<char>(PLIC::GeometryPacketFamily::Geometry);
        packet[offsetof(Header, version)] = static_cast<char>(PLIC::geometry_packet_version);
        packet[offsetof(Header, kind)] = static_cast<char>(PLIC::GeometryPacketKind::CellCertificate);
        packet[offsetof(Header, mode)] = static_cast<char>(ArithmeticMode::Ordinary);
        const auto profile = PLIC::geometry_packet_profile(ArithmeticMode::Ordinary);
        if (profile == PLIC::GeometryNumericProfile::Unsupported)
            throw std::invalid_argument("Unsupported captured cell numeric profile.");
        packet[offsetof(Header, profile)] = static_cast<char>(profile);
        const auto length = static_cast<std::uint64_t>(bytes.size());
        for (unsigned i = 0; i < 8; ++i)
            packet[offsetof(Header, body_length) + i] = static_cast<char>((length >> (8 * i)) & 255);
        packet.insert(packet.end(), bytes.begin(), bytes.end());
        return packet;
    }
    void real(long double value)
    {
        if (value == 0)
        {
            word(0);
            word(0);
            return;
        }
        int exponent = 0;
        auto fraction = std::frexp(std::abs(value), &exponent);
        word((std::uint64_t(std::uint32_t(exponent)) << 1) | (std::signbit(value) ? 1 : 0));
        word(std::numeric_limits<long double>::digits);
        for (int i = 0; i < std::numeric_limits<long double>::digits; i += 32)
        {
            fraction = std::ldexp(fraction, 32);
            const auto limb = static_cast<std::uint32_t>(fraction);
            word(limb);
            fraction -= limb;
        }
    }
    void point(const WidePoint& point)
    {
        for (const auto coordinate : point)
            real(coordinate);
    }
};
inline std::vector<char> certificate(const Row& row)
{
    CertificateWriter writer;
    writer.bytes.push_back(static_cast<char>(PLIC::GeometryTag::OrdinaryCell));
    writer.point({row.origin[0], row.origin[1], row.origin[2]});
    writer.word(row.stored_vertices.size());
    for (auto point : row.stored_vertices)
    {
        for (auto& coordinate : point)
            coordinate = std::ldexp(coordinate, row.frame_exponent);
        writer.point(point);
    }
    writer.word(faces.size());
    for (std::size_t f = 0; f < faces.size(); ++f)
    {
        writer.word(faces[f].size());
        for (const auto vertex : faces[f])
            writer.word(vertex);
        writer.support(support_keys[row.support_indices[f]]);
    }
    return writer.certificate();
}
inline Sample sample(const Row& row)
{
    return {Cell::decode_certificate(certificate(row)), row.gas, row.carrier, row.id, row.observed_mixed};
}
inline PLIC::PolyhedralReconstructionOptions options()
{
    PLIC::PolyhedralReconstructionOptions result;
    result.measure_only = true;
    result.maximum_offset_iterations = 192;
    result.maximum_fit_iterations = 24;
    result.maximum_stencil_cells = 128;
    result.maximum_clip_evaluations = 32768;
    result.volume_tolerance = 8e-15;
    result.fit_tolerance = 2e-9;
    result.maximum_condition = 1e8;
    result.orientation_tolerance = 2e-8;
    result.maximum_model_residual = .2;
    result.minimum_phase_support_fraction = .5;
    result.minimum_optimized_phase_length_ratio = .05;
    return result;
}

inline constexpr PLIC::GeometryHalfspace retained_plane{
    {0x1.f60a8909cc2b0p-1, 0x1.532aae2b8f91ep-3, 0x1.af7c2690ec529p-4}, {0x1.4000000000000p-1, 0x0.0p+0, 0x0.0p+0},
    0x1.b8be43ed21e5ep-6};
// Independent rational original-face clipping and boundary integration.
inline constexpr std::array<long double, 5> gas_fraction_residuals{
    -2.6607201832947859273411334E-19L, -4.7821259079193925504033459E-6L, 8.5348050365832131570757350E-8L, 0.L, 0.L};
inline constexpr std::array<PLIC::GeometryPoint, 5> centroids{
    {{0x1.6cccccccccccdp-1, 0x1.0000000000000p-3, 0x1.0000000000000p-3},
        {0x1.2cccccccccccdp-1, 0x1.0000000000000p-3, 0x1.0000000000000p-3},
        {0x1.6cccccccccccdp-1, 0x1.0000000000000p-3, 0x1.8000000000000p-2},
        {0x1.8666666666666p-1, 0x1.8000000000000p-2, 0x1.0000000000000p-3},
        {0x1.acccccccccccdp-1, 0x1.0000000000000p-3, 0x1.0000000000000p-2}}};
inline constexpr long double retained_rms = 0.000002391443731770367367348L;
inline constexpr long double target_gas_residual = -2.0786876431990515057352605E-21L;
} // namespace captured_native_cell8
