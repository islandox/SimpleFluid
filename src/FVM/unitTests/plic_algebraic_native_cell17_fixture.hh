/** @file plic_algebraic_native_cell17_fixture.hh
 * @brief Ordinary measure-only input captured at the native cell-17 failure.
 *
 * Source: algebraic_vof_repair_20261029/native-cell17-replay, 2026-10-29.
 * Ordered native local vertices, independent fitting labels and source supports
 * are retained here; no affine-box regeneration or majority complement is used.
 * Each support is the unique raw authority key common to its face vertices.
 * The certificate restores represented source support packets, not historical
 * plane allocations, ancestry, or pointer identities. Reconstruction measure-only
 * queries use the represented shell, not those historical allocations.
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
#include <span>
#include <vector>

namespace captured_native_cell17
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
    std::array<unsigned, 6> support_indices;
};
inline const std::vector<std::vector<std::size_t>> faces{
    {3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
inline constexpr std::array<std::uint64_t, 6> stencil_ids{2, 5, 13, 16, 18, 33};
inline constexpr std::array<Row, 7> rows{{
    {17, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.3333333333334p-3L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.3333333333334p-3L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
        }},
        0x1.f50910b65eda3p-9, 0x1.057b77a4d092ep-8, {0, 1, 2, 3, 4, 5}},
    {2, {0x1.0000000000000p-2, 0x0.0p+0, 0x0.0p+0}, 0,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {0x1.6666666666666p-2L, 0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.9999999999998p-4L, 0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {0x1.6666666666666p-2L, 0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.9999999999998p-4L, 0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
        }},
        0x1.ebdda8b7b8aabp-5, 0x1.4225748475556p-9, {6, 1, 7, 2, 5, 8}},
    {5, {0x1.0000000000000p-1, 0x0.0p+0, 0x1.0000000000000p-2}, -1,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {0x1.6666666666666p-2L, 0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.9999999999998p-4L, 0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {0x1.6666666666666p-2L, 0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.9999999999998p-4L, 0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
        }},
        0x1.fe1992024d16fp-8, 0x1.e66dfdb2e90bcp-16, {0, 1, 7, 9, 3, 2}},
    {13, {0x1.b333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {-0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {-0x1.6666666666666p-2L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {-0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {-0x1.6666666666666p-2L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
        }},
        0x1.81223f68d8a50p-13, 0x1.f3f6ee04b93aep-8, {0, 1, 5, 4, 3, 10}},
    {16, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x0.0p+0}, -1,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.3333333333334p-3L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.3333333333334p-3L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
        }},
        0x1.8a870110c0d6ep-8, 0x1.d5e3fbbcfca47p-10, {6, 0, 2, 3, 4, 5}},
    {18, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1}, -1,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.3333333333334p-3L, -0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {-0x1.9999999999998p-4L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.3333333333334p-3L, -0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
        }},
        0x1.dfa7edd6a1ff5p-9, 0x1.102c0914af006p-8, {1, 11, 2, 3, 4, 5}},
    {33, {0x1.3333333333333p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-2}, -1,
        {{
            {0x0.0p+0L, 0x0.0p+0L, 0x0.0p+0L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x0.0p+0L},
            {0x1.6666666666666p-2L, 0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x1.9999999999998p-4L, 0x1.0000000000000p-1L, 0x0.0p+0L},
            {0x0.0p+0L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {0x1.0000000000000p-2L, 0x0.0p+0L, 0x1.0000000000000p-1L},
            {0x1.6666666666666p-2L, 0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
            {0x1.9999999999998p-4L, 0x1.0000000000000p-1L, 0x1.0000000000000p-1L},
        }},
        0x1.20bcd82748de5p-17, 0x1.ff6fa193ec5b9p-8, {0, 1, 5, 12, 13, 14}},
}};
// Authority-key word zero is the byte count; remaining words are little endian.
inline const std::vector<std::vector<std::uint64_t>> support_keys{
    {0x29ULL, 0x210ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x1fffffffe00ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL, 0x0ULL,
        0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0x8000000000ULL, 0x0ULL, 0x0ULL},
    {0x19ULL, 0x210ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0x19ULL, 0x110ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x4100ULL, 0x1fffffffe0000ULL, 0x400000ULL, 0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x510000ULL, 0x1fffffffe000000ULL, 0x40000000ULL, 0xb3333333000000ULL, 0x33333000000000ULL,
        0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x1afULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x41000000ULL,
        0x0ULL, 0x4000000000ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x4100000000ULL,
        0x0ULL, 0x400000000000ULL, 0x0ULL, 0xc0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x190000000000ULL,
        0x2010000000000ULL, 0x2000000000000ULL, 0x3000000000000ULL, 0x29000000000000ULL, 0x401000000000000ULL,
        0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL, 0x400000000000000ULL,
        0x500000000000000ULL, 0x0ULL},
    {0xbbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0xd99999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x410000ULL, 0x0ULL, 0x40000000ULL, 0xc0000000000000ULL, 0x0ULL,
        0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x1000000ULL, 0x0ULL},
    {0x29ULL, 0x210ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
    {0x1cfULL, 0x615ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xd9999999000000ULL,
        0x99999800000000ULL, 0x0ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x51000000ULL,
        0x0ULL, 0x4000000000ULL, 0xb333333300000000ULL, 0x3333300000000000ULL, 0x200000000ULL, 0x4000000000ULL,
        0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x5100000000ULL, 0x0ULL, 0x400000000000ULL,
        0x3333330000000000ULL, 0x33300000000000f3ULL, 0x20000000033ULL, 0x400000000000ULL, 0x0ULL, 0x80ULL, 0x0ULL,
        0x0ULL, 0x190000000000ULL, 0x2010000000000ULL, 0x0ULL, 0x1000000000000ULL, 0x29000000000000ULL,
        0x401000000000000ULL, 0x0ULL, 0x100000000000000ULL, 0x200000000000000ULL, 0x300000000000000ULL,
        0x400000000000000ULL, 0x500000000000000ULL, 0x0ULL},
    {0x29ULL, 0x110ULL, 0x0ULL, 0x4000ULL, 0xc000000000ULL, 0x0ULL, 0x0ULL},
    {0xcbULL, 0x215ULL, 0x5100ULL, 0x0ULL, 0x400000ULL, 0x999999990000ULL, 0x999998000000ULL, 0x0ULL, 0x400000ULL,
        0x800000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x510000ULL, 0x0ULL, 0x40000000ULL, 0xb3333333000000ULL,
        0x33333000000000ULL, 0x2000000ULL, 0x40000000ULL, 0x80000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
        0x1000000ULL, 0x0ULL},
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
    return {Cell::decode_certificate(certificate(row)), row.gas, row.carrier, row.id, true};
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
    {0x1.e39e9c9186ae1p-1, 0x1.45658157e8357p-2, 0x1.5207c9abceff4p-4},
    {0x1.199999999999ap-1, 0x1.0000000000000p-2, 0x1.0000000000000p-2}, 0x1.0cec25ef09d79p-3};
// Independent rational positive-part cubic CDF on the dyadic parallelepipeds,
// evaluated at retained_plane. These are signed gas residuals divided by bulk
// volume; the production objective flips signs for carrier-minority rows.
inline constexpr std::array<long double, 7> gas_fraction_residuals{9.46660129302973653002511e-17L,
    .0390483247127528667055771L, .00349290206512296826282475L, -.0222535054814595031027687L,
    -.127219056256861473459916L, -.130492474492415133363223L, -.00104753151098557712940760L};
inline constexpr long double retained_rms = .0766444845050064915412804L;
inline constexpr long double target_gas_residual = 7.39578226017948166408211e-19L;
} // namespace captured_native_cell17
