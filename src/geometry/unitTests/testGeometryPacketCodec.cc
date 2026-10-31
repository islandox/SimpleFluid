#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/PolyhedralComplex.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto exact = ArithmeticMode::Exact;
using Header = GeometryPacketHeader;
void overwrite(std::vector<char>& bytes, std::size_t offset, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.at(offset + i) = static_cast<char>(value >> (8 * i));
}
void word(std::vector<char>& bytes, std::uint64_t value)
{
    const auto offset = bytes.size();
    bytes.resize(offset + 8);
    overwrite(bytes, offset, value);
}
void tag(std::vector<char>& bytes, GeometryTag value)
{
    bytes.push_back(static_cast<char>(value));
}
void integer(std::vector<char>& bytes, unsigned value, unsigned sign = 0)
{
    word(bytes, sign);
    word(bytes, 1);
    bytes.push_back(static_cast<char>(value));
}
TEST(GeometryPacketCodecTest, CategoryTagsUseOneByteWhileNumericWordsKeepTheirWidth)
{
    static_assert(sizeof(GeometryTag) == 1 && sizeof(GeometryPacketKind) == 1);
    static_assert(sizeof(SupportRecipeKind) == 1 && sizeof(AuthorityKeyTag) == 1);
    static_assert(sizeof(GeometryPacketFamily) == 1 && sizeof(GeometryNumericProfile) == 1);
    details::ordinary_native::Writer writer;
    writer.tag(GeometryTag::OrdinaryPoint);
    writer.word(123);
    EXPECT_EQ(writer.bytes.size(), 9u);
    writer.real(0);
    EXPECT_EQ(writer.bytes.size(), 25u); // Canonical numeric zero remains two uint64 words.
    const auto point = details::GeometryIdentityAccess::point({0, 0, 0});
    const auto key = canonical_key(point), packet = encode_identity(point);
    EXPECT_EQ(key.size(), 1 + 3 * 16);
    EXPECT_EQ(static_cast<std::uint8_t>(key.front()), static_cast<std::uint8_t>(GeometryTag::OrdinaryPoint));
    EXPECT_EQ(geometry_packet_header_bytes, sizeof(Header));
    EXPECT_EQ(packet.size(), sizeof(Header) + key.size());
    EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(Header, family)]),
        static_cast<std::uint8_t>(GeometryPacketFamily::Geometry));
    EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(Header, version)]), 3u);
    EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(Header, kind)]),
        static_cast<std::uint8_t>(GeometryPacketKind::PointIdentity));
    EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(Header, mode)]), static_cast<std::uint8_t>(ordinary));
    EXPECT_EQ(static_cast<std::uint8_t>(packet[offsetof(Header, profile)]),
        static_cast<std::uint8_t>(geometry_packet_profile(ordinary)));
}
TEST(GeometryPacketCodecTest, IdentityHeaderRejectsForeignAndUnboundedRecords)
{
    const std::array<GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto point = PlanarFace<ordinary>::make(points)->vertex_identities().front();
    const auto original = encode_identity(point);
    const auto rejects = [&](const std::vector<char>& bytes)
    { EXPECT_THROW((decode_identity<PointIdentity<ordinary>>(bytes)), std::invalid_argument); };
    rejects({});
    rejects(canonical_key(point));
    for (const auto offset : {offsetof(Header, family), offsetof(Header, version), offsetof(Header, kind),
             offsetof(Header, mode), offsetof(Header, profile)})
    {
        auto bytes = original;
        bytes[offset] = char(0xff);
        rejects(bytes);
    }
    for (std::size_t count = 0; count < original.size(); ++count)
        rejects(std::vector<char>(original.begin(), original.begin() + count));
    auto bytes = original;
    bytes.push_back(0);
    rejects(bytes);
    bytes = original;
    bytes[offsetof(Header, version)] = 2;
    rejects(bytes);
    bytes = original;
    overwrite(bytes, offsetof(Header, body_length), std::numeric_limits<std::uint64_t>::max());
    rejects(bytes);
    EXPECT_THROW((decode_identity<PointIdentity<ordinary>>(original, original.size() - 1)), std::invalid_argument);
    EXPECT_EQ(decode_identity<PointIdentity<ordinary>>(original, original.size()), point);
}
TEST(GeometryPacketCodecTest, NumericProfilesAreExplicitAndUnknownFormatsDoNotAlias)
{
    using Profile = GeometryNumericProfile;
    EXPECT_EQ(geometry_numeric_profile(2, 53, -1021, 1024), Profile::Binary64);
    EXPECT_EQ(geometry_numeric_profile(2, 64, -16381, 16384), Profile::X87Binary80);
    EXPECT_EQ(geometry_numeric_profile(2, 113, -16381, 16384), Profile::Binary128);
    for (const auto format : {std::array{2, 53, -1022, 1024}, std::array{2, 64, -16381, 16383},
             std::array{2, 106, -968, 1024}, std::array{10, 34, -6142, 6145}})
        EXPECT_EQ(geometry_numeric_profile(format[0], format[1], format[2], format[3]), Profile::Unsupported);
    EXPECT_EQ(geometry_packet_profile(exact), Profile::ExactRational);
    EXPECT_EQ(geometry_packet_profile(ArithmeticMode::BoundedCertification), Profile::Unsupported);
    for (const auto mode : {ordinary, exact})
    {
        const auto packet = encode_geometry_packet(GeometryPacketKind::DisplacementIdentity, mode, {});
        for (const auto foreign : {Profile::ExactRational, Profile::Binary64, Profile::X87Binary80, Profile::Binary128,
                 static_cast<Profile>(4), Profile::Unsupported})
        {
            if (foreign == geometry_packet_profile(mode))
                continue;
            auto corrupt = packet;
            corrupt[offsetof(Header, profile)] = static_cast<char>(foreign);
            EXPECT_THROW(decode_geometry_packet(corrupt, mode), std::invalid_argument);
        }
    }
}
TEST(GeometryPacketCodecTest, NaturalPaddingIsSerializedDeterministicallyAndUnalignedReadsAreSafe)
{
    static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header>);
    const std::vector<char> body(257, 'x'); // Opaque framing payload; numerical bodies have their own validators.
    const auto packet = encode_geometry_packet(GeometryPacketKind::PlanarFace, ordinary, body);
    EXPECT_EQ(packet.size(), sizeof(Header) + body.size());
    EXPECT_EQ(encode_geometry_packet(GeometryPacketKind::PlanarFace, ordinary, body), packet);
    EXPECT_EQ(offsetof(Header, body_length) % alignof(std::uint64_t), 0u);
    std::vector<char> expected(sizeof(Header), 0);
    expected[offsetof(Header, family)] = static_cast<char>(GeometryPacketFamily::Geometry);
    expected[offsetof(Header, version)] = 3;
    expected[offsetof(Header, kind)] = static_cast<char>(GeometryPacketKind::PlanarFace);
    expected[offsetof(Header, mode)] = static_cast<char>(ordinary);
    expected[offsetof(Header, profile)] = static_cast<char>(geometry_packet_profile(ordinary));
    overwrite(expected, offsetof(Header, body_length), 257);
    EXPECT_EQ(std::vector<char>(packet.begin(), packet.begin() + sizeof(Header)), expected);
    std::array<bool, sizeof(Header)> fields{};
    for (const auto offset : {offsetof(Header, family), offsetof(Header, version), offsetof(Header, kind),
             offsetof(Header, mode), offsetof(Header, profile)})
        fields[offset] = true;
    for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i)
        fields[offsetof(Header, body_length) + i] = true;
    for (std::size_t i = 0; i < fields.size(); ++i)
        if (!fields[i])
        {
            EXPECT_EQ(packet[i], 0);
            auto corrupt = packet;
            corrupt[i] = 1;
            EXPECT_THROW(decode_geometry_packet(corrupt, ordinary), std::invalid_argument);
        }
    for (std::size_t offset = 1; offset <= alignof(Header) + 1; ++offset)
    {
        std::vector<char> enclosing(offset + packet.size() + 1, 'z');
        std::copy(packet.begin(), packet.end(), enclosing.begin() + offset);
        const auto input = std::span<const char>(enclosing).subspan(offset, packet.size());
        const auto decoded = decode_geometry_packet(input, ordinary, packet.size());
        EXPECT_EQ(decoded.kind, GeometryPacketKind::PlanarFace);
        EXPECT_EQ(decoded.body.data(), input.data() + sizeof(Header));
        EXPECT_EQ(std::vector<char>(decoded.body.begin(), decoded.body.end()), body);
        EXPECT_THROW(decode_geometry_packet(input, ordinary, packet.size() - 1), std::invalid_argument);
    }
    EXPECT_EQ(encode_geometry_packet(GeometryPacketKind::PlanarFace, ordinary, body, packet.size()), packet);
    EXPECT_THROW(encode_geometry_packet(GeometryPacketKind::PlanarFace, ordinary, body, packet.size() - 1),
        std::invalid_argument);
}
TEST(GeometryPacketCodecTest, OldCompactEnvelopesRejectEvenWithoutBodies)
{
    for (const auto mode : {ordinary, exact})
    {
        std::vector<char> previous;
        word(previous, 0x534647454f4d5631ULL); // Actual preceding 27-byte v2 envelope.
        previous.push_back(2);
        previous.push_back(static_cast<char>(GeometryPacketKind::DisplacementIdentity));
        previous.push_back(static_cast<char>(mode));
        using Numeric = std::numeric_limits<long double>;
        const auto profile = mode == exact ? 0
                                           : std::uint64_t(Numeric::radix) | (std::uint64_t(Numeric::digits) << 8) |
                                                 (std::uint64_t(std::uint16_t(Numeric::min_exponent)) << 16) |
                                                 (std::uint64_t(std::uint16_t(Numeric::max_exponent)) << 32);
        word(previous, profile);
        word(previous, 0);
        ASSERT_EQ(previous.size(), 27u);
        EXPECT_THROW(decode_geometry_packet(previous, mode), std::invalid_argument);
    }
}
TEST(GeometryPacketCodecTest, CanonicalScalarsAndRationalsAreRequired)
{
    const std::array<GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto zero_point = PlanarFace<ordinary>::make(points)->vertex_identities().front();
    auto finite = encode_identity(zero_point);
    overwrite(
        finite, geometry_packet_header_bytes + 1, 1); // Signed/nonzero descriptor on the finite zero representation.
    EXPECT_THROW((decode_identity<PointIdentity<ordinary>>(finite)), std::invalid_argument);
    std::vector<char> body;
    tag(body, GeometryTag::ExactPoint);
    integer(body, 2);
    integer(body, 4); // Same value as 1/2, different rational encoding.
    integer(body, 0);
    integer(body, 1);
    integer(body, 0);
    integer(body, 1);
    auto rational = encode_geometry_packet(GeometryPacketKind::PointIdentity, exact, body);
    EXPECT_THROW((decode_identity<PointIdentity<exact>>(rational)), std::invalid_argument);
    body.clear();
    tag(body, GeometryTag::ExactPoint);
    integer(body, 0, 1);
    integer(body, 1); // Negative zero integer.
    integer(body, 0);
    integer(body, 1);
    integer(body, 0);
    integer(body, 1);
    rational = encode_geometry_packet(GeometryPacketKind::PointIdentity, exact, body);
    EXPECT_THROW((decode_identity<PointIdentity<exact>>(rational)), std::invalid_argument);
    body.clear();
    tag(body, GeometryTag::ExactPoint);
    word(body, 0);
    word(body, std::numeric_limits<std::uint64_t>::max());
    rational = encode_geometry_packet(GeometryPacketKind::PointIdentity, exact, body);
    EXPECT_THROW((decode_identity<PointIdentity<exact>>(rational)), std::invalid_argument);
}
TEST(GeometryPacketCodecTest, ZeroIsAnEnvelopeWithAnEmptyBody)
{
    const DisplacementIdentity<ordinary> zero;
    const auto packet = encode_identity(zero);
    ASSERT_EQ(packet.size(), geometry_packet_header_bytes);
    EXPECT_TRUE(decode_geometry_packet(packet, ordinary).body.empty());
    EXPECT_TRUE(decode_identity<DisplacementIdentity<ordinary>>(packet).is_zero());
    std::vector<char> body;
    tag(body, GeometryTag::OrdinaryDisplacement);
    for (unsigned i = 0; i < 6; ++i)
        word(body, 0);
    const auto tagged = encode_geometry_packet(GeometryPacketKind::DisplacementIdentity, ordinary, body);
    EXPECT_THROW((decode_identity<DisplacementIdentity<ordinary>>(tagged)), std::invalid_argument);
    body.clear();
    tag(body, GeometryTag::ExactDisplacement);
    for (unsigned i = 0; i < 3; ++i)
    {
        integer(body, 0);
        integer(body, 1);
    }
    EXPECT_THROW((decode_identity<DisplacementIdentity<exact>>(
                     encode_geometry_packet(GeometryPacketKind::DisplacementIdentity, exact, body))),
        std::invalid_argument);
}
template<ArithmeticMode Mode> void check_geometry_records()
{
    const std::array<GeometryPoint, 5> points{{{0, 0, 0}, {.5, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    const auto face = PlanarFace<Mode>::make(points);
    const auto canonical = face->encode(), incidence = face->encode_incidence();
    EXPECT_EQ(PlanarFace<Mode>::decode(incidence)->vertex_identities().size(), 5u);
    EXPECT_EQ(PlanarFace<Mode>::decode(canonical)->vertex_identities().size(), 4u);
    EXPECT_THROW((PlanarFace<Mode>::decode(canonical_face_key(*face))), std::invalid_argument);
    auto body = decode_geometry_packet(canonical, Mode).body;
    EXPECT_EQ(std::vector<char>(body.begin(), body.end()), canonical_face_key(*face));
    const auto cell = ConvexCellGeometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    const auto certificate = cell->encode_certificate();
    EXPECT_THROW((PlanarFace<Mode>::decode(canonical, canonical.size() - 1)), std::invalid_argument);
    EXPECT_THROW((face->encode(canonical.size() - 1)), std::invalid_argument);
    EXPECT_THROW(
        (ConvexCellGeometry<Mode>::decode_certificate(certificate, certificate.size() - 1)), std::invalid_argument);
    EXPECT_THROW((cell->encode_certificate(certificate.size() - 1)), std::invalid_argument);
    EXPECT_EQ(
        canonical_cell_key(*ConvexCellGeometry<Mode>::decode_certificate(certificate)), canonical_cell_key(*cell));
    EXPECT_THROW((ConvexCellGeometry<Mode>::decode_certificate(canonical_cell_key(*cell))), std::invalid_argument);
    const auto wrong = encode_geometry_packet(GeometryPacketKind::PlanarFace, Mode, canonical_cell_key(*cell));
    EXPECT_THROW((ConvexCellGeometry<Mode>::decode_certificate(wrong)), std::invalid_argument);
}
TEST(GeometryPacketCodecTest, GeometryKindsPreserveIncidenceAndFrozenKeys)
{
    check_geometry_records<ordinary>();
    check_geometry_records<exact>();
}
TEST(GeometryPacketCodecTest, RoundedFiniteLineKeysRemainValid)
{
    const std::array<GeometryPoint, 3> points{{{.1, -.2, 0}, {.3, .7, 0}, {-1, 2, 0}}};
    const auto face = PlanarFace<ordinary>::make(points);
    for (const auto& line : face->edge_support_identities())
        EXPECT_EQ(decode_identity<LineIdentity<ordinary>>(encode_identity(line)), line);
}
TEST(GeometryPacketCodecTest, OldEnclosingComplexVersionsRejectBeforeOptionalRecords)
{
    std::vector<char> ordinary_packet;
    tag(ordinary_packet, GeometryTag::OrdinaryComplex);
    ordinary_packet.push_back(3); // Immediately preceding compact schema, no child geometry needed.
    std::vector<char> exact_packet;
    tag(exact_packet, GeometryTag::ExactComplex);
    exact_packet.push_back(3);
    // This was a valid sparse empty local catalog under the previous version:
    // nonempty global catalogs, with zero resident cell/face/node records.
    for (unsigned i = 0; i < 3; ++i)
    {
        word(ordinary_packet, 1);
        word(exact_packet, 1);
    }
    for (unsigned i = 0; i < 3; ++i)
    {
        word(ordinary_packet, 0);
        word(exact_packet, 0);
    }
    EXPECT_THROW((PolyhedralComplex<ordinary>::decode(ordinary_packet)), std::invalid_argument);
    EXPECT_THROW((PolyhedralComplex<exact>::decode(exact_packet)), std::invalid_argument);
}

std::vector<char> exact_tuple(GeometryTag category, std::span<const int> values)
{
    std::vector<char> body;
    tag(body, category);
    for (const auto value : values)
    {
        integer(body, value < 0 ? unsigned(-value) : unsigned(value), value < 0);
        integer(body, 1);
    }
    return body;
}
TEST(GeometryPacketCodecTest, ExactProviderDirectionAndMomentNormalFormsAreRequired)
{
    const auto line = [](std::array<int, 6> values)
    {
        return encode_geometry_packet(
            GeometryPacketKind::LineIdentity, exact, exact_tuple(GeometryTag::ExactLine, values));
    };
    EXPECT_NO_THROW((decode_identity<LineIdentity<exact>>(line({1, 0, 0, 0, 2, 3}))));
    EXPECT_THROW((decode_identity<LineIdentity<exact>>(line({0, 0, 0, 0, 0, 0}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<LineIdentity<exact>>(line({2, 0, 0, 0, 2, 3}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<LineIdentity<exact>>(line({-1, 0, 0, 0, 2, 3}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<LineIdentity<exact>>(line({1, 0, 0, 1, 2, 3}))), std::invalid_argument);
    const auto support = [](std::array<int, 4> values)
    {
        return encode_geometry_packet(
            GeometryPacketKind::PlaneSupportIdentity, exact, exact_tuple(GeometryTag::ExactPlaneSupport, values));
    };
    EXPECT_NO_THROW((decode_identity<PlaneSupportIdentity<exact>>(support({0, 1, 2, 3}))));
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<exact>>(support({0, 0, 0, 0}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<exact>>(support({0, 2, 0, 3}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<exact>>(support({0, -1, 0, 3}))), std::invalid_argument);
}
TEST(GeometryPacketCodecTest, FiniteProviderFormsPreserveRoundedCoefficients)
{
    namespace on = details::ordinary_native;
    const auto line = [](on::WidePoint direction)
    {
        on::Writer writer;
        writer.tag(GeometryTag::OrdinaryLine);
        writer.word(0);
        writer.point(direction);
        writer.point({.1L, .2L, 0});
        return encode_geometry_packet(GeometryPacketKind::LineIdentity, ordinary, writer.bytes);
    };
    // The selected component is positive but not exactly one or the largest.
    // Neither reciprocal-rounding normalization nor axis selection is replayed.
    EXPECT_NO_THROW((decode_identity<LineIdentity<ordinary>>(line({std::nextafter(1.L, 0.L), 1, 0}))));
    EXPECT_THROW((decode_identity<LineIdentity<ordinary>>(line({0, 1, 0}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<LineIdentity<ordinary>>(line({-1, 0, 0}))), std::invalid_argument);
    const auto input = [](on::WidePoint normal)
    {
        on::Writer writer;
        writer.tag(GeometryTag::OrdinaryInputSupport);
        writer.point(normal);
        writer.point({});
        writer.real(0);
        return encode_geometry_packet(GeometryPacketKind::PlaneSupportIdentity, ordinary, writer.bytes);
    };
    EXPECT_NO_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(input({0, 2, -1}))));
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(input({}))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(input({0, -1, 2}))), std::invalid_argument);
    on::Writer parent;
    parent.tag(GeometryTag::OrdinaryAxisSupport);
    parent.word(0);
    parent.real(0);
    const auto translated = [&](on::WidePoint shift)
    {
        on::Writer writer;
        writer.tag(GeometryTag::OrdinaryTranslatedSupport);
        writer.packet(parent.bytes);
        writer.point(shift);
        return encode_geometry_packet(GeometryPacketKind::PlaneSupportIdentity, ordinary, writer.bytes);
    };
    EXPECT_NO_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(translated({1, 0, 0}))));
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(translated({}))), std::invalid_argument);
    const auto source = [](bool sorted)
    {
        on::Writer writer;
        writer.tag(GeometryTag::OrdinarySourceSupport);
        writer.word(3);
        const std::array<on::WidePoint, 3> points{{{-1, 0, 1}, {0, 1, 0}, {1, 0, 0}}};
        if (sorted)
            for (const auto& point : points)
                writer.point(point);
        else
        {
            writer.point(points[2]);
            writer.point(points[0]);
            writer.point(points[1]);
        }
        return encode_geometry_packet(GeometryPacketKind::PlaneSupportIdentity, ordinary, writer.bytes);
    };
    EXPECT_NO_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(source(true))));
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(source(false))), std::invalid_argument);
}
TEST(GeometryPacketCodecTest, ExtrusionDagParentsAndEndpointsKeepTheirCanonicalOrdering)
{
    namespace on = details::ordinary_native;
    const auto dag = [](bool sorted_parents, bool sorted_endpoints)
    {
        on::Writer first;
        first.byte(static_cast<std::uint8_t>(SupportRecipeKind::Original));
        first.point({0, 0, 0});
        on::Writer second;
        second.byte(static_cast<std::uint8_t>(SupportRecipeKind::Original));
        second.point({1, 1, 0});
        on::Writer derived;
        derived.byte(static_cast<std::uint8_t>(SupportRecipeKind::Derived));
        derived.word(2);
        derived.word(sorted_parents ? 0 : 1);
        derived.word(sorted_parents ? 1 : 0);
        on::Writer writer;
        writer.tag(GeometryTag::OrdinaryExtrusionDagSupport);
        writer.word(3);
        writer.packet(first.bytes);
        writer.packet(second.bytes);
        writer.packet(derived.bytes);
        writer.word(sorted_endpoints ? 0 : 2);
        writer.word(sorted_endpoints ? 2 : 0);
        return encode_geometry_packet(GeometryPacketKind::PlaneSupportIdentity, ordinary, writer.bytes);
    };
    EXPECT_NO_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(dag(true, true))));
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(dag(false, true))), std::invalid_argument);
    EXPECT_THROW((decode_identity<PlaneSupportIdentity<ordinary>>(dag(true, false))), std::invalid_argument);
}

TEST(GeometryPacketCodecTest, CoordinateCountsRejectBeforeDisproportionateReservations)
{
    namespace on = details::ordinary_native;
    const auto rejects_coordinates = [](auto&& decode)
    {
        try
        {
            decode();
            FAIL() << "Truncated coordinate count was accepted.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("encoded coordinates"), std::string::npos);
        }
    };
    const std::array<GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto face = PlanarFace<ordinary>::make(points);
    auto body = canonical_face_key(*face);
    on::Reader reader{body};
    (void) reader.tag();
    (void) reader.packet();
    (void) reader.point();
    const auto count_offset = reader.cursor;
    body.resize(count_offset + 8 + 4096, 0);
    overwrite(body, count_offset, 4096);
    const auto malformed_face = encode_geometry_packet(GeometryPacketKind::PlanarFace, ordinary, body);
    rejects_coordinates([&] { (void) PlanarFace<ordinary>::decode(malformed_face); });
    const auto cell = ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    body = canonical_cell_key(*cell);
    on::Reader cell_reader{body};
    (void) cell_reader.tag();
    (void) cell_reader.point();
    const auto cell_count_offset = cell_reader.cursor;
    body.resize(cell_count_offset + 8 + 4096, 0);
    overwrite(body, cell_count_offset, 4096);
    const auto malformed_cell = encode_geometry_packet(GeometryPacketKind::CellCertificate, ordinary, body);
    rejects_coordinates([&] { (void) ConvexCellGeometry<ordinary>::decode_certificate(malformed_cell); });
    body.clear();
    tag(body, GeometryTag::ExactCell);
    body.push_back(1);
    for (unsigned i = 0; i < 3; ++i)
        word(body, 0); // Binary64 origin.
    word(body, 512);
    body.resize(body.size() + 4096, 0);
    const auto malformed_fragment = encode_geometry_packet(GeometryPacketKind::ConvexFragment, exact, body);
    rejects_coordinates([&] { (void) geometry::swept_volume::ConvexFragment<exact>::decode(malformed_fragment); });
}
} // namespace
