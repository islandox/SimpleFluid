/** @file testVoFWireProtocol.cc
 * @brief Independent byte framing and full-width payload checks for VoF protocols.
 */
#include "geometry/VoFWireProtocol.hh"
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

namespace
{
using std::uint8_t;
using namespace SimpleFluid::packet_codec;
enum class TestKind : uint8_t
{
    First = 7,
    Second = 19
};
using H = Header<TestKind>;
struct Writer
{
    std::vector<char> data;
    void byte(uint8_t v) { data.push_back(static_cast<char>(v)); }
    void word(std::uint64_t v)
    {
        for (unsigned i = 0; i < 8; ++i)
            byte(static_cast<uint8_t>(v >> (8 * i)));
    }
};
struct Reader
{
    std::span<const char> data;
    std::size_t position = 0;
    uint8_t byte()
    {
        if (position == data.size())
            throw std::invalid_argument("Short test packet.");
        return static_cast<uint8_t>(data[position++]);
    }
    std::uint64_t word()
    {
        std::uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i)
            v |= std::uint64_t(byte()) << (8 * i);
        return v;
    }
};
static_assert(sizeof(Family) == 1 && sizeof(TestKind) == 1);
static_assert(!std::is_convertible_v<uint8_t, TestKind>);
static_assert(!std::is_convertible_v<TestKind, Record>);

TEST(VoFWireProtocolTest, NativeHeaderHasDeterministicBytesAndPreservesWidePayload)
{
    Writer writer;
    write_header(writer, Family::CapillaryPlanes, 5, TestKind::Second);
    std::vector<char> expected(sizeof(H), 0);
    expected[offsetof(H, family)] = 6;
    expected[offsetof(H, version)] = 5;
    expected[offsetof(H, kind)] = 19;
    EXPECT_EQ(writer.data, expected);
    Writer repeated;
    write_header(repeated, Family::CapillaryPlanes, 5, TestKind::Second);
    EXPECT_EQ(repeated.data, expected);
    constexpr std::array<std::uint64_t, 5> payload{
        256, 65537, std::uint64_t(1) << 40, std::uint64_t(1) << 63, std::numeric_limits<std::uint64_t>::max()};
    for (auto v : payload)
        writer.word(v);
    for (std::size_t offset = 0; offset <= alignof(std::uint64_t); ++offset)
    {
        std::vector<char> storage(offset, 'x');
        storage.insert(storage.end(), writer.data.begin(), writer.data.end());
        Reader reader{std::span<const char>(storage).subspan(offset)};
        EXPECT_EQ(
            read_header(reader, Family::CapillaryPlanes, 5, {TestKind::First, TestKind::Second}), TestKind::Second);
        EXPECT_EQ(reader.position, sizeof(H));
        for (auto v : payload)
            EXPECT_EQ(reader.word(), v);
        EXPECT_EQ(reader.position, reader.data.size());
    }
}
TEST(VoFWireProtocolTest, RejectsForeignMetadataTruncationAndNonzeroPadding)
{
    Writer writer;
    write_header(writer, Family::CapillaryPlanes, 5, TestKind::First);
    const auto reject = [](std::span<const char> bytes)
    {
        Reader reader{bytes};
        EXPECT_THROW(read_header(reader, Family::CapillaryPlanes, 5, TestKind::First), std::invalid_argument);
    };
    for (std::size_t length = 0; length < sizeof(H); ++length)
        reject(std::span<const char>(writer.data).first(length));
    for (std::size_t i = 0; i < sizeof(H); ++i)
    {
        auto corrupt = writer.data;
        corrupt[i] = char(255);
        reject(corrupt); // Every field or compiler-inserted padding byte is checked.
    }
    Reader cross_kind{writer.data};
    EXPECT_THROW(read_header(cross_kind, Family::CapillaryPlanes, 5, TestKind::Second), std::invalid_argument);
    Reader no_kind{writer.data};
    EXPECT_THROW(
        read_header(no_kind, Family::CapillaryPlanes, 5, std::initializer_list<TestKind>{}), std::invalid_argument);
}
TEST(VoFWireProtocolTest, PreviousMagicFramesRejectBeforeTheirPayload)
{
    // Independent previous protocol prefixes, not produced by the new writer.
    for (const auto& [family, magic, version] : {std::tuple{Family::CapillaryPlanes, 0x5346434150504c04ULL, 5},
             std::tuple{Family::CapillaryNodeFans, 0x5346434e46414e01ULL, 1},
             std::tuple{Family::CapillaryNodeFans, 0x5346434e46414e02ULL, 1},
             std::tuple{Family::AggregateBudgets, 0x5346414747433031ULL, 1},
             std::tuple{Family::AggregateBudgets, 0x5346414747433032ULL, 1},
             std::tuple{Family::OrdinaryCurvature, 0x5346435552560001ULL, 2},
             std::tuple{Family::ExactPhaseWorking, 0x5346455054523731ULL, 7},
             std::tuple{Family::ExactPhaseSource, 0x5346465054523731ULL, 7},
             std::tuple{Family::ExperimentalPhaseInventory, 0x5346455850493731ULL, 2}})
    {
        Writer old;
        old.word(magic);
        old.word(0); // Empty body or old context/version word cannot make it current.
        Reader reader{old.data};
        EXPECT_THROW(
            read_header(reader, family, static_cast<uint8_t>(version), {Record::Value, Record::Request, Record::Reply}),
            std::invalid_argument);
    }
}
} // namespace
