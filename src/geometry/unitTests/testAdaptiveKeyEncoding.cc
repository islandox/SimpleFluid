/** @file testAdaptiveKeyEncoding.cc
 * @brief The allocation-free canonical key length equals the emitted packet.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <gtest/gtest.h>
#include <limits>
#include <vector>
namespace
{
using namespace SimpleFluid::Meshes::AMR;
std::vector<std::uint64_t> word_path(std::size_t size, std::uint64_t value)
{
    return std::vector<std::uint64_t>(size, value);
}
TEST(AdaptiveKeyEncoding, CountedLengthMatchesEmittedPacketForEveryVariant)
{
    constexpr auto max_word = std::numeric_limits<std::uint64_t>::max();
    struct Case
    {
        AdaptiveCellKey key;
        std::size_t counted;
    };
    // Canonical 3-word header + region + variant tag, then the variant payload:
    // Cartesian level and three indices; Convex root and path; Product base and
    // axial root/path; Static provider root and native cell.
    const std::vector<Case> cases{
        {{0, CartesianCellAncestry{0, {0, 0, 0}}}, 72},
        {{7, CartesianCellAncestry{3, {1, 2, 3}}}, 72},
        {{0, ConvexCellAncestry{0, {}}}, 56},
        {{9, ConvexCellAncestry{11, {12, 13, 14}}}, 56 + 3 * 8},
        {{0, ProductCellAncestry{{0, 0, {}}, {0, {}}}}, 80},
        {{5, ProductCellAncestry{{6, 7, {8, 9}}, {10, {11}}}}, 80 + 3 * 8},
        {{0, StaticCellAncestry{0, 0}}, 56},
        {{max_word, StaticCellAncestry{max_word - 1, max_word - 2}}, 56},
    };
    for (const auto& test : cases)
    {
        const auto encoded = encode_adaptive_key(test.key);
        EXPECT_EQ(encoded_adaptive_key_size(test.key), test.counted);
        EXPECT_EQ(encoded_adaptive_key_size(test.key), encoded.size());
        EXPECT_EQ(decode_adaptive_key(encoded), test.key);
    }
}
TEST(AdaptiveKeyEncoding, CountedLengthHandlesZeroLongAndMaximumWidthWords)
{
    constexpr auto max_word = std::numeric_limits<std::uint64_t>::max();
    const std::vector<AdaptiveCellKey> keys = {
        {max_word, ConvexCellAncestry{max_word, {}}},
        {max_word, ConvexCellAncestry{max_word, word_path(129, max_word)}},
        {max_word, ProductCellAncestry{{max_word, max_word, {}}, {max_word, word_path(130, max_word)}}},
        {max_word, ProductCellAncestry{{max_word, max_word, word_path(129, max_word)}, {max_word, {}}}},
        {max_word, CartesianCellAncestry{std::numeric_limits<unsigned>::max(), {max_word, max_word - 1, max_word - 2}}},
        {max_word, StaticCellAncestry{max_word, max_word - 1}},
    };
    for (const auto& key : keys)
    {
        const auto encoded = encode_adaptive_key(key);
        EXPECT_EQ(encoded_adaptive_key_size(key), encoded.size());
        EXPECT_EQ(decode_adaptive_key(encoded), key);
    }
}
} // namespace
