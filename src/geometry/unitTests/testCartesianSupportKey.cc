#include "geometry/mesh/AMR/CartesianSupportKey.hh"
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace
{
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
std::uint64_t word_at(std::span<const char> bytes, std::size_t index)
{
    std::uint64_t value = 0;
    for (unsigned i = 0; i != 8; ++i)
        value |= std::uint64_t(static_cast<unsigned char>(bytes[index + i])) << (8 * i);
    return value;
}
D::CartesianSupportKey make(std::span<const char> domain, std::uint64_t axis = 1)
{
    D::CartesianSupportKey key;
    key.domain = domain;
    key.region = 7;
    key.axis = axis;
    key.level = 2;
    key.plane = 3;
    key.transverse = {4, 9, 5};
    return key;
}
void set_word(std::vector<char>& bytes, std::size_t offset, std::uint64_t value)
{
    for (unsigned i = 0; i != 8; ++i)
        bytes[offset + i] = char(static_cast<unsigned char>(value >> (8 * i)));
}
} // namespace

TEST(CartesianSupportKeyTest, EncodedFrameMatchesCanonicalWordOrder)
{
    const std::string domain = "s";
    const auto encoded = D::encode_cartesian_support_key(make(domain));
    ASSERT_EQ(encoded.size(), D::cartesian_support_key_size(domain));
    ASSERT_EQ(encoded.size(), 65u);
    EXPECT_EQ(word_at(encoded, 0), D::cartesian_support_key_tag);
    EXPECT_EQ(word_at(encoded, 8), 1u);  // framed domain size
    EXPECT_EQ(encoded[16], 's');         // borrowed domain bytes
    EXPECT_EQ(word_at(encoded, 17), 7u); // region
    EXPECT_EQ(word_at(encoded, 25), 1u); // axis
    EXPECT_EQ(word_at(encoded, 33), 2u); // level
    EXPECT_EQ(word_at(encoded, 41), 3u); // plane
    EXPECT_EQ(word_at(encoded, 49), 4u); // transverse[0]
    EXPECT_EQ(word_at(encoded, 57), 5u); // transverse[2], axis skipped
}

TEST(CartesianSupportKeyTest, RoundTripPreservesTypedFieldsForEveryAxis)
{
    const std::string short_domain = "s";
    const std::string long_domain = "reacting-flow";
    for (unsigned axis = 0; axis != 3; ++axis)
        for (const auto& domain : {short_domain, long_domain})
        {
            const auto key = make(domain, axis);
            const auto encoded = D::encode_cartesian_support_key(key);
            const auto decoded = D::decode_cartesian_support_key(encoded);
            ASSERT_TRUE(decoded.has_value());
            EXPECT_TRUE(*decoded == key);
            EXPECT_TRUE(D::cartesian_support_key_matches(key, encoded));
        }
}

TEST(CartesianSupportKeyTest, RejectsMalformedOrForeignFrames)
{
    const std::string domain = "s";
    const auto encoded = D::encode_cartesian_support_key(make(domain));
    EXPECT_FALSE(D::decode_cartesian_support_key({encoded.data(), 63}).has_value());
    auto wrong_tag = encoded;
    wrong_tag[0] = char(0);
    EXPECT_FALSE(D::decode_cartesian_support_key(wrong_tag).has_value());
    auto wrong_domain_size = encoded;
    wrong_domain_size[8] = char(2);
    EXPECT_FALSE(D::decode_cartesian_support_key(wrong_domain_size).has_value());
    auto bad_axis = encoded;
    set_word(bad_axis, 25, 3);
    EXPECT_FALSE(D::decode_cartesian_support_key(bad_axis).has_value());
    auto bad_level = encoded;
    set_word(bad_level, 33, 63);
    EXPECT_FALSE(D::decode_cartesian_support_key(bad_level).has_value());
    auto trailing = encoded;
    trailing.push_back(char(0));
    EXPECT_FALSE(D::decode_cartesian_support_key(trailing).has_value());
}

TEST(CartesianSupportKeyTest, MatchesEncodedRequiresSemanticEquality)
{
    const std::string domain = "s";
    const auto key = make(domain);
    const auto encoded = D::encode_cartesian_support_key(key);
    EXPECT_TRUE(D::cartesian_support_key_matches(key, encoded));
    auto different = make(domain);
    different.plane += 1;
    EXPECT_FALSE(D::cartesian_support_key_matches(different, encoded));
    auto other_domain = make(std::string("t"));
    EXPECT_FALSE(D::cartesian_support_key_matches(other_domain, encoded));
    EXPECT_FALSE(D::cartesian_support_key_matches(key, {encoded.data(), 10}));
}

TEST(CartesianSupportKeyTest, EncodeRejectsOutOfRangeIdentity)
{
    const std::string domain = "s";
    auto bad_axis = make(domain);
    bad_axis.axis = 3;
    EXPECT_THROW((void) D::encode_cartesian_support_key(bad_axis), std::invalid_argument);
    auto bad_level = make(domain);
    bad_level.level = 63;
    EXPECT_THROW((void) D::encode_cartesian_support_key(bad_level), std::invalid_argument);
}
