/**
 * @file testDBNode.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for the DBNode typed key-value container.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "dataclass/DBNode.hh"

#include <stdexcept>
#include <string>

namespace
{

/** @brief Verifies a default node has no keys or values. */
TEST(DBNodeTest, StartsEmpty)
{
    const SimpleFluid::DBNode<int> node;
    EXPECT_TRUE(node.empty());
    EXPECT_EQ(node.size(), 0u);
    EXPECT_FALSE(node.contains("missing"));
}

/** @brief Exercises insertion, lookup, containment, and size reporting. */
TEST(DBNodeTest, SetAndGetValues)
{
    SimpleFluid::DBNode<std::string> node;
    node.set("name", "fluid");
    node.set("state", std::string{"warm"});

    EXPECT_EQ(node.size(), 2u);
    EXPECT_TRUE(node.contains("name"));
    EXPECT_EQ(node.get("name"), "fluid");
    EXPECT_EQ(node.get("state"), "warm");

    const auto& const_node = node;
    EXPECT_EQ(const_node.get("name"), "fluid");
}

/** @brief Confirms assigning an existing key replaces its stored value. */
TEST(DBNodeTest, SetReplacesExistingValue)
{
    SimpleFluid::DBNode<int> node;
    node.set("count", 1);
    node.set("count", 2);

    EXPECT_EQ(node.size(), 1u);
    EXPECT_EQ(node.get("count"), 2);
}

/** @brief Verifies individual erasure and full-node clearing semantics. */
TEST(DBNodeTest, EraseAndClear)
{
    SimpleFluid::DBNode<int> node;
    node.set("a", 1);
    node.set("b", 2);

    EXPECT_TRUE(node.erase("a"));
    EXPECT_FALSE(node.contains("a"));
    EXPECT_FALSE(node.erase("a"));
    EXPECT_EQ(node.size(), 1u);

    node.clear();
    EXPECT_TRUE(node.empty());
}

/** @brief Ensures mutable and const lookup reject a missing key. */
TEST(DBNodeTest, GetMissingThrows)
{
    SimpleFluid::DBNode<int> node;

    EXPECT_THROW(node.get("missing"), std::out_of_range);
    const auto& const_node = node;
    EXPECT_THROW(const_node.get("missing"), std::out_of_range);
}

} // namespace
