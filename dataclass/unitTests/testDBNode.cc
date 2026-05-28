/**
 * @file testDBNode.cc
 * @brief Unit tests for the DBNode typed key-value container.
 */

#include <gtest/gtest.h>

#include "dataclass/DBNode.hh"

#include <stdexcept>
#include <string>

namespace
{

TEST(DBNodeTest, StartsEmpty)
{
    const SimpleFluid::DBNode<int> node;
    EXPECT_TRUE(node.empty());
    EXPECT_EQ(node.size(), 0u);
    EXPECT_FALSE(node.contains("missing"));
}

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

TEST(DBNodeTest, SetReplacesExistingValue)
{
    SimpleFluid::DBNode<int> node;
    node.set("count", 1);
    node.set("count", 2);

    EXPECT_EQ(node.size(), 1u);
    EXPECT_EQ(node.get("count"), 2);
}

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

TEST(DBNodeTest, GetMissingThrows)
{
    SimpleFluid::DBNode<int> node;

    EXPECT_THROW(node.get("missing"), std::out_of_range);
    const auto& const_node = node;
    EXPECT_THROW(const_node.get("missing"), std::out_of_range);
}

} // namespace
