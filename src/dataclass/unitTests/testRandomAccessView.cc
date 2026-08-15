/**
 * @file testRandomAccessView.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for the RandomAccessView utility class.
 *
 * Covers default construction, mutable access to vector data, const
 * iteration, and pointer-subrange views.
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "dataclass/RandomAccessView.hh"

#include <algorithm>
#include <vector>

/**
 * @brief Validates a default-constructed view is empty with size zero.
 */
TEST(RandomAccessViewTest, DefaultViewIsEmpty)
{
    SimpleFluid::RandomAccessView<int> view;

    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.size(), 0u);
}

/**
 * @brief Confirms elements can be mutated through the view and iterators work as expected.
 */
TEST(RandomAccessViewTest, ProvidesMutableAccessToVectorData)
{
    std::vector<int> data{4, 1, 3};
    SimpleFluid::RandomAccessView<int> view(data);

    ASSERT_EQ(view.size(), data.size());
    EXPECT_FALSE(view.empty());

    view[1] = 2;
    *(view.begin() + 0) = 5;

    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(view.begin()[2], 3);
    EXPECT_EQ(view.end() - view.begin(), 3);
    EXPECT_LT(view.begin(), view.end());

    std::sort(view.begin(), view.end());

    EXPECT_EQ(data, (std::vector<int>{2, 3, 5}));
}

/**
 * @brief Verifies read-only access through const iterators.
 */
TEST(RandomAccessViewTest, SupportsConstIteration)
{
    std::vector<int> data{10, 20, 30};
    const SimpleFluid::RandomAccessView<int> view(data);

    auto iter = view.begin();

    EXPECT_EQ(*iter, 10);
    EXPECT_EQ(iter[1], 20);
    EXPECT_EQ(*(iter + 2), 30);
    EXPECT_EQ(view.end() - view.begin(), 3);
    EXPECT_TRUE(view.begin() <= iter);
    EXPECT_TRUE(view.end() > iter);
}

/**
 * @brief Ensures a view constructed from a raw pointer + size represents only that subrange.
 */
TEST(RandomAccessViewTest, CanViewPointerSubrange)
{
    std::vector<double> data{0.0, 1.0, 2.0, 3.0};
    SimpleFluid::RandomAccessView<double> view(data.data() + 1, 2);

    ASSERT_EQ(view.size(), 2u);

    view[0] = 10.0;
    view[1] = 20.0;

    EXPECT_DOUBLE_EQ(data[0], 0.0);
    EXPECT_DOUBLE_EQ(data[1], 10.0);
    EXPECT_DOUBLE_EQ(data[2], 20.0);
    EXPECT_DOUBLE_EQ(data[3], 3.0);
}
