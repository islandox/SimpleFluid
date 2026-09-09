/**
 * @file testDatabase.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for the Database key-value store.
 *
 * Validates storage and retrieval of supported scalar and array types,
 * key replacement across types, erase/clear, and error handling for
 * missing or type-mismatched keys.
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "dataclass/Database.hh"

#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Verifies all supported scalar and array types can be stored and retrieved.
 */
TEST(DatabaseTest, StoresSupportedScalarAndArrayTypes)
{
    SimpleFluid::Database db;

    int dimension = 3;
    const SimpleFluid::real_t mesh_size = 0.25;
    std::string name = "box";
    const bool enabled = true;
    SimpleFluid::ArrInt indices{1, 2, 3};
    const SimpleFluid::ArrReal weights{0.5, 1.5, 2.5};

    db.set("dimension", dimension);
    db.set("mesh_size", mesh_size);
    db.set("name", name);
    db.set("enabled", enabled);
    db.set("indices", indices);
    db.set("weights", weights);
    db.set("batches", SimpleFluid::ArrString{"xmin", "xmax"});

    EXPECT_EQ(db.size(), 7u);
    EXPECT_TRUE(db.contains("dimension"));
    EXPECT_TRUE(db.contains("batches"));
    EXPECT_FALSE(db.contains("missing"));

    EXPECT_EQ(db.get<int>("dimension"), 3);
    EXPECT_DOUBLE_EQ(db.get<SimpleFluid::real_t>("mesh_size"), 0.25);
    EXPECT_EQ(db.get<std::string>("name"), "box");
    EXPECT_TRUE(db.get<bool>("enabled"));
    EXPECT_EQ(db.get<std::vector<int>>("indices"), indices);
    EXPECT_EQ(db.get<std::vector<SimpleFluid::real_t>>("weights"), weights);
    EXPECT_EQ(db.get<std::vector<std::string>>("batches"),
              (std::vector<std::string>{"xmin", "xmax"}));

    const SimpleFluid::Database& const_db = db;
    EXPECT_EQ(const_db.get<int>("dimension"), 3);
    EXPECT_EQ(const_db.get<std::vector<std::string>>("batches").size(), 2u);
}

/**
 * @brief Checks overwriting a key with values of different types replaces the old entry.
 */
TEST(DatabaseTest, ReplacesExistingKeyAcrossTypes)
{
    SimpleFluid::Database db;

    db.set("value", 1);
    EXPECT_EQ(db.size(), 1u);
    EXPECT_EQ(db.get<int>("value"), 1);

    db.set("value", 2);
    EXPECT_EQ(db.size(), 1u);
    EXPECT_EQ(db.get<int>("value"), 2);

    db.set("value", SimpleFluid::real_t{3.5});
    EXPECT_EQ(db.size(), 1u);
    EXPECT_THROW(db.get<int>("value"), std::out_of_range);
    EXPECT_DOUBLE_EQ(db.get<SimpleFluid::real_t>("value"), 3.5);

    db.set("value", SimpleFluid::ArrReal{1.0, 2.0});
    EXPECT_EQ(db.size(), 1u);
    EXPECT_THROW(db.get<SimpleFluid::real_t>("value"), std::out_of_range);
    EXPECT_EQ(db.get<SimpleFluid::ArrReal>("value"), (SimpleFluid::ArrReal{1.0, 2.0}));
}

/**
 * @brief Verifies mutable/const references and value-semantic copies remain independent.
 */
TEST(DatabaseTest, ProvidesReferencesAndIndependentCopies)
{
    SimpleFluid::Database db;
    db.set("count", 1);
    db.set("labels", SimpleFluid::ArrString{"inlet"});

    int& count = db.get<int>("count");
    count = 2;
    EXPECT_EQ(db.get<int>("count"), 2);

    SimpleFluid::ArrString& labels = db.get<SimpleFluid::ArrString>("labels");
    labels.push_back("outlet");

    const SimpleFluid::Database& const_db = db;
    const SimpleFluid::ArrString& const_labels =
        const_db.get<SimpleFluid::ArrString>("labels");
    EXPECT_EQ(const_labels, (SimpleFluid::ArrString{"inlet", "outlet"}));

    SimpleFluid::Database copy = db;
    copy.get<int>("count") = 3;
    copy.get<SimpleFluid::ArrString>("labels").push_back("wall");

    EXPECT_EQ(db.get<int>("count"), 2);
    EXPECT_EQ(db.get<SimpleFluid::ArrString>("labels").size(), 2u);
    EXPECT_EQ(copy.get<int>("count"), 3);
    EXPECT_EQ(copy.get<SimpleFluid::ArrString>("labels").size(), 3u);
}

/**
 * @brief Confirms individual entries can be erased and the entire store can be cleared.
 */
TEST(DatabaseTest, ErasesAndClearsEntries)
{
    SimpleFluid::Database db;

    db.set("a", 1);
    db.set("b", SimpleFluid::real_t{2.0});
    db.set("c", std::string{"three"});

    EXPECT_EQ(db.size(), 3u);
    EXPECT_TRUE(db.erase("b"));
    EXPECT_FALSE(db.contains("b"));
    EXPECT_EQ(db.size(), 2u);
    EXPECT_FALSE(db.erase("missing"));

    db.clear();
    EXPECT_EQ(db.size(), 0u);
    EXPECT_FALSE(db.contains("a"));
    EXPECT_FALSE(db.contains("c"));
}

/**
 * @brief Ensures accessing missing keys or using the wrong type throws std::out_of_range.
 */
TEST(DatabaseTest, ThrowsForMissingOrWrongTypedAccess)
{
    SimpleFluid::Database db;

    db.set("count", 4);

    try
    {
        (void)db.get<int>("missing");
        FAIL() << "Expected missing-key access to throw";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_STREQ(error.what(), "Database key not found: missing");
    }

    try
    {
        (void)db.get<SimpleFluid::real_t>("count");
        FAIL() << "Expected wrong-type access to throw";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_STREQ(error.what(), "Database key not found: count");
    }

    const SimpleFluid::Database& const_db = db;
    try
    {
        (void)const_db.get<std::string>("missing");
        FAIL() << "Expected const missing-key access to throw";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_STREQ(error.what(), "Database key not found: missing");
    }
}
