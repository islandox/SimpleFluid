/**
 * @file testTpetraTypes.cc
 * @brief Compile-time and smoke tests for Tpetra type packs.
 */

#include <gtest/gtest.h>

#include "dataclass/TpetraTypes.hh"

#include <type_traits>

namespace
{

static_assert(SimpleFluid::TpetraTypePack<SimpleFluid::DefaultTpetraTypes>);
static_assert(SimpleFluid::TpetraTypePack<SimpleFluid::TpetraTypes<float, int, long>>);

TEST(TpetraTypesTest, DefaultTypePackSatisfiesConcept)
{
    EXPECT_TRUE(SimpleFluid::TpetraTypePack<SimpleFluid::DefaultTpetraTypes>);
}

TEST(TpetraTypesTest, PublicAliasesMatchCamelCaseTypes)
{
    using Pack = SimpleFluid::DefaultTpetraTypes;

    EXPECT_TRUE((std::is_same_v<typename Pack::map_type, typename Pack::Map>));
    EXPECT_TRUE((std::is_same_v<typename Pack::graph_type, typename Pack::Graph>));
    EXPECT_TRUE((std::is_same_v<typename Pack::matrix_type, typename Pack::Matrix>));
    EXPECT_TRUE((std::is_same_v<typename Pack::vector_type, typename Pack::Vector>));
    EXPECT_TRUE((std::is_same_v<typename Pack::multi_vector_type,
                                typename Pack::MultiVector>));
    EXPECT_TRUE((std::is_same_v<typename Pack::import_type, typename Pack::Import>));
}

TEST(TpetraTypesTest, CustomScalarAndOrdinalsPropagate)
{
    using Pack = SimpleFluid::TpetraTypes<float, int, long>;

    EXPECT_TRUE((std::is_same_v<typename Pack::scalar_type, float>));
    EXPECT_TRUE((std::is_same_v<typename Pack::local_ordinal_type, int>));
    EXPECT_TRUE((std::is_same_v<typename Pack::global_ordinal_type, long>));
}

} // namespace
