#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <array>
#include <type_traits>
#endif

import SimpleFluid.Core;

namespace
{

TEST(CoreModuleTest, ExportsCoreRuntimeAndTypePackApis)
{
    SimpleFluid::Database database;
    database.set("iterations", 12);
    EXPECT_EQ(database.get<int>("iterations"), 12);

    std::array<int, 3> values{1, 2, 3};
    SimpleFluid::RandomAccessView<int> view(values.data(), values.size());
    EXPECT_EQ(view[1], 2);

    static_assert(SimpleFluid::TpetraTypePack<
                  SimpleFluid::DefaultTpetraTypes>);
    static_assert(std::is_same_v<
                  decltype(SimpleFluid::vec3{1.0, 2.0, 3.0}.x), double>);
}

} // namespace
