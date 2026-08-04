#include <gtest/gtest.h>

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <type_traits>
#endif

import SimpleFluid.Fields;

namespace
{

TEST(FieldsModuleTest, ExportsDefaultFieldTypes)
{
    using Pack = SimpleFluid::DefaultTpetraTypes;

    static_assert(SimpleFluid::TpetraTypePack<Pack>);
    static_assert(std::is_same_v<
                  typename SimpleFluid::CellField<Pack>::scalar_type,
                  typename Pack::scalar_type>);
    static_assert(std::is_same_v<
                  typename SimpleFluid::VectorFaceField<Pack>::vec_type,
                  SimpleFluid::vec3<typename Pack::scalar_type>>);
}

} // namespace
