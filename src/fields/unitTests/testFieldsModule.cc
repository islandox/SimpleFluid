#include <gtest/gtest.h>

#include <type_traits>

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
