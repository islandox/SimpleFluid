#include <gtest/gtest.h>

#include <type_traits>

import SimpleFluid.Mesh;

namespace
{

TEST(MeshModuleTest, ExportsMeshApisAndReexportsCore)
{
    static_assert(SimpleFluid::MeshClass<
                  SimpleFluid::Meshes::OrthogonalCartesian3D>);
    static_assert(SimpleFluid::MeshIndexer<
                  SimpleFluid::Meshes::OrthogonalIndexer>);
    static_assert(std::is_same_v<
                  SimpleFluid::Meshes::OrthogonalCylindrical3D,
                  SimpleFluid::Meshes::OrthogonalCylindrial3D>);
    static_assert(std::is_class_v<SimpleFluid::MeshHandle<>>);
    static_assert(std::is_class_v<SimpleFluid::MeshPartitioner<
                  SimpleFluid::DefaultTpetraTypes>>);
    static_assert(SimpleFluid::TpetraTypePack<
                  SimpleFluid::DefaultTpetraTypes>);
}

} // namespace
