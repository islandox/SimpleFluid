/** @file testFieldStoredCustomProvider.cc
 * @brief A custom descriptor opts into the generic stored-field provider.
 */
#include "fields/FieldStored.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <type_traits>

namespace
{
using namespace SimpleFluid;
struct CustomDescriptor : Field<double, FaceLocation>
{
    using Field<double, FaceLocation>::Field;
};
using CustomStorage = FieldStored<CustomDescriptor>;
static_assert(std::is_copy_constructible_v<CustomStorage>);
static_assert(std::is_move_constructible_v<CustomStorage>);
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(FieldStoredTest, GenericDescriptorProviderSharesTheLocationLayout)
{
    auto geometry =
        std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}});
    SP<const MeshHandle<>> mesh = std::make_shared<MeshHandle<>>(geometry);
    ScalarFaceFieldStored<> standard(mesh, 2.0, "standard");
    CustomStorage custom(CustomDescriptor{"custom"}, mesh, 3.0);
    EXPECT_EQ(custom.mesh_ptr().get(), mesh.get());
    EXPECT_EQ(&standard.owned_face_ids(), &custom.owned_face_ids());
    EXPECT_NE(standard.id(), custom.id());
    custom.set_owned_value(0, 11.0);
    EXPECT_DOUBLE_EQ(custom.value(0), 11.0);
    EXPECT_DOUBLE_EQ(custom.local_value(0), 3.0);
    EXPECT_DOUBLE_EQ(standard.value(0), 2.0);
    custom.sync_ghosts();
    EXPECT_DOUBLE_EQ(custom.local_value(0), 11.0);
}
} // namespace
