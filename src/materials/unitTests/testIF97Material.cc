#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "materials/IF97Material.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <limits>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Fields = SimpleFluid::MaterialPropertyFields<Pack>;
using Context = SimpleFluid::BoussinesqUpdateContext<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(IF97MaterialTest, ReferenceOptionsUsePhysicalWaterCoefficients)
{
    const auto options = SimpleFluid::if97_liquid_model_options(298.15, 101325.0);
    EXPECT_NEAR(options.reference_density, 997.05, 0.03);
    EXPECT_DOUBLE_EQ(options.reference_density, options.density);
    EXPECT_NEAR(options.specific_heat_capacity, 4181.9, 1.0);
    ASSERT_TRUE(options.dynamic_viscosity);
    EXPECT_NEAR(*options.dynamic_viscosity, 0.0008900, 1e-6);
    ASSERT_TRUE(options.thermal_conductivity);
    EXPECT_NEAR(*options.thermal_conductivity, 0.6065, 0.001);
    EXPECT_FALSE(options.density_feedback_enabled);
}

TEST(IF97MaterialTest, UpdatesAllOwnedFieldsAtPrescribedAbsolutePressure)
{
    const auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    Fields material(mesh, SimpleFluid::if97_liquid_model_options(298.15, 101325.0), {});
    Fields::field_type temperature(mesh, 298.15, "T"), pressure(mesh, -2e6, "p");
    Context::velocity_field_type velocity(mesh, "U");
    temperature.set_owned_value(1, 350.0);
    temperature.sync_ghosts();
    material.set_updater(SimpleFluid::make_if97_liquid_material_updater<Pack>(101325.0));
    material.update(Context{0.0, 0, *mesh, temperature, pressure, velocity});
    const auto expected = SimpleFluid::IF97Water::liquid(350.0, 101325.0);
    EXPECT_DOUBLE_EQ(material.density.value(1), expected.density);
    EXPECT_DOUBLE_EQ(material.specific_heat_capacity.value(1), expected.specific_heat_capacity);
    EXPECT_DOUBLE_EQ(material.dynamic_viscosity.value(1), expected.dynamic_viscosity);
    EXPECT_DOUBLE_EQ(material.thermal_conductivity.value(1), expected.thermal_conductivity);
    EXPECT_GT(material.density.value(0), material.density.value(1));
    EXPECT_GT(material.dynamic_viscosity.value(0), material.dynamic_viscosity.value(1));
}

TEST(IF97MaterialTest, InvalidLaterCellDoesNotPublishEarlierLocalValues)
{
    const auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    Fields material(mesh, SimpleFluid::if97_liquid_model_options(298.15, 101325.0), {});
    Fields::field_type temperature(mesh, 350.0, "T"), pressure(mesh, 0.0, "p");
    Context::velocity_field_type velocity(mesh, "U");
    const auto density = material.density.value(0);
    const auto cp = material.specific_heat_capacity.value(0);
    const auto viscosity = material.dynamic_viscosity.value(0);
    const auto conductivity = material.thermal_conductivity.value(0);
    temperature.set_owned_value(1, 400.0);
    temperature.sync_ghosts();
    material.set_updater(SimpleFluid::make_if97_liquid_material_updater<Pack>(101325.0));
    EXPECT_ANY_THROW(material.update(Context{0.0, 0, *mesh, temperature, pressure, velocity}));
    for (int cell = 0; cell < 2; ++cell)
    {
        EXPECT_DOUBLE_EQ(material.density.value(cell), density);
        EXPECT_DOUBLE_EQ(material.specific_heat_capacity.value(cell), cp);
        EXPECT_DOUBLE_EQ(material.dynamic_viscosity.value(cell), viscosity);
        EXPECT_DOUBLE_EQ(material.thermal_conductivity.value(cell), conductivity);
    }
}

TEST(IF97MaterialTest, RejectsForeignMeshAndBadPressureWithoutRestrictingValidLowPressure)
{
    const auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    const auto other = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    Fields material(mesh, SimpleFluid::if97_liquid_model_options(298.15, 101325.0), {});
    Fields::field_type temperature(other, 298.15, "T"), pressure(other, 0.0, "p");
    Context::velocity_field_type velocity(other, "U");
    material.set_updater(SimpleFluid::make_if97_liquid_material_updater<Pack>(101325.0));
    EXPECT_ANY_THROW(material.update(Context{0.0, 0, *other, temperature, pressure, velocity}));
    EXPECT_THROW(SimpleFluid::make_if97_liquid_material_updater<Pack>(0.0), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::make_if97_liquid_material_updater<Pack>(std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_THROW(SimpleFluid::make_if97_liquid_material_updater<Pack>(100.0), std::out_of_range);
    EXPECT_NO_THROW(SimpleFluid::make_if97_liquid_material_updater<Pack>(1000.0));
}

TEST(IF97MaterialTest, UpdatesNativeFieldsWithoutLegacyMeshConversion)
{
    using Mesh = SimpleFluid::MeshHandle<Pack>;
    using Material = SimpleFluid::MaterialPropertyFields<Pack, Mesh>;
    using UpdateContext = SimpleFluid::BoussinesqUpdateContext<Pack, Mesh>;
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 0.5, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    SimpleFluid::SP<const Mesh> mesh = std::make_shared<Mesh>(cartesian);
    Material material(mesh, SimpleFluid::if97_liquid_model_options(298.15, 101325.0), {});
    Material::field_type temperature(mesh, 330.0, "T"), pressure(mesh, 0.0, "p");
    UpdateContext::velocity_field_type velocity(mesh, Mesh::Vec3{}, "U");
    const auto updater = SimpleFluid::make_if97_liquid_material_updater<Pack, Mesh>(101325.0);
    material.set_updater(updater);
    material.update(UpdateContext{0.0, 0, *mesh, temperature, pressure, velocity});
    EXPECT_FALSE(mesh->legacy_mesh());
    const auto expected = SimpleFluid::IF97Water::liquid(330.0, 101325.0);
    EXPECT_DOUBLE_EQ(material.density.value(0), expected.density);
    EXPECT_DOUBLE_EQ(material.specific_heat_capacity.value(1), expected.specific_heat_capacity);
    EXPECT_DOUBLE_EQ(material.dynamic_viscosity.value(0), expected.dynamic_viscosity);
    EXPECT_DOUBLE_EQ(material.thermal_conductivity.value(1), expected.thermal_conductivity);
}

TEST(IF97MaterialTest, DistributedUpdateSynchronizesAndRejectsCollectively)
{
    const auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    Fields material(mesh, SimpleFluid::if97_liquid_model_options(298.15, 101325.0), {});
    Fields::field_type temperature(mesh, 298.15, "T"), pressure(mesh, 0.0, "p");
    Context::velocity_field_type velocity(mesh, "U");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        temperature.set_owned_value(cell, 298.15 + 50.0 * mesh->cell_centroid(cell).x);
    }
    temperature.sync_ghosts();
    material.set_updater(SimpleFluid::make_if97_liquid_material_updater<Pack>(101325.0));
    const Context context{0.0, 0, *mesh, temperature, pressure, velocity};
    material.update(context);
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(local);
        const auto water = SimpleFluid::IF97Water::liquid(temperature.local_value(cell), 101325.0);
        EXPECT_DOUBLE_EQ(material.density.local_value(cell), water.density);
        EXPECT_DOUBLE_EQ(material.specific_heat_capacity.local_value(cell), water.specific_heat_capacity);
        EXPECT_DOUBLE_EQ(material.dynamic_viscosity.local_value(cell), water.dynamic_viscosity);
        EXPECT_DOUBLE_EQ(material.thermal_conductivity.local_value(cell), water.thermal_conductivity);
    }
    const auto accepted = material.snapshot();
    if (mesh->owned_cell_map()->getComm()->getRank() == 0)
        temperature.set_owned_value(0, 400.0);
    temperature.sync_ghosts();
    EXPECT_ANY_THROW(material.update(context));
    // The owner can restore the last accepted state collectively after any
    // rank rejects a temperature; the callback adds no MPI of its own.
    EXPECT_NO_THROW(material.restore(accepted));
}
} // namespace
