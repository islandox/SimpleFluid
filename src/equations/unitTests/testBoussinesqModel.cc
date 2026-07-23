/**
 * @file testBoussinesqModel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests physical material fields and temperature-source registration.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoussinesqModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <limits>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Verifies default parsing and derivation of Boussinesq physical coefficients. */
TEST(BoussinesqModelTest, ParsesDefaultsAndDerivesPhysicalCoefficients)
{
    SimpleFluid::TimeStepperOptions time_options;
    time_options.kinematic_viscosity = 0.2;
    time_options.thermal_diffusivity = 0.3;

    SimpleFluid::Database database;
    database.set("reference_density", SimpleFluid::real_t{2.0});
    database.set("density", SimpleFluid::real_t{1.8});
    database.set(
        "specific_heat_capacity", SimpleFluid::real_t{4.0});
    database.set("density_feedback_enabled", true);
    database.set(
        "temperature_source_names",
        SimpleFluid::ArrString{"decay", "fission"});
    database.set(
        "temperature_source_power_densities",
        SimpleFluid::ArrReal{-1.0, 5.0});

    const auto options =
        SimpleFluid::boussinesq_model_options_from_database(
            database, time_options);

    EXPECT_DOUBLE_EQ(options.reference_density, 2.0);
    EXPECT_DOUBLE_EQ(options.density, 1.8);
    EXPECT_DOUBLE_EQ(options.specific_heat_capacity, 4.0);
    ASSERT_TRUE(options.dynamic_viscosity.has_value());
    ASSERT_TRUE(options.thermal_conductivity.has_value());
    EXPECT_DOUBLE_EQ(*options.dynamic_viscosity, 0.4);
    EXPECT_DOUBLE_EQ(*options.thermal_conductivity, 2.4);
    EXPECT_TRUE(options.density_feedback_enabled);
    EXPECT_EQ(
        options.temperature_source_names,
        (SimpleFluid::ArrString{"decay", "fission"}));
}

/** @brief Verifies that invalid Boussinesq database values are rejected. */
TEST(BoussinesqModelTest, RejectsInvalidDatabaseValues)
{
    const SimpleFluid::TimeStepperOptions time_options;

    SimpleFluid::Database wrong_type;
    wrong_type.set("density", std::string{"heavy"});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            wrong_type, time_options),
        std::invalid_argument);

    SimpleFluid::Database negative_viscosity;
    negative_viscosity.set(
        "dynamic_viscosity", SimpleFluid::real_t{-1.0});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            negative_viscosity, time_options),
        std::invalid_argument);

    SimpleFluid::Database zero_density;
    zero_density.set("density", SimpleFluid::real_t{0.0});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            zero_density, time_options),
        std::invalid_argument);

    SimpleFluid::Database zero_capacity;
    zero_capacity.set(
        "specific_heat_capacity", SimpleFluid::real_t{0.0});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            zero_capacity, time_options),
        std::invalid_argument);

    SimpleFluid::Database negative_conductivity;
    negative_conductivity.set(
        "thermal_conductivity", SimpleFluid::real_t{-1.0});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            negative_conductivity, time_options),
        std::invalid_argument);

    SimpleFluid::Database non_finite_density;
    non_finite_density.set(
        "density",
        std::numeric_limits<SimpleFluid::real_t>::infinity());
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            non_finite_density, time_options),
        std::invalid_argument);

    SimpleFluid::Database mismatched_sources;
    mismatched_sources.set(
        "temperature_source_names",
        SimpleFluid::ArrString{"heat"});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            mismatched_sources, time_options),
        std::invalid_argument);

    SimpleFluid::Database duplicate_sources;
    duplicate_sources.set(
        "temperature_source_names",
        SimpleFluid::ArrString{"heat", "heat"});
    duplicate_sources.set(
        "temperature_source_power_densities",
        SimpleFluid::ArrReal{1.0, 2.0});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            duplicate_sources, time_options),
        std::invalid_argument);

    SimpleFluid::Database empty_source_name;
    empty_source_name.set(
        "temperature_source_names",
        SimpleFluid::ArrString{""});
    empty_source_name.set(
        "temperature_source_power_densities",
        SimpleFluid::ArrReal{1.0});
    EXPECT_THROW(
        SimpleFluid::boussinesq_model_options_from_database(
            empty_source_name, time_options),
        std::invalid_argument);
}

/** @brief Verifies ordered source registration and spatial field initialization. */
TEST(BoussinesqModelTest, RegistryIsOrderedAndSupportsSpatialInitialization)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);

    registry.add("zeta", 2.0);
    auto& alpha = registry.add("alpha");
    alpha.initialize(
        [](const auto& centroid)
        {
            return centroid.x + 2.0 * centroid.y;
        });

    std::vector<std::string> names;
    for (const auto& [name, source] : registry.entries())
    {
        (void)source;
        names.push_back(name);
    }
    EXPECT_EQ(names, (std::vector<std::string>{"alpha", "zeta"}));

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto& centroid = mesh->cell_centroid(cell_lid);
        EXPECT_DOUBLE_EQ(
            alpha.field().value(cell_lid),
            centroid.x + 2.0 * centroid.y);
        EXPECT_DOUBLE_EQ(
            registry.total_power_density(cell_lid),
            alpha.field().value(cell_lid) + 2.0);
    }

    for (const auto* reserved_name :
         {"temperature", "wall_distance", "wall_y_plus",
          "buoyancy_production"})
    {
        EXPECT_THROW(
            registry.add(reserved_name),
            std::invalid_argument);
    }
    EXPECT_THROW(
        registry.add(
            "invalid",
            std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_EQ(registry.find("invalid"), nullptr);
    EXPECT_TRUE(registry.remove("zeta"));
    EXPECT_EQ(registry.find("zeta"), nullptr);
}

/** @brief Verifies timestep refresh skips static and disabled source fields. */
TEST(BoussinesqModelTest, RegistryUpdatesOnlyEnabledDynamicSources)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    SimpleFluid::TemperatureSourceRegistry<Pack> registry(mesh);

    auto& static_source = registry.add("static", 2.0);
    auto& disabled_source = registry.add("disabled", 3.0);
    auto& dynamic_source = registry.add("dynamic", 4.0);
    int disabled_calls = 0;
    int dynamic_calls = 0;
    disabled_source.set_updater(
        [&](const auto&, auto&)
        {
            ++disabled_calls;
        });
    disabled_source.set_enabled(false);
    dynamic_source.set_updater(
        [&](const auto&, auto& field)
        {
            ++dynamic_calls;
            field.put_scalar(5.0);
        });

    SimpleFluid::CellField<Pack> temperature(
        mesh, 300.0, "temperature");
    SimpleFluid::CellField<Pack> pressure(mesh, 0.0, "pressure");
    SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
    const SimpleFluid::BoussinesqUpdateContext<Pack> context{
        1.0, 2, *mesh, temperature, pressure, velocity};

    registry.update(context);

    EXPECT_EQ(disabled_calls, 0);
    EXPECT_EQ(dynamic_calls, 1);
    EXPECT_DOUBLE_EQ(static_source.field().value(0), 2.0);
    EXPECT_DOUBLE_EQ(disabled_source.field().value(0), 3.0);
    EXPECT_DOUBLE_EQ(dynamic_source.field().value(0), 5.0);

    dynamic_source.clear_updater();
    registry.update(context);
    EXPECT_EQ(dynamic_calls, 1);
}

/** @brief Verifies that material validation rejects non-finite field updates. */
TEST(BoussinesqModelTest, MaterialValidationRejectsNonFiniteUpdates)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
    const SimpleFluid::TimeStepperOptions time_options;
    const auto options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, options, time_options);
    material.density.set_value(
        0, std::numeric_limits<double>::quiet_NaN());

    EXPECT_THROW(
        material.validate_and_sync(), std::invalid_argument);
}

/** @brief Verifies validated initialization of spatially varying material fields. */
TEST(BoussinesqModelTest, MaterialFieldsSupportValidatedInitialization)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    const SimpleFluid::TimeStepperOptions time_options;
    const auto options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, options, time_options);

    material.initialize_density(
        [](const auto& centroid)
        {
            return 1.0 + centroid.x;
        });
    material.initialize_specific_heat_capacity(3.0);
    material.initialize_dynamic_viscosity(0.0);
    material.initialize_thermal_conductivity(2.0);

    EXPECT_DOUBLE_EQ(material.density.value(0), 1.5);
    EXPECT_DOUBLE_EQ(material.density.value(1), 2.5);
    EXPECT_DOUBLE_EQ(
        material.specific_heat_capacity.value(0), 3.0);
    EXPECT_THROW(
        material.initialize_density(0.0),
        std::invalid_argument);
    EXPECT_THROW(
        material.initialize_thermal_conductivity(-1.0),
        std::invalid_argument);
}

} // namespace
