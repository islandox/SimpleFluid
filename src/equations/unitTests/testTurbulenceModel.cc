/**
 * @file testTurbulenceModel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Direct runtime tests for two-equation turbulence model ownership.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/turbulence/TurbulenceModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
using MaterialType = SimpleFluid::MaterialPropertyFields<Pack>;
using Model = SimpleFluid::TurbulenceModel<Pack>;
using ModelOptions = SimpleFluid::TurbulenceModelOptions;
using ModelType = SimpleFluid::TurbulenceModelType;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Runtime turbulence model and its expected secondary field name. */
struct ModelCase
{
    const char* name;
    ModelType type;
    bool epsilon_family;
    bool requires_wall_distance;
};

constexpr std::array active_models{
    ModelCase{"standardKEpsilon", ModelType::StandardKEpsilon, true, false},
    ModelCase{"RNGKEpsilon", ModelType::RNGKEpsilon, true, false},
    ModelCase{"realizableKEpsilon", ModelType::RealizableKEpsilon, true, false},
    ModelCase{"standardKOmega", ModelType::StandardKOmega, false, false},
    ModelCase{"BSLKOmega", ModelType::BSLKOmega, false, true},
    ModelCase{"SSTKOmega", ModelType::SSTKOmega, false, true}};

constexpr double reference_density = 1000.0;
constexpr double density = 997.0;
constexpr double heat_capacity = 4180.0;
constexpr double molecular_viscosity = 1.5e-3;
constexpr double molecular_conductivity = 0.61;
constexpr double turbulent_prandtl_number = 0.85;

SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
}

SimpleFluid::SP<MeshType> make_two_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
}

MaterialType make_material(SimpleFluid::SP<const MeshType> mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = reference_density;
    options.density = density;
    options.specific_heat_capacity = heat_capacity;
    options.dynamic_viscosity = molecular_viscosity;
    options.thermal_conductivity = molecular_conductivity;
    return MaterialType(std::move(mesh), options, time_options);
}

ModelOptions make_model_options(ModelType type)
{
    ModelOptions options;
    options.model = type;
    options.initial_turbulent_kinetic_energy = 0.2;
    options.initial_dissipation_rate = 0.05;
    options.initial_specific_dissipation_rate = 2.0;
    options.min_turbulent_kinetic_energy = 1.0e-10;
    options.min_dissipation_rate = 2.0e-10;
    options.min_specific_dissipation_rate = 3.0e-10;
    options.turbulent_prandtl_number = turbulent_prandtl_number;
    if (type == ModelType::BSLKOmega || type == ModelType::SSTKOmega)
    {
        options.initial_wall_distance = 0.25;
    }
    return options;
}

/** @brief Verifies null-mesh rejection before boundary-cache construction. */
TEST(TurbulenceModelTest, RejectsNullMeshBeforeBuildingBoundaryCaches)
{
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    EXPECT_THROW((Model(SimpleFluid::SP<const MeshType>{}, boundary_conditions)),
                 std::invalid_argument);
}

/** @brief Verifies closure configuration, family fields, and clean model disabling. */
TEST(TurbulenceModelTest, ConfiguresEveryClosureExposesFamilyFieldsAndDisablesCleanly)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    Model model(mesh, boundary_conditions);

    EXPECT_FALSE(model.enabled());
    EXPECT_EQ(model.type(), ModelType::Laminar);
    EXPECT_TRUE(model.output_fields().empty());
    EXPECT_EQ(model.dissipation_rate(), nullptr);
    EXPECT_EQ(model.specific_dissipation_rate(), nullptr);
    EXPECT_THROW(static_cast<void>(model.turbulent_kinetic_energy()), std::logic_error);
    EXPECT_FALSE(model.disable());

    for (const auto& entry : active_models)
    {
        SCOPED_TRACE(entry.name);
        model.configure(make_model_options(entry.type), material, reference_density);

        ASSERT_TRUE(model.enabled());
        EXPECT_EQ(model.type(), entry.type);
        EXPECT_EQ(model.options().model, entry.type);
        EXPECT_EQ(model.options().initial_wall_distance.has_value(), entry.requires_wall_distance);
        EXPECT_DOUBLE_EQ(model.turbulent_kinetic_energy().value(0), 0.2);

        const auto* epsilon = model.dissipation_rate();
        const auto* omega = model.specific_dissipation_rate();
        if (entry.epsilon_family)
        {
            ASSERT_NE(epsilon, nullptr);
            EXPECT_EQ(omega, nullptr);
            EXPECT_DOUBLE_EQ(epsilon->value(0), 0.05);
        }
        else
        {
            EXPECT_EQ(epsilon, nullptr);
            ASSERT_NE(omega, nullptr);
            EXPECT_DOUBLE_EQ(omega->value(0), 2.0);
        }

        const auto& output = model.output_fields();
        ASSERT_EQ(output.size(), 5U);
        EXPECT_EQ(output.at("k"), &model.turbulent_kinetic_energy());
        EXPECT_EQ(output.at(entry.epsilon_family ? "epsilon" : "omega"),
                  entry.epsilon_family ? epsilon : omega);
        EXPECT_EQ(output.at("nu_t"), &model.turbulent_kinematic_viscosity());
        EXPECT_EQ(output.at("mu_eff"), &model.effective_dynamic_viscosity());
        EXPECT_EQ(output.at("lambda_eff"), &model.effective_thermal_conductivity());
        EXPECT_EQ(output.count(entry.epsilon_family ? "omega" : "epsilon"), 0U);
    }

    EXPECT_TRUE(model.disable());
    EXPECT_FALSE(model.enabled());
    EXPECT_EQ(model.type(), ModelType::Laminar);
    EXPECT_TRUE(model.output_fields().empty());
    EXPECT_EQ(model.dissipation_rate(), nullptr);
    EXPECT_EQ(model.specific_dissipation_rate(), nullptr);
    EXPECT_THROW(static_cast<void>(model.turbulent_kinetic_energy()), std::logic_error);
    EXPECT_FALSE(model.disable());
}

/** @brief Verifies positive eddy viscosity and effective properties for every closure. */
TEST(TurbulenceModelTest, EveryClosureInitializesPositiveEddyViscosityAndEffectiveProperties)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    Model model(mesh, boundary_conditions);

    for (const auto& entry : active_models)
    {
        SCOPED_TRACE(entry.name);
        const auto options = make_model_options(entry.type);
        model.configure(options, material, reference_density);

        const auto nu_t = model.turbulent_kinematic_viscosity().value(0);
        ASSERT_TRUE(std::isfinite(nu_t));
        EXPECT_GT(nu_t, 0.0);

        const auto expected_dynamic_viscosity =
            material.dynamic_viscosity.value(0) + reference_density * nu_t;
        const auto expected_thermal_conductivity =
            material.thermal_conductivity.value(0) + material.density.value(0) *
                                                         material.specific_heat_capacity.value(0) *
                                                         nu_t / options.turbulent_prandtl_number;
        EXPECT_DOUBLE_EQ(model.effective_dynamic_viscosity().value(0), expected_dynamic_viscosity);
        EXPECT_DOUBLE_EQ(model.effective_thermal_conductivity().value(0),
                         expected_thermal_conductivity);
        EXPECT_GT(model.effective_dynamic_viscosity().value(0),
                  material.dynamic_viscosity.value(0));
        EXPECT_GT(model.effective_thermal_conductivity().value(0),
                  material.thermal_conductivity.value(0));
    }
}

/** @brief Verifies configuration rollback after effective-property overflow. */
TEST(TurbulenceModelTest, ConfigureRejectsEffectivePropertyOverflowAndPreservesActiveModel)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    Model model(mesh, boundary_conditions);

    const auto active_options = make_model_options(ModelType::StandardKEpsilon);
    model.configure(active_options, material, reference_density);

    const auto* const active_k = &model.turbulent_kinetic_energy();
    const auto* const active_epsilon = model.dissipation_rate();
    ASSERT_NE(active_epsilon, nullptr);
    const auto initial_k = active_k->value(0);
    const auto initial_epsilon = active_epsilon->value(0);
    const auto initial_mu_eff = model.effective_dynamic_viscosity().value(0);
    const auto initial_lambda_eff = model.effective_thermal_conductivity().value(0);

    auto overflowing_options = make_model_options(ModelType::RNGKEpsilon);
    overflowing_options.initial_turbulent_kinetic_energy = 10.0;
    overflowing_options.initial_dissipation_rate = 1.0e-3;
    EXPECT_ANY_THROW(
        model.configure(overflowing_options, material, std::numeric_limits<double>::max()));

    EXPECT_TRUE(model.enabled());
    EXPECT_EQ(model.type(), ModelType::StandardKEpsilon);
    EXPECT_EQ(model.options().model, ModelType::StandardKEpsilon);
    EXPECT_DOUBLE_EQ(model.options().initial_turbulent_kinetic_energy,
                     active_options.initial_turbulent_kinetic_energy);
    EXPECT_EQ(&model.turbulent_kinetic_energy(), active_k);
    EXPECT_EQ(model.dissipation_rate(), active_epsilon);
    EXPECT_DOUBLE_EQ(model.turbulent_kinetic_energy().value(0), initial_k);
    ASSERT_NE(model.dissipation_rate(), nullptr);
    EXPECT_DOUBLE_EQ(model.dissipation_rate()->value(0), initial_epsilon);
    EXPECT_DOUBLE_EQ(model.effective_dynamic_viscosity().value(0), initial_mu_eff);
    EXPECT_DOUBLE_EQ(model.effective_thermal_conductivity().value(0), initial_lambda_eff);
}

/** @brief Verifies rejection of unsafe active scalar boundary data. */
TEST(TurbulenceModelTest, ConfigureRejectsUnsafeActiveScalarBoundaryData)
{
    const auto expect_rejected =
        [](const SimpleFluid::BoundaryConditionSet& boundaries)
    {
        auto mesh = make_single_cell_mesh();
        auto material = make_material(mesh);
        Model model(mesh, boundaries);
        EXPECT_THROW(
            model.configure(
                make_model_options(ModelType::StandardKEpsilon),
                material,
                reference_density),
            std::invalid_argument);
        EXPECT_FALSE(model.enabled());
    };

    SimpleFluid::BoundaryConditionSet non_finite_k;
    non_finite_k.turbulence.turbulent_kinetic_energy["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        std::numeric_limits<double>::quiet_NaN()};
    expect_rejected(non_finite_k);

    SimpleFluid::BoundaryConditionSet below_floor_k;
    below_floor_k.turbulence.turbulent_kinetic_energy["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    expect_rejected(below_floor_k);

    SimpleFluid::BoundaryConditionSet below_floor_epsilon;
    below_floor_epsilon.turbulence.dissipation_rate["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    expect_rejected(below_floor_epsilon);
}

/** @brief Verifies atomic rejection of effective-property overflow during refresh. */
TEST(TurbulenceModelTest, RefreshRejectsEffectivePropertyOverflowWithoutPartialPublication)
{
    auto mesh = make_two_cell_mesh();
    ASSERT_EQ(mesh->num_owned_cells(), 2U);
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    Model model(mesh, boundary_conditions);
    model.configure(make_model_options(ModelType::StandardKEpsilon), material, reference_density);

    std::array<double, 2> initial_mu_eff{};
    std::array<double, 2> initial_lambda_eff{};
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        initial_mu_eff[owned] = model.effective_dynamic_viscosity().value(cell_lid);
        initial_lambda_eff[owned] = model.effective_thermal_conductivity().value(cell_lid);
    }

    material.dynamic_viscosity.set_owned_value(0, 2.0 * molecular_viscosity);
    material.thermal_conductivity.set_owned_value(0, 2.0 * molecular_conductivity);
    material.specific_heat_capacity.set_owned_value(1, std::numeric_limits<double>::max());
    material.dynamic_viscosity.sync_ghosts();
    material.thermal_conductivity.sync_ghosts();
    material.specific_heat_capacity.sync_ghosts();

    EXPECT_ANY_THROW(model.refresh_effective_properties(material, reference_density));

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(model.effective_dynamic_viscosity().value(cell_lid),
                         initial_mu_eff[owned]);
        EXPECT_DOUBLE_EQ(model.effective_thermal_conductivity().value(cell_lid),
                         initial_lambda_eff[owned]);
    }
}

/** @brief Verifies configured and replacement wall-distance validation for BSL and SST. */
TEST(TurbulenceModelTest, BSLAndSSTValidateConfiguredAndReplacementWallDistance)
{
    const std::array menter_models{ModelType::BSLKOmega, ModelType::SSTKOmega};
    const std::array invalid_distances{0.0, -0.1, std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN()};

    for (const auto type : menter_models)
    {
        SCOPED_TRACE(SimpleFluid::to_string(type));
        auto mesh = make_single_cell_mesh();
        auto other_mesh = make_single_cell_mesh();
        auto material = make_material(mesh);
        SimpleFluid::BoundaryConditionSet boundary_conditions;
        Model model(mesh, boundary_conditions);

        auto missing_distance = make_model_options(type);
        missing_distance.initial_wall_distance.reset();
        EXPECT_THROW(model.configure(missing_distance, material, reference_density),
                     std::invalid_argument);

        for (const auto distance : invalid_distances)
        {
            auto invalid_options = make_model_options(type);
            invalid_options.initial_wall_distance = distance;
            EXPECT_THROW(model.configure(invalid_options, material, reference_density),
                         std::invalid_argument);
        }

        model.configure(make_model_options(type), material, reference_density);
        FieldType valid_distance(mesh, 0.125, "wall_distance");
        EXPECT_NO_THROW(model.set_wall_distance(valid_distance));

        for (const auto distance : invalid_distances)
        {
            FieldType invalid_distance(mesh, distance, "invalid_wall_distance");
            EXPECT_THROW(model.set_wall_distance(invalid_distance), std::invalid_argument);
        }

        FieldType wrong_mesh_distance(other_mesh, 0.125, "wrong_mesh_wall_distance");
        EXPECT_THROW(model.set_wall_distance(wrong_mesh_distance), std::invalid_argument);
        EXPECT_NO_THROW(model.set_wall_distance(valid_distance));
    }
}

/** @brief Verifies that every active closure advances both transport equations. */
TEST(TurbulenceModelTest, EveryClosureAdvancesBothTransportEquations)
{
    for (const auto& entry : active_models)
    {
        SCOPED_TRACE(entry.name);
        auto mesh = make_two_cell_mesh();
        auto material = make_material(mesh);
        SimpleFluid::BoundaryConditionSet boundary_conditions;
        boundary_conditions.velocity["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet,
                                                {0.0, 0.0, 0.0}};
        boundary_conditions.velocity["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet,
                                                {1.0, 0.25, 0.0}};
        boundary_conditions.turbulence.turbulent_kinetic_energy["xmin"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.15};
        boundary_conditions.turbulence.turbulent_kinetic_energy["xmax"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.25};
        boundary_conditions.turbulence.dissipation_rate["xmin"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.04};
        boundary_conditions.turbulence.dissipation_rate["xmax"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.06};
        boundary_conditions.turbulence.specific_dissipation_rate["xmin"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 1.8};
        boundary_conditions.turbulence.specific_dissipation_rate["xmax"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 2.2};

        Model model(mesh, boundary_conditions);
        model.configure(make_model_options(entry.type), material, reference_density);

        double maximum_initial_k_gradient = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            const auto gradient =
                model.turbulent_kinetic_energy_gradient().value(cell_lid);
            maximum_initial_k_gradient = std::max(
                maximum_initial_k_gradient,
                std::sqrt(gradient.dot(gradient)));
        }
        EXPECT_GT(maximum_initial_k_gradient, 0.0);

        VectorFieldType velocity(mesh, "velocity");
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            const auto x = mesh->cell_centroid(cell_lid).x;
            velocity.set_owned_value(cell_lid, {x, 0.25 * x, 0.0});
        }
        velocity.sync_ghosts();
        const auto boundary_cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
        SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "projected_face_fluxes");

        const auto summary =
            model.advance(velocity, zero_fluxes, boundary_cache, 1.0e-4, material,
                          reference_density, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);

        ASSERT_TRUE(summary.converged);
        EXPECT_EQ(summary.solves, 2);
        const auto* secondary =
            entry.epsilon_family ? model.dissipation_rate() : model.specific_dissipation_rate();
        ASSERT_NE(secondary, nullptr);
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
            EXPECT_TRUE(std::isfinite(model.turbulent_kinetic_energy().value(cell_lid)));
            EXPECT_TRUE(std::isfinite(secondary->value(cell_lid)));
            EXPECT_TRUE(std::isfinite(model.turbulent_kinematic_viscosity().value(cell_lid)));
            const auto k_gradient =
                model.turbulent_kinetic_energy_gradient().value(cell_lid);
            EXPECT_TRUE(std::isfinite(k_gradient.x));
            EXPECT_TRUE(std::isfinite(k_gradient.y));
            EXPECT_TRUE(std::isfinite(k_gradient.z));
            EXPECT_GT(model.turbulent_kinetic_energy().value(cell_lid), 0.0);
            EXPECT_GT(secondary->value(cell_lid), 0.0);
            EXPECT_GT(model.turbulent_kinematic_viscosity().value(cell_lid), 0.0);
        }
    }
}

/** @brief Verifies boundary-aware shear and solve reporting for standard k-epsilon. */
TEST(TurbulenceModelTest, StandardKEpsilonAdvanceUsesBoundaryAwareShearAndReportsTwoSolves)
{
    auto mesh = make_two_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.velocity["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet,
                                            {0.0, 0.0, 0.0}};
    boundary_conditions.velocity["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet,
                                            {2.0, 0.5, -0.2}};
    ASSERT_TRUE(boundary_conditions.turbulence.turbulent_kinetic_energy.empty());
    ASSERT_TRUE(boundary_conditions.turbulence.dissipation_rate.empty());

    Model model(mesh, boundary_conditions);
    const auto options = make_model_options(ModelType::StandardKEpsilon);
    model.configure(options, material, reference_density);

    VectorFieldType velocity(mesh, "velocity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const auto x = mesh->cell_centroid(cell_lid).x;
        velocity.set_owned_value(cell_lid, {x, 0.25 * x, -0.1 * x});
    }
    velocity.sync_ghosts();

    const auto velocity_boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundary_conditions);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "projected_face_fluxes");
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;

    const auto summary = model.advance(
        velocity, zero_fluxes, velocity_boundary_cache, 1.0e-2, material, reference_density,
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, linear_options);

    ASSERT_TRUE(summary.converged);
    EXPECT_EQ(summary.solves, 2);
    EXPECT_GE(summary.iterations, 0);
    EXPECT_TRUE(std::isfinite(summary.achieved_tolerance));
    EXPECT_GE(summary.achieved_tolerance, 0.0);

    const auto* epsilon = model.dissipation_rate();
    ASSERT_NE(epsilon, nullptr);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const std::array values{model.turbulent_kinetic_energy().value(cell_lid),
                                epsilon->value(cell_lid),
                                model.turbulent_kinematic_viscosity().value(cell_lid),
                                model.effective_dynamic_viscosity().value(cell_lid),
                                model.effective_thermal_conductivity().value(cell_lid)};
        for (const auto value : values)
        {
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GT(value, 0.0);
        }
        EXPECT_GT(model.turbulent_kinetic_energy().value(cell_lid),
                  options.initial_turbulent_kinetic_energy);
        EXPECT_GT(epsilon->value(cell_lid), options.initial_dissipation_rate);
    }
}

/** @brief Verifies high-Re epsilon constraints and publication of wall-face properties. */
TEST(TurbulenceModelTest, HighReWallTreatmentConstrainsEpsilonAndPublishesFaceProperties)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};

    auto options = make_model_options(ModelType::StandardKEpsilon);
    options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    options.wall_options.boundary_names = {"xmin"};
    Model model(mesh, boundary_conditions);
    model.configure(options, material, reference_density);

    ASSERT_NE(model.wall_y_plus(), nullptr);
    EXPECT_EQ(model.output_fields().at("wall_y_plus"), model.wall_y_plus());
    const auto* initial_mu_cache =
        model.effective_dynamic_viscosity_boundary_cache();
    const auto* initial_lambda_cache =
        model.effective_thermal_conductivity_boundary_cache();
    ASSERT_NE(initial_mu_cache, nullptr);
    ASSERT_NE(initial_lambda_cache, nullptr);
    ASSERT_EQ(initial_mu_cache->value.size(), 1U);

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_owned_value(0, {0.0, 2.0, 0.0});
    velocity.sync_ghosts();
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "projected_face_fluxes");

    const auto initial_k = options.initial_turbulent_kinetic_energy;
    const auto expected_epsilon =
        std::pow(options.wall_options.c_mu, 0.75) *
        initial_k * std::sqrt(initial_k) /
        (options.wall_options.kappa * 0.5);
    const auto summary = model.advance(
        velocity, zero_fluxes, velocity_cache, 1.0e-4, material,
        reference_density, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);

    ASSERT_TRUE(summary.converged);
    ASSERT_NE(model.dissipation_rate(), nullptr);
    EXPECT_NEAR(model.dissipation_rate()->value(0), expected_epsilon, 1.0e-11);
    ASSERT_NE(model.wall_y_plus(), nullptr);
    EXPECT_GT(model.wall_y_plus()->value(0), 0.0);
    const auto* wall_mu = model.effective_dynamic_viscosity_boundary_cache();
    ASSERT_NE(wall_mu, nullptr);
    ASSERT_EQ(wall_mu->value.size(), 1U);
    EXPECT_GT(wall_mu->value.begin()->second.at(0), molecular_viscosity);

    const auto k_before_second = model.turbulent_kinetic_energy().value(0);
    const auto y_plus_before_second = model.wall_y_plus()->value(0);
    const auto mu_before_second = wall_mu->value.begin()->second.at(0);
    const auto lambda_before_second =
        model.effective_thermal_conductivity_boundary_cache()
            ->value.begin()->second.at(0);
    velocity.set_owned_value(0, {0.0, 4.0, 0.0});
    velocity.sync_ghosts();
    const auto expected_second_epsilon =
        std::pow(options.wall_options.c_mu, 0.75) *
        k_before_second * std::sqrt(k_before_second) /
        (options.wall_options.kappa * 0.5);
    const auto second_summary = model.advance(
        velocity, zero_fluxes, velocity_cache, 1.0e-4, material,
        reference_density, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);

    ASSERT_TRUE(second_summary.converged);
    EXPECT_NEAR(model.dissipation_rate()->value(0), expected_second_epsilon,
                1.0e-11);
    EXPECT_NE(model.turbulent_kinetic_energy().value(0), k_before_second);
    EXPECT_NE(model.wall_y_plus()->value(0), y_plus_before_second);
    EXPECT_NE(model.effective_dynamic_viscosity_boundary_cache()
                  ->value.begin()->second.at(0),
              mu_before_second);
    EXPECT_NE(model.effective_thermal_conductivity_boundary_cache()
                  ->value.begin()->second.at(0),
              lambda_before_second);

    EXPECT_TRUE(model.disable());
    EXPECT_EQ(model.options().wall_treatment,
              SimpleFluid::TurbulenceWallTreatmentType::None);
    EXPECT_TRUE(model.options().wall_options.boundary_names.empty());
    EXPECT_EQ(model.wall_y_plus(), nullptr);
    EXPECT_EQ(model.effective_dynamic_viscosity_boundary_cache(), nullptr);
}

/** @brief Verifies propagation of high-Re wall C-mu into the standard k-epsilon closure. */
TEST(TurbulenceModelTest, HighReWallCmuAlsoConfiguresItsStandardKEpsilonClosure)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};

    auto options = make_model_options(ModelType::StandardKEpsilon);
    options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    options.wall_options.boundary_names = {"xmin"};
    options.wall_options.c_mu = 0.08;

    Model model(mesh, boundary_conditions);
    model.configure(options, material, reference_density);

    const auto expected_nu_t =
        options.wall_options.c_mu *
        options.initial_turbulent_kinetic_energy *
        options.initial_turbulent_kinetic_energy /
        options.initial_dissipation_rate;
    EXPECT_DOUBLE_EQ(model.turbulent_kinematic_viscosity().value(0),
                     expected_nu_t);
}

/** @brief Verifies molecular wall transport under resolved SST treatment. */
TEST(TurbulenceModelTest, ResolvedSSTWallTreatmentKeepsWallTransportMolecular)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    boundary_conditions.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    boundary_conditions.turbulence.turbulent_kinetic_energy["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    auto options = make_model_options(ModelType::SSTKOmega);
    options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReSST;
    options.wall_options.boundary_names = {"xmin"};
    Model model(mesh, boundary_conditions);
    model.configure(options, material, reference_density);

    const auto* wall_mu = model.effective_dynamic_viscosity_boundary_cache();
    const auto* wall_lambda = model.effective_thermal_conductivity_boundary_cache();
    ASSERT_NE(wall_mu, nullptr);
    ASSERT_NE(wall_lambda, nullptr);
    EXPECT_NEAR(wall_mu->value.begin()->second.at(0), molecular_viscosity, 1.0e-15);
    EXPECT_NEAR(wall_lambda->value.begin()->second.at(0), molecular_conductivity, 1.0e-15);

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_owned_value(0, {0.0, 1.0, 0.0});
    velocity.sync_ghosts();
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "projected_face_fluxes");
    const auto summary = model.advance(
        velocity, zero_fluxes, velocity_cache, 1.0e-6, material,
        reference_density, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);

    ASSERT_TRUE(summary.converged);
    ASSERT_NE(model.specific_dissipation_rate(), nullptr);
    EXPECT_GT(model.turbulent_kinetic_energy().value(0), 0.0);
    EXPECT_GT(model.specific_dissipation_rate()->value(0), 0.0);
    ASSERT_NE(model.wall_y_plus(), nullptr);
    EXPECT_GT(model.wall_y_plus()->value(0), 0.0);
    wall_mu = model.effective_dynamic_viscosity_boundary_cache();
    wall_lambda = model.effective_thermal_conductivity_boundary_cache();
    EXPECT_NEAR(wall_mu->value.begin()->second.at(0), molecular_viscosity, 1.0e-15);
    EXPECT_NEAR(wall_lambda->value.begin()->second.at(0), molecular_conductivity, 1.0e-15);
}

/** @brief Verifies rejection of unknown and non-no-slip wall-treatment boundaries. */
TEST(TurbulenceModelTest, WallTreatmentRejectsUnknownAndNonNoSlipBoundaries)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_material(mesh);
    auto options = make_model_options(ModelType::StandardKEpsilon);
    options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    options.wall_options.boundary_names = {"xmin"};

    SimpleFluid::BoundaryConditionSet slip;
    slip.velocity["xmin"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    Model slip_model(mesh, slip);
    EXPECT_THROW(slip_model.configure(options, material, reference_density),
                 std::invalid_argument);

    options.wall_options.boundary_names = {"notAPatch"};
    SimpleFluid::BoundaryConditionSet missing;
    missing.velocity["notAPatch"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    Model missing_model(mesh, missing);
    EXPECT_THROW(missing_model.configure(options, material, reference_density),
                 std::invalid_argument);
}

} // namespace
