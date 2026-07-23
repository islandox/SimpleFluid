/**
 * @file testTurbulenceBuoyancy.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Direct tests for turbulence buoyancy and automatic wall distance.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/turbulence/TurbulenceModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Mesh = SimpleFluid::Mesh<Pack>;
using Field = SimpleFluid::CellField<Pack>;
using VectorField = SimpleFluid::VectorCellField<Pack>;
using Material = SimpleFluid::MaterialPropertyFields<Pack>;
using Model = SimpleFluid::TurbulenceModel<Pack>;
using ModelType = SimpleFluid::TurbulenceModelType;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

struct ClosureCase
{
    const char* name;
    ModelType model;
    bool menter_family;
};

constexpr std::array closure_cases{
    ClosureCase{"standardKEpsilon", ModelType::StandardKEpsilon, false},
    ClosureCase{"RNGKEpsilon", ModelType::RNGKEpsilon, false},
    ClosureCase{"realizableKEpsilon", ModelType::RealizableKEpsilon, false},
    ClosureCase{"standardKOmega", ModelType::StandardKOmega, false},
    ClosureCase{"BSLKOmega", ModelType::BSLKOmega, true},
    ClosureCase{"SSTKOmega", ModelType::SSTKOmega, true}};

constexpr double reference_density = 1000.0;
constexpr double turbulent_prandtl = 0.8;
constexpr double buoyancy_coefficient = 1.25;
constexpr double thermal_expansion = 5.0e-3;
constexpr double temperature_intercept = 10.0;
constexpr double temperature_slope = 2.0;

SimpleFluid::SP<Mesh> make_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

Material make_material(SimpleFluid::SP<const Mesh> mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = reference_density;
    options.density = 997.0;
    options.specific_heat_capacity = 4180.0;
    options.dynamic_viscosity = 1.0e-3;
    options.thermal_conductivity = 0.6;
    return Material(std::move(mesh), options, time_options);
}

SimpleFluid::BoundaryConditionSet slip_box_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {
            SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    return boundaries;
}

SimpleFluid::BoundaryConditionSet zero_gradient_box_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {
            SimpleFluid::BoundaryConditionType::Neumann, {}};
    }
    return boundaries;
}

SimpleFluid::BoundaryConditionMap linear_temperature_boundaries()
{
    SimpleFluid::BoundaryConditionMap conditions;
    conditions["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        temperature_intercept};
    conditions["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        temperature_intercept + 2.0 * temperature_slope};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
    {
        conditions[name] = {
            SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    return conditions;
}

SimpleFluid::TurbulenceModelOptions buoyancy_options(
    const ClosureCase& closure)
{
    SimpleFluid::TurbulenceModelOptions options;
    options.model = closure.model;
    options.initial_turbulent_kinetic_energy = 0.2;
    options.initial_dissipation_rate = 0.05;
    options.initial_specific_dissipation_rate = 2.0;
    options.min_turbulent_kinetic_energy = 1.0e-12;
    options.min_dissipation_rate = 1.0e-12;
    options.min_specific_dissipation_rate = 1.0e-12;
    options.turbulent_prandtl_number = turbulent_prandtl;
    options.buoyancy_model =
        SimpleFluid::TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    options.buoyancy_coefficient = buoyancy_coefficient;
    if (closure.menter_family)
    {
        options.initial_wall_distance = 0.5;
    }
    return options;
}

void initialize_linear_temperature(Field& temperature)
{
    const auto& mesh = temperature.mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto x = mesh.cell_centroid(cell_lid).x;
        temperature.set_owned_value(
            cell_lid,
            temperature_intercept + temperature_slope * x);
    }
    temperature.sync_ghosts();
}

SimpleFluid::LinearSolveSummary advance(
    Model& model,
    const SimpleFluid::BoundaryConditionSet& boundaries,
    Material& material,
    const SimpleFluid::TurbulenceBuoyancyContext<Pack>* context,
    VectorField::vec_type velocity_value = {})
{
    const auto mesh = material.dynamic_viscosity.mesh_ptr();
    VectorField velocity(mesh, velocity_value, "velocity");
    velocity.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(
        mesh, 0.0, "projected_face_fluxes");
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundaries);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;
    linear_options.max_iterations = 400;
    return model.advance(
        velocity, zero_fluxes, velocity_cache, 1.0e-6, material,
        reference_density,
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
        linear_options, context);
}

/**
 * @brief k-epsilon C3 follows OpenFOAM's Boussinesq fvOption orientation.
 */
TEST(TurbulenceBuoyancyTest,
     KEpsilonC3CouplingDistinguishesParallelPerpendicularAndZeroVelocity)
{
    auto mesh = make_mesh();
    auto material = make_material(mesh);
    const auto boundaries = zero_gradient_box_boundaries();
    const ClosureCase closure{
        "standardKEpsilon", ModelType::StandardKEpsilon, false};
    const auto options = buoyancy_options(closure);

    Field temperature(mesh, "temperature");
    initialize_linear_temperature(temperature);
    const auto temperature_boundaries =
        linear_temperature_boundaries();
    SimpleFluid::TurbulenceBuoyancyContext<Pack> context;
    context.temperature = &temperature;
    context.temperature_boundary_conditions =
        &temperature_boundaries;
    context.gravity = {3.0, 0.0, 0.0};
    context.thermal_expansion = thermal_expansion;

    auto advance_with_velocity =
        [&](VectorField::vec_type velocity,
            bool direct_buoyancy)
    {
        auto local_options = options;
        if (!direct_buoyancy)
        {
            local_options.buoyancy_model =
                SimpleFluid::TurbulenceBuoyancyModel::None;
        }
        Model model(mesh, boundaries);
        model.configure(
            local_options, material, reference_density);
        const auto summary = advance(
            model, boundaries, material, &context, velocity);
        EXPECT_TRUE(summary.converged);
        const auto* epsilon = model.dissipation_rate();
        EXPECT_NE(epsilon, nullptr);
        return std::pair{
            model.turbulent_kinetic_energy().value(0),
            epsilon != nullptr
                ? epsilon->value(0)
                : std::numeric_limits<double>::quiet_NaN()};
    };

    const auto baseline =
        advance_with_velocity({1.0, 0.0, 0.0}, false);
    const auto parallel =
        advance_with_velocity({1.0, 0.0, 0.0}, true);
    const auto perpendicular =
        advance_with_velocity({0.0, 1.0, 0.0}, true);
    const auto zero =
        advance_with_velocity({}, true);

    // OpenFOAM fv::buoyancyTurbSource uses
    // C3 = tanh((|U_perpendicular| + SMALL) / |U_parallel|).
    EXPECT_NEAR(parallel.second, baseline.second, 1.0e-13);
    EXPECT_GT(perpendicular.second - baseline.second, 1.0e-12);
    EXPECT_GT(zero.second - baseline.second, 1.0e-12);
    EXPECT_NEAR(perpendicular.second, zero.second, 1.0e-13);

    // C3 modifies only epsilon; buoyancy production in k is orientation-free.
    EXPECT_NEAR(parallel.first, perpendicular.first, 1.0e-13);
    EXPECT_NEAR(parallel.first, zero.first, 1.0e-13);
}

/** @brief Every closure publishes the signed OpenFOAM Boussinesq source. */
TEST(TurbulenceBuoyancyTest,
     EveryClosureMatchesSignedTemperatureGradientProduction)
{
    for (const auto& closure : closure_cases)
    {
        for (const double gravity_x : {-3.0, 3.0})
        {
            SCOPED_TRACE(
                std::string{closure.name}
                + ", gravity_x=" + std::to_string(gravity_x));
            auto mesh = make_mesh();
            auto material = make_material(mesh);
            const auto boundaries = zero_gradient_box_boundaries();
            Model model(mesh, boundaries);
            const auto options = buoyancy_options(closure);
            model.configure(options, material, reference_density);

            Field temperature(mesh, "temperature");
            initialize_linear_temperature(temperature);
            const auto temperature_boundaries =
                linear_temperature_boundaries();
            SimpleFluid::TurbulenceBuoyancyContext<Pack> context;
            context.temperature = &temperature;
            context.temperature_boundary_conditions =
                &temperature_boundaries;
            context.gravity = {gravity_x, 0.0, 0.0};
            context.thermal_expansion = thermal_expansion;

            const auto summary =
                advance(
                    model, boundaries, material, &context,
                    {0.0, 1.0, 0.0});
            ASSERT_TRUE(summary.converged);
            EXPECT_EQ(summary.solves, 2);

            auto baseline_options = options;
            baseline_options.buoyancy_model =
                SimpleFluid::TurbulenceBuoyancyModel::None;
            Model baseline(mesh, boundaries);
            baseline.configure(
                baseline_options, material, reference_density);
            const auto baseline_summary =
                advance(
                    baseline, boundaries, material, &context,
                    {0.0, 1.0, 0.0});
            ASSERT_TRUE(baseline_summary.converged);

            const auto* production = model.buoyancy_production();
            ASSERT_NE(production, nullptr);
            EXPECT_EQ(
                model.output_fields().at("buoyancy_production"),
                production);

            for (size_t owned = 0;
                 owned < mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<Pack::local_ordinal_type>(owned);
                const auto nu_t =
                    model.turbulent_kinematic_viscosity().value(cell_lid);
                const auto expected =
                    buoyancy_coefficient * thermal_expansion
                  * (nu_t / turbulent_prandtl)
                  * gravity_x * temperature_slope;
                const auto actual = production->value(cell_lid);
                EXPECT_NEAR(
                    actual, expected,
                    std::max(1.0e-13, std::abs(expected) * 1.0e-10));
                if (gravity_x > 0.0)
                {
                    EXPECT_GT(actual, 0.0);
                }
                else
                {
                    EXPECT_LT(actual, 0.0);
                }

                const auto k =
                    model.turbulent_kinetic_energy().value(cell_lid);
                const auto* secondary =
                    closure.menter_family
                        || closure.model == ModelType::StandardKOmega
                      ? model.specific_dissipation_rate()
                      : model.dissipation_rate();
                ASSERT_NE(secondary, nullptr);
                const auto* baseline_secondary =
                    closure.menter_family
                        || closure.model == ModelType::StandardKOmega
                      ? baseline.specific_dissipation_rate()
                      : baseline.dissipation_rate();
                ASSERT_NE(baseline_secondary, nullptr);
                EXPECT_TRUE(std::isfinite(k));
                EXPECT_TRUE(std::isfinite(secondary->value(cell_lid)));
                EXPECT_GT(k, 0.0);
                EXPECT_GT(secondary->value(cell_lid), 0.0);
                EXPECT_GT(
                    std::abs(
                        k
                        - baseline.turbulent_kinetic_energy().value(
                            cell_lid)),
                    1.0e-13);
                EXPECT_GT(
                    std::abs(
                        secondary->value(cell_lid)
                        - baseline_secondary->value(cell_lid)),
                    1.0e-13);
                if (gravity_x < 0.0)
                {
                    EXPECT_LT(
                        k,
                        options.initial_turbulent_kinetic_energy);
                }
            }
        }
    }
}

/** @brief Density feedback implements the equivalent Boussinesq source. */
TEST(TurbulenceBuoyancyTest,
     DensityGradientFeedbackMatchesSignedReferenceDensityForm)
{
    constexpr double density_slope =
        -reference_density * thermal_expansion * temperature_slope;
    for (const double gravity_x : {-2.0, 2.0})
    {
        SCOPED_TRACE("gravity_x=" + std::to_string(gravity_x));
        auto mesh = make_mesh();
        auto material = make_material(mesh);
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<Pack::local_ordinal_type>(owned);
            material.density.set_owned_value(
                cell_lid,
                reference_density
                + density_slope * mesh->cell_centroid(cell_lid).x);
        }
        material.density.sync_ghosts();

        const auto boundaries = slip_box_boundaries();
        Model model(mesh, boundaries);
        const ClosureCase closure{
            "standardKEpsilon", ModelType::StandardKEpsilon, false};
        const auto options = buoyancy_options(closure);
        model.configure(options, material, reference_density);

        SimpleFluid::TurbulenceBuoyancyContext<Pack> context;
        context.gravity = {gravity_x, 0.0, 0.0};
        context.thermal_expansion = thermal_expansion;
        context.density_feedback_enabled = true;
        const auto summary =
            advance(model, boundaries, material, &context);
        ASSERT_TRUE(summary.converged);

        const auto* production = model.buoyancy_production();
        ASSERT_NE(production, nullptr);
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<Pack::local_ordinal_type>(owned);
            const auto nu_t =
                model.turbulent_kinematic_viscosity().value(cell_lid);
            const auto expected =
                -buoyancy_coefficient
              * (nu_t / turbulent_prandtl)
              * gravity_x * density_slope / reference_density;
            EXPECT_NEAR(
                production->value(cell_lid), expected,
                std::max(1.0e-13, std::abs(expected) * 1.0e-10));
            EXPECT_EQ(
                production->value(cell_lid) > 0.0,
                gravity_x > 0.0);
        }
    }
}

/** @brief Direct buoyancy rejects absent, incomplete, or foreign contexts. */
TEST(TurbulenceBuoyancyTest, RejectsInvalidBuoyancyContexts)
{
    auto mesh = make_mesh();
    auto material = make_material(mesh);
    const auto boundaries = slip_box_boundaries();
    Model model(mesh, boundaries);
    const ClosureCase closure{
        "standardKEpsilon", ModelType::StandardKEpsilon, false};
    model.configure(
        buoyancy_options(closure), material, reference_density);

    EXPECT_THROW(
        advance(model, boundaries, material, nullptr),
        std::invalid_argument);

    Field temperature(mesh, "temperature");
    initialize_linear_temperature(temperature);
    SimpleFluid::TurbulenceBuoyancyContext<Pack> missing_boundaries;
    missing_boundaries.temperature = &temperature;
    missing_boundaries.gravity = {1.0, 0.0, 0.0};
    missing_boundaries.thermal_expansion = thermal_expansion;
    EXPECT_THROW(
        advance(model, boundaries, material, &missing_boundaries),
        std::invalid_argument);

    const auto temperature_boundaries =
        linear_temperature_boundaries();
    auto negative_expansion = missing_boundaries;
    negative_expansion.temperature_boundary_conditions =
        &temperature_boundaries;
    negative_expansion.thermal_expansion = -thermal_expansion;
    EXPECT_THROW(
        advance(model, boundaries, material, &negative_expansion),
        std::invalid_argument);

    auto other_mesh = make_mesh();
    Field other_temperature(other_mesh, 10.0, "other_temperature");
    auto foreign_temperature = missing_boundaries;
    foreign_temperature.temperature = &other_temperature;
    foreign_temperature.temperature_boundary_conditions =
        &temperature_boundaries;
    EXPECT_THROW(
        advance(model, boundaries, material, &foreign_temperature),
        std::invalid_argument);
}

/** @brief BSL/SST configure and advance with reconstructed wall distance. */
TEST(TurbulenceBuoyancyTest,
     MenterModelsAutomaticallyReconstructAndUseWallDistance)
{
    for (const auto& [model_type, explicit_selection] :
         std::array{
             std::pair{ModelType::BSLKOmega, false},
             std::pair{ModelType::SSTKOmega, true}})
    {
        SCOPED_TRACE(SimpleFluid::to_string(model_type));
        auto mesh = make_mesh();
        auto material = make_material(mesh);
        auto boundaries = slip_box_boundaries();
        boundaries.velocity["xmin"] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};

        Model model(mesh, boundaries);
        SimpleFluid::TurbulenceModelOptions options;
        options.model = model_type;
        options.initial_turbulent_kinetic_energy = 0.2;
        options.initial_specific_dissipation_rate = 2.0;
        options.min_turbulent_kinetic_energy = 1.0e-12;
        options.min_specific_dissipation_rate = 1.0e-12;
        if (explicit_selection)
        {
            options.wall_distance_boundaries = {"xmin"};
        }
        ASSERT_FALSE(options.initial_wall_distance.has_value());
        ASSERT_NO_THROW(
            model.configure(options, material, reference_density));

        const auto* distance = model.wall_distance();
        ASSERT_NE(distance, nullptr);
        EXPECT_EQ(model.output_fields().at("wall_distance"), distance);
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_TRUE(std::isfinite(distance->value(cell_lid)));
            EXPECT_GT(distance->value(cell_lid), 0.0);
        }

        const auto summary =
            advance(model, boundaries, material, nullptr);
        ASSERT_TRUE(summary.converged);
        ASSERT_NE(model.specific_dissipation_rate(), nullptr);
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_TRUE(std::isfinite(
                model.turbulent_kinetic_energy().value(cell_lid)));
            EXPECT_TRUE(std::isfinite(
                model.specific_dissipation_rate()->value(cell_lid)));
            EXPECT_GT(
                model.turbulent_kinetic_energy().value(cell_lid), 0.0);
            EXPECT_GT(
                model.specific_dissipation_rate()->value(cell_lid), 0.0);
        }
    }
}

/** @brief Automatic distance requires at least one globally valid wall. */
TEST(TurbulenceBuoyancyTest, RejectsMissingOrUnknownAutomaticWallSelection)
{
    auto mesh = make_mesh();
    auto material = make_material(mesh);
    const auto boundaries = slip_box_boundaries();
    Model model(mesh, boundaries);

    SimpleFluid::TurbulenceModelOptions options;
    options.model = ModelType::BSLKOmega;
    EXPECT_THROW(
        model.configure(options, material, reference_density),
        std::invalid_argument);
    EXPECT_FALSE(model.enabled());

    options.model = ModelType::SSTKOmega;
    options.wall_distance_boundaries = {"not_a_boundary"};
    EXPECT_THROW(
        model.configure(options, material, reference_density),
        std::invalid_argument);
    EXPECT_FALSE(model.enabled());
}

} // namespace
