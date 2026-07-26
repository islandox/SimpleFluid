/**
 * @file testTurbulenceModelOptions.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for turbulence model configuration and parsing.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/turbulence/TurbulenceModel.hh"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

using SimpleFluid::TurbulenceModelOptions;
using SimpleFluid::TurbulenceModelType;
using SimpleFluid::TurbulenceBuoyancyModel;
using SimpleFluid::TurbulenceWallTreatmentType;

/** @brief Database spelling and expected runtime turbulence model type. */
struct ModelCase
{
    const char* name;
    TurbulenceModelType model;
    bool requires_wall_distance = false;
};

constexpr std::array canonical_models{
    ModelCase{"laminar", TurbulenceModelType::Laminar},
    ModelCase{"standardKEpsilon", TurbulenceModelType::StandardKEpsilon},
    ModelCase{"RNGKEpsilon", TurbulenceModelType::RNGKEpsilon},
    ModelCase{"realizableKEpsilon", TurbulenceModelType::RealizableKEpsilon},
    ModelCase{"standardKOmega", TurbulenceModelType::StandardKOmega},
    ModelCase{"BSLKOmega", TurbulenceModelType::BSLKOmega, true},
    ModelCase{"SSTKOmega", TurbulenceModelType::SSTKOmega, true}};

/** @brief Verifies parsing and formatting of canonical turbulence-model names. */
TEST(TurbulenceModelOptionsTest, ParsesAndFormatsCanonicalNames)
{
    for (const auto& entry : canonical_models)
    {
        SCOPED_TRACE(entry.name);
        EXPECT_EQ(SimpleFluid::parse_turbulence_model_type(entry.name), entry.model);
        EXPECT_EQ(SimpleFluid::to_string(entry.model), entry.name);
    }
}

/** @brief Verifies model aliases and rejection of unknown names. */
TEST(TurbulenceModelOptionsTest, ParsesAliasesAndRejectsUnknownNames)
{
    const std::pair<const char*, TurbulenceModelType> aliases[] = {
        {"none", TurbulenceModelType::Laminar},
        {"kEpsilon", TurbulenceModelType::StandardKEpsilon},
        {"rngKEpsilon", TurbulenceModelType::RNGKEpsilon},
        {"kOmega", TurbulenceModelType::StandardKOmega},
        {"bslKOmega", TurbulenceModelType::BSLKOmega},
        {"sstKOmega", TurbulenceModelType::SSTKOmega}};
    for (const auto& [name, expected] : aliases)
    {
        SCOPED_TRACE(name);
        EXPECT_EQ(SimpleFluid::parse_turbulence_model_type(name), expected);
    }

    for (const auto* name : {"", "standard", "SpalartAllmaras", "SST"})
    {
        SCOPED_TRACE(name);
        EXPECT_THROW(SimpleFluid::parse_turbulence_model_type(name), std::invalid_argument);
    }
}

/** @brief Verifies parsing and formatting of wall-treatment names. */
TEST(TurbulenceModelOptionsTest, ParsesAndFormatsWallTreatmentNames)
{
    const std::pair<const char*, TurbulenceWallTreatmentType> treatments[] = {
        {"none", TurbulenceWallTreatmentType::None},
        {"resolvedLowReSST", TurbulenceWallTreatmentType::ResolvedLowReSST},
        {"standardHighReKEpsilon",
         TurbulenceWallTreatmentType::StandardHighReKEpsilon}};
    for (const auto& [name, treatment] : treatments)
    {
        EXPECT_EQ(SimpleFluid::parse_turbulence_wall_treatment_type(name), treatment);
        EXPECT_EQ(SimpleFluid::to_string(treatment), name);
    }
    EXPECT_EQ(SimpleFluid::parse_turbulence_wall_treatment_type("resolvedSST"),
              TurbulenceWallTreatmentType::ResolvedLowReSST);
    EXPECT_EQ(SimpleFluid::parse_turbulence_wall_treatment_type("OpenFOAMKEpsilon"),
              TurbulenceWallTreatmentType::StandardHighReKEpsilon);
    EXPECT_THROW(SimpleFluid::parse_turbulence_wall_treatment_type("automatic"),
                 std::invalid_argument);
}

/** @brief Verifies parsing and formatting of direct buoyancy-production names. */
TEST(TurbulenceModelOptionsTest, ParsesAndFormatsBuoyancyModelNames)
{
    EXPECT_EQ(SimpleFluid::parse_turbulence_buoyancy_model("none"),
              TurbulenceBuoyancyModel::None);
    EXPECT_EQ(SimpleFluid::parse_turbulence_buoyancy_model("off"),
              TurbulenceBuoyancyModel::None);
    for (const auto* name :
         {"OpenFOAMBoussinesq", "openFOAMBoussinesq", "boussinesq"})
    {
        SCOPED_TRACE(name);
        EXPECT_EQ(SimpleFluid::parse_turbulence_buoyancy_model(name),
                  TurbulenceBuoyancyModel::OpenFOAMBoussinesq);
    }
    EXPECT_EQ(SimpleFluid::to_string(TurbulenceBuoyancyModel::None), "none");
    EXPECT_EQ(
        SimpleFluid::to_string(
            TurbulenceBuoyancyModel::OpenFOAMBoussinesq),
        "OpenFOAMBoussinesq");
    EXPECT_THROW(
        SimpleFluid::parse_turbulence_buoyancy_model("densityOnly"),
        std::invalid_argument);
}

/** @brief Verifies laminar defaults for an empty database. */
TEST(TurbulenceModelOptionsTest, EmptyDatabaseSelectsLaminarDefaults)
{
    const SimpleFluid::Database database;
    const auto options = SimpleFluid::turbulence_model_options_from_database(database);
    const TurbulenceModelOptions defaults;

    EXPECT_EQ(options.model, TurbulenceModelType::Laminar);
    EXPECT_DOUBLE_EQ(options.initial_turbulent_kinetic_energy,
                     defaults.initial_turbulent_kinetic_energy);
    EXPECT_DOUBLE_EQ(options.initial_dissipation_rate, defaults.initial_dissipation_rate);
    EXPECT_DOUBLE_EQ(options.initial_specific_dissipation_rate,
                     defaults.initial_specific_dissipation_rate);
    EXPECT_DOUBLE_EQ(options.min_turbulent_kinetic_energy, defaults.min_turbulent_kinetic_energy);
    EXPECT_DOUBLE_EQ(options.min_dissipation_rate, defaults.min_dissipation_rate);
    EXPECT_DOUBLE_EQ(options.min_specific_dissipation_rate, defaults.min_specific_dissipation_rate);
    EXPECT_DOUBLE_EQ(options.turbulent_prandtl_number, defaults.turbulent_prandtl_number);
    EXPECT_FALSE(options.initial_wall_distance.has_value());
    EXPECT_TRUE(options.wall_distance_boundaries.empty());
    EXPECT_EQ(
        options.wall_distance_equation.non_orthogonal_treatment,
        defaults.wall_distance_equation.non_orthogonal_treatment);
    EXPECT_EQ(
        options.wall_distance_equation.non_orthogonal_correctors,
        defaults.wall_distance_equation.non_orthogonal_correctors);
    EXPECT_EQ(
        options.wall_distance_equation.linear_solver.max_iterations,
        defaults.wall_distance_equation.linear_solver.max_iterations);
    EXPECT_DOUBLE_EQ(
        options.wall_distance_equation.linear_solver.tolerance,
        defaults.wall_distance_equation.linear_solver.tolerance);
    EXPECT_EQ(options.buoyancy_model, TurbulenceBuoyancyModel::None);
    EXPECT_DOUBLE_EQ(options.buoyancy_coefficient, 1.0);
    EXPECT_EQ(options.wall_treatment, TurbulenceWallTreatmentType::None);
    EXPECT_TRUE(options.wall_options.boundary_names.empty());
}

/** @brief Verifies database selection of every active turbulence model. */
TEST(TurbulenceModelOptionsTest, DatabaseSelectsEveryActiveModel)
{
    for (const auto& entry : canonical_models)
    {
        if (entry.model == TurbulenceModelType::Laminar)
        {
            continue;
        }
        SCOPED_TRACE(entry.name);
        SimpleFluid::Database database;
        database.set("turbulence_model", std::string{entry.name});

        const auto options = SimpleFluid::turbulence_model_options_from_database(database);
        EXPECT_EQ(options.model, entry.model);
        EXPECT_FALSE(options.initial_wall_distance.has_value());
    }
}

/** @brief Verifies parsing of positive scalar turbulence controls. */
TEST(TurbulenceModelOptionsTest, DatabaseReadsPositiveScalarControls)
{
    SimpleFluid::Database database;
    database.set("turbulence_model", std::string{"SSTKOmega"});
    database.set("initial_turbulent_kinetic_energy", SimpleFluid::real_t{0.1});
    database.set("initial_dissipation_rate", SimpleFluid::real_t{0.2});
    database.set("initial_specific_dissipation_rate", SimpleFluid::real_t{0.3});
    database.set("min_turbulent_kinetic_energy", SimpleFluid::real_t{1.0e-9});
    database.set("min_dissipation_rate", SimpleFluid::real_t{2.0e-9});
    database.set("min_specific_dissipation_rate", SimpleFluid::real_t{3.0e-9});
    database.set("turbulent_prandtl_number", SimpleFluid::real_t{0.85});
    database.set("wall_distance", SimpleFluid::real_t{0.4});

    const auto options = SimpleFluid::turbulence_model_options_from_database(database);
    EXPECT_EQ(options.model, TurbulenceModelType::SSTKOmega);
    EXPECT_DOUBLE_EQ(options.initial_turbulent_kinetic_energy, 0.1);
    EXPECT_DOUBLE_EQ(options.initial_dissipation_rate, 0.2);
    EXPECT_DOUBLE_EQ(options.initial_specific_dissipation_rate, 0.3);
    EXPECT_DOUBLE_EQ(options.min_turbulent_kinetic_energy, 1.0e-9);
    EXPECT_DOUBLE_EQ(options.min_dissipation_rate, 2.0e-9);
    EXPECT_DOUBLE_EQ(options.min_specific_dissipation_rate, 3.0e-9);
    EXPECT_DOUBLE_EQ(options.turbulent_prandtl_number, 0.85);
    ASSERT_TRUE(options.initial_wall_distance.has_value());
    EXPECT_DOUBLE_EQ(*options.initial_wall_distance, 0.4);
}

/** @brief Verifies rejection of non-positive values for positive database controls. */
TEST(TurbulenceModelOptionsTest, RejectsInvalidPositiveDatabaseValues)
{
    constexpr std::array positive_keys{"initial_turbulent_kinetic_energy",
                                       "initial_dissipation_rate",
                                       "initial_specific_dissipation_rate",
                                       "min_turbulent_kinetic_energy",
                                       "min_dissipation_rate",
                                       "min_specific_dissipation_rate",
                                       "turbulent_prandtl_number"};
    const std::array invalid_values{SimpleFluid::real_t{0.0}, SimpleFluid::real_t{-1.0},
                                    std::numeric_limits<SimpleFluid::real_t>::infinity(),
                                    std::numeric_limits<SimpleFluid::real_t>::quiet_NaN()};

    for (const auto* key : positive_keys)
    {
        for (const auto value : invalid_values)
        {
            SCOPED_TRACE(key);
            SimpleFluid::Database database;
            database.set(key, value);
            EXPECT_THROW(SimpleFluid::turbulence_model_options_from_database(database),
                         std::invalid_argument);
        }
    }
}

/** @brief Verifies rejection of incorrectly typed database values. */
TEST(TurbulenceModelOptionsTest, RejectsWrongDatabaseTypes)
{
    SimpleFluid::Database wrong_model_type;
    wrong_model_type.set("turbulence_model", SimpleFluid::real_t{1.0});
    EXPECT_ANY_THROW(SimpleFluid::turbulence_model_options_from_database(wrong_model_type));

    constexpr std::array scalar_keys{"initial_turbulent_kinetic_energy",
                                     "initial_dissipation_rate",
                                     "initial_specific_dissipation_rate",
                                     "min_turbulent_kinetic_energy",
                                     "min_dissipation_rate",
                                     "min_specific_dissipation_rate",
                                     "turbulent_prandtl_number",
                                     "wall_distance"};
    for (const auto* key : scalar_keys)
    {
        SCOPED_TRACE(key);
        SimpleFluid::Database database;
        database.set(key, std::string{"not-a-number"});
        EXPECT_ANY_THROW(SimpleFluid::turbulence_model_options_from_database(database));
    }
}

/** @brief Verifies rejection of initial turbulence values below configured floors. */
TEST(TurbulenceModelOptionsTest, RejectsInitialValuesBelowTheirFloors)
{
    const std::pair<const char*, const char*> key_pairs[] = {
        {"initial_turbulent_kinetic_energy", "min_turbulent_kinetic_energy"},
        {"initial_dissipation_rate", "min_dissipation_rate"},
        {"initial_specific_dissipation_rate", "min_specific_dissipation_rate"}};
    for (const auto& [initial_key, floor_key] : key_pairs)
    {
        SCOPED_TRACE(initial_key);
        SimpleFluid::Database database;
        database.set(initial_key, SimpleFluid::real_t{0.5});
        database.set(floor_key, SimpleFluid::real_t{1.0});
        EXPECT_THROW(SimpleFluid::turbulence_model_options_from_database(database),
                     std::invalid_argument);

        database.set(floor_key, SimpleFluid::real_t{0.5});
        EXPECT_NO_THROW(SimpleFluid::turbulence_model_options_from_database(database));
    }
}

/** @brief Verifies automatic distance or a positive explicit Menter override. */
TEST(TurbulenceModelOptionsTest, BSLAndSSTAcceptAutomaticOrPositiveWallDistance)
{
    for (const auto& entry : canonical_models)
    {
        if (!entry.requires_wall_distance)
        {
            continue;
        }
        SCOPED_TRACE(entry.name);
        SimpleFluid::Database database;
        database.set("turbulence_model", std::string{entry.name});
        EXPECT_NO_THROW(
            SimpleFluid::turbulence_model_options_from_database(database));

        for (const auto invalid_distance : {SimpleFluid::real_t{0.0}, SimpleFluid::real_t{-1.0},
                                            std::numeric_limits<SimpleFluid::real_t>::infinity(),
                                            std::numeric_limits<SimpleFluid::real_t>::quiet_NaN()})
        {
            database.set("wall_distance", invalid_distance);
            EXPECT_THROW(SimpleFluid::turbulence_model_options_from_database(database),
                         std::invalid_argument);
        }

        database.set("wall_distance", SimpleFluid::real_t{0.25});
        EXPECT_NO_THROW(SimpleFluid::turbulence_model_options_from_database(database));
    }
}

/** @brief Verifies database controls for wall selection and direct buoyancy. */
TEST(TurbulenceModelOptionsTest, DatabaseReadsWallDistanceAndBuoyancyControls)
{
    SimpleFluid::Database database;
    database.set("turbulence_model", std::string{"SSTKOmega"});
    database.set(
        "wall_distance_boundaries",
        SimpleFluid::ArrString{"lowerWall", "upperWall"});
    database.set(
        "turbulence_buoyancy_model",
        std::string{"OpenFOAMBoussinesq"});
    database.set(
        "turbulence_buoyancy_coefficient",
        SimpleFluid::real_t{1.25});

    const auto options =
        SimpleFluid::turbulence_model_options_from_database(database);
    EXPECT_EQ(
        options.wall_distance_boundaries,
        (SimpleFluid::ArrString{"lowerWall", "upperWall"}));
    EXPECT_EQ(
        options.buoyancy_model,
        TurbulenceBuoyancyModel::OpenFOAMBoussinesq);
    EXPECT_DOUBLE_EQ(options.buoyancy_coefficient, 1.25);
}

/** @brief Verifies parsing of automatic Poisson wall-distance controls. */
TEST(TurbulenceModelOptionsTest, DatabaseReadsWallDistanceEquationControls)
{
    SimpleFluid::Database database;
    database.set("turbulence_model", std::string{"SSTKOmega"});
    database.set(
        "wall_distance_non_orthogonal_treatment",
        std::string{"hybrid"});
    database.set("wall_distance_non_orthogonal_correctors", 4);
    database.set("wall_distance_linear_solver_max_iterations", 321);
    database.set(
        "wall_distance_linear_solver_tolerance",
        SimpleFluid::real_t{2.5e-12});
    database.set("wall_distance_linear_solver_verbosity", 7);
    database.set(
        "wall_distance_linear_solver_backend",
        std::string{"cg"});
    database.set(
        "wall_distance_linear_solver_preconditioner",
        std::string{"jacobi"});
    database.set(
        "wall_distance_linear_solver_reuse_preconditioner", true);

    const auto options =
        SimpleFluid::turbulence_model_options_from_database(database);
    EXPECT_EQ(
        options.wall_distance_equation.non_orthogonal_treatment,
        SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid);
    EXPECT_EQ(
        options.wall_distance_equation.non_orthogonal_correctors, 4);
    EXPECT_EQ(
        options.wall_distance_equation.linear_solver.max_iterations, 321);
    EXPECT_DOUBLE_EQ(
        options.wall_distance_equation.linear_solver.tolerance, 2.5e-12);
    EXPECT_EQ(
        options.wall_distance_equation.linear_solver.verbosity, 7);
    EXPECT_EQ(
        options.wall_distance_equation.linear_solver.backend,
        SimpleFluid::LinearSolverBackend::Cg);
    EXPECT_EQ(
        options.wall_distance_equation.linear_solver.preconditioner,
        SimpleFluid::LinearPreconditioner::Jacobi);
    EXPECT_TRUE(
        options.wall_distance_equation.linear_solver
            .reuse_preconditioner);
}

/** @brief Rejects invalid automatic wall-distance numerical controls. */
TEST(TurbulenceModelOptionsTest, RejectsInvalidWallDistanceEquationControls)
{
    TurbulenceModelOptions options;
    options.model = TurbulenceModelType::SSTKOmega;

    options.wall_distance_equation.non_orthogonal_correctors = -1;
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);
    options.wall_distance_equation.non_orthogonal_correctors = 2;

    options.wall_distance_equation.linear_solver.max_iterations = 0;
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);
    options.wall_distance_equation.linear_solver.max_iterations = 200;

    options.wall_distance_equation.linear_solver.tolerance = 0.0;
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);

    SimpleFluid::Database invalid_treatment;
    invalid_treatment.set(
        "turbulence_model", std::string{"SSTKOmega"});
    invalid_treatment.set(
        "wall_distance_non_orthogonal_treatment",
        std::string{"deferred"});
    EXPECT_THROW(
        SimpleFluid::turbulence_model_options_from_database(
            invalid_treatment),
        std::invalid_argument);

    SimpleFluid::Database invalid_preconditioner;
    invalid_preconditioner.set(
        "turbulence_model", std::string{"SSTKOmega"});
    invalid_preconditioner.set(
        "wall_distance_linear_solver_preconditioner",
        std::string{"ILU"});
    EXPECT_THROW(
        SimpleFluid::turbulence_model_options_from_database(
            invalid_preconditioner),
        std::invalid_argument);

    SimpleFluid::Database invalid_backend;
    invalid_backend.set(
        "turbulence_model", std::string{"SSTKOmega"});
    invalid_backend.set(
        "wall_distance_linear_solver_backend",
        std::string{"direct"});
    EXPECT_THROW(
        SimpleFluid::turbulence_model_options_from_database(
            invalid_backend),
        std::invalid_argument);
}

/** @brief Rejects invalid buoyancy controls and wall-distance selections. */
TEST(TurbulenceModelOptionsTest, RejectsInvalidBuoyancyAndWallDistanceControls)
{
    TurbulenceModelOptions options;
    options.model = TurbulenceModelType::StandardKEpsilon;
    options.buoyancy_model =
        TurbulenceBuoyancyModel::OpenFOAMBoussinesq;

    EXPECT_NO_THROW(
        SimpleFluid::validate_turbulence_model_options(options));
    for (const auto coefficient :
         {SimpleFluid::real_t{0.0}, SimpleFluid::real_t{-1.0},
          std::numeric_limits<SimpleFluid::real_t>::infinity(),
          std::numeric_limits<SimpleFluid::real_t>::quiet_NaN()})
    {
        SCOPED_TRACE(coefficient);
        options.buoyancy_coefficient = coefficient;
        EXPECT_THROW(
            SimpleFluid::validate_turbulence_model_options(options),
            std::invalid_argument);
    }

    options.buoyancy_coefficient = 1.0;
    options.model = TurbulenceModelType::Laminar;
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);

    options.model = TurbulenceModelType::StandardKEpsilon;
    options.buoyancy_model = TurbulenceBuoyancyModel::None;
    options.wall_distance_boundaries = {"xmin"};
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);

    options.model = TurbulenceModelType::SSTKOmega;
    options.wall_distance_boundaries = {"xmin", "xmin"};
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);
    options.wall_distance_boundaries = {""};
    EXPECT_THROW(
        SimpleFluid::validate_turbulence_model_options(options),
        std::invalid_argument);

    SimpleFluid::Database invalid_database;
    invalid_database.set(
        "turbulence_model", std::string{"standardKEpsilon"});
    invalid_database.set(
        "turbulence_buoyancy_model", std::string{"unknown"});
    EXPECT_THROW(
        SimpleFluid::turbulence_model_options_from_database(
            invalid_database),
        std::invalid_argument);
}

/** @brief Verifies parsing of the wall set, treatment, and associated constants. */
TEST(TurbulenceModelOptionsTest, DatabaseReadsWallTreatmentSetAndConstants)
{
    SimpleFluid::Database database;
    database.set("turbulence_model", std::string{"standardKEpsilon"});
    database.set("wall_treatment", std::string{"standardHighReKEpsilon"});
    database.set("wall_boundaries", SimpleFluid::ArrString{"lowerWall", "upperWall"});
    database.set("wall_roughness_model",
                 SimpleFluid::ArrString{"smooth", "sandGrain"});
    database.set("wall_roughness_heights",
                 SimpleFluid::ArrReal{0.0, 2.0e-4});
    database.set("wall_roughness_constants",
                 SimpleFluid::ArrReal{0.0, 0.5});
    database.set("wall_c_mu", SimpleFluid::real_t{0.08});
    database.set("wall_kappa", SimpleFluid::real_t{0.42});
    database.set("wall_e", SimpleFluid::real_t{9.7});
    database.set("wall_thermal_law", std::string{"Jayatilleke"});
    database.set("wall_thermal_turbulent_prandtl_number",
                 SimpleFluid::real_t{0.85});
    database.set("wall_epsilon_low_re_correction", true);
    database.set("wall_beta_1", SimpleFluid::real_t{0.076});
    database.set("wall_sst_omega_face_coefficient", SimpleFluid::real_t{60.0});

    const auto options = SimpleFluid::turbulence_model_options_from_database(database);
    EXPECT_EQ(options.wall_treatment,
              TurbulenceWallTreatmentType::StandardHighReKEpsilon);
    EXPECT_EQ(options.wall_options.boundary_names,
              (SimpleFluid::ArrString{"lowerWall", "upperWall"}));
    ASSERT_EQ(options.wall_options.roughness_models.size(), 2);
    EXPECT_EQ(options.wall_options.roughness_models[0],
              SimpleFluid::TurbulenceWallRoughnessModel::Smooth);
    EXPECT_EQ(options.wall_options.roughness_models[1],
              SimpleFluid::TurbulenceWallRoughnessModel::SandGrain);
    EXPECT_EQ(options.wall_options.roughness_heights,
              (SimpleFluid::ArrReal{0.0, 2.0e-4}));
    EXPECT_EQ(options.wall_options.roughness_constants,
              (SimpleFluid::ArrReal{0.0, 0.5}));
    EXPECT_DOUBLE_EQ(options.wall_options.c_mu, 0.08);
    EXPECT_DOUBLE_EQ(options.wall_options.kappa, 0.42);
    EXPECT_DOUBLE_EQ(options.wall_options.log_layer_e, 9.7);
    EXPECT_EQ(options.wall_options.thermal_wall_law,
              SimpleFluid::TurbulenceThermalWallLaw::Jayatilleke);
    ASSERT_TRUE(
        options.wall_options.thermal_turbulent_prandtl_number.has_value());
    EXPECT_DOUBLE_EQ(
        *options.wall_options.thermal_turbulent_prandtl_number, 0.85);
    EXPECT_TRUE(options.wall_options.epsilon_low_re_correction);
    EXPECT_DOUBLE_EQ(options.wall_options.sst_beta_1, 0.076);
    EXPECT_DOUBLE_EQ(options.wall_options.sst_omega_wall_coefficient, 60.0);
}

/** @brief Verifies that wall treatment requires a compatible closure and explicit walls. */
TEST(TurbulenceModelOptionsTest, WallTreatmentRequiresCompatibleClosureAndExplicitWalls)
{
    TurbulenceModelOptions options;
    options.model = TurbulenceModelType::StandardKEpsilon;
    options.wall_treatment = TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    EXPECT_THROW(SimpleFluid::validate_turbulence_model_options(options),
                 std::invalid_argument);

    options.wall_options.boundary_names = {"wall"};
    EXPECT_NO_THROW(SimpleFluid::validate_turbulence_model_options(options));

    options.model = TurbulenceModelType::RNGKEpsilon;
    EXPECT_THROW(SimpleFluid::validate_turbulence_model_options(options),
                 std::invalid_argument);

    options.model = TurbulenceModelType::SSTKOmega;
    options.initial_wall_distance = 0.1;
    options.wall_treatment = TurbulenceWallTreatmentType::ResolvedLowReSST;
    EXPECT_NO_THROW(SimpleFluid::validate_turbulence_model_options(options));

    options.wall_options.kappa = 10.0;
    EXPECT_THROW(SimpleFluid::validate_turbulence_model_options(options),
                 std::invalid_argument);
    options.wall_options.kappa = 0.42;
    options.wall_options.sst_beta_1 = 0.076;
    EXPECT_NO_THROW(SimpleFluid::validate_turbulence_model_options(options));

    options.model = TurbulenceModelType::StandardKOmega;
    EXPECT_THROW(SimpleFluid::validate_turbulence_model_options(options),
                 std::invalid_argument);

    options.wall_treatment = TurbulenceWallTreatmentType::None;
    EXPECT_THROW(SimpleFluid::validate_turbulence_model_options(options),
                 std::invalid_argument);
}

} // namespace
