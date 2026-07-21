/**
 * @file testTurbulenceModelOptions.cc
 * @brief Unit tests for turbulence model configuration and parsing.
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
using SimpleFluid::TurbulenceWallTreatmentType;

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

TEST(TurbulenceModelOptionsTest, ParsesAndFormatsCanonicalNames)
{
    for (const auto& entry : canonical_models)
    {
        SCOPED_TRACE(entry.name);
        EXPECT_EQ(SimpleFluid::parse_turbulence_model_type(entry.name), entry.model);
        EXPECT_EQ(SimpleFluid::to_string(entry.model), entry.name);
    }
}

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
    EXPECT_EQ(options.wall_treatment, TurbulenceWallTreatmentType::None);
    EXPECT_TRUE(options.wall_options.boundary_names.empty());
}

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
        if (entry.requires_wall_distance)
        {
            database.set("wall_distance", SimpleFluid::real_t{0.125});
        }

        const auto options = SimpleFluid::turbulence_model_options_from_database(database);
        EXPECT_EQ(options.model, entry.model);
        EXPECT_EQ(options.initial_wall_distance.has_value(), entry.requires_wall_distance);
    }
}

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

TEST(TurbulenceModelOptionsTest, BSLAndSSTRequirePositiveWallDistance)
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
        EXPECT_THROW(SimpleFluid::turbulence_model_options_from_database(database),
                     std::invalid_argument);

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

TEST(TurbulenceModelOptionsTest, DatabaseReadsWallTreatmentSetAndConstants)
{
    SimpleFluid::Database database;
    database.set("turbulence_model", std::string{"standardKEpsilon"});
    database.set("wall_treatment", std::string{"standardHighReKEpsilon"});
    database.set("wall_boundaries", SimpleFluid::ArrString{"lowerWall", "upperWall"});
    database.set("wall_c_mu", SimpleFluid::real_t{0.08});
    database.set("wall_kappa", SimpleFluid::real_t{0.42});
    database.set("wall_e", SimpleFluid::real_t{9.7});
    database.set("wall_epsilon_low_re_correction", true);
    database.set("wall_beta_1", SimpleFluid::real_t{0.076});
    database.set("wall_sst_omega_face_coefficient", SimpleFluid::real_t{60.0});

    const auto options = SimpleFluid::turbulence_model_options_from_database(database);
    EXPECT_EQ(options.wall_treatment,
              TurbulenceWallTreatmentType::StandardHighReKEpsilon);
    EXPECT_EQ(options.wall_options.boundary_names,
              (SimpleFluid::ArrString{"lowerWall", "upperWall"}));
    EXPECT_DOUBLE_EQ(options.wall_options.c_mu, 0.08);
    EXPECT_DOUBLE_EQ(options.wall_options.kappa, 0.42);
    EXPECT_DOUBLE_EQ(options.wall_options.log_layer_e, 9.7);
    EXPECT_TRUE(options.wall_options.epsilon_low_re_correction);
    EXPECT_DOUBLE_EQ(options.wall_options.sst_beta_1, 0.076);
    EXPECT_DOUBLE_EQ(options.wall_options.sst_omega_wall_coefficient, 60.0);
}

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
