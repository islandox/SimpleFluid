/**
 * @file testReactingGasProperties.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests reacting gas option parsing and pure property functions.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoussFiniteVapor.hh"
#include "equations/ReactingGasProperties.hh"
#include "equations/ReactingThermodynamics.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

namespace Physics = SimpleFluid::BubbleGasPhysics;
SimpleFluid::ReactingGasOptions explicit_options()
{
    namespace R = SimpleFluid::Reacting;
    using P = R::SpeciesPhase;
    auto catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{91, "finite_solvent", 1, {{"S", 1}}, {P::Liquid, P::Resolved}},
            {317, "ordinary_gas", 1, {{"G", 1}}, {P::Dissolved, P::Dispersed, P::Resolved}}},
        17);
    SimpleFluid::ReactingGasOptions options;
    options.mode = SimpleFluid::ReactingGasMode::Species;
    options.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(
        catalog, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::SensibleEnergy);
    options.bouss_reference = std::make_shared<const R::BoussReferenceVolume>(
        R::BoussReferenceVolume{"test.fixed_reference", 1, catalog->fingerprint(), 1000, 1000, 4000, {.001, .02}});
    options.bouss_sources = std::make_shared<const R::BoussCoordinatedSources>(
        R::BoussCoordinatedSources{"test.no_implicit_sources", 1, catalog->fingerprint()});
    return options;
}

TEST(ReactingGasPropertiesTest, PerSpeciesDiffusivityUsesCatalogIdentityWithoutMandatoryGasSlots)
{
    namespace R = SimpleFluid::Reacting;
    auto options = explicit_options();
    options.species_properties = {{317, 2e-9, 0}};
    // The fixed-reference vapor owner intentionally has no EOS film law.
    EXPECT_THROW(SimpleFluid::validate_reacting_gas_options(options), std::invalid_argument);
    options.bouss_reference.reset();
    options.bouss_sources.reset();
    const auto catalog = std::make_shared<const R::SpeciesCatalog>(options.reaction_mechanism->catalog());
    const R::ThermodynamicBounds bounds{250, 600, 1000, 1e7};
    options.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 0, 0, 0, 0}; },
        R::EnergyConvention::SensibleEnergy);
    options.gas_thermodynamics = R::make_ideal_gas_mixture(
        catalog, std::vector<R::IdealGasCaloricSpecies>{{20, 300, 0}, {20, 300, 0}}, bounds,
        R::EnergyConvention::SensibleEnergy);
    EXPECT_NO_THROW(SimpleFluid::validate_reacting_gas_options(options));
    options.species_properties[0].diffusivity = -1;
    EXPECT_THROW(SimpleFluid::validate_reacting_gas_options(options), std::invalid_argument);
    options.species_properties[0].diffusivity = 0;
    options.species_properties[0].species = 999;
    EXPECT_THROW(SimpleFluid::validate_reacting_gas_options(options), std::invalid_argument);
}

/** @brief No heat/yield path manufactures a schema-owned source. */
TEST(ReactingGasPropertiesTest, EnabledSpeciesCannotManufactureSchemaFromHeatOrDatabaseDefaults)
{
    SimpleFluid::ReactingGasOptions options;
    options.mode = SimpleFluid::ReactingGasMode::Species;
    EXPECT_THROW(SimpleFluid::validate_reacting_gas_options(options), std::invalid_argument);
    for (const auto* selector : {"species", "idealGasSource", "twoPopulation"})
    {
        SimpleFluid::Database database;
        database.set("reacting_species_model", std::string{selector});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
    SimpleFluid::Database enabled;
    enabled.set("enable_reacting_gas", true);
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(enabled), std::invalid_argument);
}

/** @brief Verifies rejection of invalid thermodynamic property inputs. */
TEST(ReactingGasPropertiesTest, RejectsInvalidPhysicalScalarInputs)
{
    EXPECT_THROW(Physics::require_positive(0, "temperature"), std::invalid_argument);
    EXPECT_THROW(Physics::require_positive(std::numeric_limits<double>::infinity(), "pressure"), std::invalid_argument);
    EXPECT_THROW(Physics::require_non_negative(-1, "diffusivity"), std::invalid_argument);
}

/** @brief Verifies monotonic Henry-law and Laplace-pressure terms. */
TEST(ReactingGasPropertiesTest, HenryAndLaplaceTermsAreMonotone)
{
    const auto low_pressure =
        Physics::henry_equilibrium_concentration(
            1.0e-5, 1.0e5, 0.07, 1.0e-5);
    const auto high_pressure =
        Physics::henry_equilibrium_concentration(
            1.0e-5, 2.0e5, 0.07, 1.0e-5);
    const auto small_bubble =
        Physics::henry_equilibrium_concentration(
            1.0e-5, 1.0e5, 0.07, 1.0e-6);
    EXPECT_GT(high_pressure, low_pressure);
    EXPECT_GT(small_bubble, low_pressure);
}

/** Material-specific correlations live in external libraries; test only the contract here. */
TEST(ReactingGasPropertiesTest, ExternalSurfacePropertyRetainsExplicitCallbackContract)
{
    SimpleFluid::ReactingGasOptions options;
    options.surface_tension_mode = SimpleFluid::SurfaceTensionMode::External;
    EXPECT_THROW(Physics::surface_tension(options, 300), std::invalid_argument);
    options.surface_tension_correlation = [](double t) { return .07 * t / 300; };
    EXPECT_DOUBLE_EQ(Physics::surface_tension(options, 350), .07 * 350 / 300);
    EXPECT_THROW(Physics::surface_tension(options, 0), std::invalid_argument);
    options.surface_tension_correlation = [](double) { return std::numeric_limits<double>::quiet_NaN(); };
    EXPECT_THROW(Physics::surface_tension(options, 300), std::invalid_argument);
}

TEST(ReactingGasPropertiesTest, RemovedMaterialSelectorsAreRejected)
{
    for (const auto* key : {"surface_tension_model", "hydrogen_diffusivity_model"})
    {
        SimpleFluid::Database database;
        database.set(key, std::string{"sheng2024"});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
    for (const auto* key : {"uranium_concentration_mol_per_m3", "hydrogen_yield_molecules_per_100_ev", "atmospheric_pressure"})
    {
        SimpleFluid::Database database;
        database.set(key, SimpleFluid::real_t{1000});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
}

/** @brief Verifies Hughmark-correlation branches and validity constraints. */
TEST(ReactingGasPropertiesTest, HughmarkBranchesAndValidity)
{
    const auto low =
        Physics::hughmark_sherwood(100.0, 100.0);
    const auto expected_low =
        2.0 + 0.6 * std::sqrt(100.0) * std::cbrt(100.0);
    EXPECT_NEAR(low, expected_low, 1.0e-13);

    const auto high =
        Physics::hughmark_sherwood(1000.0, 100.0);
    const auto expected_high =
        2.0 + 0.27 * std::pow(1000.0, 0.63) * std::cbrt(100.0);
    EXPECT_NEAR(high, expected_high, 1.0e-13);
    EXPECT_THROW(
        Physics::hughmark_sherwood(1.0, 250.0),
        std::invalid_argument);
}

/** A finite-slip legacy fixture checks SI units and diameter-based groups. */
TEST(ReactingGasPropertiesTest, HughmarkReportsDiameterGroupsAndCoefficient)
{
    constexpr double density = 1000.0;       // kg/m3
    constexpr double viscosity = 1.0e-3;     // Pa s
    constexpr double diffusivity = 1.0e-8;   // m2/s
    constexpr double radius = 1.0e-4;        // m
    constexpr double slip = -5.0e-6;         // m/s; Pe = 0.1
    const auto result = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::Hughmark, diffusivity, radius, density, viscosity, slip);

    EXPECT_EQ(result.model, SimpleFluid::BubbleMassTransferMode::Hughmark);
    EXPECT_EQ(result.applicability, Physics::BubbleMassTransferApplicability::HughmarkSchmidtDomain);
    EXPECT_DOUBLE_EQ(result.radius, radius);
    EXPECT_DOUBLE_EQ(result.diameter, 2.0e-4);
    EXPECT_DOUBLE_EQ(result.relative_speed, -slip);
    EXPECT_DOUBLE_EQ(result.diffusivity, diffusivity);
    EXPECT_DOUBLE_EQ(result.liquid_density, density);
    EXPECT_DOUBLE_EQ(result.dynamic_viscosity, viscosity);
    EXPECT_NEAR(result.reynolds, 0.001, 1.0e-17);
    EXPECT_NEAR(result.schmidt, 100.0, 1.0e-12);
    EXPECT_NEAR(result.peclet, 0.1, 1.0e-14);
    EXPECT_NEAR(result.peclet, result.reynolds * result.schmidt, 1.0e-14);
    EXPECT_NEAR(result.sherwood, 2.088067956057324, 1.0e-14);
    EXPECT_NEAR(result.coefficient, 1.0440339780286619e-4, 1.0e-17);
}

/** Zero slip reaches the steady spherical-diffusion endpoint at high Sc. */
TEST(ReactingGasPropertiesTest, ZeroSlipAndWinterReferenceHighSc)
{
    constexpr double density = 1546.23563;
    constexpr double viscosity = 0.00243878721;
    constexpr double radius = 1.0e-4;
    constexpr double constant_diffusivity = 4.5e-9;
    const auto constant = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::SphericalDiffusion,
        constant_diffusivity, radius, density, viscosity, 0.0);
    EXPECT_EQ(constant.applicability,
              Physics::BubbleMassTransferApplicability::DiffusionOnlyReference);
    EXPECT_DOUBLE_EQ(constant.diameter, 2.0e-4);
    EXPECT_DOUBLE_EQ(constant.reynolds, 0.0);
    EXPECT_DOUBLE_EQ(constant.peclet, 0.0);
    EXPECT_NEAR(constant.schmidt, 350.4981406574711, 1.0e-10);
    EXPECT_DOUBLE_EQ(constant.sherwood, 2.0);
    EXPECT_NEAR(constant.coefficient, 4.5e-5, 1.0e-18);

    // Winter et al. (2022) Eq. 45 gives about 5.1886e-9 m2/s at 298.85 K.
    constexpr double winter_diffusivity = 5.1886e-9;
    const auto winter = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::SphericalDiffusion,
        winter_diffusivity, radius, density, viscosity, 0.0);
    EXPECT_DOUBLE_EQ(winter.reynolds, 0.0);
    EXPECT_DOUBLE_EQ(winter.peclet, 0.0);
    EXPECT_NEAR(winter.schmidt, 303.98212098805453, 1.0e-10);
    EXPECT_DOUBLE_EQ(winter.sherwood, 2.0);
    EXPECT_NEAR(winter.coefficient, 5.1886e-5, 1.0e-18);
    EXPECT_THROW(Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::SphericalDiffusion,
        constant_diffusivity, radius, density, viscosity, 2.25e-6),
        std::domain_error);
    EXPECT_THROW(Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::SphericalDiffusion,
        constant_diffusivity, radius, density, viscosity, -2.25e-6),
        std::domain_error);

    const auto legacy = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::Hughmark, 1.0e-8, radius, 1000.0, 1.0e-3, 0.0);
    EXPECT_DOUBLE_EQ(legacy.reynolds, 0.0);
    EXPECT_DOUBLE_EQ(legacy.peclet, 0.0);
    EXPECT_DOUBLE_EQ(legacy.sherwood, 2.0);
    EXPECT_NEAR(legacy.coefficient, 1.0e-4, 1.0e-18);
}

/** The selected legacy law retains both published code branches and its Sc gate. */
TEST(ReactingGasPropertiesTest, SelectedHughmarkPreservesExistingResults)
{
    constexpr double radius = 1.0e-4;
    constexpr double diffusivity = 1.0e-8;
    constexpr double density = 1000.0;
    constexpr double viscosity = 1.0e-3;
    for (const auto [speed, reynolds] :
         {std::pair{0.5, 100.0}, std::pair{5.0, 1000.0}})
    {
        const auto result = Physics::bubble_mass_transfer(
            SimpleFluid::BubbleMassTransferMode::Hughmark, diffusivity, radius, density, viscosity, speed);
        const double expected_sherwood = reynolds < 776.06 ? 2.0 + 0.6 * std::sqrt(reynolds) * std::cbrt(100.0)
                                                           : 2.0 + 0.27 * std::pow(reynolds, 0.63) * std::cbrt(100.0);
        EXPECT_EQ(result.applicability, Physics::BubbleMassTransferApplicability::HughmarkSchmidtDomain);
        EXPECT_NEAR(result.reynolds, reynolds, 1.0e-12);
        EXPECT_NEAR(result.sherwood, expected_sherwood, 1.0e-12);
        EXPECT_NEAR(result.coefficient, expected_sherwood * diffusivity / (2.0 * radius), 1.0e-14);
        EXPECT_NEAR(result.coefficient,
                    Physics::hughmark_mass_transfer_coefficient(
                        diffusivity, radius, density, viscosity, speed), 1.0e-14);
    }
    try
    {
        (void) Physics::bubble_mass_transfer(
            SimpleFluid::BubbleMassTransferMode::Hughmark, 4.5e-9, radius, 1546.23563, 0.00243878721, 0.0);
        FAIL() << "Expected the legacy high-Sc state to be rejected.";
    }
    catch (const std::domain_error& error)
    {
        const std::string message(error.what());
        EXPECT_NE(message.find("Sc < 250"), std::string::npos);
        EXPECT_NE(message.find("R="), std::string::npos);
        EXPECT_NE(message.find("D_m="), std::string::npos);
        EXPECT_NE(message.find("Sc="), std::string::npos);
    }
    EXPECT_NO_THROW(Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::SphericalDiffusion,
        4.5e-9, radius, 1546.23563, 0.00243878721, 0.0));
    EXPECT_THROW(Physics::hughmark_sherwood(1.0, 250.0), std::invalid_argument);
}

/** Feng and Michaelides (2000), p. 67 Eq. (19) and independent Table 1 values. */
TEST(ReactingGasPropertiesTest, FengEndpointsConvertRadiusPecletAndMatchPublishedStokesValues)
{
    constexpr double density = 1546.23563;       // kg/m3
    constexpr double viscosity = 0.00243878721;  // Pa s
    constexpr double diffusivity = 4.5e-9;        // m2/s
    constexpr double radius = 1.0e-4;             // m
    constexpr double slip = 0.0045;               // m/s, source Pe_a = U R / D = 100
    const auto rigid = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid,
        diffusivity, radius, density, viscosity, slip);
    const auto clean = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::FengMichaelidesClean,
        diffusivity, radius, density, viscosity, -slip);

    // Eq. (11) defines radius Pe_a. The public result uses diameter Pe_d;
    // Eq. (19) and Table 1 already report diameter Sherwood numbers.
    constexpr double source_radius_peclet = 100.0;
    const auto expected_rigid_sh = 1.49 * std::pow(source_radius_peclet, 0.322);
    const auto expected_clean_sh = 1.49 * std::pow(source_radius_peclet, 0.435);
    for (const auto* result : {&rigid, &clean})
    {
        EXPECT_EQ(result->applicability,
                  Physics::BubbleMassTransferApplicability::SourceFitOperationalWindow);
        EXPECT_DOUBLE_EQ(result->diameter, 2.0e-4);
        EXPECT_DOUBLE_EQ(result->relative_speed, slip);
        EXPECT_NEAR(result->reynolds, 0.5706164364376832, 1.0e-12);
        EXPECT_NEAR(result->schmidt, 350.4981406574711, 1.0e-10);
        EXPECT_NEAR(result->peclet, 200.0, 1.0e-12);
        EXPECT_NEAR(result->peclet, result->reynolds * result->schmidt, 1.0e-12);
        EXPECT_NEAR(result->coefficient, result->sherwood * diffusivity / (2.0 * radius), 1.0e-17);
    }
    EXPECT_NEAR(rigid.sherwood, expected_rigid_sh, 1.0e-12);
    EXPECT_NEAR(clean.sherwood, expected_clean_sh, 1.0e-12);
    // Table 1 Stokes calculations at Pe_a=100 are 6.86 (sigma=0) and
    // 10.91 (sigma=infinity). Eq. (19) is a fit, so compare within 5%.
    EXPECT_LT(std::abs(rigid.sherwood - 6.86) / 6.86, 0.05);
    EXPECT_LT(std::abs(clean.sherwood - 10.91) / 10.91, 0.05);
    EXPECT_GT(clean.sherwood, rigid.sherwood);
    // The finite-Pe fit exponents are 0.322/0.435, not exact 1/3 or 1/2 asymptotes.
    EXPECT_GT(std::abs(rigid.sherwood - 1.49 * std::pow(100.0, 1.0 / 3.0)),
              0.01 * rigid.sherwood);
    EXPECT_GT(std::abs(clean.sherwood - 1.49 * std::sqrt(100.0)),
              0.01 * clean.sherwood);
}

/** The Winter2022 reference D remains high-Sc at a source-supported Pe_a=100. */
TEST(ReactingGasPropertiesTest, FengEndpointsAcceptWinterReferenceBeyondLegacyScGuard)
{
    constexpr double density = 1546.23563;
    constexpr double viscosity = 0.00243878721;
    constexpr double radius = 1.0e-4;
    // The external host separately tests the Winter2022 provider at 298.85 K.
    constexpr double winter_diffusivity = 5.18858348242618e-9;
    constexpr double slip = 100.0 * winter_diffusivity / radius;
    EXPECT_THROW(Physics::bubble_mass_transfer(SimpleFluid::BubbleMassTransferMode::Hughmark, winter_diffusivity,
                     radius, density, viscosity, slip),
        std::domain_error);
    for (const auto [model, exponent] : {std::pair{SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid, 0.322},
             std::pair{SimpleFluid::BubbleMassTransferMode::FengMichaelidesClean, 0.435}})
    {
        const auto result = Physics::bubble_mass_transfer(
            model, winter_diffusivity, radius, density, viscosity, slip);
        EXPECT_EQ(result.applicability,
                  Physics::BubbleMassTransferApplicability::SourceFitOperationalWindow);
        EXPECT_NEAR(result.schmidt, 303.983088698633, 1.0e-10);
        EXPECT_NEAR(result.reynolds, 0.6579313370892116, 1.0e-12);
        EXPECT_NEAR(result.peclet, 200.0, 1.0e-12);
        EXPECT_NEAR(result.sherwood, 1.49 * std::pow(100.0, exponent), 1.0e-12);
        EXPECT_NEAR(result.coefficient,
                    1.49 * std::pow(100.0, exponent) * winter_diffusivity / (2.0 * radius),
                    1.0e-17);
    }
}

/** Eq. (4) endpoint arithmetic is a low-Pe asymptotic implementation check. */
TEST(ReactingGasPropertiesTest, FengLowPecletEndpointsApproachSphericalDiffusion)
{
    constexpr double density = 1546.23563;
    constexpr double viscosity = 0.00243878721;
    constexpr double diffusivity = 4.5e-9;
    constexpr double radius = 1.0e-4;
    for (const auto model : {SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid,
                             SimpleFluid::BubbleMassTransferMode::FengMichaelidesClean})
    {
        const auto zero = Physics::bubble_mass_transfer(
            model, diffusivity, radius, density, viscosity, 0.0);
        EXPECT_DOUBLE_EQ(zero.reynolds, 0.0);
        EXPECT_DOUBLE_EQ(zero.peclet, 0.0);
        EXPECT_DOUBLE_EQ(zero.sherwood, 2.0);
        EXPECT_NEAR(zero.coefficient, diffusivity / radius, 1.0e-18);
    }

    // Public Pe_d=0.5; the p. 64 Eq. (4) correction is 1/8 for sigma=0
    // and 1/12 for sigma->infinity. These are mathematical fit checks, not
    // independent physical validation of the low-Pe asymptotic expansion.
    constexpr double slip = 1.125e-5;
    const auto rigid = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::FengMichaelidesRigid,
        diffusivity, radius, density, viscosity, slip);
    const auto clean = Physics::bubble_mass_transfer(
        SimpleFluid::BubbleMassTransferMode::FengMichaelidesClean,
        diffusivity, radius, density, viscosity, slip);
    constexpr double peclet_d = 0.5;
    const auto rigid_expected = 2.0 + peclet_d / 2.0
        + (1.0 / 8.0) * peclet_d * peclet_d * std::log(peclet_d);
    const auto clean_expected = 2.0 + peclet_d / 2.0
        + (1.0 / 12.0) * peclet_d * peclet_d * std::log(peclet_d);
    EXPECT_NEAR(rigid.peclet, peclet_d, 1.0e-14);
    EXPECT_NEAR(clean.peclet, peclet_d, 1.0e-14);
    EXPECT_NEAR(rigid.sherwood, rigid_expected, 1.0e-14);
    EXPECT_NEAR(clean.sherwood, clean_expected, 1.0e-14);
    EXPECT_NEAR(rigid.coefficient, rigid_expected * diffusivity / (2.0 * radius), 1.0e-18);
    EXPECT_NEAR(clean.coefficient, clean_expected * diffusivity / (2.0 * radius), 1.0e-18);
    EXPECT_GT(clean.sherwood, rigid.sherwood);
}

TEST(ReactingGasPropertiesTest, FengEndpointsEnforceOperationalWindowAndPhysicalInputs)
{
    using SimpleFluid::BubbleMassTransferMode;
    constexpr double density = 1546.23563;
    constexpr double viscosity = 0.00243878721;
    constexpr double diffusivity = 4.5e-9;
    constexpr double radius = 1.0e-4;
    for (const auto model : {BubbleMassTransferMode::FengMichaelidesRigid,
                             BubbleMassTransferMode::FengMichaelidesClean})
    {
        // Pe_a=99, with Re_d<1.
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, radius,
            density, viscosity, 0.004455), std::domain_error);
        // No verified interpolation connects Eq. (4) at Pe_d<1 to Eq. (19)
        // at Pe_a>=100, so the operational gap remains explicit.
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, radius,
            density, viscosity, 2.25e-5), std::domain_error); // Pe_d=1.
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, radius,
            density, viscosity, 0.0044775), std::domain_error); // Pe_d=199.
        // Pe_a=5001, with Re_d<1; isolate the upper Pe gate from the Re gate.
        EXPECT_THROW(Physics::bubble_mass_transfer(model, 1.0e-12, radius,
            density, viscosity, 5.001e-5), std::domain_error);
        // Pe_a=177.8 lies inside the fit window, but Re_d exceeds one.
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, radius,
            density, viscosity, 0.008), std::domain_error);
        EXPECT_THROW(Physics::bubble_mass_transfer(model, 0.0, radius,
            density, viscosity, 0.0045), std::invalid_argument);
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, 0.0,
            density, viscosity, 0.0045), std::invalid_argument);
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, radius,
            density, 0.0, 0.0045), std::invalid_argument);
        EXPECT_THROW(Physics::bubble_mass_transfer(model, diffusivity, radius,
            density, viscosity, std::numeric_limits<double>::infinity()), std::invalid_argument);
    }
}

/** Invalid SI inputs are distinct from finite states outside a model domain. */
TEST(ReactingGasPropertiesTest, BubbleTransferRejectsInvalidPhysicalInputs)
{
    using SimpleFluid::BubbleMassTransferMode;
    constexpr auto model = BubbleMassTransferMode::SphericalDiffusion;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double infinity = std::numeric_limits<double>::infinity();
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 0.0, 1e-4, 1000, 1e-3, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, nan, 1e-4, 1000, 1e-3, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, -1e-4, 1000, 1e-3, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, 1e-4, 0, 1e-3, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, 1e-4, 1000, infinity, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, 1e-4, 1000, 1e-3, nan), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, infinity, 1000, 1e-3, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, 1e-4, 1000, 1e-3, infinity), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(model, 1e-8, std::numeric_limits<double>::max(),
        1000, 1e-3, 0), std::invalid_argument);
    EXPECT_THROW(Physics::bubble_mass_transfer(static_cast<BubbleMassTransferMode>(-1),
        1e-8, 1e-4, 1000, 1e-3, 0), std::invalid_argument);
}

TEST(ReactingGasPropertiesTest, ParsesMassTransferSelectorAndRejectsUnknownValues)
{
    using SimpleFluid::BubbleMassTransferMode;
    EXPECT_EQ(SimpleFluid::reacting_gas_options_from_database({}).mass_transfer_mode, BubbleMassTransferMode::Hughmark);
    SimpleFluid::Database database;
    database.set("bubble_mass_transfer_model", std::string{"sphericalDiffusion"});
    EXPECT_EQ(SimpleFluid::reacting_gas_options_from_database(database).mass_transfer_mode,
              BubbleMassTransferMode::SphericalDiffusion);
    for (const auto [name, model] :
         {std::pair{std::string{"fengMichaelidesClean"}, BubbleMassTransferMode::FengMichaelidesClean},
          std::pair{std::string{"fengMichaelidesRigid"}, BubbleMassTransferMode::FengMichaelidesRigid}})
    {
        database.set("bubble_mass_transfer_model", name);
        EXPECT_EQ(SimpleFluid::reacting_gas_options_from_database(database).mass_transfer_mode, model);
    }
    database.set("bubble_mass_transfer_model", std::string{"unverified"});
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    database.set("bubble_mass_transfer_model", 7);
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
}

/** @brief Retired kinetics controls cannot silently select the disabled default. */
TEST(ReactingGasPropertiesTest, RejectsRetiredKineticsControlsEvenWhenDisabled)
{
    for (const auto* key : {"reacting_kinetics_mode", "reacting_heaviside_mode", "initial_micro_moles",
             "initial_large_moles", "initial_dissolved_hydrogen", "hydrogen_yield_mol_per_j"})
    {
        SimpleFluid::Database database;
        database.set(key, std::string{"disabled"});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
}

/** @brief Verifies that the Celata bubble-rise velocity balances drag. */
TEST(ReactingGasPropertiesTest, CelataRiseVelocityBalancesDrag)
{
    constexpr double radius = 1.0e-3;
    constexpr double liquid_density = 998.0;
    constexpr double gas_density = 1.2;
    constexpr double viscosity = 1.0e-3;
    constexpr double surface_tension = 0.072;
    constexpr double gravity = 9.80665;
    const auto result = Physics::celata2007_bubble_rise_velocity(
        radius,
        liquid_density,
        gas_density,
        viscosity,
        surface_tension,
        gravity);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.velocity, 0.0);
    EXPECT_GT(result.reynolds, 0.0);
    EXPECT_GT(result.eotvos, 0.0);
    EXPECT_GT(result.drag_coefficient, 0.0);
    EXPECT_TRUE(result.within_experimental_range);
    const auto velocity_scale = 8.0 * radius * gravity / 3.0;
    EXPECT_LE(
        std::abs(result.residual), 1.0e-9 * velocity_scale);
}

/** @brief Verifies that the computed bubble radius satisfies the equation of state. */
TEST(ReactingGasPropertiesTest, BubbleRadiusSolvesEos)
{
    constexpr double pressure = 101325.0;
    constexpr double surface_tension = 0.07;
    constexpr double gas_constant = 8.314462618;
    constexpr double temperature = 300.0;
    constexpr double radius = 2.0e-5;
    const auto moles =
        4.0 * std::numbers::pi / 3.0
        * (pressure * radius * radius * radius
           + 2.0 * surface_tension * radius * radius)
        / (gas_constant * temperature);

    const auto result = Physics::solve_bubble_radius(
        moles, pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2, 100, 1.0e-11);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.radius, radius, radius * 1.0e-9);

    const auto larger = Physics::solve_bubble_radius(
        2.0 * moles, pressure, surface_tension, gas_constant,
        temperature, 1.0e-8, 1.0e-2);
    EXPECT_TRUE(larger.converged);
    EXPECT_GT(larger.radius, result.radius);
}

/** @brief Verifies expected bubble-radius trends with thermodynamic state. */
TEST(ReactingGasPropertiesTest, BubbleRadiusTrendsWithState)
{
    constexpr double pressure = 101325.0;
    constexpr double surface_tension = 0.07;
    constexpr double gas_constant = 8.314462618;
    constexpr double temperature = 300.0;
    constexpr double moles = 1.0e-12;

    const auto base = Physics::solve_bubble_radius(
        moles, pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2);
    const auto hot = Physics::solve_bubble_radius(
        moles, pressure, surface_tension, gas_constant, 2.0 * temperature,
        1.0e-8, 1.0e-2);
    const auto compressed = Physics::solve_bubble_radius(
        moles, 2.0 * pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2);
    const auto more_bubbles = Physics::solve_bubble_radius(
        0.5 * moles, pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2);

    ASSERT_TRUE(base.converged);
    ASSERT_TRUE(hot.converged);
    ASSERT_TRUE(compressed.converged);
    ASSERT_TRUE(more_bubbles.converged);
    EXPECT_GT(hot.radius, base.radius);
    EXPECT_LT(compressed.radius, base.radius);
    EXPECT_LT(more_bubbles.radius, base.radius);
}

/** @brief Verifies reconstruction of void fraction and characteristic radius. */
TEST(ReactingGasPropertiesTest, VoidAndCharacteristicRadiusReconstruct)
{
    const auto micro_void =
        Physics::bubble_void_fraction(2.0e9, 1.0e-6);
    const auto large_void =
        Physics::bubble_void_fraction(5.0e6, 1.0e-4);
    EXPECT_GT(micro_void, 0.0);
    EXPECT_GT(large_void, micro_void);

    const auto characteristic =
        Physics::characteristic_radius(
            2.0e9, 1.0e-6, 5.0e6, 1.0e-4);
    EXPECT_GT(characteristic, 1.0e-6);
    EXPECT_LT(characteristic, 1.0e-4);
    EXPECT_DOUBLE_EQ(
        Physics::characteristic_radius(0.0, 0.0, 0.0, 0.0),
        0.0);
}

/** @brief Verifies parsing of flat runtime correlation selectors. */
TEST(ReactingGasPropertiesTest, PreservesSubstantivePressureAndSolutionSelectors)
{
    SimpleFluid::Database database;
    database.set("reacting_pressure_mode", std::string{"reconstructed"});
    database.set("solution_transport_mode", std::string{"advective"});
    auto options = SimpleFluid::reacting_gas_options_from_database(database);
    EXPECT_EQ(options.mode, SimpleFluid::ReactingGasMode::Disabled);
    EXPECT_EQ(options.pressure_mode, SimpleFluid::ReactingPressureMode::Reconstructed);
    EXPECT_EQ(options.solution_transport, SimpleFluid::ReactingTransportMode::Advective);
    database.set("reacting_pressure_mode", std::string{"prescribedHistory"});
    database.set("pressure_history_times", SimpleFluid::ArrReal{0, 2});
    database.set("pressure_history_values", SimpleFluid::ArrReal{100000, 200000});
    EXPECT_EQ(SimpleFluid::reacting_gas_options_from_database(database).pressure_mode,
        SimpleFluid::ReactingPressureMode::PrescribedHistory);
    database.set("pressure_history_times", SimpleFluid::ArrReal{0, 0});
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    database.set("reacting_pressure_mode", std::string{"inertial"});
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
}

/** @brief Verifies contextual diagnostics for incorrectly typed gas options. */
TEST(ReactingGasPropertiesTest, ReportsWrongTypedOptionContext)
{
    SimpleFluid::Database database;
    database.set("enable_reacting_gas", std::string{"yes"});

    try
    {
        SimpleFluid::reacting_gas_options_from_database(database);
        FAIL() << "Expected a typed reacting gas option failure.";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(message.find("Reacting species"), std::string::npos);
        EXPECT_NE(message.find("enable_reacting_gas"), std::string::npos);
        EXPECT_NE(message.find("wrong type"), std::string::npos);
    }
}

/** @brief Enabled schema-owned gas requires actual laws and substantive pressure history. */
TEST(ReactingGasPropertiesTest, EnabledSpeciesRequiresExplicitSchemaAndPreservesPressureHistory)
{
    auto options = explicit_options();
    EXPECT_NO_THROW(SimpleFluid::validate_reacting_gas_options(options));
    options.pressure_mode = SimpleFluid::ReactingPressureMode::PrescribedHistory;
    EXPECT_THROW(SimpleFluid::validate_reacting_gas_options(options), std::invalid_argument);
    options.pressure_history_times = {0, 1};
    options.pressure_history_values = {100000, 120000};
    EXPECT_NO_THROW(SimpleFluid::validate_reacting_gas_options(options));
    options.reaction_mechanism.reset();
    EXPECT_THROW(SimpleFluid::validate_reacting_gas_options(options), std::invalid_argument);
}

TEST(ReactingGasPropertiesTest, TransportToleranceIsOptionalValidatedAndParsed)
{
    EXPECT_DOUBLE_EQ(SimpleFluid::ReactingGasOptions{}.transport_solver_tolerance,1e-10);
    SimpleFluid::Database database;
    database.set("reacting_transport_solver_tolerance", SimpleFluid::real_t{1e-14});
    EXPECT_DOUBLE_EQ(SimpleFluid::reacting_gas_options_from_database(database).transport_solver_tolerance, 1e-14);
    for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        database.set("reacting_transport_solver_tolerance",value);
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database),std::invalid_argument);
    }
}

} // namespace


/** Retired keys never silently select the disabled default or override neutral keys. */
TEST(ReactingGasPropertiesTest, RejectsRetiredControlsBeforeDefaultingToDisabled)
{
    for (const auto* key : {
        "enable_radiolysis",
        "radiolytic_bubble_model",
        "radiolytic_pressure_mode",
        "radiolytic_heaviside_mode",
        "radiolytic_kinetics_mode",
        "radiolytic_transport_solver_tolerance",
        "max_radiolytic_subcycles",
        "radiolytic_free_surface_patches",
        "precursor_group_count",
        "precursor_decay_constants",
        "precursor_initial_concentrations",
        "precursor_source_terms",
        "precursor_power_yields",
        "precursor_effective_diffusivity"})
    {
        SimpleFluid::Database database;
        database.set(key, std::string{"disabled"});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
        database.set("reacting_bubble_model", std::string{"disabled"});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
}

TEST(ReactingGasPropertiesTest, RejectsRetiredTwoPopulationSelector)
{
    for (const auto* selector : {"sheng2024TwoPopulation", "sheng2024_two_population"})
    {
        SimpleFluid::Database database;
        database.set("reacting_bubble_model", std::string{selector});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
}

TEST(ReactingGasPropertiesTest, DisabledParserPreservesExplicitNeutralControls)
{
    using namespace SimpleFluid;
    Database database;
    database.set("enable_reacting_gas", false);
    database.set("reacting_species_model", std::string{"disabled"});
    database.set("bubble_rise_velocity_model", std::string{"constantSlip"});
    database.set("constant_slip_velocity", real_t{.4});
    database.set("surface_tension", real_t{.075});
    database.set("min_radius", real_t{2e-8});
    database.set("max_radius", real_t{.025});
    database.set("bubble_gravity", real_t{7.2});
    database.set("rise_velocity_tolerance", real_t{3e-11});
    database.set("max_rise_velocity_iterations", 61);
    database.set("max_reacting_subcycles", 37);
    database.set("reacting_free_surface_patches", ArrString{"vent_a", "vent_b"});
    const auto options = reacting_gas_options_from_database(database);
    EXPECT_EQ(options.mode, ReactingGasMode::Disabled);
    EXPECT_EQ(options.reaction_mechanism, nullptr);
    EXPECT_EQ(options.rise_velocity_mode, BubbleRiseVelocityMode::ConstantSlip);
    EXPECT_DOUBLE_EQ(options.constant_slip_velocity, .4);
    EXPECT_DOUBLE_EQ(options.surface_tension, .075);
    EXPECT_DOUBLE_EQ(options.min_radius, 2e-8);
    EXPECT_DOUBLE_EQ(options.max_radius, .025);
    EXPECT_DOUBLE_EQ(options.bubble_gravity, 7.2);
    EXPECT_DOUBLE_EQ(options.rise_velocity_tolerance, 3e-11);
    EXPECT_EQ(options.max_rise_velocity_iterations, 61);
    EXPECT_EQ(options.max_subcycles, 37);
    EXPECT_EQ(options.free_surface_patches, (ArrString{"vent_a", "vent_b"}));
}
TEST(ReactingGasPropertiesTest, DisabledParserRejectsMalformedRetainedNeutralControls)
{
    using namespace SimpleFluid;
    for (const auto* key : {"constant_slip_velocity", "surface_tension", "min_radius", "max_radius", "bubble_gravity",
             "rise_velocity_tolerance"})
        for (const real_t value : {real_t{-1}, std::numeric_limits<real_t>::quiet_NaN()})
        {
            SCOPED_TRACE(key);
            Database database;
            database.set("reacting_species_model", std::string{"disabled"});
            database.set(key, value);
            EXPECT_THROW(reacting_gas_options_from_database(database), std::invalid_argument);
        }
    for (const auto* key : {"min_radius", "rise_velocity_tolerance"})
    {
        Database database;
        database.set(key, real_t{0});
        EXPECT_THROW(reacting_gas_options_from_database(database), std::invalid_argument);
    }
    for (const auto* key : {"max_reacting_subcycles", "max_rise_velocity_iterations"})
    {
        Database database;
        database.set(key, 0);
        EXPECT_THROW(reacting_gas_options_from_database(database), std::invalid_argument);
        database.set(key, std::string{"not an integer"});
        EXPECT_THROW(reacting_gas_options_from_database(database), std::invalid_argument);
    }
    Database reversed;
    reversed.set("min_radius", real_t{.02});
    reversed.set("max_radius", real_t{.01});
    EXPECT_THROW(reacting_gas_options_from_database(reversed), std::invalid_argument);
    Database malformed_patches;
    malformed_patches.set("reacting_free_surface_patches", 7);
    EXPECT_THROW(reacting_gas_options_from_database(malformed_patches), std::invalid_argument);

    Database gravity;
    gravity.set("bubble_gravity", real_t{0});
    EXPECT_NO_THROW((void) reacting_gas_options_from_database(gravity)); // Unused by zero slip.
    gravity.set("bubble_rise_velocity_model", std::string{"celata2007"});
    EXPECT_THROW(reacting_gas_options_from_database(gravity), std::invalid_argument);
}
TEST(ReactingGasPropertiesTest, RetiredControlsStillRejectBesidePreservedDisabledSettings)
{
    using namespace SimpleFluid;
    for (const auto* key : {"max_radius_iterations", "gas_constant", "initial_micro_moles", "reacting_kinetics_mode"})
    {
        Database database;
        database.set("enable_reacting_gas", false);
        database.set("constant_slip_velocity", real_t{.4});
        database.set("reacting_free_surface_patches", ArrString{"vent"});
        database.set(key, std::string{"disabled"});
        EXPECT_THROW(reacting_gas_options_from_database(database), std::invalid_argument);
    }
}
