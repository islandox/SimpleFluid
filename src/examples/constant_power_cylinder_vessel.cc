/**
 * @file constant_power_cylinder_vessel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cylinder vessel smoke example with uniform prescribed_heat power,
 *        reacting gas, and boiling.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/BoussFiniteVapor.hh"
#include "examples/ExampleRunner.hh"
#include "examples/ReactingMixtureExampleSupport.hh"

#include <Tpetra_Core.hpp>

#include <memory>
#include <stdexcept>
#include <string_view>

/**
 * @brief Run the constant-power cylindrical vessel multiphysics smoke case.
 *
 * @param argc Argument count passed through to Tpetra.
 * @param argv Argument vector passed through to Tpetra.
 * @return Process exit code, zero on normal completion.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);

    constexpr SimpleFluid::real_t liquid_density = 1000.0;
    constexpr SimpleFluid::real_t specific_heat_capacity = 4200.0;
    constexpr SimpleFluid::real_t thermal_diffusivity = 1.4e-7;
    constexpr SimpleFluid::real_t thermal_conductivity =
        liquid_density * specific_heat_capacity * thermal_diffusivity;
    constexpr SimpleFluid::real_t wall_heat_flux = 25.0;

    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.5});
    db->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{0.5});
    db->set("height", SimpleFluid::real_t{1.0});
    db->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{"radial", "zmin", "zmax"});

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmin"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    // The transport equation adds k (dT/dn) A to the cell energy balance,
    // so q''/k supplies the same wall-to-fluid heat used by wall boiling.
    bcs.temperature["zmax"] =
        {SimpleFluid::BoundaryConditionType::Neumann,
         wall_heat_flux / thermal_conductivity};
    bcs.temperature["radial"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.velocity["zmin"] =
        {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["zmax"] =
        {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    bcs.velocity["radial"] =
        {SimpleFluid::BoundaryConditionType::NoSlip, {}};

    // Optional exterior cooling on the fluid-domain boundary (no solid wall).
    // Usage: --exterior-convection=constant|natural|forced
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        constexpr std::string_view prefix = "--exterior-convection=";
        if (!argument.starts_with(prefix))
            continue;
        const auto mode = argument.substr(prefix.size());
        auto& exterior = bcs.convection["radial"];
        exterior.ambient_temperature = 293.15;
        exterior.heat_transfer_coefficient = 5.0;
        exterior.characteristic_length = 1.0; // height (natural) or diameter (forced)
        exterior.fluid_thermal_conductivity = 0.026;
        exterior.kinematic_viscosity = 1.6e-5;
        exterior.thermal_diffusivity = 2.3e-5;
        exterior.thermal_expansion = 1.0 / 300.0;
        exterior.free_stream_speed = 1.0;
        if (mode == "natural")
            exterior.correlation = SimpleFluid::ConvectionCorrelation::ChurchillChuVerticalPlate;
        else if (mode == "forced")
            exterior.correlation = SimpleFluid::ConvectionCorrelation::ChurchillBernsteinCylinder;
        else if (mode == "constant")
            exterior.correlation = SimpleFluid::ConvectionCorrelation::Constant;
        else
            throw std::invalid_argument("--exterior-convection requires constant, natural, or forced.");
        bcs.temperature.erase("radial");
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-3;
    time_options.steps = 3;
    time_options.kinematic_viscosity = 1.0e-6;
    time_options.thermal_diffusivity = thermal_diffusivity;
    time_options.reference_temperature = 300.0;
    time_options.thermal_expansion = 2.1e-4;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 100;
    linear_options.tolerance = 1.0e-10;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = liquid_density;
    model_options.density = liquid_density;
    model_options.specific_heat_capacity = specific_heat_capacity;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = thermal_conductivity;

    SimpleFluid::run_boussinesq_example<>(
        db, bcs, time_options, linear_options, model_options,
        [=](auto& solver, auto& coupling)
        {
            coupling.initialize_bottom_hot_top_cold(300.0, 300.0);

            SimpleFluid::PrescribedHeatSourceOptions prescribed_heat;
            prescribed_heat.profile = SimpleFluid::PrescribedHeatProfile::Constant;
            prescribed_heat.power_density = 2.5e3;
            coupling.configure_prescribed_heat_source(prescribed_heat);

            namespace Mixture = SimpleFluid::Examples::ReactingMixture;
            namespace R = SimpleFluid::Reacting;
            using P = R::SpeciesPhase;
            const auto catalog = Mixture::make_catalog();
            auto mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
                std::vector<R::ReactionDefinition>{{Mixture::conversion, "finite_A_to_B_and_two_ordinary_gases",
                    {{Mixture::reactant, P::Liquid, -1}, {Mixture::product, P::Liquid, 1},
                        {Mixture::gas_one, P::Resolved, 1}, {Mixture::gas_two, P::Resolved, 1}},
                    100}},
                R::EnergyConvention::SensibleEnergy);
            auto reference = std::make_shared<R::BoussReferenceVolume>();
            reference->identity = "manufactured.cylinder.fixed_reference_volumes";
            reference->revision = 1;
            reference->schemaFingerprint = catalog->fingerprint();
            reference->liquidDensity = reference->thermalDensity = liquid_density;
            reference->heatCapacity = specific_heat_capacity;
            reference->gasMolarVolumes.resize(catalog->species().size());
            const double ordinaryGasMolarVolume = Mixture::gas_constant * 300 / 100000;
            reference->gasMolarVolumes[catalog->index(Mixture::gas_one)] = ordinaryGasMolarVolume;
            reference->gasMolarVolumes[catalog->index(Mixture::gas_two)] = ordinaryGasMolarVolume;
            reference->gasMolarVolumes[catalog->index(Mixture::solvent)] = .018; // W_solvent/rho_v with rho_v=1.
            auto sources = std::make_shared<R::BoussCoordinatedSources>();
            sources->identity = "manufactured.cylinder.finite_bulk_and_wall_vapor";
            sources->revision = 1;
            sources->schemaFingerprint = catalog->fingerprint();
            sources->vapor = R::BoussVaporBinding{Mixture::solvent, P::Resolved, {}, 299.99, 2.256e6, .05, 0};
            sources->vapor->admissionPolicy = R::BoussVaporAdmissionPolicy::ProportionalRequests;
            sources->vapor->walls = {{"zmax", wall_heat_flux, .2}};
            SimpleFluid::ReactingGasOptions reacting_gas;
            reacting_gas.mode = SimpleFluid::ReactingGasMode::Species;
            reacting_gas.alpha_max = .5;
            reacting_gas.reference_pressure = 100000;
            reacting_gas.reaction_mechanism = mechanism;
            reacting_gas.bouss_reference = reference;
            reacting_gas.bouss_sources = sources;
            reacting_gas.population_ids = {"micro", "large"};
            reacting_gas.min_radius = 1e-10;
            reacting_gas.max_radius = .1;
            reacting_gas.solution_transport = SimpleFluid::ReactingTransportMode::Advective;
            reacting_gas.initial_species_state = [catalog, ordinaryGasMolarVolume, liquid_density](
                                                     const R::ReactionEnvironment& environment, std::int64_t)
            {
                R::CellSpeciesState state(*catalog);
                const auto volume = environment.bulkVolume;
                state.dispersedPopulations = {{"micro", std::vector<double>(catalog->species().size())},
                    {"large", std::vector<double>(catalog->species().size())}};
                const double micro = 1e6 * volume * 1e-12, large = 1e4 * volume * 1e-9;
                state.set_amount(*catalog, Mixture::gas_one, P::Dispersed, .25 * micro, "micro");
                state.set_amount(*catalog, Mixture::gas_two, P::Dispersed, .75 * micro, "micro");
                state.set_amount(*catalog, Mixture::gas_one, P::Dispersed, .6 * large, "large");
                state.set_amount(*catalog, Mixture::gas_two, P::Dispersed, .4 * large, "large");
                const double liquidVolume = volume - (micro + large) * ordinaryGasMolarVolume;
                const double reactant = .1 * liquidVolume, dissolvedOne = .2 * liquidVolume,
                             dissolvedTwo = .1 * liquidVolume;
                state.set_amount(*catalog, Mixture::reactant, P::Liquid, reactant);
                state.set_amount(*catalog, Mixture::gas_one, P::Dissolved, dissolvedOne);
                state.set_amount(*catalog, Mixture::gas_two, P::Dissolved, dissolvedTwo);
                const double soluteMass = .010 * reactant + .002 * dissolvedOne + .004 * dissolvedTwo;
                const double solventMass = liquid_density * liquidVolume - soluteMass;
                if (!(liquidVolume > 0 && solventMass > 0))
                    throw std::invalid_argument("Cylinder explicit finite species exceed the initial liquid capacity.");
                state.set_amount(*catalog, Mixture::solvent, P::Liquid, solventMass / .018);
                return state;
            };
            reacting_gas.initial_population_numbers = [](const R::ReactionContext& context)
            { return std::vector<double>{1e6 * context.environment.bulkVolume, 1e4 * context.environment.bulkVolume}; };
            reacting_gas.species_source = [catalog](const R::ReactionContext&, double)
            {
                R::ReactionSourceInput input;
                R::ReactionRateProvider law;
                law.identity = "manufactured.cylinder.independent_molar_rate";
                law.revision = 1;
                law.schemaRevision = catalog->revision();
                law.schemaFingerprint = catalog->fingerprint();
                law.validityDescription = "Manufactured finite A/B/two-gas reference-CV smoke problem.";
                law.parameterUnits = "mol reaction/(m3 liquid s)";
                law.validate = [](const R::ReactionContext& context)
                {
                    if (!(context.environment.temperature > 0 && context.environment.absolutePressure > 0))
                        throw std::invalid_argument("Cylinder molar source needs positive thermodynamic inputs.");
                };
                law.evaluate = [](const R::ReactionContext&)
                {
                    return std::vector<R::ReactionRate>{
                        {Mixture::conversion, 2e-4, R::SourceVolumeBasis::LiquidVolume}};
                };
                input.rates = std::move(law);
                return input;
            };
            SimpleFluid::ScalarVoidFractionOptions alpha_options;
            alpha_options.alpha_max = .5;
            coupling.configure_scalar_void_fraction(alpha_options);
            coupling.configure_reacting_gas(reacting_gas);
            // The actual top Neumann boundary supplies q'' once. Finite wall/bulk
            // requests share solvent/void admission and debit admitted latent once.

            SimpleFluid::MaterialFeedbackOptions feedback;
            feedback.density_mode =
                SimpleFluid::DensityFeedbackMode::BoussinesqVoid;
            feedback.reference_density = 1000.0;
            feedback.liquid_density = 1000.0;
            feedback.gas_density = 1.0;
            feedback.reference_temperature = 300.0;
            feedback.thermal_expansion = 2.1e-4;
            feedback.reference_dynamic_viscosity = 1.0e-3;
            coupling.configure_material_feedback(feedback);
        },
        "constant_power_cylinder_vessel.vtu",
        SimpleFluid::SolutionOutputOptions{
            .include_sources = true, .include_material_properties = true, .include_reacting_gas_fields = true});

    return 0;
}
