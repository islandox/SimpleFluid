/** @file testBubbleRepresentation.cc
 * @brief Whole-bubble conversion, material displacement and transactional representation contracts.
 */
#include "solvers/ReactingVoFCoupler.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Solver = AdaptiveVoFSolver;
using State = VoF::LowMachState;
using Vector = VoF::Vector;
using Box = SimpleFluid::Meshes::AMR::AMRBox;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr double reference_temperature = 300.0, reference_pressure = 100000.0;
constexpr double reference_density = 1000.0, dissolved_concentration = 0.01, donor_concentration = 0.02;
constexpr double radius = 0.125, gas_constant = 8.31446261815324;
constexpr std::size_t lower_population = 0, birth_population = 1, upper_population = 2, spectator_population = 3;
const Box donor_box{{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};

double global_sum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

double total(const std::vector<double>& values)
{
    return global_sum(std::accumulate(values.begin(), values.end(), 0.0));
}

Vector total(const std::vector<Vector>& values)
{
    Vector local{}, result{};
    for (const auto& value : values)
        for (std::size_t axis = 0; axis < 3; ++axis)
            local[axis] += value[axis];
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local.data(), result.data());
    return result;
}

double volume(const Box& box)
{
    double result = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result *= box.upper[axis] - box.lower[axis];
    return result;
}

double intersection_volume(const Box& first, const Box& second)
{
    double result = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result *= std::max(
            0.0, std::min(first.upper[axis], second.upper[axis]) - std::max(first.lower[axis], second.lower[axis]));
    return result;
}

double sphere_volume(double r)
{
    return 4.0 * std::numbers::pi / 3.0 * r * r * r;
}

double sphere_moles(double r, double sigma)
{
    return (reference_pressure + 2.0 * sigma / r) * sphere_volume(r) / (gas_constant * reference_temperature);
}

VoF::LowMachOptions thermodynamics(double liquid_reference_energy = 1000.0, double gas_reference_energy = 5000.0)
{
    VoF::LinearCompressibleLiquid liquid;
    liquid.bounds = {250.0, 400.0, 10000.0, 1000000.0};
    liquid.reference_temperature = reference_temperature;
    liquid.reference_pressure = reference_pressure;
    liquid.reference_density = reference_density;
    liquid.isobaric_heat_capacity = 4000.0;
    liquid.thermal_expansion = 0.0;
    liquid.compressibility = 5e-9;
    liquid.reference_internal_energy = liquid_reference_energy;
    VoF::LowMachOptions options;
    options.liquid = VoF::make_linear_compressible_liquid(liquid);
    options.hydrogen = {0.002, 20.0, reference_temperature, gas_reference_energy, 0.0, 0.0};
    options.inert = {0.028, 25.0, reference_temperature, 1.5 * gas_reference_energy, 0.0, 0.0};
    options.volume_relative_tolerance = 2e-11;
    return options;
}

SP<Solver> make_flow(double sigma = 0.0, bool periodic = true, bool refined = false, unsigned maximum_level = 0)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 2};
    grid.block_cells = {4, 4, 4};
    grid.extent = {1.0, 1.0, 1.0};
    grid.periodic = {periodic, periodic, periodic};
    grid.max_level = std::max(maximum_level, refined ? 1U : 0U);
    std::shared_ptr<const SimpleFluid::Meshes::AMR::CartesianAMR> hierarchy =
        std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
    if (refined)
        hierarchy = hierarchy->adapt({0});
    VoFMaterial material;
    material.liquid_density = reference_density;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = 0.0;
    material.surface_tension = sigma;
    VoFSolverOptions flow_options;
    flow_options.gravity = {};
    return std::make_shared<Solver>(hierarchy, material, flow_options);
}

std::vector<VoF::DispersedPopulation> empty_populations(std::size_t cells)
{
    std::vector<VoF::DispersedPopulation> result(4);
    const std::array<std::string, 4> names{"lower", "birth", "upper", "spectator"};
    for (std::size_t q = 0; q < result.size(); ++q)
    {
        result[q].name = names[q];
        result[q].number.assign(cells, 0.0);
        result[q].hydrogen_moles.assign(cells, 0.0);
        result[q].drift_velocity.assign(cells, Vector{});
    }
    return result;
}

VoF::LowMachInitialState dispersed_initial(const Solver& flow, double bubbles = 1.0, bool spectator = true,
    Vector velocity = {0.125, -0.0625, 0.03125}, Vector drift = {0.02, -0.03, 0.04})
{
    const auto cells = flow.mesh()->num_owned_cells();
    const double sigma = flow.material().surface_tension;
    VoF::LowMachInitialState initial;
    initial.thermodynamic_pressure = reference_pressure;
    initial.temperature.assign(cells, reference_temperature);
    initial.resolved_fraction.assign(cells, 0.0);
    initial.velocity.assign(cells, velocity);
    initial.dispersed = empty_populations(cells);
    constexpr double spectator_radius = 0.015625, spectator_number = 0.25;
    for (std::size_t c = 0; c < cells; ++c)
    {
        const auto lid = static_cast<LO>(c);
        const auto box = flow.hierarchy()->cell_box(flow.mesh()->cell_global_id(lid));
        const double cell_volume = flow.mesh()->cell_volume(lid);
        const double weight = intersection_volume(box, donor_box) / volume(donor_box);
        const double number = bubbles * weight;
        auto& donor = initial.dispersed[birth_population];
        donor.number[c] = number;
        donor.hydrogen_moles[c] = number * sphere_moles(radius, sigma);
        donor.drift_velocity[c] = drift;
        double gas_volume = number * sphere_volume(radius);
        if (spectator)
        {
            auto& other = initial.dispersed[spectator_population];
            other.number[c] = spectator_number * weight;
            other.hydrogen_moles[c] = other.number[c] * sphere_moles(spectator_radius, sigma);
            other.drift_velocity[c] = {-0.01, 0.015, -0.025};
            gas_volume += other.number[c] * sphere_volume(spectator_radius);
        }
        const double liquid_volume = cell_volume - gas_volume;
        initial.dissolved_hydrogen_moles.push_back(dissolved_concentration * liquid_volume);
        initial.donor_hydrogen_moles.push_back(donor_concentration * liquid_volume);
    }
    return initial;
}

struct GasRegion
{
    Box box;
    double hydrogen_mole_fraction = 1.0;
};

VoF::LowMachInitialState resolved_initial(
    const Solver& flow, const std::vector<GasRegion>& regions, Vector velocity = {})
{
    const auto cells = flow.mesh()->num_owned_cells();
    VoF::LowMachInitialState initial;
    initial.thermodynamic_pressure = reference_pressure;
    initial.temperature.assign(cells, reference_temperature);
    initial.velocity.assign(cells, velocity);
    initial.dispersed = empty_populations(cells);
    for (std::size_t c = 0; c < cells; ++c)
    {
        const auto lid = static_cast<LO>(c);
        const auto box = flow.hierarchy()->cell_box(flow.mesh()->cell_global_id(lid));
        const double cell_volume = flow.mesh()->cell_volume(lid);
        double gas_volume = 0.0, hydrogen_volume = 0.0;
        for (const auto& region : regions)
        {
            const double overlap = intersection_volume(box, region.box);
            gas_volume += overlap;
            hydrogen_volume += overlap * region.hydrogen_mole_fraction;
        }
        initial.resolved_fraction.push_back(gas_volume / cell_volume);
        initial.resolved_hydrogen_mole_fraction.push_back(gas_volume > 0.0 ? hydrogen_volume / gas_volume : 1.0);
        initial.dissolved_hydrogen_moles.push_back(dissolved_concentration * (cell_volume - gas_volume));
        initial.donor_hydrogen_moles.push_back(donor_concentration * (cell_volume - gas_volume));
    }
    return initial;
}

void expect_scaled(double actual, double expected, double relative = 5e-11, double absolute = 1e-15)
{
    EXPECT_NEAR(actual, expected, std::max(absolute, relative * std::abs(expected)));
}

struct Inventory
{
    double liquid_mass, resolved_hydrogen, inert, dissolved, donor, energy;
    double resolved_volume = 0.0, dispersed_volume = 0.0, liquid_volume = 0.0;
    Vector momentum;
    std::vector<double> number, hydrogen, population_volume;

    Inventory(const Solver& flow, const State& state)
        : liquid_mass(total(state.liquid_mass)), resolved_hydrogen(total(state.resolved_hydrogen_moles)),
          inert(total(state.resolved_inert_moles)), dissolved(total(state.dissolved_hydrogen_moles)),
          donor(total(state.donor_hydrogen_moles)), energy(total(state.internal_energy)),
          momentum(total(state.momentum))
    {
        double gas = 0.0, aqueous = 0.0;
        for (std::size_t c = 0; c < state.resolved_fraction.size(); ++c)
        {
            gas += state.resolved_fraction[c] * flow.mesh()->cell_volume(static_cast<LO>(c));
            aqueous += state.liquid_mass[c] / state.liquid_density[c];
        }
        resolved_volume = global_sum(gas);
        liquid_volume = global_sum(aqueous);
        for (const auto& population : state.dispersed)
        {
            number.push_back(total(population.number));
            hydrogen.push_back(total(population.hydrogen_moles));
            population_volume.push_back(total(population.volume));
            dispersed_volume += population_volume.back();
        }
    }

    [[nodiscard]] double free_gas_hydrogen() const
    {
        return resolved_hydrogen + std::accumulate(hydrogen.begin(), hydrogen.end(), 0.0);
    }

    void expect_conserved(const Inventory& before) const
    {
        expect_scaled(liquid_mass, before.liquid_mass);
        expect_scaled(inert, before.inert);
        expect_scaled(dissolved, before.dissolved);
        expect_scaled(donor, before.donor);
        expect_scaled(free_gas_hydrogen(), before.free_gas_hydrogen());
        expect_scaled(energy, before.energy);
        expect_scaled(liquid_volume, before.liquid_volume);
        expect_scaled(resolved_volume + dispersed_volume, before.resolved_volume + before.dispersed_volume);
        for (std::size_t axis = 0; axis < 3; ++axis)
            expect_scaled(momentum[axis], before.momentum[axis], 5e-11, 1e-13);
    }
};

void expect_uniform_aqueous_composition(const State& state)
{
    for (std::size_t c = 0; c < state.liquid_mass.size(); ++c)
    {
        const double aqueous_volume = state.liquid_mass[c] / state.liquid_density[c];
        expect_scaled(state.dissolved_hydrogen_moles[c], dissolved_concentration * aqueous_volume);
        expect_scaled(state.donor_hydrogen_moles[c], donor_concentration * aqueous_volume);
        EXPECT_NEAR(state.temperature[c], reference_temperature, 2e-7);
    }
}

void expect_same_state(const State& actual, const State& expected)
{
    EXPECT_EQ(actual.pending_representation_relaxation, expected.pending_representation_relaxation);
    EXPECT_DOUBLE_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    EXPECT_EQ(actual.liquid_mass, expected.liquid_mass);
    EXPECT_EQ(actual.resolved_hydrogen_moles, expected.resolved_hydrogen_moles);
    EXPECT_EQ(actual.resolved_inert_moles, expected.resolved_inert_moles);
    EXPECT_EQ(actual.dissolved_hydrogen_moles, expected.dissolved_hydrogen_moles);
    EXPECT_EQ(actual.donor_hydrogen_moles, expected.donor_hydrogen_moles);
    EXPECT_EQ(actual.internal_energy, expected.internal_energy);
    EXPECT_EQ(actual.momentum, expected.momentum);
    EXPECT_EQ(actual.temperature, expected.temperature);
    EXPECT_EQ(actual.resolved_fraction, expected.resolved_fraction);
    EXPECT_EQ(actual.dispersed_fraction, expected.dispersed_fraction);
    EXPECT_EQ(actual.liquid_fraction, expected.liquid_fraction);
    EXPECT_EQ(actual.liquid_density, expected.liquid_density);
    EXPECT_EQ(actual.resolved_density, expected.resolved_density);
    EXPECT_EQ(actual.liquid_absolute_pressure, expected.liquid_absolute_pressure);
    EXPECT_EQ(actual.resolved_absolute_pressure, expected.resolved_absolute_pressure);
    EXPECT_EQ(actual.resolved_hydrogen_partial_pressure, expected.resolved_hydrogen_partial_pressure);
    ASSERT_EQ(actual.dispersed.size(), expected.dispersed.size());
    for (std::size_t q = 0; q < actual.dispersed.size(); ++q)
    {
        EXPECT_EQ(actual.dispersed[q].name, expected.dispersed[q].name);
        EXPECT_EQ(actual.dispersed[q].number, expected.dispersed[q].number);
        EXPECT_EQ(actual.dispersed[q].hydrogen_moles, expected.dispersed[q].hydrogen_moles);
        EXPECT_EQ(actual.dispersed[q].drift_velocity, expected.dispersed[q].drift_velocity);
        EXPECT_EQ(actual.dispersed[q].volume, expected.dispersed[q].volume);
        EXPECT_EQ(actual.dispersed[q].radius, expected.dispersed[q].radius);
        EXPECT_EQ(actual.dispersed[q].absolute_pressure, expected.dispersed[q].absolute_pressure);
    }
}

struct Snapshot
{
    State state;
    double time;
    bool pending_relaxation;
    std::uint64_t topology;
    std::vector<double> fraction, pressure, faces, curvature;
    std::vector<Vector> velocity;

    explicit Snapshot(const Solver& flow)
        : state(flow.low_mach_state()), time(flow.time()), pending_relaxation(flow.pending_representation_relaxation()),
          topology(flow.topology_generation()), fraction(flow.gas_fraction()), pressure(flow.pressure()),
          faces(flow.face_velocity()), curvature(flow.curvature()), velocity(flow.velocity())
    {
    }

    void expect_equal(const Solver& flow) const
    {
        expect_same_state(flow.low_mach_state(), state);
        EXPECT_DOUBLE_EQ(flow.time(), time);
        EXPECT_EQ(flow.pending_representation_relaxation(), pending_relaxation);
        EXPECT_EQ(flow.topology_generation(), topology);
        EXPECT_EQ(flow.gas_fraction(), fraction);
        EXPECT_EQ(flow.pressure(), pressure);
        EXPECT_EQ(flow.face_velocity(), faces);
        EXPECT_EQ(flow.curvature(), curvature);
        EXPECT_EQ(flow.velocity(), velocity);
    }
};

RadiolyticVoFOptions options_for(const Solver& flow)
{
    RadiolyticVoFOptions options;
    options.radiolysis.mode = RadiolyticGasMode::Disabled;
    options.radiolysis.surface_tension = flow.material().surface_tension;
    options.radiolysis.henry_coefficient = 1e-5;
    options.radiolysis.hydrogen_diffusivity = 1e-5;
    options.radiolysis.nucleation_radius = 1e-5;
    options.radiolysis.micro_to_large_conversion_coefficient = 0.0;
    options.dissolved_diffusion = false;
    const double pivot = sphere_moles(radius, flow.material().surface_tension);
    options.upper_populations = {"lower", "birth", "upper"};
    options.upper_pivots.mol_per_bubble = {0.5 * pivot, pivot, 2.0 * pivot};
    options.representation.enabled = true;
    options.representation.promotion_diameter = 0.20;
    options.representation.demotion_diameter = 0.10;
    options.representation.promotion_cells_per_diameter = 2;
    options.representation.demotion_cells_per_diameter = 4;
    options.representation.interface_guard_cells = 1;
    return options;
}

BubbleBirthSeed seed(std::uint64_t id = 100, Vector center = {0.25, 0.25, 0.25}, Box donors = donor_box)
{
    return {id, center, donors, "birth"};
}

const BubbleBirthRecord& birth(const std::vector<BubbleBirthRecord>& records, std::uint64_t id)
{
    const auto found = std::find_if(records.begin(), records.end(), [id](const auto& r) { return r.seed.id == id; });
    if (found == records.end())
        throw std::runtime_error("Expected birth identity is missing from the registry.");
    return *found;
}

bool expect_one_event(
    const BubbleRepresentationInspection& inspection, BubbleConversionKind kind, double hydrogen, double gas_volume)
{
    if (inspection.events.size() != 1U)
    {
        ADD_FAILURE() << "Expected one complete conversion event; observed " << inspection.events.size();
        return false;
    }
    const auto& event = inspection.events.front();
    EXPECT_EQ(event.kind, kind);
    expect_scaled(event.hydrogen_moles, hydrogen);
    expect_scaled(event.gas_volume, gas_volume);
    EXPECT_DOUBLE_EQ(event.dispersed_number_change, kind == BubbleConversionKind::Promotion ? -1.0 : 1.0);
    EXPECT_DOUBLE_EQ(event.resolved_number_change, kind == BubbleConversionKind::Promotion ? 1.0 : -1.0);
    const std::array<double, 2> local_id{
        static_cast<double>(event.event_id >> 32), static_cast<double>(event.event_id & UINT64_C(0xffffffff))};
    std::array<double, 2> minimum{}, maximum{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 2, local_id.data(), minimum.data());
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 2, local_id.data(), maximum.data());
    EXPECT_EQ(minimum, maximum);
    return true;
}

TEST(BubbleRepresentationTest, WholeBubblePromotionMovesCarrierAndPreservesEveryInventoryAndPhysicalMomentum)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow));
    const Snapshot accepted(*flow);
    const Inventory before(*flow, accepted.state);
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    coupler.request_births({seed()});
    ASSERT_EQ(coupler.birth_registry().size(), 1U);
    EXPECT_EQ(coupler.birth_registry().front().status, BubbleBirthStatus::Pending);
    const auto inspection = coupler.inspect_representation();
    EXPECT_EQ(birth(inspection.births, 100).reason, BubbleConversionReason::Ready);
    const auto trial = coupler.preview_representation();
    accepted.expect_equal(*flow);
    EXPECT_TRUE(coupler.conversion_history().empty());
    EXPECT_EQ(coupler.birth_registry().front().status, BubbleBirthStatus::Pending);
    ASSERT_TRUE(expect_one_event(
        trial.representation(), BubbleConversionKind::Promotion, sphere_moles(radius, 0.0), sphere_volume(radius)));
    EXPECT_EQ(trial.representation().events.front().birth_seed_id, 100U);
    EXPECT_DOUBLE_EQ(trial.representation().events.front().surface_energy_change, 0.0);
    const Inventory after(*flow, trial.state());
    after.expect_conserved(before);
    expect_scaled(after.number[birth_population], 0.0);
    expect_scaled(after.hydrogen[birth_population], 0.0);
    expect_scaled(after.resolved_hydrogen, sphere_moles(radius, 0.0));
    expect_scaled(after.resolved_volume, sphere_volume(radius));
    expect_scaled(after.number[spectator_population], before.number[spectator_population]);
    expect_scaled(after.hydrogen[spectator_population], before.hydrogen[spectator_population]);
    expect_uniform_aqueous_composition(trial.state());
    double changed_liquid_cells = 0.0;
    for (std::size_t c = 0; c < trial.state().liquid_mass.size(); ++c)
        if (std::abs(trial.state().liquid_mass[c] - accepted.state.liquid_mass[c]) > 1e-9)
            changed_liquid_cells += 1.0;
    EXPECT_GT(global_sum(changed_liquid_cells), 0.0);
    coupler.commit(trial);
    EXPECT_EQ(coupler.birth_registry().front().status, BubbleBirthStatus::Promoted);
    ASSERT_EQ(coupler.conversion_history().size(), 1U);
    EXPECT_EQ(coupler.conversion_history().front().event_id, trial.representation().events.front().event_id);
    EXPECT_DOUBLE_EQ(flow->time(), accepted.time);
    EXPECT_TRUE(flow->pending_representation_relaxation());
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_produced, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().thermal_input, 0.0);
}

TEST(BubbleRepresentationTest, FractionalDonorsCannotCreateABubbleAndCompetingSeedsConsumeOneWholeInventory)
{
    {
        auto flow = make_flow();
        flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 0.75, false, {}, {}));
        const Snapshot accepted(*flow);
        ReactingVoFCoupler coupler(flow, options_for(*flow));
        coupler.request_births({seed()});
        const auto inspection = coupler.inspect_representation();
        EXPECT_EQ(birth(inspection.births, 100).reason, BubbleConversionReason::InsufficientDonors);
        const auto trial = coupler.preview_representation();
        EXPECT_TRUE(trial.representation().events.empty());
        expect_same_state(trial.state(), accepted.state);
        accepted.expect_equal(*flow);
    }
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.5, false, {}, {}));
    const Inventory before(*flow, flow->low_mach_state());
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    const Box whole_domain{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    coupler.request_births({seed(200, {0.75, 0.25, 0.25}, whole_domain), seed(100, {0.25, 0.25, 0.25}, whole_domain)});
    const auto trial = coupler.preview_representation();
    ASSERT_TRUE(expect_one_event(
        trial.representation(), BubbleConversionKind::Promotion, sphere_moles(radius, 0.0), sphere_volume(radius)));
    EXPECT_EQ(trial.representation().events.front().birth_seed_id, 100U);
    EXPECT_EQ(birth(trial.representation().births, 200).reason, BubbleConversionReason::InsufficientDonors);
    const Inventory after(*flow, trial.state());
    after.expect_conserved(before);
    expect_scaled(after.number[birth_population], 0.5);
    expect_scaled(after.hydrogen[birth_population], 0.5 * sphere_moles(radius, 0.0));
    for (double number : trial.state().dispersed[birth_population].number)
        EXPECT_GE(number, 0.0);
    coupler.commit(trial);
    EXPECT_EQ(birth(coupler.birth_registry(), 100).status, BubbleBirthStatus::Promoted);
    EXPECT_EQ(birth(coupler.birth_registry(), 200).status, BubbleBirthStatus::Pending);
}

TEST(BubbleRepresentationTest, DisjointEventsPreserveLocalCompositionAndReportTheirOwnSurfaceArea)
{
    constexpr double sigma = 0.05, second_radius = 0.14;
    const Box first_box{{0.0, 0.0, 0.0}, {1.0, 0.5, 0.5}};
    const Box second_box{{0.0, 0.5, 0.0}, {1.0, 1.0, 0.5}};
    const std::array<Box, 2> neighborhoods{first_box, second_box};
    const std::array<double, 2> radii{radius, second_radius};
    const std::array<double, 2> concentrations{0.01, 0.03};
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 2};
    grid.block_cells = {6, 6, 6};
    grid.periodic = {true, true, true};
    grid.max_level = 0;
    VoFMaterial material;
    material.liquid_density = reference_density;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = 0.0;
    material.surface_tension = sigma;
    auto flow =
        std::make_shared<Solver>(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid), material);
    const auto cells = flow->mesh()->num_owned_cells();
    VoF::LowMachInitialState initial;
    initial.thermodynamic_pressure = reference_pressure;
    initial.temperature.assign(cells, reference_temperature);
    initial.resolved_fraction.assign(cells, 0.0);
    initial.dispersed = empty_populations(cells);
    for (std::size_t c = 0; c < cells; ++c)
    {
        const auto lid = static_cast<LO>(c);
        const auto box = flow->hierarchy()->cell_box(flow->mesh()->cell_global_id(lid));
        double gas_volume = 0.0, concentration = 0.02;
        for (std::size_t e = 0; e < neighborhoods.size(); ++e)
        {
            const double weight = intersection_volume(box, neighborhoods[e]) / volume(neighborhoods[e]);
            auto& population = initial.dispersed[e == 0 ? birth_population : upper_population];
            population.number[c] = weight;
            population.hydrogen_moles[c] = weight * sphere_moles(radii[e], sigma);
            gas_volume += weight * sphere_volume(radii[e]);
            if (weight > 0.0)
                concentration = concentrations[e];
        }
        const double liquid_volume = flow->mesh()->cell_volume(lid) - gas_volume;
        initial.dissolved_hydrogen_moles.push_back(concentration * liquid_volume);
        initial.donor_hydrogen_moles.push_back(donor_concentration * liquid_volume);
    }
    flow->initialize_low_mach(thermodynamics(), initial);
    const Snapshot accepted(*flow);
    std::array<double, 4> local_first_id, first_id;
    local_first_id.fill(std::numeric_limits<double>::infinity());
    for (std::size_t c = 0; c < cells; ++c)
    {
        const auto gid = flow->mesh()->cell_global_id(static_cast<LO>(c));
        const auto box = flow->hierarchy()->cell_box(gid);
        for (std::size_t e = 0; e < neighborhoods.size(); ++e)
            if (intersection_volume(box, neighborhoods[e]) > 0.0)
            {
                const auto slot = e + (box.lower[0] >= 0.5 ? 2U : 0U);
                local_first_id[slot] = std::min(local_first_id[slot], static_cast<double>(gid));
            }
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 4, local_first_id.data(), first_id.data());
    // Supports interleave A-left,B-left,A-right,B-right in native GID order.
    // A global min-GID carrier match would exchange their distinct solutions.
    ASSERT_LT(first_id[0], first_id[1]);
    ASSERT_LT(first_id[1], first_id[2]);
    ASSERT_LT(first_id[2], first_id[3]);
    const auto measure = [&](const State& state, const Box& region)
    {
        std::array<double, 7> local{}, result{};
        for (std::size_t c = 0; c < cells; ++c)
        {
            const auto lid = static_cast<LO>(c);
            const auto box = flow->hierarchy()->cell_box(flow->mesh()->cell_global_id(lid));
            if (intersection_volume(box, region) == 0.0)
                continue;
            local[0] += state.liquid_mass[c];
            local[1] += state.dissolved_hydrogen_moles[c];
            local[2] += state.donor_hydrogen_moles[c];
            local[3] += state.internal_energy[c];
            local[4] += state.resolved_hydrogen_moles[c];
            local[5] += state.liquid_mass[c] / state.liquid_density[c];
            local[6] += state.resolved_fraction[c] * flow->mesh()->cell_volume(lid);
            for (const auto& population : state.dispersed)
            {
                local[4] += population.hydrogen_moles[c];
                local[6] += population.volume[c];
            }
        }
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 7, local.data(), result.data());
        return result;
    };
    const std::array before{measure(accepted.state, neighborhoods[0]), measure(accepted.state, neighborhoods[1])};
    auto left_halves = neighborhoods;
    for (auto& half : left_halves)
        half.upper[0] = 0.5;
    const std::array before_left{measure(accepted.state, left_halves[0]), measure(accepted.state, left_halves[1])};
    auto options = options_for(*flow);
    options.upper_pivots.mol_per_bubble[2] = sphere_moles(second_radius, sigma);
    ReactingVoFCoupler coupler(flow, options);
    auto second = seed(200, {0.60, 0.75, 0.25}, second_box);
    second.population = "upper";
    coupler.request_births({second, seed(100, {0.40, 0.25, 0.25}, first_box)});
    const auto trial = coupler.preview_representation();
    accepted.expect_equal(*flow);
    EXPECT_TRUE(coupler.conversion_history().empty());
    ASSERT_EQ(trial.representation().events.size(), 2U);
    for (std::size_t e = 0; e < neighborhoods.size(); ++e)
    {
        const auto after = measure(trial.state(), neighborhoods[e]);
        for (std::size_t q = 0; q < after.size(); ++q)
        {
            SCOPED_TRACE(e);
            SCOPED_TRACE(q);
            expect_scaled(after[q], before[e][q]);
        }
        const auto& event = trial.representation().events[e];
        EXPECT_EQ(event.kind, BubbleConversionKind::Promotion);
        EXPECT_EQ(event.birth_seed_id, e == 0 ? 100U : 200U);
        expect_scaled(event.gas_volume, sphere_volume(radii[e]));
        expect_scaled(event.hydrogen_moles, sphere_moles(radii[e], sigma));
        EXPECT_DOUBLE_EQ(event.dispersed_number_change, -1.0);
        EXPECT_DOUBLE_EQ(event.resolved_number_change, 1.0);
        const auto after_left = measure(trial.state(), left_halves[e]);
        const double liquid_shift = after_left[0] - before_left[e][0];
        if (e == 0)
            EXPECT_LT(liquid_shift, -0.1 * reference_density * sphere_volume(radii[e]));
        else
            EXPECT_GT(liquid_shift, 0.1 * reference_density * sphere_volume(radii[e]));
        expect_scaled(liquid_shift / reference_density, before_left[e][6] - after_left[6]);
    }
    for (std::size_t c = 0; c < cells; ++c)
    {
        const auto box = flow->hierarchy()->cell_box(flow->mesh()->cell_global_id(static_cast<LO>(c)));
        double concentration = 0.02;
        bool changed_region = false;
        for (std::size_t e = 0; e < neighborhoods.size(); ++e)
            if (intersection_volume(box, neighborhoods[e]) > 0.0)
            {
                concentration = concentrations[e];
                changed_region = true;
            }
        const double liquid_volume = trial.state().liquid_mass[c] / trial.state().liquid_density[c];
        expect_scaled(trial.state().dissolved_hydrogen_moles[c], concentration * liquid_volume);
        expect_scaled(trial.state().donor_hydrogen_moles[c], donor_concentration * liquid_volume);
        EXPECT_EQ(trial.state().momentum[c], accepted.state.momentum[c]);
        EXPECT_NEAR(trial.state().temperature[c], reference_temperature, 2e-7);
        if (!changed_region)
        {
            EXPECT_DOUBLE_EQ(trial.state().liquid_mass[c], accepted.state.liquid_mass[c]);
            EXPECT_DOUBLE_EQ(trial.state().dissolved_hydrogen_moles[c], accepted.state.dissolved_hydrogen_moles[c]);
        }
    }
    VoF::AdaptiveVoF geometry(flow->hierarchy());
    std::array<double, 3> local_area{}, area_sum{};
    for (const auto& patch : geometry.interface_patches(trial.state().resolved_fraction))
    {
        bool found = false;
        for (std::size_t e = 0; e < neighborhoods.size(); ++e)
        {
            bool contained = true;
            for (std::size_t a = 0; a < 3; ++a)
                contained = contained && patch.centroid[a] >= neighborhoods[e].lower[a] &&
                            patch.centroid[a] <= neighborhoods[e].upper[a];
            if (contained)
            {
                local_area[e] += patch.area;
                found = true;
            }
        }
        if (!found)
            local_area[2] += patch.area;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local_area.data(), area_sum.data());
    EXPECT_NEAR(area_sum[2], 0.0, 1e-14);
    for (std::size_t e = 0; e < neighborhoods.size(); ++e)
    {
        const double expected_area_change = area_sum[e] - 4.0 * std::numbers::pi * radii[e] * radii[e];
        const auto& event = trial.representation().events[e];
        expect_scaled(event.surface_area_change, expected_area_change, 5e-10, 1e-13);
        expect_scaled(event.surface_energy_change, sigma * expected_area_change, 5e-10, 1e-14);
    }
    coupler.commit(trial);
    EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    ASSERT_EQ(coupler.conversion_history().size(), 2U);
    EXPECT_EQ(coupler.conversion_history()[0].birth_seed_id, 100U);
    EXPECT_EQ(coupler.conversion_history()[1].birth_seed_id, 200U);
    EXPECT_EQ(birth(coupler.birth_registry(), 100).status, BubbleBirthStatus::Promoted);
    EXPECT_EQ(birth(coupler.birth_registry(), 200).status, BubbleBirthStatus::Promoted);
}

TEST(BubbleRepresentationTest, CompletePartitionAndCoarseFineComponentsDemoteToTwoNumberConservingPivots)
{
    const Box gas_box{{0.375, 0.375, 0.375}, {0.625, 0.5, 0.5}};
    const double gas_volume = volume(gas_box);
    const double moles = reference_pressure * gas_volume / (gas_constant * reference_temperature);
    for (bool refined : {false, true})
    {
        SCOPED_TRACE(refined);
        auto flow = make_flow(0.0, false, refined);
        flow->initialize_low_mach(thermodynamics(), resolved_initial(*flow, {{gas_box}}));
        const Snapshot accepted(*flow);
        const Inventory before(*flow, accepted.state);
        auto options = options_for(*flow);
        options.upper_populations = {"lower", "upper"};
        options.upper_pivots.mol_per_bubble = {0.5 * moles, 2.0 * moles};
        options.representation.promotion_diameter = 0.4;
        options.representation.demotion_diameter = 0.3;
        ReactingVoFCoupler coupler(flow, options);
        const auto inspection = coupler.inspect_representation();
        ASSERT_EQ(inspection.components.size(), 1U);
        const auto& component = inspection.components.front();
        EXPECT_EQ(component.reason, BubbleConversionReason::Ready);
        EXPECT_FALSE(component.touches_wall);
        EXPECT_FALSE(component.touches_headspace);
        expect_scaled(component.gas_volume, gas_volume);
        expect_scaled(component.hydrogen_moles, moles);
        EXPECT_NEAR(component.centroid[0], 0.5, 1e-13);
        EXPECT_NEAR(component.centroid[1], 0.4375, 1e-13);
        EXPECT_NEAR(component.centroid[2], 0.4375, 1e-13);
        const auto trial = coupler.preview_representation();
        accepted.expect_equal(*flow);
        ASSERT_TRUE(expect_one_event(trial.representation(), BubbleConversionKind::Demotion, moles, gas_volume));
        EXPECT_EQ(trial.representation().events.front().component_label, component.label);
        const Inventory after(*flow, trial.state());
        after.expect_conserved(before);
        expect_scaled(after.resolved_hydrogen, 0.0);
        expect_scaled(after.resolved_volume, 0.0);
        expect_scaled(after.number[lower_population], 2.0 / 3.0);
        expect_scaled(after.number[upper_population], 1.0 / 3.0);
        expect_scaled(after.hydrogen[lower_population], moles / 3.0);
        expect_scaled(after.hydrogen[upper_population], 2.0 * moles / 3.0);
        expect_uniform_aqueous_composition(trial.state());
        double filled_cells = 0.0;
        for (std::size_t c = 0; c < accepted.state.resolved_fraction.size(); ++c)
            if (accepted.state.resolved_fraction[c] == 1.0 && trial.state().liquid_mass[c] > 0.0)
                filled_cells += 1.0;
        EXPECT_GT(global_sum(filled_cells), 0.0);
        coupler.commit(trial);
        EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    }
}

TEST(BubbleRepresentationTest, PromotionAndDemotionRoundTripPreservesBudgetsWithAnExplicitPolicyChange)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.0, false, {}, {}));
    const Inventory original(*flow, flow->low_mach_state());
    ReactingVoFCoupler promote(flow, options_for(*flow));
    promote.request_births({seed(100, {0.5, 0.5, 0.5}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}})});
    const auto promoted = promote.preview_representation();
    ASSERT_EQ(promoted.representation().events.size(), 1U);
    promote.commit(promoted);
    // The intervening physical step clears the explicitly retained phase history.
    // With sigma=0, uniform T and zero velocity it has the independent static solution.
    const auto physical = flow->preview_low_mach_step(1e-4, {});
    EXPECT_NEAR(physical.diagnostics().pressure_work, 0.0, 1e-8);
    flow->commit_low_mach_trial(physical);
    Inventory(*flow, flow->low_mach_state()).expect_conserved(original);
    EXPECT_FALSE(flow->pending_representation_relaxation());
    const auto unchanged_policy = promote.inspect_representation();
    ASSERT_EQ(unchanged_policy.components.size(), 1U);
    EXPECT_EQ(unchanged_policy.components.front().reason, BubbleConversionReason::AboveDemotionDiameter);
    auto demotion_options = options_for(*flow);
    demotion_options.representation.promotion_diameter = 0.4;
    demotion_options.representation.demotion_diameter = 0.3;
    demotion_options.representation.demotion_neighborhood_factor = 2.0;
    demotion_options.representation.maximum_demoted_extent_ratio = 3.0;
    ReactingVoFCoupler demote(flow, demotion_options);
    const double accepted_time = flow->time();
    const auto inspection = demote.inspect_representation();
    ASSERT_EQ(inspection.components.size(), 1U);
    const double component_moles = inspection.components.front().hydrogen_moles;
    const auto& pivots = demotion_options.upper_pivots.mol_per_bubble;
    ASSERT_GE(component_moles, pivots.front());
    ASSERT_LE(component_moles, pivots.back());
    const auto high =
        static_cast<std::size_t>(std::lower_bound(pivots.begin(), pivots.end(), component_moles) - pivots.begin());
    ASSERT_LT(high, pivots.size());
    std::array<double, 3> expected_number{};
    if (component_moles == pivots[high])
        expected_number[high] = 1.0;
    else
    {
        ASSERT_GT(high, 0U);
        const double gap = pivots[high] - pivots[high - 1];
        expected_number[high - 1] = (pivots[high] - component_moles) / gap;
        expected_number[high] = (component_moles - pivots[high - 1]) / gap;
    }
    const Inventory before_demotion(*flow, flow->low_mach_state());
    const auto demoted = demote.preview_representation();
    ASSERT_TRUE(expect_one_event(
        demoted.representation(), BubbleConversionKind::Demotion, sphere_moles(radius, 0.0), sphere_volume(radius)));
    const Inventory after(*flow, demoted.state());
    after.expect_conserved(original);
    expect_scaled(after.number[birth_population], 1.0);
    expect_scaled(after.hydrogen[birth_population], original.hydrogen[birth_population]);
    expect_scaled(after.resolved_hydrogen, 0.0);
    const std::array populations{lower_population, birth_population, upper_population};
    for (std::size_t q = 0; q < expected_number.size(); ++q)
    {
        const auto p = populations[q];
        const double increment = after.number[p] - before_demotion.number[p];
        const double molar_increment = after.hydrogen[p] - before_demotion.hydrogen[p];
        if (expected_number[q] > 0.0)
        {
            // A near-pivot component can require a positive O(epsilon) bin.
            // Its own relative oracle must fail if that small bin is dropped.
            EXPECT_GT(increment, 0.0);
            EXPECT_NEAR(increment, expected_number[q], 2e-9 * expected_number[q]);
            const double expected_moles = expected_number[q] * pivots[q];
            EXPECT_GT(molar_increment, 0.0);
            EXPECT_NEAR(molar_increment, expected_moles, 2e-9 * expected_moles);
            if (expected_number[q] < 1e-10 && Tpetra::getDefaultComm()->getRank() == 0)
                std::cout << std::setprecision(17) << "Stage4 near-pivot oracle: component_moles=" << component_moles
                          << ", pivot=" << pivots[q] << ", expected_number=" << expected_number[q]
                          << ", actual_number_increment=" << increment << ", expected_moles=" << expected_moles
                          << ", actual_molar_increment=" << molar_increment << '\n';
        }
        else
        {
            EXPECT_NEAR(increment, 0.0, 1e-30);
            EXPECT_NEAR(molar_increment, 0.0, 1e-30);
        }
    }
    demote.commit(demoted);
    EXPECT_DOUBLE_EQ(flow->time(), accepted_time);
}

TEST(BubbleRepresentationTest, PhysicalDiameterAndFinestResolutionGuardsPreservePendingBirthIdentity)
{
    {
        auto flow = make_flow();
        flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.0, false, {}, {}));
        auto too_small = options_for(*flow);
        too_small.representation.promotion_diameter = 0.30;
        ReactingVoFCoupler size_guard(flow, too_small);
        size_guard.request_births({seed()});
        EXPECT_EQ(birth(size_guard.inspect_representation().births, 100).reason,
            BubbleConversionReason::BelowPromotionDiameter);
    }
    auto flow = make_flow(0.0, false, false, 1);
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.0, false, {}, {}));
    auto options = options_for(*flow);
    options.representation.promotion_cells_per_diameter = 4;
    ReactingVoFCoupler coupler(flow, options);
    coupler.request_births({seed()});
    const auto coarse = coupler.inspect_representation();
    EXPECT_EQ(birth(coarse.births, 100).reason, BubbleConversionReason::NeedsRefinement);
    ASSERT_FALSE(coarse.refinement.refine.empty());
    const Inventory before(*flow, flow->low_mach_state());
    coupler.regrid(coarse.refinement);
    EXPECT_EQ(birth(coupler.birth_registry(), 100).seed.center, seed().center);
    EXPECT_EQ(birth(coupler.birth_registry(), 100).status, BubbleBirthStatus::Pending);
    Inventory(*flow, flow->low_mach_state()).expect_conserved(before);
    const auto fine = coupler.inspect_representation();
    EXPECT_EQ(birth(fine.births, 100).reason, BubbleConversionReason::Ready);
    const auto trial = coupler.preview_representation();
    ASSERT_TRUE(expect_one_event(
        trial.representation(), BubbleConversionKind::Promotion, sphere_moles(radius, 0.0), sphere_volume(radius)));
    Inventory(*flow, trial.state()).expect_conserved(before);
    auto production_resolution = options_for(*flow);
    production_resolution.representation.promotion_cells_per_diameter = 24;
    ReactingVoFCoupler limited(flow, production_resolution);
    limited.request_births({seed(300)});
    const auto limitation = limited.inspect_representation();
    EXPECT_EQ(birth(limitation.births, 300).reason, BubbleConversionReason::FinestResolutionLimited);
    EXPECT_TRUE(limitation.refinement.resolution_limited);
}

TEST(BubbleRepresentationTest, WallLigamentsHeadspaceAndInertComponentsRemainResolved)
{
    const Box detached{{0.25, 0.625, 0.625}, {0.375, 0.75, 0.75}};
    const Box headspace{{0.375, 0.375, 0.375}, {0.625, 0.5, 0.5}};
    const Box wall_ligament{{0.0, 0.125, 0.125}, {0.5, 0.25, 0.25}};
    const Box inert{{0.625, 0.625, 0.625}, {0.75, 0.75, 0.75}};
    auto flow = make_flow(0.0, false);
    flow->initialize_low_mach(
        thermodynamics(), resolved_initial(*flow, {{detached}, {headspace}, {wall_ligament}, {inert, 0.5}}));
    const Inventory before(*flow, flow->low_mach_state());
    const double molar_density = reference_pressure / (gas_constant * reference_temperature);
    auto options = options_for(*flow);
    options.upper_populations = {"lower", "upper"};
    options.upper_pivots.mol_per_bubble = {
        0.5 * volume(detached) * molar_density, 2.0 * volume(wall_ligament) * molar_density};
    options.representation.promotion_diameter = 0.5;
    options.representation.demotion_diameter = 0.4;
    options.representation.demotion_cells_per_diameter = 16;
    options.representation.maximum_demoted_extent_ratio = 4.0;
    options.headspace.enabled = true;
    options.headspace.regions = {{{0.5, 0.375, 0.375}, {0.625, 0.5, 0.5}}};
    options.headspace.model_name = "designated zero-withdrawal verification reservoir";
    options.headspace.validity_description = "Synthetic isothermal pure H2 gas, with zero vent rate.";
    options.headspace.validate = [](double, double, double) {};
    options.headspace.molar_rate = [](double, double, double) { return 0.0; };
    ReactingVoFCoupler coupler(flow, options);
    const auto inspection = coupler.inspect_representation();
    ASSERT_EQ(inspection.components.size(), 4U);
    std::size_t ready = 0, attached = 0, reservoir = 0, mixed_species = 0;
    for (const auto& component : inspection.components)
    {
        ready += component.reason == BubbleConversionReason::Ready;
        attached += component.reason == BubbleConversionReason::AttachedToBoundary;
        reservoir += component.reason == BubbleConversionReason::HeadspaceConnected;
        mixed_species += component.reason == BubbleConversionReason::ContainsInert;
    }
    EXPECT_EQ(ready, 1U);
    EXPECT_EQ(attached, 1U);
    EXPECT_EQ(reservoir, 1U);
    EXPECT_EQ(mixed_species, 1U);
    const auto trial = coupler.preview_representation();
    ASSERT_TRUE(expect_one_event(
        trial.representation(), BubbleConversionKind::Demotion, volume(detached) * molar_density, volume(detached)));
    const Inventory after(*flow, trial.state());
    after.expect_conserved(before);
    expect_scaled(after.resolved_volume, before.resolved_volume - volume(detached));
    expect_scaled(after.inert, before.inert);
}

TEST(BubbleRepresentationTest, DisjointGasTracesStaySeparateAndPeriodicComponentsAreNotWalls)
{
    {
        SimpleFluid::Meshes::AMR::AMRGrid grid;
        grid.block_cells = {2, 4, 2};
        grid.extent = {2.0, 4.0, 2.0};
        grid.periodic = {false, false, true};
        grid.max_level = 0;
        VoFMaterial material;
        material.liquid_density = reference_density;
        material.gas_density = 0.1;
        material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
        auto flow =
            std::make_shared<Solver>(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid), material);
        VoF::LowMachInitialState initial;
        initial.thermodynamic_pressure = reference_pressure;
        const auto cells = flow->mesh()->num_owned_cells();
        initial.temperature.assign(cells, reference_temperature);
        initial.dispersed = empty_populations(cells);
        for (std::size_t c = 0; c < cells; ++c)
        {
            const auto point = flow->mesh()->cell_centroid(static_cast<LO>(c));
            const double lower_gas = point.y < 1.0 ? 1.0 : (point.y < 2.0 ? 0.5 : 0.0);
            initial.resolved_fraction.push_back(point.x < 1.0 ? lower_gas : 1.0 - lower_gas);
        }
        flow->initialize_low_mach(thermodynamics(), initial);
        ReactingVoFCoupler coupler(flow, options_for(*flow));
        const auto inspection = coupler.inspect_representation();
        // Adjacent half-full cells have disjoint gas traces on their shared face.
        // A graph based only on f>0 would incorrectly join these two components.
        ASSERT_EQ(inspection.components.size(), 2U);
        std::vector<double> volumes;
        for (const auto& component : inspection.components)
        {
            volumes.push_back(component.gas_volume);
            EXPECT_TRUE(component.touches_wall);
        }
        std::sort(volumes.begin(), volumes.end());
        expect_scaled(volumes[0], 3.0);
        expect_scaled(volumes[1], 5.0);
    }
    {
        auto flow = make_flow();
        const Box left{{0.0, 0.375, 0.375}, {0.125, 0.5, 0.5}};
        const Box right{{0.875, 0.375, 0.375}, {1.0, 0.5, 0.5}};
        flow->initialize_low_mach(thermodynamics(), resolved_initial(*flow, {{left}, {right}}));
        ReactingVoFCoupler coupler(flow, options_for(*flow));
        const auto inspection = coupler.inspect_representation();
        ASSERT_EQ(inspection.components.size(), 1U);
        EXPECT_TRUE(inspection.components.front().crosses_periodic_seam);
        EXPECT_FALSE(inspection.components.front().touches_wall);
        expect_scaled(inspection.components.front().gas_volume, volume(left) + volume(right));
    }
}

TEST(BubbleRepresentationTest, CheckpointReplayRestoresBirthReservationsAndRejectsStaleControllerTrials)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.5, false, {}, {}));
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    coupler.request_births({seed()});
    const auto obsolete = coupler.preview_representation();
    const Snapshot initial(*flow);
    coupler.request_births({seed(200, {0.75, 0.25, 0.25}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}})});
    EXPECT_THROW(coupler.commit(obsolete), std::exception);
    initial.expect_equal(*flow);
    const auto checkpoint = coupler.checkpoint();
    const auto first = coupler.preview_representation();
    ASSERT_EQ(first.representation().events.size(), 1U);
    EXPECT_TRUE(coupler.conversion_history().empty());
    const auto event_id = first.representation().events.front().event_id;
    const State first_state = first.state();
    coupler.commit(first);
    ASSERT_EQ(coupler.conversion_history().size(), 1U);
    EXPECT_EQ(coupler.conversion_history().front().event_id, event_id);
    const Snapshot committed(*flow);
    EXPECT_THROW(coupler.commit(first), std::exception);
    committed.expect_equal(*flow);
    coupler.restore(checkpoint);
    initial.expect_equal(*flow);
    EXPECT_TRUE(coupler.conversion_history().empty());
    EXPECT_EQ(birth(coupler.birth_registry(), 100).status, BubbleBirthStatus::Pending);
    EXPECT_EQ(birth(coupler.birth_registry(), 200).status, BubbleBirthStatus::Pending);
    EXPECT_THROW(coupler.commit(first), std::exception);
    initial.expect_equal(*flow);
    const auto replay = coupler.preview_representation();
    ASSERT_EQ(replay.representation().events.size(), 1U);
    EXPECT_EQ(replay.representation().events.front().event_id, event_id);
    expect_same_state(replay.state(), first_state);
    coupler.commit(replay);
    ASSERT_EQ(coupler.conversion_history().size(), 1U);
    EXPECT_EQ(coupler.conversion_history().front().event_id, event_id);
    coupler.release_checkpoint(checkpoint);
}

TEST(BubbleRepresentationTest, PositiveSurfaceTensionConversionCarriesPressureHistoryUntilAPhysicalStep)
{
    constexpr double sigma = 5.0;
    auto flow = make_flow(sigma);
    // Zero energy references keep this oracle sensitive to the small physical
    // capillary work, without subtracting a large arbitrary caloric offset.
    flow->initialize_low_mach(thermodynamics(0.0, 0.0), dispersed_initial(*flow, 1.0, false, {}, {}));
    const State initial_state = flow->low_mach_state();
    const Inventory before(*flow, flow->low_mach_state());
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    coupler.request_births({seed()});
    const auto trial = coupler.preview_representation();
    ASSERT_TRUE(expect_one_event(
        trial.representation(), BubbleConversionKind::Promotion, sphere_moles(radius, sigma), sphere_volume(radius)));
    Inventory(*flow, trial.state()).expect_conserved(before);
    EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    const auto& event = trial.representation().events.front();
    expect_scaled(event.surface_energy_change, sigma * event.surface_area_change);
    coupler.commit(trial);
    EXPECT_TRUE(flow->pending_representation_relaxation());
    EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    EXPECT_DOUBLE_EQ(flow->low_mach_diagnostics().pressure_work, 0.0);
    EXPECT_DOUBLE_EQ(flow->low_mach_diagnostics().net_thermal_energy, 0.0);
    VoF::AdaptiveVoF geometry(flow->hierarchy());
    double local_area = 0.0;
    for (const auto& patch : geometry.interface_patches(flow->gas_fraction()))
        local_area += patch.area;
    const double area_change = global_sum(local_area) - 4.0 * std::numbers::pi * radius * radius;
    expect_scaled(event.surface_area_change, area_change, 5e-10, 1e-13);
    EXPECT_GT(std::abs(event.surface_energy_change), 1e-5);
    double local_mismatch = 0.0;
    const auto& carried = flow->low_mach_state();
    for (std::size_t c = 0; c < carried.resolved_fraction.size(); ++c)
        if (carried.resolved_fraction[c] > 1e-12 && carried.liquid_fraction[c] > 1e-12)
        {
            const double jump = carried.resolved_absolute_pressure[c] - carried.liquid_absolute_pressure[c];
            EXPECT_NEAR(jump, 2.0 * sigma / radius, 2e-7);
            local_mismatch = std::max(local_mismatch, std::abs(jump - sigma * flow->curvature()[c]));
        }
    double mismatch = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local_mismatch, &mismatch);
    EXPECT_GT(mismatch, 1e-5);
    constexpr double dt = 1e-2;
    ASSERT_LE(dt, flow->stable_timestep());
    const auto physical = flow->preview_low_mach_step(dt, {});
    const auto& diagnostic = physical.diagnostics();
    const Inventory after(*flow, physical.state());
    expect_scaled(after.liquid_mass, before.liquid_mass);
    expect_scaled(after.free_gas_hydrogen(), before.free_gas_hydrogen());
    expect_scaled(after.dissolved, before.dissolved);
    expect_scaled(after.donor, before.donor);
    long double local_energy_change = 0.0L, local_energy_magnitude = 0.0L;
    for (std::size_t c = 0; c < initial_state.internal_energy.size(); ++c)
    {
        local_energy_change += static_cast<long double>(physical.state().internal_energy[c]) -
                               static_cast<long double>(initial_state.internal_energy[c]);
        local_energy_magnitude += std::abs(static_cast<long double>(physical.state().internal_energy[c])) +
                                  std::abs(static_cast<long double>(initial_state.internal_energy[c]));
    }
    const double energy_change = global_sum(static_cast<double>(local_energy_change));
    const double roundoff =
        64.0 * std::numeric_limits<double>::epsilon() * global_sum(static_cast<double>(local_energy_magnitude));
    // 2e-9 J covers the original energy acceptance floor (8*2e-10 J),
    // independently of energy reference; require the event to exceed it clearly.
    const double energy_bound = std::max(2e-9, roundoff);
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << std::setprecision(17) << "Stage4 pressure-history oracle: sigma=" << sigma << ", dt=" << dt
                  << ", surface_energy_change=" << event.surface_energy_change
                  << ", curvature_jump_mismatch=" << mismatch << ", pressure_work=" << diagnostic.pressure_work
                  << ", internal_energy_change=" << energy_change
                  << ", energy_error=" << energy_change - diagnostic.pressure_work << ", energy_bound=" << energy_bound
                  << '\n';
    EXPECT_GT(std::abs(diagnostic.pressure_work), 200.0 * energy_bound);
    EXPECT_NEAR(energy_change, diagnostic.pressure_work, energy_bound);
    EXPECT_LE(diagnostic.maximum_occupancy_error, 2e-8);
    EXPECT_LE(diagnostic.maximum_continuity_residual, 5e-8);
    EXPECT_NEAR(diagnostic.mass_error, 0.0, 5e-10 * diagnostic.total_mass);
    EXPECT_NEAR(diagnostic.hydrogen_error, 0.0, 5e-10 * diagnostic.hydrogen_inventory);
    flow->commit_low_mach_trial(physical);
    EXPECT_DOUBLE_EQ(flow->time(), dt);
    EXPECT_FALSE(flow->pending_representation_relaxation());
    const auto& relaxed = flow->low_mach_state();
    for (std::size_t c = 0; c < relaxed.resolved_fraction.size(); ++c)
        EXPECT_NEAR(relaxed.resolved_absolute_pressure[c] - relaxed.liquid_absolute_pressure[c],
            sigma * flow->curvature()[c], 2e-7);
}

TEST(BubbleRepresentationTest, InvalidSparseParcelAndRankLocalStaleRevisionRejectWithoutPublication)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.0, false, {}, {}));
    const Snapshot accepted(*flow);
    VoF::RepresentationChangeRequest request;
    request.source_hierarchy = flow->hierarchy();
    request.accepted_state_revision = flow->state_revision();
    // A 1e-12 relative one-parcel excess is far above its arithmetic error,
    // even though it is below the global inventory acceptance tolerance.
    for (double factor : {2.0, 1.0 + 1e-12})
    {
        SCOPED_TRACE(factor);
        request.gas_parcels.clear();
        for (std::size_t c = 0; c < accepted.state.dispersed[birth_population].volume.size(); ++c)
            if (accepted.state.dispersed[birth_population].volume[c] > 0.0)
            {
                const auto gid = static_cast<std::uint64_t>(flow->mesh()->cell_global_id(static_cast<LO>(c)));
                request.gas_parcels.push_back({gid, gid, static_cast<int>(birth_population), -1,
                    factor * accepted.state.dispersed[birth_population].volume[c], 0.0});
                break;
            }
        EXPECT_THROW((void) flow->preview_representation_change(request), std::exception);
        accepted.expect_equal(*flow);
        EXPECT_EQ(flow->state_revision(), request.accepted_state_revision);
    }
    request.gas_parcels.clear();
    if (Tpetra::getDefaultComm()->getRank() == 0)
        ++request.accepted_state_revision;
    EXPECT_THROW((void) flow->preview_representation_change(request), std::exception);
    accepted.expect_equal(*flow);
}

TEST(BubbleRepresentationTest, CoupledContinuationReplaysPendingHistoryWithoutDuplicatingHeatOrRadiolysis)
{
    constexpr double sigma = 0.05, dt = 1e-4, conversion_radius = 0.25;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {4, 4, 4};
    grid.periodic = {true, true, true};
    grid.max_level = 0;
    VoFMaterial material;
    material.liquid_density = reference_density;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = 0.0;
    material.surface_tension = sigma;
    auto flow =
        std::make_shared<Solver>(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid), material);
    const auto cells = flow->mesh()->num_owned_cells();
    const double pivot = sphere_moles(conversion_radius, sigma);
    VoF::LowMachInitialState initial;
    initial.thermodynamic_pressure = reference_pressure;
    initial.temperature.assign(cells, reference_temperature);
    initial.resolved_fraction.assign(cells, 0.0);
    initial.dispersed = empty_populations(cells);
    for (std::size_t c = 0; c < cells; ++c)
    {
        const double cell_volume = flow->mesh()->cell_volume(static_cast<LO>(c));
        initial.dispersed[birth_population].number[c] = cell_volume;
        initial.dispersed[birth_population].hydrogen_moles[c] = cell_volume * pivot;
        const double liquid_volume = cell_volume * (1.0 - sphere_volume(conversion_radius));
        initial.dissolved_hydrogen_moles.push_back(dissolved_concentration * liquid_volume);
        initial.donor_hydrogen_moles.push_back(donor_concentration * liquid_volume);
    }
    VoF::DispersedPopulation micro;
    micro.name = "micro";
    micro.number.assign(initial.temperature.size(), 0.0);
    micro.hydrogen_moles = micro.number;
    micro.drift_velocity.assign(initial.temperature.size(), Vector{});
    const auto micro_index = initial.dispersed.size();
    initial.dispersed.push_back(std::move(micro));
    flow->initialize_low_mach(thermodynamics(0.0, 0.0), initial);
    auto options = options_for(*flow);
    options.upper_pivots.mol_per_bubble = {0.5 * pivot, pivot, 2.0 * pivot};
    options.radiolysis.mode = RadiolyticGasMode::Sheng2024TwoPopulation;
    options.radiolysis.hydrogen_yield_mol_per_j = 2e-7;
    options.radiolysis.gas_release_efficiency = 0.5;
    options.radiolysis.microbubble_lifetime = 1e100;
    ReactingVoFCoupler coupler(flow, options);
    coupler.request_births({seed(100, {0.5, 0.5, 0.5}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}})});
    const auto conversion = coupler.preview_representation();
    ASSERT_TRUE(expect_one_event(conversion.representation(), BubbleConversionKind::Promotion,
        sphere_moles(conversion_radius, sigma), sphere_volume(conversion_radius)));
    coupler.commit(conversion);
    ASSERT_TRUE(flow->pending_representation_relaxation());
    const Snapshot pending(*flow);
    const Inventory before(*flow, pending.state);
    const auto event_id = coupler.conversion_history().front().event_id;
    RadiolyticVoFInput input;
    for (std::size_t c = 0; c < flow->mesh()->num_owned_cells(); ++c)
    {
        const double cell_volume = flow->mesh()->cell_volume(static_cast<LO>(c));
        input.absorbed_radiation_energy.push_back(100.0 * cell_volume);
        input.net_thermal_energy.push_back(20.0 * cell_volume);
    }
    const double expected_production = 1e-7 * total(input.absorbed_radiation_energy);
    const double expected_heat = total(input.net_thermal_energy);
    const auto checkpoint = coupler.checkpoint();
    const auto trial = coupler.preview(dt, input);
    pending.expect_equal(*flow);
    EXPECT_TRUE(flow->pending_representation_relaxation());
    EXPECT_FALSE(trial.state().pending_representation_relaxation);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_produced, 0.0);
    expect_scaled(trial.statistics().hydrogen_produced, expected_production, 2e-13, 1e-20);
    // This diagnostic subtracts two donor endpoint totals. Its roundoff scale
    // is the stored donor stock, unlike the direct production-event ledger.
    const double donor_roundoff =
        32.0 * std::numeric_limits<double>::epsilon() * (before.donor + total(trial.state().donor_hydrogen_moles));
    EXPECT_NEAR(
        trial.statistics().donor_consumed, expected_production, std::max(2e-13 * expected_production, donor_roundoff));
    expect_scaled(trial.statistics().thermal_input, expected_heat, 2e-13);
    expect_scaled(total(trial.state().dispersed[micro_index].hydrogen_moles), expected_production, 2e-9, 1e-20);
    const State first_state = trial.state();
    coupler.commit(trial);
    EXPECT_DOUBLE_EQ(flow->time(), dt);
    EXPECT_FALSE(flow->pending_representation_relaxation());
    ASSERT_EQ(coupler.conversion_history().size(), 1U);
    EXPECT_EQ(coupler.conversion_history().front().event_id, event_id);
    expect_scaled(coupler.statistics().cumulative_produced, expected_production, 2e-13, 1e-20);
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_captured, 0.0);
    const Inventory after(*flow, flow->low_mach_state());
    EXPECT_NEAR(
        after.free_gas_hydrogen() - before.free_gas_hydrogen(), expected_production, 2e-9 * expected_production);
    EXPECT_NEAR(after.donor - before.donor, -expected_production, 2e-9 * expected_production);
    EXPECT_NEAR(after.liquid_mass - before.liquid_mass,
        -flow->low_mach_options().hydrogen.molar_mass * expected_production, 1e-11);
    expect_scaled(after.dissolved, before.dissolved);
    long double local_energy_change = 0.0L;
    for (std::size_t c = 0; c < pending.state.internal_energy.size(); ++c)
        local_energy_change += static_cast<long double>(flow->low_mach_state().internal_energy[c]) -
                               static_cast<long double>(pending.state.internal_energy[c]);
    const auto& diagnostic = flow->low_mach_diagnostics();
    EXPECT_NEAR(global_sum(static_cast<double>(local_energy_change)), expected_heat + diagnostic.pressure_work,
        1e-8 * expected_heat);
    expect_scaled(diagnostic.net_thermal_energy, expected_heat, 2e-13);
    expect_scaled(diagnostic.radiolysis_energy, total(input.absorbed_radiation_energy), 2e-13);
    EXPECT_LE(diagnostic.maximum_occupancy_error, 2e-8);
    EXPECT_LE(diagnostic.maximum_continuity_residual, 5e-8);
    coupler.restore(checkpoint);
    pending.expect_equal(*flow);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_produced, 0.0);
    ASSERT_EQ(coupler.conversion_history().size(), 1U);
    EXPECT_EQ(coupler.conversion_history().front().event_id, event_id);
    EXPECT_THROW(coupler.commit(trial), std::exception);
    const auto replay = coupler.preview(dt, input);
    expect_same_state(replay.state(), first_state);
    pending.expect_equal(*flow);
    coupler.commit(replay);
    EXPECT_DOUBLE_EQ(flow->time(), dt);
    expect_scaled(coupler.statistics().cumulative_produced, expected_production, 2e-13, 1e-20);
    EXPECT_EQ(coupler.conversion_history().size(), 1U);
    coupler.release_checkpoint(checkpoint);
}

TEST(BubbleRepresentationTest, DivergentBirthIdentitiesAndReusedPayloadsRejectCollectively)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.0, false, {}, {}));
    const Snapshot accepted(*flow);
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    coupler.request_births({seed()});
    EXPECT_THROW(coupler.request_births({seed(100, {0.3, 0.25, 0.25})}), std::exception);
    accepted.expect_equal(*flow);
    ASSERT_EQ(coupler.birth_registry().size(), 1U);
    EXPECT_EQ(coupler.birth_registry().front().seed.center, seed().center);
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() > 1)
    {
        EXPECT_THROW(
            coupler.request_births({seed(200U + static_cast<std::uint64_t>(comm->getRank()))}), std::exception);
        accepted.expect_equal(*flow);
        EXPECT_EQ(coupler.birth_registry().size(), 1U);
    }
    const auto trial = coupler.preview_representation();
    ASSERT_TRUE(expect_one_event(
        trial.representation(), BubbleConversionKind::Promotion, sphere_moles(radius, 0.0), sphere_volume(radius)));
    EXPECT_EQ(trial.representation().events.front().birth_seed_id, 100U);
}

TEST(BubbleRepresentationTest, IncompatibleCarriedPhaseTemperaturesRejectWithoutThermalRelaxation)
{
    auto flow = make_flow();
    auto initial = dispersed_initial(*flow, 1.0, false, {}, {});
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const auto box = flow->hierarchy()->cell_box(flow->mesh()->cell_global_id(static_cast<LO>(c)));
        if (intersection_volume(box, donor_box) > 0.0)
            initial.temperature[c] = 310.0;
    }
    flow->initialize_low_mach(thermodynamics(), initial);
    const Snapshot accepted(*flow);
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    // The complete hot donor bubble is inserted into cold mixed cells on the
    // other side of the partition. A one-temperature cell cannot preserve both
    // carried phase energies; conversion must not perform an implicit heat step.
    coupler.request_births({seed(100, {0.70, 0.70, 0.70}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}})});
    EXPECT_EQ(birth(coupler.inspect_representation().births, 100).reason, BubbleConversionReason::Ready);
    std::string failure;
    try
    {
        (void) coupler.preview_representation();
    }
    catch (const std::exception& error)
    {
        failure = error.what();
    }
    EXPECT_TRUE(failure.find("common-temperature") != std::string::npos ||
                failure.find("phase-temperature compatibility") != std::string::npos)
        << "Expected a carried-temperature incompatibility, observed: " << failure;
    accepted.expect_equal(*flow);
    EXPECT_EQ(coupler.birth_registry().front().status, BubbleBirthStatus::Pending);
    EXPECT_TRUE(coupler.conversion_history().empty());
}

TEST(BubbleRepresentationTest, EmptyOwnedRankParticipatesInRegistryPreviewAndReplay)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {1.0, 1.0, 1.0};
    grid.periodic = {true, true, true};
    grid.max_level = 0;
    VoFMaterial material;
    material.liquid_density = reference_density;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
    auto flow =
        std::make_shared<Solver>(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid), material);
    flow->initialize_low_mach(thermodynamics(), dispersed_initial(*flow, 1.0, false, {}, {}));
    if (Tpetra::getDefaultComm()->getSize() > 8)
        EXPECT_GT(global_sum(flow->mesh()->num_owned_cells() == 0 ? 1.0 : 0.0), 0.0);
    const Snapshot accepted(*flow);
    ReactingVoFCoupler coupler(flow, options_for(*flow));
    coupler.request_births({seed()});
    const auto checkpoint = coupler.checkpoint();
    const auto trial = coupler.preview_representation();
    EXPECT_TRUE(trial.representation().events.empty());
    expect_same_state(trial.state(), accepted.state);
    accepted.expect_equal(*flow);
    coupler.commit(trial);
    coupler.restore(checkpoint);
    accepted.expect_equal(*flow);
    EXPECT_EQ(coupler.birth_registry().size(), 1U);
    EXPECT_EQ(coupler.birth_registry().front().seed.id, 100U);
    EXPECT_THROW(coupler.commit(trial), std::exception);
    const auto replay = coupler.preview_representation();
    EXPECT_TRUE(replay.representation().events.empty());
    expect_same_state(replay.state(), accepted.state);
    coupler.release_checkpoint(checkpoint);
    // The controller correctly leaves this under-resolved birth pending.
    // Independently exercise the core's nonempty sparse packet exchange on
    // every rank, including the rank owning no cells in the exact-nine run.
    // This same-cell conversion is an inventory oracle, not a resolved shape.
    VoF::RepresentationChangeRequest request;
    request.source_hierarchy = flow->hierarchy();
    request.accepted_state_revision = flow->state_revision();
    for (std::size_t c = 0; c < flow->mesh()->num_owned_cells(); ++c)
    {
        const auto gid = static_cast<std::uint64_t>(flow->mesh()->cell_global_id(static_cast<LO>(c)));
        if (gid == 0)
            request.gas_parcels.push_back({gid, gid, static_cast<int>(birth_population), -1,
                flow->low_mach_state().dispersed[birth_population].volume[c], 0.0});
    }
    EXPECT_DOUBLE_EQ(global_sum(static_cast<double>(request.gas_parcels.size())), 1.0);
    const Inventory before(*flow, flow->low_mach_state());
    const auto active = flow->preview_representation_change(request);
    accepted.expect_equal(*flow);
    EXPECT_TRUE(active.is_representation_change());
    const Inventory after(*flow, active.state());
    after.expect_conserved(before);
    expect_scaled(after.number[birth_population], 0.0);
    expect_scaled(after.resolved_hydrogen, sphere_moles(radius, 0.0));
    expect_scaled(after.resolved_volume, sphere_volume(radius));
    expect_scaled(active.representation_diagnostics().gas_volume_promoted, sphere_volume(radius));
    expect_scaled(active.representation_diagnostics().hydrogen_promoted, sphere_moles(radius, 0.0));
    expect_scaled(active.representation_diagnostics().dispersed_number_removed, 1.0);
    EXPECT_DOUBLE_EQ(active.diagnostics().pressure_work, 0.0);
    flow->commit_low_mach_trial(active);
    EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    EXPECT_TRUE(flow->pending_representation_relaxation());
    const Snapshot converted(*flow);
    EXPECT_THROW(coupler.commit(replay), std::exception);
    converted.expect_equal(*flow);
}
} // namespace
