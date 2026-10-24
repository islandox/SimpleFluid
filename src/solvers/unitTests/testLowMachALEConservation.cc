/** Independent moving-volume energy and counterflow conservation oracles. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/CompensatedSum.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using Face = Flow::Traits::scalar_face_type;
using Scalar = Flow::Traits::scalar_cell_type;
constexpr double gas_constant = 8.31446261815324;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Flow::Mesh> make_mesh(bool semi, ArrReal z)
{
    Flow::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    if (semi)
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::SemiStructuredXY_Z>(
                Arr<Meshes::SemiStructuredXY_Z::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}},
                Arr<Arr<unsigned>>{{0, 1, 2, 3}}, std::move(z)),
            distribution);
    return std::make_shared<Flow::Mesh>(
        std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, std::move(z)}),
        distribution);
}
double sum(const Flow::Mesh& mesh, double value)
{
    double global = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &value, &global);
    return global;
}
double volume(const R::PhaseThermalState& phase)
{
    const double n = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.);
    if (n == 0)
        return 0.;
    double mass = 0;
    auto fractions = phase.moles;
    for (size_t i = 0; i < fractions.size(); ++i)
    {
        mass += fractions[i] * phase.thermodynamics->catalog().species()[i].molarMass;
        fractions[i] /= n;
    }
    return mass / phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, fractions).density;
}
double balance(const Flow::Mesh& mesh, const Face& flux, LO cell)
{
    double value = 0;
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const LO face = static_cast<LO>(f);
        if (mesh.owner_cell(face) == cell)
            value += flux.local_value(face);
        else if (!mesh.is_geometry_exterior_face(face) && mesh.neighbor_cell(face) == cell)
            value -= flux.local_value(face);
    }
    return value;
}
Flow::Options options()
{
    Flow::Options value;
    value.minimum_pressure = 20000;
    value.maximum_pressure = 1000000;
    value.linear.preconditioner = LinearPreconditioner::None;
    value.linear.tolerance = 1e-12;
    return value;
}

TEST(LowMachALEConservationTest, ClosedGasExpansionMatchesBackwardEulerWorkWithIndependentEnergyReference)
{
    constexpr double old_pressure = 100000, old_temperature = 300, cv = 20, target = 1.01, dt = .1;
    const double n = old_pressure / (gas_constant * old_temperature);
    // N cv (Tnew-Told) = -pnew (Vnew-Vold), pnew Vnew = N R Tnew.
    const double expected_pressure = old_pressure / (target + gas_constant / cv * (target - 1));
    const double expected_temperature = expected_pressure * target / (n * gas_constant);
    const double expected_work = -expected_pressure * (target - 1);
    for (bool semi : {false, true})
        for (double reference_shift : {0., 1e6})
        {
            auto mesh = make_mesh(semi, {0, 1});
            auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
                {47, "ordinary finite gas", .028, {{"G", 1}}, {R::SpeciesPhase::Resolved}}});
            auto gas = R::make_ideal_gas_mixture(catalog, {{cv, 300, cv * 300 + reference_shift}},
                {250, 600, 10000, 2000000}, R::EnergyConvention::SensibleEnergy);
            Flow flow(mesh, {gas}, options());
            Flow::State initial;
            initial.thermodynamic_pressure = old_pressure;
            initial.cells.resize(mesh->num_owned_cells());
            for (auto& cell : initial.cells)
                cell.phases = {{gas, {n}, old_pressure, n * (cv * 300 + reference_shift), old_temperature}};
            flow.initialize(initial);
            const auto before = flow.state();
            Flow::PlanarMotion motion(mesh);
            const auto trial = flow.preview_ale(dt, motion, target);
            EXPECT_EQ(flow.state().revision, before.revision);
            EXPECT_FALSE(motion.has_active_trial());
            EXPECT_NEAR(trial.state().thermodynamic_pressure, expected_pressure, 1e-5);
            double energy_change = 0, amount = 0;
            for (size_t c = 0; c < trial.state().cells.size(); ++c)
            {
                const auto& phase = trial.state().cells[c].phases[0];
                EXPECT_NEAR(phase.temperature, expected_temperature, 1e-7);
                EXPECT_NEAR(volume(phase), target, flow.options().volume_relative_tolerance * target);
                EXPECT_DOUBLE_EQ(trial.transport_snapshot().old_control_volumes[c], 1.);
                EXPECT_DOUBLE_EQ(trial.transport_snapshot().new_control_volumes[c], target);
                EXPECT_DOUBLE_EQ(trial.transport_snapshot().cells[c].phases[0].advected_material_volume, 1.);
                energy_change += phase.internal_energy - before.cells[c].phases[0].internal_energy;
                amount += phase.moles[0];
            }
            EXPECT_NEAR(sum(*mesh, amount), n, 1e-12 * n);
            EXPECT_NEAR(trial.diagnostics().pressure_work, expected_work, 1e-6);
            EXPECT_NEAR(
                sum(*mesh, energy_change), expected_work, trial.diagnostics().internal_energy_roundoff_bound + 1e-6);
            flow.commit_ale(trial, motion);
            for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<LO>(c)), target);
            EXPECT_EQ(
                sum(*mesh, mesh->num_owned_cells() == 0 ? 1. : 0.), mesh->owned_cell_map()->getComm()->getSize() - 1);
        }
}

TEST(LowMachALEConservationTest, MovingCounterflowKeepsNonunitDonorsAndSeparatesSweptVolumeFromPhysicalWork)
{
    for (bool semi : {false, true})
    {
        auto mesh = make_mesh(semi, {0, .25, .5, .75, 1});
        const std::vector roles{R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed};
        auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {103, "liquid carrier", .02, {{"X", 1}}, roles}, {709, "dispersed carrier", .02, {{"X", 1}}, roles}});
        auto liquid =
            R::make_linear_compressible_mixture_liquid(catalog, {250, 600, 10000, 2000000}, [](std::span<const double>)
                { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
        auto gas = R::make_ideal_gas_mixture(catalog, {{20, 300, 6000}, {20, 300, 6000}}, {250, 600, 10000, 2000000});
        auto controls = options();
        // A small population arriving two faces downstream needs componentwise
        // accuracy beyond the global RHS-norm gate. Use the supported BiCGStab
        // policy here; every physical count/material/energy gate remains active.
        controls.linear.backend = LinearSolverBackend::BiCGStab;
        controls.linear.tolerance = 1e-15;
        controls.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solution"}, {Flow::PhaseRole::DispersedGas, "micro"}};
        Flow flow(mesh, {liquid, gas}, controls);
        Flow::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const LO cell = static_cast<LO>(c);
            const double z = mesh->cell_centroid(cell).z, v = mesh->cell_volume(cell);
            const double alpha = z < .5 ? .2 : 0;
            const double mass = (1 - alpha) * v * 1000, n = alpha * v * 100000 / (gas_constant * 300);
            initial.cells[c].phases = {{liquid, {mass / .02, 0}, 100000, mass * 1.2e6, 350},
                {gas, {0, n}, 100000, n * 6000, n > 0 ? 300. : 0.}};
            initial.cells[c].phase_numbers = {std::nullopt, alpha * v * 1000};
            initial.cells[c].drift_velocity = {Flow::Vector{}, Flow::Vector{0, 0, z > .25 && z < .75 ? .1 : 0.}};
        }
        flow.initialize(initial);
        const auto accepted = flow.state();
        Flow::PlanarMotion motion(mesh);
        const auto trial = flow.preview_ale(.001, motion, 1.00001);
        const auto packets = trial.owned_phase_transport();
        const auto& snapshot = trial.transport_snapshot();
        Face phase0(mesh, 0., "independent_liquid_relative_material"),
            phase1(mesh, 0., "independent_gas_relative_material"),
            actual(mesh, 0., "independent_corrected_relative_material"),
            number_flux(mesh, 0., "independent_counterflow_number_flux");
        Scalar number_concentration(mesh, 0., "independent_counterflow_number_concentration");
        for (size_t c = 0; c < accepted.cells.size(); ++c)
            number_concentration.set_owned_value(
                static_cast<LO>(c), *snapshot.cells[c].phases[1].population_number / snapshot.new_control_volumes[c]);
        number_concentration.sync_ghosts();
        double theta_max = 0;
        for (size_t f = 0; f < packets.size(); ++f)
        {
            const LO face = static_cast<LO>(f);
            const auto& packet = packets[f];
            double theta = 0, slip = 0;
            for (size_t a = 0; a < 2; ++a)
            {
                theta += packet.material_fraction[a];
                slip += packet.material_fraction[a] * (packet.phase_volume_flux[a] - packet.carrier_volume_flux);
            }
            if (!mesh->is_geometry_exterior_face(face))
                theta_max = std::max(theta_max, theta);
            else
                EXPECT_DOUBLE_EQ(trial.owned_volume_flux()[f], packet.mesh_volume_flux);
            phase0.set_owned_value(face, packet.material_fraction[0] * packet.relative_phase_volume_flux[0]);
            phase1.set_owned_value(face, packet.material_fraction[1] * packet.relative_phase_volume_flux[1]);
            actual.set_owned_value(face, theta * (trial.owned_volume_flux()[f] - packet.mesh_volume_flux) + slip);
            if (!mesh->is_geometry_exterior_face(face))
            {
                const double phi = packet.relative_phase_volume_flux[1];
                const LO owner = mesh->owner_cell(face);
                const LO donor = phi >= 0 ? owner : mesh->opposite_or_periodic_neighbor_cell(face, owner);
                number_flux.set_owned_value(face, phi * number_concentration.local_value(donor));
            }
        }
        number_flux.sync_ghosts();
        phase0.sync_ghosts();
        phase1.sync_ghosts();
        actual.sync_ghosts();
        double maximum = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &theta_max, &maximum);
        EXPECT_GT(maximum, 1.1);
        detail::CompensatedSum<long double> energy, work, operands;
        double old_number = 0, new_number = 0;
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            const LO cell = static_cast<LO>(c);
            double old_volume = 0, final_volume = 0, advected_volume = 0;
            for (size_t a = 0; a < 2; ++a)
            {
                const auto& before = accepted.cells[c].phases[a];
                const auto& now = trial.state().cells[c].phases[a];
                const double old = volume(before), final = volume(now);
                const double advected = snapshot.cells[c].phases[a].advected_material_volume;
                const double transported = snapshot.time_step * balance(*mesh, a == 0 ? phase0 : phase1, cell);
                EXPECT_NEAR(advected, old - transported,
                    64 * std::numeric_limits<double>::epsilon() *
                        (std::abs(old) + std::abs(advected) + std::abs(transported)));
                old_volume += old;
                final_volume += final;
                advected_volume += advected;
                energy += static_cast<long double>(now.internal_energy) - before.internal_energy;
                work += -static_cast<long double>(now.absolute_pressure) * (final - advected);
                operands += std::abs(static_cast<long double>(now.absolute_pressure) * final) +
                            std::abs(static_cast<long double>(now.absolute_pressure) * advected);
            }
            const double new_cv = snapshot.new_control_volumes[c];
            EXPECT_NEAR(final_volume, new_cv, controls.volume_relative_tolerance * new_cv);
            // Closed moving material balance, with the geometric CV increment separate from physical EOS work.
            EXPECT_NEAR(balance(*mesh, actual, cell),
                (final_volume - advected_volume + old_volume - new_cv) / snapshot.time_step,
                controls.continuity_tolerance);
            const double before_count = *accepted.cells[c].phase_numbers[1];
            const double transported_count = *snapshot.cells[c].phases[1].population_number;
            const double outgoing_count = snapshot.time_step * balance(*mesh, number_flux, cell);
            EXPECT_LE(std::abs(transported_count - before_count + outgoing_count),
                controls.conservation_relative_tolerance *
                    (std::abs(transported_count) + std::abs(before_count) + std::abs(outgoing_count)));
            old_number += before_count;
            new_number += *trial.state().cells[c].phase_numbers[1];
        }
        const double global_work = sum(*mesh, static_cast<double>(work.value()));
        EXPECT_NEAR(global_work, trial.diagnostics().pressure_work,
            64 * std::numeric_limits<double>::epsilon() * sum(*mesh, static_cast<double>(operands.value())));
        EXPECT_LE(std::abs(sum(*mesh, static_cast<double>(energy.value())) - global_work),
            trial.diagnostics().internal_energy_roundoff_bound +
                64 * std::numeric_limits<double>::epsilon() * trial.diagnostics().caloric_energy_scale);
        EXPECT_NEAR(sum(*mesh, new_number), sum(*mesh, old_number), 1e-9);
    }
}

TEST(LowMachALEConservationTest, BoundedPressureAndForeignMotionFailuresPreserveEarlierCandidate)
{
    for (bool semi : {false, true})
    {
        auto mesh = make_mesh(semi, {0, 1});
        auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "ordinary finite gas", .028, {{"G", 1}}, {R::SpeciesPhase::Resolved}}});
        auto gas = R::make_ideal_gas_mixture(
            catalog, {{20, 300, 6000}}, {250, 600, 10000, 2000000}, R::EnergyConvention::SensibleEnergy);
        auto controls = options();
        controls.minimum_pressure = 99900;
        controls.maximum_pressure = 100100;
        Flow flow(mesh, {gas}, controls);
        Flow::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(mesh->num_owned_cells());
        const double n = 100000 / (gas_constant * 300);
        for (auto& cell : initial.cells)
            cell.phases = {{gas, {n}, 100000, n * 6000, 300}};
        flow.initialize(std::move(initial));
        Flow::PlanarMotion motion(mesh);
        const auto accepted = flow.state();
        const auto checkpoint = flow.checkpoint(motion);
        const auto retained = flow.preview_ale(.1, motion, 1.00001);
        const auto before_failure_epoch = mesh_geometry_epoch(*mesh);
        // The independent BE root is about 98604 Pa, outside the declared bracket.
        EXPECT_LT(100000 / (1.01 + gas_constant / 20 * .01), controls.minimum_pressure);
        EXPECT_ANY_THROW((void) flow.preview_ale(.1, motion, 1.01));
        EXPECT_GT(mesh_geometry_epoch(*mesh), before_failure_epoch); // Trial geometry was actually entered.
        EXPECT_FALSE(motion.has_active_trial());
        EXPECT_EQ(flow.state().revision, accepted.revision);
        EXPECT_DOUBLE_EQ(flow.state().time, accepted.time);
        EXPECT_DOUBLE_EQ(flow.state().thermodynamic_pressure, accepted.thermodynamic_pressure);
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<LO>(c)), 1.);
            EXPECT_EQ(flow.state().cells[c].phases[0].moles, accepted.cells[c].phases[0].moles);
            EXPECT_DOUBLE_EQ(
                flow.state().cells[c].phases[0].internal_energy, accepted.cells[c].phases[0].internal_energy);
            EXPECT_DOUBLE_EQ(flow.state().cells[c].phases[0].temperature, accepted.cells[c].phases[0].temperature);
        }
        auto foreign_mesh = make_mesh(semi, {0, 1});
        Flow::PlanarMotion foreign_motion(foreign_mesh);
        EXPECT_ANY_THROW((void) flow.preview_ale(.1, foreign_motion, 1.00001));
        EXPECT_ANY_THROW(flow.commit_ale(retained, foreign_motion));
        EXPECT_ANY_THROW(flow.restore(checkpoint, foreign_motion));
        EXPECT_FALSE(foreign_motion.has_active_trial());
        EXPECT_FALSE(motion.has_active_trial());
        EXPECT_NO_THROW(flow.commit_ale(retained, motion));
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<LO>(c)), 1.00001);
            EXPECT_DOUBLE_EQ(foreign_mesh->cell_volume(static_cast<LO>(c)), 1.);
        }
        EXPECT_EQ(sum(*mesh, mesh->num_owned_cells() == 0 ? 1. : 0.), mesh->owned_cell_map()->getComm()->getSize() - 1);
    }
}
} // namespace
