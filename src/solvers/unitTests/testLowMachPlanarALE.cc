/** Native prescribed planar motion: conservative transport and accepted-state transactions. */
#include "FVM/CellOperators.hh"
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Fixture
{
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<int> eos_calls = std::make_shared<int>(0), source_calls = std::make_shared<int>(0);
    std::shared_ptr<Flow> flow;
    explicit Fixture(bool semi)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        if (semi)
            mesh = std::make_shared<Flow::Mesh>(
                std::make_shared<Meshes::SemiStructuredXY_Z>(
                    Arr<Meshes::SemiStructuredXY_Z::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}},
                    Arr<Arr<unsigned>>{{0, 1, 2, 3}}, ArrReal{0, .5, 1}),
                distribution);
        else
            mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                    Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
                distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{{11, "finite solvent", .018, {{"L", 1}}, {R::SpeciesPhase::Liquid}},
                {83, "ordinary dispersed gas", .028, {{"G", 1}}, {R::SpeciesPhase::Dispersed}}});
        const auto wrap = [&](std::shared_ptr<const R::MixtureThermodynamics> base)
        {
            return std::make_shared<const R::MixtureThermodynamics>(
                catalog, base->bounds(), base->kind(), base->enthalpy_pressure_dependence(),
                [base, calls = eos_calls](double t, double p, std::span<const double> x)
                {
                    ++*calls;
                    return base->evaluate(t, p, x);
                },
                R::EnergyConvention::SensibleEnergy);
        };
        auto liquid = wrap(R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000}, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 1e-8, 0, 0, 0}; },
            R::EnergyConvention::SensibleEnergy));
        auto gas = wrap(R::make_ideal_gas_mixture(
            catalog, {{20, 300, 0}, {20, 300, 0}}, {250, 600, 10000, 2000000}, R::EnergyConvention::SensibleEnergy));
        Flow::Options options;
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solution"}, {Flow::PhaseRole::DispersedGas, "micro"}};
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas}, options);
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const double v = mesh->cell_volume(static_cast<LO>(c));
            state.cells[c].phases = {{liquid, {900 * v / .018, 0}, 100000, 0, 300},
                {gas, {0, .1 * v * 100000 / (8.31446261815324 * 300)}, 100000, 0, 300}};
            state.cells[c].phase_numbers = {std::nullopt, 1000 * v};
        }
        flow->initialize(std::move(state));
    }
    std::shared_ptr<const Flow::SourceStageProvider> provider(bool fail = false) const
    {
        return std::make_shared<const Flow::SourceStageProvider>(
            Flow::SourceStageProvider{fail ? "manufactured ALE failure" : "manufactured ALE zero source", 1,
                catalog->fingerprint(), [calls = source_calls, fail](const Flow::SourceStageContext& context)
                {
                    ++*calls;
                    collective_detail::collective_local_validation(context.mesh, "ALE source fault fence",
                        [&]
                        {
                            const auto comm = context.mesh.owned_cell_map()->getComm();
                            if (fail && comm->getRank() == comm->getSize() - 1)
                                throw std::runtime_error("Manufactured source failure after actual geometry mutation.");
                        });
                    return Flow::SourceStageResult{std::vector<Flow::CellSources>(context.mesh.num_owned_cells()), {}};
                }});
    }
    double sum(double value) const
    {
        double result = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &value, &result);
        return result;
    }
    std::vector<double> volumes() const
    {
        std::vector<double> result(mesh->num_local_cells());
        for (size_t c = 0; c < result.size(); ++c)
            result[c] = mesh->cell_volume(static_cast<LO>(c));
        return result;
    }
};
void expect_same(const Flow::State& a, const Flow::State& b, std::optional<double> pressure_roundoff = {})
{
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_DOUBLE_EQ(a.time, b.time);
    if (pressure_roundoff)
        EXPECT_NEAR(a.thermodynamic_pressure, b.thermodynamic_pressure, *pressure_roundoff);
    else
        EXPECT_DOUBLE_EQ(a.thermodynamic_pressure, b.thermodynamic_pressure);
    ASSERT_EQ(a.cells.size(), b.cells.size());
    for (size_t c = 0; c < a.cells.size(); ++c)
    {
        const auto& x = a.cells[c];
        const auto& y = b.cells[c];
        EXPECT_DOUBLE_EQ(x.mechanical_pressure, y.mechanical_pressure);
        EXPECT_EQ(x.phase_numbers, y.phase_numbers);
        ASSERT_EQ(x.phases.size(), y.phases.size());
        ASSERT_EQ(x.drift_velocity.size(), y.drift_velocity.size());
        for (size_t k = 0; k < 3; ++k)
            EXPECT_DOUBLE_EQ(x.velocity.component(k), y.velocity.component(k));
        for (size_t p = 0; p < x.phases.size(); ++p)
        {
            EXPECT_EQ(x.phases[p].thermodynamics.get(), y.phases[p].thermodynamics.get());
            EXPECT_EQ(x.phases[p].moles, y.phases[p].moles);
            EXPECT_DOUBLE_EQ(x.phases[p].absolute_pressure, y.phases[p].absolute_pressure);
            EXPECT_DOUBLE_EQ(x.phases[p].internal_energy, y.phases[p].internal_energy);
            EXPECT_DOUBLE_EQ(x.phases[p].temperature, y.phases[p].temperature);
            for (size_t k = 0; k < 3; ++k)
                EXPECT_DOUBLE_EQ(x.drift_velocity[p].component(k), y.drift_velocity[p].component(k));
        }
    }
}
void expect_number_packet(const Fixture& f, const Flow::Trial& trial, const Flow::State& accepted)
{
    const auto& snapshot = trial.transport_snapshot();
    int invalid =
        snapshot.cells.size() != accepted.cells.size() || snapshot.new_control_volumes.size() != accepted.cells.size();
    if (!invalid)
        for (const auto& cell : snapshot.cells)
            invalid = invalid || cell.phases.size() != 2 || !cell.phases[1].population_number.has_value();
    for (const auto& packet : trial.owned_phase_transport())
        invalid = invalid || packet.relative_phase_volume_flux.size() != 2 || packet.phase_volume_flux.size() != 2;
    int global_invalid = 0;
    Teuchos::reduceAll(*f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &invalid, &global_invalid);
    ASSERT_EQ(global_invalid, 0); // Every rank returns together before any field import.
    Flow::Traits::scalar_cell_type concentration(f.mesh, 0., "independent_Nadv_over_trial_CV");
    Flow::Traits::scalar_face_type flux(f.mesh, 0., "independent_relative_number_flux");
    for (size_t c = 0; c < accepted.cells.size(); ++c)
        concentration.set_owned_value(
            static_cast<LO>(c), *snapshot.cells[c].phases[1].population_number / snapshot.new_control_volumes[c]);
    concentration.sync_ghosts();
    const auto packets = trial.owned_phase_transport();
    for (size_t fidx = 0; fidx < packets.size(); ++fidx)
    {
        const LO face = static_cast<LO>(fidx);
        const auto& packet = packets[fidx];
        EXPECT_EQ(packet.relative_phase_volume_flux.size(), 2U);
        EXPECT_DOUBLE_EQ(packet.phase_volume_flux[1], packet.relative_phase_volume_flux[1] + packet.mesh_volume_flux);
        if (f.mesh->is_geometry_exterior_face(face))
        {
            EXPECT_DOUBLE_EQ(packet.relative_phase_volume_flux[1], 0.);
            continue;
        }
        const LO owner = f.mesh->owner_cell(face);
        const double relative = packet.relative_phase_volume_flux[1];
        const LO donor = relative >= 0 ? owner : f.mesh->opposite_or_periodic_neighbor_cell(face, owner);
        // Already a number/CV concentration: no material alpha is multiplied here.
        flux.set_owned_value(face, relative * concentration.local_value(donor));
    }
    flux.sync_ghosts();
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const LO cell = static_cast<LO>(c);
        const double before = *accepted.cells[c].phase_numbers[1];
        const double after = *snapshot.cells[c].phases[1].population_number;
        const double transport = snapshot.time_step * FVM::cell_flux_balance(*f.mesh, flux, cell);
        const double scale = std::abs(before) + std::abs(after) + std::abs(transport);
        EXPECT_LE(std::abs(after - before + transport), f.flow->options().conservation_relative_tolerance * scale);
    }
}
double zero_motion_pressure_roundoff(const Fixture& fixture, const Flow::State& accepted)
{
    // Independent fixture EOS: V_l=m_l/rho0 at reference p, V_g=nRT/p,
    // dV_l/dp=-kappa*V_l and dT/dp=V_g/(m_l*cp_l+n*(cv_g+R)).
    // ALE retains the accepted EOS occupancy residual even when swept volume is
    // zero. Bound only that arithmetic p shift, not by a physical tolerance.
    constexpr double R = 8.31446261815324, epsilon = std::numeric_limits<double>::epsilon();
    std::array<double, 3> local{}, global{};
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const auto& phases = accepted.cells[c].phases;
        const double p = accepted.thermodynamic_pressure, t = phases[1].temperature;
        const double mass = phases[0].moles[0] * .018, n = phases[1].moles[1];
        const double liquid = mass / 1000., gas = n * R * t / p;
        const double cv = fixture.mesh->cell_volume(static_cast<LO>(c));
        const double heat_capacity = mass * 4000. + n * (20. + R);
        local[0] += (liquid + gas) - cv;
        local[1] += std::abs(liquid) + std::abs(gas) + cv;
        local[2] += 1e-8 * liquid + gas / p - (gas / t) * (gas / heat_capacity);
    }
    Teuchos::reduceAll(*fixture.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local.size()),
        local.data(), global.data());
    if (!(global[2] > 0))
        throw std::runtime_error("The zero-motion fixture needs positive analytical pressure susceptibility.");
    return (std::abs(global[0]) + 32 * epsilon * global[1]) / global[2] +
           4 * epsilon * std::abs(accepted.thermodynamic_pressure);
}
TEST(LowMachPlanarALETest, ZeroMotionMatchesFixedAndFailedGeometryPreviewPreservesEarlierFixedTrial)
{
    for (bool semi : {false, true})
    {
        Fixture f(semi);
        Flow::PlanarMotion motion(f.mesh);
        const auto accepted = f.flow->state();
        const auto old_v = f.volumes();
        const auto fixed = f.flow->preview(.001);
        const auto moving = f.flow->preview_ale(.001, motion, 1.);
        expect_same(moving.state(), fixed.state(), zero_motion_pressure_roundoff(f, accepted));
        expect_same(f.flow->state(), accepted);
        EXPECT_EQ(f.volumes(), old_v);
        EXPECT_FALSE(motion.has_active_trial());
        EXPECT_ANY_THROW((void) f.flow->preview_ale_with_sources(.001, motion, 1.00001, f.provider(true)));
        expect_same(f.flow->state(), accepted);
        EXPECT_EQ(f.volumes(), old_v);
        EXPECT_NO_THROW(f.flow->commit(fixed)); // Historical epoch differs, logical accepted geometry is unchanged.
    }
}
TEST(LowMachPlanarALETest, RetainedMovingTrialRestagesExactlyWithoutCallingSourceOrEOS)
{
    for (bool semi : {false, true})
    {
        Fixture f(semi);
        Flow::PlanarMotion motion(f.mesh);
        const auto accepted = f.flow->state();
        const auto old_v = f.volumes();
        const auto trial = f.flow->preview_ale_with_sources(.001, motion, 1.00001, f.provider());
        EXPECT_EQ(f.volumes(), old_v);
        expect_same(f.flow->state(), accepted);
        expect_number_packet(f, trial, accepted);
        const auto& transport = trial.transport_snapshot();
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            EXPECT_DOUBLE_EQ(transport.old_control_volumes[c], old_v[c]);
            EXPECT_GT(transport.new_control_volumes[c], old_v[c]);
            EXPECT_TRUE(transport.cells[c].phases[1].population_number.has_value());
        }
        EXPECT_ANY_THROW(f.flow->commit(trial)); // Explicit live motion authority is mandatory.
        EXPECT_ANY_THROW((void) f.flow->preview_ale_with_sources(.001, motion, 1.00002, f.provider(true)));
        expect_same(f.flow->state(), accepted);
        EXPECT_EQ(f.volumes(), old_v);
        const int eos = *f.eos_calls, source = *f.source_calls;
        EXPECT_NO_THROW(f.flow->commit_ale(trial, motion));
        EXPECT_EQ(*f.eos_calls, eos);
        EXPECT_EQ(*f.source_calls, source);
        expect_same(f.flow->state(), trial.state());
        EXPECT_FALSE(motion.has_active_trial());
        double count = 0;
        for (const auto& cell : f.flow->state().cells)
            count += *cell.phase_numbers[1];
        EXPECT_NEAR(f.sum(count), 1000., 1e-9);
        EXPECT_EQ(f.sum(f.mesh->num_owned_cells() == 0 ? 1. : 0.),
            std::max(0, f.mesh->owned_cell_map()->getComm()->getSize() - 2));
    }
}
TEST(LowMachPlanarALETest, MotionCheckpointRestoresGeometryAllSpeciesNumbersAndEnergyWithoutCallbacks)
{
    for (bool semi : {false, true})
    {
        Fixture f(semi);
        Flow::PlanarMotion motion(f.mesh);
        const auto accepted = f.flow->state();
        const auto old_v = f.volumes();
        const auto saved = f.flow->checkpoint(motion);
        const auto trial = f.flow->preview_ale_with_sources(.001, motion, 1.00001, f.provider());
        f.flow->commit_ale(trial, motion);
        const int eos = *f.eos_calls, source = *f.source_calls;
        f.flow->restore(saved, motion);
        EXPECT_EQ(*f.eos_calls, eos);
        EXPECT_EQ(*f.source_calls, source);
        expect_same(f.flow->state(), accepted);
        EXPECT_EQ(f.volumes(), old_v);
        EXPECT_ANY_THROW(f.flow->commit_ale(trial, motion));
    }
}
} // namespace
