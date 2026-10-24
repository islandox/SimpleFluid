/** Controller-owned prescribed ALE sources, face receipts and replay. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using Controller = R::ReactingMixtureCoupler;
using LO = Flow::Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Counts
{
    int target = 0, source = 0, face = 0, number = 0;
    bool fail_source = false, divergent_target = false, finite_divergent_target = false;
    bool operator==(const Counts&) const = default;
};
struct Fixture
{
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<Flow> flow;
    std::shared_ptr<Counts> calls = std::make_shared<Counts>();
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
            std::vector<R::SpeciesDefinition>{{11, "finite reactant", .018, {{"L", 1}}, {R::SpeciesPhase::Liquid}},
                {17, "finite product", .018, {{"L", 1}}, {R::SpeciesPhase::Liquid}},
                {83, "ordinary gas", .028, {{"G", 1}}, {R::SpeciesPhase::Dispersed}}});
        auto liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000}, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 1e-8, 0, 0, 0}; },
            R::EnergyConvention::SensibleEnergy);
        auto gas = R::make_ideal_gas_mixture(catalog, {{20, 300, 0}, {20, 300, 0}, {20, 300, 0}},
            {250, 600, 10000, 2000000}, R::EnergyConvention::SensibleEnergy);
        mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{{41, "finite isomer reaction",
                {{11, R::SpeciesPhase::Liquid, -1}, {17, R::SpeciesPhase::Liquid, 1}}, 1.}},
            R::EnergyConvention::SensibleEnergy);
        Flow::Options controls;
        controls.minimum_pressure = 20000;
        controls.maximum_pressure = 1000000;
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solution"}, {Flow::PhaseRole::DispersedGas, "micro"}};
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas}, controls);
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const double v = mesh->cell_volume(static_cast<LO>(c));
            state.cells[c].phases = {{liquid, {900 * v / .018, 0, 0}, 100000, 0, 300},
                {gas, {0, 0, .1 * v * 100000 / (8.31446261815324 * 300)}, 100000, 0, 300}};
            state.cells[c].phase_numbers = {std::nullopt, 1000 * v};
        }
        flow->initialize(std::move(state));
        EXPECT_EQ(
            sum(mesh->num_owned_cells() == 0 ? 1. : 0.), std::max(0, mesh->owned_cell_map()->getComm()->getSize() - 2));
    }
    R::ReactingMixtureOptions options() const
    {
        R::ReactingMixtureOptions value;
        value.mechanism = mechanism;
        value.phases = {{R::SpeciesPhase::Liquid}, {R::SpeciesPhase::Dispersed, "micro"}};
        value.source_identity = "finite ALE chemistry";
        value.source = [count = calls, grid = mesh](const R::CoupledReactionContext& context, double dt)
        {
            ++count->source;
            if (count->fail_source && grid->owned_cell_map()->getComm()->getRank() == 0)
                throw std::runtime_error("Manufactured source rejection after actual motion.");
            EXPECT_GT(context.reaction.environment.bulkVolume, .5);
            R::ReactionSourceInput result;
            result.integratedExtents = {{41, 1e-4 * dt}};
            result.heat.rate =
                R::VolumetricHeatRate{grid->cell_centroid(static_cast<LO>(context.owned_cell)).z < .5 ? 100. : 200.,
                    R::SourceVolumeBasis::BulkVolume};
            return result;
        };
        value.population_events = std::make_shared<const R::MixturePopulationEventProvider>(
            R::MixturePopulationEventProvider{"ALE explicit number", 1, catalog->fingerprint(),
                [count = calls](const R::MixturePopulationEventContext& context)
                {
                    ++count->number;
                    return std::vector<BubbleGasPhysics::PopulationNumberEvent>{{"micro", context.time_step, 0}};
                }});
        R::MixtureFaceTransportOptions face;
        face.identity = "ALE explicit liquid thermal contact";
        face.schema_fingerprint = catalog->fingerprint();
        face.fourier = R::MixtureFourierLaw{"ALE finite heat exchange", 1, catalog->fingerprint(),
            [count = calls](const R::MixtureFaceContext& context)
            {
                ++count->face;
                EXPECT_GT(context.center[2], .5); // Law sees actual trial geometry, not restored coordinates.
                EXPECT_GT(context.owner.bulk_volume, .5);
                EXPECT_GT(context.neighbor.bulk_volume, .5);
                return 5.;
            }};
        value.face_transport = {face};
        value.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(R::MixturePlanarMotionProvider{
            mesh, {}, "explicit common planar target", 1, [count = calls, grid = mesh](double time, double dt)
            {
                ++count->target;
                const auto comm = grid->owned_cell_map()->getComm();
                if (count->divergent_target && comm->getRank() == comm->getSize() - 1)
                    return std::numeric_limits<double>::quiet_NaN();
                const double target = 1. + .01 * (time + dt);
                return target + (count->finite_divergent_target && comm->getRank() == comm->getSize() - 1 ? 1e-5 : 0.);
            }});
        return value;
    }
    double sum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    }
    std::vector<double> face_flux() const
    {
        std::vector<double> result;
        for (const auto face : flow->volume_flux().owned_face_ids())
            result.push_back(flow->volume_flux().value(face));
        return result;
    }
    void same_state(const Flow::State& expected) const
    {
        const auto& actual = flow->state();
        int local_invalid = actual.cells.size() != expected.cells.size(), invalid = 0;
        for (size_t c = 0; c < std::min(actual.cells.size(), expected.cells.size()); ++c)
            local_invalid = local_invalid || actual.cells[c].phases.size() != expected.cells[c].phases.size() ||
                            actual.cells[c].drift_velocity.size() != expected.cells[c].drift_velocity.size();
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &invalid);
        ASSERT_EQ(invalid, 0); // Every rank returns together before later fixture collectives.
        EXPECT_EQ(actual.revision, expected.revision);
        EXPECT_DOUBLE_EQ(actual.time, expected.time);
        EXPECT_DOUBLE_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
        for (size_t c = 0; c < actual.cells.size(); ++c)
        {
            EXPECT_EQ(actual.cells[c].phase_numbers, expected.cells[c].phase_numbers);
            for (size_t axis = 0; axis < 3; ++axis)
                EXPECT_EQ(actual.cells[c].velocity.component(axis), expected.cells[c].velocity.component(axis));
            for (size_t a = 0; a < actual.cells[c].drift_velocity.size(); ++a)
                for (size_t axis = 0; axis < 3; ++axis)
                    EXPECT_EQ(actual.cells[c].drift_velocity[a].component(axis),
                        expected.cells[c].drift_velocity[a].component(axis));
            EXPECT_DOUBLE_EQ(actual.cells[c].mechanical_pressure, expected.cells[c].mechanical_pressure);
            for (size_t a = 0; a < actual.cells[c].phases.size(); ++a)
            {
                EXPECT_EQ(
                    actual.cells[c].phases[a].thermodynamics.get(), expected.cells[c].phases[a].thermodynamics.get());
                EXPECT_EQ(actual.cells[c].phases[a].moles, expected.cells[c].phases[a].moles);
                EXPECT_DOUBLE_EQ(
                    actual.cells[c].phases[a].internal_energy, expected.cells[c].phases[a].internal_energy);
                EXPECT_DOUBLE_EQ(actual.cells[c].phases[a].temperature, expected.cells[c].phases[a].temperature);
                EXPECT_DOUBLE_EQ(
                    actual.cells[c].phases[a].absolute_pressure, expected.cells[c].phases[a].absolute_pressure);
            }
        }
    }
};
TEST(ReactingMixtureALETest, ActiveFaceAndNumberReceiptsCommitAndCheckpointReplayWithoutCallbacks)
{
    for (bool semi : {false, true})
    {
        Fixture f(semi);
        Controller controller(f.flow, f.options());
        const auto accepted = f.flow->state();
        const auto accepted_flux = f.face_flux();
        const auto saved = controller.checkpoint();
        const auto trial = controller.preview(.001);
        EXPECT_EQ(f.calls->target, 1);
        f.same_state(accepted);
        EXPECT_EQ(f.face_flux(), accepted_flux);
        for (size_t c = 0; c < accepted.cells.size(); ++c)
        {
            EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), .5);
            EXPECT_EQ(trial.local_population_events()[c].source_stage, trial.transport_snapshot().stage);
            EXPECT_DOUBLE_EQ(trial.local_population_events()[c].changes[1], .001);
            EXPECT_DOUBLE_EQ(trial.local_sources()[c].reactionExtents[0], 1e-7);
        }
        EXPECT_GT(f.sum(f.calls->face), 0);
        const auto counts = *f.calls;
        controller.commit(trial);
        EXPECT_EQ(*f.calls, counts);
        f.same_state(trial.state());
        for (size_t c = 0; c < accepted.cells.size(); ++c)
            EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), .500005);
        const auto result = f.flow->state();
        const auto committed_flux = f.face_flux();
        controller.restore(saved);
        EXPECT_EQ(*f.calls, counts);
        f.same_state(accepted);
        EXPECT_EQ(f.face_flux(), accepted_flux);
        for (size_t c = 0; c < accepted.cells.size(); ++c)
            EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), .5);
        EXPECT_ANY_THROW(controller.commit(trial));
        const auto replay = controller.preview(.001);
        controller.commit(replay);
        f.same_state(result);
        EXPECT_EQ(f.face_flux(), committed_flux);
    }
}
TEST(ReactingMixtureALETest, LaterSourceAndEmptyOwnerTargetFailuresKeepEarlierMovingTrialUsable)
{
    for (bool semi : {false, true})
    {
        Fixture f(semi);
        Controller controller(f.flow, f.options());
        const auto accepted = f.flow->state();
        const auto accepted_flux = f.face_flux();
        const auto valid = controller.preview(.001);
        f.calls->fail_source = true;
        EXPECT_ANY_THROW((void) controller.preview(.002));
        f.same_state(accepted);
        EXPECT_EQ(f.face_flux(), accepted_flux);
        for (size_t c = 0; c < accepted.cells.size(); ++c)
            EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), .5);
        f.calls->fail_source = false;
        f.calls->divergent_target = true;
        EXPECT_ANY_THROW((void) controller.preview(.001));
        f.same_state(accepted);
        EXPECT_EQ(f.face_flux(), accepted_flux);
        const auto counts = *f.calls;
        EXPECT_NO_THROW(controller.commit(valid));
        EXPECT_EQ(*f.calls, counts);
        f.same_state(valid.state());
    }
}
TEST(ReactingMixtureALETest, FiniteDivergentTargetOnAnEmptyOwnerRejectsWithoutConsumingEarlierTrial)
{
    for (bool semi : {false, true})
    {
        Fixture f(semi);
        if (f.mesh->owned_cell_map()->getComm()->getSize() == 1)
            GTEST_SKIP() << "Finite rank-divergent targets require multiple MPI ranks.";
        Controller controller(f.flow, f.options());
        const auto accepted = f.flow->state();
        const auto accepted_flux = f.face_flux();
        const auto valid = controller.preview(.001);
        f.calls->finite_divergent_target = true;
        EXPECT_ANY_THROW((void) controller.preview(.001));
        f.same_state(accepted);
        EXPECT_EQ(f.face_flux(), accepted_flux);
        for (size_t c = 0; c < accepted.cells.size(); ++c)
            EXPECT_DOUBLE_EQ(f.mesh->cell_volume(static_cast<LO>(c)), .5);
        const auto counts = *f.calls;
        EXPECT_NO_THROW(controller.commit(valid));
        EXPECT_EQ(*f.calls, counts);
        f.same_state(valid.state());
    }
}
} // namespace
