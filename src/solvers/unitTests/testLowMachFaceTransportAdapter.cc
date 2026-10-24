/** @file testLowMachFaceTransportAdapter.cc
 * Focused canonical-face conservation, receipt and collective failure contracts.
 */
#include "equations/CollectiveValidation.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/details/LowMachFaceSourceComposition.hh"
#include "solvers/details/LowMachFaceTransportAdapter.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
namespace F = R::low_mach_faces;
using Flow = F::Flow;
using LO = F::LO;
constexpr R::SpeciesId solvent = 9, one = 14, two = 29;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Fixture
{
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid;
    Flow::TransportSnapshot transport;
    Flow::State thermal;
    std::vector<R::CellSpeciesState> accepted, post_chemistry;
    std::vector<std::vector<std::array<long double, 2>>> post_source_parts;
    std::vector<R::SpeciesSourceLedger> chemistry_ledgers;
    R::SpeciesId finite_counter;
    explicit Fixture(std::size_t count = 2, R::SpeciesId counter_id = solvent) : finite_counter(counter_id)
    {
        S::ArrReal z;
        for (std::size_t c = 0; c <= count; ++c)
            z.push_back(double(c) / count);
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<S::Meshes::OrthogonalCartesian3D>(
                                                S::Vec3D<S::ArrReal>{S::ArrReal{0, 1}, S::ArrReal{0, 1}, z}),
            distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{
                {finite_counter, "explicit finite counter", .010, {{"X", 1}}, {R::SpeciesPhase::Liquid}},
                {one, "dissolved first molecule", .020, {{"X", 2}}, {R::SpeciesPhase::Dissolved}},
                {two, "dissolved second molecule", .040, {{"X", 4}}, {R::SpeciesPhase::Dissolved}}},
            7);
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000}, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4200, 0, 1e-9, 0, 0, 0}; },
            R::EnergyConvention::SensibleEnergy);
        transport.accepted_revision = thermal.revision = 3;
        transport.accepted_generation = 7;
        transport.schema_fingerprint = catalog->fingerprint();
        transport.old_geometry_epoch = transport.new_geometry_epoch = mesh->geometry_epoch();
        transport.time_step = .125;
        transport.stage = {"manufactured-root-source-stage", 1, 19, 0};
        thermal.thermodynamic_pressure = 100000;
        const R::ReactionMechanism chemistry(
            catalog, {{82, "finite first molecule becomes half a second molecule",
                         {{one, R::SpeciesPhase::Dissolved, -1, {}}, {two, R::SpeciesPhase::Dissolved, .5, {}}}, 0.}});
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const double v = mesh->cell_volume(cell);
            const auto gid = mesh->cell_global_id(cell);
            const double a = count == 2 ? (gid == 0 ? 2. : 0.) : (gid == 1 ? 1. : 0.);
            const double b = count == 2 ? (gid == 0 ? 0. : 1.) : 0.;
            R::CellSpeciesState state(*catalog);
            state.set_amount(*catalog, one, R::SpeciesPhase::Dissolved, a * v);
            state.set_amount(*catalog, two, R::SpeciesPhase::Dissolved, b * v);
            state.set_amount(
                *catalog, finite_counter, R::SpeciesPhase::Liquid, (1000 * v - .020 * a * v - .040 * b * v) / .010);
            accepted.push_back(state);
            R::ReactionSourceInput input;
            if (count == 2 && gid == 0)
                input.integratedExtents = {{82, .25 * v}};
            const auto produced =
                R::advance_reactions(chemistry, state, {0, 300, 100000, v, v}, transport.time_step, input);
            post_chemistry.push_back(produced.state); // Actual pure chemistry is the finite donor base.
            post_source_parts.push_back(produced.localReceipt.postSourceAmounts);
            chemistry_ledgers.push_back(produced.ledger);
            const double temperature = count == 2 && gid == 1 ? 400. : 300.;
            std::vector<double> moles{state.amount(*catalog, finite_counter, R::SpeciesPhase::Liquid),
                state.amount(*catalog, one, R::SpeciesPhase::Dissolved),
                state.amount(*catalog, two, R::SpeciesPhase::Dissolved)};
            std::vector<double> x(moles);
            double total = 0;
            for (const auto n : x)
                total += n;
            for (auto& fraction : x)
                fraction /= total;
            const double u = 1000 * v * liquid->evaluate(temperature, 100000, x).internal_energy;
            Flow::CellState candidate;
            candidate.phases = {{liquid, moles, 100000, u, temperature}};
            candidate.velocity = Flow::Vector{gid == 0 ? 3. : -2., 0., 0.};
            thermal.cells.push_back(candidate);
            // Deliberately unrelated numerical work volume: it must never be used as a concentration denominator.
            transport.cells.push_back({{{moles, u, 99. * v, std::nullopt}}});
            transport.old_control_volumes.push_back(v);
            transport.new_control_volumes.push_back(v);
        }
    }
    Flow::SourceStageContext context() const { return {transport, thermal, *mesh}; }
    F::PhaseBindings bindings() const
    {
        return {{{R::SpeciesPhase::Liquid, {}}, {R::SpeciesPhase::Dissolved, {}}, {R::SpeciesPhase::Dissolved, {}}}};
    }
    F::Options options(bool heat = true) const
    {
        F::Options result;
        result.identity = "explicit same-liquid face contacts";
        result.schema_fingerprint = catalog->fingerprint();
        F::DiffusionLaw law;
        law.binding.identity = "two finite solutes with supplied solvent counter";
        law.binding.schemaFingerprint = catalog->fingerprint();
        law.binding.counterSpecies = finite_counter;
        law.binding.energyConvention = R::EnergyConvention::SensibleEnergy;
        law.primary_species = {one, two};
        law.molar_conductance = [](const F::FaceContext&) { return std::vector<double>{0., .125, .125}; };
        law.partial_molar_enthalpy = [](R::SpeciesId id, const auto&, const F::SideSample& side)
        {
            const double molar_mass = id == one ? .020 : id == two ? .040 : .010;
            return molar_mass * (100. + 4200. * (side.temperature - 300.));
        };
        result.diffusion = std::move(law);
        if (heat)
            result.fourier = F::FourierLaw{"explicit liquid contact, G=2 W/K", 1, catalog->fingerprint(),
                [](const F::FaceContext&) { return 2.; }};
        return result;
    }
    double sum(double local) const
    {
        double global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    }
};

TEST(LowMachFaceTransportTest, ActualPostChemistryEOSDonorsGiveAnalyticTwoSoluteAndHeatPackets)
{
    Fixture f;
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, f.options(), f.bindings());
    const auto accepted = f.accepted, donor = f.post_chemistry;
    const auto receipt = adapter.preview(f.context(), donor);
    double event_owners = 0, energy = 0;
    for (const auto& event : receipt.diffusion.faces)
    {
        event_owners += event.owner.has_value();
        ASSERT_EQ(event.moles.size(), 3U);
        EXPECT_NEAR(event.moles[1], .02734375, 2e-15); // .125 s*.125 m3/s*(1.75-0) mol/m3.
        EXPECT_NEAR(event.moles[2], -.013671875, 2e-15);
        EXPECT_NEAR(event.counterMoles, 0., 2e-14);
        EXPECT_NEAR(event.enthalpy, -229.6875, 1e-10); // Supplied EOS-consistent donor h: 2 and 16804 J/mol.
        EXPECT_EQ(event.momentum, (std::array<double, 3>{}));
    }
    EXPECT_DOUBLE_EQ(f.sum(event_owners), 1.); // No duplicated global face event on the incident rank.
    for (std::size_t c = 0; c < donor.size(); ++c)
    {
        const bool lower = f.mesh->cell_global_id(static_cast<LO>(c)) == 0;
        const auto& source = receipt.rounded.sources[c];
        EXPECT_NEAR(source.species_moles[0][1], lower ? -.02734375 : .02734375, 2e-15);
        EXPECT_NEAR(source.species_moles[0][2], lower ? .013671875 : -.013671875, 2e-15);
        EXPECT_NEAR(source.species_moles[0][0], 0., 2e-14);
        EXPECT_NEAR(source.internal_energy[0], lower ? 254.6875 : -254.6875, 1e-10);
        for (std::size_t a = 0; a < 3; ++a)
            EXPECT_DOUBLE_EQ(source.momentum.component(a), 0.);
        energy += source.internal_energy[0];
        // One physical phase has constant mass and cp; independent local recovery oracle.
        auto phase = f.thermal.cells[c].phases[0];
        phase.moles = {donor[c].amount(*f.catalog, solvent, R::SpeciesPhase::Liquid),
            donor[c].amount(*f.catalog, one, R::SpeciesPhase::Dissolved),
            donor[c].amount(*f.catalog, two, R::SpeciesPhase::Dissolved)};
        for (std::size_t s = 0; s < phase.moles.size(); ++s)
            phase.moles[s] += source.species_moles[0][s];
        phase.internal_energy += source.internal_energy[0];
        const auto recovered = R::recover_phase_temperatures(std::span{&phase, 1}, phase.internal_energy);
        const double mass = 1000. * f.mesh->cell_volume(static_cast<LO>(c));
        EXPECT_NEAR(recovered.phases[0].temperature,
            f.thermal.cells[c].phases[0].temperature + source.internal_energy[0] / (mass * 4200.), 2e-8);
    }
    EXPECT_DOUBLE_EQ(f.sum(energy), 0.);
    EXPECT_EQ(f.accepted.size(), accepted.size());
    for (std::size_t c = 0; c < donor.size(); ++c)
    {
        EXPECT_EQ(f.accepted[c].amounts, accepted[c].amounts);
        EXPECT_EQ(f.post_chemistry[c].amounts, donor[c].amounts);
    }
    EXPECT_EQ(f.sum(f.mesh->num_owned_cells() == 0 ? 1. : 0.),
        std::max(0, f.mesh->owned_cell_map()->getComm()->getSize() - 2));
}

TEST(LowMachFaceTransportTest, AllIncidentFaceDemandsReserveOneFiniteDonorBeforeAnyCredit)
{
    Fixture f(3);
    auto options = f.options(false);
    options.diffusion->molar_conductance = [](const auto&) { return std::vector<double>{0., 1.6, 0.}; };
    // Center owns 1/3 mol; either demand .2 mol is admissible, both together are not.
    F::LowMachFaceTransportAdapter both(f.mesh, f.catalog, options, f.bindings());
    EXPECT_ANY_THROW((void) both.preview(f.context(), f.post_chemistry));
    options.identity = "one selected physical face";
    options.diffusion->molar_conductance = [](const F::FaceContext& face)
    { return std::vector<double>{0., face.owner.concentrations[1] < face.neighbor.concentrations[1] ? 1.6 : 0., 0.}; };
    F::LowMachFaceTransportAdapter single(f.mesh, f.catalog, options, f.bindings());
    EXPECT_NO_THROW((void) single.preview(f.context(), f.post_chemistry));
    for (std::size_t c = 0; c < f.post_chemistry.size(); ++c)
        EXPECT_EQ(f.post_chemistry[c].amounts, f.accepted[c].amounts);
}

TEST(LowMachFaceTransportTest, FiniteCounterCannotBorrowAnIncomingSoluteCredit)
{
    Fixture f;
    for (std::size_t c = 0; c < f.post_chemistry.size(); ++c)
        if (f.mesh->cell_global_id(static_cast<LO>(c)) == 1)
            f.post_chemistry[c].set_amount(*f.catalog, solvent, R::SpeciesPhase::Liquid, 0.);
    auto options = f.options(false);
    options.diffusion->primary_species = {one};
    options.diffusion->molar_conductance = [](const auto&) { return std::vector<double>{0., .125, 0.}; };
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, options, f.bindings());
    EXPECT_ANY_THROW((void) adapter.preview(f.context(), f.post_chemistry));
    for (std::size_t c = 0; c < f.post_chemistry.size(); ++c)
        if (f.mesh->cell_global_id(static_cast<LO>(c)) == 1)
            EXPECT_DOUBLE_EQ(f.post_chemistry[c].amount(*f.catalog, solvent, R::SpeciesPhase::Liquid), 0.);
}

TEST(LowMachFaceTransportTest, OwnerOnlyCoefficientFailureFencesBeforeFaceImportAndValidReplayIsPure)
{
    Fixture f;
    auto invalid = f.options();
    invalid.diffusion->molar_conductance = [](const auto&)
    { return std::vector<double>{0., std::numeric_limits<double>::quiet_NaN(), 0.}; };
    F::LowMachFaceTransportAdapter broken(f.mesh, f.catalog, invalid, f.bindings());
    EXPECT_ANY_THROW((void) broken.preview(f.context(), f.post_chemistry));
    F::LowMachFaceTransportAdapter valid(f.mesh, f.catalog, f.options(), f.bindings());
    const auto first = valid.preview(f.context(), f.post_chemistry);
    const auto second = valid.preview(f.context(), f.post_chemistry);
    EXPECT_EQ(first.stamp, second.stamp);
    for (std::size_t c = 0; c < first.rounded.sources.size(); ++c)
    {
        EXPECT_EQ(first.rounded.sources[c].species_moles, second.rounded.sources[c].species_moles);
        EXPECT_EQ(first.rounded.sources[c].internal_energy, second.rounded.sources[c].internal_energy);
        EXPECT_EQ(first.rounded.gross_events[c], second.rounded.gross_events[c]);
    }
    f.transport.stage.invocation++;
    EXPECT_ANY_THROW(valid.validate_receipt(first, f.context()));
}

TEST(LowMachFaceTransportTest, AsymmetricImmutableMetadataRejectsEvenWhenOnlyAnEmptyRankChangesIt)
{
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires actual MPI configuration disagreement.";
    auto invalid = f.options();
    if (comm->getRank() == comm->getSize() - 1)
        invalid.revision++;
    EXPECT_ANY_THROW((void) F::LowMachFaceTransportAdapter(f.mesh, f.catalog, invalid, f.bindings()));
    EXPECT_NO_THROW((void) F::LowMachFaceTransportAdapter(f.mesh, f.catalog, f.options(), f.bindings()));
}

TEST(LowMachFaceTransportTest, CandidateALEBulkVolumeNeverReplacesEOSVolumeAndStaleReceiptsReject)
{
    Fixture f;
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, f.options(), f.bindings());
    const auto fixed = adapter.preview(f.context(), f.post_chemistry);
    S::PlanarALEMeshMotion<Flow::Pack> motion(f.mesh);
    motion.begin_trial(1.2, f.transport.time_step);
    f.transport.new_geometry_epoch = f.mesh->geometry_epoch();
    for (std::size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
        f.transport.new_control_volumes[c] = f.mesh->cell_volume(static_cast<LO>(c));
    const auto moved = adapter.preview(f.context(), f.post_chemistry);
    for (std::size_t c = 0; c < moved.rounded.sources.size(); ++c)
    {
        EXPECT_EQ(moved.rounded.sources[c].species_moles, fixed.rounded.sources[c].species_moles);
        EXPECT_EQ(moved.rounded.sources[c].internal_energy, fixed.rounded.sources[c].internal_energy);
    }
    motion.rollback_trial();
    EXPECT_ANY_THROW(adapter.validate_receipt(moved, f.context()));
    f.transport.old_geometry_epoch = f.transport.new_geometry_epoch = f.mesh->geometry_epoch();
    for (std::size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
        f.transport.new_control_volumes[c] = f.mesh->cell_volume(static_cast<LO>(c));
    const auto replay = adapter.preview(f.context(), f.post_chemistry);
    for (std::size_t c = 0; c < replay.rounded.sources.size(); ++c)
        EXPECT_EQ(replay.rounded.sources[c].internal_energy, fixed.rounded.sources[c].internal_energy);
}

TEST(LowMachFaceTransportTest, OwningThermalRecoveryRejectsUnreachableFourierEnergyWithoutPublishing)
{
    Fixture f;
    auto controls = f.options();
    controls.diffusion.reset();
    controls.fourier->conductance = [](const auto&) { return 1e14; };
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, controls, f.bindings());
    const auto receipt = adapter.preview(f.context(), f.post_chemistry);
    // A finite face packet is not acceptance. The owning solver still enforces caloric/temperature bounds.
    EXPECT_ANY_THROW(S::collective_detail::collective_local_validation(*f.mesh, "Manufactured phase energy gate",
        [&]
        {
            for (std::size_t c = 0; c < f.thermal.cells.size(); ++c)
            {
                auto phase = f.thermal.cells[c].phases[0];
                phase.internal_energy += receipt.rounded.sources[c].internal_energy[0];
                (void) R::recover_phase_temperatures(std::span{&phase, 1}, phase.internal_energy);
            }
        }));
    for (std::size_t c = 0; c < f.thermal.cells.size(); ++c)
        EXPECT_DOUBLE_EQ(
            f.thermal.cells[c].phases[0].temperature, f.mesh->cell_global_id(static_cast<LO>(c)) == 1 ? 400. : 300.);
}

TEST(LowMachFaceTransportTest, RawChemistryAndFourierChannelsComposeBeforeRoundingAndRecoverTwoJoules)
{
    Fixture f;
    auto controls = f.options();
    controls.diffusion.reset();
    controls.fourier->conductance = [](const auto&) { return 8e16; }; // Q=-1e18 J owner->neighbor.
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, controls, f.bindings());
    const auto result = F::compose_post_chemistry_faces(f.context(), *f.catalog, adapter,
        [&](const Flow::SourceStageContext& context, F::LocalChemistryStage& stage)
        {
            stage.sums.reset(context.mesh.num_owned_cells(), 1, 3);
            stage.caloric_guesses = context.thermodynamic_trial.cells;
            stage.gross_events.resize(context.mesh.num_owned_cells(), std::vector<double>(6));
            const R::ReactionMechanism mechanism(f.catalog,
                {{82, "actual finite chemistry before heat contact",
                    {{one, R::SpeciesPhase::Dissolved, -1, {}}, {two, R::SpeciesPhase::Dissolved, .5, {}}}, 0.}});
            for (std::size_t c = 0; c < context.mesh.num_owned_cells(); ++c)
            {
                const auto cell = static_cast<LO>(c);
                const bool lower = context.mesh.cell_global_id(cell) == 0;
                const double bulk = context.transport.new_control_volumes[c];
                R::CellSpeciesState initial(*f.catalog);
                const auto& transported = context.transport.cells[c].phases[0].moles;
                initial.set_amount(*f.catalog, solvent, R::SpeciesPhase::Liquid, transported[0]);
                initial.set_amount(*f.catalog, one, R::SpeciesPhase::Dissolved, transported[1]);
                initial.set_amount(*f.catalog, two, R::SpeciesPhase::Dissolved, transported[2]);
                std::vector<double> x(transported);
                long double total = 0, mass = 0;
                for (std::size_t s = 0; s < x.size(); ++s)
                {
                    total += x[s];
                    mass += static_cast<long double>(x[s]) * f.catalog->species()[s].molarMass;
                }
                for (auto& value : x)
                    value = double(value / total);
                const auto& guess = context.thermodynamic_trial.cells[c].phases[0];
                const double phase_volume =
                    double(mass / f.liquid->evaluate(guess.temperature, guess.absolute_pressure, x).density);
                R::ReactionSourceInput input;
                if (lower)
                    input.integratedExtents = {{82, .25 * bulk}};
                input.heat.integratedEnergy = 1.;
                input.externalTransfers = {{"opposed-external-reservoir", solvent, R::SpeciesPhase::Liquid, 0., 0.,
                    lower ? -1e18 : 1e18, {}, {}}};
                const R::ReactionEnvironment environment{context.transport.accepted_time, guess.temperature,
                    guess.absolute_pressure, phase_volume, bulk, R::ReactionVolumeClosure::ThermodynamicPredictor};
                const auto chemistry =
                    R::advance_reactions(mechanism, initial, environment, context.transport.time_step, input);
                stage.post_chemistry.push_back(chemistry.state);
                stage.post_chemistry_parts.push_back(chemistry.localReceipt.postSourceAmounts);
                stage.ledgers.push_back(chemistry.ledger);
                for (std::size_t s = 0; s < 3; ++s)
                {
                    const auto phase = s == 0 ? R::SpeciesPhase::Liquid : R::SpeciesPhase::Dissolved;
                    stage.sums.add_material(
                        c, 0, s, chemistry.ledger.chemicalChange.amount(*f.catalog, f.catalog->species()[s].id, phase));
                }
                // Independent channels stay in ONE accumulator through the later face import/stage.
                stage.sums.add_energy(c, 0, chemistry.ledger.externalHeat);
                stage.sums.add_energy(c, 0, chemistry.ledger.reactionHeat);
                for (const auto& transfer : chemistry.ledger.externalTransfers)
                    stage.sums.add_energy(c, 0, transfer.energy);
                stage.gross_events[c] = {lower ? .25 * bulk : 0., 0., lower ? 0. : 1e18, lower ? 1e18 : 0., 1., 0.};
            }
        });
    double energy = 0;
    for (std::size_t c = 0; c < result.result.sources.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(result.result.sources[c].internal_energy[0], 1.);
        EXPECT_EQ(result.result.gross_events[c].size(), 18U);
        EXPECT_TRUE(std::ranges::any_of(result.result.gross_events[c], [](double amount) { return amount == 1e18; }));
        energy += result.result.sources[c].internal_energy[0];
        auto phase = f.thermal.cells[c].phases[0];
        for (std::size_t s = 0; s < phase.moles.size(); ++s)
            phase.moles[s] += result.result.sources[c].species_moles[0][s];
        phase.internal_energy += result.result.sources[c].internal_energy[0];
        const auto recovered = R::recover_phase_temperatures(std::span{&phase, 1}, phase.internal_energy);
        EXPECT_NEAR(recovered.phases[0].temperature,
            f.thermal.cells[c].phases[0].temperature + 1. / (1000. * f.mesh->cell_volume(static_cast<LO>(c)) * 4200.),
            2e-8);
    }
    EXPECT_DOUBLE_EQ(f.sum(energy), 2.);
    ASSERT_EQ(result.faces.size(), 1U);
    EXPECT_EQ(result.faces.front()->stamp.provider, f.transport.stage);
}

TEST(LowMachFaceTransportTest, IntegratedFaceFluxScalesWithTimeExactlyOnce)
{
    Fixture f;
    auto controls = f.options(false);
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, controls, f.bindings());
    const auto first = adapter.preview(f.context(), f.post_chemistry);
    f.transport.time_step *= .5;
    f.transport.stage.invocation++;
    const auto half = adapter.preview(f.context(), f.post_chemistry);
    for (std::size_t c = 0; c < first.rounded.sources.size(); ++c)
    {
        for (std::size_t s = 0; s < 3; ++s)
            EXPECT_NEAR(
                half.rounded.sources[c].species_moles[0][s], .5 * first.rounded.sources[c].species_moles[0][s], 2e-15);
        EXPECT_NEAR(
            half.rounded.sources[c].internal_energy[0], .5 * first.rounded.sources[c].internal_energy[0], 1e-10);
    }
    EXPECT_ANY_THROW(adapter.validate_receipt(first, f.context()));
}

TEST(LowMachFaceTransportTest, OrderedStagesConsumeActualPriorStateInsteadOfReservingOneSharedBase)
{
    Fixture f;
    auto controls = f.options(false);
    F::LowMachFaceTransportAdapter first(f.mesh, f.catalog, controls, f.bindings());
    controls.identity = "second ordered same-carrier diffusion stage";
    F::LowMachFaceTransportAdapter second(f.mesh, f.catalog, controls, f.bindings());
    const std::array<const F::LowMachFaceTransportAdapter*, 2> stages{&first, &second};
    const auto result = F::compose_post_chemistry_faces(f.context(), *f.catalog, stages,
        [&](const Flow::SourceStageContext& context, F::LocalChemistryStage& prepared)
        {
            prepared.post_chemistry = f.post_chemistry;
            prepared.post_chemistry_parts = f.post_source_parts;
            prepared.ledgers = f.chemistry_ledgers;
            prepared.caloric_guesses = context.thermodynamic_trial.cells;
            prepared.gross_events.resize(context.mesh.num_owned_cells(), std::vector<double>(2));
            prepared.sums.reset(context.mesh.num_owned_cells(), 1, 3);
            for (std::size_t c = 0; c < prepared.post_chemistry.size(); ++c)
            {
                for (std::size_t s = 0; s < 3; ++s)
                    prepared.sums.add_material(c, 0, s,
                        f.chemistry_ledgers[c].chemicalChange.amount(*f.catalog, f.catalog->species()[s].id,
                            s == 0 ? R::SpeciesPhase::Liquid : R::SpeciesPhase::Dissolved));
                prepared.gross_events[c] = {
                    f.chemistry_ledgers[c].forwardReactionExtents[0], f.chemistry_ledgers[c].reverseReactionExtents[0]};
            }
        });
    ASSERT_EQ(result.faces.size(), 2U);
    for (const auto& event : result.faces[0]->diffusion.faces)
        EXPECT_NEAR(event.moles[1], .02734375, 2e-15);
    for (const auto& event : result.faces[1]->diffusion.faces)
        EXPECT_NEAR(event.moles[1], .025634765625, 2e-15); // Updated concentration gap 1.640625, not 1.75.
    for (std::size_t c = 0; c < result.post_faces.size(); ++c)
    {
        const bool lower = f.mesh->cell_global_id(static_cast<LO>(c)) == 0;
        EXPECT_NEAR(result.post_faces[c].amount(*f.catalog, one, R::SpeciesPhase::Dissolved),
            lower ? .822021484375 : .052978515625, 2e-14);
        EXPECT_EQ(result.result.gross_events[c].size(), 26U); // 2 chemical + 2*(2*3+6) face events.
        EXPECT_EQ(result.post_faces[c].amounts, result.faces.back()->diffusion.state[c].amounts);
    }
}

TEST(LowMachFaceTransportTest, MissingCounterRejectsCollectivelyWhileExplicitZeroIsAnOrdinaryFiniteDonor)
{
    Fixture f(2, R::SpeciesId{0});
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const int local_empty_rank = f.mesh->num_owned_cells() == 0 ? comm->getRank() : -1;
    int empty_rank = -1;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_empty_rank, &empty_rank);
    const int invalid_rank = empty_rank >= 0 ? empty_rank : 0;
    auto missing = f.options(false);
    if (comm->getRank() == invalid_rank)
        missing.diffusion->binding.counterSpecies.reset();
    EXPECT_ANY_THROW((void) F::LowMachFaceTransportAdapter(f.mesh, f.catalog, missing, f.bindings()));
    auto explicit_zero = f.options(false);
    ASSERT_TRUE(explicit_zero.diffusion->binding.counterSpecies.has_value());
    EXPECT_EQ(*explicit_zero.diffusion->binding.counterSpecies, 0U);
    explicit_zero.diffusion->primary_species = {one};
    explicit_zero.diffusion->molar_conductance = [](const auto&) { return std::vector<double>{0., .125, 0.}; };
    F::LowMachFaceTransportAdapter adapter(f.mesh, f.catalog, explicit_zero, f.bindings());
    const auto receipt = adapter.preview(f.context(), f.post_chemistry);
    for (const auto& face : receipt.diffusion.faces)
    {
        EXPECT_NEAR(face.counterMoles, -.0546875, 2e-14);
        EXPECT_NEAR(face.enthalpy, -229.6875, 1e-10);
    }
    double counter_change = 0;
    for (const auto& cell : receipt.diffusion.cells)
        counter_change += cell.change.amount(*f.catalog, 0, R::SpeciesPhase::Liquid);
    EXPECT_DOUBLE_EQ(f.sum(counter_change), 0.);
}

TEST(LowMachFaceTransportTest, CustomAllocationCannotHideALostJouleAfterLargeFaceCancellation)
{
    const R::SpeciesCatalog catalog(std::vector<R::SpeciesDefinition>{
        {73, "manufactured caloric carrier", .01, {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
    const double large = std::ldexp(1., 60);
    F::LocalSourceAccumulator sums;
    sums.reset(1, 1, 1);
    sums.declare_energy(0, large);
    sums.declare_energy(0, 1.); // Original independent physical channel, absent from the lossy custom allocation.
    sums.add_allocated_energy(0, 0, large);
    F::Receipt face;
    face.physical_phase = 0;
    face.cells.resize(1);
    face.cells[0].fourier_energy = {-large};
    sums.add_faces(face, catalog);
    // Net declared budget is 1 J; final rounded packet would be zero. Gross turnover cannot set the tolerance.
    EXPECT_THROW((void) sums.finalize(), std::invalid_argument);
}

TEST(LowMachFaceTransportTest, DefaultRawEnergyChannelsPreserveOneJouleAfterFaceCancellation)
{
    const R::SpeciesCatalog catalog(std::vector<R::SpeciesDefinition>{
        {73, "manufactured caloric carrier", .01, {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
    const double large = std::ldexp(1., 60);
    F::LocalSourceAccumulator sums;
    sums.reset(1, 1, 1);
    sums.add_energy(0, 0, large);
    sums.add_energy(0, 0, 1.);
    F::Receipt face;
    face.physical_phase = 0;
    face.cells.resize(1);
    face.cells[0].fourier_energy = {-large};
    sums.add_faces(face, catalog);
    const auto packets = sums.finalize();
    ASSERT_EQ(packets.size(), 1U);
    ASSERT_EQ(packets[0].internal_energy.size(), 1U);
    EXPECT_DOUBLE_EQ(packets[0].internal_energy[0], 1.);
}
TEST(LowMachFaceTransportTest, ReceiptBindsItsProducingAdapterAndPhysicalPhase)
{
    Fixture f;
    const auto ns = f.catalog->species().size();
    const auto gas = R::make_ideal_gas_mixture(f.catalog, std::vector<R::IdealGasCaloricSpecies>(ns, {20., 300., 0.}),
        {250, 600, 10000, 2000000}, R::EnergyConvention::SensibleEnergy);
    auto bindings = f.bindings();
    bindings.push_back(std::vector<R::DiffusionCompartment>(ns, {R::SpeciesPhase::Resolved, {}}));
    for (std::size_t c = 0; c < f.thermal.cells.size(); ++c)
    {
        // A real second physical role with no occupied stock: its EOS and h laws are not needed.
        f.thermal.cells[c].phases.push_back({gas, std::vector<double>(ns), 100000., 0., 0.});
        f.transport.cells[c].phases.push_back({std::vector<double>(ns), 0., 0., std::nullopt});
    }
    const auto first_options = f.options(false);
    auto other_options = first_options;
    other_options.identity = "distinct retained ordered option instance";
    other_options.diffusion->binding.identity = "half the first conductance";
    other_options.diffusion->molar_conductance = [](const auto&) { return std::vector<double>{0., .0625, .0625}; };
    auto empty_options = f.options();
    empty_options.identity = "explicit absent gas contact";
    empty_options.physical_phase = 1;
    empty_options.diffusion.reset();
    empty_options.fourier->conductance = [](const auto&) { return 0.; };
    F::LowMachFaceTransportAdapter first(f.mesh, f.catalog, first_options, bindings);
    F::LowMachFaceTransportAdapter other(f.mesh, f.catalog, other_options, bindings);
    F::LowMachFaceTransportAdapter empty(f.mesh, f.catalog, empty_options, bindings);
    const auto first_receipt = first.preview(f.context(), f.post_chemistry);
    const auto other_receipt = other.preview(f.context(), f.post_chemistry);
    const auto empty_receipt = empty.preview(f.context(), f.post_chemistry);
    EXPECT_EQ(first_receipt.stamp, other_receipt.stamp);
    EXPECT_EQ(first_receipt.stamp, empty_receipt.stamp);
    EXPECT_EQ(first_receipt.physical_phase, other_receipt.physical_phase);
    EXPECT_NE(first_receipt.physical_phase, empty_receipt.physical_phase);
    EXPECT_NO_THROW(first.validate_receipt(first_receipt, f.context()));
    EXPECT_NO_THROW(other.validate_receipt(other_receipt, f.context()));
    EXPECT_NO_THROW(empty.validate_receipt(empty_receipt, f.context()));
    EXPECT_ANY_THROW(first.validate_receipt(other_receipt, f.context()));
    EXPECT_ANY_THROW(other.validate_receipt(first_receipt, f.context()));
    EXPECT_ANY_THROW(first.validate_receipt(empty_receipt, f.context()));
    EXPECT_ANY_THROW(empty.validate_receipt(first_receipt, f.context()));
    auto wrong_phase = first_receipt;
    wrong_phase.physical_phase = 1;
    EXPECT_ANY_THROW(first.validate_receipt(wrong_phase, f.context()));
    // Rejected foreign receipts cannot consume or invalidate their valid originals.
    EXPECT_NO_THROW(first.validate_receipt(first_receipt, f.context()));
    EXPECT_NO_THROW(other.validate_receipt(other_receipt, f.context()));
    EXPECT_NO_THROW(empty.validate_receipt(empty_receipt, f.context()));
}

TEST(LowMachFaceTransportTest, NativeInteriorPeriodicSeamsRejectBeforeLawsEvenWithEmptyOwners)
{
    Fixture f;
    auto options = f.options();
    int calls = 0;
    options.diffusion->molar_conductance = [&calls](const auto&)
    {
        ++calls; // Instrumentation only; no external physical state.
        return std::vector<double>{0., .125, .125};
    };
    options.fourier->conductance = [&calls](const auto&)
    {
        ++calls;
        return 2.;
    };
    const auto native = std::make_shared<S::Meshes::OrthogonalCartesian3D>(
        S::Vec3D<S::ArrReal>{S::ArrReal{0, 1}, S::ArrReal{0, 1}, S::ArrReal{0, .5, 1}},
        S::Vec3D<bool>{false, false, true});
    Flow::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    const auto periodic = std::make_shared<Flow::Mesh>(native, distribution);
    const bool compatibility_before = periodic->has_materialized_indexer();
    double interior = 0, empty = periodic->num_owned_cells() == 0 ? 1. : 0.;
    for (std::size_t face_index = 0; face_index < periodic->num_faces(); ++face_index)
    {
        const auto face = static_cast<LO>(face_index);
        if (periodic->is_owned_face(face) && !periodic->is_geometry_exterior_face(face))
            interior += 1.;
    }
    // The unsupported connections really are logical interiors, not paired exterior faces.
    EXPECT_GT(f.sum(interior), 0.);
    EXPECT_EQ(f.sum(empty), std::max(0, periodic->owned_cell_map()->getComm()->getSize() - 2));
    EXPECT_ANY_THROW((void) F::LowMachFaceTransportAdapter(periodic, f.catalog, options, f.bindings()));
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(periodic->geometry_identity(), native.get());
    EXPECT_EQ(periodic->has_materialized_indexer(), compatibility_before);
    // Rejection leaves a valid nonperiodic adapter usable with the same laws.
    F::LowMachFaceTransportAdapter ordinary(f.mesh, f.catalog, options, f.bindings());
    EXPECT_NO_THROW((void) ordinary.preview(f.context(), f.post_chemistry));
    EXPECT_GT(f.sum(calls), 0.);
}

TEST(LowMachFaceTransportTest, NetSourceCannotEraseFiniteStockAfterHugeDepletionAndSmallFaceCredit)
{
    F::LocalSourceAccumulator sums;
    sums.reset(1, 1, 1);
    sums.add_material(0, 0, 0, -1e20L); // Chemistry exhausts the transported donor.
    sums.add_material(0, 0, 0, 1.);     // A later physical face stage credits one real mole.
    const auto packets = sums.finalize();
    EXPECT_DOUBLE_EQ(packets[0].species_moles[0][0], -1e20);
    EXPECT_THROW(F::validate_source_endpoint(1e20, packets[0].species_moles[0][0], 1.), std::invalid_argument);
    EXPECT_NO_THROW(F::validate_source_endpoint(1e20, -1e20, 0.)); // True complete disappearance.
    EXPECT_NO_THROW(F::validate_source_endpoint(2., -1., 1.));
    EXPECT_NO_THROW(F::validate_source_endpoint(0., 1., 1.));
}

TEST(LowMachFaceTransportTest, SourceReplayBoundUsesEndpointStockAndExactEmptyState)
{
    EXPECT_THROW(F::validate_source_endpoint(1., -1., 1e-100), std::invalid_argument);
    EXPECT_THROW(F::validate_source_endpoint(1., -1. + 1e-15, 0.), std::invalid_argument);
    EXPECT_THROW(F::validate_source_endpoint(1e20, -1e20, .001), std::invalid_argument);
    EXPECT_THROW(F::validate_source_endpoint(2., -.5, 1.), std::invalid_argument);
    EXPECT_THROW(F::validate_source_endpoint(0., -1., 0.), std::invalid_argument);
    EXPECT_NO_THROW(F::validate_source_endpoint(1., -.9, .1)); // Ordinary endpoint-relative representation error.
}

} // namespace
