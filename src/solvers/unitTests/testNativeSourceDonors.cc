/** @file
 * @brief Native source donor authority with manufactured numerical conditioning.
 */
#include "equations/BubblePopulationKinetics.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <exception>
#include <string>
#include <gtest/gtest.h>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using P = R::SpeciesPhase;
constexpr R::SpeciesId primary = 0, product = 17, counter = 29;
constexpr double stock = 1e20, molar_mass = 1e-20, temperature = 300, pressure = 100000;
enum class Case
{
    FacePrimaryDebit,
    FaceCounterDebit,
    FaceProductFunding,
    FaceHighCorrectionFunding,
    InterphaseDebit,
    InterphaseProductFunding
};
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Fixture
{
    std::shared_ptr<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, manufactured_second_phase;
    std::shared_ptr<Flow> flow;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    bool interphase;
    explicit Fixture(Case selected)
        : interphase(selected == Case::InterphaseDebit || selected == Case::InterphaseProductFunding)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        const bool high_funding = selected == Case::FaceHighCorrectionFunding;
        const S::ArrReal z = high_funding ? S::ArrReal{0., 1., 2., 3.} : S::ArrReal{0., 1., 2.};
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Flow::Mesh::Cartesian>(
                                              S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, z}}),
            distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {primary, "synthetic primary", molar_mass, {{"X", 1}}, {P::Liquid, P::Resolved}},
            {product, "synthetic product", molar_mass, {{"X", 1}}, {P::Liquid, P::Resolved}},
            {counter, "synthetic finite counter", molar_mass, {{"X", 1}}, {P::Liquid, P::Resolved}}});
        mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{
                {61, "primary to product", {{primary, P::Liquid, -1}, {product, P::Liquid, 1}}, 0},
                {63, "counter to product", {{counter, P::Liquid, -1}, {product, P::Liquid, 1}}, 0},
                {65, "finite counter to primary", {{counter, P::Liquid, -1}, {primary, P::Liquid, 1}}, 0}},
            R::EnergyConvention::SensibleEnergy);
        const R::ThermodynamicBounds bounds{250, 600, 1000, 1e7};
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1, 2, 0, 0, 600, 0, 0}; },
            R::EnergyConvention::SensibleEnergy);
        // Explicit synthetic ideal-gas parameters give rho=1 and e=600 at T0,p0,
        // matching the donor reference point while retaining positive compressibility
        // and factory-supplied fugacities. This numerical gas constant is not a physical correlation.
        manufactured_second_phase = R::make_ideal_gas_mixture(catalog,
            std::vector<R::IdealGasCaloricSpecies>(
                3, R::IdealGasCaloricSpecies{2 * molar_mass, temperature, 600 * molar_mass}),
            bounds, R::EnergyConvention::SensibleEnergy, molar_mass * pressure / temperature);
        Flow::Options controls;
        controls.minimum_pressure = 50000;
        controls.maximum_pressure = 200000;
        controls.linear.preconditioner = S::LinearPreconditioner::None;
        controls.linear.tolerance = 1e-12;
        controls.mixture_dynamic_viscosity = [](auto, auto) { return 0.; };
        controls.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solution"}};
        std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos{liquid};
        if (interphase)
        {
            eos.push_back(manufactured_second_phase);
            controls.phase_roles.push_back({Flow::PhaseRole::BulkGas, "manufactured_receiver"});
        }
        flow = std::make_shared<Flow>(mesh, eos, controls);
        Flow::State initial;
        initial.thermodynamic_pressure = pressure;
        initial.cells.resize(mesh->num_owned_cells());
        for (std::size_t c = 0; c < initial.cells.size(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const bool lower = mesh->cell_centroid(cell).z < 1;
            EXPECT_DOUBLE_EQ(mesh->cell_volume(cell), 1);
            const bool center = mesh->cell_centroid(cell).z > 1 && mesh->cell_centroid(cell).z < 2;
            const std::vector<double> moles =
                high_funding ? (center ? std::vector<double>{stock, 0, 1} : std::vector<double>{0, 0, stock})
                             : (lower ? std::vector<double>{stock, 0, 0} : std::vector<double>{0, 0, stock});
            initial.cells[c].phases = {{liquid, moles, pressure, 600, temperature}};
            initial.cells[c].phase_numbers = {std::nullopt};
            if (interphase)
            {
                initial.cells[c].phases.push_back({manufactured_second_phase, {0, 0, 0}, pressure, 0, 0});
                initial.cells[c].phase_numbers.push_back(std::nullopt);
            }
        }
        flow->initialize(std::move(initial));
        int empty = mesh->num_owned_cells() == 0, actual_empty = 0;
        const auto comm = mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &actual_empty);
        EXPECT_EQ(actual_empty, std::max(0, comm->getSize() - (high_funding ? 3 : 2)));
    }
    R::ReactingMixtureOptions options(Case selected) const
    {
        R::ReactingMixtureOptions options;
        options.mechanism = mechanism;
        options.source_identity = "native finite raw predecessor";
        options.phases = {{P::Liquid, {}, {P::Liquid, P::Liquid, P::Liquid}}};
        if (interphase)
            options.phases.push_back({P::Resolved, {}, {}});
        options.source = [selected](const R::CoupledReactionContext& context, double)
        {
            R::ReactionSourceInput input;
            if (!context.reaction.environment.location)
                throw std::invalid_argument("This manufactured source requires actual geometry.");
            const bool lower = context.reaction.environment.location->centroid[2] < 1;
            if (selected == Case::FaceHighCorrectionFunding)
            {
                const double z = context.reaction.environment.location->centroid[2];
                if (z > 1 && z < 2)
                    input.integratedExtents = {{65, 1}};
            }
            else if (selected == Case::FaceCounterDebit ? !lower : lower)
                input.integratedExtents = {{selected == Case::FaceCounterDebit ? 63u : 61u, 1}};
            return input;
        };
        if (!interphase)
        {
            const auto transported = selected == Case::FaceProductFunding ? product : primary;
            R::MixtureDiffusionLaw law;
            law.binding.transported = law.binding.counter = {P::Liquid, {}};
            law.binding.counterSpecies = counter;
            law.binding.identity = "explicit synthetic same-carrier counterflux";
            law.binding.schemaFingerprint = catalog->fingerprint();
            law.binding.energyConvention = R::EnergyConvention::SensibleEnergy;
            law.primary_species = {transported};
            const auto index = catalog->index(transported);
            law.molar_conductance = [index, selected](const R::MixtureFaceContext& context)
            {
                std::vector<double> result(3, 0);
                const double contact = selected == Case::FaceHighCorrectionFunding && context.center[2] > 1.5
                                           ? 1e-20
                                           : 1;
                result[index] = contact / context.time_step; // Explicit contact, no donor-dependent admission.
                return result;
            };
            law.partial_molar_enthalpy = [](R::SpeciesId, const R::DiffusionCompartment&,
                                             const R::MixtureFacePhaseSample& sample)
            { return molar_mass * (2 * sample.temperature + sample.absolute_pressure); };
            R::MixtureFaceTransportOptions face;
            face.identity = "canonical raw predecessor face";
            face.schema_fingerprint = catalog->fingerprint();
            face.physical_phase = 0;
            face.diffusion = std::move(law);
            options.face_transport = {std::move(face)};
        }
        else
        {
            const auto transported = selected == Case::InterphaseProductFunding ? product : primary;
            R::InterphaseConfiguration configuration;
            configuration.metadata = {"raw predecessor interphase", 1, catalog->fingerprint()};
            configuration.stages = {{"explicit material relocation", R::InterphaseStageKind::Material,
                {{transported, 0, 1}}, {}}};
            auto partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(R::PartialMolarEnthalpyLaw{
                {"manufactured constant-density true partials", 1, catalog->fingerprint()}, liquid->bounds(),
                [](double t, double p, std::span<const double>)
                {
                    return R::PartialMolarEnthalpyPoint{
                        std::vector<double>(3, molar_mass * (2 * t + p)),
                        std::vector<double>(3, 2 * molar_mass), std::vector<double>(3, molar_mass)};
                }});
            configuration.phase_enthalpy = {partial, nullptr};
            options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
                R::MixtureInterphaseEventProvider{{"finite interphase rate packet", 1, catalog->fingerprint()},
                    std::make_shared<const R::InterphaseConfiguration>(std::move(configuration)),
                    [selected](const R::MixtureInterphaseContext& context)
                    {
                        const bool lower = context.location && context.location->centroid[2] < 1;
                        return std::vector<std::vector<double>>{
                            {lower ? (selected == Case::InterphaseProductFunding ? 1 : stock) : 0}};
                    }});
        }
        return options;
    }
};
void same(const Flow::State& actual, const Flow::State& expected)
{
    EXPECT_EQ(actual.revision, expected.revision);
    EXPECT_DOUBLE_EQ(actual.time, expected.time);
    EXPECT_DOUBLE_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    EXPECT_EQ(actual.cells.size(), expected.cells.size());
    for (std::size_t c = 0; c < std::min(actual.cells.size(), expected.cells.size()); ++c)
    {
        const auto& a = actual.cells[c];
        const auto& b = expected.cells[c];
        EXPECT_DOUBLE_EQ(a.mechanical_pressure, b.mechanical_pressure);
        EXPECT_EQ(a.phase_numbers, b.phase_numbers);
        EXPECT_EQ(a.phases.size(), b.phases.size());
        EXPECT_EQ(a.drift_velocity.size(), b.drift_velocity.size());
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(a.velocity.component(axis), b.velocity.component(axis));
            for (std::size_t p = 0; p < std::min(a.drift_velocity.size(), b.drift_velocity.size()); ++p)
                EXPECT_DOUBLE_EQ(a.drift_velocity[p].component(axis), b.drift_velocity[p].component(axis));
        }
        for (std::size_t p = 0; p < std::min(a.phases.size(), b.phases.size()); ++p)
        {
            EXPECT_EQ(a.phases[p].thermodynamics.get(), b.phases[p].thermodynamics.get());
            EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
            EXPECT_DOUBLE_EQ(a.phases[p].temperature, b.phases[p].temperature);
            EXPECT_DOUBLE_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
            EXPECT_DOUBLE_EQ(a.phases[p].absolute_pressure, b.phases[p].absolute_pressure);
        }
    }
}
TEST(NativeSourceDonorTest, ChemicalPrimaryAndRemoteCounterDebitsRejectWithoutChangingAcceptedState)
{
    for (const auto selected : {Case::FacePrimaryDebit, Case::FaceCounterDebit, Case::InterphaseDebit})
    {
        SCOPED_TRACE(static_cast<int>(selected));
        Fixture f(selected);
        R::ReactingMixtureCoupler controller(f.flow, f.options(selected));
        const auto accepted = f.flow->state();
        std::vector<double> flux;
        for (auto face : f.flow->volume_flux().owned_face_ids())
            flux.push_back(f.flow->volume_flux().value(face));
        const bool interphase = selected == Case::InterphaseDebit;
        const char* const donorMessage = interphase
            ? "Interphase gross outgoing events overdraw their actual predecessor donor."
            : "Canonical diffusion jointly exhausts a finite species/counter donor; recompute a smaller physical step.";
        const char* const peerMessage = interphase
            ? "Frozen post-face interphase material stage failed on another rank."
            : "Face owned donor reservation and source assembly failed on another rank.";
        bool ownsAffectedDonor = false;
        for (std::size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
        {
            const bool lower = f.mesh->cell_centroid(static_cast<LO>(c)).z < 1;
            ownsAffectedDonor |= selected == Case::FaceCounterDebit ? !lower : lower;
        }
        std::string failure;
        try
        {
            (void) controller.preview(.01);
        }
        catch (const std::exception& error)
        {
            failure = error.what();
        }
        catch (...)
        {
            failure = "Non-standard exception does not qualify the intended finite-donor rejection.";
        }
        EXPECT_TRUE(failure == donorMessage || failure == peerMessage) << failure;
        if (failure == donorMessage)
            EXPECT_TRUE(ownsAffectedDonor) << "A raw donor error must originate on its actual cell owner.";
        // Peers retain the collective fence context; empty owners are neutral.
        // An unrelated EOS, pressure or callback exception cannot qualify this test.
        int localOwner = ownsAffectedDonor, actualOwners = 0;
        int localDonorFailure = ownsAffectedDonor && failure == donorMessage, ownerSawDonorFailure = 0;
        const auto comm = f.mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &localOwner, &actualOwners);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &localDonorFailure, &ownerSawDonorFailure);
        EXPECT_EQ(actualOwners, 1);
        EXPECT_EQ(ownerSawDonorFailure, 1);
        same(f.flow->state(), accepted);
        std::size_t row = 0;
        for (auto face : f.flow->volume_flux().owned_face_ids())
            EXPECT_DOUBLE_EQ(f.flow->volume_flux().value(face), flux[row++]);
    }
}
TEST(NativeSourceDonorTest, ChemicalProductFundsFaceAndInterphaseStagesWithTheSamePhysicalGatesAndReplay)
{
    for (const auto selected : {Case::FaceProductFunding, Case::InterphaseProductFunding})
    {
        SCOPED_TRACE(static_cast<int>(selected));
        Fixture f(selected);
        R::ReactingMixtureCoupler controller(f.flow, f.options(selected));
        const auto accepted = f.flow->state();
        const auto saved = controller.checkpoint();
        const auto trial = controller.preview(.01);
        same(f.flow->state(), accepted);
        EXPECT_LE(trial.diagnostics().flow.maximum_occupancy_error, f.flow->options().volume_relative_tolerance);
        EXPECT_LE(trial.diagnostics().flow.maximum_source_residual, 1);
        EXPECT_LE(trial.diagnostics().flow.maximum_event_residual, 1);
        for (std::size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const bool lower = f.mesh->cell_centroid(static_cast<LO>(c)).z < 1;
            const auto phase = selected == Case::FaceProductFunding ? 0u : 1u;
            const double expected = selected == Case::FaceProductFunding ? (lower ? 0 : 1) : (lower ? 1 : 0);
            EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[phase].moles[f.catalog->index(product)], expected);
            for (const auto& entry : trial.state().cells[c].phases)
                if (std::ranges::any_of(entry.moles, [](double n) { return n > 0; }))
                    EXPECT_NEAR(entry.temperature, temperature, f.flow->options().thermal_relative_tolerance * temperature);
        }
        controller.commit(trial);
        same(f.flow->state(), trial.state());
        controller.restore(saved);
        same(f.flow->state(), accepted);
        const auto replay = controller.preview(.01);
        same(replay.state(), trial.state());
        EXPECT_EQ(replay.local_sources().size(), trial.local_sources().size());
        for (std::size_t c = 0; c < replay.local_sources().size(); ++c)
            EXPECT_EQ(replay.local_sources()[c].reactionExtents, trial.local_sources()[c].reactionExtents);
    }
}
TEST(NativeSourceDonorTest, ChemicalHighAndCorrectionCreditFundsBothOriginalOutgoingFaces)
{
    Fixture f(Case::FaceHighCorrectionFunding);
    R::ReactingMixtureCoupler controller(f.flow, f.options(Case::FaceHighCorrectionFunding));
    const auto accepted = f.flow->state();
    const auto saved = controller.checkpoint();
    // The supplied conductance scales as 1/dt, so integrated requests are unchanged.
    // A binary-exact duration keeps the intended one-mole contact from leaving
    // an artificial residual through the fixture's division/multiplication.
    constexpr double duration = .125;
    const auto trial = controller.preview(duration);
    same(f.flow->state(), accepted);
    EXPECT_LE(trial.diagnostics().flow.maximum_occupancy_error, f.flow->options().volume_relative_tolerance);
    EXPECT_LE(trial.diagnostics().flow.maximum_event_residual, 1);
    for (std::size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const double z = f.mesh->cell_centroid(static_cast<LO>(c)).z;
        const double expected = z < 1 ? stock : (z < 2 ? 0 : 1);
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].moles[f.catalog->index(primary)], expected);
    }
    // The source supplies 1 mol before a single face stage exports 1e20 and 1 mol jointly.
    // The final net packet is representable; source/thermal/transport gates all remain active.
    controller.commit(trial);
    controller.restore(saved);
    const auto replay = controller.preview(duration);
    same(replay.state(), trial.state());
}
} // namespace
