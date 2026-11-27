/** Independent fixed cylindrical kinetic portable continuation and restart admission. */
#include "QbmmKineticTestSupport.hh"
#include "QbmmRestartTestSupport.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <numbers>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
namespace Restart = Testing::QbmmRestart;
using Coupler = R::ReactingMixtureCoupler;
using Vector = Flow::Vector;
using Phase = R::SpeciesPhase;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double dt = 1. / 128., pressure = 64., temperature = 8., scale = 1. / 16.;
constexpr std::array geometries{"sector", "closed_annulus", "polygon_disk", "polygon_annulus"};
const std::array<std::optional<Vector>, 3> accelerations{std::nullopt, Vector{0., 0., -.125}, Vector{}};

enum Geometry : size_t
{
    Sector,
    ClosedAnnulus,
    PolygonDisk,
    PolygonAnnulus,
    Cartesian
};

SP<Flow::Mesh> make_mesh(Geometry geometry, bool different = false, bool repartition = false)
{
    const auto ranks = static_cast<size_t>(Tpetra::getDefaultComm()->getSize());
    constexpr auto turn = 2. * std::numbers::pi;
    ArrReal z{0., .5, different && geometry != ClosedAnnulus ? 1.125 : 1.};
    Flow::Mesh::DistributionOptions partition{.allow_empty_partitions = true};
    if (geometry == Cartesian || geometry == Sector || geometry == ClosedAnnulus)
    {
        partition.orthogonal_partitions =
            repartition ? std::array<size_t, 3>{1, 1, ranks} : std::array<size_t, 3>{1, ranks, 1};
        if (geometry == Cartesian)
            return std::make_shared<Flow::Mesh>(
                std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{ArrReal{0., 1.}, ArrReal{0., .5, 1.}, z}),
                partition);
        ArrReal angles{0., .25, .5};
        if (geometry == ClosedAnnulus)
        {
            angles.clear();
            const double span = different ? turn - .125 : turn;
            for (unsigned j = 0; j <= 8; ++j)
                angles.push_back(span * j / 8);
        }
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{ArrReal{1., 1.5, 2.}, angles, z}), partition);
    }

    using Semi = Flow::Mesh::SemiStructured;
    constexpr unsigned sectors = 8;
    const bool disk = geometry == PolygonDisk;
    Arr<Semi::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    if (disk)
        nodes.push_back({0., 0., 0.});
    for (unsigned j = 0; j < sectors; ++j)
    {
        const double theta = turn * j / sectors;
        const double x = std::cos(theta), y = std::sin(theta);
        nodes.push_back({x, y, 0.});
        if (!disk)
            nodes.push_back({2. * x, 2. * y, 0.});
        const unsigned next = (j + 1) % sectors;
        cells.push_back(
            disk ? Arr<unsigned>{0, j + 1, next + 1} : Arr<unsigned>{2 * j, 2 * j + 1, 2 * next + 1, 2 * next});
    }
    partition.xy_partitions = repartition ? 1 : ranks;
    return std::make_shared<Flow::Mesh>(std::make_shared<Semi>(nodes, cells, z), partition);
}

struct Fixture
{
    std::shared_ptr<const R::SpeciesCatalog> species =
        std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {47, "cylindrical restart material", 1., {{"M", 1}}, {Phase::Liquid, Phase::Dispersed}}});
    const R::ThermodynamicBounds bounds{2., 16., 16., 128.};
    std::shared_ptr<const R::MixtureThermodynamics> liquid = R::make_linear_compressible_mixture_liquid(species, bounds,
        [](auto) { return R::LinearCompressibleMixtureLiquid{temperature, pressure, 16., 2., 0., 0., 16., 0., 0.}; });
    std::shared_ptr<const R::MixtureThermodynamics> gas = R::make_ideal_gas_mixture(
        species, {{2., temperature, 16.}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
    SP<Flow::Mesh> grid;
    std::shared_ptr<Flow> flow;
    int node_calls = 0;
    bool reject_law_calls = false;

    Fixture(Geometry geometry, bool initialize, bool different = false, bool repartition = false,
        std::uint64_t law_revision = 1, std::uint64_t eos_revision = 1)
        : grid(make_mesh(geometry, different, repartition))
    {
        Flow::Options options;
        options.use_multi_rhs_transport = Restart::use_multi_rhs_transport();
        options.minimum_pressure = bounds.minimum_pressure;
        options.maximum_pressure = bounds.maximum_pressure;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1.e-12;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "solvent"}, {Flow::PhaseRole::DispersedGas, "bubbles"}};
        options.qbmm_eos_law_identities = {
            R::ThermalLawIdentity{"constant-density solvent", eos_revision, species->fingerprint()}, std::nullopt};
        options.qbmm_model.family_transport = Model::FamilyTransport::KineticEuler;
        options.qbmm_model.velocity = Model::Velocity::NodeDependentSlip;
        const R::QbmmLawMetadata metadata{
            "two axial velocities", law_revision, R::QbmmLawKind::NodeSlip, "physical atom mass in kg", "m/s"};
        R::QbmmClosureVersions versions;
        versions.laws = {metadata};
        options.qbmm_families = {
            nullptr, std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), "bubbles",
                         options.phase_roles[1], scale, std::vector<R::QbmmMassFraction>{{47, 1.}}, versions)};
        options.qbmm_node_slip_laws = {
            nullptr, std::make_shared<const R::QbmmNodeSlipLaw>(
                         R::QbmmNodeSlipLaw{metadata, [this](const R::QbmmNodeSlipInput& node)
                             {
                                 ++node_calls;
                                 if (reject_law_calls)
                                     throw std::logic_error("Restart invoked the node law.");
                                 return std::array{0., 0., node.mass_kg_per_bubble < 2. * scale ? .125 : .25};
                             }})};
        flow = std::make_shared<Flow>(grid, std::vector{liquid, gas}, options);
        if (!initialize)
            return;

        Flow::State state;
        state.thermodynamic_pressure = pressure;
        state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*options.qbmm_families[1])};
        state.cells.resize(grid->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const auto local = static_cast<LO>(c);
            const double volume = grid->cell_volume(local);
            const double void_fraction = grid->cell_centroid(local).z < .5 ? .125 : .25;
            const double liquid_moles = (1. - void_fraction) * volume * 16.;
            const double gas_moles = void_fraction * volume * pressure / temperature;
            const double heavy = gas_moles / (5. * scale), light = 2. * heavy;
            const std::array composition{1.};
            const double liquid_u = liquid->evaluate(temperature, pressure, composition).internal_energy;
            const double gas_u = gas->evaluate(temperature, pressure, composition).internal_energy;
            auto& cell = state.cells[c];
            cell.phases = {{liquid, {liquid_moles}, pressure, liquid_moles * liquid_u, temperature},
                {gas, {gas_moles}, pressure, gas_moles * gas_u, temperature}};
            cell.phase_numbers = {std::nullopt, light + heavy};
            cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{light + 9. * heavy, light + 27. * heavy}};
        }
        flow->initialize(std::move(state));
    }

    R::ReactingMixtureOptions controller_options(std::optional<Vector> force) const
    {
        R::ReactingMixtureOptions options;
        options.mechanism = std::make_shared<const R::ReactionMechanism>(
            species, std::vector<R::ReactionDefinition>{}, liquid->energy_convention());
        options.phases = {{Phase::Liquid, {}, {}}, {Phase::Dispersed, "bubbles", {}}};
        options.source_identity = "closed cylindrical continuation";
        options.source = [](auto&&, double) { return R::ReactionSourceInput{}; };
        options.body_acceleration = force;
        return options;
    }
};

std::uint64_t word(std::span<const std::byte> bytes, size_t position)
{
    std::uint64_t value = 0;
    for (size_t b = 0; b < 8; ++b)
        value |= std::to_integer<std::uint64_t>(bytes[position * 8 + b]) << (8 * b);
    return value;
}

void set_word(std::span<std::byte> bytes, size_t position, std::uint64_t value)
{
    for (size_t b = 0; b < 8; ++b)
        bytes[position * 8 + b] = static_cast<std::byte>((value >> (8 * b)) & 0xff);
}

size_t first_axial_node(std::span<const std::byte> image)
{
    // Fixed tag header, optional impulse interval, then 23 binding words.
    const size_t cell = 12 + (word(image, 11) ? 17 : 0) + 23;
    // Carrier, liquid and gas records precede the first node velocity.
    return cell + 4 + 10 + 14;
}

void expect_same_endpoint(const Flow& left, const Flow& right)
{
    const auto& x = left.state();
    const auto& y = right.state();
    EXPECT_EQ(x.time, y.time);
    EXPECT_EQ(x.revision, y.revision);
    EXPECT_EQ(x.thermodynamic_pressure, y.thermodynamic_pressure);
    EXPECT_EQ(x.qbmm_families, y.qbmm_families);
    ASSERT_EQ(x.cells.size(), y.cells.size());
    for (size_t c = 0; c < x.cells.size(); ++c)
    {
        const auto& a = x.cells[c];
        const auto& b = y.cells[c];
        EXPECT_EQ(a.velocity, b.velocity);
        EXPECT_EQ(a.mechanical_pressure, b.mechanical_pressure);
        EXPECT_EQ(a.phase_numbers, b.phase_numbers);
        EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
        EXPECT_EQ(a.drift_velocity, b.drift_velocity);
        EXPECT_EQ(a.qbmm_kinetic_closure, b.qbmm_kinetic_closure);
        EXPECT_EQ(a.qbmm_kinetic_eos_volumes, b.qbmm_kinetic_eos_volumes);
        ASSERT_EQ(a.phases.size(), b.phases.size());
        for (size_t p = 0; p < a.phases.size(); ++p)
        {
            EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
            EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
            EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
            EXPECT_EQ(a.phases[p].absolute_pressure, b.phases[p].absolute_pressure);
        }
    }
    for (size_t f = 0; f < left.mesh()->num_owned_faces(); ++f)
        EXPECT_EQ(
            left.volume_flux().owned_value(static_cast<LO>(f)), right.volume_flux().owned_value(static_cast<LO>(f)));
}

void expect_same_diagnostics(const Coupler::Diagnostics& a, const Coupler::Diagnostics& b)
{
    EXPECT_EQ(a.correctors, b.correctors);
    EXPECT_EQ(a.external_mass, b.external_mass);
    EXPECT_EQ(a.external_heat, b.external_heat);
    EXPECT_EQ(a.flow.body_acceleration_m_per_s2, b.flow.body_acceleration_m_per_s2);
    EXPECT_EQ(a.flow.impulse_parts, b.flow.impulse_parts);
    EXPECT_EQ(a.flow.pressure_impulse, b.flow.pressure_impulse);
    EXPECT_EQ(a.flow.external_impulse, b.flow.external_impulse);
    EXPECT_EQ(a.flow.pressure_work, b.flow.pressure_work);
    EXPECT_EQ(a.flow.momentum_error, b.flow.momentum_error);
    EXPECT_EQ(a.flow.kinetic_substeps, b.flow.kinetic_substeps);
    EXPECT_EQ(a.flow.maximum_kinetic_cfl, b.flow.maximum_kinetic_cfl);
}

void expect_archived(Fixture& fixture, const std::vector<std::byte>& image, std::optional<Vector> force)
{
    fixture.reject_law_calls = true;
    fixture.flow->restore_qbmm_restart(image);
    const auto report = fixture.flow->qbmm_report();
    EXPECT_EQ(fixture.node_calls, 0);
    EXPECT_EQ(report.provenance.origin, Flow::QbmmStateOrigin::PortableRestart);
    EXPECT_TRUE(report.provenance.archived_generation.has_value());
    EXPECT_TRUE(report.provenance.controller_configuration_signature.has_value());
    EXPECT_EQ(report.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    EXPECT_FALSE(report.interval);
    EXPECT_EQ(report.body_acceleration_m_per_s2, force);
    EXPECT_GT(report.kinetic_substeps, 0);
    EXPECT_TRUE(report.external_impulses_kg_m_per_s.has_value());
    EXPECT_TRUE(report.pressure_impulse_kg_m_per_s.has_value());
    EXPECT_EQ(report.balanced_kinetic_pressure_policy.has_value(), force.has_value());
    EXPECT_THROW((void) fixture.flow->preview(dt), std::exception);
    fixture.reject_law_calls = false;
}

void expect_rejected_controller(Fixture& fixture, std::optional<Vector> force)
{
    const auto baseline = fixture.flow->serialize_qbmm_restart();
    auto wrong = fixture.controller_options(force);
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    if (comm->getRank() == 0)
        wrong.body_acceleration = force ? std::nullopt : std::optional<Vector>{Vector{}};
    EXPECT_THROW(Coupler(fixture.flow, wrong), std::exception);
    wrong = fixture.controller_options(force);
    if (comm->getRank() == 0)
        wrong.source_identity += " changed";
    EXPECT_THROW(Coupler(fixture.flow, wrong), std::exception);
    EXPECT_EQ(fixture.flow->serialize_qbmm_restart(), baseline);
    EXPECT_EQ(fixture.node_calls, 0);
}

void expect_rejected_stored_node(
    Fixture& fixture, Coupler& controller, const std::vector<std::byte>& image, const Coupler::Trial& earlier)
{
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    const int changed_rank =
        reduce(*fixture.grid, Teuchos::REDUCE_MIN, fixture.grid->num_owned_cells() ? comm->getRank() : comm->getSize());
    auto corrupted = image;
    corrupted.resize(corrupted.size() - 16);
    if (comm->getRank() == changed_rank)
    {
        const size_t first_node = first_axial_node(corrupted);
        EXPECT_EQ(word(corrupted, first_node - 1), 2u);
        EXPECT_EQ(word(corrupted, first_node), 0u);
        set_word(corrupted, first_node, std::bit_cast<std::uint64_t>(.25));
    }
    Restart::append_restart_checksums(*comm, corrupted);
    const auto before = controller.serialize_qbmm_restart();
    const auto calls = fixture.node_calls;
    try
    {
        controller.restore_qbmm_restart(corrupted);
        ADD_FAILURE() << "A retained transverse node velocity was accepted.";
    }
    catch (const std::exception& error)
    {
        if (comm->getRank() == changed_rank)
            EXPECT_NE(std::string_view(error.what()).find("axial retained node slip"), std::string_view::npos);
    }
    EXPECT_EQ(fixture.node_calls, calls);
    EXPECT_EQ(controller.serialize_qbmm_restart(), before);
    EXPECT_TRUE(controller.qbmm_report(earlier).interval);
}

void expect_rejected_configuration(Geometry geometry, const std::vector<std::byte>& image)
{
    const auto reject = [&](Fixture& candidate)
    {
        EXPECT_THROW(candidate.flow->restore_qbmm_restart(image), std::exception);
        EXPECT_FALSE(candidate.flow->initialized());
        EXPECT_EQ(candidate.node_calls, 0);
    };
    Fixture changed_geometry(geometry, false, true);
    reject(changed_geometry);
    Fixture changed_eos(geometry, false, false, false, 1, 2);
    reject(changed_eos);
    Fixture changed_law(geometry, false, false, false, 2);
    reject(changed_law);
    if (changed_geometry.grid->owned_cell_map()->getComm()->getSize() > 1)
    {
        Fixture changed_partition(geometry, false, false, true);
        reject(changed_partition);
    }
}

TEST(LowMachQbmmCylindricalRestartTest, PortableContinuation)
{
    const auto* writer = std::getenv("SIMPLEFLUID_Q8D_RESTART_WRITE");
    const auto* reader = std::getenv("SIMPLEFLUID_Q8D_RESTART_READ");
    ASSERT_FALSE(writer && reader);
    const auto comm = Tpetra::getDefaultComm();
    if (writer || reader)
    {
        const auto* ranks = std::getenv("SIMPLEFLUID_Q8D_RESTART_RANKS");
        ASSERT_NE(ranks, nullptr);
        ASSERT_EQ(comm->getSize(), std::stoi(ranks));
    }
    ::testing::Test::RecordProperty("mpi_ranks", comm->getSize());

    for (size_t g = 0; g < geometries.size(); ++g)
        for (size_t a = 0; a < accelerations.size(); ++a)
        {
            const auto geometry = static_cast<Geometry>(g);
            const auto force = accelerations[a];
            const std::string stem = "." + std::string(geometries[g]) + "." + std::to_string(a);
            const auto first_path = stem + ".first";
            const auto next_path = stem + ".next";
            SCOPED_TRACE(stem);
            std::vector<std::byte> first, next;
            if (reader)
            {
                first = Restart::read_restart(*comm, reader, first_path.c_str());
                next = Restart::read_restart(*comm, reader, next_path.c_str());
            }
            else
            {
                Fixture initial(geometry, true);
                if (geometry == Sector)
                {
                    const int empty = reduce(*initial.grid, Teuchos::REDUCE_SUM, int(!initial.grid->num_owned_cells()));
                    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
                }
                Coupler controller(initial.flow, initial.controller_options(force));
                controller.advance(dt);
                first = controller.serialize_qbmm_restart();
                const auto checkpoint = controller.checkpoint();
                controller.advance(dt);
                next = controller.serialize_qbmm_restart();
                const auto calls = initial.node_calls;
                controller.restore(checkpoint);
                const auto checkpoint_report = controller.qbmm_report();
                EXPECT_EQ(initial.node_calls, calls);
                EXPECT_EQ(checkpoint_report.provenance.origin, Flow::QbmmStateOrigin::Checkpoint);
                EXPECT_EQ(checkpoint_report.receipt_history, Flow::QbmmReceiptHistory::Available);
                ASSERT_TRUE(checkpoint_report.interval);
                EXPECT_EQ(checkpoint_report.interval->local_face_transport.size(), initial.grid->num_owned_faces());
                EXPECT_FALSE(checkpoint_report.interval->local_kinetic_law_receipts.empty() &&
                             initial.grid->num_owned_cells() != 0);
                controller.advance(dt);
                Fixture uninterrupted(geometry, false);
                uninterrupted.flow->restore_qbmm_restart(next);
                expect_same_endpoint(*initial.flow, *uninterrupted.flow);
                if (writer)
                {
                    Restart::write_restart(*comm, writer, first_path.c_str(), first);
                    Restart::write_restart(*comm, writer, next_path.c_str(), next);
                    continue;
                }
            }
            EXPECT_EQ(word(first, 1), 9u);
            EXPECT_EQ(word(first, 2), 5u);
            EXPECT_EQ(word(first, 3), force.has_value());
            Fixture restored(geometry, false);
            expect_archived(restored, first, force);
            expect_rejected_controller(restored, force);
            Coupler continuation(restored.flow, restored.controller_options(force));
            EXPECT_GT(continuation.diagnostics().correctors, 0);
            const auto earlier = continuation.preview(dt);
            expect_rejected_stored_node(restored, continuation, first, earlier);
            continuation.commit(earlier);

            Fixture uninterrupted(geometry, false);
            expect_archived(uninterrupted, next, force);
            Coupler expected(uninterrupted.flow, uninterrupted.controller_options(force));
            expect_same_endpoint(*restored.flow, *uninterrupted.flow);
            expect_same_diagnostics(continuation.diagnostics(), expected.diagnostics());
            if (!force)
                expect_rejected_configuration(geometry, first);
        }
}

TEST(LowMachQbmmCylindricalRestartTest, LegacyFormats)
{
    for (const auto force : accelerations)
    {
        Fixture old(Cartesian, true);
        Coupler controller(old.flow, old.controller_options(force));
        const auto image = controller.serialize_qbmm_restart();
        EXPECT_EQ(word(image, 1), force ? 9u : 7u);
        if (force)
            EXPECT_EQ(word(image, 2), 3u);
        Fixture old_receiver(Cartesian, false);
        old_receiver.flow->restore_qbmm_restart(image);
        expect_same_endpoint(*old.flow, *old_receiver.flow);
        Fixture axial(Sector, true);
        EXPECT_THROW(axial.flow->restore_qbmm_restart(image), std::exception);
        Coupler axial_controller(axial.flow, axial.controller_options(force));
        const auto axial_image = axial_controller.serialize_qbmm_restart();
        EXPECT_EQ(word(axial_image, 2), 5u);
        EXPECT_THROW(old_receiver.flow->restore_qbmm_restart(axial_image), std::exception);
    }
}

TEST(LowMachQbmmCylindricalRestartTest, CanonicalizesRoundedArchiveDriftButRejectsInconsistentMomentum)
{
    Fixture original(Sector, true);
    Coupler controller(original.flow, original.controller_options(std::nullopt));
    controller.advance(dt);
    const auto image = controller.serialize_qbmm_restart();
    ASSERT_EQ(word(image, 1), 9u);
    ASSERT_EQ(word(image, 2), 5u);
    const auto comm = original.grid->owned_cell_map()->getComm();
    const int changed_rank = reduce(*original.grid, Teuchos::REDUCE_MIN,
        original.grid->num_owned_cells() ? comm->getRank() : comm->getSize());
    const auto revise_drift = [&](auto revision)
    {
        auto copy = image;
        copy.resize(copy.size() - 16);
        if (comm->getRank() == changed_rank)
        {
            const size_t drift_z = first_axial_node(copy) + 8; // Two node vectors, then drift x, y, z.
            const double drift = std::bit_cast<double>(word(copy, drift_z));
            EXPECT_EQ(drift, original.flow->state().cells.front().qbmm_kinetic_closure[1]->drift_momentum.z);
            EXPECT_GT(drift, 0.);
            set_word(copy, drift_z, std::bit_cast<std::uint64_t>(revision(drift)));
        }
        Restart::append_restart_checksums(*comm, copy);
        return copy;
    };

    Fixture receiver(Sector, false);
    receiver.reject_law_calls = true;
    const auto rounded = revise_drift([](double drift) { return std::nextafter(drift, 2 * drift); });
    ASSERT_NO_THROW(receiver.flow->restore_qbmm_restart(rounded));
    expect_same_endpoint(*receiver.flow, *original.flow);
    const auto baseline = receiver.flow->serialize_qbmm_restart();
    const auto inconsistent = revise_drift([](double drift) { return 2 * drift; });
    try
    {
        receiver.flow->restore_qbmm_restart(inconsistent);
        ADD_FAILURE() << "A retained drift mismatch outside node-momentum roundoff was accepted.";
    }
    catch (const std::exception& error)
    {
        if (comm->getRank() == changed_rank)
            EXPECT_NE(std::string_view(error.what()).find("archive drift momentum"), std::string_view::npos);
    }
    EXPECT_EQ(receiver.flow->serialize_qbmm_restart(), baseline);
    EXPECT_EQ(receiver.node_calls, 0);
}

TEST(LowMachQbmmCylindricalRestartTest, UnforcedCoupledHistoryRequiresTheMatchingController)
{
    Fixture fixture(Sector, true);
    Coupler::Diagnostics accepted_diagnostics;
    {
        Coupler first(fixture.flow, fixture.controller_options(std::nullopt));
        accepted_diagnostics = first.advance(dt);
    }
    ASSERT_GT(accepted_diagnostics.correctors, 0);
    const auto before = fixture.flow->serialize_qbmm_restart();
    const auto calls = fixture.node_calls;
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    auto mismatch = fixture.controller_options(std::nullopt);
    if (comm->getRank() == 0)
        mismatch.source_identity += " changed";
    EXPECT_THROW(Coupler(fixture.flow, mismatch), std::exception);
    EXPECT_THROW((void) fixture.flow->preview(dt), std::exception);
    EXPECT_EQ(fixture.node_calls, calls);
    EXPECT_EQ(fixture.flow->serialize_qbmm_restart(), before);

    Coupler matching(fixture.flow, fixture.controller_options(std::nullopt));
    EXPECT_EQ(fixture.node_calls, calls);
    expect_same_diagnostics(matching.diagnostics(), accepted_diagnostics);
    const auto report = matching.qbmm_report();
    EXPECT_TRUE(report.provenance.controller_configuration_signature.has_value());
    EXPECT_EQ(report.receipt_history, Flow::QbmmReceiptHistory::Available);
    EXPECT_TRUE(report.interval);
    matching.advance(dt);
    EXPECT_EQ(fixture.flow->state().time, 2 * dt);
}

TEST(LowMachQbmmCylindricalRestartTest, AttachedCheckpointPreservesStandaloneInterval)
{
    Fixture detached(Sector, true);
    detached.flow->advance(dt);
    const auto standalone_image = detached.flow->serialize_qbmm_restart();
    {
        Coupler signed_history(detached.flow, detached.controller_options(std::nullopt));
        signed_history.advance(dt);
    }
    const auto detached_image = detached.flow->serialize_qbmm_restart();
    EXPECT_THROW(detached.flow->restore_qbmm_restart(standalone_image), std::exception);
    EXPECT_EQ(detached.flow->serialize_qbmm_restart(), detached_image);

    Fixture fixture(Sector, true);
    fixture.flow->advance(dt);
    Coupler attached(fixture.flow, fixture.controller_options(std::nullopt));
    const auto checkpoint = attached.checkpoint();

    attached.advance(dt);
    const auto coupled_image = attached.serialize_qbmm_restart();

    Fixture incompatible(Sector, true);
    auto different_source = incompatible.controller_options(std::nullopt);
    different_source.source_identity += " changed";
    Coupler different(incompatible.flow, different_source);
    different.advance(dt);
    EXPECT_THROW(attached.restore_qbmm_restart(different.serialize_qbmm_restart()), std::exception);
    EXPECT_EQ(attached.serialize_qbmm_restart(), coupled_image);

    Fixture expected(Sector, false);
    expected.flow->restore_qbmm_restart(coupled_image);
    const auto before_rollback = fixture.node_calls;
    attached.restore_qbmm_restart(standalone_image);
    const auto rolled_back = attached.qbmm_report();
    EXPECT_EQ(fixture.node_calls, before_rollback);
    EXPECT_EQ(rolled_back.provenance.origin, Flow::QbmmStateOrigin::PortableRestart);
    EXPECT_EQ(rolled_back.provenance.time, dt);
    EXPECT_FALSE(rolled_back.provenance.controller_configuration_signature);
    EXPECT_EQ(rolled_back.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    attached.advance(dt);
    expect_same_endpoint(*fixture.flow, *expected.flow);

    const auto calls = fixture.node_calls;
    attached.restore(checkpoint);
    const auto report = attached.qbmm_report();
    EXPECT_EQ(report.provenance.origin, Flow::QbmmStateOrigin::Checkpoint);
    EXPECT_EQ(report.provenance.time, dt);
    EXPECT_FALSE(report.provenance.controller_configuration_signature);
    EXPECT_EQ(report.receipt_history, Flow::QbmmReceiptHistory::Available);
    EXPECT_TRUE(report.interval);
    const auto recoded = attached.serialize_qbmm_restart();
    EXPECT_EQ(word(recoded, 2), 5u);
    attached.restore_qbmm_restart(recoded);
    EXPECT_EQ(fixture.node_calls, calls);
    const auto portable = attached.qbmm_report();
    EXPECT_FALSE(portable.provenance.controller_configuration_signature);
    EXPECT_EQ(portable.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    attached.advance(dt);
    expect_same_endpoint(*fixture.flow, *expected.flow);
}
} // namespace
