/** Native kinetic integration, physical ledgers, and rollback across actual empty owners. */
#include "QbmmKineticTestSupport.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmKinetic;
using Range = R::QbmmRangeVariable;
using Gate = R::QbmmApplicabilityGate;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct FlowFixture
{
    static constexpr double beta = 1. / 128., dt = .25, pressure = 600., temperature = 300., e = -10.;
    SP<Flow::Mesh> grid = mesh({0., 1., 2.});
    std::shared_ptr<const R::SpeciesCatalog> species = catalog();
    R::ThermodynamicBounds bounds{200., 500., 200., 1000.};
    std::shared_ptr<const R::MixtureThermodynamics> gas = R::make_ideal_gas_mixture(
        species, {{2., temperature, e}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
    bool fail_owner = false, disable_law = false;
    int node_calls = 0, injected_failures = 0;
    Flow::Options opts;
    Flow flow;

    FlowFixture(double cfl = 1., int cap = 1024)
        : opts(options(species,
              [this](const R::QbmmNodeSlipInput& in)
              {
                  ++node_calls;
                  if (disable_law)
                      throw std::runtime_error("the node law is disabled after preview");
                  const bool left = in.location.value().centroid[0] < 1.;
                  if (fail_owner && left)
                  {
                      ++injected_failures;
                      throw std::runtime_error("injected owned left-node failure");
                  }
                  EXPECT_GT(in.diameter_m, 0.);
                  EXPECT_GT(in.count, 0.);
                  EXPECT_GT(in.phase_volume_m3, 0.);
                  EXPECT_DOUBLE_EQ(in.bulk_volume_m3, 1.);
                  EXPECT_EQ(in.catalog_fingerprint, species->fingerprint());
                  EXPECT_EQ(in.family_fingerprint, opts.qbmm_families[0]->fingerprint());
                  EXPECT_EQ(in.population, "kinetic bubbles");
                  EXPECT_EQ(in.location->geometryEpoch, grid->geometry_epoch());
                  return std::array{(left ? beta : -beta) * in.mass_kg_per_bubble, 0., 0.};
              },
              bounds, cfl, cap, Gate::ReportOnly,
              {{Range::NodeMassKgPerBubble, .5, 1.5, {}}, {Range::TimeStepS, .1, .5, {}}})),
          flow(grid, {gas}, opts)
    {
        flow.initialize(make_state());
    }

    Flow::State make_state() const
    {
        Flow::State state;
        state.thermodynamic_pressure = pressure;
        state.qbmm_families = {Flow::qbmm_family_stamp(*opts.qbmm_families[0])};
        state.cells.resize(grid->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const bool left = grid->cell_centroid(static_cast<LO>(c)).x < 1.;
            set_stock(state.cells[c], gas,
                left ? R::QbmmMomentBlock{2., 2., 2., 2.} : R::QbmmMomentBlock{1., 2., 4., 8.}, pressure, temperature,
                e);
            state.cells[c].velocity.y = left ? .02 : -.01;
        }
        return state;
    }
};

TEST(LowMachQbmmKineticTest, NativeCorrectorClosesOwnerOrientedTransportWorkAndPhysicalMomentum)
{
    FlowFixture f;
    record_ownership(*f.grid, 2);
    const auto accepted = f.flow.state();
    const auto trial = f.flow.preview(f.dt);
    const int calls = f.node_calls;
    f.disable_law = true;
    const auto report = f.flow.qbmm_report(trial);
    EXPECT_EQ(f.node_calls, calls);
    ASSERT_EQ(reduce(*f.grid, Teuchos::REDUCE_MIN, int(bool(report.interval))), 1);
    EXPECT_EQ(report.family_transport, Model::FamilyTransport::KineticEuler);
    EXPECT_EQ(report.kinetic_substeps, 1);
    EXPECT_EQ(report.interval->kinetic_substeps, 1);
    EXPECT_GT(report.maximum_kinetic_cfl, 0.);
    EXPECT_LE(report.maximum_kinetic_cfl, f.opts.qbmm_model.kinetic_cfl);
    EXPECT_EQ(report.interval->local_face_transport,
        (std::vector<Flow::PhaseFaceTransport>{
            trial.owned_phase_transport().begin(), trial.owned_phase_transport().end()}));
    for (const auto& receipt : report.interval->local_kinetic_law_receipts)
    {
        EXPECT_GE(receipt.duration, .1);
        EXPECT_LE(receipt.duration, .5);
        EXPECT_TRUE(receipt.applicability.gate_passed);
    }

    // There is one interior face. Reduce its positive-x rates so each incident owner can independently check its row.
    std::array<double, 7> packet{}, global_packet{};
    int local_faces = 0, local_partition_faces = 0;
    double max_pressure_split_error = 0.;
    const auto& carrier = f.flow.volume_flux();
    for (const auto face : carrier.owned_face_ids())
    {
        const auto& p = trial.owned_phase_transport()[carrier.owned_row(face)];
        const bool has_packet = p.qbmm_flux.size() == 1 && p.qbmm_flux[0] && p.qbmm_flux[0]->kinetic;
        EXPECT_TRUE(has_packet);
        if (!has_packet)
            continue;
        const auto& gas = *p.qbmm_flux[0];
        EXPECT_EQ(p.phase_volume_flux[0], 0.);
        EXPECT_EQ(p.relative_phase_volume_flux[0], 0.);
        if (!f.grid->is_interior_face(face))
        {
            EXPECT_EQ(gas.material_volume_flux, 0.);
            EXPECT_EQ(p.momentum_mass_flux, 0.);
            continue;
        }
        ++local_faces;
        if (!f.grid->is_owned_cell(f.grid->owner_cell(face)) || !f.grid->is_owned_cell(f.grid->neighbor_cell(face)))
            ++local_partition_faces;
        const auto& k = *gas.kinetic;
        EXPECT_GT(k.forward.mass_flux, 0.);
        EXPECT_LT(k.backward.mass_flux, 0.);
        EXPECT_GT(p.theta, 1.);
        EXPECT_DOUBLE_EQ(p.theta, k.theta);
        EXPECT_DOUBLE_EQ(p.slip_volume_flux, k.slip_volume_flux);
        EXPECT_DOUBLE_EQ(p.forward_mass_flux, k.forward.mass_flux);
        EXPECT_DOUBLE_EQ(p.backward_mass_flux, -k.backward.mass_flux);
        EXPECT_DOUBLE_EQ(p.symmetric_mass_flux, std::min(k.forward.mass_flux, -k.backward.mass_flux));
        EXPECT_NEAR(p.momentum_mass_flux, k.forward.mass_flux + k.backward.mass_flux, 2e-14);
        EXPECT_NEAR(p.drift_momentum_flux.x, k.forward.drift_momentum_flux.x + k.backward.drift_momentum_flux.x, 2e-14);
        const double implicit_tangential_flux =
            p.forward_mass_flux * p.momentum_owner_velocity.y - p.backward_mass_flux * p.momentum_neighbor_velocity.y;
        EXPECT_NEAR(p.implicit_carrier_momentum_flux.y, implicit_tangential_flux, 2e-14);
        EXPECT_GT(std::abs(p.implicit_carrier_momentum_flux.y), 1e-5);
        max_pressure_split_error =
            std::abs(gas.material_volume_flux -
                     (p.theta * p.carrier_volume_flux + p.slip_volume_flux + p.pressure_rounding_flux));
        EXPECT_LE(max_pressure_split_error, 2e-14);
        const double sign = f.grid->face_area_vector(face).x;
        packet = {sign * gas.population_number_flux, sign * gas.species_molar_flux[0], sign * gas.shape_flux.H2,
            sign * gas.shape_flux.H3, sign * gas.internal_energy_flux, sign * gas.material_volume_flux,
            sign * p.implicit_carrier_momentum_flux.y};
    }
    const auto comm = f.grid->owned_cell_map()->getComm();
    EXPECT_EQ(reduce(*f.grid, Teuchos::REDUCE_SUM, local_faces), 1);
    const int partition_faces = reduce(*f.grid, Teuchos::REDUCE_SUM, local_partition_faces);
    RecordProperty("partition_faces", partition_faces);
    EXPECT_EQ(partition_faces, int(comm->getSize() > 1));
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_SUM, static_cast<int>(packet.size()), packet.data(), global_packet.data());

    double local_work = 0., local_energy_change = 0., local_stock_error = 0., local_drift_error = 0.,
           local_tangential_error = 0.;
    std::array<double, 3> local_momentum_change{}, global_momentum_change{};
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& old = accepted.cells[c];
        const auto& state = trial.state().cells[c];
        const auto& transported = trial.transport_snapshot().cells[c].phases[0];
        const auto& output = report.cells[c];
        const auto& family = *output.families[0];
        const bool left = f.grid->cell_centroid(static_cast<LO>(c)).x < 1.;
        const double inward = left ? -1. : 1.;
        const R::QbmmMomentBlock pre_source{*transported.population_number, transported.moles[0],
            transported.qbmm_shape->H2, transported.qbmm_shape->H3};
        const auto old_moments = stock(old);
        for (size_t k = 0; k < old_moments.size(); ++k)
        {
            const double expected = old_moments[k] + inward * f.dt * global_packet[k];
            EXPECT_NEAR(pre_source[k], expected, 2e-13);
            EXPECT_NEAR(family.moments[k], expected, 2e-13);
            local_stock_error = std::max(local_stock_error, std::abs(family.moments[k] - expected));
        }
        EXPECT_NEAR(
            transported.internal_energy, old.phases[0].internal_energy + inward * f.dt * global_packet[4], 2e-13);
        const double advected_volume = 1. + inward * f.dt * global_packet[5];
        EXPECT_NEAR(transported.advected_material_volume, advected_volume, 2e-13);
        const auto& phase = state.phases[0];
        const double final_volume = phase.moles[0] * phase.temperature / phase.absolute_pressure;
        EXPECT_NEAR(final_volume, 1., 2e-9);
        const double work = -phase.absolute_pressure * (final_volume - advected_volume);
        EXPECT_NEAR(phase.internal_energy - transported.internal_energy, work, 2e-9);
        local_work += work;
        local_energy_change += phase.internal_energy - old.phases[0].internal_energy;

        const double drift = (left ? f.beta : -f.beta) * family.moments[2];
        const double old_drift = (left ? f.beta : -f.beta) * old_moments[2];
        EXPECT_TRUE(family.kinetic_closure);
        if (family.kinetic_closure)
        {
            const double actual = family.kinetic_closure->drift_momentum.x;
            EXPECT_NEAR(actual, drift, 2e-13);
            local_drift_error = std::max(local_drift_error, std::abs(actual - drift));
        }
        const double mass = phase.moles[0];
        EXPECT_NEAR(output.barycentric_velocity.x, state.velocity.x + drift / mass, 2e-13);
        // This Cartesian row has no y pressure or drift impulse. Only the solved carrier face flux changes M*u_y.
        const double tangential_error =
            std::abs(mass * state.velocity.y - old_moments[1] * old.velocity.y - inward * f.dt * global_packet[6]);
        EXPECT_LE(tangential_error, 2e-11);
        local_tangential_error = std::max(local_tangential_error, tangential_error);
        for (size_t axis = 0; axis < local_momentum_change.size(); ++axis)
        {
            const double old_physical_momentum =
                old_moments[1] * old.velocity.component(axis) + (axis == 0 ? old_drift : 0.);
            const double physical_momentum = mass * state.velocity.component(axis) + (axis == 0 ? drift : 0.);
            local_momentum_change[axis] += physical_momentum - old_physical_momentum;
        }
    }
    const double work = reduce(*f.grid, Teuchos::REDUCE_SUM, local_work);
    const double energy_change = reduce(*f.grid, Teuchos::REDUCE_SUM, local_energy_change);
    EXPECT_NEAR(work, trial.diagnostics().pressure_work, 2e-9);
    EXPECT_NEAR(energy_change, work, 2e-9);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local_momentum_change.data(), global_momentum_change.data());
    double momentum_error = 0.;
    for (size_t axis = 0; axis < global_momentum_change.size(); ++axis)
    {
        EXPECT_DOUBLE_EQ(trial.diagnostics().external_impulse.component(axis), 0.);
        const double error =
            std::abs(global_momentum_change[axis] - trial.diagnostics().pressure_impulse.component(axis));
        EXPECT_LE(error, 2e-10);
        momentum_error = std::max(momentum_error, error);
    }
    record_maximum(*f.grid, "maximum_cell_stock_error", local_stock_error);
    record_maximum(*f.grid, "maximum_pressure_split_error_m3_per_s", max_pressure_split_error);
    record_maximum(*f.grid, "maximum_drift_momentum_error_kg_m_per_s", local_drift_error);
    record_maximum(*f.grid, "maximum_cell_tangential_momentum_error_kg_m_per_s", local_tangential_error);
    record_maximum(*f.grid, "momentum_ledger_error_kg_m_per_s", momentum_error);
    record_maximum(*f.grid, "energy_ledger_error_j", std::abs(energy_change - work));
    record_maximum(*f.grid, "maximum_continuity_residual_m3_per_s", trial.diagnostics().maximum_continuity_residual);
    f.flow.commit(trial);
    const auto committed = f.flow.qbmm_report();
    EXPECT_EQ(f.node_calls, calls);
    EXPECT_EQ(committed.kinetic_substeps, 1);
    EXPECT_EQ(committed.provenance.time, f.dt);
    ASSERT_TRUE(committed.interval);
    EXPECT_EQ(committed.interval->preview_id, report.interval->preview_id);
    EXPECT_EQ(committed.interval->local_face_transport, report.interval->local_face_transport);
}

TEST(LowMachQbmmKineticTest, AllFaceCflSubcyclesAndPublishesTheExactClosedIntervalWithoutCallbacks)
{
    const auto grid = mesh({0., 1., 2.}, {0., 1., 2.});
    record_ownership(*grid, 4);
    const auto comm = grid->owned_cell_map()->getComm();
    const auto species = catalog();
    constexpr double pressure = 4., temperature = 2., energy = 3., beta = .125, cap = .125, dt = .375;
    const R::ThermodynamicBounds bounds{1., 4., 2., 8.};
    const auto gas = R::make_ideal_gas_mixture(
        species, {{2., temperature, energy}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
    size_t calls = 0;
    bool disable_law = false;
    const auto opts = options(species,
        [&](const R::QbmmNodeSlipInput& in)
        {
            ++calls;
            if (disable_law)
                throw std::runtime_error("Publication invoked the disabled node law.");
            const auto& x = in.location.value().centroid;
            const double speed = beta * in.mass_kg_per_bubble;
            return std::array{x[0] < 1. ? speed : -speed, x[1] < 1. ? speed : -speed, 0.};
        },
        bounds, cap, 2, Gate::RejectOutside, {{Range::TimeStepS, .1, .3, {}}});
    Flow flow(grid, {gas}, opts);
    Flow::State state;
    state.thermodynamic_pressure = pressure;
    state.qbmm_families = {Flow::qbmm_family_stamp(*opts.qbmm_families[0])};
    state.cells.resize(grid->num_owned_cells());
    const R::QbmmMomentBlock initial{1.5, 2., 3., 5.};
    for (auto& cell : state.cells)
        set_stock(cell, gas, initial, pressure, temperature, energy);
    flow.initialize(std::move(state));
    const auto trial = flow.preview(dt);
    const size_t preview_calls = calls;
    disable_law = true;
    const auto report = flow.qbmm_report(trial);
    EXPECT_EQ(calls, preview_calls);
    ASSERT_EQ(reduce(*grid, Teuchos::REDUCE_MIN, int(bool(report.interval))), 1);
    EXPECT_EQ(report.kinetic_substeps, 2);
    EXPECT_EQ(report.interval->kinetic_substeps, 2);
    EXPECT_EQ(report.interval->time_step, dt);
    EXPECT_EQ(trial.state().time, dt);
    EXPECT_LE(report.maximum_kinetic_cfl, cap);
    EXPECT_NEAR(report.maximum_kinetic_cfl, cap, 2e-15);

    Flow::Traits::scalar_face_type carrier(grid, 0., "retained_subcycle_carrier_rate");
    int local_partition_faces = 0;
    double local_face_error = 0.;
    constexpr R::QbmmMomentBlock expected_flux{.25, .375, .625, 1.125};
    for (const auto face : carrier.owned_face_ids())
    {
        const auto& p = trial.owned_phase_transport()[carrier.owned_row(face)];
        carrier.set_owned_value(face, p.carrier_volume_flux);
        if (!grid->is_interior_face(face))
            continue;
        if (!grid->is_owned_cell(grid->owner_cell(face)) || !grid->is_owned_cell(grid->neighbor_cell(face)))
            ++local_partition_faces;
        EXPECT_NEAR(p.carrier_volume_flux, 0., 2e-14);
        const bool has_packet = p.qbmm_flux.size() == 1 && p.qbmm_flux[0] && p.qbmm_flux[0]->kinetic;
        EXPECT_TRUE(has_packet);
        if (!has_packet)
            continue;
        const auto& k = *p.qbmm_flux[0]->kinetic;
        for (size_t i = 0; i < expected_flux.size(); ++i)
        {
            const double forward = std::abs(k.forward.moment_flux[i] - expected_flux[i]);
            const double backward = std::abs(k.backward.moment_flux[i] + expected_flux[i]);
            EXPECT_LE(forward, 2e-13);
            EXPECT_LE(backward, 2e-13);
            local_face_error = std::max({local_face_error, forward, backward});
        }
        EXPECT_NEAR(k.forward.integrated_count, dt * expected_flux[0], 2e-13);
        EXPECT_NEAR(k.backward.integrated_count, -dt * expected_flux[0], 2e-13);
    }
    carrier.sync_ghosts();
    const int partition_faces = reduce(*grid, Teuchos::REDUCE_SUM, local_partition_faces);
    if (comm->getSize() > 1)
        EXPECT_GT(partition_faces, 0);
    RecordProperty("partition_faces", partition_faces);

    std::array<double, 2> shortest{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    std::array<double, 2> longest{};
    std::array<int, 2> receipts{};
    int local_two_face_atoms = 0;
    double local_cfl = 0.;
    for (const auto& r : report.interval->local_kinetic_law_receipts)
    {
        EXPECT_TRUE(r.applicability.gate_passed);
        EXPECT_LT(r.substep, shortest.size());
        if (r.substep >= shortest.size())
            continue;
        ++receipts[r.substep];
        shortest[r.substep] = std::min(shortest[r.substep], r.duration);
        longest[r.substep] = std::max(longest[r.substep], r.duration);
        for (size_t c = 0; c < grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            if (grid->cell_geometry_global_id(cell) != r.geometry_cell_id)
                continue;
            double fraction = 0.;
            int outgoing_faces = 0;
            for (const auto face : grid->faces(cell))
                if (grid->is_interior_face(face))
                {
                    const double sign = grid->owner_cell(face) == cell ? 1. : -1.;
                    const double rate =
                        sign * (carrier.local_value(face) + r.slip_velocity.dot(grid->face_area_vector(face)));
                    if (rate > 0.)
                    {
                        ++outgoing_faces;
                        fraction += r.duration * rate / grid->cell_volume(cell);
                    }
                }
            EXPECT_EQ(outgoing_faces, 2);
            local_two_face_atoms += int(outgoing_faces == 2);
            local_cfl = std::max(local_cfl, fraction);
        }
    }
    std::array<double, 2> durations{};
    for (size_t i = 0; i < shortest.size(); ++i)
    {
        const double duration = reduce(*grid, Teuchos::REDUCE_MAX, longest[i]);
        EXPECT_EQ(reduce(*grid, Teuchos::REDUCE_MIN, shortest[i]), duration);
        EXPECT_EQ(reduce(*grid, Teuchos::REDUCE_SUM, receipts[i]), 8);
        EXPECT_NEAR(duration, i == 0 ? .25 : .125, 2e-13);
        durations[i] = duration;
    }
    // The first substep is at least half the interval, so binary64 subtraction gives the exact remainder.
    EXPECT_GE(durations[0], dt / 2.);
    EXPECT_LE(durations[0], dt);
    EXPECT_EQ(durations[1], dt - durations[0]);
    EXPECT_EQ(reduce(*grid, Teuchos::REDUCE_SUM, local_two_face_atoms), 16);
    EXPECT_NEAR(reduce(*grid, Teuchos::REDUCE_MAX, local_cfl), report.maximum_kinetic_cfl, 2e-15);
    for (const auto& cell : report.cells)
        for (size_t k = 0; k < initial.size(); ++k)
            EXPECT_NEAR(cell.families[0]->moments[k], initial[k], 2e-13);
    record_maximum(*grid, "measured_all_face_cfl", local_cfl);
    record_maximum(*grid, "maximum_subcycled_face_moment_rate_error", local_face_error);

    flow.commit(trial);
    const auto published = flow.qbmm_report();
    EXPECT_EQ(calls, preview_calls);
    EXPECT_EQ(published.provenance.time, dt);
    EXPECT_EQ(published.maximum_kinetic_cfl, report.maximum_kinetic_cfl);
    ASSERT_TRUE(published.interval);
    EXPECT_EQ(published.interval->preview_id, report.interval->preview_id);
    EXPECT_EQ(published.interval->local_face_transport, report.interval->local_face_transport);
    EXPECT_EQ(
        published.interval->local_kinetic_law_receipts.size(), report.interval->local_kinetic_law_receipts.size());
}

TEST(LowMachQbmmKineticTest, CapAndSingleOwnerLawFailurePreserveAnEarlierCommittableTrial)
{
    FlowFixture f(.01, 1);
    record_ownership(*f.grid, 2);
    const auto earlier = f.flow.preview(f.dt);
    const auto before = f.flow.serialize_qbmm_restart();
    const auto report = f.flow.qbmm_report(earlier);
    ASSERT_EQ(reduce(*f.grid, Teuchos::REDUCE_MIN, int(bool(report.interval))), 1);
    const auto retained = report.interval->local_face_transport;
    std::string cap_error;
    try
    {
        (void) f.flow.preview(4.);
    }
    catch (const std::exception& error)
    {
        cap_error = error.what();
    }
    EXPECT_NE(cap_error.find("exhausted its subcycle cap"), std::string::npos);
    EXPECT_EQ(f.flow.serialize_qbmm_restart(), before);
    f.fail_owner = true;
    EXPECT_THROW((void) f.flow.preview(f.dt), std::exception);
    EXPECT_EQ(reduce(*f.grid, Teuchos::REDUCE_SUM, f.injected_failures), 1);
    EXPECT_EQ(f.flow.serialize_qbmm_restart(), before);
    f.disable_law = true;
    const int calls = f.node_calls;
    const auto after = f.flow.qbmm_report(earlier);
    EXPECT_EQ(after.provenance.preview_id, report.provenance.preview_id);
    ASSERT_EQ(reduce(*f.grid, Teuchos::REDUCE_MIN, int(bool(after.interval))), 1);
    EXPECT_EQ(after.interval->local_face_transport, retained);
    f.flow.commit(earlier);
    EXPECT_EQ(f.node_calls, calls);
    EXPECT_DOUBLE_EQ(f.flow.state().time, f.dt);
}

TEST(LowMachQbmmKineticTest, RejectsUnsupportedModePairsAndUnqualifiedPhysicsWithoutPublishing)
{
    FlowFixture f;
    record_ownership(*f.grid, 2);
    const auto earlier = f.flow.preview(f.dt);
    const auto before = f.flow.serialize_qbmm_restart();
    const int calls = f.node_calls;
    const std::pair<const char*, void (*)(Flow::Options&)> requests[]{
        {"implicit integrator with node slip",
            [](auto& opt) { opt.qbmm_model.family_transport = Model::FamilyTransport::NativeImplicit; }},
        {"kinetic integrator with phase slip",
            [](auto& opt) { opt.qbmm_model.velocity = Model::Velocity::SuppliedPhaseSlip; }},
        {"kinetic integrator without law", [](auto& opt) { opt.qbmm_node_slip_laws.clear(); }},
        {"nonpositive CFL", [](auto& opt) { opt.qbmm_model.kinetic_cfl = 0.; }},
        {"superunit CFL", [](auto& opt) { opt.qbmm_model.kinetic_cfl = 1.001; }},
        {"no substeps", [](auto& opt) { opt.qbmm_model.maximum_kinetic_substeps = 0; }},
        {"SSPRK2", [](auto& opt) { opt.qbmm_model.family_transport = Model::FamilyTransport::RealizableLinearSSPRK2; }},
        {"BDF2", [](auto& opt) { opt.qbmm_model.transport.time = FVM::ScalarTimeScheme::BDF2; }},
        {"higher-order scalar reconstruction",
            [](auto& opt) { opt.qbmm_model.transport.convection = FVM::ScalarConvectionScheme::BoundedLinearUpwind; }},
        {"steady assembly", [](auto& opt) { opt.qbmm_model.transport.mode = FVM::AssemblyMode::Steady; }},
        {"separate phase temperatures",
            [](auto& opt) { opt.thermal_mode = R::PhaseThermalMode::SeparatePhaseTemperatures; }},
        {"independent node temperatures", [](auto& opt) { opt.qbmm_model.independent_node_temperatures = true; }},
        {"capillary pressure", [](auto& opt) { opt.qbmm_model.capillarity = true; }},
        {"bulk-gas pooling", [](auto& opt) { opt.qbmm_model.bulk_gas_conversion = true; }},
        {"cross-family coalescence", [](auto& opt) { opt.qbmm_model.cross_family_coalescence = true; }},
        {"variable family composition", [](auto& opt) { opt.qbmm_model.variable_composition = true; }}};
    for (const auto& [name, change] : requests)
    {
        SCOPED_TRACE(name);
        auto invalid = f.opts;
        change(invalid);
        EXPECT_THROW((void) Flow(f.grid, {f.gas}, invalid), std::exception);
    }
    const auto comm = f.grid->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
    {
        auto divergent = f.opts;
        if (comm->getRank() == comm->getSize() - 1)
            divergent.qbmm_model.maximum_kinetic_substeps -= 1;
        EXPECT_THROW((void) Flow(f.grid, {f.gas}, divergent), std::exception);
    }
    auto periodic_grid = std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cartesian>(
            Vec3D<ArrReal>{{{0., 1., 2.}, {0., 1.}, {0., 1.}}}, Vec3D<bool>{true, false, false}),
        Flow::Mesh::DistributionOptions{.allow_empty_partitions = true});
    EXPECT_THROW((void) Flow(periodic_grid, {f.gas}, f.opts), std::exception);

    for (const double drift : {0., f.beta})
    {
        auto supplied = f.make_state();
        for (auto& cell : supplied.cells)
            cell.drift_velocity = {Flow::Vector{drift, 0., 0.}};
        EXPECT_THROW(f.flow.initialize(std::move(supplied)), std::exception);
    }

    Flow::PlanarMotion motion(f.grid);
    const auto epoch = f.grid->geometry_epoch();
    EXPECT_THROW((void) f.flow.preview_ale(f.dt, motion, 1.001), std::exception);
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(f.grid->geometry_epoch(), epoch);
    for (const double acceleration : {0., -1.})
    {
        const auto gravity =
            std::make_shared<const Flow::SourceStageProvider>(Flow::SourceStageProvider{.identity = "supplied gravity",
                .revision = 1,
                .schema_fingerprint = f.species->fingerprint(),
                .evaluate = [](const auto&) -> Flow::SourceStageResult
                {
                    ADD_FAILURE() << "Gravity preflight must reject before evaluating the source stage.";
                    return {};
                },
                .body_acceleration = Flow::Vector{0., 0., acceleration}});
        try
        {
            (void) f.flow.preview_with_sources(f.dt, gravity);
            ADD_FAILURE() << "Supplied gravity must reject on the kinetic path.";
        }
        catch (const std::exception& error)
        {
            EXPECT_NE(std::string(error.what()).find("body acceleration disabled"), std::string::npos);
        }
    }
    EXPECT_EQ(f.flow.serialize_qbmm_restart(), before);
    EXPECT_EQ(f.node_calls, calls);
    f.flow.commit(earlier);
    EXPECT_DOUBLE_EQ(f.flow.state().time, f.dt);
}

TEST(LowMachQbmmKineticTest, SingleOwnerCompressionFailurePreservesAnEarlierCommittableTrial)
{
    const auto grid = mesh({0., 1., 2.});
    record_ownership(*grid, 2);
    const auto species = catalog();
    const R::ThermodynamicBounds bounds{1., 4., 2., 8.};
    const auto gas =
        R::make_ideal_gas_mixture(species, {{2., 2., 3.}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
    auto opts = options(
        species, [](const R::QbmmNodeSlipInput& in)
        { return std::array{in.location.value().centroid[0] < 1. ? 0x1p-26 : 0., 0., 0.}; }, bounds);
    opts.phase_roles.push_back({Flow::PhaseRole::BulkGas, {}});
    opts.qbmm_families.push_back(nullptr);
    opts.qbmm_node_slip_laws.push_back(nullptr);
    constexpr double p = 4., temperature = 2., e = 3.;
    Flow flow(grid, {gas, gas}, opts);
    Flow::State state;
    state.thermodynamic_pressure = p;
    state.qbmm_families = {Flow::qbmm_family_stamp(*opts.qbmm_families[0]), std::nullopt};
    state.cells.resize(grid->num_owned_cells());
    bool owns_receiver = false;
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const bool donor = grid->cell_centroid(static_cast<LO>(c)).x < 1.;
        owns_receiver |= !donor;
        auto& cell = state.cells[c];
        const auto h =
            donor ? R::QbmmMomentBlock{0x1p-48, 0x1p-36, 0x1p-24, 0x1p-12} : R::QbmmMomentBlock{1., 1., 1., 1.};
        set_stock(cell, gas, h, p, temperature, e);
        const double bulk_mass = 2. - h[1];
        cell.phases.push_back({gas, {bulk_mass}, p, bulk_mass * e, temperature});
        cell.phase_numbers.push_back(std::nullopt);
        cell.qbmm_shape.push_back(std::nullopt);
    }
    flow.initialize(state);
    const auto accepted = flow.serialize_qbmm_restart();
    const auto earlier = flow.preview(0x1p-12);
    const auto original = flow.qbmm_report(earlier);

    // At dt=1 the first substep adds 2^-74, 2^-62, 2^-50 and 2^-38
    // to the respective unit moments in the receiver. Its variance has
    // insufficient relative resolution while H3 excludes a singleton.
    EXPECT_EQ(R::invert_qbmm_moments({1., 1., 1. + 0x1p-50, 1. + 0x1p-38}).status, R::InversionStatus::IllConditioned);
    std::string error;
    try
    {
        (void) flow.preview(1.);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
    }
    EXPECT_EQ(error, owns_receiver ? "QBMM kinetic compression failed at the represented endpoint (IllConditioned)."
                                   : "QBMM kinetic constructive compression and balances failed on another rank.");
    EXPECT_EQ(reduce(*grid, Teuchos::REDUCE_SUM, int(owns_receiver)), 1);
    EXPECT_EQ(flow.serialize_qbmm_restart(), accepted);
    const auto retained = flow.qbmm_report(earlier);
    EXPECT_EQ(retained.provenance.preview_id, original.provenance.preview_id);
    ASSERT_TRUE(retained.interval && original.interval);
    EXPECT_EQ(retained.interval->local_face_transport, original.interval->local_face_transport);
    EXPECT_NO_THROW(flow.commit(earlier));
    EXPECT_EQ(flow.state().time, 0x1p-12);
}

TEST(LowMachQbmmKineticTest, PropertyLawUsesMassWeightedNodeSlipForGateAndReport)
{
    const auto grid = mesh({0., 1.});
    record_ownership(*grid, 1);
    const auto species = catalog();
    constexpr double speed = 1. / 32., pressure = 4., temperature = 2., e = 3.;
    const R::ThermodynamicBounds bounds{1., 4., 2., 8.};
    const auto gas = R::make_ideal_gas_mixture(
        species, {{2., temperature, e}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
    for (const double allowed_speed : {speed, 0.})
    {
        SCOPED_TRACE(allowed_speed);
        auto opts = options(species, [](const auto&) { return std::array{0., speed, 0.}; }, bounds);
        auto versions = opts.qbmm_families[0]->closure_versions();
        versions.laws.push_back({"property conditioned on family mean slip", 1, R::QbmmLawKind::TransportProperty,
            "mass-weighted slip velocity", "Pa.s", Gate::RejectOutside, true,
            {{Range::SlipSpeedMPerS, allowed_speed, allowed_speed, {}}}});
        opts.qbmm_families[0] =
            std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(), "kinetic bubbles",
                opts.phase_roles[0], 1., std::vector<R::QbmmMassFraction>{{47, 1.}}, std::move(versions));
        Flow flow(grid, {gas}, opts);
        Flow::State state;
        state.thermodynamic_pressure = pressure;
        state.qbmm_families = {Flow::qbmm_family_stamp(*opts.qbmm_families[0])};
        state.cells.resize(grid->num_owned_cells());
        for (auto& cell : state.cells)
            set_stock(cell, gas, {1., 2., 4., 8.}, pressure, temperature, e);
        flow.initialize(std::move(state));
        const auto check_report = [speed, allowed_speed](const auto& report)
        {
            for (const auto& cell : report.cells)
            {
                const auto& family = cell.families[0].value();
                ASSERT_TRUE(family.mean_node_slip_m_per_s);
                EXPECT_DOUBLE_EQ(family.mean_node_slip_m_per_s->y, speed);
                const auto& property = family.applicability[1];
                EXPECT_EQ(property.gate_passed, allowed_speed == speed);
                EXPECT_EQ(property.ranges[0].observed_minimum, speed);
                EXPECT_EQ(property.ranges[0].observed_maximum, speed);
            }
        };
        check_report(flow.qbmm_report());
        if (allowed_speed == speed)
        {
            const auto trial = flow.preview(.125);
            check_report(flow.qbmm_report(trial));
            flow.commit(trial);
            check_report(flow.qbmm_report());
        }
        else
        {
            const auto before = flow.serialize_qbmm_restart();
            EXPECT_THROW((void) flow.preview(.125), std::exception);
            EXPECT_EQ(flow.serialize_qbmm_restart(), before);
        }
    }
}

TEST(LowMachQbmmKineticTest, OrdinarySlipAndKineticStreamsShareMomentumDirections)
{
    const auto grid = mesh({0., 1., 2.});
    record_ownership(*grid, 2);
    const auto species = catalog();
    constexpr double beta = 1. / 512., pressure = 4., temperature = 2., e = 3.;
    const R::ThermodynamicBounds bounds{1., 4., 2., 8.};
    const auto gas = R::make_ideal_gas_mixture(
        species, {{2., temperature, e}}, bounds, R::EnergyConvention::IncludesFormationEnergy, 1.);
    auto opts = options(
        species, [](const R::QbmmNodeSlipInput& in)
        { return std::array{in.location.value().centroid[0] < 1. ? beta : -beta, 0., 0.}; }, bounds);
    opts.phase_roles.push_back({Flow::PhaseRole::BulkGas, "rightgoing ordinary gas"});
    opts.phase_roles.push_back({Flow::PhaseRole::BulkGas, "leftgoing ordinary gas"});
    opts.qbmm_families.resize(3);
    opts.qbmm_node_slip_laws.resize(3);
    Flow flow(grid, {gas, gas, gas}, opts);
    Flow::State state;
    state.thermodynamic_pressure = pressure;
    state.qbmm_families = {Flow::qbmm_family_stamp(*opts.qbmm_families[0]), std::nullopt, std::nullopt};
    state.cells.resize(grid->num_owned_cells());
    constexpr std::array drift{0., beta, -.5 * beta};
    for (auto& cell : state.cells)
    {
        set_stock(cell, gas, {.5, .5, .5, .5}, pressure, temperature, e);
        cell.phases.push_back({gas, {.5}, pressure, .5 * e, temperature});
        cell.phases.push_back({gas, {1.}, pressure, e, temperature});
        cell.phase_numbers.resize(3);
        cell.qbmm_shape.resize(3);
        cell.drift_velocity = {{}, {drift[1], 0., 0.}, {drift[2], 0., 0.}};
    }
    flow.initialize(state);
    const auto trial = flow.preview(.125);
    const auto& carrier = flow.volume_flux();
    const auto& pre_source = trial.transport_snapshot().cells;
    const auto comm = grid->owned_cell_map()->getComm();
    std::array<double, 4> local_moles{}, moles{};
    for (size_t c = 0; c < pre_source.size(); ++c)
    {
        const bool left = grid->cell_centroid(static_cast<LO>(c)).x < 1.;
        for (size_t a = 1; a < drift.size(); ++a)
            local_moles[2 * (a - 1) + !left] = pre_source[c].phases[a].moles[0];
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 4, local_moles.data(), moles.data());
    double pressure_error = 0., momentum_error = 0.;
    int local_partition_faces = 0;
    for (const auto face : carrier.owned_face_ids())
        if (grid->is_interior_face(face))
        {
            if (!grid->is_owned_cell(grid->owner_cell(face)) || !grid->is_owned_cell(grid->neighbor_cell(face)))
                ++local_partition_faces;
            const auto& f = trial.owned_phase_transport()[carrier.owned_row(face)];
            const auto& kinetic = *f.qbmm_flux[0]->kinetic;
            const double normal = grid->face_area_vector(face).x;
            double forward = kinetic.forward.mass_flux, backward = -kinetic.backward.mass_flux;
            double drift_momentum = f.qbmm_flux[0]->derived_drift_momentum_flux.x;
            double theta = kinetic.theta, volume_slip = kinetic.slip_volume_flux;
            EXPECT_GT(forward, 0.);
            EXPECT_GT(backward, 0.);
            for (size_t a = 1; a < drift.size(); ++a)
            {
                EXPECT_FALSE(f.qbmm_flux[a]);
                const double q = f.carrier_volume_flux + normal * drift[a];
                EXPECT_NEAR(f.phase_volume_flux[a], q, 2e-16);
                EXPECT_GT(q * normal * drift[a], 0.);
                const bool uses_right = q * normal < 0.;
                const double ordinary_mass_rate =
                    q * moles[2 * (a - 1) + uses_right]; // Unit bulk volume and molar mass.
                forward += std::max(ordinary_mass_rate, 0.);
                backward += std::max(-ordinary_mass_rate, 0.);
                drift_momentum += ordinary_mass_rate * drift[a];
                theta += f.material_fraction[a];
                volume_slip += normal * drift[a] * f.material_fraction[a];
            }
            EXPECT_GT(forward, kinetic.forward.mass_flux);
            EXPECT_GT(backward, -kinetic.backward.mass_flux);
            EXPECT_NEAR(f.forward_mass_flux, forward, 2e-16);
            EXPECT_NEAR(f.backward_mass_flux, backward, 2e-16);
            EXPECT_NEAR(f.symmetric_mass_flux, std::min(forward, backward), 2e-16);
            EXPECT_NEAR(f.momentum_mass_flux, forward - backward, 2e-16);
            EXPECT_NEAR(f.drift_momentum_flux.x, drift_momentum, 2e-16);
            EXPECT_NEAR(f.theta, theta, 2e-15);
            EXPECT_NEAR(f.slip_volume_flux, volume_slip, 2e-16);
            momentum_error = std::abs(f.drift_momentum_flux.x - drift_momentum);
            pressure_error = std::abs(f.slip_volume_flux - volume_slip);
        }
    const int partition_faces = reduce(*grid, Teuchos::REDUCE_SUM, local_partition_faces);
    RecordProperty("partition_faces", partition_faces);
    EXPECT_EQ(partition_faces, int(comm->getSize() > 1));
    record_maximum(*grid, "ordinary_and_kinetic_drift_rate_error_kg_m_per_s2", momentum_error);
    record_maximum(*grid, "ordinary_and_kinetic_slip_volume_rate_error_m3_per_s", pressure_error);
    const auto before = flow.serialize_qbmm_restart();
    for (auto& cell : state.cells)
        cell.drift_velocity[0].y = beta;
    EXPECT_THROW(flow.initialize(std::move(state)), std::exception);
    EXPECT_EQ(flow.serialize_qbmm_restart(), before);
    flow.commit(trial);
    EXPECT_EQ(flow.state().time, .125);
}
} // namespace
