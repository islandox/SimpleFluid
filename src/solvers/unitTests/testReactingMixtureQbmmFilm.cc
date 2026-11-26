/** Closed finite-liquid film exchange through the reacting solver and its portable checkpoint. */
#include "QbmmFilmTestSupport.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmFilm;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void check_first_step(const Fixture& f, const Coupler::Trial& trial, bool moving)
{
    const auto& diag = trial.diagnostics();
    const auto& families = f.flow->options().qbmm_families;
    const auto report = f.controller->qbmm_report(trial);
    ASSERT_TRUE(report.balances);
    EXPECT_NEAR(report.balances->mass_defect_kg, 0, 1e-13);
    EXPECT_NEAR(report.balances->element_defects_mol.at("X"), 0, 1e-12);
    EXPECT_NEAR(report.balances->element_defects_mol.at("Y"), 0, 1e-12);
    EXPECT_EQ(diag.external_mass, 0);
    EXPECT_NEAR(diag.net_thermal_energy, 0, 1e-7);
    EXPECT_EQ(f.calls, int(f.mesh->num_owned_cells()) * diag.flow.source_evaluations);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& old = f.flow->state().cells[c];
        const auto& cell = trial.state().cells[c];
        const auto& thermal = trial.final_thermal_evaluations()[c];
        double total_capacity = liquid_volume;
        auto capacity = [&](size_t p, R::QbmmAtom atom)
        {
            return dt * film_coefficient(families[p]->population()) * atom.count *
                   area(atom.mass_over_scale, f.frozen_temperature[p], f.frozen_pressure[p]);
        };
        for (size_t p = 1; p + 1 < families.size(); ++p)
            for (const auto atom : seed(families[p]->population()))
                total_capacity += capacity(p, atom);
        // V*C_end + sum(dt*k*A*C_end) = n_dissolved, with one concentration for all node rows.
        const double concentration = dissolved / total_capacity;
        EXPECT_NEAR(cell.phases[0].moles[0], liquid_volume * concentration, 2e-12);
        EXPECT_DOUBLE_EQ(cell.phases[0].moles[1], solvent_moles);
        const auto& transported_inert = trial.transport_snapshot().cells[c].phases.back().moles;
        EXPECT_EQ(cell.phases.back().moles, transported_inert);
        EXPECT_DOUBLE_EQ(transported_inert[1], old.phases.back().moles[1]);
        ASSERT_TRUE(trial.interphase_receipt());
        const auto& plan = trial.interphase_receipt()->cells.at(c);
        ASSERT_EQ(thermal.material_events.size(), families.size() - 2);
        const double donor_h =
            liquid_u + 1000 * (thermal.anchors.at(0).temperature[0] - T::T0) + thermal.absolute_pressure;
        double transferred = 0;
        for (size_t p = 1; p + 1 < families.size(); ++p)
        {
            const auto& name = families[p]->population();
            const auto atoms = seed(name);
            const auto& receipt = trial.local_qbmm_sources()[c][p].value();
            ASSERT_EQ(receipt.events.size(), atoms.size());
            double family_amount = 0, h2 = 0, h3 = 0;
            for (size_t a = 0; a < atoms.size(); ++a)
            {
                const auto atom = atoms[a];
                const double amount = concentration * capacity(p, atom);
                const double destination = atom.mass_over_scale + amount / (atom.count * mass_scale);
                h2 += atom.count * destination * destination;
                h3 += atom.count * destination * destination * destination;
                const auto& event = receipt.events[a];
                EXPECT_EQ(event.kind, R::QbmmEventKind::Growth);
                ASSERT_EQ(event.transfers.size(), 1u);
                EXPECT_TRUE(event.transfers[0].into_family);
                EXPECT_NEAR(event.transfers[0].transfer.moles[0], amount, 1e-12);
                EXPECT_EQ(event.transfers[0].transfer.moles[1], 0);
                family_amount += amount;
            }
            EXPECT_NEAR(cell.phases[p].moles[0] - old.phases[p].moles[0], family_amount, 2e-12);
            EXPECT_EQ(cell.phase_numbers[p], old.phase_numbers[p]);
            EXPECT_NEAR(cell.qbmm_shape[p]->H2, h2, 5e-10 * h2);
            EXPECT_NEAR(cell.qbmm_shape[p]->H3, h3, 5e-10 * h3);
            EXPECT_EQ(report.cells[c].families[p]->quadrature.status,
                name == "fine" ? R::InversionStatus::TwoNodes : R::InversionStatus::RankOne);
            EXPECT_EQ(diag.population_number_changes.at(p), 0);
            const auto* paired = film_receipt(plan, thermal, p);
            ASSERT_NE(paired, nullptr);
            EXPECT_NEAR(plan.amounts.at(paired->stage).at(paired->slot), family_amount, 2e-12);
            EXPECT_NEAR(paired->carried_energy, family_amount * donor_h, 2e-6);
            EXPECT_NEAR(thermal.raw_material_enthalpy[p], family_amount * donor_h, 2e-6);
            transferred += family_amount;
        }
        EXPECT_NEAR(thermal.raw_material_enthalpy[0], -transferred * donor_h, 2e-6);
        EXPECT_GT(transferred, dissolved / 2);
        EXPECT_LT(transferred, dissolved);
        double energy_change = 0;
        for (size_t p = 0; p < cell.phases.size(); ++p)
            energy_change += cell.phases[p].internal_energy - old.phases[p].internal_energy;
        const double bulk_change = moving ? .05 * dt : 0;
        EXPECT_NEAR(report.cells[c].bulk_volume_m3, 1 + bulk_change, 1e-14);
        const double work = -thermal.absolute_pressure * bulk_change;
        EXPECT_NEAR(diag.flow.pressure_work, work, 2e-4);
        EXPECT_NEAR(energy_change, work, 2e-4);
    }
}

void exercise_film(bool joint, bool moving)
{
    Fixture f(joint ? Evaluation::Joint : Evaluation::Integrated, moving);
    check_partition(f);
    const auto saved = f.controller->serialize_qbmm_restart();
    const auto retained = f.controller->preview(dt);
    check_first_step(f, retained, moving);
    f.failure = joint ? Failure::SeparateSolves : Failure::Explicit;
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    if (f.mesh->num_owned_cells())
        EXPECT_GT(f.attempted, dissolved);
    if (joint)
    {
        f.failure = Failure::WrongRowCount;
        EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    }
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), saved);
    if (f.mesh->num_owned_cells())
        EXPECT_EQ(f.mesh->cell_volume(0), 1.);
    const auto calls = std::array{f.calls, f.motion_calls, f.partial_calls};
    f.controller->commit(retained);
    EXPECT_EQ((std::array{f.calls, f.motion_calls, f.partial_calls}), calls);
    EXPECT_EQ(f.flow->state().time, dt);
}

TEST(ReactingMixtureQbmmFilm, SingleFamilyFixed)
{
    exercise_film(false, false);
}
TEST(ReactingMixtureQbmmFilm, SingleFamilyPlanar)
{
    exercise_film(false, true);
}
TEST(ReactingMixtureQbmmFilm, JointFamiliesFixed)
{
    exercise_film(true, false);
}
TEST(ReactingMixtureQbmmFilm, JointFamiliesPlanar)
{
    exercise_film(true, true);
}

TEST(ReactingMixtureQbmmFilm, TypedLawUsesFrozenLiquidTrace)
{
    Fixture f(Evaluation::Typed, false);
    const auto trial = f.controller->preview(dt);
    check_first_step(f, trial, false);
}

TEST(ReactingMixtureQbmmFilm, JointGateAndModeAreEnforcedBeforePublication)
{
    Fixture f(Evaluation::Joint, false);
    const auto before = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(2 * dt), std::exception);
    EXPECT_EQ(f.calls, 0);
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
    auto individual = [](const R::MixtureQbmmContext&) { return Operations{}; };
    const auto mixed = [&](R::ReactingMixtureOptions& options)
    {
        const auto& provider = *options.qbmm_population;
        options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{provider.metadata, provider.transfers, individual, {}, provider.evaluate_joint});
    };
    EXPECT_THROW((void) Fixture(Evaluation::Joint, false, false, mixed), std::exception);
    // Strip transfer slots from both fixtures so only the callback mode can
    // distinguish their controller signatures.
    const auto joint = [](R::ReactingMixtureOptions& options)
    {
        const auto& provider = *options.qbmm_population;
        options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{
            provider.metadata, nullptr, {}, {},
            [](const R::MixtureQbmmJointContext& context) { return std::vector<Operations>(context.families.size()); }});
    };
    const auto local = [&](R::ReactingMixtureOptions& options)
    {
        const auto& provider = *options.qbmm_population;
        options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{provider.metadata, nullptr, individual});
    };
    Fixture joint_only(Evaluation::Joint, false, false, joint);
    joint_only.controller->advance(dt);
    const auto joint_image = joint_only.controller->serialize_qbmm_restart();
    Fixture alternate(Evaluation::Joint, false, false, local);
    const auto local_image = alternate.controller->serialize_qbmm_restart();
    EXPECT_THROW(alternate.controller->restore_qbmm_restart(joint_image), std::exception);
    EXPECT_THROW(joint_only.controller->restore_qbmm_restart(local_image), std::exception);
    EXPECT_EQ(alternate.controller->serialize_qbmm_restart(), local_image);
    EXPECT_EQ(joint_only.controller->serialize_qbmm_restart(), joint_image);
}

TEST(ReactingMixtureQbmmFilm, PopulationRequiredJointRowRejectsBirthAtAbsence)
{
    bool emit_birth = false;
    int absent_observations = 0;
    const auto configure = [&](R::ReactingMixtureOptions& options)
    {
        const auto& provider = *options.qbmm_population;
        auto evaluate = [&, film = provider.evaluate_joint](const R::MixtureQbmmJointContext& context)
        {
            auto events = film(context);
            for (size_t i = 0; i < context.families.size(); ++i)
            {
                const auto& row = context.families[i];
                if (row.family.population() != "coarse")
                    continue;
                ++absent_observations;
                const auto& input = row.law_inputs;
                EXPECT_EQ(input.quadrature.status, R::InversionStatus::Absent);
                EXPECT_EQ(input.quadrature.active_atoms, 0u);
                EXPECT_EQ(input.geometry.value().active_atoms, 0u);
                EXPECT_FALSE(input.temperature_K);
                EXPECT_FALSE(input.phase_concentration_mol_per_m3);
                EXPECT_DOUBLE_EQ(input.carrier_liquid_volume_m3.value(), liquid_volume);
                EXPECT_FALSE(row.events_allowed);
                EXPECT_TRUE(events.at(i).empty());
                if (emit_birth)
                    events[i].emplace_back(
                        R::QbmmBirth{{{{1, 1}, {R::QbmmPhaseExchange{Phase::Dissolved}, {mass_scale, 0}}}}});
            }
            return events;
        };
        options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{provider.metadata, provider.transfers, {}, {}, std::move(evaluate)});
    };
    Fixture f(Evaluation::Joint, false, false, configure, {});
    check_partition(f);
    const auto saved = f.controller->serialize_qbmm_restart();
    const auto valid = f.controller->preview(dt);
    EXPECT_EQ(absent_observations, int(f.mesh->num_owned_cells()) * valid.diagnostics().flow.source_evaluations);
    const int successful_observations = absent_observations;
    emit_birth = true;
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(absent_observations - successful_observations, int(f.mesh->num_owned_cells()));
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), saved);
    const int before_commit = absent_observations;
    f.controller->commit(valid);
    EXPECT_EQ(absent_observations, before_commit);
    EXPECT_EQ(f.flow->state().time, dt);
    for (const auto& cell : f.flow->state().cells)
    {
        EXPECT_LT(cell.phases[0].moles[0], dissolved);
        EXPECT_EQ(cell.phase_numbers[2], 0.);
        EXPECT_EQ(cell.phases[2].moles[0], 0.);
        EXPECT_EQ(cell.qbmm_shape[2].value(), R::QbmmShapeState{});
    }
}

TEST(ReactingMixtureQbmmFilm, ReorderedPhysicalFamiliesShareTheSameDonor)
{
    Fixture forward(Evaluation::Joint, false), reversed(Evaluation::Joint, false, true);
    const auto a = forward.controller->preview(dt);
    const auto b = reversed.controller->preview(dt);
    check_first_step(forward, a, false);
    check_first_step(reversed, b, false);
    EXPECT_NEAR(a.state().thermodynamic_pressure, b.state().thermodynamic_pressure, 1e-7);
    for (size_t c = 0; c < a.state().cells.size(); ++c)
    {
        const auto& left = a.state().cells[c];
        const auto& right = b.state().cells[c];
        const auto& left_thermal = a.final_thermal_evaluations()[c];
        const auto& right_thermal = b.final_thermal_evaluations()[c];
        const auto& roles = reversed.flow->options().phase_roles;
        for (size_t p = 0; p < left.phases.size(); ++p)
        {
            const auto& name = forward.flow->options().phase_roles[p].identity;
            SCOPED_TRACE(name);
            const auto other = std::ranges::find(roles, name, &Flow::PhaseRoleDescriptor::identity);
            ASSERT_NE(other, roles.end());
            const size_t q = other - roles.begin();
            const auto compare_energy = [](long double actual, long double expected)
            { EXPECT_NEAR(actual, expected, 64 * std::numeric_limits<double>::epsilon() * std::abs(expected)); };
            compare_energy(left.phases[p].internal_energy, right.phases[q].internal_energy);
            compare_energy(left_thermal.raw_material_enthalpy[p], right_thermal.raw_material_enthalpy[q]);
            EXPECT_NEAR(left.phases[p].moles[0], right.phases[q].moles[0], 1e-13);
            if (other->role != Flow::PhaseRole::DispersedGas)
                continue;
            const auto* lp = film_receipt(a.interphase_receipt()->cells.at(c), left_thermal, p);
            const auto* rp = film_receipt(b.interphase_receipt()->cells.at(c), right_thermal, q);
            ASSERT_NE(lp, nullptr);
            ASSERT_NE(rp, nullptr);
            compare_energy(lp->carried_energy, rp->carried_energy);
            EXPECT_NEAR(left.qbmm_shape[p]->H2, right.qbmm_shape[q]->H2, 1e-10);
            EXPECT_NEAR(left.qbmm_shape[p]->H3, right.qbmm_shape[q]->H3, 1e-9);
            EXPECT_EQ(left.phase_numbers[p], right.phase_numbers[q]);
        }
    }
}

void portable_continuation(bool moving)
{
    const char* writer = std::getenv("SIMPLEFLUID_FILM_WRITE");
    const char* reader = std::getenv("SIMPLEFLUID_FILM_READ");
    ASSERT_FALSE(writer && reader);
    Fixture reference(Evaluation::Joint, moving);
    check_partition(reference);
    const auto comm = reference.mesh->owned_cell_map()->getComm();
    std::vector<std::byte> first;
    if (reader)
    {
        first = IO::read_restart(*comm, reader, ".first");
        reference.controller->restore_qbmm_restart(IO::read_restart(*comm, reader, ".next"));
    }
    else
    {
        reference.controller->advance(dt);
        first = reference.controller->serialize_qbmm_restart();
        reference.controller->advance(dt);
        if (writer)
        {
            IO::write_restart(*comm, writer, ".first", first);
            IO::write_restart(*comm, writer, ".next", reference.controller->serialize_qbmm_restart());
            return;
        }
    }
    Fixture fresh(Evaluation::Joint, moving);
    fresh.controller->restore_qbmm_restart(first);
    EXPECT_EQ(fresh.calls, 0);
    EXPECT_EQ(fresh.motion_calls, 0);
    fresh.controller->advance(dt);
    const auto& actual = fresh.flow->state();
    const auto& expected = reference.flow->state();
    EXPECT_EQ(actual.time, 2 * dt);
    EXPECT_EQ(actual.revision, expected.revision);
    EXPECT_DOUBLE_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    const std::vector<double> preserved_counts(fresh.flow->options().phase_roles.size(), 0.);
    EXPECT_EQ(fresh.controller->diagnostics().population_number_changes, preserved_counts);
    // Native fixed-grid images archive physical counts; controller source diagnostics are available for ALE.
    if (reader && !moving)
        EXPECT_TRUE(reference.controller->diagnostics().population_number_changes.empty());
    else
        EXPECT_EQ(reference.controller->diagnostics().population_number_changes, preserved_counts);
    EXPECT_DOUBLE_EQ(
        fresh.controller->diagnostics().flow.pressure_work, reference.controller->diagnostics().flow.pressure_work);
    ASSERT_EQ(actual.cells.size(), expected.cells.size());
    for (size_t c = 0; c < actual.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(fresh.mesh->cell_volume(c), reference.mesh->cell_volume(c));
        const auto& a = actual.cells[c];
        const auto& b = expected.cells[c];
        EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
        EXPECT_EQ(a.phase_numbers, b.phase_numbers);
        EXPECT_EQ(a.velocity, b.velocity);
        EXPECT_EQ(a.mechanical_pressure, b.mechanical_pressure);
        for (size_t p = 0; p < a.phases.size(); ++p)
        {
            EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
            EXPECT_DOUBLE_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
            EXPECT_DOUBLE_EQ(a.phases[p].temperature, b.phases[p].temperature);
            EXPECT_DOUBLE_EQ(a.phases[p].absolute_pressure, b.phases[p].absolute_pressure);
        }
    }
    for (size_t f = 0; f < fresh.mesh->num_owned_faces(); ++f)
        EXPECT_DOUBLE_EQ(fresh.flow->volume_flux().owned_value(f), reference.flow->volume_flux().owned_value(f));
}

TEST(ReactingMixtureQbmmFilm, FixedPortableContinuation)
{
    portable_continuation(false);
}
TEST(ReactingMixtureQbmmFilm, PlanarPortableContinuation)
{
    portable_continuation(true);
}
} // namespace
