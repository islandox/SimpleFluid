/** Adaptive local population sources against a finite-stock rate equation and chronological event replay. */
#include "QbmmFilmTestSupport.hh"
#include "solvers/details/LowMachQbmmState.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_DefaultComm.hpp>
#include <map>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmFilm;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

using Options = R::ReactingMixtureOptions;
using Law = std::function<Operations(const R::MixtureQbmmContext&)>;
constexpr double rate_constant = 1000.; // 1/(mol s)
constexpr double convergence_tolerance = .003;
constexpr char ranks_variable[] = "SIMPLEFLUID_QBMM_SUBSTEP_RANKS";

auto source_controls(double tolerance, size_t limit = 64)
{
    return [=](Options& options)
    {
        options.qbmm_source_substep_relative_tolerance = tolerance;
        options.qbmm_source_maximum_substeps = limit;
    };
}

auto integrated_controls(double tolerance, size_t limit, Law law)
{
    return [=](Options& options)
    {
        source_controls(tolerance, limit)(options);
        const auto& film = *options.qbmm_population;
        options.qbmm_population =
            std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{film.metadata, film.transfers, law});
    };
}

struct Observation
{
    size_t corrector;
    double time, step, liquid;
};

// dL/dt = -k L^2. Each birth at size one receives exactly one scaled unit of mass per bubble.
Law finite_stock_births(std::vector<Observation>& observed, double k = rate_constant)
{
    return [=, &observed](const R::MixtureQbmmContext& ctx) -> Operations
    {
        const auto& inputs = *ctx.law_inputs;
        const double liquid = ctx.material.amount(ctx.family.catalog(), tracer, Phase::Dissolved);
        EXPECT_EQ(inputs.carrier_liquid_dissolved_moles.value().at(0), liquid);
        EXPECT_EQ(inputs.time_step_s, ctx.time_step);
        observed.push_back({ctx.stage.corrector, ctx.accepted_time, ctx.time_step, liquid});
        const double amount = ctx.time_step * k * liquid * liquid;
        return {R::QbmmBirth{{{{amount / mass_scale, 1}, {R::QbmmPhaseExchange{Phase::Dissolved}, {amount, 0}}}}}};
    };
}

using Inventories = std::array<double, 5>; // Dissolved mol, dispersed mol, H0, H2, H3.

Inventories expected_from_liquid(double stock)
{
    const double transferred = dissolved - stock;
    const double births = transferred / mass_scale;
    return {stock, 5 * mass_scale + transferred, 3 + births, 11 + births, 29 + births};
}

Inventories inventories(const Flow::CellState& cell)
{
    return {cell.phases[0].moles[0], cell.phases[1].moles[0], cell.phase_numbers[1].value(), cell.qbmm_shape[1]->H2,
        cell.qbmm_shape[1]->H3};
}

void check_birth_receipts(
    const Fixture& f, const Coupler::Trial& trial, const std::vector<Observation>& observed, double k, double duration,
    double tolerance)
{
    const auto count = trial.diagnostics().maximum_qbmm_source_substeps;
    ASSERT_GE(count, 1u);
    const auto report = f.controller->qbmm_report(trial);
    ASSERT_TRUE(report.interval);
    ASSERT_TRUE(report.balances);
    EXPECT_NEAR(report.balances->mass_defect_kg, 0, 1e-13);
    EXPECT_NEAR(report.balances->element_defects_mol.at("X"), 0, 2e-13);
    EXPECT_NEAR(report.balances->internal_energy_defect_j, 0, 2e-4);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& receipt = trial.local_qbmm_sources().at(c).at(1).value();
        const auto& published = report.interval->local_receipts.at(c).at(1).value();
        ASSERT_EQ(receipt.events.size(), count);
        ASSERT_EQ(receipt.substeps.size(), count);
        ASSERT_EQ(published.substeps.size(), count);
        ASSERT_EQ(published.events.size(), count);
        double stock = dissolved;
        const double step = duration / count;
        for (size_t i = 0; i < count; ++i)
        {
            const auto& event = receipt.events[i];
            const auto& substep = receipt.substeps[i];
            const double amount = step * k * stock * stock;
            const double midpoint = stock - .5 * amount;
            const auto initial = expected_from_liquid(stock);
            const auto full = expected_from_liquid(stock - amount);
            const auto halves = expected_from_liquid(midpoint - .5 * step * k * midpoint * midpoint);
            for (size_t channel = 0; channel < 2; ++channel)
                EXPECT_LE(std::abs(full[channel] - halves[channel]), tolerance * (initial[0] + initial[1]));
            for (size_t channel = 2; channel < full.size(); ++channel)
            {
                const auto error = std::abs(full[channel] - halves[channel]);
                const auto increment = std::max(
                    std::abs(full[channel] - initial[channel]), std::abs(halves[channel] - initial[channel]));
                EXPECT_TRUE(error <= tolerance * std::max({initial[channel], full[channel], halves[channel]}) ||
                            error * (step / duration) <= tolerance * increment);
            }
            EXPECT_TRUE(std::ranges::any_of(observed,
                [&](auto call)
                {
                    return call.corrector == trial.transport_snapshot().stage.corrector &&
                           std::abs(call.time - step * i) < 1e-15 && std::abs(call.step - step) < 1e-16 &&
                           std::abs(call.liquid - stock) < 2e-14;
                }));
            stock -= amount;
            EXPECT_EQ(event.kind, R::QbmmEventKind::Birth);
            EXPECT_EQ(event.substep, i);
            EXPECT_DOUBLE_EQ(event.duration, substep.duration);
            ASSERT_EQ(event.products.size(), 1u);
            ASSERT_EQ(event.transfers.size(), 1u);
            EXPECT_NEAR(event.products.front().count, amount / mass_scale, 2e-12);
            EXPECT_NEAR(event.transfers.front().transfer.moles.at(0), amount, 2e-15);
            EXPECT_TRUE(event.transfers.front().into_family);
            EXPECT_EQ(substep.event_end, i + 1);
            EXPECT_NEAR(substep.duration, step, 4 * std::numeric_limits<double>::epsilon() * duration);
            EXPECT_EQ(substep.substep, i);
            const auto expected = expected_from_liquid(stock);
            EXPECT_NEAR(substep.final_moles.at(0), expected[1], 2e-14);
            EXPECT_NEAR(substep.final_count, expected[2], 2e-11);
            EXPECT_NEAR(substep.final_shape.H2, expected[3], 2e-11);
            EXPECT_NEAR(substep.final_shape.H3, expected[4], 2e-11);
            EXPECT_EQ(published.substeps[i].event_end, substep.event_end);
            EXPECT_EQ(published.substeps[i].substep, substep.substep);
            EXPECT_EQ(published.substeps[i].duration, substep.duration);
            EXPECT_EQ(published.substeps[i].final_count, substep.final_count);
            EXPECT_EQ(published.substeps[i].final_moles, substep.final_moles);
            EXPECT_EQ(published.substeps[i].final_shape, substep.final_shape);
            EXPECT_EQ(published.events[i].substep, i);
            EXPECT_EQ(published.events[i].transfers.at(0).transfer.moles, event.transfers.at(0).transfer.moles);
        }
        const auto actual = inventories(trial.state().cells[c]);
        const auto expected = expected_from_liquid(stock);
        for (size_t channel = 0; channel < actual.size(); ++channel)
            EXPECT_NEAR(actual[channel], expected[channel], 2e-13 * std::max(actual[channel], expected[channel]));
        const double mass = dissolved - stock;
        const double births = mass / mass_scale;
        EXPECT_NEAR(trial.diagnostics().population_births.at(1), births, 2e-11);
        // Final family gross channels include counts, mass in/out, H2/H3 created/consumed and reservoir activity.
        const auto& row = report.interval->local_gross_events.at(c);
        ASSERT_GE(row.size(), Flow::QbmmSourceReceipt::gross_channels);
        const auto gross = std::span(row).last(Flow::QbmmSourceReceipt::gross_channels);
        const std::array<double, 18> expected_gross{births, 0, 0, 0, mass, 0, births, 0, births, 0};
        for (size_t i = 0; i < gross.size(); ++i)
            EXPECT_NEAR(gross[i], expected_gross[i], 2e-12 * expected_gross[i]);
        EXPECT_EQ(trial.diagnostics().population_deaths.at(1), 0.);
        EXPECT_EQ(trial.local_sources()[c].externalTransfers.size(), 0u);
    }
}

TEST(ReactingMixtureQbmmSubstepping, NonlinearTemporalConvergenceInBothInventoriesAndAllIndependentMoments)
{
    const auto exact = expected_from_liquid(dissolved / (1 + rate_constant * dissolved * dt));
    constexpr size_t fine_steps = 16384;
    double reference_liquid = dissolved;
    for (size_t i = 0; i < fine_steps; ++i)
        reference_liquid -= (dt / fine_steps) * rate_constant * reference_liquid * reference_liquid;
    const auto reference = expected_from_liquid(reference_liquid);
    Inventories previous_error{};
    for (double tolerance : {0., convergence_tolerance, .0003})
    {
        std::vector<Observation> calls;
        Fixture f(Evaluation::Integrated, false, false, integrated_controls(tolerance, 64, finite_stock_births(calls)));
        check_partition(f, ranks_variable);
        const auto trial = f.controller->preview(dt);
        const std::string prefix =
            tolerance == 0 ? "legacy" : (tolerance == convergence_tolerance ? "adaptive" : "tight");
        RecordProperty(prefix + "_substeps", trial.diagnostics().maximum_qbmm_source_substeps);
        if (tolerance > 0)
        {
            EXPECT_GE(trial.diagnostics().maximum_qbmm_source_substeps, 4u);
            check_birth_receipts(f, trial, calls, rate_constant, dt, tolerance);
        }
        for (const auto& cell : trial.state().cells)
        {
            const auto actual = inventories(cell);
            for (size_t channel = 0; channel < actual.size(); ++channel)
            {
                SCOPED_TRACE(testing::Message() << "channel " << channel << ", tolerance " << tolerance);
                constexpr std::array names{"dissolved_mol", "dispersed_mol", "H0", "H2", "H3"};
                const double error = std::abs(actual[channel] - exact[channel]);
                RecordProperty(prefix + "_" + names[channel] + "_error", testing::PrintToString(error));
                if (tolerance > 0)
                {
                    EXPECT_LT(error, .65 * previous_error[channel]);
                    EXPECT_LT(std::abs(actual[channel] - reference[channel]), .65 * previous_error[channel]);
                }
                else
                    EXPECT_NEAR(actual[channel],
                        expected_from_liquid(dissolved - dt * rate_constant * dissolved * dissolved)[channel],
                        1e-13 * actual[channel]);
                previous_error[channel] = error;
            }
        }
    }
}

void check_retry_and_schedule(bool moving)
{
    constexpr double k = 7500.; // A full explicit first stage consumes 150% of the dissolved donor.
    std::vector<Observation> calls;
    Fixture f(Evaluation::Integrated, moving, false, integrated_controls(.05, 64, finite_stock_births(calls, k)));
    check_partition(f, ranks_variable);
    const auto checkpoint = f.controller->checkpoint();
    const auto before = f.controller->serialize_qbmm_restart();
    const auto trial = f.controller->preview(dt);
    EXPECT_GT(trial.diagnostics().maximum_qbmm_source_substeps, 1u);
    check_birth_receipts(f, trial, calls, k, dt, .05);
    if (f.mesh->num_owned_cells())
    {
        std::map<size_t, double> first_step;
        for (const auto& call : calls)
            if (first_step.emplace(call.corrector, call.step).second)
            {
                EXPECT_DOUBLE_EQ(call.time, 0.);
                EXPECT_NEAR(call.liquid, dissolved, 2e-15);
            }
        ASSERT_GE(first_step.size(), 2u);
        EXPECT_EQ(first_step.begin()->second, dt);
        EXPECT_EQ(first_step.rbegin()->second, dt / trial.diagnostics().maximum_qbmm_source_substeps);
        EXPECT_EQ(trial.diagnostics().qbmm_source_substep_retries, 0u);
        EXPECT_EQ(std::ranges::count_if(calls, [](const auto& call) { return call.step == dt; }), 1);
    }
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
    const auto callback_count = calls.size();
    f.controller->commit(trial);
    EXPECT_EQ(calls.size(), callback_count);
    if (f.mesh->num_owned_cells())
        EXPECT_NEAR(f.mesh->cell_volume(0), moving ? 1. + .05 * dt : 1., 2e-14);
    f.controller->restore(checkpoint);
    EXPECT_EQ(calls.size(), callback_count);
    calls.clear();
    const auto replay = f.controller->preview(dt);
    if (!calls.empty())
        EXPECT_EQ(calls.front().step, dt);
    EXPECT_EQ(replay.diagnostics().maximum_qbmm_source_substeps, trial.diagnostics().maximum_qbmm_source_substeps);
}

TEST(ReactingMixtureQbmmSubstepping, FixedDonorOverdrawRestartsFromTransportAndRetainsGridAcrossCorrectors)
{
    check_retry_and_schedule(false);
}

TEST(ReactingMixtureQbmmSubstepping, PlanarDonorOverdrawRestartsFromTransportAndRetainsGridAcrossCorrectors)
{
    check_retry_and_schedule(true);
}

TEST(ReactingMixtureQbmmSubstepping, MaximumGridExhaustionRollsBackAndPreservesEarlierTrial)
{
    for (const bool moving : {false, true})
    {
        std::vector<Observation> calls;
        Fixture f(Evaluation::Integrated, moving, false, integrated_controls(.001, 2, finite_stock_births(calls)));
        const auto image = f.controller->serialize_qbmm_restart();
        const auto retained = f.controller->preview(dt / 128);
        const auto epoch = f.mesh->geometry_epoch();
        calls.clear();
        EXPECT_THROW((void) f.controller->preview(dt), std::exception);
        if (f.mesh->num_owned_cells())
        {
            EXPECT_TRUE(std::ranges::any_of(calls, [](auto call) { return call.step == dt / 2; }));
            EXPECT_TRUE(std::ranges::none_of(calls, [](auto call) { return call.step < dt / 4; }));
            EXPECT_DOUBLE_EQ(f.mesh->cell_volume(0), 1.);
        }
        if (moving)
            EXPECT_GT(f.mesh->geometry_epoch(), epoch); // Restoring physical geometry invalidates trial-epoch caches.
        else
            EXPECT_EQ(f.mesh->geometry_epoch(), epoch);
        EXPECT_EQ(f.controller->serialize_qbmm_restart(), image);
        EXPECT_EQ(f.flow->state().time, 0);
        const auto callback_count = calls.size();
        f.controller->commit(retained);
        EXPECT_EQ(calls.size(), callback_count);
        EXPECT_EQ(f.flow->state().time, dt / 128);
    }
}

TEST(ReactingMixtureQbmmSubstepping, InvalidInitialApplicabilityGateRejectsBeforeAnyLaw)
{
    // The full duration violates the declared range. Its half-duration would pass.
    Fixture f(Evaluation::Joint, false, false, source_controls(.1));
    const auto image = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(2 * dt), std::exception);
    EXPECT_EQ(f.calls, 0);
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), image);
}

TEST(ReactingMixtureQbmmSubstepping, ProvisionalConcentrationGateRetriesFromOriginalFiniteStock)
{
    // dL/dt = -300*(L-.019). The first half-step overshoots the physical equilibrium and violates the gate.
    constexpr double equilibrium = .019, gate_stock = .0187;
    auto law = film_law;
    law.ranges = {{R::QbmmRangeVariable::CarrierLiquidConcentrationMolPerM3,
        gate_stock / (gate_stock + carrier_molar_mass * solvent_moles), 1., 0}};
    std::vector<Observation> calls;
    const auto birth = [&](const R::MixtureQbmmContext& ctx) -> Operations
    {
        const double stock = ctx.material.amount(ctx.family.catalog(), tracer, Phase::Dissolved);
        calls.push_back({ctx.stage.corrector, ctx.accepted_time, ctx.time_step, stock});
        EXPECT_GE(stock, gate_stock);
        const double amount = ctx.time_step * 300 * (stock - equilibrium);
        return {R::QbmmBirth{{{{amount / mass_scale, 1}, {R::QbmmPhaseExchange{Phase::Dissolved}, {amount, 0}}}}}};
    };
    Fixture f(Evaluation::Typed, false, false, integrated_controls(.5, 16, birth), coarse, law);
    const auto trial = f.controller->preview(dt);
    const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
    ASSERT_GE(n, 4u);
    const double stock = equilibrium + (dissolved - equilibrium) * std::pow(1 - 300 * dt / n, n);
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(calls.front().step, dt);
        EXPECT_NEAR(cell.phases[0].moles[0], stock, 2e-14);
    }
}

TEST(ReactingMixtureQbmmSubstepping, MalformedFirstLawOutputRejectsWithoutRefinement)
{
    size_t evaluations = 0;
    const auto invalid = [&](const R::MixtureQbmmContext&) -> Operations
    {
        ++evaluations;
        return {R::QbmmBirth{{{{1, 1}, {R::QbmmPhaseExchange{Phase::Dissolved}, {mass_scale}}}}}};
    };
    Fixture f(Evaluation::Integrated, false, false, integrated_controls(.01, 64, invalid));
    const auto before = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(evaluations, f.mesh->num_owned_cells());
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
}

TEST(ReactingMixtureQbmmSubstepping, LocalGridsDifferAcrossCellsAndReduceTheTrueMaximum)
{
    const auto turnover = [](const R::MixtureQbmmContext& ctx) -> Operations
    {
        if (ctx.law_inputs->location->centroid[2] >= 1.)
            return {};
        // Turn over three counts at unchanged mass. The original node holds only two at a time.
        const double count = 300. * ctx.time_step;
        const double size = ctx.quadrature.atoms[0].mass_over_scale;
        const R::QbmmMaterialTransfer transfer{
            R::QbmmReservoirExchange{"turnover", 0, {}}, {count * size * mass_scale, 0}};
        return {R::QbmmDisappearance{0, count, transfer}, R::QbmmBirth{{{{count, size}, transfer}}}};
    };
    Fixture f(Evaluation::Integrated, false, false, integrated_controls(.01, 8, turnover), coarse, film_law, {0, 1, 2});
    const auto trial = f.controller->preview(dt);
    EXPECT_EQ(trial.diagnostics().maximum_qbmm_source_substeps, 2u);
    EXPECT_NEAR(trial.diagnostics().population_deaths.at(1), 3., 2e-13);
    EXPECT_NEAR(trial.diagnostics().population_births.at(1), 3., 2e-13);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const bool active = f.mesh->cell_centroid(c).z < 1.;
        const auto& receipt = trial.local_qbmm_sources()[c][1].value();
        ASSERT_EQ(receipt.substeps.size(), active ? 2u : 1u);
        EXPECT_EQ(receipt.events.size(), active ? 4u : 0u);
        EXPECT_EQ(receipt.substeps.back().event_end, receipt.events.size());
        EXPECT_EQ(receipt.substeps.back().duration, active ? dt / 2 : dt);
        EXPECT_NEAR(trial.state().cells[c].phase_numbers[1].value(), 3., 2e-12);
    }
}

TEST(ReactingMixtureQbmmSubstepping, InvalidOrRankDisagreeingNumericalControlsRejectCollectively)
{
    for (auto tolerance : {-1., 1., std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW((void) Fixture(Evaluation::Integrated, false, false, source_controls(tolerance)), std::exception);
    EXPECT_THROW((void) Fixture(Evaluation::Integrated, false, false, source_controls(.1, 0)), std::exception);
    const auto comm = Teuchos::DefaultComm<int>::getComm();
    if (comm->getSize() > 1)
    {
        EXPECT_THROW(
            (void) Fixture(Evaluation::Integrated, false, false, source_controls(comm->getRank() == 0 ? .1 : .2)),
            std::exception);
        EXPECT_THROW(
            (void) Fixture(Evaluation::Integrated, false, false, source_controls(.1, comm->getRank() == 0 ? 16 : 32)),
            std::exception);
    }
}

TEST(ReactingMixtureQbmmSubstepping, NonpositiveFullStepShrinkRetriesAndPublishesOnlyAcceptedReservoirPackets)
{
    for (bool direct : {false, true})
    {
        SCOPED_TRACE(direct ? "direct integrated resize" : "mass rate adapter");
        constexpr double frequency = 150.;
        std::vector<double> steps;
        const auto shrink = [&](const R::MixtureQbmmContext& ctx)
        {
            steps.push_back(ctx.time_step);
            Operations direct_events;
            std::vector<double> rates;
            std::vector<R::QbmmTransferCounterpart> reservoir;
            for (const auto& atom : std::span(ctx.quadrature.atoms).first(ctx.quadrature.active_atoms))
            {
                rates.push_back(-frequency * atom.mass_over_scale * mass_scale);
                const double lost_moles = -ctx.time_step * atom.count * rates.back();
                reservoir.emplace_back(R::QbmmReservoirExchange{
                    "withdrawn tracer", -lost_moles * T::T0 * T::cv, {-.125 * lost_moles, 0, 0}});
                if (direct)
                {
                    const double destination = atom.mass_over_scale * (1. - frequency * ctx.time_step);
                    const double amount = (atom.mass_over_scale - destination) * atom.count * mass_scale;
                    direct_events.emplace_back(
                        R::QbmmResize{direct_events.size(), atom.count, destination, {reservoir.back(), {amount, 0}}});
                }
            }
            if (direct)
                return direct_events;
            return R::qbmm_resize_from_mass_rate(ctx.family, *ctx.law_inputs, rates, reservoir);
        };
        Fixture legacy(Evaluation::Integrated, false, false, integrated_controls(0., 64, shrink));
        const auto legacy_image = legacy.controller->serialize_qbmm_restart();
        EXPECT_THROW((void) legacy.controller->preview(dt), std::exception);
        EXPECT_EQ(legacy.controller->serialize_qbmm_restart(), legacy_image);
        steps.clear();
        Fixture f(Evaluation::Integrated, false, false, integrated_controls(.08, 64, shrink));
        const auto trial = f.controller->preview(dt);
        const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
        ASSERT_GE(n, 2u);
        const double factor = std::pow(1. - frequency * dt / n, n);
        const double mass_loss = (1 - factor) * 5 * mass_scale;
        EXPECT_NEAR(trial.diagnostics().external_mass, -mass_loss, 2e-13);
        EXPECT_NEAR(trial.diagnostics().flow.external_impulse.x, -.125 * mass_loss, 2e-14);
        const auto report = f.controller->qbmm_report(trial);
        ASSERT_TRUE(report.balances);
        EXPECT_NEAR(report.balances->external_species_moles.at(0), -mass_loss, 2e-13);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            EXPECT_EQ(steps.front(), dt);
            const auto& receipt = trial.local_qbmm_sources()[c][1].value();
            ASSERT_EQ(receipt.events.size(), 2 * n);
            double removed = 0, energy = 0;
            for (const auto& event : receipt.events)
            {
                ASSERT_EQ(event.transfers.size(), 1u);
                EXPECT_EQ(event.kind, R::QbmmEventKind::Shrinkage);
                EXPECT_FALSE(event.transfers[0].into_family);
                const auto& transfer = event.transfers[0].transfer;
                removed += transfer.moles.at(0);
                energy += std::get<R::QbmmReservoirExchange>(transfer.counterpart).energy_j;
            }
            EXPECT_NEAR(removed, mass_loss, 2e-13);
            EXPECT_NEAR(energy, -mass_loss * T::cv * T::T0, 2e-10);
            const auto& cell = trial.state().cells[c];
            EXPECT_NEAR(cell.phase_numbers[1].value(), 3, 2e-12);
            EXPECT_NEAR(cell.phases[1].moles[0], 5 * mass_scale * factor, 2e-14);
            EXPECT_NEAR(cell.qbmm_shape[1]->H2, 11 * factor * factor, 3e-11);
            EXPECT_NEAR(cell.qbmm_shape[1]->H3, 29 * std::pow(factor, 3), 3e-11);
        }
    }
}

R::QbmmLawMetadata substepped_film_contract()
{
    auto law = film_law;
    law.ranges = {{R::QbmmRangeVariable::TimeStepS, 0., dt, {}}};
    return law;
}

TEST(ReactingMixtureQbmmSubstepping, EmptyStagesPreservePredecessorAndTinySignedResizeIsAudited)
{
    const auto configure = [](Options& options)
    {
        source_controls(.01, 4)(options);
        const auto& provider = *options.qbmm_population;
        const auto joint = [](const R::MixtureQbmmJointContext& ctx)
        {
            std::vector<Operations> rows;
            for (const auto& family : ctx.families)
            {
                auto& events = rows.emplace_back();
                const auto& atom = family.law_inputs.quadrature.atoms[0];
                const R::QbmmReservoirExchange transfer{"turnover", 0., {}};
                if (family.family.population() == "fine")
                {
                    const double count = 700. * ctx.time_step; // Requires four stages for the two available donors.
                    const R::QbmmMaterialTransfer material{transfer, {count * atom.mass_over_scale * mass_scale, 0}};
                    events = {R::QbmmDisappearance{0, count, material},
                        R::QbmmBirth{{{{count, atom.mass_over_scale}, material}}}};
                }
                else if (ctx.accepted_time >= dt / 4 && ctx.accepted_time < dt / 2)
                {
                    const double destination = std::nextafter(atom.mass_over_scale, 0.);
                    const double amount = (atom.mass_over_scale - destination) * atom.count * mass_scale;
                    events.emplace_back(R::QbmmResize{0, atom.count, destination, {transfer, {amount, 0}}});
                }
            }
            return rows;
        };
        options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{provider.metadata, nullptr, {}, {}, joint});
    };
    Fixture f(Evaluation::Joint, false, false, configure, coarse, substepped_film_contract());
    const auto trial = f.controller->preview(dt);
    EXPECT_EQ(trial.diagnostics().maximum_qbmm_source_substeps, 4u);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& receipt = trial.local_qbmm_sources()[c][2].value();
        ASSERT_EQ(receipt.substeps.size(), 4u);
        ASSERT_EQ(receipt.events.size(), 1u);
        const auto& first = receipt.substeps[0];
        const auto& initial = trial.transport_snapshot().cells[c].phases[2];
        EXPECT_EQ(first.event_end, 0u);
        EXPECT_EQ(first.final_shape, initial.qbmm_shape.value());
        EXPECT_EQ(first.final_count, initial.population_number.value());
        EXPECT_EQ(first.final_moles, initial.moles);
        for (size_t i = 1; i < receipt.substeps.size(); ++i)
        {
            const auto& stage = receipt.substeps[i];
            EXPECT_EQ(stage.event_end, 1u);
            EXPECT_EQ(stage.final_shape, receipt.substeps[1].final_shape);
            EXPECT_EQ(stage.final_count, receipt.substeps[1].final_count);
            EXPECT_EQ(stage.final_moles, receipt.substeps[1].final_moles);
        }
        const auto& event = receipt.events.front();
        ASSERT_EQ(event.substep, 1u);
        ASSERT_EQ(event.kind, R::QbmmEventKind::Shrinkage);
        ASSERT_EQ(event.donors.size(), 1u);
        ASSERT_EQ(event.products.size(), 1u);
        const long double before = event.donors[0].atom.mass_over_scale, after = event.products[0].mass_over_scale;
        EXPECT_EQ(static_cast<double>(after), std::nextafter(static_cast<double>(before), 0.));
        const long double dm = (after - before) * event.donors[0].atom.count;
        const std::array expected{dm * (after + before), dm * (after * after + after * before + before * before)};
        for (size_t k = 0; k < expected.size(); ++k)
        {
            EXPECT_NEAR(event.moment_change[k + 2], expected[k], 1e-14L * std::abs(expected[k]));
            EXPECT_NEAR(receipt.event_shape_change[k], expected[k], 1e-14L * std::abs(expected[k]));
            auto corrupted = event;
            corrupted.moment_change[k + 2] *= .5L;
            std::array<R::qbmm_detail::SourceReceiptSum, 2> changes{};
            EXPECT_THROW(R::qbmm_detail::replay_source_event_moments(corrupted, changes), std::invalid_argument);
        }
    }
}

TEST(ReactingMixtureQbmmSubstepping, CrossStageCancellationPreservesSubUlpCountSpeciesAndShapeChanges)
{
    constexpr double residual = 0x1p-80;
    const std::array seed_at_one{R::QbmmAtom{1., 1.}};
    const auto configure = [](Options& options)
    {
        source_controls(.01, 2)(options);
        const auto exchange = [](const R::MixtureQbmmContext& ctx) -> Operations
        {
            if (ctx.family.population() != "coarse")
                return {};
            const double count = 2 * ctx.time_step / dt;
            const auto packet = [](double weight)
            { return R::QbmmMaterialTransfer{R::QbmmReservoirExchange{"tracer", 0., {}}, {mass_scale * weight, 0.}}; };
            if (ctx.accepted_time < dt / 2)
                return {R::QbmmBirth{{{{count, 1.}, packet(count)}}},
                    R::QbmmBirth{{{{count * residual, 1.}, packet(count * residual)}}}};
            return {R::QbmmDisappearance{0, count, packet(count)}};
        };
        options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
            R::MixtureQbmmProvider{options.qbmm_population->metadata, nullptr, exchange});
    };
    Fixture f(Evaluation::Joint, false, false, configure, seed_at_one, substepped_film_contract());
    const auto trial = f.controller->preview(dt);
    ASSERT_EQ(trial.diagnostics().maximum_qbmm_source_substeps, 2u);
    EXPECT_EQ(trial.diagnostics().population_number_changes.at(2), residual);
    EXPECT_EQ(trial.diagnostics().external_mass, residual * mass_scale);
    const auto report = f.controller->qbmm_report(trial);
    ASSERT_TRUE(report.balances);
    EXPECT_EQ(report.balances->external_species_moles.at(0), residual * mass_scale);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& receipt = trial.local_qbmm_sources()[c][2].value();
        ASSERT_EQ(receipt.events.size(), 3u);
        ASSERT_EQ(receipt.substeps.size(), 2u);
        EXPECT_EQ(receipt.substeps[0].event_end, 2u);
        EXPECT_EQ(receipt.substeps[0].final_count, 2.);
        EXPECT_EQ(receipt.substeps[1].event_end, 3u);
        EXPECT_EQ(receipt.final_count, 1.);
        EXPECT_EQ(receipt.final_moles[0], mass_scale);
        EXPECT_EQ(receipt.final_shape.H2, 1.);
        EXPECT_EQ(receipt.final_shape.H3, 1.);
        EXPECT_EQ(receipt.event_shape_change[0], residual);
        EXPECT_EQ(receipt.event_shape_change[1], residual);
    }
}

// Independent backward-Euler finite-liquid balance, including all families at once.
// V*C_end + sum(dt*k*A*C_end) = n_start with k in m/s, V in m3 and A in m2.
void check_film_substeps(Evaluation mode)
{
    constexpr double duration = dt / 8;
    Fixture f(mode, false, false, source_controls(.005), coarse, substepped_film_contract());
    const auto trial = f.controller->preview(duration);
    const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
    ASSERT_GE(n, 2u);
    const double step = duration / n;
    const auto& families = f.flow->options().qbmm_families;
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& cell = trial.state().cells[c];
        const auto& plan = trial.interphase_receipt()->cells[c];
        const auto& thermal = trial.final_thermal_evaluations()[c];
        ASSERT_EQ(plan.material_tail.substeps.size(), n);
        ASSERT_EQ(thermal.material_events.size(), n * (families.size() - 2));
        std::vector<std::vector<R::QbmmAtom>> atoms(families.size());
        for (size_t p = 1; p + 1 < families.size(); ++p)
            atoms[p].assign(seed(families[p]->population()).begin(), seed(families[p]->population()).end());
        double stock = dissolved;
        for (size_t i = 0; i < n; ++i)
        {
            const double volume = stock + solvent_moles * carrier_molar_mass;
            double capacity = volume;
            const auto node_capacity = [&](size_t p, const R::QbmmAtom& atom)
            {
                return step * film_coefficient(families[p]->population()) * atom.count *
                       area(atom.mass_over_scale, f.frozen_temperature[p], f.frozen_pressure[p]);
            };
            for (size_t p = 1; p + 1 < families.size(); ++p)
                for (const auto& atom : atoms[p])
                    capacity += node_capacity(p, atom);
            const double concentration = stock / capacity;
            stock = volume * concentration;
            for (size_t p = 1; p + 1 < families.size(); ++p)
            {
                const auto& receipt = trial.local_qbmm_sources()[c][p].value();
                ASSERT_EQ(receipt.substeps.size(), n);
                const auto& substep = receipt.substeps[i];
                EXPECT_NEAR(substep.duration, step, 4 * std::numeric_limits<double>::epsilon() * duration);
                ASSERT_EQ(substep.event_end, (i + 1) * atoms[p].size());
                ASSERT_LE(substep.event_end, receipt.events.size());
                double total = 0, h2 = 0, h3 = 0;
                std::vector<double> packets;
                for (size_t a = 0; a < atoms[p].size(); ++a)
                {
                    auto& atom = atoms[p][a];
                    const double transfer = node_capacity(p, atom) * concentration;
                    const auto& event = receipt.events[i * atoms[p].size() + a];
                    EXPECT_EQ(event.substep, i);
                    EXPECT_DOUBLE_EQ(event.duration, substep.duration);
                    ASSERT_EQ(event.transfers.size(), 1u);
                    packets.push_back(event.transfers[0].transfer.moles.at(0));
                    EXPECT_NEAR(packets.back(), transfer, 2e-12);
                    total += transfer;
                    atom.mass_over_scale += transfer / (atom.count * mass_scale);
                    h2 += atom.count * std::pow(atom.mass_over_scale, 2);
                    h3 += atom.count * std::pow(atom.mass_over_scale, 3);
                }
                EXPECT_NEAR(substep.final_shape.H2, h2, 2e-10 * h2);
                EXPECT_NEAR(substep.final_shape.H3, h3, 2e-10 * h3);
                const auto& slots = plan.configuration->stages.at(*plan.material_tail.paired_stage).material;
                const auto found = std::ranges::find(slots, R::MaterialTransferSlot{tracer, 0, p});
                ASSERT_NE(found, slots.end());
                const auto slot = static_cast<size_t>(found - slots.begin());
                const auto& accepted = plan.material_tail.substeps[i];
                EXPECT_EQ(accepted.paired_amounts.at(slot), packets);
                EXPECT_NEAR(accepted.amounts.at(slot), total, 2e-12);
                auto enthalpy = std::ranges::find_if(
                    thermal.material_events, [=](const auto& row) { return row.slot == slot && row.substep == i; });
                ASSERT_NE(enthalpy, thermal.material_events.end());
                const double temperature = thermal.anchors.at(enthalpy->stage + i).temperature.at(0);
                const double donor_h = liquid_u + 1000 * (temperature - T::T0) + thermal.absolute_pressure;
                EXPECT_NEAR(enthalpy->carried_energy, total * donor_h, 2e-6);
            }
        }
        EXPECT_NEAR(cell.phases[0].moles[0], stock, 2e-12);
        EXPECT_EQ(cell.phases[0].moles[1], solvent_moles);
    }
}

TEST(ReactingMixtureQbmmSubstepping, IntegratedTypedAndJointFilmUseEveryAcceptedFiniteLiquidPredecessor)
{
    for (const auto mode : {Evaluation::Integrated, Evaluation::Typed, Evaluation::Joint})
        check_film_substeps(mode);
}

TEST(ReactingMixtureQbmmSubstepping, ReturnedDissolvedMaterialIsAvailableOnlyInLaterThermalSubsteps)
{
    constexpr double transferred = .022; // Larger than the entire original 0.020 mol dissolved pool.
    const std::array donor_seed{R::QbmmAtom{1., 50.}};
    auto configure = [&](double tolerance)
    {
        return [=](Options& options)
        {
            source_controls(tolerance)(options);
            const auto& film = *options.qbmm_population;
            auto transfers = std::make_shared<R::InterphaseConfiguration>(*film.transfers);
            transfers->stages.front().material = {{tracer, 2, 0}, {tracer, 0, 1}};
            transfers->phase_enthalpy.at(2) = std::make_shared<const R::PartialMolarEnthalpyLaw>(
                R::PartialMolarEnthalpyLaw{{"ideal gas donor", 1, film.metadata.schema_fingerprint},
                    {250, 1200, 50000, 1000000}, [](double temperature, double, std::span<const double>)
                    {
                        return R::PartialMolarEnthalpyPoint{
                            {T::cp * temperature, T::cp * temperature}, {T::cp, T::cp}, {0, 0}};
                    }});
            const auto exchange = [](const R::MixtureQbmmJointContext& ctx)
            {
                std::vector<Operations> rows;
                for (const auto& family : ctx.families)
                {
                    const auto count = family.law_inputs.quadrature.active_atoms;
                    const double rate = (family.phase == 1 ? transferred / 3 : -transferred) / dt;
                    const std::vector<double> rates(count, rate);
                    const std::vector<R::QbmmTransferCounterpart> counterparts(
                        count, R::QbmmPhaseExchange{Phase::Dissolved});
                    rows.push_back(
                        R::qbmm_resize_from_mass_rate(family.family, family.law_inputs, rates, counterparts));
                }
                return rows;
            };
            options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
                R::MixtureQbmmProvider{film.metadata, transfers, {}, {}, exchange});
        };
    };
    Fixture legacy(Evaluation::Joint, false, false, configure(0.), donor_seed, substepped_film_contract());
    EXPECT_THROW((void) legacy.controller->preview(dt), std::exception);
    Fixture f(Evaluation::Joint, false, false, configure(.01), donor_seed, substepped_film_contract());
    const auto trial = f.controller->preview(dt);
    const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
    ASSERT_GE(n, 2u);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& cell = trial.state().cells[c];
        const auto& thermal = trial.final_thermal_evaluations()[c];
        ASSERT_EQ(thermal.material_events.size(), 2 * n);
        const auto& plan = trial.interphase_receipt()->cells[c];
        for (const auto& event : thermal.material_events)
        {
            ASSERT_TRUE(event.substep);
            const auto& slot = plan.configuration->stages.at(event.stage).material.at(event.slot);
            const double temperature = thermal.anchors.at(event.stage + *event.substep).temperature.at(slot.donor);
            const double donor_h = slot.donor == 0 ? liquid_u + 1000 * (temperature - T::T0) + thermal.absolute_pressure
                                                   : T::cp * temperature;
            EXPECT_NEAR(event.carried_energy, (transferred / n) * donor_h, 2e-6);
        }
        EXPECT_NEAR(cell.phases[0].moles[0], dissolved, 2e-14);
        EXPECT_NEAR(cell.phases[1].moles[0], 5 * mass_scale + transferred, 2e-14);
        EXPECT_NEAR(cell.phases[2].moles[0], 50 * mass_scale - transferred, 2e-14);
    }
}

void portable_continuation(bool moving)
{
    const char* writer = std::getenv("SIMPLEFLUID_QBMM_SUBSTEP_WRITE");
    const char* reader = std::getenv("SIMPLEFLUID_QBMM_SUBSTEP_READ");
    ASSERT_FALSE(writer && reader);
    std::vector<Observation> calls;
    const auto configure = integrated_controls(convergence_tolerance, 64, finite_stock_births(calls));
    Fixture reference(Evaluation::Integrated, moving, false, configure);
    check_partition(reference, ranks_variable);
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
    Fixture fresh(Evaluation::Integrated, moving, false, configure);
    const auto evaluated = calls.size();
    fresh.controller->restore_qbmm_restart(first);
    EXPECT_EQ(calls.size(), evaluated);
    EXPECT_EQ(fresh.motion_calls, 0);
    fresh.controller->advance(dt);
    if (fresh.mesh->num_owned_cells())
    {
        ASSERT_LT(evaluated, calls.size());
        EXPECT_EQ(calls[evaluated].time, dt);
        EXPECT_EQ(calls[evaluated].step, dt);
    }
    EXPECT_DOUBLE_EQ(fresh.flow->state().time, 2 * dt);
    EXPECT_DOUBLE_EQ(fresh.flow->state().thermodynamic_pressure, reference.flow->state().thermodynamic_pressure);
    for (size_t c = 0; c < fresh.mesh->num_owned_cells(); ++c)
    {
        EXPECT_EQ(inventories(fresh.flow->state().cells[c]), inventories(reference.flow->state().cells[c]));
        EXPECT_EQ(fresh.mesh->cell_volume(c), reference.mesh->cell_volume(c));
        for (size_t p = 0; p < fresh.flow->state().cells[c].phases.size(); ++p)
            EXPECT_DOUBLE_EQ(fresh.flow->state().cells[c].phases[p].internal_energy,
                reference.flow->state().cells[c].phases[p].internal_energy);
    }
    for (auto [tol, cap] : {std::pair{.01, 64}, std::pair{convergence_tolerance, 32}, std::pair{0., 64}})
    {
        Fixture mismatch(
            Evaluation::Integrated, moving, false, integrated_controls(tol, cap, finite_stock_births(calls)));
        const auto previous = mismatch.controller->serialize_qbmm_restart();
        const auto observed = calls.size();
        EXPECT_THROW(mismatch.controller->restore_qbmm_restart(first), std::exception);
        EXPECT_EQ(calls.size(), observed);
        EXPECT_EQ(mismatch.controller->serialize_qbmm_restart(), previous);
    }
}

TEST(ReactingMixtureQbmmSubstepping, FixedPortableContinuation)
{
    portable_continuation(false);
}

TEST(ReactingMixtureQbmmSubstepping, PlanarPortableContinuation)
{
    portable_continuation(true);
}
} // namespace
