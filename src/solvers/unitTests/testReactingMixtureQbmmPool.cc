/** Coupled fixed-grid BulkGas conversion with independent extensive and caloric oracles. */
#include "QbmmPoolTestSupport.hh"

namespace
{
using namespace SimpleFluid;
using namespace Testing::QbmmPool;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(ReactingMixtureQbmmPool, BidirectionalSpectrumExactAbsenceAndRegeneration)
{
    Fixture f;
    f.check_ranks();
    f.start();
    const std::array<std::array<double, 4>, 4> moments{
        {{2, 2, 2, 2}, {4, 5, 6.5, 8.75}, {0, 0, 0, 0}, {10, 5, 2.5, 1.25}}};
    const std::array<std::array<double, 4>, 4> gross{
        {{1, 3, 9, 27}, {2, 3, 4.5, 6.75}, {4, 5, 6.5, 8.75}, {10, 5, 2.5, 1.25}}};
    for (size_t i = 0; i < moments.size(); ++i)
    {
        const auto trial = f.controller->preview(dt);
        const auto report = f.controller->qbmm_report(trial);
        const auto& diag = trial.diagnostics();
        ASSERT_TRUE(diag.flow.pool_conversion);
        const auto& dir = diag.flow.pool_conversion->directions[i % 2];
        near(dir.count, gross[i][0]);
        near(dir.mass_kg, gross[i][1] * scale);
        near(dir.H2, gross[i][2]);
        near(dir.H3, gross[i][3]);
        EXPECT_EQ(dir.species_moles, Fixture::moles(gross[i][1]));
        near(dir.carried_enthalpy_j, gross[i][1] * (4 * 24 + 2 * 40) * T0);
        EXPECT_EQ(diag.flow.pool_conversion->directions[1 - i % 2].count, 0);
        EXPECT_GT(diag.flow.pool_conversion->law_evaluations, 0u);
        near(diag.external_mass, 0);
        near(diag.transfer_energy, 0);
        near(diag.external_heat, 0);
        near(trial.state().thermodynamic_pressure, f.pressure);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& cell = trial.state().cells[c];
            const auto& row = report.cells[c];
            const auto& family = *row.families[1];
            EXPECT_EQ(family.pool_conversion_law_evaluated, true);
            for (size_t k = 0; k < 4; ++k)
                near(family.moments[k], moments[i][k], 2e-13);
            EXPECT_EQ(cell.phases[1].moles, Fixture::moles(moments[i][1]));
            EXPECT_EQ(cell.phases[2].moles, Fixture::moles(5 - moments[i][1]));
            EXPECT_FALSE(cell.phase_numbers[2]);
            EXPECT_FALSE(cell.qbmm_shape[2]);
            EXPECT_FALSE(row.families[2]);
            const double gas_void = 1 - f.setup.liquid_fraction;
            near(row.gas_volume_fraction, gas_void);
            near(row.dispersed_gas_volume_fraction, gas_void * moments[i][1] / 5);
            near(row.bulk_gas_pool_volume_fraction, gas_void * (5 - moments[i][1]) / 5);
            EXPECT_EQ(row.passive_gas_volume_fraction, 0);
            near(row.temperature_K.value(), T0);
            if (i == 2)
            {
                EXPECT_EQ(family.quadrature.status, R::InversionStatus::Absent);
                EXPECT_FALSE(family.temperature_K);
                EXPECT_EQ(cell.phases[1].internal_energy, 0);
            }
            if (i % 2 == 1)
            {
                EXPECT_EQ(cell.phases[2].temperature, 0);
                EXPECT_EQ(cell.phases[2].internal_energy, 0);
            }
        }
        const auto evaluations = f.calls;
        f.controller->commit(trial);
        EXPECT_EQ(f.calls, evaluations);
    }
}

TEST(ReactingMixtureQbmmPool, PressureRootReplaysEachDonorPartialAndPhaseVolumeWork)
{
    Fixture f({.different_eos = true});
    f.start();
    const auto trial = f.controller->preview(dt);
    const double liquid_capacity = f.setup.liquid_fraction * liquid_cv;
    const double initial_capacity = liquid_capacity + 5 * 128;
    const double final_capacity = liquid_capacity + 2 * 128 + 3 * 192;
    const double t = initial_capacity * T0 / final_capacity;
    const double pressure = f.pressure * t / T0;
    const double anchor =
        (initial_capacity * T0 + pressure * (1 - f.setup.liquid_fraction)) / (initial_capacity + 30 * gas_R);
    near(trial.state().thermodynamic_pressure, pressure);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& thermal = trial.final_thermal_evaluations()[c];
        const auto& before = f.flow->state().cells[c];
        const auto& after = trial.state().cells[c];
        const auto& config = *trial.interphase_receipt()->cells[c].configuration;
        double carried = 0;
        size_t positive = 0;
        for (const auto& event : thermal.material_events)
        {
            const auto& slot = config.stages[event.stage].material[event.slot];
            if (event.carried_energy == 0)
                continue;
            EXPECT_EQ(slot.donor, 1u);
            EXPECT_EQ(slot.recipient, 2u);
            double h = anchor * (slot.species == A ? 24. : 40.);
            near(event.molar_enthalpy, h);
            near(event.carried_energy, h * (slot.species == A ? 12. : 6.));
            carried += event.carried_energy;
            ++positive;
        }
        EXPECT_EQ(positive, 2u);
        const std::array old_v{f.setup.liquid_fraction, 1 - f.setup.liquid_fraction, 0.};
        const std::array new_v{f.setup.liquid_fraction, 12 * gas_R * t / pressure, 18 * gas_R * t / pressure};
        const std::array material{0., -carried, carried};
        double work = 0., redistributed = 0.;
        for (size_t p = 0; p < 3; ++p)
        {
            const double pdv = -pressure * (new_v[p] - old_v[p]);
            near(after.phases[p].temperature, t);
            near(thermal.volumes[p], new_v[p]);
            near(thermal.raw_material_enthalpy[p], material[p]);
            const double du = after.phases[p].internal_energy - before.phases[p].internal_energy;
            near(thermal.raw_equilibrium_redistribution[p], du - material[p] - pdv);
            work += pdv;
            redistributed += thermal.raw_equilibrium_redistribution[p];
        }
        EXPECT_NEAR(work, 0., 1e-8);
        EXPECT_NEAR(redistributed, 0., 1e-8);
        EXPECT_GT(std::abs(thermal.raw_equilibrium_redistribution[2]), 1.);
    }
}

TEST(ReactingMixtureQbmmPool, TypedAndJointSubstepsDrainEachActualPredecessor)
{
    std::optional<Flow::QbmmPoolDirectionDiagnostics> typed_totals;
    size_t typed_steps = 0;
    for (bool joint : {false, true})
    {
        SCOPED_TRACE(joint ? "joint provider" : "typed provider");
        Fixture f;
        f.check_ranks();
        const auto& family = *f.controls.qbmm_families.at(1);
        const auto& contract = family.closure_versions().laws.at(0);
        const auto drain = [&](const R::QbmmLawInputs& input)
        {
            ++f.calls;
            const double fraction = .5 * input.time_step_s.value() / dt;
            return R::qbmm_pool_drain_from_fraction(
                family, input, *f.pool, std::vector<double>(input.quadrature.active_atoms, fraction));
        };
        const R::ThermalLawIdentity provider{"pool kinetics", 1, f.catalog->fingerprint()};
        if (joint)
            f.options.qbmm_population = std::make_shared<const R::MixtureQbmmProvider>(
                R::MixtureQbmmProvider{provider, f.slots, {}, {}, [&](const R::MixtureQbmmJointContext& ctx)
                    {
                        std::vector<Operations> rows;
                        for (const auto& row : ctx.families)
                            rows.push_back(drain(row.law_inputs));
                        return rows;
                    }});
        else
        {
            auto typed = std::make_shared<const R::MixtureQbmmLaw>(R::MixtureQbmmLaw{contract, drain});
            f.options.qbmm_population =
                std::make_shared<const R::MixtureQbmmProvider>(R::MixtureQbmmProvider{provider, f.slots, {}, {typed}});
        }
        f.options.qbmm_source_substep_relative_tolerance = .005;
        Coupler controller(f.flow, f.options);
        const auto trial = controller.preview(dt);
        const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
        ASSERT_GT(n, 1u);
        const auto report = controller.qbmm_report(trial);
        const auto& totals = report.pool_conversion.value().directions[0];
        const double retained = 1 - .5 / n;
        constexpr std::array initial{3., 5., 11., 29.};
        const double lost = 1 - std::pow(retained, n);
        near(totals.count, initial[0] * lost);
        near(totals.mass_kg, scale * initial[1] * lost);
        near(totals.H2, initial[2] * lost);
        near(totals.H3, initial[3] * lost);
        near(totals.carried_enthalpy_j, 5 * lost * (4 * 24 + 2 * 40) * T0);
        EXPECT_EQ(report.pool_conversion->law_evaluations, 1u);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& receipt = trial.local_qbmm_sources()[c][1].value();
            ASSERT_EQ(receipt.substeps.size(), n);
            ASSERT_EQ(receipt.events.size(), 2 * n);
            EXPECT_EQ(report.cells[c].families[1]->pool_conversion_law_evaluated, true);
            for (size_t i = 0; i < n; ++i)
            {
                const auto& stage = receipt.substeps[i];
                const double fraction = std::pow(retained, i + 1);
                near(stage.duration, dt / n);
                EXPECT_EQ(stage.event_end, 2 * (i + 1));
                EXPECT_EQ(receipt.events.at(2 * i).provider, joint ? provider.identity : contract.identity);
                near(stage.final_count, initial[0] * fraction);
                near(stage.final_shape.H2, initial[2] * fraction);
                near(stage.final_shape.H3, initial[3] * fraction);
                for (size_t s = 0; s < stage.final_moles.size(); ++s)
                    near(stage.final_moles[s], Fixture::moles(initial[1])[s] * fraction);
            }
            for (size_t s = 0; s < totals.species_moles.size(); ++s)
            {
                const double amount = Fixture::moles(initial[1])[s] * lost;
                near(totals.species_moles[s], amount);
                near(trial.state().cells[c].phases[2].moles[s], amount);
            }
        }
        if (joint)
        {
            EXPECT_EQ(n, typed_steps);
            EXPECT_EQ(totals, typed_totals.value());
        }
        else
        {
            typed_steps = n;
            typed_totals = totals;
        }
    }
}

TEST(ReactingMixtureQbmmPool, GasFractionAndTracesUsePhysicalVolumeAtPositiveLiquidOccupancy)
{
    for (const double liquid : {.75, .5, 1. / 16., 1. / 65536.})
    {
        SCOPED_TRACE(liquid);
        Fixture f({.liquid_fraction = liquid, .pool_stock = 1, .declare_liquid_law = true});
        f.events = [&](const Context& ctx)
        {
            const auto& input = *ctx.law_inputs;
            near(input.gas_volume_fraction.value(), 1 - liquid);
            near(input.carrier_liquid_volume_m3.value(), liquid);
            near(input.carrier_liquid_temperature_K.value(), T0);
            near(input.carrier_liquid_concentration_mol_per_m3.value()[2], 1.);
            const double pool_volume = (1 - liquid) / 6.;
            near(input.bulk_gas_pool_volume_m3.value(), pool_volume);
            near(input.bulk_gas_pool_temperature_K.value(), T0);
            near(input.bulk_gas_pool_concentration_mol_per_m3.value()[0], 4. / pool_volume);
            EXPECT_EQ(input.bulk_gas_pool_moles.value(), Fixture::moles(1));
            return Operations{};
        };
        f.start();
        const auto trial = f.controller->preview(dt);
        EXPECT_GT(f.sum(f.liquid_calls), 0.);
        const auto report = f.controller->qbmm_report(trial);
        for (const auto& cell : report.cells)
        {
            near(cell.gas_volume_fraction, 1 - liquid);
            near(cell.dispersed_gas_volume_fraction, 5 * (1 - liquid) / 6);
            near(cell.bulk_gas_pool_volume_fraction, (1 - liquid) / 6);
            EXPECT_EQ(cell.families[1]->pool_conversion_law_evaluated, true);
        }
    }
}

TEST(ReactingMixtureQbmmPool, SharedOriginalSpeciesAndAtomDebitsRejectWithoutPublishing)
{
    Fixture f({.pool_stock = 1, .two_families = true});
    f.check_ranks();
    for (auto id : {A, B})
        f.slots->stages.front().material.push_back({id, 1, 0});
    f.start();
    const auto unchanged = f.controller->serialize_qbmm_restart();
    for (int failure = 0; failure < 7; ++failure)
    {
        SCOPED_TRACE(failure);
        f.events = [&](const Context& in) -> Operations
        {
            if (failure == 0)
                return f.birth(in, .75, 1.);
            if (failure == 1)
                return in.phase == 1 ? f.drain(in, {0, 1}) : f.birth(in, 2, 1);
            if (in.phase != 1)
                return {};
            if (failure == 2)
            {
                auto ops = f.drain(in, {.25, 0.});
                ops.emplace_back(R::QbmmCoalescence{0, 0, .5});
                ops.emplace_back(
                    R::QbmmDisappearance{0, .75, {R::QbmmPhaseExchange{Phase::Dissolved}, Fixture::moles(.75)}});
                return ops;
            }
            auto ops = f.drain(in, {0, 1});
            auto& packet = std::get<R::QbmmDisappearance>(ops[0]).transfer;
            auto& tag = std::get<R::QbmmPhaseExchange>(packet.counterpart).resolved_pool;
            if (failure == 3)
                tag.reset();
            if (failure == 4)
                tag->direction = Direction::PoolToDispersed;
            if (failure == 5)
                tag->pool_fingerprint++;
            if (failure == 6)
            {
                packet.moles[0] -= 2;
                packet.moles[1] += 1;
            }
            return ops;
        };
        EXPECT_THROW((void) f.controller->preview(dt), std::exception);
        EXPECT_EQ(f.controller->serialize_qbmm_restart(), unchanged);
    }
    f.events = [&](const Context& in) { return f.birth(in, .5, 1); };
    const auto trial = f.controller->preview(dt);
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(cell.phases[1].moles, Fixture::moles(5.5));
        EXPECT_EQ(cell.phases[2].moles, Fixture::moles(.5));
        EXPECT_EQ(cell.phases[3].moles, Fixture::moles(0));
    }
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), unchanged);

    // Exactly two original unit parents. Coalescence consumes 1.5 and the other two uses split the remaining half.
    constexpr double dissolved = 1. / 1024.;
    for (bool overdraw : {true, false})
    {
        f.events = [&](const Context& in) -> Operations
        {
            if (in.phase != 1)
                return {};
            const double loss = overdraw ? std::nextafter(dissolved, 1.) : dissolved;
            auto ops = f.drain(in, {(.5 - dissolved) / 2, 0.});
            ops.emplace_back(R::QbmmCoalescence{0, 0, .75});
            ops.emplace_back(
                R::QbmmDisappearance{0, loss, {R::QbmmPhaseExchange{Phase::Dissolved}, Fixture::moles(loss)}});
            return ops;
        };
        if (overdraw)
        {
            EXPECT_THROW((void) f.controller->preview(dt), std::exception);
            EXPECT_EQ(f.controller->serialize_qbmm_restart(), unchanged);
            continue;
        }
        const auto exact = f.controller->preview(dt);
        for (const auto& cell : exact.state().cells)
        {
            EXPECT_EQ(cell.phase_numbers[1], 1.75);
            EXPECT_EQ(cell.qbmm_shape[1], (R::QbmmShapeState{12., 33.}));
            EXPECT_EQ(cell.phases[1].moles, Fixture::moles(4.5));
            EXPECT_EQ(cell.phases[3].moles, Fixture::moles(1.5 - dissolved));
            near(cell.phases[0].moles[0], 4 * dissolved);
            near(cell.phases[0].moles[1], 2 * dissolved);
        }
    }
}

TEST(ReactingMixtureQbmmPool, PassiveBulkGasRetainsItsStockAndCannotSupplyPoolBirths)
{
    Fixture f({.pool_stock = 1, .passive = true});
    const double gas_fraction = 1 - f.setup.liquid_fraction;
    f.events = [&](const Context& in)
    {
        const auto& inputs = *in.law_inputs;
        EXPECT_EQ(inputs.bulk_gas_pool_moles.value(), Fixture::moles(1));
        near(inputs.bulk_gas_pool_volume_m3.value(), gas_fraction / 8);
        near(inputs.gas_volume_fraction.value(), gas_fraction);
        return f.birth(in, 2, 1);
    };
    f.start();
    const auto before = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
    f.events = [&](const Context& in) { return f.birth(in, 1, 1); };
    const auto trial = f.controller->preview(dt);
    const auto report = f.controller->qbmm_report(trial);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& cell = trial.state().cells[c];
        EXPECT_EQ(cell.phases[1].moles, Fixture::moles(6));
        EXPECT_EQ(cell.phases[2].moles, Fixture::moles(0));
        EXPECT_EQ(cell.phases[3].moles, Fixture::moles(2));
        EXPECT_FALSE(cell.phase_numbers[3]);
        EXPECT_FALSE(cell.qbmm_shape[3]);
        const auto& row = report.cells[c];
        near(row.gas_volume_fraction, gas_fraction);
        near(row.dispersed_gas_volume_fraction, .75 * gas_fraction);
        EXPECT_EQ(row.bulk_gas_pool_volume_fraction, 0.);
        near(row.passive_gas_volume_fraction, .25 * gas_fraction);
    }
}

TEST(ReactingMixtureQbmmPool, EqualAndOppositeConversionsRetainDirectionalCorrectorActivity)
{
    Fixture f({.pool_stock = 1});
    f.events = [&](const Context& in)
    {
        const double n = in.stage.corrector == 0 ? .25 : .5;
        auto out = f.drain(in, {n / 2., 0});
        auto back = f.birth(in, n, 1.);
        out.push_back(std::move(back.front()));
        return out;
    };
    f.start();
    const auto trial = f.controller->preview(dt);
    EXPECT_GE(trial.diagnostics().correctors, 3);
    const auto& diag = trial.diagnostics().flow.pool_conversion.value();
    for (const auto& direction : diag.directions)
    {
        near(direction.count, .5);
        near(direction.mass_kg, .5 * scale);
        near(direction.H2, .5);
        near(direction.H3, .5);
        EXPECT_EQ(direction.species_moles, Fixture::moles(.5));
        near(direction.carried_enthalpy_j, .5 * (4 * 24 + 2 * 40) * T0);
    }
    for (const auto& cell : trial.state().cells)
    {
        EXPECT_EQ(cell.phases[1].moles, Fixture::moles(5));
        EXPECT_EQ(cell.phases[2].moles, Fixture::moles(1));
        EXPECT_EQ(cell.phase_numbers[1], 3);
        EXPECT_EQ(cell.qbmm_shape[1], (R::QbmmShapeState{11, 29}));
    }
}

TEST(ReactingMixtureQbmmPool, FailedPreviewPreservesAcceptedConversionAndACompetingTrial)
{
    Fixture f;
    f.start();
    f.controller->advance(dt);
    const auto checkpoint = f.controller->checkpoint();
    const auto before = f.controller->serialize_qbmm_restart();
    const auto accepted = f.controller->diagnostics().flow.pool_conversion;
    const auto competing = f.controller->preview(dt);
    f.events = [&](const Context& in) { return f.birth(in, 8, 1); };
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(before, f.controller->serialize_qbmm_restart());
    EXPECT_EQ(f.controller->diagnostics().flow.pool_conversion, accepted);
    const auto evaluated = f.calls;
    f.controller->commit(competing);
    near(f.controller->diagnostics().flow.pool_conversion.value().directions[1].mass_kg, 3 * scale);
    f.controller->restore(checkpoint);
    EXPECT_EQ(f.calls, evaluated);
    EXPECT_EQ(f.flow->state().time, dt);
    EXPECT_EQ(f.controller->diagnostics().flow.pool_conversion, accepted);
}

TEST(ReactingMixtureQbmmPool, DirectionalPoolWitnessDetectsChangingCounterpartAtIdenticalFamilyGross)
{
    Fixture f({.pool_stock = 1});
    f.events = [&](const Context& in)
    {
        auto death = f.drain(in, {.25, 0.});
        auto birth = f.birth(in, .5, 1.);
        // Identical parents and products on every corrector. The balanced external packets carry zero energy/impulse.
        if (in.stage.corrector == 0)
        {
            std::get<R::QbmmDisappearance>(death.front()).transfer.counterpart = R::QbmmReservoirExchange{"out"};
            std::get<R::QbmmBirth>(birth.front()).atoms.front().transfer.counterpart = R::QbmmReservoirExchange{"in"};
        }
        death.push_back(std::move(birth.front()));
        return death;
    };
    f.start();
    const auto trial = f.controller->preview(dt);
    const auto& flow = trial.diagnostics().flow;
    const auto& trace = flow.gross_residual_evaluations;
    ASSERT_GE(trace.size(), 3u);
    ASSERT_TRUE(trace.at(1).winner);
    EXPECT_EQ(trace[1].winner->channel_metadata.stage, Flow::GrossResidualStage::QbmmPoolConversion);
    EXPECT_EQ(trace[1].winner->previous_packet, 0.);
    EXPECT_GT(trace[1].winner->current_packet, 0.);
    EXPECT_GT(trace[1].normalized_residual, 1.);
    EXPECT_EQ(trace.back().normalized_residual, 0.);
    for (const auto& direction : flow.pool_conversion.value().directions)
    {
        EXPECT_EQ(direction.count, .5);
        EXPECT_EQ(direction.species_moles, Fixture::moles(.5));
    }
    EXPECT_EQ(trial.diagnostics().external_mass, 0.);
    EXPECT_EQ(trial.diagnostics().transfer_energy, 0.);
}

TEST(ReactingMixtureQbmmPool, DryLiquidCallbacksStayIdleWhileGasChemistryAndHeatAdvance)
{
    for (bool liquid_rate : {false, true})
    {
        Fixture f({.liquid_fraction = 0,
            .pool_stock = 1,
            .two_cells = true,
            .dry_chemistry = true,
            .declare_liquid_law = true});
        int rate_calls = 0, contact_calls = 0;
        f.events = [](const Context& in)
        {
            EXPECT_EQ(in.law_inputs->carrier_liquid_volume_m3, 0.);
            EXPECT_FALSE(in.law_inputs->carrier_liquid_concentration_mol_per_m3);
            EXPECT_FALSE(in.law_inputs->carrier_liquid_temperature_K);
            EXPECT_TRUE(in.law_inputs->bulk_gas_pool_concentration_mol_per_m3);
            return Operations{};
        };
        f.options.source_identity = "dry rate and heat";
        f.options.heat_phase = 0; // The nominal liquid heat recipient is absent.
        f.options.source = [&](const R::CoupledReactionContext& context, double)
        {
            EXPECT_EQ(context.reaction.environment.liquidVolume, 0.);
            EXPECT_GT(context.reaction.environment.gasVolume.value_or(0.), 0.);
            EXPECT_GT(context.reaction.environment.gasVolume.value(), 0.);
            R::ReactionRateProvider rate;
            rate.identity = "dimerization";
            rate.schemaFingerprint = f.catalog->fingerprint();
            rate.schemaRevision = f.catalog->revision();
            rate.validityDescription = "positive gas volume";
            rate.parameterUnits = "mol/(m3 s)";
            rate.requiresLiquid = liquid_rate;
            rate.validate = [&](const auto&) { ++rate_calls; };
            rate.evaluate = [&](const auto& rxn)
            {
                ++rate_calls;
                EXPECT_EQ(rxn.environment.liquidVolume, 0.);
                return std::vector<R::ReactionRate>{{7, 1., R::SourceVolumeBasis::BulkVolume}};
            };
            R::ReactionSourceInput result;
            result.rates = rate;
            result.heat.rate = R::VolumetricHeatRate{64., R::SourceVolumeBasis::BulkVolume};
            return result;
        };
        R::MixtureFaceTransportOptions contacts;
        contacts.identity = "liquid Fourier";
        contacts.physical_phase = 0;
        contacts.schema_fingerprint = f.catalog->fingerprint();
        R::MixtureDiffusionLaw diffusion;
        diffusion.binding.identity = "dry liquid solute";
        diffusion.binding.schemaFingerprint = f.catalog->fingerprint();
        diffusion.binding.counterSpecies = solvent;
        diffusion.primary_species = {A};
        diffusion.molar_conductance = [&](const auto&)
        {
            ++contact_calls;
            return std::vector<double>{1., 0., 0.};
        };
        diffusion.partial_molar_enthalpy = [&](auto, const auto&, const auto&)
        {
            ++contact_calls;
            return 0.;
        };
        contacts.diffusion = std::move(diffusion);
        contacts.fourier = R::MixtureFourierLaw{"dry contact", 1, f.catalog->fingerprint(), [&](const auto&)
            {
                ++contact_calls;
                return 1.;
            }};
        f.options.face_transport = {contacts};
        auto heat_exchange = std::make_shared<R::InterphaseConfiguration>();
        heat_exchange->metadata = {"liquid contact heat stage", 1, f.catalog->fingerprint()};
        heat_exchange->phase_enthalpy.resize(f.eos.size());
        heat_exchange->stages = {{"liquid to pool", R::InterphaseStageKind::Heat, {}, {{0, 2}}}};
        f.options.interphase = std::make_shared<const R::MixtureInterphaseEventProvider>(
            R::MixtureInterphaseEventProvider{heat_exchange->metadata, heat_exchange, [&](const auto&)
                {
                    ++contact_calls;
                    return std::vector<std::vector<double>>{{1.}};
                }});
        f.start();
        const auto trial = f.controller->preview(dt);
        EXPECT_EQ(f.liquid_calls, 0);
        EXPECT_EQ(contact_calls, 0);
        if (liquid_rate)
            EXPECT_EQ(rate_calls, 0);
        else
            EXPECT_GT(f.sum(rate_calls), 0);
        near(trial.diagnostics().reaction_extents.at(0), liquid_rate ? 0. : dt);
        near(trial.diagnostics().external_heat, 64 * dt);
        const auto report = f.controller->qbmm_report(trial);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& cell = trial.state().cells[c];
            const double v = f.mesh->cell_volume(c), extent = liquid_rate ? 0. : v * dt;
            near(cell.phases[2].moles[0], 4 * v - 2 * extent);
            near(cell.phases[2].moles[1], 2 * v + extent);
            EXPECT_EQ(cell.phases[0].temperature, 0);
            near(report.cells[c].gas_volume_fraction, 1);
            EXPECT_GT(cell.phases[2].temperature, T0);
        }
    }
}

TEST(ReactingMixtureQbmmPool, PackedAndScalarTransportCarryConversionAcrossPartitionFaces)
{
    Fixture packed({.circulating = true}), scalar({.circulating = true, .batched = false});
    packed.check_ranks(false);
    auto run = [](Fixture& f)
    {
        f.events = [&f](const Context& in)
        { return f.drain(in, std::vector<double>(in.quadrature.active_atoms, .125)); };
        f.start();
        return f.controller->preview(dt);
    };
    const auto a = run(packed), b = run(scalar);
    EXPECT_GT(a.diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
    EXPECT_EQ(b.diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
    const auto comm = packed.mesh->owned_cell_map()->getComm();
    double interiors = 0, crossing = 0, ingress = 0, total_pool = 0, total_number = 0;
    for (size_t face = 0; face < a.owned_phase_transport().size(); ++face)
    {
        if (packed.mesh->is_geometry_exterior_face(face))
            continue;
        const double flux = a.owned_phase_transport()[face].qbmm_flux.at(1)->population_number_flux;
        near(flux, b.owned_phase_transport()[face].qbmm_flux.at(1)->population_number_flux);
        interiors += std::abs(flux);
        if (static_cast<size_t>(packed.mesh->owner_cell(face)) >= packed.mesh->num_owned_cells() ||
            static_cast<size_t>(packed.mesh->neighbor_cell(face)) >= packed.mesh->num_owned_cells())
            crossing += std::abs(flux);
    }
    EXPECT_GT(packed.sum(interiors), 1e-6);
    if (comm->getSize() > 1)
        EXPECT_GT(packed.sum(crossing), 1e-6);
    near(a.state().thermodynamic_pressure, b.state().thermodynamic_pressure);
    for (size_t c = 0; c < a.state().cells.size(); ++c)
    {
        const auto& ca = a.state().cells[c];
        const auto& cb = b.state().cells[c];
        for (size_t p = 0; p < ca.phases.size(); ++p)
        {
            for (size_t s = 0; s < 3; ++s)
                near(ca.phases[p].moles[s], cb.phases[p].moles[s]);
            near(ca.phases[p].temperature, cb.phases[p].temperature);
            near(ca.phases[p].internal_energy, cb.phases[p].internal_energy);
        }
        near(*ca.phase_numbers[1], *cb.phase_numbers[1]);
        near(ca.qbmm_shape[1]->H2, cb.qbmm_shape[1]->H2);
        near(ca.qbmm_shape[1]->H3, cb.qbmm_shape[1]->H3);
        total_number += *ca.phase_numbers[1];
        total_pool += ca.phases[2].moles[0];
        if (packed.mesh->cell_centroid(c).x > .5)
            ingress += *ca.phase_numbers[1];
    }
    EXPECT_GT(packed.sum(ingress), 1e-6);
    near(packed.sum(total_number), 1.5 * .875);
    near(packed.sum(total_pool), 10 * (1 + .125));
    near(a.diagnostics().flow.pool_conversion->directions[0].mass_kg, 2.5 * .125 * scale);
}

TEST(ReactingMixtureQbmmPool, ResidualDissolvedStockWithoutSolventRejectsAtSetupAndAfterChemistry)
{
    Fixture invalid;
    invalid.check_ranks();
    auto state = invalid.flow->state();
    for (auto& cell : state.cells)
    {
        auto& liquid = cell.phases[0];
        // Equal molar masses and a composition-independent liquid EOS preserve U, density and occupied volume.
        liquid.moles[0] = std::exchange(liquid.moles[2], 0.);
    }
    invalid.flow->initialize(std::move(state));
    const auto initial = invalid.flow->serialize_qbmm_restart();
    EXPECT_THROW(invalid.start(), std::exception);
    EXPECT_EQ(invalid.calls, 0);
    EXPECT_EQ(invalid.flow->serialize_qbmm_restart(), initial);

    Fixture f;
    f.options.source_identity = "solvent removal with residual solute";
    f.options.source = [](const R::CoupledReactionContext& context, double)
    {
        const auto& rxn = context.reaction;
        const double stock = rxn.state.amount(rxn.catalog, solvent, Phase::Liquid);
        R::ReactionSourceInput input;
        input.externalTransfers = {{"solvent withdrawal", solvent, Phase::Liquid, -stock, -stock},
            {"solute supply", A, Phase::Dissolved, stock, stock}};
        return input;
    };
    f.start();
    const auto before = f.controller->serialize_qbmm_restart();
    EXPECT_THROW((void) f.controller->preview(dt), std::exception);
    EXPECT_EQ(f.calls, 0);
    EXPECT_EQ(f.controller->serialize_qbmm_restart(), before);
}

TEST(ReactingMixtureQbmmPool, UnsupportedModelsAndUnpermittedSlotsFailCollectively)
{
    using M = R::QbmmModelOptions;
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    for (int mode = 0; mode < 8; ++mode)
    {
        SCOPED_TRACE(mode);
        auto options = f.controls;
        if (comm->getRank() == comm->getSize() - 1)
        {
            auto& model = options.qbmm_model;
            if (mode == 0)
                model.family_transport = M::FamilyTransport::KineticEuler;
            if (mode == 1 || mode == 2)
                model.family_transport = M::FamilyTransport::RealizableLinearSSPRK2;
            if (mode == 2)
                model.linear_slope_limiter = M::LinearSlopeLimiter::TvdMc;
            if (mode == 3)
                model.velocity = M::Velocity::NodeDependentSlip;
            if (mode == 4)
                model.velocity = M::Velocity::IndependentPhaseMomentum;
            if (mode == 5)
                model.transport.time = FVM::ScalarTimeScheme::BDF2;
            if (mode == 6)
                model.transport.convection = FVM::ScalarConvectionScheme::BoundedLinearUpwind;
            if (mode == 7)
                options.qbmm_bulk_gas_pool.reset();
        }
        EXPECT_THROW((void) Flow(f.mesh, f.eos, options), std::exception);
    }
    int direct_calls = 0;
    const auto direct = std::make_shared<const Flow::SourceStageProvider>(
        Flow::SourceStageProvider{"direct acceleration", 1, f.catalog->fingerprint(),
            [&](const auto&)
            {
                ++direct_calls;
                return Flow::SourceStageResult{};
            },
            0., 0, Flow::Vector{0, 0, -1}});
    EXPECT_THROW((void) f.flow->preview_with_sources(dt, direct), std::exception);
    EXPECT_EQ(direct_calls, 0);
    Flow::PlanarMotion motion(f.mesh);
    EXPECT_THROW((void) f.flow->preview_ale(dt, motion, 1.001), std::exception);
    EXPECT_FALSE(motion.has_active_trial());
    int motion_calls = 0;
    f.options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(
        R::MixturePlanarMotionProvider{f.mesh, {}, "prescribed", 1, [&](double, double)
            {
                ++motion_calls;
                return 1.;
            }});
    for (const auto gravity : {Flow::Vector{}, Flow::Vector{0, 0, -1}})
    {
        f.options.body_acceleration = gravity;
        EXPECT_THROW(f.start(), std::exception);
        EXPECT_EQ(motion_calls, 0);
    }
    Fixture wrong({.direction = Direction::DispersedToPool});
    if (comm->getRank() == comm->getSize() - 1)
        wrong.slots->stages[0].material.push_back({A, 2, 1});
    EXPECT_THROW(wrong.start(), std::exception);
    Fixture missing;
    missing.slots->stages[0].material.erase(missing.slots->stages[0].material.begin());
    EXPECT_THROW(
        {
            missing.start();
            (void) missing.controller->preview(dt);
        },
        std::exception);
}

// Drain one eighth of each transported node and buy a monodisperse birth from original pool stock.
void start_exchange(Fixture& f, double reservoir_count_per_m3 = 0.)
{
    f.events = [&f, reservoir_count_per_m3](const Context& in)
    {
        if (f.setup.supplied_slip)
        {
            const auto& drift = f.flow->state().cells.at(in.owned_cell).drift_velocity.at(in.phase);
            near(in.law_inputs->slip_speed_m_per_s.value(), drift.norm(), 2e-14);
        }
        auto out = f.drain(in, std::vector<double>(in.quadrature.active_atoms, .125));
        auto back = f.birth(in, in.law_inputs->bulk_volume_m3 / 16, 1.5);
        out.push_back(std::move(back.front()));
        if (reservoir_count_per_m3 > 0)
        {
            const double count = reservoir_count_per_m3 * in.law_inputs->bulk_volume_m3;
            const auto n = Fixture::moles(count);
            const R::QbmmReservoirExchange feed{
                "external feed", Fixture::gas_capacity(n, 1.) * T0, {-.5 * count, .25 * count, .75 * count}};
            out.emplace_back(R::QbmmBirth{{{{count, 1.}, {feed, n}}}});
        }
        return out;
    };
    f.start();
}

double phase_mass(const std::vector<double>& moles)
{
    return moles[0] + 2 * moles[1] + moles[2];
}

void audit_exchange(const Fixture& f, const Coupler::Trial& trial, double reservoir_count = 0.)
{
    const auto& diag = trial.diagnostics().flow;
    const auto& directions = diag.pool_conversion.value().directions;
    std::array<double, 4> old{}, now{};
    std::array<double, 3> species_delta{};
    std::array<double, 2> enthalpy{};
    Flow::Vector delta_momentum{};
    double old_mass = 0, delta_energy = 0;
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& before = f.flow->state().cells[c];
        const auto& after = trial.state().cells[c];
        const auto accumulate = [](const Flow::CellState& cell, std::array<double, 4>& v)
        {
            v[0] += *cell.phase_numbers[1];
            v[1] += phase_mass(cell.phases[1].moles) / scale;
            v[2] += cell.qbmm_shape[1]->H2;
            v[3] += cell.qbmm_shape[1]->H3;
        };
        accumulate(before, old);
        accumulate(after, now);
        EXPECT_EQ(after.drift_velocity, before.drift_velocity);
        EXPECT_FALSE(after.phase_numbers[2]);
        EXPECT_FALSE(after.qbmm_shape[2]);
        for (size_t p = 0; p < after.phases.size(); ++p)
        {
            const auto& a = before.phases[p];
            const auto& b = after.phases[p];
            const double ma = phase_mass(a.moles), mb = phase_mass(b.moles);
            old_mass += ma;
            delta_momentum = delta_momentum + (after.velocity + after.drift_velocity[p]) * mb -
                             (before.velocity + before.drift_velocity[p]) * ma;
            delta_energy += b.internal_energy - a.internal_energy;
            for (size_t s = 0; s < species_delta.size(); ++s)
                species_delta[s] += b.moles[s] - a.moles[s];
        }
        const auto& thermal = trial.final_thermal_evaluations()[c];
        const auto& transport = trial.transport_snapshot().cells[c];
        for (const auto& event : thermal.material_events)
        {
            const auto& config = *trial.interphase_receipt()->cells.at(c).configuration;
            const auto& slot = config.stages.at(event.stage).material.at(event.slot);
            const bool drain = slot.donor == 1;
            const size_t s = slot.species == A ? 0 : 1;
            const double amount =
                drain ? .125 * transport.phases[1].moles[s] : Fixture::moles(1.5 * f.mesh->cell_volume(c) / 16)[s];
            const double h = (s == 0 ? 24. : 40.) * thermal.anchors.at(event.stage).temperature.at(slot.donor);
            near(event.carried_energy, amount * h);
            enthalpy[drain ? 0 : 1] += amount * h;
        }
    }
    const auto feed_moles = Fixture::moles(reservoir_count);
    for (size_t s = 0; s < species_delta.size(); ++s)
        near(f.sum(species_delta[s]), feed_moles[s]);
    const std::array birth{1. / 16, 1.5 / 16, 2.25 / 16, 3.375 / 16};
    const auto block = [](const Flow::QbmmPoolDirectionDiagnostics& dir)
    { return std::array{dir.count, dir.mass_kg / scale, dir.H2, dir.H3}; };
    for (size_t k = 0; k < old.size(); ++k)
    {
        const double initial = f.sum(old[k]);
        near(block(directions[0])[k], .125 * initial);
        near(block(directions[1])[k], birth[k]);
        near(f.sum(now[k]), .875 * initial + birth[k] + reservoir_count);
    }
    for (size_t d = 0; d < 2; ++d)
    {
        const auto n = Fixture::moles(directions[d].mass_kg / scale);
        for (size_t s = 0; s < n.size(); ++s)
            near(directions[d].species_moles[s], n[s]);
        near(directions[d].carried_enthalpy_j, f.sum(enthalpy[d]));
    }
    near(f.sum(delta_energy), diag.pressure_work + 128. * T0 * reservoir_count, 3e-7);
    const auto body =
        f.setup.acceleration.value_or(Flow::Vector{}) * ((f.sum(old_mass) + scale * reservoir_count) * dt);
    const auto other = Flow::Vector{-.5, .25, .75} * reservoir_count;
    const auto impulse = body + other;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        near(diag.external_impulse.component(axis), impulse.component(axis));
        near(f.sum(delta_momentum.component(axis)), impulse.component(axis) + diag.pressure_impulse.component(axis));
    }
    EXPECT_EQ(diag.body_acceleration_m_per_s2, f.setup.acceleration);
    EXPECT_EQ(diag.impulse_parts.has_value(), f.setup.acceleration.has_value());
    if (diag.impulse_parts)
    {
        near((diag.impulse_parts->body - body).norm(), 0);
        EXPECT_EQ(diag.impulse_parts->other, other);
    }
}

TEST(ReactingMixtureQbmmPool, FixedHydrostaticBidirectionalConversionWithEmptyOwners)
{
    for (double g : {0., -1.})
    {
        Fixture f({.pool_stock = 1, .two_cells = true, .acceleration = Flow::Vector{0, 0, g}});
        const auto ranks = f.mesh->owned_cell_map()->getComm()->getSize();
        if (ranks > 2)
            EXPECT_GE(f.sum(f.mesh->num_owned_cells() == 0), ranks - 2);
        auto initial = f.flow->state();
        // With two equal cells and gas density affine in p, the face-average density equals rho(p0).
        const double rho = f.setup.liquid_fraction + 6 * scale;
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            auto& cell = initial.cells[c];
            cell.mechanical_pressure = rho * g * (f.mesh->cell_centroid(c).z - .5);
            const double p = f.pressure + cell.mechanical_pressure, ratio = p / f.pressure;
            for (auto& phase : cell.phases)
                phase.absolute_pressure = p;
            for (size_t phase = 1; phase < cell.phases.size(); ++phase)
            {
                for (auto& n : cell.phases[phase].moles)
                    n *= ratio;
                cell.phases[phase].internal_energy *= ratio;
            }
            *cell.phase_numbers[1] *= ratio;
            cell.qbmm_shape[1]->H2 *= ratio;
            cell.qbmm_shape[1]->H3 *= ratio;
        }
        f.flow->initialize(initial);
        start_exchange(f);
        const auto trial = f.controller->preview(dt);
        audit_exchange(f, trial);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const double expected = f.pressure + initial.cells[c].mechanical_pressure;
            EXPECT_NEAR(
                trial.state().thermodynamic_pressure + trial.state().cells[c].mechanical_pressure, expected, 1e-7);
            near(trial.state().cells[c].velocity.norm(), 0, 1e-9);
        }
        for (const auto& face : trial.owned_phase_transport())
            near(face.carrier_volume_flux, 0, 1e-9);
        near(trial.diagnostics().flow.pressure_impulse.z, -rho * g * dt);
    }
}

TEST(ReactingMixtureQbmmPool, FixedPeriodicColumnPressureRampCannotCancelGravityAcrossSeam)
{
    constexpr double g = -1.;
    Fixture f({.liquid_fraction = .25, .pool_stock = 3, .four_layers = true, .periodic_z = true});
    f.options.body_acceleration = Flow::Vector{0, 0, g};
    auto state = f.flow->state();
    const double density = 1.;
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        // An exactly incompressible liquid and absent gas slots make the three interior pressure slopes exact.
        auto& cell = state.cells[c];
        cell.mechanical_pressure = density * g * (f.mesh->cell_centroid(c).z - .5);
        const double pressure = f.pressure + cell.mechanical_pressure;
        for (auto& phase : cell.phases)
            phase.absolute_pressure = pressure;
        const double volume = f.mesh->cell_volume(c);
        cell.phases[0].moles = {0, 0, volume};
        cell.phases[0].internal_energy = volume * liquid_cv * T0;
        for (size_t p = 1; p < 3; ++p)
        {
            cell.phases[p].moles = {0, 0, 0};
            cell.phases[p].internal_energy = 0;
            cell.phases[p].temperature = 0;
        }
        cell.phase_numbers[1] = 0;
        cell.qbmm_shape[1] = R::QbmmShapeState{};
    }
    f.flow->initialize(std::move(state));
    f.events = [](const Context&) { return Operations{}; };
    f.start();
    const auto trial = f.controller->preview(dt);
    double momentum = 0.;
    for (const auto& cell : trial.state().cells)
        for (const auto& phase : cell.phases)
            momentum += phase_mass(phase.moles) * cell.velocity.z;
    // A periodic pressure field supplies no external axial impulse. The unclosed head accelerates the column.
    near(trial.diagnostics().flow.external_impulse.z, density * g * dt);
    near(trial.diagnostics().flow.pressure_impulse.z, 0., 2e-10);
    near(f.sum(momentum), density * g * dt, 2e-10);
}

TEST(ReactingMixtureQbmmPool, FixedBuoyancyMovesDenseAndLightColumnsDuringBidirectionalConversion)
{
    Fixture f({.pool_stock = 1, .circulating = true, .acceleration = Flow::Vector{0, 0, -1}});
    auto initial = f.flow->state();
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        auto& cell = initial.cells[c];
        const double volume = f.mesh->cell_volume(c);
        const double liquid = f.mesh->cell_centroid(c).x < .5 ? .25 : .50;
        cell.velocity = {};
        cell.phases[0].moles[2] = volume * liquid;
        cell.phases[0].internal_energy = volume * liquid * liquid_cv * T0;
        const double gas = (1 - liquid) * volume * f.pressure / (6 * gas_R * T0);
        for (size_t p = 1; p < 3; ++p)
        {
            auto& phase = cell.phases[p];
            phase.moles = Fixture::moles(.5 * gas);
            phase.internal_energy = Fixture::gas_capacity(phase.moles, 1.) * T0;
            phase.temperature = T0;
        }
        const double family = .5 * gas / 5.;
        cell.phase_numbers[1] = 3 * family;
        cell.qbmm_shape[1] = R::QbmmShapeState{11 * family, 29 * family};
    }
    f.flow->initialize(std::move(initial));
    start_exchange(f);
    const auto trial = f.controller->preview(dt);
    audit_exchange(f, trial);
    // In this manufactured EOS the gas is denser than the liquid. The gas-rich left column descends.
    std::array<double, 2> vertical{};
    for (size_t face = 0; face < trial.owned_phase_transport().size(); ++face)
    {
        if (f.mesh->is_geometry_exterior_face(face))
            continue;
        const auto area = f.mesh->face_area_vector(face);
        if (area.z != 0)
            vertical[f.mesh->face_centroid(face).x < .5 ? 0 : 1] +=
                trial.owned_phase_transport()[face].carrier_volume_flux / area.z;
    }
    EXPECT_LT(f.sum(vertical[0]), -1e-6);
    EXPECT_GT(f.sum(vertical[1]), 1e-6);
}

TEST(ReactingMixtureQbmmPool, FixedSlipCarriesCountercurrentPoolMassMomentumAndFamilyMomentsAcrossOwners)
{
    for (const bool forced : {false, true})
    {
        FixtureOptions setup{.pool_stock = 1, .circulating = true, .supplied_slip = true};
        if (forced)
            setup.acceleration = Flow::Vector{0, 0, -.125};
        Fixture packed(setup);
        setup.batched = false;
        Fixture scalar(setup);
        for (auto* f : {&packed, &scalar})
        {
            auto state = f->flow->state();
            for (size_t c = 0; c < state.cells.size(); ++c)
            {
                const auto x = f->mesh->cell_centroid(c);
                auto& cell = state.cells[c];
                cell.velocity.y = x.x - 2 * x.z;
                cell.drift_velocity[1].y = x.x < .5 ? .03125 : -.0625;
                cell.drift_velocity[2].y = -.75 * cell.drift_velocity[1].y;
            }
            f->flow->initialize(std::move(state));
            start_exchange(*f);
        }
        const auto a = packed.controller->preview(dt), b = scalar.controller->preview(dt);
        audit_exchange(packed, a);
        audit_exchange(scalar, b);
        EXPECT_GT(a.diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
        EXPECT_EQ(b.diagnostics().flow.transport_multi_rhs.operator_groups, 0u);
        const auto index = [&](auto cell)
        {
            const auto point = packed.mesh->cell_centroid(cell);
            return (point.x > .5 ? 1u : 0u) + (point.z > .5 ? 2u : 0u);
        };
        const auto drift = [&](size_t p, auto cell)
        {
            const auto point = packed.mesh->cell_centroid(cell);
            const Flow::Vector loop{.5 - point.z, point.x < .5 ? .03125 : -.0625, point.x < .5 ? -.25 : .25};
            return loop * (p == 1 ? 1. : p == 2 ? -.75 : 0.);
        };
        // Globally assemble the four post-advection donor concentrations for a face oracle on partition faces.
        std::array<double, 12> densities{};
        std::array<double, 4> tangential_velocity{}, tangential_outflow{}, symmetric_outflow{};
        for (size_t c = 0; c < a.transport_snapshot().cells.size(); ++c)
        {
            const auto& ca = a.state().cells[c];
            const auto& cb = b.state().cells[c];
            tangential_velocity[index(c)] = ca.velocity.y;
            near((ca.velocity - cb.velocity).norm(), 0);
            near(ca.mechanical_pressure, cb.mechanical_pressure);
            near(*ca.phase_numbers[1], *cb.phase_numbers[1]);
            near(ca.qbmm_shape[1]->H2, cb.qbmm_shape[1]->H2);
            near(ca.qbmm_shape[1]->H3, cb.qbmm_shape[1]->H3);
            for (size_t p = 0; p < 3; ++p)
            {
                for (size_t s = 0; s < 3; ++s)
                    near(ca.phases[p].moles[s], cb.phases[p].moles[s]);
                near(ca.phases[p].internal_energy, cb.phases[p].internal_energy);
                densities[3 * index(c) + p] =
                    phase_mass(a.transport_snapshot().cells[c].phases[p].moles) / packed.mesh->cell_volume(c);
            }
        }
        for (auto& density : densities)
            density = packed.sum(density);
        for (auto& velocity : tangential_velocity)
            velocity = packed.sum(velocity);
        double countercurrent = 0., crossing = 0.;
        for (size_t face = 0; face < a.owned_phase_transport().size(); ++face)
        {
            const auto& receipt = a.owned_phase_transport()[face];
            const auto& ref = b.owned_phase_transport()[face];
            if (packed.mesh->is_geometry_exterior_face(face))
            {
                EXPECT_EQ(receipt.momentum_mass_flux, 0.);
                EXPECT_EQ(receipt.drift_momentum_flux, Flow::Vector{});
                continue;
            }
            const auto owner = packed.mesh->owner_cell(face), neighbor = packed.mesh->neighbor_cell(face);
            const auto area = packed.mesh->face_area_vector(face);
            double forward = 0., backward = 0.;
            Flow::Vector drift_flux{};
            for (size_t p = 0; p < 3; ++p)
            {
                const double rate = receipt.carrier_volume_flux + .5 * (drift(p, owner) + drift(p, neighbor)).dot(area);
                near(receipt.relative_phase_volume_flux[p], rate);
                const auto donor = rate >= 0 ? owner : neighbor;
                const double mass_rate = rate * densities[3 * index(donor) + p];
                forward += std::max(0., mass_rate);
                backward += std::max(0., -mass_rate);
                drift_flux = drift_flux + drift(p, donor) * mass_rate;
                if (p == 1)
                {
                    const auto& flux = receipt.qbmm_flux[p].value();
                    near(flux.derived_mass_flux, mass_rate);
                    near((flux.derived_drift_momentum_flux - drift(p, donor) * mass_rate).norm(), 0);
                    near(flux.population_number_flux, ref.qbmm_flux[p]->population_number_flux);
                    near(flux.shape_flux.H2, ref.qbmm_flux[p]->shape_flux.H2);
                    near(flux.shape_flux.H3, ref.qbmm_flux[p]->shape_flux.H3);
                }
            }
            const auto i = index(owner), j = index(neighbor);
            const double y_flux = forward * tangential_velocity[i] - backward * tangential_velocity[j] + drift_flux.y;
            const double symmetric = std::min(forward, backward) * (tangential_velocity[i] - tangential_velocity[j]);
            tangential_outflow[i] += y_flux;
            tangential_outflow[j] -= y_flux;
            symmetric_outflow[i] += symmetric;
            symmetric_outflow[j] -= symmetric;
            near(receipt.momentum_mass_flux, forward - backward);
            near(receipt.forward_mass_flux, forward);
            near(receipt.backward_mass_flux, backward);
            near(receipt.symmetric_mass_flux, std::min(forward, backward));
            near((receipt.implicit_carrier_momentum_flux -
                     (receipt.momentum_owner_velocity * forward - receipt.momentum_neighbor_velocity * backward))
                     .norm(),
                0);
            near((receipt.drift_momentum_flux - drift_flux).norm(), 0);
            if (receipt.relative_phase_volume_flux[1] * receipt.relative_phase_volume_flux[2] < 0 &&
                std::min(forward, backward) > 1e-6)
            {
                ++countercurrent;
                if (static_cast<size_t>(owner) >= packed.mesh->num_owned_cells() ||
                    static_cast<size_t>(neighbor) >= packed.mesh->num_owned_cells())
                    ++crossing;
            }
        }
        // In the single y layer, pressure, body force and viscous impulse all have zero y component.
        // Reconstruct the momentum update from the two carrier donors, both drifts and extensive phase masses.
        double largest_symmetric = 0.;
        for (size_t i = 0; i < tangential_outflow.size(); ++i)
        {
            tangential_outflow[i] = packed.sum(tangential_outflow[i]);
            largest_symmetric = std::max(largest_symmetric, std::abs(dt * packed.sum(symmetric_outflow[i])));
        }
        EXPECT_GT(largest_symmetric, 1e-5);
        const auto tangential_momentum = [](const Flow::CellState& cell)
        {
            double momentum = 0.;
            for (size_t p = 0; p < cell.phases.size(); ++p)
                momentum += phase_mass(cell.phases[p].moles) * (cell.velocity.y + cell.drift_velocity[p].y);
            return momentum;
        };
        for (size_t c = 0; c < a.state().cells.size(); ++c)
        {
            const double change =
                tangential_momentum(a.state().cells[c]) - tangential_momentum(packed.flow->state().cells[c]);
            EXPECT_NEAR(change, -dt * tangential_outflow[index(c)], 2e-10);
        }
        EXPECT_GT(packed.sum(countercurrent), 0.);
        if (packed.mesh->owned_cell_map()->getComm()->getSize() > 1)
            EXPECT_GT(packed.sum(crossing), 0.);
    }
}

TEST(ReactingMixtureQbmmPool, FixedModesPortableContinuation)
{
    const auto* write = std::getenv("SIMPLEFLUID_Q9C_FIXED_WRITE");
    const auto* read = std::getenv("SIMPLEFLUID_Q9C_FIXED_READ");
    ASSERT_FALSE(write && read);
    const Flow::Vector gravity{.03125, -.0625, -.125};
    for (int mode = 0; mode < 5; ++mode)
    {
        SCOPED_TRACE(mode);
        const double reservoir_count = mode == 4 ? 1. / 32 : 0.;
        FixtureOptions setup{.pool_stock = 1, .law_revision = mode == 4 ? 2u : 1u, .supplied_slip = mode != 1};
        if (mode != 0)
            setup.acceleration = mode == 2 ? Flow::Vector{} : gravity;
        const auto configure = [reservoir_count](Fixture& f)
        {
            if (f.setup.supplied_slip)
            {
                auto state = f.flow->state();
                for (auto& cell : state.cells)
                {
                    cell.drift_velocity[1] = {.25, 0, .0625};
                    cell.drift_velocity[2] = {-.0625, 0, -.125};
                }
                f.flow->initialize(std::move(state));
            }
            start_exchange(f, reservoir_count);
        };
        Fixture reference(setup);
        configure(reference);
        const auto comm = reference.mesh->owned_cell_map()->getComm();
        reference.check_ranks();
        if (const auto* ranks = std::getenv("SIMPLEFLUID_Q9C_FIXED_RANKS"))
            EXPECT_EQ(comm->getSize(), std::stoi(ranks));
        std::vector<std::byte> image;
        const std::string mid = "." + std::to_string(mode) + ".mid", end = "." + std::to_string(mode) + ".end";
        if (read)
        {
            image = IO::read_restart(*comm, read, mid.c_str());
            reference.controller->restore_qbmm_restart(IO::read_restart(*comm, read, end.c_str()));
            EXPECT_EQ(reference.calls, 0);
        }
        else
        {
            reference.controller->advance(dt);
            image = reference.controller->serialize_qbmm_restart();
            reference.controller->advance(dt);
            if (write)
            {
                IO::write_restart(*comm, write, mid.c_str(), image);
                IO::write_restart(*comm, write, end.c_str(), reference.controller->serialize_qbmm_restart());
                continue;
            }
        }
        EXPECT_EQ(image.at(8), std::byte{10});
        EXPECT_EQ(image.at(16), setup.acceleration ? std::byte{10} : std::byte{11});
        Fixture fresh(setup);
        configure(fresh);
        fresh.controller->restore_qbmm_restart(image);
        const auto report = fresh.controller->qbmm_report();
        EXPECT_EQ(report.body_acceleration_m_per_s2, setup.acceleration);
        if (setup.acceleration)
        {
            const auto& parts = report.external_impulses_kg_m_per_s.value();
            const double end_mass = setup.liquid_fraction + (6. + reservoir_count) * scale;
            near((parts.body - *setup.acceleration * (end_mass * dt)).norm(), 0);
            EXPECT_EQ(parts.other, (Flow::Vector{-.5, .25, .75} * reservoir_count));
        }
        EXPECT_EQ(report.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
        EXPECT_FALSE(report.interval);
        EXPECT_EQ(fresh.calls, 0);
        const auto trial = fresh.controller->preview(dt);
        audit_exchange(fresh, trial, reservoir_count);
        const auto baseline = fresh.controller->serialize_qbmm_restart();
        const auto calls = fresh.calls;
        // Repair both checksums after corrupting one shard. MPI4 changes an empty owner's provenance.
        const auto reject_tamper = [&](size_t offset)
        {
            auto bad = image;
            bad.resize(bad.size() - 16);
            if (comm->getRank() == comm->getSize() - 1)
                bad.at(offset) ^= std::byte{0x40};
            IO::append_restart_checksums(*comm, bad);
            EXPECT_THROW(fresh.controller->restore_qbmm_restart(bad), std::exception);
            EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), baseline);
            EXPECT_EQ(fresh.calls, calls);
        };
        reject_tamper(16);         // Payload tag.
        reject_tamper(24 + 5 * 8); // Pool binding fingerprint.
        if (setup.acceleration == gravity)
        {
            std::vector<std::byte> pattern;
            for (double component : {gravity.x, gravity.y, gravity.z})
                IO::append_restart_word(pattern, std::bit_cast<std::uint64_t>(component));
            IO::append_restart_word(pattern, 1); // Accepted force interval present.
            const auto match = std::ranges::search(image, pattern);
            const bool unique =
                !match.empty() && std::ranges::search(std::ranges::subrange(match.end(), image.end()), pattern).empty();
            ASSERT_EQ(fresh.sum(unique), comm->getSize());
            const auto force = static_cast<size_t>(match.begin() - image.begin());
            reject_tamper(force + 7);               // x acceleration.
            reject_tamper(force + (4 + 7) * 8 + 7); // Interval duration.
            reject_tamper(force + (4 + 8) * 8 + 7); // First partition-local body impulse.
        }
        fresh.controller->commit(trial);
        EXPECT_EQ(fresh.calls, calls);
        const auto& got = fresh.flow->state();
        const auto& expected = reference.flow->state();
        EXPECT_EQ(got.revision, expected.revision);
        EXPECT_EQ(got.time, expected.time);
        EXPECT_EQ(got.thermodynamic_pressure, expected.thermodynamic_pressure);
        EXPECT_EQ(fresh.controller->diagnostics().flow.pool_conversion,
            reference.controller->diagnostics().flow.pool_conversion);
        EXPECT_EQ(
            fresh.controller->diagnostics().flow.impulse_parts, reference.controller->diagnostics().flow.impulse_parts);
        for (size_t c = 0; c < got.cells.size(); ++c)
        {
            const auto& a = got.cells[c];
            const auto& b = expected.cells[c];
            EXPECT_EQ(a.velocity, b.velocity);
            EXPECT_EQ(a.drift_velocity, b.drift_velocity);
            EXPECT_EQ(a.mechanical_pressure, b.mechanical_pressure);
            EXPECT_EQ(a.phase_numbers, b.phase_numbers);
            EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
            for (size_t p = 0; p < a.phases.size(); ++p)
            {
                EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
                EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
                EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
            }
        }
        for (size_t face = 0; face < fresh.mesh->num_owned_faces(); ++face)
            EXPECT_EQ(fresh.flow->volume_flux().owned_value(face), reference.flow->volume_flux().owned_value(face));
        for (int change = 0; change < 3; ++change)
        {
            auto mismatch = setup;
            if (change == 0)
                mismatch.supplied_slip = !setup.supplied_slip;
            if (change == 1)
            {
                if (setup.acceleration)
                    mismatch.acceleration.reset();
                else
                    mismatch.acceleration = Flow::Vector{};
            }
            if (change == 2)
                mismatch.acceleration = Flow::Vector{gravity.x, gravity.y, -2};
            Fixture wrong(mismatch);
            configure(wrong);
            const auto saved = wrong.controller->serialize_qbmm_restart();
            EXPECT_THROW(wrong.controller->restore_qbmm_restart(image), std::exception);
            EXPECT_EQ(wrong.controller->serialize_qbmm_restart(), saved);
            EXPECT_EQ(wrong.calls, 0);
        }
    }
}

TEST(ReactingMixtureQbmmPool, PortableContinuation)
{
    Fixture f;
    f.check_ranks();
    f.start();
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const auto* writer = std::getenv("SIMPLEFLUID_Q9B_WRITE");
    const auto* reader = std::getenv("SIMPLEFLUID_Q9B_READ");
    ASSERT_FALSE(writer && reader);
    std::vector<std::byte> first;
    if (reader)
    {
        first = IO::read_restart(*comm, reader, ".deposit");
        f.controller->restore_qbmm_restart(IO::read_restart(*comm, reader, ".return"));
        EXPECT_EQ(f.calls, 0);
    }
    else
    {
        f.controller->advance(dt);
        first = f.controller->serialize_qbmm_restart();
        f.controller->advance(dt);
        if (writer)
        {
            IO::write_restart(*comm, writer, ".deposit", first);
            IO::write_restart(*comm, writer, ".return", f.controller->serialize_qbmm_restart());
            return;
        }
    }
    ASSERT_GE(first.size(), 24u);
    EXPECT_EQ(first[8], std::byte{10});
    EXPECT_EQ(first[16], std::byte{8});
    Fixture fresh;
    fresh.start();
    fresh.controller->restore_qbmm_restart(first);
    EXPECT_EQ(fresh.calls, 0);
    for (const auto& cell : fresh.controller->qbmm_report().cells)
        EXPECT_FALSE(cell.families[1]->pool_conversion_law_evaluated.has_value());
    near(fresh.controller->diagnostics().flow.pool_conversion.value().directions[0].mass_kg, 3 * scale);
    const auto candidate = fresh.controller->preview(dt);
    const auto saved = fresh.controller->serialize_qbmm_restart();
    auto dry_archive = first;
    dry_archive.resize(dry_archive.size() - 16);
    bool mutated = false;
    for (const auto& cell : fresh.flow->state().cells)
    {
        const auto& liquid = cell.phases[0];
        std::vector<std::byte> row;
        for (double value : {liquid.absolute_pressure, liquid.internal_energy, liquid.temperature,
                 cell.drift_velocity[0].x, cell.drift_velocity[0].y, cell.drift_velocity[0].z})
            IO::append_restart_word(row, std::bit_cast<std::uint64_t>(value));
        const auto moles_offset = row.size();
        for (double moles : liquid.moles)
            IO::append_restart_word(row, std::bit_cast<std::uint64_t>(moles));
        IO::append_restart_word(row, 0); // Liquid has neither number nor shape.
        IO::append_restart_word(row, 0);
        const auto match = std::ranges::search(dry_archive, row);
        if (!match.empty() && std::ranges::search(std::ranges::subrange(match.end(), dry_archive.end()), row).empty())
        {
            auto monomer = match.begin() + moles_offset;
            std::swap_ranges(monomer, monomer + 8, monomer + 16); // Swap dissolved A and the equal-mass solvent.
            mutated = true;
        }
    }
    ASSERT_EQ(fresh.sum(mutated), 1.);
    IO::append_restart_checksums(*comm, dry_archive);
    const auto evaluations = fresh.calls;
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(dry_archive), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), saved);
    fresh.controller->commit(candidate);
    EXPECT_EQ(fresh.calls, evaluations);
    EXPECT_DOUBLE_EQ(fresh.flow->state().time, 2 * dt);
    EXPECT_DOUBLE_EQ(fresh.flow->state().thermodynamic_pressure, f.flow->state().thermodynamic_pressure);
    EXPECT_EQ(fresh.controller->diagnostics().flow.pool_conversion, f.controller->diagnostics().flow.pool_conversion);
    for (size_t c = 0; c < fresh.mesh->num_owned_cells(); ++c)
    {
        const auto& a = fresh.flow->state().cells[c];
        const auto& b = f.flow->state().cells[c];
        EXPECT_EQ(a.phase_numbers, b.phase_numbers);
        EXPECT_EQ(a.qbmm_shape, b.qbmm_shape);
        for (size_t p = 0; p < a.phases.size(); ++p)
        {
            EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
            EXPECT_EQ(a.phases[p].temperature, b.phases[p].temperature);
            EXPECT_EQ(a.phases[p].internal_energy, b.phases[p].internal_energy);
        }
    }
    const auto archive = fresh.controller->serialize_qbmm_restart();
    auto invalid = first;
    invalid.resize(invalid.size() - 16);
    if (comm->getRank() == comm->getSize() - 1)
        invalid[16] ^= std::byte{0x10};
    IO::append_restart_checksums(*comm, invalid);
    const int calls = fresh.calls;
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(invalid), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), archive);
    EXPECT_EQ(fresh.calls, calls);
    for (auto setup : {FixtureOptions{.extra_permitted = true}, FixtureOptions{.direction = Direction::DispersedToPool},
             FixtureOptions{.law_revision = 2}, FixtureOptions{.different_eos = true},
             FixtureOptions{.pool_name = "other gas"}, FixtureOptions{.two_families = true}})
    {
        Fixture mismatch(setup);
        mismatch.start();
        const auto before = mismatch.controller->serialize_qbmm_restart();
        EXPECT_THROW(mismatch.controller->restore_qbmm_restart(first), std::exception);
        EXPECT_EQ(mismatch.controller->serialize_qbmm_restart(), before);
        EXPECT_EQ(mismatch.calls, 0);
    }
    Fixture legacy({.legacy = true});
    legacy.events = [](const Context&) { return Operations{}; };
    legacy.start();
    legacy.controller->advance(dt);
    const auto old_archive = legacy.controller->serialize_qbmm_restart();
    EXPECT_EQ(old_archive[8], std::byte{5});
    EXPECT_THROW(legacy.controller->restore_qbmm_restart(first), std::exception);
    EXPECT_EQ(legacy.controller->serialize_qbmm_restart(), old_archive);
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(old_archive), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), archive);
}
} // namespace
