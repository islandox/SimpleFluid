/** Prescribed planar motion with a thermodynamically distinct, converting BulkGas pool. */
#include "QbmmPoolTestSupport.hh"

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmPool;
using LO = Flow::Pack::local_ordinal_type;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

constexpr double displacement = 1. / 1024.;
double target_surface(double time)
{
    // Expansion on the first interval, compression on the next.
    return time <= dt ? 1. + displacement * time / dt : 1. + displacement * (3. - 2. * time / dt);
}

struct Piston : Fixture
{
    int targets = 0;
    explicit Piston(FixtureOptions setup = {}, std::uint64_t motion_revision = 1) : Fixture(setup)
    {
        options.planar_motion = std::make_shared<const R::MixturePlanarMotionProvider>(R::MixturePlanarMotionProvider{
            mesh, {}, "Q9c prescribed piston", motion_revision, [this](double time, double step)
            {
                ++targets;
                return target_surface(time + step);
            }});
        const auto comm = mesh->owned_cell_map()->getComm();
        if (const auto* ranks = std::getenv("SIMPLEFLUID_Q9C_ALE_RANKS"))
            EXPECT_EQ(comm->getSize(), std::stoi(ranks));
        if (!setup.four_layers)
            EXPECT_EQ(sum(mesh->num_owned_cells() == 0), comm->getSize() - 1);
    }
    auto callbacks() const { return std::array{targets, calls, partial_calls}; }
};

void check_piston(bool slip, bool different_eos)
{
    Piston f({.different_eos = different_eos, .supplied_slip = slip});
    double target = 0.;
    f.events = [&](const Context& in)
    {
        const auto& local = in.law_inputs->location.value();
        near(local.centroid[2], target / 2., 1e-14);
        near(in.law_inputs->bulk_volume_m3, target, 1e-14);
        EXPECT_EQ(local.geometryEpoch, f.mesh->geometry_epoch());
        return f.trajectory(in);
    };
    f.start();
    const double liquid = f.setup.liquid_fraction, pool_factor = different_eos ? 1.5 : 1.;
    constexpr double nR = 30 * gas_R;
    double previous_t = T0, previous_v = 1., previous_bubbles = 5.;
    const std::array<std::array<double, 4>, 2> expected_moments{{{2., 2., 2., 2.}, {4., 5., 6.5, 8.75}}};
    for (size_t step = 0; step < expected_moments.size(); ++step)
    {
        SCOPED_TRACE(step);
        target = target_surface((step + 1) * dt);
        const auto& moments = expected_moments[step];
        const auto capacity = [&](double bubbles)
        { return liquid * liquid_cv + 128. * (bubbles + (5. - bubbles) * pool_factor); };
        // U1 = U0 - p1*(V1-V0), p1*(V1-Vliquid) = nRT1, with independent Cv for each EOS.
        const double old_u = capacity(previous_bubbles) * previous_t;
        const double t1 = old_u / (capacity(moments[1]) + nR * (target - previous_v) / (target - liquid));
        const double p1 = nR * t1 / (target - liquid);
        const double anchor = (old_u + p1 * (previous_v - liquid)) / (capacity(previous_bubbles) + nR);
        const double donor_cv = step == 0 ? 1. : pool_factor;
        const double carried = (12 * (16 * donor_cv + gas_R) + 6 * (32 * donor_cv + gas_R)) * anchor;
        const auto trial = f.controller->preview(dt);
        const auto report = f.controller->qbmm_report(trial);
        const auto& d = trial.diagnostics();
        ASSERT_TRUE(report.pool_conversion);
        const auto& gross = report.pool_conversion->directions[step];
        near(gross.mass_kg, 3 * scale);
        near(gross.H2, step == 0 ? 9. : 4.5);
        near(gross.H3, step == 0 ? 27. : 6.75);
        near(gross.count, step == 0 ? 1. : 2.);
        EXPECT_EQ(gross.species_moles, Fixture::moles(3));
        near(gross.carried_enthalpy_j, carried);
        EXPECT_EQ(report.pool_conversion->directions[1 - step].count, 0.);
        near(trial.state().thermodynamic_pressure, p1);
        near(d.flow.pressure_work, -p1 * (target - previous_v));
        near(d.external_heat, 0.);
        near(d.external_mass, 0.);
        for (size_t c = 0; c < trial.state().cells.size(); ++c)
        {
            const auto& endpoint = trial.state().cells[c];
            const auto& row = report.cells[c];
            const auto& thermal = trial.final_thermal_evaluations()[c];
            const std::array old_v{liquid, (previous_v - liquid) * previous_bubbles / 5.,
                (previous_v - liquid) * (5. - previous_bubbles) / 5.};
            const std::array new_v{
                liquid, (target - liquid) * moments[1] / 5., (target - liquid) * (5. - moments[1]) / 5.};
            near(row.bulk_volume_m3, target, 1e-14);
            near(row.dispersed_gas_volume_fraction, new_v[1] / target);
            near(row.bulk_gas_pool_volume_fraction, new_v[2] / target);
            EXPECT_FALSE(row.families[2]);
            for (size_t k = 0; k < moments.size(); ++k)
                near(row.families[1]->moments[k], moments[k], 2e-13);
            EXPECT_EQ(endpoint.phases[1].moles, Fixture::moles(moments[1]));
            EXPECT_EQ(endpoint.phases[2].moles, Fixture::moles(5. - moments[1]));
            double sum_u = 0.;
            for (size_t p = 0; p < 3; ++p)
            {
                const double material = p == 0 ? 0. : (p == step + 1 ? -carried : carried);
                const auto& transported = trial.transport_snapshot().cells[c].phases[p];
                near(transported.advected_material_volume, old_v[p]);
                near(thermal.volumes[p], new_v[p]);
                near(thermal.raw_material_enthalpy[p], material);
                const auto& phase = endpoint.phases[p];
                near(phase.temperature, new_v[p] > 0. ? t1 : 0.);
                near(phase.internal_energy - transported.internal_energy,
                    material + thermal.raw_equilibrium_redistribution[p] - p1 * (new_v[p] - old_v[p]));
                sum_u += phase.internal_energy;
            }
            near(sum_u - old_u, -p1 * (target - previous_v), 3e-8);
        }
        double sweep = 0.;
        for (const auto& face : trial.owned_phase_transport())
        {
            EXPECT_EQ(face.carrier_volume_flux, face.mesh_volume_flux);
            for (const double q : face.relative_phase_volume_flux)
                EXPECT_EQ(q, 0.);
            EXPECT_EQ(face.qbmm_flux[1]->population_number_flux, 0.);
            sweep += face.mesh_volume_flux;
        }
        near(f.sum(sweep) * dt, target - previous_v, 2e-14);
        const auto callbacks = f.callbacks();
        f.controller->commit(trial);
        EXPECT_EQ(f.callbacks(), callbacks);
        previous_t = t1;
        previous_v = target;
        previous_bubbles = moments[1];
    }
}

TEST(ReactingMixtureQbmmPoolALE, ClosedPistonPressureWorkAndBidirectionalDonorEnthalpy)
{
    for (bool slip : {false, true})
        for (bool different : {false, true})
        {
            SCOPED_TRACE(slip);
            SCOPED_TRACE(different);
            check_piston(slip, different);
        }
}

TEST(ReactingMixtureQbmmPoolALE, SourceSubstepsUseTransportedPredecessorsOnTheTrialGeometry)
{
    Piston f({.supplied_slip = true});
    f.options.qbmm_source_substep_relative_tolerance = .005;
    f.events = [&](const Context& in)
    {
        near(in.law_inputs->bulk_volume_m3, target_surface(dt), 2e-14);
        return f.drain(in, std::vector<double>(in.quadrature.active_atoms, .5 * in.time_step / dt));
    };
    f.start();
    const auto trial = f.controller->preview(dt);
    const auto n = trial.diagnostics().maximum_qbmm_source_substeps;
    ASSERT_GT(n, 1u);
    const double retain = 1. - .5 / n;
    const auto& direction = trial.diagnostics().flow.pool_conversion.value().directions[0];
    near(direction.mass_kg, scale * 5 * (1. - std::pow(retain, n)));
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        const auto& stages = trial.local_qbmm_sources()[c][1]->substeps;
        ASSERT_EQ(stages.size(), n);
        for (size_t i = 0; i < n; ++i)
        {
            const double fraction = std::pow(retain, i + 1);
            near(stages[i].duration, dt / n);
            near(stages[i].final_count, 3 * fraction);
            near(stages[i].final_shape.H2, 11 * fraction);
            near(stages[i].final_shape.H3, 29 * fraction);
            near(stages[i].final_moles.at(0), 20 * fraction);
            near(stages[i].final_moles.at(1), 10 * fraction);
        }
        near(trial.state().cells[c].phases[2].moles[0], 20 * (1. - std::pow(retain, n)));
    }
}

TEST(ReactingMixtureQbmmPoolALE, OverdrawAfterMovingDepositPreservesGeometryLedgerAndRetainedReturn)
{
    for (bool slip : {false, true})
    {
        Piston f({.different_eos = true, .supplied_slip = slip});
        f.start();
        f.controller->advance(dt);
        const auto saved = f.controller->serialize_qbmm_restart();
        const auto epoch = f.mesh->geometry_epoch();
        const auto accepted = f.controller->qbmm_report();
        const auto earlier = f.controller->preview(dt);
        const auto retained = f.controller->qbmm_report(earlier);
        f.events = [&](const Context& in) { return f.birth(in, 100., 1.); };
        EXPECT_THROW((void) f.controller->preview(dt / 2), std::exception);
        EXPECT_EQ(f.mesh->geometry_epoch(), epoch);
        EXPECT_EQ(f.controller->serialize_qbmm_restart(), saved);
        EXPECT_EQ(f.controller->qbmm_report().pool_conversion, accepted.pool_conversion);
        const auto replay = f.controller->qbmm_report(earlier);
        EXPECT_EQ(replay.provenance.preview_id, retained.provenance.preview_id);
        EXPECT_EQ(replay.provenance.geometry_epoch, retained.provenance.geometry_epoch);
        for (size_t c = 0; c < replay.cells.size(); ++c)
        {
            near(f.mesh->cell_volume(c), target_surface(dt), 1e-14);
            near(replay.cells[c].bulk_volume_m3, target_surface(2 * dt), 1e-14);
        }
        const auto callbacks = f.callbacks();
        f.controller->commit(earlier);
        EXPECT_EQ(f.callbacks(), callbacks);
        near(f.controller->diagnostics().flow.pool_conversion->directions[1].mass_kg, 3 * scale);
    }
}

TEST(ReactingMixtureQbmmPoolALE, UniformCompositionAndSizeDistributionSurviveCellwiseGcl)
{
    for (bool slip : {false, true})
    {
        Piston f({.pool_stock = 2., .supplied_slip = slip, .four_layers = true});
        f.events = [](const Context&) { return Operations{}; };
        f.start();
        double old_height = 1.;
        for (size_t step = 1; step <= 2; ++step)
        {
            const auto trial = f.controller->preview(dt);
            const auto& transport = trial.transport_snapshot();
            const double height = target_surface(step * dt);
            std::array<double, 5> sweep{};
            for (size_t face = 0; face < trial.owned_phase_transport().size(); ++face)
            {
                const double area = f.mesh->face_area_vector(face).z;
                if (area == 0.)
                    continue;
                const auto j = static_cast<size_t>(std::lround(4 * f.mesh->face_centroid(face).z / old_height));
                sweep[j] = trial.owned_phase_transport()[face].mesh_volume_flux / area;
                near(sweep[j], .25 * j * (height - old_height) / dt, 2e-14);
            }
            for (auto& rate : sweep)
                rate = f.sum(rate);
            double transported = 0.;
            for (size_t c = 0; c < trial.state().cells.size(); ++c)
            {
                const auto j = static_cast<size_t>(4 * f.mesh->cell_centroid(c).z / old_height);
                near(transport.new_control_volumes[c] - transport.old_control_volumes[c],
                    dt * (sweep[j + 1] - sweep[j]), 2e-14);
                // A common relative flux transports every extensive component with the same operator.
                // All initial phase/species ratios and normalized bubble moments are spatially uniform.
                const auto& cell = trial.state().cells[c];
                const double unit = cell.phases[0].moles[2] / f.setup.liquid_fraction;
                transported += unit;
                for (size_t phase = 1; phase <= 2; ++phase)
                {
                    const auto expected = Fixture::moles((phase == 1 ? 5. : 2.) * unit);
                    for (size_t i = 0; i < expected.size(); ++i)
                        near(cell.phases[phase].moles[i], expected[i], 8e-10);
                }
                near(cell.phase_numbers[1].value(), 3 * unit, 8e-10);
                near(cell.qbmm_shape[1]->H2, 11 * unit, 8e-10);
                near(cell.qbmm_shape[1]->H3, 29 * unit, 8e-10);
            }
            near(f.sum(transported), 1., 8e-10);
            f.controller->commit(trial);
            old_height = height;
        }
    }
}

double axial_drift(size_t phase, size_t layer)
{
    if (layer == 0 || layer == 3)
        return 0.;
    return phase == 1 ? .002 : -.002;
}

TEST(ReactingMixtureQbmmPoolALE, SuppliedCounterflowBalancesTransportedStocksBeforeConversion)
{
    std::array<std::array<double, 4>, 2> packed{};
    for (bool batched : {true, false})
    {
        Piston f({.pool_stock = 2., .batched = batched, .supplied_slip = true, .four_layers = true});
        auto state = f.flow->state();
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            const auto layer = static_cast<size_t>(4 * f.mesh->cell_centroid(c).z);
            for (size_t p = 1; p <= 2; ++p)
                state.cells[c].drift_velocity[p].z = axial_drift(p, layer);
        }
        f.flow->initialize(std::move(state));
        f.events = [&](const Context& in)
        { return f.drain(in, std::vector<double>(in.quadrature.active_atoms, .125)); };
        f.start();
        const auto trial = f.controller->preview(dt);
        const auto& transport = trial.transport_snapshot();
        EXPECT_EQ(trial.diagnostics().flow.transport_multi_rhs.operator_groups > 0, batched);
        constexpr size_t n = 4;
        const double height = target_surface(dt), v1 = height / n;
        // Assemble the stage-used carrier flux in the positive axial direction. Reconstruct each
        // phase flux independently from that carrier, the prescribed drift and the analytic sweep.
        std::array<double, n + 1> carrier{};
        double traversed = 0.;
        for (size_t face = 0; face < trial.owned_phase_transport().size(); ++face)
        {
            const auto& packet = trial.owned_phase_transport()[face];
            const double area_z = f.mesh->face_area_vector(face).z;
            if (area_z == 0.)
                continue;
            const auto j = static_cast<size_t>(std::lround(n * f.mesh->face_centroid(face).z));
            const double sweep = (height - 1.) * j / (n * dt);
            near(packet.mesh_volume_flux, area_z * sweep, 2e-14);
            if (j == 0 || j == n)
            {
                for (const auto q : packet.relative_phase_volume_flux)
                    EXPECT_EQ(q, 0.);
                continue;
            }
            carrier[j] = packet.carrier_volume_flux / area_z;
            for (size_t p = 1; p <= 2; ++p)
            {
                const double drift = .5 * (axial_drift(p, j - 1) + axial_drift(p, j));
                near(packet.relative_phase_volume_flux[p], area_z * (carrier[j] + drift - sweep), 2e-13);
            }
            if (f.mesh->owned_cell_map()->getComm()->getSize() == 1 ||
                static_cast<size_t>(f.mesh->neighbor_cell(face)) >= f.mesh->num_owned_cells() ||
                static_cast<size_t>(f.mesh->owner_cell(face)) >= f.mesh->num_owned_cells())
                traversed += std::abs(packet.qbmm_flux[1]->population_number_flux);
        }
        EXPECT_GT(f.sum(traversed), 1e-8);
        for (auto& q : carrier)
            q = f.sum(q);
        for (size_t p = 1; p <= 2; ++p)
        {
            std::array<double, n> pre{}, final{};
            for (size_t c = 0; c < transport.cells.size(); ++c)
            {
                const auto j = static_cast<size_t>(n * f.mesh->cell_centroid(c).z);
                near(transport.new_control_volumes[c], v1, 1e-14);
                pre[j] = transport.cells[c].phases[p].moles[0];
                final[j] = trial.state().cells[c].phases[p].moles[0];
                const double converted = .125 * transport.cells[c].phases[1].moles[0];
                near(final[j], pre[j] + (p == 1 ? -converted : converted));
                if (p == 1)
                {
                    const auto& cell = trial.state().cells[c];
                    near(cell.phase_numbers[1].value(), final[j] * 3. / 20.);
                    near(cell.qbmm_shape[1]->H2, final[j] * 11. / 20.);
                    near(cell.qbmm_shape[1]->H3, final[j] * 29. / 20.);
                }
            }
            for (size_t j = 0; j < n; ++j)
            {
                pre[j] = f.sum(pre[j]);
                final[j] = f.sum(final[j]);
                if (batched)
                    packed[p - 1][j] = final[j];
                else
                    near(final[j], packed[p - 1][j], 3e-10);
            }
            std::array<double, n + 1> molar_flux{};
            for (size_t j = 1; j < n; ++j)
            {
                const double slip = .5 * (axial_drift(p, j - 1) + axial_drift(p, j));
                const double q = carrier[j] + slip - j * (height - 1.) / (n * dt);
                molar_flux[j] = q * pre[q >= 0. ? j - 1 : j] / v1;
            }
            for (size_t j = 0; j < n; ++j)
                near(pre[j] + dt * (molar_flux[j + 1] - molar_flux[j]), p == 1 ? 5. : 2., 3e-10);
        }
        near(trial.diagnostics().flow.pool_conversion->directions[0].mass_kg, 5 * scale * .125);
    }
}

TEST(ReactingMixtureQbmmPoolALE, SuppliedAccelerationRejectsBeforeMotionAndSourceEvaluation)
{
    for (bool slip : {false, true})
        for (const auto force : {Flow::Vector{}, Flow::Vector{0, 0, -.125}})
        {
            Piston f({.supplied_slip = slip, .acceleration = force});
            const auto epoch = f.mesh->geometry_epoch();
            EXPECT_THROW(f.start(), std::exception);
            EXPECT_EQ(f.callbacks(), (std::array{0, 0, 0}));
            Flow::PlanarMotion motion(f.mesh);
            int calls = 0;
            auto source = std::make_shared<const Flow::SourceStageProvider>(
                Flow::SourceStageProvider{"accelerated pool", 1, f.catalog->fingerprint(),
                    [&](const auto&)
                    {
                        ++calls;
                        return Flow::SourceStageResult{};
                    },
                    0., 0, force});
            EXPECT_THROW(
                (void) f.flow->preview_ale_with_sources(dt, motion, target_surface(dt), source), std::exception);
            EXPECT_EQ(calls, 0);
            EXPECT_FALSE(motion.has_active_trial());
            EXPECT_EQ(f.mesh->geometry_epoch(), epoch);
            EXPECT_EQ(f.flow->state().time, 0.);
        }
}

void check_tag(const std::vector<std::byte>& bytes)
{
    std::vector<std::byte> header;
    for (std::uint64_t word : std::array<std::uint64_t, 3>{0x5453524d4d425153ULL, 10, 9})
        IO::append_restart_word(header, word);
    ASSERT_GE(bytes.size(), header.size());
    EXPECT_TRUE(std::equal(header.begin(), header.end(), bytes.begin()));
}

void compare_endpoint(Piston& a, Piston& b)
{
    const auto& left = a.flow->state();
    const auto& right = b.flow->state();
    EXPECT_EQ(left.time, right.time);
    EXPECT_EQ(left.revision, right.revision);
    EXPECT_EQ(left.thermodynamic_pressure, right.thermodynamic_pressure);
    ASSERT_EQ(left.cells.size(), right.cells.size());
    for (size_t c = 0; c < left.cells.size(); ++c)
    {
        EXPECT_EQ(a.mesh->cell_volume(c), b.mesh->cell_volume(c));
        EXPECT_EQ(a.mesh->cell_centroid(c), b.mesh->cell_centroid(c));
        const auto& x = left.cells[c];
        const auto& y = right.cells[c];
        EXPECT_EQ(x.mechanical_pressure, y.mechanical_pressure);
        EXPECT_EQ(x.velocity, y.velocity);
        EXPECT_EQ(x.drift_velocity, y.drift_velocity);
        EXPECT_EQ(x.qbmm_shape, y.qbmm_shape);
        EXPECT_EQ(x.phase_numbers, y.phase_numbers);
        for (size_t p = 0; p < x.phases.size(); ++p)
        {
            EXPECT_EQ(x.phases[p].moles, y.phases[p].moles);
            EXPECT_EQ(x.phases[p].temperature, y.phases[p].temperature);
            EXPECT_EQ(x.phases[p].internal_energy, y.phases[p].internal_energy);
            EXPECT_EQ(x.phases[p].absolute_pressure, y.phases[p].absolute_pressure);
        }
    }
    for (size_t face = 0; face < a.mesh->num_faces(); ++face)
        EXPECT_EQ(a.flow->volume_flux().local_value(face), b.flow->volume_flux().local_value(face));
    const auto& x = a.controller->diagnostics();
    const auto& y = b.controller->diagnostics();
    EXPECT_EQ(x.correctors, y.correctors);
    EXPECT_EQ(x.population_births, y.population_births);
    EXPECT_EQ(x.population_deaths, y.population_deaths);
    EXPECT_EQ(x.flow.pressure_work, y.flow.pressure_work);
    EXPECT_EQ(x.flow.pool_conversion, y.flow.pool_conversion);
}

void portable_continuation(bool slip)
{
    const auto* writer = std::getenv("SIMPLEFLUID_Q9C_ALE_WRITE");
    const auto* reader = std::getenv("SIMPLEFLUID_Q9C_ALE_READ");
    ASSERT_FALSE(reader && writer);
    const FixtureOptions setup{.different_eos = true, .supplied_slip = slip};
    Piston original(setup);
    original.start();
    const auto comm = original.mesh->owned_cell_map()->getComm();
    std::vector<std::byte> deposit;
    if (reader)
    {
        deposit = IO::read_restart(*comm, reader, ".deposit");
        original.controller->restore_qbmm_restart(IO::read_restart(*comm, reader, ".return"));
        EXPECT_EQ(original.callbacks(), (std::array{0, 0, 0}));
    }
    else
    {
        original.controller->advance(dt);
        deposit = original.controller->serialize_qbmm_restart();
        check_tag(deposit);
        original.controller->advance(dt);
        if (writer)
        {
            IO::write_restart(*comm, writer, ".deposit", deposit);
            IO::write_restart(*comm, writer, ".return", original.controller->serialize_qbmm_restart());
            return;
        }
    }
    check_tag(deposit);
    Piston fresh(setup);
    fresh.start();
    fresh.controller->restore_qbmm_restart(deposit);
    EXPECT_EQ(fresh.callbacks(), (std::array{0, 0, 0}));
    const auto restored = fresh.controller->qbmm_report();
    EXPECT_EQ(restored.receipt_history, Flow::QbmmReceiptHistory::UnavailableAfterPortableRestart);
    EXPECT_FALSE(restored.interval);
    near(restored.pool_conversion.value().directions[0].mass_kg, 3 * scale);
    const auto candidate = fresh.controller->preview(dt);
    const auto before = fresh.controller->serialize_qbmm_restart();
    const auto callbacks = fresh.callbacks();
    const auto epoch = fresh.mesh->geometry_epoch();
    auto damaged = deposit;
    damaged.resize(damaged.size() - 16);
    // Locate the face-flux block by its following wall-interval receipt. On the sole
    // owning shard, replace the moving wall's accepted absolute flux with zero while
    // retaining the valid surface and timestep. Recompute both checksums.
    bool changed_wall = false;
    if (fresh.mesh->num_owned_cells() != 0)
    {
        std::vector<std::byte> wall;
        size_t moving_offset = 0;
        for (size_t face = 0; face < fresh.mesh->num_owned_faces(); ++face)
        {
            const double q = fresh.flow->volume_flux().owned_value(face);
            if (q != 0.)
                moving_offset = wall.size();
            IO::append_restart_word(wall, std::bit_cast<std::uint64_t>(q));
        }
        IO::append_restart_word(wall, 1);
        IO::append_restart_word(wall, std::bit_cast<std::uint64_t>(1.));
        IO::append_restart_word(wall, std::bit_cast<std::uint64_t>(dt));
        const auto found = std::ranges::search(damaged, wall);
        if (!found.empty() && std::ranges::search(std::ranges::subrange(found.end(), damaged.end()), wall).empty())
        {
            auto word = found.begin() + moving_offset;
            changed_wall = std::ranges::any_of(std::span(&*word, 8), [](auto byte) { return byte != std::byte{0}; });
            std::fill_n(word, 8, std::byte{0});
        }
    }
    ASSERT_EQ(fresh.sum(changed_wall), 1.);
    IO::append_restart_checksums(*comm, damaged);
    const auto other_surface = original.controller->serialize_qbmm_restart();
    const auto other_epoch = original.mesh->geometry_epoch();
    EXPECT_THROW(original.controller->restore_qbmm_restart(damaged), std::exception);
    EXPECT_EQ(original.controller->serialize_qbmm_restart(), other_surface);
    EXPECT_EQ(original.mesh->geometry_epoch(), other_epoch);
    EXPECT_THROW(fresh.controller->restore_qbmm_restart(damaged), std::exception);
    EXPECT_EQ(fresh.controller->serialize_qbmm_restart(), before);
    EXPECT_EQ(fresh.mesh->geometry_epoch(), epoch);
    fresh.controller->commit(candidate);
    EXPECT_EQ(fresh.callbacks(), callbacks);
    compare_endpoint(original, fresh);
    for (int mismatch = 0; mismatch < 3; ++mismatch)
    {
        auto changed = setup;
        if (mismatch == 1)
            ++changed.law_revision;
        if (mismatch == 2)
            changed.supplied_slip = !slip;
        Piston wrong(changed, mismatch == 0 ? 2 : 1);
        wrong.start();
        const auto image = wrong.controller->serialize_qbmm_restart();
        EXPECT_THROW(wrong.controller->restore_qbmm_restart(deposit), std::exception);
        EXPECT_EQ(wrong.controller->serialize_qbmm_restart(), image);
        EXPECT_EQ(wrong.callbacks(), (std::array{0, 0, 0}));
    }
}

TEST(ReactingMixtureQbmmPoolALE, SharedPortableContinuation)
{
    portable_continuation(false);
}

TEST(ReactingMixtureQbmmPoolALE, SuppliedSlipPortableContinuation)
{
    portable_continuation(true);
}
} // namespace
