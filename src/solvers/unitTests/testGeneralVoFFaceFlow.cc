#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
const char* backend_name(VoF::GeneralVoFTransportBackend backend)
{
    return backend == VoF::GeneralVoFTransportBackend::DonorCell ? "DonorCell" : "GeometricFace";
}
void record_metric(const std::string& name, long double value)
{
    std::ostringstream output;
    output << std::setprecision(20) << value;
    testing::Test::RecordProperty(name, output.str());
}
void expect_backend(const GeneralVoFSolver& solver, VoF::GeneralVoFTransportBackend backend)
{
    EXPECT_EQ(solver.phases()->options().transport_backend, backend);
    const auto p = solver.prepared_transport();
    ASSERT_TRUE(p);
    EXPECT_EQ(p->backend, backend);
    EXPECT_EQ(p->family, VoF::GeneralVoFTransportFamily::AlgebraicFaceVolume);
    EXPECT_EQ(p->arithmetic_mode, ArithmeticMode::Ordinary);
    testing::Test::RecordProperty(std::string(backend_name(backend)) + "_configured_backend", backend_name(backend));
    testing::Test::RecordProperty(std::string(backend_name(backend)) + "_prepared_family", "AlgebraicFaceVolume");
}
double region_gas(const GeneralVoFSolver& solver)
{
    double local = 0, global = 0;
    for (std::size_t c = 0; c < solver.mesh()->num_owned_cells(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        if (x.x - .2 * x.y > .5)
            local += solver.phases()->gas_volumes()[c];
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
// Momentum consumes these positive routes and signed face contributions.
// Compare both independently against final phase streams, including when theta<1.
void expect_final_stream_momentum_ledger(const GeneralVoFSolver& solver)
{
    const auto p = solver.prepared_transport();
    ASSERT_TRUE(p);
    for (const auto& face : p->signed_face_contributions)
    {
        const auto stream = std::ranges::find(p->streams, face.face, &VoF::GeneralVoFDirectedStream::face);
        ASSERT_NE(stream, p->streams.end());
        EXPECT_EQ(face.donor, stream->donor);
        EXPECT_EQ(face.direction, stream->direction);
        EXPECT_EQ(face.amount.bulk.value, stream->volume);
        EXPECT_EQ(face.amount.gas.value, stream->gas);
        EXPECT_EQ(face.amount.carrier.value, stream->liquid);
    }
    for (const auto& route : p->positive_transfers)
    {
        if (route.donor == route.recipient)
            continue;
        long double bulk = 0, gas = 0, liquid = 0;
        for (const auto& stream : p->streams)
            if (stream.donor == route.donor && stream.recipient == route.recipient)
            {
                bulk += stream.volume;
                gas += stream.gas;
                liquid += stream.liquid;
            }
        EXPECT_EQ(route.amount.bulk.value, bulk);
        EXPECT_EQ(route.amount.gas.value, gas);
        EXPECT_EQ(route.amount.carrier.value, liquid);
    }
}
void expect_phase_and_momentum_gates(const GeneralVoFSolver& solver, const VoFStepDiagnostics& initial)
{
    expect_final_stream_momentum_ledger(solver);
    record_qualification_cleanup(solver);
    const auto& after = solver.diagnostics();
    EXPECT_LE(std::abs(after.gas_mass - initial.gas_mass), 2e-11 * initial.gas_mass);
    EXPECT_LE(std::abs(after.liquid_mass - initial.liquid_mass), 2e-11 * initial.liquid_mass);
    EXPECT_LT(after.maximum_divergence, 1e-8);
    for (const auto& m : solver.momentum_transport_diagnostics())
        EXPECT_LE(m.physical_relative_residual, 1e-12);
    for (const auto& cell : solver.phases()->phase_volumes())
    {
        EXPECT_GE(cell.gas, 0);
        EXPECT_GE(cell.carrier, 0);
        EXPECT_NEAR(cell.gas + cell.carrier, cell.bulk, 1e-13 * cell.bulk);
    }
}
void expect_skew_coarse_fine_structure(const GeneralVoFSolver& solver, const std::string& prefix)
{
    double local[2] = {double(solver.mesh()->num_owned_cells()), 0}, global[2]{};
    for (std::size_t f = 0; f < solver.mesh()->num_owned_faces(); ++f)
    {
        if (solver.mesh()->is_geometry_exterior_face(int(f)))
            continue;
        const double a = solver.mesh()->cell_volume(solver.mesh()->owner_cell(int(f)));
        const double b = solver.mesh()->cell_volume(solver.mesh()->neighbor_cell(int(f)));
        if (std::max(a, b) > 1.5 * std::min(a, b))
            ++local[1];
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
    // The provider introduces conforming transition cells in addition to the marked splits.
    EXPECT_EQ(global[0], 50);
    EXPECT_GT(global[1], 0);
    record_metric(prefix + "_global_native_cells", global[0]);
    record_metric(prefix + "_global_coarse_fine_faces", global[1]);
}
std::vector<VoF::Vector> skew_vortex_velocity(const GeneralVoFSolver& solver)
{
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        const double xi = x.x - .2 * x.y, eta = x.y;
        const double vx = .05 * std::pow(std::sin(std::numbers::pi * xi), 2) * std::sin(2 * std::numbers::pi * eta);
        const double vy = -.05 * std::pow(std::sin(std::numbers::pi * eta), 2) * std::sin(2 * std::numbers::pi * xi);
        velocity[c] = {vx + .2 * vy, vy, 0};
    }
    return velocity;
}
std::shared_ptr<const VoF::GeneralVoF> face_backend(std::shared_ptr<const VoF::GeneralVoF> source,
    VoF::GeneralVoFTransportBackend backend = VoF::GeneralVoFTransportBackend::GeometricFace, bool pure_gas = false,
    bool cleanup = qualification_phase_cleanup_enabled())
{
    VoF::GeneralVoFOptions options = source->options();
    options.transport_backend = backend;
    auto gas = source->gas_volumes(), liquid = source->carrier_volumes();
    if (pure_gas)
        std::swap(gas, liquid);
    if (source->options().phase_cleanup.relative_cutoff > 0)
    {
        if (!cleanup || pure_gas)
            throw std::logic_error("A qualification backend clone must preserve its existing cleanup policy.");
        return std::make_shared<const VoF::GeneralVoF>(source->hierarchy(), source->owned_order(), gas, liquid, options,
            source->phase_geometry_declarations(), source->phase_cleanup_state());
    }
    qualification_phase_policy(options, gas, liquid, *source->hierarchy()->communicator(), cleanup);
    return std::make_shared<const VoF::GeneralVoF>(source->hierarchy(), source->owned_order(), gas, liquid, options);
}
TEST(GeneralVoFFaceFlow, CleanupPreservesConstantTransportedVelocityAndRawMassMomentumIdentity)
{
    constexpr double dt = 0x1p-44;
    const VoF::Vector uniform{.125, -.0625, 0};
    for (const auto backend : {VoF::GeneralVoFTransportBackend::GeometricFace})
    {
        SCOPED_TRACE(backend_name(backend));
        const auto original = phases(1, false, true, false);
        auto material = inviscid();
        material.gas_density = 1;
        material.liquid_density = 1000;
        GeneralVoFSolver raw(face_backend(original, backend, false, false), material);
        GeneralVoFSolver cleaned(face_backend(original, backend, false, true), material);
        std::vector<VoF::Vector> velocity(cleaned.mesh()->num_owned_cells(), uniform);
        std::vector<double> faces(cleaned.mesh()->num_faces());
        for (const auto& tile : cleaned.phases()->tiles())
        {
            if (tile.exterior)
                continue;
            const auto face = cleaned.mesh()->overlap_face_map()->getLocalElement(tile.face_gid);
            const auto center = tile.geometry->centroid(), normal = tile.geometry->area_vector();
            if (center.x == 1 && normal.x != 0)
                faces[face] = (center.y < 1 ? 1 : -1) * (normal.x > 0 ? 1 : -1) / 16.;
            if (center.y == 1 && normal.y != 0)
                faces[face] = (center.x > 1 ? 1 : -1) * (normal.y > 0 ? 1 : -1) / 16.;
        }
        raw.initialize(velocity, faces);
        cleaned.initialize(velocity, faces);
        for (std::size_t c = 0; c < velocity.size(); ++c)
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                EXPECT_NEAR(raw.velocity()[c][axis], uniform[axis], 1e-14);
                EXPECT_NEAR(cleaned.velocity()[c][axis], uniform[axis], 1e-14);
            }
        const auto initial = raw.diagnostics(); // Raw, before any cleanup event.
        ASSERT_NO_THROW((void) raw.step(dt));
        ASSERT_NO_THROW((void) cleaned.step(dt));
        expect_backend(cleaned, backend);
        expect_phase_and_momentum_gates(cleaned, initial);
        const auto prepared = cleaned.prepared_transport();
        ASSERT_TRUE(prepared);
        ASSERT_GT(cleaned.phases()->phase_cleanup_state().cleaned_cells, 0u);
        ASSERT_EQ(prepared->cleanup_momentum.size(), velocity.size());
        double local_nonzero = 0, global_nonzero = 0;
        for (std::size_t c = 0; c < velocity.size(); ++c)
        {
            const long double dg =
                static_cast<long double>(cleaned.phases()->gas_volumes()[c]) - raw.phases()->gas_volumes()[c];
            const long double dl =
                static_cast<long double>(cleaned.phases()->carrier_volumes()[c]) - raw.phases()->carrier_volumes()[c];
            const long double dm = dg + 1000 * dl;
            EXPECT_EQ(prepared->raw_new_mass[c], raw.prepared_transport()->new_mass[c]);
            EXPECT_EQ(prepared->cleanup_mass[c], dm);
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                // Uniform initial momentum and divergence-free closed face flux
                // give the independent raw transport solution u=uniform, before
                // pressure projection imposes the closed-wall normal condition.
                const auto expected = dm * uniform[axis];
                if (axis == 2)
                    EXPECT_EQ(prepared->cleanup_momentum[c][axis], 0);
                else
                    EXPECT_NEAR(prepared->cleanup_momentum[c][axis], expected, 1e-10L * std::abs(expected));
                local_nonzero += prepared->cleanup_momentum[c][axis] != 0;
            }
        }
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_nonzero, &global_nonzero);
        EXPECT_GT(global_nonzero, 0);
        const int local_empty = velocity.empty();
        int empty = 0;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(empty, std::max(0, Tpetra::getDefaultComm()->getSize() - 2));
        qualification_cleanup_metric(std::string(backend_name(backend)) + "_asserted_empty_owners", empty);
        record_qualification_cleanup(cleaned, std::string(backend_name(backend)) + "_");
    }
}
TEST(GeneralVoFFaceFlow, CleanupBudgetRejectionRetainsCompleteAcceptedPublication)
{
    constexpr double dt = 0x1p-44, q = 0x1p-48;
    for (const auto backend : {VoF::GeneralVoFTransportBackend::GeometricFace})
    {
        SCOPED_TRACE(backend_name(backend));
        const auto original = phases(1, false, true, false);
        auto options = original->options();
        options.transport_backend = backend;
        // This regression deliberately exhausts a small absolute allowance.
        // Physical qualification fixtures retain their original-inventory caps.
        options.phase_cleanup = {1e-14, {3 * q, 3 * q}};
        const auto source = std::make_shared<const VoF::GeneralVoF>(original->hierarchy(), original->owned_order(),
            original->gas_volumes(), original->carrier_volumes(), options);
        auto material = inviscid();
        material.gas_density = 1;
        material.liquid_density = 1000;
        GeneralVoFSolver solver(source, material);
        std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
        for (std::size_t c = 0; c < velocity.size(); ++c)
        {
            const auto center = solver.mesh()->cell_centroid(int(c));
            velocity[c] = {(1 - center.y) / 8, (center.x - 1) / 8, 0};
        }
        std::vector<double> faces(solver.mesh()->num_faces());
        for (const auto& tile : solver.phases()->tiles())
        {
            if (tile.exterior)
                continue;
            const auto face = solver.mesh()->overlap_face_map()->getLocalElement(tile.face_gid);
            const auto center = tile.geometry->centroid(), normal = tile.geometry->area_vector();
            if (center.x == 1 && normal.x != 0)
                faces[face] = (center.y < 1 ? 1 : -1) * (normal.x > 0 ? 1 : -1) / 16.;
            if (center.y == 1 && normal.y != 0)
                faces[face] = (center.x > 1 ? 1 : -1) * (normal.y > 0 ? 1 : -1) / 16.;
        }
        solver.initialize(velocity, faces);
        const auto initial = solver.diagnostics();
        EXPECT_EQ(solver.phases()->phase_cleanup_state().cleaned_cells, 0u);
        ASSERT_NO_THROW((void) solver.step(dt));
        expect_backend(solver, backend);
        expect_phase_and_momentum_gates(solver, initial);
        // Exactly two routes cross the initially aligned phase boundary.
        // Each produces a q minority and an exactly representable 1-q
        // dominant amount; their opposite corrections spend 2q per phase.
        const auto& cleanup = solver.phases()->phase_cleanup_state();
        EXPECT_EQ(cleanup.cleaned_cells, 2u);
        for (unsigned phase = 0; phase < 2; ++phase)
        {
            EXPECT_EQ(cleanup.absolute_volume[phase], 2 * q);
            EXPECT_EQ(cleanup.signed_volume[phase], 0);
            EXPECT_EQ(cleanup.largest_absolute_volume[phase], q);
        }
        const auto comm = Tpetra::getDefaultComm();
        const int local_empty = velocity.empty();
        int empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
        qualification_cleanup_metric(std::string(backend_name(backend)) + "_asserted_empty_owners", empty);
        qualification_cleanup_metric(std::string(backend_name(backend)) + "_ranks", comm->getSize());
        record_qualification_cleanup(solver, std::string(backend_name(backend)) + "_accepted_");
        const auto accepted = solver.phases();
        const auto prepared = solver.prepared_transport();
        const auto identity = accepted->accepted_identity();
        const auto control = accepted->accepted_control();
        const auto mesh = solver.mesh();
        const auto accepted_velocity = solver.velocity();
        const auto accepted_faces = solver.face_velocity();
        const auto pressure = solver.pressure();
        const auto fraction = solver.gas_fraction();
        const auto curvature = solver.curvature();
        const auto gas = accepted->gas_volumes(), carrier = accepted->carrier_volumes();
        const auto declarations = accepted->phase_geometry_declarations();
        const auto histories = accepted->accumulated_phase_errors();
        const auto roundoff = accepted->phase_roundoff_bounds();
        const auto checkpoint = accepted->phase_cleanup_checkpoint();
        const auto diagnostics = solver.diagnostics();
        const auto momentum = solver.momentum_transport_diagnostics();
        const auto capillary_model = solver.capillary_diagnostics().model;
        // Surface tension is zero, so the complete capillary vector contents
        // are empty both before and after this attempted physical step.
        EXPECT_TRUE(solver.capillary_diagnostics().fits.empty());
        EXPECT_TRUE(solver.capillary_diagnostics().components.empty());
        int rejected = 0;
        std::string rejection;
        try
        {
            (void) solver.step(dt);
        }
        catch (const std::exception& error)
        {
            rejected = 1;
            rejection = error.what();
        }
        int rejected_ranks = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &rejected_ranks);
        EXPECT_EQ(rejected_ranks, comm->getSize());
        EXPECT_NE(rejection.find("phase cleanup"), std::string::npos) << rejection;
        EXPECT_NE(rejection.find("budget"), std::string::npos) << rejection;
        EXPECT_EQ(solver.phases(), accepted);
        EXPECT_EQ(solver.phases()->accepted_identity(), identity);
        EXPECT_EQ(solver.phases()->accepted_control(), control);
        EXPECT_EQ(solver.prepared_transport(), prepared);
        EXPECT_EQ(solver.mesh(), mesh);
        EXPECT_EQ(solver.velocity(), accepted_velocity);
        EXPECT_EQ(solver.face_velocity(), accepted_faces);
        EXPECT_EQ(solver.pressure(), pressure);
        EXPECT_EQ(solver.gas_fraction(), fraction);
        EXPECT_EQ(solver.curvature(), curvature);
        EXPECT_EQ(solver.time(), dt);
        EXPECT_EQ(solver.phases()->gas_volumes(), gas);
        EXPECT_EQ(solver.phases()->carrier_volumes(), carrier);
        EXPECT_EQ(solver.phases()->phase_geometry_declarations(), declarations);
        EXPECT_EQ(solver.phases()->phase_cleanup_checkpoint(), checkpoint);
        const auto& after = solver.diagnostics();
        EXPECT_EQ(after.time, diagnostics.time);
        EXPECT_EQ(after.gas_volume, diagnostics.gas_volume);
        EXPECT_EQ(after.gas_mass, diagnostics.gas_mass);
        EXPECT_EQ(after.liquid_mass, diagnostics.liquid_mass);
        EXPECT_EQ(after.momentum, diagnostics.momentum);
        EXPECT_EQ(after.kinetic_energy, diagnostics.kinetic_energy);
        EXPECT_EQ(after.maximum_velocity, diagnostics.maximum_velocity);
        EXPECT_EQ(after.maximum_divergence, diagnostics.maximum_divergence);
        EXPECT_EQ(after.pressure_residual, diagnostics.pressure_residual);
        EXPECT_EQ(after.pressure_iterations, diagnostics.pressure_iterations);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto& m = solver.momentum_transport_diagnostics()[axis];
            EXPECT_EQ(m.physical_relative_residual, momentum[axis].physical_relative_residual);
            EXPECT_EQ(m.implicit_iterations, momentum[axis].implicit_iterations);
            EXPECT_EQ(m.defect_corrections, momentum[axis].defect_corrections);
            EXPECT_EQ(m.cleanup_momentum, momentum[axis].cleanup_momentum);
            EXPECT_EQ(m.accumulated_cleanup_momentum, momentum[axis].accumulated_cleanup_momentum);
            EXPECT_EQ(m.cleanup_identity_relative_residual, momentum[axis].cleanup_identity_relative_residual);
        }
        EXPECT_EQ(solver.capillary_diagnostics().model, capillary_model);
        EXPECT_TRUE(solver.capillary_diagnostics().fits.empty());
        EXPECT_TRUE(solver.capillary_diagnostics().components.empty());
        EXPECT_EQ(solver.phases()->accumulated_phase_errors().size(), histories.size());
        EXPECT_EQ(solver.phases()->phase_roundoff_bounds().size(), roundoff.size());
        if (solver.phases()->accumulated_phase_errors().size() == histories.size() &&
            solver.phases()->phase_roundoff_bounds().size() == roundoff.size())
            for (std::size_t c = 0; c < histories.size(); ++c)
            {
                EXPECT_EQ(solver.phases()->accumulated_phase_errors()[c].gas, histories[c].gas);
                EXPECT_EQ(solver.phases()->accumulated_phase_errors()[c].carrier, histories[c].carrier);
                EXPECT_EQ(solver.phases()->phase_roundoff_bounds()[c].gas, roundoff[c].gas);
                EXPECT_EQ(solver.phases()->phase_roundoff_bounds()[c].carrier, roundoff[c].carrier);
            }
    }
}
TEST(GeneralVoFFaceFlow, MovingPurePhasesReuseMomentumWithoutInterfaceWork)
{
    for (const auto backend :
        {VoF::GeneralVoFTransportBackend::DonorCell, VoF::GeneralVoFTransportBackend::GeometricFace})
    {
        SCOPED_TRACE(backend_name(backend));
        for (bool gas : {false, true})
        {
            GeneralVoFSolver solver(face_backend(phases(0, false, true, false), backend, gas), inviscid());
            std::vector<double> face_velocity(solver.mesh()->num_faces());
            for (const auto& tile : solver.phases()->tiles())
            {
                const auto f = solver.mesh()->overlap_face_map()->getLocalElement(tile.face_gid);
                const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
                if (tile.exterior || f < 0)
                    continue;
                if (c.x == 1 && n.x != 0)
                    face_velocity[f] = (c.y < 1 ? .05 : -.05) * (n.x > 0 ? 1 : -1);
                if (c.y == 1 && n.y != 0)
                    face_velocity[f] = (c.x > 1 ? .05 : -.05) * (n.y > 0 ? 1 : -1);
            }
            record_qualification_cleanup(solver, std::string(backend_name(backend)) + (gas ? "_gas_" : "_liquid_"));
            solver.initialize({}, face_velocity);
            const auto before = solver.diagnostics();
            EXPECT_GT(before.maximum_velocity, 0);
            for (unsigned step = 0; step < 3; ++step)
            {
                const auto after = solver.step(.001);
                expect_backend(solver, backend);
                record_qualification_cleanup(
                    solver, std::string(backend_name(backend)) + "_" + std::to_string(step) + "_");
                EXPECT_NEAR(after.gas_mass, before.gas_mass, 1e-13);
                EXPECT_NEAR(after.liquid_mass, before.liquid_mass, 1e-13);
                const auto p = solver.prepared_transport();
                ASSERT_TRUE(p);
                ASSERT_TRUE(p->momentum_preparation);
                EXPECT_EQ(p->diagnostics.plane_measurements, 0u);
                EXPECT_LE(p->momentum_builds, 1u);
                EXPECT_GE(p->momentum_reuses, 2u);
                EXPECT_EQ(p->momentum_builds + p->momentum_reuses, 3u);
                for (const auto& m : solver.momentum_transport_diagnostics())
                    EXPECT_LE(m.physical_relative_residual, 1e-12);
            }
            EXPECT_NEAR(solver.time(), .003, 1e-18);
            const auto accepted = solver.phases();
            const auto preparation = solver.prepared_transport();
            EXPECT_THROW((void) solver.step(1e8), std::runtime_error);
            EXPECT_EQ(solver.phases(), accepted);
            EXPECT_EQ(solver.prepared_transport(), preparation);
        }
    }
}
TEST(GeneralVoFFaceFlow, PureLiquidHydrostaticPressureRetainsPhysicalBalance)
{
    for (const auto backend :
        {VoF::GeneralVoFTransportBackend::DonorCell, VoF::GeneralVoFTransportBackend::GeometricFace})
    {
        SCOPED_TRACE(backend_name(backend));
        auto material = inviscid();
        material.liquid_density = 1;
        GeneralVoFSolverOptions options;
        options.flow.gravity = {.3, -.2, .1};
        GeneralVoFSolver solver(face_backend(phases(0, true, false, false), backend), material, options);
        record_qualification_cleanup(solver, std::string(backend_name(backend)) + "_initial_");
        for (unsigned step = 0; step < 4; ++step)
        {
            (void) solver.step(.01);
            expect_backend(solver, backend);
            record_qualification_cleanup(solver, std::string(backend_name(backend)) + "_" + std::to_string(step) + "_");
        }
        EXPECT_NEAR(solver.time(), .04, 1e-16);
        double local[2]{}, global[2]{};
        for (std::size_t c = 0; c < solver.velocity().size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(int(c));
            local[0] += solver.pressure()[c] - (.3 * x.x - .2 * x.y + .1 * x.z);
            local[1] += 1;
        }
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
        for (std::size_t c = 0; c < solver.velocity().size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(int(c));
            EXPECT_NEAR(solver.pressure()[c], .3 * x.x - .2 * x.y + .1 * x.z + global[0] / global[1], 2e-7);
            for (const auto u : solver.velocity()[c])
                EXPECT_NEAR(u, 0, 2e-8);
        }
        EXPECT_LT(solver.diagnostics().maximum_divergence, 1e-8);
    }
}
void high_density_baseline(VoF::GeneralVoFTransportBackend backend)
{
    SCOPED_TRACE(backend_name(backend));
    auto material = inviscid();
    material.gas_density = 1;
    material.liquid_density = 1000;
    QualificationPhaseReference raw;
    GeneralVoFSolver solver(skew_coarse_fine_plane(qualification_phase_cleanup_enabled(), backend, &raw), material);
    record_qualification_cleanup(solver, std::string(backend_name(backend)) + "_initial_");
    const std::string prefix = backend_name(backend);
    expect_skew_coarse_fine_structure(solver, prefix);
    solver.initialize(skew_vortex_velocity(solver));
    double local_region = 0, initial_region_gas = 0;
    for (std::size_t c = 0; c < raw.gas.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        if (x.x - .2 * x.y > .5)
            local_region += raw.gas[c];
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_region, &initial_region_gas);
    unsigned accepted_steps = 0;
    const auto before = raw.physical_reference(solver.diagnostics(), material);
    for (unsigned step = 0; step < 10; ++step)
    {
        record_metric(prefix + "_attempted_time", solver.time());
        record_metric(prefix + "_accepted_steps", accepted_steps);
        const auto after = solver.step(.002);
        ++accepted_steps;
        expect_backend(solver, backend);
        expect_final_stream_momentum_ledger(solver);
        record_qualification_cleanup(solver, prefix + "_" + std::to_string(step) + "_");
        EXPECT_LE(std::abs(after.gas_mass - before.gas_mass), 2e-11 * before.gas_mass);
        EXPECT_LE(std::abs(after.liquid_mass - before.liquid_mass), 2e-11 * before.liquid_mass);
        EXPECT_LT(after.maximum_divergence, 1e-8);
        for (const auto& m : solver.momentum_transport_diagnostics())
            EXPECT_LE(m.physical_relative_residual, 1e-12);
        const auto p = solver.prepared_transport();
        ASSERT_TRUE(p);
        EXPECT_EQ(p->momentum_builds, 1u);
        EXPECT_EQ(p->momentum_reuses, 2u);
        for (const auto& cell : solver.phases()->phase_volumes())
        {
            EXPECT_GE(cell.gas, 0);
            EXPECT_GE(cell.carrier, 0);
            EXPECT_NEAR(cell.gas + cell.carrier, cell.bulk, 1e-13 * cell.bulk);
        }
        const auto metric = [&](const std::string& name, long double value)
        {
            std::ostringstream output;
            output << std::setprecision(20) << value;
            testing::Test::RecordProperty(prefix + "_" + name + "_" + std::to_string(step), output.str());
        };
        metric("gas_mass_relative_drift", std::abs(after.gas_mass - before.gas_mass) / before.gas_mass);
        metric("liquid_mass_relative_drift", std::abs(after.liquid_mass - before.liquid_mass) / before.liquid_mass);
        metric("maximum_divergence", after.maximum_divergence);
        metric("maximum_occupancy_defect", p->diagnostics.maximum_occupancy_defect);
        metric("maximum_outgoing_courant", p->diagnostics.maximum_outgoing_courant);
        long double residual = 0;
        for (const auto& m : solver.momentum_transport_diagnostics())
            residual = std::max(residual, (long double) m.physical_relative_residual);
        metric("momentum_physical_relative_residual", residual);
        metric("plane_measurements", p->diagnostics.plane_measurements);
        metric("momentum_builds", p->momentum_builds);
        metric("momentum_reuses", p->momentum_reuses);
        testing::Test::RecordProperty(
            prefix + "_fallback_streams_" + std::to_string(step), p->diagnostics.fallback_streams);
        testing::Test::RecordProperty(
            prefix + "_limited_streams_" + std::to_string(step), p->diagnostics.limited_streams);
    }
    EXPECT_EQ(accepted_steps, 10u);
    record_metric(prefix + "_accepted_steps", accepted_steps);
    record_metric(prefix + "_accepted_time", solver.time());
    const double passage = region_gas(solver) - initial_region_gas;
    // At xi=.5 the initial gas occupies .37*y+.11*z<=.15. The rectangle
    // y in [.05,.2], z in [0,.5] is wholly gas and its initial inward
    // xi velocity is >=.05*sin(.1*pi). Its .02s transported-volume scale
    // is >=2.3e-5. Requiring 1e-6 tests resolved regional passage with a
    // conservative margin for interpolation and evolving coupled velocity;
    // it is not an analytic reference for the coupled flow solution.
    EXPECT_GT(passage, 1e-6);
    record_metric(prefix + "_regional_gas_passage", passage);
    EXPECT_NEAR(solver.time(), .02, 1e-16);
}
TEST(GeneralVoFFaceFlow, HighDensityRatioSkewCoarseFinePassage)
{
    high_density_baseline(VoF::GeneralVoFTransportBackend::GeometricFace);
}
TEST(GeneralVoFFaceFlow, DonorCellHighDensityRatioSkewCoarseFinePassage)
{
    high_density_baseline(VoF::GeneralVoFTransportBackend::DonorCell);
}
TEST(GeneralVoFFaceFlow, ClosedSymmetryViscousModeRetainsAnalyticFiniteDurationRate)
{
    for (const auto backend :
        {VoF::GeneralVoFTransportBackend::DonorCell, VoF::GeneralVoFTransportBackend::GeometricFace})
    {
        SCOPED_TRACE(backend_name(backend));
        auto material = inviscid();
        material.gas_density = material.liquid_density = 1;
        material.gas_viscosity = material.liquid_viscosity = .01;
        GeneralVoFSolver solver(face_backend(pure_cartesian_phases(8, 8, 2, false), backend), material);
        record_qualification_cleanup(solver, std::string(backend_name(backend)) + "_initial_");
        const auto comm = solver.mesh()->owned_cell_map()->getComm();
        const int local_cells = static_cast<int>(solver.mesh()->num_owned_cells());
        int global_cells = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
        ASSERT_EQ(global_cells, 128);
        constexpr double amplitude = 1e-6;
        std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
        for (std::size_t c = 0; c < velocity.size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
            velocity[c] = {amplitude * std::sin(std::numbers::pi * x.x) * std::cos(std::numbers::pi * x.y),
                -amplitude * std::cos(std::numbers::pi * x.x) * std::sin(std::numbers::pi * x.y), 0};
        }
        solver.initialize(velocity);
        const auto mode_amplitude = [&]
        {
            double local[2] = {}, global[2] = {};
            for (std::size_t c = 0; c < velocity.size(); ++c)
            {
                const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
                const double shape = std::sin(std::numbers::pi * x.x) * std::cos(std::numbers::pi * x.y);
                const double volume = solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c));
                local[0] += volume * solver.velocity()[c][0] * shape;
                local[1] += volume * shape * shape;
            }
            Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
            return global[0] / global[1];
        };
        const auto before = solver.diagnostics();
        const double initial = mode_amplitude();
        for (unsigned step = 0; step < 10; ++step)
        {
            const auto after = solver.step(.002);
            EXPECT_LT(after.maximum_divergence, 1e-8);
            expect_backend(solver, backend);
            expect_phase_and_momentum_gates(solver, before);
        }
        EXPECT_NEAR(solver.time(), .02, 1e-16);
        const double final = mode_amplitude();
        EXPECT_GT(final, 0);
        EXPECT_LT(final, initial);
        const double rate = -std::log(final / initial) / solver.time();
        EXPECT_NEAR(rate / (2 * std::numbers::pi * std::numbers::pi * .01), 1, .05);
        record_metric(std::string(backend_name(backend)) + "_viscous_decay_rate", rate);
        record_metric(std::string(backend_name(backend)) + "_accepted_time", solver.time());
        EXPECT_LT(solver.diagnostics().kinetic_energy, before.kinetic_energy);
        for (unsigned a = 0; a < 3; ++a)
            EXPECT_NEAR(solver.diagnostics().momentum[a], before.momentum[a], 2e-11);
    }
}
// Coupled flow has evolving velocity and pressure. These gates measure actual
// accepted redistribution and accumulated invariants, without claiming that the
// prescribed initial vortex remains an analytic solution of the coupled problem.
void high_density_long_horizon(VoF::GeneralVoFTransportBackend backend, double fraction)
{
    if (!std::getenv("SIMPLEFLUID_RUN_ALGEBRAIC_FLOW"))
        GTEST_SKIP() << "Opt-in Q3 full 8-second qualification horizon.";
    auto material = inviscid();
    material.gas_density = 1;
    material.liquid_density = 1000;
    QualificationPhaseReference raw;
    GeneralVoFSolver solver(skew_coarse_fine_plane(qualification_phase_cleanup_enabled(), backend, &raw), material);
    record_qualification_cleanup(solver, std::string(backend_name(backend)) + "_initial_");
    expect_skew_coarse_fine_structure(solver, backend_name(backend));
    solver.initialize(skew_vortex_velocity(solver));
    const auto initial = raw.physical_reference(solver.diagnostics(), material);
    const auto original_gas = raw.gas;
    const auto gas_centroid = [&](bool original = false)
    {
        double local[4]{}, global[4]{};
        for (std::size_t c = 0; c < original_gas.size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(int(c));
            const double gas = original ? original_gas[c] : solver.phases()->gas_volumes()[c];
            local[0] += gas * x.x;
            local[1] += gas * x.y;
            local[2] += gas * x.z;
            local[3] += gas;
        }
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 4, local, global);
        return std::array<double, 3>{global[0] / global[3], global[1] / global[3], global[2] / global[3]};
    };
    const auto initial_centroid = gas_centroid(true);
    // Register the full endpoint and fixed acceptance targets before stepping.
    // .125 is the refined stripe's xi width. The centroid requirement is one
    // such width, while summed phase redistribution must exceed 10% of initial gas.
    constexpr double endpoint = 8, required_centroid_displacement = .125, required_redistribution = .1;
    record_metric("physical_endpoint", endpoint);
    record_metric("completed_physical_horizon", 0);
    record_metric("stability_fraction", fraction);
    record_metric("required_centroid_displacement", required_centroid_displacement);
    record_metric("required_relative_phase_redistribution", required_redistribution);
    unsigned steps = 0, transition_steps = 0, limited_steps = 0;
    std::size_t transitions = 0, limited = 0, fallback = 0;
    double peak_courant = 0, maximum_displacement = 0;
    long double affected_bulk_volume = 0, discarded_gas_correction = 0;
    std::array<std::size_t, 10> theta_histogram{};
    while (solver.time() < endpoint)
    {
        ASSERT_LT(steps, 512u) << "accepted time=" << solver.time();
        const double stable = solver.stable_timestep();
        ASSERT_TRUE(std::isfinite(stable));
        ASSERT_GT(stable, 0);
        const double dt = std::min(fraction * stable, endpoint - solver.time());
        ASSERT_GT(dt, 0);
        ASSERT_GT(solver.time() + dt, solver.time());
        SCOPED_TRACE(::testing::Message() << "backend=" << backend_name(backend) << ", step=" << steps
                                          << ", time=" << solver.time() << ", dt=" << dt << ", stable=" << stable);
        const auto old_phases = solver.phases();
        record_metric("attempted_step", steps);
        record_metric("attempted_time", solver.time());
        record_metric("attempted_timestep", dt);
        record_metric("attempted_stable_timestep", stable);
        record_metric("attempted_stability_fraction", dt / stable);
        VoFStepDiagnostics after;
        ASSERT_NO_THROW(after = solver.step(dt));
        ++steps;
        expect_backend(solver, backend);
        expect_phase_and_momentum_gates(solver, initial);
        record_qualification_cleanup(solver, "step_" + std::to_string(steps) + "_");
        record_metric("raw_gas_mass_relative_drift", std::abs(after.gas_mass - initial.gas_mass) / initial.gas_mass);
        record_metric(
            "raw_liquid_mass_relative_drift", std::abs(after.liquid_mass - initial.liquid_mass) / initial.liquid_mass);
        const auto p = solver.prepared_transport();
        ASSERT_TRUE(p);
        EXPECT_EQ(p->momentum_builds, 1u);
        EXPECT_EQ(p->momentum_reuses, 2u);
        peak_courant = std::max(peak_courant, double(p->diagnostics.maximum_outgoing_courant));
        int local_new_transitions = 0, global_new_transitions = 0;
        for (std::size_t c = 0; c < old_phases->gas_volumes().size(); ++c)
        {
            const bool was_pure = old_phases->gas_volumes()[c] == 0 || old_phases->carrier_volumes()[c] == 0;
            const bool now_mixed = solver.phases()->gas_volumes()[c] > 0 && solver.phases()->carrier_volumes()[c] > 0;
            local_new_transitions += was_pure && now_mixed;
        }
        Teuchos::reduceAll(
            *Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_new_transitions, &global_new_transitions);
        transitions += global_new_transitions;
        transition_steps += global_new_transitions > 0;
        std::map<std::uint64_t, int> face_owner;
        for (const auto& tile : solver.phases()->tiles())
            face_owner.emplace(tile.face_gid, tile.first_owner);
        int step_limited = 0;
        for (const auto& stream : p->streams)
        {
            EXPECT_EQ(stream.mass, stream.gas + 1000 * stream.liquid);
            if (face_owner.at(stream.face) != Tpetra::getDefaultComm()->getRank())
                continue;
            // Directed algebraic streams own their theta distribution; the
            // swept geometric-face histogram intentionally excludes this family.
            const auto bin = stream.theta <= 0   ? 0u
                             : stream.theta >= 1 ? 9u
                                                 : std::min(8u, 1u + unsigned(8 * stream.theta));
            ++theta_histogram[bin];
            if (stream.theta < 1)
            {
                ++limited;
                step_limited = 1;
                affected_bulk_volume += stream.volume;
                discarded_gas_correction += std::abs(stream.gas_high - stream.gas);
            }
            fallback += stream.lower_order;
        }
        int globally_limited = 0;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &step_limited, &globally_limited);
        limited_steps += globally_limited;
        const auto centroid = gas_centroid();
        maximum_displacement =
            std::max(maximum_displacement, std::hypot(centroid[0] - initial_centroid[0],
                                               centroid[1] - initial_centroid[1], centroid[2] - initial_centroid[2]));
        record_metric("accepted_steps", steps);
        record_metric("accepted_time", solver.time());
        record_metric("maximum_outgoing_courant", peak_courant);
        record_metric("maximum_gas_centroid_displacement", maximum_displacement);
        record_metric("pure_to_mixed_transitions", transitions);
        record_metric("steps_with_pure_to_mixed_transitions", transition_steps);
        // Stream totals are canonical-owner local until this explicit reduction.
        const double local_work[4]{
            double(limited), double(fallback), double(affected_bulk_volume), double(discarded_gas_correction)};
        double global_work[4]{};
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 4, local_work, global_work);
        record_metric("limited_streams", global_work[0]);
        record_metric("fallback_streams", global_work[1]);
        record_metric("limited_affected_bulk_volume", global_work[2]);
        record_metric("discarded_gas_high_order_correction", global_work[3]);
        record_metric("steps_with_active_limiting", limited_steps);
        // Keep the first failed physical step and its metrics as the reproducer.
        // Every rank exits together, and an incomplete eight-second endpoint
        // remains a failed qualification rather than generating repeated failures.
        int local_failure = testing::Test::HasFailure(), global_failure = 0;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local_failure, &global_failure);
        ASSERT_EQ(global_failure, 0) << "Physical qualification gate failed at accepted time=" << solver.time()
                                     << "; required endpoint=" << endpoint;
    }
    record_metric("completed_physical_horizon", 1);
    double local = 0, redistribution = 0;
    for (std::size_t c = 0; c < original_gas.size(); ++c)
        local += std::abs(solver.phases()->gas_volumes()[c] - original_gas[c]);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &redistribution);
    record_metric("relative_phase_redistribution", redistribution / initial.gas_volume);
    std::array<std::size_t, 10> global_theta{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 10, theta_histogram.data(), global_theta.data());
    for (std::size_t bin = 0; bin < theta_histogram.size(); ++bin)
        record_metric("directed_stream_theta_bin_" + std::to_string(bin), global_theta[bin]);
    if (backend == VoF::GeneralVoFTransportBackend::GeometricFace && fraction == .8)
        EXPECT_GE(limited_steps, 4u) << "Sustained coupled active-limiter qualification remains open.";
    EXPECT_DOUBLE_EQ(solver.time(), endpoint);
    EXPECT_GT(steps, 10u);
    EXPECT_GT(transitions, 1u);
    EXPECT_GT(transition_steps, 1u);
    EXPECT_GE(maximum_displacement, required_centroid_displacement);
    EXPECT_GE(redistribution / initial.gas_volume, required_redistribution);
}
TEST(GeneralVoFFaceFlow, HighDensityLongHorizonDonorCellAtStability02)
{
    high_density_long_horizon(VoF::GeneralVoFTransportBackend::DonorCell, .2);
}
TEST(GeneralVoFFaceFlow, HighDensityLongHorizonDonorCellAtStability05)
{
    high_density_long_horizon(VoF::GeneralVoFTransportBackend::DonorCell, .5);
}
TEST(GeneralVoFFaceFlow, HighDensityLongHorizonDonorCellAtStability08)
{
    high_density_long_horizon(VoF::GeneralVoFTransportBackend::DonorCell, .8);
}
TEST(GeneralVoFFaceFlow, HighDensityLongHorizonGeometricFaceAtStability02)
{
    high_density_long_horizon(VoF::GeneralVoFTransportBackend::GeometricFace, .2);
}
TEST(GeneralVoFFaceFlow, HighDensityLongHorizonGeometricFaceAtStability05)
{
    high_density_long_horizon(VoF::GeneralVoFTransportBackend::GeometricFace, .5);
}
TEST(GeneralVoFFaceFlow, HighDensityLongHorizonGeometricFaceAtStability08)
{
    high_density_long_horizon(VoF::GeneralVoFTransportBackend::GeometricFace, .8);
}
} // namespace
