/** @file testGeneralVoFFaceTransactions.cc
 * @brief Moving algebraic mixed-phase AMR, spatial migration and collective rollback contracts.
 */
#include "FVM/details/VoFFailure.hh"
#include "general_vof_solver_test_helpers.hh"
#include "utils/CompensatedSum.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Backend = VoF::GeneralVoFTransportBackend;
long double sum_wide(const Teuchos::Comm<int>& comm, long double value)
{
    // Teuchos serializes binary64 here. Retain each local wide accumulator's
    // head and residual, then sum both parts with compensation in rank order.
    const double head = static_cast<double>(value);
    const double local[2]{head, static_cast<double>(value - static_cast<long double>(head))};
    std::vector<double> gathered(2 * comm.getSize());
    Teuchos::gatherAll(comm, 2, local, int(gathered.size()), gathered.data());
    SimpleFluid::detail::CompensatedSum<> result;
    for (const auto part : gathered)
        result += static_cast<long double>(part);
    return result.value();
}

void metric(const std::string& name, long double value)
{
    std::ostringstream output;
    output << std::setprecision(20) << value;
    testing::Test::RecordProperty(name, output.str());
}
std::shared_ptr<const VoF::GeneralVoF> spatial_mixed_phases(Backend backend, bool empty_owners)
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 5; ++x)
            grid.xy_nodes.push_back({double(x) + .2 * y, double(y), 0});
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 4; ++x)
        {
            const unsigned p = 5 * y + x;
            grid.base_cells.push_back({p, p + 1, p + 6, p + 5});
        }
    for (unsigned x = 0; x < 4; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({15 + x, 16 + x, "wall"});
    }
    for (unsigned y = 0; y < 3; ++y)
    {
        grid.boundaries.push_back({5 * y, 5 * (y + 1), "wall"});
        grid.boundaries.push_back({5 * y + 4, 5 * (y + 1) + 4, "wall"});
    }
    grid.z_edges = {0, .5, 1};
    const auto comm = Tpetra::getDefaultComm();
    const int active = empty_owners ? std::min(2, comm->getSize()) : comm->getSize();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 24; ++gid)
        if (int((gid % 4) * active / 4) == comm->getRank())
            owned.push_back(gid);
    AMR::ExtrudedAMR provider(grid);
    auto bootstrap = provider.detach_owned(owned, {'f', 'a', 'c', 'e', 't', 'r', 'a', 'n', 's', 'a', 'c', 't'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = ArithmeticMode::Ordinary;
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, liquid;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto shell =
            PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_geometry(*hierarchy->geometry(leaf), limits);
        const auto phase = shell->phases({{1, -.2, 0}, {0, 0, 0}, 1.75});
        order.push_back(key);
        gas.push_back(phase.gas.as_double());
        liquid.push_back(phase.carrier.as_double());
    }
    VoF::GeneralVoFOptions options;
    options.transport_backend = backend;
    options.phase_cleanup.relative_cutoff = 0;
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, liquid, options);
}
void expect_owners(const GeneralVoFSolver& solver, int active, bool rotated)
{
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    int local_empty = solver.mesh()->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - active);
    for (std::size_t c = 0; c < solver.mesh()->num_owned_cells(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        const double xi = x.x - .2 * x.y;
        int owner = std::min(active - 1, int(xi * active / 4));
        if (rotated)
            owner = (owner + 1) % active;
        EXPECT_EQ(owner, rank) << "xi=" << xi;
    }
    metric("asserted_empty_owners", empty);
}
void expect_gates(const GeneralVoFSolver& solver, const VoFStepDiagnostics& initial, Backend backend)
{
    const auto& after = solver.diagnostics();
    metric("maximum_divergence", after.maximum_divergence);
    metric("gas_mass_relative_drift", std::abs(after.gas_mass - initial.gas_mass) / initial.gas_mass);
    metric("liquid_mass_relative_drift", std::abs(after.liquid_mass - initial.liquid_mass) / initial.liquid_mass);
    EXPECT_LE(std::abs(after.gas_mass - initial.gas_mass), 2e-11 * initial.gas_mass);
    EXPECT_LE(std::abs(after.liquid_mass - initial.liquid_mass), 2e-11 * initial.liquid_mass);
    EXPECT_LT(after.maximum_divergence, 1e-8);
    EXPECT_EQ(solver.phases()->options().transport_backend, backend);
    EXPECT_EQ(solver.phases()->arithmetic_mode(), ArithmeticMode::Ordinary);
    long double local[2]{}, global[2]{};
    for (std::size_t c = 0; c < solver.phases()->phase_volumes().size(); ++c)
    {
        const auto& phase = solver.phases()->phase_volumes()[c];
        EXPECT_GE(phase.gas, 0);
        EXPECT_GE(phase.carrier, 0);
        EXPECT_NEAR(phase.gas + phase.carrier, phase.bulk, 1e-13 * phase.bulk);
        const auto history = solver.phases()->accumulated_phase_errors()[c];
        EXPECT_TRUE(std::isfinite(history.gas));
        EXPECT_TRUE(std::isfinite(history.carrier));
        EXPECT_GE(history.gas, 0);
        EXPECT_GE(history.carrier, 0);
        local[0] += history.gas;
        local[1] += history.carrier;
    }
    global[0] = sum_wide(*Tpetra::getDefaultComm(), local[0]);
    global[1] = sum_wide(*Tpetra::getDefaultComm(), local[1]);
    metric("global_gas_error_history", global[0]);
    metric("global_liquid_error_history", global[1]);
    for (const auto& m : solver.momentum_transport_diagnostics())
        EXPECT_LE(m.physical_relative_residual, 1e-12);
}
std::array<long double, 2> global_history(const GeneralVoFSolver& solver)
{
    std::array<long double, 2> local{}, global{};
    for (const auto& history : solver.phases()->accumulated_phase_errors())
    {
        local[0] += history.gas;
        local[1] += history.carrier;
    }
    global[0] = sum_wide(*Tpetra::getDefaultComm(), local[0]);
    global[1] = sum_wide(*Tpetra::getDefaultComm(), local[1]);
    return global;
}
void expect_fixed_clock_remap(const GeneralVoFSolver& solver, const VoFStepDiagnostics& before,
    const std::array<long double, 2>& history_before, bool changed = true)
{
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_LE(std::abs(solver.diagnostics().gas_mass - before.gas_mass), 2e-11 * before.gas_mass);
    EXPECT_LE(std::abs(solver.diagnostics().liquid_mass - before.liquid_mass), 2e-11 * before.liquid_mass);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], before.momentum[a], 2e-11);
    if (changed)
        EXPECT_FALSE(solver.prepared_transport());
    const auto history_after = global_history(solver);
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        const auto reduction_roundoff = 256 * std::numeric_limits<long double>::epsilon() *
                                        Tpetra::getDefaultComm()->getSize() *
                                        std::max(history_before[phase], history_after[phase]);
        EXPECT_GE(history_after[phase] + reduction_roundoff, history_before[phase]);
    }
}
using CellRow = std::array<double, 14>;
std::vector<CellRow> physical_fields(const GeneralVoFSolver& solver)
{
    const auto comm = Tpetra::getDefaultComm();
    int local_size = int(solver.mesh()->num_owned_cells()), maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_size, &maximum);
    constexpr int width = std::tuple_size_v<CellRow>;
    std::vector<double> local(maximum * width, std::numeric_limits<double>::infinity());
    std::vector<double> global(maximum * width * comm->getSize());
    for (int c = 0; c < local_size; ++c)
    {
        const auto x = solver.mesh()->cell_centroid(c);
        const auto history = solver.phases()->accumulated_phase_errors()[c];
        const CellRow row = {x.x, x.y, x.z, solver.phases()->gas_volumes()[c], solver.phases()->carrier_volumes()[c],
            solver.velocity()[c][0], solver.velocity()[c][1], solver.velocity()[c][2], solver.pressure()[c],
            solver.gas_fraction()[c], solver.curvature()[c], history.gas, history.carrier,
            solver.phases()->phase_volumes()[c].bulk};
        std::copy(row.begin(), row.end(), local.begin() + width * c);
    }
    Teuchos::gatherAll(*comm, int(local.size()), local.data(), int(global.size()), global.data());
    std::vector<CellRow> result;
    for (std::size_t i = 0; i < global.size(); i += width)
        if (std::isfinite(global[i]))
        {
            CellRow row;
            std::copy_n(global.begin() + i, width, row.begin());
            result.push_back(row);
        }
    std::ranges::sort(result);
    return result;
}
void expect_failure_preserves_publication(GeneralVoFSolver& solver)
{
    const auto phases = solver.phases();
    const auto mesh = solver.mesh();
    const auto prepared = solver.prepared_transport();
    const auto velocity = solver.velocity();
    const auto faces = solver.face_velocity();
    const auto pressure = solver.pressure();
    const auto fraction = solver.gas_fraction();
    const auto curvature = solver.curvature();
    const auto diagnostics = solver.diagnostics();
    const auto momentum = solver.momentum_transport_diagnostics();
    const auto histories = phases->accumulated_phase_errors();
    const auto gas = phases->gas_volumes();
    const auto liquid = phases->carrier_volumes();
    const auto control = phases->accepted_control();
    const auto capillary_model = solver.capillary_diagnostics().model;
    const auto capillary_fits = solver.capillary_diagnostics().fits.size();
    const auto capillary_components = solver.capillary_diagnostics().components.size();
    const auto comm = Tpetra::getDefaultComm();
    const int origin = comm->getSize() > 1 ? 1 : 0;
    std::vector<int> destinations(phases->owned_order().size(), comm->getRank());
    if (comm->getRank() == origin)
        destinations.push_back(comm->getSize()); // Rank-local invalid shape, before any accepted publication.
    bool rejected = false;
    try
    {
        solver.repartition(destinations);
    }
    catch (const SimpleFluid::detail::vof_failure::Failure& error)
    {
        EXPECT_EQ(error.record().origin_rank, origin);
        testing::Test::RecordProperty("injected_failure_stage", error.record().stage.data());
        metric("injected_failure_origin", error.record().origin_rank);
        rejected = true;
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(solver.phases(), phases);
    EXPECT_EQ(solver.phases()->accepted_control(), control);
    EXPECT_EQ(solver.phases()->gas_volumes(), gas);
    EXPECT_EQ(solver.phases()->carrier_volumes(), liquid);
    EXPECT_EQ(solver.capillary_diagnostics().model, capillary_model);
    EXPECT_EQ(solver.capillary_diagnostics().fits.size(), capillary_fits);
    EXPECT_EQ(solver.capillary_diagnostics().components.size(), capillary_components);
    EXPECT_EQ(solver.mesh(), mesh);
    EXPECT_EQ(solver.prepared_transport(), prepared);
    EXPECT_EQ(solver.velocity(), velocity);
    EXPECT_EQ(solver.face_velocity(), faces);
    EXPECT_EQ(solver.pressure(), pressure);
    EXPECT_EQ(solver.gas_fraction(), fraction);
    EXPECT_EQ(solver.curvature(), curvature);
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
    for (unsigned a = 0; a < 3; ++a)
    {
        EXPECT_EQ(solver.momentum_transport_diagnostics()[a].physical_relative_residual,
            momentum[a].physical_relative_residual);
        EXPECT_EQ(solver.momentum_transport_diagnostics()[a].implicit_iterations, momentum[a].implicit_iterations);
        EXPECT_EQ(solver.momentum_transport_diagnostics()[a].defect_corrections, momentum[a].defect_corrections);
    }
    ASSERT_EQ(solver.phases()->accumulated_phase_errors().size(), histories.size());
    for (std::size_t c = 0; c < histories.size(); ++c)
    {
        EXPECT_EQ(solver.phases()->accumulated_phase_errors()[c].gas, histories[c].gas);
        EXPECT_EQ(solver.phases()->accumulated_phase_errors()[c].carrier, histories[c].carrier);
    }
}
void moving_mixed_transactions(Backend backend, bool empty_owners)
{
    const bool cleanup = qualification_phase_cleanup_enabled() && backend != Backend::DonorCell;
    testing::Test::RecordProperty("qualification_cleanup_enabled", cleanup ? 1 : 0);
    testing::Test::RecordProperty("configured_backend", backend == Backend::DonorCell ? "DonorCell" : "GeometricFace");
    if (cleanup)
        GTEST_SKIP() << "Cleanup-enabled coupled regrid is unsupported; phase-only cleanup migration "
                        "and coupled-channel rejection are checked separately.";

    // Preregister a complete .02s accepted horizon: motion to .004, refine,
    // motion to .008, coarsen, motion to .012, repartition, motion to .02.
    constexpr unsigned total_steps = 10;
    constexpr double dt = .002, endpoint = .02;
    metric("physical_endpoint", endpoint);
    metric("configured_steps", total_steps);
    metric("refine_after_step", 2);
    metric("coarsen_after_step", 4);
    metric("repartition_after_step", 6);
    testing::Test::RecordProperty("configured_backend", backend == Backend::DonorCell ? "DonorCell" : "GeometricFace");
    auto material = inviscid();
    material.gas_density = 1;
    material.liquid_density = 1000;
    GeneralVoFSolver solver(spatial_mixed_phases(backend, empty_owners), material);
    const auto comm = Tpetra::getDefaultComm();
    const int active = empty_owners ? std::min(2, comm->getSize()) : comm->getSize();
    expect_owners(solver, active, false);
    EXPECT_EQ(solver.phases()->hierarchy()->global_cells(), 24u);
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(int(c));
        const double a = std::numbers::pi * (x.x - .2 * x.y) / 4, b = std::numbers::pi * x.y / 3;
        const double vx = .1 * std::pow(std::sin(a), 2) * std::sin(2 * b);
        const double vy = -.075 * std::pow(std::sin(b), 2) * std::sin(2 * a);
        velocity[c] = {vx + .2 * vy, vy, 0};
    }
    solver.initialize(velocity);
    const auto initial = solver.diagnostics();
    const auto initial_gas = solver.phases()->gas_volumes();
    std::shared_ptr<const VoF::GeneralVoFPreparedTransport> previous;
    std::uint64_t refined_cells = 0;
    std::vector<AMR::AdaptiveCellKey> refinement_parents;
    for (unsigned step = 0; step < total_steps; ++step)
    {
        SCOPED_TRACE(::testing::Message() << "step=" << step << ", time=" << solver.time());
        ASSERT_LE(dt, solver.stable_timestep());
        const auto source = solver.phases();
        metric("attempted_time", solver.time());
        ASSERT_NO_THROW((void) solver.step(dt));
        const auto p = solver.prepared_transport();
        ASSERT_TRUE(p);
        EXPECT_NE(p, previous);
        EXPECT_EQ(p->backend, backend);
        EXPECT_EQ(p->family, VoF::GeneralVoFTransportFamily::AlgebraicFaceVolume);
        EXPECT_EQ(p->source_identity, source->accepted_identity());
        EXPECT_EQ(p->candidate_identity, solver.phases()->accepted_identity());
        EXPECT_EQ(p->mesh_identity, source->hierarchy());
        EXPECT_EQ(p->mesh_generation, source->hierarchy()->partition_generation());
        EXPECT_EQ(p->mesh_topology_generation, source->hierarchy()->topology_generation());
        if (previous && previous->mesh_identity != source->hierarchy())
            EXPECT_NE(p->momentum_preparation, previous->momentum_preparation);
        EXPECT_EQ(p->momentum_builds, 1u);
        EXPECT_EQ(p->momentum_reuses, 2u);
        testing::Test::RecordProperty("actual_prepared_family", "AlgebraicFaceVolume");
        metric("accepted_steps", step + 1);
        metric("accepted_time", solver.time());
        expect_gates(solver, initial, backend);
        previous = p;
        if (step == 1)
        {
            double local_change = 0, change = 0;
            int local_mixed = 0, mixed = 0;
            for (std::size_t c = 0; c < initial_gas.size(); ++c)
            {
                local_change += std::abs(solver.phases()->gas_volumes()[c] - initial_gas[c]);
                local_mixed += solver.phases()->gas_volumes()[c] > 0 && solver.phases()->carrier_volumes()[c] > 0;
            }
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_change, &change);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_mixed, &mixed);
            EXPECT_GT(change, 1e-8);
            EXPECT_GT(mixed, 0);
            EXPECT_GT(solver.time(), 0);
            metric("pre_remap_gas_redistribution", change);
            expect_failure_preserves_publication(solver);
            std::vector<AMR::AdaptiveMark> marks;
            for (const auto& [key, leaf] : solver.phases()->hierarchy()->owned())
            {
                const auto x = solver.phases()->hierarchy()->geometry(leaf)->centroid();
                const double xi = x[0] - .2 * x[1];
                if (xi > 1 && xi < 2 && x[1] > 1 && x[1] < 2)
                {
                    marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
                    refinement_parents.push_back(key);
                }
            }
            const auto before = solver.diagnostics();
            const auto history_before = global_history(solver);
            testing::Test::RecordProperty("attempted_transaction", "refine_after_accepted_mixed_motion");
            ASSERT_NO_THROW(solver.regrid(marks));
            expect_fixed_clock_remap(solver, before, history_before);
            expect_gates(solver, initial, backend);
            refined_cells = solver.phases()->hierarchy()->global_cells();
            EXPECT_GT(refined_cells, 24u);
            metric("refined_global_cells", refined_cells);
        }
        if (step == 3)
        {
            // Product ancestry retains overlapping Base, Axial and Isotropic
            // capsules. Request only the inverse of each explicitly marked
            // parent, rather than retiring the same child through several capsules.
            const auto hierarchy = solver.phases()->hierarchy();
            std::vector<AMR::AdaptiveBytes> families;
            for (const auto& parent : refinement_parents)
                families.push_back(AMR::adaptive_family_key(parent, AMR::AdaptiveSplit::Isotropic));
            std::vector<AMR::AdaptiveFamily> selected;
            ASSERT_NO_THROW(selected = hierarchy->fetch_families(families));
            std::set<AMR::AdaptiveCellKey> expected_parents(refinement_parents.begin(), refinement_parents.end());
            std::set<AMR::AdaptiveCellKey> unique_children;
            std::vector<AMR::AdaptiveCellKey> children;
            int local_valid = selected.size() == families.size();
            for (const auto& family : selected)
            {
                local_valid &= family.split == AMR::AdaptiveSplit::Isotropic &&
                               expected_parents.contains(family.parent.key) && !family.children.empty();
                for (const auto& child : family.children)
                {
                    local_valid &= unique_children.insert(child).second;
                    children.push_back(child);
                }
            }
            const auto owners = hierarchy->lookup_cells(children, false);
            local_valid &= owners.size() == children.size();
            std::vector<OwnerPacket> requests;
            for (const auto& owner : owners)
            {
                local_valid &= owner.owner >= 0;
                if (owner.owner >= 0)
                    requests.push_back({owner.owner, owner.key});
            }
            // Check uniqueness at actual child owners as well: this detects
            // cross-rank overlap between requested sibling families.
            const auto received = exchange_owner_packets(requests, comm, hierarchy->options().byte_budget);
            std::set<AMR::AdaptiveBytes> retiring;
            for (const auto& packet : received.packets)
            {
                local_valid &= retiring.insert(packet.bytes).second;
                local_valid &= hierarchy->owned().contains(AMR::decode_adaptive_key(packet.bytes));
            }
            int globally_valid = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_valid, &globally_valid);
            const int local_count = int(families.size());
            int chosen_count = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_count, &chosen_count);
            metric("chosen_disjoint_isotropic_families", chosen_count);
            ASSERT_GT(chosen_count, 0);
            ASSERT_EQ(globally_valid, 1) << "Coarsening requires complete active, disjoint requested sibling families.";
            const auto before = solver.diagnostics();
            const auto history_before = global_history(solver);
            testing::Test::RecordProperty("attempted_transaction", "coarsen_after_continued_mixed_motion");
            ASSERT_NO_THROW(solver.regrid({}, families));
            expect_fixed_clock_remap(solver, before, history_before);
            expect_gates(solver, initial, backend);
            EXPECT_LT(solver.phases()->hierarchy()->global_cells(), refined_cells);
            metric("coarsened_global_cells", solver.phases()->hierarchy()->global_cells());
        }
        if (step == 5)
        {
            const auto fields = physical_fields(solver);
            const auto before = solver.diagnostics();
            const auto before_phases = solver.phases();
            const auto before_mesh = solver.mesh();
            const auto before_prepared = solver.prepared_transport();
            const auto history_before = global_history(solver);
            const auto partition_generation = solver.phases()->hierarchy()->partition_generation();
            std::vector<int> destinations(solver.phases()->owned_order().size(), (comm->getRank() + 1) % active);
            const int local_change = std::any_of(
                destinations.begin(), destinations.end(), [&](int owner) { return owner != comm->getRank(); });
            int changed = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_change, &changed);
            testing::Test::RecordProperty("attempted_transaction", "spatial_repartition_after_continued_mixed_motion");
            ASSERT_NO_THROW(solver.repartition(destinations));
            expect_fixed_clock_remap(solver, before, history_before, changed);
            if (!changed)
            {
                // Same-owner migration is a validated identity operation. It
                // preserves the exact publication, including its current ledger;
                // actual ownership changes must still invalidate preparation.
                EXPECT_EQ(solver.phases(), before_phases);
                EXPECT_EQ(solver.mesh(), before_mesh);
                EXPECT_EQ(solver.prepared_transport(), before_prepared);
            }
            testing::Test::RecordProperty("ownership_migration_changed", changed);
            expect_gates(solver, initial, backend);
            if (changed)
                EXPECT_GT(solver.phases()->hierarchy()->partition_generation(), partition_generation);
            else
                EXPECT_EQ(solver.phases()->hierarchy()->partition_generation(), partition_generation);
            expect_owners(solver, active, true);
            const auto migrated = physical_fields(solver);
            ASSERT_EQ(migrated.size(), fields.size());
            if (!changed)
                EXPECT_EQ(migrated, fields);
            for (std::size_t c = 0; c < fields.size(); ++c)
                for (std::size_t a = 0; a < fields[c].size(); ++a)
                    if (a == 3 || a == 4 || a == 11 || a == 12)
                        EXPECT_EQ(migrated[c][a], fields[c][a]); // Independent phases and histories copy.
                    else
                        EXPECT_NEAR(migrated[c][a], fields[c][a], 1e-12 * std::max(1., std::abs(fields[c][a])));
        }
    }
    const auto final_fields = physical_fields(solver);
    std::ostringstream field_manifest;
    field_manifest << std::setprecision(20);
    for (const auto& row : final_fields)
    {
        for (const auto value : row)
            field_manifest << value << ',';
        field_manifest << ';';
    }
    testing::Test::RecordProperty("final_physical_field_columns",
        "x,y,z,gas,liquid,ux,uy,uz,pressure,gas_fraction,curvature,gas_history,liquid_history,bulk");
    testing::Test::RecordProperty("final_physical_fields", field_manifest.str());
    metric("actual_ranks", comm->getSize());
    EXPECT_NEAR(solver.time(), endpoint, 1e-16);
    expect_owners(solver, active, true);
}
TEST(GeneralVoFFaceTransactions, DonorCellMovingMixedSpatialRefineCoarsenRepartition)
{
    moving_mixed_transactions(Backend::DonorCell, false);
}
TEST(GeneralVoFFaceTransactions, GeometricFaceMovingMixedSpatialRefineCoarsenRepartition)
{
    moving_mixed_transactions(Backend::GeometricFace, false);
}
TEST(GeneralVoFFaceTransactions, DonorCellMovingMixedEmptyOwnersRefineCoarsenRepartition)
{
    moving_mixed_transactions(Backend::DonorCell, true);
}
TEST(GeneralVoFFaceTransactions, GeometricFaceMovingMixedEmptyOwnersRefineCoarsenRepartition)
{
    moving_mixed_transactions(Backend::GeometricFace, true);
}
} // namespace
