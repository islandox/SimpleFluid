/** @file testGeneralVoFFinite.cc */
#include "FVM/details/BoundedVoFTransport.hh"
#include "general_vof_finite_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
void check_budget_diagnostics(const GeneralVoF& state, const GeneralVoF::Trial& trial)
{
    const auto& faces = trial.finite_face_chain_diagnostics();
    const auto& cells = trial.finite_cell_closure_diagnostics();
    const auto comm = Tpetra::getDefaultComm();
    const auto collectively_valid = [&](bool valid)
    {
        int local = valid, global = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &global);
        return global != 0;
    };
    bool valid = faces.size() == state.tiles().size() && cells.size() == state.owned_order().size();
    EXPECT_EQ(faces.size(), state.tiles().size());
    EXPECT_EQ(cells.size(), state.owned_order().size());
    std::map<std::uint64_t, const GeneralVoFFaceChainDiagnostic*> by_face;
    std::uint64_t local_max = 0, global_max = 0;
    for (const auto& face : faces)
    {
        const bool inserted = by_face.emplace(face.face, &face).second;
        EXPECT_TRUE(inserted);
        valid = valid && inserted && std::isfinite(face.control.value) &&
            std::isfinite(face.constructed.value) && std::isfinite(face.signed_difference.value) &&
            std::isfinite(face.allowed_difference);
        local_max = std::max(local_max, face.face);
        EXPECT_EQ(face.control.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(face.constructed.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(face.signed_difference.value, face.constructed.value - face.control.value);
        EXPECT_GT(face.allowed_difference, 0);
        EXPECT_LE(std::abs(face.signed_difference.value), face.allowed_difference);
    }
    for (const auto& tile : state.tiles()) valid = valid && by_face.contains(tile.face_gid);
    if (!collectively_valid(valid)) return;
    const auto geometry = state.owned_finite_geometry();
    valid = geometry.size() == cells.size();
    EXPECT_EQ(geometry.size(), cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c)
        valid = valid && cells[c].key == state.owned_order()[c];
    if (!collectively_valid(valid)) return;
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
        const auto& cell = cells[c];
        EXPECT_EQ(cell.key, state.owned_order()[c]);
        std::size_t count = 0;
        fp::Sum budgets(ArithmeticMode::Ordinary), actual(ArithmeticMode::Ordinary);
        for (const auto& tile : state.tiles())
            if (tile.first == cell.key || (tile.second && *tile.second == cell.key))
            {
                ++count;
                const auto& face = *by_face.at(tile.face_gid);
                budgets.add(fp::make_value(ArithmeticMode::Ordinary, face.allowed_difference));
                actual.add(fp::make_value(ArithmeticMode::Ordinary, std::abs(face.signed_difference.value)));
                EXPECT_LE(face.allowed_difference,
                    state.options().reconstruction.volume_tolerance * cell.quota_scale);
            }
        EXPECT_EQ(cell.canonical_face_count, count);
        EXPECT_GT(cell.quota_scale, 0);
        EXPECT_LE(cell.quota_scale * count,
            std::min(static_cast<long double>(state.phase_volumes()[c].bulk), geometry[c]->volume().value));
        EXPECT_LE(budgets.value().value, cell.allowed_closure);
        EXPECT_EQ(cell.absolute_geometry_difference, actual.value().value);
        EXPECT_EQ(cell.combined_residual.value,
            cell.canonical_residual.value + cell.signed_geometry_difference.value);
        EXPECT_EQ(cell.allowed_closure,
            state.options().reconstruction.volume_tolerance * geometry[c]->volume().value);
        EXPECT_LE(std::abs(cell.combined_residual.value), cell.allowed_closure);
    }
    // Compare actual replica quantities, not just their configured allowances.
    // A high/low double pair retains every long-double bit in these fixtures.
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max, &global_max);
    EXPECT_LT(global_max, 1000000u);
    if (global_max >= 1000000u) return; // The same reduced value governs every rank.
    const auto size = std::size_t(global_max + 1);
    std::vector<double> low(8 * size, std::numeric_limits<double>::infinity());
    std::vector<double> high(8 * size, -std::numeric_limits<double>::infinity());
    std::vector<double> minimum(low.size()), maximum(high.size());
    for (const auto& face : faces)
    {
        const std::array values{face.control.value, face.constructed.value,
            face.signed_difference.value, face.allowed_difference};
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double head = static_cast<double>(values[i]);
            const double tail = static_cast<double>(values[i] - static_cast<long double>(head));
            EXPECT_EQ(static_cast<long double>(head) + tail, values[i]);
            const auto slot = 8 * face.face + 2 * i;
            low[slot] = high[slot] = head;
            low[slot + 1] = high[slot + 1] = tail;
        }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, int(low.size()), low.data(), minimum.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(high.size()), high.data(), maximum.data());
    for (std::size_t i = 0; i < minimum.size(); ++i)
        if (std::isfinite(minimum[i])) EXPECT_EQ(minimum[i], maximum[i]);
}

TEST(GeneralVoFFinite, OrdinaryDefaultAndBoundedOptInKeepSeparateAcceptancePolicies)
{
    const auto input = fixture();
    GeneralVoF ordinary(input.hierarchy, input.order, input.gas, input.carrier);
    GeneralVoFOptions options;
    options.arithmetic_mode = ArithmeticMode::BoundedCertification;
    GeneralVoF certified(input.hierarchy, input.order, input.gas, input.carrier, options);
    EXPECT_EQ(ordinary.arithmetic_mode(), ArithmeticMode::Ordinary);
    EXPECT_EQ(certified.arithmetic_mode(), ArithmeticMode::BoundedCertification);
    auto controls = circulation(ordinary);
    for (auto& face : controls)
        face.bounded_integrated_volume = BoundedGeometryValue{face.integrated_volume, 1e-3L};
    // An Ordinary state explicitly consumes nominal controls. Opt-in interval
    // certification must still reject these unresolved source-cell controls.
    const auto trial = ordinary.plan_transport(stationary(ordinary), controls, 1);
    const auto moved = ordinary.apply(trial);
    EXPECT_EQ(moved->arithmetic_mode(), ArithmeticMode::Ordinary);
    EXPECT_THROW((void) certified.plan_transport(stationary(certified), controls, 1), std::invalid_argument);
    EXPECT_THROW((void) certified.apply(trial), std::invalid_argument);
    EXPECT_EQ(certified.gas_volumes(), input.gas);
    EXPECT_EQ(certified.carrier_volumes(), input.carrier);
}

TEST(GeneralVoFFinite, OrdinaryModeRejectsInvalidPhysicalDataAndImplicitExactViews)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    EXPECT_THROW((void) state.inventory_measures(), std::invalid_argument);
    EXPECT_THROW((void) state.geometric_phases(), std::invalid_argument);
    EXPECT_THROW((void) state.export_transfer_source(false), std::invalid_argument);
    auto gas = input.gas;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !gas.empty()) gas.front() = -1;
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, gas, input.carrier), std::invalid_argument);
    gas = input.gas;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !gas.empty())
        gas.front() = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, gas, input.carrier), std::invalid_argument);
    GeneralVoFOptions mismatch;
    mismatch.arithmetic_mode = Tpetra::getDefaultComm()->getSize() == 1 ? static_cast<ArithmeticMode>(99) :
        Tpetra::getDefaultComm()->getRank() == 0 ? ArithmeticMode::Ordinary : ArithmeticMode::BoundedCertification;
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, mismatch),
        std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}

TEST(GeneralVoFFinite, SparseOverlayRetainsDisjointAndContactAtomsAndCoefficients)
{
    const std::vector<fp::OverlayAtom> atoms{
        {BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1}), 0},
        {BoundedConvexGeometry::box({0, 0, 2}, {1, 1, 3}), 1},
        {BoundedConvexGeometry::box({0, 0, 4}, {1, 1, 5}), -1}};
    const std::array pieces{BoundedConvexGeometry::box({3, 0, 0}, {4, 1, 5}),
        BoundedConvexGeometry::box({1, 0, 0}, {2, 1, 5})};
    for (const auto& geometry : pieces)
        for (const int sign : {-1, 1})
            for (const int incidence : {-1, 1})
            {
                fp::Work work;
                work.limits.maximum_pair_tests = atoms.size();
                reset_bounded_geometry_statistics();
                const auto result = fp::overlay(atoms, {geometry, sign}, incidence, work);
                ASSERT_EQ(result.size(), atoms.size());
                EXPECT_EQ(work.pair_tests, atoms.size());
                EXPECT_GT(work.bytes, 0u);
                for (std::size_t i = 0; i < atoms.size(); ++i)
                {
                    EXPECT_EQ(result[i].geometry.get(), atoms[i].geometry.get());
                    EXPECT_EQ(result[i].coefficient, atoms[i].coefficient);
                }
                const auto stats = bounded_geometry_statistics();
                EXPECT_EQ(stats.support_distances, 0u);
                EXPECT_EQ(stats.volume_evaluations, 0u);
                EXPECT_EQ(stats.constructions, 0u);
                EXPECT_EQ(stats.intersections, 0u);
            }
}

TEST(GeneralVoFFinite, SparseOverlayRetainsZeroCoefficientAtomsForLaterActivation)
{
    const auto root = BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto far = BoundedConvexGeometry::box({3, 0, 0}, {4, 1, 1});
    const auto half = BoundedConvexGeometry::box({.5, 0, 0}, {1, 1, 1});
    fp::Work work;
    auto atoms = fp::overlay({{root, 0}}, {far, -1}, -1, work);
    ASSERT_EQ(atoms.size(), 1u);
    EXPECT_EQ(atoms.front().geometry.get(), root.get());
    EXPECT_EQ(atoms.front().coefficient, 0);
    atoms = fp::overlay(std::move(atoms), {half, 1}, 1, work);
    ASSERT_EQ(atoms.size(), 2u);
    unsigned zero = 0, one = 0;
    for (const auto& atom : atoms)
    {
        zero += atom.coefficient == 0;
        one += atom.coefficient == 1;
        const auto volume = atom.geometry->volume();
        EXPECT_LE(volume.lower(), .5L);
        EXPECT_GE(volume.upper(), .5L);
    }
    EXPECT_EQ(zero, 1u);
    EXPECT_EQ(one, 1u);
    EXPECT_EQ(root->volume().value, 1);
    EXPECT_EQ(root->volume().absolute_error, 0);
}

TEST(GeneralVoFFinite, SparseOverlayFastPathStillChargesPairAtomAndByteCaps)
{
    const auto root = BoundedConvexGeometry::box({0, 0, 0}, {1, 1, 1});
    const fp::SignedPiece far{BoundedConvexGeometry::box({3, 0, 0}, {4, 1, 1}), 1};
    const std::vector<fp::OverlayAtom> atoms{{root, 0}};
    const auto rejected = [&](fp::Work& work, const char* reason)
    {
        try
        {
            (void) fp::overlay(atoms, far, 1, work);
            FAIL() << "Sparse overlay must honor the original resource caps.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find(reason), std::string::npos);
        }
    };
    fp::Work pair;
    pair.limits.maximum_pair_tests = 0;
    rejected(pair, "pair-test budget");
    EXPECT_EQ(pair.pair_tests, 1u);
    fp::Work atom;
    atom.limits.maximum_atoms = 0;
    rejected(atom, "atom buffer capacity");
    EXPECT_EQ(atom.pair_tests, 1u);
    fp::Work bytes;
    bytes.limits.byte_budget = sizeof(fp::OverlayAtom) - 1;
    rejected(bytes, "byte budget");
    EXPECT_EQ(bytes.pair_tests, 1u);
    fp::Work exact;
    exact.limits.byte_budget = sizeof(fp::OverlayAtom);
    const auto result = fp::overlay(atoms, far, 1, exact);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().geometry.get(), root.get());
    EXPECT_EQ(result.front().coefficient, 0);
    EXPECT_EQ(exact.bytes, sizeof(fp::OverlayAtom));
    EXPECT_EQ(root->volume().value, 1);
    EXPECT_EQ(root->volume().absolute_error, 0);
}

TEST(GeneralVoFFinite, OrdinaryTransportKeepsIndependentPhasesCanonicalRoutesAndEmptyOwners)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    EXPECT_EQ(state.arithmetic_mode(), ArithmeticMode::Ordinary);
    for (const auto& geometry : state.owned_finite_geometry())
        EXPECT_EQ(geometry->arithmetic_mode(), ArithmeticMode::Ordinary);
    const auto velocities = stationary(state);
    const auto controls = circulation(state);
    struct StatisticsScope
    {
        StatisticsScope()
        {
            GeneralVoF::enable_exact_geometry_statistics(true);
            GeneralVoF::reset_geometry_statistics();
        }
        ~StatisticsScope() { GeneralVoF::enable_exact_geometry_statistics(false); }
    } statistics_scope;
    const auto trial = state.plan_transport(velocities, controls, 1);
    const auto candidate = state.apply(trial);
    check_budget_diagnostics(state, trial);
    EXPECT_TRUE(trial.transfers().empty());
    EXPECT_TRUE(trial.donor_budgets().empty());
    ASSERT_EQ(trial.finite_donor_budgets().size(), input.order.size());
    ASSERT_EQ(candidate->accumulated_phase_errors().size(), input.order.size());
    fp::Sum gas(state.arithmetic_mode()), carrier(state.arithmetic_mode());
    for (std::size_t i = 0; i < candidate->phase_volumes().size(); ++i)
    {
        const auto& p = candidate->phase_volumes()[i];
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        gas.add(fp::make_value(state.arithmetic_mode(), p.gas, trial.phase_roundoff_bounds()[i].gas));
        carrier.add(fp::make_value(state.arithmetic_mode(), p.carrier, trial.phase_roundoff_bounds()[i].carrier));
        EXPECT_GE(candidate->accumulated_phase_errors()[i].gas, trial.phase_roundoff_bounds()[i].gas);
        EXPECT_GE(candidate->accumulated_phase_errors()[i].carrier, trial.phase_roundoff_bounds()[i].carrier);
    }
    const auto comm = Tpetra::getDefaultComm();
    double local[4] = {static_cast<double>(gas.value().value), static_cast<double>(carrier.value().value),
        fp::upper_double(gas.value().absolute_error), fp::upper_double(carrier.value().absolute_error)}, global[4]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 4, local, global);
    EXPECT_NEAR(global[0], 1., global[2] + 1e-14);
    EXPECT_NEAR(global[1], 7., global[3] + 1e-14);
    for (const auto& route : trial.finite_transfers())
    {
        EXPECT_EQ(route.recipient_owner, comm->getRank());
        EXPECT_EQ(route.amount.bulk.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(route.amount.gas.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(route.amount.carrier.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_GE(route.amount.gas.value, 0);
        EXPECT_GE(route.amount.carrier.value, 0);
    }
    int local_empty = input.order.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    const auto observed = GeneralVoF::geometry_statistics();
    const auto& statistics = observed.bounded;
    EXPECT_EQ(observed.exact.source_coordinate_exports, 0);
    EXPECT_EQ(observed.exact.source_support_exports, 0);
    EXPECT_EQ(observed.exact.volume_evaluations, 0);
    EXPECT_EQ(observed.exact.moment_evaluations, 0);
    EXPECT_EQ(observed.exact.clip_constructions, 0);
    EXPECT_EQ(observed.exact.intersection_constructions, 0);
    EXPECT_EQ(observed.exact.measure_values, 0);
    EXPECT_EQ(observed.exact.measure_arithmetic, 0);
    EXPECT_EQ(observed.exact.cover_certifications, 0);
    EXPECT_EQ(statistics.source_imports, 0);
    EXPECT_EQ(statistics.interval_operations, 0);
    const std::uint64_t local_work = statistics.constructions + statistics.volume_evaluations;
    std::uint64_t global_work = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_work, &global_work);
    EXPECT_GT(global_work, 0);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, RankLocalMalformedAndCrossPolicyControlsRejectWithoutChangingPublication)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto identity = state.accepted_identity();
    const auto control = state.accepted_control();
    auto faces = circulation(state);
    if (Tpetra::getDefaultComm()->getRank() == 0 && !faces.empty()) faces.front().face ^= 1;
    EXPECT_THROW(state.plan_transport(stationary(state), faces, 1), std::invalid_argument);
    EXPECT_EQ(state.accepted_identity(), identity);
    EXPECT_EQ(state.accepted_control(), control);
    faces = circulation(state, true);
    if (Tpetra::getDefaultComm()->getRank() == 0 && !faces.empty())
        faces.front().bounded_integrated_volume = BoundedGeometryValue{0, 0, ArithmeticMode::Exact};
    EXPECT_THROW(state.plan_transport(stationary(state), faces, 1), std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, OwnerFaceBudgetsCoverCoarseFineTilesAndEmptyOwners)
{
    const auto input = coarse_fine_fixture();
    ASSERT_EQ(input.hierarchy->global_cells(), 72u);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    auto controls = circulation(state, true);
    const auto trial = state.plan_transport(stationary(state), controls, 1);
    check_budget_diagnostics(state, trial);
    std::uint64_t local_count = 0, maximum = 0;
    for (const auto& cell : trial.finite_cell_closure_diagnostics())
        local_count = std::max(local_count, std::uint64_t(cell.canonical_face_count));
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_count, &maximum);
    EXPECT_GT(maximum, 6u); // The coarse seam uses actual fine subfaces.
    int empty = input.order.empty(), total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, std::max(0, comm->getSize() - 2));
}

TEST(GeneralVoFFinite, OrdinaryFaceAllowanceCannotHideMaterialCellControlDefect)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    auto controls = circulation(state, true);
    std::uint64_t local = std::numeric_limits<std::uint64_t>::max(), chosen = 0;
    for (const auto& face : state.tiles())
        if (!face.exterior) local = std::min(local, face.face_gid);
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &chosen);
    for (auto& face : controls)
        if (face.face == chosen) face.integrated_volume = 1e-3;
    const auto identity = state.accepted_identity();
    const auto control = state.accepted_control();
    EXPECT_THROW((void) state.plan_transport(stationary(state), controls, 1), std::invalid_argument);
    EXPECT_EQ(state.accepted_identity(), identity);
    EXPECT_EQ(state.accepted_control(), control);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, UnderresolvedNeighborhoodRequestsRefinementWithoutDeletingAcceptedPhase)
{
    const auto input = fixture(true);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto requests = state.resolution_requests();
    unsigned count = unsigned(requests.size()), global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &count, &global);
    EXPECT_EQ(global, 8u);
    for (const auto& request : requests)
    {
        EXPECT_FALSE(request.assessment.admitted());
        EXPECT_GT(request.assessment.local_cell_span, request.assessment.requested_cell_span);
    }
    EXPECT_THROW(state.reconstruct(), std::invalid_argument);
    EXPECT_THROW(state.plan_transport(stationary(state), circulation(state), 1), std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
    const auto unchanged = state.plan_transport(stationary(state), circulation(state, true), 1);
    EXPECT_EQ(state.apply(unchanged)->accepted_identity(), state.accepted_identity());
}

std::shared_ptr<const DistributedAdaptiveHierarchy> bound_candidate(
    const std::shared_ptr<const DistributedAdaptiveHierarchy>& source, const DistributedAdaptiveHierarchy::Plan& plan)
{
    const auto raw = source->apply(plan);
    const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    return source->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
}
double original_cell_marker(const AdaptiveCellKey& key)
{
    const auto& ancestry = std::get<CartesianCellAncestry>(key.ancestry);
    double marker = 1;
    for (unsigned a = 0; a < 3; ++a)
        marker += (1u << a) * std::floor(std::ldexp(double(ancestry.index[a]) + .5, -int(ancestry.level)));
    return marker;
}
GeneralVoFFiniteRemapOptions channel_options()
{
    GeneralVoFFiniteRemapOptions options;
    options.phase_density = {.1, 1};
    options.channels = {{"gas_marker", GeneralVoFFiniteRemapCarrier::GasVolume, true},
        {"cell_marker", GeneralVoFFiniteRemapCarrier::BulkVolume, true}};
    return options;
}
std::vector<GeneralVoFFiniteOwnedChannels> initial_channels(const GeneralVoF& state)
{
    std::vector<GeneralVoFFiniteOwnedChannels> result;
    for (std::size_t c = 0; c < state.owned_order().size(); ++c)
    {
        const auto& key = state.owned_order()[c];
        result.push_back({key, {{2 * state.gas_volumes()[c], 0},
            {state.phase_volumes()[c].bulk * original_cell_marker(key), 0}}});
    }
    return result;
}
void check_remap(const GeneralVoF::FiniteRemapTrial& trial, unsigned expected_cells)
{
    const auto state = trial.candidate();
    EXPECT_EQ(state->inventory_policy(), PhaseInventoryPolicy::Finite);
    EXPECT_EQ(state->hierarchy()->global_cells(), expected_cells);
    std::map<AdaptiveCellKey, const GeneralVoFFiniteOwnedChannels*> channels;
    for (const auto& item : trial.owned_channels())
        ASSERT_TRUE(channels.emplace(item.key, &item).second);
    double local[2]{}, global[2]{};
    for (std::size_t c = 0; c < state->owned_order().size(); ++c)
    {
        const auto& key = state->owned_order()[c];
        const auto& p = state->phase_volumes()[c];
        const auto& output = channels.at(key)->amounts;
        ASSERT_EQ(output.size(), 2u);
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        EXPECT_NEAR(output[0].value, 2 * p.gas, output[0].absolute_error + 1e-14);
        EXPECT_NEAR(output[1].value, p.bulk * original_cell_marker(key), output[1].absolute_error + 1e-14);
        local[0] += p.gas; local[1] += p.carrier;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local, global);
    EXPECT_NEAR(global[0], 1, 2e-14);
    EXPECT_NEAR(global[1], 7, 2e-14);
    trial.validate_observations();
}
TEST(GeneralVoFFinite, FiniteRefineCoarsenPreservesPhasesWeightedChannelsAndSource)
{
    const auto input = fixture();
    const auto source = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier);
    const auto control = source->accepted_control();
    std::vector<AdaptiveMark> marks;
    for (std::size_t c = 0; c < source->owned_order().size(); ++c)
        if (source->gas_volumes()[c] == 1)
            marks.push_back({source->owned_order()[c], AdaptiveSplit::Isotropic});
    const auto plan = source->hierarchy()->plan_adaptation(marks);
    const auto target = bound_candidate(source->hierarchy(), plan);
    const auto trial = source->plan_remap(plan, target, initial_channels(*source), channel_options());
    EXPECT_FALSE(trial.ownership_only());
    // The Cartesian provider refines the complete native block containing
    // the mark: all eight original leaves acquire eight children.
    check_remap(trial, 64);
    EXPECT_EQ(source->accepted_control(), control);
    EXPECT_EQ(source->gas_volumes(), input.gas);
    EXPECT_EQ(source->carrier_volumes(), input.carrier);
    const auto refined = trial.candidate();
    std::vector<AdaptiveBytes> families;
    for (const auto& [key, family] : refined->hierarchy()->families())
        families.push_back(key);
    const auto coarsen = refined->hierarchy()->plan_adaptation({}, families);
    const auto restored = refined->plan_remap(
        coarsen, bound_candidate(refined->hierarchy(), coarsen), trial.owned_channels(), channel_options());
    check_remap(restored, 8);
}
TEST(GeneralVoFFinite, FiniteMigrationCopiesInventoriesAndRejectsForeignSchemaAtomically)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> destinations(source.hierarchy()->owned().size(), (comm->getRank() + 1) % comm->getSize());
    const auto plan = source.hierarchy()->plan_migration(destinations);
    const auto target = bound_candidate(source.hierarchy(), plan);
    const auto channels = initial_channels(source);
    const auto trial = source.plan_remap(plan, target, channels, channel_options());
    EXPECT_TRUE(trial.ownership_only());
    check_remap(trial, 8);
    const auto migrated = trial.candidate();
    for (std::size_t c = 0; c < migrated->owned_order().size(); ++c)
    {
        const auto gas = original_cell_marker(migrated->owned_order()[c]) == 1 ? 1. : 0.;
        EXPECT_EQ(migrated->gas_volumes()[c], gas);
        EXPECT_EQ(migrated->carrier_volumes()[c], 1 - gas);
        EXPECT_EQ(migrated->phase_roundoff_bounds()[c].gas, 0);
        EXPECT_EQ(migrated->phase_roundoff_bounds()[c].carrier, 0);
    }
    auto invalid = channel_options();
    if (comm->getRank() == 0) invalid.channels[0].name.clear();
    EXPECT_THROW(source.plan_remap(plan, target, channels, invalid), std::invalid_argument);
    invalid = channel_options();
    invalid.byte_budget = 1;
    EXPECT_THROW(source.plan_remap(plan, target, channels, invalid), std::invalid_argument);
    EXPECT_EQ(source.gas_volumes(), input.gas);
    EXPECT_EQ(source.carrier_volumes(), input.carrier);
}

TEST(GeneralVoFFinite, SourceOccupancyFollowsGeometricDonorSharesAndRecordsSignedFitRedistribution)
{
    auto input = fixture();
    constexpr double input_defect = 4 * std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? .9 : 0;
        input.carrier[i] = 1 - input.gas[i] + (center[0] < 1 ? input_defect : 0);
    }
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto trial = state.plan_transport(stationary(state), circulation(state), 1);
    const auto moved = state.apply(trial);
    int local_redistributed = 0;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto& before = state.initial_occupancy_errors()[i];
        const auto& after = moved->initial_occupancy_errors()[i];
        EXPECT_EQ(before.fit_allocation_bound, 0);
        EXPECT_LE(std::abs(after.difference), after.admitted_bound);
        const auto current_arithmetic = fp::bound_add(trial.phase_roundoff_bounds()[i].gas,
            trial.phase_roundoff_bounds()[i].carrier);
        // The carrier-only sweep from the left source removes the uniform
        // occupancy share from this route. That signed geometric redistribution
        // is a model contribution, not an enlarged arithmetic radius.
        if (input.gas[i] == 0 && after.fit_allocation_bound > current_arithmetic + input_defect / 256)
            ++local_redistributed;
    }
    int redistributed = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_redistributed, &redistributed);
    EXPECT_GT(redistributed, 0);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, SamePureGeometryKeepsIndependentTraceInventoryWithoutInventingAnInterface)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 1.);
    std::fill(input.carrier.begin(), input.carrier.end(), 1e-18);
    const std::vector<GeneralVoFPhaseGeometry> declarations(input.order.size(), GeneralVoFPhaseGeometry::PureGas);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, {}, declarations);
    const auto moved = state.apply(state.plan_transport(stationary(state), circulation(state), 1));
    const auto pure = moved->reconstruct();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        EXPECT_GT(moved->carrier_volumes()[i], 0);
        EXPECT_NEAR(moved->carrier_volumes()[i], 1e-18, moved->phase_roundoff_bounds()[i].carrier);
        EXPECT_TRUE(pure[i].finite_geometry);
        EXPECT_TRUE(pure[i].pure_gas);
        EXPECT_FALSE(pure[i].pure_carrier);
    }
}
TEST(GeneralVoFFinite, IncomingTinyGeometryInvalidatesPureHintAndIdentityPreservesSourceEvidence)
{
    auto input = fixture();
    constexpr double cut = 1 - 3. / 64 + 1e-6;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? cut : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto identity = state.plan_transport(stationary(state), circulation(state, true), 1);
    EXPECT_EQ(state.apply(identity)->accepted_identity(), state.accepted_identity());
    const auto moved = state.apply(state.plan_transport(stationary(state), circulation(state), 1));
    // Inspect accepted hints without fitting, resolution heuristics or exports.
    const auto declarations = moved->phase_geometry_declarations();
    ASSERT_EQ(declarations.size(), moved->owned_order().size());
    int local_tiny_receipts = 0;
    for (std::size_t i = 0; i < input.order.size(); ++i)
        if (input.gas[i] == 0 && moved->gas_volumes()[i] > 0)
        {
            ++local_tiny_receipts;
            EXPECT_LT(moved->gas_volumes()[i], 8 * std::numeric_limits<double>::epsilon());
            EXPECT_EQ(declarations[i], GeneralVoFPhaseGeometry::Infer);
        }
    int tiny_receipts = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_tiny_receipts, &tiny_receipts);
    EXPECT_GT(tiny_receipts, 0);
}
TEST(GeneralVoFFinite, TinyReceiptArithmeticRejectionPreservesAcceptedIndependentInventories)
{
    auto input = fixture();
    // A resolved planar interface lies just past the ordinary swept pyramid's
    // apex. The unit-scale donor is mostly gas, but its right recipient receives
    // only a tiny gas cap. Bulk face geometry keeps its ordinary accuracy gate.
    constexpr double cut = 1 - 3. / 64 + 1e-6;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? cut : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    GeneralVoFOptions bounded;
    bounded.arithmetic_mode = ArithmeticMode::BoundedCertification;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, bounded);
    const auto identity = state.accepted_identity();
    const auto control = state.accepted_control();
    const auto phases = state.phase_volumes();
    const auto history = state.accumulated_phase_errors();
    auto faces = circulation(state);
    // A bounded canonical control can satisfy its unit-scale occupancy budget
    // while leaving a much smaller phase receipt uncertified. The same radius
    // is shared by every owner incidence and is never hidden in phase history.
    for (auto& face : faces)
        if (face.integrated_volume != 0)
            face.bounded_integrated_volume = BoundedGeometryValue{face.integrated_volume, 1e-19L};
    std::string diagnostic;
    try
    {
        (void) state.plan_transport(stationary(state), faces, 1);
        FAIL() << "An uncertified tiny receipt must not borrow its donor's phase budget.";
    }
    catch (const std::invalid_argument& error)
    {
        diagnostic = error.what();
    }
    // Geometry may reject the uncertain tiny cut before recipient publication.
    // Both paths are arithmetic failures; neither requests spatial refinement
    // solely because the amount is small. The own-phase admission formula is
    // covered separately by the scalar ReceiptAdmissionUsesItsOwnPhaseAndExcludesFinalNarrowing regression.
    const int local_arithmetic_gate = diagnostic.find("recipient gas phase current-step") != std::string::npos ||
        diagnostic.find("support/topology sign unresolved") != std::string::npos;
    int observed_arithmetic_gate = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1,
        &local_arithmetic_gate, &observed_arithmetic_gate);
    EXPECT_EQ(observed_arithmetic_gate, 1) << diagnostic;
    EXPECT_EQ(state.accepted_identity(), identity);
    EXPECT_EQ(state.accepted_control(), control);
    ASSERT_EQ(state.phase_volumes().size(), phases.size());
    ASSERT_EQ(state.accumulated_phase_errors().size(), history.size());
    for (std::size_t i = 0; i < phases.size(); ++i)
    {
        EXPECT_EQ(state.phase_volumes()[i].gas, phases[i].gas);
        EXPECT_EQ(state.phase_volumes()[i].carrier, phases[i].carrier);
        EXPECT_EQ(state.accumulated_phase_errors()[i].gas, history[i].gas);
        EXPECT_EQ(state.accumulated_phase_errors()[i].carrier, history[i].carrier);
    }
}
} // namespace
