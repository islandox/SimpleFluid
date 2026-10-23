/** @file testGeneralVoFFinite.cc */
#include "general_vof_finite_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
TEST(GeneralVoFFinite, BoundedTransportKeepsIndependentPhasesCanonicalRoutesAndEmptyOwners)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
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
    EXPECT_TRUE(trial.transfers().empty());
    EXPECT_TRUE(trial.donor_budgets().empty());
    ASSERT_EQ(trial.finite_donor_budgets().size(), input.order.size());
    ASSERT_EQ(candidate->accumulated_phase_errors().size(), input.order.size());
    fp::Sum gas, carrier;
    for (std::size_t i = 0; i < candidate->phase_volumes().size(); ++i)
    {
        const auto& p = candidate->phase_volumes()[i];
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        gas.add({p.gas, trial.phase_roundoff_bounds()[i].gas});
        carrier.add({p.carrier, trial.phase_roundoff_bounds()[i].carrier});
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
    EXPECT_GT(statistics.constructions + statistics.volume_evaluations, 0);
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
        faces.front().exact_integrated_volume = ExactGeometryMeasure::from_value(0);
    EXPECT_THROW(state.plan_transport(stationary(state), faces, 1), std::invalid_argument);
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





} // namespace
