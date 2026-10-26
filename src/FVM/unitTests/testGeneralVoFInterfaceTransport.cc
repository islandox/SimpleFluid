#include "general_vof_finite_test_helpers.hh"
#include <algorithm>
#include <array>
#include <string>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Family = GeneralVoFTransportFamily;
Fixture planar_fixture(unsigned nx = 4, unsigned ny = 2, bool diagonal = false)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {nx, ny, 2};
    grid.extent = {double(nx), double(ny), 2};
    CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < nx * ny * 2; ++id)
        if (int(id % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(id);
    auto source = mesh.detach_owned(owned, {'i', 'n', 't', 'e', 'r', 'f', 'a', 'c', 'e'});
    Fixture input;
    input.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(source.owned), std::move(source.families), source.kernel, comm);
    for (const auto& [key, leaf] : input.hierarchy->owned())
    {
        const auto c = input.hierarchy->geometry(leaf)->centroid();
        const double t = diagonal ? 2.5 - int(c[0]) - int(c[1]) : 1.5 - int(c[0]);
        const double g = diagonal ? (t <= 0      ? 0
                                        : t >= 2 ? 1
                                        : t <= 1 ? t * t / 2
                                                 : 1 - (2 - t) * (2 - t) / 2)
                                  : std::clamp(t, 0., 1.);
        input.order.push_back(key);
        input.gas.push_back(g);
        input.carrier.push_back(1 - g);
    }
    int local_empty = input.order.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    return input;
}
std::vector<GeneralVoFFaceVolume> loops(const GeneralVoF& source, double q, bool competing = false)
{
    std::vector<GeneralVoFFaceVolume> result;
    for (const auto& tile : source.tiles())
    {
        const auto p = tile.geometry->centroid(), n = tile.geometry->area_vector();
        double value = 0;
        const auto loop = [&](double x, double y, double direction)
        {
            if (p.x == x && p.y > y - 1 && p.y < y + 1 && n.x != 0)
                value += direction * q * (p.y < y ? 1 : -1) * (n.x > 0 ? 1 : -1);
            if (p.y == y && p.x > x - 1 && p.x < x + 1 && n.y != 0)
                value += direction * q * (p.x > x ? 1 : -1) * (n.y > 0 ? 1 : -1);
        };
        if (!tile.exterior)
        {
            if (competing)
            {
                loop(1, 2, -1);
                loop(2, 1, 1);
            }
            else
                loop(1, 1, 1);
        }
        result.push_back({tile.face_gid, value});
    }
    return result;
}
void check_signed_budget(const GeneralVoFPreparedTransport& prepared)
{
    for (const auto& b : prepared.signed_donor_budgets)
        for (const auto& p : {std::array{b.old.bulk.value, b.positive_remote.bulk.value, b.negative_remote.bulk.value,
                                  b.diagonal.bulk.value},
                 std::array{
                     b.old.gas.value, b.positive_remote.gas.value, b.negative_remote.gas.value, b.diagonal.gas.value},
                 std::array{b.old.carrier.value, b.positive_remote.carrier.value, b.negative_remote.carrier.value,
                     b.diagonal.carrier.value}})
        {
            EXPECT_GE(p[1], 0);
            EXPECT_GE(p[2], 0);
            EXPECT_LE(p[1], p[0]);
            EXPECT_NEAR(p[3] + p[1] - p[2], p[0], 8e-15L * p[0]);
        }
}
TEST(GeneralVoFInterfaceTransport, PlanarCycleHasIndependentGeometricPhaseUpdate)
{
    const auto input = planar_fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    constexpr double q = 1. / 64;
    const auto trial = source.plan_transport(stationary(source), loops(source, q), 1);
    const auto next = source.apply(trial);
    const auto prepared = trial.prepare_transport({1, 1000}, 29);
    EXPECT_EQ(prepared->family, Family::InterfaceLocalSwept);
    EXPECT_EQ(prepared->backend, GeneralVoFTransportBackend::SweptPartition);
    EXPECT_TRUE(prepared->positive_transfers.empty());
    EXPECT_TRUE(prepared->donor_budgets.empty());
    EXPECT_TRUE(trial.finite_transfers().empty());
    EXPECT_EQ(prepared->signed_donor_budgets.size(), input.order.size());
    EXPECT_EQ(prepared->node_trajectories.size(), source.nodes().size());
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        // The left face of the half-filled column is wholly gas; its horizontal
        // face is half gas. These integrals are independent of the transport code.
        const double expected = int(x[0]) == 1 ? .5 + (x[1] < 1 ? q / 2 : -q / 2) : input.gas[c];
        EXPECT_NEAR(next->gas_volumes()[c], expected, 8e-14);
        EXPECT_NEAR(next->carrier_volumes()[c], 1 - expected, 8e-14);
    }
    check_signed_budget(*prepared);
    EXPECT_EQ(source.gas_volumes(), input.gas);
    EXPECT_NO_THROW(source.validate_prepared_transport_source(*prepared));
    auto forged = *prepared;
    forged.family = Family::AlgebraicFaceVolume;
    EXPECT_THROW(source.validate_prepared_transport_source(forged), std::invalid_argument);
}
TEST(GeneralVoFInterfaceTransport, PurePaddingKeepsGeometricWorkLocal)
{
    std::array<std::array<double, 7>, 2> global{};
    for (unsigned sample = 0; sample < 2; ++sample)
    {
        const auto input = planar_fixture(sample ? 8 : 4);
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
        const auto trial = source.plan_transport(stationary(source), loops(source, 1. / 64), 1);
        const auto d = trial.prepare_transport()->diagnostics;
        const std::array<double, 7> local{double(d.geometric_faces), double(d.swept_pieces), double(d.plane_requests),
            double(d.indexed_cells), double(d.index_queries), double(d.index_node_visits), double(d.donor_candidates)};
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 7, local.data(), global[sample].data());
        for (std::size_t i = 0; i < local.size(); ++i)
            RecordProperty("work_" + std::to_string(sample) + "_" + std::to_string(i), global[sample][i]);
    }
    EXPECT_GT(global[0][0], 0);
    EXPECT_GT(global[0][1], 0);
    EXPECT_GT(global[0][2], 0);
    EXPECT_EQ(global[1][0], global[0][0]);
    EXPECT_EQ(global[1][1], global[0][1]);
    EXPECT_EQ(global[1][2], global[0][2]);
    EXPECT_GT(global[1][3], global[0][3]);
    EXPECT_GT(global[0][4], 0);
    EXPECT_GT(global[0][5], 0);
    EXPECT_LT(global[1][5], 4 * global[0][5]);
    EXPECT_LT(global[1][6], 3 * global[0][6]);
}
TEST(GeneralVoFInterfaceTransport, ThreePlanarCyclesUseAggregateWitnessesAndPreserveIndependentInventories)
{
    const auto input = planar_fixture();
    auto source = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier);
    for (unsigned step = 0; step < 3; ++step)
    {
        SCOPED_TRACE(::testing::Message() << "step=" << step);
        const auto identity = source->accepted_identity();
        const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
        const auto trial = source->plan_transport(stationary(*source), loops(*source, 1. / 64), 1);
        const auto prepared = trial.prepare_transport({1, 1000}, 29 + step);
        const auto next = source->apply(trial);
        EXPECT_EQ(prepared->family, Family::InterfaceLocalSwept);
        EXPECT_EQ(prepared->backend, GeneralVoFTransportBackend::SweptPartition);
        EXPECT_TRUE(prepared->positive_transfers.empty());
        check_signed_budget(*prepared);
        std::array<double, 6> local{0, 0, double(prepared->diagnostics.aggregate_groups),
            double(prepared->diagnostics.aggregate_witnesses), double(prepared->diagnostics.cancelled_boundaries),
            double(prepared->diagnostics.aggregate_retries)}, global{};
        for (const auto& amount : next->phase_volumes())
        {
            EXPECT_GE(amount.gas, 0);
            EXPECT_GE(amount.carrier, 0);
            local[0] += amount.gas;
            local[1] += amount.carrier;
        }
        Teuchos::reduceAll(
            *Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
        // Four unit cells are gas and four are half gas. The remaining eight
        // are wholly carrier, independently giving six gas and ten carrier units.
        EXPECT_NEAR(global[0], 6, 2e-11 * 6);
        EXPECT_NEAR(global[1], 10, 2e-11 * 10);
        EXPECT_GT(global[2], 0);
        EXPECT_GT(global[3], 0);
        RecordProperty("aggregate_groups_" + std::to_string(step), global[2]);
        RecordProperty("aggregate_witnesses_" + std::to_string(step), global[3]);
        RecordProperty("cancelled_boundaries_" + std::to_string(step), global[4]);
        RecordProperty("aggregate_retries_" + std::to_string(step), global[5]);
        EXPECT_EQ(source->accepted_identity(), identity);
        EXPECT_EQ(source->gas_volumes(), gas);
        EXPECT_EQ(source->carrier_volumes(), carrier);
        source = next;
    }
}
TEST(GeneralVoFInterfaceTransport, FaceAlignedInterfaceWithoutMixedCellsUsesGeometry)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto trial = source.plan_transport(stationary(source), circulation(source), 1);
    const auto p = trial.prepare_transport();
    EXPECT_EQ(p->family, Family::InterfaceLocalSwept);
    double local[3]{double(p->diagnostics.geometric_faces), 0, 0}, global[3]{};
    for (const auto& phases : trial.candidate()->phase_volumes())
    {
        EXPECT_GE(phases.gas, 0);
        EXPECT_GE(phases.carrier, 0);
        local[1] += phases.gas;
        local[2] += phases.carrier;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_GT(global[0], 0);
    EXPECT_NEAR(global[1], 1, 1e-13);
    EXPECT_NEAR(global[2], 7, 1e-13);
    check_signed_budget(*p);
}
TEST(GeneralVoFInterfaceTransport, ZeroNetFaceRetainsOpposingPhaseStreamsAndCorrelatedHistory)
{
    auto input = fixture();
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        input.gas[c] = x[1] < 1 ? .5 : 0;
        input.carrier[c] = 1 - input.gas[c];
    }
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    constexpr double dt = 1. / 16;
    std::vector<GeneralVoFNodeVelocity> velocity;
    for (const auto& node : source.nodes())
        velocity.push_back({node.gid, {node.coordinate[0] == 1 ? node.coordinate[1] - 1 : 0, 0, 0}});
    const auto zero = circulation(source, true);
    const auto first = source.plan_transport(velocity, zero, dt);
    const auto next = first.candidate();
    const auto prepared = first.prepare_transport();
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        // Integrating the corrected departure fan gives dt/12 from its bottom
        // triangle and dt/48 from each side: a net gas transfer dt/8 at q=0.
        const double expected = x[1] < 1 ? .5 + (x[0] < 1 ? dt / 8 : -dt / 8) : 0;
        EXPECT_NEAR(next->gas_volumes()[c], expected, 3e-13);
        EXPECT_NEAR(next->carrier_volumes()[c], 1 - expected, 3e-13);
    }
    double local_counter = double(prepared->diagnostics.gross_counterstreams), counter = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_counter, &counter);
    EXPECT_GE(counter, 2);
    std::map<std::uint64_t, std::array<bool, 2>> signs;
    for (const auto& f : prepared->signed_face_contributions)
        signs[f.face][f.direction > 0 ? 1 : 0] = true;
    unsigned local_both = 0, both = 0;
    for (const auto& [face, sign] : signs)
        local_both += sign[0] && sign[1];
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_both, &both);
    EXPECT_GE(both, 2u);
    check_signed_budget(*prepared);

    const auto second = next->plan_transport(velocity, zero, dt).prepare_transport();
    std::array<double, 32> local{}, global{}; // Eight cells, G/L and their old histories.
    for (std::size_t c = 0; c < next->owned_order().size(); ++c)
    {
        const auto gid = std::size_t(next->mesh()->owned_cell_map()->getGlobalElement(c));
        ASSERT_LT(gid, 8u);
        local[4 * gid] = next->gas_volumes()[c];
        local[4 * gid + 1] = next->carrier_volumes()[c];
        local[4 * gid + 2] = next->accumulated_phase_errors()[c].gas;
        local[4 * gid + 3] = next->accumulated_phase_errors()[c].carrier;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 32, local.data(), global.data());
    for (const auto& receipt : second->signed_receipts)
        for (unsigned phase = 0; phase < 2; ++phase)
        {
            const auto old = global[4 * receipt.donor + phase];
            if (old == 0)
                continue;
            const auto coefficient = phase == 0 ? receipt.amount.gas.value : receipt.amount.carrier.value;
            const auto expected = std::abs(coefficient / old) * global[4 * receipt.donor + 2 + phase];
            const auto observed = phase == 0 ? receipt.inherited_error.gas : receipt.inherited_error.carrier;
            EXPECT_GE(observed, expected);
            EXPECT_NEAR(observed, expected, 8 * std::numeric_limits<double>::epsilon() * expected);
        }
}

TEST(GeneralVoFInterfaceTransport, CorrectedCenterBeyondStationaryNodesRejectsUncoveredReach)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto identity = source.accepted_identity();
    auto controls = circulation(source);
    for (auto& c : controls)
        c.integrated_volume *= .6 * 64;
    std::string cause;
    try
    {
        (void) source.plan_transport(stationary(source), controls, 1);
        ADD_FAILURE() << "A corrected center outside the admitted fluid shell must reject.";
    }
    catch (const std::invalid_argument& error)
    {
        cause = error.what();
    }
    EXPECT_NE(cause.find("authenticated retained shell"), std::string::npos) << cause;
    EXPECT_EQ(source.accepted_identity(), identity);
    EXPECT_EQ(source.gas_volumes(), input.gas);
    EXPECT_NO_THROW((void) source.plan_transport(stationary(source), circulation(source), 1));
}
TEST(GeneralVoFInterfaceTransport, RemotePureTraceRetainsInventoryAndCompensatedOccupancy)
{
    auto input = planar_fixture(8);
    std::vector<GeneralVoFPhaseGeometry> declarations;
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        if (x[0] >= 3)
            input.gas[c] = 1e-310;
        declarations.push_back(input.carrier[c] == 0              ? GeneralVoFPhaseGeometry::PureGas
                               : (input.gas[c] == 0 || x[0] >= 3) ? GeneralVoFPhaseGeometry::PureCarrier
                                                                  : GeneralVoFPhaseGeometry::Infer);
    }
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, {}, declarations);
    const auto next = source.plan_transport(stationary(source), loops(source, 1. / 64), 1).candidate();
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        if (x[0] >= 3)
        {
            EXPECT_EQ(next->gas_volumes()[c], 1e-310);
            EXPECT_EQ(next->carrier_volumes()[c], 1);
            EXPECT_EQ(next->initial_occupancy_errors()[c].difference, 1e-310);
            EXPECT_EQ(next->phase_geometry_declarations()[c], GeneralVoFPhaseGeometry::PureCarrier);
        }
    }
}

TEST(GeneralVoFInterfaceTransport, SharedCorrectionLimitsCompetingOldPhaseExports)
{
    const auto input = planar_fixture(4, 4, true);
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto trial = source.plan_transport(stationary(source), loops(source, .25, true), 1);
    const auto p = trial.prepare_transport();
    const auto next = trial.candidate();
    double limited = double(p->diagnostics.limited_faces), all_limited = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &limited, &all_limited);
    EXPECT_GT(all_limited, 0);
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        if (int(x[0]) == 1 && int(x[1]) == 1)
        {
            // The gas wedge x+y<2.5 has volume1/8. Two outgoing q=1/4
            // pyramids each contain13/200 gas, exceeding that old inventory.
            // Low order alone would leave1/16; a nontrivial bounded correction
            // must use geometry without exporting more than the original1/8.
            EXPECT_GE(next->gas_volumes()[c], 0);
            EXPECT_LT(next->gas_volumes()[c], 1. / 16 - 1e-6);
        }
    }
    check_signed_budget(*p);
}
} // namespace
