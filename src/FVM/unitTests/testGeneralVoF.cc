/** @file testGeneralVoF.cc */
#include "general_vof_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::vector<ExactPhaseMeasures> published_phase_lifts(const GeneralVoF& state)
{
    std::vector<ExactPhaseMeasures> result;
    for (const auto& value : state.phase_volumes())
        result.push_back({ExactGeometryMeasure::from_value(value.bulk), ExactGeometryMeasure::from_value(value.gas),
            ExactGeometryMeasure::from_value(value.carrier)});
    return result;
}

TEST(GeneralVoF, CanonicalNodeFanUsesAllIncidentOwnersBeforeFixedPhysicalSupport)
{
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier);
    ASSERT_TRUE(state.hierarchy()->native_partition());
    EXPECT_NE(state.hierarchy().get(), f.hierarchy.get());
    const auto support = state.support_cells();
    for (const auto& node : state.nodes())
    {
        const auto actual = state.node_cells(node.gid);
        std::vector<std::uint64_t> expected;
        for (const auto& cell : support)
        {
            const auto center = cell.geometry->centroid();
            unsigned distance = 0;
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                const auto p = node.coordinate[axis];
                if ((p == 0 && center[axis] > 1) || (p == 2 && center[axis] < 1))
                    ++distance;
            }
            if (distance <= 2)
                expected.push_back(cell.cell);
        }
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(actual, expected);
        if (node.coordinate == GeometryPoint{1, 1, 1})
            EXPECT_EQ(actual.size(), 8);
    }
}
TEST(GeneralVoF, GeometricFractionPreservesPhysicalInventoriesNativeConversionAndSubnormalPresence)
{
    const auto comm = Tpetra::getDefaultComm();
    auto make = [&](GeometryPoint extent)
    {
        AMRGrid grid;
        grid.block_cells = {2, 2, 2};
        grid.extent = extent;
        CartesianAMR source(grid);
        std::vector<std::uint64_t> owned;
        for (std::uint64_t gid = 0; gid < 8; ++gid)
            if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
                owned.push_back(gid);
        auto boot = source.detach_owned(owned, {'g', 'e', 'o', '_', 'f', 'r', 'a', 'c', 't', 'i', 'o', 'n'});
        return std::make_shared<const DistributedAdaptiveHierarchy>(
            std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    };
    const auto hierarchy = make({.7, .8, .9});
    std::vector<AdaptiveCellKey> order;
    std::vector<double> amount, zero;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        order.push_back(key);
        amount.push_back(std::nextafter(hierarchy->metrics(leaf).volume, std::numeric_limits<double>::infinity()));
        zero.push_back(0);
    }
    GeneralVoF gas(hierarchy, order, amount, zero);
    GeneralVoF carrier(hierarchy, order, zero, amount);
    const auto gas_fraction = gas.geometric_gas_fractions(), carrier_fraction = carrier.geometric_gas_fractions();
    const auto labels = gas.geometric_phases();
    int local_conversion = 0;
    for (std::size_t i = 0; i < order.size(); ++i)
    {
        EXPECT_EQ(gas_fraction[i], 1);
        EXPECT_EQ(carrier_fraction[i], 0);
        EXPECT_EQ(gas.gas_volumes()[i], amount[i]);
        EXPECT_EQ(carrier.carrier_volumes()[i], amount[i]);
        EXPECT_EQ(labels[i].gas.compare(labels[i].bulk), 0);
        EXPECT_GT(published_phase_lifts(gas)[i].gas.compare(labels[i].gas), 0);
        local_conversion += gas.initial_occupancy_errors()[i].native_conversion != 0;
    }
    int conversion = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_conversion, &conversion);
    EXPECT_GT(conversion, 0);
    auto invalid = amount;
    if (comm->getRank() == 0 && !invalid.empty())
        invalid.front() *= 1.01;
    EXPECT_THROW((GeneralVoF(hierarchy, order, invalid, zero)), std::invalid_argument);
    const auto large = make({4, 2, 2}); // Each actual cell volume is exactly2.
    order.clear();
    amount.clear();
    std::vector<double> bulk;
    for (const auto& [key, leaf] : large->owned())
    {
        order.push_back(key);
        amount.push_back(std::numeric_limits<double>::denorm_min());
        bulk.push_back(2);
    }
    GeneralVoF tiny(large, order, amount, bulk);
    const auto fraction = tiny.geometric_gas_fractions();
    const auto exact = tiny.geometric_phases();
    for (std::size_t i = 0; i < order.size(); ++i)
    {
        EXPECT_EQ(fraction[i], 0);
        EXPECT_EQ(tiny.gas_volumes()[i], std::numeric_limits<double>::denorm_min());
        EXPECT_EQ(exact[i].gas.compare(ExactGeometryMeasure::from_value(std::numeric_limits<double>::denorm_min())), 0);
        EXPECT_EQ(published_phase_lifts(tiny)[i].gas.compare(exact[i].gas), 0);
    }
}
TEST(GeneralVoF, InferredCoarseFineCirculationUsesActualFourSubfaceSeam)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    CartesianAMR source(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 16; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto boot = source.detach_owned(owned, {'c', 'o', 'a', 'r', 's', 'e', '_', 'f', 'i', 'n', 'e'});
    auto initial = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    std::vector<AdaptiveMark> marks;
    if (initial->owned().contains(source.cell_key(0)))
        marks.push_back({source.cell_key(0), AdaptiveSplit::Isotropic});
    const auto hierarchy = initial->apply(initial->plan_adaptation(marks));
    EXPECT_EQ(hierarchy->global_cells(), 72);
    std::vector<AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = hierarchy->geometry(leaf);
        const auto center = geometry->centroid();
        const double volume = hierarchy->metrics(leaf).volume;
        const double amount = center[1] < .5 ? volume : (center[1] == .5 ? .5 * volume : 0);
        order.push_back(key);
        gas.push_back(amount);
        carrier.push_back(volume - amount);
    }
    GeneralVoF state(hierarchy, order, gas, carrier);
    constexpr double phi = 1. / 1024;
    std::vector<GeneralVoFFaceVolume> controls;
    std::uint64_t local_seam = 0;
    for (const auto& face : state.tiles())
    {
        const auto p = face.geometry->centroid(), area = face.geometry->area_vector();
        double value = 0;
        if (!face.exterior && p.x == 2 && p.y == .25 && p.z == .25)
            value = area.x > 0 ? phi : -phi;
        if (!face.exterior && p.x == 2.5 && p.y == 1 && p.z == .5)
            value = area.y > 0 ? phi : -phi;
        if (!face.exterior && p.x == 2 && p.y == 1.25 && p.z == .25)
            value = area.x < 0 ? phi : -phi;
        if (!face.exterior && p.x == 1.75 && (p.y == 1 || p.y == .5) && p.z == .25)
            value = area.y < 0 ? phi : -phi;
        controls.push_back({face.face_gid, value});
        if (face.first_owner == comm->getRank() && p.x == 2 && p.y < 1 && p.z < 1)
            ++local_seam;
    }
    std::uint64_t seam = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_seam, &seam);
    EXPECT_EQ(seam, 4);
    const auto trial = state.plan_transport(nodes(state), controls, 1);
    const auto candidate = state.apply(trial);
    for (std::size_t i = 0; i < candidate->owned_order().size(); ++i)
    {
        const auto& key = candidate->owned_order()[i];
        const auto center = hierarchy->geometry(hierarchy->owned().at(key))->centroid();
        const auto source_index = std::find(order.begin(), order.end(), key) - order.begin();
        double expected = gas.at(source_index);
        if (center == GeometryPoint{1.75, .25, .25})
            expected -= phi;
        if (center == GeometryPoint{2.5, .5, .5})
            expected += phi;
        EXPECT_EQ(candidate->gas_volumes()[i], expected);
        EXPECT_EQ(candidate->carrier_volumes()[i], hierarchy->metrics(hierarchy->owned().at(key)).volume - expected);
    }
    if (comm->getRank() >= 2)
        EXPECT_TRUE(candidate->owned_order().empty());
}
TEST(GeneralVoF, InferredObliquePLICDrivesActualSkewPolyhedralUnsplitTransport)
{
    auto fixture = skew_setup();
    GeneralVoF source(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier);
    const auto reconstruction = source.reconstruct();
    const GeometryPoint normal{-1, 0, 1};
    const double norm = std::hypot(normal[0], normal[1], normal[2]);
    for (std::size_t i = 0; i < reconstruction.size(); ++i)
        if (fixture.gas[i] > 0)
            for (unsigned axis = 0; axis < 3; ++axis)
                EXPECT_NEAR(reconstruction[i].gas.normal[axis], normal[axis] / norm, 2e-7);
    std::vector<GeneralVoFFaceVolume> flux;
    for (const auto& tile : source.tiles())
    {
        const auto p = tile.geometry->centroid(), area = tile.geometry->area_vector();
        const double u = .5 * (p.x + p.z) - .25 * p.y, v = p.y;
        double amount = 0;
        if (!tile.exterior && u == 1 && area.x != 0)
            amount = (v < 1 ? 1 : -1) * (area.x > 0 ? 1 : -1) / 64.;
        if (!tile.exterior && v == 1 && area.x == 0 && area.z == 0)
            amount = (u > 1 ? 1 : -1) * (area.y > 0 ? 1 : -1) / 64.;
        flux.push_back({tile.face_gid, amount});
    }
    auto trial = source.plan_transport(nodes(source), flux, 1);
    auto result = source.apply(trial);
    // Reflection of each actual swept pyramid in reference w=1/2 splits
    // transported bottom-layer volume exactly in half. No plane is prescribed
    // to the production reconstruction or transport call.
    for (std::size_t i = 0; i < source.owned_order().size(); ++i)
    {
        EXPECT_NEAR(result->gas_volumes()[i], source.gas_volumes()[i], 3e-13);
        EXPECT_NEAR(result->carrier_volumes()[i], source.carrier_volumes()[i], 3e-13);
    }
    for (const auto& stream : trial.finite_face_contributions())
    {
        EXPECT_NEAR(stream.amount.gas.value + stream.amount.carrier.value, stream.amount.bulk.value, stream.amount.gas.absolute_error + stream.amount.carrier.absolute_error + stream.amount.bulk.absolute_error);
        if (stream.amount.gas.value > 0)
            EXPECT_NEAR(
                stream.amount.gas.as_double(), .5 * stream.amount.bulk.as_double(), 3e-13);
    }
}
TEST(GeneralVoF, ThreeAxisCornerTransferUsesIncidentFaceDonorUnion)
{
    auto fixture = setup();
    GeneralVoF state(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier);
    auto velocity = nodes(state);
    for (std::size_t i = 0; i < state.nodes().size(); ++i)
        if (state.nodes()[i].coordinate == GeometryPoint{1, 1, 1})
            velocity[i].velocity = {1. / 64, 1. / 64, 1. / 64};
    std::vector<GeneralVoFFaceVolume> zero;
    for (const auto& face : state.tiles())
        zero.push_back({face.face_gid, 0});
    // Backtrace the shared center into the opposite octant. Its original
    // donor is three face hops from the upper/right/front recipient, despite
    // arbitrarily small CFL. A target-only two-ring donor set is incomplete.
    const auto trial = state.plan_transport(velocity, zero, 1);
    const auto candidate = state.apply(trial);
    const auto support = state.support_cells();
    int local_corner = 0, global_corner = 0;
    for (const auto& transfer : trial.finite_transfers())
    {
        const auto donor =
            std::find_if(support.begin(), support.end(), [&](const auto& cell) { return cell.cell == transfer.donor; });
        const auto recipient = std::find_if(
            support.begin(), support.end(), [&](const auto& cell) { return cell.cell == transfer.recipient; });
        if (donor != support.end() && recipient != support.end() &&
            donor->geometry->centroid() == GeometryPoint{.5, .5, .5} &&
            recipient->geometry->centroid() == GeometryPoint{1.5, 1.5, 1.5})
        {
            EXPECT_GT(transfer.amount.bulk.value, transfer.amount.bulk.absolute_error);
            EXPECT_NEAR(transfer.amount.gas.value, transfer.amount.bulk.value, transfer.amount.gas.absolute_error + transfer.amount.bulk.absolute_error);
            ++local_corner;
        }
    }
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_corner, &global_corner);
    EXPECT_EQ(global_corner, 1);
    for (const auto& budget : trial.finite_donor_budgets())
    {
        EXPECT_NEAR(budget.exported.gas.value + budget.retained.gas.value, budget.old.gas.value, budget.exported.gas.absolute_error + budget.retained.gas.absolute_error);
        EXPECT_NEAR(budget.exported.carrier.value + budget.retained.carrier.value, budget.old.carrier.value, budget.exported.carrier.absolute_error + budget.retained.carrier.absolute_error);
    }
    if (comm->getRank() >= 2)
        EXPECT_TRUE(candidate->owned_order().empty());
}
TEST(GeneralVoF, HighValenceNativeVertexFansIncludeDonorsBeyondBothFaceNeighborhoods)
{
    const auto comm = Tpetra::getDefaultComm();
    ExtrudedAMRGrid grid;
    grid.xy_nodes.push_back({0, 0, 0});
    const std::array<GeometryPoint, 8> rays{
        {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {-1, 1, 0}, {-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}}};
    for (const auto& point : rays)
        grid.xy_nodes.push_back({point[0], point[1], 0});
    for (std::size_t sector = 0; sector < rays.size(); ++sector)
    {
        const auto a = sector + 1, b = (sector + 1) % rays.size() + 1;
        grid.base_cells.push_back({0, a, b});
        grid.boundaries.push_back({a, b, "wall"});
    }
    grid.z_edges = {0, 1, 2};
    ExtrudedAMR source(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 16; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto boot = source.detach_owned(owned, {'h', 'i', 'g', 'h', '_', 'v', 'a', 'l', 'e', 'n', 'c', 'e'});
    const auto hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    std::vector<AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        order.push_back(key);
        gas.push_back(0);
        carrier.push_back(hierarchy->metrics(leaf).volume);
    }
    GeneralVoF state(hierarchy, order, gas, carrier);
    auto velocity = nodes(state);
    for (std::size_t i = 0; i < state.nodes().size(); ++i)
        if (state.nodes()[i].coordinate == GeometryPoint{0, 0, 1})
            velocity[i].velocity = {1. / 64, 0, 0};
    std::vector<GeneralVoFFaceVolume> zero;
    for (const auto& face : state.tiles())
        zero.push_back({face.face_gid, 0});
    const auto trial = state.plan_transport(velocity, zero, 1);
    const auto candidate = state.apply(trial);
    const auto support = state.support_cells();
    int local_distant = 0, distant = 0;
    for (const auto& transfer : trial.finite_transfers())
    {
        if (transfer.donor == transfer.recipient || transfer.amount.bulk.value <= 0)
            continue;
        const auto donor =
            std::find_if(support.begin(), support.end(), [&](const auto& cell) { return cell.cell == transfer.donor; });
        const auto recipient = std::find_if(
            support.begin(), support.end(), [&](const auto& cell) { return cell.cell == transfer.recipient; });
        if (donor == support.end() || recipient == support.end())
            continue;
        const auto a = donor->geometry->centroid(), b = recipient->geometry->centroid();
        // Opposite sectors touch only at the shared vertical native edge;
        // their angular face-graph distance is at least three, independently
        // of the number of radial/Z halo rings one might assume for boxes.
        if (a[0] < -.5 && b[0] > .5)
            ++local_distant;
        EXPECT_EQ(transfer.amount.gas.value, 0);
        EXPECT_NEAR(transfer.amount.carrier.value, transfer.amount.bulk.value, transfer.amount.carrier.absolute_error + transfer.amount.bulk.absolute_error);
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_distant, &distant);
    EXPECT_GT(distant, 0);
    for (const auto& budget : trial.finite_donor_budgets())
        EXPECT_NEAR(budget.exported.carrier.value + budget.retained.carrier.value, budget.old.carrier.value, budget.exported.carrier.absolute_error + budget.retained.carrier.absolute_error);
    if (comm->getRank() >= 2)
        EXPECT_TRUE(candidate->owned_order().empty());
}
TEST(GeneralVoF, CrossedFaceKeepsOppositeDonorStreamsWhenProjectedNetVolumeIsZero)
{
    auto fixture = setup();
    for (std::size_t i = 0; i < fixture.order.size(); ++i)
    {
        auto p = fixture.hierarchy->geometry(fixture.hierarchy->owned().at(fixture.order[i]))->centroid();
        fixture.gas[i] = p[1] < 1 ? .5 : 0;
        fixture.carrier[i] = 1 - fixture.gas[i];
    }
    GeneralVoF source(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier);
    std::vector<GeneralVoFNodeVelocity> velocity;
    constexpr double dt = 1. / 16;
    for (const auto& node : source.nodes())
        velocity.push_back({node.gid, {node.coordinate[0] == 1 ? node.coordinate[1] - 1 : 0, 0, 0}});
    std::vector<GeneralVoFFaceVolume> phi;
    for (const auto& tile : source.tiles())
        phi.push_back({tile.face_gid, 0});
    auto trial = source.plan_transport(velocity, phi, dt);
    auto result = source.apply(trial);
    for (std::size_t i = 0; i < source.owned_order().size(); ++i)
    {
        auto p = fixture.hierarchy->geometry(fixture.hierarchy->owned().at(source.owned_order()[i]))->centroid();
        const double expected = p[1] < 1 ? .5 + (p[0] < 1 ? dt / 8 : -dt / 8) : 0;
        EXPECT_NEAR(result->gas_volumes()[i], expected, 3e-13);
        EXPECT_NEAR(result->carrier_volumes()[i], 1 - expected, 3e-13);
    }
    unsigned local_faces = 0;
    auto comm = Tpetra::getDefaultComm();
    for (const auto& tile : source.tiles())
        if (tile.first_owner == comm->getRank())
        {
            const auto p = tile.geometry->centroid();
            if (p.x != 1 || p.y != .5)
                continue;
            SimpleFluid::VoF::finite_detail::Sum bulk;
            double gas = 0, carrier = 0;
            bool positive = false, negative = false;
            for (const auto& stream : trial.finite_face_contributions())
                if (stream.face == tile.face_gid)
                {
                    bulk.add(stream.direction > 0 ? stream.amount.bulk : SimpleFluid::VoF::finite_detail::negate(stream.amount.bulk));
                    gas += stream.direction * stream.amount.gas.as_double();
                    carrier += stream.direction * stream.amount.carrier.as_double();
                    positive |= stream.direction > 0;
                    negative |= stream.direction < 0;
                }
            EXPECT_LE(std::abs(bulk.value().value), bulk.value().absolute_error);
            EXPECT_TRUE(positive);
            EXPECT_TRUE(negative);
            // Independent integration of the corrected departure fan: bottom gas
            // triangle contributes dt/12 and its two side pieces dt/48 each.
            const int orientation = tile.geometry->area_vector().x > 0 ? 1 : -1;
            EXPECT_NEAR(gas, -orientation * dt / 8, 3e-13);
            EXPECT_NEAR(carrier, orientation * dt / 8, 3e-13);
            ++local_faces;
        }
    unsigned faces = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_faces, &faces);
    EXPECT_EQ(faces, 2u);
}
TEST(GeneralVoF, DeclaredSubnormalInventorySurvivesWithoutInventingAnIdentifiableNormal)
{
    auto f = setup(true);
    std::vector<GeneralVoFPhaseGeometry> declaration(f.order.size(), GeneralVoFPhaseGeometry::PureCarrier);
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier, {}, declaration);
    std::vector<GeneralVoFFaceVolume> zero;
    for (const auto& face : state.tiles())
        zero.push_back({face.face_gid, 0});
    auto trial = state.plan_transport(nodes(state), zero, 1);
    auto unchanged = state.apply(trial);
    for (auto gas : unchanged->gas_volumes())
        EXPECT_EQ(gas, 1e-310);
    for (auto carrier : unchanged->carrier_volumes())
        EXPECT_EQ(carrier, 1);
    ASSERT_EQ(state.initial_occupancy_errors().size(), state.owned_order().size());
    for (std::size_t i = 0; i < state.owned_order().size(); ++i)
    {
        const auto& error = state.initial_occupancy_errors()[i];
        EXPECT_EQ(error.difference, 1e-310);
        EXPECT_LE(error.difference, error.admitted_bound);
        EXPECT_EQ(unchanged->initial_occupancy_errors()[i].difference, error.difference);
        EXPECT_EQ(published_phase_lifts(*unchanged)[i].gas.compare(published_phase_lifts(state)[i].gas), 0);
        EXPECT_EQ(published_phase_lifts(*unchanged)[i].carrier.compare(published_phase_lifts(state)[i].carrier), 0);
    }
    for (const auto& p : state.reconstruct())
    {
        EXPECT_TRUE(p.pure_carrier);
        EXPECT_TRUE(p.finite_geometry);
        EXPECT_LT(p.finite_target_volume_difference.value, 0);
    }
    const auto transported = state.plan_transport(nodes(state), cycle(state), 1);
    const auto moved = state.apply(transported);
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        EXPECT_EQ(moved->gas_volumes()[i], 1e-310);
        EXPECT_EQ(moved->carrier_volumes()[i], 1);
        EXPECT_EQ(published_phase_lifts(*moved)[i].gas.compare(ExactGeometryMeasure::from_value(1e-310)), 0);
        EXPECT_EQ(moved->initial_occupancy_errors()[i].difference, 1e-310);
    }
    ASSERT_EQ(transported.finite_donor_budgets().size(), state.owned_order().size());
    for (const auto& budget : transported.finite_donor_budgets())
    {
        const auto gas = SimpleFluid::VoF::finite_detail::add(budget.exported.gas, budget.retained.gas);
        EXPECT_LE(std::abs(gas.value - budget.old.gas.value), gas.absolute_error);
    }
    GeneralVoF unresolved(f.hierarchy, f.order, f.gas, f.carrier);
    EXPECT_THROW(unresolved.reconstruct(), std::invalid_argument);
    EXPECT_NO_THROW(unresolved.plan_transport(nodes(unresolved), zero, 1));
}
TEST(GeneralVoF, MaterialProjectedDivergenceAndRankLocalMalformedControlsRejectPrivateTrial)
{
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier);
    auto controls = cycle(state);
    for (std::size_t i = 0; i < state.tiles().size(); ++i)
    {
        const auto p = state.tiles()[i].geometry->centroid();
        if (p.x == 1 && p.y == .5 && p.z == .5)
            controls[i].integrated_volume += 1. / 64;
    }
    EXPECT_THROW(state.plan_transport(nodes(state), controls, 1), std::invalid_argument);
    auto velocities = nodes(state);
    if (Tpetra::getDefaultComm()->getRank() == 0 && !velocities.empty())
        velocities.pop_back();
    EXPECT_THROW(state.plan_transport(velocities, cycle(state), 1), std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), f.gas);
    EXPECT_EQ(state.carrier_volumes(), f.carrier);
    auto excessive = cycle(state);
    for (auto& face : excessive)
        face.integrated_volume *= 256;
    EXPECT_THROW(state.plan_transport(nodes(state), excessive, 1), std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), f.gas);
    EXPECT_EQ(state.carrier_volumes(), f.carrier);
    GeneralVoFOptions budget;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        budget.byte_budget = 1;
    EXPECT_THROW((GeneralVoF(f.hierarchy, f.order, f.gas, f.carrier, budget)), std::invalid_argument);
    EXPECT_NO_THROW(state.plan_transport(nodes(state), cycle(state), 1));
}

} // namespace

namespace
{
TEST(GeneralVoF, FiniteInventoriesPublishOnceAndSubsequentTrialsLiftOnlyPublishedValues)
{
    static_assert(sizeof(PhaseVolumes) == 3 * sizeof(double));
    static_assert(sizeof(PhaseError) == 8 * sizeof(double));
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier);
    auto control = cycle(state);
    for (auto& face : control)
        face.integrated_volume = face.integrated_volume > 0 ? .0125 : face.integrated_volume < 0 ? -.0125 : 0;
    const auto trial = state.plan_transport(nodes(state), control, 1);
    const auto moved = state.apply(trial);
    int local_rounded = 0;
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto& phase = moved->phase_volumes()[i];
        const auto& rounding = trial.phase_roundoff_bounds()[i];
        EXPECT_EQ(phase.gas, moved->gas_volumes()[i]);
        EXPECT_EQ(phase.carrier, moved->carrier_volumes()[i]);
        EXPECT_GE(phase.gas, 0);
        EXPECT_GE(phase.carrier, 0);
        EXPECT_TRUE(std::isfinite(phase.gas));
        EXPECT_TRUE(std::isfinite(phase.carrier));
        local_rounded += rounding.gas > 0 || rounding.carrier > 0;
    }
    int rounded = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_rounded, &rounded);
    EXPECT_GT(rounded, 0);
    std::vector<GeneralVoFFaceVolume> stationary;
    for (const auto& face : moved->tiles())
        stationary.push_back({face.face_gid, 0});
    const auto next = moved->plan_transport(nodes(*moved), stationary, 1);
    ASSERT_EQ(next.finite_donor_budgets().size(), moved->phase_volumes().size());
    for (std::size_t i = 0; i < moved->phase_volumes().size(); ++i)
    {
        const auto& phase = moved->phase_volumes()[i];
        const auto& old = next.finite_donor_budgets()[i].old;
        EXPECT_EQ(old.gas.value, phase.gas);
        EXPECT_EQ(old.gas.absolute_error, 0);
        EXPECT_EQ(old.carrier.value, phase.carrier);
        EXPECT_EQ(old.carrier.absolute_error, 0);
        EXPECT_EQ(next.phase_roundoff_bounds()[i].gas, 0);
        EXPECT_EQ(next.phase_roundoff_bounds()[i].carrier, 0);
    }
    EXPECT_EQ(state.gas_volumes(), f.gas); // accepted source was never changed
}
} // namespace

namespace
{
TEST(GeneralVoF, NonDyadicPublicationContinuesWithDerivedErrorInsteadOfFreshInputAllowance)
{
    auto f = setup();
    const double input_defect = 8 * std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < f.gas.size(); ++i)
        if (f.gas[i] == 1)
            f.carrier[i] = input_defect;
    std::vector<GeneralVoFPhaseGeometry> declarations;
    for (double gas : f.gas)
        declarations.push_back(gas == 1 ? GeneralVoFPhaseGeometry::PureGas : GeneralVoFPhaseGeometry::PureCarrier);
    GeneralVoF source(f.hierarchy, f.order, f.gas, f.carrier, {}, declarations);
    auto controls = cycle(source);
    for (auto& face : controls)
        face.integrated_volume = face.integrated_volume > 0 ? .0125 : face.integrated_volume < 0 ? -.0125 : 0;
    const auto trial = source.plan_transport(nodes(source), controls, 1);
    const auto moved = source.apply(trial);
    int local_beyond_fresh = 0;
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto& error = moved->initial_occupancy_errors()[i];
        const auto& phase = moved->phase_volumes()[i];
        if (std::abs(error.difference) > input_defect * phase.bulk)
        {
            ++local_beyond_fresh;
            EXPECT_LE(std::abs(error.difference), error.admitted_bound);
            EXPECT_GT(trial.phase_roundoff_bounds()[i].gas + trial.phase_roundoff_bounds()[i].carrier, 0);
        }
    }
    int beyond_fresh = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_beyond_fresh, &beyond_fresh);
    EXPECT_GT(beyond_fresh, 0);
    // The publication is admitted by its derived arithmetic provenance. It
    // cannot be reintroduced as an unproved fresh input under the initial8eps.
    EXPECT_THROW((GeneralVoF(f.hierarchy, moved->owned_order(), moved->gas_volumes(), moved->carrier_volumes())),
        std::invalid_argument);
    std::vector<GeneralVoFFaceVolume> stationary;
    for (const auto& tile : moved->tiles())
        stationary.push_back({tile.face_gid, 0});
    const auto continued = moved->apply(moved->plan_transport(nodes(*moved), stationary, 1));
    EXPECT_EQ(continued->gas_volumes(), moved->gas_volumes());
    EXPECT_EQ(continued->carrier_volumes(), moved->carrier_volumes());
}
} // namespace
