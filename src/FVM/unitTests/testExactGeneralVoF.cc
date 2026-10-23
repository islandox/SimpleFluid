/** @file testExactGeneralVoF.cc */
#include "general_vof_test_helpers.hh"
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/ExactPhaseAccumulator.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "FVM/details/PhaseArithmetic.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::map<std::uint64_t, ExactPhaseMeasures> delivered_phases(const GeneralVoF::Trial& trial)
{
    std::map<std::uint64_t, ExactPhaseMeasures> result;
    for (const auto& route : trial.transfers())
    {
        auto& total = result[route.recipient];
        total.bulk = total.bulk + route.exact.bulk;
        total.gas = total.gas + route.exact.gas;
        total.carrier = total.carrier + route.exact.carrier;
    }
    return result;
}

GeneralVoFOptions exact_reference_options()
{
    GeneralVoFOptions options;
    options.inventory_policy = PhaseInventoryPolicy::ExperimentalExact;
    return options;
}
TEST(ExactGeneralVoF, ComplementaryDonorLabelCorrectionIsLinearBoundedAndRejectsUnprovedDefect)
{
    using M = ExactGeometryMeasure;
    const auto one = M::from_value(1), quarter = M::from_value(.25), tail = M::from_value(1e-310);
    const ExactPhaseMeasures full{one, quarter, M::from_value(.75)};
    const auto map = SimpleFluid::VoF::detail::ExactPhaseLabelMap::create(full, true, quarter + tail, tail, 1 << 20);
    const ExactPhaseMeasures a{M::from_value(.25), M::from_value(.125), M::from_value(.125)};
    const ExactPhaseMeasures b{M::from_value(.75), M::from_value(.125), M::from_value(.625)};
    const auto first = map.apply(a), second = map.apply(b), total = map.apply(full);
    const auto bounded = map.apply(a, 1 << 20);
    EXPECT_EQ(bounded.bulk.compare(first.bulk), 0);
    EXPECT_EQ(bounded.gas.compare(first.gas), 0);
    EXPECT_EQ(bounded.carrier.compare(first.carrier), 0);
    EXPECT_THROW(map.apply(a, 0), std::invalid_argument);
    EXPECT_THROW(map.apply(a, (1 << 20) + 1), std::invalid_argument);
    EXPECT_THROW(map.apply(a, a.bulk.storage_bytes() + a.gas.storage_bytes() + a.carrier.storage_bytes()),
        std::invalid_argument);
    EXPECT_EQ(map.apply(a).gas.compare(first.gas), 0); // Rejection never mutates the donor map.
    EXPECT_EQ((first.gas + second.gas).compare(quarter + tail), 0);
    EXPECT_EQ((first.carrier + second.carrier).compare(total.carrier), 0);
    EXPECT_EQ((first.gas + first.carrier).compare(a.bulk), 0);
    EXPECT_EQ((second.gas + second.carrier).compare(b.bulk), 0);
    EXPECT_GE(first.gas.sign(), 0);
    EXPECT_GE(second.carrier.sign(), 0);
    const auto reversed = map.apply({a.bulk.scaled(-1), a.gas.scaled(-1), a.carrier.scaled(-1)});
    EXPECT_EQ((first.gas + reversed.gas).sign(), 0);
    const auto bounded_reversed = map.apply({a.bulk.scaled(-1), a.gas.scaled(-1), a.carrier.scaled(-1)}, 1 << 20);
    EXPECT_EQ(bounded_reversed.gas.compare(reversed.gas), 0);
    EXPECT_EQ(bounded_reversed.carrier.compare(reversed.carrier), 0);
    EXPECT_THROW(SimpleFluid::VoF::detail::ExactPhaseLabelMap::create(full, true, quarter + tail, {}, 1 << 20),
        std::invalid_argument);
    EXPECT_THROW(SimpleFluid::VoF::detail::ExactPhaseLabelMap::create(full, true, one + tail, one, 1 << 20),
        std::invalid_argument);
}

TEST(ExactGeneralVoF, ActualUnsplitCirculationHasPositiveRoutesIndependentComplementsAndEmptyOwners)
{
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier, exact_reference_options());
    auto controls = cycle(state);
    auto trial = state.plan_transport(nodes(state), controls, 1);
    auto moved = state.apply(trial);
    // A nonidentity transport publication owns new phase arrays while retaining
    // the same immutable native topology and canonical incidence storage.
    EXPECT_EQ(&moved->tiles(), &state.tiles());
    EXPECT_EQ(&moved->nodes(), &state.nodes());
    EXPECT_EQ(&moved->owned_order(), &state.owned_order());
    EXPECT_NE(&moved->gas_volumes(), &state.gas_volumes());
    for (const auto& node : state.nodes())
        EXPECT_EQ(&moved->node_tiles(node.gid), &state.node_tiles(node.gid));
    double gas = 0, carrier = 0;
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto center = f.hierarchy->geometry(f.hierarchy->owned().at(moved->owned_order()[i]))->centroid();
        const double expected = center[2] < 1 && center[1] < 1 ? (center[0] < 1 ? 1 - 1. / 64 : 1. / 64) : 0;
        EXPECT_EQ(moved->gas_volumes()[i], expected);
        EXPECT_EQ(moved->carrier_volumes()[i], 1 - expected);
        gas += moved->gas_volumes()[i];
        carrier += moved->carrier_volumes()[i];
    }
    auto comm = Tpetra::getDefaultComm();
    double global_gas = 0, global_carrier = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &gas, &global_gas);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &carrier, &global_carrier);
    EXPECT_EQ(global_gas, 1);
    EXPECT_EQ(global_carrier, 7);
    for (const auto& transfer : trial.transfers())
    {
        EXPECT_EQ(transfer.recipient_owner, comm->getRank());
        EXPECT_GE(transfer.exact.gas.sign(), 0);
        EXPECT_GE(transfer.exact.carrier.sign(), 0);
        EXPECT_EQ((transfer.exact.gas + transfer.exact.carrier).compare(transfer.exact.bulk), 0);
    }
    // Delivered exact self/remote phase records must agree independently with
    // the accepted target inventories; a local reference cannot alias a later
    // mutable column accumulator or swap gas/carrier during packet decoding.
    std::map<std::uint64_t, Meshes::ExactPhaseMeasures> delivered;
    for (const auto& route : trial.transfers())
    {
        auto& total = delivered[route.recipient];
        total.bulk = total.bulk + route.exact.bulk;
        total.gas = total.gas + route.exact.gas;
        total.carrier = total.carrier + route.exact.carrier;
    }
    const auto support = state.support_cells();
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto key = moved->owned_order()[i];
        const auto cell =
            std::find_if(support.begin(), support.end(), [&](const auto& item) { return item.key == key; });
        ASSERT_NE(cell, support.end());
        const auto& total = delivered.at(cell->cell);
        const auto& value = moved->phase_volumes()[i];
        const auto& error = moved->phase_roundoff_bounds()[i];
        const auto gas_error = total.gas - ExactGeometryMeasure::from_value(value.gas);
        const auto carrier_error = total.carrier - ExactGeometryMeasure::from_value(value.carrier);
        EXPECT_LE((gas_error.sign() < 0 ? gas_error.scaled(-1) : gas_error)
                      .compare(ExactGeometryMeasure::from_value(error.gas)),
            0);
        EXPECT_LE((carrier_error.sign() < 0 ? carrier_error.scaled(-1) : carrier_error)
                      .compare(ExactGeometryMeasure::from_value(error.carrier)),
            0);
    }
    int self_routes = 0, remote_routes = 0;
    for (const auto& route : trial.transfers())
        if (route.exact.bulk.sign() > 0)
            (route.donor_owner == route.recipient_owner ? self_routes : remote_routes)++;
    int global_self = 0, global_remote = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &self_routes, &global_self);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &remote_routes, &global_remote);
    EXPECT_GT(global_self, 0); // Retained atoms exercise same-rank proof reuse.
    if (comm->getSize() > 1)
        EXPECT_GT(global_remote, 0); // Actual cross-owner flow retains strict decode.
    for (const auto& stream : trial.face_contributions())
    {
        EXPECT_EQ((stream.exact.gas + stream.exact.carrier).compare(stream.exact.bulk), 0);
        EXPECT_EQ((stream.exact.gas + stream.exact.carrier).compare(stream.exact.bulk), 0);
    }
    for (const auto& budget : trial.donor_budgets())
    {
        EXPECT_EQ((budget.exact_exported.gas + budget.exact_retained.gas).compare(budget.exact_old.gas), 0);
        EXPECT_EQ((budget.exact_exported.carrier + budget.exact_retained.carrier).compare(budget.exact_old.carrier), 0);
    }
    int local_empty = state.owned_order().empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    GeneralVoF another(f.hierarchy, f.order, f.gas, f.carrier);
    EXPECT_THROW(another.apply(trial), std::invalid_argument);
    if (comm->getSize() > 1)
    {
        std::vector<GeneralVoFFaceVolume> zero;
        for (const auto& face : state.tiles())
            zero.push_back({face.face_gid, 0});
        auto other_trial = state.plan_transport(nodes(state), zero, 1);
        EXPECT_THROW(state.apply(comm->getRank() % 2 ? trial : other_trial), std::invalid_argument);
        const auto& hybrid = comm->getRank() % 2 ? *moved : state;
        EXPECT_THROW(hybrid.geometric_phases(), std::invalid_argument);
        EXPECT_THROW(hybrid.geometric_gas_fractions(), std::invalid_argument);
        EXPECT_THROW(hybrid.plan_transport(nodes(hybrid), zero, 1), std::invalid_argument);
    }
    EXPECT_EQ(state.gas_volumes(), f.gas);
}

TEST(ExactGeneralVoF, ExactSkewWallAndEdgeTrajectoriesAreOwnershipIndependentAndDoNotLeak)
{
    const auto comm = Tpetra::getDefaultComm();
    for (unsigned shift = 0; shift < 2; ++shift)
    {
        auto fixture = skew_setup(shift);
        GeneralVoF state(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier, exact_reference_options());
        auto velocity = nodes(state);
        for (std::size_t i = 0; i < state.nodes().size(); ++i)
        {
            const auto p = state.nodes()[i].coordinate;
            const double u = (p[0] + p[2] - .5 * p[1]) / 2, w = (p[2] - p[0]) / 2;
            const double a = p[1] == 1 ? (u - 1) / 64 : 0;
            // The extra component is normal to the dyadic skew w wall.
            // At u/w edges it is orthogonal to the second wall's normal.
            const double b = p[1] == 1 && (w == 0 || w == 2) ? 1. / 128 : 0;
            velocity[i].velocity = {.25 * a + b, a, .25 * a - b};
        }
        std::vector<GeneralVoFFaceVolume> zero;
        std::set<std::uint64_t> exterior;
        for (const auto& face : state.tiles())
        {
            zero.push_back({face.face_gid, 0});
            if (face.exterior)
                exterior.insert(face.face_gid);
        }
        const auto trial = state.plan_transport(velocity, zero, 1);
        const auto moved = state.apply(trial);
        EXPECT_EQ(trial.node_trajectories().size(), state.nodes().size());
        int local_tangent = 0, local_edge = 0;
        for (const auto& record : trial.node_trajectories())
        {
            const auto node = std::find_if(state.nodes().begin(), state.nodes().end(),
                [&](const auto& value) { return value.gid == record.node; });
            EXPECT_NE(node, state.nodes().end());
            if (node == state.nodes().end())
                continue;
            const auto p = node->coordinate;
            const double u = (p[0] + p[2] - .5 * p[1]) / 2, w = (p[2] - p[0]) / 2;
            const double a = p[1] == 1 ? (u - 1) / 64 : 0;
            const auto expected = ExactNodeTrajectory::backtrace(node->exact_identity, {.25 * a, a, .25 * a}, 1);
            // The same analytic certificate is required under both owner
            // permutations, without receive-order or rounded-coordinate ties.
            EXPECT_EQ(record.trajectory.point, expected.point);
            EXPECT_EQ(record.trajectory.departure, expected.departure);
            EXPECT_EQ(record.trajectory.displacement_squared().compare(expected.displacement_squared()), 0);
            if (a && (w == 0 || w == 2))
            {
                ++local_tangent;
                if (u == 0 || u == 2)
                    ++local_edge;
            }
        }
        int tangent = 0, edge = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_tangent, &tangent);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_edge, &edge);
        EXPECT_GT(tangent, 0);
        EXPECT_GT(edge, 0);
        std::uint64_t local_routes = 0, routes = 0;
        for (const auto& stream : trial.face_contributions())
            EXPECT_FALSE(exterior.contains(stream.face));
        for (const auto& transfer : trial.transfers())
            local_routes += transfer.donor != transfer.recipient && transfer.exact.bulk.sign() > 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_routes, &routes);
        EXPECT_GT(routes, 0);
        for (const auto& budget : trial.donor_budgets())
        {
            EXPECT_EQ((budget.exact_exported.gas + budget.exact_retained.gas).compare(budget.exact_old.gas), 0);
            EXPECT_EQ(
                (budget.exact_exported.carrier + budget.exact_retained.carrier).compare(budget.exact_old.carrier), 0);
        }
        for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
        {
            EXPECT_NEAR(moved->gas_volumes()[i], state.gas_volumes()[i], 3e-13);
            EXPECT_NEAR(moved->carrier_volumes()[i], state.carrier_volumes()[i], 3e-13);
        }
        if (comm->getRank() >= 2)
            EXPECT_TRUE(moved->owned_order().empty());
        std::vector<double> microscopic(fixture.order.size(), 1e-310), full(fixture.order.size(), 2);
        GeneralVoF unresolved(fixture.hierarchy, fixture.order, microscopic, full);
        auto normal_only = nodes(unresolved);
        for (std::size_t i = 0; i < unresolved.nodes().size(); ++i)
        {
            const auto p = unresolved.nodes()[i].coordinate;
            const double w = (p[2] - p[0]) / 2;
            normal_only[i].velocity = w == 0 || w == 2 ? GeometryPoint{1. / 128, 0, -1. / 128} : GeometryPoint{};
        }
        EXPECT_THROW(unresolved.reconstruct(), std::invalid_argument);
        const auto identity = unresolved.plan_transport(normal_only, zero, 1);
        const auto retained = unresolved.apply(identity);
        EXPECT_TRUE(identity.face_contributions().empty());
        for (const auto& record : identity.node_trajectories())
            EXPECT_EQ(record.trajectory.point, record.trajectory.departure);
        EXPECT_EQ(retained->gas_volumes(), microscopic);
        EXPECT_EQ(retained->carrier_volumes(), full);
        if (comm->getSize() > 1)
        {
            auto disagree = velocity;
            if (comm->getRank() == 0)
                for (std::size_t i = 0; i < state.nodes().size(); ++i)
                    if (state.nodes()[i].coordinate == GeometryPoint{1.25, 1, 1.25})
                    {
                        disagree[i].velocity[0] += 1. / 128;
                        disagree[i].velocity[2] -= 1. / 128;
                    }
            // Raw model controls must agree even when their discrepancy is
            // removed by exactly the same wall projection.
            EXPECT_THROW(state.plan_transport(disagree, zero, 1), std::invalid_argument);
        }
    }
}

TEST(ExactGeneralVoF, ExactSubquantumStreamsRemainExplicitAndFiniteRejectsCrossPolicyControls)
{
    auto fixture = setup();
    for (std::size_t i = 0; i < fixture.order.size(); ++i)
    {
        const auto center = fixture.hierarchy->geometry(fixture.hierarchy->owned().at(fixture.order[i]))->centroid();
        fixture.gas[i] = center[2] < 1 ? .5 : 0;
        fixture.carrier[i] = 1 - fixture.gas[i];
    }
    GeneralVoF source(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier, exact_reference_options());
    auto phi = cycle(source);
    const auto tiny = std::numeric_limits<double>::denorm_min();
    for (auto& face : phi)
        if (face.integrated_volume != 0)
        {
            face.exact_integrated_volume =
                ExactGeometryMeasure::from_value(tiny).scaled(face.integrated_volume > 0 ? .5 : -.5);
            face.integrated_volume = 0; // Explicit diagnostic underflow, NOT physical zero.
        }
    auto trial = source.plan_transport(nodes(source), phi, 1);
    auto result = source.apply(trial);
    const auto half_min = ExactGeometryMeasure::from_value(tiny).scaled(.5);
    int local = 0;
    const auto inspect = [&](const ExactGeometryMeasure& measure, const CertifiedGeometryValue& diagnostic)
    {
        if (measure.sign() > 0 && measure.compare(half_min) <= 0)
        {
            ++local;
            EXPECT_TRUE(diagnostic.exact_nonzero);
            // Consuming a diagnostic as a finite positive physical double is
            // forbidden; exact records remain available for all arithmetic.
            EXPECT_THROW(measure.certified().as_double(), std::runtime_error);
        }
    };
    for (const auto& stream : trial.face_contributions())
    {
        inspect(stream.exact.gas, stream.diagnostic.gas);
        inspect(stream.exact.carrier, stream.diagnostic.carrier);
        EXPECT_EQ((stream.exact.gas + stream.exact.carrier).compare(stream.exact.bulk), 0);
    }
    for (const auto& transfer : trial.transfers())
    {
        inspect(transfer.exact.gas, transfer.diagnostic.gas);
        inspect(transfer.exact.carrier, transfer.diagnostic.carrier);
    }
    int count = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &count);
    EXPECT_GT(count, 0);
    EXPECT_EQ(result->gas_volumes(), source.gas_volumes());
    EXPECT_EQ(result->carrier_volumes(), source.carrier_volumes());
    for (const auto& budget : trial.donor_budgets())
    {
        EXPECT_EQ((budget.exact_exported.gas + budget.exact_retained.gas).compare(budget.exact_old.gas), 0);
        EXPECT_EQ((budget.exact_exported.carrier + budget.exact_retained.carrier).compare(budget.exact_old.carrier), 0);
    }
    // The finite numerical route cannot acquire rational face authority. It
    // rejects the unsupported control atomically rather than deleting a phase.
    auto pure_fixture = setup();
    GeneralVoF pure(pure_fixture.hierarchy, pure_fixture.order, pure_fixture.gas, pure_fixture.carrier);
    auto tiny_phi = cycle(pure);
    for (auto& face : tiny_phi)
        if (face.integrated_volume != 0)
        {
            face.exact_integrated_volume = ExactGeometryMeasure::from_value(tiny).scaled(
                face.integrated_volume > 0 ? .5 : -.5);
            face.integrated_volume = 0;
        }
    EXPECT_THROW(pure.plan_transport(nodes(pure), tiny_phi, 1), std::invalid_argument);
    auto malformed = tiny_phi;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (auto& face : malformed)
            if (face.exact_integrated_volume)
            {
                face.exact_integrated_volume = *face.exact_integrated_volume + ExactGeometryMeasure::from_value(tiny);
                break;
            }
    EXPECT_THROW((void) pure.plan_transport(nodes(pure), malformed, 1), std::invalid_argument);
    EXPECT_EQ(pure.gas_volumes(), pure_fixture.gas);
    EXPECT_EQ(pure.carrier_volumes(), pure_fixture.carrier);
    EXPECT_NO_THROW(pure.plan_transport(nodes(pure), cycle(pure), 1));
}

TEST(ExactGeneralVoF, DirectAdoptionWitnessRejectsLookAlikesAndDoesNotRetainOldPublication)
{
    const auto comm = Tpetra::getDefaultComm();
    std::weak_ptr<const void> old_identity;
    std::shared_ptr<const GeneralVoF> retained_target;
    {
        auto fixture = setup();
        fixture.gas.assign(fixture.order.size(), 0);
        fixture.carrier.assign(fixture.order.size(), 1);
        auto source =
            std::make_shared<const GeneralVoF>(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier);
        old_identity = source->accepted_identity();
        EXPECT_NE(source->accepted_control(), 0);
        EXPECT_EQ(source->accepted_identity(), source->accepted_identity());
        std::vector<int> destinations(
            source->hierarchy()->owned().size(), (comm->getRank() + 1) % std::min(2, comm->getSize()));
        const auto plan = source->hierarchy()->plan_migration(destinations);
        const auto raw = source->hierarchy()->apply(plan);
        const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
        const auto target_h =
            source->hierarchy()->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
        const auto transfer = ExactPhaseTransfer::prepare(source->export_transfer_source(false), plan, target_h);
        retained_target = source->adopt_transfer(transfer);
        EXPECT_NO_THROW(retained_target->validate_transfer_target(transfer));
        EXPECT_NE(retained_target->accepted_identity(), source->accepted_identity());
        EXPECT_NE(retained_target->accepted_control(), source->accepted_control());
        GeneralVoF look_alike(retained_target->hierarchy(), retained_target->owned_order(),
            retained_target->gas_volumes(), retained_target->carrier_volumes(), {},
            std::vector<GeneralVoFPhaseGeometry>(
                retained_target->owned_order().size(), GeneralVoFPhaseGeometry::PureCarrier));
        EXPECT_EQ(look_alike.gas_volumes(), retained_target->gas_volumes());
        EXPECT_EQ(look_alike.carrier_volumes(), retained_target->carrier_volumes());
        EXPECT_THROW(look_alike.validate_transfer_target(transfer), std::invalid_argument);
        const auto moved_trial = retained_target->plan_transport(nodes(*retained_target), cycle(*retained_target), 1);
        const auto moved = retained_target->apply(moved_trial);
        EXPECT_EQ(moved->gas_volumes(), retained_target->gas_volumes());
        EXPECT_EQ(moved->carrier_volumes(), retained_target->carrier_volumes());
        EXPECT_NE(moved->accepted_control(), retained_target->accepted_control());
        EXPECT_THROW(moved->validate_transfer_target(transfer), std::invalid_argument);
        EXPECT_NO_THROW(retained_target->validate_transfer_target(transfer));
    }
    // The witness holds only a weak identity. It must not retain a chain of
    // source phase arrays, geometry, native maps or old accepted publications.
    EXPECT_TRUE(old_identity.expired());
    EXPECT_TRUE(retained_target);
}

TEST(ExactGeneralVoF, ExactZeroBootstrapRetainsPureEvidenceWithoutClassifyingNonzeroTails)
{
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier, exact_reference_options());
    constexpr double tail = 1e-26;
    auto controls = cycle(state);
    for (auto& face : controls)
        face.integrated_volume = face.integrated_volume > 0 ? tail : face.integrated_volume < 0 ? -tail : 0;
    const auto trial = state.plan_transport(nodes(state), controls, 1);
    const auto moved = state.apply(trial);
    const auto delivered = delivered_phases(trial);
    const auto source = moved->export_transfer_source(false);
    int local_tails = 0;
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto& phase = moved->phase_volumes()[i];
        const bool was_gas = f.carrier[i] == 0;
        const auto& minority = was_gas ? phase.carrier : phase.gas;
        const auto& capsule = source.owned()[i];
        EXPECT_TRUE(capsule.geometry_ready);
        EXPECT_EQ(capsule.geometry.pure_gas, was_gas);
        EXPECT_EQ(capsule.geometry.pure_carrier, !was_gas);
        const auto gid = std::uint64_t(moved->mesh()->cell_global_id(int(i)));
        const auto expected = was_gas ? delivered.at(gid).carrier : delivered.at(gid).gas;
        const auto error = was_gas ? trial.phase_roundoff_bounds()[i].carrier : trial.phase_roundoff_bounds()[i].gas;
        const auto difference = expected - ExactGeometryMeasure::from_value(minority);
        EXPECT_LE((difference.sign() < 0 ? difference.scaled(-1) : difference)
                      .compare(ExactGeometryMeasure::from_value(error)),
            0);
        if (expected.sign() > 0)
        {
            ++local_tails;
            EXPECT_GT(minority, 0); // representable minorities are never classified away
            EXPECT_NEAR(minority, tail, 2 * std::numeric_limits<double>::epsilon() * tail);
        }
        // Exact reference transport preserves the occupancy allowance and the
        // independent tail. Its pure-geometry fit bound records that actual
        // minority residual rather than claiming a zero residual.
        const auto& before = state.initial_occupancy_errors()[i];
        const auto& after = moved->initial_occupancy_errors()[i];
        EXPECT_LE(std::abs(after.difference - before.difference),
            moved->phase_roundoff_bounds()[i].gas + moved->phase_roundoff_bounds()[i].carrier);
        EXPECT_GE(after.admitted_bound, before.admitted_bound);
        EXPECT_EQ(after.fit_bound, std::max(before.fit_bound, minority));
    }
    int tails = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_tails, &tails);
    EXPECT_GT(tails, 0);
    expect_finite_phase_balance(state, *moved, trial);
    // Reconstruction uses the preserved proof, without fitting an arbitrary
    // normal to the transported microscopic minority.
    const auto planes = moved->reconstruct();
    for (std::size_t i = 0; i < planes.size(); ++i)
        EXPECT_EQ(planes[i].pure_gas, f.carrier[i] == 0);

    std::vector<GeneralVoFPhaseGeometry> infer(f.order.size(), GeneralVoFPhaseGeometry::Infer);
    GeneralVoF explicit_infer(f.hierarchy, f.order, f.gas, f.carrier, exact_reference_options(), infer);
    const auto inferred = explicit_infer.apply(explicit_infer.plan_transport(nodes(explicit_infer), controls, 1));
    const auto inferred_source = inferred->export_transfer_source(false);
    int local_unresolved = 0;
    for (const auto& capsule : inferred_source.owned())
        if (capsule.phases.gas > 0 && capsule.phases.carrier > 0)
        {
            ++local_unresolved;
            EXPECT_FALSE(capsule.geometry_ready);
            EXPECT_FALSE(capsule.geometry.pure_gas);
            EXPECT_FALSE(capsule.geometry.pure_carrier);
        }
    int unresolved = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_unresolved, &unresolved);
    EXPECT_GT(unresolved, 0);

    auto initial_tail = setup(true);
    GeneralVoF nonzero(initial_tail.hierarchy, initial_tail.order, initial_tail.gas, initial_tail.carrier);
    const auto nonzero_source = nonzero.export_transfer_source(false);
    for (const auto& capsule : nonzero_source.owned())
    {
        EXPECT_GT(capsule.phases.gas, 0);
        EXPECT_FALSE(capsule.geometry_ready);
        EXPECT_FALSE(capsule.geometry.pure_gas);
        EXPECT_FALSE(capsule.geometry.pure_carrier);
    }
}

TEST(ExactGeneralVoF, DonorLiveLeasesReleaseReplacedValuesWithoutLosingExactColumns)
{
    using Live = SimpleFluid::VoF::detail::ExactLiveStorage;
    using M = ExactGeometryMeasure;
    SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{64 * 1024};
    auto quantum = M::from_value(1).divided_by(M::from_value(3));
    for (unsigned i = 0; i < 8; ++i)
        quantum = quantum.squared();
    M total;
    const auto source_identity = quantum.storage_identity();
    const auto source_wire = quantum.encode();
    {
        Live live(budget, 32 * 1024);
        {
            Live::Scope scope(live, {&quantum, &total});
            const auto held = budget.used;
            {
                auto first = live.pin(quantum);
                auto second = live.pin(quantum);
                EXPECT_EQ(budget.used, held); // immutable aliases do not duplicate payload
            }
            for (unsigned i = 0; i < 4096; ++i)
                live.add(total, quantum);
            EXPECT_EQ(total.compare(quantum.scaled(4096)), 0);
            EXPECT_LT(budget.used, budget.limit);
            const auto accepted = total;
            auto too_large = M::from_value(2);
            for (unsigned i = 0; i < 20; ++i)
                too_large = too_large.squared();
            ASSERT_GT(too_large.storage_bytes(), live.operation_bytes());
            EXPECT_THROW(live.add(total, too_large), std::invalid_argument);
            EXPECT_EQ(total.compare(accepted), 0);
            EXPECT_NO_THROW(live.add(total, quantum));
            EXPECT_EQ(total.compare(quantum.scaled(4097)), 0);
        }
        EXPECT_EQ(budget.used, live.operation_bytes());
    }
    EXPECT_EQ(budget.used, 0u);
    EXPECT_EQ(quantum.storage_identity(), source_identity);
    EXPECT_EQ(quantum.encode(), source_wire);
}

TEST(ExactGeneralVoF, DonorLinearRawGroupsMatchPerPieceHistoryAndRepairExactly)
{
    using M = ExactGeometryMeasure;
    using Accumulator = SimpleFluid::VoF::detail::ExactPhaseAccumulator;
    using Live = SimpleFluid::VoF::detail::ExactLiveStorage;
    using Map = SimpleFluid::VoF::detail::ExactPhaseLabelMap;
    const auto ratio = [](double a, double b) { return M::from_value(a).divided_by(M::from_value(b)); };
    auto first_denominator = M::from_value(17), second_denominator = M::from_value(19);
    for (unsigned i = 0; i < 11; ++i)
    {
        first_denominator = first_denominator.squared();
        second_denominator = second_denominator.squared();
    }
    const auto history = M::from_value(1).divided_by(first_denominator).divided_by(second_denominator);
    ASSERT_GT(history.sign(), 0);
    ASSERT_GT(history.storage_bytes(), 1024u);
    for (bool gas : {false, true})
        for (int correction : {-1, 1})
        {
            const ExactPhaseMeasures full{
                M::from_value(1), gas ? ratio(2, 5) : ratio(3, 5), gas ? ratio(3, 5) : ratio(2, 5)};
            const std::array<ExactPhaseMeasures, 3> pieces{{{ratio(1, 3), ratio(1, 7), ratio(1, 3) - ratio(1, 7)},
                {ratio(1, 5), ratio(1, 11), ratio(1, 5) - ratio(1, 11)},
                {ratio(7, 15), full.gas - ratio(1, 7) - ratio(1, 11),
                    ratio(7, 15) - full.gas + ratio(1, 7) + ratio(1, 11)}}};
            const auto target = (gas ? full.gas : full.carrier) + history.scaled(correction);
            const auto map = Map::create(full, gas, target, history, 8 * 1024 * 1024);
            for (int sign : {-1, 1})
            {
                SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{8 * 1024 * 1024};
                Live live(budget, 2 * 1024 * 1024);
                Accumulator grouped(live);
                ExactPhaseMeasures expected;
                M old_absolute, old_bound;
                for (const auto& piece : pieces)
                {
                    const auto paired = map.apply(piece);
                    expected.bulk = expected.bulk + paired.bulk.scaled(sign);
                    expected.gas = expected.gas + paired.gas.scaled(sign);
                    expected.carrier = expected.carrier + paired.carrier.scaled(sign);
                    const auto delta = paired.gas - piece.gas;
                    old_absolute = old_absolute + (delta.sign() < 0 ? delta.scaled(-1) : delta);
                    old_bound = old_bound + map.bound();
                    grouped.add(piece, sign);
                }
                const auto actual = grouped.apply(map);
                EXPECT_EQ(actual.bulk.encode(), expected.bulk.encode());
                EXPECT_EQ(actual.gas.encode(), expected.gas.encode());
                EXPECT_EQ(actual.carrier.encode(), expected.carrier.encode());
                EXPECT_EQ(grouped.count(), pieces.size());
                const auto delta = actual.gas - grouped.raw().gas;
                const auto absolute = delta.sign() < 0 ? delta.scaled(-1) : delta;
                EXPECT_EQ(absolute.encode(), old_absolute.encode());
                EXPECT_EQ(map.bound().scaled(double(grouped.count())).encode(), old_bound.encode());
            }
        }
}

TEST(ExactGeneralVoF, DonorRawGroupingPreservesMappedZeroForSingularAndIdentityMaps)
{
    using M = ExactGeometryMeasure;
    using Map = SimpleFluid::VoF::detail::ExactPhaseLabelMap;
    const ExactPhaseMeasures full{M::from_value(1), M::from_value(.25), M::from_value(.75)};
    const ExactPhaseMeasures first{M::from_value(1), M::from_value(.25), M::from_value(.75)};
    const ExactPhaseMeasures second{M::from_value(1), M::from_value(.5), M::from_value(.5)};
    for (bool gas : {false, true})
        for (double target : {0., 1.})
        {
            const auto map = Map::create(full, gas, M::from_value(target), M::from_value(1), 1 << 20);
            SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{1 << 20};
            SimpleFluid::VoF::detail::ExactLiveStorage live(budget, 1 << 18);
            SimpleFluid::VoF::detail::ExactPhaseAccumulator row(live);
            row.add(first);
            row.add(second, -1);
            ASSERT_NE(row.raw().gas.sign(), 0); // raw-zero would change the admitted singular-map case
            const auto paired = row.apply(map);
            EXPECT_EQ(paired.bulk.sign(), 0);
            EXPECT_EQ(paired.gas.sign(), 0);
            EXPECT_EQ(paired.carrier.sign(), 0);
        }
    const auto identity = Map::create(full, true, full.gas, {}, 1 << 20);
    SimpleFluid::Meshes::AMR::adaptive_detail::Budget budget{1 << 20};
    SimpleFluid::VoF::detail::ExactLiveStorage live(budget, 1 << 18);
    SimpleFluid::VoF::detail::ExactPhaseAccumulator zero(live);
    EXPECT_EQ(zero.apply(identity).bulk.sign(), 0);
    EXPECT_EQ(zero.apply(identity).gas.sign(), 0);
    EXPECT_EQ(zero.apply(identity).carrier.sign(), 0);
    zero.add(first);
    zero.add(first, -1);
    EXPECT_EQ(zero.apply(identity).gas.sign(), 0);
    EXPECT_THROW(zero.add({M::from_value(1), M::from_value(1), M::from_value(1)}), std::invalid_argument);
    EXPECT_EQ(zero.apply(identity).carrier.sign(), 0);
}

TEST(ExactGeneralVoF, FinitePublicationVerifiesOutwardBoundsAndKeepsRepresentableMinorities)
{
    using M = ExactGeometryMeasure;
    using SimpleFluid::VoF::detail::phase_upper_bound;
    using SimpleFluid::VoF::detail::round_phase_value;
    const auto third = M::from_value(1).divided_by(M::from_value(3));
    const auto rounded = round_phase_value(third);
    const auto error = M::from_value(rounded.value) - third;
    EXPECT_LE((error.sign() < 0 ? error.scaled(-1) : error).compare(M::from_value(rounded.absolute_error)), 0);
    EXPECT_GE(M::from_value(phase_upper_bound(third)).compare(third), 0);
    const auto eta = std::numeric_limits<double>::denorm_min();
    EXPECT_EQ(round_phase_value(M::from_value(eta)).value, eta);
    const auto subquantum = M::from_value(eta).divided_by(M::from_value(3));
    EXPECT_EQ(round_phase_value(subquantum).value, 0); // no retained rational phase tail
    EXPECT_GE(M::from_value(phase_upper_bound(subquantum)).compare(subquantum), 0);
    EXPECT_THROW(round_phase_value(M::from_value(-eta)), std::invalid_argument);
}

TEST(ExactGeneralVoF, ExperimentalInventoryPolicyIsExplicitCollectiveAndIsolatedFromFinite)
{
    auto f = setup();
    GeneralVoF finite(f.hierarchy, f.order, f.gas, f.carrier);
    EXPECT_EQ(finite.inventory_policy(), PhaseInventoryPolicy::Finite);
    EXPECT_THROW((void) finite.experimental_phases(), std::invalid_argument);
    const auto finite_source = finite.export_transfer_source(false);
    for (const auto& capsule : finite_source.owned())
    {
        EXPECT_EQ(capsule.policy, PhaseInventoryPolicy::Finite);
        EXPECT_FALSE(capsule.experimental_exact);
    }
    GeneralVoFOptions options;
    options.inventory_policy = PhaseInventoryPolicy::ExperimentalExact;
    GeneralVoF exact(f.hierarchy, f.order, f.gas, f.carrier, options);
    EXPECT_EQ(exact.inventory_policy(), PhaseInventoryPolicy::ExperimentalExact);
    EXPECT_EQ(exact.experimental_phases().size(), exact.owned_order().size());
    int local_empty = exact.owned_order().empty(), empty = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (comm->getSize() == 4)
        EXPECT_GT(empty, 0);
    if (comm->getSize() > 1)
    {
        auto mixed = options;
        if (comm->getRank() != 0)
            mixed.inventory_policy = PhaseInventoryPolicy::Finite;
        EXPECT_THROW((GeneralVoF(f.hierarchy, f.order, f.gas, f.carrier, mixed)), std::invalid_argument);
    }
    auto invalid = options;
    invalid.inventory_policy = static_cast<PhaseInventoryPolicy>(2);
    EXPECT_THROW((GeneralVoF(f.hierarchy, f.order, f.gas, f.carrier, invalid)), std::invalid_argument);
    std::vector<GeneralVoFFaceVolume> zero;
    for (const auto& face : exact.tiles())
        zero.push_back({face.face_gid, 0});
    const auto exact_trial = exact.plan_transport(nodes(exact), zero, 1);
    EXPECT_THROW((void) finite.apply(exact_trial), std::invalid_argument);
    const auto finite_trial = finite.plan_transport(nodes(finite), zero, 1);
    EXPECT_THROW((void) exact.apply(finite_trial), std::invalid_argument);
    EXPECT_NO_THROW((void) exact.apply(exact_trial));
    EXPECT_NO_THROW((void) finite.apply(finite_trial));
}

TEST(ExactGeneralVoF, ExperimentalNonDyadicHistorySurvivesOnlyAfterExplicitOptIn)
{
    using M = ExactGeometryMeasure;
    auto f = setup();
    GeneralVoFOptions options;
    options.inventory_policy = PhaseInventoryPolicy::ExperimentalExact;
    GeneralVoF exact(f.hierarchy, f.order, f.gas, f.carrier, options);
    auto controls = cycle(exact);
    const auto q = M::from_value(1).divided_by(M::from_value(79));
    for (auto& face : controls)
        if (face.integrated_volume != 0)
        {
            face.exact_integrated_volume = q.scaled(face.integrated_volume > 0 ? 1 : -1);
            face.integrated_volume = static_cast<double>(face.exact_integrated_volume->enclosure().value);
        }
    const auto trial = exact.plan_transport(nodes(exact), controls, 1);
    const auto moved = exact.apply(trial);
    const auto delivered = delivered_phases(trial);
    int local_difference = 0;
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto gid = std::uint64_t(moved->mesh()->cell_global_id(int(i)));
        const auto& authority = moved->experimental_phases()[i];
        EXPECT_EQ(authority.gas.encode(), delivered.at(gid).gas.encode());
        EXPECT_EQ(authority.carrier.encode(), delivered.at(gid).carrier.encode());
        local_difference += authority.gas.compare(M::from_value(moved->phase_volumes()[i].gas)) != 0;
    }
    int difference = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_difference, &difference);
    EXPECT_GT(difference, 0); // Exact authority has not been replaced by its display.
    const auto accepted = moved->inventory_measures();
    std::vector<GeneralVoFFaceVolume> zero;
    for (const auto& face : moved->tiles())
        zero.push_back({face.face_gid, 0});
    const auto replay = moved->plan_transport(nodes(*moved), zero, 1);
    const auto unchanged = moved->apply(replay);
    for (std::size_t i = 0; i < accepted.size(); ++i)
        EXPECT_EQ(unchanged->experimental_phases()[i].gas.encode(), accepted[i].gas.encode());
    EXPECT_THROW((void) moved->apply(trial), std::invalid_argument); // old-source trial
    EXPECT_EQ(exact.gas_volumes(), f.gas);
    EXPECT_NO_THROW((void) moved->apply(replay));
}
} // namespace
