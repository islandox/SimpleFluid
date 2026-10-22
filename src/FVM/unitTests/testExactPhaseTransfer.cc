/** @file testExactPhaseTransfer.cc
 * @brief Finite phase remap, bounded rounding and transaction-local tiny momentum.
 */
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::Meshes::AMR;
using namespace SimpleFluid::VoF;
using M = ExactGeometryMeasure;
using H = DistributedAdaptiveHierarchy;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const H> bootstrap(const CartesianAMR& root, std::uint64_t count)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < count; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto b = root.detach_owned(owned, {'f', 'i', 'n', 'i', 't', 'e', '_', 'r', 'e', 'm', 'a', 'p'});
    return std::make_shared<const H>(std::move(b.owned), std::move(b.families), b.kernel, comm);
}
std::shared_ptr<const H> bind(const H& source, const H::Plan& plan)
{
    auto raw = source.apply(plan);
    auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    return source.apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
}
M absolute(const M& value)
{
    return value.sign() < 0 ? value.scaled(-1) : value;
}
M global(const M& local)
{
    const auto comm = Tpetra::getDefaultComm();
    const std::array<OwnerPacket, 1> packets{{{0, local.encode(1 << 20)}}};
    const auto received = exchange_owner_packets(packets, comm, 1 << 20);
    M sum;
    if (comm->getRank() == 0)
        for (const auto& p : received.packets)
            sum = sum + M::decode(p.bytes, 1 << 20);
    return sum;
}
void empty_owners(const H& h)
{
    int local = h.owned().empty(), total = 0;
    Teuchos::reduceAll(*h.communicator(), Teuchos::REDUCE_SUM, 1, &local, &total);
    if (h.communicator()->getSize() == 4)
        EXPECT_GT(total, 0);
}
ExactPhaseTransferSource channels(const GeneralVoF& state)
{
    auto source = state.export_transfer_source(false);
    std::vector<ExactTransferOwnedChannels> values;
    for (const auto& c : source.owned())
    {
        ExactTransferPhaseMass mass{
            M::from_value(c.phases.gas).scaled(2), M::from_value(c.phases.carrier).scaled(1000)};
        values.push_back(
            {c.key, {(mass.gas + mass.carrier).scaled(-.5), M::from_value(c.phases.bulk).scaled(3)}, mass});
    }
    return source.with_channels({{"momentum", ExactTransferCarrier::MixtureMass, false},
                                    {"pressure_volume", ExactTransferCarrier::BulkVolume, true}},
        std::move(values));
}
void conservation(const ExactPhaseTransferSource& source, const ExactPhaseTransferResult& target)
{
    M sg, sc, tg, tc, rg, rc, sm, tm, sp, tp;
    for (const auto& c : source.owned())
    {
        sg = sg + M::from_value(c.phases.gas);
        sc = sc + M::from_value(c.phases.carrier);
        if (!c.amounts.empty())
        {
            sm = sm + c.amounts[0];
            sp = sp + c.amounts[1];
        }
    }
    for (const auto& c : target.owned())
    {
        tg = tg + M::from_value(c.phases.gas);
        tc = tc + M::from_value(c.phases.carrier);
        rg = rg + M::from_value(c.roundoff.gas);
        rc = rc + M::from_value(c.roundoff.carrier);
        if (!c.amounts.empty())
        {
            tm = tm + c.amounts[0];
            tp = tp + c.amounts[1];
        }
        EXPECT_TRUE(std::isfinite(c.phases.gas));
        EXPECT_TRUE(std::isfinite(c.phases.carrier));
        EXPECT_GE(c.phases.gas, 0);
        EXPECT_GE(c.phases.carrier, 0);
    }
    sg = global(sg);
    sc = global(sc);
    tg = global(tg);
    tc = global(tc);
    rg = global(rg);
    rc = global(rc);
    sm = global(sm);
    tm = global(tm);
    sp = global(sp);
    tp = global(tp);
    if (Tpetra::getDefaultComm()->getRank() == 0)
    {
        EXPECT_LE(absolute(tg - sg).compare(rg), 0);
        EXPECT_LE(absolute(tc - sc).compare(rc), 0);
        EXPECT_EQ(sm.compare(tm), 0);
        EXPECT_EQ(sp.compare(tp), 0);
    }
}
TEST(ExactPhaseTransfer, FiniteInventoriesMigrateSplitAndCoarsenWithinActualPublicationBounds)
{
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, .7};
    CartesianAMR root(grid);
    auto h = bootstrap(root, 8);
    std::vector<AdaptiveCellKey> keys;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : h->owned())
    {
        keys.push_back(key);
        gas.push_back(1e-200);
        carrier.push_back(h->metrics(leaf).volume);
    }
    std::vector<GeneralVoFPhaseGeometry> declarations(keys.size(), GeneralVoFPhaseGeometry::PureCarrier);
    GeneralVoF initial(h, keys, gas, carrier, {}, declarations);
    auto source = channels(initial);
    EXPECT_EQ(source.inventory_policy(), PhaseInventoryPolicy::Finite);
    for (const auto& c : source.owned())
    {
        EXPECT_EQ(c.policy, PhaseInventoryPolicy::Finite);
        EXPECT_FALSE(c.experimental_exact);
    }
    std::vector<int> destinations(
        keys.size(), (h->communicator()->getRank() + 1) % std::min(2, h->communicator()->getSize()));
    const auto migration = source.hierarchy()->plan_migration(destinations);
    const auto moved_h = bind(*source.hierarchy(), migration);
    const auto moved = ExactPhaseTransfer::prepare(source, migration, moved_h);
    ASSERT_TRUE(moved.ownership_only());
    empty_owners(*moved_h);
    for (const auto& c : moved.owned())
        EXPECT_EQ(std::bit_cast<std::uint64_t>(c.phases.gas), std::bit_cast<std::uint64_t>(1e-200));
    conservation(source, moved);
    auto state = initial.adopt_transfer(moved);
    std::vector<AdaptiveMark> marks;
    for (const auto& [key, leaf] : state->hierarchy()->owned())
        if (key == root.cell_key(0))
            marks.push_back({key, AdaptiveSplit::Isotropic});
    const auto plan = state->hierarchy()->plan_adaptation(marks);
    const auto fine_h = bind(*state->hierarchy(), plan);
    const auto finite_source = channels(*state);
    const auto refined = ExactPhaseTransfer::prepare(finite_source, plan, fine_h);
    conservation(finite_source, refined);
    empty_owners(*fine_h);
    for (const auto& c : refined.owned())
    {
        EXPECT_GT(c.phases.gas, 0);
        EXPECT_EQ(c.phases.bulk, fine_h->metrics(fine_h->owned().at(c.key)).volume);
        // Per-phase publication errors scale with the tiny phase, not cell V.
        EXPECT_LE(c.roundoff.gas, 16 * std::numeric_limits<double>::epsilon() * c.phases.gas);
    }
    ExactPhaseTransferOptions exhausted;
    exhausted.byte_budget = 1;
    EXPECT_THROW(ExactPhaseTransfer::prepare(finite_source, plan, fine_h, exhausted), std::invalid_argument);
    ExactPhaseTransferOptions no_fit;
    no_fit.maximum_relative_target_fit = 0;
    EXPECT_THROW(ExactPhaseTransfer::prepare(finite_source, plan, fine_h, no_fit), std::invalid_argument);
    const auto fine = state->adopt_transfer(refined);
    std::vector<AdaptiveBytes> inverse;
    if (h->communicator()->getRank() == 0)
        inverse.push_back(adaptive_family_key(root.cell_key(0), AdaptiveSplit::Isotropic));
    const auto coarsening = fine->hierarchy()->plan_adaptation({}, inverse);
    const auto coarse_h = bind(*fine->hierarchy(), coarsening);
    const auto fine_source = channels(*fine);
    const auto coarse = ExactPhaseTransfer::prepare(fine_source, coarsening, coarse_h);
    conservation(fine_source, coarse);
    const auto restored = fine->adopt_transfer(coarse);
    for (std::size_t i = 0; i < restored->phase_volumes().size(); ++i)
    {
        EXPECT_EQ(restored->phase_volumes()[i].gas, coarse.owned()[i].phases.gas);
        EXPECT_EQ(restored->phase_volumes()[i].carrier, coarse.owned()[i].phases.carrier);
    }
    EXPECT_THROW(initial.adopt_transfer(coarse), std::invalid_argument);
    EXPECT_THROW(ExactPhaseTransfer::prepare(source, coarsening, coarse_h), std::invalid_argument);
}
TEST(ExactPhaseTransfer, FiniteInheritedDefectAndRoundingAreNotResetByAdoption)
{
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    CartesianAMR root(grid);
    const auto h = bootstrap(root, 8);
    const double defect = 6 * std::numeric_limits<double>::epsilon();
    std::vector<AdaptiveCellKey> keys;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : h->owned())
    {
        keys.push_back(key);
        const bool mixed = std::get<CartesianCellAncestry>(key.ancestry).index[0] == 0;
        gas.push_back(mixed ? .5 : 0);
        carrier.push_back(mixed ? .5 + defect : 1);
    }
    GeneralVoF initial(h, keys, gas, carrier);
    const auto source = initial.export_transfer_source(true);
    std::vector<AdaptiveMark> marks;
    for (const auto& [key, leaf] : initial.hierarchy()->owned())
        if (key == root.cell_key(0))
            marks.push_back({key, AdaptiveSplit::Isotropic});
    const auto plan = initial.hierarchy()->plan_adaptation(marks);
    const auto target = bind(*initial.hierarchy(), plan);
    const auto result = ExactPhaseTransfer::prepare(source, plan, target);
    const auto adopted = initial.adopt_transfer(result);
    conservation(source, result);
    int local_inherited = 0;
    for (std::size_t i = 0; i < result.owned().size(); ++i)
    {
        const auto& c = result.owned()[i];
        const auto& error = adopted->initial_occupancy_errors()[i];
        EXPECT_EQ(error.admitted_bound, c.error.admitted_bound);
        EXPECT_EQ(error.difference, c.error.difference);
        EXPECT_EQ(adopted->phase_roundoff_bounds()[i].gas, c.roundoff.gas);
        const auto bulk = M::volume(*target->geometry(target->owned().at(c.key)));
        const auto actual = M::from_value(c.phases.gas) + M::from_value(c.phases.carrier) - bulk;
        EXPECT_LE(absolute(actual).compare(M::from_value(c.error.admitted_bound)), 0);
        if (absolute(actual).compare(bulk.scaled(8 * std::numeric_limits<double>::epsilon())) > 0)
            ++local_inherited;
    }
    int inherited = 0;
    Teuchos::reduceAll(*h->communicator(), Teuchos::REDUCE_SUM, 1, &local_inherited, &inherited);
    EXPECT_GT(inherited, 0);
    EXPECT_THROW(
        GeneralVoF(adopted->hierarchy(), adopted->owned_order(), adopted->gas_volumes(), adopted->carrier_volumes()),
        std::invalid_argument);
    result.validate_observations();
}
TEST(ExactPhaseTransfer, TinyMomentumRemainsTransactionLocalUntilAggregationAndVelocityPublication)
{
    // Eight admitted even-block cells each have exact volume 2^-1074.
    // Per-cell rho=.5,u=.5 gives momentum eta/4 below binary64, while
    // the eight-cell aggregate 2*eta is representable.
    const double width = std::ldexp(1., -358);
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2 * width, 2 * width, 2 * width};
    CartesianAMR root(grid);
    const auto h = bootstrap(root, 8);
    std::vector<AdaptiveCellKey> keys;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : h->owned())
    {
        keys.push_back(key);
        gas.push_back(h->metrics(leaf).volume);
        carrier.push_back(0);
    }
    int local_cells = int(keys.size()), global_cells = 0;
    Teuchos::reduceAll(*h->communicator(), Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
    EXPECT_EQ(global_cells, 8);
    GeneralVoF initial(h, keys, gas, carrier);
    auto source = initial.export_transfer_source(false);
    std::vector<ExactTransferOwnedChannels> values;
    for (const auto& c : source.owned())
    {
        EXPECT_EQ(c.phases.gas, std::numeric_limits<double>::denorm_min());
        ExactTransferPhaseMass mass{M::from_value(c.phases.gas).scaled(.5), {}};
        const auto momentum = mass.gas.scaled(.5);
        EXPECT_GT(momentum.sign(), 0);
        EXPECT_EQ(static_cast<double>(momentum.enclosure().value), 0);
        values.push_back({c.key, {momentum}, mass});
    }
    source = source.with_channels({{"momentum", ExactTransferCarrier::MixtureMass, false}}, std::move(values));
    std::vector<int> destinations(
        keys.size(), (h->communicator()->getRank() + 1) % std::min(2, h->communicator()->getSize()));
    const auto plan = source.hierarchy()->plan_migration(destinations);
    const auto target = bind(*source.hierarchy(), plan);
    const auto result = ExactPhaseTransfer::prepare(source, plan, target);
    const auto adopted = initial.adopt_transfer(result);
    M local_momentum;
    for (const auto& c : result.owned())
    {
        ASSERT_TRUE(c.phase_mass);
        EXPECT_EQ(c.amounts[0].divided_by(c.phase_mass->gas).certified().as_double(), .5);
        local_momentum = local_momentum + c.amounts[0];
        EXPECT_GT(c.phases.gas, 0);
    }
    const auto total = global(local_momentum);
    if (h->communicator()->getRank() == 0)
    {
        EXPECT_GT(total.certified().as_double(), 0);
        EXPECT_EQ(total.compare(M::from_value(std::numeric_limits<double>::denorm_min()).scaled(2)), 0);
    }
    for (const auto& p : adopted->phase_volumes())
        EXPECT_GT(p.gas, 0);
    empty_owners(*target);
}
// Runtime opt-in only: these tests do not change the finite default policy.
TEST(ExactPhaseTransfer, ExperimentalExactAuthorityIsPinnedAcrossSmallHandoffsAndRejectsCrossPolicy)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, .7};
    CartesianAMR root(grid);
    const auto h = bootstrap(root, 8);
    const auto corner = [](const CartesianCellAncestry& c)
    {
        const auto x = c.index[0] >> c.level, y = c.index[1] >> c.level;
        return unsigned(y ? (x ? 2 : 3) : (x ? 1 : 0));
    };
    const auto expected = [&](const CartesianCellAncestry& c)
    {
        const auto xy = corner(c), z = unsigned(c.index[2] >> c.level);
        const auto old = M::from_value(std::ldexp(1e-200, int(xy + 4 * z)));
        const auto previous = M::from_value(std::ldexp(1e-200, int((xy + 3) % 4 + 4 * z)));
        return old + (previous - old).scaled(1. / 64).divided_by(M::from_value(.35));
    };
    std::vector<AdaptiveCellKey> keys;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : h->owned())
    {
        const auto& c = std::get<CartesianCellAncestry>(key.ancestry);
        keys.push_back(key);
        gas.push_back(std::ldexp(1e-200, int(corner(c) + 4 * c.index[2])));
        carrier.push_back(h->metrics(leaf).volume);
    }
    std::vector<GeneralVoFPhaseGeometry> pure(keys.size(), GeneralVoFPhaseGeometry::PureCarrier);
    GeneralVoF finite(h, keys, gas, carrier, {}, pure);
    EXPECT_EQ(finite.inventory_policy(), PhaseInventoryPolicy::Finite);
    EXPECT_THROW((void) finite.experimental_phases(), std::invalid_argument);
    GeneralVoFOptions options;
    options.inventory_policy = PhaseInventoryPolicy::ExperimentalExact;
    GeneralVoF exact(h, keys, gas, carrier, options, pure);
    std::vector<GeneralVoFNodeVelocity> nodes;
    for (const auto& n : exact.nodes())
        nodes.push_back({n.gid, {0, 0, 0}});
    std::vector<GeneralVoFFaceVolume> faces;
    for (const auto& face : exact.tiles())
    {
        const auto center = face.geometry->centroid(), area = face.geometry->area_vector();
        double phi = 0;
        if (!face.exterior && center.x == 1 && area.x != 0)
            phi = (center.y < 1 ? 1 : -1) * (area.x > 0 ? 1 : -1) / 64.;
        if (!face.exterior && center.y == 1 && area.y != 0)
            phi = (center.x > 1 ? 1 : -1) * (area.y > 0 ? 1 : -1) / 64.;
        faces.push_back({face.face_gid, phi});
    }
    const auto transported = exact.apply(exact.plan_transport(nodes, faces, 1));
    auto source = transported->export_transfer_source(false);
    EXPECT_EQ(source.inventory_policy(), PhaseInventoryPolicy::ExperimentalExact);
    int local_remainder = 0;
    for (const auto& c : source.owned())
    {
        EXPECT_EQ(c.policy, PhaseInventoryPolicy::ExperimentalExact);
        ASSERT_TRUE(c.experimental_exact);
        EXPECT_EQ(
            c.experimental_exact->phases.gas.compare(expected(std::get<CartesianCellAncestry>(c.key.ancestry))), 0);
        local_remainder += c.experimental_exact->phases.gas.compare(M::from_value(c.phases.gas)) != 0;
    }
    int remainder = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_remainder, &remainder);
    EXPECT_GT(remainder, 0);
    std::vector<int> destinations(keys.size(), (comm->getRank() + 1) % std::min(2, comm->getSize()));
    const auto migration = source.hierarchy()->plan_migration(destinations);
    const auto moved_h = bind(*source.hierarchy(), migration);
    const auto moved = ExactPhaseTransfer::prepare(source, migration, moved_h);
    EXPECT_EQ(moved.inventory_policy(), PhaseInventoryPolicy::ExperimentalExact);
    const auto state = transported->adopt_transfer(moved);
    empty_owners(*state->hierarchy());
    for (const auto& c : moved.owned())
    {
        ASSERT_TRUE(c.experimental_exact);
        EXPECT_EQ(
            c.experimental_exact->phases.gas.compare(expected(std::get<CartesianCellAncestry>(c.key.ancestry))), 0);
    }
    bool policy_rejected = false;
    try
    {
        (void) finite.adopt_transfer(moved);
    }
    catch (const std::invalid_argument& error)
    {
        policy_rejected = true;
        EXPECT_NE(std::string(error.what()).find("policy"), std::string::npos);
    }
    EXPECT_TRUE(policy_rejected);
    const auto finite_source = finite.export_transfer_source(false);
    EXPECT_EQ(finite_source.inventory_policy(), PhaseInventoryPolicy::Finite);
    for (const auto& c : finite_source.owned())
    {
        EXPECT_EQ(c.policy, PhaseInventoryPolicy::Finite);
        EXPECT_FALSE(c.experimental_exact);
    }
    std::vector<AdaptiveMark> marks;
    for (const auto& [key, leaf] : state->hierarchy()->owned())
        if (key == root.cell_key(0))
            marks.push_back({key, AdaptiveSplit::Isotropic});
    const auto plan = state->hierarchy()->plan_adaptation(marks);
    const auto target = bind(*state->hierarchy(), plan);
    const auto result = ExactPhaseTransfer::prepare(state->export_transfer_source(false), plan, target);
    for (const auto& c : result.owned())
    {
        ASSERT_TRUE(c.experimental_exact);
        const auto& ancestry = std::get<CartesianCellAncestry>(c.key.ancestry);
        const auto child = M::volume(*target->geometry(target->owned().at(c.key)));
        const auto wanted =
            expected(ancestry).divided_by(M::from_value(.35)).divided_by(M::from_value(1).divided_by(child));
        EXPECT_EQ(c.experimental_exact->phases.gas.compare(wanted), 0);
        EXPECT_EQ((c.experimental_exact->phases.gas + c.experimental_exact->phases.carrier - child)
                      .compare(c.experimental_exact->error.difference),
            0);
    }
    const auto refined = state->adopt_transfer(result);
    EXPECT_EQ(refined->inventory_policy(), PhaseInventoryPolicy::ExperimentalExact);
    std::vector<AdaptiveBytes> inverse;
    if (comm->getRank() == 0)
        inverse.push_back(adaptive_family_key(root.cell_key(0), AdaptiveSplit::Isotropic));
    const auto inverse_plan = refined->hierarchy()->plan_adaptation({}, inverse);
    const auto inverse_target = bind(*refined->hierarchy(), inverse_plan);
    const auto inverse_result =
        ExactPhaseTransfer::prepare(refined->export_transfer_source(false), inverse_plan, inverse_target);
    for (const auto& c : inverse_result.owned())
    {
        ASSERT_TRUE(c.experimental_exact);
        const auto& key = std::get<CartesianCellAncestry>(c.key.ancestry);
        EXPECT_EQ(c.experimental_exact->phases.gas.compare(expected(key)), 0);
    }
    EXPECT_EQ(refined->adopt_transfer(inverse_result)->inventory_policy(), PhaseInventoryPolicy::ExperimentalExact);
    EXPECT_THROW(
        exact.adopt_transfer(result), std::invalid_argument); // wrong accepted source/control despite same mode
    ExactPhaseTransferOptions exhausted;
    exhausted.byte_budget = 1;
    EXPECT_THROW(ExactPhaseTransfer::prepare(state->export_transfer_source(false), plan, target, exhausted),
        std::invalid_argument);
    result.validate_observations();
}

} // namespace
