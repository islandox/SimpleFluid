/** @file testGeneralVoF.cc
 * @brief Native finite-inventory transport, transient geometry proofs and rejection checks.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/ExactPhaseAccumulator.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "FVM/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/CompensatedSum.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::Meshes::AMR;
using namespace SimpleFluid::VoF;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
// Test-only transient lift of the published finite state, never an alternate
// accepted-state inventory or a preserved rational remainder.
std::vector<ExactPhaseMeasures> published_phase_lifts(const GeneralVoF& state)
{
    std::vector<ExactPhaseMeasures> result;
    for (const auto& value : state.phase_volumes())
        result.push_back({ExactGeometryMeasure::from_value(value.bulk), ExactGeometryMeasure::from_value(value.gas),
            ExactGeometryMeasure::from_value(value.carrier)});
    return result;
}
void expect_finite_phase_balance(const GeneralVoF& source, const GeneralVoF& target, const GeneralVoF::Trial& trial)
{
    std::array<::SimpleFluid::detail::CompensatedSum<>, 6> sums;
    for (std::size_t i = 0; i < source.phase_volumes().size(); ++i)
    {
        const auto& old = source.phase_volumes()[i];
        const auto& next = target.phase_volumes()[i];
        const auto& bound = trial.phase_roundoff_bounds()[i];
        sums[0] += static_cast<long double>(next.gas) - old.gas;
        sums[1] += static_cast<long double>(next.carrier) - old.carrier;
        sums[2] += bound.gas;
        sums[3] += bound.carrier;
        sums[4] += static_cast<long double>(std::abs(next.gas)) + std::abs(old.gas);
        sums[5] += static_cast<long double>(std::abs(next.carrier)) + std::abs(old.carrier);
    }
    std::array<double, 6> local{}, global{};
    for (std::size_t i = 0; i < local.size(); ++i)
        local[i] = static_cast<double>(sums[i].value());
    const auto comm = source.hierarchy()->communicator();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    // Conservative finite arithmetic envelope for the six local/wire sums;
    // each phase uses its own L1 inventory and actual publication errors.
    const double operations = 64 * double(source.hierarchy()->global_cells()) + 32 * comm->getSize() + 128;
    const double u = operations * std::numeric_limits<double>::epsilon();
    ASSERT_LT(u, .25);
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        const double tolerance = global[phase + 2] + u / (1 - u) * (global[phase + 4] + global[phase + 2]);
        EXPECT_LE(std::abs(global[phase]), tolerance);
    }
}
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
struct Fixture
{
    std::shared_ptr<const DistributedAdaptiveHierarchy> hierarchy;
    std::vector<AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
};
Fixture setup(bool trace = false)
{
    auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    auto source = std::make_shared<CartesianAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 8; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto boot = source->detach_owned(owned, {'g', 'e', 'n', 'e', 'r', 'a', 'l', '_', 'v', 'o', 'f'});
    Fixture f;
    f.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    for (const auto& [key, leaf] : f.hierarchy->owned())
    {
        f.order.push_back(key);
        auto c = f.hierarchy->geometry(leaf)->centroid();
        const double gas = trace ? 1e-310 : ((c[0] < 1 && c[1] < 1 && c[2] < 1) ? 1 : 0);
        f.gas.push_back(gas);
        f.carrier.push_back(trace ? 1 : 1 - gas);
    }
    return f;
}
std::vector<GeneralVoFNodeVelocity> nodes(const GeneralVoF& state)
{
    std::vector<GeneralVoFNodeVelocity> result;
    for (const auto& node : state.nodes())
        result.push_back({node.gid, {0, 0, 0}});
    return result;
}
std::vector<GeneralVoFFaceVolume> cycle(const GeneralVoF& state)
{
    std::vector<GeneralVoFFaceVolume> result;
    for (const auto& face : state.tiles())
    {
        const auto p = face.geometry->centroid(), area = face.geometry->area_vector();
        double phi = 0;
        if (!face.exterior && p.x == 1 && area.x != 0)
            phi = (p.y < 1 ? 1 : -1) * (area.x > 0 ? 1 : -1) / 64.;
        if (!face.exterior && p.y == 1 && area.y != 0)
            phi = (p.x > 1 ? 1 : -1) * (area.y > 0 ? 1 : -1) / 64.;
        result.push_back({face.face_gid, phi});
    }
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
TEST(GeneralVoF, ComplementaryDonorLabelCorrectionIsLinearBoundedAndRejectsUnprovedDefect)
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
TEST(GeneralVoF, ActualUnsplitCirculationHasPositiveRoutesIndependentComplementsAndEmptyOwners)
{
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier);
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
Fixture skew_setup(unsigned ownership_shift = 0)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<AdaptiveLeaf> owned;
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned j = 0; j < 2; ++j)
            for (unsigned i = 0; i < 2; ++i)
            {
                const auto id = i + 2 * j + 4 * k;
                if (int((id + ownership_shift) % std::min(2, comm->getSize())) != comm->getRank())
                    continue;
                AdaptiveLeaf leaf;
                leaf.key = {7, ConvexCellAncestry{id, {}}};
                leaf.domain = {'s', 'k', 'e', 'w'};
                std::vector<GeometryPoint> points;
                for (unsigned z = 0; z < 2; ++z)
                    for (unsigned y = 0; y < 2; ++y)
                        for (unsigned x = 0; x < 2; ++x)
                        {
                            const double u = i + x, v = j + y, w = k + z;
                            points.push_back({u + .25 * v - w, v, u + .25 * v + w});
                        }
                leaf.geometry = ConvexCellGeometry::make(
                    points, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}});
                for (unsigned f = 0; f < 6; ++f)
                {
                    const unsigned axis = f / 2, coordinate = std::array<unsigned, 3>{i, j, k}[axis] + f % 2;
                    AdaptiveSurface side;
                    side.key = {'s', char(axis), char(coordinate)};
                    side.geometry = leaf.geometry->face_geometry(f);
                    side.orientation = f % 2 ? 1 : -1;
                    side.exterior = coordinate == 0 || coordinate == 2;
                    side.boundary = side.exterior ? "wall" : "";
                    leaf.surfaces.push_back(std::move(side));
                }
                owned.push_back(std::move(leaf));
            }
    Fixture fixture;
    fixture.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(owned), std::vector<AdaptiveFamily>{}, convex_adaptive_kernel(2, 0), comm);
    for (const auto& [key, leaf] : fixture.hierarchy->owned())
    {
        fixture.order.push_back(key);
        const auto root = std::get<ConvexCellAncestry>(key.ancestry).root;
        fixture.gas.push_back(root < 4 ? 1 : 0);
        fixture.carrier.push_back(root < 4 ? 1 : 2);
    }
    return fixture;
}
TEST(GeneralVoF, ExactSkewWallAndEdgeTrajectoriesAreOwnershipIndependentAndDoNotLeak)
{
    const auto comm = Tpetra::getDefaultComm();
    for (unsigned shift = 0; shift < 2; ++shift)
    {
        auto fixture = skew_setup(shift);
        GeneralVoF state(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier);
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
    for (const auto& stream : trial.face_contributions())
    {
        EXPECT_EQ((stream.exact.gas + stream.exact.carrier).compare(stream.exact.bulk), 0);
        if (stream.exact.gas.sign() > 0)
            EXPECT_NEAR(
                stream.exact.gas.certified().as_double(), .5 * stream.exact.bulk.certified().as_double(), 3e-13);
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
    for (const auto& transfer : trial.transfers())
    {
        const auto donor =
            std::find_if(support.begin(), support.end(), [&](const auto& cell) { return cell.cell == transfer.donor; });
        const auto recipient = std::find_if(
            support.begin(), support.end(), [&](const auto& cell) { return cell.cell == transfer.recipient; });
        if (donor != support.end() && recipient != support.end() &&
            donor->geometry->centroid() == GeometryPoint{.5, .5, .5} &&
            recipient->geometry->centroid() == GeometryPoint{1.5, 1.5, 1.5})
        {
            EXPECT_GT(transfer.exact.bulk.sign(), 0);
            EXPECT_EQ(transfer.exact.gas.compare(transfer.exact.bulk), 0);
            ++local_corner;
        }
    }
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_corner, &global_corner);
    EXPECT_EQ(global_corner, 1);
    for (const auto& budget : trial.donor_budgets())
    {
        EXPECT_EQ((budget.exact_exported.gas + budget.exact_retained.gas).compare(budget.exact_old.gas), 0);
        EXPECT_EQ((budget.exact_exported.carrier + budget.exact_retained.carrier).compare(budget.exact_old.carrier), 0);
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
    for (const auto& transfer : trial.transfers())
    {
        if (transfer.donor == transfer.recipient || transfer.exact.bulk.sign() <= 0)
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
        EXPECT_EQ(transfer.exact.gas.sign(), 0);
        EXPECT_EQ(transfer.exact.carrier.compare(transfer.exact.bulk), 0);
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_distant, &distant);
    EXPECT_GT(distant, 0);
    for (const auto& budget : trial.donor_budgets())
        EXPECT_EQ((budget.exact_exported.carrier + budget.exact_retained.carrier).compare(budget.exact_old.carrier), 0);
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
            ExactGeometryMeasure bulk;
            double gas = 0, carrier = 0;
            bool positive = false, negative = false;
            for (const auto& stream : trial.face_contributions())
                if (stream.face == tile.face_gid)
                {
                    bulk = bulk + stream.exact.bulk.scaled(stream.direction);
                    gas += stream.direction * stream.exact.gas.certified().as_double();
                    carrier += stream.direction * stream.exact.carrier.certified().as_double();
                    positive |= stream.direction > 0;
                    negative |= stream.direction < 0;
                }
            EXPECT_EQ(bulk.sign(), 0);
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
TEST(GeneralVoF, UnderflowingIndividualPhaseStreamsRemainExactWithRepresentableFinalInventories)
{
    auto fixture = setup();
    for (std::size_t i = 0; i < fixture.order.size(); ++i)
    {
        const auto center = fixture.hierarchy->geometry(fixture.hierarchy->owned().at(fixture.order[i]))->centroid();
        fixture.gas[i] = center[2] < 1 ? .5 : 0;
        fixture.carrier[i] = 1 - fixture.gas[i];
    }
    GeneralVoF source(fixture.hierarchy, fixture.order, fixture.gas, fixture.carrier);
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
    // A subquantum arrival is now rounded once, with measured arithmetic
    // uncertainty, rather than retained as an accepted rational inventory.
    auto pure_fixture = setup();
    GeneralVoF pure(pure_fixture.hierarchy, pure_fixture.order, pure_fixture.gas, pure_fixture.carrier);
    auto tiny_phi = cycle(pure);
    for (auto& face : tiny_phi)
        if (face.integrated_volume != 0)
        {
            face.exact_integrated_volume =
                ExactGeometryMeasure::from_value(tiny).scaled(face.integrated_volume > 0 ? .5 : -.5);
            face.integrated_volume = 0;
        }
    const auto tiny_trial = pure.plan_transport(nodes(pure), tiny_phi, 1);
    const auto finite = pure.apply(tiny_trial);
    const auto delivered = delivered_phases(tiny_trial);
    int local_unrepresented = 0;
    for (std::size_t i = 0; i < finite->owned_order().size(); ++i)
    {
        const auto gid = std::uint64_t(finite->mesh()->cell_global_id(int(i)));
        const auto& raw = delivered.at(gid);
        const auto& phase = finite->phase_volumes()[i];
        const auto& bound = tiny_trial.phase_roundoff_bounds()[i];
        for (const auto& entry :
            {std::tuple{raw.gas, phase.gas, bound.gas}, std::tuple{raw.carrier, phase.carrier, bound.carrier}})
        {
            const auto& exact = std::get<0>(entry);
            const double value = std::get<1>(entry), error = std::get<2>(entry);
            const auto difference = exact - ExactGeometryMeasure::from_value(value);
            EXPECT_LE((difference.sign() < 0 ? difference.scaled(-1) : difference)
                          .compare(ExactGeometryMeasure::from_value(error)),
                0);
            if (exact.compare(ExactGeometryMeasure::from_value(tiny)) >= 0)
                EXPECT_GT(value, 0);
            if (exact.sign() > 0 && value == 0)
            {
                ++local_unrepresented;
                EXPECT_GT(error, 0);
            }
        }
    }
    int unrepresented = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_unrepresented, &unrepresented);
    EXPECT_GT(unrepresented, 0);
    expect_finite_phase_balance(pure, *finite, tiny_trial);
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
    EXPECT_EQ(finite->gas_volumes(), pure.apply(tiny_trial)->gas_volumes());
    EXPECT_NO_THROW(pure.plan_transport(nodes(pure), cycle(pure), 1));
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
        EXPECT_TRUE(p.target_volume_difference.exact_nonzero);
        EXPECT_LT(p.target_volume_difference.value, 0);
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
    for (const auto& budget : transported.donor_budgets())
    {
        EXPECT_LE(budget.signed_repair_absolute.compare(budget.signed_repair_bound), 0);
        EXPECT_LE(budget.route_repair_absolute.compare(budget.route_repair_bound), 0);
        EXPECT_EQ(budget.route_repair_absolute.compare(ExactGeometryMeasure::from_value(1e-310)), 0);
        EXPECT_EQ((budget.exact_exported.gas + budget.exact_retained.gas).compare(budget.exact_old.gas), 0);
        EXPECT_EQ((budget.exact_exported.carrier + budget.exact_retained.carrier).compare(budget.exact_old.carrier), 0);
    }
    GeneralVoF unresolved(f.hierarchy, f.order, f.gas, f.carrier);
    EXPECT_THROW(unresolved.reconstruct(), std::invalid_argument);
    EXPECT_NO_THROW(unresolved.plan_transport(nodes(unresolved), zero, 1));
}
TEST(GeneralVoF, DirectAdoptionWitnessRejectsLookAlikesAndDoesNotRetainOldPublication)
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
TEST(GeneralVoF, ExactZeroBootstrapRetainsPureEvidenceWithoutClassifyingNonzeroTails)
{
    auto f = setup();
    GeneralVoF state(f.hierarchy, f.order, f.gas, f.carrier);
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
        // Neither pure-geometry retention nor exact physical transport refreshes
        // the original occupancy/fit allowance or erases the independent tail.
        const auto& before = state.initial_occupancy_errors()[i];
        const auto& after = moved->initial_occupancy_errors()[i];
        EXPECT_LE(std::abs(after.difference - before.difference),
            moved->phase_roundoff_bounds()[i].gas + moved->phase_roundoff_bounds()[i].carrier);
        EXPECT_GE(after.admitted_bound, before.admitted_bound);
        EXPECT_EQ(after.fit_bound, before.fit_bound);
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
    GeneralVoF explicit_infer(f.hierarchy, f.order, f.gas, f.carrier, {}, infer);
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

} // namespace

namespace
{
TEST(GeneralVoF, DonorLiveLeasesReleaseReplacedValuesWithoutLosingExactColumns)
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

} // namespace

namespace
{
TEST(GeneralVoF, DonorLinearRawGroupsMatchPerPieceHistoryAndRepairExactly)
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
TEST(GeneralVoF, DonorRawGroupingPreservesMappedZeroForSingularAndIdentityMaps)
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
    ASSERT_EQ(next.donor_budgets().size(), moved->phase_volumes().size());
    for (std::size_t i = 0; i < moved->phase_volumes().size(); ++i)
    {
        const auto& phase = moved->phase_volumes()[i];
        const auto& old = next.donor_budgets()[i].exact_old;
        EXPECT_EQ(old.gas.encode(), ExactGeometryMeasure::from_value(phase.gas).encode());
        EXPECT_EQ(old.carrier.encode(), ExactGeometryMeasure::from_value(phase.carrier).encode());
        EXPECT_EQ(next.phase_roundoff_bounds()[i].gas, 0);
        EXPECT_EQ(next.phase_roundoff_bounds()[i].carrier, 0);
    }
    EXPECT_EQ(state.gas_volumes(), f.gas); // accepted source was never changed
}
TEST(GeneralVoF, FinitePublicationVerifiesOutwardBoundsAndKeepsRepresentableMinorities)
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

namespace
{
TEST(GeneralVoF, ExperimentalInventoryPolicyIsExplicitCollectiveAndIsolatedFromFinite)
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
TEST(GeneralVoF, ExperimentalNonDyadicHistorySurvivesOnlyAfterExplicitOptIn)
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
