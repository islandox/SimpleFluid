/** @file testExactFaceTransfer.cc
 * @brief Exact geometric face-rate transfer and native transaction rejection checks.
 */
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::Meshes::AMR;
using namespace SimpleFluid::VoF;
using M = ExactGeometryMeasure;
using H = DistributedAdaptiveHierarchy;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const GeneralVoF> setup(Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::null, bool cubic = false)
{
    ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x)
            grid.xy_nodes.push_back({double(x), double(y), 0});
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
        {
            const unsigned n = x + 4 * y;
            grid.base_cells.push_back({n, n + 1, n + 5, n + 4});
        }
    grid.z_edges = cubic ? std::vector<double>{0, 1, 2, 3} : std::vector<double>{0, 1};
    auto root = std::make_shared<const ExtrudedAMR>(grid);
    auto fine = cubic ? root : root->adapt({}, {0});
    if (comm.is_null())
        comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t i = 0; i < (cubic ? 27 : 18); ++i)
        if (int(i % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(i);
    auto boot = fine->detach_owned(owned, {'f', 'a', 'c', 'e', '_', 't', 'r', 'a', 'n', 's', 'f', 'e', 'r'});
    auto hierarchy = std::make_shared<const H>(std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    std::vector<AdaptiveCellKey> keys;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        keys.push_back(key);
        gas.push_back(0);
        carrier.push_back(cubic ? 1 : .5);
    }
    return std::make_shared<const GeneralVoF>(hierarchy, keys, gas, carrier);
}
std::shared_ptr<const H> bind(const H& source, const H::Plan& plan)
{
    auto raw = source.apply(plan);
    auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    return source.apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
}
M cycle(const AdaptiveTile& tile)
{
    if (tile.exterior)
        return {};
    const auto p = tile.geometry->centroid();
    const auto a = tile.geometry->area_vector();
    if (a.x != 0 && (p.y < 1 || p.y > 2))
        return tile.geometry->area_vector_dot({p.y < 1 ? 1. : -1., 0, 0});
    if (a.y != 0 && (p.x < 1 || p.x > 2))
        return tile.geometry->area_vector_dot({0, p.x > 2 ? 1. : -1., 0});
    return {};
}
std::vector<ExactOwnedFaceRate> rates(const GeneralVoF& state)
{
    std::map<std::uint64_t, const AdaptiveTile*> tiles;
    for (const auto& tile : state.tiles())
        tiles.emplace(tile.face_gid, &tile);
    std::vector<ExactOwnedFaceRate> result;
    for (std::size_t i = 0; i < state.mesh()->num_owned_faces(); ++i)
    {
        const auto id = std::uint64_t(state.mesh()->face_global_id(i));
        result.push_back({id, cycle(*tiles.at(id))});
    }
    return result;
}
TEST(ExactFaceTransfer, IndependentAcceptedRatesMigrateByActualFaceIdentityWithoutVelocitySeed)
{
    const auto source = setup();
    auto old = rates(*source);
    std::vector<int> destinations(source->hierarchy()->owned().size(),
        (Tpetra::getDefaultComm()->getRank() + 1) % std::min(2, Tpetra::getDefaultComm()->getSize()));
    const auto plan = source->hierarchy()->plan_migration(destinations);
    const auto h = bind(*source->hierarchy(), plan);
    const auto phase = ExactPhaseTransfer::prepare(source->export_transfer_source(false), plan, h);
    const auto target = source->adopt_transfer(phase);
    const auto result = ExactFaceTransfer::prepare(*source, *target, phase, old);
    std::map<std::uint64_t, const AdaptiveTile*> actual;
    for (const auto& tile : target->tiles())
        actual.emplace(tile.face_gid, &tile);
    for (const auto& item : result.owned_rates())
        EXPECT_EQ(item.rate.compare(cycle(*actual.at(item.face))), 0);
    int localempty = target->hierarchy()->owned().empty(), empty = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &localempty, &empty);
    if (Tpetra::getDefaultComm()->getSize() == 4)
        EXPECT_GT(empty, 0);
    auto missing = old;
    if (!missing.empty())
        missing.pop_back();
    EXPECT_THROW((void) ExactFaceTransfer::prepare(*source, *target, phase, missing), std::invalid_argument);
    auto duplicate = old;
    if (duplicate.size() > 1)
        duplicate[1] = duplicate[0];
    EXPECT_THROW((void) ExactFaceTransfer::prepare(*source, *target, phase, duplicate), std::invalid_argument);
    ExactFaceTransferOptions limits;
    limits.byte_budget = 1;
    EXPECT_THROW((void) ExactFaceTransfer::prepare(*source, *target, phase, old, {}, limits), std::invalid_argument);
    auto wall = old, divergent = old;
    bool changed_wall = false, changed_internal = false;
    for (auto& item : wall)
        if (!changed_wall)
        {
            const auto local = source->mesh()->owned_face_map()->getLocalElement(item.face);
            if (source->mesh()->is_geometry_exterior_face(local))
            {
                item.rate = M::from_value(1);
                changed_wall = true;
            }
        }
    for (auto& item : divergent)
    {
        const auto local = source->mesh()->owned_face_map()->getLocalElement(item.face);
        if (!changed_internal && !source->mesh()->is_geometry_exterior_face(local))
        {
            item.rate = item.rate + M::from_value(1);
            changed_internal = true;
        }
    }
    EXPECT_THROW((void) ExactFaceTransfer::prepare(*source, *target, phase, wall), std::invalid_argument);
    EXPECT_THROW((void) ExactFaceTransfer::prepare(*source, *target, phase, divergent), std::invalid_argument);
    ExactFaceTransferOptions work_limit;
    work_limit.maximum_pair_tests = 1;
    EXPECT_THROW(
        (void) ExactFaceTransfer::prepare(*source, *target, phase, old, {}, work_limit), std::invalid_argument);
    const auto recovered = ExactFaceTransfer::prepare(*source, *target, phase, old);
    EXPECT_EQ(recovered.owned_rates().size(), result.owned_rates().size());
    std::vector<AdaptiveCellKey> needed;
    for (const auto& [key, leaf] : source->hierarchy()->owned())
        needed.push_back(key);
    EXPECT_THROW((void) source->hierarchy()->lookup_cells(needed, true, 1), std::exception);
    EXPECT_THROW((void) source->hierarchy()->fetch_leaves(needed, 1), std::exception);
    const auto cap = std::min<std::size_t>(1024 * 1024, source->hierarchy()->options().byte_budget);
    EXPECT_EQ(source->hierarchy()->lookup_cells(needed, true, cap).size(), needed.size());
    EXPECT_EQ(source->hierarchy()->fetch_leaves(needed, cap).size(), needed.size());
    const auto local_cap =
        Tpetra::getDefaultComm()->getRank() + 1 == Tpetra::getDefaultComm()->getSize() ? std::size_t(1) : cap;
    EXPECT_THROW((void) source->hierarchy()->lookup_cells(needed, true, local_cap), std::exception);
    EXPECT_THROW((void) source->hierarchy()->fetch_leaves(needed, local_cap), std::exception);
    EXPECT_EQ(source->hierarchy()->lookup_cells(needed, true, cap).size(), needed.size());
    EXPECT_EQ(source->hierarchy()->fetch_leaves(needed, cap).size(), needed.size());
    EXPECT_THROW((void) source->hierarchy()->lookup_cells(needed, true, source->hierarchy()->options().byte_budget + 1),
        std::invalid_argument);
    GeneralVoF foreign(source->hierarchy(), source->owned_order(), source->gas_volumes(), source->carrier_volumes());
    EXPECT_THROW((void) ExactFaceTransfer::prepare(foreign, *target, phase, old), std::invalid_argument);
    const auto serial_target = setup(Teuchos::rcp(new Teuchos::SerialComm<int>()));
    const auto& mismatched = Tpetra::getDefaultComm()->getRank() == 0 ? *serial_target : *target;
    EXPECT_THROW((void) ExactFaceTransfer::prepare(*source, mismatched, phase, old), std::invalid_argument);
    result.validate_observations(*source, *target);
}
TEST(ExactFaceTransfer, CrossedBaseRefinementAxialCoarseningPreservesConstantStraightLegAndClosesAllRows)
{
    const auto source = setup();
    const auto old = rates(*source);
    std::vector<AdaptiveBytes> coarsen;
    std::vector<ExactOwnedCellVelocity> seed;
    for (const auto& [key, leaf] : source->hierarchy()->owned())
    {
        const auto& p = std::get<ProductCellAncestry>(key.ancestry);
        if (p.axial.path.back() == 0)
        {
            auto parent = key;
            std::get<ProductCellAncestry>(parent.ancestry).axial.path.pop_back();
            coarsen.push_back(adaptive_family_key(parent, AdaptiveSplit::Axial));
        }
        seed.push_back({key, p.base.root == 1 ? GeometryPoint{1, 0, 0} : GeometryPoint{0, 0, 0}});
    }
    const auto coarsening = source->hierarchy()->plan_adaptation({}, coarsen);
    const auto intermediate = bind(*source->hierarchy(), coarsening);
    std::vector<AdaptiveMark> marks;
    for (const auto& [key, leaf] : intermediate->owned())
        if (std::get<ProductCellAncestry>(key.ancestry).base.root == 1)
            marks.push_back({key, AdaptiveSplit::Base});
    const auto refinement = intermediate->plan_adaptation(marks);
    const auto h = bind(*intermediate, refinement);
    const std::array<ExactTransferPlanStep, 2> chain{{{coarsening, intermediate}, {refinement, h}}};
    const auto original_source = source->export_transfer_source(false);
    const auto phase = ExactPhaseTransfer::prepare(original_source, chain);
    const auto target = source->adopt_transfer(phase);
    int crossed_local = 0, crossed = 0;
    for (const auto& [key, leaf] : target->hierarchy()->owned())
    {
        const auto& p = std::get<ProductCellAncestry>(key.ancestry);
        if (p.base.root == 1)
        {
            EXPECT_EQ(p.base.path.size(), 1u);
            EXPECT_TRUE(p.axial.path.empty());
            ++crossed_local;
        }
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &crossed_local, &crossed);
    EXPECT_EQ(crossed, 4);
    auto reordered = chain;
    std::swap(reordered[0], reordered[1]);
    EXPECT_THROW((void) ExactPhaseTransfer::prepare(original_source, reordered), std::invalid_argument);
    // The second plan pins this EXACT intermediate native map. An equivalent
    // rebuilt binding must not be substituted after it has been prepared.
    const auto alternative_native = assemble_adaptive_native(*intermediate, intermediate->configuration_descriptor());
    const auto rebound =
        intermediate->with_partition(alternative_native.partition, alternative_native.mesh->owned_cell_map());
    auto stale = chain;
    stale[0].candidate = rebound;
    EXPECT_THROW((void) ExactPhaseTransfer::prepare(original_source, stale), std::invalid_argument);
    const auto foreign_source = setup();
    const auto foreign_plan = foreign_source->hierarchy()->plan_migration(
        std::vector<int>(foreign_source->hierarchy()->owned().size(), Tpetra::getDefaultComm()->getRank()));
    auto foreign = chain;
    foreign[0] = {foreign_plan, bind(*foreign_source->hierarchy(), foreign_plan)};
    EXPECT_THROW((void) ExactPhaseTransfer::prepare(original_source, foreign), std::invalid_argument);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        const std::span<const ExactTransferPlanStep> different(
            chain.data(), Tpetra::getDefaultComm()->getRank() == 0 ? 1 : 2);
        EXPECT_THROW((void) ExactPhaseTransfer::prepare(original_source, different), std::invalid_argument);
    }
    phase.validate_observations();
    const auto result = ExactFaceTransfer::prepare(*source, *target, phase, old, seed);
    std::map<std::uint64_t, const AdaptiveTile*> actual;
    for (const auto& tile : target->tiles())
        actual.emplace(tile.face_gid, &tile);
    int local = 0, total = 0;
    for (const auto& item : result.owned_rates())
    {
        const auto& tile = *actual.at(item.face);
        const auto p = tile.geometry->centroid();
        if (p.x == 1.5 && p.y < 1 && tile.geometry->area_vector().x != 0)
        {
            EXPECT_EQ(item.rate.compare(tile.geometry->area_vector_dot({1, 0, 0})), 0);
            ++local;
        }
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &total);
    EXPECT_GT(total, 0);
    result.validate_observations(*source, *target);
    std::vector<AdaptiveBytes> reverse_coarsen;
    std::vector<ExactOwnedCellVelocity> reverse_seed;
    for (const auto& [key, leaf] : target->hierarchy()->owned())
    {
        const auto& p = std::get<ProductCellAncestry>(key.ancestry);
        if (!p.base.path.empty() && p.base.path.back() == 0)
        {
            auto parent = key;
            std::get<ProductCellAncestry>(parent.ancestry).base.path.pop_back();
            reverse_coarsen.push_back(adaptive_family_key(parent, AdaptiveSplit::Base));
        }
        reverse_seed.push_back({key, p.base.root == 1 ? GeometryPoint{1, 0, 0} : GeometryPoint{0, 0, 0}});
    }
    const auto inverse_base = target->hierarchy()->plan_adaptation({}, reverse_coarsen);
    const auto inverse_mid = bind(*target->hierarchy(), inverse_base);
    std::vector<AdaptiveMark> reverse_marks;
    for (const auto& [key, leaf] : inverse_mid->owned())
        reverse_marks.push_back({key, AdaptiveSplit::Axial});
    const auto inverse_z = inverse_mid->plan_adaptation(reverse_marks);
    const auto inverse_h = bind(*inverse_mid, inverse_z);
    const std::array<ExactTransferPlanStep, 2> inverse_chain{{{inverse_base, inverse_mid}, {inverse_z, inverse_h}}};
    const auto inverse_phase = ExactPhaseTransfer::prepare(target->export_transfer_source(false), inverse_chain);
    const auto restored = target->adopt_transfer(inverse_phase);
    const auto inverse =
        ExactFaceTransfer::prepare(*target, *restored, inverse_phase, result.owned_rates(), reverse_seed);
    std::map<std::uint64_t, const AdaptiveTile*> restored_tiles;
    for (const auto& tile : restored->tiles())
        restored_tiles.emplace(tile.face_gid, &tile);
    for (const auto& item : inverse.owned_rates())
        EXPECT_EQ(item.rate.compare(cycle(*restored_tiles.at(item.face))), 0);
    inverse.validate_observations(*target, *restored);
}
} // namespace

TEST(ExactFaceTransfer, TwentySevenCellCentralRegridUsesPhaseLocalDecodeLease)
{
    const auto source = setup(Teuchos::null, true);
    std::vector<AdaptiveMark> marks;
    std::vector<ExactOwnedCellVelocity> seeds;
    for (const auto& [key, leaf] : source->hierarchy()->owned())
    {
        const auto center = source->hierarchy()->geometry(leaf)->centroid();
        if (center[0] == 1.5 && center[1] == 1.5 && center[2] == 1.5)
            marks.push_back({key, AdaptiveSplit::Isotropic});
        seeds.push_back({key, {0, 0, 0}});
    }
    const auto plan = source->hierarchy()->plan_adaptation(marks);
    const auto h = bind(*source->hierarchy(), plan);
    const auto phase = ExactPhaseTransfer::prepare(source->export_transfer_source(false), plan, h);
    const auto target = source->adopt_transfer(phase);
    EXPECT_EQ(source->hierarchy()->global_cells(), 27);
    EXPECT_GT(target->hierarchy()->global_cells(), source->hierarchy()->global_cells());
    auto old = rates(*source);
    for (auto& rate : old)
        rate.rate = {};
    ExactFaceTransferOptions options;
    options.byte_budget = 32 * 1024 * 1024;
    const auto result = ExactFaceTransfer::prepare(*source, *target, phase, old, seeds, options);
    for (const auto& rate : result.owned_rates())
        EXPECT_EQ(rate.rate.sign(), 0);
    EXPECT_EQ(result.owned_rates().size(), target->mesh()->num_owned_faces());
    result.validate_observations(*source, *target);
}
