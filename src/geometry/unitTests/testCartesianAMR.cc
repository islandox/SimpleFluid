/**
 * @file testCartesianAMR.cc
 * @brief Dyadic hierarchy, conservative native seams, and collective map checks.
 */
#include "geometry/CartesianAMR.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "parallel/MeshPartitioner.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Import.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Handle = MeshHandle<>;
using ID = std::uint64_t;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

double volume(const AMRBox& box)
{
    return (box.upper[0] - box.lower[0]) * (box.upper[1] - box.lower[1]) * (box.upper[2] - box.lower[2]);
}

std::size_t leaf_ordinal(const CartesianAMR& hierarchy, const AMRBlock& key)
{
    const auto found = std::find(hierarchy.blocks().begin(), hierarchy.blocks().end(), key);
    if (found == hierarchy.blocks().end())
        throw std::logic_error("AMR test expected a missing leaf.");
    return static_cast<std::size_t>(found - hierarchy.blocks().begin());
}

void check_native_closure(const MultiRegionMesh& mesh)
{
    std::vector<unsigned> incidence(mesh.num_faces(), 0);
    for (ID cell = 0; cell < mesh.num_cells(); ++cell)
    {
        MeshUtils::Vec3 closure{};
        for (const auto face : mesh.cell_faces(cell))
        {
            ++incidence[face];
            closure = closure + mesh.face_area_vector_outward(face, cell);
        }
        EXPECT_NEAR(closure.norm(), 0.0, 2.0e-13);
    }
    for (ID face = 0; face < mesh.num_faces(); ++face)
        EXPECT_EQ(incidence[face], mesh.is_exterior_face(face) ? 1U : 2U);
}

void check_physical_boundaries(const CartesianAMR& hierarchy)
{
    const auto& mesh = *hierarchy.geometry();
    const auto& config = hierarchy.config();
    std::array<double, 6> areas{};
    for (ID face = 0; face < mesh.num_faces(); ++face)
    {
        if (!mesh.is_exterior_face(face))
            continue;
        const auto normal = mesh.face_normal(face);
        const auto center = mesh.face_centroid(face);
        std::size_t axis = 0;
        while (axis < 2 && std::abs(normal.component(axis)) < 0.5)
            ++axis;
        const bool upper = normal.component(axis) > 0.0;
        const double boundary = config.origin[axis] + (upper ? config.extent[axis] : 0.0);
        EXPECT_FALSE(config.periodic[axis]);
        EXPECT_NEAR(center.component(axis), boundary, 2.0e-13);
        EXPECT_NEAR(normal.component(axis), upper ? 1.0 : -1.0, 2.0e-14);
        for (std::size_t other = 0; other < 3; ++other)
            if (other != axis)
                EXPECT_NEAR(normal.component(other), 0.0, 2.0e-14);
        areas[2 * axis + upper] += mesh.face_area(face);
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double expected =
            config.periodic[axis] ? 0.0 : config.extent[(axis + 1) % 3] * config.extent[(axis + 2) % 3];
        EXPECT_NEAR(areas[2 * axis], expected, 2.0e-13);
        EXPECT_NEAR(areas[2 * axis + 1], expected, 2.0e-13);
    }
}

void check_overlap_lookup(const CartesianAMR& hierarchy, const AMRBox& query)
{
    const auto actual = hierarchy.overlaps(query);
    std::vector<AMROverlap> reference;
    for (ID cell = 0; cell < hierarchy.geometry()->num_cells(); ++cell)
    {
        auto box = hierarchy.cell_box(cell);
        bool positive = true;
        for (std::size_t a = 0; a < 3; ++a)
        {
            box.lower[a] = std::max(box.lower[a], query.lower[a]);
            box.upper[a] = std::min(box.upper[a], query.upper[a]);
            positive = positive && box.upper[a] > box.lower[a];
        }
        if (positive)
            reference.push_back({cell, box});
    }
    ASSERT_EQ(actual.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
    {
        EXPECT_EQ(actual[i].cell, reference[i].cell);
        EXPECT_EQ(actual[i].box.lower, reference[i].box.lower);
        EXPECT_EQ(actual[i].box.upper, reference[i].box.upper);
    }
}

TEST(CartesianAMRTest, RootCellKeysAndOverlapBoxesAgreeWithNativeGeometry)
{
    AMRGrid config;
    config.root_blocks = {2, 1, 1};
    config.block_cells = {4, 2, 2};
    config.origin = {-1.0, 0.25, 2.0};
    config.extent = {2.0, 1.0, 0.5};
    const CartesianAMR hierarchy(config);
    ASSERT_EQ(hierarchy.blocks().size(), 2U);
    EXPECT_EQ(hierarchy.geometry()->num_cells(), 32U);
    EXPECT_EQ(hierarchy.topology_generation(), 0U);
    double total = 0.0;
    for (ID cell = 0; cell < hierarchy.geometry()->num_cells(); ++cell)
    {
        const auto box = hierarchy.cell_box(cell);
        const auto center = hierarchy.geometry()->cell_centroid(cell);
        for (std::size_t a = 0; a < 3; ++a)
            EXPECT_DOUBLE_EQ(center.component(a), 0.5 * (box.lower[a] + box.upper[a]));
        EXPECT_NEAR(hierarchy.geometry()->cell_volume(cell), volume(box), 1.0e-15);
        EXPECT_EQ(hierarchy.cell_block(cell), static_cast<std::size_t>(cell / 16));
        total += volume(box);
    }
    EXPECT_DOUBLE_EQ(total, 1.0);
    const AMRBox query{{-0.1, 0.4, 2.1}, {0.65, 0.9, 2.4}};
    const auto overlaps = hierarchy.overlaps(query);
    ASSERT_FALSE(overlaps.empty());
    total = 0.0;
    std::set<ID> seen;
    for (const auto& overlap : overlaps)
    {
        EXPECT_TRUE(seen.insert(overlap.cell).second);
        total += volume(overlap.box);
        for (std::size_t a = 0; a < 3; ++a)
        {
            EXPECT_GE(overlap.box.lower[a], query.lower[a]);
            EXPECT_LE(overlap.box.upper[a], query.upper[a]);
        }
    }
    EXPECT_NEAR(total, volume(query), 2.0e-15);
    auto domain = AMRBox{config.origin, {1.0, 1.25, 2.5}};
    for (std::size_t a = 0; a < 3; ++a)
    {
        domain.lower[a] = std::nextafter(domain.lower[a], -std::numeric_limits<double>::infinity());
        domain.upper[a] = std::nextafter(domain.upper[a], std::numeric_limits<double>::infinity());
    }
    EXPECT_EQ(hierarchy.overlaps(domain).size(), 32U);
    domain.lower[0] -= 1.0e-5;
    EXPECT_THROW((void) hierarchy.overlaps(domain), std::invalid_argument);
    EXPECT_THROW((void) hierarchy.cell_box(32), std::out_of_range);
    EXPECT_TRUE(hierarchy.overlaps({query.lower, query.lower}).empty());
}

TEST(CartesianAMRTest, RefineCoarsenAndPartialSubfacesPreserveCanonicalArea)
{
    AMRGrid config;
    config.root_blocks = {2, 1, 1};
    const CartesianAMR original(config);
    const auto refined = original.adapt({0});
    ASSERT_EQ(refined->blocks().size(), 9U);
    EXPECT_EQ(original.geometry()->num_cells(), 128U);
    EXPECT_EQ(refined->geometry()->num_cells(), 576U);
    EXPECT_EQ(refined->topology_generation(), 1U);
    EXPECT_EQ(refined->partition_generation(), 1U);
    check_native_closure(*refined->geometry());
    check_physical_boundaries(*refined);
    check_overlap_lookup(*refined, {{0.4, 0.1, 0.15}, {0.7, 0.8, 0.9}});
    std::size_t partial = 0;
    double seam_area = 0.0;
    for (const auto& item : refined->geometry()->interfaces())
        if (const auto* interface = std::get_if<NonconformingInterface>(&item))
        {
            ++partial;
            EXPECT_TRUE(interface->partial_coarse_patch);
            EXPECT_EQ(interface->faces.size(), 4U);
            for (const auto& mapping : interface->faces)
            {
                ASSERT_EQ(mapping.fine_faces.size(), 4U);
                const RegionFace coarse{interface->coarse_region, mapping.coarse_face};
                EXPECT_EQ(refined->geometry()->logical_faces(coarse).size(), 4U);
                for (const auto& weight : refined->geometry()->face_flux_weights(coarse))
                    EXPECT_DOUBLE_EQ(weight.coefficient, -0.25);
                EXPECT_DOUBLE_EQ(refined->geometry()->restrict_face_flux(coarse, [](ID) { return 2.0; }), -8.0);
                for (const auto face : refined->geometry()->logical_faces(coarse))
                    seam_area += refined->geometry()->face_area(face);
            }
        }
    EXPECT_EQ(partial, 4U);
    EXPECT_DOUBLE_EQ(seam_area, 1.0);
    const auto unchanged = refined->adapt({});
    EXPECT_EQ(unchanged->geometry(), refined->geometry());
    EXPECT_EQ(unchanged->topology_generation(), refined->topology_generation());
    const auto restored = refined->adapt({}, {{0, {0, 0, 0}}});
    EXPECT_EQ(restored->blocks(), original.blocks());
    EXPECT_EQ(restored->topology_generation(), 2U);
    check_physical_boundaries(*restored);
    EXPECT_EQ(refined->blocks().size(), 9U);
    for (ID cell = 0; cell < original.geometry()->num_cells(); ++cell)
    {
        EXPECT_EQ(restored->cell_box(cell).lower, original.cell_box(cell).lower);
        EXPECT_EQ(restored->cell_box(cell).upper, original.cell_box(cell).upper);
    }
}

TEST(CartesianAMRTest, BalanceIncludesCornerNeighborsAndPeriodicFacePartners)
{
    AMRGrid config;
    config.root_blocks = {2, 2, 2};
    config.block_cells = {2, 2, 2};
    config.max_level = 3;
    const CartesianAMR root(config);
    const auto once = root.adapt({leaf_ordinal(root, {0, {0, 0, 0}})});
    const auto twice = once->adapt({leaf_ordinal(*once, {1, {1, 1, 1}})});
    EXPECT_EQ(twice->blocks().size(), 71U);
    for (const auto& block : twice->blocks())
        EXPECT_GE(block.level, 1U);
    check_native_closure(*twice->geometry());
    check_physical_boundaries(*twice);
    check_overlap_lookup(*twice, {{0.35, 0.35, 0.35}, {0.61, 0.66, 0.71}});

    config.root_blocks = {2, 1, 1};
    config.periodic = {true, false, false};
    const CartesianAMR periodic(config);
    const auto first = periodic.adapt({0});
    ASSERT_EQ(first->blocks().size(), 16U);
    for (const auto& block : first->blocks())
        EXPECT_EQ(block.level, 1U);
    const auto second = first->adapt({leaf_ordinal(*first, {1, {0, 0, 0}})});
    EXPECT_EQ(second->blocks().size(), 30U);
    std::size_t periodic_seams = 0;
    for (const auto& item : second->geometry()->interfaces())
        if (const auto* seam = std::get_if<StructuredPatchInterface>(&item); seam && seam->periodic_translation)
        {
            ++periodic_seams;
            EXPECT_EQ(second->blocks()[seam->first.region].level, second->blocks()[seam->second.region].level);
            EXPECT_DOUBLE_EQ(seam->periodic_translation->x, -1.0);
        }
    EXPECT_GT(periodic_seams, 0U);
    const auto rejected_by_balance = second->adapt({}, {{1, {0, 0, 0}}});
    EXPECT_EQ(rejected_by_balance->blocks(), second->blocks());
    EXPECT_EQ(rejected_by_balance->geometry(), second->geometry());
    check_native_closure(*second->geometry());
    check_physical_boundaries(*second);
    check_overlap_lookup(*second, {{0.0, 0.1, 0.15}, {0.04, 0.4, 0.45}});
}

TEST(CartesianAMRTest, InvalidRequestsPreserveTheAcceptedHierarchy)
{
    AMRGrid invalid;
    invalid.block_cells[0] = 3;
    EXPECT_THROW(CartesianAMR{invalid}, std::invalid_argument);
    invalid = {};
    invalid.max_level = 63;
    EXPECT_THROW(CartesianAMR{invalid}, std::invalid_argument);
    invalid = {};
    invalid.extent[2] = 0.0;
    EXPECT_THROW(CartesianAMR{invalid}, std::invalid_argument);
    const CartesianAMR root(AMRGrid{});
    const auto first = root.adapt({0});
    const auto second = first->adapt({0});
    const auto accepted = second->geometry();
    EXPECT_THROW((void) second->adapt({}, {{0, {0, 0, 0}}}), std::invalid_argument);
    EXPECT_THROW((void) first->adapt({0}, {{0, {0, 0, 0}}}), std::invalid_argument);
    const auto finest =
        std::find_if(second->blocks().begin(), second->blocks().end(), [](const auto& b) { return b.level == 2; });
    ASSERT_NE(finest, second->blocks().end());
    EXPECT_THROW(
        (void) second->adapt({static_cast<std::size_t>(finest - second->blocks().begin())}), std::invalid_argument);
    EXPECT_THROW((void) second->adapt({second->blocks().size()}), std::out_of_range);
    EXPECT_EQ(second->geometry(), accepted);
}

TEST(CartesianAMRTest, PartialCoarsePatchCodecRetainsCoverageAndDefaultStrictness)
{
    AMRGrid config;
    config.root_blocks = {2, 1, 1};
    const auto refined = CartesianAMR(config).adapt({0});
    const auto& mesh = *refined->geometry();
    auto interfaces = mesh.interfaces();
    const auto selected = std::find_if(interfaces.begin(), interfaces.end(),
        [](const auto& item) { return std::holds_alternative<NonconformingInterface>(item); });
    ASSERT_NE(selected, interfaces.end());
    std::get<NonconformingInterface>(*selected).partial_coarse_patch = false;
    EXPECT_THROW((MultiRegionMesh(mesh.regions(), interfaces)), std::invalid_argument);
    std::get<NonconformingInterface>(*selected).partial_coarse_patch = true;
    auto& mapping = std::get<NonconformingInterface>(*selected).faces.front();
    mapping.fine_faces.back() = mapping.fine_faces.front();
    EXPECT_THROW((MultiRegionMesh(mesh.regions(), interfaces)), std::invalid_argument);
    const auto packet = mesh.serialize_partition(std::span<const ID>{});
    const auto restored = MultiRegionMesh::deserialize_partition(packet, Tpetra::getDefaultComm());
    EXPECT_EQ(restored->configuration_signature(), mesh.configuration_signature());
    EXPECT_EQ(restored->num_cells(), mesh.num_cells());
    EXPECT_EQ(restored->num_faces(), mesh.num_faces());
    check_native_closure(*restored);
    auto old_version = packet;
    const std::uint64_t old_magic = 0x5346524547490002ULL;
    std::memcpy(old_version.data(), &old_magic, sizeof(old_magic));
    EXPECT_THROW(
        (void) MultiRegionMesh::deserialize_partition(old_version, Tpetra::getDefaultComm()), std::invalid_argument);
}

TEST(CartesianAMRTest, NativeOwnedMapsAndHalosUseCanonicalGlobalCellsAndFaces)
{
    AMRGrid config;
    config.root_blocks = {2, 1, 1};
    const auto refined = CartesianAMR(config).adapt({0});
    const auto& geometry = *refined->geometry();
    const Handle mesh(refined->geometry());
    EXPECT_EQ(mesh.owned_cell_map()->getGlobalNumElements(), geometry.num_cells());
    EXPECT_EQ(mesh.owned_face_map()->getGlobalNumElements(), geometry.num_faces());
    EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(mesh.has_materialized_indexer());
    const auto comm = Tpetra::getDefaultComm();
    const long long local_count = static_cast<long long>(mesh.num_owned_cells());
    long long total_count = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_count, &total_count);
    EXPECT_EQ(total_count, static_cast<long long>(geometry.num_cells()));
    if (comm->getSize() > 1)
        EXPECT_LT(mesh.num_owned_cells(), geometry.num_cells());
    using Pack = DefaultTpetraTypes;
    Pack::vector_type owned(mesh.owned_cell_map(), true);
    Pack::vector_type overlap(mesh.overlap_cell_map(), true);
    for (std::size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
        owned.replaceLocalValue(static_cast<Pack::local_ordinal_type>(cell), mesh.cell_global_id(cell) + 0.25);
    Tpetra::Import<Pack::local_ordinal_type, Pack::global_ordinal_type, Pack::node_type> importer(
        mesh.owned_cell_map(), mesh.overlap_cell_map());
    overlap.doImport(owned, importer, Tpetra::INSERT);
    const auto data = overlap.getData();
    for (std::size_t cell = 0; cell < mesh.num_cells(); ++cell)
    {
        EXPECT_DOUBLE_EQ(data[cell], mesh.cell_global_id(cell) + 0.25);
        EXPECT_NEAR(mesh.cell_volume(cell), volume(refined->cell_box(mesh.cell_global_id(cell))), 1.0e-15);
        for (const auto face : mesh.faces(cell))
            if (mesh.is_owned_cell(cell) && !mesh.is_geometry_exterior_face(face))
                EXPECT_NE(mesh.opposite_cell(face, cell), Handle::invalid_local_id());
    }
}

TEST(CartesianAMRMultiRankTest, DivergentConfigurationRequestsAndPartialPatchFlagsRejectCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Requires multiple MPI ranks.";
    AMRGrid config;
    config.max_level = comm->getRank() == 0 ? 2U : 3U;
    EXPECT_THROW(CartesianAMR{config}, std::invalid_argument);
    config = {};
    if (comm->getRank() == 0)
        config.block_cells[0] = 0;
    EXPECT_THROW(CartesianAMR{config}, std::invalid_argument);
    const CartesianAMR hierarchy(AMRGrid{});
    const std::vector<std::size_t> request =
        comm->getRank() == 0 ? std::vector<std::size_t>{0} : std::vector<std::size_t>{};
    EXPECT_THROW((void) hierarchy.adapt(request), std::invalid_argument);
    const auto geometry = test::coarse_fine_regions();
    auto interfaces = geometry->interfaces();
    if (comm->getRank() == 0)
        std::get<NonconformingInterface>(interfaces[0]).partial_coarse_patch = true;
    EXPECT_THROW((MultiRegionMesh(geometry->regions(), interfaces)), std::invalid_argument);
    EXPECT_EQ(hierarchy.topology_generation(), 0U);
}

TEST(CartesianAMRTest, DirectoryClosedBlockQueriesMatchExhaustiveIntersections)
{
    AMRGrid config;
    config.root_blocks = {3, 2, 1};
    config.block_cells = {2, 2, 2};
    config.max_level = 2;
    const auto root = std::make_shared<CartesianAMR>(config);
    const auto hierarchy = root->adapt({1, 4});
    std::vector<AMRBox> queries{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}},
        {{-1.0, -1.0, -1.0}, {0.0, 0.0, 0.0}}, {{1.1, 0.0, 0.0}, {2.0, 1.0, 1.0}},
        {{-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0}}};
    for (std::size_t b = 0; b < hierarchy->blocks().size(); ++b)
    {
        const auto box = hierarchy->block_box(b);
        queries.push_back(box);
        queries.push_back({box.lower, box.lower});
        queries.push_back({box.upper, box.upper});
        auto expanded = box;
        for (std::size_t a = 0; a < 3; ++a)
        {
            expanded.lower[a] -= 0.013;
            expanded.upper[a] += 0.037;
        }
        queries.push_back(expanded);
    }
    for (const auto& query : queries)
    {
        std::vector<std::size_t> expected;
        for (std::size_t b = 0; b < hierarchy->blocks().size(); ++b)
        {
            const auto box = hierarchy->block_box(b);
            bool intersects = true;
            for (std::size_t a = 0; a < 3; ++a)
                intersects = intersects && !(query.upper[a] < box.lower[a] || box.upper[a] < query.lower[a]);
            if (intersects)
                expected.push_back(b);
        }
        EXPECT_EQ(hierarchy->intersecting_blocks(query), expected);
    }
    EXPECT_THROW((void) hierarchy->intersecting_blocks({{NAN, 0.0, 0.0}, {1.0, 1.0, 1.0}}), std::invalid_argument);
    EXPECT_THROW((void) hierarchy->intersecting_blocks({{1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}}), std::invalid_argument);
}
TEST(CartesianAMRTest, StableCellAncestryAndExactSourcePlans)
{
    AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.region_id = 41;
    grid.material_id = 73;
    CartesianAMR source(grid), foreign(grid);
    const AdaptiveHierarchy& provider = source;
    EXPECT_EQ(provider.native_geometry(), source.geometry());
    const auto original_key = source.cell_key(8);
    const auto plan = source.plan_adaptation({0});
    auto refined = source.apply_adaptation(plan);
    EXPECT_THROW((void)foreign.apply_adaptation(plan), std::invalid_argument);
    EXPECT_THROW((void)refined->apply_adaptation(plan), std::invalid_argument);
    EXPECT_EQ(source.observation(), plan.source_observation());
    // The untouched level-zero block sorts before the refined level-one leaves.
    EXPECT_EQ(refined->cell_key(0), original_key);
    EXPECT_EQ(refined->cell_material(0), 73U);
    EXPECT_EQ(refined->cell_key(0).region, 41U);
    EXPECT_EQ(refined->cell_fragment(0), 0U);
    std::set<AdaptiveCellKey> keys;
    std::map<CartesianCellAncestry, unsigned> parent_counts;
    for (ID cell = 0; cell < refined->geometry()->num_cells(); ++cell)
    {
        const auto key = refined->cell_key(cell);
        EXPECT_TRUE(keys.insert(key).second);
        const auto ancestry = std::get<CartesianCellAncestry>(key.ancestry);
        if (ancestry.level)
            ++parent_counts[ancestry.parent()];
    }
    EXPECT_EQ(parent_counts.size(), 8U);
    for (const auto& [parent, count] : parent_counts)
        EXPECT_EQ(count, 8U);
    EXPECT_THROW((void)std::get<CartesianCellAncestry>(original_key.ancestry).parent(), std::invalid_argument);
    EXPECT_EQ(source.adapt({})->observation(), source.observation());
    const auto other_plan = source.plan_adaptation({});
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        const auto& mixed = Tpetra::getDefaultComm()->getRank() == 0 ? plan : other_plan;
        EXPECT_THROW((void)source.apply_adaptation(mixed), std::invalid_argument);
    }
}

TEST(CartesianAMRTest, NativePartitionChangesOnlyOwnershipObservation)
{
    AMRGrid grid;
    grid.block_cells = {8, 2, 2};
    grid.extent = {4.0, 1.0, 1.0};
    CartesianAMR source(grid), foreign(grid);
    const auto comm = Tpetra::getDefaultComm();
    const MeshHandle<> original(source.geometry());
    CompositePartitionOptions options;
    options.policy = CompositePartitionOptions::Policy::TopologyAware;
    options.target_cells_per_fragment = 8;
    options.require_balance = false;
    auto partition = std::make_shared<const CompositePartition<>>(MeshPartitioner<DefaultTpetraTypes>::partition(
        CompositeMeshSource(source.geometry()), options, comm));
    const auto pending = source.plan_adaptation({});
    auto rebound = source.with_partition(partition);
    const auto before = source.observation(), after = rebound->observation();
    EXPECT_EQ(before.topology_identity, after.topology_identity);
    EXPECT_EQ(before.topology, after.topology);
    EXPECT_EQ(before.geometry, after.geometry);
    EXPECT_NE(before.partition_identity, after.partition_identity);
    EXPECT_NE(before.identity, after.identity);
    EXPECT_EQ(after.partition, before.partition + 1);
    EXPECT_EQ(rebound->geometry(), source.geometry());
    EXPECT_EQ(rebound->cell_key(0), source.cell_key(0));
    EXPECT_THROW((void)rebound->apply_adaptation(pending), std::invalid_argument);
    EXPECT_THROW((void)foreign.with_partition(partition), std::invalid_argument);
    MeshHandle<> native(*rebound->native_partition());
    const auto owned = native.owned_cell_map()->getLocalNumElements();
    EXPECT_EQ(owned, partition->plan().owned_cells().size());
    std::set<ID> native_ids, original_ids, planned_ids;
    for (std::size_t cell = 0; cell < native.num_owned_cells(); ++cell)
        native_ids.insert(native.cell_global_id(cell));
    for (std::size_t cell = 0; cell < original.num_owned_cells(); ++cell)
        original_ids.insert(original.cell_global_id(cell));
    for (const auto cell : partition->plan().owned_cells())
        planned_ids.insert(cell);
    EXPECT_EQ(native_ids, planned_ids);
    const int changed = native_ids != original_ids;
    const int empty = native_ids.empty();
    int changed_ranks = 0, empty_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &changed, &changed_ranks);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_ranks);
    if (comm->getRank() == 0)
        std::cout << "Partition observation: ranks=" << comm->getSize()
                  << " global_cells=" << source.geometry()->num_cells()
                  << " changed_owner_ranks=" << changed_ranks
                  << " empty_owner_ranks=" << empty_ranks << '\n';
    if (comm->getSize() > 1)
    {
        if (comm->getSize() == 2 || comm->getSize() == 4)
            EXPECT_GT(changed_ranks, 0);
        // The anisotropic graph favors cuts across x over the default canonical
        // slices. Empty ownership is checked only when ranks outnumber cells.
        const auto cells = static_cast<int>(source.geometry()->num_cells());
        if (comm->getSize() > cells)
            EXPECT_GE(empty_ranks, comm->getSize() - cells);
    }
    else
    {
        EXPECT_EQ(changed_ranks, 0);
        EXPECT_EQ(empty_ranks, 0);
    }
    auto repeated = rebound->with_partition(partition);
    EXPECT_EQ(repeated->observation(), after);
    if (comm->getSize() > 1)
    {
        // Same hierarchy and collective partition plan on every rank; only rank
        // zero supplies the already retained wrapper. The pointer no-op status
        // must agree before any replacement identity broadcast.
        auto copy = std::make_shared<const CompositePartition<>>(*partition);
        const auto mixed = comm->getRank() == 0 ? partition : copy;
        EXPECT_THROW((void)rebound->with_partition(mixed), std::invalid_argument);
    }
    EXPECT_EQ(source.observation(), before);
}
TEST(CartesianAMRTest, RejectsMovedResidentBeforePartitionBinding)
{
    CartesianAMR source(AMRGrid{});
    const auto observation = source.observation();
    const auto box = source.cell_box(0);
    auto partition = std::make_shared<const CompositePartition<>>(MeshPartitioner<DefaultTpetraTypes>::partition(
        CompositeMeshSource(source.geometry()), {}, Tpetra::getDefaultComm()));
    EXPECT_TRUE(partition->geometry_unchanged());
    ASSERT_NO_THROW((void)source.with_partition(partition));
    auto mesh = std::make_shared<MeshHandle<>>(*partition);
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, 0.5);
    motion.accept_trial();
    EXPECT_FALSE(partition->geometry_unchanged());
    EXPECT_THROW((void)source.with_partition(partition), std::invalid_argument);
    EXPECT_TRUE(source.geometry_unchanged());
    EXPECT_EQ(source.observation(), observation);
    EXPECT_EQ(source.cell_box(0).lower, box.lower);
    EXPECT_EQ(source.cell_box(0).upper, box.upper);
}
} // namespace
