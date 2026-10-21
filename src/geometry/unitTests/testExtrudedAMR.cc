/** @file testExtrudedAMR.cc
 * @brief Whole-product ancestry, exact refinement, inverse families and native contracts.
 */
#include "geometry/ExtrudedAMR.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
ExtrudedAMRGrid quad()
{
    ExtrudedAMRGrid g;
    g.xy_nodes = {{{.1, .1, 0}}, {{.2, .1, 0}}, {{.2, .2, 0}}, {{.1, .2, 0}}};
    g.base_cells = {{0, 1, 2, 3}};
    g.z_edges = {.1, .2};
    g.boundaries = {{0, 1, "south"}, {1, 2, "east"}, {2, 3, "north"}, {3, 0, "west"}};
    g.boundary_aliases = {{"zmin", "floor"}, {"zmax", "roof"}};
    return g;
}
ExtrudedAMRGrid triangles()
{
    ExtrudedAMRGrid g;
    g.xy_nodes = {{{0, 0, 0}}, {{1, 0, 0}}, {{1, 1, 0}}, {{0, 1, 0}}};
    g.base_cells = {{0, 1, 2}, {0, 2, 3}};
    g.z_edges = {0, 1};
    return g;
}
std::size_t base_ordinal(const ExtrudedAMR& h, const BaseCellKey& k)
{
    auto it = std::find(h.base_keys().begin(), h.base_keys().end(), k);
    if (it == h.base_keys().end())
        throw std::logic_error("Missing test base key.");
    return it - h.base_keys().begin();
}
void same_cells(const ExtrudedAMR& a, const ExtrudedAMR& b)
{
    ASSERT_EQ(a.geometry()->num_cells(), b.geometry()->num_cells());
    EXPECT_EQ(a.base_keys(), b.base_keys());
    EXPECT_EQ(a.interval_keys(), b.interval_keys());
    for (std::uint64_t c = 0; c < a.geometry()->num_cells(); ++c)
    {
        EXPECT_EQ(a.cell_key(c), b.cell_key(c));
        EXPECT_TRUE(a.cell_geometry(c)->equivalent(*b.cell_geometry(c)));
    }
    for (std::size_t c = 0; c < a.base_keys().size(); ++c)
        EXPECT_EQ(a.base_edge_keys(c), b.base_edge_keys(c));
}
std::shared_ptr<const AdaptiveOverlapPlan> certificate(const ExtrudedAMR& a, const ExtrudedAMR& b)
{
    std::vector<AdaptiveCellBinding> source, target;
    for (std::uint64_t c = 0; c < a.geometry()->num_cells(); ++c)
        source.push_back({a.cell_key(c), c, a.cell_material(c), a.cell_geometry(c)});
    for (std::uint64_t c = 0; c < b.geometry()->num_cells(); ++c)
        target.push_back({b.cell_key(c), c, b.cell_material(c), b.cell_geometry(c)});
    return AdaptiveOverlapPlan::build(a.observation(), b.observation(), source, target, a.overlap_relations(b));
}
TEST(ExtrudedAMRTest, DecimalQuadFactorOrderAndExactInversePreservePatches)
{
    ExtrudedAMR parent(quad());
    auto xy = parent.adapt({0});
    auto z = parent.adapt({}, {0});
    auto xyz = xy->adapt({}, {0});
    auto zxy = z->adapt({0});
    ASSERT_EQ(xyz->base_keys().size(), 4U);
    ASSERT_EQ(xyz->interval_keys().size(), 2U);
    same_cells(*xyz, *zxy);
    EXPECT_EQ(xyz->geometry()->configuration_signature(), zxy->geometry()->configuration_signature());
    EXPECT_NO_THROW((void) certificate(parent, *xyz));
    EXPECT_NO_THROW((void) certificate(*xyz, parent));
    const auto coarsened_xy = xyz->adapt({}, {}, parent.base_keys());
    same_cells(*coarsened_xy, *z);
    const auto restored = coarsened_xy->adapt({}, {}, {}, parent.interval_keys());
    same_cells(parent, *restored);
    EXPECT_EQ(parent.geometry()->configuration_signature(), restored->geometry()->configuration_signature());
    std::set<std::string> patches;
    for (auto b : xyz->geometry()->boundary_batch_ids())
        patches.insert(xyz->geometry()->boundary_batch_name(b));
    EXPECT_TRUE(patches.contains("product/floor") || patches.contains("floor"));
    EXPECT_TRUE(patches.contains("product/roof") || patches.contains("roof"));
    for (const auto& side : {"south", "east", "north", "west"})
        EXPECT_TRUE(std::any_of(patches.begin(), patches.end(), [&](const auto& p) { return p.ends_with(side); }));
}
TEST(ExtrudedAMRTest, TriangleNVBClosureAndCompleteFactorFamilies)
{
    ExtrudedAMR parent(triangles());
    auto refined = parent.adapt({0});
    EXPECT_EQ(refined->base_keys().size(), 4U);
    EXPECT_NO_THROW((void) certificate(parent, *refined));
    EXPECT_THROW((void) refined->adapt({}, {}, {parent.base_keys()[0]}), std::invalid_argument);
    auto restored = refined->adapt({}, {}, parent.base_keys());
    same_cells(parent, *restored);
    auto deep = refined->adapt({0});
    EXPECT_GT(deep->base_keys().size(), refined->base_keys().size());
    EXPECT_NO_THROW((void) certificate(parent, *deep));
    EXPECT_THROW((void) deep->adapt({}, {}, parent.base_keys()), std::invalid_argument);
    MeshHandle<> native(deep->geometry());
    EXPECT_EQ(native.connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(native.has_materialized_indexer());
    auto bad = triangles();
    bad.triangle_reference_edges = {0, 0};
    EXPECT_THROW((void) ExtrudedAMR(bad), std::invalid_argument);
}
TEST(ExtrudedAMRTest, CircularStylePolygonFanAndSavedParentRecovery)
{
    ExtrudedAMRGrid g;
    constexpr double pi = 3.14159265358979323846;
    for (unsigned i = 0; i < 7; ++i)
    {
        const auto theta = 2 * pi * i / 7;
        g.xy_nodes.push_back({std::cos(theta), std::sin(theta), 0});
    }
    g.base_cells = {{0, 1, 2, 3, 4, 5, 6}};
    g.z_edges = {-.1, .3};
    ExtrudedAMR parent(g);
    auto fan = parent.adapt({0});
    EXPECT_EQ(fan->base_keys().size(), 5U);
    EXPECT_NO_THROW((void) certificate(parent, *fan));
    auto refined = fan->adapt({0});
    EXPECT_GT(refined->base_keys().size(), 5U);
    EXPECT_NO_THROW((void) certificate(parent, *refined));
    auto restored = fan->adapt({}, {}, parent.base_keys());
    same_cells(parent, *restored);
    EXPECT_EQ(parent.geometry()->configuration_signature(), restored->geometry()->configuration_signature());
}
TEST(ExtrudedAMRTest, StableEdgesAcrossDifferentIndependentXYHistories)
{
    auto g = triangles();
    // Independent outer-edge bisections preserve one connected constituent;
    // the next split then closes the shared diagonal from both histories.
    g.triangle_reference_edges = {0, 1};
    ExtrudedAMR parent(g);
    auto first = parent.adapt({0});
    auto second = parent.adapt({1});
    auto ab = first->adapt({base_ordinal(*first, parent.base_keys()[1])});
    auto ba = second->adapt({base_ordinal(*second, parent.base_keys()[0])});
    same_cells(*ab, *ba);
    auto deeper_ab = ab->adapt({0}), deeper_ba = ba->adapt({0});
    same_cells(*deeper_ab, *deeper_ba);
}
TEST(ExtrudedAMRTest, RootAndTemplateInteriorEdgesHaveDisjointPersistentNamespaces)
{
    ExtrudedAMR parent(quad());
    const auto root_edge = parent.base_edge_keys(0)[0];
    EXPECT_EQ(root_edge.kind, BaseEdgeKey::Kind::Root);
    auto refined = parent.adapt({0});
    bool matched_ordinal = false;
    for (std::size_t c = 0; c < refined->base_keys().size(); ++c)
        for (const auto& edge : refined->base_edge_keys(c))
            if (edge.kind == BaseEdgeKey::Kind::TemplateInterior && edge.creator == root_edge.creator &&
                edge.ordinal == root_edge.ordinal)
            {
                matched_ordinal = true;
                EXPECT_NE(edge, root_edge);
                EXPECT_EQ(edge.path, root_edge.path);
            }
    EXPECT_TRUE(matched_ordinal);
    EXPECT_NO_THROW((void) certificate(parent, *refined));
}
TEST(ExtrudedAMRTest, RepeatedCyclesAndLayerCountsKeepFactoredStorageBounded)
{
    auto g = quad();
    g.z_edges = {0, 1};
    ExtrudedAMR parent(g);
    auto current = std::make_shared<const ExtrudedAMR>(parent);
    const auto initial = parent.storage_counts();
    const auto initial_bytes = parent.storage_report().measured_bytes();
    for (unsigned cycle = 0; cycle < 5; ++cycle)
    {
        auto fine = current->adapt({0}, {0});
        current = fine->adapt({}, {}, parent.base_keys(), parent.interval_keys());
        EXPECT_EQ(current->storage_counts().point_recipes, initial.point_recipes);
        EXPECT_EQ(current->storage_counts().z_recipes, initial.z_recipes);
        EXPECT_EQ(current->storage_counts().base_history_nodes, initial.base_history_nodes);
        EXPECT_EQ(current->storage_counts().axial_history_nodes, initial.axial_history_nodes);
        EXPECT_LE(current->storage_report().measured_bytes(), initial_bytes + 1024);
    }
    auto many = g;
    many.z_edges.clear();
    for (unsigned k = 0; k <= 64; ++k)
        many.z_edges.push_back(double(k) / 64);
    ExtrudedAMR layers(many);
    EXPECT_EQ(layers.storage_counts().base_vertices, initial.base_vertices);
    EXPECT_EQ(layers.storage_counts().point_recipes, initial.point_recipes);
    EXPECT_EQ(layers.storage_counts().z_edges, 65U);
    const auto one = parent.storage_report().measured_bytes(), sixty_four = layers.storage_report().measured_bytes();
    EXPECT_LT(sixty_four, one + 64 * 2048U); // per-interval state, no base*layer shell cache
    EXPECT_EQ(layers.storage_counts().base_history_nodes, 1U);
    auto base_four = parent.adapt({0});
    auto layers_four = layers.adapt({0});
    const std::vector<std::size_t> all_four{0, 1, 2, 3};
    auto base_sixteen = base_four->adapt(all_four);
    auto layers_sixteen = layers_four->adapt(all_four);
    ASSERT_EQ(base_sixteen->storage_counts().base_cells, 16U);
    ASSERT_EQ(layers_sixteen->storage_counts().base_cells, 16U);
    EXPECT_EQ(base_sixteen->storage_counts().point_recipes, layers_sixteen->storage_counts().point_recipes);
    const auto sixteen_one = base_sixteen->storage_report().measured_bytes();
    const auto sixteen_many = layers_sixteen->storage_report().measured_bytes();
    ASSERT_GT(sixty_four, one);
    ASSERT_GT(sixteen_many, sixteen_one);
    const auto axial_growth_one_base = sixty_four - one;
    const auto axial_growth_sixteen_bases = sixteen_many - sixteen_one;
    // A retained per-cell shell/metric table would multiply axial growth by
    // sixteen. Shared exact base state must be independent of layer growth.
    EXPECT_LE(axial_growth_sixteen_bases, axial_growth_one_base + 2048U);
    EXPECT_LE(axial_growth_one_base, axial_growth_sixteen_bases + 2048U);
    std::cout << "Product storage: one_layer=" << one << " sixty_four_layers=" << sixty_four
              << " base_vertices=" << initial.base_vertices << " axial_growth_one_base=" << axial_growth_one_base
              << " axial_growth_sixteen_bases=" << axial_growth_sixteen_bases << '\n';
}
TEST(ExtrudedAMRTest, CrossedFactorPairsAndEmptySelectionsAreExplicit)
{
    ExtrudedAMR parent(quad());
    auto xy = parent.adapt({0}), z = parent.adapt({}, {0});
    const auto pairs = z->overlap_relations(*xy);
    ASSERT_EQ(pairs.size(), 8U);
    for (const auto& pair : pairs)
        EXPECT_EQ(pair.kind, AdaptiveAncestryRelation::Kind::ProductIntersection);
    EXPECT_NO_THROW((void) certificate(*z, *xy));
    EXPECT_TRUE(z->overlap_relations(*xy, std::vector<std::uint64_t>{}).empty());
    auto altered = quad();
    altered.boundary_aliases["zmin"] = "different";
    ExtrudedAMR foreign(altered);
    EXPECT_THROW((void) parent.overlap_relations(foreign), std::invalid_argument);
}
TEST(ExtrudedAMRTest, FailedCandidatesAndStalePlansLeaveSourceUsable)
{
    auto g = quad();
    g.max_depth = 1;
    ExtrudedAMR source(g);
    const auto before = source.observation();
    auto plan = source.plan_adaptation({0});
    auto fine = source.apply_adaptation(plan);
    EXPECT_THROW((void) fine->apply_adaptation(plan), std::invalid_argument);
    ExtrudedAMR foreign(g);
    EXPECT_THROW((void) foreign.apply_adaptation(plan), std::invalid_argument);
    EXPECT_THROW((void) fine->adapt({0}), std::invalid_argument);
    EXPECT_THROW((void) source.adapt({9}), std::out_of_range);
    EXPECT_EQ(source.observation(), before);
    EXPECT_EQ(source.adapt()->observation(), before);
    EXPECT_GT(source.cell_geometry(0)->moments().volume.value, 0);
    auto quality = quad();
    quality.xy_nodes.clear();
    for (unsigned i = 0; i < 5; ++i)
    {
        double t = 6.283185307179586 * i / 5;
        quality.xy_nodes.push_back({std::cos(t), std::sin(t), 0});
    }
    quality.base_cells = {{0, 1, 2, 3, 4}};
    quality.boundaries.clear();
    quality.minimum_scaled_area = .5;
    ExtrudedAMR accepted(quality);
    EXPECT_THROW((void) accepted.adapt({0}), std::invalid_argument);
    auto collapsed = quad();
    collapsed.xy_nodes = {{{1e16, 0, 0}}, {{1e16 + 2, 0, 0}}, {{1e16 + 2, 1, 0}}, {{1e16, 1, 0}}};
    EXPECT_THROW((void) ExtrudedAMR(collapsed), std::invalid_argument);
    auto near = quad();
    near.z_edges = {1e16, 1e16 + 2};
    EXPECT_THROW((void) ExtrudedAMR(near), std::invalid_argument);
}
TEST(ExtrudedAMRTest, NativePartitionPinsResidentIdentityAndRejectsMixedPlans)
{
    ExtrudedAMR source(quad());
    auto comm = Tpetra::getDefaultComm();
    auto pending = source.plan_adaptation();
    auto partition = std::make_shared<const CompositePartition<>>(
        MeshPartitioner<DefaultTpetraTypes>::partition(CompositeMeshSource(source.geometry()), {}, comm));
    auto bound = source.with_partition(partition);
    EXPECT_EQ(bound->observation().topology_identity, source.observation().topology_identity);
    EXPECT_NE(bound->observation().partition_identity, source.observation().partition_identity);
    EXPECT_EQ(bound->observation().native_geometry, partition->geometry().get());
    EXPECT_TRUE(bound->geometry_unchanged());
    EXPECT_THROW((void) bound->apply_adaptation(pending), std::invalid_argument);
    auto mesh = std::make_shared<MeshHandle<>>(*bound->native_partition());
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
    EXPECT_THROW((void) PlanarALEMeshMotion<>(mesh), std::invalid_argument);
    if (comm->getSize() > 1)
    {
        auto copy = std::make_shared<const CompositePartition<>>(*partition);
        EXPECT_THROW((void) bound->with_partition(comm->getRank() ? copy : partition), std::invalid_argument);
        auto a = source.plan_adaptation({0}), b = source.plan_adaptation({}, {0});
        EXPECT_THROW((void) source.apply_adaptation(comm->getRank() ? a : b), std::invalid_argument);
        EXPECT_THROW((void) source.adapt(comm->getRank() ? std::vector<std::size_t>{0} : std::vector<std::size_t>{}),
            std::invalid_argument);
        EXPECT_THROW((void) source.adapt(comm->getRank() ? std::vector<std::size_t>{9} : std::vector<std::size_t>{}),
            std::exception);
        auto divergent = quad();
        if (comm->getRank())
            divergent.xy_nodes[1][0] = .21;
        EXPECT_THROW((void) ExtrudedAMR(divergent), std::invalid_argument);
    }
}
TEST(ExtrudedAMRTest, NestedCoarseningMarksCannotBecomeEligibleMidTransaction)
{
    ExtrudedAMR root(quad());
    auto first = root.adapt({0}, {0});
    auto deep = first->adapt({0}, {0});
    const auto observation = deep->observation();
    const auto child = first->base_keys()[0], parent = root.base_keys()[0];
    EXPECT_THROW((void) deep->adapt({}, {}, {child, parent}), std::invalid_argument);
    EXPECT_THROW((void) deep->adapt({}, {}, {parent, child}), std::invalid_argument);
    const auto zchild = first->interval_keys()[0], zparent = root.interval_keys()[0];
    EXPECT_THROW((void) deep->adapt({}, {}, {}, {zchild, zparent}), std::invalid_argument);
    EXPECT_THROW((void) deep->adapt({}, {}, {}, {zparent, zchild}), std::invalid_argument);
    EXPECT_EQ(deep->observation(), observation);
    EXPECT_GT(deep->cell_geometry(0)->moments().volume.value, 0);
}
TEST(ExtrudedAMRTest, RejectsConnectedRootOverlapAndUnresolvedRootTJunction)
{
    ExtrudedAMRGrid spiral;
    spiral.xy_nodes.push_back({0, 0, 0});
    for (unsigned i = 0; i <= 4; ++i)
    {
        const double a = 100.0 * i * 3.14159265358979323846 / 180;
        spiral.xy_nodes.push_back({std::cos(a), std::sin(a), 0});
    }
    for (unsigned i = 0; i < 4; ++i)
        spiral.base_cells.push_back({0, i + 1, i + 2});
    spiral.z_edges = {0, 1};
    EXPECT_THROW((void) ExtrudedAMR(spiral), std::invalid_argument);
    ExtrudedAMRGrid hanging;
    hanging.xy_nodes = {
        {{0, 0, 0}}, {{1, 0, 0}}, {{1, 2, 0}}, {{0, 2, 0}}, {{1, 1, 0}}, {{2, 0, 0}}, {{2, 1, 0}}, {{2, 2, 0}}};
    hanging.base_cells = {{0, 1, 2, 3}, {1, 5, 6, 4}, {4, 6, 7, 2}};
    hanging.z_edges = {0, 1};
    EXPECT_THROW((void) ExtrudedAMR(hanging), std::invalid_argument);
}
} // namespace
