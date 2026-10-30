/** @file testGeneralVoFSpatialCache.cc
 * @brief Independent retained-domain cache lifetime and admission checks.
 */
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "general_vof_finite_test_helpers.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
namespace local = SimpleFluid::VoF::general_vof_detail;
using Cache = local::LocalSpatialCache;
using Geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
struct CacheTrial
{
    SimpleFluid::geometry::swept_volume::Work work{};
    GeneralVoFTransportDiagnostics diagnostics;
    std::shared_ptr<const Cache> cache;
    explicit CacheTrial(const GeneralVoF::Data& data) : cache(local::retained_spatial_cache(data, work, diagnostics)) {}
};
GeneralVoF::Data catalog()
{
    GeneralVoF::Data data;
    data.topology = std::make_shared<GeneralVoF::Data::Topology>();
    data.observation.identity = std::make_shared<const int>(1);
    data.supplied_observation = data.observation;
    for (unsigned i = 0; i < 8; ++i)
    {
        GeneralVoF::Key key;
        key.region = i;
        local::Cell cell;
        cell.key = key;
        cell.gid = 10 + i;
        cell.owner = Tpetra::getDefaultComm()->getRank();
        cell.finite_geometry = Geometry::box({2. * i, 0, 0}, {2. * i + 1, 1, 1})->dynamic_geometry();
        data.cells.emplace(key, std::move(cell));
    }
    return data;
}
std::vector<std::size_t> query(const Cache& cache, long double x, SimpleFluid::geometry::swept_volume::Work& work,
    GeneralVoFTransportDiagnostics& diagnostics)
{
    local::local_spatial_detail::Box box;
    box.add(std::array{x, .5L, .5L});
    return cache.donor_index.query(box, work, diagnostics);
}
TEST(GeneralVoFSpatialCache, ImmutableCatalogReusesStorageAndCountsOnlyExecutedProbes)
{
    auto data = catalog();
    const CacheTrial first(data);
    EXPECT_EQ(first.diagnostics.spatial_index_builds, 1u);
    EXPECT_EQ(first.diagnostics.spatial_index_reuses, 0u);
    EXPECT_EQ(first.diagnostics.index_queries, 0u);
    EXPECT_EQ(first.diagnostics.index_node_visits, 0u);
    EXPECT_GT(first.diagnostics.spatial_index_bytes, 0u);
    // Publication copies carry different Cell objects and phase state. Neither
    // phase values nor retained planes are a static spatial-index dependency.
    auto next = data;
    for (auto& [key, cell] : next.cells)
    {
        cell.phase.gas_volume = .25;
        cell.phase.carrier_volume = .75;
        cell.requires_interface = true;
    }
    CacheTrial second(next);
    EXPECT_EQ(second.cache, first.cache);
    EXPECT_EQ(second.diagnostics.spatial_index_builds, 0u);
    EXPECT_EQ(second.diagnostics.spatial_index_reuses, 1u);
    EXPECT_EQ(second.diagnostics.spatial_index_bytes, first.diagnostics.spatial_index_bytes);
    EXPECT_EQ(second.work.bytes, first.work.bytes);
    EXPECT_LT(second.work.pair_tests, first.work.pair_tests);
    EXPECT_EQ(query(*second.cache, 4.5L, second.work, second.diagnostics), std::vector<std::size_t>{2});
    EXPECT_TRUE(query(*second.cache, 5.5L, second.work, second.diagnostics).empty());
    EXPECT_EQ(second.diagnostics.index_queries, 2u);
    EXPECT_GT(second.diagnostics.index_node_visits, 0u);
}
TEST(GeneralVoFSpatialCache, ChangedRetainedGeometryRebuildsWithoutMutatingPreviousReaders)
{
    auto data = catalog();
    CacheTrial first(data);
    auto changed = data;
    changed.cells.begin()->second.finite_geometry = Geometry::box({20, 0, 0}, {21, 1, 1})->dynamic_geometry();
    CacheTrial replacement(changed);
    EXPECT_NE(replacement.cache, first.cache);
    EXPECT_EQ(replacement.diagnostics.spatial_index_builds, 1u);
    EXPECT_EQ(query(*first.cache, .5L, first.work, first.diagnostics), std::vector<std::size_t>{0});
    EXPECT_TRUE(query(*replacement.cache, .5L, replacement.work, replacement.diagnostics).empty());
    EXPECT_EQ(query(*replacement.cache, 20.5L, replacement.work, replacement.diagnostics), std::vector<std::size_t>{0});
    // A changed key or owner invalidates even when represented coordinates agree.
    auto renamed = changed;
    auto node = renamed.cells.extract(renamed.cells.begin());
    node.key().region = 50;
    node.mapped().key = node.key();
    renamed.cells.insert(std::move(node));
    const CacheTrial renamed_trial(renamed);
    EXPECT_EQ(renamed_trial.diagnostics.spatial_index_builds, 1u);
    auto migrated = renamed;
    ++migrated.cells.begin()->second.owner;
    const CacheTrial migrated_trial(migrated);
    EXPECT_EQ(migrated_trial.diagnostics.spatial_index_builds, 1u);
}
TEST(GeneralVoFSpatialCache, GeometryTopologyAndPartitionObservationsInvalidate)
{
    const auto original = catalog();
    for (unsigned change = 0; change < 8; ++change)
    {
        SCOPED_TRACE(change);
        auto data = original;
        const CacheTrial before(data);
        switch (change)
        {
            case 0:
                ++data.observation.geometry;
                break;
            case 1:
                ++data.observation.topology;
                break;
            case 2:
                ++data.observation.partition;
                break;
            case 3:
                data.observation.identity = std::make_shared<const int>(1);
                break;
            case 4:
                data.observation.topology_identity = std::make_shared<const int>(1);
                break;
            case 5:
                data.observation.partition_identity = std::make_shared<const int>(1);
                break;
            case 6:
                data.observation.native_geometry = &data;
                break;
            case 7:
                ++data.supplied_observation.geometry;
                break;
        }
        const CacheTrial after(data);
        EXPECT_NE(before.cache, after.cache);
        EXPECT_EQ(after.diagnostics.spatial_index_builds, 1u);
        EXPECT_EQ(after.diagnostics.spatial_index_reuses, 0u);
        // A new topology cannot inherit a cache by matching shape or GIDs.
        data.topology = std::make_shared<GeneralVoF::Data::Topology>();
        const CacheTrial rebuilt(data);
        EXPECT_EQ(rebuilt.diagnostics.spatial_index_builds, 1u);
    }
}
TEST(GeneralVoFSpatialCache, FrontierIndexRetainsImmutableFiniteFacetGeometry)
{
    auto data = catalog();
    auto topology = std::make_shared<GeneralVoF::Data::Topology>();
    AMR::AdaptiveTile tile;
    tile.first = data.cells.begin()->first;
    tile.first_owner = data.cells.begin()->second.owner;
    tile.face_gid = 5;
    tile.identity = {'w', 'a', 'l', 'l'};
    tile.exterior = true;
    tile.boundary = "wall";
    const auto retained = Geometry::from_dynamic(data.cells.begin()->second.finite_geometry);
    tile.geometry = retained->face_geometry(0);
    topology->tiles.push_back(tile);
    topology->frontier_tile_references.push_back({decltype(topology)::element_type::SurfaceTileOwner::OwnedTiles, 0});
    data.topology = topology;
    const CacheTrial first(data);
    CacheTrial next(data);
    EXPECT_EQ(next.cache, first.cache);
    ASSERT_EQ(next.cache->frontiers.size(), 1u);
    EXPECT_EQ(next.cache->frontiers[0].geometry, tile.geometry);
    EXPECT_TRUE(next.cache->frontiers[0].first);
    const auto& frontier = next.cache->frontiers[0];
    EXPECT_EQ(frontier.face_gid, tile.face_gid);
    EXPECT_EQ(frontier.tile_identity, tile.identity);
    EXPECT_EQ(frontier.boundary, "wall");
    EXPECT_EQ(frontier.kind, local::local_spatial_detail::Frontier::Kind::PhysicalBoundary);
    EXPECT_EQ(frontier.retained_key, data.cells.begin()->first);
    EXPECT_EQ(frontier.retained_gid, data.cells.begin()->second.gid);
    EXPECT_EQ(frontier.retained_owner, tile.first_owner);
    EXPECT_EQ(frontier.retained_geometry->dynamic_geometry(), data.cells.begin()->second.finite_geometry);
    ASSERT_TRUE(frontier.support_identity);
    EXPECT_EQ(*frontier.support_identity, PLIC::planar_support<ArithmeticMode::Ordinary>(*retained->face_geometry(0)));
    ASSERT_TRUE(frontier.original_boundary);
    EXPECT_FALSE(frontier.outside_key);
    local::local_spatial_detail::Box bounds;
    const auto center = tile.geometry->centroid();
    bounds.add(std::array<long double, 3>{center.x, center.y, center.z});
    EXPECT_EQ(next.cache->frontier_index.query(bounds, next.work, next.diagnostics), std::vector<std::size_t>{0});
    local::local_spatial_detail::Box far;
    far.add(std::array{40.L, .5L, .5L});
    EXPECT_TRUE(next.cache->frontier_index.query(far, next.work, next.diagnostics).empty());
}
TEST(GeneralVoFSpatialCache, HaloFrontierBindsRetainedSecondCellAndOutsideAuthority)
{
    auto data = catalog();
    auto topology = std::make_shared<GeneralVoF::Data::Topology>();
    AMR::AdaptiveTile tile;
    tile.first.region = 50;
    tile.first_owner = 7;
    tile.second = data.cells.begin()->first;
    tile.second_owner = data.cells.begin()->second.owner;
    tile.face_gid = 91;
    tile.identity = {'h', 'a', 'l', 'o'};
    const auto retained = Geometry::from_dynamic(data.cells.begin()->second.finite_geometry);
    tile.geometry = retained->face_geometry(1);
    topology->tiles.push_back(tile);
    topology->frontier_tile_references.push_back({decltype(topology)::element_type::SurfaceTileOwner::OwnedTiles, 0});
    data.topology = topology;
    const CacheTrial trial(data);
    ASSERT_EQ(trial.cache->frontiers.size(), 1u);
    const auto& frontier = trial.cache->frontiers[0];
    EXPECT_EQ(frontier.kind, local::local_spatial_detail::Frontier::Kind::RetainedHalo);
    EXPECT_FALSE(frontier.first);
    EXPECT_EQ(frontier.retained_key, *tile.second);
    EXPECT_EQ(frontier.retained_owner, tile.second_owner);
    ASSERT_TRUE(frontier.outside_key);
    EXPECT_EQ(*frontier.outside_key, tile.first);
    EXPECT_EQ(frontier.outside_owner, tile.first_owner);
    EXPECT_EQ(frontier.retained_face, 1u);
    ASSERT_TRUE(frontier.original_boundary);
    const PLIC::Point<ArithmeticMode::Ordinary> center(std::array{.5L, .5L, .5L});
    EXPECT_EQ(frontier.original_boundary->classify(center), PLIC::OrdinaryBoundaryClass::Inside);
}
TEST(GeneralVoFSpatialCache, WarmAndColdCachesHonorEachConsumersLimits)
{
    const auto data = catalog();
    const CacheTrial admitted(data);
    const auto cache = data.topology->spatial_cache;
    SimpleFluid::geometry::swept_volume::Work limited{};
    limited.limits.byte_budget = admitted.diagnostics.spatial_index_bytes - 1;
    GeneralVoFTransportDiagnostics diagnostics;
    EXPECT_THROW((void) local::retained_spatial_cache(data, limited, diagnostics), PLIC::GeometryError);
    EXPECT_EQ(data.topology->spatial_cache, cache);
    EXPECT_EQ(diagnostics.spatial_index_reuses, 0u);
    limited = {};
    limited.limits.maximum_atoms = data.cells.size() - 1;
    EXPECT_THROW((void) local::retained_spatial_cache(data, limited, diagnostics), PLIC::GeometryError);
    limited = {};
    limited.limits.maximum_pair_tests = 0;
    EXPECT_THROW((void) local::retained_spatial_cache(data, limited, diagnostics), PLIC::GeometryError);
    auto cold = data;
    cold.topology = std::make_shared<GeneralVoF::Data::Topology>();
    limited = {};
    limited.limits.byte_budget = 1;
    EXPECT_THROW((void) local::retained_spatial_cache(cold, limited, diagnostics), PLIC::GeometryError);
    EXPECT_FALSE(cold.topology->spatial_cache);
    const CacheTrial retried(data);
    EXPECT_EQ(retried.cache, cache);
    EXPECT_EQ(retried.diagnostics.spatial_index_reuses, 1u);
}
TEST(GeneralVoFSpatialCache, CombinedRetainedAdmissionFailureLeavesTheSourceRetryable)
{
    auto data = catalog();
    const CacheTrial size_probe(data);
    auto cold = data;
    cold.topology = std::make_shared<GeneralVoF::Data::Topology>();
    // Independent admission oracle: the source header and cache each fit, but
    // their combined storage exceeds this budget by exactly one byte.
    cold.options.byte_budget = sizeof(cold) + size_probe.cache->bytes - 1;
    ASSERT_GT(cold.options.byte_budget, sizeof(cold));
    ASSERT_GT(cold.options.byte_budget, size_probe.cache->bytes);
    const auto combined_admission = [](const GeneralVoF::Data& source, std::size_t bytes)
    {
        local::wire::Budget retained{source.options.byte_budget};
        retained.add(sizeof(source));
        retained.add(bytes);
    };
    SimpleFluid::geometry::swept_volume::Work work{};
    GeneralVoFTransportDiagnostics diagnostics;
    EXPECT_THROW(
        (void) local::retained_spatial_cache(cold, work, diagnostics, combined_admission), PLIC::GeometryError);
    EXPECT_FALSE(cold.topology->spatial_cache);
    EXPECT_EQ(diagnostics.spatial_index_builds, 0u);
    ++cold.options.byte_budget;
    work = {};
    EXPECT_NO_THROW((void) local::retained_spatial_cache(cold, work, diagnostics, combined_admission));
    EXPECT_TRUE(cold.topology->spatial_cache);
    EXPECT_EQ(diagnostics.spatial_index_builds, 1u);
    // A failed replacement likewise preserves already admitted readers.
    auto changed = data;
    ++changed.cells.begin()->second.gid;
    changed.options.byte_budget = sizeof(changed) + size_probe.cache->bytes - 1;
    work = {};
    EXPECT_THROW(
        (void) local::retained_spatial_cache(changed, work, diagnostics, combined_admission), PLIC::GeometryError);
    EXPECT_EQ(data.topology->spatial_cache, size_probe.cache);
    const CacheTrial original_retry(data);
    EXPECT_EQ(original_retry.cache, size_probe.cache);
    EXPECT_EQ(original_retry.diagnostics.spatial_index_reuses, 1u);
}
TEST(GeneralVoFSpatialCache, EmptyOwnerBuildsAndReusesAnEmptyImmutableIndex)
{
    GeneralVoF::Data empty;
    empty.topology = std::make_shared<GeneralVoF::Data::Topology>();
    const CacheTrial first(empty);
    CacheTrial second(empty);
    EXPECT_EQ(first.diagnostics.spatial_index_builds, 1u);
    EXPECT_EQ(second.diagnostics.spatial_index_reuses, 1u);
    EXPECT_EQ(second.cache, first.cache);
    EXPECT_TRUE(query(*second.cache, .5L, second.work, second.diagnostics).empty());
    EXPECT_EQ(second.diagnostics.index_queries, 1u);
    EXPECT_EQ(second.diagnostics.index_node_visits, 0u);
}
} // namespace
