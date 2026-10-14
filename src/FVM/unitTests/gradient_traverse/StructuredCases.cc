#include "reconstruction_helpers.hh"

TEST(GradientTraverseTest, StructuredNumericalCachesPreserveScalarVectorOrderAndLiveBoundaries)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial canonical maps.";
    for (unsigned mask = 0; mask < 8; ++mask)
    {
        SCOPED_TRACE(::testing::Message() << "periodic mask=" << mask);
        auto geometry = std::make_shared<Meshes::OrthogonalCartesian3D>(
            Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 7, 8}}},
            Vec3D<bool>{{bool(mask & 1U), bool(mask & 2U), bool(mask & 4U)}});
        expect_native_cache_layout(std::make_shared<Handle>(geometry), true);
    }
    for (const auto& edges : {
        Vec3D<ArrReal>{{{0, 1}, {0, 1}, {0, 1}}},
        Vec3D<ArrReal>{{{0, 1}, {0, 0.5, 2, 4}, {0, 2, 7}}},
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 1}, {0, 2, 7}}},
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 2, 7}, {0, 1}}}})
        for (bool periodic : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "thin periodic=" << periodic);
            auto geometry = std::make_shared<Meshes::OrthogonalCartesian3D>(
                edges, Vec3D<bool>{{periodic && edges[0].size() > 2,
                    periodic && edges[1].size() > 2, periodic && edges[2].size() > 2}});
            expect_native_cache_layout(std::make_shared<Handle>(geometry), true);
        }
    const auto pi = std::numbers::pi_v<real_t>;
    for (const auto& edges : {
        Vec3D<ArrReal>{{{1, 2}, {0, 0.5}, {0, 1}}},
        Vec3D<ArrReal>{{{1, 2, 4}, {0, 0.2, 0.7, 1.5}, {0, 0.5, 3, 4}}},
        Vec3D<ArrReal>{{{1, 2}, {0, pi, 2 * pi}, {0, 1}}},
        Vec3D<ArrReal>{{{1, 2, 3, 4}, {0, pi / 2, pi, 3 * pi / 2, 2 * pi}, {0, 0.5, 2, 3}}}})
    {
        auto geometry = std::make_shared<Meshes::OrthogonalCylindrical3D>(edges);
        SCOPED_TRACE(::testing::Message() << "cylindrical periodic=" << geometry->is_theta_periodic());
        expect_native_cache_layout(std::make_shared<Handle>(geometry), true);
    }
}

TEST(GradientTraverseTest, NumericalCachesRetainGenericFallbackForMappedAndCompositeMeshes)
{
    expect_native_cache_layout(std::make_shared<Handle>(test::coarse_fine_regions()), false);
    auto mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 7, 8}}}));
    if (Tpetra::getDefaultComm()->getSize() == 1)
    {
        auto reordered = MeshReorderingFactory<>::selected_cells_first(
            std::move(mesh), [](auto, const Vec& point) { return point.x > 1; });
        expect_native_cache_layout(reordered.mesh, false);
    }
    else
        expect_native_cache_layout(mesh, false);
}

TEST(GradientTraverseTest, StructuredCompatibilityViewsPublishOnceAndSupportCacheMovesAndCopies)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial canonical maps.";
    auto mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 7, 8}}}));
    Cache original(mesh);
    Cache moved(std::move(original));
    Cache cache(mesh);
    cache = std::move(moved);
    GaussCache gauss(mesh);
    GaussCache copy = gauss;
    EXPECT_TRUE(cache.uses_structured_storage());
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_EQ(copy.structured_geometry(), gauss.structured_geometry());
    EXPECT_FALSE(copy.legacy_geometry_materialized());
    expect_cached_reconstruction(mesh, cache, gauss);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_FALSE(gauss.legacy_geometry_materialized());

    using Published = std::tuple<const void*, const void*, const void*, const void*>;
    std::latch start(1);
    std::vector<std::future<Published>> readers;
    for (size_t reader = 0; reader < 6; ++reader)
        readers.push_back(std::async(std::launch::async, [&, reader]
        {
            start.wait();
            // Const compatibility reads may race on the first materialization.
            return Published{&cache.interior_geometry(), &cache.boundary_geometry(),
                &cache.boundary_locations(), &(reader % 2 ? copy : gauss).geometry()};
        }));
    start.count_down();
    const auto published = readers.front().get();
    for (size_t reader = 1; reader < readers.size(); ++reader)
        EXPECT_EQ(readers[reader].get(), published);
    EXPECT_TRUE(cache.legacy_geometry_materialized());
    EXPECT_TRUE(gauss.legacy_geometry_materialized());
    EXPECT_TRUE(copy.legacy_geometry_materialized());
    expect_least_squares_geometry(cache);
    expect_gauss_geometry(gauss);
    expect_cached_reconstruction(mesh, cache, gauss);

    const auto& old_geometry = copy.geometry();
    const auto old_volumes = old_geometry.volumes;
    const auto old_epoch = copy.geometry_epoch();
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.25, 0.2);
    cache.refresh();
    gauss.refresh();
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_FALSE(gauss.legacy_geometry_materialized());
    EXPECT_NE(gauss.structured_geometry(), copy.structured_geometry());
    EXPECT_EQ(copy.geometry_epoch(), old_epoch);
    EXPECT_EQ(old_geometry.volumes, old_volumes);
    EXPECT_THROW(copy.geometry(), std::invalid_argument);
    expect_cached_reconstruction(mesh, cache, gauss);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_FALSE(gauss.legacy_geometry_materialized());
    motion.rollback_trial();
    cache.refresh();
    gauss.refresh();
    expect_cached_reconstruction(mesh, cache, gauss);
    expect_least_squares_geometry(cache);
    expect_gauss_geometry(gauss);
    EXPECT_EQ(gauss.geometry().volumes, old_volumes);
}

TEST(GradientTraverseTest, StructuredRefreshFailurePreservesUnmaterializedSnapshot)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial canonical maps.";
    // Stay below the growth-ratio limit so the ALE trial reaches the injected
    // cache-refresh failure independently of geometry-query rounding.
    auto mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 6, 8}}}));
    bool fail = false;
    using Call = std::pair<Local, Local>;
    std::vector<Call> calls, expected;
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        FVM::detail::visit_gradient_geometry_samples(*mesh, static_cast<Local>(cell), [&](const auto& sample)
        { if (!sample.interior) expected.emplace_back(sample.face_lid, cell); });
    Cache cache(mesh, [&](Local face, Local cell)
    {
        calls.emplace_back(face, cell);
        if (fail && calls.size() == 2) throw std::runtime_error("injected structured boundary failure");
        return mesh->face_centroid(face) - mesh->cell_centroid(cell);
    });
    EXPECT_EQ(calls, expected);
    GaussCache gauss(mesh);
    const auto epoch = cache.geometry_epoch();
    const auto* plan = &cache.traverse_plan();
    const auto* geometry = cache.structured_geometry();
    ASSERT_NE(geometry, nullptr);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    expect_cached_reconstruction(mesh, cache, gauss);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    // An unsuccessful rebuild at the same epoch leaves the original compact
    // snapshot usable, including its numerical values and lazy state.
    fail = true;
    calls.clear();
    EXPECT_THROW(cache.refresh(), std::runtime_error);
    EXPECT_EQ(cache.geometry_epoch(), epoch);
    EXPECT_EQ(cache.structured_geometry(), geometry);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    expect_cached_reconstruction(mesh, cache, gauss);
    fail = false;
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.35, 0.2);
    fail = true;
    calls.clear();
    EXPECT_THROW(cache.refresh(), std::runtime_error);
    EXPECT_EQ(cache.geometry_epoch(), epoch);
    EXPECT_EQ(&cache.traverse_plan(), plan);
    EXPECT_THROW(cache.structured_geometry(), std::invalid_argument);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_THROW(cache.interior_geometry(), std::invalid_argument);
    fail = false;
    calls.clear();
    cache.refresh();
    EXPECT_EQ(calls, expected);
    gauss.refresh();
    expect_cached_reconstruction(mesh, cache, gauss);
    motion.rollback_trial();
    calls.clear();
    cache.refresh();
    EXPECT_EQ(calls, expected);
    gauss.refresh();
    expect_cached_reconstruction(mesh, cache, gauss);
    expect_least_squares_geometry(cache);
    expect_gauss_geometry(gauss);
    EXPECT_EQ(calls, expected);
}

