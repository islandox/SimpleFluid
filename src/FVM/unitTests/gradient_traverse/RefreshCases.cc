/** @file RefreshCases.cc
 * @brief Gradient cache refresh and geometry lifetime checks.
 */
#include "reconstruction_helpers.hh"

TEST(GradientTraverseTest, BoundaryDirectionOrderAndRefreshFailurePreservePreviousGeometry)
{
    auto mesh = std::make_shared<Handle>(test::two_regions(2));
    using Call = std::pair<Local, Local>;
    std::vector<Call> calls, expected;
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        FVM::detail::visit_gradient_geometry_samples(*mesh, static_cast<Local>(cell), [&](const auto& sample)
        { if (!sample.interior) expected.emplace_back(sample.face_lid, cell); });
    bool fail = false;
    Cache cache(mesh, [&](Local face, Local cell)
    {
        calls.emplace_back(face, cell);
        if (fail && calls.size() == 2) throw std::runtime_error("injected boundary direction failure");
        return mesh->face_centroid(face) - mesh->cell_centroid(cell);
    });
    EXPECT_EQ(calls, expected);
    const auto* plan = &cache.traverse_plan();
    const auto& interior = cache.interior_geometry();
    const auto& boundary = cache.boundary_geometry();
    const auto old_interior = snapshot(interior), old_boundary = snapshot(boundary);
    const auto* interior_data = interior.data();
    const auto* boundary_data = boundary.data();
    const auto old_epoch = cache.geometry_epoch();
    calls.clear();
    fail = true;
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.2, 0.1);
    if (expected.size() >= 2)
    {
        EXPECT_THROW(cache.refresh(), std::runtime_error);
        EXPECT_EQ(cache.geometry_epoch(), old_epoch);
        EXPECT_EQ(snapshot(interior), old_interior);
        EXPECT_EQ(snapshot(boundary), old_boundary);
        EXPECT_EQ(interior.data(), interior_data);
        EXPECT_EQ(boundary.data(), boundary_data);
        EXPECT_THROW(cache.boundary_geometry(), std::invalid_argument);
    }
    fail = false;
    calls.clear();
    cache.refresh();
    EXPECT_EQ(calls, expected);
    EXPECT_EQ(&cache.traverse_plan(), plan);
    expect_least_squares_geometry(cache);
    // A callback exception must also release the geometry lease for rollback.
    motion.rollback_trial();
    cache.refresh();
    EXPECT_EQ(snapshot(cache.interior_geometry()), old_interior);
    EXPECT_EQ(snapshot(cache.boundary_geometry()), old_boundary);
}

TEST(GradientTraverseTest, MotionAndRollbackReusePlansWithCurrentGeometryForEveryRegionFamily)
{
    for (const auto& [geometry, planar_faces] : composite_fixtures())
    {
        auto mesh = std::make_shared<Handle>(geometry);
        Cache least_squares(mesh);
        GaussCache gauss(mesh);
        const auto* ls_plan = &least_squares.traverse_plan();
        const auto* gauss_plan = &gauss.traverse_plan();
        const auto ls_bytes = ls_plan->storage_bytes();
        const auto gauss_bytes = gauss_plan->storage_bytes();
        PlanarALEMeshMotion<> motion(mesh);
        auto check = [&]
        {
            least_squares.refresh();
            gauss.refresh();
            EXPECT_EQ(&least_squares.traverse_plan(), ls_plan);
            EXPECT_EQ(&gauss.traverse_plan(), gauss_plan);
            EXPECT_EQ(ls_plan->storage_bytes(), ls_bytes);
            EXPECT_EQ(gauss_plan->storage_bytes(), gauss_bytes);
            expect_least_squares_geometry(least_squares);
            expect_gauss_geometry(gauss);
            expect_stencil_geometry(*mesh);
            expect_linear_reconstruction(mesh, planar_faces);
        };
        motion.begin_trial(1.35, 0.2);
        EXPECT_THROW(least_squares.boundary_geometry(), std::invalid_argument);
        EXPECT_THROW(gauss.geometry(), std::invalid_argument);
        check();
        motion.accept_trial();
        motion.begin_trial(0.8, 0.2);
        check();
        motion.rollback_trial();
        check();
    }
}

TEST(GradientTraverseTest, EmptyRanksBuildRefreshAndApplyWithoutSyntheticEntries)
{
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(std::vector<Meshes::MultiRegionMesh::Region>{
        Meshes::cartesian_region("single", {{{0, 1}, {0, 1}, {0, 1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    SP<const Handle> mesh = std::make_shared<Handle>(geometry);
    compare_caches(mesh);
    expect_linear_reconstruction(mesh);
    // Missing incidence is invalid even on a rank with no cells to traverse.
    const auto cells_only = std::make_shared<const MeshTraversePlan<Handle>>(*mesh, MeshTraverseEntities::Cells);
    const auto locations = FVM::detail::boundary_face_locations(*mesh);
    EXPECT_THROW(static_cast<void>(FVM::detail::least_squares_gradient_stencils(*mesh, *cells_only)),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(FVM::detail::boundary_aware_gradient_geometry(*mesh, locations, *cells_only)),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(TransportCache::SharedGeometry(*mesh, cells_only)), std::invalid_argument);
    if (Tpetra::getDefaultComm()->getRank() > 0) EXPECT_EQ(mesh->num_owned_cells(), 0U);
}

TEST(GradientTraverseTest, SharedLeastSquaresPlansRetainGeometryIndependenceAndRejectForeignMeshes)
{
    auto mesh = std::make_shared<Handle>(test::coarse_fine_regions());
    auto foreign = std::make_shared<Handle>(test::coarse_fine_regions());
    Cache original(mesh);
    const auto plan = original.shared_traverse_plan();
    Cache shared(mesh, Cache::boundary_direction_provider_type{}, plan);
    EXPECT_EQ(shared.shared_traverse_plan(), plan);
    EXPECT_EQ(snapshot(shared.interior_geometry()), snapshot(original.interior_geometry()));
    EXPECT_EQ(snapshot(shared.boundary_geometry()), snapshot(original.boundary_geometry()));
    EXPECT_NE(shared.interior_geometry().data(), original.interior_geometry().data());
    EXPECT_THROW(static_cast<void>(Cache(foreign, Cache::boundary_direction_provider_type{}, plan)),
        std::invalid_argument);
    const auto faces_only = std::make_shared<const Cache::traverse_plan_type>(*mesh, MeshTraverseEntities::Faces);
    EXPECT_THROW(static_cast<void>(Cache(mesh, Cache::boundary_direction_provider_type{}, faces_only)),
        std::invalid_argument);

    const auto& old_interior = original.interior_geometry();
    const auto& old_boundary = original.boundary_geometry();
    const auto interior_values = snapshot(old_interior), boundary_values = snapshot(old_boundary);
    const auto old_epoch = original.geometry_epoch();
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.25, 0.2);
    shared.refresh();
    EXPECT_EQ(shared.shared_traverse_plan(), plan);
    EXPECT_EQ(original.geometry_epoch(), old_epoch);
    EXPECT_EQ(snapshot(old_interior), interior_values);
    EXPECT_EQ(snapshot(old_boundary), boundary_values);
    EXPECT_THROW(original.interior_geometry(), std::invalid_argument);
    Cache fresh(mesh);
    EXPECT_EQ(snapshot(shared.interior_geometry()), snapshot(fresh.interior_geometry()));
    EXPECT_EQ(snapshot(shared.boundary_geometry()), snapshot(fresh.boundary_geometry()));
    motion.rollback_trial();
}

TEST(GradientTraverseTest, TransportAndPressureRefreshSharePlansWithoutChangingRetainedSnapshots)
{
    for (const auto& geometry : {test::coarse_fine_regions(), test::periodic_regions()})
    {
        auto mesh = std::make_shared<Handle>(geometry);
        TransportCache transport(*mesh);
        PressureWorkspace pressure(mesh);
        const auto transport_snapshot = transport.shared_geometry();
        const auto pressure_snapshot = pressure.shared_geometry();
        PressureWorkspace retained_pressure(mesh, pressure_snapshot);
        const auto* transport_plan = &transport.traverse_plan();
        const auto pressure_plan = pressure.gradient_cache().shared_traverse_plan();
        const auto old_transport = snapshot(*transport_snapshot);
        const auto& old_pressure_interior = retained_pressure.gradient_cache().interior_geometry();
        const auto& old_pressure_boundary = retained_pressure.gradient_cache().boundary_geometry();
        const auto old_interior = snapshot(old_pressure_interior), old_boundary = snapshot(old_pressure_boundary);
        const auto transport_epoch = transport.geometry_epoch();
        const auto pressure_epoch = retained_pressure.gradient_cache().geometry_epoch();
        transport.refresh();
        pressure.refresh_geometry();
        EXPECT_EQ(transport.shared_geometry(), transport_snapshot);
        EXPECT_EQ(pressure.shared_geometry(), pressure_snapshot);

        PlanarALEMeshMotion<> motion(mesh);
        auto refresh_and_compare = [&]
        {
            transport.refresh();
            pressure.refresh_geometry();
            EXPECT_EQ(&transport.traverse_plan(), transport_plan);
            EXPECT_EQ(pressure.gradient_cache().shared_traverse_plan(), pressure_plan);
            EXPECT_NE(transport.shared_geometry(), transport_snapshot);
            EXPECT_NE(pressure.shared_geometry(), pressure_snapshot);
            EXPECT_EQ(retained_pressure.shared_geometry(), pressure_snapshot);
            EXPECT_EQ(transport_snapshot->epoch, transport_epoch);
            EXPECT_EQ(retained_pressure.gradient_cache().geometry_epoch(), pressure_epoch);
            EXPECT_EQ(snapshot(*transport_snapshot), old_transport);
            EXPECT_EQ(snapshot(old_pressure_interior), old_interior);
            EXPECT_EQ(snapshot(old_pressure_boundary), old_boundary);
            TransportCache fresh_transport(*mesh);
            PressureWorkspace fresh_pressure(mesh);
            EXPECT_EQ(snapshot(*transport.shared_geometry()), snapshot(*fresh_transport.shared_geometry()));
            EXPECT_EQ(snapshot(pressure.gradient_cache().interior_geometry()),
                snapshot(fresh_pressure.gradient_cache().interior_geometry()));
            EXPECT_EQ(snapshot(pressure.gradient_cache().boundary_geometry()),
                snapshot(fresh_pressure.gradient_cache().boundary_geometry()));
            expect_least_squares_geometry(pressure.gradient_cache());
        };
        motion.begin_trial(1.3, 0.2);
        refresh_and_compare();
        EXPECT_THROW(retained_pressure.gradient_cache().interior_geometry(), std::invalid_argument);
        motion.rollback_trial();
        refresh_and_compare();
        EXPECT_EQ(snapshot(*transport.shared_geometry()), old_transport);
        EXPECT_EQ(snapshot(pressure.gradient_cache().interior_geometry()), old_interior);
        EXPECT_EQ(snapshot(pressure.gradient_cache().boundary_geometry()), old_boundary);
    }
}

