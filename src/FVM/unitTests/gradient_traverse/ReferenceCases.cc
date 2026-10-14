/** @file ReferenceCases.cc
 * @brief Gradient traversal comparisons against reference reconstructions.
 */
#include "reconstruction_helpers.hh"

TEST(GradientTraverseTest, CompositeCacheBuildsAndRefreshesMatchOrderedPointGeometry)
{
    for (const auto& [geometry, planar_faces] : composite_fixtures())
    {
        SP<const Handle> mesh = std::make_shared<Handle>(geometry);
        const auto indexer = mesh->has_materialized_indexer();
        compare_caches(mesh);
        expect_linear_reconstruction(mesh, planar_faces);
        EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
        EXPECT_EQ(mesh->has_materialized_indexer(), indexer);
    }
}

TEST(GradientTraverseTest, NativeMappedReorderedAndLegacyCachesMatchPointGeometry)
{
    auto native = std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 1, 1.5, 2}, {0, 0.5, 1}, {0, 0.5, 1}}});
    compare_caches<Handle>(std::make_shared<Handle>(native));
    compare_caches<Mesh<Pack>>(test::build_mesh<Pack>(test::make_3x3x3_database()));
    auto mesh = std::make_shared<Handle>(native);
    auto reordered = MeshReorderingFactory<>::selected_cells_first(
        std::move(mesh), [](auto, const Vec& point) { return point.y > 0.5; });
    compare_caches(reordered.mesh);
    expect_linear_reconstruction(reordered.mesh);
}

TEST(GradientTraverseTest, RegionNativeCachesKeepMappedFieldsAndUntaggedExteriors)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Explicit native regions require serial handles.";
    using Native = Meshes::UnstructuredMesh;
    auto native = std::make_shared<Native>(
        Arr<Vec>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                 {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
                 {2, 0, 0}, {2, 1, 0}, {2, 0, 1}, {2, 1, 1}},
        Arr<Native::CellDefinition>{
            {Native::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}},
            {Native::CellType::HEXAHEDRON, {1, 8, 9, 2, 5, 10, 11, 6}}},
        Arr<Native::BoundaryFaceDefinition>{
            {{3, 7, 4, 0}, 7, "inlet"}, {{8, 10, 11, 9}, 8, "outlet"}});
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::native_region("pair", native)},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    const auto check = [&](SP<const Handle> mesh, bool native_execution)
    {
        ASSERT_EQ(mesh->supports_region_execution(), native_execution);
        const auto indexer = mesh->has_materialized_indexer();
        const auto owned = mesh->owned_cell_map(), overlap = mesh->overlap_cell_map();
        const auto faces = mesh->overlap_face_map();
        Cache least_squares(mesh);
        GaussCache gauss(mesh);
        const auto locations = FVM::detail::boundary_face_locations(*mesh);
        size_t tagged = 0, untagged = 0;
        for (const auto& face : gauss.geometry().faces)
            if (!face.interior)
            {
                EXPECT_TRUE(mesh->is_geometry_exterior_face(face.face_lid));
                EXPECT_EQ(face.boundary.active, locations.at(face.face_lid).active);
                (face.boundary.active ? tagged : untagged)++;
            }
        if (mesh->num_owned_cells())
        {
            EXPECT_GT(tagged, 0U);
            EXPECT_GT(untagged, 0U);
        }
        for (bool refresh : {false, true})
        {
            if (refresh)
            {
                least_squares.refresh();
                gauss.refresh();
            }
            expect_least_squares_geometry(least_squares);
            expect_gauss_geometry(gauss);
            expect_cached_reconstruction(mesh, least_squares, gauss);
        }
        EXPECT_EQ(mesh->owned_cell_map(), owned);
        EXPECT_EQ(mesh->overlap_cell_map(), overlap);
        EXPECT_EQ(mesh->overlap_face_map(), faces);
        EXPECT_FALSE(mesh->has_materialized_connectivity());
        EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
        EXPECT_EQ(mesh->has_materialized_indexer(), indexer);
    };
    check(std::make_shared<Handle>(geometry), true);
    if (Tpetra::getDefaultComm()->getSize() == 1)
    {
        // Composite reordering is unsupported; the same native constituent
        // exercises the mapped fallback against its independent point oracle.
        auto mesh = std::make_shared<Handle>(native);
        auto reordered = MeshReorderingFactory<>::selected_cells_first(
            std::move(mesh), [](auto, const Vec& center) { return center.x > 1; });
        check(reordered.mesh, false);
    }
}

TEST(GradientTraverseTest, RegionNativeCachesKeepPartitionSeamsInterior)
{
    if (Tpetra::getDefaultComm()->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    auto mesh = std::make_shared<Handle>(test::two_regions(2));
    ASSERT_TRUE(mesh->supports_region_execution());
    ASSERT_GT(mesh->num_local_cells(), mesh->num_owned_cells());
    Cache least_squares(mesh);
    GaussCache gauss(mesh);
    expect_least_squares_geometry(least_squares);
    expect_gauss_geometry(gauss);
    const auto locations = FVM::detail::boundary_face_locations(*mesh);
    size_t partition_faces = 0;
    for (const auto& face : gauss.geometry().faces)
        if (face.interior && static_cast<size_t>(face.other_lid) >= mesh->num_owned_cells())
        {
            ++partition_faces;
            EXPECT_FALSE(mesh->is_geometry_exterior_face(face.face_lid));
            EXPECT_FALSE(locations.at(face.face_lid).active);
            EXPECT_FALSE(face.boundary.active);
        }
    EXPECT_GT(partition_faces, 0U);
    expect_cached_reconstruction(mesh, least_squares, gauss);
    EXPECT_FALSE(mesh->has_materialized_connectivity());
}

