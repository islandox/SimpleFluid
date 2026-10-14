#include "test_helpers.hh"

TEST(MeshTraverseTest, StructuredCartesianRangesPreserveAllPeriodicMasks)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    for (unsigned mask = 0; mask < 8; ++mask)
    {
        SCOPED_TRACE(mask);
        auto geometry = std::make_shared<Cartesian>(
            Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 7}}},
            Vec3D<bool>{{bool(mask & 1U), bool(mask & 2U), bool(mask & 4U)}});
        Handle mesh(geometry);
        expect_structured_plan(*geometry, mesh);
        expect_structured_plan(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredCartesianRangesPreserveThinAndSingleCellAxes)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    for (const auto shape : {std::array<size_t, 3>{1, 1, 1}, {1, 4, 2}, {4, 1, 2}, {4, 2, 1}, {2, 2, 2}})
    {
        SCOPED_TRACE(::testing::PrintToString(shape));
        auto geometry = cartesian(shape[0], shape[1], shape[2]);
        Handle mesh(geometry);
        expect_structured_plan(*geometry, mesh);
        expect_structured_plan(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredCylindricalRangesPreserveSectorsAndPeriodicTheta)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto pi = std::numbers::pi_v<real_t>;
    for (const auto& edges : {
        Vec3D<ArrReal>{{{1, 2}, {0, 0.5}, {0, 1}}},
        Vec3D<ArrReal>{{{1, 2, 4}, {0, 0.2, 0.7, 1.5}, {0, 0.5, 3}}},
        Vec3D<ArrReal>{{{1, 2}, {0, pi, 2 * pi}, {0, 1}}},
        Vec3D<ArrReal>{{{1, 2, 4}, {0, pi / 2, pi, 3 * pi / 2, 2 * pi}, {0, 0.5, 2, 3}}}})
    {
        auto geometry = std::make_shared<Cylindrical>(edges);
        SCOPED_TRACE(geometry->is_theta_periodic());
        Handle mesh(geometry);
        expect_structured_plan(*geometry, mesh);
        expect_structured_plan(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredPlanStorageDoesNotGrowWithCellOrFaceCount)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto small = cartesian(3, 3, 3), large = cartesian(24, 24, 24);
    Handle small_handle(small), large_handle(large);
    for (const auto entities : {MeshTraverseEntities::Cells, MeshTraverseEntities::Faces,
                               MeshTraverseEntities::All, MeshTraverseEntities::AllWithCellFaces})
    {
        const auto small_native_plan = make_mesh_traverse_plan(*small, entities);
        const auto large_native_plan = make_mesh_traverse_plan(*large, entities);
        const auto small_handle_plan = make_mesh_traverse_plan(small_handle, entities);
        const auto large_handle_plan = make_mesh_traverse_plan(large_handle, entities);
        EXPECT_TRUE(small_native_plan.uses_structured_ranges());
        EXPECT_TRUE(large_native_plan.uses_structured_ranges());
        EXPECT_TRUE(small_handle_plan.uses_structured_ranges());
        EXPECT_TRUE(large_handle_plan.uses_structured_ranges());
        EXPECT_EQ(small_native_plan.storage_bytes(), large_native_plan.storage_bytes());
        EXPECT_EQ(small_handle_plan.storage_bytes(), large_handle_plan.storage_bytes());
    }
}

TEST(MeshTraverseTest, StructuredPlansRetainTopologyAndRefreshGeometryAcrossALE)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto check = [](const auto& geometry)
    {
        auto mesh = std::make_shared<Handle>(geometry);
        const auto plan = make_mesh_traverse_plan(*mesh, MeshTraverseEntities::AllWithCellFaces);
        const auto native_plan = make_mesh_traverse_plan(*geometry, MeshTraverseEntities::AllWithCellFaces);
        ASSERT_TRUE(plan.uses_structured_ranges());
        ASSERT_TRUE(native_plan.uses_structured_ranges());
        const auto storage = plan.storage_bytes();
        const auto capture = [&](const auto& retained_plan)
        {
            std::vector<real_t> metrics;
            const auto traverse = retained_plan.acquire();
            traverse.cells().apply([&](const auto& cell)
            {
                EXPECT_DOUBLE_EQ(cell.volume(), mesh->cell_volume(cell.id()));
                metrics.push_back(cell.volume());
                std::vector<Local> expected;
                for (const auto face : mesh->faces(cell.id())) expected.push_back(face);
                EXPECT_EQ(selected_ids(traverse.cell_faces(cell.id())), expected);
                traverse.cell_faces(cell.id()).apply(cell.geometry(), [&](const auto& face)
                {
                    EXPECT_DOUBLE_EQ(face.area(), mesh->face_area(face.id()));
                    expect_vector_equal(face.centroid(), mesh->face_centroid(face.id()));
                    expect_vector_equal(face.outward_normal(), mesh->face_normal_outward(face.id(), cell.id()));
                    metrics.push_back(face.area());
                });
            });
            return metrics;
        };
        const auto before = capture(plan);
        EXPECT_EQ(capture(native_plan), before);
        PlanarALEMeshMotion<Pack> motion(mesh);
        motion.begin_trial(4.0, 0.1);
        const auto during = capture(plan);
        EXPECT_NE(during, before);
        EXPECT_EQ(capture(native_plan), during);
        EXPECT_EQ(plan.storage_bytes(), storage);
        motion.rollback_trial();
        EXPECT_EQ(capture(plan), before);
        EXPECT_EQ(capture(native_plan), before);
        EXPECT_EQ(plan.storage_bytes(), storage);
    };
    check(cartesian(3, 2, 2));
    check(std::make_shared<Cylindrical>(
        Vec3D<ArrReal>{{{1, 2, 3}, {0, 0.5, 1}, {0, 1, 2}}}));
}

