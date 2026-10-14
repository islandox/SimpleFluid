#include "reconstruction_helpers.hh"

TEST(GradientTraverseTest, OnePassCompactCachesPreserveBothNormalsAndBoundaryCallbackOrder)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial canonical maps.";
    std::vector<SP<Handle>> meshes;
    meshes.push_back(std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 3}, {0, 0.4, 2, 4}}})));
    meshes.push_back(std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCylindrical3D>(
        Vec3D<ArrReal>{{{1, 2, 3, 5}, {0, 0.2, 0.7, 1.5}, {0, 0.5, 2, 3}}})));
    meshes.push_back(layered_gradient_mesh(5, false));
    meshes.push_back(layered_gradient_mesh(5, true));
    for (const auto& mesh : meshes)
    {
        using Call = std::pair<Local, Local>;
        std::vector<Call> calls;
        const auto direction = [&](Local face, Local cell)
        {
            const auto delta = mesh->face_centroid(face) - mesh->cell_centroid(cell);
            return Vec{delta.x + 0.17 * delta.y, delta.y - 0.11 * delta.z, delta.z + 0.09 * delta.x};
        };
        const auto expected_calls = [&]
        {
            std::vector<Call> expected;
            const auto locations = FVM::detail::boundary_face_locations(*mesh);
            for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
                FVM::detail::visit_gradient_geometry_samples(*mesh, static_cast<Local>(cell), [&](const auto& sample)
                {
                    if (!sample.interior && locations.at(sample.face_lid).active)
                        expected.emplace_back(sample.face_lid, static_cast<Local>(cell));
                });
            return expected;
        };
        Cache cache(mesh, [&](Local face, Local cell)
        {
            calls.emplace_back(face, cell);
            return direction(face, cell);
        });
        GaussCache gauss(mesh);
        EXPECT_EQ(calls, expected_calls());
        const auto check = [&]
        {
            // Independently form each normal matrix through point geometry.
            // In particular the overridden direction must not change the
            // interior-only matrix or the raw boundary normal distance.
            expect_least_squares_geometry(cache, direction);
            expect_gauss_geometry(gauss);
            const auto execution = cache.traverse_plan().acquire();
            EXPECT_FALSE(execution.cells().interior().empty());
            EXPECT_FALSE(execution.cells().boundary().empty());
        };
        check();
        const auto interior_before = snapshot(cache.interior_geometry());
        const auto boundary_before = snapshot(cache.boundary_geometry());
        const auto topology = gauss.layered_geometry() ? gauss.layered_geometry()->topology : nullptr;
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.2, 0.2);
        calls.clear();
        cache.refresh();
        gauss.refresh();
        EXPECT_EQ(calls, expected_calls());
        if (topology) EXPECT_EQ(gauss.layered_geometry()->topology, topology);
        check();
        motion.rollback_trial();
        calls.clear();
        cache.refresh();
        gauss.refresh();
        EXPECT_EQ(calls, expected_calls());
        if (topology) EXPECT_EQ(gauss.layered_geometry()->topology, topology);
        check();
        EXPECT_EQ(snapshot(cache.interior_geometry()), interior_before);
        EXPECT_EQ(snapshot(cache.boundary_geometry()), boundary_before);
    }
}
