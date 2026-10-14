#include "reconstruction_helpers.hh"

namespace
{
void expect_layered_gauss_payload_capacity(const GaussCache& cache)
{
    const auto* compact = cache.layered_geometry();
    ASSERT_NE(compact, nullptr);
    const auto& mesh = *cache.mesh_ptr();
    size_t interior = 0, boundary = 0;
    // An independent 3D point-query count checks the XY/Z reservation formula,
    // including variable degree and one/two-layer boundary-only meshes.
    for (size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
    {
        bool exterior = false;
        size_t degree = 0;
        for (auto face : mesh.faces(static_cast<Local>(cell)))
        {
            ++degree;
            exterior = exterior || mesh.is_geometry_exterior_face(face);
        }
        (exterior ? boundary : interior) += degree;
    }
    EXPECT_EQ(compact->areas.size(), interior);
    EXPECT_EQ(compact->cell_distances.size(), interior);
    EXPECT_EQ(compact->other_distances.size(), interior);
    EXPECT_EQ(compact->topology->other_lids.size(), interior);
    EXPECT_EQ(compact->boundary.size(), boundary);
    EXPECT_EQ(compact->topology->boundary.size(), boundary);
    // Supported std::vector implementations allocate the requested capacity.
    // Growth slack would indicate that the exact payload reservation regressed.
    EXPECT_EQ(compact->areas.capacity(), interior);
    EXPECT_EQ(compact->cell_distances.capacity(), interior);
    EXPECT_EQ(compact->other_distances.capacity(), interior);
    EXPECT_EQ(compact->topology->other_lids.capacity(), interior);
    EXPECT_EQ(compact->boundary.capacity(), boundary);
    EXPECT_EQ(compact->topology->boundary.capacity(), boundary);
}
}

TEST(GradientTraverseTest, LayeredNumericalCachesPreserveVariableDegreeOrderAndLiveBoundaries)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "XY/Z remains serial only.";
    for (size_t layers : {1U, 2U, 5U})
        for (bool triangles : {false, true})
            for (bool reserved : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "layers=" << layers
                    << " triangles=" << triangles << " reserved=" << reserved);
                if (reserved)
                {
                    EXPECT_THROW(layered_gradient_mesh(layers, triangles, true), std::invalid_argument);
                    continue;
                }
                auto mesh = layered_gradient_mesh(layers, triangles);
                Cache cache(mesh);
                GaussCache gauss(mesh);
                expect_layered_gauss_payload_capacity(gauss);
                const auto retained_topology = gauss.layered_geometry()->topology;
                const auto row_data = retained_topology->rows.data();
                const auto interior_data = retained_topology->other_lids.data();
                const auto boundary_data = retained_topology->boundary.data();
                EXPECT_TRUE(cache.uses_layered_storage());
                EXPECT_TRUE(gauss.uses_layered_storage());
                EXPECT_FALSE(cache.uses_structured_storage());
                EXPECT_FALSE(gauss.uses_structured_storage());
                expect_cached_reconstruction(mesh, cache, gauss);
                EXPECT_FALSE(cache.legacy_geometry_materialized());
                EXPECT_FALSE(gauss.legacy_geometry_materialized());
                expect_least_squares_geometry(cache);
                expect_gauss_geometry(gauss);
                const auto locations = FVM::detail::boundary_face_locations(*mesh);
                const auto& actual = cache.boundary_locations();
                ASSERT_EQ(actual.size(), locations.size());
                for (size_t face = 0; face < locations.size(); ++face)
                {
                    EXPECT_EQ(actual[face].active, locations[face].active);
                    EXPECT_EQ(actual[face].batch_id, locations[face].batch_id);
                    EXPECT_EQ(actual[face].in_batch_id, locations[face].in_batch_id);
                }
                cache.refresh();
                gauss.refresh();
                EXPECT_EQ(gauss.layered_geometry()->topology, retained_topology);
                EXPECT_EQ(gauss.layered_geometry()->topology->rows.data(), row_data);
                EXPECT_EQ(gauss.layered_geometry()->topology->other_lids.data(), interior_data);
                EXPECT_EQ(gauss.layered_geometry()->topology->boundary.data(), boundary_data);
                expect_layered_gauss_payload_capacity(gauss);
                expect_cached_reconstruction(mesh, cache, gauss);
                EXPECT_FALSE(cache.legacy_geometry_materialized());
                EXPECT_FALSE(gauss.legacy_geometry_materialized());
            }
    // A row larger than unsigned char must retain every original incidence.
    using Native = Meshes::SemiStructuredXY_Z;
    Arr<Vec> nodes;
    Arr<unsigned> polygon;
    for (unsigned i = 0; i < 260; ++i)
    {
        const auto angle = 2 * std::numbers::pi_v<real_t> * i / 260;
        nodes.push_back({std::cos(angle), std::sin(angle), 0});
        polygon.push_back(i);
    }
    auto mesh = std::make_shared<Handle>(std::make_shared<Native>(
        nodes, Arr<Arr<unsigned>>{polygon}, Arr<real_t>{0, 0.4, 1.7, 3}));
    Cache cache(mesh);
    GaussCache gauss(mesh);
    expect_layered_gauss_payload_capacity(gauss);
    ASSERT_TRUE(cache.uses_layered_storage());
    ASSERT_TRUE(gauss.uses_layered_storage());
    expect_cached_reconstruction(mesh, cache, gauss);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_FALSE(gauss.legacy_geometry_materialized());
    expect_least_squares_geometry(cache);
    expect_gauss_geometry(gauss);
    EXPECT_EQ(gauss.geometry().offsets[1], 262U);
}

TEST(GradientTraverseTest, LayeredCompatibilityViewsPublishOnceAndSupportCacheMovesAndCopies)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial canonical maps.";
    auto mesh = layered_gradient_mesh(5);
    Cache original(mesh);
    Cache moved(std::move(original));
    Cache cache(mesh);
    cache = std::move(moved);
    GaussCache gauss(mesh);
    GaussCache copy = gauss;
    const auto retained_topology = gauss.layered_geometry()->topology;
    const auto rows = retained_topology->rows;
    const auto interior_offsets = retained_topology->interior_offsets;
    const auto boundary_offsets = retained_topology->boundary_offsets;
    const auto neighbors = retained_topology->other_lids;
    const auto topology_arrays = std::tuple{retained_topology->rows.data(),
        retained_topology->interior_offsets.data(), retained_topology->boundary_offsets.data(),
        retained_topology->other_lids.data(), retained_topology->boundary.data()};
    EXPECT_EQ(copy.layered_geometry()->topology, retained_topology);
    EXPECT_TRUE(cache.uses_layered_storage());
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_EQ(copy.layered_geometry(), gauss.layered_geometry());
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
    EXPECT_EQ(gauss.layered_geometry()->topology, retained_topology);
    EXPECT_EQ(copy.layered_geometry()->topology, retained_topology);
    EXPECT_EQ(retained_topology->rows, rows);
    EXPECT_EQ(retained_topology->interior_offsets, interior_offsets);
    EXPECT_EQ(retained_topology->boundary_offsets, boundary_offsets);
    EXPECT_EQ(retained_topology->other_lids, neighbors);
    EXPECT_EQ((std::tuple{retained_topology->rows.data(), retained_topology->interior_offsets.data(),
        retained_topology->boundary_offsets.data(), retained_topology->other_lids.data(),
        retained_topology->boundary.data()}), topology_arrays);
    expect_layered_gauss_payload_capacity(gauss);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_FALSE(gauss.legacy_geometry_materialized());
    EXPECT_NE(gauss.layered_geometry(), copy.layered_geometry());
    EXPECT_EQ(copy.geometry_epoch(), old_epoch);
    EXPECT_EQ(old_geometry.volumes, old_volumes);
    EXPECT_THROW(copy.geometry(), std::invalid_argument);
    expect_cached_reconstruction(mesh, cache, gauss);
    EXPECT_FALSE(cache.legacy_geometry_materialized());
    EXPECT_FALSE(gauss.legacy_geometry_materialized());
    motion.rollback_trial();
    cache.refresh();
    gauss.refresh();
    EXPECT_EQ(gauss.layered_geometry()->topology, retained_topology);
    EXPECT_EQ(copy.layered_geometry()->topology, retained_topology);
    EXPECT_EQ(retained_topology->rows, rows);
    EXPECT_EQ(retained_topology->interior_offsets, interior_offsets);
    EXPECT_EQ(retained_topology->boundary_offsets, boundary_offsets);
    EXPECT_EQ(retained_topology->other_lids, neighbors);
    EXPECT_EQ((std::tuple{retained_topology->rows.data(), retained_topology->interior_offsets.data(),
        retained_topology->boundary_offsets.data(), retained_topology->other_lids.data(),
        retained_topology->boundary.data()}), topology_arrays);
    expect_layered_gauss_payload_capacity(gauss);
    expect_cached_reconstruction(mesh, cache, gauss);
    expect_least_squares_geometry(cache);
    expect_gauss_geometry(gauss);
    EXPECT_EQ(gauss.geometry().volumes, old_volumes);
}

TEST(GradientTraverseTest, LayeredRefreshFailurePreservesUnmaterializedSnapshot)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial canonical maps.";
    auto mesh = layered_gradient_mesh(5);
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
    const auto* geometry = cache.layered_geometry();
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
    EXPECT_EQ(cache.layered_geometry(), geometry);
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
    EXPECT_THROW(cache.layered_geometry(), std::invalid_argument);
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

TEST(GradientTraverseTest, LayeredGaussInvalidationFailurePreservesRetainedSnapshot)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "XY/Z remains serial only.";
    auto mesh = layered_gradient_mesh(5);
    GaussCache gauss(mesh);
    GaussCache copy = gauss;
    const auto* snapshot = gauss.layered_geometry();
    const auto topology = snapshot->topology;
    const auto volumes = snapshot->volumes;
    const auto epoch = gauss.geometry_epoch();
    const auto storage = gauss.numerical_storage_bytes();
    *mesh = *layered_gradient_mesh(2);
    // Unsupported backend/topology replacement must fail before publishing
    // either topology or metrics, preserving snapshots retained by copies.
    EXPECT_THROW(gauss.refresh(), std::logic_error);
    EXPECT_EQ(gauss.layered_geometry(), snapshot);
    EXPECT_EQ(copy.layered_geometry(), snapshot);
    EXPECT_EQ(gauss.layered_geometry()->topology, topology);
    EXPECT_EQ(gauss.layered_geometry()->volumes, volumes);
    EXPECT_EQ(gauss.geometry_epoch(), epoch);
    EXPECT_EQ(gauss.numerical_storage_bytes(), storage);
    EXPECT_FALSE(gauss.legacy_geometry_materialized());
}

TEST(GradientTraverseTest, LayeredSplitConstructionMapsInterleavedOwnedRowsExactly)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "XY/Z remains serial only.";
    for (size_t layers : {1U, 2U, 5U})
    {
        auto mesh = layered_gradient_mesh(layers);
        std::vector<Local> inner, boundary;
        std::vector<std::vector<Local>> neighbors(mesh->num_owned_cells());
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            bool exterior = false;
            for (auto face : mesh->faces(static_cast<Local>(cell)))
            {
                const bool face_exterior = mesh->is_geometry_exterior_face(face);
                exterior = exterior || face_exterior;
                if (!face_exterior)
                    neighbors[cell].push_back(mesh->owner_cell(face) == static_cast<Local>(cell)
                        ? mesh->neighbor_cell(face) : mesh->owner_cell(face));
            }
            (exterior ? boundary : inner).push_back(static_cast<Local>(cell));
        }
        if (layers < 3) EXPECT_TRUE(inner.empty());
        else
        {
            ASSERT_FALSE(inner.empty());
            ASSERT_FALSE(boundary.empty());
            EXPECT_LT(boundary.front(), inner.front());
            EXPECT_GT(boundary.back(), inner.back());
        }
        Cache cache(mesh);
        GaussCache gauss(mesh);
        const auto check_mapping = [&]
        {
            const auto* compact = cache.layered_geometry();
            ASSERT_NE(compact, nullptr);
            size_t offset = 0;
            for (const auto* group : {&inner, &boundary})
                for (Local cell : *group)
                {
                    const auto& row = compact->rows[cell];
                    EXPECT_EQ(row.interior_begin, offset);
                    EXPECT_EQ(row.interior_count, neighbors[cell].size());
                    for (size_t slot = 0; slot < row.interior_count; ++slot)
                        EXPECT_EQ(compact->interior[offset + slot].other_lid, neighbors[cell][slot]);
                    offset += row.interior_count;
                }
            EXPECT_EQ(offset, compact->interior.size());
            size_t visited = 0;
            compact->template visit_rows<false>([&](size_t cell, const auto& row)
            {
                EXPECT_EQ(cell, visited++);
                ASSERT_EQ(row.interior_samples.size(), neighbors[cell].size());
                size_t slot = 0;
                for (const auto& sample : row.interior_samples)
                    EXPECT_EQ(sample.other_lid, neighbors[cell][slot++]);
            });
            EXPECT_EQ(visited, mesh->num_owned_cells());
        };
        check_mapping();
        expect_cached_reconstruction(mesh, cache, gauss);
        EXPECT_FALSE(cache.legacy_geometry_materialized());
        expect_least_squares_geometry(cache);
        const auto original = snapshot(cache.interior_geometry());
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.2, 0.2);
        cache.refresh(); gauss.refresh();
        check_mapping();
        expect_cached_reconstruction(mesh, cache, gauss);
        expect_least_squares_geometry(cache);
        motion.rollback_trial();
        cache.refresh(); gauss.refresh();
        check_mapping();
        expect_cached_reconstruction(mesh, cache, gauss);
        expect_least_squares_geometry(cache);
        EXPECT_EQ(snapshot(cache.interior_geometry()), original);
    }
}

