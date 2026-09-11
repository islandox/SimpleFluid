/** @file testStoredGradientViews.cc @brief Bulk field and resolved region gradient contracts. */
#include "FVM/CellOperators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<>;
using Scalar = ScalarCellFieldStored<Pack>;
using Vector = VectorCellFieldStored<Pack>;
using Tensor = TensorCellFieldStored<Pack>;
using Cache = FVM::CellGradientCache<Pack, Handle>;
using Vec = MeshUtils::Vec3;
using Calls = std::vector<std::tuple<char, int, size_t>>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

real_t scalar_value(Vec p)
{
    return 0.7 + p.x * p.x - 0.3 * p.x * p.y + std::sin(p.z);
}
Vec vector_value(Vec p)
{
    return {scalar_value(p), 0.2 + p.y * p.z, p.z * p.z - 0.4 * p.x};
}

auto condition(const Handle& mesh, int batch, size_t index)
{
    const auto face = mesh.boundary_face_batch(batch).face_lids[index];
    return mesh.face_normal(face).z < -0.5 ? BoundaryCondition{BoundaryConditionType::Neumann, 0.13}
                                           : BoundaryCondition{BoundaryConditionType::Dirichlet, 0};
}

void initialize(Scalar& scalar, Vector& vector)
{
    const auto& mesh = scalar.mesh();
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        scalar.set_owned_value(c, scalar_value(mesh.cell_centroid(c)));
        vector.set_owned_value(c, vector_value(mesh.cell_centroid(c)));
    }
    scalar.sync_ghosts();
    vector.sync_ghosts();
    // Deliberately leave both local owned copies and remote ghosts published
    // from the previous state. Cached centers read owned; neighbors and all
    // uncached/Gauss samples continue to read that published overlap state.
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        scalar.set_owned_value(c, scalar.value(c) + 2.1);
        vector.set_owned_value(c, vector.value(c) + Vec{0.4, -0.8, 1.3});
    }
}

void expect_same(const auto& actual, const auto& expected)
{
    const auto a = actual.owned_read_view(), b = expected.owned_read_view();
    ASSERT_EQ(a.extent(0), b.extent(0));
    ASSERT_EQ(a.extent(1), b.extent(1));
    for (size_t c = 0; c < a.extent(0); ++c)
        for (size_t k = 0; k < a.extent(1); ++k)
            EXPECT_EQ(a(c, k), b(c, k)) << "row " << c << " component " << k;
}

void expect_unpublished(const auto& field, real_t sentinel)
{
    const auto values = field.local_read_view();
    for (size_t c = 0; c < values.extent(0); ++c)
        for (size_t k = 0; k < values.extent(1); ++k)
            EXPECT_EQ(values(c, k), sentinel);
}

// A point-access oracle is intentionally retained here: it verifies exactly
// which field storage each bulk read borrows, independent of host view access.
void point_scalar_gradient(const Scalar& field, Vector& output, const Cache& cache, bool cached, bool boundaries)
{
    const auto& mesh = field.mesh();
    auto bc = [&](int b, size_t i) { return condition(mesh, b, i); };
    auto bv = [&](int b, size_t i)
    { return scalar_value(mesh.face_centroid(mesh.boundary_face_batch(b).face_lids[i])); };
    if (!cached)
    {
        if (boundaries)
        {
            const auto stencils = FVM::detail::scalar_affine_gradient_stencils(mesh, bc, bv);
            for (size_t c = 0; c < stencils.size(); ++c)
            {
                auto gradient = stencils[c].constant;
                for (const auto& entry : stencils[c].entries)
                    gradient = gradient + entry.coefficient * field.local_value(entry.cell_lid);
                output.set_owned_value(c, gradient);
            }
        }
        else
        {
            const auto stencils = FVM::detail::least_squares_gradient_stencils(mesh);
            for (size_t c = 0; c < stencils.size(); ++c)
            {
                Vec gradient{};
                for (const auto& entry : stencils[c])
                    gradient = gradient + entry.coefficient * field.local_value(entry.cell_lid);
                output.set_owned_value(c, gradient);
            }
        }
        return;
    }
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto& geometry = boundaries && !cache.boundary_geometry()[c].boundary_samples.empty()
                                   ? cache.boundary_geometry()[c]
                                   : cache.interior_geometry()[c];
        const auto center = field.value(c);
        Vec gradient{};
        for (const auto& sample : geometry.interior_samples)
            gradient = gradient + sample.weight * (field.local_value(sample.other_lid) - center);
        for (const auto& sample : geometry.boundary_samples)
        {
            const auto cond = bc(sample.location.batch_id, sample.location.in_batch_id);
            const auto delta = cond.type == BoundaryConditionType::Dirichlet
                                   ? bv(sample.location.batch_id, sample.location.in_batch_id) - center
                                   : cond.value * sample.normal_distance;
            gradient = gradient + sample.weight * delta;
        }
        output.set_owned_value(c, gradient);
    }
}

void compare_gauss(const SP<const Handle>& mesh)
{
    const auto indexer = mesh->has_materialized_indexer();
    Scalar scalar(mesh, "scalar");
    Vector vector(mesh, "vector"), actual(mesh, Vec{-91, -91, -91}, "actual"), expected(mesh, "expected");
    Tensor actual_tensor(mesh, "actual_tensor"), expected_tensor(mesh, "expected_tensor");
    initialize(scalar, vector);
    Calls actual_calls, expected_calls;
    auto run = [&](bool reference, Calls& calls, Vector& scalar_gradient, Tensor& vector_gradient)
    {
        auto bc = [&](int b, size_t i)
        {
            calls.emplace_back('c', b, i);
            return condition(*mesh, b, i);
        };
        auto bv = [&](int b, size_t i)
        {
            calls.emplace_back('s', b, i);
            const auto face = mesh->boundary_face_batch(b).face_lids[i];
            // Callbacks may still read the original field while views live.
            return scalar_value(mesh->face_centroid(face)) + 0.1 * scalar.value(mesh->owner_cell(face));
        };
        auto vv = [&](int b, size_t i)
        {
            calls.emplace_back('v', b, i);
            const auto face = mesh->boundary_face_batch(b).face_lids[i];
            return vector_value(mesh->face_centroid(face)) + vector.value(mesh->owner_cell(face)) * 0.1;
        };
        if (reference)
        {
            FVM::detail::stored_gauss_linear_cell_gradient_reference(scalar, bc, bv, scalar_gradient);
            FVM::detail::stored_gauss_linear_cell_gradient_reference(vector, vv, vector_gradient);
        }
        else
        {
            FVM::gauss_linear_cell_gradient(scalar, bc, bv, scalar_gradient);
            FVM::gauss_linear_cell_gradient(vector, vv, vector_gradient);
        }
    };
    run(false, actual_calls, actual, actual_tensor);
    run(true, expected_calls, expected, expected_tensor);
    EXPECT_EQ(actual_calls, expected_calls);
    expect_same(actual, expected);
    expect_same(actual_tensor, expected_tensor);
    expect_unpublished(actual, -91);
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
    EXPECT_EQ(mesh->has_materialized_indexer(), indexer);
}
} // namespace

TEST(StoredGradientViewsTest, CachedAndUncachedScalarReadsPreserveOwnedAndOverlapState)
{
    for (const auto& geometry : {test::two_regions(2), test::coarse_fine_regions(), test::periodic_regions()})
    {
        SP<const Handle> mesh = std::make_shared<Handle>(geometry);
        Scalar scalar(mesh, "scalar");
        Vector vector(mesh, "vector");
        initialize(scalar, vector);
        Cache cache(mesh);
        for (bool cached : {false, true})
            for (bool boundaries : {false, true})
            {
                Vector actual(mesh, Vec{-77, -77, -77}, "actual"), expected(mesh, "expected");
                auto bc = [&](int b, size_t i) { return condition(*mesh, b, i); };
                auto bv = [&](int b, size_t i)
                { return scalar_value(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
                point_scalar_gradient(scalar, expected, cache, cached, boundaries);
                if (cached && boundaries)
                    FVM::cell_gradient(scalar, bc, bv, actual, cache);
                else if (cached)
                    FVM::cell_gradient(scalar, actual, cache);
                else if (boundaries)
                    FVM::cell_gradient(scalar, bc, bv, actual);
                else
                    FVM::cell_gradient(scalar, actual);
                expect_same(actual, expected);
                expect_unpublished(actual, -77);
            }
    }
}

TEST(StoredGradientViewsTest, RegionGaussPreservesExactOrderForAllProvidersAndInterfaces)
{
    for (const auto& geometry : {test::two_regions(3), test::mixed_regions(), test::coarse_fine_regions(),
             test::periodic_regions(), test::cylindrical_regions(), test::independent_extruded_regions()})
        compare_gauss(std::make_shared<Handle>(geometry));
}

TEST(StoredGradientViewsTest, MotionRefreshRollbackAndStaleCachePreserveGradientContracts)
{
    for (const auto& geometry : {test::two_regions(), test::coarse_fine_regions(), test::periodic_regions(),
             test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        auto mesh = std::make_shared<Handle>(geometry);
        Cache cache(mesh);
        Scalar scalar(mesh, 1.0, "scalar");
        Vector gradient(mesh, Vec{-39, -39, -39}, "gradient");
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.35, 0.2);
        motion.accept_trial();
        EXPECT_THROW(FVM::cell_gradient(scalar, gradient, cache), std::invalid_argument);
        expect_unpublished(gradient, -39);
        cache.refresh();
        EXPECT_NO_THROW(FVM::cell_gradient(scalar, gradient, cache));
        compare_gauss(mesh);
        motion.begin_trial(0.8, 0.2);
        compare_gauss(mesh);
        motion.rollback_trial();
        compare_gauss(mesh);
    }
}

TEST(StoredGradientViewsTest, EmptyRanksAndNativeFallbackUseExistingMaps)
{
    compare_gauss(std::make_shared<Handle>(test::two_regions(1)));
    auto native = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 1, 1.5, 2}, {0, 0.5, 1}, {0, 0.5, 1}}}));
    compare_gauss(native);
    if (Tpetra::getDefaultComm()->getSize() == 1)
    {
        auto layout = MeshReorderingFactory<>::selected_cells_first(
            std::move(native), [](auto, const Vec& center) { return center.x > 1; });
        EXPECT_FALSE(layout.mesh->supports_region_execution());
        compare_gauss(layout.mesh);
    }
}

TEST(StoredGradientViewsTest, InvalidBoundaryAndWrongMeshDoNotBypassValidation)
{
    auto mesh = std::make_shared<Handle>(test::two_regions(1));
    auto other = std::make_shared<Handle>(test::two_regions(1));
    Scalar scalar(mesh, 1.0, "scalar");
    Vector actual(mesh, "actual"), wrong(other, "wrong");
    Cache cache(mesh);
    EXPECT_THROW(FVM::cell_gradient(scalar, wrong, cache), std::invalid_argument);
    EXPECT_THROW(FVM::gauss_linear_cell_gradient(scalar, wrong), std::invalid_argument);
    if (mesh->num_owned_cells())
        EXPECT_THROW(FVM::gauss_linear_cell_gradient(
                         scalar, [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Robin, 1}; },
                         [](int, size_t) { return 1.0; }, actual),
            std::invalid_argument);
}

TEST(StoredGradientViewsTest, VectorLeastSquaresMatchesComponentReadsWithStaleOverlap)
{
    SP<const Handle> mesh = std::make_shared<Handle>(test::two_regions(2));
    Scalar scalar(mesh, "scalar");
    Vector vector(mesh, "vector");
    initialize(scalar, vector);
    Cache cache(mesh);
    auto boundary = [&](int b, size_t i)
    { return vector_value(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
    for (bool cached : {false, true})
        for (bool boundaries : {false, true})
        {
            Tensor actual(mesh, "actual");
            if (cached && boundaries)
                FVM::cell_gradient(vector, boundary, actual, cache);
            else if (cached)
                FVM::cell_gradient(vector, actual, cache);
            else if (boundaries)
                FVM::cell_gradient(vector, boundary, actual);
            else
                FVM::cell_gradient(vector, actual);
            for (size_t component = 0; component < 3; ++component)
            {
                for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                    scalar.set_owned_value(c, vector.value(c).component(component));
                {
                    auto local = scalar.local_write_view();
                    for (size_t c = 0; c < mesh->num_local_cells(); ++c)
                        local(c, 0) = vector.local_value(c).component(component);
                }
                Vector expected(mesh, "expected");
                auto bc = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 0}; };
                auto bv = [&](int b, size_t i) { return boundary(b, i).component(component); };
                if (cached && boundaries)
                    FVM::cell_gradient(scalar, bc, bv, expected, cache);
                else if (cached)
                    FVM::cell_gradient(scalar, expected, cache);
                else if (boundaries)
                    FVM::cell_gradient(scalar, bc, bv, expected);
                else
                    FVM::cell_gradient(scalar, expected);
                for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                {
                    const auto value = actual.value(c)[component], reference = expected.value(c);
                    EXPECT_EQ(value.x, reference.x);
                    EXPECT_EQ(value.y, reference.y);
                    EXPECT_EQ(value.z, reference.z);
                }
            }
        }
}

TEST(StoredGradientViewsTest, ScalarSourceMayBorrowAnOutputComponent)
{
    SP<const Handle> mesh = std::make_shared<Handle>(test::two_regions(2));
    for (bool cached : {false, true})
    {
        Scalar scalar(mesh, "scalar");
        Vector actual(mesh, "aliased_output"), vector(mesh, "vector"), expected(mesh, "expected");
        // FieldStored exposes Tpetra storage; a vector column can be retained
        // by a scalar source. Borrowed host views must not assume non-aliasing.
        scalar.owned_data() = *actual.owned_data().getVectorNonConst(0);
        ASSERT_EQ(scalar.owned_read_view().data(), actual.owned_read_view().data());
        initialize(scalar, vector);
        Cache cache(mesh);
        point_scalar_gradient(scalar, expected, cache, cached, false);
        if (cached)
            FVM::cell_gradient(scalar, actual, cache);
        else
            FVM::cell_gradient(scalar, actual);
        expect_same(actual, expected);
    }
}

TEST(StoredGradientViewsTest, PublishedSourceAliasObservesEarlierOutputRows)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "The output owned map aliases the source overlap map in serial.";
    SP<const Handle> mesh = std::make_shared<Handle>(test::two_regions(2));
    for (int mode = 0; mode < 3; ++mode)
    {
        Scalar source(mesh, "source"), reference_source(mesh, "reference_source");
        Vector actual(mesh, "actual"), expected(mesh, "expected"), vector(mesh, "vector");
        source.overlap_data() = *actual.owned_data().getVectorNonConst(0);
        reference_source.overlap_data() = *expected.owned_data().getVectorNonConst(0);
        ASSERT_EQ(source.local_read_view().data(), actual.owned_read_view().data());
        initialize(source, vector);
        initialize(reference_source, vector);
        Cache cache(mesh);
        if (mode < 2)
        {
            point_scalar_gradient(reference_source, expected, cache, mode == 1, false);
            if (mode == 1)
                FVM::cell_gradient(source, actual, cache);
            else
                FVM::cell_gradient(source, actual);
        }
        else
        {
            auto bc = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; };
            auto bv = [](int, size_t) { return 0.0; };
            FVM::detail::stored_gauss_linear_cell_gradient_reference(reference_source, bc, bv, expected);
            FVM::gauss_linear_cell_gradient(source, bc, bv, actual);
        }
        expect_same(actual, expected);
    }
}
