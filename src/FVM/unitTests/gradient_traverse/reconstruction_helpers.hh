#pragma once

// Helpers retain translation-unit-local linkage. Register the test
// environment only in the executable's top-level source.

#include "geometry_reference.hh"

namespace
{
using Scalar = ScalarCellFieldStored<Pack>;
using Vector = VectorCellFieldStored<Pack>;
using Tensor = TensorCellFieldStored<Pack>;
using BoundaryCalls = std::vector<std::tuple<char, int, size_t>>;

real_t nonlinear_scalar(Vec point)
{
    return 0.7 + point.x * point.x - 0.3 * point.x * point.y + 0.2 * point.z * point.z;
}

Vec nonlinear_vector(Vec point)
{
    return {nonlinear_scalar(point), 0.2 + point.y * point.z, point.z * point.z - 0.4 * point.x};
}

void expect_same_owned(const auto& actual, const auto& expected)
{
    const auto a = actual.owned_read_view(), b = expected.owned_read_view();
    ASSERT_EQ(a.extent(0), b.extent(0));
    ASSERT_EQ(a.extent(1), b.extent(1));
    for (size_t cell = 0; cell < a.extent(0); ++cell)
        for (size_t component = 0; component < a.extent(1); ++component)
        {
            SCOPED_TRACE(::testing::Message() << "cell=" << cell << " component=" << component);
            expect_roundoff_equal(a(cell, component), b(cell, component));
        }
}

// Preserve the previous pointwise cached kernels as independent arithmetic
// oracles. These deliberately read the compatibility view of a separate cache,
// so evaluating the cache under test cannot accidentally materialize its view.
void point_cached_scalar(const Scalar& field, Vector& gradient, const Cache& cache,
    bool boundaries, auto boundary_condition, auto boundary_value)
{
    const auto& interior = cache.interior_geometry();
    const auto& boundary = cache.boundary_geometry();
    for (size_t cell = 0; cell < interior.size(); ++cell)
    {
        const auto& geometry = boundaries && !boundary[cell].boundary_samples.empty()
            ? boundary[cell] : interior[cell];
        const auto center = field.value(cell);
        Vec value{};
        for (const auto& sample : geometry.interior_samples)
            value = value + sample.weight * (field.local_value(sample.other_lid) - center);
        for (const auto& sample : geometry.boundary_samples)
        {
            const auto condition = boundary_condition(sample.location.batch_id, sample.location.in_batch_id);
            const auto delta = condition.type == BoundaryConditionType::Dirichlet
                ? boundary_value(sample.location.batch_id, sample.location.in_batch_id) - center
                : condition.value * sample.normal_distance;
            value = value + sample.weight * delta;
        }
        gradient.set_owned_value(cell, value);
    }
}

void point_cached_vector(const Vector& field, Tensor& gradient, const Cache& cache,
    bool boundaries, auto boundary_value)
{
    const auto& interior = cache.interior_geometry();
    const auto& boundary = cache.boundary_geometry();
    for (size_t cell = 0; cell < interior.size(); ++cell)
    {
        const auto& geometry = boundaries && !boundary[cell].boundary_samples.empty()
            ? boundary[cell] : interior[cell];
        const auto center = field.value(cell);
        std::array<Vec, 3> value{};
        for (const auto& sample : geometry.interior_samples)
        {
            const auto delta = field.local_value(sample.other_lid) - center;
            for (size_t component = 0; component < 3; ++component)
                value[component] = value[component] + sample.weight * delta.component(component);
        }
        for (const auto& sample : geometry.boundary_samples)
        {
            const auto delta = boundary_value(sample.location.batch_id, sample.location.in_batch_id) - center;
            for (size_t component = 0; component < 3; ++component)
                value[component] = value[component] + sample.weight * delta.component(component);
        }
        gradient.set_owned_value(cell, value);
    }
}

void expect_cached_reconstruction(SP<const Handle> mesh, const Cache& cache, const GaussCache& gauss)
{
    Scalar scalar(mesh, "structured_scalar");
    Vector vector(mesh, "structured_vector"), actual(mesh, Vec{-91, -91, -91}, "actual"),
        expected(mesh, "expected");
    Tensor actual_tensor(mesh, "actual_tensor"), expected_tensor(mesh, "expected_tensor");
    Cache oracle(mesh);
    expect_least_squares_geometry(oracle);
    BoundaryCalls actual_calls, expected_calls;
    BoundaryCalls* calls = &actual_calls;
    int phase = 0;
    auto condition = [&](int batch, size_t index)
    {
        calls->emplace_back('c', batch, index);
        return phase == 1 || ((static_cast<size_t>(batch) + index) % 2 == 0)
            ? BoundaryCondition{BoundaryConditionType::Neumann, phase ? -0.41 : 0.13}
            : BoundaryCondition{BoundaryConditionType::Dirichlet, 0};
    };
    auto scalar_boundary = [&](int batch, size_t index)
    {
        calls->emplace_back('s', batch, index);
        const auto face = mesh->boundary_face_batch(batch).face_lids[index];
        return nonlinear_scalar(mesh->face_centroid(face)) + phase * 0.8
            + 0.1 * scalar.value(mesh->owner_cell(face));
    };
    auto vector_boundary = [&](int batch, size_t index)
    {
        calls->emplace_back('v', batch, index);
        const auto face = mesh->boundary_face_batch(batch).face_lids[index];
        return nonlinear_vector(mesh->face_centroid(face)) + Vec{0.2, -0.7, 0.4} * phase
            + vector.value(mesh->owner_cell(face)) * 0.1;
    };
    for (; phase < 2; ++phase)
    {
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto center = mesh->cell_centroid(cell);
            scalar.set_owned_value(cell, nonlinear_scalar(center) + phase * 1.3);
            vector.set_owned_value(cell, nonlinear_vector(center) + Vec{1.3, 0.2, -0.8} * phase);
        }
        scalar.sync_ghosts();
        vector.sync_ghosts();
        // Centers in cached LS use owned values; neighbors and Gauss use the
        // previously published overlap values, even for owned local entries.
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            scalar.set_owned_value(cell, scalar.value(cell) + 2.1);
            vector.set_owned_value(cell, vector.value(cell) + Vec{0.4, -0.8, 1.3});
        }
        for (bool boundaries : {false, true})
        {
            actual_calls.clear();
            expected_calls.clear();
            calls = &actual_calls;
            if (boundaries)
            {
                FVM::cell_gradient(scalar, condition, scalar_boundary, actual, cache);
                FVM::cell_gradient(vector, vector_boundary, actual_tensor, cache);
            }
            else
            {
                FVM::cell_gradient(scalar, actual, cache);
                FVM::cell_gradient(vector, actual_tensor, cache);
            }
            calls = &expected_calls;
            point_cached_scalar(scalar, expected, oracle, boundaries, condition, scalar_boundary);
            point_cached_vector(vector, expected_tensor, oracle, boundaries, vector_boundary);
            EXPECT_EQ(actual_calls, expected_calls);
            expect_same_owned(actual, expected);
            expect_same_owned(actual_tensor, expected_tensor);
        }
        actual_calls.clear();
        expected_calls.clear();
        calls = &actual_calls;
        FVM::gauss_linear_cell_gradient(scalar, condition, scalar_boundary, actual, gauss);
        FVM::gauss_linear_cell_gradient(vector, vector_boundary, actual_tensor, gauss);
        calls = &expected_calls;
        FVM::detail::stored_gauss_linear_cell_gradient_reference(scalar, condition, scalar_boundary, expected);
        FVM::detail::stored_gauss_linear_cell_gradient_reference(vector, vector_boundary, expected_tensor);
        EXPECT_EQ(actual_calls, expected_calls);
        expect_same_owned(actual, expected);
        expect_same_owned(actual_tensor, expected_tensor);
        const auto unpublished = actual.local_read_view();
        for (size_t cell = 0; cell < unpublished.extent(0); ++cell)
            for (size_t component = 0; component < unpublished.extent(1); ++component)
                EXPECT_EQ(unpublished(cell, component), -91);
    }
}

// Mixed XY degrees with a genuine interior column and repeated side patches.
// The reserved-name fixture deliberately probes the existing provider rejection.
auto layered_gradient_mesh(size_t layers, bool triangles = false, bool reserved = false)
{
    using Native = Meshes::SemiStructuredXY_Z;
    Arr<Vec> nodes;
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) nodes.push_back({real_t(x), real_t(y), 0});
    Arr<Arr<unsigned>> cells;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
        {
            const unsigned a = x + 4 * y, b = a + 1, c = b + 4, d = a + 4;
            if (triangles || (x == 0 && y == 0))
            {
                cells.push_back({a, b, d});
                cells.push_back({b, c, d});
            }
            else cells.push_back({a, b, c, d});
        }
    Arr<Native::BoundaryEdge> edges;
    for (unsigned i = 0; i < 3; ++i)
    {
        edges.push_back({i, i + 1, reserved ? "zmin" : "south"});
        edges.push_back({12 + i, 13 + i, "north"});
        edges.push_back({4 * i, 4 * (i + 1), i % 2 ? "odd" : "even"});
        edges.push_back({3 + 4 * i, 3 + 4 * (i + 1), i % 2 ? "odd" : "even"});
    }
    Arr<real_t> z{0};
    for (size_t layer = 0; layer < layers; ++layer) z.push_back(z.back() + 0.4 + 0.13 * layer);
    return std::make_shared<Handle>(std::make_shared<Native>(nodes, cells, z, edges));
}

void expect_native_cache_layout(SP<const Handle> mesh, bool structured)
{
    Cache cache(mesh);
    GaussCache gauss(mesh);
    EXPECT_EQ(cache.uses_structured_storage(), structured);
    EXPECT_EQ(gauss.uses_structured_storage(), structured);
    EXPECT_EQ(cache.legacy_geometry_materialized(), !structured);
    EXPECT_EQ(gauss.legacy_geometry_materialized(), !structured);
    expect_cached_reconstruction(mesh, cache, gauss);
    EXPECT_EQ(cache.legacy_geometry_materialized(), !structured);
    EXPECT_EQ(gauss.legacy_geometry_materialized(), !structured);
    expect_least_squares_geometry(cache);
    expect_gauss_geometry(gauss);
    EXPECT_TRUE(cache.legacy_geometry_materialized());
    EXPECT_TRUE(gauss.legacy_geometry_materialized());
    const auto expected_locations = FVM::detail::boundary_face_locations(*mesh);
    const auto& locations = cache.boundary_locations();
    ASSERT_EQ(locations.size(), expected_locations.size());
    for (size_t face = 0; face < locations.size(); ++face)
    {
        EXPECT_EQ(locations[face].active, expected_locations[face].active);
        EXPECT_EQ(locations[face].batch_id, expected_locations[face].batch_id);
        EXPECT_EQ(locations[face].in_batch_id, expected_locations[face].in_batch_id);
    }
}
} // namespace

