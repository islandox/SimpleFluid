/** @file testGradientTraverse.cc
 * @brief Ordered geometry and retained traversal contracts for gradient builds.
 */
#include "FVM/CellOperators.hh"
#include "FVM/FaceFlux.hh"
#include "FVM/TransportSystem.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <future>
#include <latch>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Local = Pack::local_ordinal_type;
using Vec = MeshUtils::Vec3;
using Cache = FVM::CellGradientCache<Pack, Handle>;
using GaussCache = FVM::GaussLinearGradientCache<Pack, Handle>;
using TransportCache = FVM::TransportGeometryCache<Handle>;
using PressureWorkspace = FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Handle>;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void expect_exact(Vec actual, Vec expected)
{
    EXPECT_EQ(actual.x, expected.x);
    EXPECT_EQ(actual.y, expected.y);
    EXPECT_EQ(actual.z, expected.z);
}

struct CompositeFixture
{
    SP<Meshes::MultiRegionMesh> geometry;
    bool planar_faces;
};

auto composite_fixtures()
{
    return std::array{CompositeFixture{test::two_regions(3), true}, CompositeFixture{test::mixed_regions(), true},
        CompositeFixture{test::coarse_fine_regions(), true}, CompositeFixture{test::periodic_regions(), true},
        CompositeFixture{test::cylindrical_regions(), false},
        CompositeFixture{test::independent_extruded_regions(), true}};
}

auto normal_matrix(const auto& samples)
{
    std::array<std::array<real_t, 3>, 3> normal{};
    for (const auto& sample : samples)
        for (size_t row = 0; row < 3; ++row)
            for (size_t column = row; column < 3; ++column)
                normal[row][column] += sample.direction.component(row) * sample.direction.component(column);
    for (size_t row = 1; row < 3; ++row)
        for (size_t column = 0; column < row; ++column)
            normal[row][column] = normal[column][row];
    return normal;
}

// The reference uses the original native-dispatch sample visitor, independently
// of traverse selection and incidence descriptors. Preserve the numerical
// backend and arithmetic order so even one changed sample is observable.
template<class MeshType>
void expect_least_squares_geometry(const FVM::CellGradientCache<Pack, MeshType>& cache,
    const typename FVM::CellGradientCache<Pack, MeshType>::boundary_direction_provider_type& boundary_direction = {})
{
    const auto& mesh = *cache.mesh_ptr();
    const auto locations = FVM::detail::boundary_face_locations(mesh);
    for (bool boundaries : {false, true})
    {
        const auto& geometry = boundaries ? cache.boundary_geometry() : cache.interior_geometry();
        ASSERT_EQ(geometry.size(), mesh.num_owned_cells());
        for (size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
        {
            SCOPED_TRACE(::testing::Message() << "cell=" << cell << " boundaries=" << boundaries);
            std::vector<FVM::detail::GradientGeometrySample<MeshType>> samples;
            bool has_boundary = false;
            FVM::detail::visit_gradient_geometry_samples(mesh, static_cast<Local>(cell), [&](auto sample)
            {
                if (!sample.interior && !locations.at(sample.face_lid).active) return;
                has_boundary = has_boundary || !sample.interior;
                if (!sample.interior && boundary_direction)
                    sample.direction = boundary_direction(sample.face_lid, static_cast<Local>(cell));
                samples.push_back(sample);
            }, boundaries);
            if (boundaries && !has_boundary)
            {
                EXPECT_TRUE(geometry[cell].interior_samples.empty());
                EXPECT_TRUE(geometry[cell].boundary_samples.empty());
                continue;
            }
            const auto normal = normal_matrix(samples);
            std::array<Vec, 3> inverse;
            for (size_t column = 0; column < 3; ++column)
            {
                auto matrix = normal;
                Vec basis{};
                basis.component(column) = 1;
                inverse[column] = FVM::detail::solve_3x3(matrix, basis);
            }
            size_t interior_index = 0, boundary_index = 0;
            for (const auto& sample : samples)
            {
                const auto weight = inverse[0] * sample.direction.x + inverse[1] * sample.direction.y
                    + inverse[2] * sample.direction.z;
                if (sample.interior)
                {
                    ASSERT_LT(interior_index, geometry[cell].interior_samples.size());
                    const auto& actual = geometry[cell].interior_samples[interior_index++];
                    EXPECT_EQ(actual.other_lid, sample.other_lid);
                    expect_exact(actual.weight, weight);
                }
                else
                {
                    ASSERT_LT(boundary_index, geometry[cell].boundary_samples.size());
                    const auto& actual = geometry[cell].boundary_samples[boundary_index++];
                    const auto& location = locations.at(sample.face_lid);
                    EXPECT_EQ(actual.location.active, location.active);
                    EXPECT_EQ(actual.location.batch_id, location.batch_id);
                    EXPECT_EQ(actual.location.in_batch_id, location.in_batch_id);
                    EXPECT_EQ(actual.normal_distance, sample.normal_distance);
                    expect_exact(actual.weight, weight);
                }
            }
            EXPECT_EQ(interior_index, geometry[cell].interior_samples.size());
            EXPECT_EQ(boundary_index, geometry[cell].boundary_samples.size());
        }
    }
}

// Stencil builders solve each direction directly, whereas the reusable cache
// multiplies inverse columns. Check each builder's original arithmetic exactly.
template<class MeshType>
void expect_stencil_geometry(const MeshType& mesh)
{
    const auto locations = FVM::detail::boundary_face_locations(mesh);
    const auto interior = FVM::detail::least_squares_gradient_stencils(mesh);
    const auto boundary = FVM::detail::boundary_aware_gradient_geometry(mesh, locations);
    ASSERT_EQ(interior.size(), mesh.num_owned_cells());
    ASSERT_EQ(boundary.size(), mesh.num_owned_cells());
    for (bool boundaries : {false, true})
        for (size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
        {
            std::vector<FVM::detail::GradientGeometrySample<MeshType>> samples;
            FVM::detail::visit_gradient_geometry_samples(mesh, static_cast<Local>(cell), [&](auto sample)
            {
                if (sample.interior || locations.at(sample.face_lid).active) samples.push_back(sample);
            }, boundaries);
            const auto normal = normal_matrix(samples);
            std::unordered_map<Local, Vec> coefficients;
            if (!boundaries) coefficients.reserve(samples.size() + 1);
            size_t boundary_index = 0;
            for (const auto& sample : samples)
            {
                auto matrix = normal;
                auto direction = sample.direction;
                const auto basis = FVM::detail::solve_3x3(matrix, direction);
                if (sample.interior)
                {
                    coefficients[sample.other_lid] = coefficients[sample.other_lid] + basis;
                    coefficients[static_cast<Local>(cell)] = coefficients[static_cast<Local>(cell)]
                        + Vec{-basis.x, -basis.y, -basis.z};
                }
                else
                {
                    ASSERT_LT(boundary_index, boundary[cell].boundary_samples.size());
                    const auto& actual = boundary[cell].boundary_samples[boundary_index++];
                    EXPECT_EQ(actual.face_lid, sample.face_lid);
                    EXPECT_EQ(actual.location.active, locations.at(sample.face_lid).active);
                    EXPECT_EQ(actual.location.batch_id, locations.at(sample.face_lid).batch_id);
                    EXPECT_EQ(actual.location.in_batch_id, locations.at(sample.face_lid).in_batch_id);
                    EXPECT_EQ(actual.normal_distance, sample.normal_distance);
                    expect_exact(actual.basis, basis);
                }
            }
            const auto& actual = boundaries ? boundary[cell].interior_entries : interior[cell];
            ASSERT_EQ(actual.size(), coefficients.size());
            size_t entry = 0;
            for (const auto& [id, coefficient] : coefficients)
            {
                EXPECT_EQ(actual[entry].cell_lid, id);
                expect_exact(actual[entry++].coefficient, coefficient);
            }
            if (boundaries) EXPECT_EQ(boundary_index, boundary[cell].boundary_samples.size());
        }
}

template<class MeshType>
void expect_gauss_geometry(const FVM::GaussLinearGradientCache<Pack, MeshType>& cache)
{
    const auto& mesh = *cache.mesh_ptr();
    const auto& actual = cache.geometry();
    const auto locations = FVM::detail::boundary_face_locations(mesh);
    ASSERT_EQ(actual.volumes.size(), mesh.num_owned_cells());
    ASSERT_EQ(actual.offsets.size(), mesh.num_owned_cells() + 1);
    size_t entry = 0;
    for (size_t cell = 0; cell < mesh.num_owned_cells(); ++cell)
    {
        const auto cell_id = FVM::detail::query_cell_id(mesh, static_cast<Local>(cell));
        EXPECT_EQ(actual.offsets[cell], entry);
        EXPECT_EQ(actual.volumes[cell], mesh.cell_volume(cell_id));
        for (const auto face_id : mesh.faces(cell_id))
        {
            ASSERT_LT(entry, actual.faces.size());
            const auto& face = actual.faces[entry++];
            EXPECT_EQ(face.face_lid, FVM::detail::packed_face_local_id(mesh, face_id));
            expect_exact(face.area, mesh.face_area_vector_outward(face_id, cell_id));
            EXPECT_EQ(face.interior, mesh.is_interior_face(face_id));
            if (face.interior)
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                EXPECT_EQ(face.other_lid, FVM::detail::packed_cell_local_id(mesh, other));
                EXPECT_EQ(face.cell_distance, mesh.cell_to_face_distance(face_id, cell_id));
                EXPECT_EQ(face.other_distance, mesh.cell_to_face_distance(face_id, other));
            }
            else
            {
                const auto& location = locations.at(face.face_lid);
                EXPECT_EQ(face.boundary.active, location.active);
                EXPECT_EQ(face.boundary.batch_id, location.batch_id);
                EXPECT_EQ(face.boundary.in_batch_id, location.in_batch_id);
                if (location.active)
                    EXPECT_EQ(face.normal_distance,
                        FVM::detail::boundary_normal_distance(mesh, face.face_lid, static_cast<Local>(cell)));
            }
        }
    }
    EXPECT_EQ(actual.offsets.back(), entry);
    EXPECT_EQ(actual.faces.size(), entry);
}

template<class MeshType>
void compare_caches(SP<const MeshType> mesh)
{
    FVM::CellGradientCache<Pack, MeshType> least_squares(mesh);
    FVM::GaussLinearGradientCache<Pack, MeshType> gauss(mesh);
    expect_least_squares_geometry(least_squares);
    expect_gauss_geometry(gauss);
    expect_stencil_geometry(*mesh);
    const auto* ls_plan = &least_squares.traverse_plan();
    const auto* gauss_plan = &gauss.traverse_plan();
    least_squares.refresh();
    gauss.refresh();
    EXPECT_EQ(&least_squares.traverse_plan(), ls_plan);
    EXPECT_EQ(&gauss.traverse_plan(), gauss_plan);
    expect_least_squares_geometry(least_squares);
    expect_gauss_geometry(gauss);
}

// Flatten only public cache data, retaining order and lengths as well as values.
auto snapshot(const std::vector<Cache::CellGeometry>& geometry)
{
    std::vector<real_t> result;
    for (const auto& cell : geometry)
    {
        result.push_back(cell.interior_samples.size());
        result.push_back(cell.boundary_samples.size());
        for (const auto& sample : cell.interior_samples)
            result.insert(result.end(), {static_cast<real_t>(sample.other_lid),
                sample.weight.x, sample.weight.y, sample.weight.z});
        for (const auto& sample : cell.boundary_samples)
            result.insert(result.end(), {static_cast<real_t>(sample.location.active),
                static_cast<real_t>(sample.location.batch_id), static_cast<real_t>(sample.location.in_batch_id),
                sample.weight.x, sample.weight.y, sample.weight.z, sample.normal_distance});
    }
    return result;
}

auto snapshot(const TransportCache::SharedGeometry& geometry)
{
    std::vector<real_t> result;
    auto append_entries = [&](const auto& entries)
    {
        result.push_back(entries.size());
        for (const auto& entry : entries)
            result.insert(result.end(), {static_cast<real_t>(entry.cell_lid),
                entry.coefficient.x, entry.coefficient.y, entry.coefficient.z});
    };
    for (const auto& entries : geometry.interior) append_entries(entries);
    for (const auto& cell : geometry.boundary)
    {
        append_entries(cell.interior_entries);
        result.push_back(cell.boundary_samples.size());
        for (const auto& sample : cell.boundary_samples)
            result.insert(result.end(), {static_cast<real_t>(sample.face_lid),
                static_cast<real_t>(sample.location.active), static_cast<real_t>(sample.location.batch_id),
                static_cast<real_t>(sample.location.in_batch_id), sample.basis.x, sample.basis.y, sample.basis.z,
                sample.normal_distance});
    }
    for (const auto& location : geometry.locations)
        result.insert(result.end(), {static_cast<real_t>(location.active),
            static_cast<real_t>(location.batch_id), static_cast<real_t>(location.in_batch_id)});
    const auto& assembly = geometry.assembly;
    result.insert(result.end(), assembly.volumes.begin(), assembly.volumes.end());
    result.insert(result.end(), assembly.face_offsets.begin(), assembly.face_offsets.end());
    for (const auto& face : assembly.faces)
        result.insert(result.end(), {static_cast<real_t>(face.face_lid), static_cast<real_t>(face.other),
            static_cast<real_t>(face.interior), static_cast<real_t>(face.boundary),
            static_cast<real_t>(face.owned_orientation)});
    for (bool boundary : assembly.physical_boundary_faces) result.push_back(boundary);
    for (const auto& [location, index] : assembly.boundary_indices)
        result.insert(result.end(), {static_cast<real_t>(location.first),
            static_cast<real_t>(location.second), static_cast<real_t>(index)});
    result.push_back(geometry.has_non_orthogonal_faces);
    return result;
}

void expect_linear_reconstruction(SP<const Handle> mesh, bool planar_faces = true)
{
    ScalarCellFieldStored<Pack> scalar(mesh, "linear_scalar");
    VectorCellFieldStored<Pack> gradient(mesh, "linear_gradient");
    VectorCellFieldStored<Pack> gauss_reference(mesh, "gauss_point_reference");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        scalar.set_owned_value(cell, 1.3 + 0.7 * mesh->cell_centroid(cell).z);
    scalar.sync_ghosts();
    auto condition = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 0}; };
    auto boundary = [&](int batch, size_t index)
    { return 1.3 + 0.7 * mesh->face_centroid(mesh->boundary_face_batch(batch).face_lids[index]).z; };
    Cache least_squares(mesh);
    GaussCache gauss(mesh);
    FVM::detail::stored_gauss_linear_cell_gradient_reference(scalar, condition, boundary, gauss_reference);
    for (int mode = 0; mode < 4; ++mode)
    {
        if (mode == 0) FVM::cell_gradient(scalar, condition, boundary, gradient, least_squares);
        if (mode == 1) FVM::cell_gradient(scalar, condition, boundary, gradient);
        if (mode == 2) FVM::gauss_linear_cell_gradient(scalar, condition, boundary, gradient, gauss);
        if (mode == 3) FVM::gauss_linear_cell_gradient(scalar, condition, boundary, gradient);
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            SCOPED_TRACE(::testing::Message() << "mode=" << mode << " cell=" << cell);
            const auto value = gradient.value(cell);
            if (mode >= 2) expect_exact(value, gauss_reference.value(cell));
            // Cylindrical radial faces use midpoint normals rather than exact
            // integrated area vectors. Gauss is checked against its pointwise
            // operator there; affine exactness of transverse components holds
            // for planar faces and for least-squares on every fixture.
            if (mode < 2 || planar_faces)
            {
                EXPECT_NEAR(value.x, 0, 2e-12);
                EXPECT_NEAR(value.y, 0, 2e-12);
            }
            EXPECT_NEAR(value.z, 0.7, 2e-12);
        }
    }
}

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
            EXPECT_EQ(a(cell, component), b(cell, component))
                << "cell=" << cell << " component=" << component;
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
    auto mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 7, 8}}}));
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
