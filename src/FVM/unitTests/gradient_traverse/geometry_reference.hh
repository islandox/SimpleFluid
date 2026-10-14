#pragma once

// Helpers retain translation-unit-local linkage. Register the test
// environment only in the executable's top-level source.

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
#include <limits>
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

// The compiled cache and the inline reference can contract arithmetic
// differently. Allow only roundoff in computed values; IDs, sample ordering,
// callback ordering and retained snapshots are still checked exactly.
void expect_roundoff_equal(real_t actual, real_t expected)
{
    const auto scale = std::max({real_t{1}, std::abs(actual), std::abs(expected)});
    EXPECT_NEAR(actual, expected, 16 * std::numeric_limits<real_t>::epsilon() * scale);
}

void expect_roundoff_equal(Vec actual, Vec expected)
{
    expect_roundoff_equal(actual.x, expected.x);
    expect_roundoff_equal(actual.y, expected.y);
    expect_roundoff_equal(actual.z, expected.z);
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
                    expect_roundoff_equal(actual.weight, weight);
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
                    expect_roundoff_equal(actual.weight, weight);
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
                    expect_roundoff_equal(actual.basis, basis);
                }
            }
            const auto& actual = boundaries ? boundary[cell].interior_entries : interior[cell];
            ASSERT_EQ(actual.size(), coefficients.size());
            size_t entry = 0;
            for (const auto& [id, coefficient] : coefficients)
            {
                EXPECT_EQ(actual[entry].cell_lid, id);
                expect_roundoff_equal(actual[entry++].coefficient, coefficient);
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
            expect_roundoff_equal(face.area, mesh.face_area_vector_outward(face_id, cell_id));
            EXPECT_EQ(face.interior, mesh.is_interior_face(face_id));
            if (face.interior)
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_id, cell_id);
                EXPECT_EQ(face.other_lid, FVM::detail::packed_cell_local_id(mesh, other));
                expect_roundoff_equal(face.cell_distance, mesh.cell_to_face_distance(face_id, cell_id));
                expect_roundoff_equal(face.other_distance, mesh.cell_to_face_distance(face_id, other));
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
            if (mode >= 2) expect_roundoff_equal(value, gauss_reference.value(cell));
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

} // namespace
