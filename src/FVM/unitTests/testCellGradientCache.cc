/**
 * @file testCellGradientCache.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for reusable least-squares cell-gradient geometry.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/CellOperators.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
using TensorFieldType = SimpleFluid::TensorCellField<Pack>;
using GradientCache = SimpleFluid::FVM::CellGradientCache<Pack>;

static_assert(!std::is_copy_constructible_v<GradientCache>);
static_assert(!std::is_copy_assignable_v<GradientCache>);
static_assert(std::is_move_constructible_v<GradientCache>);
static_assert(std::is_move_assignable_v<GradientCache>);

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::real_t scalar_value(
    const MeshType::Vec3& point,
    SimpleFluid::real_t scale)
{
    return 0.25 + scale * (0.5 * point.x - 0.2 * point.y)
         + 0.125 * point.z + 0.3 * point.x * point.y
         - 0.15 * point.z * point.z;
}

MeshType::Vec3 vector_value(
    const MeshType::Vec3& point,
    SimpleFluid::real_t scale)
{
    return {
        1.0 + scale * point.x - 0.4 * point.y
            + 0.2 * point.z * point.z,
        -2.0 + 0.3 * point.x * point.y
            + scale * point.z,
        0.5 - 0.1 * point.x + scale * point.y * point.z};
}

void expect_scalar_gradients_near(
    const VectorFieldType& actual,
    const VectorFieldType& expected,
    double tolerance = 2.0e-12)
{
    ASSERT_EQ(actual.num_owned_cells(), expected.num_owned_cells());
    for (size_t owned = 0; owned < actual.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto difference =
            actual.value(cell_lid) - expected.value(cell_lid);
        EXPECT_NEAR(difference.x, 0.0, tolerance);
        EXPECT_NEAR(difference.y, 0.0, tolerance);
        EXPECT_NEAR(difference.z, 0.0, tolerance);
    }
}

void expect_vector_gradients_near(
    const TensorFieldType& actual,
    const TensorFieldType& expected,
    double tolerance = 2.0e-12)
{
    ASSERT_EQ(actual.num_owned_cells(), expected.num_owned_cells());
    for (size_t owned = 0; owned < actual.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto actual_value = actual.value(cell_lid);
        const auto expected_value = expected.value(cell_lid);
        for (size_t component = 0; component < 3; ++component)
        {
            const auto difference =
                actual_value[component] - expected_value[component];
            EXPECT_NEAR(difference.x, 0.0, tolerance);
            EXPECT_NEAR(difference.y, 0.0, tolerance);
            EXPECT_NEAR(difference.z, 0.0, tolerance);
        }
    }
}

TEST(CellGradientCacheTest,
     ReusesInteriorWeightsForChangingScalarAndVectorFields)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    GradientCache cache(mesh);
    FieldType scalar(mesh, "cached_scalar");
    VectorFieldType vector(mesh, "cached_vector");
    VectorFieldType direct_scalar_gradient(
        mesh, "direct_scalar_gradient");
    VectorFieldType cached_scalar_gradient(
        mesh, "cached_scalar_gradient");
    TensorFieldType direct_vector_gradient(
        mesh, "direct_vector_gradient");
    TensorFieldType cached_vector_gradient(
        mesh, "cached_vector_gradient");

    for (const auto scale :
         {SimpleFluid::real_t{0.7}, SimpleFluid::real_t{-1.3}})
    {
        for (size_t owned = 0;
             owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            const auto& center = mesh->cell_centroid(cell_lid);
            scalar.set_owned_value(
                cell_lid, scalar_value(center, scale));
            vector.set_owned_value(
                cell_lid, vector_value(center, scale));
        }
        scalar.sync_ghosts();
        vector.sync_ghosts();

        SimpleFluid::FVM::cell_gradient(
            scalar, direct_scalar_gradient);
        SimpleFluid::FVM::cell_gradient(
            scalar, cached_scalar_gradient, cache);
        SimpleFluid::FVM::cell_gradient(
            vector, direct_vector_gradient);
        SimpleFluid::FVM::cell_gradient(
            vector, cached_vector_gradient, cache);

        expect_scalar_gradients_near(
            cached_scalar_gradient, direct_scalar_gradient);
        expect_vector_gradients_near(
            cached_vector_gradient, direct_vector_gradient);
    }
}

TEST(CellGradientCacheTest,
     MatchesDynamicScalarAndVectorBoundaryReconstructions)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    GradientCache cache(mesh);
    FieldType scalar(mesh, "boundary_scalar");
    VectorFieldType vector(mesh, "boundary_vector");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto& center = mesh->cell_centroid(cell_lid);
        scalar.set_owned_value(
            cell_lid, scalar_value(center, 1.1));
        vector.set_owned_value(
            cell_lid, vector_value(center, 0.9));
    }
    scalar.sync_ghosts();
    vector.sync_ghosts();

    auto scalar_condition =
        [&](int batch_id, size_t) -> SimpleFluid::BoundaryCondition
    {
        if (mesh->boundary_batch_name(batch_id) == "xmin")
        {
            return {
                SimpleFluid::BoundaryConditionType::Dirichlet,
                0.0};
        }
        return {
            SimpleFluid::BoundaryConditionType::Neumann,
            0.17};
    };
    auto scalar_boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id)
                .face_lids[in_batch_id];
        return scalar_value(
            mesh->face_centroid(face_lid), 1.1);
    };
    auto vector_boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id)
                .face_lids[in_batch_id];
        return vector_value(
            mesh->face_centroid(face_lid), 0.9);
    };

    VectorFieldType direct_scalar_gradient(
        mesh, "direct_boundary_scalar_gradient");
    VectorFieldType cached_scalar_gradient(
        mesh, "cached_boundary_scalar_gradient");
    TensorFieldType direct_vector_gradient(
        mesh, "direct_boundary_vector_gradient");
    TensorFieldType cached_vector_gradient(
        mesh, "cached_boundary_vector_gradient");
    SimpleFluid::FVM::cell_gradient(
        scalar, scalar_condition, scalar_boundary_value,
        direct_scalar_gradient);
    SimpleFluid::FVM::cell_gradient(
        scalar, scalar_condition, scalar_boundary_value,
        cached_scalar_gradient, cache);
    SimpleFluid::FVM::cell_gradient(
        vector, vector_boundary_value, direct_vector_gradient);
    SimpleFluid::FVM::cell_gradient(
        vector, vector_boundary_value,
        cached_vector_gradient, cache);

    expect_scalar_gradients_near(
        cached_scalar_gradient, direct_scalar_gradient);
    expect_vector_gradients_near(
        cached_vector_gradient, direct_vector_gradient);

    SimpleFluid::BoundaryConditionMap scalar_boundaries;
    scalar_boundaries["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        1.35};
    scalar_boundaries["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann,
        -0.08};
    SimpleFluid::FVM::cell_gradient(
        scalar, scalar_boundaries, direct_scalar_gradient);
    SimpleFluid::FVM::cell_gradient(
        scalar, scalar_boundaries,
        cached_scalar_gradient, cache);
    expect_scalar_gradients_near(
        cached_scalar_gradient, direct_scalar_gradient);
}

TEST(CellGradientCacheTest,
     MatchesRankDeficientPeriodicReconstruction)
{
    auto mesh =
        SimpleFluid::test::build_mesh<Pack>(
            SimpleFluid::test::make_box_database(
                3, 1, 1, 1.0 / 3.0));
    MeshType::local_ordinal_type xmin_face = -1;
    MeshType::local_ordinal_type xmax_face = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin")
        {
            xmin_face = batch.face_lids.front();
        }
        else if (name == "xmax")
        {
            xmax_face = batch.face_lids.front();
        }
    }
    ASSERT_GE(xmin_face, 0);
    ASSERT_GE(xmax_face, 0);
    const auto xmin_owner = mesh->owner_cell(xmin_face);
    const auto xmax_owner = mesh->owner_cell(xmax_face);
    mesh->set_periodic_face(xmin_face, xmax_owner);
    mesh->set_periodic_face(xmax_face, xmin_owner);

    GradientCache cache(mesh);
    FieldType scalar(mesh, "periodic_scalar");
    VectorFieldType vector(mesh, "periodic_vector");
    for (size_t owned = 0;
         owned < mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto& center = mesh->cell_centroid(cell_lid);
        scalar.set_owned_value(
            cell_lid, scalar_value(center, 0.8));
        vector.set_owned_value(
            cell_lid, vector_value(center, 1.2));
    }
    scalar.sync_ghosts();
    vector.sync_ghosts();

    VectorFieldType direct_scalar_gradient(
        mesh, "direct_periodic_scalar_gradient");
    VectorFieldType cached_scalar_gradient(
        mesh, "cached_periodic_scalar_gradient");
    TensorFieldType direct_vector_gradient(
        mesh, "direct_periodic_vector_gradient");
    TensorFieldType cached_vector_gradient(
        mesh, "cached_periodic_vector_gradient");
    SimpleFluid::FVM::cell_gradient(
        scalar, direct_scalar_gradient);
    SimpleFluid::FVM::cell_gradient(
        scalar, cached_scalar_gradient, cache);
    SimpleFluid::FVM::cell_gradient(
        vector, direct_vector_gradient);
    SimpleFluid::FVM::cell_gradient(
        vector, cached_vector_gradient, cache);

    expect_scalar_gradients_near(
        cached_scalar_gradient, direct_scalar_gradient);
    expect_vector_gradients_near(
        cached_vector_gradient, direct_vector_gradient);
}

TEST(CellGradientCacheTest,
     RetainsItsMeshAndRejectsFieldsOnAnotherMesh)
{
    EXPECT_THROW(
        GradientCache(SimpleFluid::SP<const MeshType>{}),
        std::invalid_argument);

    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    std::weak_ptr<const MeshType> retained_mesh = mesh;
    auto cache = std::make_unique<GradientCache>(mesh);
    mesh.reset();
    ASSERT_FALSE(retained_mesh.expired());
    EXPECT_EQ(cache->mesh_ptr().get(), retained_mesh.lock().get());

    auto other_mesh =
        SimpleFluid::test::build_mesh<Pack>(
            SimpleFluid::test::make_2x2x2_database());
    FieldType field(other_mesh, 1.0, "wrong_mesh_field");
    VectorFieldType gradient(other_mesh, "wrong_mesh_gradient");
    EXPECT_THROW(
        SimpleFluid::FVM::cell_gradient(
            field, gradient, *cache),
        std::invalid_argument);

    cache.reset();
    EXPECT_TRUE(retained_mesh.expired());
}

} // namespace
