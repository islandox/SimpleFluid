/**
 * @file testGeometryEpochCaches.cc
 * @brief Tests explicit geometry-epoch rejection and refresh for FVM caches.
 */

#include <gtest/gtest.h>

#include "FVM/CellOperators.hh"
#include "FVM/FaceFlux.hh"
#include "FVM/TransportSystem.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "utils/testing_environment.hh"

#include <memory>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorField = SimpleFluid::VectorCellFieldStored<Pack>;
using GeometryCache = SimpleFluid::FVM::TransportGeometryCache<Mesh>;
using GradientCache = SimpleFluid::FVM::CellGradientCache<Pack, Mesh>;
using FluxWorkspace = SimpleFluid::FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

std::vector<double> transport_coefficients(const GeometryCache& cache)
{
    std::vector<double> result;
    for (const auto& stencil : cache.interior_stencils())
    {
        for (const auto& entry : stencil)
        {
            for (size_t component = 0; component < 3; ++component)
            {
                result.push_back(entry.coefficient.component(component));
            }
        }
    }
    return result;
}

std::vector<double> gradient_coefficients(const GradientCache& cache)
{
    std::vector<double> result;
    for (const auto& cell : cache.interior_geometry())
    {
        for (const auto& sample : cell.interior_samples)
        {
            for (size_t component = 0; component < 3; ++component)
            {
                result.push_back(sample.weight.component(component));
            }
        }
    }
    return result;
}

} // namespace

TEST(GeometryEpochCacheTest, MotionAndRollbackRejectStaleCachesUntilExplicitRefresh)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    auto alias_mesh = std::make_shared<Mesh>(geometry);
    GeometryCache transport_cache(*mesh);
    GeometryCache alias_transport_cache(*alias_mesh);
    GradientCache gradient_cache(mesh);
    FluxWorkspace flux_workspace(mesh);
    EXPECT_EQ(transport_cache.geometry_epoch(), 0U);
    EXPECT_EQ(gradient_cache.geometry_epoch(), 0U);
    EXPECT_EQ(flux_workspace.gradient_cache().geometry_epoch(), 0U);
    const auto old_transport_coefficients = transport_coefficients(transport_cache);
    const auto old_gradient_coefficients = gradient_coefficients(gradient_cache);

    ScalarField scalar(mesh, "linear_scalar");
    VectorField gradient(mesh, "linear_scalar_gradient");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        scalar.set_owned_value(cell, mesh->cell_centroid(cell).z);
    }
    scalar.sync_ghosts();

    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(3.0, 1.0);
    ASSERT_EQ(mesh->geometry_epoch(), 1U);
    ASSERT_EQ(alias_mesh->geometry_epoch(), 1U);

    EXPECT_THROW(transport_cache.require_mesh(*mesh), std::invalid_argument);
    EXPECT_THROW(alias_transport_cache.require_mesh(*alias_mesh), std::invalid_argument);
    EXPECT_THROW(gradient_cache.require_mesh(*mesh), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(transport_cache.interior_stencils()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(gradient_cache.interior_geometry()), std::invalid_argument);
    EXPECT_THROW(SimpleFluid::FVM::cell_gradient(scalar, gradient, gradient_cache), std::invalid_argument);
    EXPECT_THROW(flux_workspace.gradient_cache().require_mesh(*mesh), std::invalid_argument);

    transport_cache.refresh();
    alias_transport_cache.refresh();
    gradient_cache.refresh();
    flux_workspace.refresh_geometry();
    EXPECT_NO_THROW(transport_cache.require_mesh(*mesh));
    EXPECT_NO_THROW(gradient_cache.require_mesh(*mesh));
    EXPECT_NO_THROW(flux_workspace.gradient_cache().require_mesh(*mesh));
    EXPECT_EQ(transport_cache.geometry_epoch(), 1U);
    EXPECT_EQ(gradient_cache.geometry_epoch(), 1U);
    EXPECT_EQ(flux_workspace.gradient_cache().geometry_epoch(), 1U);
    EXPECT_EQ(alias_transport_cache.geometry_epoch(), 1U);
    EXPECT_NE(transport_coefficients(transport_cache), old_transport_coefficients);
    EXPECT_NE(gradient_coefficients(gradient_cache), old_gradient_coefficients);
    EXPECT_NO_THROW(SimpleFluid::FVM::cell_gradient(scalar, gradient, gradient_cache));
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto value = gradient.value(cell);
        EXPECT_NEAR(value.x, 0.0, 1.0e-12);
        EXPECT_NEAR(value.y, 0.0, 1.0e-12);
        EXPECT_NEAR(value.z, 2.0 / 3.0, 1.0e-12);
    }

    motion.rollback_trial();
    ASSERT_EQ(mesh->geometry_epoch(), 2U);
    EXPECT_THROW(transport_cache.require_mesh(*mesh), std::invalid_argument);
    EXPECT_THROW(gradient_cache.require_mesh(*mesh), std::invalid_argument);
    EXPECT_THROW(flux_workspace.gradient_cache().require_mesh(*mesh), std::invalid_argument);

    transport_cache.refresh();
    gradient_cache.refresh();
    flux_workspace.refresh_geometry();
    EXPECT_EQ(transport_cache.geometry_epoch(), 2U);
    EXPECT_EQ(gradient_cache.geometry_epoch(), 2U);
    EXPECT_EQ(flux_workspace.gradient_cache().geometry_epoch(), 2U);
    SimpleFluid::FVM::cell_gradient(scalar, gradient, gradient_cache);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto value = gradient.value(cell);
        EXPECT_NEAR(value.x, 0.0, 1.0e-12);
        EXPECT_NEAR(value.y, 0.0, 1.0e-12);
        EXPECT_NEAR(value.z, 1.0, 1.0e-12);
    }
}
