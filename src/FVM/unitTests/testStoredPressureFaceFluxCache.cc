/**
 * @file testStoredPressureFaceFluxCache.cc
 * @brief Analytic pressure-flux checks across workspace reuse and ALE epochs.
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "utils/testing_environment.hh"

#include <limits>
#include <memory>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorField = SimpleFluid::VectorCellFieldStored<Pack>;
using FluxField = SimpleFluid::ScalarFaceFieldStored<Pack>;
using Workspace = SimpleFluid::FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack>;
using Vec = SimpleFluid::vec3<double>;
using BoundaryType = SimpleFluid::BoundaryConditionType;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

auto make_mesh()
{
    return std::make_shared<Mesh>(std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 0.25, 0.75, 1.5, 3.0}, {0.0, 0.5, 2.0}, {0.0, 0.25, 1.0, 2.0, 4.0}}}));
}

Vec affine_velocity(double z, double shift)
{
    return {0.25 + shift + 0.1 * z, -0.3 + 0.2 * z, 0.4 - shift - 0.125 * z};
}

/**
 * @brief Quadratic axial pressure has an analytic two-point face derivative.
 *
 * A graded mesh makes that derivative differ from the linearly interpolated
 * exact cell gradient, exercising an actual Rhie--Chow correction. Physical
 * boundary values change independently of the reusable geometry workspace.
 */
std::vector<double> check_quadratic_flux(const std::shared_ptr<Mesh>& mesh, Workspace& workspace,
    double height, double shift)
{
    VectorField velocity(mesh, "velocity");
    ScalarField pressure(mesh, "pressure");
    VectorField gradient(mesh, "pressure_gradient");
    FluxField normal(mesh, "normal_flux");
    FluxField flux(mesh, "pressure_flux");
    FluxField expected_flux(mesh, "expected_pressure_flux");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto z = mesh->cell_centroid(cell).z;
        velocity.set_owned_value(cell, affine_velocity(z, shift));
        pressure.set_owned_value(cell, 3.0 * z * z);
        gradient.set_owned_value(cell, {0.0, 0.0, 6.0 * z});
    }
    velocity.sync_ghosts();
    pressure.sync_ghosts();
    gradient.sync_ghosts();

    SimpleFluid::BoundaryConditionSet conditions;
    conditions.velocity["xmin"] = {BoundaryType::Slip, {}};
    conditions.velocity["xmax"] = {BoundaryType::Dirichlet, {1.0 + shift, -2.0, 0.5}};
    conditions.velocity["ymin"] = {BoundaryType::NoSlip, {}};
    conditions.velocity["ymax"] = {BoundaryType::Neumann, {}};
    conditions.velocity["zmin"] = {BoundaryType::Dirichlet, {-1.0, 2.0, 0.25 - shift}};
    conditions.velocity["zmax"] = {BoundaryType::Neumann, {}};
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(const_mesh, conditions);
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) == "ymax")
        {
            for (auto& value : boundaries.value.at(batch_id))
            {
                value = {0.5, 0.75 + shift, -0.5};
            }
        }
    }
    SimpleFluid::BoundaryConditionMap pressure_boundaries;
    pressure_boundaries["zmax"] = {BoundaryType::Dirichlet, 3.0 * height * height};
    SimpleFluid::FVM::face_fluxes(velocity, boundaries, normal);
    normal.sync_ghosts();
    constexpr double coefficient = 0.13;
    flux.put_value(173.0);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, gradient, coefficient, boundaries, pressure_boundaries, workspace, flux);

    // Fringe overlap faces can have an adjacent cell outside the local cell
    // map. Evaluate their analytical value on the owning rank, then import it.
    for (const auto face : flux.owned_face_ids())
    {
        const auto owner = mesh->owner_cell(face);
        const auto zo = mesh->cell_centroid(owner).z;
        const auto zf = mesh->face_centroid(face).z;
        const auto area = mesh->face_area(face);
        const auto normal_vector = mesh->face_normal(face);
        double expected = normal.local_value(face);
        if (mesh->is_interior_face(face))
        {
            const auto neighbor = mesh->opposite_or_periodic_neighbor_cell(face, owner);
            const auto zn = mesh->cell_centroid(neighbor).z;
            const auto direct = 3.0 * (zo + zn) * normal_vector.z * area;
            const auto interpolated = 6.0 * zf * normal_vector.z * area;
            expected = affine_velocity(zf, shift).dot(normal_vector) * area
                     - coefficient * (direct - interpolated);
        }
        else if (normal_vector.z > 0.5)
        {
            const auto direct = 3.0 * (height + zo) * area;
            const auto interpolated = 6.0 * zo * area;
            expected = affine_velocity(zo, shift).z * area - coefficient * (direct - interpolated);
        }
        expected_flux.set_owned_value(face, expected);
    }
    expected_flux.sync_ghosts();
    std::vector<double> result;
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_NEAR(flux.local_value(face), expected_flux.local_value(face), 3.0e-13) << "face " << face;
        result.push_back(flux.local_value(face));
    }

    // The disabled correction must not consume the supplied gradient at all.
    gradient.put_value({std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0});
    flux.put_value(-173.0);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, gradient, 0.0, boundaries, pressure_boundaries, workspace, flux);
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(flux.local_value(face), normal.local_value(face));
    }
    return result;
}

std::vector<double> reconstruct_flux(const std::shared_ptr<Mesh>& mesh, Workspace& workspace,
    double scale, double boundary_shift)
{
    VectorField velocity(mesh, "shared_geometry_velocity");
    ScalarField pressure(mesh, "shared_geometry_pressure");
    FluxField flux(mesh, "shared_geometry_flux");
    velocity.put_value({0.2, 0.1, -0.3});
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell);
        pressure.set_owned_value(cell, scale * (0.2 * center.x + center.z * center.z));
    }
    pressure.sync_ghosts();
    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
        conditions.velocity[mesh->boundary_batch_name(batch_id)] = {BoundaryType::Neumann, {}};
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    const auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(const_mesh, conditions);
    SimpleFluid::BoundaryConditionMap pressure_boundaries;
    pressure_boundaries["zmax"] = {BoundaryType::Dirichlet, 16.0 * scale + boundary_shift};
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, 0.13, boundaries, pressure_boundaries, workspace, flux);
    std::vector<double> result;
    for (size_t local = 0; local < mesh->num_faces(); ++local)
        result.push_back(flux.local_value(static_cast<Pack::local_ordinal_type>(local)));
    return result;
}

std::vector<double> gradient_values(const Workspace& workspace)
{
    const auto& gradient = workspace.pressure_gradient();
    const auto values = gradient.local_read_view();
    std::vector<double> result;
    for (size_t owned = 0; owned < gradient.num_owned_cells(); ++owned)
        for (size_t component = 0; component < 3; ++component)
            result.push_back(values(owned, component));
    return result;
}

} // namespace

TEST(StoredPressureFaceFluxCacheTest, ReuseReadsUpdatedFieldsAndPhysicalBoundaryValues)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto* geometry = workspace.face_geometry().data();
    const auto first = check_quadratic_flux(mesh, workspace, 4.0, 0.0);
    const auto second = check_quadratic_flux(mesh, workspace, 4.0, 0.375);
    EXPECT_EQ(workspace.face_geometry().data(), geometry);
    EXPECT_NE(first, second);
    EXPECT_EQ(check_quadratic_flux(mesh, workspace, 4.0, 0.0), first);
}

TEST(StoredPressureFaceFluxCacheTest, MotionAndRollbackRequireRefreshWithPrecomputedGradient)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto original = check_quadratic_flux(mesh, workspace, 4.0, 0.0);
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(6.0, 1.0);
    EXPECT_THROW(static_cast<void>(workspace.face_geometry()), std::invalid_argument);
    EXPECT_THROW(check_quadratic_flux(mesh, workspace, 6.0, 0.0), std::invalid_argument);
    workspace.refresh_geometry();
    const auto moved = check_quadratic_flux(mesh, workspace, 6.0, 0.0);
    EXPECT_NE(moved, original);
    Workspace fresh(mesh);
    EXPECT_EQ(moved, check_quadratic_flux(mesh, fresh, 6.0, 0.0));

    motion.rollback_trial();
    EXPECT_THROW(static_cast<void>(workspace.face_geometry()), std::invalid_argument);
    workspace.refresh_geometry();
    EXPECT_EQ(check_quadratic_flux(mesh, workspace, 4.0, 0.0), original);
}

TEST(StoredPressureFaceFluxCacheTest, SharedGeometryKeepsScratchAndBoundaryReconstructionIndependent)
{
    auto mesh = make_mesh();
    auto first = std::make_unique<Workspace>(mesh);
    Workspace second(mesh, first->shared_geometry());
    Workspace fresh(mesh);
    EXPECT_EQ(second.shared_geometry(), first->shared_geometry());
    EXPECT_EQ(&second.gradient_cache(), &first->gradient_cache());
    EXPECT_EQ(second.face_geometry().data(), first->face_geometry().data());
    EXPECT_NE(&second.pressure_gradient(), &first->pressure_gradient());

    const auto first_flux = reconstruct_flux(mesh, *first, 1.0, 0.0);
    const auto first_gradient = gradient_values(*first);
    const auto second_flux = reconstruct_flux(mesh, second, 2.0, 0.75);
    EXPECT_EQ(second_flux, reconstruct_flux(mesh, fresh, 2.0, 0.75));
    EXPECT_NE(second_flux, first_flux);
    EXPECT_NE(gradient_values(second), first_gradient);
    EXPECT_EQ(gradient_values(*first), first_gradient);
    EXPECT_EQ(reconstruct_flux(mesh, *first, 1.0, 0.0), first_flux);

    // The shared snapshot owns its mesh and survives the creating workspace.
    first.reset();
    EXPECT_EQ(reconstruct_flux(mesh, second, 2.0, 0.75), second_flux);
}

TEST(StoredPressureFaceFluxCacheTest, RefreshReplacesSharedSnapshotAndRejectsForeignOrStaleGeometry)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto original_geometry = workspace.shared_geometry();
    Workspace retained(mesh, original_geometry);
    const auto original_flux = check_quadratic_flux(mesh, workspace, 4.0, 0.0);
    const auto foreign = make_mesh();
    EXPECT_THROW(static_cast<void>(Workspace(foreign, original_geometry)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, {})), std::invalid_argument);

    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(6.0, 1.0);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, original_geometry)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(retained.face_geometry()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(retained.boundary_locations()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(retained.gradient_cache().interior_geometry()), std::invalid_argument);
    workspace.refresh_geometry();
    EXPECT_NE(workspace.shared_geometry(), original_geometry);
    EXPECT_EQ(retained.shared_geometry(), original_geometry);
    EXPECT_THROW(static_cast<void>(retained.face_geometry()), std::invalid_argument);
    Workspace current(mesh, workspace.shared_geometry());
    EXPECT_EQ(check_quadratic_flux(mesh, workspace, 6.0, 0.0), check_quadratic_flux(mesh, current, 6.0, 0.0));

    motion.rollback_trial();
    EXPECT_THROW(static_cast<void>(current.face_geometry()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, original_geometry)), std::invalid_argument);
    retained.refresh_geometry();
    EXPECT_NE(retained.shared_geometry(), original_geometry);
    EXPECT_EQ(check_quadratic_flux(mesh, retained, 4.0, 0.0), original_flux);
    EXPECT_THROW(static_cast<void>(workspace.face_geometry()), std::invalid_argument);
}
