/** @file testVariableDensityMomentumEquation.cc
 * @brief Independent density-weighted momentum storage, force and rejection checks.
 */
#include "equations/VariableDensityMomentumEquation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>
#include <limits>
#include <array>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<Handle> mesh()
{
    auto geometry = std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 1}, {0, 1}, {0, 1}}});
    Handle::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Handle>(geometry, distribution);
}

template<class MeshType>
void momentum_balance(SP<const MeshType> grid)
{
    using Traits = MeshFieldTraits<Pack, MeshType>;
    typename Traits::scalar_cell_type old_density(grid, 2.0, "rho_old"), density(grid, 3.0, "rho"),
        viscosity(grid, 0.0, "mu");
    typename Traits::scalar_face_type mass_flux(grid, 0.0, "mass_flux");
    typename Traits::vector_cell_type old_velocity(grid, vec3<double>{1, -2, 3}, "old_u"),
        velocity(grid, vec3<double>{9, 9, 9}, "u");
    VariableDensityMomentumEquation<Pack, MeshType> equation(grid);
    const auto statistics = equation.advance(old_velocity, old_density, density, mass_flux, viscosity,
        0.25, velocity, [](auto) { return vec3<double>{4, 8, -4}; });
    EXPECT_TRUE(statistics.converged);
    for (size_t c = 0; c < grid->num_owned_cells(); ++c)
    {
        const auto value = velocity.value(static_cast<typename Pack::local_ordinal_type>(c));
        EXPECT_NEAR(3 * value.x, 2 * 1 + 0.25 * 4, 1e-12);
        EXPECT_NEAR(3 * value.y, 2 * -2 + 0.25 * 8, 1e-12);
        EXPECT_NEAR(3 * value.z, 2 * 3 - 0.25 * 4, 1e-12);
    }
}

TEST(VariableDensityMomentumTest, NativeStorageUsesActualOldAndNewDensityAndPhysicalForce)
{
    auto grid = mesh();
    const auto comm = grid->owned_cell_map()->getComm();
    const int local_empty = grid->num_owned_cells() == 0 ? 1 : 0;
    int empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - 1);
    momentum_balance<Handle>(grid);
}

TEST(VariableDensityMomentumTest, CanonicalFaceMassFluxIsNotMultipliedByDensityAgain)
{
    auto geometry = std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{{0, 1, 2}, {0, 1}, {0, 1}}});
    Handle::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    auto grid = std::make_shared<Handle>(geometry, distribution);
    using Traits = MeshFieldTraits<Pack, Handle>;
    Traits::scalar_cell_type old_density(grid, 0.0, "old_rho"), density(grid, 0.0, "rho"),
        viscosity(grid, 0.0, "mu");
    Traits::scalar_face_type flux(grid, 0.0, "mass_flux");
    Traits::vector_cell_type old_velocity(grid, vec3<double>{}, "old_u"), velocity(grid, vec3<double>{}, "u");
    for (size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const bool left = grid->cell_centroid(cell).x < 1.;
        old_density.set_owned_value(cell, left ? 2. : 4.);
        density.set_owned_value(cell, left ? 1.5 : 4.5);
        old_velocity.set_owned_value(cell, {left ? 3. : -1., 0., 0.});
    }
    old_density.sync_ghosts();
    density.sync_ghosts();
    old_velocity.sync_ghosts();
    for (const auto face : flux.owned_face_ids())
        if (grid->is_interior_face(face))
            flux.set_owned_value(face, grid->face_area_vector(face).x);
    flux.sync_ghosts();
    VariableDensityMomentumEquation<Pack, Handle> equation(grid);
    EXPECT_TRUE(equation.advance(old_velocity, old_density, density, flux, viscosity, .5, velocity).converged);
    double local_momentum = 0.;
    for (size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const double expected = grid->cell_centroid(cell).x < 1. ? 3. : -2.5 / 4.5;
        EXPECT_NEAR(velocity.value(cell).x, expected, 2e-12);
        local_momentum += density.value(cell) * velocity.value(cell).x * grid->cell_volume(cell);
    }
    double global_momentum = 0.;
    Teuchos::reduceAll(*grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_momentum, &global_momentum);
    EXPECT_NEAR(global_momentum, 2., 2e-12);
}

TEST(VariableDensityMomentumTest, LegacyStorageRetainsTheSameMomentumBalance)
{
    auto grid = test::build_mesh<Pack>(test::make_single_hex_database());
    if (grid->owned_cell_map()->getComm()->getSize() != 1) GTEST_SKIP() << "Legacy mesh fixture is serial.";
    momentum_balance<Mesh<Pack>>(grid);
}

TEST(VariableDensityMomentumTest, RejectsInvalidOwnedDensityCollectivelyWithoutPublishing)
{
    auto grid = mesh();
    using Traits = MeshFieldTraits<Pack, Handle>;
    Traits::scalar_cell_type old_density(grid, 2.0, "old_rho"), density(grid, 2.0, "rho"), viscosity(grid, 0.0, "mu");
    Traits::scalar_face_type flux(grid, 0.0, "mass_flux");
    Traits::vector_cell_type old_velocity(grid, vec3<double>{1, 2, 3}, "old_u"), velocity(grid, vec3<double>{4, 5, 6}, "u");
    if (grid->num_owned_cells()) density.set_owned_value(0, -1);
    VariableDensityMomentumEquation<Pack, Handle> equation(grid);
    EXPECT_THROW(equation.advance(old_velocity, old_density, density, flux, viscosity, 0.1, velocity), std::exception);
    for (size_t c = 0; c < grid->num_owned_cells(); ++c)
    {
        const auto actual = velocity.value(static_cast<typename Pack::local_ordinal_type>(c));
        EXPECT_DOUBLE_EQ(actual.x, 4); EXPECT_DOUBLE_EQ(actual.y, 5); EXPECT_DOUBLE_EQ(actual.z, 6);
    }
}

TEST(VariableDensityMomentumTest, NativeALEStorageAndForceUseAcceptedOldAndTrialNewVolumes)
{
    using Traits = MeshFieldTraits<Pack, Handle>;
    for (const double height : {2., .5})
    {
        SCOPED_TRACE(height);
        auto grid = mesh();
        PlanarALEMeshMotion<Pack> motion(grid);
        constexpr double dt = .25;
        motion.begin_trial(height, dt);
        const auto ale = FVM::make_ale_control_volume_state(*grid, motion);
        Traits::scalar_cell_type old_density(grid, 2., "ale_old_rho"), density(grid, 0., "ale_trial_rho"),
            viscosity(grid, 0., "ale_mu");
        Traits::scalar_face_type flux(grid, 0., "ale_relative_mass_flux");
        Traits::vector_cell_type old_velocity(grid, vec3<double>{1., -2., 3.}, "ale_old_u"),
            velocity(grid, vec3<double>{9., 8., 7.}, "ale_u");
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            EXPECT_DOUBLE_EQ(ale.old_cell_volumes()[row], 1.);
            EXPECT_DOUBLE_EQ(ale.new_cell_volumes()[row], height);
            density.set_owned_value(static_cast<Pack::local_ordinal_type>(row), 2. / height);
        }
        old_density.sync_ghosts();
        density.sync_ghosts();
        viscosity.sync_ghosts();
        old_velocity.sync_ghosts();
        flux.sync_ghosts();
        VariableDensityMomentumEquation<Pack, Handle> equation(grid);
        const vec3<double> force{4., 8., -4.};
        const auto statistics = equation.advance(
            old_velocity, old_density, density, flux, viscosity, dt, velocity, [force](auto) { return force; }, {}, {},
            {}, FVM::NonOrthogonalTreatment::Implicit, &ale);
        EXPECT_TRUE(statistics.converged);
        std::array<double, 3> local{}, global{};
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(row);
            const auto actual = velocity.value(cell);
            const auto old_u = old_velocity.value(cell);
            const double new_mass = density.value(cell) * height;
            EXPECT_DOUBLE_EQ(new_mass, 2.);
            for (std::size_t component = 0; component < 3; ++component)
            {
                // Closed moving material parcel: P_new=P_old+dt*V_new*f.
                const auto expected = 2. * old_u.component(component) + dt * height * force.component(component);
                EXPECT_NEAR(new_mass * actual.component(component), expected, 2e-12);
                local[component] += new_mass * actual.component(component);
            }
        }
        Teuchos::reduceAll(*grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 3, local.data(), global.data());
        EXPECT_NEAR(global[0], 2. + dt * height * 4., 2e-12);
        EXPECT_NEAR(global[1], -4. + dt * height * 8., 2e-12);
        EXPECT_NEAR(global[2], 6. - dt * height * 4., 2e-12);
        motion.rollback_trial();
    }
}

TEST(VariableDensityMomentumTest, NativeALEFaceMassFluxConservesMomentumAfterLateFailureAndGeometryReplay)
{
    using Traits = MeshFieldTraits<Pack, Handle>;
    auto geometry = std::make_shared<Handle::Cartesian>(Vec3D<ArrReal>{{{0., 1.}, {0., 1.}, {0., .25, 1.}}});
    auto grid = std::make_shared<Handle>(geometry, Handle::DistributionOptions{.allow_empty_partitions = true});
    const auto comm = grid->owned_cell_map()->getComm();
    const int local_empty = grid->num_owned_cells() == 0 ? 1 : 0;
    int empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    const int local_failing_rank = grid->num_owned_cells() ? comm->getRank() : std::numeric_limits<int>::max();
    int failing_rank = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_failing_rank, &failing_rank);
    PlanarALEMeshMotion<Pack> motion(grid);
    constexpr double dt = .125;
    motion.begin_trial(2., dt);
    const auto first_ale = FVM::make_ale_control_volume_state(*grid, motion);
    Traits::scalar_cell_type old_density(grid, 0., "ale_face_old_rho"), density(grid, 0., "ale_face_rho"),
        viscosity(grid, 0., "ale_face_mu");
    Traits::scalar_face_type flux(grid, 0., "ale_face_relative_mass_flux");
    Traits::vector_cell_type old_velocity(grid, vec3<double>{}, "ale_face_old_u"),
        velocity(grid, vec3<double>{9., 8., 7.}, "ale_face_u");
    for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const bool lower = grid->cell_geometry_global_id(cell) == 0;
        const double old_mass = lower ? .5 : 3.;
        const double new_mass = old_mass + (lower ? -dt : dt); // One kg/s from lower to upper.
        EXPECT_DOUBLE_EQ(first_ale.old_cell_volumes()[row], lower ? .25 : .75);
        EXPECT_DOUBLE_EQ(first_ale.new_cell_volumes()[row], lower ? .5 : 1.5);
        old_density.set_owned_value(cell, lower ? 2. : 4.);
        density.set_owned_value(cell, new_mass / first_ale.new_cell_volumes()[row]);
        old_velocity.set_owned_value(cell, lower ? vec3<double>{3., -1., 2.} : vec3<double>{-1., 2., -3.});
    }
    for (const auto face : flux.owned_face_ids())
        if (grid->is_interior_face(face))
            flux.set_owned_value(face, grid->face_area_vector(face).z); // Area one, canonical sign.
    old_density.sync_ghosts();
    density.sync_ghosts();
    viscosity.sync_ghosts();
    old_velocity.sync_ghosts();
    flux.sync_ghosts();
    VariableDensityMomentumEquation<Pack, Handle> equation(grid);
    LinearSolverOptions linear;
    linear.tolerance = 1e-12;
    const auto advance = [&](const FVM::ALEControlVolumeState& ale,
                             VariableDensityMomentumEquation<Pack, Handle>::boundary_type boundary = {})
    {
        return equation.advance(old_velocity, old_density, density, flux, viscosity, dt, velocity, {},
            std::move(boundary), {}, linear, FVM::NonOrthogonalTreatment::Implicit, &ale);
    };
    const auto expect_unpublished = [&]
    {
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            const auto u = velocity.value(static_cast<Pack::local_ordinal_type>(row));
            EXPECT_DOUBLE_EQ(u.x, 9.);
            EXPECT_DOUBLE_EQ(u.y, 8.);
            EXPECT_DOUBLE_EQ(u.z, 7.);
        }
    };
    const auto reject_third_component = [&, failing_rank](int, std::size_t, std::size_t component) -> BoundaryCondition
    {
        if (component == 2 && comm->getRank() == failing_rank)
            throw std::runtime_error("Manufactured final momentum component boundary failure.");
        return {BoundaryConditionType::Neumann, 0.};
    };
    EXPECT_THROW((void) advance(first_ale, reject_third_component), std::exception);
    expect_unpublished();
    EXPECT_TRUE(advance(first_ale).converged);
    std::vector<vec3<double>> first;
    std::array<double, 3> local_momentum{}, global_momentum{};
    for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(row);
        const bool lower = grid->cell_geometry_global_id(cell) == 0;
        const vec3<double> expected = lower ? vec3<double>{3., -1., 2.} : vec3<double>{-.84, 1.88, -2.8};
        const auto u = velocity.value(cell);
        first.push_back(u);
        for (std::size_t component = 0; component < 3; ++component)
        {
            EXPECT_NEAR(u.component(component), expected.component(component), 2e-12);
            local_momentum[component] +=
                density.value(cell) * first_ale.new_cell_volumes()[row] * u.component(component);
        }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local_momentum.data(), global_momentum.data());
    EXPECT_NEAR(global_momentum[0], -1.5, 2e-12);
    EXPECT_NEAR(global_momentum[1], 5.5, 2e-12);
    EXPECT_NEAR(global_momentum[2], -8., 2e-12);
    motion.rollback_trial();
    for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        velocity.set_owned_value(static_cast<Pack::local_ordinal_type>(row), {9., 8., 7.});
    motion.begin_trial(2., dt);
    const auto replay_ale = FVM::make_ale_control_volume_state(*grid, motion);
    EXPECT_THROW((void) advance(first_ale), std::exception);
    expect_unpublished();
    EXPECT_TRUE(advance(replay_ale).converged);
    for (std::size_t row = 0; row < first.size(); ++row)
        for (std::size_t component = 0; component < 3; ++component)
            EXPECT_NEAR(velocity.value(static_cast<Pack::local_ordinal_type>(row)).component(component),
                first[row].component(component), 2e-12);
    motion.rollback_trial();
}

TEST(VariableDensityMomentumTest, LegacyStorageLargeDensityRatioRetainsSmallAcceptedMomentum)
{
    auto grid = test::build_mesh<Pack>(test::make_single_hex_database());
    if (grid->owned_cell_map()->getComm()->getSize() != 1)
        GTEST_SKIP() << "Legacy mesh fixture is serial.";
    using Traits = MeshFieldTraits<Pack, Mesh<Pack>>;
    Traits::scalar_cell_type old_density(grid, 1., "legacy_small_old_rho"),
        density(grid, 1e20, "legacy_large_trial_rho"), viscosity(grid, 0., "legacy_mu");
    Traits::scalar_face_type flux(grid, 0., "legacy_zero_mass_flux");
    Traits::vector_cell_type old_velocity(grid, vec3<double>{1., -2., 3.}, "legacy_old_u"),
        velocity(grid, vec3<double>{9., 8., 7.}, "legacy_u");
    VariableDensityMomentumEquation<Pack, Mesh<Pack>> equation(grid);
    LinearSolverOptions linear;
    linear.tolerance = 1e-12;
    EXPECT_TRUE(equation.advance(old_velocity, old_density, density, flux, viscosity, .25, velocity, {}, {}, {}, linear)
            .converged);
    const auto u = velocity.value(0);
    EXPECT_NEAR(u.x * 1e20, 1., 1e-12);
    EXPECT_NEAR(u.y * 1e20, -2., 1e-12);
    EXPECT_NEAR(u.z * 1e20, 3., 1e-12);
}

TEST(VariableDensityMomentumTest, MalformedAndRankAsymmetricLinearPoliciesRejectBeforePublishingAndCanReplay)
{
    auto grid = mesh();
    using Traits = MeshFieldTraits<Pack, Handle>;
    Traits::scalar_cell_type old_density(grid, 2., "linear_old_rho"), density(grid, 2., "linear_rho"),
        viscosity(grid, 0., "linear_mu");
    Traits::scalar_face_type flux(grid, 0., "linear_mass_flux");
    Traits::vector_cell_type old_velocity(grid, vec3<double>{1., 2., 3.}, "linear_old_u"),
        velocity(grid, vec3<double>{4., 5., 6.}, "linear_u");
    VariableDensityMomentumEquation<Pack, Handle> equation(grid);
    const auto comm = grid->owned_cell_map()->getComm();
    const auto rejects = [&](const LinearSolverOptions& linear)
    {
        int rejected = 0;
        try
        {
            (void) equation.advance(
                old_velocity, old_density, density, flux, viscosity, .1, velocity, {}, {}, {}, linear);
        }
        catch (const std::exception&)
        {
            rejected = 1;
        }
        int all_rejected = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
        EXPECT_EQ(all_rejected, 1);
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            const auto u = velocity.value(static_cast<Pack::local_ordinal_type>(row));
            EXPECT_DOUBLE_EQ(u.x, 4.);
            EXPECT_DOUBLE_EQ(u.y, 5.);
            EXPECT_DOUBLE_EQ(u.z, 6.);
        }
    };
    LinearSolverOptions invalid;
    invalid.max_iterations = 0;
    rejects(invalid);
    invalid = {};
    if (comm->getRank() == comm->getSize() - 1)
        invalid.tolerance = std::numeric_limits<double>::quiet_NaN(); // Empty owner in MPI.
    rejects(invalid);
    if (comm->getSize() > 1)
    {
        for (int choice = 0; choice < 6; ++choice)
        {
            SCOPED_TRACE(choice);
            LinearSolverOptions selected;
            if (comm->getRank() != 0)
            {
                if (choice == 0)
                    ++selected.max_iterations;
                if (choice == 1)
                    selected.tolerance *= 10.;
                if (choice == 2)
                    selected.backend = LinearSolverBackend::BiCGStab;
                if (choice == 3)
                    selected.preconditioner = LinearPreconditioner::Jacobi;
                if (choice == 4)
                    selected.verbosity = 0;
                if (choice == 5)
                    selected.reuse_preconditioner = true;
            }
            rejects(selected);
        }
    }
    EXPECT_TRUE(equation.advance(old_velocity, old_density, density, flux, viscosity, .1, velocity).converged);
    for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
    {
        const auto u = velocity.value(static_cast<Pack::local_ordinal_type>(row));
        EXPECT_NEAR(u.x, 1., 1e-12);
        EXPECT_NEAR(u.y, 2., 1e-12);
        EXPECT_NEAR(u.z, 3., 1e-12);
    }
}

TEST(VariableDensityMomentumTest, NonfiniteOwnedMassFluxRejectsCollectivelyAndValidFluxCanReplay)
{
    auto grid = mesh();
    using Traits = MeshFieldTraits<Pack, Handle>;
    Traits::scalar_cell_type old_density(grid, 2., "flux_old_rho"), density(grid, 2., "flux_rho"),
        viscosity(grid, 0., "flux_mu");
    Traits::scalar_face_type flux(grid, 0., "nonfinite_mass_flux");
    Traits::vector_cell_type old_velocity(grid, vec3<double>{1., 2., 3.}, "flux_old_u"),
        velocity(grid, vec3<double>{4., 5., 6.}, "flux_u");
    VariableDensityMomentumEquation<Pack, Handle> equation(grid);
    for (const auto invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
            velocity.set_owned_value(static_cast<Pack::local_ordinal_type>(row), {4., 5., 6.});
        bool first = true;
        for (const auto face : flux.owned_face_ids())
        {
            flux.set_owned_value(face, first ? invalid : 0.);
            first = false;
        }
        flux.sync_ghosts();
        EXPECT_THROW(
            (void) equation.advance(old_velocity, old_density, density, flux, viscosity, .1, velocity), std::exception);
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            const auto u = velocity.value(static_cast<Pack::local_ordinal_type>(row));
            EXPECT_DOUBLE_EQ(u.x, 4.);
            EXPECT_DOUBLE_EQ(u.y, 5.);
            EXPECT_DOUBLE_EQ(u.z, 6.);
        }
        for (const auto face : flux.owned_face_ids())
            flux.set_owned_value(face, 0.);
        flux.sync_ghosts();
        EXPECT_TRUE(equation.advance(old_velocity, old_density, density, flux, viscosity, .1, velocity).converged);
        for (std::size_t row = 0; row < grid->num_owned_cells(); ++row)
        {
            const auto u = velocity.value(static_cast<Pack::local_ordinal_type>(row));
            EXPECT_NEAR(u.x, 1., 1e-12);
            EXPECT_NEAR(u.y, 2., 1e-12);
            EXPECT_NEAR(u.z, 3., 1e-12);
        }
    }
}

} // namespace
