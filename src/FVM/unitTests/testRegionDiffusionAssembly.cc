/** @file testRegionDiffusionAssembly.cc @brief Production/reference region diffusion parity. */
#include <gtest/gtest.h>
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<>;
using Scalar = ScalarCellFieldStored<Pack>;
using Vec = MeshUtils::Vec3;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

real_t value(Vec p) { return 1 + p.x*p.x + 0.2*p.x*p.y + std::sin(p.z) + p.y; }

void compare_assembly(const SP<const Handle>& mesh)
{
    const auto initial_indexer = mesh->has_materialized_indexer();
    const auto execution = mesh->acquire_execution_view();
    constexpr real_t diffusion = 1.7;
    auto condition = [&](int b, size_t i)
    {
        const auto face = mesh->boundary_face_batch(b).face_lids[i];
        // Exercise both boundary formulas, with enough Dirichlet data to anchor
        // the system in all fixtures, including the translated periodic ring.
        return mesh->face_normal(face).z < -0.5
            ? BoundaryCondition{BoundaryConditionType::Neumann, 0.13}
            : BoundaryCondition{BoundaryConditionType::Dirichlet, value(mesh->face_centroid(face))};
    };
    std::vector<int> fast_sources, reference_sources;
    std::vector<std::pair<int,size_t>> fast_boundaries, reference_boundaries;
    auto source = [&](int cell) { return 0.2 + std::cos(mesh->cell_centroid(cell).x); };
    const auto fast = FVM::diffusion_system<Pack>(*mesh, diffusion,
        [&](int b,size_t i) { fast_boundaries.emplace_back(b,i); return condition(b,i); },
        [&](int c) { fast_sources.push_back(c); return source(c); });
    const auto reference = FVM::detail::diffusion_system_reference_impl<Pack>(*mesh, diffusion,
        [&](int b,size_t i) { reference_boundaries.emplace_back(b,i); return condition(b,i); },
        [&](int c) { reference_sources.push_back(c); return source(c); });
    EXPECT_EQ(fast_sources, reference_sources);
    EXPECT_EQ(fast_boundaries, reference_boundaries);
    EXPECT_EQ(fast.matrix->getGlobalNumEntries(), reference.matrix->getGlobalNumEntries());

    Pack::vector_type input(mesh->owned_cell_map(),true), action(mesh->owned_cell_map(),true),
        expected(mesh->owned_cell_map(),true), difference(mesh->owned_cell_map(),true);
    for (size_t c=0; c<mesh->num_owned_cells(); ++c) input.replaceLocalValue(c,value(mesh->cell_centroid(c)));
    fast.matrix->apply(input,action);
    reference.matrix->apply(input,expected);
    difference.update(1.0,action,-1.0,expected,0.0);
    EXPECT_LE(difference.norm2(), 3e-13*(1+expected.norm2()));
    difference.update(1.0,*fast.rhs,-1.0,*reference.rhs,0.0);
    EXPECT_LE(difference.norm2(), 3e-13*(1+reference.rhs->norm2()));

    Pack::vector_type fast_values(mesh->owned_cell_map(),true), reference_values(mesh->owned_cell_map(),true);
    BelosLinearSolver<> solver;
    LinearSolverOptions options;
    options.tolerance=1e-12;
    options.max_iterations=1000;
    ASSERT_TRUE(solver.solve(fast.matrix,*fast.rhs,fast_values,options));
    ASSERT_TRUE(solver.solve(reference.matrix,*reference.rhs,reference_values,options));
    difference.update(1.0,fast_values,-1.0,reference_values,0.0);
    EXPECT_LE(difference.norm2(), 2e-10*(1+reference_values.norm2()));
    fast.matrix->apply(fast_values,action);
    action.update(-1.0,*fast.rhs,1.0);
    EXPECT_LE(action.norm2(), 2e-10*(1+fast.rhs->norm2()));

    Scalar fast_field(mesh,"fast"), reference_field(mesh,"reference");
    const auto fv=fast_values.getData(), rv=reference_values.getData();
    for (size_t c=0; c<mesh->num_owned_cells(); ++c)
    {
        fast_field.set_owned_value(c,fv[c]);
        reference_field.set_owned_value(c,rv[c]);
    }
    fast_field.sync_ghosts(); reference_field.sync_ghosts();
    // Compare integrated diffusion fluxes from the converged fields using the
    // production/reference geometry formulas independently on every owned face.
    real_t local_flux_error=0, global_flux_error=0;
    if (mesh->supports_region_execution())
        for (size_t f=0; f<mesh->num_owned_faces(); ++f)
        {
            const auto resolved=mesh->resolve_region_face(f);
            if (!resolved.interior()) continue;
            const auto owner=mesh->owner_cell(f), other=mesh->neighbor_cell(f);
            const FVM::detail::ResolvedDiffusionGeometry<decltype(resolved),real_t> geometry(resolved);
            const auto a=FVM::detail::interior_diffusion_coefficient(
                geometry,resolved.face,resolved.owner,resolved.neighbor,diffusion);
            const auto b=FVM::detail::interior_diffusion_coefficient(*mesh,static_cast<int>(f),owner,other,diffusion);
            const auto qfast=a*(fast_field.local_value(owner)-fast_field.local_value(other));
            const auto qreference=b*(reference_field.local_value(owner)-reference_field.local_value(other));
            local_flux_error=std::max(local_flux_error,std::abs(qfast-qreference));
        }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(),Teuchos::REDUCE_MAX,1,&local_flux_error,&global_flux_error);
    EXPECT_LT(global_flux_error,2e-10);
    EXPECT_EQ(mesh->connectivity_storage_bytes(),0U);
    EXPECT_EQ(mesh->has_materialized_indexer(),initial_indexer);
}
}

TEST(RegionDiffusionAssemblyTest, StaticFamiliesPreserveOperatorSolutionFluxAndResidual)
{
    for (const auto& geometry : {test::two_regions(3), test::mixed_regions(), test::coarse_fine_regions(),
             test::periodic_regions(), test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        auto mesh=std::make_shared<Handle>(geometry);
        ASSERT_TRUE(mesh->supports_region_execution());
        compare_assembly(mesh);
    }
}

TEST(RegionDiffusionAssemblyTest, AcceptedMotionAndRollbackUseTransformedGeometry)
{
    for (const auto& geometry : {test::two_regions(), test::coarse_fine_regions(), test::periodic_regions(),
             test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        auto mesh=std::make_shared<Handle>(geometry);
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.35,0.2);
        motion.accept_trial();
        compare_assembly(mesh);
        motion.begin_trial(0.8,0.2);
        compare_assembly(mesh);
        motion.rollback_trial();
        compare_assembly(mesh);
    }
}

TEST(RegionDiffusionAssemblyTest, EmptyRanksParticipateWithoutChangingMaps)
{
    auto mesh=std::make_shared<Handle>(test::two_regions(1));
    compare_assembly(mesh);
}

TEST(RegionDiffusionAssemblyTest, ReorderedHandlesRetainGenericAssembly)
{
    if (Tpetra::getDefaultComm()->getSize()!=1) GTEST_SKIP() << "Serial reordered reference.";
    auto composite=std::make_shared<Handle>(test::two_regions());
    EXPECT_THROW(MeshReorderingFactory<>::selected_cells_first(std::move(composite),
        [](auto, const Vec& center) { return center.x>1; }),std::invalid_argument);
    auto mesh=std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0,0.5,1,1.5,2},{0,0.5,1},{0,0.5,1}}}));
    auto layout=MeshReorderingFactory<>::selected_cells_first(std::move(mesh),
        [](auto, const Vec& center) { return center.x>1; });
    ASSERT_TRUE(layout.mesh->has_reordered_cells());
    EXPECT_FALSE(layout.mesh->supports_region_execution());
    compare_assembly(layout.mesh);
}
