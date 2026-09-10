/** @file testSteadyDiffusionConvergence.cc @brief Actual steady-helper correction and MPI contracts. */
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_DefaultSerialComm.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Field = ScalarCellFieldStored<Pack>;
using Treatment = FVM::NonOrthogonalTreatment;
using Vec = MeshUtils::Vec3;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

ArrReal axis(int n, real_t lo = 0, real_t hi = 1)
{
    ArrReal result;
    for (int i = 0; i <= n; ++i) result.push_back(lo + (hi - lo) * i / n);
    return result;
}

SP<const Handle> skewed_grid(const Teuchos::RCP<const Pack::comm_type>& comm)
{
    using Geometry = Meshes::UnstructuredMesh;
    using Partitioned = Meshes::PartitionedMesh<Geometry, Pack>;
    constexpr int nx = 8, ny = 4;
    Arr<Vec> nodes;
    const auto node = [](int i, int j, int k) { return Geometry::NodeID(i + (nx + 1) * (j + (ny + 1) * k)); };
    for (int k = 0; k <= 1; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                nodes.push_back({real_t(i) / nx + 0.3 * j / ny, real_t(j) / ny, real_t(k)});
    Arr<Geometry::CellDefinition> cells;
    Arr<Geometry::BoundaryFaceDefinition> boundaries;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            const auto a = node(i, j, 0), b = node(i + 1, j, 0), c = node(i + 1, j + 1, 0), d = node(i, j + 1, 0);
            const auto e = node(i, j, 1), f = node(i + 1, j, 1), g = node(i + 1, j + 1, 1), h = node(i, j + 1, 1);
            cells.push_back({Geometry::CellType::HEXAHEDRON, {a, b, c, d, e, f, g, h}});
            if (i == 0) boundaries.push_back({{a, d, h, e}, 0, "xmin"});
            if (i + 1 == nx) boundaries.push_back({{b, c, g, f}, 1, "xmax"});
            if (j == 0) boundaries.push_back({{a, b, f, e}, 2, "ymin"});
            if (j + 1 == ny) boundaries.push_back({{d, c, g, h}, 3, "ymax"});
            boundaries.push_back({{a, b, c, d}, 4, "zmin"});
            boundaries.push_back({{e, f, g, h}, 5, "zmax"});
        }
    auto geometry = std::make_shared<Geometry>(nodes, cells, boundaries);
    auto partition = MeshPartitioner<Pack>::partition(*geometry, comm);
    return std::make_shared<Handle>(std::make_shared<Partitioned>(geometry, std::move(partition.indexer), comm));
}

SP<const Handle> coarse_fine_grid()
{
    using namespace Meshes;
    auto coarse = cartesian_region("coarse", {{axis(2, 0, 0.5), axis(2), {0, 1}}});
    auto fine = cartesian_region("fine", {{axis(2, 0.5, 1), axis(4), {0, 1}}});
    NonconformingInterface interface{0, 1, 1, 0, {}};
    for (size_t cf = 0; cf < coarse.layout().faces; ++cf)
    {
        if (coarse.topology().boundary_id(cf) != 1) continue;
        std::vector<uint64_t> children;
        const auto y = coarse.geometry().face_centroid(cf).y;
        for (size_t ff = 0; ff < fine.layout().faces; ++ff)
            if (fine.topology().boundary_id(ff) == 0 && std::abs(fine.geometry().face_centroid(ff).y - y) < 0.25)
                children.push_back(ff);
        interface.faces.push_back({cf, std::move(children)});
    }
    return std::make_shared<Handle>(std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{coarse, fine}, std::vector<MultiRegionMesh::Interface>{interface}));
}

real_t boundary_value(Vec p) { return 1 + p.x * p.x + 0.7 * p.x * p.y + 0.3 * std::sin(2 * p.y) + 0.2 * p.z; }

template<class MeshType>
auto conditions(const MeshType& mesh)
{
    return [&mesh](int batch, size_t index)
    {
        return BoundaryCondition{BoundaryConditionType::Dirichlet,
            boundary_value(mesh.face_centroid(mesh.boundary_face_batch(batch).face_lids[index]))};
    };
}

LinearSolverOptions linear_controls()
{
    LinearSolverOptions result;
    result.tolerance = 1e-12;
    result.max_iterations = 1000;
    return result;
}

real_t full_residual(const Field& solution)
{
    const auto& mesh = solution.mesh();
    // The Explicit assembly from the final field evaluates the entire lagged
    // correction independently of the helper's implicit matrix/RHS split.
    const auto system = FVM::non_orthogonal_diffusion_system<Pack>(mesh, 0.7, conditions(mesh),
        [](int) { return 0.0; }, Treatment::Explicit, &solution);
    Pack::vector_type residual(mesh.owned_cell_map(), true);
    system.matrix->apply(solution.owned_data(), residual);
    residual.update(-1.0, *system.rhs, 1.0);
    const auto norm = system.rhs->norm2();
    return residual.norm2() / (norm > 0 ? norm : 1);
}

int partition_corrections(const Handle& mesh)
{
    int local = 0, global = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
        for (const auto face : mesh.faces(c))
            if (mesh.is_interior_face(face) && !mesh.is_owned_cell(mesh.opposite_or_periodic_neighbor_cell(face, c)) &&
                FVM::detail::non_orthogonal_area_vector(mesh.face_area_vector_outward(face, c),
                    mesh.cell_center_vector(face, c)).norm() > 1e-12)
                ++local;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

TEST(SteadyDiffusionConvergence, ImplicitMatchesSerialFromNonExactInitialFields)
{
    const auto serial = skewed_grid(Teuchos::rcp(new Teuchos::SerialComm<int>));
    Field reference(serial, 0.0, "serial_reference");
    auto correction = FVM::NonOrthogonalConvergenceOptions{};
    correction.max_iterations = 1; // No lagged remote gradients in serial.
    ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*serial, 0.7, conditions(*serial), reference,
        Treatment::Implicit, 0, linear_controls(), correction));
    EXPECT_LT(full_residual(reference), 2e-11);

    const auto mesh = skewed_grid(Tpetra::getDefaultComm());
    if (mesh->owned_cell_map()->getComm()->getSize() > 1) EXPECT_GT(partition_corrections(*mesh), 0);
    for (const auto initial : {0.0, -7.0})
    {
        Field solution(mesh, 0.0, "distributed_solution");
        solution.owned_data().putScalar(initial);
        // Deliberately do not synchronize the non-exact initial overlap values.
        ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, conditions(*mesh), solution,
            Treatment::Implicit, 0, linear_controls()));
        EXPECT_LT(full_residual(solution), 2e-10);
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto gid = mesh->owned_cell_map()->getGlobalElement(c);
            const auto ref = serial->owned_cell_map()->getLocalElement(gid);
            EXPECT_NEAR(solution.value(c), reference.value(ref), 2e-9) << "cell " << gid;
        }
    }
}

TEST(SteadyDiffusionConvergence, ExhaustionAndFreshResidualCannotReportSuccess)
{
    const auto mesh = skewed_grid(Tpetra::getDefaultComm());
    Field solution(mesh, 0.0, "limited_solution");
    FVM::NonOrthogonalConvergenceOptions correction;
    correction.max_iterations = 1;
    // Make the update criterion deliberately permissive: only the fresh full
    // equation residual can reject this lagged solve.
    correction.update_tolerance = 1e9;
    const auto success = FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, conditions(*mesh), solution,
        Treatment::Implicit, 0, linear_controls(), correction);
    if (mesh->owned_cell_map()->getComm()->getSize() > 1)
    {
        EXPECT_FALSE(success);
        EXPECT_GT(full_residual(solution), 1e-6);
    }
    else EXPECT_TRUE(success);
    std::vector<real_t> returned_overlap(mesh->num_local_cells());
    for (size_t c = 0; c < mesh->num_local_cells(); ++c)
    {
        returned_overlap[c] = solution.local_value(c);
        EXPECT_TRUE(std::isfinite(returned_overlap[c]));
    }
    solution.sync_ghosts();
    for (size_t c = 0; c < mesh->num_local_cells(); ++c) EXPECT_EQ(solution.local_value(c), returned_overlap[c]);

    correction = {};
    correction.update_tolerance = 1e-30;
    correction.residual_tolerance = 1e9;
    correction.max_iterations = 1;
    solution.owned_data().putScalar(0.0);
    const auto update_success = FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, conditions(*mesh), solution,
        Treatment::Implicit, 0, linear_controls(), correction);
    EXPECT_EQ(update_success, mesh->owned_cell_map()->getComm()->getSize() == 1);
}

TEST(SteadyDiffusionConvergence, HomogeneousProblemConvergesFromNonconstantInitialField)
{
    const auto mesh = skewed_grid(Tpetra::getDefaultComm());
    Field solution(mesh, 0.0, "homogeneous_nonconstant_initial");
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto point = mesh->cell_centroid(c);
        solution.set_owned_value(c, 2.0 + point.x * point.x + std::sin(3 * point.y));
    }
    auto boundary = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 0.0}; };
    const auto converged = FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, boundary, solution,
        Treatment::Implicit, 0, linear_controls());
    const auto system = FVM::non_orthogonal_diffusion_system<Pack>(*mesh, 0.7, boundary,
        [](int) { return 0.0; }, Treatment::Explicit, &solution);
    Pack::vector_type residual(mesh->owned_cell_map(), true);
    system.matrix->apply(solution.owned_data(), residual);
    residual.update(-1.0, *system.rhs, 1.0);
    EXPECT_TRUE(converged) << "solution norm " << solution.owned_data().normInf()
                          << ", full absolute residual " << residual.norm2()
                          << ", corrected RHS norm " << system.rhs->norm2();
    EXPECT_LT(solution.owned_data().normInf(), 1e-10);
    EXPECT_LT(residual.norm2(), 1e-10);
}

TEST(SteadyDiffusionConvergence, SmallNonzeroLoadRetainsRelativeResidualGate)
{
    constexpr real_t scale = 1e-16;
    const auto mesh = skewed_grid(Tpetra::getDefaultComm());
    Field solution(mesh, 0.0, "small_nonzero_load");
    const auto values = conditions(*mesh);
    auto boundary = [&](int batch, size_t index)
    {
        auto condition = values(batch, index);
        condition.value *= scale;
        return condition;
    };
    FVM::NonOrthogonalConvergenceOptions correction;
    correction.max_iterations = 1;
    const auto one_solve = FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, boundary, solution,
        Treatment::Implicit, 0, linear_controls(), correction);
    EXPECT_EQ(one_solve, mesh->owned_cell_map()->getComm()->getSize() == 1);
    ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, boundary, solution,
        Treatment::Implicit, 0, linear_controls()));
    const auto system = FVM::non_orthogonal_diffusion_system<Pack>(*mesh, 0.7, boundary,
        [](int) { return 0.0; }, Treatment::Explicit, &solution);
    Pack::vector_type residual(mesh->owned_cell_map(), true);
    system.matrix->apply(solution.owned_data(), residual);
    residual.update(-1.0, *system.rhs, 1.0);
    EXPECT_LT(residual.norm2() / system.rhs->norm2(), 2e-10);
}

TEST(SteadyDiffusionConvergence, InvalidAndRankDivergentControlsAreCollective)
{
    const auto mesh = skewed_grid(Tpetra::getDefaultComm());
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    Field solution(mesh, -2.0, "unchanged_on_invalid_input");
    for (int scenario = 0; scenario < 9; ++scenario)
    {
        FVM::NonOrthogonalConvergenceOptions correction;
        auto linear = linear_controls();
        auto treatment = Treatment::Implicit;
        int correctors = 0;
        if (rank == 0)
        {
            switch (scenario)
            {
                case 0: correction.max_iterations = 0; break;
                case 1: correction.update_tolerance = -1; break;
                case 2: correction.residual_tolerance = std::numeric_limits<real_t>::quiet_NaN(); break;
                case 3: correctors = -1; break;
                case 4: linear.tolerance = std::numeric_limits<real_t>::infinity(); break;
                case 5: treatment = static_cast<Treatment>(99); break;
                case 6: correction.max_iterations = 39; break;
                case 7: correction.update_tolerance *= 2; break;
                case 8: treatment = Treatment::Hybrid; break;
            }
        }
        if (scenario >= 6 && mesh->owned_cell_map()->getComm()->getSize() == 1) continue;
        EXPECT_THROW(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, conditions(*mesh), solution,
            treatment, correctors, linear, correction), std::invalid_argument) << scenario;
    }
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c) EXPECT_EQ(solution.value(c), -2.0);
}

TEST(SteadyDiffusionConvergence, ExplicitAndHybridKeepFixedSweepContract)
{
    const auto mesh = skewed_grid(Tpetra::getDefaultComm());
    for (const auto treatment : {Treatment::Explicit, Treatment::Hybrid})
    {
        Field solution(mesh, 0.0, "fixed_sweep_solution");
        FVM::NonOrthogonalConvergenceOptions correction;
        correction.max_iterations = 1;
        correction.update_tolerance = 1e-30;
        correction.residual_tolerance = 1e-30;
        ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, conditions(*mesh), solution,
            treatment, 0, linear_controls(), correction));
        EXPECT_GT(full_residual(solution), 1e-6);
    }
}

TEST(SteadyDiffusionConvergence, CoarseFineCompositeConvergesActualHelper)
{
    const auto mesh = coarse_fine_grid();
    if (mesh->owned_cell_map()->getComm()->getSize() > 1) EXPECT_GT(partition_corrections(*mesh), 0);
    Field first(mesh, 0.0, "zero_initial"), second(mesh, -3.0, "nonzero_initial");
    for (auto* solution : {&first, &second})
    {
        ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, conditions(*mesh), *solution,
            Treatment::Implicit, 0, linear_controls()));
        EXPECT_LT(full_residual(*solution), 2e-10);
    }
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c) EXPECT_NEAR(first.value(c), second.value(c), 2e-9);
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(mesh->has_materialized_indexer());
}

TEST(SteadyDiffusionConvergence, EmptyRanksParticipateInConvergence)
{
    using namespace Meshes;
    SP<const Handle> mesh = std::make_shared<Handle>(std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{cartesian_region("one", {{{0, 1}, {0, 1}, {0, 1}}})},
        std::vector<MultiRegionMesh::Interface>{}));
    Field solution(mesh, 0.0, "one_cell");
    ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7,
        [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 2.5}; }, solution,
        Treatment::Implicit, 0, linear_controls()));
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c) EXPECT_NEAR(solution.value(c), 2.5, 1e-12);
}

TEST(SteadyDiffusionConvergence, NativeImplicitAlsoChecksFreshFullEquation)
{
    const auto mesh = test::make_skewed_prism_mesh<Pack>();
    CellField<Pack> solution(mesh, "native_zero_initial");
    solution.owned_data().putScalar(0.0);
    const auto boundary = conditions(*mesh);
    ASSERT_TRUE(FVM::solve_non_orthogonal_diffusion<Pack>(*mesh, 0.7, boundary, solution,
        Treatment::Implicit, 0, linear_controls()));
    const auto residual = FVM::full_diffusion_residual<Pack>(solution, 0.7, boundary);
    const auto system = FVM::non_orthogonal_diffusion_system<Pack>(*mesh, 0.7, boundary,
        [](int) { return 0.0; }, Treatment::Explicit, &solution);
    EXPECT_LT(residual->norm2() / system.rhs->norm2(), 2e-10);
}
} // namespace
