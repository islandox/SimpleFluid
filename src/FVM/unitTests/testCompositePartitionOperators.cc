/** @file testCompositePartitionOperators.cc
 * @brief Numerical parity and conservation for native mixed-region partitions.
 */
#include <gtest/gtest.h>
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/composite_partition_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <map>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<>;
using Partitioner = MeshPartitioner<Pack>;
using Policy = CompositePartitionOptions::Policy;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
real_t analytic(MeshUtils::Vec3 point) { return 1.0 + 2.0 * point.z; }

std::map<Pack::global_ordinal_type, real_t> row(const Pack::matrix_type& matrix, size_t local)
{
    Pack::matrix_type::local_inds_host_view_type columns;
    Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(local, columns, values);
    std::map<Pack::global_ordinal_type, real_t> result;
    for (size_t i = 0; i < columns.size(); ++i)
        result.emplace(matrix.getColMap()->getGlobalElement(columns[i]), values[i]);
    return result;
}
}

TEST(CompositePartitionOperatorsTest, MixedRegionDiffusionAndGradientMatchSerialCanonicalRows)
{
    const auto comm = Tpetra::getDefaultComm();
    // This small serial reference is deliberately independent of MPI ownership.
    SP<const Handle> reference = std::make_shared<Handle>(test::mixed_partition_source());
    auto reference_boundary = [&](int b, size_t i) {
        return BoundaryCondition{BoundaryConditionType::Dirichlet,
            analytic(reference->face_centroid(reference->boundary_face_batch(b).face_lids[i]))};
    };
    const auto serial = FVM::diffusion_system<Pack>(*reference, 1.0, reference_boundary);
    for (const bool distributed_source : {false, true})
    for (const auto policy : {Policy::TopologyAware, Policy::CellGraph, Policy::ContiguousCanonical})
    {
        SCOPED_TRACE(distributed_source ? "distributed source" : "root source");
        CompositePartitionOptions partition_options; partition_options.policy = policy;
        const auto partition = [&]
        {
            auto initial_options = partition_options;
            if (distributed_source) initial_options.policy = policy == Policy::ContiguousCanonical
                ? Policy::TopologyAware : Policy::ContiguousCanonical;
            auto first = [&]
            {
                const auto source = comm->getRank() == 0 ? test::mixed_partition_source() : nullptr;
                return Partitioner::partition(CompositeMeshSource(source), initial_options, comm);
            }();
            if (!distributed_source) return first;
            const auto owned = first.plan().owned_cells();
            const auto source = CompositeMeshSource::distributed(first.geometry(),
                std::vector<uint64_t>(owned.begin(), owned.end()), comm->getSize() - 1);
            return Partitioner::partition(source, partition_options, comm);
        }();
        SP<const Handle> mesh = std::make_shared<Handle>(partition);
        ScalarCellFieldStored<Pack> scalar(mesh, "phi");
        VectorCellFieldStored<Pack> gradient(mesh, "gradient");
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
            scalar.set_owned_value(cell, analytic(mesh->cell_centroid(cell)));
        scalar.sync_ghosts();
        auto boundary_value = [&](int b, size_t i) {
            return analytic(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i]));
        };
        auto boundary = [&](int b, size_t i) {
            return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary_value(b, i)};
        };
        FVM::cell_gradient(scalar, boundary, boundary_value, gradient);
        const auto distributed = FVM::diffusion_system<Pack>(*mesh, 1.0, boundary);
        const auto rhs = distributed.rhs->getData(), reference_rhs = serial.rhs->getData();
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            EXPECT_NEAR((gradient.value(cell) - MeshUtils::Vec3{0, 0, 2}).norm(), 0, 2e-11);
            const auto canonical = mesh->cell_global_id(cell);
            const auto expected = row(*serial.matrix, canonical), actual = row(*distributed.matrix, cell);
            ASSERT_EQ(actual.size(), expected.size());
            for (auto [column, value] : expected)
            {
                ASSERT_TRUE(actual.contains(column));
                EXPECT_NEAR(actual.at(column), value, 2e-13);
            }
            EXPECT_NEAR(rhs[cell], reference_rhs[canonical], 2e-13);
        }
        Pack::vector_type solution(mesh->owned_cell_map(), true);
        BelosLinearSolver<> solver; LinearSolverOptions options; options.tolerance = 1e-12;
        ASSERT_TRUE(solver.solve(distributed.matrix, *distributed.rhs, solution, options));
        const auto values = solution.getData();
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
            EXPECT_NEAR(values[cell], analytic(mesh->cell_centroid(cell)), 2e-10);
        EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
    }
}

TEST(CompositePartitionOperatorsTest, MixedRegionSeamTransportConservesInventory)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto source = comm->getRank() == 0 ? test::mixed_partition_source() : nullptr;
    const auto partition = Partitioner::partition(CompositeMeshSource(source), {}, comm);
    SP<const Handle> mesh = std::make_shared<Handle>(partition);
    ScalarCellFieldStored<Pack> old(mesh, "old");
    ScalarFaceFieldStored<Pack> flux(mesh, 0.0, "flux");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        old.set_owned_value(cell, 1 + mesh->cell_centroid(cell).x);
    for (size_t face = 0; face < mesh->num_owned_faces(); ++face)
    {
        const auto x = mesh->face_centroid(face).x;
        if (mesh->is_interior_face(face) && (std::abs(x - 1) < 1e-12 || std::abs(x - 2) < 1e-12))
            flux.set_owned_value(face, 0.125 * mesh->face_area_vector(face).x);
    }
    old.sync_ghosts(); flux.sync_ghosts();
    const auto system = FVM::transport_system<Pack>(old, flux, 0.03, 0.0,
        [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; },
        [](int, size_t) { return 0.0; }, [](int) { return 0.0; });
    Pack::vector_type result(mesh->owned_cell_map(), true);
    BelosLinearSolver<> solver; LinearSolverOptions options; options.tolerance = 1e-12;
    ASSERT_TRUE(solver.solve(system.matrix, *system.rhs, result, options));
    const auto values = result.getData();
    real_t local[3]{}, global[3]{};
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        local[0] += mesh->cell_volume(cell) * old.value(cell);
        local[1] += mesh->cell_volume(cell) * values[cell];
        local[2] += std::abs(values[cell] - old.value(cell));
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_NEAR(global[0], global[1], 2e-10);
    EXPECT_GT(global[2], 1e-4);
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
}
