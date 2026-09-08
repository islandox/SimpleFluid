/** @file testStoredTransportReuse.cc @brief Repeated mapped transport and immutable graph contracts. */
#include <gtest/gtest.h>

#include "FVM/Operators.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_Array.hpp>

#include <cmath>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::FVM;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Scalar = ScalarCellFieldStored<Pack>;
using Flux = ScalarFaceFieldStored<Pack>;
using Vector = VectorCellFieldStored<Pack>;
using Tensor = TensorCellFieldStored<Pack>;
using LO = Pack::local_ordinal_type;
using Plan = FVM::detail::StoredTransportSymbolicPlan<Pack>;
using Row = FVM::detail::StoredTransportMatrixRow<Pack>;

testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const Handle> make_mesh()
{
    using Native = Meshes::UnstructuredMesh;
    using Partitioned = Meshes::PartitionedMesh<Native, Pack>;
    auto native = test::make_unstructured_hex_line(8, 0.125);
    auto comm = Tpetra::getDefaultComm();
    auto partition = MeshPartitioner<Pack>::partition(*native, comm);
    return std::make_shared<Handle>(
        std::make_shared<Partitioned>(native, std::move(partition.indexer), comm));
}

void expect_same_system(const TransportSystem<Pack>& actual, const TransportSystem<Pack>& expected)
{
    ASSERT_EQ(actual.matrix->getLocalNumRows(), expected.matrix->getLocalNumRows());
    for (size_t row = 0; row < actual.matrix->getLocalNumRows(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type actual_columns, expected_columns;
        Pack::matrix_type::values_host_view_type actual_values, expected_values;
        actual.matrix->getLocalRowView(static_cast<LO>(row), actual_columns, actual_values);
        expected.matrix->getLocalRowView(static_cast<LO>(row), expected_columns, expected_values);
        // A retained graph may keep zero-valued stencil slots when diffusion
        // or boundary types change. Compare every column, including those
        // absent from either graph, so extra nonzero entries still fail.
        const auto column_count = actual.matrix->getColMap()->getLocalNumElements();
        ASSERT_EQ(column_count, expected.matrix->getColMap()->getLocalNumElements());
        std::vector<double> actual_row(column_count, 0.0), expected_row(column_count, 0.0);
        for (size_t entry = 0; entry < actual_columns.extent(0); ++entry)
        {
            actual_row.at(actual_columns(entry)) = actual_values(entry);
        }
        for (size_t entry = 0; entry < expected_columns.extent(0); ++entry)
        {
            expected_row.at(expected_columns(entry)) = expected_values(entry);
        }
        for (size_t column = 0; column < column_count; ++column)
        {
            EXPECT_EQ(actual_row[column], expected_row[column]);
        }
        EXPECT_EQ(actual.rhs->getData()[row], expected.rhs->getData()[row]);
    }
}

TEST(StoredTransportReuseTest, FrozenGraphRefreshesNumericAndBoundaryInputs)
{
    const auto mesh = make_mesh();
    Scalar values(mesh, 0.0, "old"), storage(mesh, 1.0, "storage"), diffusion(mesh, 0.2, "diffusion");
    Flux flux(mesh, 0.0, "flux");
    TransportGeometryCache<Handle> geometry(*mesh);
    Plan plan;
    Teuchos::RCP<Pack::matrix_type> matrix;
    for (int generation = 0; generation < 4; ++generation)
    {
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<LO>(owned);
            const auto center = mesh->cell_centroid(cell);
            values.set_owned_value(cell, 1.0 + center.x + generation);
            storage.set_owned_value(cell, 1.0 + center.x + 0.25 * generation);
            diffusion.set_owned_value(cell, generation == 2 ? 0.0 : 0.3 + center.x);
        }
        for (const auto face : flux.owned_face_ids())
        {
            flux.set_owned_value(face, (generation % 2 ? -0.3 : 0.2) * mesh->face_area_vector(face).x);
        }
        values.sync_ghosts();
        storage.sync_ghosts();
        diffusion.sync_ghosts();
        flux.sync_ghosts();
        const auto assemble = [&](Teuchos::RCP<Pack::matrix_type> cached, Plan* selected_plan)
        {
            return weighted_scalar_transport_system<Pack>(MeshWeightedScalarTransportRequest<Pack, Handle>{
                .old_values = values,
                .face_fluxes = flux,
                .time_step = 0.5,
                .storage_weight = storage,
                .advection_weight = storage,
                .diffusivity = diffusion,
                .boundary_condition = [&](int batch, size_t)
                {
                    return BoundaryCondition{(batch + generation) % 2 ? BoundaryConditionType::Dirichlet
                                                                                     : BoundaryConditionType::Neumann,
                        0.125 * generation};
                },
                .boundary_value = [&](int, size_t) { return 0.75 + generation; },
                .source = [](LO) { return 0.125; },
                .treatment = NonOrthogonalTreatment::Hybrid,
                .correction_field = &values,
                .cached_matrix = std::move(cached),
                .geometry_cache = &geometry,
                .symbolic_plan = selected_plan});
        };
        const auto fresh = assemble(Teuchos::null, nullptr);
        const auto reused = assemble(matrix, &plan);
        EXPECT_TRUE(reused.matrix->isStaticGraph());
        if (!matrix.is_null())
        {
            EXPECT_EQ(matrix.get(), reused.matrix.get());
        }
        expect_same_system(reused, fresh);
        matrix = reused.matrix;
    }
}

TEST(StoredTransportReuseTest, SymbolicPlanValidatesChangedGraphsEpochsAndRankCategories)
{
    const auto mesh = make_mesh();
    struct EpochMesh
    {
        const Handle& mesh;
        std::uint64_t epoch = 0;
        auto owned_cell_map() const { return mesh.owned_cell_map(); }
        auto overlap_cell_map() const { return mesh.overlap_cell_map(); }
        auto num_owned_cells() const { return mesh.num_owned_cells(); }
        auto geometry_epoch() const { return epoch; }
    } epoch_mesh{*mesh};
    std::vector<Row> rows(mesh->num_owned_cells());
    for (size_t row = 0; row < rows.size(); ++row)
    {
        rows[row] = {{static_cast<LO>(row)}, {2.0}};
    }
    Plan plan;
    auto matrix = FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, Teuchos::null, 2, rows, &plan);
    ASSERT_TRUE(matrix->isStaticGraph());
    EXPECT_TRUE(FVM::detail::prepare_stored_transport_matrix<Pack>(epoch_mesh, matrix, 2, rows, &plan).symbolic_reuse);
    Pack::vector_type ones(mesh->owned_cell_map()), applied(mesh->owned_cell_map());
    ones.putScalar(1.0);
    matrix->apply(ones, applied);
    const auto global_rows = static_cast<double>(matrix->getGlobalNumRows());
    EXPECT_NEAR(matrix->getFrobeniusNorm(), 2.0 * std::sqrt(global_rows), 1.0e-12);
    for (auto& row : rows)
    {
        row.values[0] = 3.25;
    }
    matrix = FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, matrix, 2, rows, &plan);
    matrix->apply(ones, applied);
    EXPECT_NEAR(matrix->getFrobeniusNorm(), 3.25 * std::sqrt(global_rows), 1.0e-12);
    for (size_t row = 0; row < rows.size(); ++row)
    {
        EXPECT_EQ(applied.getData()[row], 3.25);
    }
    ++epoch_mesh.epoch;
    EXPECT_FALSE(plan.matches(epoch_mesh, matrix, rows));
    matrix = FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, matrix, 2, rows, &plan);
    EXPECT_TRUE(plan.matches(epoch_mesh, matrix, rows));

    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    auto changed = rows;
    if (rank == 0 && !changed.empty())
    {
        changed[0].columns.push_back(1);
        changed[0].values.push_back(-0.5);
    }
    EXPECT_THROW(FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, matrix, 2, changed, &plan),
        std::invalid_argument);
    EXPECT_TRUE(matrix->isFillComplete());

    // An external graph receives the same full validation and is not enrolled.
    auto external = FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, Teuchos::null, 2, rows, nullptr);
    Plan external_plan;
    auto validated = FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, external, 2, rows, &external_plan);
    EXPECT_EQ(validated.get(), external.get());
    EXPECT_FALSE(external_plan.owns(external));
    EXPECT_THROW(FVM::detail::finish_stored_transport_matrix<Pack>(epoch_mesh, external, 2, changed, &external_plan),
        std::invalid_argument);
    if (mesh->owned_cell_map()->getComm()->getSize() > 1)
    {
        EXPECT_THROW(FVM::detail::finish_stored_transport_matrix<Pack>(
                         epoch_mesh, matrix, 2, rows, rank == 0 ? nullptr : &plan),
            std::invalid_argument);
    }
}

TEST(StoredTransportReuseTest, MomentumScratchRecomputesGradientsAndRejectsForeignStorage)
{
    const auto mesh = make_mesh();
    Vector velocity(mesh, "velocity");
    Tensor scratch(mesh, "stress_gradient");
    Scalar viscosity(mesh, 0.2, "viscosity");
    Flux flux(mesh, 0.0, "flux");
    TransportGeometryCache<Handle> geometry(*mesh);
    const auto* original_storage = scratch.owned_data().getData(0).getRawPtr();
    for (int generation = 0; generation < 3; ++generation)
    {
        const auto velocity_at = [&](const Handle::Vec3& x)
        {
            return Handle::Vec3{generation + 2.0 * x.x + x.y, 0.5 * x.x - generation * x.y, x.z + x.x};
        };
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<LO>(owned);
            velocity.set_owned_value(cell, velocity_at(mesh->cell_centroid(cell)));
            viscosity.set_owned_value(cell, 0.2 + 0.1 * generation + mesh->cell_centroid(cell).x);
        }
        velocity.sync_ghosts();
        viscosity.sync_ghosts();
        const auto assemble = [&](Tensor* workspace)
        {
            return physical_momentum_transport_system<Pack>(velocity, flux, 0.5, viscosity, 2.0,
                [&](int batch, size_t face)
                {
                    return velocity_at(mesh->face_centroid(mesh->boundary_face_batch(batch).face_lids[face]));
                },
                [](LO) { return Handle::Vec3{}; }, NonOrthogonalTreatment::Hybrid, &velocity, Teuchos::null,
                [](int batch, size_t) { return batch % 2 == 0; }, nullptr, &geometry,
                FaceCoefficientInterpolation::Linear, nullptr, workspace);
        };
        const auto fresh = assemble(nullptr);
        const auto retained = assemble(&scratch);
        EXPECT_EQ(original_storage, scratch.owned_data().getData(0).getRawPtr());
        for (size_t component = 0; component < 3; ++component)
        {
            const auto actual = retained.rhs->getData(component);
            const auto expected = fresh.rhs->getData(component);
            for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
            {
                EXPECT_EQ(actual[row], expected[row]);
            }
        }
        const auto gradients = scratch.owned_read_view();
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            EXPECT_NEAR(gradients(row, 0), 2.0, 1.0e-12);
            EXPECT_NEAR(gradients(row, 4), -static_cast<double>(generation), 1.0e-12);
            EXPECT_NEAR(gradients(row, 8), 1.0, 1.0e-12);
        }
    }
    const auto foreign_mesh = make_mesh();
    Tensor foreign(foreign_mesh, "foreign");
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    EXPECT_THROW(physical_momentum_transport_system<Pack>(velocity, flux, 0.5, viscosity, 2.0,
                     [](int, size_t) { return Handle::Vec3{}; }, [](LO) { return Handle::Vec3{}; },
                     NonOrthogonalTreatment::Explicit, nullptr, Teuchos::null,
                     FVM::detail::AlwaysDiffuseBoundary{}, nullptr, &geometry,
                     FaceCoefficientInterpolation::Linear, nullptr, rank == 0 ? &foreign : &scratch),
        std::invalid_argument);
}
} // namespace
