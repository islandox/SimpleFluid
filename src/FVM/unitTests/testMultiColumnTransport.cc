/**
 * @file testMultiColumnTransport.cc
 * @brief Phase 1 tests for CarrierTransportPlan and multi-column FVM weighted transport.
 */
#include "FVM/CarrierTransportPlan.hh"
#include "FVM/TransportSystemFactory.hh"
#include <gtest/gtest.h>

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/FaceFlux.hh"
#include "FVM/Operators.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace
{

using namespace SimpleFluid;
using namespace SimpleFluid::FVM;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using ScalarField = ScalarCellFieldStored<Pack>;
using FaceField = ScalarFaceFieldStored<Pack>;
using Motion = PlanarALEMeshMotion<Pack>;
using Factory = TransportSystemFactory<Pack, Handle>;
using LO = Pack::local_ordinal_type;
using scalar_type = Pack::scalar_type;

testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Handle> make_distributed_line(const std::size_t num_cells)
{
    using Native = Meshes::UnstructuredMesh;
    using Partitioned = Meshes::PartitionedMesh<Native, Pack>;
    auto native = test::make_unstructured_hex_line(num_cells, 1.0 / static_cast<double>(std::max<std::size_t>(1, num_cells)));
    auto comm = Tpetra::getDefaultComm();
    auto partition = MeshPartitioner<Pack>::partition(*native, comm);
    return std::make_shared<Handle>(
        std::make_shared<Partitioned>(native, std::move(partition.indexer), comm));
}

std::unordered_map<LO, scalar_type> matrix_row(const Pack::matrix_type& matrix, const LO row)
{
    Pack::matrix_type::local_inds_host_view_type cols;
    Pack::matrix_type::values_host_view_type vals;
    matrix.getLocalRowView(row, cols, vals);
    std::unordered_map<LO, scalar_type> result;
    for (std::size_t i = 0; i < cols.extent(0); ++i)
    {
        result[cols(i)] = vals(i);
    }
    return result;
}

void expect_matrix_equal(const Pack::matrix_type& expected, const Pack::matrix_type& actual)
{
    ASSERT_TRUE(expected.getRowMap()->isSameAs(*actual.getRowMap()));
    ASSERT_EQ(expected.getLocalNumRows(), actual.getLocalNumRows());
    for (std::size_t r = 0; r < expected.getLocalNumRows(); ++r)
    {
        const auto lid = static_cast<LO>(r);
        EXPECT_EQ(matrix_row(actual, lid), matrix_row(expected, lid));
    }
}

TransportOperatorKey make_base_key(const void* mesh_id)
{
    TransportOperatorKey key;
    key.mesh_identity = mesh_id;
    key.geometry_epoch = 1;
    key.time_step = 0.25;
    key.storage_weight = {reinterpret_cast<const void*>(0x1001), 1};
    key.advection_flux = {reinterpret_cast<const void*>(0x1002), 1};
    key.advection_weight = {reinterpret_cast<const void*>(0x1003), 1};
    key.diffusivity = {reinterpret_cast<const void*>(0x1004), 1};
    key.matrix_boundary = {1, nullptr, 0};
    return key;
}

TEST(MultiColumnTransportTest, CarrierGroupVsOperatorGroupSplitting)
{
    const auto* mesh_id = reinterpret_cast<const void*>(0x42);
    const CarrierId liquid{CarrierKind::Liquid, "liquid"};
    const CarrierId gas{CarrierKind::BulkGas, "bulk_gas"};
    const CarrierId pop{CarrierKind::Population, "microbubble"};

    const auto base = make_base_key(mesh_id);

    auto diff_diff = base;
    diff_diff.diffusivity.version_epoch = 2;
    auto diff_bc = base;
    diff_bc.matrix_boundary.condition_epoch = 2;
    auto diff_sink = base;
    diff_sink.implicit_sink = FieldVersionKey{reinterpret_cast<const void*>(0x2001), 1};
    auto diff_corr = base;
    diff_corr.implicit_correction = FieldVersionKey{reinterpret_cast<const void*>(0x2002), 1};
    auto diff_storage = base;
    diff_storage.storage_weight.version_epoch = 2;
    auto diff_adv_w = base;
    diff_adv_w.advection_weight.version_epoch = 2;
    auto diff_flux = base;
    diff_flux.advection_flux.version_epoch = 2;
    auto diff_dt = base;
    diff_dt.time_step = 0.25 + 1.0e-12;
    auto diff_ale = base;
    diff_ale.has_ale = true;
    diff_ale.ale_new_geometry_epoch = 2;
    auto diff_disc = base;
    diff_disc.discretization.time = ScalarTimeScheme::BDF2;
    auto diff_fixed = base;
    diff_fixed.constraints.has_fixed_rows = true;

    std::vector<PlannedColumnEntry> entries{
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 1u, "liquid", {}, 0}, base},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 2u, "liquid", {}, 1}, base},
        {TransportColumn{liquid, QuantityKind::InternalEnergy, std::nullopt, "liquid", {}, 2}, diff_diff},
        {TransportColumn{gas, QuantityKind::SpeciesMoles, 1u, "bulk_gas", {}, 3}, base},
        {TransportColumn{pop, QuantityKind::PopulationNumber, std::nullopt, "microbubble", {}, 4}, base},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 3u, "liquid", {}, 5}, diff_bc},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 4u, "liquid", {}, 6}, diff_sink},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 5u, "liquid", {}, 7}, diff_corr},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 6u, "liquid", {}, 8}, diff_storage},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 7u, "liquid", {}, 9}, diff_adv_w},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 8u, "liquid", {}, 10}, diff_flux},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 9u, "liquid", {}, 11}, diff_dt},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 10u, "liquid", {}, 12}, diff_ale},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 11u, "liquid", {}, 13}, diff_disc},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 12u, "liquid", {}, 14}, diff_fixed},
    };

    const auto strict_plan = build_carrier_transport_plan(entries);
    EXPECT_EQ(strict_plan.carrier_groups.size(), 3U);
    // Species 1 & 2 in Liquid share one OperatorGroup; Gas & Pop are NOT merged with Liquid by default;
    // all 11 remaining Liquid variants each form their own OperatorGroup -> 1 + 1 + 1 + 11 = 14 groups!
    EXPECT_EQ(strict_plan.operator_groups.size(), 14U);

    // InternalEnergy must automatically allow negative values.
    bool found_energy = false;
    for (const auto& col : strict_plan.columns)
    {
        if (col.quantity == QuantityKind::InternalEnergy)
        {
            found_energy = true;
            EXPECT_TRUE(col.acceptance_policy.allow_negative);
        }
    }
    EXPECT_TRUE(found_energy);

    // Explicit opt-in cross-carrier sharing merges ONLY entries with identical full TransportOperatorKey.
    const auto cross_plan = build_carrier_transport_plan(entries, {.allow_cross_carrier_sharing = true});
    EXPECT_EQ(cross_plan.carrier_groups.size(), 3U);
    EXPECT_EQ(cross_plan.operator_groups.size(), 12U);
}

TEST(MultiColumnTransportTest, PermutationInvarianceAndDuplicateColumns)
{
    const auto* mesh_id = reinterpret_cast<const void*>(0x99);
    const CarrierId liquid{CarrierKind::Liquid, "liquid"};
    const CarrierId gas{CarrierKind::BulkGas, "bulk_gas"};
    const auto base = make_base_key(mesh_id);

    std::vector<PlannedColumnEntry> original{
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 1u, "liq", {}, 0}, base},
        {TransportColumn{liquid, QuantityKind::SpeciesMoles, 1u, "liq", {}, 1}, base}, // duplicate
        {TransportColumn{liquid, QuantityKind::InternalEnergy, std::nullopt, "liq", {}, 2}, base},
        {TransportColumn{gas, QuantityKind::SpeciesMoles, 2u, "gas", {}, 3}, base},
    };
    std::vector<PlannedColumnEntry> permuted{original[3], original[2], original[0], original[1]};

    const auto plan_a = build_carrier_transport_plan(original);
    const auto plan_b = build_carrier_transport_plan(permuted);
    ASSERT_EQ(plan_a.columns.size(), plan_b.columns.size());
    ASSERT_EQ(plan_a.operator_groups.size(), plan_b.operator_groups.size());
    for (std::size_t i = 0; i < plan_a.columns.size(); ++i)
    {
        EXPECT_EQ(plan_a.columns[i], plan_b.columns[i]);
    }
    for (std::size_t g = 0; g < plan_a.operator_groups.size(); ++g)
    {
        EXPECT_EQ(plan_a.operator_groups[g].operator_key, plan_b.operator_groups[g].operator_key);
        EXPECT_EQ(plan_a.operator_groups[g].plan_column_indices, plan_b.operator_groups[g].plan_column_indices);
    }
}

TEST(MultiColumnTransportTest, ScalarVsMultiColumnEquivalenceAndRhsOnlyRefresh)
{
    const auto mesh = make_distributed_line(8);
    const auto owned_cells = mesh->num_owned_cells();

    ScalarField storage(mesh, 1.0, "storage");
    ScalarField old_storage(mesh, 1.0, "old_storage");
    ScalarField advection(mesh, 1.0, "advection");
    ScalarField diffusivity(mesh, 0.15, "diffusivity");
    FaceField flux(mesh, 0.0, "flux");
    for (std::size_t i = 0; i < owned_cells; ++i)
    {
        const auto lid = static_cast<LO>(i);
        const auto x = mesh->cell_centroid(lid).x;
        storage.set_owned_value(lid, 1.2 + 0.3 * x);
        old_storage.set_owned_value(lid, 1.1 + 0.2 * x);
        advection.set_owned_value(lid, 0.9 + 0.25 * x);
        diffusivity.set_owned_value(lid, 0.1 + 0.05 * x);
    }
    for (const auto face : flux.owned_face_ids())
    {
        flux.set_owned_value(face, 0.35 * mesh->face_area_vector(face).x);
    }
    storage.sync_ghosts();
    old_storage.sync_ghosts();
    advection.sync_ghosts();
    diffusivity.sync_ghosts();
    flux.sync_ghosts();

    constexpr std::size_t num_cols = 4;
    std::vector<std::unique_ptr<ScalarField>> old_fields;
    for (std::size_t c = 0; c < num_cols; ++c)
    {
        old_fields.push_back(std::make_unique<ScalarField>(mesh, 0.0, "old_" + std::to_string(c)));
        for (std::size_t i = 0; i < owned_cells; ++i)
        {
            const auto lid = static_cast<LO>(i);
            const auto x = mesh->cell_centroid(lid).x;
            double v = 0.0;
            if (c == 0) v = 1.0e6 * (1.0 + x);     // huge
            if (c == 1) v = 0.0;                   // zero
            if (c == 2) v = 1.0e-12 * (1.0 + x);   // trace
            if (c == 3) v = -25.0 + 60.0 * x;      // signed InternalEnergy
            old_fields.back()->set_owned_value(lid, v);
        }
        old_fields.back()->sync_ghosts();
    }

    auto bc_cond = [](int batch, std::size_t)
    {
        return BoundaryCondition{
            batch % 2 == 0 ? BoundaryConditionType::Dirichlet : BoundaryConditionType::Neumann, 0.0};
    };
    auto implicit_sink = [&](LO cell)
    {
        return 0.05 * (1.0 + mesh->cell_centroid(cell).x);
    };
    auto fixed_mask = [&](LO cell)
    {
        return mesh->cell_global_id(cell) == 0;
    };

    const scalar_type dt = 0.25;
    TransportGeometryCache<Handle> geometry(*mesh);
    Factory::WeightedScalarOptions options;
    options.discretization = {.time = ScalarTimeScheme::BackwardEuler, .convection = ScalarConvectionScheme::Upwind};
    options.correction = {NonOrthogonalTreatment::Hybrid, nullptr};

    // Assemble N scalar reference systems.
    std::vector<TransportSystem<Pack>> scalar_systems;
    for (std::size_t c = 0; c < num_cols; ++c)
    {
        auto src = [&, c](LO cell) -> scalar_type
        {
            if (c == 1) return 0.0;
            if (c == 3) return -4.0 + 8.0 * mesh->cell_centroid(cell).x;
            return (c == 0 ? 10.0 : 1.0e-13) * (1.0 + mesh->cell_centroid(cell).x);
        };
        auto bv = [c](int, std::size_t) -> scalar_type
        {
            if (c == 1) return 0.0;
            if (c == 3) return -12.0;
            return c == 0 ? 500.0 : 2.0e-12;
        };
        auto cond = [&, c, bv](int batch, std::size_t face)
        {
            auto base = bc_cond(batch, face);
            if (base.type == BoundaryConditionType::Neumann && c != 1)
            {
                base.value = (c == 3 ? -1.5 : (c == 0 ? 2.0 : 1.0e-13));
            }
            return base;
        };
        auto fixed_val = [&](LO cell) -> std::optional<scalar_type>
        {
            if (!fixed_mask(cell)) return std::nullopt;
            if (c == 1) return 0.0;
            if (c == 3) return -3.0;
            return c == 0 ? 100.0 : 1.0e-12;
        };
        auto col_options = options;
        col_options.correction.lagged_field = old_fields[c].get();
        scalar_systems.push_back(Factory::assemble(
            Factory::WeightedScalarTerms{
                .history = {*old_fields[c], dt},
                .face_fluxes = flux,
                .coefficients = {storage, advection, diffusivity, &old_storage},
                .sources = {.source = src, .implicit_sink = implicit_sink},
                .constraints = {.fixed_cell_value = fixed_val}},
            Factory::WeightedScalarBoundary{.condition = cond, .value = bv},
            col_options,
            Factory::WeightedScalarContext{.geometry_cache = &geometry}));
    }

    // Assemble once via multi-column factory.
    Factory::MultiColumnWeightedTerms mc_terms{
        .time_step = dt,
        .face_fluxes = flux,
        .coefficients = {storage, advection, diffusivity, &old_storage},
        .sources = {.implicit_sink = implicit_sink},
        .constraints = {.fixed_cell_mask = fixed_mask},
        .columns = {}};
    for (std::size_t c = 0; c < num_cols; ++c)
    {
        mc_terms.columns.push_back(MultiColumnScalarInput<Pack, Handle>{
            .old_values = old_fields[c].get(),
            .source = [&, c](LO cell) -> scalar_type
            {
                if (c == 1) return 0.0;
                if (c == 3) return -4.0 + 8.0 * mesh->cell_centroid(cell).x;
                return (c == 0 ? 10.0 : 1.0e-13) * (1.0 + mesh->cell_centroid(cell).x);
            },
            .boundary_value = [c](int, std::size_t) -> scalar_type
            {
                if (c == 1) return 0.0;
                if (c == 3) return -12.0;
                return c == 0 ? 500.0 : 2.0e-12;
            },
            .neumann_flux = [c](int, std::size_t) -> scalar_type
            {
                if (c == 1) return 0.0;
                return c == 3 ? -1.5 : (c == 0 ? 2.0 : 1.0e-13);
            },
            .fixed_cell_value = [&, c](LO) -> scalar_type
            {
                if (c == 1) return 0.0;
                if (c == 3) return -3.0;
                return c == 0 ? 100.0 : 1.0e-12;
            },
            .lagged_correction_field = old_fields[c].get(),
            .allow_negative = (c == 3)});
    }

    FVM::detail::StoredTransportSymbolicPlan<Pack> symbolic_plan;
    auto mc_system = Factory::assemble_multi_column(
        mc_terms,
        Factory::MultiColumnWeightedBoundary{.condition = bc_cond},
        options,
        Factory::MultiColumnWeightedContext{.geometry_cache = &geometry, .symbolic_plan = &symbolic_plan});

    ASSERT_EQ(mc_system.active_columns.size(), num_cols);
    EXPECT_TRUE(mc_system.active_columns[0]);
    EXPECT_FALSE(mc_system.active_columns[1]);
    EXPECT_TRUE(mc_system.active_columns[2]);
    EXPECT_TRUE(mc_system.active_columns[3]);

    for (std::size_t c = 0; c < num_cols; ++c)
    {
        expect_matrix_equal(*scalar_systems[c].matrix, *mc_system.matrix);
        const auto s_rhs = scalar_systems[c].rhs->getData();
        const auto m_rhs = mc_system.rhs->getData(c);
        for (std::size_t r = 0; r < owned_cells; ++r)
        {
            EXPECT_DOUBLE_EQ(m_rhs[r], s_rhs[r]);
        }
    }

    // Solve batched vs N scalar solves and verify identical solutions.
    BelosLinearSolver<Pack> solver;
    LinearSolverOptions linear_opts;
    linear_opts.backend = LinearSolverBackend::Gmres;
    linear_opts.tolerance = 1.0e-12;
    linear_opts.max_iterations = 200;
    Pack::multi_vector_type mc_sol(mesh->owned_cell_map(), num_cols, true);
    const auto mc_solve_stats = solver.solve_multi_rhs_from_zero_with_statistics(
        mc_system.matrix, *mc_system.rhs, mc_sol,
        MultiRhsSolveOptions{linear_opts, {}, {}, mc_system.active_columns});
    ASSERT_TRUE(mc_solve_stats.converged);
    EXPECT_EQ(mc_solve_stats.execution_mode, MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(mc_solve_stats.max_batch_width, 3U);

    for (std::size_t c = 0; c < num_cols; ++c)
    {
        Pack::vector_type s_sol(mesh->owned_cell_map(), true);
        ASSERT_TRUE(solver.solve_from_zero_with_statistics(
            scalar_systems[c].matrix, *scalar_systems[c].rhs, s_sol, linear_opts).converged);
        const auto m_data = mc_sol.getData(c);
        const auto s_data = s_sol.getData();
        for (std::size_t r = 0; r < owned_cells; ++r)
        {
            EXPECT_NEAR(m_data[r], s_data[r], 1.0e-10 * std::max(1.0, std::abs(s_data[r])));
        }
    }

    // Now test RHS-only refresh: mutate old_values and sources without touching A_g.
    for (std::size_t c = 0; c < num_cols; ++c)
    {
        for (std::size_t i = 0; i < owned_cells; ++i)
        {
            const auto lid = static_cast<LO>(i);
            const auto cur = old_fields[c]->owned_read_view()(i, 0);
            old_fields[c]->set_owned_value(lid, c == 1 ? 0.0 : cur * 1.5 + (c == 3 ? -3.0 : 0.0));
        }
        old_fields[c]->sync_ghosts();
    }
    const auto* original_matrix_ptr = mc_system.matrix.get();
    const auto original_frob = mc_system.matrix->getFrobeniusNorm();
    auto refreshed = Factory::refresh_multi_column_rhs(
        mc_terms,
        Factory::MultiColumnWeightedBoundary{.condition = bc_cond},
        options,
        Factory::MultiColumnWeightedContext{
            .cached_matrix = mc_system.matrix,
            .geometry_cache = &geometry,
            .cached_rhs = mc_system.rhs});
    EXPECT_EQ(refreshed.matrix.get(), original_matrix_ptr);
    EXPECT_DOUBLE_EQ(refreshed.matrix->getFrobeniusNorm(), original_frob);
    EXPECT_DOUBLE_EQ(refreshed.timings.matrix_assembly_seconds, 0.0);

    for (std::size_t c = 0; c < num_cols; ++c)
    {
        auto col_options = options;
        col_options.correction.lagged_field = old_fields[c].get();
        auto fresh_scalar = Factory::assemble(
            Factory::WeightedScalarTerms{
                .history = {*old_fields[c], dt},
                .face_fluxes = flux,
                .coefficients = {storage, advection, diffusivity, &old_storage},
                .sources = {.source = mc_terms.columns[c].source, .implicit_sink = implicit_sink},
                .constraints = {.fixed_cell_value = [&](LO cell) -> std::optional<scalar_type> {
                    if (!fixed_mask(cell)) return std::nullopt;
                    return mc_terms.columns[c].fixed_cell_value(cell);
                }}},
            Factory::WeightedScalarBoundary{
                .condition = [&, c](int batch, std::size_t face) {
                    auto base = bc_cond(batch, face);
                    if (base.type == BoundaryConditionType::Neumann)
                        base.value = mc_terms.columns[c].neumann_flux(batch, face);
                    return base;
                },
                .value = mc_terms.columns[c].boundary_value},
            col_options,
            Factory::WeightedScalarContext{.geometry_cache = &geometry});
        expect_matrix_equal(*fresh_scalar.matrix, *refreshed.matrix);
        const auto s_rhs = fresh_scalar.rhs->getData();
        const auto m_rhs = refreshed.rhs->getData(c);
        for (std::size_t r = 0; r < owned_cells; ++r)
        {
            EXPECT_DOUBLE_EQ(m_rhs[r], s_rhs[r]);
        }
    }
}

TEST(MultiColumnTransportTest, SignedInternalEnergyZeroSumAndGlobalActiveMask)
{
    const auto mesh = make_distributed_line(8);
    const auto comm = mesh->owned_cell_map()->getComm();
    const auto owned_cells = mesh->num_owned_cells();
    const int rank = comm->getRank();

    ScalarField unit(mesh, 1.0, "unit");
    ScalarField zero(mesh, 0.0, "zero");
    FaceField flux(mesh, 0.0, "flux");
    for (const auto face : flux.owned_face_ids())
    {
        if (!mesh->is_boundary_face(face))
        {
            flux.set_owned_value(face, 0.1 * mesh->face_area_vector(face).x);
        }
    }
    flux.sync_ghosts();

    // 4 columns using extensive_old_inventories:
    // col 0: identically zero on all ranks
    // col 1: zero on rank 0, non-zero on last rank
    // col 2: signed InternalEnergy with global sum == 0 (+10 on gid 0, -10 on gid 7)
    // col 3: trace 1e-15
    std::vector<std::vector<double>> inv(4, std::vector<double>(owned_cells, 0.0));
    for (std::size_t i = 0; i < owned_cells; ++i)
    {
        const auto gid = mesh->cell_global_id(static_cast<LO>(i));
        if (rank == comm->getSize() - 1)
        {
            inv[1][i] = 2.5;
        }
        if (gid == 0) inv[2][i] = 10.0;
        if (gid == 7) inv[2][i] = -10.0;
        inv[3][i] = 1.0e-15 * static_cast<double>(gid + 1);
    }

    Factory::MultiColumnWeightedTerms terms{
        .time_step = 0.5,
        .face_fluxes = flux,
        .coefficients = {unit, unit, zero}};
    for (std::size_t c = 0; c < 4; ++c)
    {
        terms.columns.push_back(MultiColumnScalarInput<Pack, Handle>{
            .extensive_old_inventories = inv[c],
            .allow_negative = (c == 2)});
    }
    auto boundary = Factory::MultiColumnWeightedBoundary{
        .condition = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; }};
    Factory::WeightedScalarOptions options;
    options.discretization = {.time = ScalarTimeScheme::BackwardEuler, .convection = ScalarConvectionScheme::Upwind};

    auto sys = Factory::assemble_multi_column(terms, boundary, options);
    ASSERT_EQ(sys.active_columns.size(), 4U);
    EXPECT_FALSE(sys.active_columns[0]);
    EXPECT_TRUE(sys.active_columns[1]);
    EXPECT_TRUE(sys.active_columns[2]);
    EXPECT_TRUE(sys.active_columns[3]);

    // Also verify refresh_multi_column_rhs preserves active_columns detection (no matrix re-assembly).
    auto refreshed = Factory::refresh_multi_column_rhs(
        terms, boundary, options,
        Factory::MultiColumnWeightedContext{.cached_matrix = sys.matrix, .cached_rhs = sys.rhs});
    EXPECT_EQ(refreshed.active_columns, sys.active_columns);
    EXPECT_DOUBLE_EQ(refreshed.timings.matrix_assembly_seconds, 0.0);

    BelosLinearSolver<Pack> solver;
    LinearSolverOptions lopts;
    lopts.backend = LinearSolverBackend::Gmres;
    lopts.tolerance = 1.0e-13;
    lopts.max_iterations = 200;
    Pack::multi_vector_type sol(mesh->owned_cell_map(), 4, true);
    sol.getVectorNonConst(0)->putScalar(42.0);
    const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(
        refreshed.matrix, *refreshed.rhs, sol, MultiRhsSolveOptions{lopts, {}, {}, refreshed.active_columns});
    ASSERT_TRUE(stats.converged);
    EXPECT_TRUE(stats.column_statistics[0].skipped_zero);
    for (const auto val : sol.getData(0))
    {
        EXPECT_DOUBLE_EQ(val, 0.0);
    }

    // Verify signed InternalEnergy and trace inventory conservation (closed Neumann box).
    CarrierMultiVectorBuffer<Pack> buffer;
    std::vector<double> unpacked_energy(owned_cells, 0.0);
    std::vector<double> cell_vols(owned_cells, 0.0);
    for (std::size_t i = 0; i < owned_cells; ++i)
    {
        cell_vols[i] = mesh->cell_volume(static_cast<LO>(i));
    }
    auto& buf_mv = buffer.ensure(mesh->owned_cell_map(), 1);
    {
        const auto s = sol.getData(2);
        auto d = buf_mv.getDataNonConst(0);
        std::copy(s.begin(), s.end(), d.begin());
    }
    (void)buffer.unpack_owned_spans({{std::span<double>(unpacked_energy)}}, cell_vols);
    double local_energy_sum = std::accumulate(unpacked_energy.begin(), unpacked_energy.end(), 0.0);
    double global_energy_sum = 0.0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_energy_sum, &global_energy_sum);
    EXPECT_NEAR(global_energy_sum, 0.0, 1.0e-12);
}

TEST(MultiColumnTransportTest, MpiEmptyOwnerRankAndAleOldNewVolumes)
{
    const auto comm = Tpetra::getDefaultComm();

    // 1. Empty owner rank test: 1-cell mesh on 2 ranks leaves rank 1 with 0 owned cells.
    {
        const auto tiny_mesh = make_distributed_line(1);
        const auto owned = tiny_mesh->num_owned_cells();
        ScalarField unit(tiny_mesh, 1.0, "unit");
        ScalarField zero(tiny_mesh, 0.0, "zero");
        FaceField flux(tiny_mesh, 0.0, "flux");
        ScalarField old0(tiny_mesh, 3.0, "old0");
        ScalarField old1(tiny_mesh, -7.0, "old1");

        Factory::MultiColumnWeightedTerms terms{
            .time_step = 0.5,
            .face_fluxes = flux,
            .coefficients = {unit, unit, zero},
            .columns = {
                MultiColumnScalarInput<Pack, Handle>{.old_values = &old0},
                MultiColumnScalarInput<Pack, Handle>{.old_values = &old1, .allow_negative = true}}};
        auto boundary = Factory::MultiColumnWeightedBoundary{
            .condition = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; }};
        Factory::WeightedScalarOptions options;
        auto sys = Factory::assemble_multi_column(terms, boundary, options);
        EXPECT_EQ(sys.active_columns, (std::vector<bool>{true, true}));
        auto refreshed = Factory::refresh_multi_column_rhs(
            terms, boundary, options,
            Factory::MultiColumnWeightedContext{.cached_matrix = sys.matrix, .cached_rhs = sys.rhs});
        EXPECT_EQ(refreshed.active_columns, (std::vector<bool>{true, true}));

        CarrierMultiVectorBuffer<Pack> buffer;
        std::vector<const ScalarField*> src_fields{&old0, &old1};
        (void)buffer.pack_owned_fields(
            tiny_mesh->owned_cell_map(),
            std::span<const ScalarField* const>(src_fields));

        Pack::multi_vector_type sol(tiny_mesh->owned_cell_map(), 2, true);
        BelosLinearSolver<Pack> solver;
        LinearSolverOptions lopts;
        lopts.backend = LinearSolverBackend::Gmres;
        lopts.tolerance = 1.0e-12;
        const auto stats = solver.solve_multi_rhs_from_zero_with_statistics(
            refreshed.matrix, *refreshed.rhs, sol, MultiRhsSolveOptions{lopts, {}, {}, refreshed.active_columns});
        ASSERT_TRUE(stats.converged);
        if (owned == 1)
        {
            EXPECT_NEAR(sol.getData(0)[0], 3.0, 1.0e-12);
            EXPECT_NEAR(sol.getData(1)[0], -7.0, 1.0e-12);
        }
    }

    // 2. ALE old/new volumes & mesh-relative flux contract (no double subtraction of mesh flux).
    {
        auto geometry = std::make_shared<Handle::Cartesian>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.5, 1.0, 1.5, 2.0}}});
        auto mesh = std::make_shared<Handle>(std::move(geometry));
        const auto owned = mesh->num_owned_cells();
        const scalar_type dt = 0.25;
        Motion motion(mesh);
        const double z1 = 2.4;
        motion.begin_trial(z1, dt);
        const auto ale = make_validated_planar_ale_control_volume_state(*mesh, motion);

        ScalarField storage(mesh, 1.0, "storage");
        ScalarField advection(mesh, 1.0, "advection");
        ScalarField zero(mesh, 0.0, "zero");
        FaceField rel_flux(mesh, 0.0, "relative_flux");
        ScalarField q0(mesh, 0.0, "q0");
        ScalarField q1(mesh, 0.0, "q1");
        for (std::size_t i = 0; i < owned; ++i)
        {
            const auto lid = static_cast<LO>(i);
            const auto z = mesh->cell_centroid(lid).z;
            storage.set_owned_value(lid, 1.0 + 0.1 * z);
            q0.set_owned_value(lid, 2.0 + 0.5 * z);
            q1.set_owned_value(lid, -3.0 + 1.5 * z);
        }
        // Zero mesh-relative flux tests pure volume expansion/contraction conservation:
        // V_new * w * q_new = V_old * w * q_old.
        storage.sync_ghosts();
        q0.sync_ghosts();
        q1.sync_ghosts();
        rel_flux.sync_ghosts();

        auto cond = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; };
        Factory::WeightedScalarOptions opts;

        const auto scalar0 = Factory::assemble(
            Factory::WeightedScalarTerms{
                .history = {q0, dt}, .face_fluxes = rel_flux, .coefficients = {storage, advection, zero}},
            Factory::WeightedScalarBoundary{.condition = cond, .value = [](int, std::size_t) { return 0.0; }},
            opts,
            Factory::WeightedScalarContext{.ale = &ale});

        Factory::MultiColumnWeightedTerms mc_terms{
            .time_step = dt,
            .face_fluxes = rel_flux,
            .coefficients = {storage, advection, zero},
            .columns = {
                MultiColumnScalarInput<Pack, Handle>{.old_values = &q0},
                MultiColumnScalarInput<Pack, Handle>{.old_values = &q1, .allow_negative = true}}};
        auto mc_sys = Factory::assemble_multi_column(
            mc_terms,
            Factory::MultiColumnWeightedBoundary{.condition = cond},
            opts,
            Factory::MultiColumnWeightedContext{.ale = &ale});

        expect_matrix_equal(*scalar0.matrix, *mc_sys.matrix);
        for (std::size_t r = 0; r < owned; ++r)
        {
            EXPECT_DOUBLE_EQ(mc_sys.rhs->getData(0)[r], scalar0.rhs->getData()[r]);
        }

        Pack::multi_vector_type sol(mesh->owned_cell_map(), 2, true);
        BelosLinearSolver<Pack> solver;
        LinearSolverOptions lopts;
        lopts.backend = LinearSolverBackend::Gmres;
        lopts.tolerance = 1.0e-13;
        ASSERT_TRUE(solver.solve_multi_rhs_from_zero_with_statistics(
            mc_sys.matrix, *mc_sys.rhs, sol, MultiRhsSolveOptions{lopts, {}, {}, mc_sys.active_columns}).converged);

        const auto w = storage.owned_read_view();
        for (std::size_t i = 0; i < owned; ++i)
        {
            const auto expected0 = (ale.old_cell_volumes()[i] / ale.new_cell_volumes()[i]) * q0.owned_read_view()(i, 0);
            const auto expected1 = (ale.old_cell_volumes()[i] / ale.new_cell_volumes()[i]) * q1.owned_read_view()(i, 0);
            EXPECT_NEAR(sol.getData(0)[i], expected0, 1.0e-12);
            EXPECT_NEAR(sol.getData(1)[i], expected1, 1.0e-12);
            EXPECT_NEAR(ale.new_cell_volumes()[i] * w(i, 0) * sol.getData(0)[i],
                ale.old_cell_volumes()[i] * w(i, 0) * q0.owned_read_view()(i, 0), 1.0e-12);
        }
        motion.accept_trial();
    }
}

TEST(MultiColumnTransportTest, BoundedLinearUpwindWithInhomogeneousBoundaries)
{
    const auto mesh = make_distributed_line(8);
    const auto owned = mesh->num_owned_cells();

    ScalarField storage(mesh, 1.0, "storage");
    ScalarField advection(mesh, 1.0, "advection");
    ScalarField zero(mesh, 0.0, "zero");
    FaceField flux(mesh, 0.0, "flux");
    for (const auto face : flux.owned_face_ids())
    {
        flux.set_owned_value(face, 0.25 * mesh->face_area_vector(face).x);
    }
    flux.sync_ghosts();

    ScalarField old0(mesh, 0.0, "old0");
    ScalarField old1(mesh, 0.0, "old1");
    for (std::size_t i = 0; i < owned; ++i)
    {
        const auto lid = static_cast<LO>(i);
        const auto x = mesh->cell_centroid(lid).x;
        old0.set_owned_value(lid, 1.0 + 3.0 * x * x);
        old1.set_owned_value(lid, -5.0 + 4.0 * x);
    }
    old0.sync_ghosts();
    old1.sync_ghosts();

    // Raw boundary.condition sets a nonzero Neumann value (99.0), while columns
    // explicitly override neumann_flux (col 0 = 0.0, col 1 = -2.5) and Dirichlet values.
    auto bc_cond = [](int batch, std::size_t)
    {
        return BoundaryCondition{
            batch % 2 == 0 ? BoundaryConditionType::Dirichlet : BoundaryConditionType::Neumann, 99.0};
    };

    const scalar_type dt = 0.2;
    TransportGeometryCache<Handle> geometry(*mesh);
    Factory::WeightedScalarOptions options;
    options.discretization = {
        .time = ScalarTimeScheme::BackwardEuler,
        .convection = ScalarConvectionScheme::BoundedLinearUpwind};

    std::vector<TransportSystem<Pack>> scalar_ref;
    for (std::size_t c = 0; c < 2; ++c)
    {
        const auto* old_f = (c == 0) ? &old0 : &old1;
        auto bv = [c](int, std::size_t) -> scalar_type { return c == 0 ? 4.0 : -3.5; };
        auto cond = [&, c](int batch, std::size_t face)
        {
            auto b = bc_cond(batch, face);
            if (b.type == BoundaryConditionType::Neumann)
            {
                b.value = (c == 0) ? 0.0 : -2.5;
            }
            return b;
        };
        scalar_ref.push_back(Factory::assemble(
            Factory::WeightedScalarTerms{
                .history = {*old_f, dt},
                .face_fluxes = flux,
                .coefficients = {storage, advection, zero}},
            Factory::WeightedScalarBoundary{.condition = cond, .value = bv},
            options,
            Factory::WeightedScalarContext{.geometry_cache = &geometry}));
    }

    Factory::MultiColumnWeightedTerms mc_terms{
        .time_step = dt,
        .face_fluxes = flux,
        .coefficients = {storage, advection, zero},
        .columns = {
            MultiColumnScalarInput<Pack, Handle>{
                .old_values = &old0,
                .boundary_value = [](int, std::size_t) { return 4.0; },
                .neumann_flux = [](int, std::size_t) { return 0.0; }},
            MultiColumnScalarInput<Pack, Handle>{
                .old_values = &old1,
                .boundary_value = [](int, std::size_t) { return -3.5; },
                .neumann_flux = [](int, std::size_t) { return -2.5; },
                .allow_negative = true}}};

    auto mc_sys = Factory::assemble_multi_column(
        mc_terms,
        Factory::MultiColumnWeightedBoundary{.condition = bc_cond},
        options,
        Factory::MultiColumnWeightedContext{.geometry_cache = &geometry});

    for (std::size_t c = 0; c < 2; ++c)
    {
        expect_matrix_equal(*scalar_ref[c].matrix, *mc_sys.matrix);
        const auto s_rhs = scalar_ref[c].rhs->getData();
        const auto m_rhs = mc_sys.rhs->getData(c);
        for (std::size_t r = 0; r < owned; ++r)
        {
            EXPECT_DOUBLE_EQ(m_rhs[r], s_rhs[r]);
        }
    }

    // Mutate old0/old1 and refresh RHS only: must match fresh scalar assembly bit-for-bit.
    for (std::size_t i = 0; i < owned; ++i)
    {
        const auto lid = static_cast<LO>(i);
        old0.set_owned_value(lid, 2.0 * old0.owned_read_view()(i, 0));
        old1.set_owned_value(lid, old1.owned_read_view()(i, 0) - 1.0);
    }
    old0.sync_ghosts();
    old1.sync_ghosts();

    auto refreshed = Factory::refresh_multi_column_rhs(
        mc_terms,
        Factory::MultiColumnWeightedBoundary{.condition = bc_cond},
        options,
        Factory::MultiColumnWeightedContext{
            .cached_matrix = mc_sys.matrix, .geometry_cache = &geometry, .cached_rhs = mc_sys.rhs});
    EXPECT_DOUBLE_EQ(refreshed.timings.matrix_assembly_seconds, 0.0);

    for (std::size_t c = 0; c < 2; ++c)
    {
        const auto* old_f = (c == 0) ? &old0 : &old1;
        auto bv = [c](int, std::size_t) -> scalar_type { return c == 0 ? 4.0 : -3.5; };
        auto cond = [&, c](int batch, std::size_t face)
        {
            auto b = bc_cond(batch, face);
            if (b.type == BoundaryConditionType::Neumann)
            {
                b.value = (c == 0) ? 0.0 : -2.5;
            }
            return b;
        };
        auto fresh = Factory::assemble(
            Factory::WeightedScalarTerms{
                .history = {*old_f, dt},
                .face_fluxes = flux,
                .coefficients = {storage, advection, zero}},
            Factory::WeightedScalarBoundary{.condition = cond, .value = bv},
            options,
            Factory::WeightedScalarContext{.geometry_cache = &geometry});
        const auto s_rhs = fresh.rhs->getData();
        const auto m_rhs = refreshed.rhs->getData(c);
        for (std::size_t r = 0; r < owned; ++r)
        {
            EXPECT_DOUBLE_EQ(m_rhs[r], s_rhs[r]);
        }
    }
}

TEST(MultiColumnTransportTest, CollectiveUniformityRejectsRankNonuniformMultiColumnMetadata)
{
    const auto mesh = make_distributed_line(8);
    const auto comm = mesh->owned_cell_map()->getComm();
    const std::size_t owned = mesh->num_owned_cells();
    const double dt = 0.2;

    ScalarField storage(mesh, 1.0, "w");
    ScalarField advection(mesh, 1.0, "wa");
    ScalarField diffusion(mesh, 0.0, "D");
    ScalarField init_phi(mesh, 1.0, "phi");
    ScalarField old_phi(mesh, 0.5, "old_phi");
    FaceField flux(mesh, 0.0, "F");
    TransportGeometryCache<Handle> geometry(*mesh);
    Factory::WeightedScalarOptions options;

    auto make_bc = []()
    {
        return Factory::MultiColumnWeightedBoundary{
            .condition = [](int, std::size_t)
            { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; }};
    };

    // Rank-local invalid extensive inventory span size throws collectively without hanging.
    {
        std::vector<double> bad_extensive(owned + 1, 1.0);
        std::vector<MultiColumnScalarInput<Pack, Handle>> cols(1);
        cols[0].extensive_old_inventories = bad_extensive;
        EXPECT_THROW(
            Factory::assemble_multi_column(
                Factory::MultiColumnWeightedTerms{
                    .time_step = dt,
                    .face_fluxes = flux,
                    .coefficients = {storage, advection, diffusion},
                    .columns = cols},
                make_bc(),
                options,
                Factory::MultiColumnWeightedContext{.geometry_cache = &geometry}),
            std::invalid_argument);
    }

    if (comm->getSize() >= 2)
    {
        // Rank 0 uses old_values while rank 1 uses extensive_old_inventories -> collective throw.
        {
            std::vector<double> ok_extensive(owned, 0.5);
            std::vector<MultiColumnScalarInput<Pack, Handle>> cols(1);
            if (comm->getRank() == 0)
            {
                cols[0].old_values = &old_phi;
            }
            else
            {
                cols[0].extensive_old_inventories = ok_extensive;
            }
            EXPECT_THROW(
                Factory::assemble_multi_column(
                    Factory::MultiColumnWeightedTerms{
                        .time_step = dt,
                        .face_fluxes = flux,
                        .coefficients = {storage, advection, diffusion},
                        .columns = cols},
                    make_bc(),
                    options,
                    Factory::MultiColumnWeightedContext{.geometry_cache = &geometry}),
                std::invalid_argument);
        }

        // Rank 0 supplies lagged_correction_field while rank 1 leaves it null -> collective throw.
        {
            std::vector<MultiColumnScalarInput<Pack, Handle>> cols(1);
            cols[0].old_values = &old_phi;
            cols[0].lagged_correction_field = (comm->getRank() == 0) ? &init_phi : nullptr;
            EXPECT_THROW(
                Factory::assemble_multi_column(
                    Factory::MultiColumnWeightedTerms{
                        .time_step = dt,
                        .face_fluxes = flux,
                        .coefficients = {storage, advection, diffusion},
                        .columns = cols},
                    make_bc(),
                    options,
                    Factory::MultiColumnWeightedContext{.geometry_cache = &geometry}),
                std::invalid_argument);
        }
    }
}

TEST(MultiColumnTransportTest, NonOrthogonalDefaultOptionsMatchScalarOldValuesAndExtensiveInventories)
{
    const auto mesh = test::make_skewed_prism_mesh<Pack>(4, 3, 3);
    const std::size_t owned = mesh->num_owned_cells();
    const double dt = 0.15;

    ScalarField storage(mesh, 1.25, "storage");
    ScalarField advection(mesh, 1.25, "advection");
    ScalarField diffusion(mesh, 0.2, "diffusion");
    ScalarField init_phi(mesh, 0.0, "init_phi");
    ScalarField old_phi(mesh, 0.0, "old_phi");
    ScalarField scaled_old_phi(mesh, 0.0, "scaled_old_phi");
    ScalarField src_0(mesh, 0.3, "src_0");
    ScalarField src_1(mesh, -0.45, "src_1");
    FaceField flux(mesh, 0.0, "flux");
    TransportGeometryCache<Handle> geometry(*mesh);

    std::vector<double> ext_0(owned, 0.0);
    std::vector<double> ext_1(owned, 0.0);
    std::vector<double> ext_refreshed_1(owned, 0.0);

    for (std::size_t r = 0; r < owned; ++r)
    {
        const auto cell = static_cast<LO>(r);
        const auto p = mesh->cell_centroid(cell);
        const double phi0 = 1.0 + 2.0 * p.x + 3.0 * p.y + 0.5 * p.z;
        const double old0 = 0.4 + 0.7 * p.x + 1.1 * p.y + 0.3 * p.z;
        init_phi.set_owned_value(cell, phi0);
        old_phi.set_owned_value(cell, old0);
        scaled_old_phi.set_owned_value(cell, 2.0 * old0 + 0.1);
        const double cap = storage.value(cell) * mesh->cell_volume(cell);
        ext_0[r] = cap * old0;
        ext_1[r] = cap * (2.0 * old0 + 0.1);
        ext_refreshed_1[r] = cap * (3.5 * old0 + 0.2);
    }
    init_phi.sync_ghosts();
    old_phi.sync_ghosts();
    scaled_old_phi.sync_ghosts();

    for (const auto face : flux.owned_face_ids())
    {
        const auto s = mesh->face_area_vector(face);
        flux.set_owned_value(face, 0.3 * s.x + 0.15 * s.y - 0.1 * s.z);
    }
    flux.sync_ghosts();

    // Default assembly options: Explicit non-orthogonal treatment with null lagged_field,
    // so both scalar and multi-column must leave explicit cross-diffusion lagged_field null
    // instead of falling back to old_values (while still honoring explicit per-column lagged_correction_field).
    Factory::WeightedScalarOptions options;

    auto base_cond = [&mesh](int batch, std::size_t)
    {
        const auto patch = mesh->boundary_batch_name(batch);
        if (patch == "xmin")
        {
            return BoundaryCondition{BoundaryConditionType::Dirichlet, 0.0};
        }
        if (patch == "xmax")
        {
            return BoundaryCondition{BoundaryConditionType::Neumann, 0.0};
        }
        return BoundaryCondition{BoundaryConditionType::Neumann, 0.0};
    };

    auto make_scalar_bc = [&](double scale)
    {
        return Factory::WeightedScalarBoundary{
            .condition = [&mesh, scale](int batch, std::size_t)
            {
                const auto patch = mesh->boundary_batch_name(batch);
                if (patch == "xmin")
                {
                    return BoundaryCondition{BoundaryConditionType::Dirichlet, 1.0 * scale};
                }
                if (patch == "xmax")
                {
                    return BoundaryCondition{BoundaryConditionType::Neumann, -0.25 * scale};
                }
                return BoundaryCondition{BoundaryConditionType::Neumann, 0.0};
            },
            .value = [&mesh, scale](int batch, std::size_t)
            { return mesh->boundary_batch_name(batch) == "xmin" ? 1.0 * scale : 0.0; }};
    };

    auto src_0_fn = [&src_0](LO r) { return src_0.value(r); };
    auto src_1_fn = [&src_1](LO r) { return src_1.value(r); };

    const auto bc0 = make_scalar_bc(1.0);
    const auto bc1 = make_scalar_bc(2.0);
    const auto scalar0 = Factory::assemble(
        Factory::WeightedScalarTerms{
            .history = {old_phi, dt},
            .face_fluxes = flux,
            .coefficients = {storage, advection, diffusion},
            .sources = {.source = src_0_fn}},
        bc0, options, Factory::WeightedScalarContext{.geometry_cache = &geometry});
    const auto scalar1 = Factory::assemble(
        Factory::WeightedScalarTerms{
            .history = {scaled_old_phi, dt},
            .face_fluxes = flux,
            .coefficients = {storage, advection, diffusion},
            .sources = {.source = src_1_fn}},
        bc1, options, Factory::WeightedScalarContext{.geometry_cache = &geometry});

    auto mc_bv = [&mesh](double scale)
    {
        return [&mesh, scale](int batch, std::size_t)
        { return mesh->boundary_batch_name(batch) == "xmin" ? 1.0 * scale : 0.0; };
    };
    auto mc_nf = [&mesh](double scale)
    {
        return [&mesh, scale](int batch, std::size_t)
        { return mesh->boundary_batch_name(batch) == "xmax" ? -0.25 * scale : 0.0; };
    };

    // Case A: multi-column with old_values.
    Factory::MultiColumnWeightedTerms terms_old{
        .time_step = dt,
        .face_fluxes = flux,
        .coefficients = {storage, advection, diffusion},
        .columns = {
            MultiColumnScalarInput<Pack, Handle>{
                .old_values = &old_phi,
                .source = src_0_fn,
                .boundary_value = mc_bv(1.0),
                .neumann_flux = mc_nf(1.0)},
            MultiColumnScalarInput<Pack, Handle>{
                .old_values = &scaled_old_phi,
                .source = src_1_fn,
                .boundary_value = mc_bv(2.0),
                .neumann_flux = mc_nf(2.0)}}};
    auto multi_old = Factory::assemble_multi_column(
        terms_old,
        Factory::MultiColumnWeightedBoundary{.condition = base_cond},
        options,
        Factory::MultiColumnWeightedContext{.geometry_cache = &geometry});

    // Case B: multi-column with equivalent extensive_old_inventories.
    Factory::MultiColumnWeightedTerms terms_ext{
        .time_step = dt,
        .face_fluxes = flux,
        .coefficients = {storage, advection, diffusion},
        .columns = {
            MultiColumnScalarInput<Pack, Handle>{
                .extensive_old_inventories = ext_0,
                .source = src_0_fn,
                .boundary_value = mc_bv(1.0),
                .neumann_flux = mc_nf(1.0)},
            MultiColumnScalarInput<Pack, Handle>{
                .extensive_old_inventories = ext_1,
                .source = src_1_fn,
                .boundary_value = mc_bv(2.0),
                .neumann_flux = mc_nf(2.0)}}};
    auto multi_ext = Factory::assemble_multi_column(
        terms_ext,
        Factory::MultiColumnWeightedBoundary{.condition = base_cond},
        options,
        Factory::MultiColumnWeightedContext{.geometry_cache = &geometry});

    expect_matrix_equal(*scalar0.matrix, *multi_old.matrix);
    expect_matrix_equal(*scalar0.matrix, *multi_ext.matrix);

    const std::array<const Pack::vector_type*, 2> scalar_rhs{scalar0.rhs.get(), scalar1.rhs.get()};
    for (std::size_t col = 0; col < 2; ++col)
    {
        const auto sr = scalar_rhs[col]->getData();
        const auto mr_old = multi_old.rhs->getData(col);
        const auto mr_ext = multi_ext.rhs->getData(col);
        for (std::size_t r = 0; r < owned; ++r)
        {
            EXPECT_DOUBLE_EQ(mr_old[r], sr[r]);
            EXPECT_DOUBLE_EQ(mr_ext[r], sr[r]);
        }
    }

    // Solve both scalar and multi-column systems and compare solutions.
    BelosLinearSolver<Pack> solver;
    LinearSolverOptions lin_opts;
    lin_opts.tolerance = 1.0e-12;
    Pack::vector_type sol_s0(scalar0.rhs->getMap());
    Pack::vector_type sol_s1(scalar1.rhs->getMap());
    Pack::multi_vector_type sol_m(multi_ext.rhs->getMap(), 2);
    ASSERT_TRUE(solver.solve(scalar0.matrix, *scalar0.rhs, sol_s0, lin_opts));
    ASSERT_TRUE(solver.solve(scalar1.matrix, *scalar1.rhs, sol_s1, lin_opts));
    MultiRhsSolveOptions multi_opts;
    multi_opts.solver = lin_opts;
    ASSERT_TRUE(solver.solve_multi_rhs_from_zero_with_statistics(
        multi_ext.matrix, *multi_ext.rhs, sol_m, multi_opts).converged);
    for (std::size_t r = 0; r < owned; ++r)
    {
        EXPECT_NEAR(sol_m.getData(0)[r], sol_s0.getData()[r], 1.0e-11);
        EXPECT_NEAR(sol_m.getData(1)[r], sol_s1.getData()[r], 1.0e-11);
    }

    // Verify refresh_multi_column_rhs matches fresh scalar assembly with mixed old_values + extensive_old_inventories,
    // and also verify explicit per-column lagged_correction_field matches scalar options.correction.lagged_field.
    ScalarField old_refreshed_1(mesh, 0.0, "old_ref1");
    for (std::size_t r = 0; r < owned; ++r)
    {
        const auto cell = static_cast<LO>(r);
        old_refreshed_1.set_owned_value(cell, ext_refreshed_1[r] / (storage.value(cell) * mesh->cell_volume(cell)));
    }
    old_refreshed_1.sync_ghosts();
    auto lag_opts = options;
    lag_opts.correction.lagged_field = &init_phi;
    const auto fresh1 = Factory::assemble(
        Factory::WeightedScalarTerms{
            .history = {old_refreshed_1, dt},
            .face_fluxes = flux,
            .coefficients = {storage, advection, diffusion},
            .sources = {.source = src_0_fn}},
        bc1, lag_opts, Factory::WeightedScalarContext{.geometry_cache = &geometry});

    Factory::MultiColumnWeightedTerms refresh_terms{
        .time_step = dt,
        .face_fluxes = flux,
        .coefficients = {storage, advection, diffusion},
        .columns = {
            MultiColumnScalarInput<Pack, Handle>{
                .old_values = &old_phi,
                .source = src_0_fn,
                .boundary_value = mc_bv(1.0),
                .neumann_flux = mc_nf(1.0)},
            MultiColumnScalarInput<Pack, Handle>{
                .extensive_old_inventories = ext_refreshed_1,
                .source = src_0_fn,
                .boundary_value = mc_bv(2.0),
                .neumann_flux = mc_nf(2.0),
                .lagged_correction_field = &init_phi}}};
    const auto refreshed = Factory::refresh_multi_column_rhs(
        refresh_terms,
        Factory::MultiColumnWeightedBoundary{.condition = base_cond},
        options,
        Factory::MultiColumnWeightedContext{
            .cached_matrix = multi_old.matrix, .geometry_cache = &geometry, .cached_rhs = multi_old.rhs});
    for (std::size_t r = 0; r < owned; ++r)
    {
        EXPECT_DOUBLE_EQ(refreshed.rhs->getData(0)[r], scalar0.rhs->getData()[r]);
        EXPECT_DOUBLE_EQ(refreshed.rhs->getData(1)[r], fresh1.rhs->getData()[r]);
    }
}

} // namespace
