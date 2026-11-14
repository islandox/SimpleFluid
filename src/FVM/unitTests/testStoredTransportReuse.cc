/**
 * @file testStoredTransportReuse.cc
 * @brief  Repeated mapped transport and immutable graph contracts.
 */
#include "FVM/TransportSystemFactory.hh"
#include <gtest/gtest.h>
#include <type_traits>

#include "FVM/Operators.hh"
#include "transport_providers/TestMeshes.hh"
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
template<class Field>
using TransportFactory = SimpleFluid::FVM::TransportSystemFactory<Pack, typename std::remove_cvref_t<Field>::mesh_type>;
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
    Teuchos::RCP<Pack::vector_type> rhs;
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
        const auto assemble = [&](Teuchos::RCP<Pack::matrix_type> cached, Plan* selected_plan,
                                  Teuchos::RCP<Pack::vector_type> cached_rhs = Teuchos::null)
        {
            return TransportFactory<decltype(values)>::assemble(
                typename TransportFactory<decltype(values)>::WeightedScalarTerms{
                    .history = {.old_values = values, .time_step = 0.5},
                    .face_fluxes = flux,
                    .coefficients = {.storage_weight = storage, .advection_weight = storage, .diffusivity = diffusion},
                    .sources = {.source = [](LO) { return 0.125; }},
                    .constraints = {}},
                {.condition =
                        [&](int batch, size_t)
                    {
                        return BoundaryCondition{(batch + generation) % 2 ? BoundaryConditionType::Dirichlet
                                                                          : BoundaryConditionType::Neumann,
                            0.125 * generation};
                    },
                    .value = [&](int, size_t) { return 0.75 + generation; }},
                {.correction = {.treatment = NonOrthogonalTreatment::Hybrid, .lagged_field = &values}},
                {.cached_matrix = std::move(cached),
                    .geometry_cache = &geometry,
                    .symbolic_plan = selected_plan,
                    .cached_rhs = std::move(cached_rhs)});
        };
        const auto fresh = assemble(Teuchos::null, nullptr);
        const auto reused = assemble(matrix, &plan, rhs);
        if (!rhs.is_null()) EXPECT_EQ(rhs.get(), reused.rhs.get());
        EXPECT_TRUE(reused.matrix->isStaticGraph());
        if (!matrix.is_null())
        {
            EXPECT_EQ(matrix.get(), reused.matrix.get());
        }
        expect_same_system(reused, fresh);
        matrix = reused.matrix;
        rhs = reused.rhs;
        if (generation == 0)
        {
            const auto foreign_mesh = make_mesh();
            auto foreign_rhs = Teuchos::rcp(new Pack::vector_type(foreign_mesh->owned_cell_map()));
            if (mesh->owned_cell_map()->getComm()->getRank() != 0) foreign_rhs = rhs;
            EXPECT_THROW(assemble(matrix, &plan, foreign_rhs), std::invalid_argument);
            expect_same_system(reused, fresh);
        }
    }
}

TEST(StoredTransportReuseTest, SymbolicPlanValidatesChangedGraphsEpochsAndRankCategories)
{
    const auto mesh = make_mesh();
    test::transport_providers::EpochMesh epoch_mesh{*mesh};
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
    // Motion changes coefficients, not row slots. Required columns and exact
    // map/graph identities still guard every symbolic reuse.
    EXPECT_TRUE(plan.matches(epoch_mesh, matrix, rows));
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
            return TransportFactory<decltype(velocity)>::assemble(
                typename TransportFactory<decltype(velocity)>::MomentumTerms{
                    .history = {.old_values = velocity, .time_step = 0.5},
                    .face_fluxes = flux,
                    .material = {.dynamic_viscosity = viscosity, .reference_density = 2.0},
                    .acceleration_source = [](LO) { return Handle::Vec3{}; }},
                {.value = [&](int batch, size_t face)
                    { return velocity_at(mesh->face_centroid(mesh->boundary_face_batch(batch).face_lids[face])); },
                    .diffusion = [](int batch, size_t) { return batch % 2 == 0; },
                    .coefficients = nullptr},
                {.correction = {.treatment = NonOrthogonalTreatment::Hybrid, .lagged_field = &velocity},
                    .coefficient_interpolation = FaceCoefficientInterpolation::Linear},
                {.cached_matrix = Teuchos::null,
                    .geometry_cache = &geometry,
                    .ale = nullptr,
                    .gradient_workspace = workspace});
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
    EXPECT_THROW(TransportFactory<decltype(velocity)>::assemble(
                     typename TransportFactory<decltype(velocity)>::MomentumTerms{
                         .history = {.old_values = velocity, .time_step = 0.5},
                         .face_fluxes = flux,
                         .material = {.dynamic_viscosity = viscosity, .reference_density = 2.0},
                         .acceleration_source = [](LO) { return Handle::Vec3{}; }},
                     {.value = [](int, size_t) { return Handle::Vec3{}; },
                         .diffusion = FVM::detail::AlwaysDiffuseBoundary{},
                         .coefficients = nullptr},
                     {.correction = {.treatment = NonOrthogonalTreatment::Explicit, .lagged_field = nullptr},
                         .coefficient_interpolation = FaceCoefficientInterpolation::Linear},
                     {.cached_matrix = Teuchos::null,
                         .geometry_cache = &geometry,
                         .ale = nullptr,
                         .gradient_workspace = rank == 0 ? &foreign : &scratch}),
        std::invalid_argument);
}

TEST(StoredTransportReuseTest, PhysicalCapacitySharesWeightedAlgebraAndIndependentOldFactors)
{
    const auto mesh = make_mesh();
    Scalar values(mesh, 0.0, "temperature"), density(mesh, 1.0, "density"), cp(mesh, 1.0, "cp"),
        old_density(mesh, 1.0, "old_density"), old_cp(mesh, 1.0, "old_cp"), capacity(mesh, 1.0, "capacity"),
        old_capacity(mesh, 1.0, "old_capacity"), diffusion(mesh, 0.0, "conductivity"), zero(mesh, 0.0, "zero");
    Flux flux(mesh, 0.0, "flux");
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto x = mesh->cell_centroid(cell).x;
        values.set_owned_value(cell, 2.0 + 0.5 * x);
        density.set_owned_value(cell, 1.5 + x);
        cp.set_owned_value(cell, 4.0 - 0.25 * x);
        old_density.set_owned_value(cell, 0.75 + 0.5 * x);
        old_cp.set_owned_value(cell, 2.0 + 0.125 * x);
        capacity.set_owned_value(cell, (1.5 + x) * (4.0 - 0.25 * x));
        diffusion.set_owned_value(cell, 0.2 + 0.1 * x);
    }
    values.sync_ghosts();
    density.sync_ghosts();
    cp.sync_ghosts();
    old_density.sync_ghosts();
    old_cp.sync_ghosts();
    capacity.sync_ghosts();
    diffusion.sync_ghosts();
    TransportGeometryCache<Handle> geometry(*mesh);
    const auto boundary = [](int batch, size_t)
    { return BoundaryCondition{batch % 2 ? BoundaryConditionType::Dirichlet : BoundaryConditionType::Neumann, 0.125}; };
    const auto boundary_value = [](int batch, size_t) { return 1.0 + 0.25 * batch; };
    const auto source = [](LO cell) { return 0.125 + 0.0625 * cell; };
    constexpr double dt = 0.5;
    for (int selection = 0; selection < 4; ++selection)
    {
        SCOPED_TRACE(selection);
        const auto* accepted_density = selection & 1 ? &old_density : nullptr;
        const auto* accepted_cp = selection & 2 ? &old_cp : nullptr;
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto x = mesh->cell_centroid(cell).x;
            const auto rho = selection & 1 ? 0.75 + 0.5 * x : 1.5 + x;
            const auto heat_capacity = selection & 2 ? 2.0 + 0.125 * x : 4.0 - 0.25 * x;
            old_capacity.set_owned_value(cell, rho * heat_capacity);
        }
        old_capacity.sync_ghosts();
        // Independent storage oracle with no face contributions, including
        // each individual accepted-property fallback.
        flux.data().putScalar(0.0);
        flux.sync_ghosts();
        const auto storage = TransportFactory<decltype(values)>::assemble(
            typename TransportFactory<decltype(values)>::TemperatureTerms{
                .history = {.old_values = values, .time_step = dt},
                .face_fluxes = flux,
                .material = {.density = density,
                    .specific_heat_capacity = cp,
                    .thermal_conductivity = zero,
                    .old_density = accepted_density,
                    .old_specific_heat_capacity = accepted_cp},
                .power_density = source},
            {.condition = boundary, .value = boundary_value, .coefficients = nullptr},
            {.correction = {.treatment = NonOrthogonalTreatment::Implicit, .lagged_field = nullptr},
                .coefficient_interpolation = FaceCoefficientInterpolation::Harmonic},
            {.cached_matrix = Teuchos::null, .geometry_cache = &geometry, .ale = nullptr});
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const auto volume = mesh->cell_volume(cell);
            const auto entries = [&]
            {
                Pack::matrix_type::local_inds_host_view_type columns;
                Pack::matrix_type::values_host_view_type coefficients;
                storage.matrix->getLocalRowView(cell, columns, coefficients);
                double diagonal = 0.0;
                for (size_t i = 0; i < columns.extent(0); ++i)
                {
                    if (columns[i] == cell)
                        diagonal += coefficients[i];
                    else
                        EXPECT_DOUBLE_EQ(coefficients[i], 0.0);
                }
                return diagonal;
            }();
            EXPECT_DOUBLE_EQ(entries, capacity.value(cell) * volume / dt);
            EXPECT_NEAR(storage.rhs->getData()[row],
                old_capacity.value(cell) * volume / dt * values.value(cell) + volume * source(cell), 1.0e-12);
        }
        for (const auto face : flux.owned_face_ids())
            flux.set_owned_value(face, (selection % 2 ? -0.3 : 0.2) * mesh->face_area_vector(face).x);
        flux.sync_ghosts();
        for (auto interpolation : {FaceCoefficientInterpolation::Harmonic, FaceCoefficientInterpolation::Linear})
        {
            for (auto treatment :
                {NonOrthogonalTreatment::Explicit, NonOrthogonalTreatment::Implicit, NonOrthogonalTreatment::Hybrid})
            {
                const auto* correction = treatment == NonOrthogonalTreatment::Implicit ? nullptr : &values;
                const auto weighted = TransportFactory<decltype(values)>::assemble(
                    typename TransportFactory<decltype(values)>::WeightedScalarTerms{
                        .history = {.old_values = values, .time_step = dt},
                        .face_fluxes = flux,
                        .coefficients = {.storage_weight = capacity,
                            .advection_weight = capacity,
                            .diffusivity = diffusion,
                            .old_storage_weight = selection == 0 ? nullptr : &old_capacity},
                        .sources = {.source = source},
                        .constraints = {}},
                    {.condition = boundary, .value = boundary_value},
                    {.correction = {.treatment = treatment, .lagged_field = correction},
                        .coefficient_interpolation = interpolation},
                    {.geometry_cache = &geometry});
                const auto physical = TransportFactory<decltype(values)>::assemble(
                    typename TransportFactory<decltype(values)>::TemperatureTerms{
                        .history = {.old_values = values, .time_step = dt},
                        .face_fluxes = flux,
                        .material = {.density = density,
                            .specific_heat_capacity = cp,
                            .thermal_conductivity = diffusion,
                            .old_density = accepted_density,
                            .old_specific_heat_capacity = accepted_cp},
                        .power_density = source},
                    {.condition = boundary, .value = boundary_value, .coefficients = nullptr},
                    {.correction = {.treatment = treatment, .lagged_field = correction},
                        .coefficient_interpolation = interpolation},
                    {.cached_matrix = Teuchos::null, .geometry_cache = &geometry, .ale = nullptr});
                expect_same_system(physical, weighted);
                physical.matrix->setAllToScalar(7.0);
                const auto rhs_only = TransportFactory<decltype(values)>::refresh_temperature_rhs(
                    typename TransportFactory<decltype(values)>::TemperatureTerms{
                        .history = {.old_values = values, .time_step = dt},
                        .face_fluxes = flux,
                        .material = {.density = density,
                            .specific_heat_capacity = cp,
                            .thermal_conductivity = diffusion,
                            .old_density = accepted_density,
                            .old_specific_heat_capacity = accepted_cp},
                        .power_density = source},
                    {.condition = boundary, .value = boundary_value, .coefficients = nullptr},
                    {.correction = {.treatment = treatment, .lagged_field = correction},
                        .coefficient_interpolation = interpolation},
                    {.cached_matrix = physical.matrix, .geometry_cache = &geometry, .ale = nullptr});
                EXPECT_EQ(rhs_only.matrix.get(), physical.matrix.get());
                for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
                {
                    EXPECT_DOUBLE_EQ(rhs_only.rhs->getData()[row], weighted.rhs->getData()[row]);
                    Pack::matrix_type::local_inds_host_view_type columns;
                    Pack::matrix_type::values_host_view_type coefficients;
                    rhs_only.matrix->getLocalRowView(static_cast<LO>(row), columns, coefficients);
                    for (size_t i = 0; i < coefficients.extent(0); ++i)
                        EXPECT_DOUBLE_EQ(coefficients[i], 7.0);
                }
            }
        }
    }
}

} // namespace
