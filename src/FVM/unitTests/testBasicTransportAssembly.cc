/** @file testBasicTransportAssembly.cc
 * @brief Basic scalar/vector assembly parity and callback contracts.
 */
#include "FVM/TransportSystem.hh"
#include "FVM/TransportSystemFactory.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <type_traits>

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
template<class Field>
using TransportFactory = SimpleFluid::FVM::TransportSystemFactory<Pack, typename std::remove_cvref_t<Field>::mesh_type>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

auto make_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
}

double local_matrix_entry(
    const Pack::matrix_type& matrix, Pack::local_ordinal_type row, Pack::local_ordinal_type column)
{
    Pack::matrix_type::local_inds_host_view_type columns;
    Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(row, columns, values);
    for (size_t i = 0; i < columns.extent(0); ++i)
        if (columns[i] == column)
            return values[i];
    return 0.0;
}

/** Basic vector assembly visits callbacks once and retains scalar component algebra. */
TEST(BasicTransportAssemblyTest, BasicTransportSharesMatrixAndPreservesComponentCallbacks)
{
    using namespace SimpleFluid;
    using namespace SimpleFluid::FVM;
    using LO = Pack::local_ordinal_type;
    using Vec = vec3<Pack::scalar_type>;
    const auto mesh = make_mesh();
    VectorFieldType velocity(mesh, Vec{1.25, -2.5, 4.0}, "velocity");
    FaceField<Pack> flux(mesh, 0.0, "flux");
    const auto source_value = [](LO cell) { return Vec{0.5 + cell, -1.0 - cell, 3.0 + 2.0 * cell}; };
    const auto boundary_value = [](int batch, size_t face)
    { return Vec{2.0 + batch + 0.1 * face, -3.0 + 0.2 * face, 5.0 - 0.3 * batch}; };
    const auto boundary_type = [](int batch)
    {
        switch (batch % 3)
        {
            case 0:
                return BoundaryConditionType::Neumann;
            case 1:
                return BoundaryConditionType::NoSlip;
            default:
                return BoundaryConditionType::Dirichlet;
        }
    };
    const Vec gradient{0.25, -0.5, 0.75};
    Teuchos::RCP<Pack::matrix_type> cached;
    for (int generation = 0; generation < 3; ++generation)
    {
        const double diffusion = generation == 1 ? 0.0 : 0.4;
        for (const auto face : flux.owned_face_ids())
        {
            flux.set_value(face, generation == 0 ? 0.0
                                                 : (generation == 1 ? 0.2 : -0.3) *
                                                       mesh->face_area_vector(face).dot(Vec{1.0, -0.5, 0.25}));
        }
        std::vector<LO> source_calls;
        std::vector<std::pair<int, size_t>> value_calls, condition_calls;
        bool awaiting_condition = false;
        const auto vector = TransportFactory<decltype(velocity)>::assemble(
            typename TransportFactory<decltype(velocity)>::VectorTerms{
                .history = {.old_values = velocity, .time_step = 0.25},
                .face_fluxes = flux,
                .coefficients = {.diffusivity = diffusion},
                .source =
                    [&](LO cell)
                {
                    source_calls.push_back(cell);
                    return source_value(cell);
                }},
            typename TransportFactory<decltype(velocity)>::BasicVectorBoundary{
                .condition =
                    [&](int batch, size_t face)
                {
                    EXPECT_TRUE(awaiting_condition);
                    if (!value_calls.empty())
                        EXPECT_EQ(value_calls.back(), (std::pair{batch, face}));
                    awaiting_condition = false;
                    condition_calls.emplace_back(batch, face);
                    return VectorBoundaryCondition{boundary_type(batch), gradient};
                },
                .value =
                    [&](int batch, size_t face)
                {
                    EXPECT_FALSE(awaiting_condition);
                    awaiting_condition = true;
                    value_calls.emplace_back(batch, face);
                    return boundary_value(batch, face);
                }},
            {}, {.cached_matrix = cached});
        EXPECT_FALSE(awaiting_condition);
        ASSERT_EQ(source_calls.size(), mesh->num_owned_cells());
        for (size_t cell = 0; cell < source_calls.size(); ++cell)
            EXPECT_EQ(source_calls[cell], static_cast<LO>(cell));
        size_t expected_boundary_calls = 0;
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
            for (const auto face : batch.face_lids)
                if (mesh->is_owned_face(face) && mesh->is_boundary_face(face))
                    ++expected_boundary_calls;
        EXPECT_EQ(value_calls.size(), expected_boundary_calls);
        EXPECT_EQ(value_calls, condition_calls);
        if (!cached.is_null())
            EXPECT_EQ(vector.matrix.get(), cached.get());
        cached = vector.matrix;
        for (size_t component = 0; component < 3; ++component)
        {
            FieldType scalar(mesh, Vec{1.25, -2.5, 4.0}.component(component), "scalar");
            const auto reference = TransportFactory<decltype(scalar)>::assemble(
                typename TransportFactory<decltype(scalar)>::ScalarTerms{
                    .history = {.old_values = scalar, .time_step = 0.25},
                    .face_fluxes = flux,
                    .coefficients = {.diffusivity = diffusion},
                    .source = [&](LO cell) { return source_value(cell).component(component); }},
                {.condition =
                        [&](int batch, size_t)
                    {
                        const auto type = boundary_type(batch);
                        return BoundaryCondition{
                            type == BoundaryConditionType::NoSlip ? BoundaryConditionType::Dirichlet : type,
                            gradient.component(component)};
                    },
                    .value = [&](int batch, size_t face) { return boundary_value(batch, face).component(component); }},
                {}, {});
            for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                EXPECT_DOUBLE_EQ(vector.rhs->getData(component)[row], reference.rhs->getData()[row]);
                for (size_t column = 0; column < mesh->num_local_cells(); ++column)
                    EXPECT_DOUBLE_EQ(local_matrix_entry(*vector.matrix, cell, static_cast<LO>(column)),
                        local_matrix_entry(*reference.matrix, cell, static_cast<LO>(column)));
            }
        }
    }
    // Independent transient/source oracle: the shared matrix must not be
    // accumulated once for each RHS component.
    flux.data().putScalar(0.0);
    const auto transient = TransportFactory<decltype(velocity)>::assemble(
        typename TransportFactory<decltype(velocity)>::VectorTerms{
            .history = {.old_values = velocity, .time_step = 0.25},
            .face_fluxes = flux,
            .coefficients = {.diffusivity = 0.0},
            .source = source_value},
        typename TransportFactory<decltype(velocity)>::BasicVectorBoundary{
            .condition = [](int, size_t)
            { return SimpleFluid::VectorBoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = [](int, size_t) { return Vec{}; }},
        {}, {});
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto volume = mesh->cell_volume(cell);
        EXPECT_DOUBLE_EQ(local_matrix_entry(*transient.matrix, cell, cell), volume / 0.25);
        for (size_t component = 0; component < 3; ++component)
            EXPECT_DOUBLE_EQ(
                transient.rhs->getData(component)[row], volume / 0.25 * velocity.value(cell).component(component) +
                                                            volume * source_value(cell).component(component));
    }
}

} // namespace
