/**
 * @file testTurbulenceScalarTransportEquation.cc
 * @brief Tests for positive semi-implicit turbulence scalar transport.
 */

#include <gtest/gtest.h>

#include "equations/turbulence/TurbulenceScalarTransportEquation.hh"
#include "fields/FaceField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using Equation = SimpleFluid::TurbulenceScalarTransportEquation<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
}

SimpleFluid::SP<MeshType> make_two_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
}

SimpleFluid::SP<MeshType> make_2x2x2_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
}

Equation::scalar_provider_type zero_provider()
{
    return [](MeshType::local_ordinal_type) -> Pack::scalar_type { return 0.0; };
}

TEST(TurbulenceScalarTransportEquationTest, AdvancesAcceptedStateWithExplicitSourceAndImplicitSink)
{
    auto mesh = make_single_cell_mesh();
    FieldType state(mesh, 2.0, "state");
    FieldType diffusivity(mesh, 0.0, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    Equation equation(mesh);

    const Equation::scalar_provider_type source = [](MeshType::local_ordinal_type) { return 3.0; };
    const Equation::scalar_provider_type sink = [](MeshType::local_ordinal_type) { return 4.0; };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;

    const auto statistics =
        equation.advance(state, zero_flux, 0.5, diffusivity, state, source, sink, 1.0e-12,
                         SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, linear_options);

    EXPECT_TRUE(statistics.converged);
    // Backward Euler: (2 / dt + source) / (1 / dt + sink).
    EXPECT_NEAR(state.value(0), 7.0 / 6.0, 1.0e-12);
}

TEST(TurbulenceScalarTransportEquationTest,
     UsesVariableDiffusivityAndConservesWithZeroFluxBoundaries)
{
    auto mesh = make_two_cell_mesh();
    FieldType state(mesh, "state");
    state.set_owned_value(0, 1.0);
    state.set_owned_value(1, 3.0);
    state.sync_ghosts();
    FieldType diffusivity(mesh, "diffusivity");
    diffusivity.set_owned_value(0, 0.1);
    diffusivity.set_owned_value(1, 0.3);
    diffusivity.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    Equation equation(mesh);

    const auto statistics = equation.advance(state, zero_flux, 0.25, diffusivity, state,
                                             zero_provider(), zero_provider(), 1.0e-12);

    ASSERT_TRUE(statistics.converged);
    const auto harmonic_diffusivity = 0.15;
    const auto transient = 4.0;
    const auto expected_difference = 2.0 * transient / (transient + 2.0 * harmonic_diffusivity);
    EXPECT_NEAR(state.value(0), 2.0 - 0.5 * expected_difference, 1.0e-10);
    EXPECT_NEAR(state.value(1), 2.0 + 0.5 * expected_difference, 1.0e-10);
    EXPECT_NEAR(state.value(0) + state.value(1), 4.0, 1.0e-12);
}

TEST(TurbulenceScalarTransportEquationTest, AppliesPositiveFloorAfterStrongImplicitDestruction)
{
    auto mesh = make_single_cell_mesh();
    FieldType state(mesh, 1.0e-4, "state");
    FieldType diffusivity(mesh, 0.0, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    Equation equation(mesh);
    const Equation::scalar_provider_type sink = [](MeshType::local_ordinal_type) { return 1.0e9; };

    const auto statistics =
        equation.advance(state, zero_flux, 1.0, diffusivity, state, zero_provider(), sink, 1.0e-6);

    EXPECT_TRUE(statistics.converged);
    EXPECT_DOUBLE_EQ(state.value(0), 1.0e-6);
}

TEST(TurbulenceScalarTransportEquationTest, RejectsInvalidInputsWithoutPublishingOverAcceptedState)
{
    auto mesh = make_single_cell_mesh();
    FieldType old_state(mesh, 2.0, "old_state");
    FieldType accepted_state(mesh, 7.0, "accepted_state");
    FieldType diffusivity(mesh, 0.1, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    Equation equation(mesh);
    const Equation::scalar_provider_type negative_source = [](MeshType::local_ordinal_type)
    { return -1.0; };
    const Equation::scalar_provider_type negative_sink = [](MeshType::local_ordinal_type)
    { return -1.0; };
    const Equation::scalar_provider_type infinite_source = [](MeshType::local_ordinal_type)
    { return std::numeric_limits<double>::infinity(); };

    EXPECT_THROW(equation.advance(old_state, zero_flux, 0.1, diffusivity, accepted_state,
                                  negative_source, zero_provider(), 1.0e-6),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(accepted_state.value(0), 7.0);

    EXPECT_THROW(equation.advance(old_state, zero_flux, 0.1, diffusivity, accepted_state,
                                  zero_provider(), negative_sink, 1.0e-6),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(accepted_state.value(0), 7.0);

    EXPECT_THROW(equation.advance(old_state, zero_flux, 0.1, diffusivity, accepted_state,
                                  infinite_source, zero_provider(), 1.0e-6),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(accepted_state.value(0), 7.0);

    diffusivity.put_scalar(-0.1);
    EXPECT_THROW(equation.advance(old_state, zero_flux, 0.1, diffusivity, accepted_state,
                                  zero_provider(), zero_provider(), 1.0e-6),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(accepted_state.value(0), 7.0);

    diffusivity.put_scalar(0.1);
    EXPECT_THROW(equation.advance(old_state, zero_flux, 0.1, diffusivity, accepted_state,
                                  zero_provider(), zero_provider(), 0.0),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(accepted_state.value(0), 7.0);
}

TEST(TurbulenceScalarTransportEquationTest, LinearSolveRejectionPreservesAliasedAcceptedState)
{
    auto mesh = make_2x2x2_mesh();
    FieldType state(mesh, "state");
    std::vector<Pack::scalar_type> accepted(mesh->num_local_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<MeshType::local_ordinal_type>(owned);
        const auto value = 1.0 + static_cast<double>(owned * owned);
        state.set_owned_value(cell_lid, value);
    }
    state.sync_ghosts();
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        accepted[local] = state.local_value(static_cast<MeshType::local_ordinal_type>(local));
    }

    FieldType diffusivity(mesh, 1.0, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    Equation equation(mesh);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1;
    linear_options.tolerance = 1.0e-14;

    try
    {
        equation.advance(state, zero_flux, 1.0, diffusivity, state, zero_provider(),
                         zero_provider(), 1.0e-12,
                         SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, linear_options);
        FAIL() << "Expected the under-iterated turbulence solve to fail.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("did not converge"), std::string::npos);
    }

    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        EXPECT_DOUBLE_EQ(state.local_value(static_cast<MeshType::local_ordinal_type>(local)),
                         accepted[local]);
    }
}

TEST(TurbulenceScalarTransportEquationTest, RebuildsCachedGraphWhenNonOrthogonalStencilExpands)
{
    auto mesh = make_2x2x2_mesh();
    FieldType state(mesh, "state");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        state.set_owned_value(static_cast<MeshType::local_ordinal_type>(owned),
                              1.0 + static_cast<double>(owned));
    }
    state.sync_ghosts();
    FieldType diffusivity(mesh, 0.2, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    Equation equation(mesh);

    const auto explicit_statistics = equation.advance(
        state, zero_flux, 0.1, diffusivity, state, zero_provider(), zero_provider(), 1.0e-12,
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    ASSERT_TRUE(explicit_statistics.converged);

    const auto implicit_statistics = equation.advance(
        state, zero_flux, 0.1, diffusivity, state, zero_provider(), zero_provider(), 1.0e-12,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);
    EXPECT_TRUE(implicit_statistics.converged);
}

TEST(TurbulenceScalarTransportEquationTest, HonorsConfiguredDirichletBoundaryValue)
{
    auto mesh = make_single_cell_mesh();
    FieldType state(mesh, 1.0, "state");
    FieldType diffusivity(mesh, 1.0, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    SimpleFluid::BoundaryConditionMap boundaries;
    boundaries["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 4.0};
    Equation equation(mesh, boundaries);

    const auto statistics = equation.advance(state, zero_flux, 1.0, diffusivity, state,
                                             zero_provider(), zero_provider(), 1.0e-12);

    EXPECT_TRUE(statistics.converged);
    // Unit transient plus a boundary diffusion coefficient of two.
    EXPECT_NEAR(state.value(0), 3.0, 1.0e-10);
}

TEST(TurbulenceScalarTransportEquationTest, ExtrapolatesNeumannDataForAdvectiveBoundaryInflow)
{
    auto mesh = make_single_cell_mesh();
    FieldType state(mesh, 1.0, "state");
    FieldType diffusivity(mesh, 0.0, "diffusivity");
    SimpleFluid::FaceField<Pack> face_flux(mesh, 0.0, "face_flux");
    MeshType::local_ordinal_type xmin_face = -1;
    MeshType::local_ordinal_type xmax_face = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin")
            xmin_face = batch.face_lids.front();
        if (name == "xmax")
            xmax_face = batch.face_lids.front();
    }
    ASSERT_GE(xmin_face, 0);
    ASSERT_GE(xmax_face, 0);
    face_flux.set_value(xmin_face, -1.0);
    face_flux.set_value(xmax_face, 1.0);

    SimpleFluid::BoundaryConditionMap boundaries;
    boundaries["xmin"] = {SimpleFluid::BoundaryConditionType::Neumann, 2.0};
    Equation equation(mesh, boundaries);

    const auto statistics = equation.advance(state, face_flux, 1.0, diffusivity, state,
                                             zero_provider(), zero_provider(), 1.0e-12);

    ASSERT_TRUE(statistics.converged);
    // The unit cell has a 0.5 normal distance, so phi_face = 1 + 2*0.5.
    EXPECT_NEAR(state.value(0), 1.5, 1.0e-10);
}

} // namespace
