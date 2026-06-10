/**
 * @file testFrameworkEquations.cc
 * @brief Tests for generic equations, coupled blocks, and Problem assembly.
 */

#include <gtest/gtest.h>

#include "FVM/DiffusionSystem.hh"
#include "equations/CoupledEquation.hh"
#include "equations/Equation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "problems/Problem.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using ScalarStored = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorStored = SimpleFluid::VectorCellFieldStored<Pack>;
using ScalarEquation = SimpleFluid::Equation<ScalarStored, Pack>;
using VectorEquation = SimpleFluid::Equation<VectorStored, Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class RecordingCoupledBackend final
    : public SimpleFluid::CoupledSolverBackend<Pack>
{
public:
    explicit RecordingCoupledBackend(bool& called)
        : d_called(called)
    {
    }

    bool solve(
        SimpleFluid::AssembledCoupledEquation<Pack>& equation) override
    {
        d_called = true;
        bool converged = true;
        for (auto& block : equation.diagonal_blocks())
        {
            converged = block->solve() && converged;
        }
        return converged;
    }

private:
    bool& d_called;
};

SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>
make_cartesian_handle()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 0.5, 1.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(mesh);
}

std::vector<std::pair<int, double>> row_values(
    const Pack::matrix_type& matrix,
    int row)
{
    Pack::matrix_type::local_inds_host_view_type columns;
    Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(row, columns, values);
    std::vector<std::pair<int, double>> result;
    for (size_t entry = 0; entry < columns.extent(0); ++entry)
    {
        result.emplace_back(columns[entry], values[entry]);
    }
    std::ranges::sort(result);
    return result;
}

} // namespace

TEST(FrameworkEquationTest, SolvesScalarDiffusionOnCartesianHandle)
{
    auto mesh = make_cartesian_handle();
    auto temperature = std::make_shared<ScalarStored>(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"),
        mesh);
    ScalarEquation equation(temperature);
    equation.add_lhs(
        SimpleFluid::FVM::DiffusionOperator<Pack>{1.0});
    equation.add_rhs(
        SimpleFluid::FVM::SourceOperator<Pack>{
            [](int, size_t) { return 0.0; }});
    equation.set_boundary_providers(
        [](int patch_id)
        {
            return patch_id < 2
                 ? SimpleFluid::BoundaryConditionType::Dirichlet
                 : SimpleFluid::BoundaryConditionType::Neumann;
        },
        [](int patch_id, size_t, size_t)
        {
            return patch_id == 1 ? 1.0 : 0.0;
        });

    auto assembled = equation.assemble();
    ASSERT_TRUE(assembled.solve());
    EXPECT_NEAR(temperature->value(0), 0.25, 1.0e-12);
    EXPECT_NEAR(temperature->value(1), 0.75, 1.0e-12);
}

TEST(FrameworkEquationTest, MatchesExistingDiffusionAssembly)
{
    auto legacy = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    auto handle =
        std::make_shared<SimpleFluid::MeshHandle<Pack>>(legacy);
    auto field = std::make_shared<ScalarStored>(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("phi"),
        handle);

    ScalarEquation equation(field);
    equation.add_lhs(
        SimpleFluid::FVM::DiffusionOperator<Pack>{2.0});
    auto generic = equation.assemble();

    auto existing = SimpleFluid::FVM::diffusion_system<Pack>(
        *legacy,
        2.0,
        [](int, size_t)
        {
            return SimpleFluid::BoundaryCondition{};
        });

    ASSERT_EQ(generic.matrix()->getLocalNumRows(),
              existing.matrix->getLocalNumRows());
    for (int row = 0;
         row < static_cast<int>(generic.matrix()->getLocalNumRows());
         ++row)
    {
        EXPECT_EQ(row_values(*generic.matrix(), row),
                  row_values(*existing.matrix, row));
    }
    EXPECT_EQ(generic.rhs().getData()[0],
              existing.rhs->getData()[0]);
}

TEST(FrameworkEquationTest, SolvesVectorTransientEquation)
{
    auto mesh = make_cartesian_handle();
    auto velocity = std::make_shared<VectorStored>(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("velocity"),
        mesh);
    VectorEquation equation(velocity);
    equation.add_lhs(
        SimpleFluid::FVM::TransientOperator<Pack>{
            0.5,
            [](int, size_t component)
            {
                return static_cast<double>(component + 1);
            }});

    auto assembled = equation.assemble();
    ASSERT_TRUE(assembled.solve());
    const auto result = velocity->value(0);
    EXPECT_NEAR(result.x, 1.0, 1.0e-12);
    EXPECT_NEAR(result.y, 2.0, 1.0e-12);
    EXPECT_NEAR(result.z, 3.0, 1.0e-12);
}

TEST(FrameworkEquationTest, ProblemAssemblesAndSolvesNamedEquation)
{
    auto mesh = make_cartesian_handle();
    SimpleFluid::Problem<Pack> problem(mesh);
    auto& field = problem.add_field(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("temperature"));
    auto field_ptr = std::shared_ptr<ScalarStored>(
        problem.mesh_ptr(),
        &field);
    ScalarEquation equation(field_ptr);
    equation.add_lhs(
        SimpleFluid::FVM::TransientOperator<Pack>{
            1.0,
            [](int, size_t) { return 4.0; }});
    problem.add_equation("temperature_equation", equation);

    using Assembled =
        SimpleFluid::AssembledEquation<ScalarStored, Pack>;
    problem.assemble<ScalarEquation>(
        "temperature_equation", "temperature_system");
    EXPECT_TRUE(problem.solve<Assembled>("temperature_system"));
    EXPECT_DOUBLE_EQ(field.value(0), 4.0);
}

TEST(FrameworkEquationTest, ValidatesCoupledBlockStructure)
{
    auto mesh = make_cartesian_handle();
    auto scalar = std::make_shared<ScalarStored>(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("scalar"),
        mesh);
    auto vector = std::make_shared<VectorStored>(
        SimpleFluid::VectorCellFieldDescriptor<Pack>("vector"),
        mesh);

    ScalarEquation scalar_equation(scalar);
    scalar_equation.add_lhs(
        SimpleFluid::FVM::TransientOperator<Pack>{
            1.0, [](int, size_t) { return 1.0; }});
    VectorEquation vector_equation(vector);
    vector_equation.add_lhs(
        SimpleFluid::FVM::TransientOperator<Pack>{
            1.0, [](int, size_t) { return 2.0; }});

    SimpleFluid::CoupledEquation<Pack> coupled;
    coupled.add_diagonal(
        [scalar_equation]
        {
            return SimpleFluid::make_assembled_equation_model<
                decltype(scalar_equation.assemble()), Pack>(
                    scalar_equation.assemble());
        });
    coupled.add_diagonal(
        [vector_equation]
        {
            return SimpleFluid::make_assembled_equation_model<
                decltype(vector_equation.assemble()), Pack>(
                    vector_equation.assemble());
        });

    auto assembled = coupled.assemble();
    EXPECT_TRUE(assembled.solve());
    EXPECT_DOUBLE_EQ(scalar->value(0), 1.0);
    const auto result = vector->value(0);
    EXPECT_NEAR(result.x, 2.0, 1.0e-12);
    EXPECT_NEAR(result.y, 2.0, 1.0e-12);
    EXPECT_NEAR(result.z, 2.0, 1.0e-12);
}

TEST(FrameworkEquationTest, DispatchesCustomCoupledBackend)
{
    auto mesh = make_cartesian_handle();
    auto scalar = std::make_shared<ScalarStored>(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("scalar"),
        mesh);
    ScalarEquation scalar_equation(scalar);
    scalar_equation.add_lhs(
        SimpleFluid::FVM::TransientOperator<Pack>{
            1.0, [](int, size_t) { return 3.0; }});

    auto off_diagonal = Teuchos::rcp(new Pack::matrix_type(
        mesh->owned_cell_map(), mesh->overlap_cell_map(), 1));
    for (int row = 0;
         row < static_cast<int>(mesh->num_owned_cells());
         ++row)
    {
        Teuchos::Array<int> columns{row};
        Teuchos::Array<double> values{0.0};
        off_diagonal->insertLocalValues(
            row, columns(), values());
    }
    off_diagonal->fillComplete();

    SimpleFluid::CoupledEquation<Pack> coupled;
    coupled.add_diagonal(
        [scalar_equation]
        {
            return SimpleFluid::make_assembled_equation_model<
                decltype(scalar_equation.assemble()), Pack>(
                    scalar_equation.assemble());
        });
    EXPECT_THROW(
        coupled.add_off_diagonal({1, 0, off_diagonal}).assemble(),
        std::out_of_range);

    SimpleFluid::CoupledEquation<Pack> two_block;
    for (int block = 0; block < 2; ++block)
    {
        two_block.add_diagonal(
            [scalar_equation]
            {
                return SimpleFluid::make_assembled_equation_model<
                    decltype(scalar_equation.assemble()), Pack>(
                        scalar_equation.assemble());
            });
    }
    two_block.add_off_diagonal({0, 1, off_diagonal});
    bool called = false;
    auto assembled = two_block.assemble(
        std::make_shared<RecordingCoupledBackend>(called));
    EXPECT_TRUE(assembled.solve());
    EXPECT_TRUE(called);
}

TEST(FrameworkEquationTest, ValidatesOperatorConfiguration)
{
    auto mesh = make_cartesian_handle();
    auto field = std::make_shared<ScalarStored>(
        SimpleFluid::ScalarCellFieldDescriptor<Pack>("scalar"),
        mesh);
    ScalarEquation equation(field);

    EXPECT_THROW(
        equation.add_lhs(
            SimpleFluid::FVM::TransientOperator<Pack>{
                0.0, [](int, size_t) { return 0.0; }}),
        std::invalid_argument);
    EXPECT_THROW(
        equation.add_lhs(
            SimpleFluid::FVM::ConvectionOperator<Pack>{}),
        std::invalid_argument);
    EXPECT_THROW(
        equation.add_lhs(
            SimpleFluid::FVM::DiffusionOperator<Pack>{-1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        equation.add_rhs(
            SimpleFluid::FVM::SourceOperator<Pack>{}),
        std::invalid_argument);
    EXPECT_THROW(
        equation.add_lhs(SimpleFluid::FVM::GradientOperator{}),
        std::invalid_argument);
    EXPECT_THROW(
        equation.set_boundary_providers(
            {}, [](int, size_t, size_t) { return 0.0; }),
        std::invalid_argument);
}

TEST(FrameworkEquationTest, RejectsTermsOnWrongSideAndRobinBoundaries)
{
    auto mesh = make_cartesian_handle();
    auto make_field = [&]
    {
        return std::make_shared<ScalarStored>(
            SimpleFluid::ScalarCellFieldDescriptor<Pack>("scalar"),
            mesh);
    };

    ScalarEquation source_on_lhs(make_field());
    source_on_lhs.add_lhs(
        SimpleFluid::FVM::SourceOperator<Pack>{
            [](int, size_t) { return 1.0; }});
    EXPECT_THROW(source_on_lhs.assemble(), std::invalid_argument);

    ScalarEquation transient_on_rhs(make_field());
    transient_on_rhs.add_rhs(
        SimpleFluid::FVM::TransientOperator<Pack>{
            1.0, [](int, size_t) { return 1.0; }});
    EXPECT_THROW(transient_on_rhs.assemble(), std::invalid_argument);

    ScalarEquation coupling(make_field());
    coupling.add_lhs(
        SimpleFluid::FVM::GradientOperator{"pressure"});
    EXPECT_THROW(coupling.assemble(), std::logic_error);

    ScalarEquation robin(make_field());
    robin.add_lhs(
        SimpleFluid::FVM::DiffusionOperator<Pack>{1.0});
    robin.set_boundary_providers(
        [](int) { return SimpleFluid::BoundaryConditionType::Robin; },
        [](int, size_t, size_t) { return 0.0; });
    EXPECT_THROW(robin.assemble(), std::runtime_error);
}
