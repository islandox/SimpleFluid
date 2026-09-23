/** @file testPolyhedralDiffusion.cc @brief Conservative assembly beyond hexahedral row sizes. */
#include "FVM/DiffusionSystem.hh"
#include "FVM/MatrixOperators.hh"
#include "geometry/MeshHandle.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Unstructured = Meshes::UnstructuredMesh;
using Vec = MeshUtils::Vec3;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

/** A central octagonal prism has eight different neighboring ring cells. */
SP<Handle> octagonal_prism_mesh()
{
    Arr<Vec> nodes;
    for (int z = 0; z < 2; ++z)
        for (int radius = 1; radius <= 2; ++radius)
            for (int i = 0; i < 8; ++i)
            {
                const auto angle = 2 * std::numbers::pi_v<real_t> * i / 8;
                nodes.emplace_back(radius * std::cos(angle), radius * std::sin(angle), z);
            }

    Unstructured::CellDefinition center;
    center.type = MeshUtils::CellType::POLYHEDRON;
    center.face_node_ids = {{7, 6, 5, 4, 3, 2, 1, 0}, {16, 17, 18, 19, 20, 21, 22, 23}};
    for (Unstructured::NodeID i = 0; i < 8; ++i)
    {
        const auto next = (i + 1) % 8;
        center.face_node_ids.push_back({i, next, next + 16, i + 16});
    }
    Arr<Unstructured::CellDefinition> cells{std::move(center)};
    for (Unstructured::NodeID i = 0; i < 8; ++i)
    {
        const auto next = (i + 1) % 8;
        cells.push_back({MeshUtils::CellType::HEXAHEDRON,
            {i, i + 8, next + 8, next, i + 16, i + 24, next + 24, next + 16}});
    }

    const Unstructured untagged(nodes, cells);
    Arr<Unstructured::BoundaryFaceDefinition> boundaries;
    for (size_t f = 0; f < untagged.num_faces(); ++f)
        if (untagged.is_boundary_face(f))
            boundaries.push_back({untagged.face_nodes(f), 0, "wall"});
    return std::make_shared<Handle>(std::make_shared<Unstructured>(nodes, cells, boundaries));
}

void expect_nine_column_row(const Pack::matrix_type& matrix)
{
    Pack::matrix_type::local_inds_host_view_type columns;
    Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(0, columns, values);
    ASSERT_EQ(columns.extent(0), 9U);
    std::vector<bool> found(9, false);
    for (size_t i = 0; i < columns.extent(0); ++i)
    {
        ASSERT_GE(columns(i), 0);
        ASSERT_LT(columns(i), 9);
        EXPECT_FALSE(found[columns(i)]);
        found[columns(i)] = true;
    }
}
} // namespace

TEST(PolyhedralDiffusionTest, NineColumnOperatorsPreserveConservationAndConstantSolutions)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Serial native-polyhedron stencil regression.";
    const auto mesh = octagonal_prism_mesh();
    ASSERT_EQ(mesh->num_owned_cells(), 9U);
    ASSERT_EQ(mesh->faces(0).size(), 10U);

    const auto diffusion = FVM::diffusion_matrix<Pack>(*mesh, 1.0);
    expect_nine_column_row(*diffusion);
    Pack::vector_type constant(mesh->owned_cell_map(), true), action(mesh->owned_cell_map(), true);
    constant.putScalar(1.0);
    diffusion->apply(constant, action);
    EXPECT_LT(action.norm2(), 1e-12);

    const auto neumann = [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0}; };
    const auto pressure = FVM::pressure_poisson_matrix<Pack>(*mesh,
        std::optional<Pack::global_ordinal_type>{1}, neumann);
    expect_nine_column_row(*pressure);

    ScalarFaceFieldStored<Pack> flux(ScalarFaceFieldDescriptor<Pack>("flux"), mesh, 0.0);
    for (const auto face : mesh->faces(0))
        if (mesh->is_interior_face(face))
            flux.set_value(face, mesh->owner_cell(face) == 0 ? -1.0 : 1.0);
    const auto convection = FVM::upwind_convection_matrix<Pack>(*mesh, flux);
    expect_nine_column_row(*convection);
    convection->apply(constant, action);
    EXPECT_DOUBLE_EQ(action.getData()[0], -8.0);
    real_t net_flux = 0;
    for (const auto value : action.getData()) net_flux += value;
    EXPECT_NEAR(net_flux, 0.0, 1e-13);

    const auto scalar = FVM::diffusion_system<Pack>(*mesh, 1.7,
        [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 3.0}; });
    expect_nine_column_row(*scalar.matrix);
    constant.putScalar(3.0);
    scalar.matrix->apply(constant, action);
    action.update(-1.0, *scalar.rhs, 1.0);
    EXPECT_LT(action.norm2(), 1e-11);

    const Vec value{3.0, -2.0, 0.5};
    const auto vector = FVM::vector_diffusion_system<Pack>(*mesh, 1.7,
        [&](int, size_t) { return VectorBoundaryCondition{BoundaryConditionType::Dirichlet, value}; });
    expect_nine_column_row(*vector.matrix);
    Pack::multi_vector_type vector_constant(mesh->owned_cell_map(), 3, true),
        vector_action(mesh->owned_cell_map(), 3, true);
    for (size_t component = 0; component < 3; ++component)
        vector_constant.getVectorNonConst(component)->putScalar(value.component(component));
    vector.matrix->apply(vector_constant, vector_action);
    vector_action.update(-1.0, *vector.rhs, 1.0);
    for (size_t component = 0; component < 3; ++component)
        EXPECT_LT(vector_action.getVector(component)->norm2(), 1e-11);
}
