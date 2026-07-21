/**
 * @file testFvmOperators.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief unit tests for finite-volume helper operators
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "fields/CellField.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/Operators.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
using TensorFieldType = SimpleFluid::TensorCellField<Pack>;
using FaceFluxWorkspace =
    SimpleFluid::FVM::PressureWeightedFaceFluxWorkspace<Pack>;

static_assert(!std::is_copy_constructible_v<FaceFluxWorkspace>);
static_assert(!std::is_copy_assignable_v<FaceFluxWorkspace>);
static_assert(std::is_move_constructible_v<FaceFluxWorkspace>);
static_assert(std::is_move_assignable_v<FaceFluxWorkspace>);

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

std::vector<Pack::scalar_type> local_values(const FieldType& field)
{
    std::vector<Pack::scalar_type> values(field.num_local_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(field.num_local_cells());
         ++lid)
    {
        values[static_cast<size_t>(lid)] = field.local_value(lid);
    }
    return values;
}

MeshType::local_ordinal_type boundary_face_lid(const MeshType& mesh,
                                               const char* boundary_name)
{
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        if (mesh.boundary_batch_name(batch_id) == boundary_name
            && !batch.face_lids.empty())
        {
            return batch.face_lids.front();
        }
    }

    throw std::runtime_error("Requested boundary face was not found.");
}

Pack::scalar_type local_matrix_entry(
    const Pack::matrix_type& matrix,
    MeshType::local_ordinal_type row,
    MeshType::local_ordinal_type column)
{
    const auto row_entries = matrix.getNumEntriesInLocalRow(row);
    typename Pack::matrix_type::nonconst_local_inds_host_view_type columns(
        "columns", row_entries);
    typename Pack::matrix_type::nonconst_values_host_view_type values(
        "values", row_entries);
    size_t num_entries = 0;
    matrix.getLocalRowCopy(row, columns, values, num_entries);

    Pack::scalar_type entry = 0.0;
    for (size_t i = 0; i < num_entries; ++i)
    {
        if (columns(i) == column)
        {
            entry += values(i);
        }
    }

    return entry;
}

std::vector<Pack::scalar_type> local_matrix_action(
    const Pack::matrix_type& matrix,
    const FieldType& field)
{
    std::vector<Pack::scalar_type> result(field.num_owned_cells(), 0.0);
    for (MeshType::local_ordinal_type row = 0;
         row < static_cast<MeshType::local_ordinal_type>(field.num_owned_cells());
         ++row)
    {
        const auto row_entries = matrix.getNumEntriesInLocalRow(row);
        typename Pack::matrix_type::nonconst_local_inds_host_view_type columns(
            "columns", row_entries);
        typename Pack::matrix_type::nonconst_values_host_view_type values(
            "values", row_entries);
        size_t num_entries = 0;
        matrix.getLocalRowCopy(row, columns, values, num_entries);

        auto value = Pack::scalar_type{};
        for (size_t i = 0; i < num_entries; ++i)
        {
            value += values(i) * field.local_value(columns(i));
        }
        result[static_cast<size_t>(row)] = value;
    }
    return result;
}

double nonlinear_scalar(const SimpleFluid::vec3<>& point)
{
    return 0.25 + 0.5 * point.x - 0.2 * point.y + 0.125 * point.z
         + 0.3 * point.x * point.y - 0.15 * point.z * point.z;
}

} // namespace

/**
 * @brief Sets a linear scalar field and verifies the cell_gradient operator recovers the exact gradient.
 */
TEST(FvmOperatorsTest, RecoversLinearCellGradientOnStructuredBox)
{
    auto mesh = make_mesh();
    FieldType phi(mesh, "phi");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& c = mesh->cell_centroid(lid);
        phi.set_value(lid, 2.0 * c.x + 3.0 * c.y + 4.0 * c.z);
    }
    phi.sync_ghosts();

    VectorFieldType gradients(mesh, "gradient");
    SimpleFluid::FVM::cell_gradient(phi, gradients);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto gradient = gradients.value(
            static_cast<MeshType::local_ordinal_type>(owned));
        EXPECT_NEAR(gradient.x, 2.0, 1.0e-12);
        EXPECT_NEAR(gradient.y, 3.0, 1.0e-12);
        EXPECT_NEAR(gradient.z, 4.0, 1.0e-12);
    }
}

/** @brief Verifies in-plane gradients are recovered on a one-cell-thick box. */
TEST(FvmOperatorsTest, RecoversInPlaneGradientOnOneCellThickBox)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(3, 3, 1, 0.25));
    FieldType phi(mesh, "planar_phi");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
                   mesh->num_owned_cells());
         ++lid)
    {
        const auto& center = mesh->cell_centroid(lid);
        phi.set_value(lid, 1.0 + 2.0 * center.x - 3.0 * center.y);
    }
    phi.sync_ghosts();

    VectorFieldType gradients(mesh, "gradient");
    SimpleFluid::FVM::cell_gradient(phi, gradients);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto gradient = gradients.value(
            static_cast<MeshType::local_ordinal_type>(owned));
        EXPECT_NEAR(gradient.x, 2.0, 1.0e-12);
        EXPECT_NEAR(gradient.y, -3.0, 1.0e-12);
        EXPECT_NEAR(gradient.z, 0.0, 1.0e-12);
    }
}

/** @brief Verifies the normal-system solver handles a rank-deficient oblique plane. */
TEST(FvmOperatorsTest, SolvesRankDeficientNormalSystemInObliquePlane)
{
    const MeshType::Vec3 first_direction{1.0, 0.0, 1.0};
    const MeshType::Vec3 second_direction{0.0, 1.0, 1.0};
    const MeshType::Vec3 expected =
        first_direction * 2.0 - second_direction * 3.0;

    std::array<std::array<SimpleFluid::real_t, 3>, 3> normal{};
    MeshType::Vec3 rhs{};
    for (const auto& direction :
         std::array<MeshType::Vec3, 2>{
             first_direction, second_direction})
    {
        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t column = 0; column < 3; ++column)
            {
                normal[row][column] +=
                    direction.component(row)
                  * direction.component(column);
            }
        }
        rhs = rhs + direction * direction.dot(expected);
    }

    const auto actual =
        SimpleFluid::FVM::detail::solve_3x3(normal, rhs);

    EXPECT_NEAR(actual.x, expected.x, 1.0e-12);
    EXPECT_NEAR(actual.y, expected.y, 1.0e-12);
    EXPECT_NEAR(actual.z, expected.z, 1.0e-12);
}

/**
 * @brief Verifies the vector cell gradient operator recovers the exact gradient of a linear velocity field.
 */
TEST(FvmOperatorsTest, RecoversLinearVectorCellGradientOnStructuredBox)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& c = mesh->cell_centroid(lid);
        velocity.set_value(lid,
                           {1.0 + 2.0 * c.x - 3.0 * c.y + 4.0 * c.z,
                            -2.0 + c.x + 0.5 * c.y - 1.5 * c.z,
                            3.0 - 4.0 * c.x + 2.0 * c.y + c.z});
    }
    velocity.sync_ghosts();

    TensorFieldType gradients(mesh, "velocity_gradient");
    SimpleFluid::FVM::cell_gradient(velocity, gradients);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto gradient = gradients.value(
            static_cast<MeshType::local_ordinal_type>(owned));
        EXPECT_NEAR(gradient[0].x, 2.0, 1.0e-12);
        EXPECT_NEAR(gradient[0].y, -3.0, 1.0e-12);
        EXPECT_NEAR(gradient[0].z, 4.0, 1.0e-12);

        EXPECT_NEAR(gradient[1].x, 1.0, 1.0e-12);
        EXPECT_NEAR(gradient[1].y, 0.5, 1.0e-12);
        EXPECT_NEAR(gradient[1].z, -1.5, 1.0e-12);

        EXPECT_NEAR(gradient[2].x, -4.0, 1.0e-12);
        EXPECT_NEAR(gradient[2].y, 2.0, 1.0e-12);
        EXPECT_NEAR(gradient[2].z, 1.0, 1.0e-12);
    }
}

/** @brief Verifies boundary face samples recover an affine vector gradient. */
TEST(FvmOperatorsTest,
     RecoversAffineVectorGradientFromBoundaryFaceSamples)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
    auto affine_value = [](const MeshType::Vec3& point)
    {
        return MeshType::Vec3{
            1.0 + 2.0 * point.x - 3.0 * point.y + 4.0 * point.z,
            -2.0 + point.x + 0.5 * point.y - 1.5 * point.z,
            3.0 - 4.0 * point.x + 2.0 * point.y + point.z};
    };

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, affine_value(mesh->cell_centroid(0)));

    TensorFieldType cell_only_gradients(mesh, "cell_only_gradient");
    SimpleFluid::FVM::cell_gradient(velocity, cell_only_gradients);
    for (const auto& component : cell_only_gradients.value(0))
    {
        EXPECT_EQ(component, MeshType::Vec3{});
    }

    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return affine_value(mesh->face_centroid(face_lid));
    };
    TensorFieldType boundary_gradients(mesh, "boundary_gradient");
    SimpleFluid::FVM::cell_gradient(
        velocity, boundary_value, boundary_gradients);

    const auto gradient = boundary_gradients.value(0);
    EXPECT_NEAR(gradient[0].x, 2.0, 1.0e-12);
    EXPECT_NEAR(gradient[0].y, -3.0, 1.0e-12);
    EXPECT_NEAR(gradient[0].z, 4.0, 1.0e-12);
    EXPECT_NEAR(gradient[1].x, 1.0, 1.0e-12);
    EXPECT_NEAR(gradient[1].y, 0.5, 1.0e-12);
    EXPECT_NEAR(gradient[1].z, -1.5, 1.0e-12);
    EXPECT_NEAR(gradient[2].x, -4.0, 1.0e-12);
    EXPECT_NEAR(gradient[2].y, 2.0, 1.0e-12);
    EXPECT_NEAR(gradient[2].z, 1.0, 1.0e-12);
}

/** @brief Verifies face-area vectors decompose into orthogonal and tangential parts. */
TEST(FvmOperatorsTest, DecomposesFaceAreaIntoOrthogonalAndTangentialParts)
{
    const SimpleFluid::vec3<> area_vector{2.0, 3.0, -1.0};
    const SimpleFluid::vec3<> cell_center_vector{4.0, 1.0, 2.0};

    const auto orthogonal =
        SimpleFluid::FVM::detail::orthogonal_area_vector(
            area_vector, cell_center_vector);
    const auto tangential =
        SimpleFluid::FVM::detail::non_orthogonal_area_vector(
            area_vector, cell_center_vector);

    const auto reconstructed = orthogonal + tangential;
    EXPECT_NEAR(reconstructed.x, area_vector.x, 1.0e-12);
    EXPECT_NEAR(reconstructed.y, area_vector.y, 1.0e-12);
    EXPECT_NEAR(reconstructed.z, area_vector.z, 1.0e-12);
    EXPECT_NEAR(tangential.dot(cell_center_vector), 0.0, 1.0e-12);
}

/** @brief Verifies non-orthogonal treatment strings parse and format correctly. */
TEST(FvmOperatorsTest, ParsesNonOrthogonalTreatmentSwitch)
{
    EXPECT_EQ(
        SimpleFluid::FVM::non_orthogonal_treatment_from_string("explicit"),
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    EXPECT_EQ(
        SimpleFluid::FVM::non_orthogonal_treatment_from_string("implicit"),
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);
    EXPECT_EQ(
        SimpleFluid::FVM::non_orthogonal_treatment_from_string("hybrid"),
        SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid);

    EXPECT_EQ(
        SimpleFluid::FVM::to_string(
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit),
        "implicit");
    EXPECT_THROW(
        SimpleFluid::FVM::non_orthogonal_treatment_from_string("sideways"),
        std::invalid_argument);
}

/** @brief Verifies least-squares gradient stencils reproduce cell gradients. */
TEST(FvmOperatorsTest, LeastSquaresGradientStencilMatchesCellGradient)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType phi(mesh, "phi");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        phi.set_value(lid, nonlinear_scalar(mesh->cell_centroid(lid)));
    }
    phi.sync_ghosts();

    VectorFieldType gradients(mesh, "gradient");
    SimpleFluid::FVM::cell_gradient(phi, gradients);
    const auto stencils =
        SimpleFluid::FVM::detail::least_squares_gradient_stencils(*mesh);

    ASSERT_EQ(stencils.size(), gradients.num_owned_cells());
    for (size_t row = 0; row < stencils.size(); ++row)
    {
        MeshType::Vec3 reconstructed{};
        for (const auto& entry : stencils[row])
        {
            const auto value = phi.local_value(entry.cell_lid);
            reconstructed = reconstructed + entry.coefficient * value;
        }

        const auto gradient = gradients.value(
            static_cast<MeshType::local_ordinal_type>(row));
        EXPECT_NEAR(reconstructed.x, gradient.x, 1.0e-12);
        EXPECT_NEAR(reconstructed.y, gradient.y, 1.0e-12);
        EXPECT_NEAR(reconstructed.z, gradient.z, 1.0e-12);
    }
}

/** @brief Verifies skew-boundary Neumann gradients use the normal distance. */
TEST(FvmOperatorsTest,
     PressureGradientUsesNormalDistanceForSkewBoundaryNeumannData)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType pressure(mesh, 0.0, "pressure");
    constexpr Pack::scalar_type prescribed_normal_gradient = 2.5;

    SimpleFluid::BoundaryConditionMap pressure_boundaries;
    pressure_boundaries["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann,
        prescribed_normal_gradient};

    VectorFieldType gradient(mesh, "pressure_gradient");
    SimpleFluid::FVM::cell_gradient(
        pressure, pressure_boundaries, gradient);

    const auto target_face = boundary_face_lid(*mesh, "xmin");
    const auto target_cell = mesh->owner_cell(target_face);
    const auto boundary_locations =
        SimpleFluid::FVM::detail::boundary_face_locations(*mesh);
    auto reconstruct = [&](bool use_normal_distance)
    {
        std::array<std::array<SimpleFluid::real_t, 3>, 3> normal{};
        MeshType::Vec3 rhs{};
        for (const auto face_lid : mesh->faces(target_cell))
        {
            const auto direction =
                mesh->is_interior_face(face_lid)
              ? mesh->cell_center_vector(face_lid, target_cell)
              : mesh->face_centroid(face_lid)
                    - mesh->cell_centroid(target_cell);

            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;

            Pack::scalar_type delta = 0.0;
            if (!mesh->is_interior_face(face_lid))
            {
                const auto location =
                    boundary_locations[static_cast<size_t>(face_lid)];
                if (location.active
                    && mesh->boundary_batch_name(location.batch_id)
                       == "xmin")
                {
                    const auto distance =
                        use_normal_distance
                      ? direction.dot(mesh->face_normal_outward(
                            face_lid, target_cell))
                      : mesh->cell_to_face_distance(
                            face_lid, target_cell);
                    delta = prescribed_normal_gradient * distance;
                }
            }
            rhs = rhs + direction * delta;
        }
        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        return SimpleFluid::FVM::detail::solve_3x3(normal, rhs);
    };

    const auto expected = reconstruct(true);
    const auto euclidean_result = reconstruct(false);
    ASSERT_GT((expected - euclidean_result).norm(), 1.0e-6);

    const auto actual = gradient.value(target_cell);
    EXPECT_NEAR(actual.x, expected.x, 1.0e-12);
    EXPECT_NEAR(actual.y, expected.y, 1.0e-12);
    EXPECT_NEAR(actual.z, expected.z, 1.0e-12);
}

/** @brief Verifies an empty pressure boundary map implies homogeneous Neumann data. */
TEST(FvmOperatorsTest,
     EmptyPressureBoundaryMapDefaultsToHomogeneousNeumann)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType pressure(mesh, "pressure");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
                   mesh->num_owned_cells());
         ++lid)
    {
        pressure.set_value(
            lid, nonlinear_scalar(mesh->cell_centroid(lid)));
    }
    pressure.sync_ghosts();

    SimpleFluid::BoundaryConditionMap empty_boundaries;
    SimpleFluid::BoundaryConditionMap partial_boundaries;
    partial_boundaries["xmin"] = {
        SimpleFluid::BoundaryConditionType::Neumann, 0.0};

    VectorFieldType empty_gradient(mesh, "empty_map_gradient");
    VectorFieldType partial_gradient(mesh, "partial_map_gradient");
    VectorFieldType cell_only_gradient(mesh, "cell_only_gradient");
    SimpleFluid::FVM::cell_gradient(
        pressure, empty_boundaries, empty_gradient);
    SimpleFluid::FVM::cell_gradient(
        pressure, partial_boundaries, partial_gradient);
    SimpleFluid::FVM::cell_gradient(pressure, cell_only_gradient);

    Pack::scalar_type maximum_map_difference = 0.0;
    Pack::scalar_type maximum_boundary_effect = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        maximum_map_difference = std::max(
            maximum_map_difference,
            (empty_gradient.value(cell_lid)
             - partial_gradient.value(cell_lid)).norm());
        maximum_boundary_effect = std::max(
            maximum_boundary_effect,
            (empty_gradient.value(cell_lid)
             - cell_only_gradient.value(cell_lid)).norm());
    }

    EXPECT_NEAR(maximum_map_difference, 0.0, 1.0e-12);
    EXPECT_GT(maximum_boundary_effect, 1.0e-6);
}

/** @brief Verifies implicit non-orthogonal diffusion expands the gradient graph. */
TEST(FvmOperatorsTest, ImplicitNonOrthogonalMatrixExpandsGradientGraph)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    auto boundary_condition =
        [](int, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    };
    auto source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto orthogonal = SimpleFluid::FVM::diffusion_system<Pack>(
        *mesh, 0.5, boundary_condition, source);
    const auto implicit =
        SimpleFluid::FVM::fully_implicit_non_orthogonal_diffusion_system<Pack>(
            *mesh, 0.5, boundary_condition, source);

    bool saw_expanded_row = false;
    for (MeshType::local_ordinal_type row = 0;
         row < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++row)
    {
        if (implicit.matrix->getNumEntriesInLocalRow(row)
            > orthogonal.matrix->getNumEntriesInLocalRow(row))
        {
            saw_expanded_row = true;
            break;
        }
    }

    EXPECT_TRUE(saw_expanded_row);
}

/** @brief Verifies implicit non-orthogonal vector transport expands the gradient graph. */
TEST(FvmOperatorsTest, VectorTransportImplicitNonOrthogonalDiffusionExpandsGradientGraph)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    VectorFieldType velocity(mesh, "velocity");
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [](int, MeshType::local_ordinal_type)
    {
        return SimpleFluid::vec3<Pack::scalar_type>{};
    };
    auto source =
        [](MeshType::local_ordinal_type)
    {
        return SimpleFluid::vec3<Pack::scalar_type>{};
    };

    const auto orthogonal = SimpleFluid::FVM::transport_system<Pack>(
        velocity, zero_fluxes, 0.25, 0.5, boundary_value, source);
    const auto implicit = SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(
        velocity, zero_fluxes, 0.25, 0.5, boundary_value, source,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);

    bool saw_expanded_row = false;
    for (MeshType::local_ordinal_type row = 0;
         row < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++row)
    {
        if (implicit.matrix->getNumEntriesInLocalRow(row)
            > orthogonal.matrix->getNumEntriesInLocalRow(row))
        {
            saw_expanded_row = true;
            break;
        }
    }

    EXPECT_TRUE(saw_expanded_row);
}

/** @brief Verifies the implicit non-orthogonal matrix matches the full residual. */
TEST(FvmOperatorsTest, ImplicitNonOrthogonalMatrixMatchesFullResidual)
{
    constexpr double diffusivity = 0.7;
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType phi(mesh, "phi");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        phi.set_value(lid, nonlinear_scalar(mesh->cell_centroid(lid)));
    }
    phi.sync_ghosts();

    auto boundary_condition =
        [&](int batch_id, size_t in_batch_id)
            -> SimpleFluid::BoundaryCondition
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return {SimpleFluid::BoundaryConditionType::Dirichlet,
                nonlinear_scalar(mesh->face_centroid(face_lid))};
    };
    auto source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto system =
        SimpleFluid::FVM::fully_implicit_non_orthogonal_diffusion_system<Pack>(
            *mesh, diffusivity, boundary_condition, source);
    const auto residual =
        SimpleFluid::FVM::full_diffusion_residual<Pack>(
            phi, diffusivity, boundary_condition);
    const auto applied = local_matrix_action(*system.matrix, phi);
    const auto rhs_view = system.rhs->getData();
    const auto residual_view = residual->getData();

    for (size_t row = 0; row < applied.size(); ++row)
    {
        EXPECT_NEAR(applied[row] - rhs_view[row], residual_view[row], 1.0e-10);
    }
}

/** @brief Verifies identity and diffusion matrix construction. */
TEST(FvmOperatorsTest, BuildsIdentityAndDiffusionMatrices)
{
    auto mesh = make_mesh();

    auto identity = SimpleFluid::FVM::identity_matrix<Pack>(
        mesh->owned_cell_map());
    auto diffusion = SimpleFluid::FVM::diffusion_matrix<Pack>(*mesh, 1.0);

    EXPECT_EQ(identity->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(diffusion->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
}

/**
 * @brief Confirms face_fluxes uses all three velocity components to compute volumetric fluxes on interior faces.
 */
TEST(FvmOperatorsTest, FaceFluxesUseAllThreeVelocityComponents)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "velocity");

    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FVM::face_fluxes(velocity, fluxes);

    bool saw_x_face = false;
    bool saw_y_face = false;
    bool saw_z_face = false;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_interior_face(fid))
        {
            continue;
        }

        const auto& normal = mesh->face_normal(fid);
        const auto magnitude = fluxes.is_owned_face(fid) ? std::abs(fluxes.value(fid)) : 0.0;
        if (std::abs(normal.x) > 0.5)
        {
            saw_x_face = true;
            EXPECT_NEAR(magnitude, 1.0, 1.0e-12);
        }
        if (std::abs(normal.y) > 0.5)
        {
            saw_y_face = true;
            EXPECT_NEAR(magnitude, 2.0, 1.0e-12);
        }
        if (std::abs(normal.z) > 0.5)
        {
            saw_z_face = true;
            EXPECT_NEAR(magnitude, 3.0, 1.0e-12);
        }
    }

    EXPECT_TRUE(saw_x_face);
    EXPECT_TRUE(saw_y_face);
    EXPECT_TRUE(saw_z_face);
}

/** @brief Verifies pressure-weighted fluxes preserve a linear pressure field. */
TEST(FvmOperatorsTest, PressureWeightedFluxPreservesLinearPressure)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
                   mesh->num_owned_cells());
         ++lid)
    {
        const auto center = mesh->cell_centroid(lid);
        pressure.set_value(
            lid, 1.0 + 2.0 * center.x - 3.0 * center.y
                       + 0.5 * center.z);
    }
    pressure.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> fluxes(mesh, "pressure_weighted_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, 0.1, cache, fluxes);

    for (MeshType::local_ordinal_type face_lid = 0;
         face_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_faces());
         ++face_lid)
    {
        if (fluxes.is_owned_face(face_lid))
        {
            EXPECT_NEAR(fluxes.value(face_lid), 0.0, 1.0e-12);
        }
    }
}

/** @brief Verifies pressure-weighted fluxes suppress a checkerboard pressure mode. */
TEST(FvmOperatorsTest, PressureWeightedFluxSuppressesCheckerboardMode)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_4x4x4_database());
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
                   mesh->num_owned_cells());
         ++lid)
    {
        const auto center = mesh->cell_centroid(lid);
        const auto parity =
            static_cast<int>(center.x)
          + static_cast<int>(center.y)
          + static_cast<int>(center.z);
        pressure.set_value(
            lid, parity % 2 == 0 ? 1.0 : -1.0);
    }
    pressure.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> fluxes(mesh, "checkerboard_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, 0.1, cache, fluxes);

    Pack::scalar_type flux_norm = 0.0;
    Pack::scalar_type pressure_work = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto balance =
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                *mesh, fluxes, cell_lid);
        flux_norm += balance * balance;
        pressure_work += pressure.value(cell_lid) * balance;
    }

    EXPECT_GT(flux_norm, 1.0e-12);
    EXPECT_GT(pressure_work, 0.0);
}

/** @brief Verifies pressure-weighted flux workspaces reuse storage and overwrite stale values. */
TEST(FvmOperatorsTest,
     PressureWeightedFluxWorkspaceReusesStorageAndOverwritesValues)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 2.5};
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FVM::PressureWeightedFaceFluxWorkspace<Pack>
        workspace(mesh);
    SimpleFluid::FaceField<Pack> fluxes(mesh, "reused_workspace_flux");

    auto storage_pointers = [&]()
    {
        std::array<const Pack::scalar_type*, 6> pointers{};
        for (size_t component = 0; component < 3; ++component)
        {
            pointers[component] =
                workspace.pressure_gradient().owned_data()
                    .getData(component).getRawPtr();
            pointers[3 + component] =
                workspace.pressure_gradient().overlap_data()
                    .getData(component).getRawPtr();
        }
        return pointers;
    };
    const auto initial_storage = storage_pointers();
    const auto* const initial_boundary_locations =
        workspace.boundary_locations().data();
    ASSERT_EQ(workspace.boundary_locations().size(), mesh->num_faces());

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        velocity.set_owned_value(
            cell_lid,
            {0.1 + center.x,
             -0.2 + 0.5 * center.y,
             0.3 - 0.25 * center.z});
        pressure.set_owned_value(
            cell_lid, nonlinear_scalar(center));
    }
    velocity.sync_ghosts();
    pressure.sync_ghosts();

    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity,
        pressure,
        0.2,
        velocity_cache,
        bcs.pressure,
        workspace,
        fluxes);
    EXPECT_EQ(storage_pointers(), initial_storage);
    EXPECT_EQ(
        workspace.boundary_locations().data(),
        initial_boundary_locations);

    std::vector<Pack::scalar_type> first_fluxes(mesh->num_faces());
    for (MeshType::local_ordinal_type face_lid = 0;
         face_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_faces());
         ++face_lid)
    {
        if (fluxes.is_owned_face(face_lid))
        {
            first_fluxes[static_cast<size_t>(face_lid)] =
                fluxes.value(face_lid);
        }
    }
    std::vector<SimpleFluid::vec3<Pack::scalar_type>> first_gradient(
        mesh->num_local_cells());
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_local_cells());
         ++cell_lid)
    {
        first_gradient[static_cast<size_t>(cell_lid)] =
            workspace.pressure_gradient().local_value(cell_lid);
    }

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        velocity.set_owned_value(
            cell_lid,
            {1.0 + 2.0 * center.x,
             0.75 - 1.5 * center.y,
             -0.5 + center.z});
        pressure.set_owned_value(
            cell_lid,
            4.0 + 1.7 * nonlinear_scalar(center)
                + 0.4 * center.x * center.x);
    }
    velocity.sync_ghosts();
    pressure.sync_ghosts();
    bcs.pressure["xmax"].value = 8.0;

    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity,
        pressure,
        0.2,
        velocity_cache,
        bcs.pressure,
        workspace,
        fluxes);
    EXPECT_EQ(storage_pointers(), initial_storage);
    EXPECT_EQ(
        workspace.boundary_locations().data(),
        initial_boundary_locations);

    SimpleFluid::FVM::PressureWeightedFaceFluxWorkspace<Pack>
        fresh_workspace(mesh);
    SimpleFluid::FaceField<Pack> fresh_fluxes(
        mesh, "fresh_workspace_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity,
        pressure,
        0.2,
        velocity_cache,
        bcs.pressure,
        fresh_workspace,
        fresh_fluxes);

    bool face_flux_changed = false;
    for (MeshType::local_ordinal_type face_lid = 0;
         face_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_faces());
         ++face_lid)
    {
        if (!fluxes.is_owned_face(face_lid))
        {
            continue;
        }
        EXPECT_DOUBLE_EQ(
            fluxes.value(face_lid), fresh_fluxes.value(face_lid));
        face_flux_changed = face_flux_changed
                         || fluxes.value(face_lid)
                            != first_fluxes[static_cast<size_t>(face_lid)];
    }
    EXPECT_TRUE(face_flux_changed);

    bool pressure_gradient_changed = false;
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_local_cells());
         ++cell_lid)
    {
        const auto reused_gradient =
            workspace.pressure_gradient().local_value(cell_lid);
        const auto fresh_gradient =
            fresh_workspace.pressure_gradient().local_value(cell_lid);
        EXPECT_EQ(reused_gradient, fresh_gradient);
        const auto gradient_delta =
            reused_gradient
          - first_gradient[static_cast<size_t>(cell_lid)];
        pressure_gradient_changed = pressure_gradient_changed
                                 || gradient_delta.dot(gradient_delta)
                                        > 1.0e-20;
    }
    EXPECT_TRUE(pressure_gradient_changed);
}

/** @brief Verifies a pressure-weighted flux workspace rejects a different mesh. */
TEST(FvmOperatorsTest,
     PressureWeightedFluxWorkspaceRejectsAnotherMesh)
{
    auto workspace_mesh = make_mesh();
    auto field_mesh = make_mesh();
    SimpleFluid::FVM::PressureWeightedFaceFluxWorkspace<Pack>
        workspace(workspace_mesh);
    VectorFieldType velocity(field_mesh, "velocity");
    FieldType pressure(field_mesh, "pressure");
    SimpleFluid::BoundaryConditionSet bcs;
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            field_mesh, bcs);
    SimpleFluid::FaceField<Pack> fluxes(field_mesh, "face_flux");

    EXPECT_THROW(
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity,
            pressure,
            0.1,
            velocity_cache,
            workspace,
            fluxes),
        std::invalid_argument);
}

/** @brief Verifies fused face fluxes match the face-velocity composition. */
TEST(FvmOperatorsTest,
     FusedFaceFluxMatchesFaceVelocityComposition)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");
    FaceFluxWorkspace workspace(mesh);
    SimpleFluid::VectorFaceField<Pack> reference_face_velocity(
        mesh, "reference_face_velocity");
    SimpleFluid::FaceField<Pack> reference_fluxes(
        mesh, "reference_face_flux");
    SimpleFluid::FaceField<Pack> fused_fluxes(
        mesh, "fused_face_flux");
    SimpleFluid::FaceField<Pack> pressure_weighted_fluxes(
        mesh, "pressure_weighted_face_flux");

    const auto set_velocity = [&](Pack::scalar_type offset)
    {
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            const auto center = mesh->cell_centroid(cell_lid);
            velocity.set_owned_value(
                cell_lid,
                {offset + 0.75 + 1.5 * center.x - 0.2 * center.y,
                 -0.4 + 0.3 * center.x + 2.0 * center.y,
                 0.6 - 0.5 * center.y + 1.25 * center.z});
        }
        velocity.sync_ghosts();
    };

    const auto make_uniform_boundaries = [](
        SimpleFluid::BoundaryConditionType type)
    {
        SimpleFluid::BoundaryConditionSet conditions;
        for (const auto* name :
             {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        {
            conditions.velocity[name] = {type, {}};
        }
        return conditions;
    };

    SimpleFluid::BoundaryConditionSet default_boundaries;
    const auto neumann_boundaries = make_uniform_boundaries(
        SimpleFluid::BoundaryConditionType::Neumann);
    const auto no_slip_boundaries = make_uniform_boundaries(
        SimpleFluid::BoundaryConditionType::NoSlip);
    const auto slip_boundaries = make_uniform_boundaries(
        SimpleFluid::BoundaryConditionType::Slip);
    auto dirichlet_boundaries = make_uniform_boundaries(
        SimpleFluid::BoundaryConditionType::Dirichlet);
    for (auto& entry : dirichlet_boundaries.velocity)
    {
        entry.second.value = {1.25, -0.75, 2.5};
    }
    const std::array<std::pair<const char*,
                              const SimpleFluid::BoundaryConditionSet*>, 5>
        cases{{
            {"default", &default_boundaries},
            {"explicit Neumann", &neumann_boundaries},
            {"no-slip", &no_slip_boundaries},
            {"slip", &slip_boundaries},
            {"Dirichlet", &dirichlet_boundaries},
        }};

    const auto compare = [&](const char* input_name)
    {
        for (const auto& [boundary_name, boundaries] : cases)
        {
            SCOPED_TRACE(input_name);
            SCOPED_TRACE(boundary_name);
            const auto cache =
                SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
                    mesh, *boundaries);

            SimpleFluid::FVM::face_velocities(
                velocity, cache, reference_face_velocity);
            SimpleFluid::FVM::normal_face_fluxes(
                reference_face_velocity, reference_fluxes);
            SimpleFluid::FVM::face_fluxes(
                velocity, cache, fused_fluxes);
            SimpleFluid::FVM::pressure_weighted_face_fluxes(
                velocity,
                pressure,
                0.0,
                cache,
                workspace,
                pressure_weighted_fluxes);

            bool saw_interior_face = false;
            bool saw_boundary_face = false;
            for (MeshType::local_ordinal_type face_lid = 0;
                 face_lid < static_cast<MeshType::local_ordinal_type>(
                                mesh->num_faces());
                 ++face_lid)
            {
                if (!fused_fluxes.is_owned_face(face_lid))
                {
                    continue;
                }
                saw_interior_face = saw_interior_face
                                 || mesh->is_interior_face(face_lid);
                saw_boundary_face = saw_boundary_face
                                 || mesh->is_boundary_face(face_lid);
                EXPECT_DOUBLE_EQ(
                    fused_fluxes.value(face_lid),
                    reference_fluxes.value(face_lid));
                EXPECT_DOUBLE_EQ(
                    pressure_weighted_fluxes.value(face_lid),
                    reference_fluxes.value(face_lid));
            }
            EXPECT_TRUE(saw_interior_face);
            EXPECT_TRUE(saw_boundary_face);
        }
    };

    set_velocity(0.0);
    compare("initial velocity");
    set_velocity(4.0);
    compare("changed velocity");
}

/** @brief Verifies Dirichlet pressure rejects a prescribed velocity boundary flux. */
TEST(FvmOperatorsTest,
     DirichletPressureRejectsPrescribedVelocityBoundaryFlux)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        {3.0, 0.0, 0.0}};
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");

    EXPECT_THROW(
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity,
            pressure,
            0.1,
            cache,
            bcs.pressure,
            fluxes),
        std::invalid_argument);
}

/**
 * @brief Verifies interior face velocities are the arithmetic average of owner and neighbor cell values.
 */
TEST(FvmOperatorsTest, FaceVelocitiesInterpolateInteriorFaces)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& center = mesh->cell_centroid(lid);
        velocity.set_value(lid,
                           {center.x + 1.0,
                            2.0 * center.y + 3.0,
                            3.0 * center.z + 5.0});
    }
    velocity.sync_ghosts();

    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");
    SimpleFluid::FVM::face_velocities(velocity, face_velocity);

    bool saw_interior_face = false;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_interior_face(fid))
        {
            continue;
        }

        saw_interior_face = true;
        const auto owner = mesh->owner_cell(fid);
        const auto neighbor = mesh->neighbor_cell(fid);
        const auto expected =
            (velocity.local_value(owner) + velocity.local_value(neighbor)) / 2.0;
        const auto actual = face_velocity.value(fid);

        EXPECT_NEAR(actual.x, expected.x, 1.0e-12);
        EXPECT_NEAR(actual.y, expected.y, 1.0e-12);
        EXPECT_NEAR(actual.z, expected.z, 1.0e-12);
    }

    EXPECT_TRUE(saw_interior_face);
}

/** @brief Verifies no-slip boundaries produce zero face velocity. */
TEST(FvmOperatorsTest, NoSlipBoundaryProducesZeroFaceVelocity)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");
    SimpleFluid::FVM::face_velocities(velocity, cache, face_velocity);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            EXPECT_EQ(face_velocity.value(fid), (SimpleFluid::vec3{}));
        }
    }
}

/** @brief Verifies no-slip boundaries produce zero exterior flux. */
TEST(FvmOperatorsTest, NoSlipBoundaryProducesZeroExteriorFlux)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FVM::face_fluxes(velocity, cache, fluxes);

    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (mesh->is_boundary_face(fid))
        {
            EXPECT_DOUBLE_EQ(fluxes.is_owned_face(fid) ? fluxes.value(fid) : 0.0, 0.0);
        }
    }
}

/**
 * @brief Checks slip boundary removes the normal component of face velocity and produces zero normal flux.
 */
TEST(FvmOperatorsTest, SlipBoundaryRemovesNormalVelocityAndFlux)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, "velocity");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& center = mesh->cell_centroid(lid);
        velocity.set_value(lid,
                           {center.x + 1.0,
                            2.0 * center.y + 3.0,
                            3.0 * center.z + 5.0});
    }
    velocity.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    }

    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::VectorFaceField<Pack> face_velocity(mesh, "face_velocity");
    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FVM::face_velocities(velocity, cache, face_velocity);
    SimpleFluid::FVM::normal_face_fluxes(face_velocity, fluxes);

    bool saw_boundary_face = false;
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        if (!mesh->is_boundary_face(fid))
        {
            continue;
        }

        saw_boundary_face = true;
        const auto owner = mesh->owner_cell(fid);
        const auto& normal = mesh->face_normal_outward(fid, owner);
        const auto owner_velocity = velocity.local_value(owner);
        const auto expected =
            owner_velocity - normal * owner_velocity.dot(normal);
        const auto actual = face_velocity.value(fid);

        EXPECT_NEAR(actual.x, expected.x, 1.0e-12);
        EXPECT_NEAR(actual.y, expected.y, 1.0e-12);
        EXPECT_NEAR(actual.z, expected.z, 1.0e-12);
        EXPECT_NEAR(actual.dot(normal), 0.0, 1.0e-12);
        EXPECT_NEAR(fluxes.value(fid), 0.0, 1.0e-12);
    }

    EXPECT_TRUE(saw_boundary_face);
}

/** @brief Verifies upwind and pressure-Poisson matrix construction. */
TEST(FvmOperatorsTest, BuildsUpwindAndPressurePoissonMatrices)
{
    auto mesh = make_mesh();
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 0.0, 0.0}, "velocity");

    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FVM::face_fluxes(velocity, fluxes);
    auto convection =
        SimpleFluid::FVM::upwind_convection_matrix<Pack>(*mesh, fluxes);
    auto pressure = SimpleFluid::FVM::pressure_poisson_matrix<Pack>(
        *mesh, mesh->owned_cell_map()->getMinAllGlobalIndex());

    EXPECT_EQ(convection->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(pressure->getGlobalNumRows(),
              mesh->owned_cell_map()->getGlobalNumElements());
}

/**
 * @brief Verifies the scalar transport RHS includes the volume-scaled source term.
 */
TEST(FvmOperatorsTest, ScalarTransportRhsIncludesCellSourceTerm)
{
    auto mesh = make_mesh();
    FieldType old_values(mesh, "old_values");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, 3.0 + static_cast<double>(lid));
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [](int, MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto source =
        [&](MeshType::local_ordinal_type cell_lid) -> Pack::scalar_type
    {
        return 2.0 + mesh->cell_centroid(cell_lid).x;
    };

    constexpr double time_step = 0.25;
    const auto system = SimpleFluid::FVM::transport_system<Pack>(
        old_values, zero_fluxes, time_step, 0.0, boundary_value, source);

    const auto rhs_data = system.rhs->getData();
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto volume = mesh->cell_volume(lid);
        const auto expected =
            volume / time_step * old_values.value(lid) + volume * source(lid);
        EXPECT_NEAR(rhs_data[lid], expected, 1.0e-12);
    }
}

/**
 * @brief Verifies the vector transport RHS includes the volume-scaled source term for each component.
 */
TEST(FvmOperatorsTest, VectorTransportRhsIncludesCellSourceTerm)
{
    auto mesh = make_mesh();
    VectorFieldType old_values(mesh, "old_values");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(
            lid,
            {1.0 + static_cast<double>(lid),
             2.0 + static_cast<double>(lid),
             3.0 + static_cast<double>(lid)});
    }
    old_values.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [](int, MeshType::local_ordinal_type) -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return {};
    };
    auto source =
        [&](MeshType::local_ordinal_type cell_lid)
            -> SimpleFluid::vec3<Pack::scalar_type>
    {
        const auto& center = mesh->cell_centroid(cell_lid);
        return {2.0 + center.x, 3.0 + center.y, 4.0 + center.z};
    };

    constexpr double time_step = 0.25;
    const auto system = SimpleFluid::FVM::transport_system<Pack>(
        old_values, zero_fluxes, time_step, 0.0, boundary_value, source);

    for (size_t component = 0;
         component < VectorFieldType::num_components;
         ++component)
    {
        const auto rhs_data = system.rhs->getData(component);
        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(
                 mesh->num_owned_cells());
             ++lid)
        {
            const auto volume = mesh->cell_volume(lid);
            const auto expected =
                volume / time_step * old_values.value(lid).component(component)
              + volume * source(lid).component(component);
            EXPECT_NEAR(rhs_data[lid], expected, 1.0e-12);
        }
    }
}

/** @brief Verifies scalar transport treats Neumann boundary data as flux. */
TEST(FvmOperatorsTest, ScalarTransportTreatsNeumannBoundaryAsFlux)
{
    auto mesh = make_mesh();
    FieldType old_values(mesh, 0.0, "old_values");
    old_values.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");

    const int neumann_batch = mesh->boundary_batches().begin()->first;
    constexpr double diffusivity = 2.0;
    constexpr double gradient = 3.0;
    constexpr double time_step = 1.0;
    auto boundary_condition =
        [&](int batch_id, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {
            batch_id == neumann_batch
                ? SimpleFluid::BoundaryConditionType::Neumann
                : SimpleFluid::BoundaryConditionType::Dirichlet,
            batch_id == neumann_batch ? gradient : 0.0};
    };
    auto boundary_value =
        [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto zero_source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto system = SimpleFluid::FVM::transport_system<Pack>(
        old_values, zero_fluxes, time_step, diffusivity,
        boundary_condition, boundary_value, zero_source);

    std::vector<double> expected_diagonal(mesh->num_owned_cells(), 0.0);
    std::vector<double> expected_rhs(mesh->num_owned_cells(), 0.0);
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        expected_diagonal[static_cast<size_t>(cell_lid)] =
            mesh->cell_volume(cell_lid) / time_step;
        for (const auto face_lid : mesh->faces(cell_lid))
        {
            if (!mesh->is_interior_face(face_lid))
            {
                continue;
            }
            const auto other =
                mesh->opposite_or_periodic_neighbor_cell(
                    face_lid, cell_lid);
            expected_diagonal[static_cast<size_t>(cell_lid)] +=
                SimpleFluid::FVM::detail::interior_diffusion_coefficient(
                    *mesh, face_lid, cell_lid, other, diffusivity);
        }
    }

    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        for (const auto face_lid : batch.face_lids)
        {
            if (!mesh->is_owned_face(face_lid)
                || !mesh->is_boundary_face(face_lid))
            {
                continue;
            }
            const auto owner = mesh->owner_cell(face_lid);
            if (batch_id == neumann_batch)
            {
                expected_rhs[static_cast<size_t>(owner)] +=
                    diffusivity * gradient * mesh->face_area(face_lid);
                continue;
            }

            expected_diagonal[static_cast<size_t>(owner)] +=
                SimpleFluid::FVM::detail::boundary_diffusion_coefficient(
                    *mesh, face_lid, owner, diffusivity);
        }
    }

    const auto rhs_data = system.rhs->getData();
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        EXPECT_NEAR(
            local_matrix_entry(*system.matrix, cell_lid, cell_lid),
            expected_diagonal[static_cast<size_t>(cell_lid)],
            1.0e-12);
        EXPECT_NEAR(
            rhs_data[cell_lid],
            expected_rhs[static_cast<size_t>(cell_lid)],
            1.0e-12);
    }
}

/** @brief Verifies vector transport treats Neumann boundary data as flux. */
TEST(FvmOperatorsTest, VectorTransportTreatsNeumannBoundaryAsFlux)
{
    auto mesh = make_mesh();
    VectorFieldType old_values(mesh, "old_values");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++lid)
    {
        old_values.set_value(lid, {});
    }
    old_values.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");

    const int neumann_batch = mesh->boundary_batches().begin()->first;
    constexpr double diffusivity = 2.0;
    const SimpleFluid::vec3<Pack::scalar_type> gradient{1.0, 2.0, 3.0};
    auto boundary_condition =
        [&](int batch_id, size_t) -> SimpleFluid::VectorBoundaryCondition
    {
        return {
            batch_id == neumann_batch
                ? SimpleFluid::BoundaryConditionType::Neumann
                : SimpleFluid::BoundaryConditionType::Dirichlet,
            batch_id == neumann_batch
                ? gradient
                : SimpleFluid::vec3<Pack::scalar_type>{}};
    };
    auto boundary_value =
        [](int, size_t) -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return {};
    };
    auto zero_source =
        [](MeshType::local_ordinal_type)
            -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return {};
    };

    const auto system = SimpleFluid::FVM::transport_system<Pack>(
        old_values, zero_fluxes, 1.0, diffusivity,
        boundary_condition, boundary_value, zero_source);

    std::array<double, VectorFieldType::num_components> expected_rhs{};
    for (const auto face_lid :
         mesh->boundary_batches().at(neumann_batch).face_lids)
    {
        if (!mesh->is_owned_face(face_lid)
            || !mesh->is_boundary_face(face_lid))
        {
            continue;
        }
        for (size_t component = 0;
             component < VectorFieldType::num_components;
             ++component)
        {
            expected_rhs[component] +=
                diffusivity * gradient.component(component)
              * mesh->face_area(face_lid);
        }
    }

    for (size_t component = 0;
         component < VectorFieldType::num_components;
         ++component)
    {
        const auto rhs_data = system.rhs->getData(component);
        double actual = 0.0;
        for (MeshType::local_ordinal_type cell_lid = 0;
             cell_lid < static_cast<MeshType::local_ordinal_type>(
                 mesh->num_owned_cells());
             ++cell_lid)
        {
            actual += rhs_data[cell_lid];
        }
        EXPECT_NEAR(actual, expected_rhs[component], 1.0e-12);
    }
}

/** @brief Verifies scalar and vector transport systems reuse supplied matrices. */
TEST(FvmOperatorsTest, ReusesScalarAndVectorTransportMatrices)
{
    auto mesh = make_mesh();
    FieldType scalar(mesh, 2.0, "scalar");
    VectorFieldType vector(
        mesh, SimpleFluid::vec3{1.0, 2.0, 3.0}, "vector");
    SimpleFluid::FaceField<Pack> zero_fluxes(
        mesh, 0.0, "face_flux");

    auto scalar_boundary = [](int, size_t) { return 0.0; };
    auto vector_boundary = [](int, size_t)
    {
        return SimpleFluid::vec3<Pack::scalar_type>{};
    };

    auto scalar_first = SimpleFluid::FVM::transport_system<Pack>(
        scalar, zero_fluxes, 1.0, 1.0, scalar_boundary);
    auto scalar_second = SimpleFluid::FVM::transport_system<Pack>(
        scalar, zero_fluxes, 0.5, 0.0, scalar_boundary,
        scalar_first.matrix);
    EXPECT_EQ(
        scalar_second.matrix.getRawPtr(),
        scalar_first.matrix.getRawPtr());

    auto vector_first = SimpleFluid::FVM::transport_system<Pack>(
        vector, zero_fluxes, 1.0, 1.0, vector_boundary);
    auto vector_second = SimpleFluid::FVM::transport_system<Pack>(
        vector, zero_fluxes, 0.25, 0.0, vector_boundary,
        vector_first.matrix);
    EXPECT_EQ(
        vector_second.matrix.getRawPtr(),
        vector_first.matrix.getRawPtr());

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_NEAR(
            local_matrix_entry(*scalar_second.matrix, lid, lid),
            mesh->cell_volume(lid) / 0.5,
            1.0e-12);
        EXPECT_NEAR(
            local_matrix_entry(*vector_second.matrix, lid, lid),
            mesh->cell_volume(lid) / 0.25,
            1.0e-12);
    }
}

// =========================================================================
//  Periodic Boundary Condition Tests
// =========================================================================

/**
 * @brief Create a 2×1×1 box mesh and pair the xmin face of cell 0 with
 *        the xmax face of cell 1 as a periodic boundary.
 *
 * @return Shared pointer to the configured mesh with periodic pairs set.
 */
SimpleFluid::SP<MeshType> make_periodic_box_mesh()
{
    constexpr int n = 2;
    auto db = SimpleFluid::test::make_box_database(n, 1, 1, 1.0 / n);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);

    // Find xmin and xmax boundary faces.
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmin_face = -1;
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmax_face = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin" && !batch.face_lids.empty()) xmin_face = batch.face_lids[0];
        if (name == "xmax" && !batch.face_lids.empty()) xmax_face = batch.face_lids[0];
    }

    // The cell adjacent to the xmin face is cell 0; its paired cell is the
    // cell adjacent to the xmax face (cell 1), and vice versa.
    const auto owner0 = mesh->owner_cell(xmin_face);
    const auto owner1 = mesh->owner_cell(xmax_face);
    mesh->set_periodic_face(xmin_face, owner1);
    mesh->set_periodic_face(xmax_face, owner0);

    return mesh;
}

/** @brief Verifies fused periodic face fluxes match the face-velocity composition. */
TEST(FvmOperatorsTest,
     FusedFaceFluxMatchesFaceVelocityCompositionOnPeriodicMesh)
{
    auto mesh = make_periodic_box_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType pressure(mesh, "pressure");

    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::Periodic, {}};
    boundaries.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Periodic, {}};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundaries);

    SimpleFluid::VectorFaceField<Pack> reference_face_velocity(
        mesh, "periodic_reference_face_velocity");
    SimpleFluid::FaceField<Pack> reference_fluxes(
        mesh, "periodic_reference_flux");
    SimpleFluid::FaceField<Pack> fused_fluxes(
        mesh, "periodic_fused_flux");
    SimpleFluid::FaceField<Pack> pressure_weighted_fluxes(
        mesh, "periodic_pressure_weighted_flux");
    FaceFluxWorkspace workspace(mesh);

    const auto compare = [&](const char* input_name)
    {
        SCOPED_TRACE(input_name);
        SimpleFluid::FVM::face_velocities(
            velocity, cache, reference_face_velocity);
        SimpleFluid::FVM::normal_face_fluxes(
            reference_face_velocity, reference_fluxes);
        SimpleFluid::FVM::face_fluxes(
            velocity, cache, fused_fluxes);
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity,
            pressure,
            0.0,
            cache,
            workspace,
            pressure_weighted_fluxes);

        for (MeshType::local_ordinal_type face_lid = 0;
             face_lid < static_cast<MeshType::local_ordinal_type>(
                            mesh->num_faces());
             ++face_lid)
        {
            if (!fused_fluxes.is_owned_face(face_lid))
            {
                continue;
            }
            EXPECT_DOUBLE_EQ(
                fused_fluxes.value(face_lid),
                reference_fluxes.value(face_lid));
            EXPECT_DOUBLE_EQ(
                pressure_weighted_fluxes.value(face_lid),
                reference_fluxes.value(face_lid));
        }

        const auto xmin_face = boundary_face_lid(*mesh, "xmin");
        const auto xmax_face = boundary_face_lid(*mesh, "xmax");
        EXPECT_TRUE(mesh->is_interior_face(xmin_face));
        EXPECT_TRUE(mesh->is_interior_face(xmax_face));
        EXPECT_NE(reference_fluxes.value(xmin_face), 0.0);
        EXPECT_NE(reference_fluxes.value(xmax_face), 0.0);
    };

    velocity.set_owned_value(0, {1.0, -0.5, 0.25});
    velocity.set_owned_value(1, {3.0, 1.5, -0.75});
    velocity.sync_ghosts();
    compare("initial velocity");

    velocity.set_owned_value(0, {-2.0, 0.75, 1.25});
    velocity.set_owned_value(1, {0.5, -1.0, 2.0});
    velocity.sync_ghosts();
    compare("changed velocity");
}

/**
 * @brief Verifies that periodic faces are correctly identified and that
 *        face velocities average the owner and paired-cell values.
 */
TEST(FvmOperatorsTest, PeriodicBoundaryFaceVelocityIsAveraged)
{
    auto mesh = make_periodic_box_mesh();

    // Set velocity: cell 0 → (1,0,0), cell 1 → (3,0,0)
    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {1.0, 0.0, 0.0});
    velocity.set_value(1, {3.0, 0.0, 0.0});
    velocity.sync_ghosts();

    // Assemble face velocities without boundary-condition treatment.
    SimpleFluid::VectorFaceField<Pack> face_vel(mesh, "face_vel");
    SimpleFluid::FVM::face_velocities(velocity, face_vel);

    // Find periodic faces.
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmin_face = -1;
    SimpleFluid::Mesh<Pack>::local_ordinal_type xmax_face = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        if (name == "xmin" && !batch.face_lids.empty()) xmin_face = batch.face_lids[0];
        if (name == "xmax" && !batch.face_lids.empty()) xmax_face = batch.face_lids[0];
    }

    ASSERT_NE(xmin_face, -1);
    ASSERT_NE(xmax_face, -1);
    EXPECT_TRUE(mesh->is_interior_face(xmin_face));
    EXPECT_TRUE(mesh->is_interior_face(xmax_face));
    EXPECT_FALSE(mesh->is_boundary_face(xmin_face));
    EXPECT_FALSE(mesh->is_boundary_face(xmax_face));
    EXPECT_EQ(mesh->opposite_or_periodic_neighbor_cell(
                  xmin_face, mesh->owner_cell(xmin_face)),
              mesh->owner_cell(xmax_face));
    EXPECT_EQ(mesh->opposite_or_periodic_neighbor_cell(
                  xmax_face, mesh->owner_cell(xmax_face)),
              mesh->owner_cell(xmin_face));

    // Periodic faces should get the average of the owner and paired cell.
    const auto v_xmin = face_vel.value(xmin_face);
    const auto v_xmax = face_vel.value(xmax_face);
    EXPECT_NEAR(v_xmin.x, 2.0, 1.0e-12);  // (1 + 3) / 2
    EXPECT_NEAR(v_xmax.x, 2.0, 1.0e-12);  // (3 + 1) / 2
}

/** @brief Verifies the periodic velocity cache preserves paired-face values. */
TEST(FvmOperatorsTest, PeriodicVelocityCacheDoesNotOverwritePairedFaceVelocity)
{
    auto mesh = make_periodic_box_mesh();

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {1.0, 0.0, 0.0});
    velocity.set_value(1, {3.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.velocity["xmin"] = {SimpleFluid::BoundaryConditionType::Periodic, {}};
    bcs.velocity["xmax"] = {SimpleFluid::BoundaryConditionType::Periodic, {}};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    EXPECT_EQ(cache.value.size(), mesh->boundary_batches().size());

    SimpleFluid::VectorFaceField<Pack> face_vel(mesh, "face_vel");
    EXPECT_NO_THROW(SimpleFluid::FVM::face_velocities(
        velocity, cache, face_vel));

    const auto xmin_face = boundary_face_lid(*mesh, "xmin");
    const auto xmax_face = boundary_face_lid(*mesh, "xmax");
    EXPECT_NEAR(face_vel.value(xmin_face).x, 2.0, 1.0e-12);
    EXPECT_NEAR(face_vel.value(xmax_face).x, 2.0, 1.0e-12);
}

/** @brief Verifies the velocity boundary cache rejects nonzero Neumann derivatives. */
TEST(FvmOperatorsTest, VelocityBoundaryCacheRejectsNonzeroNeumannDerivative)
{
    auto mesh = make_mesh();
    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann,
        {1.0, 0.0, 0.0}};

    EXPECT_THROW(
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundaries),
        std::invalid_argument);
}

/**
 * @brief Verifies that periodic face fluxes are computed correctly,
 *        treating the periodic boundary like an interior face.
 *
 * A 2×1×1 mesh with mesh_size = 0.5 gives face areas of 0.25 and
 * cell volumes of 0.125.  Cell 0 has velocity (2,0,0), cell 1 has (0,0,0).
 * The averaged face velocity at the periodic xmax face is (1,0,0).
 * With face normal (+1,0,0) and area 0.25: flux = 1 × 1 × 0.25 = 0.25.
 */
TEST(FvmOperatorsTest, PeriodicBoundaryFluxIsComputed)
{
    auto mesh = make_periodic_box_mesh();

    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {2.0, 0.0, 0.0});
    velocity.set_value(1, {0.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
    SimpleFluid::FVM::face_fluxes(velocity, fluxes);

    SimpleFluid::Mesh<Pack>::local_ordinal_type xmax_face = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) == "xmax" && !batch.face_lids.empty())
            xmax_face = batch.face_lids[0];
    }
    ASSERT_NE(xmax_face, -1);
    ASSERT_TRUE(fluxes.is_owned_face(xmax_face));

    // Averaged face velocity: (2 + 0)/2 = 1.0 along +x.
    // Face area = dy*dz = 0.5 * 0.5 = 0.25.
    // Normal on xmax = (+1, 0, 0).
    // Flux = v·n·A = 1.0 * 1.0 * 0.25 = 0.25.
    EXPECT_NEAR(fluxes.value(xmax_face), 0.25, 1.0e-12);
}

/**
 * @brief Verifies that explicit diffusion with periodic boundary
 *        conditions correctly exchanges heat between the paired cells.
 *
 * A 2×1×1 mesh with mesh_size = 0.5 has face area 0.25 and cell volume
 * 0.125.  Cell 0 (T=1) at x=0.25, cell 1 (T=0) at x=0.75.  The interior
 * face connects them (distance 0.5).  The periodic xmin face of cell 0
 * also connects to cell 1 (distance 0.5), doubling the laplacian.
 *
 * laplacian(cell 0) = -1.0,  laplacian/vol = -8.0
 * T0_new = 1.0 + 0.1 * (-8.0) = 0.2
 * T1_new = 0.0 + 0.1 * 8.0  = 0.8
 */
TEST(FvmOperatorsTest, PeriodicBoundaryExplicitDiffusionTransfersHeat)
{
    auto mesh = make_periodic_box_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 1.0);
    temperature.set_value(1, 0.0);
    temperature.sync_ghosts();

    // Insulated on Y and Z faces; periodic on X faces.
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
        bcs.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Periodic, 0.0};
    bcs.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Periodic, 0.0};

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    const auto old_t = local_values(temperature);
    equation.advance_explicit(old_t, 0.1, 1.0, temperature);

    // With doubled connectivity (interior + periodic), the heat transfer is
    // amplified: each cell exchanges with the other through two faces.
    // area = 0.25, dist = 0.5, vol = 0.125
    // laplacian magnitude = 2 * (1.0 - 0.0) * 0.25 / 0.5 = 1.0
    // laplacian/vol = 1.0 / 0.125 = 8.0
    // T0_new = 1.0 - 0.1 * 8.0 = 0.2
    // T1_new = 0.0 + 0.1 * 8.0 = 0.8
    EXPECT_NEAR(temperature.value(0), 0.2, 1.0e-12);
    EXPECT_NEAR(temperature.value(1), 0.8, 1.0e-12);
}

/** @brief Verifies periodic scalar transport couples to the paired cell. */
TEST(FvmOperatorsTest, PeriodicBoundaryScalarTransportMatrixUsesPairedCell)
{
    auto mesh = make_periodic_box_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 1.0);
    temperature.set_value(1, 0.0);
    temperature.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [&](int batch_id, MeshType::local_ordinal_type in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[
                static_cast<size_t>(in_batch_id)];
        EXPECT_TRUE(mesh->is_boundary_face(face_lid));
        return 0.0;
    };

    const auto system = SimpleFluid::FVM::transport_system<Pack>(
        temperature, zero_fluxes, 1.0, 1.0, boundary_value);

    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 0), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 1), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 1), -1.0, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 0), -1.0, 1.0e-12);
}

/** @brief Verifies periodic upwind transport couples to the paired cell. */
TEST(FvmOperatorsTest, PeriodicBoundaryUpwindMatrixUsesPairedCell)
{
    auto mesh = make_periodic_box_mesh();
    const auto xmin_face = boundary_face_lid(*mesh, "xmin");
    const auto xmax_face = boundary_face_lid(*mesh, "xmax");

    SimpleFluid::FaceField<Pack> fluxes(mesh, 0.0, "face_flux");
    fluxes.set_value(xmin_face, -0.25);
    fluxes.set_value(xmax_face, -0.5);

    const auto matrix =
        SimpleFluid::FVM::upwind_convection_matrix<Pack>(
            *mesh, fluxes);

    EXPECT_NEAR(local_matrix_entry(*matrix, 0, 0), 0.0, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*matrix, 1, 1), 0.0, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*matrix, 0, 1), -0.25, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*matrix, 1, 0), -0.5, 1.0e-12);
}

/** @brief Verifies periodic vector transport couples to the paired cell. */
TEST(FvmOperatorsTest, PeriodicBoundaryVectorTransportMatrixUsesPairedCell)
{
    auto mesh = make_periodic_box_mesh();
    VectorFieldType velocity(mesh, "velocity");
    velocity.set_value(0, {1.0, 2.0, 3.0});
    velocity.set_value(1, {0.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");
    auto boundary_value =
        [&](int batch_id, MeshType::local_ordinal_type in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[
                static_cast<size_t>(in_batch_id)];
        EXPECT_TRUE(mesh->is_boundary_face(face_lid));
        return SimpleFluid::vec3<Pack::scalar_type>{};
    };

    const auto system = SimpleFluid::FVM::transport_system<Pack>(
        velocity, zero_fluxes, 1.0, 1.0, boundary_value);

    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 0), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 1), 5.125, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 0, 1), -1.0, 1.0e-12);
    EXPECT_NEAR(local_matrix_entry(*system.matrix, 1, 0), -1.0, 1.0e-12);
}

/** @brief Verifies harmonic face interpolation uses cell-to-face distances. */
TEST(FvmOperatorsTest, HarmonicFaceValueUsesCellToFaceDistances)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    FieldType coefficient(mesh, "coefficient");
    coefficient.set_value(0, 1.0);
    coefficient.set_value(1, 4.0);
    coefficient.sync_ghosts();

    MeshType::local_ordinal_type interior_face =
        static_cast<MeshType::local_ordinal_type>(-1);
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<MeshType::local_ordinal_type>(face);
        if (mesh->is_interior_face(face_lid))
        {
            interior_face = face_lid;
            break;
        }
    }
    ASSERT_GE(interior_face, 0);

    EXPECT_NEAR(
        SimpleFluid::FVM::detail::harmonic_face_value(
            *mesh, interior_face, 0, 1, coefficient),
        1.6,
        1.0e-12);
}

/** @brief Verifies harmonic face interpolation rejects negative cell values. */
TEST(FvmOperatorsTest, HarmonicFaceValueRejectsNegativeCellValues)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    FieldType coefficient(mesh, "coefficient");
    coefficient.set_value(0, 1.0);
    coefficient.set_value(1, -1.0);
    coefficient.sync_ghosts();

    MeshType::local_ordinal_type interior_face =
        static_cast<MeshType::local_ordinal_type>(-1);
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<MeshType::local_ordinal_type>(face);
        if (mesh->is_interior_face(face_lid))
        {
            interior_face = face_lid;
            break;
        }
    }
    ASSERT_GE(interior_face, 0);

    EXPECT_THROW(
        SimpleFluid::FVM::detail::harmonic_face_value(
            *mesh, interior_face, 0, 1, coefficient),
        std::invalid_argument);
}

/** @brief Verifies weighted scalar transport treats Neumann data as flux. */
TEST(FvmOperatorsTest, WeightedScalarTransportTreatsNeumannBoundaryAsFlux)
{
    auto mesh = make_mesh();
    FieldType scalar(mesh, "scalar");
    FieldType storage(mesh, 1.0, "storage");
    FieldType advection(mesh, 0.0, "advection");
    FieldType diffusivity(mesh, 2.0, "diffusivity");
    scalar.sync_ghosts();
    diffusivity.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");

    const int neumann_batch = mesh->boundary_batches().begin()->first;
    constexpr double gradient = 3.0;
    auto boundary_condition =
        [&](int batch_id, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {
            batch_id == neumann_batch
                ? SimpleFluid::BoundaryConditionType::Neumann
                : SimpleFluid::BoundaryConditionType::Dirichlet,
            batch_id == neumann_batch ? gradient : 0.0};
    };
    auto boundary_value =
        [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto zero_source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto system =
        SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            scalar,
            zero_fluxes,
            1.0,
            storage,
            advection,
            diffusivity,
            boundary_condition,
            boundary_value,
            zero_source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);

    std::vector<double> expected_rhs(mesh->num_owned_cells(), 0.0);
    for (const auto face_lid :
         mesh->boundary_batches().at(neumann_batch).face_lids)
    {
        if (!mesh->is_owned_face(face_lid)
            || !mesh->is_boundary_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        expected_rhs[static_cast<size_t>(owner)] +=
            diffusivity.value(owner) * gradient * mesh->face_area(face_lid);
    }

    const auto rhs_data = system.rhs->getData();
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        EXPECT_NEAR(
            rhs_data[cell_lid],
            expected_rhs[static_cast<size_t>(cell_lid)],
            1.0e-12);
    }
}

/** @brief Verifies weighted scalar transport adds and validates an implicit sink. */
TEST(FvmOperatorsTest,
     WeightedScalarTransportAddsAndValidatesImplicitSink)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
    FieldType scalar(mesh, 5.0, "scalar");
    FieldType storage(mesh, 2.0, "storage");
    FieldType advection(mesh, 0.0, "advection");
    FieldType diffusivity(mesh, 0.0, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");

    auto boundary_condition =
        [](int, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    };
    auto boundary_value = [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto source = [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 1.0;
    };
    using SinkProvider = std::function<Pack::scalar_type(
        MeshType::local_ordinal_type)>;
    auto assemble = [&](Pack::scalar_type sink)
    {
        return SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            scalar,
            zero_fluxes,
            0.5,
            storage,
            advection,
            diffusivity,
            boundary_condition,
            boundary_value,
            source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            nullptr,
            Teuchos::null,
            SinkProvider{[sink](MeshType::local_ordinal_type)
            {
                return sink;
            }});
    };

    const auto system = assemble(3.0);
    const auto volume = mesh->cell_volume(0);
    const auto expected_diagonal = volume * (2.0 / 0.5 + 3.0);
    const auto expected_rhs = volume * (2.0 / 0.5 * 5.0 + 1.0);
    EXPECT_NEAR(
        local_matrix_entry(*system.matrix, 0, 0),
        expected_diagonal,
        1.0e-12);
    EXPECT_NEAR(system.rhs->getData()[0], expected_rhs, 1.0e-12);

    FieldType expected_solution(mesh, expected_rhs / expected_diagonal,
                                "expected_solution");
    const auto matrix_action =
        local_matrix_action(*system.matrix, expected_solution);
    ASSERT_EQ(matrix_action.size(), 1U);
    EXPECT_NEAR(matrix_action[0], expected_rhs, 1.0e-12);
    EXPECT_NEAR(expected_solution.value(0), 3.0, 1.0e-12);

    EXPECT_THROW(assemble(-1.0), std::invalid_argument);
    EXPECT_THROW(
        assemble(std::numeric_limits<Pack::scalar_type>::infinity()),
        std::invalid_argument);
}

/** @brief Verifies fixed weighted-scalar cells remain exact under explicit correction. */
TEST(FvmOperatorsTest,
     WeightedScalarTransportFixedCellIsExactIdentityWithExplicitCorrection)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType scalar(mesh, "scalar");
    FieldType storage(mesh, 1.0, "storage");
    FieldType advection(mesh, 1.0, "advection");
    FieldType diffusivity(mesh, 2.0, "diffusivity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        scalar.set_owned_value(
            static_cast<MeshType::local_ordinal_type>(owned),
            1.0 + static_cast<double>(owned));
    }
    scalar.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");

    auto boundary_condition =
        [](int, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    };
    auto boundary_value = [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto source = [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 3.0;
    };
    using FixedProvider = std::function<std::optional<Pack::scalar_type>(
        MeshType::local_ordinal_type)>;
    constexpr Pack::scalar_type fixed_value = 7.25;
    FixedProvider fixed = [fixed_value](MeshType::local_ordinal_type cell_lid)
        -> std::optional<Pack::scalar_type>
    {
        return cell_lid == 0
             ? std::optional<Pack::scalar_type>{fixed_value}
             : std::nullopt;
    };

    const auto system =
        SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            scalar,
            zero_fluxes,
            0.5,
            storage,
            advection,
            diffusivity,
            boundary_condition,
            boundary_value,
            source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
            &scalar,
            Teuchos::null,
            {},
            fixed);

    EXPECT_NEAR(system.rhs->getData()[0], fixed_value, 1.0e-14);
    const auto row_entries = system.matrix->getNumEntriesInLocalRow(0);
    typename Pack::matrix_type::nonconst_local_inds_host_view_type columns(
        "columns", row_entries);
    typename Pack::matrix_type::nonconst_values_host_view_type values(
        "values", row_entries);
    size_t num_entries = 0;
    system.matrix->getLocalRowCopy(
        0, columns, values, num_entries);
    for (size_t entry = 0; entry < num_entries; ++entry)
    {
        EXPECT_NEAR(
            values(entry),
            columns(entry) == 0 ? 1.0 : 0.0,
            1.0e-14);
    }

    FixedProvider non_finite = [](MeshType::local_ordinal_type)
        -> std::optional<Pack::scalar_type>
    {
        return std::numeric_limits<Pack::scalar_type>::infinity();
    };
    EXPECT_THROW(
        SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            scalar, zero_fluxes, 0.5, storage, advection, diffusivity,
            boundary_condition, boundary_value, source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
            &scalar, Teuchos::null, {}, non_finite),
        std::invalid_argument);
}

/** @brief Verifies explicit variable diffusion uses boundary values and sparse coefficients. */
TEST(FvmOperatorsTest,
     ExplicitVariableDiffusionUsesBoundaryValuesAndSparseCoefficients)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType scalar(mesh, 0.0, "scalar");
    FieldType unit_weight(mesh, 1.0, "unit_weight");
    FieldType diffusivity(mesh, 2.0, "diffusivity");
    VectorFieldType velocity(mesh, "velocity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "zero_fluxes");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto centroid = mesh->cell_centroid(cell_lid);
        scalar.set_owned_value(
            cell_lid,
            0.3 + centroid.x * centroid.y - 0.2 * centroid.z);
        velocity.set_owned_value(
            cell_lid,
            {centroid.y + 0.1, centroid.x * centroid.z, -0.3 * centroid.z});
    }
    scalar.sync_ghosts();
    velocity.sync_ghosts();

    const auto& [cached_batch_id, cached_batch] =
        *mesh->boundary_batches().begin();
    SimpleFluid::FVM::BoundaryCache<Pack> cache{{}, mesh};
    cache.value[cached_batch_id] = SimpleFluid::Arr<Pack::scalar_type>(
        cached_batch.face_lids.size(), 6.0);
    auto boundary_coefficient =
        [&](int batch_id, size_t in_batch_id,
            Pack::scalar_type owner_value)
    {
        return SimpleFluid::FVM::boundary_coefficient<Pack>(
            &cache, batch_id, in_batch_id, owner_value);
    };

    auto boundary_condition = [](int, size_t)
    {
        return SimpleFluid::BoundaryCondition{
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    };
    auto scalar_boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_batches().at(batch_id).face_lids.at(in_batch_id);
        const auto centroid = mesh->face_centroid(face_lid);
        return 1.0 + centroid.x + 2.0 * centroid.y - centroid.z;
    };
    auto zero_source = [](MeshType::local_ordinal_type)
    {
        return Pack::scalar_type{};
    };
    auto assemble_scalar =
        [&](SimpleFluid::FVM::NonOrthogonalTreatment treatment,
            const FieldType* correction_field)
    {
        return SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            scalar, zero_fluxes, 1.0, unit_weight, unit_weight, diffusivity,
            boundary_condition, scalar_boundary_value, zero_source,
            treatment, correction_field, Teuchos::null, {}, {}, &cache);
    };
    const auto scalar_base = assemble_scalar(
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr);
    const auto scalar_corrected = assemble_scalar(
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, &scalar);
    Pack::vector_type expected_scalar(mesh->owned_cell_map(), true);
    SimpleFluid::FVM::add_variable_explicit_non_orthogonal_correction<Pack>(
        scalar, diffusivity, boundary_condition, scalar_boundary_value,
        expected_scalar, 1.0, boundary_coefficient);

    Pack::scalar_type scalar_correction_norm{};
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto actual = scalar_corrected.rhs->getData()[owned]
                          - scalar_base.rhs->getData()[owned];
        const auto expected = expected_scalar.getData()[owned];
        EXPECT_NEAR(actual, expected, 1.0e-11);
        scalar_correction_norm += std::abs(expected);
    }
    EXPECT_GT(scalar_correction_norm, 1.0e-8);

    const auto scalar_implicit = assemble_scalar(
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr);
    const auto scalar_hybrid = assemble_scalar(
        SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid, &scalar);
    const auto scalar_explicit_action =
        local_matrix_action(*scalar_corrected.matrix, scalar);
    const auto scalar_implicit_action =
        local_matrix_action(*scalar_implicit.matrix, scalar);
    const auto scalar_hybrid_action =
        local_matrix_action(*scalar_hybrid.matrix, scalar);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto explicit_residual = scalar_explicit_action[owned]
            - scalar_corrected.rhs->getData()[owned];
        const auto implicit_residual = scalar_implicit_action[owned]
            - scalar_implicit.rhs->getData()[owned];
        const auto hybrid_residual = scalar_hybrid_action[owned]
            - scalar_hybrid.rhs->getData()[owned];
        EXPECT_NEAR(implicit_residual, explicit_residual, 1.0e-11);
        EXPECT_NEAR(hybrid_residual, explicit_residual, 1.0e-11);
    }

    auto vector_boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_batches().at(batch_id).face_lids.at(in_batch_id);
        const auto centroid = mesh->face_centroid(face_lid);
        return MeshType::Vec3{
            1.0 + centroid.x, 2.0 * centroid.y, 0.5 - centroid.z};
    };
    auto zero_acceleration = [](MeshType::local_ordinal_type)
    {
        return MeshType::Vec3{};
    };
    constexpr Pack::scalar_type reference_density = 2.0;
    auto assemble_vector =
        [&](SimpleFluid::FVM::NonOrthogonalTreatment treatment,
            const VectorFieldType* correction_field)
    {
        return SimpleFluid::FVM::physical_momentum_transport_system<Pack>(
            velocity, zero_fluxes, 1.0, diffusivity, reference_density,
            vector_boundary_value, zero_acceleration, treatment,
            correction_field, Teuchos::null,
            SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{}, &cache);
    };
    const auto vector_base = assemble_vector(
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr);
    const auto vector_corrected = assemble_vector(
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, &velocity);
    Pack::multi_vector_type expected_vector(
        mesh->owned_cell_map(), 3, true);
    SimpleFluid::FVM::add_variable_explicit_non_orthogonal_correction<Pack>(
        velocity, diffusivity, vector_boundary_value, expected_vector,
        1.0 / reference_density,
        SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{},
        boundary_coefficient);

    Pack::scalar_type vector_correction_norm{};
    for (size_t component = 0; component < 3; ++component)
    {
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto actual =
                vector_corrected.rhs->getData(component)[owned]
              - vector_base.rhs->getData(component)[owned];
            const auto expected = expected_vector.getData(component)[owned];
            EXPECT_NEAR(actual, expected, 1.0e-11);
            vector_correction_norm += std::abs(expected);
        }
    }
    EXPECT_GT(vector_correction_norm, 1.0e-8);

    const auto vector_implicit = assemble_vector(
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, nullptr);
    const auto vector_hybrid = assemble_vector(
        SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid, &velocity);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto row =
            static_cast<MeshType::local_ordinal_type>(owned);
        for (size_t component = 0; component < 3; ++component)
        {
            auto matrix_action =
                [&](const Pack::matrix_type& matrix)
            {
                const auto entries = matrix.getNumEntriesInLocalRow(row);
                typename Pack::matrix_type::nonconst_local_inds_host_view_type
                    local_columns("local_columns", entries);
                typename Pack::matrix_type::nonconst_values_host_view_type
                    local_values("local_values", entries);
                size_t local_entries = 0;
                matrix.getLocalRowCopy(
                    row, local_columns, local_values, local_entries);
                Pack::scalar_type result{};
                for (size_t entry = 0; entry < local_entries; ++entry)
                {
                    result += local_values(entry)
                        * velocity.local_value(local_columns(entry))
                              .component(component);
                }
                return result;
            };
            const auto explicit_residual =
                matrix_action(*vector_corrected.matrix)
              - vector_corrected.rhs->getData(component)[owned];
            const auto implicit_residual =
                matrix_action(*vector_implicit.matrix)
              - vector_implicit.rhs->getData(component)[owned];
            const auto hybrid_residual =
                matrix_action(*vector_hybrid.matrix)
              - vector_hybrid.rhs->getData(component)[owned];
            EXPECT_NEAR(implicit_residual, explicit_residual, 1.0e-11);
            EXPECT_NEAR(hybrid_residual, explicit_residual, 1.0e-11);
        }
    }
}

/** @brief Verifies physical temperature transport treats Neumann data as flux. */
TEST(FvmOperatorsTest, PhysicalTemperatureTransportTreatsNeumannBoundaryAsFlux)
{
    auto mesh = make_mesh();
    FieldType temperature(mesh, 0.0, "temperature");
    FieldType density(mesh, 1.0, "density");
    FieldType heat_capacity(mesh, 1.0, "heat_capacity");
    FieldType conductivity(mesh, 2.0, "conductivity");
    temperature.sync_ghosts();
    conductivity.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "face_flux");

    const int neumann_batch = mesh->boundary_batches().begin()->first;
    constexpr double gradient = 3.0;
    auto boundary_condition =
        [&](int batch_id, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {
            batch_id == neumann_batch
                ? SimpleFluid::BoundaryConditionType::Neumann
                : SimpleFluid::BoundaryConditionType::Dirichlet,
            batch_id == neumann_batch ? gradient : 0.0};
    };
    auto boundary_value =
        [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto zero_source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    const auto system =
        SimpleFluid::FVM::physical_temperature_transport_system<Pack>(
            temperature,
            zero_fluxes,
            1.0,
            density,
            heat_capacity,
            conductivity,
            boundary_condition,
            boundary_value,
            zero_source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);

    std::vector<double> expected_rhs(mesh->num_owned_cells(), 0.0);
    for (const auto face_lid :
         mesh->boundary_batches().at(neumann_batch).face_lids)
    {
        if (!mesh->is_owned_face(face_lid)
            || !mesh->is_boundary_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        expected_rhs[static_cast<size_t>(owner)] +=
            conductivity.value(owner) * gradient * mesh->face_area(face_lid);
    }

    const auto rhs_data = system.rhs->getData();
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        EXPECT_NEAR(
            rhs_data[cell_lid],
            expected_rhs[static_cast<size_t>(cell_lid)],
            1.0e-12);
    }
}

/** @brief Verifies physical transport harmonically interpolates material coefficients. */
TEST(FvmOperatorsTest, PhysicalTransportUsesHarmonicMaterialCoefficients)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    FieldType temperature(mesh, 0.0, "temperature");
    FieldType density(mesh, 1.0, "density");
    FieldType heat_capacity(mesh, 1.0, "heat_capacity");
    FieldType conductivity(mesh, "conductivity");
    conductivity.set_value(0, 1.0);
    conductivity.set_value(1, 4.0);
    conductivity.sync_ghosts();
    SimpleFluid::FaceField<Pack> zero_fluxes(
        mesh, 0.0, "face_flux");

    auto boundary_condition =
        [](int, size_t)
    {
        return SimpleFluid::BoundaryCondition{};
    };
    auto boundary_value =
        [](int, size_t)
    {
        return 0.0;
    };
    auto zero_source =
        [](MeshType::local_ordinal_type)
    {
        return 0.0;
    };

    for (const auto treatment : {
             SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
             SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
             SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid})
    {
        const auto system =
            SimpleFluid::FVM::physical_temperature_transport_system<Pack>(
                temperature,
                zero_fluxes,
                1.0,
                density,
                heat_capacity,
                conductivity,
                boundary_condition,
                boundary_value,
                zero_source,
                treatment,
                treatment
                        == SimpleFluid::FVM::NonOrthogonalTreatment::Implicit
                    ? nullptr
                    : &temperature);
        EXPECT_NEAR(
            local_matrix_entry(*system.matrix, 0, 1),
            -1.6,
            1.0e-12);
        EXPECT_NEAR(
            local_matrix_entry(*system.matrix, 1, 0),
            -1.6,
            1.0e-12);
    }

    VectorFieldType velocity(mesh, "velocity");
    FieldType dynamic_viscosity(mesh, "dynamic_viscosity");
    dynamic_viscosity.set_value(0, 2.0);
    dynamic_viscosity.set_value(1, 8.0);
    dynamic_viscosity.sync_ghosts();
    auto velocity_boundary =
        [](int, MeshType::local_ordinal_type)
    {
        return SimpleFluid::vec3<>{};
    };
    auto zero_acceleration =
        [](MeshType::local_ordinal_type)
    {
        return SimpleFluid::vec3<>{};
    };
    for (const auto treatment : {
             SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
             SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
             SimpleFluid::FVM::NonOrthogonalTreatment::Hybrid})
    {
        const auto momentum =
            SimpleFluid::FVM::physical_momentum_transport_system<Pack>(
                velocity,
                zero_fluxes,
                1.0,
                dynamic_viscosity,
                2.0,
                velocity_boundary,
                zero_acceleration,
                treatment,
                treatment
                        == SimpleFluid::FVM::NonOrthogonalTreatment::Implicit
                    ? nullptr
                    : &velocity);
        EXPECT_NEAR(
            local_matrix_entry(*momentum.matrix, 0, 1),
            -1.6,
            1.0e-12);
    }
}

/** @brief Verifies physical transport consumes sparse boundary coefficient caches. */
TEST(FvmOperatorsTest,
     PhysicalTransportUsesSparseBoundaryCoefficientCaches)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(1, 1, 1, 1.0));
    FieldType temperature(mesh, 0.0, "temperature");
    FieldType density(mesh, 1.0, "density");
    FieldType heat_capacity(mesh, 1.0, "heat_capacity");
    FieldType conductivity(mesh, 2.0, "conductivity");
    FieldType dynamic_viscosity(mesh, 2.0, "dynamic_viscosity");
    VectorFieldType velocity(mesh, "velocity");
    SimpleFluid::FaceField<Pack> zero_fluxes(
        mesh, 0.0, "zero_fluxes");

    const auto& [batch_id, batch] = *mesh->boundary_batches().begin();
    constexpr Pack::scalar_type cached_coefficient = 6.0;
    SimpleFluid::FVM::BoundaryCache<Pack> cache{{}, mesh};
    cache.value[batch_id] = SimpleFluid::Arr<Pack::scalar_type>(
        batch.face_lids.size(), cached_coefficient);

    auto boundary_condition = [](int, size_t)
    {
        return SimpleFluid::BoundaryCondition{
            SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    };
    auto scalar_boundary_value = [](int, size_t)
    {
        return Pack::scalar_type{};
    };
    auto zero_source = [](MeshType::local_ordinal_type)
    {
        return Pack::scalar_type{};
    };
    auto assemble_temperature =
        [&](const SimpleFluid::FVM::BoundaryCache<Pack>* boundary_cache)
    {
        return SimpleFluid::FVM::physical_temperature_transport_system<Pack>(
            temperature,
            zero_fluxes,
            1.0,
            density,
            heat_capacity,
            conductivity,
            boundary_condition,
            scalar_boundary_value,
            zero_source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            nullptr,
            Teuchos::null,
            boundary_cache);
    };

    const auto temperature_without_cache = assemble_temperature(nullptr);
    const auto temperature_with_cache = assemble_temperature(&cache);
    Pack::scalar_type expected_temperature_delta{};
    for (const auto face_lid : batch.face_lids)
    {
        if (!mesh->is_boundary_face(face_lid)
            || !mesh->is_owned_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        expected_temperature_delta +=
            SimpleFluid::FVM::detail::boundary_diffusion_coefficient(
                *mesh,
                face_lid,
                owner,
                cached_coefficient - conductivity.value(owner));
    }
    EXPECT_NEAR(
        local_matrix_entry(*temperature_with_cache.matrix, 0, 0)
      - local_matrix_entry(*temperature_without_cache.matrix, 0, 0),
        expected_temperature_delta,
        1.0e-12);

    auto vector_boundary_value = [](int, size_t)
    {
        return MeshType::Vec3{};
    };
    auto zero_acceleration = [](MeshType::local_ordinal_type)
    {
        return MeshType::Vec3{};
    };
    auto assemble_momentum =
        [&](const SimpleFluid::FVM::BoundaryCache<Pack>* boundary_cache)
    {
        return SimpleFluid::FVM::physical_momentum_transport_system<Pack>(
            velocity,
            zero_fluxes,
            1.0,
            dynamic_viscosity,
            2.0,
            vector_boundary_value,
            zero_acceleration,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            nullptr,
            Teuchos::null,
            SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{},
            boundary_cache);
    };
    const auto momentum_without_cache = assemble_momentum(nullptr);
    const auto momentum_with_cache = assemble_momentum(&cache);
    EXPECT_NEAR(
        local_matrix_entry(*momentum_with_cache.matrix, 0, 0)
      - local_matrix_entry(*momentum_without_cache.matrix, 0, 0),
        expected_temperature_delta / 2.0,
        1.0e-12);

    auto invalid_cache = cache;
    invalid_cache.value[batch_id][0] = -1.0;
    EXPECT_THROW(assemble_temperature(&invalid_cache), std::invalid_argument);

    auto other_mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(1, 1, 1, 1.0));
    invalid_cache.value.clear();
    invalid_cache.mesh = other_mesh;
    EXPECT_THROW(assemble_momentum(&invalid_cache), std::invalid_argument);
}

/** @brief Verifies physical momentum adds lagged variable-viscosity transpose stress. */
TEST(FvmOperatorsTest,
     PhysicalMomentumAddsLaggedVariableViscosityTransposeStress)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(2, 2, 1, 0.5));
    VectorFieldType velocity(mesh, "shear_velocity");
    FieldType variable_viscosity(mesh, "variable_viscosity");
    FieldType constant_viscosity(mesh, 2.0, "constant_viscosity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        // Divergence-free simple shear: grad(u_y) = e_x.  A viscosity
        // gradient in y therefore drives the x component only through
        // div(mu * grad(U)^T).
        velocity.set_owned_value(cell_lid, {0.0, center.x, 0.0});
        variable_viscosity.set_owned_value(
            cell_lid, center.y < 0.5 ? 1.0 : 3.0);
    }
    velocity.sync_ghosts();
    variable_viscosity.sync_ghosts();
    constant_viscosity.sync_ghosts();

    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return MeshType::Vec3{0.0, mesh->face_centroid(face_lid).x, 0.0};
    };
    auto zero_acceleration = [](MeshType::local_ordinal_type)
    {
        return MeshType::Vec3{};
    };
    SimpleFluid::FaceField<Pack> zero_fluxes(
        mesh, 0.0, "zero_fluxes");
    constexpr Pack::scalar_type reference_density = 2.0;
    auto assemble = [&](const FieldType& viscosity)
    {
        return SimpleFluid::FVM::physical_momentum_transport_system<Pack>(
            velocity,
            zero_fluxes,
            1.0,
            viscosity,
            reference_density,
            boundary_value,
            zero_acceleration,
            SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    };

    const auto variable = assemble(variable_viscosity);
    const auto constant = assemble(constant_viscosity);
    const auto variable_x = variable.rhs->getData(0);
    const auto constant_x = constant.rhs->getData(0);

    double maximum_transpose_stress = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        Pack::scalar_type expected_integrated_stress = 0.0;
        for (const auto face_lid : mesh->faces(cell_lid))
        {
            auto face_viscosity =
                variable_viscosity.local_value(cell_lid);
            if (mesh->is_interior_face(face_lid))
            {
                const auto other =
                    mesh->opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                face_viscosity =
                    SimpleFluid::FVM::detail::harmonic_face_value(
                        *mesh,
                        face_lid,
                        cell_lid,
                        other,
                        variable_viscosity);
            }
            expected_integrated_stress +=
                face_viscosity
              * mesh->face_area_vector_outward(face_lid, cell_lid).y
              / reference_density;
        }

        EXPECT_NEAR(
            variable_x[cell_lid], expected_integrated_stress, 1.0e-12);
        EXPECT_NEAR(constant_x[cell_lid], 0.0, 1.0e-12);
        maximum_transpose_stress = std::max(
            maximum_transpose_stress,
            std::abs(variable_x[cell_lid]));
    }
    EXPECT_GT(maximum_transpose_stress, 1.0e-6);
}

/** @brief Verifies explicit transpose stress removes two-thirds of the divergence trace. */
TEST(FvmOperatorsTest,
     ExplicitTransposeStressRemovesTwoThirdsDivergenceTrace)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    VectorFieldType velocity(mesh, "expansion_velocity");
    FieldType viscosity(mesh, "variable_viscosity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        velocity.set_owned_value(
            cell_lid,
            {mesh->cell_centroid(cell_lid).x, 0.0, 0.0});
        viscosity.set_owned_value(cell_lid, owned == 0 ? 1.0 : 3.0);
    }
    velocity.sync_ghosts();
    viscosity.sync_ghosts();
    auto boundary_value = [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            mesh->boundary_face_batch(batch_id).face_lids[in_batch_id];
        return MeshType::Vec3{
            mesh->face_centroid(face_lid).x, 0.0, 0.0};
    };
    Pack::multi_vector_type rhs(mesh->owned_cell_map(), 3, true);

    SimpleFluid::FVM::add_explicit_deviatoric_transpose_gradient_stress<Pack>(
        velocity, viscosity, 1.0, boundary_value, rhs);

    // grad(U)=diag(1,0,0), so the explicit tensor is
    // diag(1/3,-2/3,-2/3). Harmonic interior viscosity is 1.5.
    EXPECT_NEAR(rhs.getData(0)[0], 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(rhs.getData(0)[1], 0.5, 1.0e-12);
    EXPECT_NEAR(rhs.getData(1)[0], 0.0, 1.0e-12);
    EXPECT_NEAR(rhs.getData(1)[1], 0.0, 1.0e-12);
    EXPECT_NEAR(rhs.getData(2)[0], 0.0, 1.0e-12);
    EXPECT_NEAR(rhs.getData(2)[1], 0.0, 1.0e-12);
}
