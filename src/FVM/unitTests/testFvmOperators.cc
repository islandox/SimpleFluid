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
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
using TensorFieldType = SimpleFluid::TensorCellField<Pack>;

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

TEST(FvmOperatorsTest, WeightedScalarTransportTreatsNeumannBoundaryAsFlux)
{
    auto mesh = make_mesh();
    FieldType scalar(mesh, 0.0, "scalar");
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
