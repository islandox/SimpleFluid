/**
 * @file testTurbulenceWallTreatment.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Formula and aggregation tests for policy-based turbulence walls.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "FVM/OperatorDetails.hh"
#include "equations/turbulence/TurbulenceWallTreatment.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::Mesh<Pack>;
using Field = SimpleFluid::CellField<Pack>;
using Velocity = SimpleFluid::VectorCellField<Pack>;
using Material = SimpleFluid::MaterialPropertyFields<Pack>;
using Type = SimpleFluid::BoundaryConditionType;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::BoundaryConditionSet wall_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.velocity["xmin"] = {Type::NoSlip, {}};
    boundaries.velocity["ymin"] = {Type::NoSlip, {}};
    return boundaries;
}

Material make_material(const SimpleFluid::SP<const Mesh>& mesh,
                       double viscosity = 1.0e-6,
                       double conductivity = 0.6)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1.0;
    options.density = 1.0;
    options.specific_heat_capacity = 1000.0;
    options.dynamic_viscosity = viscosity;
    options.thermal_conductivity = conductivity;
    return Material(mesh, options, time_options);
}

int boundary_id(const Mesh& mesh, const std::string& name)
{
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        (void)batch;
        if (mesh.boundary_batch_name(batch_id) == name)
            return batch_id;
    }
    throw std::out_of_range("missing test boundary");
}

SimpleFluid::SP<Mesh> rectangular_single_cell_mesh()
{
    auto database = SimpleFluid::test::make_single_hex_database();
    auto mutable_database = std::make_shared<SimpleFluid::Database>(*database);
    mutable_database->set("X", SimpleFluid::ArrReal{0.0, 2.0});
    mutable_database->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    mutable_database->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    return SimpleFluid::test::build_mesh<Pack>(mutable_database);
}

} // namespace

/** @brief Verifies Menter face values and molecular transport for resolved SST walls. */
TEST(TurbulenceWallTreatmentTest, ResolvedSSTUsesMenterFaceValueAndMolecularWallTransport)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(1, 1, 1, 2.0e-4));
    auto boundaries = wall_boundaries();
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        mesh, boundaries);
    auto material = make_material(mesh);
    Field k(mesh, 0.2, "k");
    Velocity velocity(mesh, "velocity");
    velocity.set_owned_value(0, {0.0, 1.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::TurbulenceWallTreatmentOptions options;
    options.boundary_names = {"xmin"};
    SimpleFluid::ResolvedLowReSSTWallTreatment<Pack> treatment(
        mesh, options, boundaries.velocity);
    const auto evaluation = treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);

    const auto id = boundary_id(*mesh, "xmin");
    const auto& face = evaluation.face(id, 0);
    EXPECT_EQ(face.turbulent_kinetic_energy.type, Type::Dirichlet);
    EXPECT_DOUBLE_EQ(face.turbulent_kinetic_energy.value, 0.0);
    EXPECT_EQ(face.secondary.type, Type::Dirichlet);
    EXPECT_NEAR(face.wall_distance, 1.0e-4, 1.0e-14);
    EXPECT_NEAR(face.secondary.value, 80000.0, 1.0e-8);
    EXPECT_DOUBLE_EQ(face.turbulent_kinematic_viscosity, 0.0);
    EXPECT_NEAR(face.y_plus, 10.0, 1.0e-12);
    EXPECT_NEAR(face.effective_dynamic_viscosity, 1.0e-6, 1.0e-18);
    EXPECT_NEAR(face.effective_thermal_conductivity, 0.6, 1.0e-14);
    EXPECT_FALSE(evaluation.secondary_constraint(0).has_value());
    EXPECT_FALSE(evaluation.production_override(0).has_value());
    ASSERT_TRUE(evaluation.cell_y_plus(0).has_value());
    EXPECT_NEAR(*evaluation.cell_y_plus(0), 10.0, 1.0e-12);
    EXPECT_NEAR(evaluation.boundary_dynamic_viscosity().value.at(id).at(0),
                1.0e-6, 1.0e-18);
    EXPECT_NEAR(evaluation.boundary_scalar_diffusivity().value.at(id).at(0),
                1.0e-6, 1.0e-18);
}

/** @brief Verifies resolved SST treatment with skewed STK wall geometry. */
TEST(TurbulenceWallTreatmentTest, ResolvedSSTUsesSkewedSTKWallGeometry)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    auto boundaries = wall_boundaries();
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        mesh, boundaries);
    constexpr double nu = 1.0e-5;
    auto material = make_material(mesh, nu);
    Field k(mesh, 0.2, "k");
    Velocity velocity(mesh, "velocity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto centroid = mesh->cell_centroid(cell);
        velocity.set_owned_value(
            cell, {0.2 + centroid.y, 1.0 + centroid.z, 0.3});
    }
    velocity.sync_ghosts();

    SimpleFluid::TurbulenceWallTreatmentOptions options;
    options.boundary_names = {"xmin"};
    SimpleFluid::ResolvedLowReSSTWallTreatment<Pack> treatment(
        mesh, options, boundaries.velocity);
    const auto evaluation = treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);

    const auto id = boundary_id(*mesh, "xmin");
    const auto& batch = mesh->boundary_batches().at(id);
    ASSERT_FALSE(batch.face_lids.empty());
    for (size_t in_batch = 0; in_batch < batch.face_lids.size(); ++in_batch)
    {
        const auto face_lid = batch.face_lids[in_batch];
        const auto owner = mesh->owner_cell(face_lid);
        const auto distance = SimpleFluid::FVM::detail::boundary_normal_distance(
            *mesh, face_lid, owner);
        const auto normal = mesh->face_normal_outward(face_lid, owner);
        const auto owner_velocity = velocity.local_value(owner);
        const auto tangential_velocity =
            owner_velocity - normal * owner_velocity.dot(normal);
        const auto expected_y_plus = distance *
            std::sqrt(nu * tangential_velocity.norm() / distance) / nu;
        const auto expected_omega = options.sst_omega_wall_coefficient * nu /
            (options.sst_beta_1 * distance * distance);

        ASSERT_TRUE(evaluation.contains_face(id, in_batch));
        const auto& face = evaluation.face(id, in_batch);
        EXPECT_NEAR(face.wall_distance, distance, 1.0e-13);
        EXPECT_NEAR(face.secondary.value, expected_omega,
                    1.0e-12 * expected_omega);
        EXPECT_NEAR(face.y_plus, expected_y_plus,
                    1.0e-12 * std::max(expected_y_plus, 1.0));
        EXPECT_DOUBLE_EQ(face.turbulent_kinematic_viscosity, 0.0);
    }
}

/** @brief Verifies high-Re k-epsilon wall values against OpenFOAM log-layer formulas. */
TEST(TurbulenceWallTreatmentTest, HighReKEpsilonMatchesOpenFOAMLogLayerFormulas)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
    auto boundaries = wall_boundaries();
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        mesh, boundaries);
    constexpr double nu = 1.0e-5;
    constexpr double k_value = 0.375;
    auto material = make_material(mesh, nu);
    Field k(mesh, k_value, "k");
    Velocity velocity(mesh, "velocity");
    velocity.set_owned_value(0, {10.0, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::TurbulenceWallTreatmentOptions options;
    options.boundary_names = {"ymin"};
    SimpleFluid::StandardHighReKEpsilonWallTreatment<Pack> treatment(
        mesh, options, boundaries.velocity);
    const auto evaluation = treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);

    const auto id = boundary_id(*mesh, "ymin");
    const auto& face = evaluation.face(id, 0);
    const double y = 0.5;
    const double c_mu_quarter = std::pow(options.c_mu, 0.25);
    const double c_mu_three_quarters = std::pow(options.c_mu, 0.75);
    const double sqrt_k = std::sqrt(k_value);
    const double y_plus = c_mu_quarter * y * sqrt_k / nu;
    const double nut = nu *
        (y_plus * options.kappa / std::log(options.log_layer_e * y_plus) - 1.0);
    const double epsilon = c_mu_three_quarters * k_value * sqrt_k /
        (options.kappa * y);
    const double production = (nu + nut) * (10.0 / y) *
        c_mu_quarter * sqrt_k / (options.kappa * y);

    EXPECT_EQ(face.turbulent_kinetic_energy.type, Type::Neumann);
    EXPECT_EQ(face.secondary.type, Type::Neumann);
    EXPECT_NEAR(face.y_plus, y_plus, 1.0e-10);
    EXPECT_NEAR(face.turbulent_kinematic_viscosity, nut, 1.0e-14);
    ASSERT_TRUE(evaluation.secondary_constraint(0).has_value());
    ASSERT_TRUE(evaluation.production_override(0).has_value());
    EXPECT_NEAR(*evaluation.secondary_constraint(0), epsilon, 1.0e-14);
    EXPECT_NEAR(*evaluation.production_override(0), production, 1.0e-12);
}

/** @brief Verifies wall-normal motion does not create high-Re wall shear. */
TEST(TurbulenceWallTreatmentTest,
     HighReKEpsilonIgnoresWallNormalRelativeVelocity)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
    auto boundaries = wall_boundaries();
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        mesh, boundaries);
    constexpr double nu = 1.0e-3;
    auto material = make_material(mesh, nu);
    Field k(mesh, 1.0e-4, "k");
    Velocity velocity(mesh, "velocity");
    velocity.set_owned_value(0, {0.0, 1.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::TurbulenceWallTreatmentOptions options;
    options.boundary_names = {"ymin"};
    SimpleFluid::StandardHighReKEpsilonWallTreatment<Pack> treatment(
        mesh, options, boundaries.velocity);
    const auto evaluation = treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);

    const auto id = boundary_id(*mesh, "ymin");
    EXPECT_DOUBLE_EQ(evaluation.face(id, 0).y_plus, 0.0);
    ASSERT_TRUE(evaluation.production_override(0).has_value());
    EXPECT_DOUBLE_EQ(*evaluation.production_override(0), 0.0);
}

/** @brief Verifies OpenFOAM v2606 high-Re defaults and the optional low-Re correction. */
TEST(TurbulenceWallTreatmentTest,
     HighReDefaultsToOpenFOAMV2606AndOffersLowReCorrection)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
    auto boundaries = wall_boundaries();
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        mesh, boundaries);
    constexpr double nu = 1.0e-3;
    auto material = make_material(mesh, nu);
    Velocity velocity(mesh, "velocity");
    velocity.set_owned_value(0, {0.01, 0.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::TurbulenceWallTreatmentOptions options;
    options.boundary_names = {"ymin"};
    SimpleFluid::StandardHighReKEpsilonWallTreatment<Pack> treatment(
        mesh, options, boundaries.velocity);
    EXPECT_NEAR(treatment.y_plus_lam(), 11.530107304327, 1.0e-11);

    const double y = 0.5;
    const double c_mu_quarter = std::pow(options.c_mu, 0.25);
    const double below = 0.99 * treatment.y_plus_lam();
    const double k_below = std::pow(below * nu / (c_mu_quarter * y), 2);
    Field k(mesh, k_below, "k");
    const auto high_re = treatment.evaluate(k, velocity, cache, material, 1.0, 0.9);
    const auto id = boundary_id(*mesh, "ymin");
    EXPECT_DOUBLE_EQ(high_re.face(id, 0).turbulent_kinematic_viscosity, 0.0);
    EXPECT_GT(*high_re.production_override(0), 0.0);
    EXPECT_NEAR(*high_re.secondary_constraint(0),
                std::pow(options.c_mu, 0.75) * k_below *
                    std::sqrt(k_below) / (options.kappa * y),
                1.0e-14);
    const double mag_grad_u = 0.01 / y;
    EXPECT_NEAR(high_re.face(id, 0).y_plus,
                y * std::sqrt(nu * mag_grad_u) / nu, 1.0e-13);

    auto corrected_options = options;
    corrected_options.epsilon_low_re_correction = true;
    SimpleFluid::StandardHighReKEpsilonWallTreatment<Pack> corrected_treatment(
        mesh, corrected_options, boundaries.velocity);
    const auto viscous = corrected_treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);
    EXPECT_DOUBLE_EQ(*viscous.production_override(0), 0.0);
    EXPECT_NEAR(*viscous.secondary_constraint(0),
                2.0 * nu * k_below / (y * y), 1.0e-14);

    const double k_equal = std::pow(
        corrected_treatment.y_plus_lam() * nu / (c_mu_quarter * y), 2);
    k.put_scalar(k_equal);
    k.sync_ghosts();
    const auto at_switch = corrected_treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);
    EXPECT_NEAR(at_switch.face(id, 0).y_plus,
                corrected_treatment.y_plus_lam(), 1.0e-12);
    EXPECT_DOUBLE_EQ(at_switch.face(id, 0).turbulent_kinematic_viscosity, 0.0);
    EXPECT_DOUBLE_EQ(*at_switch.production_override(0), 0.0);
    EXPECT_NEAR(*at_switch.secondary_constraint(0),
                std::pow(options.c_mu, 0.75) * k_equal *
                    std::sqrt(k_equal) / (options.kappa * y),
                1.0e-14);

    k.put_scalar(k_below * std::pow(1.02 / 0.99, 2));
    k.sync_ghosts();
    const auto logarithmic = corrected_treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);
    EXPECT_GT(logarithmic.face(id, 0).turbulent_kinematic_viscosity, 0.0);
    EXPECT_GT(*logarithmic.production_override(0), 0.0);
}

/** @brief Verifies equal face-count averaging for high-Re corner cells. */
TEST(TurbulenceWallTreatmentTest, HighReCornerCellsUseEqualFaceCountAveraging)
{
    auto mesh = rectangular_single_cell_mesh();
    auto boundaries = wall_boundaries();
    const auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        mesh, boundaries);
    auto material = make_material(mesh, 1.0e-2);
    Field k(mesh, 4.0, "k");
    Velocity velocity(mesh, "velocity");
    velocity.set_owned_value(0, {3.0, 4.0, 0.0});
    velocity.sync_ghosts();

    SimpleFluid::TurbulenceWallTreatmentOptions options;
    options.boundary_names = {"xmin", "ymin"};
    SimpleFluid::StandardHighReKEpsilonWallTreatment<Pack> treatment(
        mesh, options, boundaries.velocity);
    const auto evaluation = treatment.evaluate(
        k, velocity, cache, material, 1.0, 0.9);

    const auto xmin = boundary_id(*mesh, "xmin");
    const auto ymin = boundary_id(*mesh, "ymin");
    const auto& xface = evaluation.face(xmin, 0);
    const auto& yface = evaluation.face(ymin, 0);
    ASSERT_TRUE(xface.secondary_constraint.has_value());
    ASSERT_TRUE(yface.secondary_constraint.has_value());
    ASSERT_TRUE(xface.production_override.has_value());
    ASSERT_TRUE(yface.production_override.has_value());
    EXPECT_NEAR(*evaluation.secondary_constraint(0),
                0.5 * (*xface.secondary_constraint + *yface.secondary_constraint),
                1.0e-13);
    EXPECT_NEAR(*evaluation.production_override(0),
                0.5 * (*xface.production_override + *yface.production_override),
                1.0e-12);
    EXPECT_DOUBLE_EQ(*evaluation.cell_y_plus(0),
                     std::max(xface.y_plus, yface.y_plus));
}

/** @brief Verifies rejection of missing, duplicate, and non-wall patches. */
TEST(TurbulenceWallTreatmentTest, RejectsMissingDuplicateAndNonWallPatches)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
    auto boundaries = wall_boundaries();

    SimpleFluid::TurbulenceWallTreatmentOptions empty;
    EXPECT_THROW((SimpleFluid::ResolvedLowReSSTWallTreatment<Pack>(
                     mesh, empty, boundaries.velocity)),
                 std::invalid_argument);

    auto duplicate = empty;
    duplicate.boundary_names = {"xmin", "xmin"};
    EXPECT_THROW((SimpleFluid::ResolvedLowReSSTWallTreatment<Pack>(
                     mesh, duplicate, boundaries.velocity)),
                 std::invalid_argument);

    auto missing = empty;
    missing.boundary_names = {"notAPatch"};
    boundaries.velocity["notAPatch"] = {Type::NoSlip, {}};
    EXPECT_THROW((SimpleFluid::ResolvedLowReSSTWallTreatment<Pack>(
                     mesh, missing, boundaries.velocity)),
                 std::invalid_argument);

    auto non_wall = empty;
    non_wall.boundary_names = {"xmax"};
    boundaries.velocity["xmax"] = {Type::Slip, {}};
    EXPECT_THROW((SimpleFluid::ResolvedLowReSSTWallTreatment<Pack>(
                     mesh, non_wall, boundaries.velocity)),
                 std::invalid_argument);
}
