/**
 * @file testSolidHeatConductionEquation.cc
 * @brief Focused tests for heat conduction on a selected solid subdomain.
 */

#include <gtest/gtest.h>

#include "FVM/details/OperatorDetails.hh"
#include "equations/SolidHeatConductionEquation.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using ParentMesh = SimpleFluid::MeshHandle<Pack>;
using SolidMesh = SimpleFluid::SolidSubdomain<Pack>;
using Field = SimpleFluid::ScalarCellFieldStored<Pack, SolidMesh>;
using Material = SimpleFluid::MaterialPropertyFields<Pack, SolidMesh>;
using Equation = SimpleFluid::SolidHeatConductionEquation<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

template<class Selector>
SimpleFluid::SP<const SolidMesh> make_solid_subdomain(
    const SimpleFluid::SP<const SimpleFluid::Database>& database, Selector&& selector)
{
    auto legacy_mesh = SimpleFluid::test::build_mesh<Pack>(database);
    auto parent_mesh = std::make_shared<ParentMesh>(std::move(legacy_mesh));
    return std::make_shared<SolidMesh>(std::move(parent_mesh), std::forward<Selector>(selector));
}

Material make_material(
    const SimpleFluid::SP<const SolidMesh>& mesh, double density, double heat_capacity, double conductivity)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = density;
    model_options.density = density;
    model_options.specific_heat_capacity = heat_capacity;
    model_options.thermal_conductivity = conductivity;
    return Material(mesh, model_options, time_options);
}

std::vector<double> local_values(const Field& field)
{
    std::vector<double> result(field.num_local_cells());
    for (size_t local = 0; local < result.size(); ++local)
    {
        result[local] = field.local_value(static_cast<Pack::local_ordinal_type>(local));
    }
    return result;
}

} // namespace

/** @brief Volumetric heating uses rho-cp storage without an advective contribution. */
TEST(SolidHeatConductionEquationTest, InsulatedCellAddsVolumetricPower)
{
    auto mesh = make_solid_subdomain(SimpleFluid::test::make_single_hex_database(),
        [](Pack::global_ordinal_type, const SolidMesh::Vec3&) { return true; });
    ASSERT_EQ(mesh->num_owned_cells(), 1U);

    constexpr double initial_temperature = 300.0;
    constexpr double density = 2.0;
    constexpr double heat_capacity = 4.0;
    constexpr double power_density = 16.0;
    constexpr double time_step = 0.5;
    Field temperature(mesh, initial_temperature, "solid_temperature");
    auto material = make_material(mesh, density, heat_capacity, 1.0);
    SimpleFluid::BoundaryConditionSet boundaries;
    Equation equation(mesh, boundaries);

    const auto statistics = equation.advance(temperature, time_step, material, temperature,
        [power_density](Pack::local_ordinal_type) { return power_density; });

    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(
        temperature.value(0), initial_temperature + time_step * power_density / (density * heat_capacity), 1.0e-12);
}

/** @brief A face cut by the solid selection is assembled as a thermal boundary. */
TEST(SolidHeatConductionEquationTest, CutInterfaceAppliesDirichletTemperature)
{
    auto mesh = make_solid_subdomain(SimpleFluid::test::make_two_hex_database(),
        [](Pack::global_ordinal_type, const SolidMesh::Vec3& centroid) { return centroid.x < 1.0; });
    ASSERT_EQ(mesh->num_owned_cells(), 1U);

    constexpr double initial_temperature = 300.0;
    constexpr double interface_temperature = 400.0;
    constexpr double density = 2.0;
    constexpr double heat_capacity = 3.0;
    constexpr double conductivity = 5.0;
    constexpr double time_step = 0.25;

    int interface_batch_id = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) == "solid_interface")
        {
            interface_batch_id = batch_id;
            EXPECT_EQ(batch.face_lids.size(), 1U);
            break;
        }
    }
    ASSERT_GE(interface_batch_id, 0);
    const auto interface_face = mesh->boundary_face_batch(interface_batch_id).face_lids.front();
    const auto owner = mesh->owner_cell(interface_face);

    Field temperature(mesh, initial_temperature, "solid_temperature");
    auto material = make_material(mesh, density, heat_capacity, conductivity);
    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.temperature["solid_interface"] = {SimpleFluid::BoundaryConditionType::Dirichlet, interface_temperature};
    Equation equation(mesh, boundaries);

    const auto statistics = equation.advance(temperature, time_step, material, temperature);

    const auto volume = mesh->cell_volume(owner);
    const auto transient = density * heat_capacity * volume / time_step;
    const auto conductance =
        SimpleFluid::FVM::detail::boundary_diffusion_coefficient(*mesh, interface_face, owner, conductivity);
    const auto expected =
        (transient * initial_temperature + conductance * interface_temperature) / (transient + conductance);
    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(temperature.value(owner), expected, 1.0e-10);
}

/** @brief Two radial control volumes solve the transient vessel-wall balance. */
TEST(SolidHeatConductionEquationTest, SolvesCylindricalVesselWall)
{
    using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrical3D;
    auto parent = std::make_shared<ParentMesh>(std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{1.0, 2.0, 3.0}, {0.0, 1.0}, {0.0, 1.0}}}));
    auto mesh = std::make_shared<SolidMesh>(std::move(parent));
    ASSERT_EQ(mesh->num_owned_cells(), 2U);
    ASSERT_TRUE(mesh->interface_faces().empty());

    constexpr double initial_temperature = 300.0;
    constexpr double inner_temperature = 500.0;
    constexpr double outer_temperature = 250.0;
    constexpr double density = 2.0;
    constexpr double heat_capacity = 3.0;
    constexpr double conductivity = 4.0;
    constexpr double time_step = 0.5;

    const auto radius = [&](Pack::local_ordinal_type cell_lid)
    {
        const auto center = mesh->cell_centroid(cell_lid);
        return std::hypot(center.x, center.y);
    };
    const auto inner = radius(0) < radius(1) ? 0 : 1;
    const auto outer = inner == 0 ? 1 : 0;
    auto radial_interface = SolidMesh::invalid_local_id();
    for (const auto face_lid : mesh->faces(inner))
    {
        if (mesh->is_interior_face(face_lid) && mesh->opposite_cell(face_lid, inner) == outer)
        {
            radial_interface = face_lid;
            break;
        }
    }
    ASSERT_NE(radial_interface, SolidMesh::invalid_local_id());

    const auto inner_face = mesh->boundary_face_batch(0).face_lids.front();
    const auto outer_face = mesh->boundary_face_batch(1).face_lids.front();
    ASSERT_EQ(mesh->boundary_batch_name(0), "rmin");
    ASSERT_EQ(mesh->boundary_batch_name(1), "rmax");

    Field temperature(mesh, initial_temperature, "solid_temperature");
    auto material = make_material(mesh, density, heat_capacity, conductivity);
    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.temperature["rmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, inner_temperature};
    boundaries.temperature["rmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, outer_temperature};
    Equation equation(mesh, boundaries);
    const auto statistics = equation.advance(temperature, time_step, material, temperature);

    const auto inner_storage = density * heat_capacity * mesh->cell_volume(inner) / time_step;
    const auto outer_storage = density * heat_capacity * mesh->cell_volume(outer) / time_step;
    const auto inner_conductance =
        SimpleFluid::FVM::detail::boundary_diffusion_coefficient(*mesh, inner_face, inner, conductivity);
    const auto interface_conductance =
        SimpleFluid::FVM::detail::interior_diffusion_coefficient(*mesh, radial_interface, inner, outer, conductivity);
    const auto outer_conductance =
        SimpleFluid::FVM::detail::boundary_diffusion_coefficient(*mesh, outer_face, outer, conductivity);
    const auto a = inner_storage + inner_conductance + interface_conductance;
    const auto b = -interface_conductance;
    const auto d = outer_storage + outer_conductance + interface_conductance;
    const auto inner_rhs = inner_storage * initial_temperature + inner_conductance * inner_temperature;
    const auto outer_rhs = outer_storage * initial_temperature + outer_conductance * outer_temperature;
    const auto determinant = a * d - b * b;
    const auto expected_inner = (d * inner_rhs - b * outer_rhs) / determinant;
    const auto expected_outer = (a * outer_rhs - b * inner_rhs) / determinant;

    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(temperature.value(inner), expected_inner, 1.0e-10);
    EXPECT_NEAR(temperature.value(outer), expected_outer, 1.0e-10);
    EXPECT_GT(temperature.value(inner), temperature.value(outer));
}

/** @brief Transient vessel boundary data can be replaced between advances. */
TEST(SolidHeatConductionEquationTest, UpdatesBoundaryConditionsBetweenSteps)
{
    auto legacy_mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    auto parent = std::make_shared<ParentMesh>(std::move(legacy_mesh));
    auto mesh = std::make_shared<SolidMesh>(std::move(parent));
    Field temperature(mesh, 300.0, "solid_temperature");
    auto material = make_material(mesh, 1.0, 1.0, 1.0);

    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 300.0};
    Equation equation(mesh, boundaries);
    const auto unchanged = equation.advance(temperature, 0.1, material, temperature);
    EXPECT_TRUE(unchanged.converged);
    EXPECT_NEAR(temperature.value(0), 300.0, 1.0e-12);

    boundaries.temperature["xmin"].value = 400.0;
    equation.set_boundary_conditions(boundaries);
    const auto heated = equation.advance(temperature, 0.1, material, temperature);
    EXPECT_TRUE(heated.converged);
    EXPECT_GT(temperature.value(0), 300.0);
}

/** @brief A rejected solve must not overwrite an aliased accepted solid state. */
TEST(SolidHeatConductionEquationTest, RejectedSolvePreservesAcceptedTemperature)
{
    auto mesh = make_solid_subdomain(SimpleFluid::test::make_2x2x2_database(),
        [](Pack::global_ordinal_type, const SolidMesh::Vec3&) { return true; });
    Field temperature(mesh, 0.0, "solid_temperature");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto index = static_cast<double>(owned + 1);
        temperature.set_owned_value(cell_lid, 1.0 + index * index);
    }
    temperature.sync_ghosts();
    const auto accepted = local_values(temperature);

    auto material = make_material(mesh, 1.0, 1.0, 1.0);
    SimpleFluid::BoundaryConditionSet boundaries;
    Equation equation(mesh, boundaries);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1;
    linear_options.tolerance = 1.0e-14;

    EXPECT_THROW(equation.advance(temperature, 1.0, material, temperature,
                     SimpleFluid::FVM::NonOrthogonalTreatment::Implicit, linear_options),
        std::runtime_error);

    for (size_t local = 0; local < accepted.size(); ++local)
    {
        EXPECT_DOUBLE_EQ(temperature.local_value(static_cast<Pack::local_ordinal_type>(local)), accepted[local]);
    }
}
