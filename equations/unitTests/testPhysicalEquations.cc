/**
 * @file testPhysicalEquations.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for separated physical equation classes.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "fields/VectorCellField.hh"
#include "geometry/MeshFactory.hh"
#include "FVM/FvmOperators.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_hex_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

SimpleFluid::SP<MeshType> make_2x2x2_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

std::vector<Pack::scalar_type> local_values(const FieldType& field)
{
    std::vector<Pack::scalar_type> values(field.num_local_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(field.num_local_cells());
         ++lid)
    {
        values[static_cast<std::size_t>(lid)] = field.local_value(lid);
    }

    return values;
}

std::vector<MeshType::Vec3> local_vector_values(const VectorFieldType& field)
{
    std::vector<MeshType::Vec3> values(field.num_local_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(field.num_local_cells());
         ++lid)
    {
        values[static_cast<std::size_t>(lid)] = field.local_value(lid);
    }

    return values;
}

} // namespace

TEST(PhysicalEquationsTest, TemperatureDiffusionAppliesDirichletBoundary)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 0.0);
    temperature.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    equation.advance_explicit(local_values(temperature), 0.1, 1.0, temperature);

    EXPECT_NEAR(temperature.value(0), 0.2, 1.0e-12);
}

TEST(PhysicalEquationsTest, BoussinesqMomentumAdvancesAllVelocityComponents)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    VectorFieldType velocity(mesh, "velocity");

    temperature.set_value(0, 1.0);
    velocity.set_value(0, {});
    temperature.sync_ghosts();
    velocity.sync_ghosts();

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.0;
    options.thermal_expansion = 2.0;
    options.gravity_x = -10.0;
    options.gravity_y = -20.0;
    options.gravity_z = -30.0;
    options.reference_temperature = 0.5;

    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::BoussinesqMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity(local_vector_values(velocity),
                              std::vector<Pack::scalar_type>(mesh->num_faces(), 0.0),
                              temperature,
                              bcs,
                              options,
                              velocity);

    EXPECT_NEAR(velocity.value(0).x, 1.0, 1.0e-10);
    EXPECT_NEAR(velocity.value(0).y, 2.0, 1.0e-10);
    EXPECT_NEAR(velocity.value(0).z, 3.0, 1.0e-10);
}

/**
 * @brief Runs semi-implicit temperature advection in X, Y, and Z directions and verifies changes.
 */
TEST(PhysicalEquationsTest, TemperatureSemiImplicitAdvectionRunsInEachDirection)
{
    auto mesh = make_2x2x2_mesh();

    for (std::size_t component = 0; component < 3; ++component)
    {
        FieldType temperature(mesh, "temperature");
        VectorFieldType velocity(mesh, "velocity");

        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
             ++lid)
        {
            const auto& center = mesh->cell_centroid(lid);
            const auto value = component == 0 ? center.x
                             : (component == 1 ? center.y : center.z);
            temperature.set_value(lid, value);
            velocity.set_value(lid,
                               {component == 0 ? 1.0 : 0.0,
                                component == 1 ? 1.0 : 0.0,
                                component == 2 ? 1.0 : 0.0});
        }
        temperature.sync_ghosts();
        velocity.sync_ghosts();

        const auto old_temperature = local_values(temperature);
        const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
            *mesh, velocity);

        SimpleFluid::BoundaryConditionSet bcs;
        SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
        equation.advance_semi_implicit(old_temperature, fluxes, 0.1, 0.0,
                                       temperature);

        bool changed = false;
        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
             ++lid)
        {
            EXPECT_TRUE(std::isfinite(temperature.value(lid)));
            changed = changed
                   || std::abs(temperature.value(lid)
                             - old_temperature[static_cast<std::size_t>(lid)])
                      > 1.0e-12;
        }
        EXPECT_TRUE(changed);
    }
}

TEST(PhysicalEquationsTest, PressureProjectionSolvesIdentitySystem)
{
    auto mesh = make_single_hex_mesh();
    FieldType pressure(mesh, "pressure");
    pressure.set_value(0, 7.0);
    pressure.sync_ghosts();

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;

    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh, options);
    equation.solve(pressure);

    EXPECT_NEAR(pressure.value(0), 0.0, 1.0e-12);
}

TEST(PhysicalEquationsTest, PressureProjectionReducesFluxDivergence)
{
    auto mesh = make_2x2x2_mesh();
    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 0.0, 0.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    auto divergence_norm = [&]()
    {
        const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
            *mesh, velocity, bcs);
        const auto divergence =
            SimpleFluid::FvmOperators::cell_divergence_from_fluxes<Pack>(
                *mesh, fluxes);

        Pack::scalar_type norm = 0.0;
        for (const auto value : divergence)
        {
            norm += value * value;
        }
        return std::sqrt(norm);
    };

    const auto before = divergence_norm();

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;

    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh, options);
    equation.project(pressure, 0.1, bcs, velocity);

    const auto after = divergence_norm();
    EXPECT_LT(after, before);
}
