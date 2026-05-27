/**
 * @file testBoussinesqSolver.cc
 * @brief smoke tests for the Boussinesq solver
 */

#include <gtest/gtest.h>

#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_box_mesh()
{
    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    return factory.template build<Pack>();
}

} // namespace

TEST(BoussinesqSolverTest, RunsHeatedBoxSmokeCase)
{
    auto mesh = make_box_mesh();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 2;
    time_options.thermal_diffusivity = 1.0e-2;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, bcs, time_options, linear_options);
    solver.initialize_heated_box(1.0, 0.0);
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_GT(solver.time(), 0.0);

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_TRUE(std::isfinite(solver.temperature().value(lid)));
        EXPECT_TRUE(std::isfinite(solver.velocity_z().value(lid)));
    }
}
