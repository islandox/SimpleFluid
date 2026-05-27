/**
 * @file natural_convection_cylinder.cc
 * @brief Runnable bottom-heated cylinder Boussinesq smoke example.
 */

#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"

#include <Tpetra_Core.hpp>

#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);

    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("height", SimpleFluid::real_t{2.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"radial", "zmin", "zmax"});

    SimpleFluid::MeshFactory factory(db);
    auto mesh = factory.build<>();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    bcs.temperature["radial"] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 5;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.reference_temperature = 0.5;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 50;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::BoussinesqSolver<> solver(mesh, bcs, time_options, linear_options);
    solver.initialize_bottom_hot_top_cold(1.0, 0.0);
    solver.run();
    solver.write_solution_vtu("natural_convection_cylinder.vtu");

    if (Tpetra::getDefaultComm()->getRank() == 0)
    {
        std::cout << "Wrote natural_convection_cylinder.vtu at t="
                  << solver.time() << "\n";
    }

    return 0;
}
