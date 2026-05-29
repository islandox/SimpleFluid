/**
 * @file natural_convection_boundary_layer_box.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Runnable boundary-layer box case with three-direction Boussinesq forcing.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "examples/ExampleRunner.hh"

#include <Tpetra_Core.hpp>

#include <memory>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);

    auto db = std::make_shared<SimpleFluid::Database>();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{0.25});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    db->set("X", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("Y", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("Z", SimpleFluid::ArrReal{0.0, 0.25, 0.5, 0.75, 1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    db->set("boundary_layer_boundary_names",
            SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    db->set("boundary_layer_counts", SimpleFluid::ArrInt{1, 1, 1, 1, 1, 1});
    db->set("boundary_layer_first_cell_heights",
            SimpleFluid::ArrReal{0.05, 0.05, 0.05, 0.05, 0.05, 0.05});
    db->set("boundary_layer_growth_ratios",
            SimpleFluid::ArrReal{1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 5;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.gravity_x = -1.0;
    time_options.gravity_y = -2.0;
    time_options.gravity_z = -9.81;
    time_options.reference_temperature = 0.5;

    SimpleFluid::LinearSolverOptions linear_options;

    SimpleFluid::run_boussinesq_example<>(
        db, bcs, time_options, linear_options,
        [](auto& solver) { solver.initialize_heated_box(1.0, 0.0); },
        "natural_convection_boundary_layer_box.vtu");

    return 0;
}
