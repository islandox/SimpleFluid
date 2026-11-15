/** @file testNativeMeshFactory.cc
 * @brief Factory handles retain native identity through fluid and generic equations.
 */
#include "equations/Equation.hh"
#include "geometry/MeshFactory.hh"
#include "solvers/FluidSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using ScalarStored = ScalarCellFieldStored<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<Handle> make_factory_box()
{
    auto db = std::make_shared<Database>();
    db->set("dimension", 3);
    db->set("domain_type", static_cast<int>(MeshFactory::DomainType::BOX));
    db->set("mesh_size", 1.0);
    db->set("X", ArrReal{0, 2});
    db->set("Y", ArrReal{0, 3});
    db->set("Z", ArrReal{0, 4});
    db->set("domain_exterior_face_types", ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return MeshFactory(db).build_handle<Pack>();
}

class InspectFluidSolver : public FluidSolver<Pack>
{
public:
    using FluidSolver<Pack>::FluidSolver;
    bool legacy_backend() const { return uses_legacy_backend(); }
};
} // namespace

TEST(NativeMeshFactoryTest, FluidSolverKeepsFactoryHandleAndZeroFlowConservation)
{
    const auto mesh = make_factory_box();
    EXPECT_FALSE(mesh->is_stk());
    EXPECT_FALSE(mesh->legacy_mesh());
    EXPECT_TRUE(mesh->visit([](const auto& geometry)
        { return std::is_same_v<std::remove_cvref_t<decltype(geometry)>, Meshes::UnstructuredMesh>; }));
    TimeStepperOptions time;
    time.time_step = .25;
    time.steps = 1;
    time.kinematic_viscosity = 0;
    BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        boundaries.pressure[name] = {BoundaryConditionType::Neumann, 0};
    }
    LinearSolverOptions linear;
    linear.preconditioner = LinearPreconditioner::None;
    InspectFluidSolver solver(mesh, boundaries, time, linear);
    auto pressure_options = solver.pressure_linear_solver_options();
    pressure_options.preconditioner = LinearPreconditioner::None;
    solver.set_pressure_linear_solver_options(pressure_options);
    EXPECT_FALSE(solver.legacy_backend());
    EXPECT_EQ(solver.velocity().mesh_ptr().get(), mesh.get());
    EXPECT_EQ(solver.pressure().mesh_ptr().get(), mesh.get());
    solver.step();
    EXPECT_FALSE(solver.legacy_backend());
    EXPECT_EQ(solver.velocity().mesh_ptr().get(), mesh.get());
    EXPECT_EQ(solver.pressure().mesh_ptr().get(), mesh.get());
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_DOUBLE_EQ(solver.time(), .25);
    double local_inventory = 0, global_inventory = 0;
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<int>(c);
        EXPECT_DOUBLE_EQ(mesh->cell_volume(cell), 24);
        EXPECT_NEAR(solver.velocity().value(cell).norm(), 0, 1e-14);
        EXPECT_NEAR(solver.pressure().value(cell), 0, 1e-14);
        local_inventory += mesh->cell_volume(cell);
    }
    const auto comm = mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_inventory, &global_inventory);
    EXPECT_DOUBLE_EQ(global_inventory, 24);
    const auto& flux = solver.pressure_corrected_face_fluxes();
    for (size_t f = 0; f < mesh->num_owned_faces(); ++f)
        EXPECT_NEAR(flux.value(static_cast<int>(f)), 0, 1e-14);
    EXPECT_NEAR(solver.last_step_statistics().continuity, 0, 1e-14);
}

TEST(NativeMeshFactoryTest, GenericTransientSourceEquationSolvesOnSameNativeFactoryHandle)
{
    const auto mesh = make_factory_box();
    auto unknown = std::make_shared<ScalarStored>(ScalarCellFieldDescriptor<Pack>("factory_transient"), mesh);
    Equation<ScalarStored, Pack> equation(unknown);
    equation.add_lhs(FVM::TransientOperator<Pack>{.25, [](int, size_t) { return 3.0; }});
    equation.add_rhs(FVM::SourceOperator<Pack>{[](int, size_t) { return 2.0; }});
    LinearSolverOptions options;
    options.preconditioner = LinearPreconditioner::None;
    auto assembled = equation.assemble(options);
    EXPECT_TRUE(assembled.solve());
    EXPECT_EQ(unknown->mesh_ptr().get(), mesh.get());
    double local_integral = 0, global_integral = 0;
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<int>(c);
        EXPECT_NEAR(unknown->value(cell), 3.5, 1e-13);
        local_integral += unknown->value(cell) * mesh->cell_volume(cell);
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_integral, &global_integral);
    EXPECT_NEAR(global_integral, 84, 3e-12);
}

TEST(NativeMeshFactoryTest, NativeAndExplicitLegacyBackendsPreserveNonzeroUniformThroughFlow)
{
    auto db = std::make_shared<Database>();
    db->set("dimension", 3);
    db->set("domain_type", static_cast<int>(MeshFactory::DomainType::BOX));
    db->set("mesh_size", 1.0);
    db->set("X", ArrReal{0, 1, 2});
    db->set("Y", ArrReal{0, 1.5, 3});
    db->set("Z", ArrReal{0, 2, 4});
    db->set("domain_exterior_face_types", ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    MeshFactory factory(db);
    const auto native = factory.build_handle<Pack>();
    const auto legacy = factory.build<Pack>();
    EXPECT_FALSE(native->legacy_mesh());

    TimeStepperOptions time;
    time.time_step = .1;
    time.steps = 1;
    time.kinematic_viscosity = .1;
    BoundaryConditionSet boundaries;
    boundaries.velocity["xmin"] = {BoundaryConditionType::Dirichlet, {1, 0, 0}};
    // Prescribed-pressure outlets require zero-gradient velocity.
    boundaries.velocity["xmax"] = {BoundaryConditionType::Neumann, {}};
    for (const auto* name : {"ymin", "ymax", "zmin", "zmax"})
        boundaries.velocity[name] = {BoundaryConditionType::Slip, {}};
    for (const auto* name : {"xmin", "ymin", "ymax", "zmin", "zmax"})
        boundaries.pressure[name] = {BoundaryConditionType::Neumann, 0};
    boundaries.pressure["xmax"] = {BoundaryConditionType::Dirichlet, 0};
    LinearSolverOptions linear;
    linear.preconditioner = LinearPreconditioner::None;
    linear.tolerance = 1e-13;
    linear.max_iterations = 500;
    InspectFluidSolver native_solver(native, boundaries, time, linear);
    InspectFluidSolver legacy_solver(legacy, boundaries, time, linear);

    const auto advance_and_check = [&](InspectFluidSolver& solver, bool legacy_backend)
    {
        SCOPED_TRACE(legacy_backend ? "explicit legacy backend" : "native factory backend");
        auto pressure_options = solver.pressure_linear_solver_options();
        pressure_options.preconditioner = LinearPreconditioner::None;
        pressure_options.tolerance = 1e-13;
        solver.set_pressure_linear_solver_options(pressure_options);
        EXPECT_EQ(solver.legacy_backend(), legacy_backend);
        solver.velocity().put_value({1, 0, 0});
        solver.pressure().put_value(0);
        solver.step();
        EXPECT_EQ(solver.step_index(), 1);
        EXPECT_NEAR(solver.time(), .1, 1e-15);
        EXPECT_EQ(solver.legacy_backend(), legacy_backend);
        const auto mesh = solver.velocity().mesh_ptr();
        EXPECT_EQ(solver.pressure().mesh_ptr().get(), mesh.get());
        const auto& flux = solver.pressure_corrected_face_fluxes();
        // Inventories and physical boundary fluxes have independent exact
        // expectations and do not require matching partition-local cell orders.
        std::array<double, 8> local{}, global{};
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<int>(c);
            const auto velocity = solver.velocity().value(cell);
            const auto pressure = solver.pressure().value(cell);
            EXPECT_NEAR(velocity.x, 1, 1e-11);
            EXPECT_NEAR(velocity.y, 0, 1e-11);
            EXPECT_NEAR(velocity.z, 0, 1e-11);
            EXPECT_NEAR(pressure, 0, 1e-11);
            const auto volume = mesh->cell_volume(cell);
            EXPECT_DOUBLE_EQ(volume, 3);
            local[0] += volume;
            local[1] += volume * velocity.x;
            local[2] += volume * velocity.y;
            local[3] += volume * velocity.z;
            local[4] += volume * pressure;
            double cell_balance = 0;
            for (const auto face : mesh->faces(cell))
            {
                const auto outward = flux.value(face) * (mesh->owner_cell(face) == cell ? 1 : -1);
                cell_balance += outward;
                if (!mesh->is_boundary_face(face))
                    continue;
                const auto& name = mesh->boundary_batch_name(mesh->boundary_id(face));
                if (name == "xmin")
                    local[5] += outward;
                else if (name == "xmax")
                    local[6] += outward;
                else
                    EXPECT_NEAR(outward, 0, 1e-11);
            }
            EXPECT_NEAR(cell_balance, 0, 1e-11);
            local[7] += cell_balance;
        }
        const auto comm = mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(local.size()), local.data(), global.data());
        const std::array<double, 8> exact{24, 24, 0, 0, 0, -12, 12, 0};
        for (size_t i = 0; i < global.size(); ++i)
            EXPECT_NEAR(global[i], exact[i], 1e-11);
        return global;
    };
    const auto native_inventories = advance_and_check(native_solver, false);
    const auto legacy_inventories = advance_and_check(legacy_solver, true);
    for (size_t i = 0; i < native_inventories.size(); ++i)
        EXPECT_NEAR(native_inventories[i], legacy_inventories[i], 1e-11);
}
