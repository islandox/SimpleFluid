/**
 * @file testBoussinesqPlanarALE.cc
 * @brief Solver-level acceptance and rollback tests for constrained planar ALE.
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Handle = SimpleFluid::MeshHandle<Pack>;
using BaseSolver = SimpleFluid::BoussinesqSolver<Pack>;
class Solver final : public BaseSolver
{
public:
    using BaseSolver::BaseSolver;

    SimpleFluid::BoundaryConditionType cached_velocity_boundary_type(const std::string& name)
    {
        return SimpleFluid::FluidSolver<Pack>::native_velocity_boundary_cache().type_by_name.at(name);
    }

    const canonical_velocity_boundary_cache_type& cached_velocity_boundary_cache()
    {
        return SimpleFluid::FluidSolver<Pack>::native_velocity_boundary_cache();
    }

    auto& mutable_pressure_corrected_face_fluxes()
    {
        return SimpleFluid::FluidSolver<Pack>::projected_face_fluxes();
    }

    auto solution_topology() const { return SimpleFluid::FluidSolver<Pack>::fluid_solution_writer().topology_handle(); }
};
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using FaceFlux = SimpleFluid::ScalarFaceFieldStored<Pack>;
using FaceVelocity = SimpleFluid::VectorFaceFieldStored<Pack>;
using Source = SimpleFluid::VolumetricScalarSource<Pack, Handle>;
using Coupling = SimpleFluid::PressureVelocityCoupling;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

/** Build a unit-area, two-cell vertical column on the default communicator. */
SimpleFluid::SP<Handle> make_column()
{
    auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.5, 1.0}}});
    return std::make_shared<Handle>(std::move(geometry));
}

SimpleFluid::BoundaryConditionSet planar_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
        boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    boundaries.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    return boundaries;
}

SimpleFluid::TimeStepperOptions time_options(Coupling coupling)
{
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-2;
    options.steps = 1;
    options.thermal_diffusivity = 0.0;
    options.kinematic_viscosity = 0.0;
    options.thermal_expansion = 0.0;
    options.gravity_x = 0.0;
    options.gravity_y = 0.0;
    options.gravity_z = 0.0;
    options.reference_temperature = 300.0;
    options.pressure_velocity_coupling = coupling;
    options.n_pressure_correctors = 2;
    options.n_outer_correctors = 2;
    return options;
}

SimpleFluid::BoussinesqModelOptions physical_options()
{
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 10.0;
    options.density = 10.0;
    options.specific_heat_capacity = 2.0;
    options.dynamic_viscosity = 1.0e-2;
    options.thermal_conductivity = 0.0;
    return options;
}

SimpleFluid::MaterialFeedbackOptions material_feedback_options()
{
    SimpleFluid::MaterialFeedbackOptions options;
    options.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    options.reference_density = 10.0;
    options.liquid_density = 10.0;
    options.gas_density = 1.0;
    options.reference_temperature = 300.0;
    options.thermal_expansion = 1.0e-3;
    options.reference_dynamic_viscosity = 1.0e-2;
    options.min_density = 1.0;
    return options;
}

SimpleFluid::FreeSurfaceOptions free_surface_options(int maximum_correctors)
{
    SimpleFluid::FreeSurfaceOptions options;
    options.enabled = true;
    options.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    options.gravity_axis = SimpleFluid::Dimension::Z;
    options.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    options.initial_liquid_volume = 1.0;
    options.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    options.vessel.bottom_elevation = 0.0;
    options.vessel.top_elevation = 2.0;
    options.vessel.cross_section_area = 1.0;
    options.vessel.total_internal_volume = 2.0;
    options.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    options.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    options.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    options.headspace.ambient_pressure = 101325.0;
    options.headspace.initial_pressure = 101325.0;
    options.headspace.initial_temperature = 300.0;
    options.ale.top_boundary = "zmax";
    options.ale.maximum_correctors = maximum_correctors;
    options.ale.level_absolute_tolerance = 1.0e-13;
    options.ale.level_relative_tolerance = 0.0;
    options.ale.relaxation = 1.0;
    return options;
}

SimpleFluid::RadiolyticGasOptions gas_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    options.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    options.bubble_transport = SimpleFluid::BubbleTransportMode::General;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ZeroSlip;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.max_source_alpha_rate = 1.0;
    options.henry_coefficient = 1.0e-5;
    options.surface_tension = 0.07;
    // mu/(rho*D)=100, inside the implemented Hughmark Sc range.
    options.hydrogen_diffusivity = 1.0e-5;
    options.uranium_concentration_mol_per_m3 = 1000.0;
    options.hydrogen_yield_molecules_per_100_ev = 1.8;
    options.reference_pressure = 101325.0;
    options.initial_dissolved_hydrogen = 1.0;
    options.free_surface_patches = {"zmax"};
    return options;
}

struct ConfiguredCase
{
    SimpleFluid::SP<Handle> mesh;
    std::unique_ptr<Solver> solver;
    Source* heat_source = nullptr;
};

ConfiguredCase make_case(Coupling coupling, double power_density, int maximum_correctors, bool include_gas = false,
    double bubble_slip_velocity = 0.0, bool zero_gas_tolerance = false,
    SimpleFluid::RadiolyticPressureMode pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant,
    bool use_celata_slip = false)
{
    auto mesh = make_column();
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    linear_options.max_iterations = 500;
    auto solver =
        std::make_unique<Solver>(mesh, planar_boundaries(), time_options(coupling), linear_options, physical_options());

    solver->configure_material_feedback(material_feedback_options());
    auto* heat_source = &solver->add_temperature_source("planar_ale_heat", power_density);
    if (include_gas)
    {
        solver->add_fission_power_source().initialize_constant(0.0);
        auto gas = gas_options();
        gas.pressure_mode = pressure_mode;
        if (bubble_slip_velocity > 0.0)
        {
            gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
            gas.constant_slip_velocity = bubble_slip_velocity;
        }
        if (use_celata_slip)
        {
            gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::Celata2007;
        }
        solver->configure_radiolytic_gas(gas);
    }
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto surface_options = free_surface_options(maximum_correctors);
    if (zero_gas_tolerance)
    {
        surface_options.coupling.gas_absolute_tolerance = 0.0;
        surface_options.coupling.gas_relative_tolerance = 0.0;
    }
    if (solver->configure_free_surface(surface_options) == nullptr)
    {
        throw std::logic_error("The planar-ALE test fixture did not create a free-surface model.");
    }
    return {std::move(mesh), std::move(solver), heat_source};
}

template<class Mutator> void expect_free_surface_option_rejected(Mutator&& mutator)
{
    auto mesh = make_column();
    auto solver = std::make_unique<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(4);
    std::forward<Mutator>(mutator)(options);
    EXPECT_THROW(static_cast<void>(solver->configure_free_surface(options)), std::invalid_argument);
    EXPECT_FALSE(solver->planar_ale_enabled());
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
}

template<class Configure> void expect_active_model_rejected(Configure&& configure)
{
    auto mesh = make_column();
    auto solver = std::make_unique<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    std::forward<Configure>(configure)(*solver);
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    EXPECT_THROW(static_cast<void>(solver->configure_free_surface(free_surface_options(4))), std::invalid_argument);
    EXPECT_FALSE(solver->planar_ale_enabled());
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
}

double global_mesh_volume(const Handle& mesh)
{
    double local_volume = 0.0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        local_volume += mesh.cell_volume(static_cast<Pack::local_ordinal_type>(owned));
    }
    double global_volume = 0.0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);
    return global_volume;
}

double top_elevation(const Handle& mesh)
{
    double local_top = -std::numeric_limits<double>::infinity();
    int local_faces = 0;
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        if (mesh.boundary_batch_name(batch_id) != "zmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (mesh.is_owned_face(face_lid))
            {
                local_top = std::max(local_top, mesh.face_centroid(face_lid).z);
                ++local_faces;
            }
        }
    }
    double global_top = local_top;
    int global_faces = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_top, &global_top);
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    if (global_faces == 0 || !std::isfinite(global_top))
    {
        throw std::logic_error("The planar-ALE test mesh has no owned zmax face.");
    }
    return global_top;
}

void expect_zero_relative_top_flux(const ConfiguredCase& state)
{
    const auto& flux = state.solver->mesh_relative_face_fluxes();
    double local_maximum = 0.0;
    int local_faces = 0;
    for (const auto& [batch_id, batch] : state.mesh->boundary_batches())
    {
        if (state.mesh->boundary_batch_name(batch_id) != "zmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (flux.is_owned_face(face_lid))
            {
                local_maximum = std::max(local_maximum, std::abs(flux.value(face_lid)));
                ++local_faces;
            }
        }
    }
    double global_maximum = 0.0;
    int global_faces = 0;
    Teuchos::reduceAll(
        *state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum, &global_maximum);
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    EXPECT_GT(global_faces, 0);
    EXPECT_NEAR(global_maximum, 0.0, 1.0e-14);
}

double independent_maximum_courant_number(const ConfiguredCase& state, const FaceFlux& flux)
{
    double local_maximum = 0.0;
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        double absolute_flux_sum = 0.0;
        for (const auto face_lid : state.mesh->faces(cell))
        {
            absolute_flux_sum += std::abs(flux.local_value(face_lid));
        }
        local_maximum = std::max(
            local_maximum, 0.5 * state.solver->time_step() * absolute_flux_sum / state.mesh->cell_volume(cell));
    }
    double global_maximum = 0.0;
    Teuchos::reduceAll(
        *state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum, &global_maximum);
    return global_maximum;
}

void record_ale_residuals(const Solver::PlanarALEStepDiagnostics& diagnostics)
{
    testing::Test::RecordProperty("gcl_max_m3_per_s", diagnostics.maximum_gcl_residual);
    testing::Test::RecordProperty("continuity_max_m3_per_s", diagnostics.continuity.maximum);
    testing::Test::RecordProperty("liquid_mass_residual_kg", diagnostics.liquid_mass_residual);
    testing::Test::RecordProperty("gas_inventory_residual_mol", diagnostics.gas_inventory_residual);
    testing::Test::RecordProperty("mesh_pool_mismatch_m3", diagnostics.mesh_vessel_mismatch);
    testing::Test::RecordProperty(
        "source_pool_closure_m3_per_s", diagnostics.volume_source.source_pool_closure_residual);
    testing::Test::RecordProperty("energy_residual_j", diagnostics.energy_residual);
}

struct GeometrySnapshot
{
    std::uint64_t epoch = 0;
    std::vector<double> cell_volumes;
    std::vector<Handle::Vec3> cell_centroids;
    std::vector<Handle::Vec3> face_centroids;
};

GeometrySnapshot capture_geometry(const Handle& mesh)
{
    GeometrySnapshot result;
    result.epoch = mesh.geometry_epoch();
    result.cell_volumes.reserve(mesh.num_local_cells());
    result.cell_centroids.reserve(mesh.num_local_cells());
    result.face_centroids.reserve(mesh.num_faces());
    for (size_t local = 0; local < mesh.num_local_cells(); ++local)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(local);
        result.cell_volumes.push_back(mesh.cell_volume(cell));
        result.cell_centroids.push_back(mesh.cell_centroid(cell));
    }
    for (size_t local = 0; local < mesh.num_faces(); ++local)
    {
        result.face_centroids.push_back(mesh.face_centroid(static_cast<Pack::local_ordinal_type>(local)));
    }
    return result;
}

void expect_geometry_restored(const Handle& mesh, const GeometrySnapshot& expected)
{
    EXPECT_GT(mesh.geometry_epoch(), expected.epoch);
    ASSERT_EQ(mesh.num_local_cells(), expected.cell_volumes.size());
    ASSERT_EQ(mesh.num_faces(), expected.face_centroids.size());
    for (size_t local = 0; local < mesh.num_local_cells(); ++local)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(mesh.cell_volume(cell), expected.cell_volumes[local]);
        const auto centroid = mesh.cell_centroid(cell);
        EXPECT_DOUBLE_EQ(centroid.x, expected.cell_centroids[local].x);
        EXPECT_DOUBLE_EQ(centroid.y, expected.cell_centroids[local].y);
        EXPECT_DOUBLE_EQ(centroid.z, expected.cell_centroids[local].z);
    }
    for (size_t local = 0; local < mesh.num_faces(); ++local)
    {
        const auto centroid = mesh.face_centroid(static_cast<Pack::local_ordinal_type>(local));
        EXPECT_DOUBLE_EQ(centroid.x, expected.face_centroids[local].x);
        EXPECT_DOUBLE_EQ(centroid.y, expected.face_centroids[local].y);
        EXPECT_DOUBLE_EQ(centroid.z, expected.face_centroids[local].z);
    }
}

struct PrimarySnapshot
{
    std::vector<double> temperature;
    std::vector<double> pressure;
    std::vector<Handle::Vec3> velocity;
};

PrimarySnapshot capture_primary(const Solver& solver)
{
    PrimarySnapshot result;
    const auto owned_cells = solver.temperature().num_owned_cells();
    result.temperature.reserve(owned_cells);
    result.pressure.reserve(owned_cells);
    result.velocity.reserve(owned_cells);
    for (size_t owned = 0; owned < owned_cells; ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        result.temperature.push_back(solver.temperature().value(cell));
        result.pressure.push_back(solver.pressure().value(cell));
        result.velocity.push_back(solver.velocity().value(cell));
    }
    return result;
}

void expect_primary_restored(const Solver& solver, const PrimarySnapshot& expected)
{
    ASSERT_EQ(solver.temperature().num_owned_cells(), expected.temperature.size());
    for (size_t owned = 0; owned < expected.temperature.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(solver.temperature().value(cell), expected.temperature[owned]);
        EXPECT_DOUBLE_EQ(solver.pressure().value(cell), expected.pressure[owned]);
        const auto velocity = solver.velocity().value(cell);
        EXPECT_DOUBLE_EQ(velocity.x, expected.velocity[owned].x);
        EXPECT_DOUBLE_EQ(velocity.y, expected.velocity[owned].y);
        EXPECT_DOUBLE_EQ(velocity.z, expected.velocity[owned].z);
    }
}

using FieldValues = std::map<std::string, std::vector<double>>;

std::vector<double> capture_owned_values(const ScalarField& field)
{
    std::vector<double> result;
    result.reserve(field.num_owned_cells());
    for (size_t owned = 0; owned < field.num_owned_cells(); ++owned)
    {
        result.push_back(field.value(static_cast<Pack::local_ordinal_type>(owned)));
    }
    return result;
}

void expect_owned_values(const ScalarField& field, const std::vector<double>& expected)
{
    ASSERT_EQ(field.num_owned_cells(), expected.size());
    for (size_t owned = 0; owned < expected.size(); ++owned)
    {
        EXPECT_DOUBLE_EQ(field.value(static_cast<Pack::local_ordinal_type>(owned)), expected[owned]);
    }
}

struct FaceValues
{
    std::vector<Pack::local_ordinal_type> local_ids;
    std::vector<double> values;
};

FaceValues capture_owned_face_values(const Solver::face_flux_field_type& field)
{
    FaceValues result;
    for (const auto face_lid : field.owned_face_ids())
    {
        result.local_ids.push_back(face_lid);
        result.values.push_back(field.value(face_lid));
    }
    return result;
}

void expect_owned_face_values(const Solver::face_flux_field_type& field, const FaceValues& expected)
{
    ASSERT_EQ(field.owned_face_ids().size(), expected.local_ids.size());
    ASSERT_EQ(expected.local_ids.size(), expected.values.size());
    for (size_t index = 0; index < expected.local_ids.size(); ++index)
    {
        EXPECT_DOUBLE_EQ(field.value(expected.local_ids[index]), expected.values[index]);
    }
}

FieldValues capture_gas_fields(const Solver::radiolytic_gas_model_type& gas)
{
    FieldValues result;
    for (const auto& [name, field] : gas.output_fields())
    {
        auto& values = result[name];
        values.reserve(field->num_owned_cells());
        for (size_t owned = 0; owned < field->num_owned_cells(); ++owned)
        {
            values.push_back(field->value(static_cast<Pack::local_ordinal_type>(owned)));
        }
    }
    return result;
}

void expect_gas_fields_restored(const Solver::radiolytic_gas_model_type& gas, const FieldValues& expected)
{
    const auto actual = capture_gas_fields(gas);
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [name, values] : expected)
    {
        const auto found = actual.find(name);
        ASSERT_NE(found, actual.end()) << name;
        ASSERT_EQ(found->second.size(), values.size()) << name;
        for (size_t index = 0; index < values.size(); ++index)
        {
            EXPECT_DOUBLE_EQ(found->second[index], values[index]) << name;
        }
    }
}

const char* coupling_name(const testing::TestParamInfo<Coupling>& parameter)
{
    switch (parameter.param)
    {
        case Coupling::SIMPLE:
            return "SIMPLE";
        case Coupling::PISO:
            return "PISO";
        case Coupling::PIMPLE:
            return "PIMPLE";
        case Coupling::CoupledKrylov:
            return "CoupledKrylov";
    }
    return "Unknown";
}

class BoussinesqPlanarALECouplingTest : public testing::TestWithParam<Coupling>
{
};

TEST_P(BoussinesqPlanarALECouplingTest, AcceptsStationaryStepWithoutChangingExtensiveState)
{
    auto state = make_case(GetParam(), 0.0, 3);
    const auto initial_epoch = state.mesh->geometry_epoch();
    const auto initial_volume = global_mesh_volume(*state.mesh);
    const auto initial_top = top_elevation(*state.mesh);
    const auto initial_mass = state.solver->liquid_mass_inventory().totalMass();
    const auto initial_history = state.solver->free_surface_history().size();

    ASSERT_TRUE(state.solver->planar_ale_enabled());
    ASSERT_NO_THROW(state.solver->step());

    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_NEAR(state.solver->time(), 1.0e-2, 1.0e-15);
    EXPECT_EQ(state.solver->free_surface_history().size(), initial_history + 1);
    EXPECT_GT(state.mesh->geometry_epoch(), initial_epoch);
    EXPECT_DOUBLE_EQ(global_mesh_volume(*state.mesh), initial_volume);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), initial_top);
    EXPECT_DOUBLE_EQ(state.solver->liquid_mass_inventory().totalMass(), initial_mass);
    EXPECT_DOUBLE_EQ(state.solver->free_surface_diagnostics().pool_level, initial_top);
    EXPECT_DOUBLE_EQ(state.solver->free_surface_diagnostics().pool_volume, initial_volume);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    EXPECT_LE(state.solver->last_volume_continuity_residuals().maximum, 1.0e-12);

    const auto& diagnostics = state.solver->planar_ale_diagnostics();
    EXPECT_TRUE(diagnostics.initialized);
    EXPECT_EQ(diagnostics.outer_correctors, 1);
    EXPECT_DOUBLE_EQ(diagnostics.old_mesh_volume, initial_volume);
    EXPECT_DOUBLE_EQ(diagnostics.new_mesh_volume, initial_volume);
    EXPECT_NEAR(diagnostics.maximum_gcl_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-14);
    expect_zero_relative_top_flux(state);
}

TEST_P(BoussinesqPlanarALECouplingTest, AppliesOneNonzeroThermalVolumeTargetInEveryCouplingAlgorithm)
{
    auto state = make_case(GetParam(), 1.0e-4, 8);

    ASSERT_NO_THROW(state.solver->step());

    const auto surface = state.solver->free_surface_diagnostics();
    const auto& diagnostics = state.solver->planar_ale_diagnostics();
    record_ale_residuals(diagnostics);
    EXPECT_GT(diagnostics.volume_source.global_material_source, 0.0);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), surface.pool_volume, 2.0e-12);
    EXPECT_NEAR(top_elevation(*state.mesh), surface.pool_level, 2.0e-12);
    EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-10);
    EXPECT_LE(diagnostics.continuity.maximum, 3.0e-10);
    EXPECT_NEAR(diagnostics.liquid_mass_residual, 0.0, 1.0e-12);
    expect_zero_relative_top_flux(state);
}

TEST_P(BoussinesqPlanarALECouplingTest, PreservesTangentialFlowAtMovingSlipTop)
{
    auto state = make_case(GetParam(), 1.0e-4, 8);
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        state.solver->velocity().set_owned_value(
            static_cast<Pack::local_ordinal_type>(owned), SimpleFluid::vec3<double>{0.2, -0.1, 0.0});
    }
    state.solver->velocity().sync_ghosts();
    const auto old_top = top_elevation(*state.mesh);

    ASSERT_NO_THROW(state.solver->step());
    ASSERT_GT(top_elevation(*state.mesh), old_top);
    ASSERT_EQ(state.solver->cached_velocity_boundary_type("zmax"), SimpleFluid::BoundaryConditionType::Slip);

    FaceVelocity face_velocity(state.mesh, "accepted_moving_slip_face_velocity");
    SimpleFluid::FVM::face_velocities(
        state.solver->velocity(), state.solver->cached_velocity_boundary_cache(), face_velocity);
    double local_maximum_tangential_speed = 0.0;
    int local_faces = 0;
    for (const auto& [batch_id, batch] : state.mesh->boundary_batches())
    {
        if (state.mesh->boundary_batch_name(batch_id) != "zmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (!face_velocity.is_owned_face(face_lid))
            {
                continue;
            }
            ++local_faces;
            const auto owner = state.mesh->owner_cell(face_lid);
            const auto normal = state.mesh->face_normal_outward(face_lid, owner);
            const auto owner_velocity = state.solver->velocity().local_value(owner);
            const auto expected = owner_velocity - normal * owner_velocity.dot(normal);
            const auto actual = face_velocity.value(face_lid);
            EXPECT_NEAR(actual.x, expected.x, 1.0e-14);
            EXPECT_NEAR(actual.y, expected.y, 1.0e-14);
            EXPECT_NEAR(actual.z, expected.z, 1.0e-14);
            EXPECT_NEAR(actual.dot(normal), 0.0, 1.0e-14);
            local_maximum_tangential_speed = std::max(local_maximum_tangential_speed, actual.norm());
        }
    }
    double global_maximum_tangential_speed = 0.0;
    int global_faces = 0;
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1,
        &local_maximum_tangential_speed, &global_maximum_tangential_speed);
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    EXPECT_GT(global_faces, 0);
    EXPECT_GT(global_maximum_tangential_speed, 0.1);
    expect_zero_relative_top_flux(state);
}

TEST_P(BoussinesqPlanarALECouplingTest, CourantUsesMeshRelativeTransportFluxWithZeroRelativeTopFlux)
{
    auto state = make_case(GetParam(), 0.1, 8);
    const auto old_top = top_elevation(*state.mesh);
    ASSERT_NO_THROW(state.solver->step());
    ASSERT_GT(top_elevation(*state.mesh), old_top);
    expect_zero_relative_top_flux(state);

    const auto relative_courant = independent_maximum_courant_number(state, state.solver->mesh_relative_face_fluxes());
    const auto absolute_courant =
        independent_maximum_courant_number(state, state.solver->pressure_corrected_face_fluxes());
    ASSERT_GT(absolute_courant, 1.0e-9);
    EXPECT_LT(relative_courant, 1.0e-11);
    EXPECT_LT(relative_courant, absolute_courant * 1.0e-3);
    EXPECT_NEAR(state.solver->maximum_courant_number(), relative_courant, 1.0e-15);

    const auto& base_solver = static_cast<const SimpleFluid::FluidSolver<Pack>&>(*state.solver);
    EXPECT_NEAR(base_solver.maximum_courant_number(), relative_courant, 1.0e-15);
}

INSTANTIATE_TEST_SUITE_P(PlanarALECouplingModes, BoussinesqPlanarALECouplingTest,
    testing::Values(Coupling::SIMPLE, Coupling::PISO, Coupling::PIMPLE, Coupling::CoupledKrylov), coupling_name);

TEST(BoussinesqPlanarALETest, UniformHeatingMovesTopAndClosesConservativeBalances)
{
    constexpr double power_density = 1.0e-3;
    constexpr double time_step = 1.0e-2;
    constexpr double density = 10.0;
    constexpr double heat_capacity = 2.0;
    constexpr double expansion = 1.0e-3;
    constexpr double initial_temperature = 300.0;

    auto state = make_case(Coupling::PISO, power_density, 4);
    const auto old_epoch = state.mesh->geometry_epoch();
    const auto old_top = top_elevation(*state.mesh);
    const auto old_mass = state.solver->liquid_mass_inventory().totalMass();
    const auto old_history = state.solver->free_surface_history().size();

    ASSERT_NO_THROW(state.solver->step());

    const auto expected_temperature = initial_temperature + power_density * time_step / (density * heat_capacity);
    const auto expected_density = density * (1.0 - expansion * (expected_temperature - initial_temperature));
    const auto expected_level = old_mass / expected_density;
    const auto new_top = top_elevation(*state.mesh);
    const auto mesh_volume = global_mesh_volume(*state.mesh);
    const auto surface = state.solver->free_surface_diagnostics();
    const auto& diagnostics = state.solver->planar_ale_diagnostics();
    record_ale_residuals(diagnostics);

    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_NEAR(state.solver->time(), time_step, 1.0e-15);
    EXPECT_EQ(state.solver->free_surface_history().size(), old_history + 1);
    EXPECT_GT(state.mesh->geometry_epoch(), old_epoch);
    EXPECT_GT(new_top, old_top);
    EXPECT_NEAR(new_top, expected_level, 2.0e-12);
    EXPECT_NEAR(mesh_volume, expected_level, 2.0e-12);
    EXPECT_NEAR(surface.pool_level, new_top, 2.0e-12);
    EXPECT_NEAR(surface.pool_volume, mesh_volume, 2.0e-12);
    EXPECT_NEAR(state.solver->liquid_mass_inventory().totalMass(), old_mass, 1.0e-12);
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        // The strict pressure correction redistributes the uniform heating by
        // at most 1.197e-9 K between these two moving control volumes.
        EXPECT_NEAR(state.solver->temperature().value(cell), expected_temperature, 2.0e-9);
        EXPECT_NEAR(state.solver->rho_liquid().value(cell), expected_density, 2.0e-10);
    }
    EXPECT_GE(diagnostics.outer_correctors, 2);
    EXPECT_LE(diagnostics.outer_correctors, 4);
    EXPECT_EQ(diagnostics.level_residual_history.size(), static_cast<size_t>(diagnostics.outer_correctors));
    EXPECT_EQ(diagnostics.target_change_history.size(), static_cast<size_t>(diagnostics.outer_correctors));
    EXPECT_EQ(diagnostics.continuity_maximum_history.size(), static_cast<size_t>(diagnostics.outer_correctors));
    EXPECT_EQ(diagnostics.material_state_residual_history.size(), static_cast<size_t>(diagnostics.outer_correctors));
    EXPECT_EQ(diagnostics.gas_state_residual_history.size(), static_cast<size_t>(diagnostics.outer_correctors));
    ASSERT_FALSE(diagnostics.level_residual_history.empty());
    EXPECT_DOUBLE_EQ(diagnostics.level_residual_history.back(), diagnostics.level_residual);
    EXPECT_DOUBLE_EQ(diagnostics.target_change_history.back(), diagnostics.volume_source.maximum_target_change);
    EXPECT_DOUBLE_EQ(diagnostics.continuity_maximum_history.back(), diagnostics.continuity.maximum);
    EXPECT_DOUBLE_EQ(diagnostics.material_state_residual_history.back(), diagnostics.material_state_residual);
    EXPECT_DOUBLE_EQ(diagnostics.gas_state_residual_history.back(), diagnostics.gas_state_residual);
    EXPECT_NEAR(diagnostics.mesh_vessel_mismatch, 0.0, 2.0e-12);
    EXPECT_NEAR(diagnostics.maximum_gcl_residual, 0.0, 1.0e-12);
    EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-10);
    EXPECT_LE(diagnostics.continuity.maximum, 3.0e-10);
    EXPECT_LE(std::abs(diagnostics.liquid_mass_residual), 1.0e-12);
    EXPECT_LE(std::abs(diagnostics.energy_residual), 6.1e-6);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, ConfiguringAlePreservesAcceptedStationaryCourantFlux)
{
    auto mesh = make_column();
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    linear_options.max_iterations = 500;
    auto solver = std::make_unique<Solver>(
        mesh, planar_boundaries(), time_options(Coupling::PISO), linear_options, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);

    auto& accepted_absolute_flux = solver->mutable_pressure_corrected_face_fluxes();
    for (const auto face_lid : accepted_absolute_flux.owned_face_ids())
    {
        accepted_absolute_flux.set_owned_value(face_lid, 0.01 * (1.0 + static_cast<double>(face_lid)));
    }
    accepted_absolute_flux.sync_ghosts();
    const auto fixed_grid_courant = solver->maximum_courant_number();
    ASSERT_GT(fixed_grid_courant, 0.0);

    ASSERT_NE(solver->configure_free_surface(free_surface_options(4)), nullptr);
    ASSERT_TRUE(solver->planar_ale_enabled());
    const auto& accepted_relative_flux = solver->mesh_relative_face_fluxes();
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        EXPECT_DOUBLE_EQ(accepted_relative_flux.local_value(face_lid), accepted_absolute_flux.local_value(face_lid));
    }
    EXPECT_DOUBLE_EQ(solver->maximum_courant_number(), fixed_grid_courant);
    const auto& base_solver = static_cast<const SimpleFluid::FluidSolver<Pack>&>(*solver);
    EXPECT_DOUBLE_EQ(base_solver.maximum_courant_number(), fixed_grid_courant);
}

TEST(BoussinesqPlanarALETest, HeatingThenCoolingContractsAndRecoversTheAcceptedState)
{
    constexpr double power_density = 1.0e-3;
    constexpr double time_step = 1.0e-2;
    auto state = make_case(Coupling::PISO, power_density, 5);
    const auto initial_top = top_elevation(*state.mesh);
    const auto initial_volume = global_mesh_volume(*state.mesh);
    const auto initial_mass = state.solver->liquid_mass_inventory().totalMass();
    const auto initial_temperature = state.solver->temperature().value(0);

    ASSERT_NO_THROW(state.solver->step());
    const auto expanded_top = top_elevation(*state.mesh);
    const auto expanded_volume = global_mesh_volume(*state.mesh);
    ASSERT_GT(expanded_top, initial_top);
    ASSERT_GT(expanded_volume, initial_volume);

    // The source is volumetric on the trial-new mesh.  Scale the cooling so
    // its final-volume integral exactly removes the preceding accepted heat.
    state.heat_source->initialize(-power_density * expanded_volume / initial_volume);
    ASSERT_NO_THROW(state.solver->step());

    const auto& diagnostics = state.solver->planar_ale_diagnostics();
    EXPECT_LT(top_elevation(*state.mesh), expanded_top);
    EXPECT_NEAR(top_elevation(*state.mesh), initial_top, 3.0e-12);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), initial_volume, 3.0e-12);
    EXPECT_NEAR(state.solver->temperature().value(0), initial_temperature, 3.0e-9);
    EXPECT_NEAR(state.solver->liquid_mass_inventory().totalMass(), initial_mass, 2.0e-12);
    EXPECT_NEAR(diagnostics.maximum_gcl_residual, 0.0, 1.0e-12);
    EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-10);
    EXPECT_LE(diagnostics.continuity.maximum, 3.0e-10);
    EXPECT_LE(std::abs(diagnostics.liquid_mass_residual), 1.0e-12);
    EXPECT_LE(std::abs(diagnostics.energy_residual), 6.1e-6);
    EXPECT_EQ(state.solver->step_index(), 2);
    EXPECT_NEAR(state.solver->time(), 2.0 * time_step, 1.0e-15);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, KeepsConfiguredSlipBoundaryDuringAndAfterAle)
{
    auto state = make_case(Coupling::PISO, 1.0, 6);
    const auto old_top = top_elevation(*state.mesh);
    ASSERT_NO_THROW(state.solver->step());
    ASSERT_GT(top_elevation(*state.mesh), old_top);
    EXPECT_EQ(state.solver->cached_velocity_boundary_type("zmax"), SimpleFluid::BoundaryConditionType::Slip);

    ASSERT_TRUE(state.solver->remove_free_surface_model());
    EXPECT_EQ(state.solver->cached_velocity_boundary_type("zmax"), SimpleFluid::BoundaryConditionType::Slip);
}

TEST(BoussinesqPlanarALETest, AcceptedMotionUpdatesVtuPointsAndPublishesAleDiagnostics)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
    {
        GTEST_SKIP() << "This accepted-file assertion uses one serial output path.";
    }
    auto state = make_case(Coupling::PISO, 1.0, 6);
    ASSERT_NO_THROW(state.solver->step());
    const auto accepted_top = top_elevation(*state.mesh);
    const auto topology = state.solver->solution_topology();
    ASSERT_TRUE(topology);
    const auto maximum_point_z = std::ranges::max(topology->points, {}, [](const auto& point) { return point.z; }).z;
    EXPECT_NEAR(maximum_point_z, accepted_top, 2.0e-12);

    const auto filename = std::filesystem::temp_directory_path() / "simplefluid_planar_ale_accepted.vtu";
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_free_surface_fields = true;
    ASSERT_NO_THROW(state.solver->write_solution_vtu(filename.string(), output_options));
    std::ifstream input(filename, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("Name=\"meshVolumeRate\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"volumeSourceRate\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"continuityTarget\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"continuityResidual\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"liquidMassInventory\""), std::string::npos);
    input.close();
    std::filesystem::remove(filename);
}

TEST(BoussinesqPlanarALETest, AcceptedMotionWritesCurrentParallelPiecesAndPvtuDiagnostics)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This PVTU assertion requires exactly two ranks.";
    }
    auto state = make_case(Coupling::PISO, 1.0, 6);
    ASSERT_NO_THROW(state.solver->step());
    const auto accepted_top = top_elevation(*state.mesh);
    const auto topology = state.solver->solution_topology();
    ASSERT_TRUE(topology);
    const auto local_maximum_point_z =
        std::ranges::max(topology->points, {}, [](const auto& point) { return point.z; }).z;
    double maximum_point_z{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_maximum_point_z, &maximum_point_z);
    EXPECT_NEAR(maximum_point_z, accepted_top, 2.0e-12);

    const auto filename = std::filesystem::temp_directory_path() / "simplefluid_planar_ale_parallel_accepted.vtu";
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_free_surface_fields = true;
    ASSERT_NO_THROW(state.solver->write_parallel_solution_vtu(filename.string(), output_options));

    const auto piece_filename =
        SimpleFluid::VTUWriter::rank_piece_filename(filename.string(), comm->getRank(), comm->getSize());
    std::ifstream piece_input(piece_filename, std::ios::binary);
    EXPECT_TRUE(piece_input);
    const std::string piece_contents{std::istreambuf_iterator<char>(piece_input), std::istreambuf_iterator<char>()};
    for (const auto* field :
        {"meshVolumeRate", "volumeSourceRate", "bubbleSlipVolumeRate", "continuityTarget", "continuityResidual"})
    {
        EXPECT_NE(piece_contents.find("Name=\"" + std::string(field) + "\""), std::string::npos);
    }
    piece_input.close();

    if (comm->getRank() == 0)
    {
        const auto index_filename = SimpleFluid::VTUWriter::parallel_index_filename(filename.string());
        std::ifstream index_input(index_filename);
        EXPECT_TRUE(index_input);
        const std::string index_contents{std::istreambuf_iterator<char>(index_input), std::istreambuf_iterator<char>()};
        EXPECT_NE(index_contents.find("type=\"PUnstructuredGrid\""), std::string::npos);
        EXPECT_NE(index_contents.find("_rank0.vtu\""), std::string::npos);
        EXPECT_NE(index_contents.find("_rank1.vtu\""), std::string::npos);
        EXPECT_NE(index_contents.find("Name=\"continuityResidual\""), std::string::npos);
        index_input.close();
    }
    comm->barrier();
    std::filesystem::remove(piece_filename);
    comm->barrier();
    if (comm->getRank() == 0)
    {
        std::filesystem::remove(SimpleFluid::VTUWriter::parallel_index_filename(filename.string()));
    }
    comm->barrier();
}

TEST(BoussinesqPlanarALETest, ShengHydrogenGenerationUsesTheAcceptedMovingGeometryAndClosesInventory)
{
    auto state = make_case(Coupling::PISO, 0.0, 8, true);
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(fission, nullptr);
    fission->initialize_constant(1.0e-3);
    const auto old_generated = gas->cumulative_hydrogen_produced();
    const auto old_epoch = state.mesh->geometry_epoch();

    ASSERT_NO_THROW(state.solver->step());

    const auto surface = state.solver->free_surface_diagnostics();
    const auto& ale = state.solver->planar_ale_diagnostics();
    record_ale_residuals(ale);
    EXPECT_GT(gas->cumulative_hydrogen_produced(), old_generated);
    EXPECT_GT(state.mesh->geometry_epoch(), old_epoch);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), surface.pool_volume, 2.0e-12);
    EXPECT_NEAR(top_elevation(*state.mesh), surface.pool_level, 2.0e-12);
    EXPECT_NEAR(gas->last_statistics().inventory_error, 0.0, 1.0e-12);
    EXPECT_NEAR(ale.gas_inventory_residual, 0.0, 1.0e-12);
    EXPECT_GE(ale.outer_correctors, 2);
    EXPECT_LE(ale.gas_state_residual, 1.0e-10);
    EXPECT_EQ(ale.gas_state_residual_history.size(), static_cast<size_t>(ale.outer_correctors));
    EXPECT_NEAR(ale.volume_source.source_pool_closure_residual, 0.0, 1.0e-10);
    EXPECT_LE(ale.continuity.maximum, 3.0e-10);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, MovingTopBubbleEscapeTransfersToTheVentExactlyOnceAfterCommit)
{
    auto state = make_case(Coupling::PISO, 0.0, 10, true, 100.0);
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(fission, nullptr);
    fission->initialize_constant(1.0e3);

    // The Sheng split transports accepted populations before applying this
    // step's local kinetics.  Prime one accepted step so the following
    // moving-mesh transport step has a nonzero bubble population to vent.
    ASSERT_NO_THROW(state.solver->step());
    const auto old_cumulative_escape = gas->cumulative_submerged_bubble_hydrogen_escaped();
    const auto old_vented = state.solver->free_surface_diagnostics().vented_gas_moles;
    const auto old_vented_h2 = old_vented.contains("H2") ? old_vented.at("H2") : 0.0;

    ASSERT_NO_THROW(state.solver->step());

    const auto statistics = gas->last_statistics();
    const auto surface = state.solver->free_surface_diagnostics();
    record_ale_residuals(state.solver->planar_ale_diagnostics());
    ASSERT_GT(statistics.submerged_bubble_hydrogen_escaped, 0.0);
    ASSERT_TRUE(surface.escaped_gas_moles_this_step.contains("H2"));
    ASSERT_TRUE(surface.vented_gas_moles.contains("H2"));
    EXPECT_NEAR(surface.escaped_gas_moles_this_step.at("H2"), statistics.submerged_bubble_hydrogen_escaped, 1.0e-14);
    EXPECT_NEAR(gas->cumulative_submerged_bubble_hydrogen_escaped() - old_cumulative_escape,
        statistics.submerged_bubble_hydrogen_escaped, 1.0e-14);
    EXPECT_NEAR(
        surface.vented_gas_moles.at("H2") - old_vented_h2, statistics.submerged_bubble_hydrogen_escaped, 1.0e-14);
    EXPECT_NEAR(state.solver->planar_ale_diagnostics().gas_inventory_residual, 0.0, 1.0e-12);
    EXPECT_GE(state.solver->planar_ale_diagnostics().outer_correctors, 2);
    EXPECT_LE(state.solver->planar_ale_diagnostics().gas_state_residual, 1.0e-10);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), surface.pool_volume, 2.0e-12);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, ReconstructedPressureRunsInsideTheAcceptedAleTransaction)
{
    auto state =
        make_case(Coupling::PISO, 0.0, 10, true, 0.0, false, SimpleFluid::RadiolyticPressureMode::Reconstructed);
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(fission, nullptr);
    fission->initialize_constant(1.0e-3);

    ASSERT_NO_THROW(state.solver->step());
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        EXPECT_GT(gas->absolute_pressure().value(static_cast<Pack::local_ordinal_type>(owned)), 0.0);
    }
    EXPECT_NEAR(gas->last_statistics().inventory_error, 0.0, 1.0e-12);
    EXPECT_LE(state.solver->planar_ale_diagnostics().continuity.maximum, 3.0e-10);
}

TEST(BoussinesqPlanarALETest, CelataSlipRunsThroughSolverIntegratedMovingTopEscape)
{
    auto state =
        make_case(Coupling::PISO, 0.0, 12, true, 0.0, false, SimpleFluid::RadiolyticPressureMode::Constant, true);
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(fission, nullptr);
    fission->initialize_constant(1.0e3);

    ASSERT_NO_THROW(state.solver->step());
    ASSERT_NO_THROW(state.solver->step());
    EXPECT_GT(gas->last_statistics().submerged_bubble_hydrogen_escaped, 0.0);
    EXPECT_NEAR(gas->last_statistics().inventory_error, 0.0, 1.0e-12);
    EXPECT_LE(state.solver->planar_ale_diagnostics().continuity.maximum, 3.0e-10);
}

TEST(BoussinesqPlanarALETest, StrictGasStepClosureFailureRollsBackTheWholeTransaction)
{
    auto state = make_case(Coupling::PISO, 0.0, 8, true, 0.0, true);
    auto* gas = state.solver->find_radiolytic_gas_model();
    auto* fission = state.solver->find_fission_power_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(fission, nullptr);
    fission->initialize_constant(1.0e-3);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto old_generated = gas->cumulative_hydrogen_produced();
    const auto old_escape = gas->cumulative_submerged_bubble_hydrogen_escaped();
    const auto old_history = state.solver->free_surface_history().size();

    std::string failure;
    try
    {
        state.solver->step();
        FAIL() << "Expected zero-tolerance gas closure to reject roundoff.";
    }
    catch (const std::runtime_error& error)
    {
        failure = error.what();
    }
    EXPECT_NE(failure.find("radiolytic H2 step-inventory closure"), std::string::npos) << failure;
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(gas->cumulative_hydrogen_produced(), old_generated);
    EXPECT_DOUBLE_EQ(gas->cumulative_submerged_bubble_hydrogen_escaped(), old_escape);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.solver->free_surface_history().size(), old_history);
    EXPECT_EQ(state.solver->planar_ale_diagnostics().rejected_transactions, 1U);
}

TEST(BoussinesqPlanarALETest, MeshMotionLimitFailureLeavesAcceptedStateUntouched)
{
    auto mesh = make_column();
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    auto solver = std::make_unique<Solver>(
        mesh, planar_boundaries(), time_options(Coupling::PISO), linear_options, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->add_temperature_source("motion_limit_heat", 1.0);
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(6);
    options.ale.maximum_level_change = 1.0e-12;
    ASSERT_NE(solver->configure_free_surface(options), nullptr);
    const auto geometry = capture_geometry(*mesh);
    const auto primary = capture_primary(*solver);
    const auto history = solver->free_surface_history().size();

    EXPECT_THROW(solver->step(), std::invalid_argument);
    expect_geometry_restored(*mesh, geometry);
    expect_primary_restored(*solver, primary);
    EXPECT_EQ(solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(solver->time(), 0.0);
    EXPECT_EQ(solver->free_surface_history().size(), history);
    EXPECT_EQ(solver->planar_ale_diagnostics().rejected_transactions, 1U);
}

TEST(BoussinesqPlanarALETest, MomentumTransportPolicyFailureRollsBackAndCanRetry)
{
    auto state = make_case(Coupling::PISO, 1.0, 6);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto history = state.solver->free_surface_history().size();
    const auto valid_options = state.solver->linear_solver_options();
    auto invalid_options = valid_options;
    invalid_options.max_iterations = 0;
    state.solver->set_linear_solver_options(invalid_options);

    EXPECT_THROW(state.solver->step(), std::invalid_argument);
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.solver->free_surface_history().size(), history);

    state.solver->set_linear_solver_options(valid_options);
    state.heat_source->set_enabled(false);
    ASSERT_NO_THROW(state.solver->step());
    EXPECT_EQ(state.solver->step_index(), 1);
}

TEST(BoussinesqPlanarALETest, FailedOuterTrialRestoresGeometryFieldsLedgersAndCanRetry)
{
    auto state = make_case(Coupling::PISO, 1.0, 2, true);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto initial_time = state.solver->time();
    const auto initial_step = state.solver->step_index();
    const auto initial_history = state.solver->free_surface_history().size();
    const auto initial_surface = state.solver->free_surface_diagnostics();
    const auto initial_liquid = state.solver->liquid_mass_inventory().diagnostics();
    const auto initial_liquid_field = capture_owned_values(state.solver->liquid_mass_inventory().cellMassInventory());
    const auto initial_density = capture_owned_values(state.solver->liquid_mass_inventory().pureLiquidDensity());
    const auto initial_material_density = capture_owned_values(state.solver->material_properties().density);
    const auto initial_heat_capacity = capture_owned_values(state.solver->material_properties().specific_heat_capacity);
    const auto initial_clear_level_field = capture_owned_values(state.solver->clear_level());
    const auto initial_pool_level_field = capture_owned_values(state.solver->pool_level());
    const auto initial_headspace_pressure_field = capture_owned_values(state.solver->headspace_pressure());
    const auto initial_pool_occupancy_field = capture_owned_values(state.solver->pool_occupancy());
    const auto initial_absolute_flux = capture_owned_face_values(state.solver->pressure_corrected_face_fluxes());
    const auto initial_relative_flux = capture_owned_face_values(state.solver->mesh_relative_face_fluxes());
    const auto initial_occupancy_error = state.solver->pool_occupancy_volume_error();
    const auto* initial_feedback = state.solver->find_material_feedback_model();
    ASSERT_NE(initial_feedback, nullptr);
    const auto initial_feedback_density = capture_owned_values(initial_feedback->density_feedback());
    const auto initial_feedback_viscosity = capture_owned_values(initial_feedback->viscosity_feedback());

    auto* gas = state.solver->find_radiolytic_gas_model();
    ASSERT_NE(gas, nullptr);
    const auto initial_gas_fields = capture_gas_fields(*gas);
    const auto initial_dissolved = gas->global_dissolved_hydrogen_moles();
    const auto initial_micro = gas->global_microbubble_hydrogen_moles();
    const auto initial_large = gas->global_large_bubble_hydrogen_moles();
    const auto initial_generated = gas->cumulative_hydrogen_produced();
    const auto initial_escape = gas->cumulative_submerged_bubble_hydrogen_escaped();
    const auto initial_pressure_offset = gas->absolute_pressure_offset();
    const auto initial_committed_escape = state.solver->find_free_surface_model()->committedEscapedMoles();

    std::string failure;
    try
    {
        state.solver->step();
        FAIL() << "Expected the bounded planar-ALE trial to fail.";
    }
    catch (const std::runtime_error& error)
    {
        failure = error.what();
    }

    EXPECT_NE(failure.find("outer level/continuity corrector did not converge"), std::string::npos) << failure;
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(state.solver->time(), initial_time);
    EXPECT_EQ(state.solver->step_index(), initial_step);
    EXPECT_EQ(state.solver->free_surface_history().size(), initial_history);

    const auto restored_surface = state.solver->free_surface_diagnostics();
    EXPECT_DOUBLE_EQ(restored_surface.clear_level, initial_surface.clear_level);
    EXPECT_DOUBLE_EQ(restored_surface.pool_level, initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(restored_surface.liquid_volume, initial_surface.liquid_volume);
    EXPECT_DOUBLE_EQ(restored_surface.pool_volume, initial_surface.pool_volume);
    EXPECT_DOUBLE_EQ(restored_surface.headspace.pressure, initial_surface.headspace.pressure);
    EXPECT_EQ(state.solver->find_free_surface_model()->committedEscapedMoles(), initial_committed_escape);

    const auto restored_liquid = state.solver->liquid_mass_inventory().diagnostics();
    EXPECT_DOUBLE_EQ(restored_liquid.total_mass, initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(restored_liquid.liquid_volume, initial_liquid.liquid_volume);
    EXPECT_DOUBLE_EQ(restored_liquid.step_mass_balance_residual, initial_liquid.step_mass_balance_residual);
    const auto restored_liquid_field = capture_owned_values(state.solver->liquid_mass_inventory().cellMassInventory());
    const auto restored_density = capture_owned_values(state.solver->liquid_mass_inventory().pureLiquidDensity());
    ASSERT_EQ(restored_liquid_field.size(), initial_liquid_field.size());
    ASSERT_EQ(restored_density.size(), initial_density.size());
    for (size_t owned = 0; owned < initial_liquid_field.size(); ++owned)
    {
        EXPECT_DOUBLE_EQ(restored_liquid_field[owned], initial_liquid_field[owned]);
        EXPECT_DOUBLE_EQ(restored_density[owned], initial_density[owned]);
    }
    expect_owned_values(state.solver->material_properties().density, initial_material_density);
    expect_owned_values(state.solver->material_properties().specific_heat_capacity, initial_heat_capacity);
    expect_owned_values(state.solver->clear_level(), initial_clear_level_field);
    expect_owned_values(state.solver->pool_level(), initial_pool_level_field);
    expect_owned_values(state.solver->headspace_pressure(), initial_headspace_pressure_field);
    expect_owned_values(state.solver->pool_occupancy(), initial_pool_occupancy_field);
    expect_owned_face_values(state.solver->pressure_corrected_face_fluxes(), initial_absolute_flux);
    expect_owned_face_values(state.solver->mesh_relative_face_fluxes(), initial_relative_flux);
    EXPECT_DOUBLE_EQ(state.solver->pool_occupancy_volume_error(), initial_occupancy_error);
    const auto* restored_feedback = state.solver->find_material_feedback_model();
    ASSERT_NE(restored_feedback, nullptr);
    expect_owned_values(restored_feedback->density_feedback(), initial_feedback_density);
    expect_owned_values(restored_feedback->viscosity_feedback(), initial_feedback_viscosity);

    expect_gas_fields_restored(*gas, initial_gas_fields);
    EXPECT_DOUBLE_EQ(gas->global_dissolved_hydrogen_moles(), initial_dissolved);
    EXPECT_DOUBLE_EQ(gas->global_microbubble_hydrogen_moles(), initial_micro);
    EXPECT_DOUBLE_EQ(gas->global_large_bubble_hydrogen_moles(), initial_large);
    EXPECT_DOUBLE_EQ(gas->cumulative_hydrogen_produced(), initial_generated);
    EXPECT_DOUBLE_EQ(gas->cumulative_submerged_bubble_hydrogen_escaped(), initial_escape);
    EXPECT_DOUBLE_EQ(gas->absolute_pressure_offset(), initial_pressure_offset);

    const auto& rejected = state.solver->planar_ale_diagnostics();
    EXPECT_EQ(rejected.rejected_transactions, 1U);
    EXPECT_EQ(rejected.last_rejection_reason, failure);

    state.heat_source->set_enabled(false);
    ASSERT_NO_THROW(state.solver->step());

    EXPECT_EQ(state.solver->step_index(), initial_step + 1);
    EXPECT_NEAR(state.solver->time(), initial_time + 1.0e-2, 1.0e-15);
    EXPECT_EQ(state.solver->free_surface_history().size(), initial_history + 1);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(state.solver->liquid_mass_inventory().totalMass(), initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(gas->global_dissolved_hydrogen_moles(), initial_dissolved);
    EXPECT_DOUBLE_EQ(gas->cumulative_submerged_bubble_hydrogen_escaped(), initial_escape);
    EXPECT_EQ(state.solver->planar_ale_diagnostics().rejected_transactions, 1U);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    expect_zero_relative_top_flux(state);

    // A later rejected step must retain the preceding accepted numerical
    // report; begin_step() must not make a cleared trial report observable.
    const auto accepted_statistics = state.solver->last_step_statistics();
    const auto accepted_volume_residuals = state.solver->last_volume_continuity_residuals();
    auto invalid_pressure_options = state.solver->pressure_linear_solver_options();
    invalid_pressure_options.max_iterations = 0;
    state.solver->set_pressure_linear_solver_options(invalid_pressure_options);
    EXPECT_THROW(state.solver->step(), std::invalid_argument);
    const auto& restored_statistics = state.solver->last_step_statistics();
    EXPECT_EQ(restored_statistics.converged, accepted_statistics.converged);
    EXPECT_EQ(restored_statistics.nonlinear_iterations, accepted_statistics.nonlinear_iterations);
    EXPECT_EQ(restored_statistics.linear_solves, accepted_statistics.linear_solves);
    EXPECT_EQ(restored_statistics.krylov_iterations, accepted_statistics.krylov_iterations);
    EXPECT_DOUBLE_EQ(restored_statistics.achieved_tolerance, accepted_statistics.achieved_tolerance);
    EXPECT_DOUBLE_EQ(restored_statistics.momentum, accepted_statistics.momentum);
    EXPECT_DOUBLE_EQ(restored_statistics.pressure, accepted_statistics.pressure);
    EXPECT_DOUBLE_EQ(restored_statistics.temperature, accepted_statistics.temperature);
    EXPECT_DOUBLE_EQ(restored_statistics.continuity, accepted_statistics.continuity);
    const auto& restored_volume_residuals = state.solver->last_volume_continuity_residuals();
    EXPECT_DOUBLE_EQ(restored_volume_residuals.l2, accepted_volume_residuals.l2);
    EXPECT_DOUBLE_EQ(restored_volume_residuals.maximum, accepted_volume_residuals.maximum);
    EXPECT_DOUBLE_EQ(restored_volume_residuals.normalized_l2, accepted_volume_residuals.normalized_l2);
    EXPECT_DOUBLE_EQ(restored_volume_residuals.normalization, accepted_volume_residuals.normalization);
}

TEST(BoussinesqPlanarALETest, PressureSolveFailureInsideTrialRollsBackAndCanRetry)
{
    auto state = make_case(Coupling::PISO, 0.0, 3);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto initial_time = state.solver->time();
    const auto initial_step = state.solver->step_index();
    const auto initial_history = state.solver->free_surface_history().size();
    const auto initial_surface = state.solver->free_surface_diagnostics();
    const auto initial_liquid = state.solver->liquid_mass_inventory().diagnostics();
    const auto initial_liquid_field = capture_owned_values(state.solver->liquid_mass_inventory().cellMassInventory());
    const auto initial_density = capture_owned_values(state.solver->liquid_mass_inventory().pureLiquidDensity());
    const auto initial_clear_level_field = capture_owned_values(state.solver->clear_level());
    const auto initial_pool_level_field = capture_owned_values(state.solver->pool_level());
    const auto initial_headspace_pressure_field = capture_owned_values(state.solver->headspace_pressure());
    const auto initial_pool_occupancy_field = capture_owned_values(state.solver->pool_occupancy());
    const auto initial_absolute_flux = capture_owned_face_values(state.solver->pressure_corrected_face_fluxes());

    const auto valid_pressure_options = state.solver->pressure_linear_solver_options();
    auto invalid_pressure_options = valid_pressure_options;
    invalid_pressure_options.max_iterations = 0;
    state.solver->set_pressure_linear_solver_options(invalid_pressure_options);

    std::string failure;
    try
    {
        state.solver->step();
        FAIL() << "Expected the in-trial pressure solve to reject its policy.";
    }
    catch (const std::invalid_argument& error)
    {
        failure = error.what();
    }

    EXPECT_NE(failure.find("BelosLinearSolver requires positive maximum iterations"), std::string::npos) << failure;
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(state.solver->time(), initial_time);
    EXPECT_EQ(state.solver->step_index(), initial_step);
    EXPECT_EQ(state.solver->free_surface_history().size(), initial_history);

    const auto restored_surface = state.solver->free_surface_diagnostics();
    EXPECT_DOUBLE_EQ(restored_surface.clear_level, initial_surface.clear_level);
    EXPECT_DOUBLE_EQ(restored_surface.pool_level, initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(restored_surface.liquid_volume, initial_surface.liquid_volume);
    EXPECT_DOUBLE_EQ(restored_surface.pool_volume, initial_surface.pool_volume);
    EXPECT_DOUBLE_EQ(restored_surface.headspace.pressure, initial_surface.headspace.pressure);
    const auto restored_liquid = state.solver->liquid_mass_inventory().diagnostics();
    EXPECT_DOUBLE_EQ(restored_liquid.total_mass, initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(restored_liquid.liquid_volume, initial_liquid.liquid_volume);
    expect_owned_values(state.solver->liquid_mass_inventory().cellMassInventory(), initial_liquid_field);
    expect_owned_values(state.solver->liquid_mass_inventory().pureLiquidDensity(), initial_density);
    expect_owned_values(state.solver->clear_level(), initial_clear_level_field);
    expect_owned_values(state.solver->pool_level(), initial_pool_level_field);
    expect_owned_values(state.solver->headspace_pressure(), initial_headspace_pressure_field);
    expect_owned_values(state.solver->pool_occupancy(), initial_pool_occupancy_field);
    expect_owned_face_values(state.solver->pressure_corrected_face_fluxes(), initial_absolute_flux);
    EXPECT_EQ(state.solver->planar_ale_diagnostics().rejected_transactions, 1U);
    EXPECT_EQ(state.solver->planar_ale_diagnostics().last_rejection_reason, failure);

    state.solver->set_pressure_linear_solver_options(valid_pressure_options);
    ASSERT_NO_THROW(state.solver->step());

    EXPECT_EQ(state.solver->step_index(), initial_step + 1);
    EXPECT_NEAR(state.solver->time(), initial_time + 1.0e-2, 1.0e-15);
    EXPECT_EQ(state.solver->free_surface_history().size(), initial_history + 1);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(state.solver->liquid_mass_inventory().totalMass(), initial_liquid.total_mass);
    EXPECT_EQ(state.solver->planar_ale_diagnostics().rejected_transactions, 1U);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    expect_zero_relative_top_flux(state);
}

enum class MatrixMeshFamily
{
    Cartesian,
    Cylindrical,
    SemiStructured
};

struct MatrixCase
{
    MatrixMeshFamily family = MatrixMeshFamily::Cartesian;
    SimpleFluid::Dimension axis = SimpleFluid::Dimension::Z;
    const char* moving_boundary = "zmax";
    double cross_section_area = 1.0;
};

SimpleFluid::SP<Handle> make_matrix_mesh(const MatrixCase& test_case)
{
    switch (test_case.family)
    {
        case MatrixMeshFamily::Cartesian: {
            SimpleFluid::Vec3D<SimpleFluid::ArrReal> edges{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}};
            edges[static_cast<size_t>(test_case.axis)] = {0.0, 0.5, 1.0};
            return std::make_shared<Handle>(
                std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(std::move(edges)));
        }
        case MatrixMeshFamily::Cylindrical:
            return std::make_shared<Handle>(
                std::make_shared<SimpleFluid::Meshes::OrthogonalCylindrial3D>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{
                    {{1.0, 2.0}, {0.0, 0.5 * std::numbers::pi}, {0.0, 0.5, 1.0}}}));
        case MatrixMeshFamily::SemiStructured:
            return std::make_shared<Handle>(std::make_shared<SimpleFluid::Meshes::SemiStructuredXY_Z>(
                SimpleFluid::Arr<SimpleFluid::Meshes::SemiStructuredXY_Z::Vec3>{
                    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
                SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2, 3}}, SimpleFluid::ArrReal{0.0, 0.5, 1.0}));
    }
    throw std::logic_error("Unknown planar-ALE support-matrix mesh family.");
}

SimpleFluid::BoundaryConditionSet matrix_boundaries(const Handle& mesh, const std::string& moving_boundary)
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        (void) batch;
        const auto name = mesh.boundary_batch_name(batch_id);
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
        boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }
    boundaries.velocity[moving_boundary] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    boundaries.pressure[moving_boundary] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    return boundaries;
}

SimpleFluid::TimeStepperOptions matrix_time_options(SimpleFluid::Dimension gravity_axis)
{
    auto options = time_options(Coupling::PISO);
    options.gravity_x = 0.0;
    options.gravity_y = 0.0;
    options.gravity_z = 0.0;
    switch (gravity_axis)
    {
        case SimpleFluid::Dimension::X:
            options.gravity_x = -9.81;
            break;
        case SimpleFluid::Dimension::Y:
            options.gravity_y = -9.81;
            break;
        case SimpleFluid::Dimension::Z:
            options.gravity_z = -9.81;
            break;
    }
    return options;
}

SimpleFluid::FreeSurfaceOptions matrix_free_surface_options(const MatrixCase& test_case)
{
    auto options = free_surface_options(5);
    options.gravity_axis = test_case.axis;
    options.initial_liquid_volume = test_case.cross_section_area;
    options.vessel.cross_section_area = test_case.cross_section_area;
    options.vessel.total_internal_volume = 2.0 * test_case.cross_section_area;
    options.ale.top_boundary = test_case.moving_boundary;
    return options;
}

struct MatrixConfiguredCase
{
    SimpleFluid::SP<Handle> mesh;
    std::unique_ptr<Solver> solver;
    MatrixCase definition;
};

MatrixConfiguredCase configure_matrix_case(const MatrixCase& test_case, double power_density)
{
    auto mesh = make_matrix_mesh(test_case);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    linear_options.max_iterations = 500;
    auto solver = std::make_unique<Solver>(mesh, matrix_boundaries(*mesh, test_case.moving_boundary),
        matrix_time_options(test_case.axis), linear_options, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->add_temperature_source("matrix_uniform_heat", power_density);

    Handle::Vec3 direction{};
    direction.component(static_cast<size_t>(test_case.axis)) = 1.0;
    solver->initialize_linear_temperature(direction, 300.0, 300.0);
    if (solver->configure_free_surface(matrix_free_surface_options(test_case)) == nullptr)
    {
        throw std::logic_error("The planar-ALE support-matrix case did not create a model.");
    }
    return {std::move(mesh), std::move(solver), test_case};
}

double boundary_elevation(const Handle& mesh, const MatrixCase& test_case)
{
    const auto axis = static_cast<size_t>(test_case.axis);
    double local_minimum = std::numeric_limits<double>::infinity();
    double local_maximum = -std::numeric_limits<double>::infinity();
    int local_faces = 0;
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        if (mesh.boundary_batch_name(batch_id) != test_case.moving_boundary)
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (!mesh.is_owned_face(face_lid))
            {
                continue;
            }
            const auto elevation = mesh.face_centroid(face_lid).component(axis);
            local_minimum = std::min(local_minimum, elevation);
            local_maximum = std::max(local_maximum, elevation);
            ++local_faces;
        }
    }

    double global_minimum{};
    double global_maximum{};
    int global_faces = 0;
    const auto communicator = mesh.owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_minimum, &global_minimum);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_maximum, &global_maximum);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    if (global_faces == 0 || !std::isfinite(global_minimum) || !std::isfinite(global_maximum))
    {
        throw std::logic_error("The support-matrix moving boundary has no owned face.");
    }
    EXPECT_NEAR(global_minimum, global_maximum, 1.0e-14);
    return global_maximum;
}

void expect_matrix_top_kinematics(const MatrixConfiguredCase& state, double old_surface_elevation, double time_step)
{
    const auto& absolute = state.solver->pressure_corrected_face_fluxes();
    const auto& relative = state.solver->mesh_relative_face_fluxes();
    const auto new_surface_elevation = boundary_elevation(*state.mesh, state.definition);
    const auto speed = (new_surface_elevation - old_surface_elevation) / time_step;

    double local_absolute_error = 0.0;
    int local_relative_not_exactly_zero = 0;
    int local_faces = 0;
    for (const auto& [batch_id, batch] : state.mesh->boundary_batches())
    {
        if (state.mesh->boundary_batch_name(batch_id) != state.definition.moving_boundary)
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (!absolute.is_owned_face(face_lid))
            {
                continue;
            }
            const auto expected_mesh_flux = speed * state.mesh->face_area(face_lid);
            local_absolute_error =
                std::max(local_absolute_error, std::abs(absolute.value(face_lid) - expected_mesh_flux));
            local_relative_not_exactly_zero = local_relative_not_exactly_zero || relative.value(face_lid) != 0.0;
            ++local_faces;
        }
    }

    double global_absolute_error = 0.0;
    int any_relative_not_exactly_zero = 0;
    int global_faces = 0;
    const auto communicator = state.mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_absolute_error, &global_absolute_error);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1, &local_relative_not_exactly_zero, &any_relative_not_exactly_zero);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    EXPECT_GT(global_faces, 0);
    EXPECT_LE(global_absolute_error, 1.0e-12);
    EXPECT_EQ(any_relative_not_exactly_zero, 0);
}

void exercise_supported_matrix_case(const MatrixCase& test_case)
{
    if (test_case.family == MatrixMeshFamily::SemiStructured && Tpetra::getDefaultComm()->getSize() != 1)
    {
        GTEST_SKIP() << "SemiStructuredXY_Z planar ALE is intentionally "
                        "serial-only.";
    }

    constexpr double power_density = 1.0e-3;
    constexpr double time_step = 1.0e-2;
    auto state = configure_matrix_case(test_case, power_density);
    const auto old_epoch = state.mesh->geometry_epoch();
    const auto old_surface = boundary_elevation(*state.mesh, test_case);
    const auto old_volume = global_mesh_volume(*state.mesh);
    const auto old_mass = state.solver->liquid_mass_inventory().totalMass();

    ASSERT_TRUE(state.solver->planar_ale_enabled());
    ASSERT_NO_THROW(state.solver->step());

    const auto new_surface = boundary_elevation(*state.mesh, test_case);
    const auto new_volume = global_mesh_volume(*state.mesh);
    const auto surface = state.solver->free_surface_diagnostics();
    const auto& diagnostics = state.solver->planar_ale_diagnostics();
    record_ale_residuals(diagnostics);
    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_NEAR(state.solver->time(), time_step, 1.0e-15);
    EXPECT_GT(state.mesh->geometry_epoch(), old_epoch);
    EXPECT_GT(new_surface, old_surface);
    EXPECT_GT(new_volume, old_volume);
    EXPECT_NEAR(surface.pool_level, new_surface, 2.0e-11);
    EXPECT_NEAR(surface.pool_volume, new_volume, 2.0e-11);
    EXPECT_NEAR(new_volume, test_case.cross_section_area * new_surface, 2.0e-11);
    EXPECT_NEAR(state.solver->liquid_mass_inventory().totalMass(), old_mass, 2.0e-11);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    EXPECT_TRUE(diagnostics.initialized);
    EXPECT_GE(diagnostics.outer_correctors, 2);
    EXPECT_LE(diagnostics.outer_correctors, 5);
    EXPECT_NEAR(diagnostics.old_mesh_volume, old_volume, 2.0e-13);
    EXPECT_NEAR(diagnostics.new_mesh_volume, new_volume, 2.0e-13);
    EXPECT_NEAR(diagnostics.mesh_vessel_mismatch, 0.0, 2.0e-11);
    EXPECT_NEAR(diagnostics.maximum_gcl_residual, 0.0, 2.0e-12);
    EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 2.0e-10);
    EXPECT_LE(diagnostics.continuity.maximum, 5.0e-10);
    EXPECT_LE(state.solver->last_volume_continuity_residuals().maximum, 5.0e-10);
    expect_matrix_top_kinematics(state, old_surface, time_step);
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsCartesianMotionAndGravityAlongX)
{
    exercise_supported_matrix_case({MatrixMeshFamily::Cartesian, SimpleFluid::Dimension::X, "xmax", 1.0});
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsCartesianMotionAndGravityAlongY)
{
    exercise_supported_matrix_case({MatrixMeshFamily::Cartesian, SimpleFluid::Dimension::Y, "ymax", 1.0});
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsCylindricalAxialMotionAndGravityAlongZ)
{
    constexpr double area = 0.75 * std::numbers::pi;
    exercise_supported_matrix_case({MatrixMeshFamily::Cylindrical, SimpleFluid::Dimension::Z, "zmax", area});
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsSerialSemiStructuredAxialMotionAndGravityAlongZ)
{
    exercise_supported_matrix_case({MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Z, "zmax", 1.0});
}

void expect_nonaxial_family_rejected(MatrixCase test_case)
{
    if (test_case.family == MatrixMeshFamily::SemiStructured && Tpetra::getDefaultComm()->getSize() != 1)
    {
        GTEST_SKIP() << "The separate multi-rank test covers the earlier "
                        "SemiStructuredXY_Z distribution rejection.";
    }

    auto mesh = make_matrix_mesh(test_case);
    SimpleFluid::LinearSolverOptions linear_options;
    auto solver = std::make_unique<Solver>(mesh, matrix_boundaries(*mesh, test_case.moving_boundary),
        matrix_time_options(test_case.axis), linear_options, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);

    std::string diagnostic;
    try
    {
        static_cast<void>(solver->configure_free_surface(matrix_free_surface_options(test_case)));
        FAIL() << "Expected a nonaxial planar-ALE mesh family to be rejected.";
    }
    catch (const std::invalid_argument& error)
    {
        diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("axial"), std::string::npos) << diagnostic;
    EXPECT_FALSE(solver->planar_ale_enabled());
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsNonaxialCylindricalMotion)
{
    constexpr double area = 0.75 * std::numbers::pi;
    expect_nonaxial_family_rejected({MatrixMeshFamily::Cylindrical, SimpleFluid::Dimension::X, "zmax", area});
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsNonaxialSemiStructuredMotion)
{
    expect_nonaxial_family_rejected({MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Y, "zmax", 1.0});
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsUnsupportedVesselLiquidRangeAndHeadspaceOwnership)
{
    expect_free_surface_option_rejected(
        [](auto& options)
        {
            options.vessel.mode = SimpleFluid::VesselVolumeMapMode::Tabulated;
            options.vessel.height_table = {0.0, 1.0, 2.0};
            options.vessel.volume_table = {0.0, 1.0, 2.0};
        });
    expect_free_surface_option_rejected(
        [](auto& options) { options.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::GlobalConstantMass; });
    expect_free_surface_option_rejected(
        [](auto& options) { options.range_policy = SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport; });
    expect_free_surface_option_rejected(
        [](auto& options)
        {
            options.headspace.mode = SimpleFluid::HeadspaceMode::Closed;
            options.headspace.total_internal_volume = 2.0;
        });
    expect_free_surface_option_rejected(
        [](auto& options) { options.headspace.temperature_mode = SimpleFluid::HeadspaceTemperatureMode::BulkLiquid; });
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsUnmigratedOptionalTransportModelsBeforeMotion)
{
    expect_active_model_rejected(
        [](Solver& solver) { solver.configure_scalar_void_fraction(SimpleFluid::ScalarVoidFractionOptions{}); });
    expect_active_model_rejected(
        [](Solver& solver)
        {
            SimpleFluid::BoilingSourceOptions options;
            options.enable_bulk_boiling = true;
            solver.configure_boiling_source(options);
        });
    expect_active_model_rejected(
        [](Solver& solver)
        {
            SimpleFluid::DelayedNeutronPrecursorOptions options;
            options.group_count = 1;
            solver.configure_precursors(options);
        });
    expect_active_model_rejected(
        [](Solver& solver)
        {
            SimpleFluid::TurbulenceModelOptions options;
            options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
            solver.configure_turbulence(options);
        });
    expect_active_model_rejected([](Solver& solver) { solver.set_material_updater([](const auto&, auto&) {}); });
    expect_active_model_rejected([](Solver& solver)
        { solver.add_temperature_source("dynamic_source", 0.0).set_updater([](const auto&, auto&) {}); });
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsNonadiabaticAndOpenNonmovingBoundaries)
{
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        auto boundaries = planar_boundaries();
        if (scenario == 0)
        {
            boundaries.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 300.0};
        }
        else
        {
            boundaries.velocity["xmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, Handle::Vec3{1.0, 0.0, 0.0}};
        }
        auto mesh = make_column();
        auto solver = std::make_unique<Solver>(
            mesh, boundaries, time_options(Coupling::PISO), SimpleFluid::LinearSolverOptions{}, physical_options());
        solver->configure_material_feedback(material_feedback_options());
        solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
        EXPECT_THROW(static_cast<void>(solver->configure_free_surface(free_surface_options(4))), std::invalid_argument);
        EXPECT_FALSE(solver->planar_ale_enabled());
        EXPECT_EQ(solver->find_free_surface_model(), nullptr);
    }
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsShengRadiolysisWithoutFissionSourceAtSetup)
{
    auto mesh = make_column();
    auto solver = std::make_unique<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->configure_radiolytic_gas(gas_options());
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    EXPECT_THROW(static_cast<void>(solver->configure_free_surface(free_surface_options(4))), std::invalid_argument);
    EXPECT_FALSE(solver->planar_ale_enabled());
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsShengRadiolysisWithOnlyOneOuterCorrector)
{
    auto mesh = make_column();
    auto solver = std::make_unique<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->add_fission_power_source().initialize_constant(0.0);
    solver->configure_radiolytic_gas(gas_options());
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(1);
    EXPECT_THROW(static_cast<void>(solver->configure_free_surface(options)), std::invalid_argument);
    EXPECT_FALSE(solver->planar_ale_enabled());
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsGeometryDependentFissionProfileBeforeMotion)
{
    auto state = make_case(Coupling::PISO, 0.0, 5, true);
    auto* fission = state.solver->find_fission_power_source();
    ASSERT_NE(fission, nullptr);
    fission->initialize_gaussian(1.0, Handle::Vec3{0.5, 0.5, 0.0}, Handle::Vec3{1.0, 1.0, 0.5});
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);

    EXPECT_THROW(state.solver->step(), std::invalid_argument);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.mesh->geometry_epoch(), geometry.epoch);
    expect_primary_restored(*state.solver, primary);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsGeometryDependentThermalFissionProfileWithoutRadiolysis)
{
    auto state = make_case(Coupling::PISO, 0.0, 5);
    auto& fission = state.solver->add_fission_power_source();
    fission.initialize_gaussian(1.0, Handle::Vec3{0.5, 0.5, 0.0}, Handle::Vec3{1.0, 1.0, 0.5});
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);

    EXPECT_THROW(state.solver->step(), std::invalid_argument);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.mesh->geometry_epoch(), geometry.epoch);
    expect_primary_restored(*state.solver, primary);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsGravityThatIsNotInwardAndAxial)
{
    for (const auto gravity : {Handle::Vec3{-1.0, 0.0, 0.0}, Handle::Vec3{0.0, 0.0, 1.0}})
    {
        auto mesh = make_column();
        auto options = time_options(Coupling::PISO);
        options.gravity_x = gravity.x;
        options.gravity_y = gravity.y;
        options.gravity_z = gravity.z;
        auto solver = std::make_unique<Solver>(
            mesh, planar_boundaries(), options, SimpleFluid::LinearSolverOptions{}, physical_options());
        solver->configure_material_feedback(material_feedback_options());
        solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
        EXPECT_THROW(static_cast<void>(solver->configure_free_surface(free_surface_options(4))), std::invalid_argument);
        EXPECT_FALSE(solver->planar_ale_enabled());
    }
}

TEST(BoussinesqPlanarALESupportMatrixTest, ConfigurationRejectsGeometryBeforePrimaryFieldsOrHistory)
{
    auto mesh = make_column();
    auto solver = std::make_unique<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    auto invalid_geometry = free_surface_options(4);
    invalid_geometry.vessel.cross_section_area = 2.0;
    invalid_geometry.vessel.total_internal_volume = 4.0;
    const auto initial_primary = capture_primary(*solver);
    EXPECT_THROW(static_cast<void>(solver->configure_free_surface(invalid_geometry)), std::invalid_argument);
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
    EXPECT_TRUE(solver->free_surface_history().empty());
    EXPECT_FALSE(solver->planar_ale_enabled());
    expect_primary_restored(*solver, initial_primary);
}

TEST(BoussinesqPlanarALESupportMatrixTest, CollectivelyRejectsRankDivergentRuntimeChoicesBeforeMotion)
{
    const auto communicator = Tpetra::getDefaultComm();
    if (communicator->getSize() != 2)
    {
        GTEST_SKIP() << "This collective preflight check requires two ranks.";
    }
    auto mesh = make_column();
    const auto coupling = communicator->getRank() == 0 ? Coupling::SIMPLE : Coupling::CoupledKrylov;
    auto solver = std::make_unique<Solver>(
        mesh, planar_boundaries(), time_options(coupling), SimpleFluid::LinearSolverOptions{}, physical_options());
    solver->configure_material_feedback(material_feedback_options());
    solver->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    EXPECT_THROW(static_cast<void>(solver->configure_free_surface(free_surface_options(4))), std::invalid_argument);
    EXPECT_FALSE(solver->planar_ale_enabled());
    EXPECT_EQ(solver->find_free_surface_model(), nullptr);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsMultiRankSemiStructuredBeforeSolverConfiguration)
{
    if (Tpetra::getDefaultComm()->getSize() == 1)
    {
        GTEST_SKIP() << "This rejection requires a multi-rank communicator.";
    }

    const MatrixCase test_case{MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Z, "zmax", 1.0};
    std::string diagnostic;
    try
    {
        static_cast<void>(make_matrix_mesh(test_case));
        FAIL() << "Expected multi-rank SemiStructuredXY_Z construction to fail.";
    }
    catch (const std::runtime_error& error)
    {
        diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("does not yet support multi-rank"), std::string::npos) << diagnostic;
}

} // namespace
