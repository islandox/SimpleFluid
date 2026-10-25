/**
 * @file testBoussinesqPlanarALE.cc
 * @brief Solver-level acceptance and rollback tests for constrained planar ALE.
 */

#include "solvers/unitTests/CoupledBackendTestSupport.hh"
#include <gtest/gtest.h>

#include "ReactingMixtureTestSupport.hh"
#include "SurfaceGasFixtureSupport.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
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
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
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

    auto& mutable_pressure_corrected_face_fluxes() { return SimpleFluid::FluidSolver<Pack>::projected_face_fluxes(); }

    auto solution_topology() const { return SimpleFluid::FluidSolver<Pack>::fluid_solution_writer().topology_handle(); }
};
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using FaceFlux = SimpleFluid::ScalarFaceFieldStored<Pack>;
using FaceVelocity = SimpleFluid::VectorFaceFieldStored<Pack>;
using Source = SimpleFluid::VolumetricScalarSource<Pack, Handle>;
using Coupling = SimpleFluid::PressureVelocityCoupling;
namespace SurfaceGas = SimpleFluid::Testing::SurfaceGas;

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

SimpleFluid::ReactingGasOptions gas_options(const ScalarField* prescribed = nullptr)
{
    auto options = SurfaceGas::options({}, prescribed ? SurfaceGas::prescribed_power(prescribed) : SurfaceGas::Power{});
    options.pressure_mode = SimpleFluid::ReactingPressureMode::Constant;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ZeroSlip;
    options.free_surface_patches = {"zmax"};
    return options;
}

struct ConfiguredCase
{
    SimpleFluid::SP<Handle> mesh;
    std::shared_ptr<Solver> solver;
    std::unique_ptr<SimpleFluid::MultiphysicsSolverCoupler<Pack>> coupling;
    Source* heat_source = nullptr;
};

ConfiguredCase make_case(Coupling coupling, double power_density, int maximum_correctors, bool include_gas = false,
    double bubble_slip_velocity = 0.0, bool zero_gas_tolerance = false,
    SimpleFluid::ReactingPressureMode pressure_mode = SimpleFluid::ReactingPressureMode::Constant,
    bool use_celata_slip = false,
    SimpleFluid::CoupledOperatorBackend backend = SimpleFluid::CoupledOperatorBackend::Assembled,
    SimpleFluid::CoupledWorkspacePolicy workspace = SimpleFluid::CoupledWorkspacePolicy::CachedProducts,
    SimpleFluid::FVM::CellGradientScheme gradient = SimpleFluid::FVM::CellGradientScheme::LeastSquares,
    bool track_donor = false,
    SimpleFluid::BubbleMassTransferMode mass_transfer_mode = SimpleFluid::BubbleMassTransferMode::Hughmark)
{
    auto mesh = make_column();
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    linear_options.max_iterations = 500;
    auto options = time_options(coupling);
    options.coupled_operator_backend = backend;
    options.coupled_workspace_policy = workspace;
    options.pressure_gradient_scheme = gradient;
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), options, linear_options, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);

    solver_coupling->configure_material_feedback(material_feedback_options());
    auto* heat_source = &solver->add_temperature_source("planar_ale_heat", power_density);
    if (include_gas)
    {
        solver_coupling->add_prescribed_heat_source().initialize_constant(0.0);
        auto gas = gas_options(&solver_coupling->find_prescribed_heat_source()->field());
        gas.pressure_mode = pressure_mode;
        gas.mass_transfer_mode = mass_transfer_mode;
        if (bubble_slip_velocity > 0.0)
        {
            gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
            gas.constant_slip_velocity = bubble_slip_velocity;
        }
        if (use_celata_slip)
        {
            gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::Celata2007;
        }
        auto& gas_model = solver_coupling->configure_reacting_gas(gas);
        (void) gas_model;
        (void) track_donor; // Typed histories retain actual finite inventories, never a fabricated deficit.
    }
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto surface_options = free_surface_options(maximum_correctors);
    if (zero_gas_tolerance)
    {
        surface_options.coupling.gas_absolute_tolerance = 0.0;
        surface_options.coupling.gas_relative_tolerance = 0.0;
    }
    if (solver_coupling->configure_free_surface(surface_options) == nullptr)
    {
        throw std::logic_error("The planar-ALE test fixture did not create a free-surface model.");
    }
    return {std::move(mesh), std::move(solver), std::move(solver_coupling), heat_source};
}

template<class Mutator> void expect_free_surface_option_rejected(Mutator&& mutator)
{
    auto mesh = make_column();
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(4);
    std::forward<Mutator>(mutator)(options);
    EXPECT_THROW(static_cast<void>(solver_coupling->configure_free_surface(options)), std::invalid_argument);
    EXPECT_FALSE(solver_coupling->planar_ale_enabled());
    EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
}

template<class Configure> void expect_active_model_rejected(Configure&& configure)
{
    auto mesh = make_column();
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    std::forward<Configure>(configure)(*solver_coupling);
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    EXPECT_THROW(
        static_cast<void>(solver_coupling->configure_free_surface(free_surface_options(4))), std::invalid_argument);
    EXPECT_FALSE(solver_coupling->planar_ale_enabled());
    EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
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
    const auto& flux = state.coupling->mesh_relative_face_fluxes();
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

void record_ale_residuals(const SimpleFluid::MultiphysicsSolverCoupler<Pack>::PlanarALEStepDiagnostics& diagnostics)
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

FieldValues capture_gas_fields(const SimpleFluid::MultiphysicsSolverCoupler<Pack>::reacting_gas_model_type& gas)
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

void expect_gas_fields_restored(
    const SimpleFluid::MultiphysicsSolverCoupler<Pack>::reacting_gas_model_type& gas, const FieldValues& expected)
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
    const auto initial_mass = state.coupling->liquid_mass_inventory().totalMass();
    const auto initial_history = state.coupling->free_surface_history().size();

    ASSERT_TRUE(state.coupling->planar_ale_enabled());
    ASSERT_NO_THROW(state.coupling->step());

    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_NEAR(state.solver->time(), 1.0e-2, 1.0e-15);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history + 1);
    EXPECT_GT(state.mesh->geometry_epoch(), initial_epoch);
    EXPECT_DOUBLE_EQ(global_mesh_volume(*state.mesh), initial_volume);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), initial_top);
    EXPECT_DOUBLE_EQ(state.coupling->liquid_mass_inventory().totalMass(), initial_mass);
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().pool_level, initial_top);
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().pool_volume, initial_volume);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    EXPECT_LE(state.solver->last_volume_continuity_residuals().maximum, 1.0e-12);

    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
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

    ASSERT_NO_THROW(state.coupling->step());

    const auto surface = state.coupling->free_surface_diagnostics();
    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
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

    ASSERT_NO_THROW(state.coupling->step());
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
    ASSERT_NO_THROW(state.coupling->step());
    ASSERT_GT(top_elevation(*state.mesh), old_top);
    expect_zero_relative_top_flux(state);

    const auto relative_courant =
        independent_maximum_courant_number(state, state.coupling->mesh_relative_face_fluxes());
    const auto absolute_courant =
        independent_maximum_courant_number(state, state.solver->pressure_corrected_face_fluxes());
    ASSERT_GT(absolute_courant, 1.0e-9);
    EXPECT_LT(relative_courant, 1.0e-11);
    EXPECT_LT(relative_courant, absolute_courant * 1.0e-3);
    EXPECT_NEAR(state.coupling->maximum_courant_number(), relative_courant, 1.0e-15);

    const auto& base_solver = static_cast<const SimpleFluid::FluidSolver<Pack>&>(*state.solver);
    // The core keeps its standalone absolute-flux diagnostic; ALE diagnostics belong to the external controller.
    EXPECT_NEAR(base_solver.maximum_courant_number(), absolute_courant, 1.0e-15);
}

INSTANTIATE_TEST_SUITE_P(PlanarALECouplingModes, BoussinesqPlanarALECouplingTest,
    testing::Values(Coupling::SIMPLE, Coupling::PISO, Coupling::PIMPLE, Coupling::CoupledKrylov), coupling_name);

TEST_P(BoussinesqPlanarALECouplingTest, RemovesAcceptedMeshPoolDefectAfterHeatingStops)
{
    auto state = make_case(GetParam(), 1.0e-4, 40);
    auto options = free_surface_options(40);
    options.ale.level_absolute_tolerance = 2.0e-10;
    options.ale.relaxation = 0.5;
    state.coupling->configure_free_surface(options);
    const auto initial_mass = state.coupling->liquid_mass_inventory().totalMass();
    ASSERT_NO_THROW(state.coupling->step());
    const auto initial_defect =
        state.coupling->free_surface_diagnostics().pool_volume - global_mesh_volume(*state.mesh);
    ASSERT_GT(initial_defect, 1.0e-13);
    ASSERT_LT(initial_defect, options.ale.level_absolute_tolerance);

    state.heat_source->initialize(0.0);
    for (int step = 0; step < 3; ++step)
    {
        SCOPED_TRACE(step);
        ASSERT_NO_THROW(state.coupling->step());
        const auto& diagnostics = state.coupling->planar_ale_diagnostics();
        if (step > 0)
            EXPECT_LT(std::abs(diagnostics.mesh_vessel_mismatch), initial_defect * 0.1);
        EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), initial_mass, 1.0e-12);
        EXPECT_LE(diagnostics.continuity.maximum, 2.0e-10);
        EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-12);
        EXPECT_NEAR(diagnostics.energy_residual, 0.0, 1.0e-8);
        expect_zero_relative_top_flux(state);
    }
}

TEST(BoussinesqPlanarALETest, CorrectsBothDefectSignsAcrossCheckpointReplayAndTimeStepChanges)
{
    for (double defect : {-8.0e-11, 8.0e-11})
    {
        SCOPED_TRACE(defect);
        auto state = make_case(Coupling::PISO, 0.0, 12);
        auto options = free_surface_options(12);
        options.initial_liquid_volume = 1.0 + defect;
        options.ale.level_absolute_tolerance = 2.0e-10;
        state.coupling->configure_free_surface(options);
        auto& solver = *state.solver;
        auto& solver_coupling = *state.coupling;
        auto checkpoint = solver_coupling.create_coupling_checkpoint();
        const auto initial_mass = solver_coupling.liquid_mass_inventory().totalMass();
        const auto initial_volume = global_mesh_volume(*state.mesh);
        const auto initial_defect = solver_coupling.free_surface_diagnostics().pool_volume - initial_volume;
        ASSERT_GT(std::abs(initial_defect), 5.0e-11);
        for (double dt : {0.01, 0.004})
        {
            SCOPED_TRACE(dt);
            solver_coupling.restore_coupling_checkpoint(checkpoint);
            EXPECT_DOUBLE_EQ(global_mesh_volume(*state.mesh), initial_volume);
            solver.set_time_step(dt);
            ASSERT_NO_THROW(solver_coupling.step());
            const auto& diagnostics = solver_coupling.planar_ale_diagnostics();
            EXPECT_NEAR(diagnostics.volume_source.global_mesh_volume_correction, initial_defect / dt, 1.0e-18);
            EXPECT_NEAR(diagnostics.volume_source.global_material_source, 0.0, 1.0e-12);
            EXPECT_NEAR(diagnostics.mesh_vessel_mismatch, 0.0, 2.0e-14);
            EXPECT_NEAR(diagnostics.maximum_gcl_residual, 0.0, 2.0e-13);
            EXPECT_NEAR(solver_coupling.liquid_mass_inventory().totalMass(), initial_mass, 1.0e-12);
            EXPECT_NEAR(diagnostics.energy_residual, 0.0, 1.0e-8);
            expect_zero_relative_top_flux(state);
        }
        solver_coupling.accept_coupling_checkpoint(checkpoint);
    }
}

TEST(BoussinesqPlanarALETest, VolumeBudgetLevelPolicyAcceptsAndCorrectsAnInitialDefect)
{
    constexpr double area = 0.25;
    for (double defect : {-8.0e-11, 8.0e-11})
    {
        SCOPED_TRACE(defect);
        auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, area}, {0.0, 1.0}, {0.0, 0.5, 1.0}}});
        auto mesh = std::make_shared<Handle>(std::move(geometry));
        auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
            SimpleFluid::LinearSolverOptions{}, physical_options());
        auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
        solver_coupling->configure_material_feedback(material_feedback_options());
        solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
        ConfiguredCase state{mesh, std::move(solver), std::move(solver_coupling), nullptr};
        auto options = free_surface_options(12);
        options.initial_liquid_volume = area + defect;
        options.vessel.cross_section_area = area;
        options.vessel.total_internal_volume = 2.0 * area;
        options.ale.level_absolute_tolerance = 0.0;
        options.ale.level_relative_tolerance = 0.0;
        EXPECT_THROW(state.coupling->configure_free_surface(options), std::invalid_argument);
        options.ale.level_tolerance_from_volume = true;
        options.initial_liquid_volume = area + 2.0e-10;
        EXPECT_THROW(state.coupling->configure_free_surface(options), std::invalid_argument);
        options.initial_liquid_volume = area + defect;
        ASSERT_NO_THROW(state.coupling->configure_free_surface(options));
        const auto mass = state.coupling->liquid_mass_inventory().totalMass();
        ASSERT_NO_THROW(state.coupling->step());
        const auto& diagnostics = state.coupling->planar_ale_diagnostics();
        EXPECT_NEAR(diagnostics.mesh_vessel_mismatch, 0.0, 2.0e-14);
        EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), mass, 1.0e-12);
        EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-12);
        expect_zero_relative_top_flux(state);
    }
}

TEST(BoussinesqPlanarALETest, RepeatedWeakHeatingKeepsMeshPoolDefectBounded)
{
    auto state = make_case(Coupling::PISO, 0.0, 40);
    auto options = free_surface_options(40);
    options.ale.level_absolute_tolerance = 2.0e-10;
    options.ale.relaxation = 0.5;
    state.coupling->configure_free_surface(options);
    const auto initial_mass = state.coupling->liquid_mass_inventory().totalMass();
    for (int step = 0; step < 40; ++step)
    {
        SCOPED_TRACE(step);
        state.heat_source->initialize(1.0e-4 * (1 + step % 3));
        ASSERT_NO_THROW(state.coupling->step());
        const auto& diagnostics = state.coupling->planar_ale_diagnostics();
        EXPECT_LT(std::abs(diagnostics.mesh_vessel_mismatch), 5.0e-12);
        EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), initial_mass, 1.0e-12);
        EXPECT_LE(diagnostics.continuity.maximum, 2.0e-10);
        EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-12);
        expect_zero_relative_top_flux(state);
    }
}

TEST(BoussinesqPlanarALETest, AitkenHeatingClosesBalancesAndRestartsHistoryAfterCheckpointReplay)
{
    auto state = make_case(Coupling::PISO, 1.0e-3, 40);
    auto options = free_surface_options(40);
    options.ale.relaxation = 0.5;
    options.ale.aitken = true;
    state.coupling->configure_free_surface(options);
    auto checkpoint = state.coupling->create_coupling_checkpoint();
    const auto initial_mass = state.coupling->liquid_mass_inventory().totalMass();
    const auto geometry = capture_geometry(*state.mesh);
    state.coupling->set_ale_convergence_observer([](const auto& row) {
        if (row.corrector == 2) throw std::runtime_error("Reject an accelerated trial before commit.");
    });
    EXPECT_THROW(state.coupling->step(), std::runtime_error);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    expect_geometry_restored(*state.mesh, geometry);
    state.coupling->set_ale_convergence_observer({});
    std::vector<double> first_history;
    double first_top = 0.0;
    for (int replay = 0; replay < 2; ++replay)
    {
        SCOPED_TRACE(replay);
        if (replay) state.coupling->restore_coupling_checkpoint(checkpoint);
        ASSERT_NO_THROW(state.coupling->step());
        const auto& diagnostics = state.coupling->planar_ale_diagnostics();
        EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), initial_mass, 1.0e-12);
        EXPECT_NEAR(diagnostics.mesh_vessel_mismatch, 0.0, 2.0e-12);
        EXPECT_NEAR(diagnostics.maximum_gcl_residual, 0.0, 1.0e-12);
        EXPECT_NEAR(diagnostics.volume_source.source_pool_closure_residual, 0.0, 1.0e-10);
        EXPECT_LE(diagnostics.continuity.maximum, 3.0e-10);
        EXPECT_LE(std::abs(diagnostics.liquid_mass_residual), 1.0e-12);
        EXPECT_LE(std::abs(diagnostics.energy_residual), 6.1e-6);
        expect_zero_relative_top_flux(state);
        if (!replay)
        {
            first_history = diagnostics.target_change_history;
            first_top = top_elevation(*state.mesh);
        }
        else
        {
            EXPECT_EQ(diagnostics.target_change_history, first_history);
            EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), first_top);
        }
    }
    state.coupling->accept_coupling_checkpoint(checkpoint);
    // A subsequent accepted substep also starts with a fresh accelerator.
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), initial_mass, 1.0e-12);
    EXPECT_LE(state.coupling->planar_ale_diagnostics().continuity.maximum, 3.0e-10);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, CompositeCoupledHeatingAndRollback)
{
    for (const auto selection : SimpleFluid::test::coupled_backend_selections)
        for (const auto gradient :
            {SimpleFluid::FVM::CellGradientScheme::LeastSquares, SimpleFluid::FVM::CellGradientScheme::GaussLinear})
        {
            SCOPED_TRACE(SimpleFluid::test::backend_name(selection));
            SCOPED_TRACE(static_cast<int>(gradient));
            auto state = make_case(Coupling::CoupledKrylov, 1.0e-4, 8, false, 0., false,
                SimpleFluid::ReactingPressureMode::Constant, false, selection.backend, selection.workspace, gradient);
            ASSERT_NO_THROW(state.coupling->step());
            const auto& diagnostics = state.coupling->planar_ale_diagnostics();
            EXPECT_GT(diagnostics.volume_source.global_material_source, 0.);
            EXPECT_LE(diagnostics.continuity.maximum, 3.0e-10);
            EXPECT_NEAR(diagnostics.liquid_mass_residual, 0., 1.0e-12);
            EXPECT_NEAR(
                global_mesh_volume(*state.mesh), state.coupling->free_surface_diagnostics().pool_volume, 2.0e-12);
            expect_zero_relative_top_flux(state);

            auto rejected = make_case(Coupling::CoupledKrylov, 1.0, 2, false, 0., false,
                SimpleFluid::ReactingPressureMode::Constant, false, selection.backend, selection.workspace, gradient);
            const auto volume = global_mesh_volume(*rejected.mesh);
            const auto top = top_elevation(*rejected.mesh);
            EXPECT_THROW(rejected.coupling->step(), std::runtime_error);
            EXPECT_EQ(rejected.solver->step_index(), 0);
            EXPECT_DOUBLE_EQ(global_mesh_volume(*rejected.mesh), volume);
            EXPECT_DOUBLE_EQ(top_elevation(*rejected.mesh), top);
        }
}

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
    const auto old_mass = state.coupling->liquid_mass_inventory().totalMass();
    const auto old_history = state.coupling->free_surface_history().size();

    state.solver->reset_solver_phase_timings();
    ASSERT_NO_THROW(state.coupling->step());

    const auto expected_temperature = initial_temperature + power_density * time_step / (density * heat_capacity);
    const auto expected_density = density * (1.0 - expansion * (expected_temperature - initial_temperature));
    const auto expected_level = old_mass / expected_density;
    const auto new_top = top_elevation(*state.mesh);
    const auto mesh_volume = global_mesh_volume(*state.mesh);
    const auto surface = state.coupling->free_surface_diagnostics();
    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
    record_ale_residuals(diagnostics);

    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_NEAR(state.solver->time(), time_step, 1.0e-15);
    EXPECT_EQ(state.coupling->free_surface_history().size(), old_history + 1);
    EXPECT_GT(state.mesh->geometry_epoch(), old_epoch);
    EXPECT_GT(new_top, old_top);
    EXPECT_NEAR(new_top, expected_level, 2.0e-12);
    EXPECT_NEAR(mesh_volume, expected_level, 2.0e-12);
    EXPECT_NEAR(surface.pool_level, new_top, 2.0e-12);
    EXPECT_NEAR(surface.pool_volume, mesh_volume, 2.0e-12);
    EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), old_mass, 1.0e-12);
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        // The strict pressure correction redistributes the uniform heating by
        // at most 1.197e-9 K between these two moving control volumes.
        EXPECT_NEAR(state.solver->temperature().value(cell), expected_temperature, 2.0e-9);
        EXPECT_NEAR(state.coupling->rho_liquid().value(cell), expected_density, 2.0e-10);
    }
    EXPECT_GE(diagnostics.outer_correctors, 2);
    EXPECT_LE(diagnostics.outer_correctors, 4);
    // Successful Picard replay refreshes only candidate geometry, never the
    // transient accepted mesh between two candidate trials.
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::GeometryRefresh).calls,
        state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALEGeometry).calls);
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALERestore).calls,
        static_cast<std::uint64_t>(diagnostics.outer_correctors - 1));
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
    auto solver = std::make_shared<Solver>(
        mesh, planar_boundaries(), time_options(Coupling::PISO), linear_options, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);

    auto& accepted_absolute_flux = solver->mutable_pressure_corrected_face_fluxes();
    for (const auto face_lid : accepted_absolute_flux.owned_face_ids())
    {
        accepted_absolute_flux.set_owned_value(face_lid, 0.01 * (1.0 + static_cast<double>(face_lid)));
    }
    accepted_absolute_flux.sync_ghosts();
    const auto fixed_grid_courant = solver_coupling->maximum_courant_number();
    ASSERT_GT(fixed_grid_courant, 0.0);

    ASSERT_NE(solver_coupling->configure_free_surface(free_surface_options(4)), nullptr);
    ASSERT_TRUE(solver_coupling->planar_ale_enabled());
    const auto& accepted_relative_flux = solver_coupling->mesh_relative_face_fluxes();
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        EXPECT_DOUBLE_EQ(accepted_relative_flux.local_value(face_lid), accepted_absolute_flux.local_value(face_lid));
    }
    EXPECT_DOUBLE_EQ(solver_coupling->maximum_courant_number(), fixed_grid_courant);
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
    const auto initial_mass = state.coupling->liquid_mass_inventory().totalMass();
    const auto initial_temperature = state.solver->temperature().value(0);

    ASSERT_NO_THROW(state.coupling->step());
    const auto expanded_top = top_elevation(*state.mesh);
    const auto expanded_volume = global_mesh_volume(*state.mesh);
    ASSERT_GT(expanded_top, initial_top);
    ASSERT_GT(expanded_volume, initial_volume);

    // The source is volumetric on the trial-new mesh.  Scale the cooling so
    // its final-volume integral exactly removes the preceding accepted heat.
    state.heat_source->initialize(-power_density * expanded_volume / initial_volume);
    ASSERT_NO_THROW(state.coupling->step());

    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
    EXPECT_LT(top_elevation(*state.mesh), expanded_top);
    EXPECT_NEAR(top_elevation(*state.mesh), initial_top, 3.0e-12);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), initial_volume, 3.0e-12);
    EXPECT_NEAR(state.solver->temperature().value(0), initial_temperature, 3.0e-9);
    EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), initial_mass, 2.0e-12);
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
    ASSERT_NO_THROW(state.coupling->step());
    ASSERT_GT(top_elevation(*state.mesh), old_top);
    EXPECT_EQ(state.solver->cached_velocity_boundary_type("zmax"), SimpleFluid::BoundaryConditionType::Slip);

    ASSERT_TRUE(state.coupling->remove_free_surface_model());
    EXPECT_EQ(state.solver->cached_velocity_boundary_type("zmax"), SimpleFluid::BoundaryConditionType::Slip);
}

TEST(BoussinesqPlanarALETest, AcceptedMotionUpdatesVtuPointsAndPublishesAleDiagnostics)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
    {
        GTEST_SKIP() << "This accepted-file assertion uses one serial output path.";
    }
    auto state = make_case(Coupling::PISO, 1.0, 6);
    ASSERT_NO_THROW(state.coupling->step());
    const auto accepted_top = top_elevation(*state.mesh);
    const auto topology = state.solver->solution_topology();
    ASSERT_TRUE(topology);
    const auto maximum_point_z = std::ranges::max(topology->points, {}, [](const auto& point) { return point.z; }).z;
    EXPECT_NEAR(maximum_point_z, accepted_top, 2.0e-12);

    const auto filename = std::filesystem::temp_directory_path() / "simplefluid_planar_ale_accepted.vtu";
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_free_surface_fields = true;
    ASSERT_NO_THROW(state.coupling->write_solution_vtu(filename.string(), output_options));
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
    ASSERT_NO_THROW(state.coupling->step());
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
    ASSERT_NO_THROW(state.coupling->write_parallel_solution_vtu(filename.string(), output_options));

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

TEST(BoussinesqPlanarALETest, StationaryGasCanConvergeOnTheFirstAleTrial)
{
    auto state = make_case(Coupling::PISO, 0.0, 4, true);
    std::vector<SimpleFluid::ALEConvergenceObservation> observations;
    state.coupling->set_ale_convergence_observer([&](const auto& row) { observations.push_back(row); });
    auto checkpoint = state.coupling->create_coupling_checkpoint();
    for (double dt : {0.01, 0.004})
    {
        state.coupling->restore_coupling_checkpoint(checkpoint);
        state.solver->set_time_step(dt);
        ASSERT_NO_THROW(state.coupling->step());
        const auto& diagnostics = state.coupling->planar_ale_diagnostics();
        EXPECT_EQ(diagnostics.outer_correctors, 1);
        ASSERT_EQ(diagnostics.gas_state_residual_history.size(), 1u);
        EXPECT_LE(diagnostics.gas_state_residual_history.front(), 8 * std::numeric_limits<double>::epsilon());
        ASSERT_FALSE(observations.empty());
        EXPECT_EQ(observations.back().corrector, 1);
        EXPECT_EQ(observations.back().blocking_mask, 0u);
        for (const auto value : observations.back().gas_relative_change)
            EXPECT_LE(value, 8 * std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(SurfaceGas::compartment(*state.coupling->find_reacting_gas_model(), SurfaceGas::Phase::Dissolved),
            1.0, 1.e-14);
        expect_zero_relative_top_flux(state);
    }
    state.coupling->accept_coupling_checkpoint(checkpoint);
    ASSERT_EQ(observations.size(), 2u);
    EXPECT_EQ(observations[1].check, observations[0].check + 1);
}

TEST(BoussinesqPlanarALETest, GasProductionStillRequiresAConvergedTrialComparison)
{
    auto state = make_case(Coupling::PISO, 0.0, 40, true);
    std::vector<SimpleFluid::ALEConvergenceObservation> observations;
    state.coupling->set_ale_convergence_observer([&](const auto& row) { observations.push_back(row); });
    state.coupling->find_prescribed_heat_source()->initialize_constant(2000.0);
    ASSERT_NO_THROW(state.coupling->step());
    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
    ASSERT_GE(diagnostics.gas_state_residual_history.size(), 2u);
    // New bubble populations start from zero; their normalized physical
    // increment is one, even though the artificial first-pass sentinel is gone.
    EXPECT_DOUBLE_EQ(diagnostics.gas_state_residual_history.front(), 1.0);
    EXPECT_LE(diagnostics.gas_state_residual_history.back(), 1.e-10);
    ASSERT_EQ(observations.size(), diagnostics.gas_state_residual_history.size());
    for (size_t i = 0; i < observations.size(); ++i)
    {
        const auto& row = observations[i];
        EXPECT_EQ(row.corrector, i + 1);
        for (const auto legacy : row.gas_relative_change)
            EXPECT_DOUBLE_EQ(legacy, 0);
        EXPECT_EQ(
            (row.blocking_mask & (size_t{1} << 6)) != 0, diagnostics.gas_state_residual_history[i] > row.gas_tolerance);
    }
    const auto* gas = state.coupling->find_reacting_gas_model();
    const double produced = SurfaceGas::supplied(*gas);
    EXPECT_NEAR(produced, 2000.0 * 0.01 * 2.e-7 * global_mesh_volume(*state.mesh), 1.e-15);
    EXPECT_NEAR(gas->global_species_moles(SurfaceGas::gas_id) + SurfaceGas::escaped(*gas), 1.0 + produced, 1.e-12);
}

TEST(BoussinesqPlanarALETest, AitkenGasProductionKeepsUnrelaxedDiagnosticsAndInventoryGates)
{
    auto state = make_case(Coupling::PISO, 0.0, 40, true);
    auto options = free_surface_options(40);
    options.ale.aitken = true;
    options.ale.relaxation = 0.5;
    state.coupling->configure_free_surface(options);
    std::vector<SimpleFluid::ALEConvergenceObservation> observations;
    state.coupling->set_ale_convergence_observer([&](const auto& row) { observations.push_back(row); });
    state.coupling->find_prescribed_heat_source()->initialize_constant(2000.0);
    ASSERT_NO_THROW(state.coupling->step());
    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
    ASSERT_GE(diagnostics.gas_state_residual_history.size(), 2u);
    // New bubble populations start from zero; their normalized physical
    // increment is one, even though the artificial first-pass sentinel is gone.
    EXPECT_DOUBLE_EQ(diagnostics.gas_state_residual_history.front(), 1.0);
    EXPECT_LE(diagnostics.gas_state_residual_history.back(), 1.e-10);
    ASSERT_EQ(observations.size(), diagnostics.gas_state_residual_history.size());
    for (size_t i = 0; i < observations.size(); ++i)
    {
        const auto& row = observations[i];
        EXPECT_EQ(row.corrector, i + 1);
        for (const auto legacy : row.gas_relative_change)
            EXPECT_DOUBLE_EQ(legacy, 0);
        EXPECT_EQ(
            (row.blocking_mask & (size_t{1} << 6)) != 0, diagnostics.gas_state_residual_history[i] > row.gas_tolerance);
    }
    const auto* gas = state.coupling->find_reacting_gas_model();
    const double produced = SurfaceGas::supplied(*gas);
    EXPECT_NEAR(produced, 2000.0 * 0.01 * 2.e-7 * global_mesh_volume(*state.mesh), 1.e-15);
    EXPECT_NEAR(gas->global_species_moles(SurfaceGas::gas_id) + SurfaceGas::escaped(*gas), 1.0 + produced, 1.e-12);
}

TEST(BoussinesqPlanarALETest, ConvergenceObserverFailureRejectsTheStepCollectively)
{
    auto state = make_case(Coupling::PISO, 0.0, 4, true);
    const auto history_size = state.coupling->free_surface_history().size();
    const auto rank = state.mesh->owned_cell_map()->getComm()->getRank();
    state.coupling->set_ale_convergence_observer([rank](const auto&) {
        if (rank == 0) throw std::runtime_error("diagnostic output failed");
    });
    EXPECT_THROW(state.coupling->step(), std::runtime_error);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.);
    EXPECT_EQ(state.coupling->free_surface_history().size(), history_size);
    const auto checks = state.coupling->ale_convergence_statistics().checks;
    EXPECT_GT(checks, 0u);
    state.coupling->set_ale_convergence_observer({});
    EXPECT_NO_THROW(state.coupling->step());
    EXPECT_GT(state.coupling->ale_convergence_statistics().checks, checks);
    EXPECT_DOUBLE_EQ(state.solver->time(), .01);
}

TEST(BoussinesqPlanarALETest, SuppliedGasGenerationUsesTheAcceptedMovingGeometryAndClosesInventory)
{
    // The explicitly supplied external-feed/count law checks ALE dispatch,
    // accepted-state conservation and independent ideal-mixture/Laplace reconstruction.
    auto state = make_case(Coupling::PISO, 0.0, 8, true, 0.0, false, SimpleFluid::ReactingPressureMode::Constant, false,
        SimpleFluid::CoupledOperatorBackend::Assembled, SimpleFluid::CoupledWorkspacePolicy::CachedProducts,
        SimpleFluid::FVM::CellGradientScheme::LeastSquares, false,
        SimpleFluid::BubbleMassTransferMode::SphericalDiffusion);
    auto* gas = state.coupling->find_reacting_gas_model();
    auto* prescribed_heat = state.coupling->find_prescribed_heat_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(prescribed_heat, nullptr);
    EXPECT_EQ(gas->options().mass_transfer_mode, SimpleFluid::BubbleMassTransferMode::SphericalDiffusion);
    prescribed_heat->initialize_constant(1.0e-3);
    const auto old_generated = SurfaceGas::supplied(*gas);
    const auto old_epoch = state.mesh->geometry_epoch();

    ASSERT_NO_THROW(state.coupling->step());

    const auto surface = state.coupling->free_surface_diagnostics();
    const auto& ale = state.coupling->planar_ale_diagnostics();
    record_ale_residuals(ale);
    EXPECT_GT(SurfaceGas::supplied(*gas), old_generated);
    EXPECT_GT(state.mesh->geometry_epoch(), old_epoch);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), surface.pool_volume, 2.0e-12);
    EXPECT_NEAR(top_elevation(*state.mesh), surface.pool_level, 2.0e-12);
    EXPECT_NEAR(SurfaceGas::inventory_error(*gas), 0.0, 1.0e-12);
    EXPECT_NEAR(ale.gas_inventory_residual, 0.0, 1.0e-12);
    EXPECT_GE(ale.outer_correctors, 2);
    EXPECT_LE(ale.gas_state_residual, 1.0e-10);
    EXPECT_EQ(ale.gas_state_residual_history.size(), static_cast<size_t>(ale.outer_correctors));
    EXPECT_NEAR(ale.volume_source.source_pool_closure_residual, 0.0, 1.0e-10);
    EXPECT_LE(ale.continuity.maximum, 3.0e-10);
    // The D/r accepted-transfer oracle is the two-gas fixed-grid test. Here retain
    // the moving geometry/radius reconstruction purpose independently of legacy fields.
    double local_bubble_volume = 0;
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const double volume = state.mesh->cell_volume(cell);
        const double pressure = gas->absolute_pressure().value(cell);
        const double temperature = state.solver->temperature().value(cell);
        const auto material = gas->cell_species_state(cell, volume);
        for (const auto& population : gas->options().population_ids)
        {
            const double count = gas->population_number_field(population).value(cell) * volume;
            double moles = 0;
            for (const auto& species : gas->options().reaction_mechanism->catalog().species())
                moles += material.amount(
                    gas->options().reaction_mechanism->catalog(), species.id, SurfaceGas::Phase::Dispersed, population);
            if (count == 0)
            {
                EXPECT_DOUBLE_EQ(moles, 0);
                continue;
            }
            const double rhs = 3 * moles * SurfaceGas::M::gas_constant * temperature / (4 * std::numbers::pi * count);
            double lo = 0, hi = .1;
            for (int iteration = 0; iteration < 100; ++iteration)
            {
                const double radius = std::midpoint(lo, hi);
                if (pressure * radius * radius * radius + 2 * .07 * radius * radius < rhs)
                    lo = radius;
                else
                    hi = radius;
            }
            const double radius = std::midpoint(lo, hi);
            ASSERT_GT(radius, 0);
            local_bubble_volume += count * 4 * std::numbers::pi * radius * radius * radius / 3;
        }
    }
    double global_bubble_volume = 0;
    Teuchos::reduceAll(
        *state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_bubble_volume, &global_bubble_volume);
    EXPECT_NEAR(surface.submerged_bubble_volume, global_bubble_volume, global_bubble_volume * 1e-12);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, MovingTopBubbleEscapeTransfersToTheVentExactlyOnceAfterCommit)
{
    auto state = make_case(Coupling::PISO, 0.0, 10, true, 100.0);
    auto* gas = state.coupling->find_reacting_gas_model();
    auto* prescribed_heat = state.coupling->find_prescribed_heat_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(prescribed_heat, nullptr);
    prescribed_heat->initialize_constant(1.0e3);

    // Accepted populations are transported before this step's supplied
    // reservoir/nucleation packets.  Prime one accepted step so the following
    // moving-mesh transport step has a nonzero bubble population to vent.
    ASSERT_NO_THROW(state.coupling->step());
    const auto old_cumulative_escape = SurfaceGas::escaped(*gas);
    const auto old_vented = state.coupling->free_surface_diagnostics().vented_gas_moles;
    const auto old_vented_h2 = old_vented.contains(SurfaceGas::gas_name) ? old_vented.at(SurfaceGas::gas_name) : 0.0;

    ASSERT_NO_THROW(state.coupling->step());

    const auto statistics = gas->last_statistics();
    const auto surface = state.coupling->free_surface_diagnostics();
    record_ale_residuals(state.coupling->planar_ale_diagnostics());
    ASSERT_GT(SurfaceGas::step_escaped(*gas), 0.0);
    ASSERT_TRUE(surface.escaped_gas_moles_this_step.contains(SurfaceGas::gas_name));
    ASSERT_TRUE(surface.vented_gas_moles.contains(SurfaceGas::gas_name));
    EXPECT_NEAR(surface.escaped_gas_moles_this_step.at(SurfaceGas::gas_name), SurfaceGas::step_escaped(*gas), 1.0e-14);
    EXPECT_NEAR(SurfaceGas::escaped(*gas) - old_cumulative_escape, SurfaceGas::step_escaped(*gas), 1.0e-14);
    EXPECT_NEAR(
        surface.vented_gas_moles.at(SurfaceGas::gas_name) - old_vented_h2, SurfaceGas::step_escaped(*gas), 1.0e-14);
    EXPECT_NEAR(state.coupling->planar_ale_diagnostics().gas_inventory_residual, 0.0, 1.0e-12);
    EXPECT_GE(state.coupling->planar_ale_diagnostics().outer_correctors, 2);
    EXPECT_LE(state.coupling->planar_ale_diagnostics().gas_state_residual, 1.0e-10);
    EXPECT_NEAR(global_mesh_volume(*state.mesh), surface.pool_volume, 2.0e-12);
    expect_zero_relative_top_flux(state);
}

TEST(BoussinesqPlanarALETest, ReconstructedPressureRunsInsideTheAcceptedAleTransaction)
{
    auto state = make_case(Coupling::PISO, 0.0, 10, true, 0.0, false, SimpleFluid::ReactingPressureMode::Reconstructed);
    auto* gas = state.coupling->find_reacting_gas_model();
    auto* prescribed_heat = state.coupling->find_prescribed_heat_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(prescribed_heat, nullptr);
    prescribed_heat->initialize_constant(1.0e-3);

    ASSERT_NO_THROW(state.coupling->step());
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        EXPECT_GT(gas->absolute_pressure().value(static_cast<Pack::local_ordinal_type>(owned)), 0.0);
    }
    EXPECT_NEAR(SurfaceGas::inventory_error(*gas), 0.0, 1.0e-12);
    EXPECT_LE(state.coupling->planar_ale_diagnostics().continuity.maximum, 3.0e-10);
}

TEST(BoussinesqPlanarALETest, CelataSlipRunsThroughSolverIntegratedMovingTopEscape)
{
    auto state =
        make_case(Coupling::PISO, 0.0, 12, true, 0.0, false, SimpleFluid::ReactingPressureMode::Constant, true);
    auto* gas = state.coupling->find_reacting_gas_model();
    auto* prescribed_heat = state.coupling->find_prescribed_heat_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(prescribed_heat, nullptr);
    prescribed_heat->initialize_constant(1.0e3);

    ASSERT_NO_THROW(state.coupling->step());
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_GT(SurfaceGas::step_escaped(*gas), 0.0);
    EXPECT_NEAR(SurfaceGas::inventory_error(*gas), 0.0, 1.0e-12);
    EXPECT_LE(state.coupling->planar_ale_diagnostics().continuity.maximum, 3.0e-10);
}

TEST(BoussinesqPlanarALETest, StrictGasStepClosureFailureRollsBackTheWholeTransaction)
{
    auto state = make_case(Coupling::PISO, 0.0, 8, true, 0.0, true);
    auto* gas = state.coupling->find_reacting_gas_model();
    auto* prescribed_heat = state.coupling->find_prescribed_heat_source();
    ASSERT_NE(gas, nullptr);
    ASSERT_NE(prescribed_heat, nullptr);
    prescribed_heat->initialize_constant(1.0e-3);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto old_generated = SurfaceGas::supplied(*gas);
    const auto old_escape = SurfaceGas::escaped(*gas);
    const auto old_history = state.coupling->free_surface_history().size();

    std::string failure;
    try
    {
        state.coupling->step();
        FAIL() << "Expected zero-tolerance gas closure to reject roundoff.";
    }
    catch (const std::runtime_error& error)
    {
        failure = error.what();
    }
    EXPECT_NE(failure.find("gas-inventory closure for species"), std::string::npos) << failure;
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(SurfaceGas::supplied(*gas), old_generated);
    EXPECT_DOUBLE_EQ(SurfaceGas::escaped(*gas), old_escape);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.coupling->free_surface_history().size(), old_history);
    EXPECT_EQ(state.coupling->planar_ale_diagnostics().rejected_transactions, 1U);
}

TEST(BoussinesqPlanarALETest, PublicScalarVoidRestoreRefreshesRollbackGeometry)
{
    auto mesh = make_column();
    SimpleFluid::ScalarVoidFractionOptions options;
    options.alpha_diffusivity = 0.1;
    SimpleFluid::ScalarVoidFractionModel<Pack, Handle> model(mesh, options);
    ScalarField initial(mesh, "initial_alpha");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        initial.set_owned_value(cell, mesh->cell_centroid(cell).z < 0.5 ? 0.8 : 0.1);
    }
    initial.sync_ghosts();
    model.initialize_from(initial);
    const auto accepted = model.snapshot();
    model.update_explicit(0.1, nullptr, nullptr);
    const auto expected = capture_owned_values(model.alpha_g());
    model.restore(accepted);
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(1.2, 0.1);
    model.refresh_geometry();
    model.update_explicit(0.1, nullptr, nullptr);
    motion.rollback_trial();
    // The public restore alone must refresh rollback metrics and invalidate
    // trial diffusion state; no additional caller refresh is supplied.
    ASSERT_NO_THROW(model.restore(accepted));
    ASSERT_NO_THROW(model.update_explicit(0.1, nullptr, nullptr));
    expect_owned_values(model.alpha_g(), expected);
    SimpleFluid::ScalarVoidFractionModel<Pack, Handle> foreign(mesh, options);
    EXPECT_THROW(model.restore(foreign.snapshot()), std::invalid_argument);
    expect_owned_values(model.alpha_g(), expected);
}

TEST(BoussinesqPlanarALETest, MeshMotionLimitFailureLeavesAcceptedStateUntouched)
{
    auto mesh = make_column();
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    auto solver = std::make_shared<Solver>(
        mesh, planar_boundaries(), time_options(Coupling::PISO), linear_options, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver->add_temperature_source("motion_limit_heat", 1.0);
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(6);
    options.ale.maximum_level_change = 1.0e-12;
    ASSERT_NE(solver_coupling->configure_free_surface(options), nullptr);
    const auto geometry = capture_geometry(*mesh);
    const auto primary = capture_primary(*solver);
    const auto history = solver_coupling->free_surface_history().size();

    solver->reset_solver_phase_timings();
    EXPECT_THROW(solver_coupling->step(), std::invalid_argument);
    // The stationary first candidate completes and rolls back internally;
    // the next candidate fails begin_trial's movement limit. The outward
    // catch must still refresh the accepted mesh before returning.
    EXPECT_EQ(solver->solver_phase_timing(SimpleFluid::SolverPhase::ALEGeometry).calls, 2u);
    EXPECT_EQ(solver->solver_phase_timing(SimpleFluid::SolverPhase::ALERestore).calls, 2u);
    EXPECT_EQ(solver->solver_phase_timing(SimpleFluid::SolverPhase::GeometryRefresh).calls, 2u);
    expect_geometry_restored(*mesh, geometry);
    expect_primary_restored(*solver, primary);
    EXPECT_EQ(solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(solver->time(), 0.0);
    EXPECT_EQ(solver_coupling->free_surface_history().size(), history);
    EXPECT_EQ(solver_coupling->planar_ale_diagnostics().rejected_transactions, 1U);
}

TEST(BoussinesqPlanarALETest, MomentumTransportPolicyFailureRollsBackAndCanRetry)
{
    auto state = make_case(Coupling::PISO, 1.0, 6);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto history = state.coupling->free_surface_history().size();
    const auto valid_options = state.solver->linear_solver_options();
    auto invalid_options = valid_options;
    invalid_options.max_iterations = 0;
    state.solver->set_linear_solver_options(invalid_options);

    EXPECT_THROW(state.coupling->step(), std::invalid_argument);
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.coupling->free_surface_history().size(), history);

    state.solver->set_linear_solver_options(valid_options);
    state.heat_source->set_enabled(false);
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_EQ(state.solver->step_index(), 1);
}

TEST(BoussinesqPlanarALETest, FailedOuterTrialRestoresGeometryFieldsLedgersAndCanRetry)
{
    auto state = make_case(Coupling::PISO, 1.0, 2, true);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto initial_time = state.solver->time();
    const auto initial_step = state.solver->step_index();
    const auto initial_history = state.coupling->free_surface_history().size();
    const auto initial_surface = state.coupling->free_surface_diagnostics();
    const auto initial_liquid = state.coupling->liquid_mass_inventory().diagnostics();
    const auto initial_liquid_field = capture_owned_values(state.coupling->liquid_mass_inventory().cellMassInventory());
    const auto initial_density = capture_owned_values(state.coupling->liquid_mass_inventory().pureLiquidDensity());
    const auto initial_material_density = capture_owned_values(state.solver->material_properties().density);
    const auto initial_heat_capacity = capture_owned_values(state.solver->material_properties().specific_heat_capacity);
    const auto initial_clear_level_field = capture_owned_values(state.coupling->clear_level());
    const auto initial_pool_level_field = capture_owned_values(state.coupling->pool_level());
    const auto initial_headspace_pressure_field = capture_owned_values(state.coupling->headspace_pressure());
    const auto initial_pool_occupancy_field = capture_owned_values(state.coupling->pool_occupancy());
    const auto initial_absolute_flux = capture_owned_face_values(state.solver->pressure_corrected_face_fluxes());
    const auto initial_relative_flux = capture_owned_face_values(state.coupling->mesh_relative_face_fluxes());
    const auto initial_occupancy_error = state.coupling->pool_occupancy_volume_error();
    const auto* initial_feedback = state.coupling->find_material_feedback_model();
    ASSERT_NE(initial_feedback, nullptr);
    const auto initial_feedback_density = capture_owned_values(initial_feedback->density_feedback());
    const auto initial_feedback_viscosity = capture_owned_values(initial_feedback->viscosity_feedback());

    auto* gas = state.coupling->find_reacting_gas_model();
    ASSERT_NE(gas, nullptr);
    const auto initial_gas_fields = capture_gas_fields(*gas);
    const auto initial_species_receipt = gas->species_event_receipt();
    const auto initial_species_histories = gas->species_histories();
    const auto initial_source_ledgers = gas->source_ledgers();
    const auto initial_dissolved = SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dissolved);
    const auto initial_micro = SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dispersed, SurfaceGas::micro);
    const auto initial_large = SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dispersed, SurfaceGas::large);
    const auto initial_generated = SurfaceGas::supplied(*gas);
    const auto initial_escape = SurfaceGas::escaped(*gas);
    const auto initial_pressure_offset = gas->absolute_pressure_offset();
    const auto initial_committed_escape = state.coupling->find_free_surface_model()->committedEscapedMoles();

    state.solver->reset_solver_phase_timings();
    std::string failure;
    try
    {
        state.coupling->step();
        FAIL() << "Expected the bounded planar-ALE trial to fail.";
    }
    catch (const std::runtime_error& error)
    {
        failure = error.what();
    }

    EXPECT_NE(failure.find("outer level/continuity corrector did not converge"), std::string::npos) << failure;
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALEGeometry).calls, 2u);
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALERestore).calls, 3u);
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::GeometryRefresh).calls, 3u);
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::GasRestore).calls, 3u);
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(state.solver->time(), initial_time);
    EXPECT_EQ(state.solver->step_index(), initial_step);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history);

    const auto restored_surface = state.coupling->free_surface_diagnostics();
    EXPECT_DOUBLE_EQ(restored_surface.clear_level, initial_surface.clear_level);
    EXPECT_DOUBLE_EQ(restored_surface.pool_level, initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(restored_surface.liquid_volume, initial_surface.liquid_volume);
    EXPECT_DOUBLE_EQ(restored_surface.pool_volume, initial_surface.pool_volume);
    EXPECT_DOUBLE_EQ(restored_surface.headspace.pressure, initial_surface.headspace.pressure);
    EXPECT_EQ(state.coupling->find_free_surface_model()->committedEscapedMoles(), initial_committed_escape);

    const auto restored_liquid = state.coupling->liquid_mass_inventory().diagnostics();
    EXPECT_DOUBLE_EQ(restored_liquid.total_mass, initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(restored_liquid.liquid_volume, initial_liquid.liquid_volume);
    EXPECT_DOUBLE_EQ(restored_liquid.step_mass_balance_residual, initial_liquid.step_mass_balance_residual);
    const auto restored_liquid_field =
        capture_owned_values(state.coupling->liquid_mass_inventory().cellMassInventory());
    const auto restored_density = capture_owned_values(state.coupling->liquid_mass_inventory().pureLiquidDensity());
    ASSERT_EQ(restored_liquid_field.size(), initial_liquid_field.size());
    ASSERT_EQ(restored_density.size(), initial_density.size());
    for (size_t owned = 0; owned < initial_liquid_field.size(); ++owned)
    {
        EXPECT_DOUBLE_EQ(restored_liquid_field[owned], initial_liquid_field[owned]);
        EXPECT_DOUBLE_EQ(restored_density[owned], initial_density[owned]);
    }
    expect_owned_values(state.solver->material_properties().density, initial_material_density);
    expect_owned_values(state.solver->material_properties().specific_heat_capacity, initial_heat_capacity);
    expect_owned_values(state.coupling->clear_level(), initial_clear_level_field);
    expect_owned_values(state.coupling->pool_level(), initial_pool_level_field);
    expect_owned_values(state.coupling->headspace_pressure(), initial_headspace_pressure_field);
    expect_owned_values(state.coupling->pool_occupancy(), initial_pool_occupancy_field);
    expect_owned_face_values(state.solver->pressure_corrected_face_fluxes(), initial_absolute_flux);
    expect_owned_face_values(state.coupling->mesh_relative_face_fluxes(), initial_relative_flux);
    EXPECT_DOUBLE_EQ(state.coupling->pool_occupancy_volume_error(), initial_occupancy_error);
    const auto* restored_feedback = state.coupling->find_material_feedback_model();
    ASSERT_NE(restored_feedback, nullptr);
    expect_owned_values(restored_feedback->density_feedback(), initial_feedback_density);
    expect_owned_values(restored_feedback->viscosity_feedback(), initial_feedback_viscosity);

    expect_gas_fields_restored(*gas, initial_gas_fields);
    EXPECT_EQ(gas->species_event_receipt(), initial_species_receipt);
    EXPECT_EQ(gas->species_histories(), initial_species_histories);
    SurfaceGas::M::expect_same_ledgers(gas->source_ledgers(), initial_source_ledgers);
    EXPECT_DOUBLE_EQ(SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dissolved), initial_dissolved);
    EXPECT_DOUBLE_EQ(SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dispersed, SurfaceGas::micro), initial_micro);
    EXPECT_DOUBLE_EQ(SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dispersed, SurfaceGas::large), initial_large);
    EXPECT_DOUBLE_EQ(SurfaceGas::supplied(*gas), initial_generated);
    EXPECT_DOUBLE_EQ(SurfaceGas::escaped(*gas), initial_escape);
    EXPECT_DOUBLE_EQ(gas->absolute_pressure_offset(), initial_pressure_offset);

    const auto& rejected = state.coupling->planar_ale_diagnostics();
    EXPECT_EQ(rejected.rejected_transactions, 1U);
    EXPECT_EQ(rejected.last_rejection_reason, failure);

    state.heat_source->set_enabled(false);
    ASSERT_NO_THROW(state.coupling->step());

    EXPECT_EQ(state.solver->step_index(), initial_step + 1);
    EXPECT_NEAR(state.solver->time(), initial_time + 1.0e-2, 1.0e-15);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history + 1);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(state.coupling->liquid_mass_inventory().totalMass(), initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(SurfaceGas::compartment(*gas, SurfaceGas::Phase::Dissolved), initial_dissolved);
    EXPECT_DOUBLE_EQ(SurfaceGas::escaped(*gas), initial_escape);
    EXPECT_EQ(state.coupling->planar_ale_diagnostics().rejected_transactions, 1U);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    expect_zero_relative_top_flux(state);

    // A later rejected step must retain the preceding accepted numerical
    // report; begin_step() must not make a cleared trial report observable.
    const auto accepted_statistics = state.solver->last_step_statistics();
    const auto accepted_volume_residuals = state.solver->last_volume_continuity_residuals();
    auto invalid_pressure_options = state.solver->pressure_linear_solver_options();
    invalid_pressure_options.max_iterations = 0;
    state.solver->set_pressure_linear_solver_options(invalid_pressure_options);
    EXPECT_THROW(state.coupling->step(), std::invalid_argument);
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
    const auto initial_history = state.coupling->free_surface_history().size();
    const auto initial_surface = state.coupling->free_surface_diagnostics();
    const auto initial_liquid = state.coupling->liquid_mass_inventory().diagnostics();
    const auto initial_liquid_field = capture_owned_values(state.coupling->liquid_mass_inventory().cellMassInventory());
    const auto initial_density = capture_owned_values(state.coupling->liquid_mass_inventory().pureLiquidDensity());
    const auto initial_clear_level_field = capture_owned_values(state.coupling->clear_level());
    const auto initial_pool_level_field = capture_owned_values(state.coupling->pool_level());
    const auto initial_headspace_pressure_field = capture_owned_values(state.coupling->headspace_pressure());
    const auto initial_pool_occupancy_field = capture_owned_values(state.coupling->pool_occupancy());
    const auto initial_absolute_flux = capture_owned_face_values(state.solver->pressure_corrected_face_fluxes());

    state.solver->reset_solver_phase_timings();
    const auto valid_pressure_options = state.solver->pressure_linear_solver_options();
    auto invalid_pressure_options = valid_pressure_options;
    invalid_pressure_options.max_iterations = 0;
    state.solver->set_pressure_linear_solver_options(invalid_pressure_options);

    std::string failure;
    try
    {
        state.coupling->step();
        FAIL() << "Expected the in-trial pressure solve to reject its policy.";
    }
    catch (const std::invalid_argument& error)
    {
        failure = error.what();
    }

    EXPECT_NE(failure.find("BelosLinearSolver requires positive maximum iterations"), std::string::npos) << failure;
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 1u);
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALE).calls, 1u);
    EXPECT_GT(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::PressureProjection).calls, 0u);
    const auto rejected_trial_calls = state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALETrial).calls;
    EXPECT_GT(rejected_trial_calls, 0u);
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(state.solver->time(), initial_time);
    EXPECT_EQ(state.solver->step_index(), initial_step);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history);

    const auto restored_surface = state.coupling->free_surface_diagnostics();
    EXPECT_DOUBLE_EQ(restored_surface.clear_level, initial_surface.clear_level);
    EXPECT_DOUBLE_EQ(restored_surface.pool_level, initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(restored_surface.liquid_volume, initial_surface.liquid_volume);
    EXPECT_DOUBLE_EQ(restored_surface.pool_volume, initial_surface.pool_volume);
    EXPECT_DOUBLE_EQ(restored_surface.headspace.pressure, initial_surface.headspace.pressure);
    const auto restored_liquid = state.coupling->liquid_mass_inventory().diagnostics();
    EXPECT_DOUBLE_EQ(restored_liquid.total_mass, initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(restored_liquid.liquid_volume, initial_liquid.liquid_volume);
    expect_owned_values(state.coupling->liquid_mass_inventory().cellMassInventory(), initial_liquid_field);
    expect_owned_values(state.coupling->liquid_mass_inventory().pureLiquidDensity(), initial_density);
    expect_owned_values(state.coupling->clear_level(), initial_clear_level_field);
    expect_owned_values(state.coupling->pool_level(), initial_pool_level_field);
    expect_owned_values(state.coupling->headspace_pressure(), initial_headspace_pressure_field);
    expect_owned_values(state.coupling->pool_occupancy(), initial_pool_occupancy_field);
    expect_owned_face_values(state.solver->pressure_corrected_face_fluxes(), initial_absolute_flux);
    EXPECT_EQ(state.coupling->planar_ale_diagnostics().rejected_transactions, 1U);
    EXPECT_EQ(state.coupling->planar_ale_diagnostics().last_rejection_reason, failure);

    state.solver->set_pressure_linear_solver_options(valid_pressure_options);
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 2u);
    EXPECT_EQ(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALE).calls, 2u);
    EXPECT_GT(state.solver->solver_phase_timing(SimpleFluid::SolverPhase::ALETrial).calls, rejected_trial_calls);

    EXPECT_EQ(state.solver->step_index(), initial_step + 1);
    EXPECT_NEAR(state.solver->time(), initial_time + 1.0e-2, 1.0e-15);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history + 1);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), initial_surface.pool_level);
    EXPECT_DOUBLE_EQ(state.coupling->liquid_mass_inventory().totalMass(), initial_liquid.total_mass);
    EXPECT_EQ(state.coupling->planar_ale_diagnostics().rejected_transactions, 1U);
    EXPECT_TRUE(state.solver->last_step_statistics().converged);
    expect_zero_relative_top_flux(state);
}

enum class MatrixMeshFamily
{
    Cartesian,
    Cylindrical,
    SemiStructured,
    Composite
};

struct MatrixCase
{
    MatrixMeshFamily family = MatrixMeshFamily::Cartesian;
    SimpleFluid::Dimension axis = SimpleFluid::Dimension::Z;
    const char* moving_boundary = "zmax";
    double cross_section_area = 1.0;
    bool single_layer_xy_partition = false;
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
        case MatrixMeshFamily::Composite:
            return std::make_shared<Handle>(std::make_shared<SimpleFluid::Meshes::MultiRegionMesh>(
                std::vector<SimpleFluid::Meshes::MultiRegionMesh::Region>{
                    SimpleFluid::Meshes::cartesian_region("left",{{{0,0.5},{0,1},{0,0.5,1}}}),
                    SimpleFluid::Meshes::cartesian_region("right",{{{0.5,1},{0,1},{0,0.5,1}}})},
                std::vector<SimpleFluid::Meshes::MultiRegionMesh::Interface>{SimpleFluid::Meshes::StructuredPatchInterface{{0,1},{1,0}}},
                SimpleFluid::Meshes::InterfaceTolerance{}, SimpleFluid::Meshes::MultiRegionMesh::BoundaryNamePolicy::MergeMatchingNames));
        case MatrixMeshFamily::SemiStructured: {
            Handle::DistributionOptions distribution;
            distribution.allow_empty_partitions = true;
            if (test_case.single_layer_xy_partition)
                return std::make_shared<Handle>(std::make_shared<SimpleFluid::Meshes::SemiStructuredXY_Z>(
                    SimpleFluid::Arr<SimpleFluid::Meshes::SemiStructuredXY_Z::Vec3>{
                        {0., 0., 0.}, {.45, 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {.55, 1., 0.}, {1., 1., 0.}},
                    SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 4, 3}, {1, 2, 5, 4}},
                    SimpleFluid::ArrReal{0., 1.}), distribution);
            return std::make_shared<Handle>(std::make_shared<SimpleFluid::Meshes::SemiStructuredXY_Z>(
                SimpleFluid::Arr<SimpleFluid::Meshes::SemiStructuredXY_Z::Vec3>{
                    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
                SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2, 3}}, SimpleFluid::ArrReal{0.0, 0.5, 1.0}),
                distribution);
        }
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
    std::shared_ptr<Solver> solver;
    std::unique_ptr<SimpleFluid::MultiphysicsSolverCoupler<Pack>> coupling;
    MatrixCase definition;
};

MatrixConfiguredCase configure_matrix_case(const MatrixCase& test_case, double power_density,
    Coupling coupling = Coupling::PISO, SimpleFluid::test::CoupledBackendSelection selection = {},
    SimpleFluid::FVM::CellGradientScheme gradient = SimpleFluid::FVM::CellGradientScheme::LeastSquares)
{
    auto mesh = make_matrix_mesh(test_case);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;
    linear_options.max_iterations = 500;
    auto options = SimpleFluid::test::with_coupled_backend(matrix_time_options(test_case.axis), selection);
    options.pressure_velocity_coupling = coupling;
    options.pressure_gradient_scheme = gradient;
    auto solver = std::make_shared<Solver>(
        mesh, matrix_boundaries(*mesh, test_case.moving_boundary), options, linear_options, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver->add_temperature_source("matrix_uniform_heat", power_density);

    Handle::Vec3 direction{};
    direction.component(static_cast<size_t>(test_case.axis)) = 1.0;
    solver_coupling->initialize_linear_temperature(direction, 300.0, 300.0);
    if (solver_coupling->configure_free_surface(matrix_free_surface_options(test_case)) == nullptr)
    {
        throw std::logic_error("The planar-ALE support-matrix case did not create a model.");
    }
    return {std::move(mesh), std::move(solver), std::move(solver_coupling), test_case};
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
    const auto& relative = state.coupling->mesh_relative_face_fluxes();
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

void exercise_supported_matrix_case(const MatrixCase& test_case, Coupling coupling = Coupling::PISO,
    SimpleFluid::test::CoupledBackendSelection selection = {},
    SimpleFluid::FVM::CellGradientScheme gradient = SimpleFluid::FVM::CellGradientScheme::LeastSquares)
{
    constexpr double power_density = 1.0e-3;
    constexpr double time_step = 1.0e-2;
    auto state = configure_matrix_case(test_case, power_density, coupling, selection, gradient);
    if (test_case.family == MatrixMeshFamily::SemiStructured)
    {
        const auto comm = state.mesh->owned_cell_map()->getComm();
        const int local_empty = state.mesh->num_owned_cells() == 0 ? 1 : 0;
        int empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    }
    const auto old_epoch = state.mesh->geometry_epoch();
    const auto old_surface = boundary_elevation(*state.mesh, test_case);
    const auto old_volume = global_mesh_volume(*state.mesh);
    const auto old_mass = state.coupling->liquid_mass_inventory().totalMass();

    ASSERT_TRUE(state.coupling->planar_ale_enabled());
    ASSERT_NO_THROW(state.coupling->step());

    const auto new_surface = boundary_elevation(*state.mesh, test_case);
    const auto new_volume = global_mesh_volume(*state.mesh);
    const auto surface = state.coupling->free_surface_diagnostics();
    const auto& diagnostics = state.coupling->planar_ale_diagnostics();
    record_ale_residuals(diagnostics);
    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_NEAR(state.solver->time(), time_step, 1.0e-15);
    EXPECT_GT(state.mesh->geometry_epoch(), old_epoch);
    EXPECT_GT(new_surface, old_surface);
    EXPECT_GT(new_volume, old_volume);
    EXPECT_NEAR(surface.pool_level, new_surface, 2.0e-11);
    EXPECT_NEAR(surface.pool_volume, new_volume, 2.0e-11);
    EXPECT_NEAR(new_volume, test_case.cross_section_area * new_surface, 2.0e-11);
    EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), old_mass, 2.0e-11);
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

TEST(BoussinesqPlanarALESupportMatrixTest, CoupledBackendsCoverSupportedMotionAxes)
{
    const MatrixCase cases[] = {{MatrixMeshFamily::Cartesian, SimpleFluid::Dimension::X, "xmax", 1.},
        {MatrixMeshFamily::Cartesian, SimpleFluid::Dimension::Y, "ymax", 1.},
        {MatrixMeshFamily::Cartesian, SimpleFluid::Dimension::Z, "zmax", 1.},
        {MatrixMeshFamily::Cylindrical, SimpleFluid::Dimension::Z, "zmax", .75 * std::numbers::pi}};
    for (const auto& definition : cases)
        for (const auto selection : SimpleFluid::test::coupled_backend_selections)
            for (const auto gradient :
                {SimpleFluid::FVM::CellGradientScheme::LeastSquares, SimpleFluid::FVM::CellGradientScheme::GaussLinear})
            {
                SCOPED_TRACE(static_cast<int>(definition.family));
                SCOPED_TRACE(static_cast<int>(definition.axis));
                SCOPED_TRACE(SimpleFluid::test::backend_name(selection));
                SCOPED_TRACE(static_cast<int>(gradient));
                exercise_supported_matrix_case(definition, Coupling::CoupledKrylov, selection, gradient);
            }
}

TEST(BoussinesqPlanarALESupportMatrixTest, CoupledBackendsCoverSemiStructuredMotion)
{
    for (const auto selection : SimpleFluid::test::coupled_backend_selections)
        for (const auto gradient :
            {SimpleFluid::FVM::CellGradientScheme::LeastSquares, SimpleFluid::FVM::CellGradientScheme::GaussLinear})
        {
            SCOPED_TRACE(SimpleFluid::test::backend_name(selection));
            exercise_supported_matrix_case({MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Z, "zmax", 1.},
                Coupling::CoupledKrylov, selection, gradient);
        }
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsCompositeAxialMotion)
{
    for (const auto coupling : {Coupling::PISO, Coupling::CoupledKrylov})
        for (const auto selection : SimpleFluid::test::backends_for(coupling))
            exercise_supported_matrix_case({MatrixMeshFamily::Composite,SimpleFluid::Dimension::Z,"zmax",1.0},coupling,selection);
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

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsSemiStructuredAxialMotionAndGravityAlongZ)
{
    exercise_supported_matrix_case({MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Z, "zmax", 1.0});
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsSemiStructuredSingleLayerXYPartition)
{
    exercise_supported_matrix_case(
        {MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Z, "zmax", 1.0, true});
}

TEST(BoussinesqPlanarALESupportMatrixTest, SemiStructuredTrialFailureRestoresGeometryFieldsAndCanRetry)
{
    auto state = configure_matrix_case(
        {MatrixMeshFamily::SemiStructured, SimpleFluid::Dimension::Z, "zmax", 1.0, true}, 1.e-3);
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);
    const auto initial_epoch = state.mesh->geometry_epoch();
    const auto initial_time = state.solver->time();
    const auto initial_mass = state.coupling->liquid_mass_inventory().totalMass();
    const auto initial_history = state.coupling->free_surface_history().size();
    const auto valid = state.solver->pressure_linear_solver_options();
    auto invalid = valid;
    invalid.max_iterations = 0;
    state.solver->set_pressure_linear_solver_options(invalid);
    EXPECT_THROW(state.coupling->step(), std::invalid_argument);
    EXPECT_GT(state.mesh->geometry_epoch(), initial_epoch);
    expect_geometry_restored(*state.mesh, geometry);
    expect_primary_restored(*state.solver, primary);
    EXPECT_DOUBLE_EQ(state.solver->time(), initial_time);
    EXPECT_DOUBLE_EQ(state.coupling->liquid_mass_inventory().totalMass(), initial_mass);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history);
    state.solver->set_pressure_linear_solver_options(valid);
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_EQ(state.solver->step_index(), 1);
    EXPECT_EQ(state.coupling->free_surface_history().size(), initial_history + 1);
    EXPECT_NEAR(state.coupling->liquid_mass_inventory().totalMass(), initial_mass, 2.e-11);
    EXPECT_LE(state.coupling->planar_ale_diagnostics().continuity.maximum, 5.e-10);
}

void expect_nonaxial_family_rejected(MatrixCase test_case)
{
    auto mesh = make_matrix_mesh(test_case);
    SimpleFluid::LinearSolverOptions linear_options;
    auto solver = std::make_shared<Solver>(mesh, matrix_boundaries(*mesh, test_case.moving_boundary),
        matrix_time_options(test_case.axis), linear_options, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);

    std::string diagnostic;
    try
    {
        static_cast<void>(solver_coupling->configure_free_surface(matrix_free_surface_options(test_case)));
        FAIL() << "Expected a nonaxial planar-ALE mesh family to be rejected.";
    }
    catch (const std::invalid_argument& error)
    {
        diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("axial"), std::string::npos) << diagnostic;
    EXPECT_FALSE(solver_coupling->planar_ale_enabled());
    EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
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
    expect_active_model_rejected([](SimpleFluid::MultiphysicsSolverCoupler<Pack>& coupling)
        { coupling.configure_scalar_void_fraction(SimpleFluid::ScalarVoidFractionOptions{}); });
    expect_active_model_rejected(
        [](SimpleFluid::MultiphysicsSolverCoupler<Pack>& coupling)
        {
            SimpleFluid::BoilingSourceOptions options;
            options.enable_bulk_boiling = true;
            coupling.configure_boiling_source(options);
        });
    expect_active_model_rejected(
        [](SimpleFluid::MultiphysicsSolverCoupler<Pack>& coupling)
        {
            SimpleFluid::TurbulenceModelOptions options;
            options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
            coupling.boussinesq_solver().configure_turbulence(options);
        });
    expect_active_model_rejected([](SimpleFluid::MultiphysicsSolverCoupler<Pack>& coupling)
        { coupling.set_material_updater([](const auto&, auto&) {}); });
    expect_active_model_rejected([](SimpleFluid::MultiphysicsSolverCoupler<Pack>& coupling)
        { coupling.add_temperature_source("dynamic_source", 0.0).set_updater([](const auto&, auto&) {}); });
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
        auto solver = std::make_shared<Solver>(
            mesh, boundaries, time_options(Coupling::PISO), SimpleFluid::LinearSolverOptions{}, physical_options());
        auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
        solver_coupling->configure_material_feedback(material_feedback_options());
        solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
        EXPECT_THROW(
            static_cast<void>(solver_coupling->configure_free_surface(free_surface_options(4))), std::invalid_argument);
        EXPECT_FALSE(solver_coupling->planar_ale_enabled());
        EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
    }
}

TEST(BoussinesqPlanarALESupportMatrixTest, AcceptsTypedIndependentGasWithoutPrescribedHeatSourceAtSetup)
{
    auto mesh = make_column();
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->configure_reacting_gas(gas_options());
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    ASSERT_NE(solver_coupling->configure_free_surface(free_surface_options(4)), nullptr);
    EXPECT_TRUE(solver_coupling->planar_ale_enabled());
    EXPECT_NO_THROW(solver_coupling->step());
    EXPECT_DOUBLE_EQ(SurfaceGas::supplied(*solver_coupling->find_reacting_gas_model()), 0);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsTypedReactingGasWithOnlyOneOuterCorrector)
{
    auto mesh = make_column();
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->add_prescribed_heat_source().initialize_constant(0.0);
    solver_coupling->configure_reacting_gas(gas_options());
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(1);
    EXPECT_THROW(static_cast<void>(solver_coupling->configure_free_surface(options)), std::invalid_argument);
    EXPECT_FALSE(solver_coupling->planar_ale_enabled());
    EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsGeometryDependentPrescribedHeatProfileBeforeMotion)
{
    auto state = make_case(Coupling::PISO, 0.0, 5, true);
    auto* prescribed_heat = state.coupling->find_prescribed_heat_source();
    ASSERT_NE(prescribed_heat, nullptr);
    prescribed_heat->initialize_gaussian(1.0, Handle::Vec3{0.5, 0.5, 0.0}, Handle::Vec3{1.0, 1.0, 0.5});
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);

    EXPECT_THROW(state.coupling->step(), std::invalid_argument);
    EXPECT_EQ(state.solver->step_index(), 0);
    EXPECT_DOUBLE_EQ(state.solver->time(), 0.0);
    EXPECT_EQ(state.mesh->geometry_epoch(), geometry.epoch);
    expect_primary_restored(*state.solver, primary);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsGeometryDependentThermalPrescribedHeatProfileWithoutReactingGas)
{
    auto state = make_case(Coupling::PISO, 0.0, 5);
    auto& prescribed_heat = state.coupling->add_prescribed_heat_source();
    prescribed_heat.initialize_gaussian(1.0, Handle::Vec3{0.5, 0.5, 0.0}, Handle::Vec3{1.0, 1.0, 0.5});
    const auto geometry = capture_geometry(*state.mesh);
    const auto primary = capture_primary(*state.solver);

    EXPECT_THROW(state.coupling->step(), std::invalid_argument);
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
        auto solver = std::make_shared<Solver>(
            mesh, planar_boundaries(), options, SimpleFluid::LinearSolverOptions{}, physical_options());
        auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
        solver_coupling->configure_material_feedback(material_feedback_options());
        solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
        EXPECT_THROW(
            static_cast<void>(solver_coupling->configure_free_surface(free_surface_options(4))), std::invalid_argument);
        EXPECT_FALSE(solver_coupling->planar_ale_enabled());
    }
}

TEST(BoussinesqPlanarALESupportMatrixTest, ConfigurationRejectsGeometryBeforePrimaryFieldsOrHistory)
{
    auto mesh = make_column();
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    auto invalid_geometry = free_surface_options(4);
    invalid_geometry.vessel.cross_section_area = 2.0;
    invalid_geometry.vessel.total_internal_volume = 4.0;
    const auto initial_primary = capture_primary(*solver);
    EXPECT_THROW(static_cast<void>(solver_coupling->configure_free_surface(invalid_geometry)), std::invalid_argument);
    EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
    EXPECT_TRUE(solver_coupling->free_surface_history().empty());
    EXPECT_FALSE(solver_coupling->planar_ale_enabled());
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
    auto solver = std::make_shared<Solver>(
        mesh, planar_boundaries(), time_options(coupling), SimpleFluid::LinearSolverOptions{}, physical_options());
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    EXPECT_THROW(
        static_cast<void>(solver_coupling->configure_free_surface(free_surface_options(4))), std::invalid_argument);
    EXPECT_FALSE(solver_coupling->planar_ale_enabled());
    EXPECT_EQ(solver_coupling->find_free_surface_model(), nullptr);
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsRankDivergentLevelTolerancePolicyBeforeMotion)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    auto mesh = make_column();
    auto solver_shared = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_material_feedback(material_feedback_options());
    solver_coupling.initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto options = free_surface_options(4);
    options.ale.level_tolerance_from_volume = comm->getRank() == 1;
    EXPECT_THROW(static_cast<void>(solver_coupling.configure_free_surface(options)), std::invalid_argument);
    EXPECT_FALSE(solver_coupling.planar_ale_enabled());
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsRankDivergentAitkenOptionsBeforeMotion)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    auto state = make_case(Coupling::PISO, 0.0, 4);
    for (int choice = 0; choice < 4; ++choice)
    {
        auto options = free_surface_options(4);
        if (comm->getRank() == 1)
        {
            if (choice == 0) options.ale.aitken = true;
            if (choice == 1) options.ale.initial_relaxation = 0.8;
            if (choice == 2) options.ale.minimum_relaxation = 0.2;
            if (choice == 3) { options.ale.maximum_relaxation = 0.9; options.ale.initial_relaxation = 0.9; }
        }
        EXPECT_THROW(state.coupling->configure_free_surface(options), std::invalid_argument);
    }
}

TEST(BoussinesqPlanarALESupportMatrixTest, RejectsDivergentMomentumPolicyBeforeMotion)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    auto mesh = make_column();
    auto solver_shared = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_material_feedback(material_feedback_options());
    solver_coupling.initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto momentum = solver.linear_solver_options();
    momentum.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    momentum.preconditioner = SimpleFluid::LinearPreconditioner::ILU0;
    solver.set_momentum_linear_solver_options(momentum);
    if (comm->getRank() == 1) solver.clear_momentum_linear_solver_options();
    EXPECT_THROW(
        static_cast<void>(solver_coupling.configure_free_surface(free_surface_options(4))), std::invalid_argument);
    EXPECT_FALSE(solver_coupling.planar_ale_enabled());
}

/** Integrate physical liquid sensible energy independently of the solver report. */
double coupling_sensible_energy(const ConfiguredCase& state)
{
    double local = 0.0;
    const auto& mass = state.coupling->liquid_mass_inventory().cellMassInventory();
    const auto& cp = state.solver->material_properties().specific_heat_capacity;
    for (size_t owned = 0; owned < state.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        local += state.mesh->cell_volume(cell) * mass.value(cell) * cp.value(cell) *
            state.solver->temperature().value(cell);
    }
    double global = 0.0;
    Teuchos::reduceAll(*state.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

TEST(BoussinesqCouplingIntervalTest, ReplaysMovingNonuniformEnergyAndGasAfterTwoAcceptedSubsteps)
{
    auto state = make_case(Coupling::PISO, 0.0, 25, true, 0.0, false, SimpleFluid::ReactingPressureMode::Constant,
        false, SimpleFluid::CoupledOperatorBackend::Assembled, SimpleFluid::CoupledWorkspacePolicy::CachedProducts,
        SimpleFluid::FVM::CellGradientScheme::LeastSquares, true);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    auto& coupling = solver_coupling;
    auto* gas = coupling.find_reacting_gas_model();
    const auto initial_energy = coupling_sensible_energy(state);
    const auto initial_species_moles = gas->global_species_moles(SurfaceGas::gas_id);
    const auto initial_history = solver_coupling.free_surface_history().size();
    std::vector<double> energy(state.mesh->num_owned_cells());
    for (size_t owned = 0; owned < energy.size(); ++owned)
        energy[owned] = state.mesh->cell_centroid(static_cast<Pack::local_ordinal_type>(owned)).z < 0.5 ? 0.01 : 0.02;
    auto checkpoint = coupling.create_coupling_checkpoint();
    EXPECT_THROW(static_cast<void>(solver_coupling.create_coupling_checkpoint()), std::logic_error);
    solver.reset_solver_phase_timings();
    coupling.set_coupling_interval_energy(energy, 0.02);
    coupling.step();
    EXPECT_NEAR(solver_coupling.find_prescribed_heat_source()->integrated_power(), 1.5, 1.e-13);
    EXPECT_THROW(solver_coupling.accept_coupling_checkpoint(checkpoint), std::logic_error);
    EXPECT_THROW(solver_coupling.set_coupling_interval_energy(energy, 0.02), std::logic_error);
    solver_coupling.step();
    const auto accepted_time = solver.time();
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::GeometryRefresh).calls,
        solver.solver_phase_timing(SimpleFluid::SolverPhase::ALEGeometry).calls);
    EXPECT_GT(solver.solver_phase_timing(SimpleFluid::SolverPhase::ALERestore).calls, 0u);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::GasRestore).calls,
        solver.solver_phase_timing(SimpleFluid::SolverPhase::ALERestore).calls);
    const auto accepted_level = top_elevation(*state.mesh);
    const auto accepted_energy = coupling_sensible_energy(state);
    const auto accepted_species_moles = gas->global_species_moles(SurfaceGas::gas_id);
    const auto accepted_generation = SurfaceGas::supplied(*gas);
    const auto advanced_epoch = state.mesh->geometry_epoch();
    const auto accepted_geometry = capture_geometry(*state.mesh);
    const auto accepted_primary = capture_primary(solver);
    const auto accepted_absolute_flux = capture_owned_face_values(solver.pressure_corrected_face_fluxes());
    const auto accepted_relative_flux = capture_owned_face_values(solver_coupling.mesh_relative_face_fluxes());
    const auto accepted_mass = capture_owned_values(solver_coupling.liquid_mass_inventory().cellMassInventory());
    const auto accepted_gas_fields = capture_gas_fields(*gas);
    const auto accepted_species_receipt = gas->species_event_receipt();
    const auto accepted_species_histories = gas->species_histories();
    const auto accepted_source_ledgers = gas->source_ledgers();
    const auto accepted_history = solver_coupling.free_surface_history();
    std::vector<double> accepted_temperature;
    for (size_t owned = 0; owned < energy.size(); ++owned)
        accepted_temperature.push_back(solver.temperature().value(static_cast<Pack::local_ordinal_type>(owned)));
    EXPECT_NEAR(accepted_energy - initial_energy, 0.03, 2.e-8);
    EXPECT_GT(accepted_level, 1.0);
    EXPECT_NEAR(accepted_generation, 0.03 * SurfaceGas::supplied_mol_per_joule, 1.e-16);
    EXPECT_NEAR(accepted_species_moles - initial_species_moles, accepted_generation, 2.e-12);
    EXPECT_NEAR(SurfaceGas::inventory_error(*gas), 0.0, 2.e-12);
    EXPECT_NEAR(SurfaceGas::history(*gas).external_supply, accepted_generation, 2e-12);
    EXPECT_NEAR(SurfaceGas::history(*gas).current_moles + SurfaceGas::history(*gas).physical_outflow,
        SurfaceGas::history(*gas).initial_moles + accepted_generation, 2e-12);

    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_GT(state.mesh->geometry_epoch(), advanced_epoch);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), 1.0);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), initial_history);
    EXPECT_DOUBLE_EQ(SurfaceGas::supplied(*gas), 0.0);
    EXPECT_NEAR(gas->global_species_moles(SurfaceGas::gas_id), initial_species_moles, 1.e-14);
    EXPECT_DOUBLE_EQ(SurfaceGas::history(*gas).external_supply, 0);
    EXPECT_NEAR(SurfaceGas::history(*gas).current_moles, initial_species_moles, 1e-14);
    auto alternate_energy = energy;
    for (auto& value : alternate_energy)
        value *= 0.5;
    solver_coupling.set_coupling_interval_energy(alternate_energy, 0.02);
    solver_coupling.step();
    solver_coupling.step();
    EXPECT_LT(coupling_sensible_energy(state), accepted_energy);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), 1.0);
    EXPECT_DOUBLE_EQ(SurfaceGas::supplied(*gas), 0.0);
    solver_coupling.set_coupling_interval_energy(energy, 0.02);
    solver_coupling.step();
    solver_coupling.step();
    EXPECT_DOUBLE_EQ(solver.time(), accepted_time);
    EXPECT_NEAR(top_elevation(*state.mesh), accepted_level, 1.e-13);
    EXPECT_NEAR(coupling_sensible_energy(state), accepted_energy, 2.e-8);
    EXPECT_NEAR(SurfaceGas::supplied(*gas), accepted_generation, 1.e-16);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), initial_history + 2);
    expect_geometry_restored(*state.mesh, accepted_geometry);
    expect_primary_restored(solver, accepted_primary);
    expect_owned_face_values(solver.pressure_corrected_face_fluxes(), accepted_absolute_flux);
    expect_owned_face_values(solver_coupling.mesh_relative_face_fluxes(), accepted_relative_flux);
    expect_owned_values(solver_coupling.liquid_mass_inventory().cellMassInventory(), accepted_mass);
    expect_gas_fields_restored(*gas, accepted_gas_fields);
    EXPECT_EQ(gas->species_histories(), accepted_species_histories);
    SurfaceGas::M::expect_same_ledgers(gas->source_ledgers(), accepted_source_ledgers);
    EXPECT_EQ(gas->species_event_receipt().revision, accepted_species_receipt.revision);
    EXPECT_DOUBLE_EQ(gas->species_event_receipt().interval_end, accepted_species_receipt.interval_end);
    EXPECT_DOUBLE_EQ(gas->species_event_receipt().time_step, accepted_species_receipt.time_step);
    EXPECT_GT(gas->species_event_receipt().publication_nonce, accepted_species_receipt.publication_nonce);
    ASSERT_EQ(solver_coupling.free_surface_history().size(), accepted_history.size());
    for (size_t step = 0; step < accepted_history.size(); ++step)
    {
        EXPECT_DOUBLE_EQ(solver_coupling.free_surface_history()[step].free_surface.pool_level,
            accepted_history[step].free_surface.pool_level);
        EXPECT_DOUBLE_EQ(solver_coupling.free_surface_history()[step].liquid_mass.total_mass,
            accepted_history[step].liquid_mass.total_mass);
    }
    for (size_t owned = 0; owned < energy.size(); ++owned)
        EXPECT_NEAR(solver.temperature().value(static_cast<Pack::local_ordinal_type>(owned)),
            accepted_temperature[owned], 1.e-11);
    coupling.accept_coupling_checkpoint(checkpoint);
    EXPECT_THROW(solver_coupling.restore_coupling_checkpoint(checkpoint), std::invalid_argument);
}

TEST(BoussinesqCouplingIntervalTest, RepeatedReplayPreservesAcceptedHistoryPrefix)
{
    auto state = make_case(Coupling::PISO, 0.0, 4);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    solver_coupling.step();
    solver_coupling.step();
    const auto prefix = solver_coupling.free_surface_history();
    ASSERT_GE(prefix.size(), 2U);
    const auto accepted_time = solver.time();
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    for (int replay = 0; replay < 2; ++replay)
    {
        solver_coupling.step();
        ASSERT_EQ(solver_coupling.free_surface_history().size(), prefix.size() + 1);
        solver_coupling.restore_coupling_checkpoint(checkpoint);
        EXPECT_DOUBLE_EQ(solver.time(), accepted_time);
        ASSERT_EQ(solver_coupling.free_surface_history().size(), prefix.size());
        for (size_t record = 0; record < prefix.size(); ++record)
        {
            const auto& actual = solver_coupling.free_surface_history()[record];
            EXPECT_DOUBLE_EQ(actual.free_surface.pool_level, prefix[record].free_surface.pool_level);
            EXPECT_DOUBLE_EQ(actual.liquid_mass.total_mass, prefix[record].liquid_mass.total_mass);
            ASSERT_EQ(actual.planar_ale.has_value(), prefix[record].planar_ale.has_value());
            if (actual.planar_ale)
            {
                EXPECT_EQ(actual.planar_ale->level_residual_history, prefix[record].planar_ale->level_residual_history);
                EXPECT_EQ(actual.planar_ale->target_change_history, prefix[record].planar_ale->target_change_history);
                EXPECT_EQ(actual.planar_ale->continuity_maximum_history, prefix[record].planar_ale->continuity_maximum_history);
                EXPECT_EQ(actual.planar_ale->material_state_residual_history, prefix[record].planar_ale->material_state_residual_history);
                EXPECT_EQ(actual.planar_ale->gas_state_residual_history, prefix[record].planar_ale->gas_state_residual_history);
            }
        }
    }
    solver_coupling.step();
    solver_coupling.accept_coupling_checkpoint(checkpoint);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), prefix.size() + 1);
}

TEST(BoussinesqCouplingIntervalTest, SignedCoolingIntervalConservesEnergyAndReplaysMovingGeometry)
{
    auto state = make_case(Coupling::PISO, 0.0, 25);
    auto& coupling = *state.coupling;
    coupling.add_prescribed_heat_source().initialize_constant(0.0);
    const auto initial_energy = coupling_sensible_energy(state);
    const auto initial_history = coupling.free_surface_history().size();
    auto checkpoint = coupling.create_coupling_checkpoint();
    std::vector<double> energy(state.mesh->num_owned_cells());
    for (size_t owned = 0; owned < energy.size(); ++owned)
        energy[owned] = -2.0 * state.mesh->cell_volume(static_cast<Pack::local_ordinal_type>(owned));
    coupling.set_coupling_interval_energy(energy, 0.02);
    coupling.step();
    coupling.step();
    EXPECT_NEAR(coupling_sensible_energy(state) - initial_energy, -2.0, 2.e-8);
    EXPECT_LT(top_elevation(*state.mesh), 1.0);
    const auto accepted_level = top_elevation(*state.mesh);
    coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_NEAR(coupling_sensible_energy(state), initial_energy, 2.e-8);
    EXPECT_EQ(coupling.free_surface_history().size(), initial_history);
    coupling.set_coupling_interval_energy(energy, 0.02);
    coupling.step();
    coupling.step();
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), accepted_level);
    EXPECT_NEAR(coupling_sensible_energy(state) - initial_energy, -2.0, 2.e-8);
    coupling.accept_coupling_checkpoint(checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, RejectsInvalidBudgetAndRestoresSourceAfterFailedTrial)
{
    auto state = make_case(Coupling::PISO, 0.0, 1);
    state.coupling->add_prescribed_heat_source().initialize_constant(0.0);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    std::vector<double> invalid(state.mesh->num_owned_cells(), 0.0);
    if (state.mesh->owned_cell_map()->getComm()->getRank() == 0 && !invalid.empty())
        invalid[0] = std::numeric_limits<double>::quiet_NaN(); // Signed cooling is a valid budget.
    EXPECT_THROW(solver_coupling.set_coupling_interval_energy(invalid, 0.01), std::exception);
    std::vector<double> energy(state.mesh->num_owned_cells(), 0.01);
    EXPECT_THROW(solver_coupling.set_coupling_interval_energy(energy, 0.0), std::invalid_argument);
    solver_coupling.set_coupling_interval_energy(energy, 0.01);
    const auto before_source = solver_coupling.find_prescribed_heat_source()->integrated_power();
    EXPECT_THROW(solver_coupling.step(), std::runtime_error);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), 1.0);
    EXPECT_DOUBLE_EQ(solver_coupling.find_prescribed_heat_source()->integrated_power(), before_source);
    const Solver& observer = solver;
    auto& observer_coupling = solver_coupling;
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::logic_error);
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::logic_error);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_DOUBLE_EQ(solver_coupling.find_prescribed_heat_source()->integrated_power(), 0.0);
    EXPECT_FALSE(solver_coupling.find_prescribed_heat_source()->has_interval_energy());
    EXPECT_NO_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint));
    solver_coupling.accept_coupling_checkpoint(checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, AcceptancePreflightPreservesPartialAndCompleteCheckpoints)
{
    auto state = make_case(Coupling::PISO, 0.0, 25);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    const Solver& observer = solver;
    auto& observer_coupling = solver_coupling;
    auto& source = solver_coupling.add_prescribed_heat_source();
    source.initialize_constant(0.0);
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    const std::vector<double> energy(state.mesh->num_owned_cells(), 0.01);
    solver_coupling.set_coupling_interval_energy(energy, 0.02);
    solver_coupling.step();
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::logic_error);
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::logic_error);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_FALSE(source.has_interval_energy());
    solver_coupling.set_coupling_interval_energy(energy, 0.02);
    solver_coupling.step();
    solver_coupling.step();

    const auto primary = capture_primary(observer);
    const auto source_values = capture_owned_values(source.field());
    const auto mass_values = capture_owned_values(solver_coupling.liquid_mass_inventory().cellMassInventory());
    const auto geometry = capture_geometry(*state.mesh);
    const auto history = solver_coupling.free_surface_history();
    const double power = source.integrated_power();
    EXPECT_NO_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint));
    EXPECT_NO_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint));
    EXPECT_DOUBLE_EQ(solver.time(), 0.02);
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_EQ(state.mesh->geometry_epoch(), geometry.epoch);
    EXPECT_EQ(capture_geometry(*state.mesh).cell_volumes, geometry.cell_volumes);
    expect_primary_restored(observer, primary);
    expect_owned_values(source.field(), source_values);
    expect_owned_values(solver_coupling.liquid_mass_inventory().cellMassInventory(), mass_values);
    EXPECT_TRUE(source.has_interval_energy());
    EXPECT_DOUBLE_EQ(source.interval_duration(), 0.02);
    EXPECT_DOUBLE_EQ(source.interval_end_time(), 0.02);
    EXPECT_DOUBLE_EQ(source.integrated_power(), power);
    ASSERT_EQ(solver_coupling.free_surface_history().size(), history.size());
    for (size_t record = 0; record < history.size(); ++record)
    {
        EXPECT_DOUBLE_EQ(solver_coupling.free_surface_history()[record].free_surface.pool_level,
            history[record].free_surface.pool_level);
        EXPECT_DOUBLE_EQ(solver_coupling.free_surface_history()[record].liquid_mass.total_mass,
            history[record].liquid_mass.total_mass);
    }
    EXPECT_THROW(static_cast<void>(solver_coupling.create_coupling_checkpoint()), std::logic_error);
    solver_coupling.accept_coupling_checkpoint(checkpoint);
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::invalid_argument);
    EXPECT_THROW(solver_coupling.accept_coupling_checkpoint(checkpoint), std::invalid_argument);
}

TEST(BoussinesqCouplingIntervalTest, AcceptancePreflightRejectsForeignAndRankLocalForeignTokens)
{
    auto first = make_case(Coupling::PISO, 0.0, 25);
    auto second = make_case(Coupling::PISO, 0.0, 25);
    auto first_checkpoint = first.coupling->create_coupling_checkpoint();
    auto second_checkpoint = second.coupling->create_coupling_checkpoint();
    const Solver& first_observer = *first.solver;
    auto& first_observer_coupling = *first.coupling;
    const Solver& second_observer = *second.solver;
    auto& second_observer_coupling = *second.coupling;
    EXPECT_THROW(
        first_observer_coupling.validate_coupling_checkpoint_acceptance(second_checkpoint), std::invalid_argument);
    const auto& rank_local_token = Tpetra::getDefaultComm()->getRank() == 0
        ? second_checkpoint : first_checkpoint;
    EXPECT_THROW(first_observer_coupling.validate_coupling_checkpoint_acceptance(rank_local_token), std::exception);
    EXPECT_NO_THROW(first_observer_coupling.validate_coupling_checkpoint_acceptance(first_checkpoint));
    EXPECT_NO_THROW(second_observer_coupling.validate_coupling_checkpoint_acceptance(second_checkpoint));
    first.coupling->restore_coupling_checkpoint(first_checkpoint);
    second.coupling->restore_coupling_checkpoint(second_checkpoint);
    first.coupling->accept_coupling_checkpoint(first_checkpoint);
    second.coupling->accept_coupling_checkpoint(second_checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, ShortEnergyIntervalCannotBeAcceptedWithoutAdvancing)
{
    auto state = make_case(Coupling::PISO, 0.0, 25);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    const std::vector<double> energy(state.mesh->num_owned_cells(), 1.e-6);
    solver_coupling.set_coupling_interval_energy(energy, 1.e-16);
    const Solver& observer = solver;
    auto& observer_coupling = solver_coupling;
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::logic_error);
    EXPECT_THROW(solver_coupling.accept_coupling_checkpoint(checkpoint), std::logic_error);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_FALSE(solver_coupling.find_prescribed_heat_source()->has_interval_energy());
    solver_coupling.accept_coupling_checkpoint(checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, AcceptanceRejectsOneUlpShortOfCanonicalIntervalEnd)
{
    auto state = make_case(Coupling::PISO, 0.0, 25);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    const Solver& observer = solver;
    auto& observer_coupling = solver_coupling;
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    solver.set_time_step(0.002);
    solver_coupling.step();
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    const double start = solver.time();
    const double canonical_end = start + 5.e-5;
    const double duration = canonical_end - start;
    const std::vector<double> energy(state.mesh->num_owned_cells(), 0.0);
    solver_coupling.set_coupling_interval_energy(energy, duration);
    const double partial_end = std::nextafter(canonical_end, start);
    solver.set_time_step(partial_end - start);
    solver_coupling.step();
    EXPECT_DOUBLE_EQ(solver.time(), partial_end);
    ASSERT_GT(canonical_end - solver.time(),
        32.0 * std::numeric_limits<double>::epsilon() * duration);
    EXPECT_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint), std::logic_error);
    EXPECT_THROW(solver_coupling.accept_coupling_checkpoint(checkpoint), std::logic_error);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    solver_coupling.set_coupling_interval_energy(energy, duration);
    solver.set_time_step(duration);
    solver_coupling.step();
    EXPECT_DOUBLE_EQ(solver.time(), canonical_end);
    EXPECT_NO_THROW(observer_coupling.validate_coupling_checkpoint_acceptance(checkpoint));
    solver_coupling.accept_coupling_checkpoint(checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, RejectsEndpointOverrunAndUnresolvableSourceTimes)
{
    auto state = make_case(Coupling::PISO, 0.0, 25);
    auto& source = state.coupling->add_prescribed_heat_source();
    const std::vector<double> energy(state.mesh->num_owned_cells(), 1.e-6);
    source.set_interval_energy(energy, 0.0, 1.e-16);
    const double short_power = source.integrated_power();
    EXPECT_THROW(source.refresh_interval_energy(1.e-16, 1.e-16), std::invalid_argument);
    EXPECT_THROW(source.refresh_interval_energy(0.0, 2.e-16), std::invalid_argument);
    EXPECT_DOUBLE_EQ(source.integrated_power(), short_power);

    source.set_interval_energy(energy, 1.0, 1.0);
    const double power = source.integrated_power();
    EXPECT_THROW(source.refresh_interval_energy(1.0, std::numeric_limits<double>::denorm_min()),
                 std::invalid_argument);
    EXPECT_THROW(source.set_interval_energy(energy, 1.e8, 1.e-8), std::invalid_argument);
    EXPECT_DOUBLE_EQ(source.integrated_power(), power);
    EXPECT_DOUBLE_EQ(source.interval_end_time(), 2.0);

    // Even advancing by one representable clock tick must not silently round
    // a requested substep by a significant fraction of the energy interval.
    const double start = 1.e8;
    const double tick = std::nextafter(start, std::numeric_limits<double>::infinity()) - start;
    source.set_interval_energy(energy, start, 2.0 * tick);
    const double large_clock_power = source.integrated_power();
    EXPECT_THROW(source.refresh_interval_energy(start, 1.25 * tick), std::invalid_argument);
    EXPECT_DOUBLE_EQ(source.integrated_power(), large_clock_power);
    EXPECT_NO_THROW(source.refresh_interval_energy(start, tick));
    EXPECT_NO_THROW(source.refresh_interval_energy(start + tick, tick));
}

TEST(BoussinesqCouplingIntervalTest, UnequalSubcyclesReplayTheSameEnergyAndHydrogenBudgets)
{
    auto state = make_case(Coupling::PISO, 0.0, 25, true, 0.0, false, SimpleFluid::ReactingPressureMode::Constant,
        false, SimpleFluid::CoupledOperatorBackend::Assembled, SimpleFluid::CoupledWorkspacePolicy::CachedProducts,
        SimpleFluid::FVM::CellGradientScheme::LeastSquares, true);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    auto* gas = solver_coupling.find_reacting_gas_model();
    const auto before_energy = coupling_sensible_energy(state);
    const auto before_species_moles = gas->global_species_moles(SurfaceGas::gas_id);
    const auto history_size = solver_coupling.free_surface_history().size();
    std::vector<double> energy(state.mesh->num_owned_cells());
    for (size_t owned = 0; owned < energy.size(); ++owned)
        energy[owned] = state.mesh->cell_centroid(static_cast<Pack::local_ordinal_type>(owned)).z < 0.5 ? 0.01 : 0.02;
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    std::vector<double> first_temperatures;
    double first_level = 0.0;
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        SCOPED_TRACE(attempt);
        if (attempt != 0)
            solver_coupling.restore_coupling_checkpoint(checkpoint);
        solver_coupling.set_coupling_interval_energy(energy, 0.02);
        for (double dt : {0.003, 0.007, 0.01})
        {
            solver.set_time_step(dt);
            solver_coupling.step();
        }
        EXPECT_NEAR(solver.time(), 0.02, 1.e-17);
        EXPECT_NEAR(coupling_sensible_energy(state) - before_energy, 0.03, 2.e-8);
        const double generated = 0.03 * SurfaceGas::supplied_mol_per_joule;
        EXPECT_NEAR(SurfaceGas::supplied(*gas), generated, 1.e-16);
        EXPECT_NEAR(gas->global_species_moles(SurfaceGas::gas_id) - before_species_moles, generated, 2.e-12);
        EXPECT_NEAR(SurfaceGas::inventory_error(*gas), 0.0, 2.e-12);
        EXPECT_EQ(solver_coupling.free_surface_history().size(), history_size + 3);
        for (size_t owned = 0; owned < energy.size(); ++owned)
        {
            const double temperature = solver.temperature().value(static_cast<Pack::local_ordinal_type>(owned));
            if (attempt == 0)
                first_temperatures.push_back(temperature);
            else
                EXPECT_NEAR(temperature, first_temperatures[owned], 1.e-11);
        }
        if (attempt == 0)
            first_level = top_elevation(*state.mesh);
        else
            EXPECT_NEAR(top_elevation(*state.mesh), first_level, 1.e-13);
    }
    solver_coupling.accept_coupling_checkpoint(checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, ConsecutiveSmallAnnularIntervalsReplayEnergyAndHydrogen)
{
    // Match the physical scales and solver configuration of the external host's
    // annular consumer: the thermal increment is tiny relative to M*cp*T.
    constexpr double inner_radius = 0.04;
    constexpr double outer_radius = 0.25;
    constexpr double initial_height = 0.5;
    constexpr double initial_mass = 148.20868454718243;
    constexpr double specific_heat = 4200.0;
    constexpr double initial_temperature = 300.0;
    constexpr double interval_duration = 1.e-4;
    constexpr double interval_energy = 0.10133263268392909;
    constexpr double external_mol_per_joule = 1.5e-7;
    const double area = std::numbers::pi * (outer_radius * outer_radius - inner_radius * inner_radius);
    const double density = initial_mass / (area * initial_height);
    const auto uniform_edges = [](double begin, double end, int cells)
    {
        SimpleFluid::ArrReal result(static_cast<size_t>(cells) + 1);
        for (int i = 0; i <= cells; ++i)
            result[static_cast<size_t>(i)] = std::lerp(begin, end, static_cast<double>(i) / cells);
        return result;
    };
    auto geometry = std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{uniform_edges(inner_radius, outer_radius, 2),
            uniform_edges(0.0, 2.0 * std::numbers::pi, 4), uniform_edges(0.0, initial_height, 4)}});
    const auto communicator = Tpetra::getDefaultComm();
    Handle::DistributionOptions distribution;
    distribution.partition = static_cast<size_t>(communicator->getRank());
    distribution.partitions = static_cast<size_t>(communicator->getSize());
    distribution.allow_empty_partitions = true;
    auto mesh = std::make_shared<Handle>(std::move(geometry), distribution);

    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"rmin", "rmax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    boundaries.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    SimpleFluid::TimeStepperOptions time;
    time.time_step = interval_duration;
    time.steps = 1;
    time.reference_temperature = initial_temperature;
    time.thermal_expansion = 2.1e-4;
    time.gravity_z = 0.0;
    time.pressure_velocity_coupling = Coupling::PISO;
    time.n_pressure_correctors = 2;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.e-11;
    linear.max_iterations = 1000;
    SimpleFluid::BoussinesqModelOptions material;
    material.reference_density = density;
    material.density = density;
    material.specific_heat_capacity = specific_heat;
    material.dynamic_viscosity = 1.e-3;
    material.thermal_conductivity = 0.6;
    auto solver_owner = std::make_shared<Solver>(mesh, boundaries, time, linear, material);
    auto solver_owner_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver_owner);
    auto& solver = *solver_owner;
    auto& solver_coupling = *solver_owner_coupling;
    SimpleFluid::MaterialFeedbackOptions thermal;
    thermal.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    thermal.reference_density = density;
    thermal.liquid_density = density;
    thermal.reference_temperature = initial_temperature;
    thermal.thermal_expansion = 2.1e-4;
    thermal.reference_dynamic_viscosity = 1.e-3;
    solver_coupling.configure_material_feedback(thermal);
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    SurfaceGas::Initial initial;
    initial.liquid_density = density;
    initial.reference_temperature = initial_temperature;
    initial.expansion = 2.1e-4;
    initial.compressibility = 4.5e-10;
    initial.dissolved_concentration = 0;
    auto gas_options = SurfaceGas::options(
        initial, SurfaceGas::prescribed_power(&solver_coupling.find_prescribed_heat_source()->field()),
        external_mol_per_joule);
    gas_options.pressure_mode = SimpleFluid::ReactingPressureMode::Constant;
    gas_options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    gas_options.constant_slip_velocity = 0.0;
    gas_options.free_surface_patches = {"zmax"};
    auto& gas = solver_coupling.configure_reacting_gas(gas_options);
    // Accepted catalog-keyed source/escape histories own replay, not a donor-deficit proxy.
    solver_coupling.initialize_linear_temperature({0.0, 0.0, 1.0}, initial_temperature, initial_temperature);
    SimpleFluid::FreeSurfaceOptions surface;
    surface.enabled = true;
    surface.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    surface.gravity_axis = SimpleFluid::Dimension::Z;
    surface.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.initial_liquid_volume = area * initial_height;
    surface.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    surface.vessel.bottom_elevation = 0.0;
    surface.vessel.top_elevation = 1.0;
    surface.vessel.cross_section_area = area;
    surface.vessel.total_internal_volume = area;
    surface.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    surface.headspace.ambient_pressure = 101325.0;
    surface.headspace.initial_pressure = 101325.0;
    surface.headspace.initial_temperature = initial_temperature;
    surface.ale.top_boundary = "zmax";
    surface.ale.maximum_correctors = 30;
    ASSERT_NE(solver_coupling.configure_free_surface(surface), nullptr);
    ConfiguredCase state{mesh, std::move(solver_owner), std::move(solver_owner_coupling), nullptr};

    // Axisymmetric, nonuniform radial/axial source; normalize owned volumes
    // so serial and independently partitioned MPI runs receive the same joules.
    const std::array<double, 4> axial_shape{0.8, 1.2, 1.5, 0.9};
    std::vector<double> energy(mesh->num_owned_cells());
    double local_weight = 0.0;
    for (size_t owned = 0; owned < energy.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto gid = mesh->cell_global_id(cell);
        const size_t radial = static_cast<size_t>(gid) % 2;
        const size_t axial = static_cast<size_t>(gid) / 8;
        energy[owned] = mesh->cell_volume(cell) * axial_shape[axial] * (radial == 0 ? 1.0 : 0.7);
        local_weight += energy[owned];
    }
    double global_weight = 0.0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_weight, &global_weight);
    for (auto& value : energy)
        value *= interval_energy / global_weight;
    const auto initial_sensible_energy = coupling_sensible_energy(state);
    const auto initial_history_size = solver_coupling.free_surface_history().size();
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().totalMass(), initial_mass, 2.e-11);

    for (int interval = 0; interval < 2; ++interval)
    {
        SCOPED_TRACE(interval);
        auto checkpoint = solver_coupling.create_coupling_checkpoint();
        std::vector<double> first_temperatures;
        double first_level = 0.0;
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            SCOPED_TRACE(attempt);
            solver_coupling.restore_coupling_checkpoint(checkpoint);
            solver_coupling.set_coupling_interval_energy(energy, interval_duration);
            ASSERT_NO_THROW(solver_coupling.step());
            const double cumulative_energy = (interval + 1) * interval_energy;
            const double produced = cumulative_energy * external_mol_per_joule;
            EXPECT_NEAR(solver.time(), (interval + 1) * interval_duration, 1.e-19);
            EXPECT_NEAR(coupling_sensible_energy(state) - initial_sensible_energy, cumulative_energy, 2.e-6);
            // Keep the established cumulative liquid-mass conservation gate
            // at this 148 kg scale, independently of the transport solver.
            EXPECT_NEAR(solver_coupling.liquid_mass_inventory().totalMass(), initial_mass,
                std::max(4096.0 * std::numeric_limits<double>::epsilon(), 1.e-10) * initial_mass);
            EXPECT_NEAR(SurfaceGas::supplied(gas), produced, 1.e-18);
            EXPECT_NEAR(gas.global_species_moles(SurfaceGas::gas_id) + SurfaceGas::escaped(gas), produced, 1.e-17);
            EXPECT_NEAR(SurfaceGas::inventory_error(gas), 0.0, 1.e-17);
            EXPECT_NEAR(top_elevation(*mesh), solver_coupling.free_surface_diagnostics().pool_level,
                surface.ale.level_absolute_tolerance + surface.ale.level_relative_tolerance);
            EXPECT_EQ(solver_coupling.free_surface_history().size(), initial_history_size + interval + 1);
            for (size_t owned = 0; owned < energy.size(); ++owned)
            {
                const auto cell = static_cast<Pack::local_ordinal_type>(owned);
                const auto temperature = solver.temperature().value(cell);
                if (attempt == 0)
                    first_temperatures.push_back(temperature);
                else
                    EXPECT_NEAR(temperature, first_temperatures[owned], 1.e-11);
            }
            if (attempt == 0)
                first_level = top_elevation(*mesh);
            else
                EXPECT_NEAR(top_elevation(*mesh), first_level, 1.e-13);
        }
        solver_coupling.accept_coupling_checkpoint(checkpoint);
    }
}

TEST(BoussinesqCouplingIntervalTest, R100ConsecutiveIntervalsPreserveLiquidMass)
{
    // The real R100 coupling case, including its bulk/gas solver tolerances.
    // A 1 W source must not accumulate liquid mass error over accepted intervals.
    constexpr double inner_radius = 0.03815;
    constexpr double outer_radius = 0.25;
    constexpr double bottom = 0.1;
    constexpr double initial_top = 0.60852;
    constexpr double vessel_top = 2.1;
    constexpr double initial_mass = 151.09200565656749;
    constexpr double initial_temperature = 298.85;
    constexpr double interval_duration = 0.01;
    constexpr double interval_energy = 0.01;
    constexpr double external_mol_per_joule = 1.5e-7;
    constexpr int radial_cells = 4, angular_cells = 8, axial_cells = 8;
    const double area = std::numbers::pi * (outer_radius * outer_radius - inner_radius * inner_radius);
    const double density = initial_mass / (area * (initial_top - bottom));
    const double mass_tolerance = 4096.0 * std::numeric_limits<double>::epsilon() * initial_mass;
    const auto uniform_edges = [](double begin, double end, int cells)
    {
        SimpleFluid::ArrReal result(static_cast<size_t>(cells) + 1);
        for (int i = 0; i <= cells; ++i)
            result[static_cast<size_t>(i)] = std::lerp(begin, end, static_cast<double>(i) / cells);
        return result;
    };
    auto geometry = std::make_shared<Handle::Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{uniform_edges(inner_radius, outer_radius, radial_cells),
            uniform_edges(0.0, 2.0 * std::numbers::pi, angular_cells),
            uniform_edges(bottom, initial_top, axial_cells)}});
    const auto communicator = Tpetra::getDefaultComm();
    Handle::DistributionOptions distribution;
    distribution.partition = static_cast<size_t>(communicator->getRank());
    distribution.partitions = static_cast<size_t>(communicator->getSize());
    distribution.allow_empty_partitions = true;
    auto mesh = std::make_shared<Handle>(std::move(geometry), distribution);
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"rmin", "rmax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    boundaries.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    SimpleFluid::TimeStepperOptions time;
    time.time_step = 0.005;
    time.steps = 1;
    time.reference_temperature = initial_temperature;
    time.thermal_expansion = 2.1e-4;
    time.gravity_z = -9.81;
    time.pressure_velocity_coupling = Coupling::PISO;
    time.n_pressure_correctors = 2;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.e-11;
    linear.max_iterations = 1000;
    SimpleFluid::BoussinesqModelOptions material;
    material.reference_density = density;
    material.density = density;
    material.specific_heat_capacity = 4200.0;
    material.dynamic_viscosity = 1.e-3;
    material.thermal_conductivity = 0.6;
    auto solver_shared = std::make_shared<Solver>(mesh, boundaries, time, linear, material);
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::MaterialFeedbackOptions thermal;
    thermal.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    thermal.reference_density = density;
    thermal.liquid_density = density;
    thermal.reference_temperature = initial_temperature;
    thermal.thermal_expansion = 2.1e-4;
    thermal.reference_dynamic_viscosity = 1.e-3;
    solver_coupling.configure_material_feedback(thermal);
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    SurfaceGas::Initial initial;
    initial.liquid_density = density;
    initial.reference_temperature = initial_temperature;
    initial.expansion = 2.1e-4;
    initial.compressibility = 4.5e-10;
    initial.dissolved_concentration = 0;
    auto gas_options = SurfaceGas::options(
        initial, SurfaceGas::prescribed_power(&solver_coupling.find_prescribed_heat_source()->field()),
        external_mol_per_joule);
    gas_options.pressure_mode = SimpleFluid::ReactingPressureMode::Constant;
    gas_options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    gas_options.constant_slip_velocity = 0.01;
    gas_options.free_surface_patches = {"zmax"};
    gas_options.transport_solver_tolerance = 1e-14;
    auto& gas = solver_coupling.configure_reacting_gas(gas_options);
    // Accepted catalog-keyed source/escape histories own replay, not a donor-deficit proxy.
    solver_coupling.initialize_linear_temperature({0.0, 0.0, 1.0}, initial_temperature, initial_temperature);
    SimpleFluid::FreeSurfaceOptions surface;
    surface.enabled = true;
    surface.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    surface.gravity_axis = SimpleFluid::Dimension::Z;
    surface.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.initial_liquid_volume = area * (initial_top - bottom);
    surface.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    surface.vessel.bottom_elevation = bottom;
    surface.vessel.top_elevation = vessel_top;
    surface.vessel.cross_section_area = area;
    surface.vessel.total_internal_volume = area * (vessel_top - bottom);
    surface.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    surface.headspace.ambient_pressure = 101325.0;
    surface.headspace.initial_pressure = 101325.0;
    surface.headspace.initial_temperature = initial_temperature;
    surface.ale.top_boundary = "zmax";
    surface.ale.maximum_correctors = 30;
    ASSERT_NE(solver_coupling.configure_free_surface(surface), nullptr);

    const std::array<double, axial_cells> axial_shape{0.5, 0.8, 1.2, 1.5, 1.4, 1.1, 0.8, 0.5};
    const std::array<double, radial_cells> radial_shape{1.0, 1.2, 0.9, 0.7};
    std::vector<double> energy(mesh->num_owned_cells());
    double local_weight = 0.0;
    for (size_t owned = 0; owned < energy.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto gid = static_cast<size_t>(mesh->cell_global_id(cell));
        const size_t radial = gid % radial_cells;
        const size_t axial = gid / (radial_cells * angular_cells);
        energy[owned] = mesh->cell_volume(cell) * axial_shape[axial] * radial_shape[radial];
        local_weight += energy[owned];
    }
    double global_weight = 0.0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_weight, &global_weight);
    for (auto& value : energy)
        value *= interval_energy / global_weight;
    const auto initial_history_size = solver_coupling.free_surface_history().size();
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().totalMass(), initial_mass, mass_tolerance);

    size_t accepted_substeps = 0;
    for (int interval = 0; interval < 12; ++interval)
    {
        SCOPED_TRACE(interval);
        const double start = interval * interval_duration;
        const double end = (interval + 1) * interval_duration;
        const double duration = end - start;
        // Match the consumer's endpoint arithmetic: a represented 0.01 s
        // interval may need three substeps under the 0.005 s maximum.
        const int substeps = static_cast<int>(std::ceil(duration / time.time_step));
        auto checkpoint = solver_coupling.create_coupling_checkpoint();
        std::vector<double> first_temperatures;
        double first_level = 0.0, first_mass = 0.0, first_external_supply = 0.0;
        const int attempts = interval == 5 ? 2 : 1;
        for (int attempt = 0; attempt < attempts; ++attempt)
        {
            SCOPED_TRACE(attempt);
            solver_coupling.restore_coupling_checkpoint(checkpoint);
            EXPECT_DOUBLE_EQ(solver.time(), start);
            solver_coupling.set_coupling_interval_energy(energy, duration);
            for (int substep = 0; substep < substeps; ++substep)
            {
                const double endpoint = substep + 1 == substeps ? end : start + duration * (substep + 1) / substeps;
                solver.set_time_step(endpoint - solver.time());
                ASSERT_NO_THROW(solver_coupling.step());
            }
            const double mass = solver_coupling.liquid_mass_inventory().totalMass();
            const double produced = (interval + 1) * interval_energy * external_mol_per_joule;
            EXPECT_NEAR(mass, initial_mass, mass_tolerance);
            EXPECT_DOUBLE_EQ(solver.time(), end);
            EXPECT_NEAR(SurfaceGas::supplied(gas), produced, 1.e-18);
            EXPECT_NEAR(gas.global_species_moles(SurfaceGas::gas_id) + SurfaceGas::escaped(gas), produced, 1.e-17);
            EXPECT_EQ(
                solver_coupling.free_surface_history().size(), initial_history_size + accepted_substeps + substeps);
            EXPECT_NEAR(top_elevation(*mesh), solver_coupling.free_surface_diagnostics().pool_level,
                surface.ale.level_absolute_tolerance + surface.ale.level_relative_tolerance);
            if (interval == 5)
            {
                for (size_t owned = 0; owned < energy.size(); ++owned)
                {
                    const auto temperature = solver.temperature().value(static_cast<Pack::local_ordinal_type>(owned));
                    if (attempt == 0)
                        first_temperatures.push_back(temperature);
                    else
                        EXPECT_NEAR(temperature, first_temperatures[owned], 1.e-11);
                }
                if (attempt == 0)
                {
                    first_level = top_elevation(*mesh);
                    first_mass = mass;
                    first_external_supply = SurfaceGas::supplied(gas);
                }
                else
                {
                    EXPECT_NEAR(top_elevation(*mesh), first_level, 1.e-13);
                    EXPECT_NEAR(mass, first_mass, mass_tolerance);
                    EXPECT_NEAR(SurfaceGas::supplied(gas), first_external_supply, 1.e-18);
                }
            }
        }
        solver_coupling.accept_coupling_checkpoint(checkpoint);
        accepted_substeps += static_cast<size_t>(substeps);
    }
}

TEST(BoussinesqCouplingIntervalTest, RejectsForeignCheckpointAndPreservesTrialState)
{
    auto first = make_case(Coupling::PISO, 0.0, 25, true);
    auto second = make_case(Coupling::PISO, 0.0, 25, true);
    auto checkpoint = first.coupling->create_coupling_checkpoint();
    EXPECT_THROW(second.coupling->restore_coupling_checkpoint(checkpoint), std::invalid_argument);
    EXPECT_DOUBLE_EQ(second.solver->time(), 0.0);
    first.coupling->accept_coupling_checkpoint(checkpoint);
}


TEST(BoussinesqCouplingIntervalTest, RestoresAnnularGeometryAfterAcceptedSubcycles)
{
    constexpr double area = 0.75 * std::numbers::pi;
    auto state = configure_matrix_case({MatrixMeshFamily::Cylindrical, SimpleFluid::Dimension::Z, "zmax", area}, 0.0);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    const auto original_geometry = capture_geometry(*state.mesh);
    const auto original_map = state.mesh->owned_cell_map();
    solver.reset_solver_phase_timings();
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    std::vector<double> energy(state.mesh->num_owned_cells());
    for (size_t owned = 0; owned < energy.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        energy[owned] = state.mesh->cell_volume(cell) * 1.e-6 *
            (state.mesh->cell_centroid(cell).z < 0.5 ? 1.0 : 2.0);
    }
    solver_coupling.set_coupling_interval_energy(energy, 0.02);
    solver_coupling.step();
    solver_coupling.step();
    const auto epoch = state.mesh->geometry_epoch();
    EXPECT_GT(boundary_elevation(*state.mesh, state.definition), 1.0);
    const auto attempt_timings = solver.solver_phase_timings();
    const auto gates = solver_coupling.ale_convergence_statistics();
    EXPECT_EQ(gates.checks,solver.solver_phase_timing(SimpleFluid::SolverPhase::ALETrial).calls);
    EXPECT_EQ(gates.converged_checks,2u);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 2u);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::ALE).calls, 2u);
    EXPECT_GE(solver.solver_phase_timing(SimpleFluid::SolverPhase::ALETrial).calls, 2u);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    const auto restored_timings = solver.solver_phase_timings();
    EXPECT_EQ(solver_coupling.ale_convergence_statistics().checks,gates.checks);
    EXPECT_EQ(solver_coupling.ale_convergence_statistics().blocking_combinations,gates.blocking_combinations);
    for (std::size_t phase = 0; phase < attempt_timings.size(); ++phase)
    {
        SCOPED_TRACE(attempt_timings[phase].name);
        if (phase == static_cast<std::size_t>(SimpleFluid::SolverPhase::GeometryRefresh))
        {
            // Restoring geometry adds real work to cumulative diagnostics.
            EXPECT_EQ(restored_timings[phase].calls, attempt_timings[phase].calls + 1u);
            EXPECT_GE(restored_timings[phase].seconds, attempt_timings[phase].seconds);
        }
        else
        {
            EXPECT_EQ(restored_timings[phase].calls, attempt_timings[phase].calls);
            EXPECT_EQ(restored_timings[phase].seconds, attempt_timings[phase].seconds);
        }
    }
    EXPECT_GT(state.mesh->geometry_epoch(), epoch);
    EXPECT_EQ(state.mesh->owned_cell_map().getRawPtr(), original_map.getRawPtr());
    expect_geometry_restored(*state.mesh, original_geometry);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    solver_coupling.accept_coupling_checkpoint(checkpoint);
}

TEST(BoussinesqCouplingIntervalTest, ReplayedBubbleEscapeCommitsTheVentLedgerOnlyOnce)
{
    auto state = make_case(Coupling::PISO, 0.0, 20, true, 100.0);
    auto& solver = *state.solver;
    auto& solver_coupling = *state.coupling;
    auto* gas = solver_coupling.find_reacting_gas_model();
    solver_coupling.find_prescribed_heat_source()->initialize_constant(1.e3);
    solver_coupling.step(); // Create populations that can escape during the coupled interval.
    const auto history_size = solver_coupling.free_surface_history().size();
    const auto escape_before = SurfaceGas::escaped(*gas);
    solver.reset_solver_phase_timings();
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    std::vector<double> no_energy(state.mesh->num_owned_cells(), 0.0);
    solver_coupling.set_coupling_interval_energy(no_energy, 0.02);
    solver_coupling.step();
    solver_coupling.step();
    const auto escaped = SurfaceGas::escaped(*gas);
    const auto vented = solver_coupling.free_surface_diagnostics().vented_gas_moles.at(SurfaceGas::gas_name);
    ASSERT_GT(escaped, escape_before);
    const auto first_step_seconds = solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).seconds;
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 2u);
    const auto gas_attempts = solver.solver_phase_timing(SimpleFluid::SolverPhase::Gas).calls;
    const auto gate_attempts = solver_coupling.ale_convergence_statistics().checks;
    EXPECT_GT(gas_attempts, 0u);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 2u);
    EXPECT_EQ(solver_coupling.ale_convergence_statistics().checks, gate_attempts);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Gas).calls, gas_attempts);
    EXPECT_DOUBLE_EQ(SurfaceGas::escaped(*gas), escape_before);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), history_size);
    solver_coupling.set_coupling_interval_energy(no_energy, 0.02);
    solver_coupling.step();
    solver_coupling.step();
    solver_coupling.accept_coupling_checkpoint(checkpoint);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 4u);
    EXPECT_GT(solver_coupling.ale_convergence_statistics().checks, gate_attempts);
    EXPECT_GT(solver.solver_phase_timing(SimpleFluid::SolverPhase::Gas).calls, gas_attempts);
    EXPECT_GE(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).seconds, first_step_seconds);
    EXPECT_NEAR(SurfaceGas::escaped(*gas), escaped, 1.e-14);
    EXPECT_NEAR(solver_coupling.free_surface_diagnostics().vented_gas_moles.at(SurfaceGas::gas_name), vented, 1.e-14);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), history_size + 2);
}

TEST(BoussinesqCouplingIntervalTest, CheckpointInitializesLazyMaterialAndGasDependencies)
{
    auto mesh = make_column();
    auto solver_shared = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_material_feedback(material_feedback_options());
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    auto& gas = solver_coupling.configure_reacting_gas(gas_options());
    // Accepted catalog-keyed source/escape histories own replay, not a donor-deficit proxy.
    // Supported direct field initialization without initialize_linear_temperature().
    solver.temperature().put_scalar(300.0);
    ASSERT_NE(solver_coupling.configure_free_surface(free_surface_options(10)), nullptr);
    EXPECT_FALSE(gas.initial_state_initialized());
    auto checkpoint = solver_coupling.create_coupling_checkpoint();
    EXPECT_TRUE(gas.initial_state_initialized());
    EXPECT_TRUE(solver_coupling.find_free_surface_model()->initialized());
    EXPECT_NEAR(SurfaceGas::compartment(gas, SurfaceGas::Phase::Dissolved), 1.0, 1.e-14);
    solver_coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_TRUE(solver_coupling.planar_ale_enabled());
    solver_coupling.accept_coupling_checkpoint(checkpoint);
}

namespace TypedSpeciesSurface
{
namespace M = SimpleFluid::Testing::ReactingMixture;
namespace R = SimpleFluid::Reacting;
using Model = SimpleFluid::MultiphysicsSolverCoupler<Pack>::reacting_gas_model_type;
double global_mass(const Model& model)
{
    double local = 0, global = 0;
    const auto& field = model.liquid_mass_density();
    for (std::size_t c = 0; c < field.mesh().num_owned_cells(); ++c)
        local += field.value(static_cast<Pack::local_ordinal_type>(c)) *
                 field.mesh().cell_volume(static_cast<Pack::local_ordinal_type>(c));
    Teuchos::reduceAll(*field.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}
ConfiguredCase make(double heat_rate = .4, bool bubbles = false, bool fail_expanded_volume = false)
{
    Handle::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    auto geometry = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0., 1.}, {0., 1.}, {0., .5, 1.}}});
    auto grid = std::make_shared<Handle>(std::move(geometry), distribution);
    // Resolve the small bubble displacement against the much larger finite liquid inventory.
    // Physical convergence/conservation tolerances retain the common fixture's original values.
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1e-13;
    linear.preconditioner = SimpleFluid::LinearPreconditioner::None;
    auto solver = std::make_shared<Solver>(grid, planar_boundaries(), time_options(Coupling::PISO), linear,
        physical_options());
    auto coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    coupling->configure_material_feedback(material_feedback_options());
    M::Mixture mixture(false, false, R::EnergyConvention::SensibleEnergy, 0);
    mixture.liquid = R::make_linear_compressible_mixture_liquid(
        mixture.catalog, M::bounds, [](std::span<const double>)
        {
            // Valid caloric derivative over the EOS bounds; the Boussinesq temperature owner still uses cp=2.
            return R::LinearCompressibleMixtureLiquid{300, 101325, 10, 100000, 1e-3, 1e-9, 0, .01, 0};
        },
        R::EnergyConvention::SensibleEnergy);
    M::InitialMixture seed;
    seed.product_concentration = .2;
    if (bubbles)
        seed.populations = {{"micro", 1e4, 1e-9, .25}, {"large", 2e3, 2e-9, .6}};
    auto settings = M::field_options(mixture, seed);
    settings.reference_pressure = 101325;
    settings.species_properties = {mixture.transport(M::gas_one, 0, 1e-5), mixture.transport(M::gas_two, 0, 2e-5)};
    settings.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    settings.constant_slip_velocity = bubbles ? .02 : 0;
    settings.free_surface_patches = {"zmax"};
    settings.initial_species_state = [mixture, seed](const R::ReactionEnvironment& environment, std::int64_t gid)
    {
        auto local = seed;
        local.reactant_concentration = gid == 0 ? .2 : .1;
        return M::initial_cell(mixture, environment, local).species;
    };
    settings.species_source = [heat_rate, fail_expanded_volume](const R::ReactionContext& context, double duration)
    {
        const double reactant = context.state.amount(context.catalog, M::reactant, R::SpeciesPhase::Liquid);
        if (fail_expanded_volume && duration > 1e-3 && context.environment.bulkVolume > .5 * (1 + 1e-8) &&
            reactant / context.environment.bulkVolume > .15)
            throw std::runtime_error("manufactured bottom-cell expanded-volume source failure");
        R::ReactionSourceInput source;
        source.heat.integratedEnergy = heat_rate * duration * context.environment.bulkVolume;
        return source;
    };
    auto& gas = coupling->configure_reacting_gas(settings);
    gas.set_transport_linear_solver_options(linear);
    coupling->initialize_heated_box(300, 300);
    auto options = free_surface_options(60);
    options.initial_liquid_volume.reset();
    options.liquid_mass.initial_liquid_mass = global_mass(gas);
    coupling->configure_free_surface(options);
    return {std::move(grid), std::move(solver), std::move(coupling), nullptr};
}
} // namespace TypedSpeciesSurface

TEST(BoussinesqSpeciesPlanarALETest, FollowsTransportedLiquidAndExactEOSFacePacketsOnActualEmptyOwners)
{
    auto configured = TypedSpeciesSurface::make();
    auto& coupling = *configured.coupling;
    auto& gas = *coupling.find_reacting_gas_model();
    const double old_mass = coupling.liquid_mass_inventory().totalMass();
    const auto before = gas.species_event_receipt();
    coupling.step();
    EXPECT_TRUE(coupling.liquid_mass_inventory().speciesBound());
    EXPECT_NEAR(coupling.liquid_mass_inventory().totalMass(), old_mass, 2e-12);
    EXPECT_EQ(gas.species_event_receipt().revision, before.revision + 1);
    EXPECT_EQ(gas.liquid_transport_volume_receipt().event, gas.species_event_receipt());
    EXPECT_EQ(gas.liquid_transport_volume_receipt().geometry_epoch, SimpleFluid::mesh_geometry_epoch(*configured.mesh));
    EXPECT_TRUE(gas.liquid_transport_volume_receipt().complete_carrier);
    EXPECT_GT(coupling.free_surface_diagnostics().pool_level, 1.);
    EXPECT_NEAR(coupling.planar_ale_diagnostics().liquid_mass_residual, 0., 1e-12);
    int local_empty = configured.mesh->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*configured.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (configured.mesh->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(empty, 2);
}
TEST(BoussinesqSpeciesPlanarALETest, MixedPopulationEscapeAndIntervalReplayCommitEverySpeciesOnce)
{
    auto configured = TypedSpeciesSurface::make(0, true);
    auto& coupling = *configured.coupling;
    auto& gas = *coupling.find_reacting_gas_model();
    auto checkpoint = coupling.create_coupling_checkpoint();
    const auto accepted_receipt = gas.species_event_receipt();
    const auto* scalar = coupling.find_scalar_void_fraction_model();
    ASSERT_NE(scalar, nullptr);
    std::vector<double> accepted_alpha, accepted_source;
    for (std::size_t owned = 0; owned < configured.mesh->num_owned_cells(); ++owned)
        accepted_alpha.push_back(scalar->alpha_g().value(static_cast<Pack::local_ordinal_type>(owned)));
    coupling.step();
    const auto accepted_surface = coupling.free_surface_diagnostics();
    const auto accepted_events = gas.physical_escape_events();
    const auto accepted_step = gas.species_event_receipt();
    const auto accepted_mass = coupling.liquid_mass_inventory().totalMass();
    double local_change = 0., global_change = 0.;
    for (std::size_t owned = 0; owned < accepted_alpha.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const double change = scalar->alpha_g().value(cell) - accepted_alpha[owned];
        local_change = std::max(local_change, std::abs(change));
        EXPECT_DOUBLE_EQ(scalar->alpha_g().value(cell), gas.alpha_g().value(cell));
        EXPECT_NEAR(scalar->source_alpha_total().value(cell), change / accepted_step.time_step, 1e-13);
        accepted_source.push_back(scalar->source_alpha_total().value(cell));
    }
    Teuchos::reduceAll(
        *configured.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_change, &global_change);
    EXPECT_GT(global_change, 0.); // The full-step assertion must see actual population transport/escape.
    SimpleFluid::GasMolesBySpecies expected;
    for (const auto& event : accepted_events)
        if (event.phase == SimpleFluid::Reacting::SpeciesPhase::Dispersed ||
            event.phase == SimpleFluid::Reacting::SpeciesPhase::Resolved)
            expected[gas.options()
                    .reaction_mechanism->catalog()
                    .species()[gas.options().reaction_mechanism->catalog().index(event.species)]
                    .name] += event.moles;
    for (const auto& name : {"gas_one", "gas_two"})
    {
        EXPECT_GT(expected.at(name), 0);
        EXPECT_DOUBLE_EQ(accepted_surface.vented_gas_moles.at(name), expected.at(name));
    }
    coupling.restore_coupling_checkpoint(checkpoint);
    EXPECT_EQ(gas.species_event_receipt(), accepted_receipt);
    EXPECT_TRUE(coupling.free_surface_diagnostics().vented_gas_moles.empty());
    coupling.step();
    EXPECT_EQ(gas.species_event_receipt().revision, accepted_step.revision);
    EXPECT_DOUBLE_EQ(gas.species_event_receipt().interval_end, accepted_step.interval_end);
    EXPECT_DOUBLE_EQ(gas.species_event_receipt().time_step, accepted_step.time_step);
    EXPECT_GT(gas.species_event_receipt().publication_nonce, accepted_step.publication_nonce);
    EXPECT_EQ(gas.physical_escape_events(), accepted_events);
    EXPECT_EQ(coupling.free_surface_diagnostics().vented_gas_moles, accepted_surface.vented_gas_moles);
    EXPECT_DOUBLE_EQ(coupling.liquid_mass_inventory().totalMass(), accepted_mass);
    for (std::size_t owned = 0; owned < accepted_source.size(); ++owned)
        EXPECT_DOUBLE_EQ(
            scalar->source_alpha_total().value(static_cast<Pack::local_ordinal_type>(owned)), accepted_source[owned]);
    coupling.accept_coupling_checkpoint(checkpoint);
}
TEST(BoussinesqSpeciesPlanarALETest, RankLocalSourceFailureAfterMotionRestoresGeometryReceiptAndRetries)
{
    auto configured = TypedSpeciesSurface::make(.4, false, true);
    auto& coupling = *configured.coupling;
    auto& gas = *coupling.find_reacting_gas_model();
    const auto receipt = gas.species_event_receipt();
    const auto surface = coupling.free_surface_diagnostics();
    const auto history = coupling.free_surface_history().size();
    const auto mass = coupling.liquid_mass_inventory().totalMass();
    const SimpleFluid::Testing::ReactingMixture::SavedReactingState<TypedSpeciesSurface::Model> saved(gas);
    std::vector<double> volumes;
    for (std::size_t c = 0; c < configured.mesh->num_owned_cells(); ++c)
        volumes.push_back(configured.mesh->cell_volume(static_cast<Pack::local_ordinal_type>(c)));
    EXPECT_ANY_THROW(coupling.step());
    saved.expect_restored(gas);
    EXPECT_EQ(gas.species_event_receipt(), receipt);
    EXPECT_DOUBLE_EQ(configured.solver->time(), 0.);
    EXPECT_DOUBLE_EQ(coupling.liquid_mass_inventory().totalMass(), mass);
    EXPECT_DOUBLE_EQ(coupling.free_surface_diagnostics().pool_level, surface.pool_level);
    EXPECT_EQ(coupling.free_surface_history().size(), history);
    for (std::size_t c = 0; c < volumes.size(); ++c)
        EXPECT_DOUBLE_EQ(configured.mesh->cell_volume(static_cast<Pack::local_ordinal_type>(c)), volumes[c]);
    // Remain below the manufactured duration threshold while resolving the
    // accepted volume increment before division by dt on double-only platforms.
    configured.solver->set_time_step(1e-3);
    EXPECT_NO_THROW(coupling.step());
    EXPECT_EQ(gas.species_event_receipt().revision, receipt.revision + 1);
}

} // namespace

namespace
{
class ConstantThermalParticipant final : public SimpleFluid::ThermalStepParticipant<Pack>
{
public:
    struct Saved final : State { double energy{}, trial{}, dt{}, cavity = 300.; int commits{}; std::vector<double> guess; };
    explicit ConstantThermalParticipant(double flux) : d_flux(flux) {}
    std::string configuration_signature() const override { return "constant:" + std::to_string(d_flux); }
    SimpleFluid::SP<const State> capture_state() const override { return std::make_shared<const Saved>(d_state); }
    void restore_state(const State& state) override { d_state = dynamic_cast<const Saved&>(state); }
    void begin_substep(double, double dt) override { d_state.dt = dt; d_state.trial = 0; }
    SimpleFluid::SP<const boundary_cache_type> preview_trial(
        SimpleFluid::SP<const mesh_type> mesh, const field_type& guess) override
    {
        d_state.guess.resize(mesh->num_owned_cells());
        const auto values = guess.owned_read_view();
        for (size_t owned = 0; owned < d_state.guess.size(); ++owned)
            d_state.guess[owned] = values(owned, 0);
        auto flux = std::make_shared<boundary_cache_type>();
        flux->mesh = mesh;
        double local_power = 0;
        for (const auto& [id, batch] : mesh->boundary_batches())
            if (mesh->boundary_batch_name(id) == "xmax")
            {
                flux->value[id] = SimpleFluid::Arr<double>(batch.face_lids.size(), d_flux);
                for (const auto face : batch.face_lids)
                    if (mesh->is_owned_cell(mesh->owner_cell(face)))
                        local_power += mesh->face_area(face) * d_flux;
            }
        double power = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_power, &power);
        d_state.trial = power * d_state.dt;
        d_state.cavity = 310.;
        return flux;
    }
    double residual(const field_type& solved) const override
    {
        double local = 0, global = 0;
        const auto values = solved.owned_read_view();
        for (size_t owned = 0; owned < d_state.guess.size(); ++owned)
            local = std::max(local, std::abs(values(owned, 0) - d_state.guess[owned]) / 1.e-8);
        Teuchos::reduceAll(*solved.mesh().owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
        return global;
    }
    std::optional<double> trial_cavity_temperature() const override { return d_state.cavity; }
    void validate_commit() const override {}
    void commit() noexcept override { d_state.energy += d_state.trial; ++d_state.commits; }
    double energy() const { return d_state.energy; }
    int commits() const { return d_state.commits; }
private:
    double d_flux;
    Saved d_state;
};

ConfiguredCase make_thermal_case(double flux, int correctors, bool external_temperature = false)
{
    auto mesh = make_column();
    auto physical = physical_options();
    physical.thermal_conductivity = 1.0;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.e-13;
    linear.max_iterations = 500;
    auto solver = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO), linear, physical);
    auto solver_coupling = std::make_unique<SimpleFluid::MultiphysicsSolverCoupler<Pack>>(solver);
    solver_coupling->configure_material_feedback(material_feedback_options());
    solver_coupling->initialize_linear_temperature({0., 0., 1.}, 300., 300.);
    solver_coupling->set_thermal_step_participant(std::make_shared<ConstantThermalParticipant>(flux));
    auto surface = free_surface_options(correctors);
    if (external_temperature) surface.headspace.temperature_mode = SimpleFluid::HeadspaceTemperatureMode::External;
    solver_coupling->configure_free_surface(surface);
    return {mesh, std::move(solver), std::move(solver_coupling), nullptr};
}
}

TEST(BoussinesqPlanarALETest, ThermalParticipantUsesMovingAreaAndCheckpointReplay)
{
    auto state = make_thermal_case(1.e-3, 8);
    auto participant =
        std::dynamic_pointer_cast<ConstantThermalParticipant>(state.coupling->thermal_step_participant());
    const auto old_energy = coupling_sensible_energy(state);
    auto checkpoint = state.coupling->create_coupling_checkpoint();
    EXPECT_THROW(state.coupling->set_thermal_step_participant(nullptr), std::logic_error);
    ASSERT_NO_THROW(state.coupling->step());
    const auto accepted_energy = coupling_sensible_energy(state);
    const auto escaped = participant->energy();
    const auto accepted_top = top_elevation(*state.mesh);
    EXPECT_EQ(participant->commits(), 1);
    EXPECT_LT(accepted_top, 1.0);
    EXPECT_NEAR(escaped, 1.e-5 * accepted_top, 1.e-14);
    EXPECT_NEAR(accepted_energy - old_energy + escaped, 0., 2.e-8);
    EXPECT_NEAR(state.coupling->planar_ale_diagnostics().energy_residual, 0., 2.e-8);
    ASSERT_NO_THROW(state.coupling->restore_coupling_checkpoint(checkpoint));
    EXPECT_EQ(participant->commits(), 0);
    EXPECT_DOUBLE_EQ(participant->energy(), 0.);
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_EQ(participant->commits(), 1);
    EXPECT_DOUBLE_EQ(participant->energy(), escaped);
    EXPECT_NEAR(coupling_sensible_energy(state), accepted_energy, 2.e-8);
    EXPECT_NO_THROW(state.coupling->accept_coupling_checkpoint(checkpoint));
}

TEST(BoussinesqPlanarALETest, ThermalParticipantFailedCorrectorRestoresExternalInventory)
{
    auto state = make_thermal_case(1.e-3, 1);
    auto participant =
        std::dynamic_pointer_cast<ConstantThermalParticipant>(state.coupling->thermal_step_participant());
    const auto old_energy = coupling_sensible_energy(state);
    EXPECT_THROW(state.coupling->step(), std::runtime_error);
    EXPECT_EQ(participant->commits(), 0);
    EXPECT_DOUBLE_EQ(participant->energy(), 0.);
    EXPECT_NEAR(coupling_sensible_energy(state), old_energy, 1.e-12);
    EXPECT_DOUBLE_EQ(top_elevation(*state.mesh), 1.);
}

TEST(BoussinesqPlanarALETest, ZeroThermalParticipantPreservesDefaultFields)
{
    auto coupled = make_thermal_case(0., 4);
    auto reference = make_thermal_case(0., 4);
    reference.coupling->set_thermal_step_participant(nullptr);
    ASSERT_NO_THROW(coupled.coupling->step());
    ASSERT_NO_THROW(reference.coupling->step());
    for (size_t owned = 0; owned < coupled.mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(coupled.solver->temperature().value(cell), reference.solver->temperature().value(cell));
        EXPECT_DOUBLE_EQ(coupled.solver->pressure().value(cell), reference.solver->pressure().value(cell));
    }
    EXPECT_DOUBLE_EQ(top_elevation(*coupled.mesh), top_elevation(*reference.mesh));
}

TEST(BoussinesqPlanarALETest, ExternalThermalTemperatureUsesParticipantAndRollsBack)
{
    auto state = make_thermal_case(1.e-3, 8, true);
    auto checkpoint = state.coupling->create_coupling_checkpoint();
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().headspace.temperature, 300.);
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().headspace.temperature, 310.);
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().headspace.pressure, 101325.);
    ASSERT_NO_THROW(state.coupling->restore_coupling_checkpoint(checkpoint));
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().headspace.temperature, 300.);
    ASSERT_NO_THROW(state.coupling->step());
    EXPECT_DOUBLE_EQ(state.coupling->free_surface_diagnostics().headspace.temperature, 310.);
    EXPECT_NO_THROW(state.coupling->accept_coupling_checkpoint(checkpoint));
}

/** The external controller shares the core solver and owns its own optional models. */
TEST(MultiphysicsSolverCouplerTest, ConfiguresExternalModelsAndKeepsSolverInstancesIndependent)
{
    auto mesh = make_column();
    auto solver_shared = std::make_shared<Solver>(mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    auto& coupling = solver_coupling;
    EXPECT_EQ(&coupling.boussinesq_solver(), &solver);
    const auto& observer = std::as_const(solver_coupling);
    EXPECT_EQ(&coupling, &observer);
    EXPECT_FALSE(observer.planar_ale_enabled());
    EXPECT_EQ(observer.find_reacting_gas_model(), nullptr);

    auto& feedback = coupling.configure_material_feedback(material_feedback_options());
    auto& gas = coupling.configure_reacting_gas(gas_options());
    auto& heat = coupling.add_temperature_source("coupler_heat", 0.01);
    coupling.add_prescribed_heat_source().initialize_constant(0.0);
    solver_coupling.initialize_linear_temperature({0.0, 0.0, 1.0}, 300.0, 300.0);
    auto* surface = coupling.configure_free_surface(free_surface_options(25));
    ASSERT_NE(surface, nullptr);
    EXPECT_EQ(solver_coupling.find_material_feedback_model(), &feedback);
    EXPECT_EQ(solver_coupling.find_reacting_gas_model(), &gas);
    EXPECT_EQ(std::as_const(solver_coupling).find_reacting_gas_model(), observer.find_reacting_gas_model());
    EXPECT_EQ(solver.find_temperature_source("coupler_heat"), &heat);
    EXPECT_EQ(solver_coupling.find_free_surface_model(), surface);
    EXPECT_EQ(&solver_coupling.liquid_mass_inventory(), &observer.liquid_mass_inventory());
    EXPECT_EQ(&solver_coupling.planar_ale_diagnostics(), &observer.planar_ale_diagnostics());
    EXPECT_EQ(&solver_coupling.free_surface_history(), &observer.free_surface_history());

    auto other_mesh = make_column();
    auto other_shared = std::make_shared<Solver>(other_mesh, planar_boundaries(), time_options(Coupling::PISO),
        SimpleFluid::LinearSolverOptions{}, physical_options());
    auto& other = *other_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> other_coupling(other_shared);
    EXPECT_EQ(other_coupling.find_reacting_gas_model(), nullptr);
    EXPECT_FALSE(other_coupling.planar_ale_enabled());
    EXPECT_EQ(other.find_temperature_source("coupler_heat"), nullptr);

    EXPECT_TRUE(solver_coupling.remove_reacting_gas_model());
    EXPECT_EQ(observer.find_reacting_gas_model(), nullptr);
    EXPECT_EQ(observer.find_free_surface_model(), nullptr);
    EXPECT_FALSE(observer.planar_ale_enabled());
}

/** Controller lifetime prevents bypass and releases the solver without a reverse call. */
TEST(MultiphysicsSolverCouplerTest, SharedBindingIsExclusiveAndReleasesStandaloneAdvancement)
{
    auto mesh = make_column();
    auto boundaries = planar_boundaries();
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    auto solver_shared = std::make_shared<Solver>(
        mesh, boundaries, time_options(Coupling::PISO), SimpleFluid::LinearSolverOptions{}, physical_options());
    auto& solver = *solver_shared;
    solver.initialize_heated_box(300.0, 300.0);
    {
        SimpleFluid::MultiphysicsSolverCoupler<Pack> coupling(solver_shared);
        EXPECT_THROW(SimpleFluid::MultiphysicsSolverCoupler<Pack> duplicate(solver_shared), std::exception);
        EXPECT_THROW(solver.step(), std::exception);
        EXPECT_EQ(solver.step_index(), 0);
        coupling.step();
        EXPECT_EQ(solver.step_index(), 1);
    }
    EXPECT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 2);
    SimpleFluid::MultiphysicsSolverCoupler<Pack> replacement(solver_shared);
    EXPECT_NO_THROW(replacement.step());
    EXPECT_EQ(solver.step_index(), 3);
}

/** Shared ownership keeps the core alive through stepping and checkpoint replay. */
TEST(MultiphysicsSolverCouplerTest, RetainsSolverAfterCallerReleaseAndReleasesItOnDestruction)
{
    auto state = make_case(Coupling::PISO, 0.01, 25, true);
    const std::weak_ptr<Solver> lifetime = state.solver;
    const auto* identity = state.solver.get();
    auto coupling = std::move(state.coupling);
    state.solver.reset();
    ASSERT_FALSE(lifetime.expired());
    EXPECT_EQ(&coupling->boussinesq_solver(), identity);

    auto checkpoint = coupling->create_coupling_checkpoint();
    ASSERT_NO_THROW(coupling->step());
    EXPECT_EQ(coupling->boussinesq_solver().step_index(), 1);
    ASSERT_NO_THROW(coupling->restore_coupling_checkpoint(checkpoint));
    EXPECT_EQ(coupling->boussinesq_solver().step_index(), 0);
    EXPECT_DOUBLE_EQ(coupling->boussinesq_solver().time(), 0.0);
    ASSERT_NO_THROW(coupling->accept_coupling_checkpoint(checkpoint));

    coupling.reset();
    EXPECT_TRUE(lifetime.expired());
}

/** The controller requires a real shared owner instead of a borrowed solver. */
TEST(MultiphysicsSolverCouplerTest, RejectsNullSolverOwnership)
{
    using Coupler = SimpleFluid::MultiphysicsSolverCoupler<Pack>;
    static_assert(std::is_constructible_v<Coupler, std::shared_ptr<Solver>>);
    static_assert(!std::is_constructible_v<Coupler, Solver&>);
    EXPECT_THROW(Coupler(Coupler::solver_pointer{}), std::invalid_argument);
}

/** Disabled optional models preserve independent core flow/heat advancement. */
TEST(MultiphysicsSolverCouplerTest, DisabledControllerMatchesStandaloneSolver)
{
    auto bare_mesh = make_column();
    auto coupled_mesh = make_column();
    auto time = time_options(Coupling::PISO);
    auto material = physical_options();
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.e-13;
    auto boundaries = planar_boundaries();
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    Solver bare(bare_mesh, boundaries, time, linear, material);
    auto core_shared = std::make_shared<Solver>(coupled_mesh, boundaries, time, linear, material);
    auto& core = *core_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> coupling(core_shared);
    bare.add_temperature_source("heat", 0.1);
    coupling.add_temperature_source("heat", 0.1);
    bare.initialize_heated_box(300.0, 300.0);
    coupling.initialize_heated_box(300.0, 300.0);
    for (int step = 0; step < 3; ++step)
    {
        bare.step();
        coupling.step();
        expect_primary_restored(core, capture_primary(bare));
        EXPECT_DOUBLE_EQ(core.time(), bare.time());
        EXPECT_EQ(core.step_index(), bare.step_index());
    }
}
