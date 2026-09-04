/**
 * @file testALEEquations.cc
 * @brief Equation-layer forwarding tests for mapped ALE transport state.
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "utils/testing_environment.hh"

#include <memory>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorField = SimpleFluid::VectorCellFieldStored<Pack>;
using FaceField = SimpleFluid::ScalarFaceFieldStored<Pack>;
using Motion = SimpleFluid::PlanarALEMeshMotion<Pack>;
using scalar_type = Pack::scalar_type;
using local_ordinal_type = Pack::local_ordinal_type;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

std::shared_ptr<Handle> make_column()
{
    auto geometry = std::make_shared<Handle::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    return std::make_shared<Handle>(std::move(geometry));
}

FaceField mesh_relative_zero_absolute_flux(
    const std::shared_ptr<Handle>& mesh, const SimpleFluid::FVM::ALEControlVolumeState& ale)
{
    FaceField absolute(mesh, 0.0, "absolute_zero_flux");
    FaceField relative(mesh, "relative_flux");
    SimpleFluid::FVM::mesh_relative_face_fluxes(absolute, ale, relative);
    return relative;
}

FaceField zero_relative_flux(const std::shared_ptr<Handle>& mesh, const SimpleFluid::FVM::ALEControlVolumeState& ale)
{
    FaceField absolute(mesh, 0.0, "absolute_mesh_flux");
    for (const auto face_lid : absolute.owned_face_ids())
    {
        absolute.set_owned_value(face_lid, ale.face_mesh_fluxes()[static_cast<size_t>(face_lid)]);
    }
    absolute.sync_ghosts();
    FaceField relative(mesh, "zero_relative_flux");
    SimpleFluid::FVM::mesh_relative_face_fluxes(absolute, ale, relative);
    return relative;
}

auto zero_vector_source()
{
    return [](local_ordinal_type) -> Handle::Vec3 { return {}; };
}

auto zero_scalar_source()
{
    return [](local_ordinal_type) -> scalar_type { return {}; };
}

void expect_vector_value(const VectorField& field, const Handle::Vec3& expected, scalar_type tolerance)
{
    for (size_t owned = 0; owned < field.mesh().num_owned_cells(); ++owned)
    {
        const auto value = field.value(static_cast<local_ordinal_type>(owned));
        for (size_t component = 0; component < 3; ++component)
        {
            EXPECT_NEAR(value.component(component), expected.component(component), tolerance);
        }
    }
}

void expect_vector_systems_exact(const SimpleFluid::FVM::VectorTransportSystem<Pack>& expected,
    const SimpleFluid::FVM::VectorTransportSystem<Pack>& actual)
{
    ASSERT_EQ(expected.matrix->getLocalNumRows(), actual.matrix->getLocalNumRows());
    for (size_t row = 0; row < expected.matrix->getLocalNumRows(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type expected_columns;
        Pack::matrix_type::values_host_view_type expected_values;
        Pack::matrix_type::local_inds_host_view_type actual_columns;
        Pack::matrix_type::values_host_view_type actual_values;
        expected.matrix->getLocalRowView(static_cast<local_ordinal_type>(row), expected_columns, expected_values);
        actual.matrix->getLocalRowView(static_cast<local_ordinal_type>(row), actual_columns, actual_values);
        ASSERT_EQ(expected_columns.extent(0), actual_columns.extent(0));
        for (size_t entry = 0; entry < expected_columns.extent(0); ++entry)
        {
            EXPECT_EQ(expected_columns(entry), actual_columns(entry));
            EXPECT_DOUBLE_EQ(expected_values(entry), actual_values(entry));
        }
    }
    for (size_t component = 0; component < 3; ++component)
    {
        const auto expected_rhs = expected.rhs->getData(component);
        const auto actual_rhs = actual.rhs->getData(component);
        ASSERT_EQ(expected_rhs.size(), actual_rhs.size());
        for (size_t row = 0; row < expected_rhs.size(); ++row)
        {
            EXPECT_DOUBLE_EQ(expected_rhs[row], actual_rhs[row]);
        }
    }
}

} // namespace

TEST(ALEEquationTest, MomentumForwardsALEAndPreservesAConstantField)
{
    auto mesh = make_column();
    Motion motion(mesh);
    constexpr scalar_type time_step = 0.5;
    motion.begin_trial(3.0, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto relative_flux = mesh_relative_zero_absolute_flux(mesh, ale);

    const Handle::Vec3 constant_velocity{1.25, -0.75, 0.5};
    VectorField old_velocity(mesh, constant_velocity, "old_velocity");
    VectorField generic_velocity(mesh, "generic_velocity");
    VectorField physical_velocity(mesh, "physical_velocity");
    ScalarField viscosity(mesh, 0.0, "dynamic_viscosity");
    const auto boundary_cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        std::shared_ptr<const Handle>(mesh), SimpleFluid::BoundaryConditionSet{});
    SimpleFluid::TimeStepperOptions options;
    options.time_step = time_step;
    options.kinematic_viscosity = 0.0;
    options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;

    SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
    equation.advance_velocity(old_velocity, relative_flux, boundary_cache, options, generic_velocity,
        SimpleFluid::LinearSolverOptions{}, &ale);
    equation.advance_velocity_physical(old_velocity, relative_flux, boundary_cache, options, viscosity, 1.0,
        physical_velocity, zero_vector_source(), SimpleFluid::LinearSolverOptions{}, nullptr, &ale);

    expect_vector_value(generic_velocity, constant_velocity, 2.0e-12);
    expect_vector_value(physical_velocity, constant_velocity, 2.0e-12);
    motion.rollback_trial();
}

TEST(ALEEquationTest, PhysicalTemperatureForwardsOldVolumesAndAcceptedOldProperties)
{
    auto mesh = make_column();
    Motion motion(mesh);
    constexpr scalar_type time_step = 1.0;
    motion.begin_trial(3.0, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto relative_flux = zero_relative_flux(mesh, ale);

    ScalarField old_temperature(mesh, "old_temperature");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        old_temperature.set_owned_value(static_cast<local_ordinal_type>(owned), 10.0 + static_cast<scalar_type>(owned));
    }
    old_temperature.sync_ghosts();
    ScalarField temperature(mesh, "temperature");
    ScalarField old_density(mesh, 2.0, "old_density");
    ScalarField old_heat_capacity(mesh, 3.0, "old_heat_capacity");

    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.density = 4.0;
    material_options.reference_density = 4.0;
    material_options.specific_heat_capacity = 5.0;
    material_options.dynamic_viscosity = 0.0;
    material_options.thermal_conductivity = 0.0;
    SimpleFluid::MaterialPropertyFields<Pack, Handle> material(mesh, material_options, time_options);
    SimpleFluid::TemperatureDiffusionEquation<Pack, Handle> equation(mesh, SimpleFluid::BoundaryConditionSet{});

    equation.advance_physical(old_temperature, relative_flux, time_step, material, temperature, zero_scalar_source(),
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, SimpleFluid::LinearSolverOptions{}, nullptr, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic, &ale, &old_density, &old_heat_capacity, nullptr);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto expected = ale.old_cell_volumes()[owned] * 2.0 * 3.0 * old_temperature.value(cell_lid) /
                              (ale.new_cell_volumes()[owned] * 4.0 * 5.0);
        EXPECT_NEAR(temperature.value(cell_lid), expected, 2.0e-12);
    }
    motion.rollback_trial();
}

TEST(ALEEquationTest, PhysicalTemperaturePreservesAConstantFieldUnderALE)
{
    auto mesh = make_column();
    Motion motion(mesh);
    constexpr scalar_type time_step = 0.5;
    motion.begin_trial(3.0, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto relative_flux = mesh_relative_zero_absolute_flux(mesh, ale);

    constexpr scalar_type constant_temperature = 321.0;
    ScalarField old_temperature(mesh, constant_temperature, "old_temperature");
    ScalarField temperature(mesh, "temperature");
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.density = 2.0;
    material_options.reference_density = 2.0;
    material_options.specific_heat_capacity = 3.0;
    material_options.dynamic_viscosity = 0.0;
    material_options.thermal_conductivity = 0.0;
    SimpleFluid::MaterialPropertyFields<Pack, Handle> material(mesh, material_options, time_options);
    SimpleFluid::TemperatureDiffusionEquation<Pack, Handle> equation(mesh, SimpleFluid::BoundaryConditionSet{});

    equation.advance_physical(old_temperature, relative_flux, time_step, material, temperature, zero_scalar_source(),
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, SimpleFluid::LinearSolverOptions{}, nullptr, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic, &ale, nullptr, nullptr, nullptr);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        EXPECT_NEAR(temperature.value(static_cast<local_ordinal_type>(owned)), constant_temperature, 2.0e-12);
    }
    motion.rollback_trial();
}

TEST(ALEEquationTest, ExplicitNullRetainsFixedGridAssemblyAndAdvance)
{
    auto mesh = make_column();
    FaceField zero_flux(mesh, 0.0, "zero_flux");
    const Handle::Vec3 velocity_value{0.5, -1.0, 1.5};
    VectorField old_velocity(mesh, velocity_value, "old_velocity");
    ScalarField viscosity(mesh, 0.0, "dynamic_viscosity");
    const auto boundary_cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        std::shared_ptr<const Handle>(mesh), SimpleFluid::BoundaryConditionSet{});
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.25;
    options.kinematic_viscosity = 0.0;
    options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;

    SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> omitted_momentum(mesh);
    SimpleFluid::IncompressibleMomentumEquation<Pack, Handle> null_momentum(mesh);
    const auto omitted_system = omitted_momentum.assemble_system(old_velocity, zero_flux, boundary_cache, options);
    const auto null_system =
        null_momentum.assemble_system(old_velocity, zero_flux, boundary_cache, options, nullptr, nullptr);
    expect_vector_systems_exact(omitted_system, null_system);

    const auto omitted_physical = omitted_momentum.assemble_physical_system(
        old_velocity, zero_flux, boundary_cache, options, viscosity, 1.0, zero_vector_source());
    const auto null_physical = null_momentum.assemble_physical_system(old_velocity, zero_flux, boundary_cache, options,
        viscosity, 1.0, zero_vector_source(), nullptr, nullptr, nullptr);
    expect_vector_systems_exact(omitted_physical, null_physical);

    constexpr scalar_type initial_temperature = 275.0;
    ScalarField old_temperature(mesh, initial_temperature, "old_temperature");
    ScalarField omitted_temperature(mesh, "omitted_temperature");
    ScalarField null_temperature(mesh, "null_temperature");
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.dynamic_viscosity = 0.0;
    material_options.thermal_conductivity = 0.0;
    SimpleFluid::MaterialPropertyFields<Pack, Handle> material(mesh, material_options, options);
    SimpleFluid::TemperatureDiffusionEquation<Pack, Handle> omitted_temperature_equation(
        mesh, SimpleFluid::BoundaryConditionSet{});
    SimpleFluid::TemperatureDiffusionEquation<Pack, Handle> null_temperature_equation(
        mesh, SimpleFluid::BoundaryConditionSet{});
    omitted_temperature_equation.advance_physical(old_temperature, zero_flux, options.time_step, material,
        omitted_temperature, zero_scalar_source(), SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    null_temperature_equation.advance_physical(old_temperature, zero_flux, options.time_step, material,
        null_temperature, zero_scalar_source(), SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
        SimpleFluid::LinearSolverOptions{}, nullptr, nullptr, SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic,
        nullptr, nullptr, nullptr, nullptr);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(omitted_temperature.value(cell_lid), null_temperature.value(cell_lid));
    }
}
