/**
 * @file testALETransport.cc
 * @brief Conservative old/new-volume and relative-flux ALE operator tests.
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "FVM/TransportSystem.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_Array.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

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

auto dirichlet_condition()
{
    return [](int, size_t)
    { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 0.0}; };
}

auto zero_source()
{
    return [](local_ordinal_type) { return scalar_type{}; };
}

auto zero_vector_source()
{
    return [](local_ordinal_type) { return Handle::Vec3{}; };
}

std::unordered_map<local_ordinal_type, scalar_type> matrix_row(const Pack::matrix_type& matrix, local_ordinal_type row)
{
    typename Pack::matrix_type::local_inds_host_view_type columns;
    typename Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(row, columns, values);
    std::unordered_map<local_ordinal_type, scalar_type> result;
    for (size_t entry = 0; entry < columns.extent(0); ++entry)
    {
        result[columns[entry]] = values[entry];
    }
    return result;
}

void expect_systems_equal(const SimpleFluid::FVM::TransportSystem<Pack>& expected,
    const SimpleFluid::FVM::TransportSystem<Pack>& actual, size_t rows)
{
    ASSERT_TRUE(expected.matrix->getRowMap()->isSameAs(*actual.matrix->getRowMap()));
    const auto expected_rhs = expected.rhs->getData();
    const auto actual_rhs = actual.rhs->getData();
    for (size_t row = 0; row < rows; ++row)
    {
        const auto lid = static_cast<local_ordinal_type>(row);
        EXPECT_EQ(matrix_row(*actual.matrix, lid), matrix_row(*expected.matrix, lid));
        EXPECT_DOUBLE_EQ(actual_rhs[lid], expected_rhs[lid]);
    }
}

void expect_scalar_residual(const SimpleFluid::FVM::TransportSystem<Pack>& system,
    const std::vector<scalar_type>& candidate, scalar_type tolerance)
{
    Pack::vector_type x(system.matrix->getDomainMap(), true);
    for (size_t row = 0; row < candidate.size(); ++row)
    {
        x.replaceLocalValue(static_cast<local_ordinal_type>(row), candidate[row]);
    }
    Pack::vector_type applied(system.matrix->getRangeMap(), true);
    system.matrix->apply(x, applied);
    const auto values = applied.getData();
    const auto rhs = system.rhs->getData();
    for (size_t row = 0; row < candidate.size(); ++row)
    {
        EXPECT_NEAR(values[row], rhs[row], tolerance);
    }
}

void expect_vector_residual(
    const SimpleFluid::FVM::VectorTransportSystem<Pack>& system, const Handle::Vec3& candidate, scalar_type tolerance)
{
    Pack::multi_vector_type x(system.matrix->getDomainMap(), 3, true);
    for (size_t row = 0; row < system.matrix->getLocalNumRows(); ++row)
    {
        const auto lid = static_cast<local_ordinal_type>(row);
        for (size_t component = 0; component < 3; ++component)
        {
            x.replaceLocalValue(lid, component, candidate.component(component));
        }
    }
    Pack::multi_vector_type applied(system.matrix->getRangeMap(), 3, true);
    system.matrix->apply(x, applied);
    for (size_t component = 0; component < 3; ++component)
    {
        const auto values = applied.getData(component);
        const auto rhs = system.rhs->getData(component);
        for (size_t row = 0; row < system.matrix->getLocalNumRows(); ++row)
        {
            EXPECT_NEAR(values[row], rhs[row], tolerance);
        }
    }
}

FaceField absolute_flux_matching_mesh(
    const std::shared_ptr<Handle>& mesh, const SimpleFluid::FVM::ALEControlVolumeState& ale)
{
    FaceField result(mesh, 0.0, "absolute_mesh_flux");
    for (const auto face_lid : result.owned_face_ids())
    {
        result.set_owned_value(face_lid, ale.face_mesh_fluxes()[static_cast<size_t>(face_lid)]);
    }
    result.sync_ghosts();
    return result;
}

void expect_constant_preservation(SimpleFluid::real_t target_elevation)
{
    auto mesh = make_column();
    Motion motion(mesh);
    constexpr scalar_type time_step = 0.5;
    motion.begin_trial(target_elevation, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);

    FaceField absolute_flux(mesh, 0.0, "absolute_flux");
    FaceField relative_flux(mesh, "relative_flux");
    SimpleFluid::FVM::mesh_relative_face_fluxes(absolute_flux, ale, relative_flux);

    constexpr scalar_type scalar_value = 2.75;
    ScalarField old_scalar(mesh, scalar_value, "old_scalar");
    ScalarField unit(mesh, 1.0, "unit");
    ScalarField zero(mesh, 0.0, "zero");
    const auto scalar_system = SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
        SimpleFluid::FVM::MeshWeightedScalarTransportRequest<Pack, Handle>{.old_values = old_scalar,
            .face_fluxes = relative_flux,
            .time_step = time_step,
            .storage_weight = unit,
            .advection_weight = unit,
            .diffusivity = zero,
            .boundary_condition = dirichlet_condition(),
            .boundary_value = [](int, size_t) { return scalar_value; },
            .source = zero_source(),
            .treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
            .ale = &ale});
    expect_scalar_residual(scalar_system, std::vector<scalar_type>(mesh->num_owned_cells(), scalar_value), 2.0e-12);

    const Handle::Vec3 vector_value{1.25, -0.75, 0.5};
    VectorField old_vector(mesh, vector_value, "old_vector");
    const auto vector_system = SimpleFluid::FVM::non_orthogonal_transport_system<Pack>(
        old_vector, relative_flux, time_step, scalar_type{}, [=](int, size_t) { return vector_value; },
        zero_vector_source(), SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr, Teuchos::null,
        SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{}, nullptr, &ale);
    expect_vector_residual(vector_system, vector_value, 2.0e-12);

    ScalarField viscosity(mesh, 0.0, "viscosity");
    const auto momentum_system = SimpleFluid::FVM::physical_momentum_transport_system<Pack>(
        old_vector, relative_flux, time_step, viscosity, scalar_type{1}, [=](int, size_t) { return vector_value; },
        zero_vector_source(), SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr, Teuchos::null,
        SimpleFluid::FVM::detail::AlwaysDiffuseBoundary{}, nullptr, nullptr,
        SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic, &ale);
    expect_vector_residual(momentum_system, vector_value, 2.0e-12);
}

/** Minimal active-motion fixture used to inject invalid swept flux. */
class CorruptMotion final : public SimpleFluid::MeshMotionModel
{
public:
    explicit CorruptMotion(std::shared_ptr<Handle> mesh)
        : d_mesh(std::move(mesh)), d_old(d_mesh->num_local_cells()), d_new(d_mesh->num_local_cells()),
          d_flux(d_mesh->num_faces(), 0.0)
    {
        for (size_t local = 0; local < d_old.size(); ++local)
        {
            const auto lid = static_cast<local_ordinal_type>(local);
            d_old[local] = d_mesh->cell_volume(lid);
            d_new[local] = d_old[local];
        }
        for (size_t face = 0; face < d_mesh->num_faces(); ++face)
        {
            const auto lid = static_cast<local_ordinal_type>(face);
            if (d_mesh->is_boundary_face(lid))
            {
                d_flux[face] = 1.0;
                break;
            }
        }
        d_diagnostics.time_step = 1.0;
        d_diagnostics.old_geometry_epoch = d_mesh->geometry_epoch();
        d_diagnostics.new_geometry_epoch = d_mesh->geometry_epoch();
        d_diagnostics.trial_active = true;
    }

    void begin_trial(SimpleFluid::real_t, SimpleFluid::real_t) override {}
    void accept_trial() override { d_active = false; }
    void rollback_trial() override { d_active = false; }
    bool has_active_trial() const noexcept override { return d_active; }
    std::string_view mesh_family() const noexcept override { return "corrupt"; }
    std::span<const SimpleFluid::real_t> old_cell_volumes() const noexcept override { return d_old; }
    std::span<const SimpleFluid::real_t> new_cell_volumes() const noexcept override { return d_new; }
    std::span<const SimpleFluid::real_t> face_mesh_fluxes() const noexcept override { return d_flux; }
    const SimpleFluid::MeshMotionDiagnostics& diagnostics() const noexcept override { return d_diagnostics; }
    const std::shared_ptr<Handle>& mesh_ptr() const noexcept { return d_mesh; }

private:
    std::shared_ptr<Handle> d_mesh;
    std::vector<SimpleFluid::real_t> d_old;
    std::vector<SimpleFluid::real_t> d_new;
    std::vector<SimpleFluid::real_t> d_flux;
    SimpleFluid::MeshMotionDiagnostics d_diagnostics;
    bool d_active = true;
};

} // namespace

TEST(ALETransportTest, StationaryTrialExactlyMatchesFixedGridWeightedAssembly)
{
    auto mesh = make_column();
    ScalarField old_values(mesh, "old_values");
    ScalarField storage(mesh, "storage");
    ScalarField advection(mesh, "advection");
    ScalarField diffusion(mesh, 0.0, "diffusion");
    FaceField absolute_flux(mesh, 0.0, "absolute_flux");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        old_values.set_owned_value(cell, 1.0 + owned);
        storage.set_owned_value(cell, 2.0 + owned);
        advection.set_owned_value(cell, 0.5 + owned);
    }
    old_values.sync_ghosts();
    storage.sync_ghosts();
    advection.sync_ghosts();

    constexpr scalar_type time_step = 0.25;
    auto assemble = [&](const FaceField& flux, const SimpleFluid::FVM::ALEControlVolumeState* ale)
    {
        return SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            SimpleFluid::FVM::MeshWeightedScalarTransportRequest<Pack, Handle>{.old_values = old_values,
                .face_fluxes = flux,
                .time_step = time_step,
                .storage_weight = storage,
                .advection_weight = advection,
                .diffusivity = diffusion,
                .boundary_condition = dirichlet_condition(),
                .boundary_value = [](int, size_t) { return 0.0; },
                .source = [](local_ordinal_type cell) { return 0.125 * static_cast<scalar_type>(cell + 1); },
                .treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
                .ale = ale});
    };
    const auto fixed = assemble(absolute_flux, nullptr);

    Motion motion(mesh);
    motion.begin_trial(2.0, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    FaceField relative_flux(mesh, "relative_flux");
    SimpleFluid::FVM::mesh_relative_face_fluxes(absolute_flux, ale, relative_flux);
    const auto stationary_ale = assemble(relative_flux, &ale);

    expect_systems_equal(fixed, stationary_ale, mesh->num_owned_cells());
}

TEST(ALETransportTest, GclPreservesUniformScalarVectorAndMomentumDuringExpansionAndContraction)
{
    expect_constant_preservation(3.0);
    expect_constant_preservation(1.5);
}

TEST(ALETransportTest, DistinctOldScalarAndTemperatureCapacityConserveStoredInventory)
{
    auto mesh = make_column();
    Motion motion(mesh);
    constexpr scalar_type time_step = 1.0;
    motion.begin_trial(3.0, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto absolute_flux = absolute_flux_matching_mesh(mesh, ale);
    FaceField relative_flux(mesh, "relative_flux");
    SimpleFluid::FVM::mesh_relative_face_fluxes(absolute_flux, ale, relative_flux);

    ScalarField old_temperature(mesh, "old_temperature");
    ScalarField scalar_storage(mesh, 4.0, "new_scalar_storage");
    ScalarField old_scalar_storage(mesh, 2.0, "old_scalar_storage");
    ScalarField unit(mesh, 1.0, "unit");
    ScalarField density(mesh, 4.0, "new_density");
    ScalarField heat_capacity(mesh, 5.0, "new_heat_capacity");
    ScalarField old_density(mesh, 2.0, "old_density");
    ScalarField old_heat_capacity(mesh, 3.0, "old_heat_capacity");
    ScalarField conductivity(mesh, 0.0, "conductivity");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        old_temperature.set_owned_value(static_cast<local_ordinal_type>(owned), 10.0 + owned);
    }
    old_temperature.sync_ghosts();

    const auto scalar_system = SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
        SimpleFluid::FVM::MeshWeightedScalarTransportRequest<Pack, Handle>{.old_values = old_temperature,
            .face_fluxes = relative_flux,
            .time_step = time_step,
            .storage_weight = scalar_storage,
            .advection_weight = unit,
            .diffusivity = conductivity,
            .boundary_condition = dirichlet_condition(),
            .boundary_value = [](int, size_t) { return 0.0; },
            .source = zero_source(),
            .treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
            .old_storage_weight = &old_scalar_storage,
            .ale = &ale});
    std::vector<scalar_type> expected_scalar(mesh->num_owned_cells());
    for (size_t owned = 0; owned < expected_scalar.size(); ++owned)
    {
        expected_scalar[owned] = ale.old_cell_volumes()[owned] * 2.0 *
                                 old_temperature.value(static_cast<local_ordinal_type>(owned)) /
                                 (ale.new_cell_volumes()[owned] * 4.0);
    }
    expect_scalar_residual(scalar_system, expected_scalar, 2.0e-12);

    const auto system = SimpleFluid::FVM::physical_temperature_transport_system<Pack>(
        old_temperature, relative_flux, time_step, density, heat_capacity, conductivity, dirichlet_condition(),
        [](int, size_t) { return 0.0; }, zero_source(), SimpleFluid::FVM::NonOrthogonalTreatment::Explicit, nullptr,
        Teuchos::null, nullptr, nullptr, SimpleFluid::FVM::FaceCoefficientInterpolation::Harmonic, &ale, &old_density,
        &old_heat_capacity);

    std::vector<scalar_type> expected(mesh->num_owned_cells());
    scalar_type old_energy{};
    scalar_type new_energy{};
    for (size_t owned = 0; owned < expected.size(); ++owned)
    {
        expected[owned] = ale.old_cell_volumes()[owned] * 2.0 * 3.0 *
                          old_temperature.value(static_cast<local_ordinal_type>(owned)) /
                          (ale.new_cell_volumes()[owned] * 4.0 * 5.0);
        old_energy +=
            ale.old_cell_volumes()[owned] * 2.0 * 3.0 * old_temperature.value(static_cast<local_ordinal_type>(owned));
        new_energy += ale.new_cell_volumes()[owned] * 4.0 * 5.0 * expected[owned];
    }
    expect_scalar_residual(system, expected, 2.0e-12);
    EXPECT_NEAR(new_energy, old_energy, 2.0e-12);
}

TEST(ALETransportTest, ClosedRelativeFluxConservesInventoryAndRoundTripsGeometry)
{
    auto mesh = make_column();
    Motion motion(mesh);
    constexpr scalar_type time_step = 1.0;
    ScalarField unit(mesh, 1.0, "unit");
    ScalarField zero(mesh, 0.0, "zero");
    ScalarField initial(mesh, "initial");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        initial.set_owned_value(cell, mesh->cell_global_id(cell) == 0 ? 2.0 : 5.0);
    }
    initial.sync_ghosts();

    auto assemble =
        [&](const ScalarField& old_values, const FaceField& flux, const SimpleFluid::FVM::ALEControlVolumeState& ale)
    {
        return SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            SimpleFluid::FVM::MeshWeightedScalarTransportRequest<Pack, Handle>{.old_values = old_values,
                .face_fluxes = flux,
                .time_step = time_step,
                .storage_weight = unit,
                .advection_weight = unit,
                .diffusivity = zero,
                .boundary_condition = dirichlet_condition(),
                .boundary_value = [](int, size_t) { return 0.0; },
                .source = zero_source(),
                .treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
                .ale = &ale});
    };

    motion.begin_trial(3.0, time_step);
    const auto expanded_ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto expanded_absolute = absolute_flux_matching_mesh(mesh, expanded_ale);
    FaceField expanded_relative(mesh, "expanded_relative");
    SimpleFluid::FVM::mesh_relative_face_fluxes(expanded_absolute, expanded_ale, expanded_relative);
    std::vector<scalar_type> expanded(mesh->num_owned_cells());
    scalar_type initial_inventory{};
    scalar_type expanded_inventory{};
    for (size_t owned = 0; owned < expanded.size(); ++owned)
    {
        const auto value = initial.value(static_cast<local_ordinal_type>(owned));
        expanded[owned] = expanded_ale.old_cell_volumes()[owned] * value / expanded_ale.new_cell_volumes()[owned];
        initial_inventory += expanded_ale.old_cell_volumes()[owned] * value;
        expanded_inventory += expanded_ale.new_cell_volumes()[owned] * expanded[owned];
    }
    expect_scalar_residual(assemble(initial, expanded_relative, expanded_ale), expanded, 1.0e-12);
    EXPECT_NEAR(expanded_inventory, initial_inventory, 1.0e-12);
    motion.accept_trial();

    ScalarField expanded_field(mesh, "expanded_field");
    for (size_t owned = 0; owned < expanded.size(); ++owned)
    {
        expanded_field.set_owned_value(static_cast<local_ordinal_type>(owned), expanded[owned]);
    }
    expanded_field.sync_ghosts();
    motion.begin_trial(2.0, time_step);
    const auto contracted_ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto contracted_absolute = absolute_flux_matching_mesh(mesh, contracted_ale);
    FaceField contracted_relative(mesh, "contracted_relative");
    SimpleFluid::FVM::mesh_relative_face_fluxes(contracted_absolute, contracted_ale, contracted_relative);
    std::vector<scalar_type> recovered(mesh->num_owned_cells());
    for (size_t owned = 0; owned < recovered.size(); ++owned)
    {
        recovered[owned] =
            contracted_ale.old_cell_volumes()[owned] * expanded[owned] / contracted_ale.new_cell_volumes()[owned];
        EXPECT_NEAR(recovered[owned], initial.value(static_cast<local_ordinal_type>(owned)), 1.0e-14);
    }
    expect_scalar_residual(assemble(expanded_field, contracted_relative, contracted_ale), recovered, 1.0e-12);
}

TEST(ALETransportTest, RejectsWrongGclFluxMismatchedTimestepAndBdf2)
{
    auto mesh = make_column();
    CorruptMotion corrupt(mesh);
    EXPECT_THROW(SimpleFluid::FVM::make_ale_control_volume_state(*mesh, corrupt), std::invalid_argument);

    Motion motion(mesh);
    motion.begin_trial(3.0, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_THROW(ale.validate(*mesh, 0.5), std::invalid_argument);

    FaceField absolute_flux(mesh, 0.0, "absolute_flux");
    EXPECT_THROW(SimpleFluid::FVM::mesh_relative_face_fluxes(absolute_flux, ale, absolute_flux), std::invalid_argument);

    ScalarField old_values(mesh, 1.0, "old_values");
    ScalarField older_values(mesh, 1.0, "older_values");
    ScalarField unit(mesh, 1.0, "unit");
    ScalarField zero(mesh, 0.0, "zero");
    FaceField relative_flux(mesh, 0.0, "relative_flux");
    EXPECT_THROW(SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
                     SimpleFluid::FVM::MeshWeightedScalarTransportRequest<Pack, Handle>{.old_values = old_values,
                         .face_fluxes = relative_flux,
                         .time_step = 1.0,
                         .storage_weight = unit,
                         .advection_weight = unit,
                         .diffusivity = zero,
                         .boundary_condition = dirichlet_condition(),
                         .boundary_value = [](int, size_t) { return 1.0; },
                         .source = zero_source(),
                         .treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
                         .discretization = {SimpleFluid::FVM::ScalarTimeScheme::BDF2,
                             SimpleFluid::FVM::ScalarConvectionScheme::Upwind},
                         .older_values = &older_values,
                         .ale = &ale}),
        std::invalid_argument);
}
