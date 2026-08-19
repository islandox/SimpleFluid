/**
 * @file testScalarTransportDiscretization.cc
 * @brief Accuracy, boundedness, and backend-parity tests for scalar schemes.
 */

#include <gtest/gtest.h>

#include "FVM/TransportSystem.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using LegacyMesh = SimpleFluid::Mesh<Pack>;
using MappedMesh = SimpleFluid::MeshHandle<Pack>;
using LegacyScalar = SimpleFluid::CellField<Pack>;
using LegacyFlux = SimpleFluid::FaceField<Pack>;
using MappedScalar = SimpleFluid::ScalarCellFieldStored<Pack, MappedMesh>;
using MappedFlux = SimpleFluid::ScalarFaceFieldStored<Pack, MappedMesh>;
using TransportSystem = SimpleFluid::FVM::TransportSystem<Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<LegacyMesh> make_legacy_line()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(3, 1, 1));
}

SimpleFluid::SP<const MappedMesh> make_mapped_line()
{
    return std::make_shared<MappedMesh>(SimpleFluid::test::make_unstructured_hex_line(3));
}

SimpleFluid::SP<const MappedMesh> make_distributed_mapped_line()
{
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<MappedMesh>(std::move(cartesian));
}

Pack::scalar_type matrix_entry(
    const Pack::matrix_type& matrix, Pack::local_ordinal_type row, Pack::local_ordinal_type column)
{
    const auto row_entries = matrix.getNumEntriesInLocalRow(row);
    Pack::matrix_type::nonconst_local_inds_host_view_type columns("scalar_scheme_columns", row_entries);
    Pack::matrix_type::nonconst_values_host_view_type values("scalar_scheme_values", row_entries);
    size_t num_entries = 0;
    matrix.getLocalRowCopy(row, columns, values, num_entries);

    auto result = Pack::scalar_type{};
    for (size_t entry = 0; entry < num_entries; ++entry)
    {
        if (columns(entry) == column)
        {
            result += values(entry);
        }
    }
    return result;
}

void expect_systems_equal(const TransportSystem& actual, const TransportSystem& expected, size_t cells)
{
    ASSERT_EQ(actual.matrix->getLocalNumRows(), cells);
    ASSERT_EQ(expected.matrix->getLocalNumRows(), cells);
    for (size_t row = 0; row < cells; ++row)
    {
        const auto row_lid = static_cast<Pack::local_ordinal_type>(row);
        for (size_t column = 0; column < cells; ++column)
        {
            const auto column_lid = static_cast<Pack::local_ordinal_type>(column);
            EXPECT_NEAR(matrix_entry(*actual.matrix, row_lid, column_lid),
                matrix_entry(*expected.matrix, row_lid, column_lid), 1.0e-12);
        }
        EXPECT_NEAR(actual.rhs->getData()[row], expected.rhs->getData()[row], 1.0e-12);
    }
}

template<class Field> void set_cell_values(Field& field, const std::array<double, 3>& values)
{
    for (size_t cell = 0; cell < values.size(); ++cell)
    {
        field.set_value(static_cast<Pack::local_ordinal_type>(cell), values[cell]);
    }
    field.sync_ghosts();
}

template<class MeshType, class FaceField> void set_positive_x_flux(const MeshType& mesh, FaceField& flux)
{
    for (const auto face_lid : flux.owned_face_ids())
    {
        flux.set_value(face_lid, 0.2 * mesh.face_area_vector(face_lid).x);
    }
}

auto dirichlet_condition()
{
    return [](int, size_t)
    { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 0.0}; };
}

auto zero_boundary_value()
{
    return [](int, size_t) { return 0.0; };
}

auto cell_source()
{
    return [](Pack::local_ordinal_type cell_lid) { return 0.05 * static_cast<double>(cell_lid + 1); };
}

TEST(ScalarTransportDiscretizationTest, Bdf2IsQuadraticExactAndLinearUpwindIsLocallyBounded)
{
    using namespace SimpleFluid::FVM;

    const auto backward_euler = detail::scalar_transient_coefficients<double>(ScalarTimeScheme::BackwardEuler);
    EXPECT_DOUBLE_EQ(backward_euler.diagonal, 1.0);
    EXPECT_DOUBLE_EQ(backward_euler.previous, 1.0);
    EXPECT_DOUBLE_EQ(backward_euler.older, 0.0);

    const auto bdf2 = detail::scalar_transient_coefficients<double>(ScalarTimeScheme::BDF2);
    EXPECT_DOUBLE_EQ(bdf2.diagonal, 1.5);
    EXPECT_DOUBLE_EQ(bdf2.previous, 2.0);
    EXPECT_DOUBLE_EQ(bdf2.older, -0.5);

    // For phi(t)=t^2 at t={0,1,2}, constant-step BDF2 gives phi'(2)=4.
    EXPECT_DOUBLE_EQ(bdf2.diagonal * 4.0 - (bdf2.previous * 1.0 + bdf2.older * 0.0), 4.0);

    const SimpleFluid::vec3<double> displacement{0.5, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(
        detail::bounded_linear_upwind_face_value(1.0, 3.0, SimpleFluid::vec3<double>{2.0, 0.0, 0.0}, displacement),
        2.0);
    EXPECT_DOUBLE_EQ(
        detail::bounded_linear_upwind_face_value(1.0, 3.0, SimpleFluid::vec3<double>{20.0, 0.0, 0.0}, displacement),
        3.0);
    EXPECT_DOUBLE_EQ(
        detail::bounded_linear_upwind_face_value(1.0, 3.0, SimpleFluid::vec3<double>{-20.0, 0.0, 0.0}, displacement),
        1.0);

    const auto owner_correction = detail::deferred_convection_rhs_correction(2.0, 3.0, 1.0, 1.5);
    const auto neighbor_correction = detail::deferred_convection_rhs_correction(-2.0, 3.0, 1.0, 1.5);
    EXPECT_DOUBLE_EQ(owner_correction, -3.0);
    EXPECT_DOUBLE_EQ(neighbor_correction, 3.0);
    EXPECT_DOUBLE_EQ(owner_correction + neighbor_correction, 0.0);
}

TEST(ScalarTransportDiscretizationTest, HistoricalOverloadsForwardToBackwardEulerUpwind)
{
    using namespace SimpleFluid::FVM;

    auto legacy_mesh = make_legacy_line();
    LegacyScalar old(legacy_mesh, 1.0, "old");
    LegacyScalar storage(legacy_mesh, 2.0, "storage");
    LegacyScalar advection(legacy_mesh, 2.0, "advection");
    LegacyScalar diffusion(legacy_mesh, 0.25, "diffusion");
    LegacyFlux flux(legacy_mesh, 0.0, "flux");
    set_positive_x_flux(*legacy_mesh, flux);

    const auto historical = weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion,
        dirichlet_condition(), zero_boundary_value(), cell_source(), NonOrthogonalTreatment::Explicit);
    const auto explicit_default =
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), cell_source(), ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit);
    expect_systems_equal(explicit_default, historical, 3);

    auto mapped_mesh = make_mapped_line();
    MappedScalar mapped_old(mapped_mesh, 1.0, "old");
    MappedScalar mapped_storage(mapped_mesh, 2.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 2.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.25, "diffusion");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    set_positive_x_flux(*mapped_mesh, mapped_flux);

    const auto mapped_historical = weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25, mapped_storage,
        mapped_advection, mapped_diffusion, dirichlet_condition(), zero_boundary_value(), cell_source(),
        NonOrthogonalTreatment::Explicit);
    const auto mapped_explicit_default = weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25,
        mapped_storage, mapped_advection, mapped_diffusion, dirichlet_condition(), zero_boundary_value(), cell_source(),
        ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit);
    expect_systems_equal(mapped_explicit_default, mapped_historical, 3);
}

TEST(ScalarTransportDiscretizationTest, Bdf2RequiresACompatibleOlderFieldForLegacyAndMappedStorage)
{
    using namespace SimpleFluid::FVM;

    const ScalarTransportDiscretization bdf2{ScalarTimeScheme::BDF2, ScalarConvectionScheme::Upwind};

    auto legacy_mesh = make_legacy_line();
    LegacyScalar old(legacy_mesh, 1.0, "old");
    LegacyScalar storage(legacy_mesh, 2.0, "storage");
    LegacyScalar advection(legacy_mesh, 2.0, "advection");
    LegacyScalar diffusion(legacy_mesh, 0.25, "diffusion");
    LegacyFlux flux(legacy_mesh, 0.0, "flux");

    EXPECT_THROW(
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), cell_source(), bdf2, NonOrthogonalTreatment::Explicit),
        std::invalid_argument);

    auto other_legacy_mesh = make_legacy_line();
    LegacyScalar incompatible_older(other_legacy_mesh, 0.5, "incompatible_older");
    EXPECT_THROW(
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), cell_source(), bdf2, NonOrthogonalTreatment::Explicit, &incompatible_older),
        std::invalid_argument);

    auto mapped_mesh = make_mapped_line();
    MappedScalar mapped_old(mapped_mesh, 1.0, "old");
    MappedScalar mapped_storage(mapped_mesh, 2.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 2.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.25, "diffusion");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25, mapped_storage, mapped_advection,
                     mapped_diffusion, dirichlet_condition(), zero_boundary_value(), cell_source(), bdf2,
                     NonOrthogonalTreatment::Explicit),
        std::invalid_argument);
}

TEST(ScalarTransportDiscretizationTest, AdvancedAssemblersRejectRankDivergentPreflightInputsCollectively)
{
    using namespace SimpleFluid::FVM;

    const auto legacy_mesh = make_legacy_line();
    const auto communicator = legacy_mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "Requires at least two MPI ranks.";
    }

    LegacyScalar old(legacy_mesh, 1.0, "old");
    LegacyScalar older(legacy_mesh, 0.5, "older");
    LegacyScalar storage(legacy_mesh, 2.0, "storage");
    LegacyScalar advection(legacy_mesh, 2.0, "advection");
    LegacyScalar diffusion(legacy_mesh, 0.25, "diffusion");
    LegacyFlux flux(legacy_mesh, 0.0, "flux");
    const ScalarCellValueProvider<Pack> finite_source = [](Pack::local_ordinal_type) { return 0.0; };

    const ScalarTransportDiscretization rank_dependent_policy{
        communicator->getRank() == 0 ? ScalarTimeScheme::BackwardEuler : ScalarTimeScheme::BDF2,
        ScalarConvectionScheme::Upwind};
    EXPECT_THROW(
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), finite_source, rank_dependent_policy, NonOrthogonalTreatment::Explicit, &older),
        std::invalid_argument);

    const auto rank_dependent_time_step = communicator->getRank() == 0 ? -0.25 : 0.25;
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(old, flux, rank_dependent_time_step, storage, advection,
                     diffusion, dirichlet_condition(), zero_boundary_value(), finite_source,
                     ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit),
        std::invalid_argument);

    if (communicator->getRank() == 0 && storage.num_owned_cells() != 0)
    {
        storage.set_value(0, -1.0);
    }
    storage.sync_ghosts();
    EXPECT_THROW(
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), finite_source, ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit),
        std::invalid_argument);
    if (communicator->getRank() == 0 && storage.num_owned_cells() != 0)
    {
        storage.set_value(0, 2.0);
    }
    storage.sync_ghosts();

    const std::function<double(Pack::local_ordinal_type)> invalid_sink =
        [rank = communicator->getRank()](Pack::local_ordinal_type) { return rank == 0 ? -1.0 : 0.0; };
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion,
                     dirichlet_condition(), zero_boundary_value(), finite_source, ScalarTransportDiscretization{},
                     NonOrthogonalTreatment::Explicit, nullptr, nullptr, Teuchos::null, invalid_sink),
        std::invalid_argument);

    const std::function<std::optional<double>(Pack::local_ordinal_type)> invalid_fixed =
        [rank = communicator->getRank()](Pack::local_ordinal_type) -> std::optional<double>
    { return rank == 0 ? std::optional<double>{std::numeric_limits<double>::infinity()} : std::nullopt; };
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion,
                     dirichlet_condition(), zero_boundary_value(), finite_source, ScalarTransportDiscretization{},
                     NonOrthogonalTreatment::Explicit, nullptr, nullptr, Teuchos::null, {}, invalid_fixed),
        std::invalid_argument);

    const ScalarCellValueProvider<Pack> invalid_source = [rank = communicator->getRank()](Pack::local_ordinal_type)
    { return rank == 0 ? std::numeric_limits<double>::quiet_NaN() : 0.0; };
    EXPECT_THROW(
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), invalid_source, ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit),
        std::invalid_argument);

    const auto mapped_mesh = make_distributed_mapped_line();
    MappedScalar mapped_old(mapped_mesh, 1.0, "old");
    MappedScalar mapped_storage(mapped_mesh, 2.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 2.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.25, "diffusion");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25, mapped_storage, mapped_advection,
                     mapped_diffusion, dirichlet_condition(), zero_boundary_value(), finite_source,
                     ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit,
                     static_cast<const MappedScalar*>(nullptr), static_cast<const MappedScalar*>(nullptr),
                     Teuchos::null, invalid_sink),
        std::invalid_argument);
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25, mapped_storage, mapped_advection,
                     mapped_diffusion, dirichlet_condition(), zero_boundary_value(), finite_source,
                     ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit,
                     static_cast<const MappedScalar*>(nullptr), static_cast<const MappedScalar*>(nullptr),
                     Teuchos::null, {}, invalid_fixed),
        std::invalid_argument);
    EXPECT_THROW(weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25, mapped_storage, mapped_advection,
                     mapped_diffusion, dirichlet_condition(), zero_boundary_value(), invalid_source,
                     ScalarTransportDiscretization{}, NonOrthogonalTreatment::Explicit),
        std::invalid_argument);
}

TEST(ScalarTransportDiscretizationTest, Bdf2BoundedLinearUpwindMatchesLegacyMappedAndPhysicalPaths)
{
    using namespace SimpleFluid::FVM;

    const ScalarTransportDiscretization advanced{ScalarTimeScheme::BDF2, ScalarConvectionScheme::BoundedLinearUpwind};
    const ScalarTransportDiscretization bdf2_upwind{ScalarTimeScheme::BDF2, ScalarConvectionScheme::Upwind};

    auto legacy_mesh = make_legacy_line();
    LegacyScalar old(legacy_mesh, "old");
    LegacyScalar older(legacy_mesh, "older");
    LegacyScalar storage(legacy_mesh, 2.0, "storage");
    LegacyScalar advection(legacy_mesh, 2.0, "advection");
    LegacyScalar diffusion(legacy_mesh, 0.25, "diffusion");
    LegacyFlux flux(legacy_mesh, 0.0, "flux");
    set_cell_values(old, {0.0, 1.0, 0.25});
    set_cell_values(older, {0.25, 0.5, 0.75});
    set_positive_x_flux(*legacy_mesh, flux);

    const auto legacy_advanced =
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), cell_source(), advanced, NonOrthogonalTreatment::Explicit, &older);
    const auto legacy_upwind =
        weighted_scalar_transport_system<Pack>(old, flux, 0.25, storage, advection, diffusion, dirichlet_condition(),
            zero_boundary_value(), cell_source(), bdf2_upwind, NonOrthogonalTreatment::Explicit, &older);

    auto mapped_mesh = make_mapped_line();
    MappedScalar mapped_old(mapped_mesh, "old");
    MappedScalar mapped_older(mapped_mesh, "older");
    MappedScalar mapped_storage(mapped_mesh, 2.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 2.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.25, "diffusion");
    MappedScalar density(mapped_mesh, 1.0, "density");
    MappedScalar specific_heat(mapped_mesh, 2.0, "specific_heat");
    MappedScalar conductivity(mapped_mesh, 0.25, "conductivity");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    set_cell_values(mapped_old, {0.0, 1.0, 0.25});
    set_cell_values(mapped_older, {0.25, 0.5, 0.75});
    set_positive_x_flux(*mapped_mesh, mapped_flux);

    const auto mapped_advanced = weighted_scalar_transport_system<Pack>(mapped_old, mapped_flux, 0.25, mapped_storage,
        mapped_advection, mapped_diffusion, dirichlet_condition(), zero_boundary_value(), cell_source(), advanced,
        NonOrthogonalTreatment::Explicit, &mapped_older);
    const auto physical_advanced = physical_temperature_transport_system<Pack>(mapped_old, mapped_flux, 0.25, density,
        specific_heat, conductivity, dirichlet_condition(), zero_boundary_value(), cell_source(), advanced,
        NonOrthogonalTreatment::Explicit, &mapped_older);

    expect_systems_equal(mapped_advanced, legacy_advanced, 3);
    expect_systems_equal(physical_advanced, mapped_advanced, 3);

    auto correction_norm = 0.0;
    for (size_t cell = 0; cell < 3; ++cell)
    {
        correction_norm = std::max(
            correction_norm, std::abs(legacy_advanced.rhs->getData()[cell] - legacy_upwind.rhs->getData()[cell]));
    }
    EXPECT_GT(correction_norm, 1.0e-12);
}

} // namespace
