/**
 * @file testScalarTransportDiscretization.cc
 * @brief Accuracy, boundedness, and backend-parity tests for scalar schemes.
 */

#include "geometry/MeshConcepts.hh"
#include <gtest/gtest.h>

#include "FVM/TransportSystemFactory.hh"
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
#include <string_view>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using CartesianMesh = SimpleFluid::MeshHandle<Pack>;
using MappedMesh = SimpleFluid::MeshHandle<Pack>;
using CartesianScalar = SimpleFluid::CellField<Pack>;
using CartesianFlux = SimpleFluid::FaceField<Pack>;
using MappedScalar = SimpleFluid::ScalarCellFieldStored<Pack, MappedMesh>;
using MappedFlux = SimpleFluid::ScalarFaceFieldStored<Pack, MappedMesh>;
using TransportSystem = SimpleFluid::FVM::TransportSystem<Pack>;
using CartesianFactory = SimpleFluid::FVM::TransportSystemFactory<Pack, CartesianMesh>;
using MappedFactory = SimpleFluid::FVM::TransportSystemFactory<Pack, MappedMesh>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<CartesianMesh> make_cartesian_line()
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

SimpleFluid::SP<CartesianMesh> make_periodic_cartesian_line()
{
    auto geometry = std::make_shared<CartesianMesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0., 1., 2., 3.}, {0., 1.}, {0., 1.}}},
        SimpleFluid::Vec3D<bool>{{true, false, false}});
    return std::make_shared<CartesianMesh>(std::move(geometry));
}

template<class MeshPtr> auto periodic_scalar_condition(MeshPtr mesh)
{
    return [mesh = std::move(mesh)](int batch_id, size_t)
    {
        const auto& name = mesh->boundary_batch_name(batch_id);
        return SimpleFluid::BoundaryCondition{name == "xmin" || name == "xmax"
                                                  ? SimpleFluid::BoundaryConditionType::Periodic
                                                  : SimpleFluid::BoundaryConditionType::Neumann,
            0.0};
    };
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

void expect_system_scaled(const TransportSystem& actual, const TransportSystem& expected, size_t cells, double scale)
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
                scale * matrix_entry(*expected.matrix, row_lid, column_lid), 1.0e-12);
        }
        EXPECT_NEAR(actual.rhs->getData()[row], scale * expected.rhs->getData()[row], 1.0e-12);
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

template<SimpleFluid::MeshType MeshType, class FaceField>
void set_positive_x_flux(const MeshType& mesh, FaceField& flux)
{
    for (const auto face_lid : flux.owned_face_ids())
    {
        flux.set_value(face_lid, 0.2 * mesh.face_area_vector(face_lid).x);
    }
    flux.sync_ghosts();
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

TEST(ScalarTransportDiscretizationTest, WeightedPacksConserveIndependentStorageSourcesAndConstraints)
{
    using namespace SimpleFluid;
    using namespace SimpleFluid::FVM;
    auto check = []<SimpleFluid::MeshType MeshType>(SP<MeshType> mesh)
    {
        using Factory = TransportSystemFactory<Pack, std::remove_const_t<MeshType>>;
        using Field = typename Factory::WeightedScalarTerms::Field;
        using Flux = typename MeshFieldTraits<Pack, std::remove_const_t<MeshType>>::scalar_face_type;
        Field old(mesh, 7.0, "old"), storage(mesh, 3.0, "storage"), accepted(mesh, 5.0, "accepted"),
            advection(mesh, 0.0, "advection"), diffusion(mesh, 0.0, "diffusion");
        Flux flux(mesh, 0.0, "flux");
        int calls = 0;
        typename Factory::WeightedScalarTerms terms{.history = {old, 0.5},
            .face_fluxes = flux,
            .coefficients = {storage, advection, diffusion, &accepted},
            .sources = {[&](auto)
                {
                    ++calls;
                    return 11.0;
                },
                [](auto) { return 13.0; }}};
        typename Factory::WeightedScalarBoundary boundary{
            [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; }, zero_boundary_value()};
        EXPECT_EQ(calls, 0); // Pack construction must not evaluate callbacks.
        const auto system = Factory::assemble(terms, boundary, {});
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto local = static_cast<Pack::local_ordinal_type>(row);
            const auto column =
                system.matrix->getColMap()->getLocalElement(system.matrix->getRowMap()->getGlobalElement(local));
            ASSERT_DOUBLE_EQ(mesh->cell_volume(local), 2.0);
            // V*(w/dt + sink) and V*(w_old*old/dt + source).
            EXPECT_DOUBLE_EQ(matrix_entry(*system.matrix, local, column), 38.0);
            EXPECT_DOUBLE_EQ(system.rhs->getData()[row], 162.0);
        }
        EXPECT_EQ(calls, mesh->num_owned_cells());
        terms.constraints.fixed_cell_value = [](auto) { return std::optional<double>{4.0}; };
        const auto fixed = Factory::assemble(terms, boundary, {}, {.cached_matrix = system.matrix});
        EXPECT_EQ(fixed.matrix.get(), system.matrix.get());
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto local = static_cast<Pack::local_ordinal_type>(row);
            const auto column =
                fixed.matrix->getColMap()->getLocalElement(fixed.matrix->getRowMap()->getGlobalElement(local));
            EXPECT_DOUBLE_EQ(matrix_entry(*fixed.matrix, local, column), 1.0);
            EXPECT_DOUBLE_EQ(fixed.rhs->getData()[row], 4.0);
        }
        terms.constraints = {};
        terms.sources = {}; // Missing source and sink are both zero.
        const auto zero_source = Factory::assemble(terms, boundary, {}, {.cached_matrix = fixed.matrix});
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto local = static_cast<Pack::local_ordinal_type>(row);
            const auto column = zero_source.matrix->getColMap()->getLocalElement(
                zero_source.matrix->getRowMap()->getGlobalElement(local));
            EXPECT_DOUBLE_EQ(matrix_entry(*zero_source.matrix, local, column), 12.0);
            EXPECT_DOUBLE_EQ(zero_source.rhs->getData()[row], 140.0);
        }
    };
    auto db = std::make_shared<Database>(*test::make_single_hex_database());
    db->set("X", ArrReal{0.0, 2.0});
    check(test::build_mesh<Pack>(db));
    check(std::make_shared<const MappedMesh>(test::make_unstructured_hex_line(1, 2.0)));
}

TEST(ScalarTransportDiscretizationTest, AdvancedPacksMatchAnalyticalRowsAndRebuildCachedValues)
{
    using namespace SimpleFluid;
    using namespace SimpleFluid::FVM;
    auto check = []<SimpleFluid::MeshType MeshType>(SP<MeshType> mesh)
    {
        using Backend = std::remove_const_t<MeshType>;
        using Factory = TransportSystemFactory<Pack, Backend>;
        using Field = typename Factory::WeightedScalarTerms::Field;
        using Flux = typename MeshFieldTraits<Pack, Backend>::scalar_face_type;
        Field old(mesh, "old"), older(mesh, "older"), storage(mesh, 2.0, "storage"), advection(mesh, 1.5, "advection"),
            diffusion(mesh, 0.25, "diffusion");
        Flux flux(mesh, 0.0, "flux");
        set_cell_values(old, {0.2, 1.1, 0.4});
        set_cell_values(older, {0.1, 0.7, 0.3});
        MeshBoundaryCache<Pack, Backend> coefficients{{}, mesh};
        for (const auto& [id, batch] : mesh->boundary_batches())
            coefficients.value.emplace(id, Arr<double>(batch.face_lids.size(), 0.35));
        TransportGeometryCache<Backend> geometry(*mesh);
        typename Factory::WeightedScalarTerms terms{.history = {old, 0.25, &older},
            .face_fluxes = flux,
            .coefficients = {storage, advection, diffusion},
            .sources = {cell_source(), [](auto cell) { return 0.02 * (cell + 1); }},
            .constraints = {[](auto cell) -> std::optional<double>
                { return cell == 1 ? std::optional<double>{0.75} : std::nullopt; }}};
        typename Factory::WeightedScalarBoundary boundary{
            dirichlet_condition(), [](int, size_t) { return 0.4; }, &coefficients};
        typename Factory::WeightedScalarOptions options{
            .discretization = {ScalarTimeScheme::BDF2, ScalarConvectionScheme::BoundedLinearUpwind},
            .correction = {NonOrthogonalTreatment::Hybrid, &old},
            .coefficient_interpolation = FaceCoefficientInterpolation::Linear};
        auto system = Factory::assemble(terms, boundary, options, {.geometry_cache = &geometry});
        for (int generation = 0; generation < 2; ++generation)
        {
            // BDF2 storage is 12; five boundary conductances total 3.5;
            // the remaining interior conductance is 0.25. Fixed row stays identity.
            const std::array<double, 3> diagonals{15.77 + 12.0 * generation, 1.0, 15.81 + 12.0 * generation};
            const std::array<double, 3> rhs{4.25 + 2.8 * generation, 0.75, 6.75 + 5.2 * generation};
            for (Pack::local_ordinal_type row = 0; row < 3; ++row)
            {
                EXPECT_NEAR(matrix_entry(*system.matrix, row, row), diagonals[row], 1.0e-12);
                EXPECT_NEAR(system.rhs->getData()[row], rhs[row], 1.0e-12);
                for (Pack::local_ordinal_type column = 0; column < 3; ++column)
                    if (row != column)
                        EXPECT_NEAR(
                            matrix_entry(*system.matrix, row, column), row != 1 && column == 1 ? -0.25 : 0.0, 1.0e-12);
            }
            if (generation == 0)
            {
                storage.owned_data().putScalar(4.0);
                storage.sync_ghosts();
                auto rebuilt = Factory::assemble(
                    terms, boundary, options, {.cached_matrix = system.matrix, .geometry_cache = &geometry});
                EXPECT_EQ(rebuilt.matrix.get(), system.matrix.get());
                system = std::move(rebuilt);
            }
        }
    };
    check(make_cartesian_line());
    check(make_mapped_line());
}

TEST(ScalarTransportDiscretizationTest, TemperatureRhsRefreshRequiresOperatorAndPreservesMatrixValues)
{
    using namespace SimpleFluid;
    using namespace SimpleFluid::FVM;
    auto check = []<SimpleFluid::MeshType MeshType>(SP<MeshType> mesh)
    {
        using Backend = std::remove_const_t<MeshType>;
        using Factory = TransportSystemFactory<Pack, Backend>;
        using Field = typename Factory::TemperatureTerms::Field;
        using Flux = typename MeshFieldTraits<Pack, Backend>::scalar_face_type;
        Field old(mesh, 7.0, "old"), density(mesh, 2.0, "density"), heat_capacity(mesh, 3.0, "heat_capacity"),
            conductivity(mesh, 0.0, "conductivity");
        Flux flux(mesh, 0.0, "flux");
        typename Factory::TemperatureTerms terms{
            .history = {old, 0.5}, .face_fluxes = flux, .material = {density, heat_capacity, conductivity}};
        typename Factory::TemperatureBoundary boundary{
            [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; }, zero_boundary_value()};
        typename Factory::TemperatureOptions options{};
        EXPECT_THROW(Factory::refresh_temperature_rhs(terms, boundary, options, {}), std::invalid_argument);
        const auto system = Factory::assemble(terms, boundary, options);
        ASSERT_EQ(system.matrix->getLocalNumRows(), 1);
        EXPECT_DOUBLE_EQ(matrix_entry(*system.matrix, 0, 0), 12.0);
        EXPECT_DOUBLE_EQ(system.rhs->getData()[0], 84.0);
        // A sentinel makes any accidental numeric rebuild observable.
        system.matrix->resumeFill();
        system.matrix->setAllToScalar(29.0);
        system.matrix->fillComplete(mesh->owned_cell_map(), mesh->owned_cell_map());
        terms.power_density = [](auto) { return 17.0; };
        const auto refreshed =
            Factory::refresh_temperature_rhs(terms, boundary, options, {.cached_matrix = system.matrix});
        EXPECT_EQ(refreshed.matrix.get(), system.matrix.get());
        EXPECT_DOUBLE_EQ(matrix_entry(*refreshed.matrix, 0, 0), 29.0);
        EXPECT_DOUBLE_EQ(refreshed.rhs->getData()[0], 101.0);
        {
            terms.history.older_values = &old;
            options.discretization.time = ScalarTimeScheme::BDF2;
            EXPECT_THROW(Factory::refresh_temperature_rhs(terms, boundary, options, {.cached_matrix = system.matrix}),
                std::invalid_argument);
            options.discretization.time = ScalarTimeScheme::BackwardEuler;
            options.discretization.convection = ScalarConvectionScheme::BoundedLinearUpwind;
            EXPECT_THROW(Factory::refresh_temperature_rhs(terms, boundary, options, {.cached_matrix = system.matrix}),
                std::invalid_argument);
            EXPECT_DOUBLE_EQ(matrix_entry(*system.matrix, 0, 0), 29.0);
        }
    };
    check(test::build_mesh<Pack>(test::make_single_hex_database()));
    check(std::make_shared<const MappedMesh>(test::make_unstructured_hex_line(1)));
}

TEST(ScalarTransportDiscretizationTest, Bdf2RequiresACompatibleOlderFieldForCartesianAndMappedStorage)
{
    using namespace SimpleFluid::FVM;

    const ScalarTransportDiscretization bdf2{ScalarTimeScheme::BDF2, ScalarConvectionScheme::Upwind};

    auto cartesian_mesh = make_cartesian_line();
    CartesianScalar old(cartesian_mesh, 1.0, "old");
    CartesianScalar storage(cartesian_mesh, 2.0, "storage");
    CartesianScalar advection(cartesian_mesh, 2.0, "advection");
    CartesianScalar diffusion(cartesian_mesh, 0.25, "diffusion");
    CartesianFlux flux(cartesian_mesh, 0.0, "flux");

    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {cell_source(), {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);

    auto other_cartesian_mesh = make_cartesian_line();
    CartesianScalar incompatible_older(other_cartesian_mesh, 0.5, "incompatible_older");
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, &incompatible_older},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {cell_source(), {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);

    auto mapped_mesh = make_mapped_line();
    MappedScalar mapped_old(mapped_mesh, 1.0, "old");
    MappedScalar mapped_storage(mapped_mesh, 2.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 2.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.25, "diffusion");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    EXPECT_THROW(MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                             .face_fluxes = mapped_flux,
                                             .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                             .sources = {cell_source(), {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);
}

TEST(ScalarTransportDiscretizationTest, Bdf2UsesCurrentStorageAsDerivativeCoefficient)
{
    using namespace SimpleFluid::FVM;

    const ScalarTransportDiscretization bdf2{ScalarTimeScheme::BDF2, ScalarConvectionScheme::Upwind};
    constexpr auto base_storage = 2.0;
    constexpr auto changed_storage = 5.0;
    constexpr auto storage_ratio = changed_storage / base_storage;

    auto cartesian_mesh = make_cartesian_line();
    CartesianScalar old(cartesian_mesh, 3.0, "old");
    CartesianScalar older(cartesian_mesh, 1.0, "older");
    CartesianScalar storage(cartesian_mesh, base_storage, "storage");
    CartesianScalar changed(cartesian_mesh, changed_storage, "changed_storage");
    CartesianScalar advection(cartesian_mesh, 0.0, "advection");
    CartesianScalar diffusion(cartesian_mesh, 0.0, "diffusion");
    CartesianFlux flux(cartesian_mesh, 0.0, "flux");

    const auto cartesian_base = CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, &older},
                                                         .face_fluxes = flux,
                                                         .coefficients = {storage, advection, diffusion},
                                                         .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
        {dirichlet_condition(), zero_boundary_value()},
        {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto cartesian_changed =
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, &older},
                                    .face_fluxes = flux,
                                    .coefficients = {changed, advection, diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    expect_system_scaled(cartesian_changed, cartesian_base, 3, storage_ratio);

    auto mapped_mesh = make_mapped_line();
    MappedScalar mapped_old(mapped_mesh, 3.0, "old");
    MappedScalar mapped_older(mapped_mesh, 1.0, "older");
    MappedScalar mapped_storage(mapped_mesh, base_storage, "storage");
    MappedScalar mapped_changed(mapped_mesh, changed_storage, "changed_storage");
    MappedScalar mapped_advection(mapped_mesh, 0.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.0, "diffusion");
    MappedScalar density(mapped_mesh, 1.0, "density");
    MappedScalar changed_density(mapped_mesh, storage_ratio, "changed_density");
    MappedScalar specific_heat(mapped_mesh, base_storage, "specific_heat");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");

    const auto mapped_base =
        MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25, &mapped_older},
                                    .face_fluxes = mapped_flux,
                                    .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto mapped_changed_system =
        MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25, &mapped_older},
                                    .face_fluxes = mapped_flux,
                                    .coefficients = {mapped_changed, mapped_advection, mapped_diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    expect_system_scaled(mapped_changed_system, mapped_base, 3, storage_ratio);

    const auto physical_base =
        MappedFactory::assemble(MappedFactory::TemperatureTerms{.history = {mapped_old, 0.25, &mapped_older},
                                    .face_fluxes = mapped_flux,
                                    .material = {density, specific_heat, mapped_diffusion},
                                    .power_density = [](Pack::local_ordinal_type) { return 0.0; }},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto physical_changed =
        MappedFactory::assemble(MappedFactory::TemperatureTerms{.history = {mapped_old, 0.25, &mapped_older},
                                    .face_fluxes = mapped_flux,
                                    .material = {changed_density, specific_heat, mapped_diffusion},
                                    .power_density = [](Pack::local_ordinal_type) { return 0.0; }},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = bdf2, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    expect_system_scaled(physical_changed, physical_base, 3, storage_ratio);
}

TEST(ScalarTransportDiscretizationTest, PeriodicBoundedLinearUpwindUsesWrappedConservativeDisplacement)
{
    using namespace SimpleFluid::FVM;

    const ScalarTransportDiscretization advanced{
        ScalarTimeScheme::BackwardEuler, ScalarConvectionScheme::BoundedLinearUpwind};
    const ScalarTransportDiscretization upwind{ScalarTimeScheme::BackwardEuler, ScalarConvectionScheme::Upwind};

    auto cartesian_mesh = make_periodic_cartesian_line();
    CartesianScalar old(cartesian_mesh, "old");
    CartesianScalar storage(cartesian_mesh, 1.0, "storage");
    CartesianScalar advection(cartesian_mesh, 1.0, "advection");
    CartesianScalar diffusion(cartesian_mesh, 0.0, "diffusion");
    CartesianFlux flux(cartesian_mesh, 0.0, "flux");
    set_cell_values(old, {0.0, 0.0, 1.0});
    for (const auto face : flux.owned_face_ids())
        if (cartesian_mesh->is_periodic_face(face))
            flux.set_value(face, 0.2 * cartesian_mesh->face_normal(face).x);
    flux.sync_ghosts();
    const auto cartesian_condition = periodic_scalar_condition(cartesian_mesh);

    const auto cartesian_advanced =
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                    .face_fluxes = flux,
                                    .coefficients = {storage, advection, diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {cartesian_condition, zero_boundary_value()},
            {.discretization = advanced, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto cartesian_upwind =
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                    .face_fluxes = flux,
                                    .coefficients = {storage, advection, diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {cartesian_condition, zero_boundary_value()},
            {.discretization = upwind, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    expect_systems_equal(cartesian_advanced, cartesian_upwind, 3);

    const auto mapped_mesh = cartesian_mesh;
    MappedScalar mapped_old(mapped_mesh, "old");
    MappedScalar mapped_storage(mapped_mesh, 1.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 1.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.0, "diffusion");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    set_cell_values(mapped_old, {0.0, 0.0, 1.0});
    for (const auto face : mapped_flux.owned_face_ids())
        if (cartesian_mesh->is_periodic_face(face))
            mapped_flux.set_value(face, 0.2 * cartesian_mesh->face_normal(face).x);
    mapped_flux.sync_ghosts();
    const auto mapped_condition = periodic_scalar_condition(mapped_mesh);

    const auto mapped_advanced =
        MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                    .face_fluxes = mapped_flux,
                                    .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {mapped_condition, zero_boundary_value()},
            {.discretization = advanced, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto mapped_upwind =
        MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                    .face_fluxes = mapped_flux,
                                    .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {mapped_condition, zero_boundary_value()},
            {.discretization = upwind, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    expect_systems_equal(mapped_advanced, mapped_upwind, 3);
    TransportGeometryCache<MappedMesh> mapped_geometry(*mapped_mesh);
    const auto mapped_cached =
        MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                    .face_fluxes = mapped_flux,
                                    .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                    .sources = {[](Pack::local_ordinal_type) { return 0.0; }, {}}},
            {mapped_condition, zero_boundary_value()},
            {.discretization = advanced, .correction = {NonOrthogonalTreatment::Explicit, nullptr}},
            {.geometry_cache = &mapped_geometry});
    expect_systems_equal(mapped_cached, mapped_advanced, 3);
}

TEST(ScalarTransportDiscretizationTest, AdvancedAssemblersRejectRankDivergentPreflightInputsCollectively)
{
    using namespace SimpleFluid::FVM;

    const auto cartesian_mesh = make_cartesian_line();
    const auto communicator = cartesian_mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "Requires at least two MPI ranks.";
    }

    CartesianScalar old(cartesian_mesh, 1.0, "old");
    CartesianScalar older(cartesian_mesh, 0.5, "older");
    CartesianScalar storage(cartesian_mesh, 2.0, "storage");
    CartesianScalar advection(cartesian_mesh, 2.0, "advection");
    CartesianScalar diffusion(cartesian_mesh, 0.25, "diffusion");
    CartesianFlux flux(cartesian_mesh, 0.0, "flux");
    const ScalarCellValueProvider<Pack> finite_source = [](Pack::local_ordinal_type) { return 0.0; };
    // Every rank enters the same factory overload, but selects a different
    // correction category. Reject before choosing different collective paths.
    CartesianFactory::ScalarOptions scalar_options{};
    if (communicator->getRank() == 0)
        scalar_options.correction = TransportCorrection<CartesianScalar>{NonOrthogonalTreatment::Explicit, &old};
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::ScalarTerms{{old, 0.25}, flux, {0.25}},
                     {dirichlet_condition(), zero_boundary_value()}, scalar_options),
        std::invalid_argument);

    const ScalarTransportDiscretization rank_dependent_policy{
        communicator->getRank() == 0 ? ScalarTimeScheme::BackwardEuler : ScalarTimeScheme::BDF2,
        ScalarConvectionScheme::Upwind};
    EXPECT_THROW(
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, &older},
                                    .face_fluxes = flux,
                                    .coefficients = {storage, advection, diffusion},
                                    .sources = {finite_source, {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = rank_dependent_policy, .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);

    const auto positive_rank_dependent_time_step = communicator->getRank() == 0 ? 0.25 : 0.5;
    EXPECT_THROW(
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, positive_rank_dependent_time_step},
                                    .face_fluxes = flux,
                                    .coefficients = {storage, advection, diffusion},
                                    .sources = {finite_source, {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = ScalarTransportDiscretization{},
                .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);

    const auto rank_dependent_time_step = communicator->getRank() == 0 ? -0.25 : 0.25;
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, rank_dependent_time_step},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);

    if (communicator->getRank() == 0 && storage.num_owned_cells() != 0)
    {
        storage.set_value(0, -1.0);
    }
    storage.sync_ghosts();
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);
    if (communicator->getRank() == 0 && storage.num_owned_cells() != 0)
    {
        storage.set_value(0, 2.0);
    }
    storage.sync_ghosts();

    const std::function<double(Pack::local_ordinal_type)> invalid_sink =
        [rank = communicator->getRank()](Pack::local_ordinal_type) { return rank == 0 ? -1.0 : 0.0; };
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, nullptr},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, invalid_sink}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}},
                     {.cached_matrix = Teuchos::null}),
        std::invalid_argument);

    const std::function<std::optional<double>(Pack::local_ordinal_type)> invalid_fixed =
        [rank = communicator->getRank()](Pack::local_ordinal_type) -> std::optional<double>
    { return rank == 0 ? std::optional<double>{std::numeric_limits<double>::infinity()} : std::nullopt; };
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, nullptr},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, {}},
                                             .constraints = {invalid_fixed}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}},
                     {.cached_matrix = Teuchos::null}),
        std::invalid_argument);

    const ScalarCellValueProvider<Pack> invalid_source = [rank = communicator->getRank()](Pack::local_ordinal_type)
    { return rank == 0 ? std::numeric_limits<double>::quiet_NaN() : 0.0; };
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {invalid_source, {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);

    const ScalarBoundaryConditionProvider<Pack> throwing_boundary_condition = [rank = communicator->getRank()](
                                                                                  int, size_t)
    {
        if (rank == 0)
        {
            throw std::runtime_error("rank-local boundary condition failure");
        }
        return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    };
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, {}}},
                     {throwing_boundary_condition, zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Implicit, nullptr}}),
        std::runtime_error);

    const ScalarBoundaryValueProvider<Pack> throwing_boundary_value = [rank = communicator->getRank()](int, size_t)
    {
        if (rank == 0)
        {
            throw std::runtime_error("rank-local boundary value failure");
        }
        return 0.0;
    };
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, {}}},
                     {dirichlet_condition(), throwing_boundary_value},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Implicit, nullptr}}),
        std::runtime_error);

    const ScalarBoundaryConditionProvider<Pack> rank_dependent_robin = [rank = communicator->getRank()](int, size_t)
    {
        return SimpleFluid::BoundaryCondition{
            rank == 0 ? SimpleFluid::BoundaryConditionType::Robin : SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    };
    EXPECT_THROW(CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25},
                                             .face_fluxes = flux,
                                             .coefficients = {storage, advection, diffusion},
                                             .sources = {finite_source, {}}},
                     {rank_dependent_robin, zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Implicit, nullptr}}),
        std::invalid_argument);

    const auto mapped_mesh = make_distributed_mapped_line();
    MappedScalar mapped_old(mapped_mesh, 1.0, "old");
    MappedScalar mapped_storage(mapped_mesh, 2.0, "storage");
    MappedScalar mapped_advection(mapped_mesh, 2.0, "advection");
    MappedScalar mapped_diffusion(mapped_mesh, 0.25, "diffusion");
    MappedFlux mapped_flux(mapped_mesh, 0.0, "flux");
    MappedFactory::ScalarOptions mapped_scalar_options{};
    if (communicator->getRank() == 0)
        mapped_scalar_options.correction =
            TransportCorrection<MappedScalar>{NonOrthogonalTreatment::Explicit, &mapped_old};
    EXPECT_THROW(MappedFactory::assemble(MappedFactory::ScalarTerms{{mapped_old, 0.25}, mapped_flux, {0.25}},
                     {dirichlet_condition(), zero_boundary_value()}, mapped_scalar_options),
        std::invalid_argument);
    EXPECT_THROW(MappedFactory::assemble(
                     MappedFactory::WeightedScalarTerms{.history = {mapped_old, positive_rank_dependent_time_step},
                         .face_fluxes = mapped_flux,
                         .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                         .sources = {finite_source, {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);
    EXPECT_THROW(
        MappedFactory::assemble(
            MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25, static_cast<const MappedScalar*>(nullptr)},
                .face_fluxes = mapped_flux,
                .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                .sources = {finite_source, invalid_sink}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = ScalarTransportDiscretization{},
                .correction = {NonOrthogonalTreatment::Explicit, static_cast<const MappedScalar*>(nullptr)}},
            {.cached_matrix = Teuchos::null}),
        std::invalid_argument);
    EXPECT_THROW(
        MappedFactory::assemble(
            MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25, static_cast<const MappedScalar*>(nullptr)},
                .face_fluxes = mapped_flux,
                .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                .sources = {finite_source, {}},
                .constraints = {invalid_fixed}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = ScalarTransportDiscretization{},
                .correction = {NonOrthogonalTreatment::Explicit, static_cast<const MappedScalar*>(nullptr)}},
            {.cached_matrix = Teuchos::null}),
        std::invalid_argument);
    EXPECT_THROW(MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                             .face_fluxes = mapped_flux,
                                             .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                             .sources = {invalid_source, {}}},
                     {dirichlet_condition(), zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Explicit, nullptr}}),
        std::invalid_argument);
    EXPECT_THROW(MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                             .face_fluxes = mapped_flux,
                                             .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                             .sources = {finite_source, {}}},
                     {throwing_boundary_condition, zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Implicit, nullptr}}),
        std::runtime_error);
    EXPECT_THROW(MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                             .face_fluxes = mapped_flux,
                                             .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                             .sources = {finite_source, {}}},
                     {dirichlet_condition(), throwing_boundary_value},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Implicit, nullptr}}),
        std::runtime_error);
    EXPECT_THROW(MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25},
                                             .face_fluxes = mapped_flux,
                                             .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                             .sources = {finite_source, {}}},
                     {rank_dependent_robin, zero_boundary_value()},
                     {.discretization = ScalarTransportDiscretization{},
                         .correction = {NonOrthogonalTreatment::Implicit, nullptr}}),
        std::invalid_argument);
}

TEST(ScalarTransportDiscretizationTest, Bdf2BoundedLinearUpwindMatchesCartesianMappedAndPhysicalPaths)
{
    using namespace SimpleFluid::FVM;

    const ScalarTransportDiscretization advanced{ScalarTimeScheme::BDF2, ScalarConvectionScheme::BoundedLinearUpwind};
    const ScalarTransportDiscretization bdf2_upwind{ScalarTimeScheme::BDF2, ScalarConvectionScheme::Upwind};

    auto cartesian_mesh = make_cartesian_line();
    CartesianScalar old(cartesian_mesh, "old");
    CartesianScalar older(cartesian_mesh, "older");
    CartesianScalar storage(cartesian_mesh, 2.0, "storage");
    CartesianScalar advection(cartesian_mesh, 2.0, "advection");
    CartesianScalar diffusion(cartesian_mesh, 0.25, "diffusion");
    CartesianFlux flux(cartesian_mesh, 0.0, "flux");
    set_cell_values(old, {0.0, 1.0, 0.25});
    set_cell_values(older, {0.25, 0.5, 0.75});
    set_positive_x_flux(*cartesian_mesh, flux);

    const auto cartesian_advanced =
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, &older},
                                    .face_fluxes = flux,
                                    .coefficients = {storage, advection, diffusion},
                                    .sources = {cell_source(), {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = advanced, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto cartesian_upwind =
        CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, 0.25, &older},
                                    .face_fluxes = flux,
                                    .coefficients = {storage, advection, diffusion},
                                    .sources = {cell_source(), {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = bdf2_upwind, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});

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

    const auto mapped_advanced =
        MappedFactory::assemble(MappedFactory::WeightedScalarTerms{.history = {mapped_old, 0.25, &mapped_older},
                                    .face_fluxes = mapped_flux,
                                    .coefficients = {mapped_storage, mapped_advection, mapped_diffusion},
                                    .sources = {cell_source(), {}}},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = advanced, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});
    const auto physical_advanced =
        MappedFactory::assemble(MappedFactory::TemperatureTerms{.history = {mapped_old, 0.25, &mapped_older},
                                    .face_fluxes = mapped_flux,
                                    .material = {density, specific_heat, conductivity},
                                    .power_density = cell_source()},
            {dirichlet_condition(), zero_boundary_value()},
            {.discretization = advanced, .correction = {NonOrthogonalTreatment::Explicit, nullptr}});

    expect_systems_equal(mapped_advanced, cartesian_advanced, 3);
    expect_systems_equal(physical_advanced, mapped_advanced, 3);

    auto correction_norm = 0.0;
    for (size_t cell = 0; cell < 3; ++cell)
    {
        correction_norm = std::max(
            correction_norm, std::abs(cartesian_advanced.rhs->getData()[cell] - cartesian_upwind.rhs->getData()[cell]));
    }
    EXPECT_GT(correction_norm, 1.0e-12);
}

TEST(ScalarTransportDiscretizationTest, CartesianAcceptedOldStorageAvoidsEndpointCancellation)
{
    using namespace SimpleFluid::FVM;
    const auto mesh = make_cartesian_line();
    CartesianScalar old(mesh, 3., "old"), accepted(mesh, 1., "accepted_storage"), trial(mesh, 1e20, "trial_storage"),
        unit(mesh, 1., "unit"), zero(mesh, 0., "zero");
    CartesianFlux flux(mesh, 0., "flux");
    const auto system = CartesianFactory::assemble(CartesianFactory::WeightedScalarTerms{.history = {old, .5},
                                                    .face_fluxes = flux,
                                                    .coefficients = {trial, unit, zero, &accepted},
                                                    .sources = {[](auto) { return 0.; }, {}}},
        {[](int, size_t) { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Neumann, 0.}; },
            zero_boundary_value()},
        {.correction = {NonOrthogonalTreatment::Implicit, nullptr}});
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto local = static_cast<Pack::local_ordinal_type>(row);
        const auto column =
            system.matrix->getColMap()->getLocalElement(system.matrix->getRowMap()->getGlobalElement(local));
        const auto volume = mesh->cell_volume(local);
        EXPECT_NEAR(matrix_entry(*system.matrix, local, column) / (2e20 * volume), 1., 1e-14);
        EXPECT_NEAR(system.rhs->getData()[row] / (6. * volume), 1., 1e-14);
    }
}

TEST(ScalarTransportDiscretizationTest, CartesianAcceptedStorageRejectsUnsupportedAndDivergentInputsCollectively)
{
    using namespace SimpleFluid::FVM;
    const auto mesh = make_cartesian_line();
    CartesianScalar old(mesh, 3., "old"), accepted(mesh, 1., "accepted_storage"), trial(mesh, 2., "trial_storage"),
        unit(mesh, 1., "unit"), zero(mesh, 0., "zero");
    CartesianFlux flux(mesh, 0., "flux");
    const auto assemble = [&](const CartesianScalar* previous, ScalarTimeScheme scheme = ScalarTimeScheme::BackwardEuler)
    {
        return CartesianFactory::assemble(
            CartesianFactory::WeightedScalarTerms{.history = {old, .5, scheme == ScalarTimeScheme::BDF2 ? &old : nullptr},
                .face_fluxes = flux,
                .coefficients = {trial, unit, zero, previous},
                .sources = {[](auto) { return 0.; }, {}}},
            {[](int, size_t)
                { return SimpleFluid::BoundaryCondition{SimpleFluid::BoundaryConditionType::Neumann, 0.}; },
                zero_boundary_value()},
            {.discretization = {.time = scheme}, .correction = {NonOrthogonalTreatment::Implicit, nullptr}});
    };
    EXPECT_THROW((void) assemble(&accepted, ScalarTimeScheme::BDF2), std::invalid_argument);
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
        EXPECT_THROW((void) assemble(comm->getRank() == 0 ? nullptr : &accepted), std::invalid_argument);
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto local = static_cast<Pack::local_ordinal_type>(row);
        if (mesh->owned_cell_map()->getGlobalElement(local) == 0)
            accepted.set_owned_value(local, std::numeric_limits<double>::quiet_NaN());
    }
    // set_owned_value updates only owned storage; coefficient assembly reads the
    // synchronized overlap view even in serial.
    accepted.sync_ghosts();
    int local_invalid = 0;
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        local_invalid += !std::isfinite(accepted.local_value(static_cast<Pack::local_ordinal_type>(row)));
    int global_invalid = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_invalid, &global_invalid);
    EXPECT_EQ(global_invalid, 1);
    EXPECT_THROW((void) assemble(&accepted), std::invalid_argument);
    accepted.owned_data().putScalar(1.);
    accepted.sync_ghosts();
    EXPECT_NO_THROW((void) assemble(&accepted));
}

} // namespace
