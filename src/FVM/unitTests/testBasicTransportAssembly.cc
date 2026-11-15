/** @file testBasicTransportAssembly.cc
 * @brief Basic scalar/vector assembly parity and callback contracts.
 */
#include "FVM/TransportSystem.hh"
#include "FVM/TransportSystemFactory.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <type_traits>

#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <limits>
#include <stdexcept>

#include <utility>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
template<class Field>
using TransportFactory = SimpleFluid::FVM::TransportSystemFactory<Pack, typename std::remove_cvref_t<Field>::mesh_type>;
using MeshType = SimpleFluid::Mesh<Pack>;
using NativeMesh = SimpleFluid::MeshHandle<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

auto make_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
}

double local_matrix_entry(
    const Pack::matrix_type& matrix, Pack::local_ordinal_type row, Pack::local_ordinal_type column)
{
    Pack::matrix_type::local_inds_host_view_type columns;
    Pack::matrix_type::values_host_view_type values;
    matrix.getLocalRowView(row, columns, values);
    for (size_t i = 0; i < columns.extent(0); ++i)
        if (columns[i] == column)
            return values[i];
    return 0.0;
}

/** Basic vector assembly visits callbacks once and retains scalar component algebra. */
TEST(BasicTransportAssemblyTest, BasicTransportSharesMatrixAndPreservesComponentCallbacks)
{
    using namespace SimpleFluid;
    using namespace SimpleFluid::FVM;
    using LO = Pack::local_ordinal_type;
    using Vec = vec3<Pack::scalar_type>;
    const auto mesh = make_mesh();
    VectorFieldType velocity(mesh, Vec{1.25, -2.5, 4.0}, "velocity");
    FaceField<Pack> flux(mesh, 0.0, "flux");
    const auto source_value = [](LO cell) { return Vec{0.5 + cell, -1.0 - cell, 3.0 + 2.0 * cell}; };
    const auto boundary_value = [](int batch, size_t face)
    { return Vec{2.0 + batch + 0.1 * face, -3.0 + 0.2 * face, 5.0 - 0.3 * batch}; };
    const auto boundary_type = [](int batch)
    {
        switch (batch % 3)
        {
            case 0:
                return BoundaryConditionType::Neumann;
            case 1:
                return BoundaryConditionType::NoSlip;
            default:
                return BoundaryConditionType::Dirichlet;
        }
    };
    const Vec gradient{0.25, -0.5, 0.75};
    Teuchos::RCP<Pack::matrix_type> cached;
    for (int generation = 0; generation < 3; ++generation)
    {
        const double diffusion = generation == 1 ? 0.0 : 0.4;
        for (const auto face : flux.owned_face_ids())
        {
            flux.set_value(face, generation == 0 ? 0.0
                                                 : (generation == 1 ? 0.2 : -0.3) *
                                                       mesh->face_area_vector(face).dot(Vec{1.0, -0.5, 0.25}));
        }
        std::vector<LO> source_calls;
        std::vector<std::pair<int, size_t>> value_calls, condition_calls;
        bool awaiting_condition = false;
        const auto vector = TransportFactory<decltype(velocity)>::assemble(
            typename TransportFactory<decltype(velocity)>::VectorTerms{
                .history = {.old_values = velocity, .time_step = 0.25},
                .face_fluxes = flux,
                .coefficients = {.diffusivity = diffusion},
                .source =
                    [&](LO cell)
                {
                    source_calls.push_back(cell);
                    return source_value(cell);
                }},
            typename TransportFactory<decltype(velocity)>::BasicVectorBoundary{
                .condition =
                    [&](int batch, size_t face)
                {
                    EXPECT_TRUE(awaiting_condition);
                    if (!value_calls.empty())
                        EXPECT_EQ(value_calls.back(), (std::pair{batch, face}));
                    awaiting_condition = false;
                    condition_calls.emplace_back(batch, face);
                    return VectorBoundaryCondition{boundary_type(batch), gradient};
                },
                .value =
                    [&](int batch, size_t face)
                {
                    EXPECT_FALSE(awaiting_condition);
                    awaiting_condition = true;
                    value_calls.emplace_back(batch, face);
                    return boundary_value(batch, face);
                }},
            {}, {.cached_matrix = cached});
        EXPECT_FALSE(awaiting_condition);
        ASSERT_EQ(source_calls.size(), mesh->num_owned_cells());
        for (size_t cell = 0; cell < source_calls.size(); ++cell)
            EXPECT_EQ(source_calls[cell], static_cast<LO>(cell));
        size_t expected_boundary_calls = 0;
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
            for (const auto face : batch.face_lids)
                if (mesh->is_owned_face(face) && mesh->is_boundary_face(face))
                    ++expected_boundary_calls;
        EXPECT_EQ(value_calls.size(), expected_boundary_calls);
        EXPECT_EQ(value_calls, condition_calls);
        if (!cached.is_null())
            EXPECT_EQ(vector.matrix.get(), cached.get());
        cached = vector.matrix;
        for (size_t component = 0; component < 3; ++component)
        {
            FieldType scalar(mesh, Vec{1.25, -2.5, 4.0}.component(component), "scalar");
            const auto reference = TransportFactory<decltype(scalar)>::assemble(
                typename TransportFactory<decltype(scalar)>::ScalarTerms{
                    .history = {.old_values = scalar, .time_step = 0.25},
                    .face_fluxes = flux,
                    .coefficients = {.diffusivity = diffusion},
                    .source = [&](LO cell) { return source_value(cell).component(component); }},
                {.condition =
                        [&](int batch, size_t)
                    {
                        const auto type = boundary_type(batch);
                        return BoundaryCondition{
                            type == BoundaryConditionType::NoSlip ? BoundaryConditionType::Dirichlet : type,
                            gradient.component(component)};
                    },
                    .value = [&](int batch, size_t face) { return boundary_value(batch, face).component(component); }},
                {}, {});
            for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
            {
                const auto cell = static_cast<LO>(row);
                EXPECT_DOUBLE_EQ(vector.rhs->getData(component)[row], reference.rhs->getData()[row]);
                for (size_t column = 0; column < mesh->num_local_cells(); ++column)
                    EXPECT_DOUBLE_EQ(local_matrix_entry(*vector.matrix, cell, static_cast<LO>(column)),
                        local_matrix_entry(*reference.matrix, cell, static_cast<LO>(column)));
            }
        }
    }
    // Independent transient/source oracle: the shared matrix must not be
    // accumulated once for each RHS component.
    flux.data().putScalar(0.0);
    const auto transient = TransportFactory<decltype(velocity)>::assemble(
        typename TransportFactory<decltype(velocity)>::VectorTerms{
            .history = {.old_values = velocity, .time_step = 0.25},
            .face_fluxes = flux,
            .coefficients = {.diffusivity = 0.0},
            .source = source_value},
        typename TransportFactory<decltype(velocity)>::BasicVectorBoundary{
            .condition = [](int, size_t)
            { return SimpleFluid::VectorBoundaryCondition{SimpleFluid::BoundaryConditionType::Dirichlet, {}}; },
            .value = [](int, size_t) { return Vec{}; }},
        {}, {});
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        const auto volume = mesh->cell_volume(cell);
        EXPECT_DOUBLE_EQ(local_matrix_entry(*transient.matrix, cell, cell), volume / 0.25);
        for (size_t component = 0; component < 3; ++component)
            EXPECT_DOUBLE_EQ(
                transient.rhs->getData(component)[row], volume / 0.25 * velocity.value(cell).component(component) +
                                                            volume * source_value(cell).component(component));
    }
}

/** One conservative unit-cell balance fixes the outflow coefficient independently. */
TEST(BasicTransportAssemblyTest, NativeBoundaryOutflowContributesOnceAndRetainsCallbacks)
{
    using namespace SimpleFluid;
    using Factory = FVM::TransportSystemFactory<Pack, NativeMesh>;
    using LO = Pack::local_ordinal_type;
    using Vec = vec3<Pack::scalar_type>;
    auto mesh = std::make_shared<NativeMesh>(std::make_shared<NativeMesh::Cartesian>(
        Vec3D<ArrReal>{{{0., 1.}, {0., 1.}, {0., 1.}}}),
        NativeMesh::DistributionOptions{.allow_empty_partitions = true});
    SimpleFluid::VectorCellFieldStored<Pack, NativeMesh> old(mesh, Vec{1., -2., 3.}, "old");
    ScalarFaceFieldStored<Pack, NativeMesh> flux(mesh, 0., "flux");
    for (const auto face : flux.owned_face_ids())
    {
        const auto center = mesh->face_centroid(face);
        const auto normal = mesh->face_normal(face);
        if (normal.x < -.5 && center.x == 0.) flux.set_value(face, -.25);
        if (normal.x > .5 && center.x == 1.) flux.set_value(face, .75);
    }
    old.sync_ghosts();
    flux.sync_ghosts();
    Teuchos::RCP<Pack::matrix_type> cached;
    Teuchos::RCP<Pack::multi_vector_type> previous_rhs;
    for (int generation = 0; generation < 2; ++generation)
    {
        std::vector<LO> source_calls;
        std::vector<std::pair<int, size_t>> value_calls, condition_calls;
        bool awaiting_condition = false;
        const auto system = Factory::assemble(Factory::VectorTerms{
                .history = {old, .5}, .face_fluxes = flux, .coefficients = {0.},
                .source = [&](LO cell)
                {
                    source_calls.push_back(cell);
                    return Vec{.2, -.4, .6};
                }},
            Factory::BasicVectorBoundary{
                .condition = [&](int batch, size_t face)
                {
                    EXPECT_TRUE(awaiting_condition);
                    if (!value_calls.empty()) EXPECT_EQ(value_calls.back(), (std::pair{batch, face}));
                    awaiting_condition = false;
                    condition_calls.emplace_back(batch, face);
                    return VectorBoundaryCondition{BoundaryConditionType::Dirichlet, {}};
                },
                .value = [&](int batch, size_t face)
                {
                    EXPECT_FALSE(awaiting_condition);
                    awaiting_condition = true;
                    value_calls.emplace_back(batch, face);
                    return Vec{4., 5., -6.};
                }}, {}, {.cached_matrix = cached});
        EXPECT_FALSE(awaiting_condition);
        ASSERT_EQ(source_calls.size(), mesh->num_owned_cells());
        for (size_t local = 0; local < source_calls.size(); ++local)
            EXPECT_EQ(source_calls[local], static_cast<LO>(local));
        size_t boundary_faces = 0;
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
            for (const auto face : batch.face_lids)
                if (mesh->is_owned_face(face) && mesh->is_boundary_face(face)) ++boundary_faces;
        EXPECT_EQ(value_calls.size(), boundary_faces);
        EXPECT_EQ(value_calls, condition_calls);
        EXPECT_EQ(system.matrix->getGlobalNumRows(), 1U);
        EXPECT_EQ(system.rhs->getNumVectors(), 3U);
        if (!cached.is_null()) EXPECT_EQ(system.matrix.get(), cached.get());
        if (!previous_rhs.is_null()) EXPECT_NE(system.rhs.get(), previous_rhs.get());
        // V/dt=2; the only outflow adds .75 once. Source and inflow enter each RHS component.
        const std::array<double, 3> expected_rhs{3.2, -3.15, 5.1};
        for (size_t local = 0; local < mesh->num_owned_cells(); ++local)
        {
            const auto row = static_cast<LO>(local);
            const auto column = system.matrix->getColMap()->getLocalElement(0);
            ASSERT_GE(column, 0);
            EXPECT_DOUBLE_EQ(local_matrix_entry(*system.matrix, row, column), 2.75);
            for (size_t component = 0; component < expected_rhs.size(); ++component)
                EXPECT_NEAR(system.rhs->getData(component)[local], expected_rhs[component], 1e-14);
        }
        cached = system.matrix;
        previous_rhs = system.rhs;
    }
}

/** Two owned global balances exercise the imported partition-face flux on two ranks. */
TEST(BasicTransportAssemblyTest, NativePartitionFluxMatchesIndependentTwoCellBalance)
{
    using namespace SimpleFluid;
    using Factory = FVM::TransportSystemFactory<Pack, NativeMesh>;
    using LO = Pack::local_ordinal_type;
    using Vec = vec3<Pack::scalar_type>;
    auto mesh = std::make_shared<NativeMesh>(std::make_shared<NativeMesh::Cartesian>(
        Vec3D<ArrReal>{{{0., 1., 2.}, {0., 1.}, {0., 1.}}}));
    SimpleFluid::VectorCellFieldStored<Pack, NativeMesh> old(mesh, "old");
    for (size_t local = 0; local < mesh->num_owned_cells(); ++local)
    {
        const auto cell = static_cast<LO>(local);
        const auto gid = mesh->owned_cell_map()->getGlobalElement(cell);
        ASSERT_TRUE(gid == 0 || gid == 1);
        old.set_value(cell, gid == 0 ? Vec{1., 2., 3.} : Vec{4., 5., 6.});
    }
    old.sync_ghosts();
    ScalarFaceFieldStored<Pack, NativeMesh> flux(mesh, 0., "flux");
    for (const auto face : flux.owned_face_ids())
        flux.set_value(face, .5 * mesh->face_normal(face).x * mesh->face_area(face));
    flux.sync_ghosts();
    const auto system = Factory::assemble(Factory::VectorTerms{
            .history = {old, 1.}, .face_fluxes = flux, .coefficients = {0.},
            .source = [](LO) { return Vec{}; }},
        Factory::BasicVectorBoundary{
            .condition = [](int, size_t) { return VectorBoundaryCondition{BoundaryConditionType::Dirichlet, {}}; },
            .value = [](int, size_t) { return Vec{2., -1., 4.}; }}, {}, {});
    const std::array<std::array<double, 2>, 2> expected_matrix{{{{1.5, 0.}}, {{-.5, 1.5}}}};
    const std::array<std::array<double, 3>, 2> expected_rhs{{{{2., 1.5, 5.}}, {{4., 5., 6.}}}};
    int local_remote_faces = 0;
    for (size_t local = 0; local < mesh->num_owned_cells(); ++local)
    {
        const auto row = static_cast<LO>(local);
        const auto gid = static_cast<size_t>(mesh->owned_cell_map()->getGlobalElement(row));
        ASSERT_LT(gid, expected_matrix.size());
        for (size_t other = 0; other < expected_matrix[gid].size(); ++other)
        {
            const auto column = system.matrix->getColMap()->getLocalElement(
                static_cast<Pack::global_ordinal_type>(other));
            ASSERT_GE(column, 0);
            EXPECT_DOUBLE_EQ(local_matrix_entry(*system.matrix, row, column), expected_matrix[gid][other]);
        }
        for (size_t component = 0; component < expected_rhs[gid].size(); ++component)
            EXPECT_DOUBLE_EQ(system.rhs->getData(component)[local], expected_rhs[gid][component]);
        for (const auto face : mesh->faces(row))
            if (!flux.is_owned_face(face)) ++local_remote_faces;
    }
    const auto comm = mesh->owned_cell_map()->getComm();
    int remote_faces = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_remote_faces, &remote_faces);
    if (comm->getSize() == 2) EXPECT_GT(remote_faces, 0);
}


/** A rejected local callback or boundary policy cannot mutate the accepted cached operator. */
TEST(BasicTransportAssemblyTest, NativeFailuresAreCollectiveAndPreserveCachedOperator)
{
    using namespace SimpleFluid;
    using Factory = FVM::TransportSystemFactory<Pack, NativeMesh>;
    using LO = Pack::local_ordinal_type;
    using Vec = vec3<Pack::scalar_type>;
    auto mesh = std::make_shared<NativeMesh>(std::make_shared<NativeMesh::Cartesian>(
        Vec3D<ArrReal>{{{0., 1.}, {0., 1.}, {0., 1.}}}),
        NativeMesh::DistributionOptions{.allow_empty_partitions = true});
    const auto comm = mesh->owned_cell_map()->getComm();
    const int local_owner = mesh->num_owned_cells() ? comm->getRank() : -1;
    int owner_rank = -1;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_owner, &owner_rank);
    ASSERT_EQ(owner_rank, 0);
    SimpleFluid::VectorCellFieldStored<Pack, NativeMesh> old(mesh, Vec{1., -2., 3.}, "failure_old");
    ScalarFaceFieldStored<Pack, NativeMesh> flux(mesh, 0., "failure_flux");
    old.sync_ghosts();
    flux.sync_ghosts();
    enum class Fault { None, Source, BoundaryValue, BoundaryCondition, Robin, BoundaryOverflow };
    Teuchos::RCP<Pack::matrix_type> cached;
    const auto assemble = [&](Fault fault)
    {
        const bool inject = comm->getRank() == owner_rank;
        return Factory::assemble(Factory::VectorTerms{
                .history = {old, .5}, .face_fluxes = flux, .coefficients = {.5},
                .source = [&](LO)
                {
                    if (inject && fault == Fault::Source) throw std::runtime_error("Injected source failure.");
                    return Vec{.2, -.4, .6};
                }},
            Factory::BasicVectorBoundary{
                .condition = [&](int, size_t)
                {
                    if (inject && fault == Fault::BoundaryCondition)
                        throw std::runtime_error("Injected boundary condition failure.");
                    return VectorBoundaryCondition{
                        inject && fault == Fault::Robin
                            ? BoundaryConditionType::Robin : BoundaryConditionType::Dirichlet,
                        {}};
                },
                .value = [&](int, size_t)
                {
                    if (inject && fault == Fault::BoundaryValue)
                        throw std::runtime_error("Injected boundary value failure.");
                    if (inject && fault == Fault::BoundaryOverflow)
                        return Vec{std::numeric_limits<double>::max(), 0., 0.};
                    return Vec{4., 5., -6.};
                }}, {}, {.cached_matrix = cached});
    };
    const auto accepted = assemble(Fault::None);
    cached = accepted.matrix;
    const auto* const accepted_matrix = accepted.matrix.get();
    const auto snapshot_matrix = [&]
    {
        std::vector<std::vector<std::pair<LO, double>>> rows;
        for (size_t local = 0; local < cached->getLocalNumRows(); ++local)
        {
            Pack::matrix_type::local_inds_host_view_type columns;
            Pack::matrix_type::values_host_view_type values;
            cached->getLocalRowView(static_cast<LO>(local), columns, values);
            auto& row = rows.emplace_back();
            for (size_t entry = 0; entry < columns.extent(0); ++entry)
                row.emplace_back(columns[entry], values[entry]);
        }
        return rows;
    };
    const auto matrix_snapshot = snapshot_matrix();
    const auto snapshot_rhs = [](const Pack::multi_vector_type& rhs)
    {
        std::array<std::vector<double>, 3> rows;
        for (size_t component = 0; component < rows.size(); ++component)
        {
            const auto values = rhs.getData(component);
            for (size_t row = 0; row < rhs.getLocalLength(); ++row)
                rows[component].push_back(values[row]);
        }
        return rows;
    };
    const auto rhs_snapshot = snapshot_rhs(*accepted.rhs);
    for (const auto fault : {Fault::Source, Fault::BoundaryValue, Fault::BoundaryCondition,
             Fault::Robin, Fault::BoundaryOverflow})
    {
        SCOPED_TRACE(static_cast<int>(fault));
        int caught = 0;
        try
        {
            (void) assemble(fault);
        }
        catch (const std::exception&)
        {
            caught = 1;
        }
        int total_caught = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &caught, &total_caught);
        EXPECT_EQ(total_caught, comm->getSize());
        EXPECT_EQ(cached.get(), accepted_matrix);
        EXPECT_TRUE(cached->isFillComplete());
        if (cached->isFillComplete()) EXPECT_EQ(snapshot_matrix(), matrix_snapshot);
        EXPECT_EQ(snapshot_rhs(*accepted.rhs), rhs_snapshot);
        const auto valid = assemble(Fault::None);
        EXPECT_EQ(valid.matrix.get(), accepted_matrix);
        EXPECT_TRUE(valid.matrix->isFillComplete());
        EXPECT_EQ(snapshot_matrix(), matrix_snapshot);
        EXPECT_NE(valid.rhs.get(), accepted.rhs.get());
        EXPECT_EQ(snapshot_rhs(*valid.rhs), rhs_snapshot);
    }
}

} // namespace
