/** @file testConservativeVoFTransport.cc
 * @brief Direct passive-tracer bounds and local momentum-ledger checks before projection.
 */
#include "FVM/AdaptiveVoF.hh"
#include "geometry/PLIC/SweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "solvers/GeneralVoFSolver.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>

namespace
{
using namespace SimpleFluid;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(ConservativeVoFTransportTest, FiniteMomentumReusesThreeRhsAndInvalidatesChangedFlux)
{
    Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto map = mesh->owned_cell_map();
    const auto n = mesh->num_owned_cells();
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transport = detail::ConservativeVoFTransport;
    Transport transport(mesh);
    std::vector<long double> capacity(n), old_mass(n), outgoing(n), incoming_mass(n), source(n), old(n);
    std::vector<Transport::FiniteTransfer> routes;
    std::vector<Transport::FiniteFaceContribution> faces;
    const auto ledger = [&](long double scale)
    {
        routes.clear();
        faces.clear();
        std::fill(outgoing.begin(), outgoing.end(), 0);
        std::fill(incoming_mass.begin(), incoming_mass.end(), 0);
        std::vector<std::map<GO, long double>> rows(n);
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t f = 0; f < mesh->num_faces(); ++f)
        {
            const auto face = LO(f);
            // A varying positive x flux gives a nonzero storage change. The
            // old/new storage contrast remains close to the 1000:1 density jump.
            const auto amount = scale * mesh->face_normal(face).x * (1 + .25L * mesh->face_centroid(face).x);
            if (amount == 0)
                continue;
            const auto owner = mesh->cell_global_id(mesh->owner_cell(face));
            const auto neighbor = mesh->cell_global_id(mesh->neighbor_cell(face));
            const GO donor = amount > 0 ? owner : neighbor, recipient = amount > 0 ? neighbor : owner;
            if (f < mesh->num_owned_faces())
                faces.push_back({face, donor, amount, 0});
            const auto donor_lid = map->getLocalElement(donor), recipient_lid = map->getLocalElement(recipient);
            if (donor_lid >= 0)
                outgoing[donor_lid] += std::abs(amount);
            if (recipient_lid >= 0)
            {
                incoming_mass[recipient_lid] += std::abs(amount);
                rows[recipient_lid][donor] += std::abs(amount);
            }
        }
        for (std::size_t c = 0; c < n; ++c)
            for (const auto& [donor, amount] : rows[c])
                routes.push_back({donor, map->getGlobalElement(LO(c)), amount, 0});
    };
    ledger(.125L);
    for (std::size_t c = 0; c < n; ++c)
    {
        old_mass[c] = mesh->cell_centroid(LO(c)).x < 2 ? 1000 : 1;
        capacity[c] = old_mass[c] - outgoing[c] + incoming_mass[c];
        ASSERT_GT(capacity[c], 0);
        // Independent row-sum identity for the physical conservative equation.
        EXPECT_EQ(capacity[c] + outgoing[c] - incoming_mass[c], old_mass[c]);
    }
    const auto check_constant = [&](double velocity)
    {
        for (std::size_t c = 0; c < n; ++c)
            old[c] = old_mass[c] * velocity;
        const auto result = transport.solve(old, source, capacity, outgoing, routes, faces, false, 3000, 1e-12);
        EXPECT_LE(result.physical_relative_residual, 1e-12);
        for (std::size_t c = 0; c < n; ++c)
        {
            EXPECT_NEAR(result.cell_trace[c], velocity, 1e-11);
            EXPECT_NEAR(result.amount[c], static_cast<double>(capacity[c] * velocity), 1e-9);
        }
    };
    for (double velocity : {.5, -.75, 1.25})
        check_constant(velocity);
    EXPECT_EQ(transport.preparation_statistics().finite_builds, 1);
    EXPECT_EQ(transport.preparation_statistics().finite_hits, 2);

    // Reusing admitted coefficients must still admit every new RHS collectively,
    // including calls involving empty owners. Rejection does not count as a hit.
    if (n)
        old[0] = std::numeric_limits<long double>::quiet_NaN();
    EXPECT_THROW(
        (void) transport.solve(old, source, capacity, outgoing, routes, faces, false, 3000, 1e-12), std::runtime_error);
    EXPECT_EQ(transport.preparation_statistics().finite_builds, 1);
    EXPECT_EQ(transport.preparation_statistics().finite_hits, 2);

    // Keep capacities fixed and change the physical flux coefficients. A new
    // old mass follows from the independent row sum; constant velocity remains
    // the analytic solution, so stale matrix/raw-row reuse cannot pass silently.
    ledger(.25L);
    for (std::size_t c = 0; c < n; ++c)
        old_mass[c] = capacity[c] + outgoing[c] - incoming_mass[c];
    check_constant(.375);
    EXPECT_EQ(transport.preparation_statistics().finite_builds, 2);
    EXPECT_EQ(transport.preparation_statistics().finite_hits, 2);
    check_constant(-.625);
    EXPECT_EQ(transport.preparation_statistics().finite_builds, 2);
    EXPECT_EQ(transport.preparation_statistics().finite_hits, 3);
    EXPECT_THROW((void) transport.solve(old, source, capacity, outgoing, routes, faces, false, 3000, 1e-12, 1),
        std::runtime_error);
    EXPECT_EQ(transport.preparation_statistics().finite_builds, 2);
    EXPECT_EQ(transport.preparation_statistics().finite_hits, 3);
    check_constant(.125);
    EXPECT_EQ(transport.preparation_statistics().finite_builds, 3);
    EXPECT_EQ(transport.preparation_statistics().finite_hits, 3);
}

TEST(ConservativeVoFTransportTest, OrdinaryWideMomentumPreservesPhysicalStorageWithoutIntervalAdmission)
{
    Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    std::vector<long double> old(n, 1), source(n), capacity(n, 2), outgoing(n), uncertainty(n, 4);
    const std::vector<detail::ConservativeVoFTransport::FiniteTransfer> routes;
    const std::vector<detail::ConservativeVoFTransport::FiniteFaceContribution> faces;
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() > 1)
    {
        detail::ConservativeVoFTransport mismatched(mesh);
        const auto mode = comm->getRank() + 1 == comm->getSize() ?
            ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
        EXPECT_THROW((void) mismatched.solve(old, source, capacity, outgoing, routes, faces, true, 3000, 1e-12,
                         64 * 1024 * 1024, {}, {}, {}, {}, mode), std::runtime_error);
    }
    detail::ConservativeVoFTransport transport(mesh);
    const auto carried = transport.solve(old, source, capacity, outgoing, routes, faces, true, 3000, 1e-12,
        64 * 1024 * 1024, uncertainty, {}, uncertainty, {});
    EXPECT_LE(carried.physical_relative_residual, 1e-12);
    double local_norm[2]{}, global_norm[2]{};
    for (std::size_t c = 0; c < n; ++c)
    {
        // This diagonal analytic case admits a few rounding steps in distributed
        // Krylov normalization. Google Test's four-ULP comparison checks that
        // roundoff without imposing bitwise analytic values on an iterative solve.
        EXPECT_DOUBLE_EQ(carried.cell_trace[c], .5);
        EXPECT_DOUBLE_EQ(carried.amount[c], 1);
        const auto stored = capacity[c] * carried.cell_trace[c];
        EXPECT_EQ(carried.amount[c], static_cast<double>(stored));
        const auto residual = stored - old[c] - source[c];
        local_norm[0] += static_cast<double>(residual * residual);
        local_norm[1] += static_cast<double>((old[c] + source[c]) * (old[c] + source[c]));
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_norm, global_norm);
    ASSERT_GT(global_norm[1], 0);
    const double physical_residual = std::sqrt(global_norm[0] / global_norm[1]);
    EXPECT_LE(physical_residual, 1e-12);
    EXPECT_DOUBLE_EQ(carried.physical_relative_residual, physical_residual);
    EXPECT_THROW((void) transport.solve(old, source, capacity, outgoing, routes, faces, true, 3000, 1e-12,
                     64 * 1024 * 1024, {}, {}, {}, {}, ArithmeticMode::BoundedCertification),
        std::runtime_error);
    detail::ConservativeVoFTransport bounded(mesh);
    EXPECT_THROW((void) bounded.solve(old, source, capacity, outgoing, routes, faces, true, 3000, 1e-12,
                     64 * 1024 * 1024, uncertainty, {}, uncertainty, {}, ArithmeticMode::BoundedCertification),
        std::runtime_error);
    for (auto& value : capacity)
        value = -1;
    EXPECT_THROW(
        (void) transport.solve(old, source, capacity, outgoing, routes, faces, true, 3000), std::runtime_error);
    for (auto& value : capacity)
        value = 2;
    EXPECT_NO_THROW((void) transport.solve(old, source, capacity, outgoing, routes, faces, true, 3000));
}

TEST(ConservativeVoFTransportTest, SplitCounterexampleHasBoundedConservativeImplicitTracesForEveryFirstAxis)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 4, 200};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    detail::ConservativeVoFTransport tracer(mesh);
    for (std::size_t first_axis = 0; first_axis < 3; ++first_axis)
        for (int fixture = 0; fixture < 4; ++fixture)
        {
            SCOPED_TRACE(::testing::Message() << first_axis << ',' << fixture);
            std::vector<double> f(n), rhs(n), velocity(mesh->num_faces()), capacity(n), zero(n);
            for (std::size_t c = 0; c < n; ++c)
            {
                const auto x = mesh->cell_centroid(static_cast<LO>(c));
                f[c] = x.x < 1 ? 0 : 1;
                if (fixture == 1)
                    f[c] = x.x < 1 ? 1e-12 : 1 - 1e-12;
                if (fixture == 2)
                    f[c] = 0;
                if (fixture == 3)
                    f[c] = 1;
                const double passive = x.x >= 1 && x.x < 2 && x.y >= 1 && x.y < 2 ? 1 : 0;
                rhs[c] = mesh->cell_volume(static_cast<LO>(c)) * (1000 * (1 - f[c]) + f[c]) * passive;
            }
            for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
            {
                const auto fid = static_cast<LO>(id);
                const auto normal = mesh->face_normal(fid);
                const auto owner = mesh->owner_cell(fid);
                const auto x = mesh->cell_centroid(owner);
                velocity[id] = normal.x + normal.y + normal.z * (x.x >= 1 && x.x < 2 && x.y >= 1 && x.y < 2 ? 1 : 0);
            }
            const auto phase = geometry.advect(f, velocity, 0.1, first_axis);
            std::vector<double> mass_flux(velocity.size());
            for (std::size_t c = 0; c < n; ++c)
                capacity[c] =
                    mesh->cell_volume(static_cast<LO>(c)) * (1000 * (1 - phase.fraction[c]) + phase.fraction[c]);
            for (std::size_t id = 0; id < mass_flux.size(); ++id)
                mass_flux[id] = 1000 * phase.liquid_volume[id] + phase.gas_volume[id];
            const auto transported = tracer.solve(rhs, zero, capacity, mass_flux, false, 3000, 1e-14);
            for (std::size_t c = 0; c < n; ++c)
            {
                const double u = transported.amount[c] / capacity[c];
                EXPECT_GE(u, -2e-11);
                EXPECT_LE(u, 1 + 2e-11);
                double ledger = transported.amount[c] - rhs[c];
                for (const auto face : mesh->faces(static_cast<LO>(c)))
                    ledger += (mesh->owner_cell(face) == static_cast<LO>(c) ? 1 : -1) * transported.flux[face];
                EXPECT_NEAR(ledger, 0, 2e-11 * std::max(1.0, rhs[c]));
            }
            double before = 0, after = 0;
            for (std::size_t c = 0; c < n; ++c)
            {
                before += rhs[c];
                after += transported.amount[c];
            }
            double local[2] = {before, after}, global[2]{};
            Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local, global);
            EXPECT_NEAR(global[1], global[0], 2e-11 * std::max(1.0, global[0]));
        }
}
TEST(ConservativeVoFTransportTest, ChangedDirectionAndStorageRefreshTheReusableOperator)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    detail::ConservativeVoFTransport tracer(mesh);
    std::vector<double> rhs(n), zero(n), capacity(n), flux(mesh->num_faces());
    for (std::size_t c = 0; c < n; ++c)
        rhs[c] = mesh->cell_centroid(static_cast<LO>(c)).x < 1 ? 1 : 0;
    // Alternating RHS groups reuse A; reversal, zero flow, then new capacity
    // force distinct numerical operators on the same graph, including replay.
    for (int stage : {1, 1, -1, 0, 2, -2, 1})
    {
        const double mass = std::abs(stage) == 2 ? 3 : 2;
        std::fill(capacity.begin(), capacity.end(), mass);
        for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
            flux[face] = stage == 0 ? 0 : (stage > 0 ? 0.25 : -0.25) * mesh->face_normal(static_cast<LO>(face)).x;
        flux = geometry.exchange_faces(flux);
        const auto result = tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-14);
        const double r = 0.25 / (mass + 0.25);
        const double x0 = 1 / ((mass + 0.25) * (1 - std::pow(r, 4)));
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto x = static_cast<unsigned>(mesh->cell_centroid(static_cast<LO>(c)).x);
            const unsigned distance = stage > 0 ? x : (4 - x) % 4;
            const double expected = stage == 0 ? rhs[c] : mass * x0 * std::pow(r, distance);
            EXPECT_NEAR(result.amount[c], expected, 2e-13);
        }
    }
    // An unrelated failed zero-storage refresh must not make the old key
    // eligible for reuse while the matrix contains failed candidate values.
    std::fill(capacity.begin(), capacity.end(), 0);
    for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
    {
        const auto id = static_cast<LO>(face);
        const auto owner = mesh->owner_cell(id);
        const double x = mesh->cell_centroid(owner).x + mesh->face_center_vector(id, owner).x;
        flux[face] = std::abs(x - 1) < 1e-12 ? 0.25 * mesh->face_normal(id).x : 0;
    }
    flux = geometry.exchange_faces(flux);
    EXPECT_THROW((void) tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-14), std::exception);
    std::fill(capacity.begin(), capacity.end(), 2);
    for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
        flux[face] = 0.25 * mesh->face_normal(static_cast<LO>(face)).x;
    flux = geometry.exchange_faces(flux);
    EXPECT_NO_THROW((void) tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-14));
}

TEST(ConservativeVoFTransportTest, EquilibratedMicroscopicCarrierPreservesConstantConcentration)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    detail::ConservativeVoFTransport tracer(mesh);
    std::vector<double> zero(n), capacity(n), rhs(n), flux(mesh->num_faces());
    // The reciprocal of 1e-310 overflows double; direct coefficient division
    // still gives an ordinary finite M-matrix and concentration seven.
    for (double storage : {1.0, 1e-30, 1e-310, 1.0})
    {
        std::fill(capacity.begin(), capacity.end(), storage);
        std::fill(rhs.begin(), rhs.end(), 7 * storage);
        for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
            flux[face] = 0.25 * storage * mesh->face_normal(static_cast<LO>(face)).x;
        flux = geometry.exchange_faces(flux);
        const auto result = tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-12, true);
        for (std::size_t c = 0; c < n; ++c)
        {
            EXPECT_NEAR(result.amount[c] / storage, 7, 3e-12);
            EXPECT_GE(result.amount[c], 0);
        }
        // Changing the scaling policy on the same ordinary physical matrix
        // must refresh the numerical cache as well as its acceptance checks.
        if (storage == 1.0)
        {
            const auto unscaled = tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-12, false);
            for (std::size_t c = 0; c < n; ++c)
                EXPECT_NEAR(unscaled.amount[c], result.amount[c], 3e-12);
        }
    }
}

TEST(ConservativeVoFTransportTest, EquilibratedStationaryTracerHasAnOverflowSafePhysicalNormRatio)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 4, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    detail::ConservativeVoFTransport tracer(mesh);
    // First reproduce overflow of the squared normalized norm. Then exercise
    // overflow of max(D)*max(bhat), with representable input/output and target.
    for (const auto scales : {std::array<double, 3>{1e-200, 1.0, 1.0},
             std::array<double, 3>{std::ldexp(1.0, -100), std::ldexp(1.0, 900), std::ldexp(1.0, 150)}})
    {
        SCOPED_TRACE(::testing::Message() << scales[0] << ',' << scales[1] << ',' << scales[2]);
        std::vector<double> zero(n), capacity(n), rhs(n), flux(mesh->num_faces());
        for (std::size_t c = 0; c < n; ++c)
        {
            const bool stationary = mesh->cell_centroid(static_cast<LO>(c)).x < 1.0;
            capacity[c] = stationary ? scales[0] : scales[1];
            rhs[c] = stationary ? capacity[c] * scales[2] : 0.0;
        }
        for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
        {
            const auto id = static_cast<LO>(face);
            const auto owner = mesh->owner_cell(id);
            if (mesh->cell_centroid(owner).x >= 1.0)
                flux[face] = 0.125 * scales[1] * mesh->face_normal(id).y;
        }
        flux = geometry.exchange_faces(flux);
        const auto result = tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-12, true);
        for (std::size_t c = 0; c < n; ++c)
        {
            EXPECT_TRUE(std::isfinite(result.amount[c]));
            EXPECT_GE(result.amount[c], 0.0);
            if (rhs[c] > 0.0)
                EXPECT_NEAR(result.amount[c] / rhs[c], 1.0, 2e-12);
            else
                EXPECT_DOUBLE_EQ(result.amount[c], 0.0);
        }
        for (const auto amount : result.flux)
            EXPECT_DOUBLE_EQ(amount, 0.0);
    }
}

TEST(ConservativeVoFTransportTest, PhysicalDefectCorrectionHonorsTheSharedIterationBudgetAndRecovers)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 4, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    double local_empty = n == 0 ? 1.0 : 0.0, empty_ranks = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty_ranks);
    if (static_cast<std::size_t>(Tpetra::getDefaultComm()->getSize()) > mesh->owned_cell_map()->getGlobalNumElements())
        EXPECT_GT(empty_ranks, 0);
    detail::ConservativeVoFTransport tracer(mesh);
    constexpr double dt = 1.0 / 7.0, heavy_amount = 0x1p-24, weak_capacity = 0x1p-80, weak_amount = 0x1p-60;
    constexpr std::array<double, 4> weights{0.7, 0.9, 0.6, 0.8};
    std::vector<double> zero(n), capacity(n), rhs(n), flux(mesh->num_faces());
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto x = mesh->cell_centroid(static_cast<LO>(c));
        const bool weak = x.x < 1.0;
        capacity[c] = weak ? weak_capacity : 1.0;
        rhs[c] = weak ? weak_amount : heavy_amount * weights[static_cast<std::size_t>(x.y)];
    }
    for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
    {
        const auto id = static_cast<LO>(face);
        if (mesh->cell_centroid(mesh->owner_cell(id)).x >= 1.0)
            flux[face] = dt * mesh->face_normal(id).y;
    }
    flux = geometry.exchange_faces(flux);
    const auto original_rhs = rhs;
    // The weak identity block dominates the scaled RHS by about 14 orders of
    // magnitude. A first scaled solve may converge while the physical defect
    // in the flowing block is still large; one iteration cannot resolve both.
    try
    {
        (void) tracer.solve(rhs, zero, capacity, flux, true, 1, 1e-12, true);
        ADD_FAILURE() << "An exhausted physical residual budget must reject the transport.";
    }
    catch (const std::exception& error)
    {
        EXPECT_NE(std::string(error.what()).find("original physical residual"), std::string::npos) << error.what();
    }
    EXPECT_EQ(rhs, original_rhs);
    for (std::size_t shift : {0u, 1u})
    {
        SCOPED_TRACE(::testing::Message() << "RHS shift=" << shift);
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto x = mesh->cell_centroid(static_cast<LO>(c));
            rhs[c] = x.x < 1.0 ? weak_amount
                               : heavy_amount * weights[(static_cast<std::size_t>(x.y) + shift) % weights.size()];
        }
        const auto result = tracer.solve(rhs, zero, capacity, flux, true, 20, 1e-12, true);
        EXPECT_GE(result.physical_refinements, 1);
        EXPECT_GT(result.iterations, 1);
        EXPECT_LE(result.iterations, 20);
        EXPECT_TRUE(std::isfinite(result.physical_relative_residual));
        EXPECT_LE(result.physical_relative_residual, 1e-12);
        double local_scale = 0.0, scale = 0.0;
        for (const auto amount : rhs)
            local_scale = std::max(local_scale, std::abs(amount));
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local_scale, &scale);
        double local[4]{};
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto lid = static_cast<LO>(c);
            const auto x = mesh->cell_centroid(lid);
            EXPECT_GE(result.amount[c], 0.0);
            EXPECT_TRUE(std::isfinite(result.amount[c]));
            if (x.x < 1.0)
                EXPECT_NEAR(result.amount[c] / weak_amount, 1.0, 2e-12);
            else
            {
                const auto j = static_cast<std::size_t>(x.y);
                double expected = 0.0, factor = 1.0;
                for (std::size_t distance = 0; distance < weights.size(); ++distance)
                {
                    expected += factor * weights[(j + shift + weights.size() - distance) % weights.size()];
                    factor *= 0.125;
                }
                expected *= heavy_amount / ((1.0 + dt) * (1.0 - factor));
                EXPECT_NEAR(result.amount[c], expected, 3e-12 * heavy_amount);
            }
            double residual = result.amount[c] - rhs[c];
            for (const auto face : mesh->faces(lid))
                residual += (mesh->owner_cell(face) == lid ? 1.0 : -1.0) * result.flux[face];
            local[0] += std::pow(residual / scale, 2);
            local[1] += std::pow(rhs[c] / scale, 2);
            local[2] += rhs[c];
            local[3] += result.amount[c];
        }
        double global[4]{};
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 4, local, global);
        const double physical_residual = std::sqrt(global[0] / global[1]);
        EXPECT_LE(physical_residual, 1e-12);
        EXPECT_NEAR(global[3] / global[2], 1.0, 2e-12);
        if (Tpetra::getDefaultComm()->getRank() == 0)
            std::cout << std::setprecision(17) << "physical_defect_correction,rhs_shift=" << shift
                      << ",refinements=" << result.physical_refinements << ",total_iterations=" << result.iterations
                      << ",physical_relative_residual=" << physical_residual << '\n';
    }
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, ZeroNetCounterflowRetainsImplicitMixingAndPhysicalResidual)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    detail::ConservativeVoFTransport transport(mesh);
    std::vector<double> old(n), capacity(n, 1), source(n), forward(mesh->num_faces()), reverse(mesh->num_faces());
    for (std::size_t c = 0; c < n; ++c)
        old[c] = mesh->cell_centroid(static_cast<LO>(c)).x < 2 ? 1 : 0;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        forward[f] = reverse[f] = std::abs(mesh->face_normal(static_cast<LO>(f)).x) * 0.25;
    forward = geometry.exchange_faces(forward);
    reverse = geometry.exchange_faces(reverse);
    for (double scale : {1.0, 1e-200, 1e-310, 1.0})
    {
        SCOPED_TRACE(scale);
        auto rhs = old, storage = capacity, a = forward, b = reverse;
        for (double& v : rhs)
            v *= scale;
        for (double& v : storage)
            v *= scale;
        for (double& v : a)
            v *= scale;
        for (double& v : b)
            v *= scale;
        const auto result =
            transport.solve_streams(rhs, source, storage, a, b, true, 3000, scale < 1e-300 ? 1e-11 : 1e-13, true);
        EXPECT_LE(result.physical_relative_residual, scale < 1e-300 ? 1e-11 : 1e-13);
        double local[3]{}, global[3]{};
        for (std::size_t c = 0; c < n; ++c)
        {
            const double u = result.amount[c] / storage[c];
            EXPECT_GE(u, 0);
            EXPECT_LE(u, 1 + 1e-12);
            local[0] += rhs[c] / scale;
            local[1] += result.amount[c] / scale;
            local[2] += std::abs(u - old[c]);
        }
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 3, local, global);
        EXPECT_NEAR(global[1], global[0], 1e-10);
        EXPECT_GT(global[2], 0.1); // A net-flux adapter incorrectly leaves this discontinuity unchanged.
        std::fill(rhs.begin(), rhs.end(), 0.37 * scale);
        const auto uniform = transport.solve_streams(rhs, source, storage, a, b, true, 3000, 1e-11, true);
        for (std::size_t c = 0; c < n; ++c)
            EXPECT_NEAR(uniform.amount[c] / storage[c], 0.37, 1e-11);
    }
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, OneWaySubnormalFluxPreservesUnequalDonorTracesExactly)
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    detail::ConservativeVoFTransport transport(mesh);
    std::vector<double> old(n), source(n), capacity(n, 1), forward(mesh->num_faces()), reverse(mesh->num_faces());
    for (std::size_t c = 0; c < n; ++c)
        old[c] = .125 + .03125 * mesh->cell_centroid(LO(c)).x;
    for (int direction : {1, -1})
    {
        std::fill(forward.begin(), forward.end(), 0);
        std::fill(reverse.begin(), reverse.end(), 0);
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            if (!mesh->is_geometry_exterior_face(LO(f)))
                (direction > 0 ? forward : reverse)[f] = std::numeric_limits<double>::denorm_min();
        forward = geometry.exchange_faces(forward);
        reverse = geometry.exchange_faces(reverse);
        const auto result = transport.solve_streams(old, source, capacity, forward, reverse, true, 3000, 1e-12, true);
        const auto trace = geometry.exchange_cells(result.amount);
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            if (!mesh->is_geometry_exterior_face(LO(f)))
            {
                const auto donor = direction > 0 ? mesh->owner_cell(LO(f)) : mesh->neighbor_cell(LO(f));
                EXPECT_EQ(result.trace[f], trace[donor]);
                // The amount legitimately rounds to zero; the donor trace must
                // not be recovered by division through that rounded product.
                EXPECT_EQ(result.flux[f], 0);
            }
    }
}

TEST(ConservativeVoFTransportTest, NonincidentDonorRoutesPreserveRawFaceResidualAndRecoverAfterExhaustion)
{
    struct StatisticsScope
    {
        StatisticsScope()
        {
            GeneralVoFSolver::enable_geometry_statistics(true);
            GeneralVoFSolver::reset_geometry_statistics();
        }
        ~StatisticsScope() { GeneralVoFSolver::enable_geometry_statistics(false); }
    } statistics_scope;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transfer = detail::ConservativeVoFTransport::Transfer;
    using Stream = detail::ConservativeVoFTransport::FaceContribution;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    std::array<GO, 16> local_ids, ids;
    local_ids.fill(-1);
    const auto index = [](const auto& x)
    { return int(std::floor(x.x)) + 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z)); };
    for (std::size_t c = 0; c < n; ++c)
        local_ids[index(mesh->cell_centroid(LO(c)))] = mesh->cell_global_id(LO(c));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 16, local_ids.data(), ids.data());
    detail::ConservativeVoFTransport transport(mesh);
    for (double scale : {1.0, 1e-200, 1e-310, 1.0})
    {
        SCOPED_TRACE(scale);
        std::vector<double> old(n), capacity(n, scale), outgoing(n), source(n);
        std::vector<Transfer> incoming;
        std::vector<Stream> streams;
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto x = mesh->cell_centroid(LO(c));
            const int ix = int(std::floor(x.x));
            old[c] = scale * (ix == 0 ? 1 : ix == 1 ? .5 : 0);
            if (ix == 0)
            {
                capacity[c] = .75 * scale;
                outgoing[c] = .25 * scale;
            }
            if (ix == 2)
            {
                capacity[c] = 1.25 * scale;
                incoming.push_back({ids[index(x) - 2], mesh->cell_global_id(LO(c)), .25 * scale});
            }
        }
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        {
            const auto x = mesh->face_centroid(LO(f));
            const auto normal = mesh->face_normal(LO(f));
            if (std::abs(normal.x) > .5 && (std::abs(x.x - 1) < 1e-12 || std::abs(x.x - 2) < 1e-12))
            {
                const int donor = 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z));
                streams.push_back({LO(f), ids[donor], normal.x * .25 * scale});
            }
        }
        const auto result =
            transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-11);
        EXPECT_LE(result.physical_relative_residual, 1e-11);
        double local[2]{}, global[2]{};
        for (std::size_t c = 0; c < n; ++c)
        {
            const int ix = int(std::floor(mesh->cell_centroid(LO(c)).x));
            const double expected = ix == 0 ? 1 : ix == 1 ? .5 : ix == 2 ? .2 : 0;
            EXPECT_NEAR(result.amount[c] / capacity[c], expected, 1e-11);
            local[0] += old[c] / scale;
            local[1] += result.amount[c] / scale;
        }
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
        EXPECT_NEAR(global[0], global[1], 1e-10);
        if (scale == 1)
        {
            // An inconsistent raw face ledger cannot be accepted merely because
            // the positive sparse route equation solved successfully.
            auto broken = streams;
            for (auto& face : broken)
                if (std::abs(mesh->face_centroid(face.face).x - 2) < 1e-12)
                    face.amount = 0;
            EXPECT_THROW(
                (void) transport.solve_transfers(old, source, capacity, outgoing, incoming, broken, true, 3000, 1e-11),
                std::runtime_error);
            EXPECT_NO_THROW((void) transport.solve_transfers(
                old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-11));
        }
    }
    const auto exact = GeneralVoFSolver::geometry_statistics().exact;
    EXPECT_EQ(exact.measure_values, 0u);
    EXPECT_EQ(exact.measure_arithmetic, 0u);
    EXPECT_EQ(exact.volume_evaluations, 0u);
    EXPECT_EQ(exact.clip_constructions, 0u);
    EXPECT_EQ(exact.intersection_constructions, 0u);
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, SparsePositivePhysicalAmountCannotDisappearBelowPublishedRange)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transfer = detail::ConservativeVoFTransport::Transfer;
    using Stream = detail::ConservativeVoFTransport::FaceContribution;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    std::array<GO, 16> local_ids, ids;
    local_ids.fill(-1);
    const auto index = [](const auto& x)
    { return int(std::floor(x.x)) + 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z)); };
    for (std::size_t c = 0; c < n; ++c)
        local_ids[index(mesh->cell_centroid(LO(c)))] = mesh->cell_global_id(LO(c));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 16, local_ids.data(), ids.data());
    detail::ConservativeVoFTransport transport(mesh);
    for (int quanta : {1, 4})
    {
        const double mass = quanta * std::numeric_limits<double>::denorm_min();
        std::vector<double> old(n), capacity(n, 1), outgoing(n), source(n);
        std::vector<Transfer> incoming;
        std::vector<Stream> streams;
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto x = mesh->cell_centroid(LO(c));
            const int ix = int(std::floor(x.x));
            if (ix == 0)
            {
                old[c] = mass;
                capacity[c] = mass;
                outgoing[c] = mass;
            }
            if (ix == 2)
            {
                capacity[c] = 2 * mass;
                incoming.push_back({ids[index(x) - 2], mesh->cell_global_id(LO(c)), mass});
            }
        }
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        {
            const auto x = mesh->face_centroid(LO(f));
            const auto normal = mesh->face_normal(LO(f));
            if (std::abs(normal.x) > .5 && (std::abs(x.x - 1) < 1e-12 || std::abs(x.x - 2) < 1e-12))
            {
                const int donor = 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z));
                streams.push_back({LO(f), ids[donor], normal.x * mass});
            }
        }
        // The analytic implicit traces are 1/2 at the donor and 1/4 at
        // the recipient. At one quantum, both retained positive amounts
        // are half a quantum and cannot be published without deletion.
        if (quanta == 1)
            EXPECT_THROW(
                (void) transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-12),
                std::runtime_error);
        else
        {
            const auto result =
                transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-12);
            for (std::size_t c = 0; c < n; ++c)
            {
                const int ix = int(std::floor(mesh->cell_centroid(LO(c)).x));
                if (ix == 0 || ix == 2)
                    EXPECT_EQ(result.amount[c], 2 * std::numeric_limits<double>::denorm_min());
            }
        }
    }
}
} // namespace
