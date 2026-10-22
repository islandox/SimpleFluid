/** @file testConservativeVoFTransport.cc
 * @brief Direct passive-tracer bounds and local momentum-ledger checks before projection.
 */
#include "FVM/AdaptiveVoF.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFTransferBinding.hh"
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
#include <string>

namespace
{
using namespace SimpleFluid;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

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
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, NonincidentDonorRoutesPreserveRawFaceResidualAndRecoverAfterExhaustion)
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
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, ExactOppositeGroupsRetainMicroscopicNetBeforeImplicitTrace)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Measure = SimpleFluid::Meshes::ExactGeometryMeasure;
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
    constexpr double tiny = 1e-310;
    const auto forward = Measure::from_value(1), reverse = forward - Measure::from_value(tiny);
    detail::ConservativeVoFTransport transport(mesh);
    std::vector<double> old(n), capacity(n, 1), outgoing(n), source(n);
    std::vector<Transfer> incoming;
    std::vector<Stream> streams;
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto x = mesh->cell_centroid(LO(c));
        const int ix = int(std::floor(x.x));
        old[c] = ix == 0 ? 1 : ix == 1 ? .5 : 0;
        if (ix == 0)
            outgoing[c] = tiny;
        if (ix == 2)
        {
            capacity[c] = 2 * tiny;
            incoming.push_back({ids[index(x) - 2], mesh->cell_global_id(LO(c)), tiny});
        }
    }
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto x = mesh->face_centroid(LO(f));
        const auto normal = mesh->face_normal(LO(f));
        if (std::abs(normal.x) > .5 && (std::abs(x.x - 1) < 1e-12 || std::abs(x.x - 2) < 1e-12))
        {
            const int donor = 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z));
            streams.push_back({LO(f), ids[donor], normal.x, std::make_shared<const Measure>(forward.scaled(normal.x))});
            streams.push_back(
                {LO(f), ids[donor], -normal.x, std::make_shared<const Measure>(reverse.scaled(-normal.x))});
        }
    }
    const auto result =
        transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-12);
    for (std::size_t c = 0; c < n; ++c)
        if (int(std::floor(mesh->cell_centroid(LO(c)).x)) == 2)
            EXPECT_NEAR(result.amount[c] / capacity[c], .5, 1e-12);
    for (const auto& stream : streams)
    {
        const double sign = mesh->face_normal(stream.face).x;
        EXPECT_EQ(result.flux[stream.face], sign * tiny);
    }
    // The displayed doubles still agree, but this exact net is materially
    // different from the certified microscopic routed amount. Its provenance
    // must reject before a global residual can hide the tiny recipient defect.
    auto inconsistent = streams;
    for (std::size_t i = 0; i < inconsistent.size(); i += 2)
    {
        const double sign = mesh->face_normal(inconsistent[i].face).x;
        inconsistent[i].exact_amount =
            std::make_shared<const Measure>((forward + Measure::from_value(1e-20)).scaled(sign));
    }
    EXPECT_THROW(
        (void) transport.solve_transfers(old, source, capacity, outgoing, incoming, inconsistent, true, 3000, 1e-12),
        std::runtime_error);
    auto duplicated = incoming;
    duplicated.insert(duplicated.end(), incoming.begin(), incoming.end());
    EXPECT_THROW(
        (void) transport.solve_transfers(old, source, capacity, outgoing, duplicated, streams, true, 3000, 1e-12),
        std::runtime_error);
    EXPECT_NO_THROW(
        (void) transport.solve_transfers(old, source, capacity, outgoing, incoming, streams, true, 3000, 1e-12));
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

namespace
{
TEST(ConservativeVoFTransportTest, ExactInputNormalizesBeforeNarrowingAndRecoversFromRangeFailure)
{
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using M = SimpleFluid::Meshes::ExactGeometryMeasure;
    using Transport = detail::ConservativeVoFTransport;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    const auto n = mesh->num_owned_cells();
    std::array<GO, 16> local, ids;
    local.fill(-1);
    const auto index = [](const auto& x)
    { return int(std::floor(x.x)) + 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z)); };
    for (std::size_t c = 0; c < n; ++c)
        local[index(mesh->cell_centroid(LO(c)))] = mesh->cell_global_id(LO(c));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 16, local.data(), ids.data());
    Transport transport(mesh);
    const double eta = std::numeric_limits<double>::denorm_min();
    for (int scenario : {0, 1, 0, 2, 3, 4})
    {
        const auto mu = scenario == 4  ? M::from_value(.1)
                        : scenario < 2 ? M::from_value(eta).divided_by(M::from_value(scenario == 1 ? 3 : 4))
                                       : M::from_value((scenario == 2 ? 3 : 24) * eta);
        std::vector<M> old(n), source(n), capacity(n, M::from_value(1)), outgoing(n);
        std::vector<Transport::ExactTransfer> routes;
        std::vector<Transport::ExactFaceContribution> faces;
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto x = mesh->cell_centroid(LO(c));
            const int ix = int(std::floor(x.x));
            if (ix == 0)
            {
                outgoing[c] = mu;
                capacity[c] = scenario < 2 ? M::from_value(1) - mu : M::from_value((scenario == 2 ? 1 : 8) * eta);
                old[c] = scenario < 2 ? M::from_value(8) : capacity[c];
                if (scenario == 4)
                {
                    capacity[c] = M::from_value(.9);
                    old[c] = M::from_value(.7);
                }
            }
            if (ix == 2)
            {
                capacity[c] = scenario < 2 ? M::from_value(1e-315) + mu : M::from_value((scenario == 2 ? 4 : 32) * eta);
                if (scenario == 4)
                    capacity[c] = M::from_value(1.1);
                routes.push_back({ids[index(x) - 2], mesh->cell_global_id(LO(c)), mu});
            }
        }
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        {
            const auto x = mesh->face_centroid(LO(f));
            const auto normal = mesh->face_normal(LO(f));
            if (std::abs(normal.x) > .5 && (std::abs(x.x - 1) < 1e-12 || std::abs(x.x - 2) < 1e-12))
            {
                const int donor = 4 * int(std::floor(x.y)) + 8 * int(std::floor(x.z));
                faces.push_back({LO(f), ids[donor], mu.scaled(normal.x)});
            }
        }
        if (scenario == 2)
        {
            EXPECT_THROW((void) transport.solve_exact(old, source, capacity, outgoing, routes, faces, true, 3000),
                std::runtime_error);
            continue;
        }
        if (scenario == 4)
        {
            EXPECT_THROW(
                (void) transport.solve_exact(old, source, capacity, outgoing, routes, faces, true, 3000, 1e-30),
                std::runtime_error);
            EXPECT_THROW(
                (void) transport.solve_exact(old, source, capacity, outgoing, routes, faces, true, 3000, 1e-12, 1024),
                std::runtime_error);
            const auto comm = mesh->owned_cell_map()->getComm();
            if (comm->getSize() > 1)
            {
                EXPECT_THROW((void) transport.solve_exact(
                                 old, source, capacity, outgoing, routes, faces, comm->getRank() == 0, 3000),
                    std::runtime_error);
                EXPECT_THROW((void) transport.solve_exact(old, source, capacity, outgoing, routes, faces, true,
                                 comm->getRank() == 0 ? 2999 : 3000),
                    std::runtime_error);
                EXPECT_THROW((void) transport.solve_exact(old, source, capacity, outgoing, routes, faces, true, 3000,
                                 comm->getRank() == 0 ? 1e-11 : 1e-12),
                    std::runtime_error);
            }
        }
        const auto result = transport.solve_exact(old, source, capacity, outgoing, routes, faces, true, 3000);
        EXPECT_TRUE(result.exact_input);
        EXPECT_TRUE(result.flux.empty());
        EXPECT_LE(result.physical_relative_residual, 1e-12);
        if (n > 0)
            EXPECT_FALSE(result.inner_conversion_enclosures.empty());
        for (std::size_t c = 0; c < n; ++c)
        {
            const int ix = int(std::floor(mesh->cell_centroid(LO(c)).x));
            if (ix == 2 && scenario < 4)
                EXPECT_EQ(result.amount[c], (scenario == 1 ? 3 : scenario == 3 ? 6 : 2) * eta);
            if (scenario == 4 && ix == 0)
                EXPECT_NEAR(result.cell_trace[c], .7, 1e-12);
            if (scenario == 4 && ix == 2)
                EXPECT_NEAR(result.cell_trace[c], .07 / 1.1, 1e-12);
            if (ix == 0 && scenario == 3)
                EXPECT_EQ(result.amount[c], 2 * eta);
        }
    }
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, ExactStorageDeduplicatesSharedPayloadAndSeparatesDomains)
{
    using Domain = detail::GeneralVoFStorageDomain;
    using M = Meshes::ExactGeometryMeasure;
    const auto value = M::from_value(1e-310).squared();
    detail::GeneralVoFExactStorage audit(16384);
    audit.capacity(Domain::Input, 10, sizeof(M));
    audit.measure(Domain::Input, value);
    audit.measure(Domain::Scratch, value);
    audit.measure(Domain::Result, value);
    const auto shared = audit.finish();
    EXPECT_EQ(shared.bytes[0], 10 * sizeof(M) + value.storage_bytes());
    EXPECT_EQ(shared.bytes[1], 0u);
    EXPECT_EQ(shared.bytes[2], 0u);
    audit.release(Domain::Input, value);
    const auto moved = audit.finish();
    EXPECT_EQ(moved.bytes[0], 10 * sizeof(M));
    EXPECT_EQ(moved.bytes[1], value.storage_bytes());
    EXPECT_THROW(audit.capacity(Domain::Scratch, 16384, 1), std::runtime_error);
    EXPECT_EQ(audit.finish().total(), moved.total());
    EXPECT_THROW(audit.release(Domain::Input, value), std::runtime_error);
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, DistinctExactPayloadsExhaustCumulativeBudgetWhileSharedValuesFit)
{
    using Domain = detail::GeneralVoFStorageDomain;
    using M = Meshes::ExactGeometryMeasure;
    std::vector<M> distinct, shared;
    distinct.reserve(128);
    shared.reserve(128);
    const auto common = M::from_value(1e-310).squared();
    for (unsigned i = 1; i <= 128; ++i)
    {
        distinct.push_back(common.scaled(double(i)));
        shared.push_back(common);
    }
    constexpr std::size_t budget = 16384;
    for (const auto& value : distinct)
        ASSERT_LT(value.storage_bytes(), budget);
    detail::GeneralVoFExactStorage sharing(budget);
    sharing.capacity(Domain::Input, shared.capacity(), sizeof(M));
    for (const auto& value : shared)
        sharing.measure(Domain::Input, value);
    EXPECT_LT(sharing.finish().total(), budget);
    detail::GeneralVoFExactStorage exhausted(budget);
    exhausted.capacity(Domain::Input, distinct.capacity(), sizeof(M));
    EXPECT_THROW(
        {
            for (const auto& value : distinct)
                exhausted.measure(Domain::Input, value);
        },
        std::runtime_error);
    EXPECT_LE(exhausted.finish().total(), budget);
    detail::GeneralVoFExactStorage recovered(256 * 1024);
    recovered.capacity(Domain::Input, distinct.capacity(), sizeof(M));
    for (const auto& value : distinct)
        recovered.measure(Domain::Input, value);
    EXPECT_GT(recovered.finish().total(), budget);
    EXPECT_LT(recovered.finish().total(), 256 * 1024u);
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, ExactRawBindingReusesOnlyAuthenticatedLocalCoefficients)
{
    using M = Meshes::ExactGeometryMeasure;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transport = detail::ConservativeVoFTransport;
    namespace AMR = Meshes::AMR;
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
            grid.xy_nodes.push_back({double(x), double(y), 0});
    for (unsigned x = 0; x < 3; ++x)
        grid.base_cells.push_back({x, x + 1, x + 5, x + 4});
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR root(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 3; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto boot = root.detach_owned(owned, {'b', 'i', 'n', 'd', 'i', 'n', 'g'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    const auto native = AMR::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor());
    const auto mesh = native.mesh;
    // An exact, unrepresentably small non-dyadic body. Its size, rather than
    // a diagnostic floating view, stresses repeated self-route serialization.
    auto positive = M::from_value(1).divided_by(M::from_value(3));
    for (unsigned i = 0; i < 17; ++i)
        positive = positive.squared();
    const auto negative = positive.scaled(-1);
    auto plus = std::make_shared<const M>(positive), minus = std::make_shared<const M>(negative);
    std::vector<Transport::FaceContribution> faces;
    int local_self = 0, local_remote = 0;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto normal = mesh->face_normal(LO(f));
        if (mesh->is_geometry_exterior_face(LO(f)) || std::abs(normal.x) < .5)
            continue;
        const auto first = mesh->cell_global_id(mesh->owner_cell(LO(f)));
        const auto second = mesh->cell_global_id(mesh->neighbor_cell(LO(f)));
        for (unsigned repeat = 0; repeat < 2; ++repeat)
        {
            faces.push_back({LO(f), first, 0, plus});
            faces.push_back({LO(f), first, 0, minus});
        }
        for (const auto endpoint : {first, second})
            (mesh->owned_cell_map()->getLocalElement(endpoint) >= 0 ? local_self : local_remote)++;
    }
    int global_self = 0, global_remote = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_self, &global_self);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_remote, &global_remote);
    EXPECT_GT(global_self, 0);
    if (comm->getSize() > 1)
        EXPECT_GT(global_remote, 0);
    const auto n = mesh->num_owned_cells();
    int local_empty = n == 0, global_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &global_empty);
    if (comm->getSize() == 4)
        EXPECT_GT(global_empty, 0);
    std::vector<M> outgoing(n);
    std::vector<std::map<GO, M>> incoming(n);
    constexpr std::size_t budget = 4 * 1024 * 1024;
    const auto verify = [&]
    { return detail::verify_general_vof_transfer_binding(mesh, {}, {}, faces, budget, &outgoing, &incoming, true); };
    const auto rows = verify();
    ASSERT_EQ(rows.size(), n);
    for (const auto& row : rows)
        for (const auto& [donor, coefficient] : row.coefficients)
            EXPECT_EQ(coefficient.sign(), 0);
    if (!outgoing.empty())
        outgoing.front() = M::from_value(1);
    EXPECT_THROW((void) verify(), std::runtime_error);
    for (auto& amount : outgoing)
        amount = {};
    EXPECT_THROW((void) detail::verify_general_vof_transfer_binding(mesh, {}, {}, faces, 1, &outgoing, &incoming, true),
        std::exception);
    EXPECT_NO_THROW((void) verify());
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, ExactBindingSharesRemainingCapacityBetweenOperationAndRetainedResult)
{
    using M = Meshes::ExactGeometryMeasure;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using Transport = detail::ConservativeVoFTransport;
    namespace AMR = Meshes::AMR;
    AMR::ExtrudedAMRGrid grid;
    grid.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}};
    grid.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}};
    grid.z_edges = {0, 1};
    AMR::ExtrudedAMR root(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    if (!comm->getRank())
        owned = {0, 1};
    auto boot = root.detach_owned(owned, {'b', 'i', 'n', 'd', '_', 'l', 'e', 'a', 's', 'e'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    const auto mesh = AMR::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor()).mesh;
    auto quantum = M::from_value(1).divided_by(M::from_value(3));
    auto other = M::from_value(1).divided_by(M::from_value(5));
    for (unsigned i = 0; i < 15; ++i)
    {
        quantum = quantum.squared();
        other = other.squared();
    }
    const auto positive = std::make_shared<const M>(quantum);
    const auto second = std::make_shared<const M>(other);
    const auto negative = std::make_shared<const M>((quantum + other).scaled(-1));
    std::vector<Transport::FaceContribution> contributions;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        if (!mesh->is_geometry_exterior_face(LO(f)))
        {
            const auto donor = mesh->cell_global_id(mesh->owner_cell(LO(f)));
            // Distinct coprime denominators grow the nonzero partial sum before
            // its exact cancellation; shared-body doubling misses that growth.
            contributions.push_back({LO(f), donor, 0, positive});
            contributions.push_back({LO(f), donor, 0, second});
            contributions.push_back({LO(f), donor, 0, negative});
        }
    const auto n = mesh->num_owned_cells();
    int local_empty = n == 0, global_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &global_empty);
    EXPECT_EQ(global_empty, comm->getSize() - 1);
    std::vector<M> outgoing(n);
    std::vector<std::map<GO, M>> incoming(n);
    constexpr std::size_t budget = 2 * 1024 * 1024;
    const auto verify = [&](std::size_t cap)
    {
        return detail::verify_general_vof_transfer_binding(
            mesh, {}, {}, contributions, cap, &outgoing, &incoming, true);
    };
    const auto rows = verify(budget);
    // Matched old/new admission:640KiB retains the exact result and operation,
    // while the old half-remainder subdivision rejects this same fixed input.
    EXPECT_NO_THROW((void) verify(640 * 1024));
    for (const auto& row : rows)
        for (const auto& [donor, amount] : row.coefficients)
            EXPECT_EQ(amount.sign(), 0);
    if (!outgoing.empty())
        outgoing.front() = quantum;
    EXPECT_THROW((void) verify(budget), std::runtime_error);
    for (auto& amount : outgoing)
        amount = {};
    EXPECT_THROW((void) verify(comm->getRank() == 0 ? 1 : budget), std::exception);
    EXPECT_NO_THROW((void) verify(budget));
}
} // namespace

namespace
{
TEST(ConservativeVoFTransportTest, LeasedPhysicalNormSeparatesAggregateOperationAndCommunicationStorage)
{
    using M = Meshes::ExactGeometryMeasure;
    namespace E = detail::vof_exact;
    const auto comm = Tpetra::getDefaultComm();
    DefaultTpetraTypes::map_type map(3, 0, comm);
    auto p = M::from_value(1).divided_by(M::from_value(3));
    auto q = M::from_value(1).divided_by(M::from_value(5));
    for (unsigned i = 0; i < 13; ++i)
    {
        p = p.squared();
        q = q.squared();
    }
    std::vector<M> residual(map.getLocalNumElements()), rhs(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i)
    {
        const auto gid = map.getGlobalElement(LO(i));
        residual[i] = gid == 0 ? p : gid == 1 ? q : M::from_value(1);
    }
    // Original zero-RHS reference is exactly1: 1+p²+q² >1, despite a
    // rounded relative diagnostic of1. No tiny squared defect may disappear.
    constexpr std::size_t budget = 512 * 1024;
    const auto rejected = E::leased_physical_norm(residual, rhs, 1, map, budget);
    EXPECT_FALSE(rejected.accepted);
    EXPECT_DOUBLE_EQ(rejected.relative_residual, 1);
    const auto accepted = E::leased_physical_norm(residual, rhs, 2, map, budget);
    EXPECT_TRUE(accepted.accepted);
    EXPECT_THROW((void) E::physical_norm(residual, rhs, 1, map, budget / 8), std::exception);
    EXPECT_THROW(
        (void) E::leased_physical_norm(residual, rhs, 1, map, comm->getRank() == 0 ? 1 : budget), std::exception);
    EXPECT_FALSE(E::leased_physical_norm(residual, rhs, 1, map, budget).accepted);
    int local_empty = residual.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (comm->getSize() > 3)
        EXPECT_EQ(empty, comm->getSize() - 3);
}
} // namespace
