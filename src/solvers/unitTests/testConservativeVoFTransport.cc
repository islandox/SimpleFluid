/** @file testConservativeVoFTransport.cc
 * @brief Direct passive-tracer bounds and local momentum-ledger checks before projection.
 */
#include "solvers/details/ConservativeVoFTransport.hh"
#include "FVM/AdaptiveVoF.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace
{
using namespace SimpleFluid;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(ConservativeVoFTransportTest, SplitCounterexampleHasBoundedConservativeImplicitTracesForEveryFirstAxis)
{
    Meshes::AMRGrid grid;
    grid.root_blocks = {2, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 4, 200};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::CartesianAMR>(grid));
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
                rhs[c] = mesh->cell_volume(static_cast<LO>(c)) * (1000*(1-f[c]) + f[c]) * passive;
            }
            for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
            {
                const auto fid = static_cast<LO>(id);
                const auto normal = mesh->face_normal(fid);
                const auto owner = mesh->owner_cell(fid);
                const auto x = mesh->cell_centroid(owner);
                velocity[id] = normal.x + normal.y + normal.z *
                    (x.x >= 1 && x.x < 2 && x.y >= 1 && x.y < 2 ? 1 : 0);
            }
            const auto phase = geometry.advect(f, velocity, 0.1, first_axis);
            std::vector<double> mass_flux(velocity.size());
            for (std::size_t c = 0; c < n; ++c)
                capacity[c] = mesh->cell_volume(static_cast<LO>(c)) * (1000*(1-phase.fraction[c]) + phase.fraction[c]);
            for (std::size_t id = 0; id < mass_flux.size(); ++id)
                mass_flux[id] = 1000*phase.liquid_volume[id] + phase.gas_volume[id];
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
    Meshes::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::CartesianAMR>(grid));
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
    EXPECT_THROW((void)tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-14), std::exception);
    std::fill(capacity.begin(), capacity.end(), 2);
    for (std::size_t face = 0; face < mesh->num_owned_faces(); ++face)
        flux[face] = 0.25 * mesh->face_normal(static_cast<LO>(face)).x;
    flux = geometry.exchange_faces(flux);
    EXPECT_NO_THROW((void)tracer.solve(rhs, zero, capacity, flux, true, 3000, 1e-14));
}

TEST(ConservativeVoFTransportTest, EquilibratedMicroscopicCarrierPreservesConstantConcentration)
{
    Meshes::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::CartesianAMR>(grid));
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

} // namespace
