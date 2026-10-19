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
            for (std::size_t id = 0; id < velocity.size(); ++id)
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
} // namespace
