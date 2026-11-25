/** Exact two-cell balances for the packed common-map backward-Euler transport. */
#include "FVM/TransportSystemFactory.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/details/LowMachQbmmTransport.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace
{
using namespace SimpleFluid;
namespace Kernel = Reacting::transport_consistency_detail;
using Flow = Kernel::Flow;
using LO = Kernel::LO;
using Factory = FVM::TransportSystemFactory<Flow::Pack, Flow::Mesh>;

const auto environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t species = 2;
constexpr double dt = .25;
constexpr double physical_tolerance = 0x1p-40;
using Stock = std::array<double, species + 4>;
constexpr std::array<Stock, 2> stock{{{2., 0., 6., 12., 14., -8.}, {.5, 0., 10., 18., 26., 3.}}};

void expect_bits(double observed, double expected)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(observed), std::bit_cast<std::uint64_t>(expected));
}

struct TwoCellBlock
{
    SP<Flow::Mesh> mesh = std::make_shared<Flow::Mesh>(
        std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{ArrReal{0., .5, 1.}, {0., 1.}, {0., 1.}}),
        Flow::Mesh::DistributionOptions{.allow_empty_partitions = true});
    Kernel::Scalar history{mesh, "count history", false};
    Kernel::Scalar unit{mesh, 1., "unit coefficients"};
    Kernel::Scalar zero{mesh, 0., "zero diffusion"};
    Kernel::Face flux{mesh, "prescribed closed carrier flux", false};
    FVM::TransportGeometryCache<Flow::Mesh> geometry{*mesh};
    Kernel::BatchedWorkspace workspace;

    std::size_t cell_index(LO id) const { return mesh->cell_geometry_global_id(id); }

    void check_layout() const
    {
        const auto comm = mesh->owned_cell_map()->getComm();
        if (const char* expected = std::getenv("SIMPLEFLUID_QBMM_PACKED_EXPECTED_RANKS"))
            EXPECT_EQ(comm->getSize(), std::stoi(expected));
        int local[2]{mesh->num_owned_cells() == 0, 0}, global[2]{};
        for (const LO face : flux.owned_face_ids())
            if (mesh->is_interior_face(face))
            {
                const LO owner = mesh->owner_cell(face);
                const LO neighbor = mesh->opposite_or_periodic_neighbor_cell(face, owner);
                local[1] += static_cast<std::size_t>(neighbor) >= mesh->num_owned_cells();
            }
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, global);
        EXPECT_EQ(global[0], std::max(0, comm->getSize() - 2));
        EXPECT_EQ(global[1], comm->getSize() > 1 ? 1 : 0);
    }

    void check_transport(std::size_t donor, bool moving_stock, const FVM::ALEControlVolumeState* ale = nullptr)
    {
        SCOPED_TRACE(testing::Message() << "donor " << donor << ", transport " << moving_stock << ", ALE " << !!ale);
        const double new_volume = ale ? 1. : .5;
        const double rate = moving_stock ? new_volume / dt : 0.;
        for (const LO face : flux.owned_face_ids())
        {
            const double sign = cell_index(mesh->owner_cell(face)) == donor ? 1. : -1.;
            flux.set_owned_value(face, mesh->is_geometry_exterior_face(face) ? 0. : sign * rate);
        }
        flux.sync_ghosts();

        const std::size_t cells = mesh->num_owned_cells();
        std::vector<std::vector<double>> before(species + 4, std::vector<double>(cells)), after;
        for (std::size_t c = 0; c < cells; ++c)
        {
            const LO cell = static_cast<LO>(c);
            const auto& values = stock[cell_index(cell)];
            expect_bits(mesh->cell_volume(cell), new_volume);
            if (ale)
            {
                expect_bits(ale->old_cell_volumes()[c], .5);
                expect_bits(ale->new_cell_volumes()[c], new_volume);
            }
            for (std::size_t q = 0; q < values.size(); ++q)
                before[q][c] = values[q];
            history.set_owned_value(cell, values[species] / .5);
        }
        history.sync_ghosts();
        geometry.refresh();
        const auto system = Factory::assemble(Factory::WeightedScalarTerms{.history = {history, dt},
                                                  .face_fluxes = flux,
                                                  .coefficients = {unit, unit, zero},
                                                  .sources = {.source = [](LO) { return 0.; }}},
            {.condition = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.}; },
                .value = [](int, std::size_t) { return 0.; }},
            {.discretization = {FVM::ScalarTimeScheme::BackwardEuler, FVM::ScalarConvectionScheme::Upwind}},
            {.geometry_cache = &geometry, .ale = ale});

        const auto retained_cells = workspace.conc_owned;
        const auto retained_faces = workspace.face_overlap;
        const auto* retained_cell_import = workspace.cell_import.get();
        const auto* retained_face_import = workspace.face_import.get();
        LinearSolverOptions options;
        options.max_iterations = 2;
        options.tolerance = physical_tolerance;
        const auto result = Kernel::enforce_block(mesh, before, flux, dt, after, *system.matrix, *system.rhs, options,
            physical_tolerance, species, ale, workspace);
        if (!retained_cells.is_null())
        {
            EXPECT_EQ(workspace.conc_owned.getRawPtr(), retained_cells.getRawPtr());
            EXPECT_EQ(workspace.face_overlap.getRawPtr(), retained_faces.getRawPtr());
            EXPECT_EQ(workspace.cell_import.get(), retained_cell_import);
            EXPECT_EQ(workspace.face_import.get(), retained_face_import);
        }

        // dt * |phi| = Vnew. The donor row halves its stock in sweep one,
        // and sweep two lets the receiver use that updated donor value.
        ASSERT_EQ(result.sweeps, moving_stock ? 2 : 0);
        EXPECT_EQ(result.halo_cell_imports, std::size_t(result.sweeps) + 1);
        EXPECT_EQ(result.halo_face_imports, std::size_t(result.sweeps) + 1);
        EXPECT_EQ(workspace.conc_owned->getNumVectors(), species + 4);
        EXPECT_EQ(workspace.face_overlap->getNumVectors(), species + 4);
        ASSERT_EQ(result.components.size(), species + 4);
        for (std::size_t q = 0; q < result.components.size(); ++q)
        {
            SCOPED_TRACE(q);
            const double sent = stock[donor][q];
            const double received = stock[1 - donor][q];
            const auto& component = result.components[q];
            for (std::size_t c = 0; c < cells; ++c)
            {
                const bool from_cell = cell_index(static_cast<LO>(c)) == donor;
                const double delta = moving_stock ? (from_cell ? -.5 : .5) * sent : 0.;
                expect_bits(after[q][c], before[q][c] + delta);
            }
            long double initial = 0.;
            if (moving_stock && sent != 0.)
            {
                const long double magnitude = std::abs(sent);
                initial = std::max(1.L / 3, magnitude / (2.L * std::abs(received) + magnitude)) / physical_tolerance;
            }
            expect_bits(component.initial_residual, static_cast<double>(initial));
            expect_bits(component.final_residual, 0.);
            EXPECT_EQ(component.linear.iterations, 0);
            EXPECT_TRUE(component.linear.converged);
            EXPECT_DOUBLE_EQ(component.linear.rhs_norm, std::hypot(sent, received) / dt);
            expect_bits(component.linear.achieved_tolerance, 0.);
        }
        for (const LO face : flux.owned_face_ids())
        {
            const auto& packet = result.owned_flux[face];
            const Stock amounts{packet.species_molar_flux.at(0), packet.species_molar_flux.at(1),
                packet.population_number_flux, packet.shape_flux.H2, packet.shape_flux.H3, packet.internal_energy_flux};
            const bool exterior = mesh->is_geometry_exterior_face(face);
            // One face sees the final donor. A zero signed carrier uses the
            // owner trace, preserving the product's sign even for zero rates.
            const auto donor_for_packet = moving_stock ? donor : cell_index(mesh->owner_cell(face));
            for (std::size_t q = 0; q < amounts.size(); ++q)
            {
                const double factor = moving_stock ? .5 : 1.;
                expect_bits(amounts[q],
                    exterior ? 0. : flux.owned_value(face) * (factor * stock[donor_for_packet][q] / new_volume));
            }
        }
    }
};

TEST(LowMachQbmmPackedTransportTest, ExactFixedAndMovingMapsReusePackedHalos)
{
    TwoCellBlock block;
    block.check_layout();
    block.check_transport(0, true);
    block.check_transport(1, true);

    Flow::PlanarMotion motion(block.mesh, {.axis = Dimension::X});
    motion.begin_trial(2., dt);
    const auto ale = FVM::make_validated_planar_ale_control_volume_state(*block.mesh, motion);
    block.check_transport(0, true, &ale);
    block.check_transport(1, true, &ale);
    motion.rollback_trial();

    // Initial convergence still replays one packed cell/face exchange and
    // overwrites the rates left by the previous ALE solve.
    block.check_transport(1, false);
}

TEST(LowMachQbmmPackedTransportTest, RefreshesIncompleteWorkspaceCollectively)
{
    TwoCellBlock block;
    block.check_layout();
    auto& workspace = block.workspace;
    constexpr auto channels = species + 4;
    workspace.ensure(*block.mesh, block.flux, channels);
    const auto comm = block.mesh->owned_cell_map()->getComm();
    const auto previous_cells = workspace.conc_owned;
    const auto previous_faces = workspace.face_owned;
    const auto previous_cell_import = workspace.cell_import;
    if (comm->getRank() == comm->getSize() - 1)
        workspace.conc_overlap = Teuchos::null;
    workspace.ensure(*block.mesh, block.flux, channels);
    EXPECT_NE(workspace.conc_owned.getRawPtr(), previous_cells.getRawPtr());
    EXPECT_EQ(workspace.conc_overlap->getNumVectors(), channels);
    EXPECT_EQ(workspace.face_owned.getRawPtr(), previous_faces.getRawPtr());
    EXPECT_EQ(workspace.cell_import.get(), previous_cell_import.get());

    workspace.ensure(*block.mesh, block.flux, channels + 1);
    EXPECT_EQ(workspace.conc_overlap->getNumVectors(), channels + 1);
    EXPECT_EQ(workspace.face_overlap->getNumVectors(), channels + 1);
    workspace.ensure(*block.mesh, block.flux, channels);
    block.check_transport(0, true);
}
} // namespace
