/** @file testGeneralVoFSolver.cc
 * @brief Transactional source-free flow integration on actual detached native cells.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/GeneralVoFSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
namespace AMR = SimpleFluid::Meshes::AMR;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const VoF::GeneralVoF> phases(unsigned fixture, bool skew = false)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    const auto owning_ranks = std::min(2, comm->getSize());
    for (std::uint64_t gid = 0; gid < 27; ++gid)
        if (int(gid % owning_ranks) == comm->getRank())
            owned.push_back(gid);
    AMR::ExtrudedAMRGrid grid;
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned i = 0; i < 4; ++i)
            grid.xy_nodes.push_back({double(i) + (skew ? .3 * j : 0), double(j), 0});
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 3; ++i)
        {
            const unsigned a = 4 * j + i;
            grid.base_cells.push_back({a, a + 1, a + 5, a + 4});
        }
    for (unsigned i = 0; i < 3; ++i)
    {
        grid.boundaries.push_back({i, i + 1, "wall"});
        grid.boundaries.push_back({12 + i, 13 + i, "wall"});
        grid.boundaries.push_back({4 * i, 4 * (i + 1), "wall"});
        grid.boundaries.push_back({4 * i + 3, 4 * (i + 1) + 3, "wall"});
    }
    grid.z_edges = {0, 1, 2, 3};
    auto source = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap = source->detach_owned(owned, {'g', 'e', 'n', 'e', 'r', 'a', 'l', 'f', 'l', 'o', 'w'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = hierarchy->geometry(leaf);
        const auto volume = geometry->moments().volume.as_double();
        double g = 0, c = volume;
        if (fixture == 1)
        {
            const bool gas_cell = geometry->centroid()[0] < 1;
            g = gas_cell ? volume : 0;
            c = gas_cell ? 0 : volume;
        }
        if (fixture == 2)
        {
            const auto cut = geometry->clip({{1, 0, 0}, {0, 0, 0}, .75});
            g = cut.inside ? cut.inside->moments().volume.as_double() : 0;
            c = cut.outside ? cut.outside->moments().volume.as_double() : 0;
        }
        order.push_back(key);
        gas.push_back(g);
        carrier.push_back(c);
    }
    auto result =
        std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
    int local_present = result->mesh()->num_owned_cells() > 0, present = 0;
    int local_empty = !local_present, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_present, &present);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(present, owning_ranks);
    EXPECT_EQ(empty, comm->getSize() - owning_ranks);
    return result;
}
std::shared_ptr<const VoF::GeneralVoF> cylinder_sector_phases()
{
    AMR::ExtrudedAMRGrid grid;
    const std::array<double, 4> radii = {.8, .95, 1.05, 1.2};
    for (unsigned a = 0; a < 5; ++a)
    {
        const double angle = -.2 + .1 * a;
        for (double radius : radii)
            grid.xy_nodes.push_back({radius * std::cos(angle), radius * std::sin(angle), 0});
    }
    for (unsigned a = 0; a < 4; ++a)
        for (unsigned r = 0; r < 3; ++r)
        {
            const unsigned p = 4 * a + r;
            grid.base_cells.push_back({p, p + 1, p + 5, p + 4});
        }
    for (unsigned a = 0; a < 4; ++a)
    {
        grid.boundaries.push_back({4 * a, 4 * (a + 1), "wall"});
        grid.boundaries.push_back({4 * a + 3, 4 * (a + 1) + 3, "wall"});
    }
    for (unsigned r = 0; r < 3; ++r)
    {
        grid.boundaries.push_back({r, r + 1, "symmetry"});
        grid.boundaries.push_back({16 + r, 17 + r, "symmetry"});
    }
    grid.z_edges = {0, .2 / 3, 2 * .2 / 3, .2};
    grid.boundary_aliases = {{"zmin", "symmetry_zmin"}, {"zmax", "symmetry_zmax"}};
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 36; ++c)
        if (int(c % comm->getSize()) == comm->getRank())
            owned.push_back(c);
    auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap =
        provider->detach_owned(owned, {'c', 'y', 'l', 'i', 'n', 'd', 'e', 'r', 's', 'e', 'c', 't', 'o', 'r'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto shell = hierarchy->geometry(leaf);
        const auto center = shell->centroid();
        const double radial = std::hypot(center[0], center[1]);
        const unsigned layer = radial < .95 ? 0 : radial < 1.05 ? 1 : 2;
        const double height = shell->axis_spans()[2].as_double(), half_angle = .05;
        double gas_area = 0, carrier_area = 0;
        if (layer == 0)
            gas_area = .5 * (radii[1] * radii[1] - radii[0] * radii[0]) * std::sin(.1);
        else if (layer == 2)
            carrier_area = .5 * (radii[3] * radii[3] - radii[2] * radii[2]) * std::sin(.1);
        else
        {
            // Independent polar integrals: inner gas is 1/2∫(1-r_in²)dθ;
            // outer carrier is 1/2∫(r_out²-1)dθ. The chord bounds are
            // r_i*cos(Δθ/2)/cos(θ-θ_mid), integrated analytically by tan.
            const double chord_integral = std::cos(half_angle) * std::cos(half_angle) * std::tan(half_angle);
            gas_area = half_angle - radii[1] * radii[1] * chord_integral;
            carrier_area = radii[2] * radii[2] * chord_integral - half_angle;
        }
        order.push_back(key);
        gas.push_back(layer == 0 ? shell->moments().volume.as_double() : gas_area * height);
        carrier.push_back(layer == 2 ? shell->moments().volume.as_double() : carrier_area * height);
    }
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
}
std::shared_ptr<const VoF::GeneralVoF> pure_cartesian_phases(unsigned nx, unsigned ny, unsigned nz)
{
    AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {nx, ny, nz};
    grid.extent = {1, 1, 1};
    auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < std::uint64_t(nx) * ny * nz; ++c)
        if (int(c % comm->getSize()) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider->detach_owned(owned, {'v', 'i', 's', 'c', 'o', 'u', 's', 'm', 'o', 'd', 'e'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        order.push_back(key);
        gas.push_back(0);
        carrier.push_back(hierarchy->geometry(leaf)->moments().volume.as_double());
    }
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
}
std::shared_ptr<const VoF::GeneralVoF> skew_coarse_fine_plane()
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 5; ++x)
            grid.xy_nodes.push_back({x / 4.0 + .2 * y / 2.0, y / 2.0, 0});
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
        {
            const unsigned p = 5 * y + x;
            grid.base_cells.push_back({p, p + 1, p + 6, p + 5});
        }
    for (unsigned x = 0; x < 4; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({10 + x, 11 + x, "wall"});
    }
    for (unsigned y = 0; y < 2; ++y)
    {
        grid.boundaries.push_back({5 * y, 5 * (y + 1), "wall"});
        grid.boundaries.push_back({5 * y + 4, 5 * (y + 1) + 4, "wall"});
    }
    grid.z_edges = {0, .5, 1};
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 16; ++c)
        if (int(c % comm->getSize()) == comm->getRank())
            owned.push_back(c);
    auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    auto bootstrap = provider->detach_owned(owned, {'s', 'k', 'e', 'w', 'p', 'a', 's', 's', 'a', 'g', 'e'});
    auto raw = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> zero, full;
    for (const auto& [key, leaf] : raw->owned())
    {
        order.push_back(key);
        zero.push_back(0);
        full.push_back(raw->geometry(leaf)->moments().volume.as_double());
    }
    auto bound = std::make_shared<const VoF::GeneralVoF>(raw, order, zero, full);
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& [key, leaf] : bound->hierarchy()->owned())
    {
        const auto center = bound->hierarchy()->geometry(leaf)->centroid();
        const double xi = center[0] - .2 * center[1];
        if (xi > .5 && xi < .75)
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    const auto plan = bound->hierarchy()->plan_adaptation(marks);
    const auto refined = bound->hierarchy()->apply(plan);
    order.clear();
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : refined->owned())
    {
        const auto shell = refined->geometry(leaf);
        const auto clip = shell->clip({{1, .17, .11}, {0, 0, 0}, .65});
        order.push_back(key);
        gas.push_back(clip.inside ? clip.inside->moments().volume.as_double() : 0);
        carrier.push_back(clip.outside ? clip.outside->moments().volume.as_double() : 0);
    }
    return std::make_shared<const VoF::GeneralVoF>(refined, std::move(order), std::move(gas), std::move(carrier));
}
VoFMaterial inviscid()
{
    VoFMaterial material;
    material.gas_viscosity = material.liquid_viscosity = material.surface_tension = 0;
    return material;
}
TEST(GeneralVoFSolverTest, NativeSourceFreeStationaryAdvanceAndRejectedStepRetainAcceptedPublication)
{
    GeneralVoFSolver solver(phases(0), inviscid());
    const auto initial = solver.phases();
    const auto fraction = solver.gas_fraction();
    EXPECT_THROW((void) solver.step(-1), std::runtime_error);
    EXPECT_EQ(solver.phases(), initial);
    EXPECT_EQ(solver.gas_fraction(), fraction);
    EXPECT_EQ(solver.time(), 0);
    for (int step = 0; step < 3; ++step)
    {
        const auto diagnostic = solver.step(.01);
        EXPECT_NEAR(diagnostic.time, .01 * (step + 1), 1e-15);
        EXPECT_EQ(diagnostic.gas_volume, 0);
        EXPECT_EQ(diagnostic.maximum_velocity, 0);
        EXPECT_EQ(diagnostic.maximum_divergence, 0);
    }
}
TEST(GeneralVoFSolverTest, IndependentMixedPlaneRetainsBothPhaseMassesInSourceFreeStationaryFlow)
{
    GeneralVoFSolver solver(phases(2), inviscid());
    const auto before = solver.diagnostics();
    for (int step = 0; step < 3; ++step)
    {
        const auto after = solver.step(.01);
        EXPECT_NEAR(after.gas_volume, before.gas_volume, 1e-12);
        EXPECT_NEAR(after.gas_mass, before.gas_mass, 1e-12);
        EXPECT_NEAR(after.liquid_mass, before.liquid_mass, 1e-10);
        EXPECT_EQ(after.maximum_velocity, 0);
    }
}
TEST(GeneralVoFSolverTest, BinaryInterfaceCannotSilentlyBecomeZeroCurvatureCapillarySuccess)
{
    auto source = phases(1);
    auto material = inviscid();
    material.surface_tension = .07;
    EXPECT_THROW((void) GeneralVoFSolver(source, material), std::runtime_error);
}
TEST(GeneralVoFSolverTest, SymmetricNativeRegridRetainsAcceptedPhysicalMomentumAtFixedClock)
{
    GeneralVoFSolver solver(phases(0), inviscid());
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        const double a = std::numbers::pi * x.x / 3, b = std::numbers::pi * x.y / 3;
        velocity[c] = {std::sin(a) * std::sin(a) * std::sin(2 * b), -std::sin(b) * std::sin(b) * std::sin(2 * a), 0};
    }
    solver.initialize(velocity);
    const auto immutable_pre_step = solver.phases();
    (void) solver.step(.001);
    const auto before = solver.diagnostics();
    ASSERT_GT(before.time, 0);
    const auto source = solver.phases();
    EXPECT_NE(source, immutable_pre_step);
    std::vector<AMR::AdaptiveMark> refine;
    for (const auto& key : source->owned_order())
    {
        const auto center = source->hierarchy()->geometry(source->hierarchy()->owned().at(key))->centroid();
        if (std::abs(center[0] - 1.5) < 1e-12 && std::abs(center[1] - 1.5) < 1e-12 && std::abs(center[2] - 1.5) < 1e-12)
            refine.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    const auto count = [](const GeneralVoFSolver& flow)
    {
        double local = double(flow.mesh()->num_owned_cells()), global = 0;
        Teuchos::reduceAll(*flow.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    const double old_cells = count(solver);
    EXPECT_NO_THROW(solver.regrid(refine));
    EXPECT_GT(count(solver), old_cells);
    EXPECT_NE(solver.phases(), source);
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_NEAR(solver.diagnostics().liquid_mass, before.liquid_mass, 2e-11);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], before.momentum[a], 2e-11);
    const auto remapped = solver.phases();
    const auto after_regrid = solver.diagnostics();
    std::vector<int> destination(remapped->owned_order().size(),
        (Tpetra::getDefaultComm()->getRank() + 1) % Tpetra::getDefaultComm()->getSize());
    EXPECT_NO_THROW(solver.repartition(destination));
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_NEAR(solver.diagnostics().gas_mass, after_regrid.gas_mass, 2e-11);
    EXPECT_NEAR(solver.diagnostics().liquid_mass, after_regrid.liquid_mass, 2e-11);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], after_regrid.momentum[a], 2e-11);
    // Retained sources from both accepted epochs remain valid and unchanged.
    EXPECT_NE(immutable_pre_step, source);
    EXPECT_NE(source, remapped);
    EXPECT_EQ(immutable_pre_step->gas_volumes(), std::vector<double>(immutable_pre_step->owned_order().size(), 0));
}
TEST(GeneralVoFSolverTest, PhysicalHydrostaticPressureBalancesSkewNativeCentersOverFourSteps)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolverOptions options;
    options.flow.gravity = {.3, -.2, .1};
    GeneralVoFSolver solver(phases(0, true), material, options);
    for (unsigned step = 0; step < 4; ++step)
        (void) solver.step(.01);
    using GO = DefaultTpetraTypes::global_ordinal_type;
    using LO = DefaultTpetraTypes::local_ordinal_type;
    GO local_gid = std::numeric_limits<GO>::max(), reference_gid = 0;
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
        local_gid = std::min(local_gid, solver.mesh()->cell_global_id(LO(c)));
    const auto comm = solver.mesh()->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_gid, &reference_gid);
    double local[4]{}, reference[4]{};
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
        if (solver.mesh()->cell_global_id(LO(c)) == reference_gid)
        {
            const auto x = solver.mesh()->cell_centroid(LO(c));
            local[0] = solver.pressure()[c];
            local[1] = x.x;
            local[2] = x.y;
            local[3] = x.z;
        }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 4, local, reference);
    for (std::size_t c = 0; c < solver.velocity().size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(LO(c));
        const double expected = .3 * (x.x - reference[1]) - .2 * (x.y - reference[2]) + .1 * (x.z - reference[3]);
        EXPECT_NEAR(solver.pressure()[c] - reference[0], expected, 2e-7);
        for (double value : solver.velocity()[c])
            EXPECT_NEAR(value, 0, 2e-8);
    }
    EXPECT_LT(solver.diagnostics().maximum_divergence, 1e-8);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, IndependentlyIntegratedReflectedCylinderRetainsFiniteDurationCapillaryBalance)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    material.surface_tension = 1;
    GeneralVoFSolverOptions options;
    options.flow.pressure_relative_tolerance = 1e-12;
    GeneralVoFSolver solver(cylinder_sector_phases(), material, options);
    const auto initial = solver.diagnostics();
    constexpr double physical_end = .02, step_ceiling = .0025, stability_fraction = .9;
    const double initial_limit = solver.stable_timestep();
    ASSERT_TRUE(std::isfinite(initial_limit));
    ASSERT_GT(initial_limit, 0);
    const double initial_step = std::min(step_ceiling, stability_fraction * initial_limit);
    ASSERT_GT(initial_step, 0);
    const double nominal_steps = std::ceil(physical_end / initial_step);
    ASSERT_LT(nominal_steps, double(std::numeric_limits<std::size_t>::max() / 2));
    // Twice the initial-limit step count permits moderate subsequent variation
    // while detecting a collapsing timestep instead of shortening physical time.
    const auto maximum_steps = 2 * std::size_t(nominal_steps);
    std::size_t steps = 0;
    double peak = 0;
    while (solver.time() < physical_end)
    {
        ASSERT_LT(steps, maximum_steps) << "initial stability limit=" << initial_limit;
        const double before = solver.time(), limit = solver.stable_timestep();
        ASSERT_TRUE(std::isfinite(limit));
        ASSERT_GT(limit, 0);
        const double dt = std::min({step_ceiling, stability_fraction * limit, physical_end - before});
        ASSERT_TRUE(std::isfinite(dt));
        ASSERT_GT(dt, 0);
        ASSERT_GT(before + dt, before);
        SCOPED_TRACE(::testing::Message() << "time=" << before << ", dt=" << dt << ", stable limit=" << limit);
        VoFStepDiagnostics after;
        ASSERT_NO_THROW(after = solver.step(dt));
        ++steps;
        peak = std::max(peak, after.maximum_velocity);
        EXPECT_LT(after.maximum_divergence, 1e-8);
        EXPECT_LT(std::abs(after.gas_mass - initial.gas_mass) / initial.gas_mass, 2e-11);
        EXPECT_LT(std::abs(after.liquid_mass - initial.liquid_mass) / initial.liquid_mass, 2e-11);
    }
    double local[4] = {}, global[4] = {};
    for (std::size_t c = 0; c < solver.mesh()->num_owned_cells(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        const double radial = std::hypot(x.x, x.y),
                     volume = solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c));
        if (radial < .95)
        {
            local[0] += volume * solver.pressure()[c];
            local[1] += volume;
        }
        if (radial > 1.05)
        {
            local[2] += volume * solver.pressure()[c];
            local[3] += volume;
        }
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
    ASSERT_GT(global[1], 0);
    ASSERT_GT(global[3], 0);
    const double jump = global[0] / global[1] - global[2] / global[3];
    EXPECT_LT(std::abs(jump - 1), .08);
    EXPECT_LT(peak, .05);
    EXPECT_DOUBLE_EQ(solver.time(), physical_end);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, ClosedSymmetryViscousModeRetainsAnalyticFiniteDurationRate)
{
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    material.gas_viscosity = material.liquid_viscosity = .01;
    GeneralVoFSolver solver(pure_cartesian_phases(8, 8, 2), material);
    const auto comm = solver.mesh()->owned_cell_map()->getComm();
    const int local_cells = static_cast<int>(solver.mesh()->num_owned_cells());
    int global_cells = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
    ASSERT_EQ(global_cells, 128);
    constexpr double amplitude = 1e-6;
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        velocity[c] = {amplitude * std::sin(std::numbers::pi * x.x) * std::cos(std::numbers::pi * x.y),
            -amplitude * std::cos(std::numbers::pi * x.x) * std::sin(std::numbers::pi * x.y), 0};
    }
    solver.initialize(velocity);
    const auto mode_amplitude = [&]
    {
        double local[2] = {}, global[2] = {};
        for (std::size_t c = 0; c < velocity.size(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
            const double shape = std::sin(std::numbers::pi * x.x) * std::cos(std::numbers::pi * x.y);
            const double volume = solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c));
            local[0] += volume * solver.velocity()[c][0] * shape;
            local[1] += volume * shape * shape;
        }
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
        return global[0] / global[1];
    };
    const auto before = solver.diagnostics();
    const double initial = mode_amplitude();
    for (unsigned step = 0; step < 10; ++step)
    {
        const auto after = solver.step(.002);
        EXPECT_LT(after.maximum_divergence, 1e-8);
    }
    const double final = mode_amplitude();
    EXPECT_GT(final, 0);
    EXPECT_LT(final, initial);
    const double rate = -std::log(final / initial) / solver.time();
    EXPECT_NEAR(rate / (2 * std::numbers::pi * std::numbers::pi * .01), 1, .05);
    EXPECT_LT(solver.diagnostics().kinetic_energy, before.kinetic_energy);
    for (unsigned a = 0; a < 3; ++a)
        EXPECT_NEAR(solver.diagnostics().momentum[a], before.momentum[a], 2e-11);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, MicroscopicDeclaredTraceAggregatesPhaseMassBeforeDiagnosticNarrowing)
{
    const auto base = phases(0);
    const double eta = std::numeric_limits<double>::denorm_min();
    std::vector<double> gas(base->owned_order().size(), eta), carrier = base->carrier_volumes();
    std::vector<VoF::GeneralVoFPhaseGeometry> geometry(gas.size(), VoF::GeneralVoFPhaseGeometry::PureCarrier);
    auto source = std::make_shared<const VoF::GeneralVoF>(
        base->hierarchy(), base->owned_order(), gas, carrier, base->options(), geometry);
    auto material = inviscid();
    material.gas_density = .5;
    material.liquid_density = 1;
    GeneralVoFSolver solver(source, material);
    // The accepted phase eta is representable. Its weighted mass eta/2 is a
    // transient product; aggregate before narrowing rather than dropping it.
    const auto expected = Meshes::ExactGeometryMeasure::from_value(eta).scaled(13.5).certified().as_double();
    EXPECT_EQ(solver.diagnostics().gas_mass, expected);
    const auto before = solver.phases();
    const auto after = solver.step(.001);
    EXPECT_EQ(after.gas_mass, expected);
    EXPECT_EQ(after.maximum_velocity, 0);
    for (double value : solver.phases()->gas_volumes())
        EXPECT_EQ(value, eta);
    EXPECT_NE(solver.phases(), before);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, NonzeroSkewCoarseFinePhasePassagePreservesIndependentInventories)
{
    auto material = inviscid();
    material.gas_density = .1;
    material.liquid_density = 1;
    GeneralVoFSolver solver(skew_coarse_fine_plane(), material);
    double local_structure[2] = {double(solver.mesh()->num_owned_cells()), 0}, structure[2]{};
    for (std::size_t f = 0; f < solver.mesh()->num_owned_faces(); ++f)
    {
        const auto face = DefaultTpetraTypes::local_ordinal_type(f);
        if (solver.mesh()->is_geometry_exterior_face(face))
            continue;
        const double left = solver.mesh()->cell_volume(solver.mesh()->owner_cell(face)),
                     right = solver.mesh()->cell_volume(solver.mesh()->neighbor_cell(face));
        if (std::max(left, right) > 1.5 * std::min(left, right))
            local_structure[1] += 1;
    }
    Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local_structure, structure);
    // The provider also introduces conforming transition cells. Check its
    // actual refined native topology, rather than assuming only marked splits.
    EXPECT_GT(structure[0], 16);
    EXPECT_LE(structure[0], 64);
    EXPECT_GT(structure[1], 0);
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < velocity.size(); ++c)
    {
        const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
        const double xi = x.x - .2 * x.y, eta = x.y;
        const double vx = .05 * std::pow(std::sin(std::numbers::pi * xi), 2) * std::sin(2 * std::numbers::pi * eta);
        const double vy = -.05 * std::pow(std::sin(std::numbers::pi * eta), 2) * std::sin(2 * std::numbers::pi * xi);
        velocity[c] = {vx + .2 * vy, vy, 0};
    }
    solver.initialize(velocity);
    const auto region_gas = [&]
    {
        double local = 0, global = 0;
        for (std::size_t c = 0; c < solver.mesh()->num_owned_cells(); ++c)
        {
            const auto x = solver.mesh()->cell_centroid(DefaultTpetraTypes::local_ordinal_type(c));
            if (x.x - .2 * x.y > .5)
                local += solver.phases()->gas_volumes()[c];
        }
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    const auto before = solver.diagnostics();
    const double initial_region = region_gas();
    for (unsigned step = 0; step < 10; ++step)
    {
        const auto after = solver.step(.002);
        EXPECT_LT(std::abs(after.gas_mass - before.gas_mass) / before.gas_mass, 2e-11);
        EXPECT_LT(std::abs(after.liquid_mass - before.liquid_mass) / before.liquid_mass, 2e-11);
        for (const auto& ledger : solver.momentum_transport_diagnostics())
            EXPECT_LE(ledger.physical_relative_residual, 1e-12);
        EXPECT_LT(after.maximum_divergence, 1e-8);
        for (std::size_t c = 0; c < solver.phases()->gas_volumes().size(); ++c)
        {
            EXPECT_GE(solver.phases()->gas_volumes()[c], 0);
            EXPECT_GE(solver.phases()->carrier_volumes()[c], 0);
            EXPECT_GE(solver.gas_fraction()[c], 0);
            EXPECT_LE(solver.gas_fraction()[c], 1);
        }
    }
    // The region's only open internal boundary is the actual refined stripe
    // interface xi=.5; external walls are impermeable. The initial analytic
    // velocity carries gas into this region, at O(1e−4) volume over .02s.
    EXPECT_GT(region_gas() - initial_region, 1e-5);
    EXPECT_DOUBLE_EQ(solver.time(), .02);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, DominantGasOccupancyDefectUsesSharedBoundedGeometricView)
{
    const auto base = phases(0);
    std::vector<double> gas = base->carrier_volumes(), carrier(gas.size(), std::numeric_limits<double>::denorm_min());
    for (auto& amount : gas)
        amount = std::nextafter(amount, std::numeric_limits<double>::infinity());
    std::vector<VoF::GeneralVoFPhaseGeometry> geometry(gas.size(), VoF::GeneralVoFPhaseGeometry::PureGas);
    auto source = std::make_shared<const VoF::GeneralVoF>(
        base->hierarchy(), base->owned_order(), gas, carrier, base->options(), geometry);
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolver solver(source, material);
    for (std::size_t c = 0; c < gas.size(); ++c)
    {
        EXPECT_GT(
            solver.phases()->gas_volumes()[c], solver.mesh()->cell_volume(DefaultTpetraTypes::local_ordinal_type(c)));
        EXPECT_EQ(solver.gas_fraction()[c], 1);
    }
    (void) solver.step(.001);
    EXPECT_EQ(solver.phases()->gas_volumes(), gas);
    EXPECT_EQ(solver.phases()->carrier_volumes(), carrier);
    for (double fraction : solver.gas_fraction())
        EXPECT_EQ(fraction, 1);
}
} // namespace

namespace
{
TEST(GeneralVoFSolverTest, TinyCellRepartitionRetainsIndependentZeroRatesAndSubnormalAggregateMomentum)
{
    const double length = std::ldexp(1.0, -358);
    const double eta = std::numeric_limits<double>::denorm_min();
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2 * length, 2 * length, 2 * length};
    auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    const auto comm = Tpetra::getDefaultComm();
    const int owning_ranks = std::min(2, comm->getSize());
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 8; ++c)
        if (int(c % owning_ranks) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider->detach_owned(owned, {'t', 'i', 'n', 'y', 'm', 'o', 'm', 'e', 'n', 't', 'u', 'm'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto exact_volume = Meshes::ExactGeometryMeasure::volume(*hierarchy->geometry(leaf));
        ASSERT_EQ(exact_volume.compare(Meshes::ExactGeometryMeasure::from_value(eta)), 0);
        order.push_back(key);
        gas.push_back(0);
        carrier.push_back(eta);
    }
    auto phase =
        std::make_shared<const VoF::GeneralVoF>(hierarchy, std::move(order), std::move(gas), std::move(carrier));
    auto material = inviscid();
    material.gas_density = material.liquid_density = 1;
    GeneralVoFSolver solver(phase, material);
    std::vector<VoF::Vector> velocity(solver.mesh()->num_owned_cells(), {.5, 0, 0});
    std::vector<double> independent_faces(solver.mesh()->num_faces(), 0);
    ASSERT_NO_THROW(solver.initialize(velocity, independent_faces));
    const auto check_owners = [&]
    {
        const int local_present = solver.mesh()->num_owned_cells() > 0;
        const int local_empty = !local_present;
        int present = 0, empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_present, &present);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
        EXPECT_EQ(present, owning_ranks);
        EXPECT_EQ(empty, comm->getSize() - owning_ranks);
    };
    check_owners();
    const auto before = solver.diagnostics();
    EXPECT_EQ(before.momentum[0], 4 * eta);
    EXPECT_EQ(before.kinetic_energy, eta);
    const auto source = solver.phases();
    const auto source_finite = source->phase_volumes();
    const auto source_pressure = solver.pressure();
    std::vector<int> destination(source->owned_order().size(), (comm->getRank() + 1) % comm->getSize());
    ASSERT_NO_THROW(solver.repartition(destination));
    check_owners();
    EXPECT_EQ(solver.time(), before.time);
    EXPECT_EQ(solver.diagnostics().momentum, before.momentum);
    EXPECT_EQ(solver.diagnostics().liquid_mass, before.liquid_mass);
    EXPECT_EQ(solver.diagnostics().kinetic_energy, before.kinetic_energy);
    for (const auto& u : solver.velocity())
    {
        EXPECT_EQ(u[0], .5);
        EXPECT_EQ(u[1], 0);
        EXPECT_EQ(u[2], 0);
    }
    for (double value : solver.face_velocity())
        EXPECT_EQ(value, 0);
    for (double value : solver.pressure())
        EXPECT_EQ(value, 0);
    EXPECT_EQ(solver.diagnostics().maximum_divergence, 0);
    for (const auto& p : solver.phases()->phase_volumes())
    {
        EXPECT_EQ(p.gas, 0);
        EXPECT_EQ(p.carrier, eta);
    }
    // Immutable accepted sources survive a later ownership publication.
    ASSERT_EQ(source->phase_volumes().size(), source_finite.size());
    for (std::size_t c = 0; c < source_finite.size(); ++c)
    {
        EXPECT_EQ(source->phase_volumes()[c].gas, source_finite[c].gas);
        EXPECT_EQ(source->phase_volumes()[c].carrier, source_finite[c].carrier);
    }
    EXPECT_EQ(source_pressure, std::vector<double>(source_finite.size(), 0));
}
} // namespace

namespace
{
std::shared_ptr<const VoF::GeneralVoF> inventory_fixture(
    VoF::PhaseInventoryPolicy policy, bool seed_remainder, bool cell_local_refinement = false)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    const auto owners = std::min(2, comm->getSize());
    for (std::uint64_t c = 0; c < 8; ++c)
        if (int(c % owners) == comm->getRank())
            owned.push_back(c);
    auto boot = [&]
    {
        const AMR::AdaptiveBytes domain = {'o', 'p', 't', '_', 'i', 'n', 'v', 'e', 'n', 't', 'o', 'r', 'y'};
        if (cell_local_refinement)
        {
            AMR::ExtrudedAMRGrid grid;
            for (unsigned y = 0; y < 3; ++y)
                for (unsigned x = 0; x < 3; ++x)
                    grid.xy_nodes.push_back({double(x), double(y), 0});
            for (unsigned y = 0; y < 2; ++y)
                for (unsigned x = 0; x < 2; ++x)
                {
                    const unsigned a = 3 * y + x;
                    grid.base_cells.push_back({a, a + 1, a + 4, a + 3});
                }
            grid.z_edges = {0, 1, 2};
            return AMR::ExtrudedAMR(grid).detach_owned(owned, domain);
        }
        AMR::AMRGrid grid;
        grid.block_cells = {2, 2, 2};
        grid.extent = {2, 2, 2};
        return AMR::CartesianAMR(grid).detach_owned(owned, domain);
    }();
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto x = hierarchy->geometry(leaf)->centroid();
        const double g = seed_remainder && x[0] < 1 && x[1] < 1 && x[2] < 1 ? 1 : 0;
        order.push_back(key);
        gas.push_back(g);
        carrier.push_back(1 - g);
    }
    VoF::GeneralVoFOptions options;
    options.inventory_policy = policy;
    auto source = std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier, options);
    int local_empty = source->mesh()->num_owned_cells() == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - owners);
    if (!seed_remainder)
        return source;
    std::vector<VoF::GeneralVoFNodeVelocity> nodes;
    for (const auto& node : source->nodes())
        nodes.push_back({node.gid, {0, 0, 0}});
    std::vector<VoF::GeneralVoFFaceVolume> controls;
    for (const auto& face : source->tiles())
    {
        const auto x = face.geometry->centroid(), area = face.geometry->area_vector();
        double phi = 0;
        if (!face.exterior && x.x == 1 && area.x != 0)
            phi = (x.y < 1 ? 1 : -1) * (area.x > 0 ? 1 : -1) * .1;
        if (!face.exterior && x.y == 1 && area.y != 0)
            phi = (x.x > 1 ? 1 : -1) * (area.y > 0 ? 1 : -1) * .1;
        controls.push_back({face.face_gid, phi});
    }
    return source->apply(source->plan_transport(nodes, controls, 1));
}
TEST(GeneralVoFSolverTest, ExperimentalInventoryIsExplicitAndRollbackPreservesItsSelectedAuthority)
{
    using Policy = VoF::PhaseInventoryPolicy;
    auto finite = phases(0);
    EXPECT_EQ(finite->inventory_policy(), Policy::Finite);
    EXPECT_THROW((void) finite->experimental_phases(), std::exception);
    auto source = inventory_fixture(Policy::ExperimentalExact, true);
    const auto before = source->experimental_phases();
    int local_remainder = 0, remainders = 0;
    for (std::size_t c = 0; c < before.size(); ++c)
        local_remainder +=
            before[c].gas.compare(Meshes::ExactGeometryMeasure::from_value(source->gas_volumes()[c])) != 0 ||
            before[c].carrier.compare(Meshes::ExactGeometryMeasure::from_value(source->carrier_volumes()[c])) != 0;
    Teuchos::reduceAll(
        *source->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_remainder, &remainders);
    ASSERT_GT(remainders, 0);
    GeneralVoFSolver solver(source, inviscid());
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
    EXPECT_THROW((void) solver.step(-.001), std::exception);
    EXPECT_EQ(solver.phases(), source);
    EXPECT_EQ(solver.time(), 0);
    ASSERT_NO_THROW((void) solver.step(.001));
    EXPECT_DOUBLE_EQ(solver.time(), .001);
    ASSERT_EQ(solver.phases()->experimental_phases().size(), before.size());
    for (std::size_t c = 0; c < before.size(); ++c)
    {
        EXPECT_EQ(solver.phases()->experimental_phases()[c].gas.compare(before[c].gas), 0);
        EXPECT_EQ(solver.phases()->experimental_phases()[c].carrier.compare(before[c].carrier), 0);
    }
    const auto snapshot = solver.phases();
    std::vector<int> destination(snapshot->owned_order().size(),
        (Tpetra::getDefaultComm()->getRank() + 1) % Tpetra::getDefaultComm()->getSize());
    namespace wire = AMR::adaptive_detail;
    std::vector<OwnerPacket> expected_packets;
    for (std::size_t c = 0; c < snapshot->owned_order().size(); ++c)
    {
        wire::Writer w;
        w.bytes(AMR::encode_adaptive_key(snapshot->owned_order()[c]));
        w.bytes(snapshot->experimental_phases()[c].gas.encode());
        w.bytes(snapshot->experimental_phases()[c].carrier.encode());
        expected_packets.push_back({destination[c], std::move(w.data)});
    }
    const auto expected_packets_received =
        exchange_owner_packets(expected_packets, snapshot->mesh()->owned_cell_map()->getComm(), 1024 * 1024);
    std::map<AMR::AdaptiveCellKey, std::pair<Meshes::ExactGeometryMeasure, Meshes::ExactGeometryMeasure>> expected;
    bool expected_valid = true;
    for (const auto& packet : expected_packets_received.packets)
    {
        wire::Reader r{packet.bytes};
        auto key = AMR::decode_adaptive_key(r.bytes());
        auto gas = Meshes::ExactGeometryMeasure::decode(r.bytes());
        auto carrier = Meshes::ExactGeometryMeasure::decode(r.bytes());
        r.done();
        expected_valid =
            expected.emplace(std::move(key), std::pair{std::move(gas), std::move(carrier)}).second && expected_valid;
    }
    int local_valid = expected_valid, global_valid = 0;
    Teuchos::reduceAll(
        *snapshot->mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_valid, &global_valid);
    ASSERT_TRUE(global_valid);
    ASSERT_NO_THROW(solver.repartition(destination));
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
    EXPECT_DOUBLE_EQ(solver.time(), .001);
    const auto& migrated = solver.phases()->experimental_phases();
    ASSERT_EQ(expected.size(), migrated.size());
    for (std::size_t c = 0; c < migrated.size(); ++c)
    {
        const auto& value = expected.at(solver.phases()->owned_order()[c]);
        EXPECT_EQ(migrated[c].gas.compare(value.first), 0);
        EXPECT_EQ(migrated[c].carrier.compare(value.second), 0);
    }
    EXPECT_TRUE(snapshot->hierarchy()->geometry_unchanged());
}
TEST(GeneralVoFSolverTest, ExperimentalInventorySamePolicyRegridRetainsClockAndPhaseAuthority)
{
    using Policy = VoF::PhaseInventoryPolicy;
    // A Cartesian mark refines its entire 8-cell block, so the 15-cell oracle
    // requires the equivalent XY/Z provider's cell-local isotropic split.
    GeneralVoFSolver solver(inventory_fixture(Policy::ExperimentalExact, false, true), inviscid());
    const auto cells = [&]
    {
        double local = solver.mesh()->num_owned_cells(), global = 0;
        Teuchos::reduceAll(*solver.mesh()->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        return global;
    };
    ASSERT_EQ(cells(), 8);
    ASSERT_NO_THROW((void) solver.step(.001));
    const auto before = solver.diagnostics();
    const auto source = solver.phases();
    std::vector<AMR::AdaptiveMark> marks;
    for (const auto& key : source->owned_order())
    {
        const auto x = source->hierarchy()->geometry(source->hierarchy()->owned().at(key))->centroid();
        if (x[0] < 1 && x[1] < 1 && x[2] < 1)
            marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
    }
    EXPECT_NO_THROW(solver.regrid(marks));
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
    EXPECT_DOUBLE_EQ(solver.time(), before.time);
    EXPECT_EQ(solver.diagnostics().liquid_mass, before.liquid_mass);
    EXPECT_EQ(solver.diagnostics().momentum, before.momentum);
    EXPECT_EQ(cells(), 15);
}
TEST(GeneralVoFSolverTest, ExperimentalInventoryRejectsForeignPolicyTrialWithoutChangingAcceptedSolver)
{
    using Policy = VoF::PhaseInventoryPolicy;
    auto finite = inventory_fixture(Policy::Finite, false);
    auto exact = inventory_fixture(Policy::ExperimentalExact, false);
    GeneralVoFSolver solver(exact, inviscid());
    std::vector<VoF::GeneralVoFNodeVelocity> nodes;
    for (const auto& node : finite->nodes())
        nodes.push_back({node.gid, {0, 0, 0}});
    std::vector<VoF::GeneralVoFFaceVolume> controls;
    for (const auto& face : finite->tiles())
        controls.push_back({face.face_gid, 0});
    const auto foreign = finite->plan_transport(nodes, controls, 1);
    EXPECT_THROW((void) solver.phases()->apply(foreign), std::exception);
    EXPECT_EQ(solver.phases(), exact);
    EXPECT_EQ(solver.time(), 0);
    EXPECT_EQ(solver.phases()->inventory_policy(), Policy::ExperimentalExact);
}
} // namespace
