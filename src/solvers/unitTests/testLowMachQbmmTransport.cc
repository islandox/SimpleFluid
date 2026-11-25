/** Native common-map QBMM transport and owner-oriented receipt regressions. */
#include "equations/CollectiveValidation.hh"
#include "equations/details/LowMachScaledFaceSum.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/details/LowMachQbmmState.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using FaceFlux = Flow::QbmmFaceFlux;
static_assert(std::is_same_v<decltype(FaceFlux::species_molar_flux), std::vector<double>>);
static_assert(std::is_same_v<decltype(FaceFlux::population_number_flux), double>);
static_assert(std::is_same_v<decltype(FaceFlux::shape_flux), R::QbmmShapeState>);
static_assert(std::is_same_v<decltype(R::QbmmShapeState::H2), double>);
static_assert(std::is_same_v<decltype(R::QbmmShapeState::H3), double>);
static_assert(std::is_same_v<decltype(FaceFlux::internal_energy_flux), double>);
static_assert(std::is_same_v<decltype(FaceFlux::material_volume_flux), double>);
static_assert(std::is_same_v<decltype(FaceFlux::derived_mass_flux), double>);
static_assert(std::is_same_v<decltype(FaceFlux::derived_drift_momentum_flux), vec3<double>>);

constexpr size_t channels = 7; // nA,nB,H0,derived H1,H2,H3,U.
constexpr double dt = .01;
constexpr double gas_constant = 8.31446261815324;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct FixtureOptions
{
    unsigned cells = 4;
    double stock_scale = 1;
    bool vacuum = false;
    bool vacuum_cells = false;
    double x_split = .5;
    double gas_reference = 6000; // J/mol at 300 K.
};

using FamilyTransport = R::QbmmModelOptions::FamilyTransport;
using Velocity = R::QbmmModelOptions::Velocity;
using SlopeLimiter = R::QbmmModelOptions::LinearSlopeLimiter;

struct MixedCarrierCase
{
    FamilyTransport family = FamilyTransport::NativeImplicit;
    Velocity velocity = Velocity::SharedCarrier;
    SlopeLimiter limiter = SlopeLimiter::RealizabilityOnly;
    LinearSolverBackend backend = LinearSolverBackend::Gmres;
    MultiRhsExecutionPolicy policy = MultiRhsExecutionPolicy::Auto;
    int width = 0;
    bool ale = false;
};

struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos;
    Flow::Options options;
    std::shared_ptr<Flow> flow;
    unsigned cells;

    explicit Fixture(FixtureOptions input = {}, std::optional<MixedCarrierCase> mixed = {}, bool batched = true)
        : cells(input.cells)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{
                cells == 4 ? ArrReal{0, input.x_split, 1} : ArrReal{0, 1}, ArrReal{0, 1},
                cells == 3 ? ArrReal{0, 1. / 3, 2. / 3, 1} : ArrReal{0, .5, 1}}),
            distribution);
        const std::vector phases{R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed};
        catalog = std::make_shared<const R::SpeciesCatalog>(
            std::vector<R::SpeciesDefinition>{{11, "A", .02, {{"X", 1}}, phases}, {29, "B", .04, {{"Y", 1}}, phases}});
        const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
        auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
        auto gas = R::make_ideal_gas_mixture(
            catalog, {{20, 300, input.gas_reference}, {20, 300, input.gas_reference}}, bounds);
        eos = {liquid, gas, gas};
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.preconditioner = LinearPreconditioner::None;
        options.linear.tolerance = 1e-12;
        options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "fine"},
            {Flow::PhaseRole::DispersedGas, "coarse"}};
        options.qbmm_families = {nullptr,
            std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "fine",
                options.phase_roles[1], .001, std::vector<R::QbmmMassFraction>{{11, .5}, {29, .5}}),
            std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "coarse",
                options.phase_roles[2], .003, std::vector<R::QbmmMassFraction>{{11, .25}, {29, .75}})};
        Flow::State state;
        state.thermodynamic_pressure = 100000;
        state.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*options.qbmm_families[1]),
            Flow::qbmm_family_stamp(*options.qbmm_families[2])};
        state.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < state.cells.size(); ++c)
        {
            auto& cell = state.cells[c];
            const auto center = mesh->cell_centroid(static_cast<LO>(c));
            const auto index = global_cell(static_cast<LO>(c));
            const double volume = mesh->cell_volume(static_cast<LO>(c));
            cell.phases.resize(3);
            cell.phase_numbers.resize(3);
            cell.qbmm_shape.resize(3);
            double gas_volume = 0;
            for (size_t a = 1; a < 3; ++a)
            {
                // Independent atom arithmetic: alternating two-node/rank-one fine cells,
                // and a common rank-one size with spatially varying weight in the coarse family.
                const double factor = input.stock_scale * volume *
                                      (a == 1 && (input.vacuum || (input.vacuum_cells && index % 2 == 1)) ? 0. : 1.);
                const double n0 = a == 1 ? (index % 2 == 0 ? 5 : 3) : 2 + index;
                const double h1 = a == 1 ? (index % 2 == 0 ? 14 : 6) : 2 * n0;
                const double h2 = a == 1 ? (index % 2 == 0 ? 50 : 12) : 4 * n0;
                const double h3 = a == 1 ? (index % 2 == 0 ? 194 : 24) : 8 * n0;
                const double mass = factor * h1 * (a == 1 ? .001 : .003);
                const double fraction = a == 1 ? .5 : .25;
                std::vector<double> moles{mass * fraction / .02, mass * (1 - fraction) / .04};
                const double total = moles[0] + moles[1];
                gas_volume += total * gas_constant * 300 / 100000;
                cell.phases[a] = {gas, moles, 100000, input.gas_reference * total, total > 0 ? 300. : 0.};
                cell.phase_numbers[a] = factor * n0;
                cell.qbmm_shape[a] = R::QbmmShapeState{factor * h2, factor * h3};
            }
            const double mass = 1000 * (volume - gas_volume);
            cell.phases[0] = {liquid, {mass / .02, 0}, 100000, mass * 1e6, 300};
            if (cells == 4)
            {
                // Circulate 1/8 m3/s on each interior face, including unequal-width cells.
                const double vertical = center.x < input.x_split ? -.125 / input.x_split : .125 / (1 - input.x_split);
                cell.velocity = Flow::Vector{.5 - center.z, 0, vertical};
            }
        }
        if (mixed)
        {
            options.use_multi_rhs_transport = batched;
            options.linear.multi_rhs_policy = mixed->policy;
            options.linear.max_batch_width = mixed->width;
            options.linear.backend = mixed->backend;
            options.qbmm_model.family_transport = mixed->family;
            options.qbmm_model.velocity = mixed->velocity;
            options.qbmm_model.linear_slope_limiter = mixed->limiter;
            options.qbmm_families[2].reset(); // Ordinary dispersed gas retains its finite count.
            state.qbmm_families[2].reset();
            if (mixed->family == FamilyTransport::KineticEuler)
            {
                R::QbmmLawMetadata slip{"mass-dependent z slip", 1, R::QbmmLawKind::NodeSlip, "mass in kg", "m/s"};
                R::QbmmClosureVersions versions;
                versions.laws.push_back(slip);
                options.qbmm_families[1] =
                    std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(), "fine",
                        options.phase_roles[1], .001, std::vector<R::QbmmMassFraction>{{11, .5}, {29, .5}}, versions);
                state.qbmm_families[1] = Flow::qbmm_family_stamp(*options.qbmm_families[1]);
                options.qbmm_node_slip_laws = {nullptr,
                    std::make_shared<const R::QbmmNodeSlipLaw>(
                        R::QbmmNodeSlipLaw{slip, [](const R::QbmmNodeSlipInput& atom)
                            { return std::array{0., 0., 2. * atom.mass_kg_per_bubble}; }}),
                    nullptr};
            }
            for (size_t c = 0; c < state.cells.size(); ++c)
            {
                auto& cell = state.cells[c];
                cell.qbmm_shape[2].reset();
                // The second gas species starts on just one cell. Preserve total gas moles and EOS volume.
                auto& ordinary = cell.phases[2];
                const double total = ordinary.moles[0] + ordinary.moles[1];
                const size_t index = global_cell(static_cast<LO>(c));
                ordinary.moles[1] = index == 0 ? .01 * total : 0.;
                ordinary.moles[0] = total - ordinary.moles[1];
                cell.phase_numbers[2] = mesh->cell_volume(static_cast<LO>(c)) * (5 - index);
                if (mixed->velocity == Velocity::SuppliedPhaseSlip && index == 1)
                    cell.drift_velocity = {{}, {0, 0, .004}, {0, 0, -.002}};
            }
        }
        flow = std::make_shared<Flow>(mesh, eos, options);
        flow->initialize(state);
    }

    size_t global_cell(LO cell) const
    {
        const auto center = mesh->cell_centroid(cell);
        if (cells == 3)
            return static_cast<size_t>(center.z * 3);
        return (center.z > .5 ? (cells == 4 ? 2 : 1) : 0) + (cells == 4 && center.x > .5 ? 1 : 0);
    }
};

std::array<double, channels> values(
    const std::vector<double>& moles, double count, const R::QbmmShapeState& shape, double energy, double scale)
{
    return {moles[0], moles[1], count, (.02 * moles[0] + .04 * moles[1]) / scale, shape.H2, shape.H3, energy};
}

void near_stock(double actual, double expected)
{
    if (expected == 0)
        EXPECT_DOUBLE_EQ(actual, 0);
    else
        EXPECT_NEAR(actual, expected, 2e-10 * std::abs(expected)); // No absolute floor for tiny stock.
}

// Independent dense elimination; coefficients are assembled only from published face rates.
std::vector<double> solve(std::vector<double> matrix, std::vector<double> rhs)
{
    const size_t n = rhs.size();
    for (size_t p = 0; p < n; ++p)
    {
        const double diagonal = matrix[p * n + p];
        for (size_t j = p; j < n; ++j)
            matrix[p * n + j] /= diagonal;
        rhs[p] /= diagonal;
        for (size_t i = 0; i < n; ++i)
            if (i != p)
            {
                const double multiplier = matrix[i * n + p];
                for (size_t j = p; j < n; ++j)
                    matrix[i * n + j] -= multiplier * matrix[p * n + j];
                rhs[i] -= multiplier * rhs[p];
            }
    }
    return rhs;
}

void audit_transport(Fixture& f, bool require_motion)
{
    const auto before = f.flow->state();
    const auto trial = f.flow->preview(dt);
    // Only the unbound liquid enters the carrier plan. The two family blocks have their own common map.
    EXPECT_EQ(trial.diagnostics().transport_multi_rhs.operator_groups, 1U);
    EXPECT_EQ(trial.diagnostics().transport_multi_rhs.transported_columns, 3U);
    const auto& snapshot = trial.transport_snapshot();
    const auto receipts = trial.owned_phase_transport();
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const size_t n = f.cells;
    int empty = before.cells.empty(), empty_total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_total);
    EXPECT_EQ(empty_total, std::max(0, comm->getSize() - int(n)));
    R::qbmm_detail::validate_state(f.options, trial.state());
    EXPECT_EQ(snapshot.qbmm_families, before.qbmm_families);
    double motion = 0;
    int crossing = 0;
    for (size_t a = 1; a < 3; ++a)
    {
        const double scale = f.options.qbmm_families[a]->mass_scale_kg();
        std::vector<double> local_old(n * channels), old(n * channels), local_new(n * channels), now(n * channels);
        std::vector<double> local_volume(n), volume(n), local_q(n * n), q(n * n);
        std::vector<double> local_flux(n * n * channels), flux(n * n * channels);
        collective_detail::collective_local_validation(*f.mesh, "QBMM receipt test layout",
            [&]
            {
                if (snapshot.cells.size() != before.cells.size() || receipts.size() != f.mesh->num_owned_faces())
                    throw std::runtime_error("QBMM receipt layout differs.");
                for (size_t c = 0; c < before.cells.size(); ++c)
                {
                    const auto i = f.global_cell(static_cast<LO>(c));
                    const auto& initial = before.cells[c];
                    const auto& transported = snapshot.cells[c].phases.at(a);
                    const auto first = values(initial.phases[a].moles, *initial.phase_numbers[a],
                        *initial.qbmm_shape[a], initial.phases[a].internal_energy, scale);
                    const auto last = values(transported.moles, *transported.population_number, *transported.qbmm_shape,
                        transported.internal_energy, scale);
                    std::copy(first.begin(), first.end(), local_old.begin() + i * channels);
                    std::copy(last.begin(), last.end(), local_new.begin() + i * channels);
                    local_volume[i] = f.mesh->cell_volume(static_cast<LO>(c));
                }
                for (size_t face = 0; face < receipts.size(); ++face)
                {
                    const auto fid = static_cast<LO>(face);
                    const auto& receipt = receipts[face];
                    const auto& packet = receipt.qbmm_flux.at(a).value();
                    const auto packets = values(packet.species_molar_flux, packet.population_number_flux,
                        packet.shape_flux, packet.internal_energy_flux, scale);
                    if (f.mesh->is_geometry_exterior_face(fid))
                    {
                        for (double value : packets)
                            if (value != 0)
                                throw std::runtime_error("Closed QBMM boundary has a nonzero packet.");
                        continue;
                    }
                    const auto owner = f.mesh->owner_cell(fid), neighbor = f.mesh->neighbor_cell(fid);
                    const auto i = f.global_cell(owner), j = f.global_cell(neighbor);
                    local_q[i * n + j] += receipt.relative_phase_volume_flux.at(a);
                    for (size_t k = 0; k < channels; ++k)
                        local_flux[(i * n + j) * channels + k] += packets[k];
                    if (receipt.relative_phase_volume_flux[a] != 0 &&
                        (size_t(owner) >= f.mesh->num_owned_cells() || size_t(neighbor) >= f.mesh->num_owned_cells()))
                        ++crossing;
                }
            });
        const auto sum = [&](const auto& local, auto& global)
        {
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(local.size()), local.data(), global.data());
        };
        sum(local_old, old);
        sum(local_new, now);
        sum(local_volume, volume);
        sum(local_q, q);
        sum(local_flux, flux);
        std::vector<double> matrix(n * n);
        for (size_t i = 0; i < n; ++i)
            matrix[i * n + i] = 1;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                const double rate = q[i * n + j];
                motion += std::abs(rate);
                const size_t donor = rate >= 0 ? i : j;
                matrix[i * n + donor] += dt * rate / volume[donor];
                matrix[j * n + donor] -= dt * rate / volume[donor];
                for (size_t k = 0; k < channels; ++k)
                    near_stock(flux[(i * n + j) * channels + k], rate * now[donor * channels + k] / volume[donor]);
            }
        for (size_t k = 0; k < channels; ++k)
        {
            std::vector<double> initial(n);
            for (size_t i = 0; i < n; ++i)
                initial[i] = old[i * channels + k];
            const auto expected = solve(matrix, initial);
            double total_before = 0, total_after = 0;
            for (size_t i = 0; i < n; ++i)
            {
                near_stock(now[i * channels + k], expected[i]);
                double outgoing = 0;
                for (size_t j = 0; j < n; ++j)
                    outgoing += flux[(i * n + j) * channels + k] - flux[(j * n + i) * channels + k];
                const double endpoint = now[i * channels + k];
                EXPECT_LE(std::abs(endpoint + dt * outgoing - initial[i]),
                    2e-10 * (std::abs(endpoint) + std::abs(initial[i]) + std::abs(dt * outgoing)));
                total_before += initial[i];
                total_after += now[i * channels + k];
            }
            near_stock(total_after, total_before);
        }
        for (size_t c = 0; c < before.cells.size(); ++c)
        {
            const auto& transported = snapshot.cells[c].phases[a];
            EXPECT_EQ(trial.state().cells[c].phases[a].moles, transported.moles);
            EXPECT_EQ(trial.state().cells[c].phase_numbers[a], transported.population_number);
            EXPECT_EQ(trial.state().cells[c].qbmm_shape[a], transported.qbmm_shape);
            const auto status = trial.qbmm_inversion_status()[c][a];
            EXPECT_TRUE(status == R::InversionStatus::Absent || status == R::InversionStatus::RankOne ||
                        status == R::InversionStatus::TwoNodes);
            if (a == 2)
                EXPECT_EQ(status, R::InversionStatus::RankOne);
        }
    }
    if (require_motion)
    {
        EXPECT_GT(motion, .01);
        int global_crossing = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &crossing, &global_crossing);
        if (comm->getSize() > 1)
            EXPECT_GT(global_crossing, 0);
    }
    EXPECT_EQ(f.flow->state().time, before.time);
    EXPECT_EQ(f.flow->state().revision, before.revision);
    for (size_t c = 0; c < before.cells.size(); ++c)
    {
        EXPECT_EQ(f.flow->state().cells[c].qbmm_shape, before.cells[c].qbmm_shape);
        EXPECT_EQ(f.flow->state().cells[c].phase_numbers, before.cells[c].phase_numbers);
    }
    f.flow->commit(trial);
    for (size_t c = 0; c < before.cells.size(); ++c)
        EXPECT_EQ(f.flow->state().cells[c].qbmm_shape, trial.state().cells[c].qbmm_shape);
}

TEST(LowMachQbmmTransportTest, ClosedMovingFamiliesMatchIndependentImplicitMapAndPublishedReceipts)
{
    Fixture f;
    audit_transport(f, true);
}

TEST(LowMachQbmmTransportTest, MovingTinyPositiveStocksUseRelativeGatesWithoutInventedVacuum)
{
    Fixture f({.stock_scale = 1e-30});
    audit_transport(f, true);
}

TEST(LowMachQbmmTransportTest, NegativeGasInternalEnergyUsesTheCommonMapAcrossUnequalCellVolumes)
{
    Fixture f({.x_split = .25, .gas_reference = -6000});
    for (size_t c = 0; c < f.flow->state().cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        EXPECT_DOUBLE_EQ(f.mesh->cell_volume(cell), f.global_cell(cell) % 2 == 0 ? .125 : .375);
        for (size_t a = 1; a < 3; ++a)
            EXPECT_LT(f.flow->state().cells[c].phases[a].internal_energy, 0);
    }
    audit_transport(f, true);
}

TEST(LowMachQbmmTransportTest, ExactVacuumFamilyRemainsAbsentBesideMovingPositiveFamily)
{
    Fixture f({.vacuum = true});
    audit_transport(f, true);
    for (const auto& cell : f.flow->state().cells)
    {
        EXPECT_EQ(cell.phases[1].moles, (std::vector<double>{0, 0}));
        EXPECT_DOUBLE_EQ(*cell.phase_numbers[1], 0);
        EXPECT_EQ(*cell.qbmm_shape[1], R::QbmmShapeState{});
    }
}

TEST(LowMachQbmmTransportTest, MovingTinyPopulationFillsInitiallyExactVacuumCells)
{
    Fixture f({.stock_scale = 1e-30, .vacuum_cells = true});
    for (size_t c = 0; c < f.flow->state().cells.size(); ++c)
        if (f.global_cell(static_cast<LO>(c)) % 2 == 1)
        {
            const auto& cell = f.flow->state().cells[c];
            EXPECT_EQ(cell.phases[1].moles, (std::vector<double>{0, 0}));
            EXPECT_DOUBLE_EQ(*cell.phase_numbers[1], 0);
            EXPECT_EQ(*cell.qbmm_shape[1], R::QbmmShapeState{});
        }
    audit_transport(f, true);
    for (const auto& cell : f.flow->state().cells)
    {
        EXPECT_GT(*cell.phase_numbers[1], 0);
        EXPECT_GT(cell.qbmm_shape[1]->H2, 0);
        EXPECT_GT(cell.qbmm_shape[1]->H3, 0);
    }
}

TEST(LowMachQbmmTransportTest, TwoCellsIncludeActualEmptyOwnersWithTinyStocksAndVacuum)
{
    Fixture f({.cells = 2, .stock_scale = 1e-30, .vacuum = true});
    audit_transport(f, false);
}

TEST(LowMachQbmmTransportTest, SubnormalFacePacketsRequireRelativePrecisionAndRetainExactRates)
{
    constexpr double quantum = std::numeric_limits<double>::denorm_min();
    constexpr double rate = 8 * quantum;
    for (const bool exactly_representable : {false, true})
    {
        SCOPED_TRACE(exactly_representable);
        Fixture f({.cells = 2});
        auto input = f.flow->state();
        for (size_t c = 0; c < input.cells.size(); ++c)
        {
            auto& cell = input.cells[c];
            cell.velocity = Flow::Vector{0, 0, rate};
            if (!exactly_representable)
                continue;

            // A dyadic 50:50 mass mixture and one size-five node. V=1/2 makes
            // the species donor traces 1/2 and 1/4, so their face rates are exact.
            cell.phases[1].moles = {.25, .125};
            cell.phases[1].internal_energy = 6000 * .375;
            cell.phase_numbers[1] = 2.;
            cell.qbmm_shape[1] = R::QbmmShapeState{50, 250};
            cell.phases[2].moles = {0, 0};
            cell.phases[2].internal_energy = 0;
            cell.phases[2].temperature = 0;
            cell.phase_numbers[2] = 0.;
            cell.qbmm_shape[2] = R::QbmmShapeState{};
            const double volume = f.mesh->cell_volume(static_cast<LO>(c));
            const double liquid_mass = 1000 * (volume - .375 * gas_constant * 300 / 100000);
            cell.phases[0].moles = {liquid_mass / .02, 0};
            cell.phases[0].internal_energy = liquid_mass * 1e6;
        }
        f.flow->initialize(input);
        bool owns_carrier_face = false;
        for (const LO face : f.flow->volume_flux().owned_face_ids())
            if (!f.mesh->is_geometry_exterior_face(face))
            {
                owns_carrier_face = true;
                EXPECT_EQ(std::abs(f.flow->volume_flux().owned_value(face)), rate);
            }
        const auto accepted = f.flow->serialize_qbmm_restart();
        if (!exactly_representable)
        {
            // The original lower cell has species traces about .35 and .175.
            // Ordinary rounding gives 3 and 1 quanta, a 60:40 mass packet,
            // even though the donor has 50:50 mass and all scalar rates are nonzero.
            std::string error;
            try
            {
                (void) f.flow->preview(dt);
            }
            catch (const std::exception& exception)
            {
                error = exception.what();
            }
            EXPECT_EQ(error, owns_carrier_face ? "QBMM canonical face packet loses relative precision."
                                               : "QBMM common canonical packets failed on another rank.");
            EXPECT_EQ(f.flow->serialize_qbmm_restart(), accepted);
            continue;
        }
        const auto trial = f.flow->preview(dt);
        const auto receipts = trial.owned_phase_transport();
        for (size_t fidx = 0; fidx < receipts.size(); ++fidx)
            if (!f.mesh->is_geometry_exterior_face(static_cast<LO>(fidx)))
            {
                const auto& receipt = receipts[fidx];
                const auto& packet = *receipt.qbmm_flux[1];
                EXPECT_EQ(std::abs(receipt.relative_phase_volume_flux[1]), rate);
                EXPECT_EQ(std::abs(packet.species_molar_flux[0]), 4 * quantum);
                EXPECT_EQ(std::abs(packet.species_molar_flux[1]), 2 * quantum);
                EXPECT_EQ(std::abs(packet.population_number_flux), 32 * quantum);
                EXPECT_EQ(std::abs(packet.shape_flux.H2), 800 * quantum);
                EXPECT_EQ(std::abs(packet.shape_flux.H3), 4000 * quantum);
                EXPECT_EQ(std::abs(packet.internal_energy_flux), 36000 * quantum);
            }
        EXPECT_EQ(f.flow->serialize_qbmm_restart(), accepted);
    }
}

TEST(LowMachQbmmTransportTest, SubnormalCounterflowRoundsPhaseAndNativeMomentumPacketsIndependently)
{
    constexpr double quantum = std::numeric_limits<double>::denorm_min();
    constexpr double pressure = 100000, temperature = 300, molar_mass = .25, gas_energy = 6000;
    Flow::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    const auto geometry = std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}});
    const auto mesh = std::make_shared<Flow::Mesh>(geometry, distribution);
    const auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {11, "A", molar_mass, {{"X", 1}}, {R::SpeciesPhase::Liquid, R::SpeciesPhase::Dispersed}}});
    const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
    auto liquid = R::make_linear_compressible_mixture_liquid(catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; });
    auto gas = R::make_ideal_gas_mixture(catalog, {{20, temperature, gas_energy}}, bounds);
    Flow::Options options;
    options.minimum_pressure = 20000;
    options.maximum_pressure = 1000000;
    options.linear.preconditioner = LinearPreconditioner::None;
    options.linear.tolerance = 1e-12;
    options.qbmm_model.velocity = R::QbmmModelOptions::Velocity::SuppliedPhaseSlip;
    options.phase_roles = {{Flow::PhaseRole::BulkLiquid, "liquid"}, {Flow::PhaseRole::DispersedGas, "fine"},
        {Flow::PhaseRole::DispersedGas, "coarse"}};
    options.qbmm_families.resize(3);
    for (size_t phase = 1; phase < 3; ++phase)
        options.qbmm_families[phase] = std::make_shared<const R::QbmmPopulationSchema>(catalog, catalog->fingerprint(),
            options.phase_roles[phase].identity, options.phase_roles[phase], .125,
            std::vector<R::QbmmMassFraction>{{11, 1}});

    for (const double direction : {1., -1.})
    {
        SCOPED_TRACE(direction);
        Flow flow(mesh, {liquid, gas, gas}, options);
        Flow::State input;
        input.thermodynamic_pressure = pressure;
        input.qbmm_families = {std::nullopt, Flow::qbmm_family_stamp(*options.qbmm_families[1]),
            Flow::qbmm_family_stamp(*options.qbmm_families[2])};
        input.cells.resize(mesh->num_owned_cells());
        for (auto& cell : input.cells)
        {
            // Each gas has exactly one mole per bulk cubic metre and one unit-size node per cell.
            const double liquid_mass = 1000 * (.5 - gas_constant * temperature / pressure);
            cell.phases = {{liquid, {liquid_mass / molar_mass}, pressure, liquid_mass * 1e6, temperature},
                {gas, {.5}, pressure, .5 * gas_energy, temperature},
                {gas, {.5}, pressure, .5 * gas_energy, temperature}};
            cell.phase_numbers = {std::nullopt, 1., 1.};
            cell.qbmm_shape = {std::nullopt, R::QbmmShapeState{1, 1}, R::QbmmShapeState{1, 1}};
            cell.drift_velocity = {{}, {0, 8, direction * quantum}, {0, -8, -direction * 3 * quantum}};
        }
        flow.initialize(std::move(input));
        const auto trial = flow.preview(dt);
        const auto receipts = trial.owned_phase_transport();
        int local_interior = 0, global_interior = 0;
        for (size_t f = 0; f < receipts.size(); ++f)
        {
            const LO face = static_cast<LO>(f);
            if (mesh->is_geometry_exterior_face(face))
                continue;
            ++local_interior;
            const auto& receipt = receipts[f];
            const auto& fine = receipt.qbmm_flux.at(1).value();
            const auto& coarse = receipt.qbmm_flux.at(2).value();
            const double signed_quantum = direction * mesh->face_area_vector(face).z * quantum;
            EXPECT_EQ(std::abs(signed_quantum), quantum);
            EXPECT_EQ(receipt.relative_phase_volume_flux[0], 0);
            EXPECT_EQ(receipt.relative_phase_volume_flux[1], signed_quantum);
            EXPECT_EQ(receipt.relative_phase_volume_flux[2], -3 * signed_quantum);
            EXPECT_EQ(fine.species_molar_flux.at(0), signed_quantum);
            EXPECT_EQ(coarse.species_molar_flux.at(0), -3 * signed_quantum);
            // In units of quantum the phase mass rates are +1/4 and -3/4.
            EXPECT_EQ(fine.derived_mass_flux, 0);
            EXPECT_EQ(coarse.derived_mass_flux, -signed_quantum);
            EXPECT_EQ(fine.derived_drift_momentum_flux, (Flow::Vector{0, 2 * signed_quantum, 0}));
            EXPECT_EQ(coarse.derived_drift_momentum_flux, (Flow::Vector{0, 6 * signed_quantum, 0}));
            // The unrounded total is a half-quantum tie, which rounds to the even value zero.
            EXPECT_EQ(receipt.momentum_mass_flux, 0);
            EXPECT_EQ(receipt.drift_momentum_flux, (Flow::Vector{0, 8 * signed_quantum, 0}));
        }
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_interior, &global_interior);
        EXPECT_EQ(global_interior, 1);
    }
}

TEST(LowMachQbmmTransportTest, ScaledFaceArithmeticRetainsCancellationAndRoundsAtBinary64Boundaries)
{
    using FaceSum = R::qbmm_face_detail::ScaledFaceSum;
    constexpr double q = std::numeric_limits<double>::denorm_min();
    constexpr double max = std::numeric_limits<double>::max();
    for (const double direction : {1., -1.})
    {
        SCOPED_TRACE(direction);
        FaceSum mass, drift;
        mass.add_product(direction * q, .25);
        drift.add_weighted(mass, 8.);
        EXPECT_EQ(mass.rounded(), 0);
        EXPECT_EQ(drift.rounded(), direction * 2 * q);

        FaceSum tiny, huge, opposite_half, total;
        tiny.add_product(direction * q, 1.);
        huge.add_product(direction * std::scalbn(1., 1000), std::scalbn(1., 897));
        opposite_half.add_product(-direction * std::scalbn(1., 1000), std::scalbn(1., 896));
        EXPECT_EQ(huge.rounded(), direction * std::numeric_limits<double>::infinity());
        total.add_weighted(huge, 1.);
        total.add_weighted(tiny, 1.);
        total.add_weighted(opposite_half, 1.);
        total.add_weighted(opposite_half, 1.);
        EXPECT_EQ(total.rounded(), direction * q);

        FaceSum three_scales;
        for (const double term : {1., std::scalbn(1., -70), std::scalbn(1., -140), -1., -std::scalbn(1., -70)})
            three_scales.add_product(direction, term);
        EXPECT_EQ(three_scales.rounded(), direction * std::scalbn(1., -140));

        FaceSum partial_cancellation;
        const double remainder = std::scalbn(1., -10);
        const double low_order = std::scalbn(1., -56);
        partial_cancellation.add_product(direction, 1.);
        partial_cancellation.add_product(direction, low_order);
        partial_cancellation.add_product(-direction, 1. - remainder);
        EXPECT_EQ(partial_cancellation.rounded(), direction * (remainder + low_order));
        EXPECT_EQ(partial_cancellation.rounded(3.), direction * ((remainder + low_order) / 3.));

        FaceSum product_residual;
        constexpr double delta = std::numeric_limits<double>::epsilon();
        product_residual.add_product(direction * (1. + delta), 1. - delta);
        product_residual.add_product(-direction, 1.);
        EXPECT_EQ(product_residual.rounded(), -direction * delta * delta);

        FaceSum tie;
        tie.add_product(direction * q, .5);
        auto above = tie, below = tie;
        above.add_product(direction * q, q);
        below.add_product(-direction * q, q);
        EXPECT_EQ(tie.rounded(), 0);
        EXPECT_EQ(above.rounded(), direction * q);
        EXPECT_EQ(below.rounded(), 0);

        FaceSum normal_tie;
        normal_tie.add_product(direction, 1.);
        normal_tie.add_product(direction, std::scalbn(1., -53));
        auto normal_above = normal_tie, normal_below = normal_tie;
        normal_above.add_product(direction * q, q);
        normal_below.add_product(-direction * q, q);
        EXPECT_EQ(normal_tie.rounded(), direction);
        EXPECT_EQ(normal_above.rounded(), direction * std::nextafter(1., 2.));
        EXPECT_EQ(normal_below.rounded(), direction);

        FaceSum divided_tie;
        divided_tie.add_weighted(normal_tie, 3.);
        auto divided_above = divided_tie, divided_below = divided_tie;
        divided_above.add_product(direction * q, q);
        divided_below.add_product(-direction * q, q);
        EXPECT_EQ(divided_tie.rounded(3.), direction);
        EXPECT_EQ(divided_above.rounded(3.), direction * std::nextafter(1., 2.));
        EXPECT_EQ(divided_below.rounded(3.), direction);

        FaceSum overflow_tie;
        overflow_tie.add_product(direction, max);
        overflow_tie.add_product(direction, std::scalbn(1., 970));
        auto overflow_below = overflow_tie;
        overflow_below.add_product(-direction * q, q);
        EXPECT_EQ(overflow_tie.rounded(), direction * std::numeric_limits<double>::infinity());
        EXPECT_EQ(overflow_below.rounded(), direction * max);
    }
}

TEST(LowMachQbmmTransportTest, ScaledMaterialVolumeRetainsFluxBelowTheFractionFloor)
{
    using namespace R::qbmm_face_detail;
    constexpr double q = std::numeric_limits<double>::denorm_min();
    ScaledFaceSum tiny;
    tiny.add_product(q, .25);
    const auto [zero_fraction, recovered_flux] = material_volume_rates(tiny, 4., -32.);
    EXPECT_EQ(zero_fraction, 0.);
    EXPECT_EQ(recovered_flux, -2 * q);

    ScaledFaceSum normal;
    normal.add_product(1., 3.);
    const auto [normal_fraction, normal_flux] = material_volume_rates(normal, 7., 3.);
    EXPECT_EQ(normal_fraction, 3. / 7.);
    EXPECT_EQ(normal_flux, 3. * normal_fraction);
}

TEST(LowMachQbmmTransportTest, ScaledRatioRetainsTheFullMassDenominator)
{
    using R::qbmm_face_detail::ScaledFaceSum;
    constexpr double max = std::numeric_limits<double>::max();
    constexpr double q = std::numeric_limits<double>::denorm_min();
    const double tail = std::scalbn(1., -53);
    ScaledFaceSum mass;
    mass.add_product(1., 1.);
    mass.add_product(1., tail);
    ASSERT_EQ(mass.rounded(), 1.);

    for (const double direction : {1., -1.})
    {
        SCOPED_TRACE(direction);
        ScaledFaceSum momentum;
        momentum.add_product(direction, max);
        momentum.add_product(direction * max, tail);
        EXPECT_EQ(momentum.rounded(mass.rounded()), direction * std::numeric_limits<double>::infinity());
        EXPECT_EQ(momentum.rounded(mass), direction * max);

        // (1 + tail)^2 / (1 + tail) is exactly the normal midpoint.
        ScaledFaceSum normal_tie;
        normal_tie.add_product(direction, 1.);
        normal_tie.add_product(direction, 2 * tail);
        normal_tie.add_product(direction * tail, tail);
        auto normal_above = normal_tie;
        normal_above.add_product(direction * q, q);
        EXPECT_EQ(normal_tie.rounded(mass), direction);
        EXPECT_EQ(normal_above.rounded(mass), direction * std::nextafter(1., 2.));

        // (q/2)(1 + tail) / (1 + tail) is exactly the zero-to-q midpoint.
        ScaledFaceSum tiny_tie;
        tiny_tie.add_product(direction * q, .5);
        tiny_tie.add_product(direction * q, .5, tail);
        auto tiny_above = tiny_tie;
        tiny_above.add_product(direction * q, q);
        EXPECT_EQ(tiny_tie.rounded(mass), 0.);
        EXPECT_EQ(tiny_above.rounded(mass), direction * q);
    }
}

TEST(LowMachQbmmTransportTest, DifferentPositiveConservativeScalarMapFailsTheProductionBlockGate)
{
    Fixture f({.cells = 2});
    const auto retained = f.flow->preview(dt);
    auto mismatched = retained.state();
    double local_before = 0, local_after = 0, before = 0, after = 0;
    for (size_t c = 0; c < mismatched.cells.size(); ++c)
    {
        auto& cell = mismatched.cells[c];
        local_before += cell.qbmm_shape[2]->H2;
        // H2 alone uses a positive conservative permutation of the two donor stocks:
        // [4,6] -> [6,4]. Species/H0/H3 retain their actual identity transport map.
        cell.qbmm_shape[2]->H2 = f.global_cell(static_cast<LO>(c)) == 0 ? 6 : 4;
        local_after += cell.qbmm_shape[2]->H2;
        EXPECT_GT(*cell.phase_numbers[2], 0);
        EXPECT_GT(cell.qbmm_shape[2]->H2, 0);
        EXPECT_GT(cell.qbmm_shape[2]->H3, 0);
    }
    const auto comm = f.mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_before, &before);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_after, &after);
    EXPECT_DOUBLE_EQ(before, after);
    EXPECT_THROW(collective_detail::collective_local_validation(*f.mesh, "Mismatched transported QBMM block",
                     [&] { R::qbmm_detail::validate_transport(f.options, mismatched); }),
        std::exception);
    EXPECT_NO_THROW(f.flow->commit(retained));
}

TEST(LowMachQbmmTransportTest, NativePreviewRejectsUnresolvedArrivalOnOneOwnerAndPreservesRetainedTrial)
{
    Fixture f({.cells = 2});
    auto state = f.flow->state();
    bool owns_receiver = false;
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto cell_id = static_cast<LO>(c);
        auto& cell = state.cells[c];
        const bool donor = f.global_cell(cell_id) == 0;
        owns_receiver |= !donor;
        const double count = donor ? 0x1p-48 : 1.;
        const double size = donor ? 0x1p12 : 1.;
        const double mass = .001 * count * size;
        auto& fine = cell.phases[1];
        fine.moles = {mass * .5 / .02, mass * .5 / .04};
        fine.internal_energy = 6000 * (fine.moles[0] + fine.moles[1]);
        cell.phase_numbers[1] = count;
        cell.qbmm_shape[1] = R::QbmmShapeState{count * size * size, count * size * size * size};
        const auto& coarse = cell.phases[2];
        const double gas_moles = fine.moles[0] + fine.moles[1] + coarse.moles[0] + coarse.moles[1];
        const double liquid_mass = 1000 * (f.mesh->cell_volume(cell_id) - gas_moles * gas_constant * 300 / 100000);
        cell.phases[0].moles = {liquid_mass / .02, 0};
        cell.phases[0].internal_energy = liquid_mass * 1e6;
        cell.velocity = Flow::Vector{0, 0, 0x1p-27};
    }
    f.flow->initialize(state);
    const auto accepted = f.flow->serialize_qbmm_restart();
    const auto retained = f.flow->preview(0x1p-14);

    // On the first BE solve at dt=1, the incoming fraction is C/(1+C), C=2^-26.
    // The unit-size receiver gains about 2^-50 in H2 and 2^-38 in H3.
    // Thus the variance is unresolved in binary64 while H3 rules out a single atom.
    // Both accepted cells were monodisperse. Only the receiver fails inversion.
    std::string error;
    try
    {
        (void) f.flow->preview(1.);
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
    }
    if (owns_receiver)
        EXPECT_EQ(error, "Native QBMM family 'fine' failed inversion (IllConditioned).");
    else
        EXPECT_EQ(error, "Native represented QBMM transport block failed on another rank.");
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int local[2]{owns_receiver, state.cells.empty()}, totals[2]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, totals);
    EXPECT_EQ(totals[0], 1);
    EXPECT_EQ(totals[1], std::max(0, comm->getSize() - 2));

    // The checkpoint covers accepted material, moments, time, revision, face flux, and diagnostics.
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), accepted);
    EXPECT_NO_THROW(f.flow->commit(retained));
    EXPECT_EQ(f.flow->state().revision, state.revision + 1);
}

void expect_carrier_value(double value, double reference)
{
    EXPECT_NEAR(value, reference, 2e-10 * std::abs(reference) + 1e-12);
}

void expect_carrier_vector(const Flow::Vector& value, const Flow::Vector& reference)
{
    for (int axis = 0; axis < 3; ++axis)
        expect_carrier_value(value.component(axis), reference.component(axis));
}

void expect_carrier_array(std::span<const double> value, std::span<const double> reference)
{
    ASSERT_EQ(value.size(), reference.size());
    for (size_t i = 0; i < value.size(); ++i)
        expect_carrier_value(value[i], reference[i]);
}

void expect_carrier_trial(const Flow::Trial& trial, const Flow::Trial& reference)
{
    const auto& state = trial.state();
    const auto& expected = reference.state();
    expect_carrier_value(state.thermodynamic_pressure, expected.thermodynamic_pressure);
    const auto& diagnostics = trial.diagnostics();
    const auto& expected_diagnostics = reference.diagnostics();
    EXPECT_NEAR(diagnostics.caloric_energy_scale, expected_diagnostics.caloric_energy_scale,
        2e-10 * expected_diagnostics.caloric_energy_scale);
    EXPECT_NEAR(diagnostics.internal_energy_roundoff_bound, expected_diagnostics.internal_energy_roundoff_bound,
        2e-10 * expected_diagnostics.internal_energy_roundoff_bound);
    EXPECT_EQ(state.revision, expected.revision);
    EXPECT_EQ(state.time, expected.time);
    EXPECT_EQ(state.qbmm_families, expected.qbmm_families);
    EXPECT_EQ(trial.qbmm_inversion_status(), reference.qbmm_inversion_status());
    const auto& snapshot = trial.transport_snapshot();
    const auto& old = reference.transport_snapshot();
    EXPECT_EQ(snapshot.old_control_volumes, old.old_control_volumes);
    EXPECT_EQ(snapshot.new_control_volumes, old.new_control_volumes);
    ASSERT_EQ(state.cells.size(), expected.cells.size());
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        const auto& cell = state.cells[c];
        const auto& other = expected.cells[c];
        expect_carrier_vector(cell.velocity, other.velocity);
        EXPECT_EQ(cell.drift_velocity, other.drift_velocity);
        // Pressure perturbation is gauge-pinned, including cells whose true perturbation vanishes.
        EXPECT_NEAR(
            cell.mechanical_pressure, other.mechanical_pressure, 1e-8 + 2e-10 * std::abs(other.mechanical_pressure));
        for (size_t a = 0; a < 3; ++a)
        {
            SCOPED_TRACE(::testing::Message() << "owned cell " << c << ", phase " << a);
            const auto& phase = cell.phases.at(a);
            const auto& previous = other.phases.at(a);
            expect_carrier_array(phase.moles, previous.moles);
            expect_carrier_value(phase.internal_energy, previous.internal_energy);
            expect_carrier_value(phase.temperature, previous.temperature);
            expect_carrier_value(phase.absolute_pressure, previous.absolute_pressure);
            const auto& inventory = snapshot.cells.at(c).phases.at(a);
            const auto& ref = old.cells.at(c).phases.at(a);
            expect_carrier_array(inventory.moles, ref.moles);
            expect_carrier_value(inventory.internal_energy, ref.internal_energy);
            expect_carrier_value(inventory.advected_material_volume, ref.advected_material_volume);
            ASSERT_EQ(cell.phase_numbers.at(a).has_value(), a != 0);
            ASSERT_EQ(inventory.population_number.has_value(), a != 0);
            ASSERT_EQ(cell.qbmm_shape.at(a).has_value(), a == 1);
            ASSERT_EQ(inventory.qbmm_shape.has_value(), a == 1);
            if (a != 0)
            {
                expect_carrier_value(*cell.phase_numbers[a], other.phase_numbers[a].value());
                expect_carrier_value(*inventory.population_number, ref.population_number.value());
            }
            if (a == 1)
            {
                expect_carrier_value(cell.qbmm_shape[a]->H2, other.qbmm_shape[a].value().H2);
                expect_carrier_value(cell.qbmm_shape[a]->H3, other.qbmm_shape[a].value().H3);
                expect_carrier_value(inventory.qbmm_shape->H2, ref.qbmm_shape.value().H2);
                expect_carrier_value(inventory.qbmm_shape->H3, ref.qbmm_shape.value().H3);
            }
            if (a == 0)
                EXPECT_DOUBLE_EQ(inventory.moles[1], 0.);
        }
    }
    expect_carrier_array(trial.owned_volume_flux(), reference.owned_volume_flux());
    const auto faces = trial.owned_phase_transport(), others = reference.owned_phase_transport();
    ASSERT_EQ(faces.size(), others.size());
    for (size_t f = 0; f < faces.size(); ++f)
    {
        const auto& face = faces[f];
        const auto& other = others[f];
        expect_carrier_array(face.relative_phase_volume_flux, other.relative_phase_volume_flux);
        expect_carrier_array(face.material_fraction, other.material_fraction);
        expect_carrier_value(face.material_volume_flux, other.material_volume_flux);
        expect_carrier_value(face.momentum_mass_flux, other.momentum_mass_flux);
        expect_carrier_vector(face.drift_momentum_flux, other.drift_momentum_flux);
        expect_carrier_vector(face.implicit_carrier_momentum_flux, other.implicit_carrier_momentum_flux);
        ASSERT_EQ(face.qbmm_flux.size(), 3u);
        EXPECT_FALSE(face.qbmm_flux[0]);
        EXPECT_FALSE(face.qbmm_flux[2]);
        const auto& packet = face.qbmm_flux[1].value();
        const auto& ref = other.qbmm_flux[1].value();
        expect_carrier_array(packet.species_molar_flux, ref.species_molar_flux);
        expect_carrier_value(packet.population_number_flux, ref.population_number_flux);
        expect_carrier_value(packet.shape_flux.H2, ref.shape_flux.H2);
        expect_carrier_value(packet.shape_flux.H3, ref.shape_flux.H3);
        expect_carrier_value(packet.internal_energy_flux, ref.internal_energy_flux);
        expect_carrier_value(packet.material_volume_flux, ref.material_volume_flux);
        expect_carrier_value(packet.derived_mass_flux, ref.derived_mass_flux);
        expect_carrier_vector(packet.derived_drift_momentum_flux, ref.derived_drift_momentum_flux);
        ASSERT_EQ(packet.kinetic.has_value(), ref.kinetic.has_value());
        if (packet.kinetic)
            for (auto direction : {&Flow::QbmmKineticFaceFlux::forward, &Flow::QbmmKineticFaceFlux::backward})
            {
                const auto& actual = (*packet.kinetic).*direction;
                const auto& expected_packet = (*ref.kinetic).*direction;
                expect_carrier_array(actual.moment_flux, expected_packet.moment_flux);
                expect_carrier_array(actual.species_molar_flux, expected_packet.species_molar_flux);
                expect_carrier_value(actual.integrated_count, expected_packet.integrated_count);
                expect_carrier_value(actual.integrated_mass, expected_packet.integrated_mass);
            }
    }
}

// Independent backward-Euler solution for the numbered unbound gas. Assemble from published
// owner-oriented relative rates and trial volumes, in geometry order independent of the MPI layout.
void expect_unbound_gas_balance(const Fixture& f, const Flow::Trial& trial)
{
    const size_t n = f.cells;
    std::vector<double> local(n * (n + 4)), global(local.size());
    const auto& snapshot = trial.transport_snapshot();
    for (size_t c = 0; c < f.mesh->num_owned_cells(); ++c)
    {
        const size_t i = f.global_cell(static_cast<LO>(c));
        const auto& cell = f.flow->state().cells[c];
        local[i * n + i] = snapshot.new_control_volumes[c];
        local[n * n + i] = cell.phases[2].moles[0];
        local[n * (n + 1) + i] = cell.phases[2].moles[1];
        local[n * (n + 2) + i] = *cell.phase_numbers[2];
        local[n * (n + 3) + i] = cell.phases[2].internal_energy;
    }
    for (size_t fidx = 0; fidx < f.mesh->num_owned_faces(); ++fidx)
    {
        const auto face = static_cast<LO>(fidx);
        if (f.mesh->is_geometry_exterior_face(face))
            continue;
        const size_t i = f.global_cell(f.mesh->owner_cell(face)), j = f.global_cell(f.mesh->neighbor_cell(face));
        const double transported_volume = dt * trial.owned_phase_transport()[fidx].relative_phase_volume_flux[2];
        const size_t donor = transported_volume >= 0 ? i : j;
        local[i * n + donor] += transported_volume;
        local[j * n + donor] -= transported_volume;
    }
    Teuchos::reduceAll(
        *f.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    const std::vector<double> matrix(global.begin(), global.begin() + n * n);
    for (size_t k = 0; k < 4; ++k)
    {
        const auto begin = global.begin() + n * (n + k);
        const auto concentration = solve(matrix, {begin, begin + n});
        for (size_t c = 0; c < snapshot.cells.size(); ++c)
        {
            const auto& gas = snapshot.cells[c].phases[2];
            const std::array amounts{gas.moles[0], gas.moles[1], *gas.population_number, gas.internal_energy};
            near_stock(amounts[k], concentration[f.global_cell(static_cast<LO>(c))] * snapshot.new_control_volumes[c]);
        }
    }
}

void check_mixed_carriers(MixedCarrierCase controls, unsigned cells = 4)
{
    SCOPED_TRACE(::testing::Message() << "family " << static_cast<int>(controls.family) << ", ALE " << controls.ale
                                      << ", backend " << to_string(controls.backend) << ", policy "
                                      << to_string(controls.policy) << ", width " << controls.width);
    const FixtureOptions setup{.cells = cells, .gas_reference = -6000};
    Fixture scalar(setup, controls, false), multi(setup, controls);
    Flow::PlanarMotion scalar_motion(scalar.mesh), multi_motion(multi.mesh);
    const auto preview = [&](Fixture& f, Flow::PlanarMotion& motion)
    { return controls.ale ? f.flow->preview_ale(dt, motion, 1.00001) : f.flow->preview(dt); };
    const auto reference = preview(scalar, scalar_motion);
    const auto trial = preview(multi, multi_motion);
    expect_carrier_trial(trial, reference);
    expect_unbound_gas_balance(multi, trial);
    const auto& diag = trial.diagnostics();
    EXPECT_EQ(diag.family_transport, controls.family);
    EXPECT_EQ(diag.kinetic_substeps > 0, controls.family == FamilyTransport::KineticEuler);
    EXPECT_EQ(diag.linear_substeps > 0, controls.family == FamilyTransport::RealizableLinearSSPRK2);
    EXPECT_LE(diag.maximum_population_transport_residual, 1.);
    EXPECT_LE(diag.maximum_species_transport_residual, 1.);
    EXPECT_LE(diag.maximum_energy_transport_residual, 1.);
    EXPECT_LE(diag.maximum_shape_transport_residual, 1.);
    EXPECT_EQ(reference.diagnostics().transport_multi_rhs.operator_groups, 0u);
    const auto& block = diag.qbmm_block_transport;
    EXPECT_EQ(block.families, controls.family == FamilyTransport::NativeImplicit ? 1u : 0u);
    EXPECT_EQ(block.halo_cell_imports, block.sweeps + block.families);
    EXPECT_EQ(block.halo_face_imports, block.sweeps + block.families);
    const auto& plan = diag.transport_multi_rhs;
    // Two ordinary carriers: [nA, nB=0, U] and [nA, nB, N, U]. No family channels.
    EXPECT_EQ(plan.operator_groups, 2u);
    EXPECT_EQ(plan.matrix_assemblies, 2u);
    EXPECT_EQ(plan.transported_columns, 7u);
    EXPECT_EQ(plan.active_columns, 6u);
    EXPECT_EQ(plan.column_consistency_sweeps.size(), 7u);
    const bool serial = controls.policy == MultiRhsExecutionPolicy::ColumnSerial || controls.width == 1;
    EXPECT_EQ(plan.execution_mode, serial ? MultiRhsExecutionMode::ColumnSerial : MultiRhsExecutionMode::BatchedKrylov);
    EXPECT_EQ(plan.batched_krylov_solves, serial ? 0u : 2u);
    EXPECT_EQ(plan.column_serial_solves, serial ? 2u : 0u);
    EXPECT_EQ(plan.max_batch_width, serial ? 1 : controls.width == 0 ? 4 : controls.width);
    if (!serial)
        EXPECT_LT(plan.operator_apply_calls, plan.operator_column_applies);
    const auto comm = multi.mesh->owned_cell_map()->getComm();
    if (const char* expected = std::getenv("SIMPLEFLUID_MIXED_CARRIER_RANKS"))
        EXPECT_EQ(comm->getSize(), std::stoi(expected));
    int local[3]{int(multi.mesh->num_owned_cells() == 0), 0, 0}, totals[3]{};
    for (size_t f = 0; f < multi.mesh->num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        if (multi.mesh->is_geometry_exterior_face(face))
            continue;
        const auto& receipt = trial.owned_phase_transport()[f];
        if (std::abs(receipt.relative_phase_volume_flux[2]) > 1e-12)
        {
            ++local[1];
            if (size_t(multi.mesh->owner_cell(face)) >= multi.mesh->num_owned_cells() ||
                size_t(multi.mesh->neighbor_cell(face)) >= multi.mesh->num_owned_cells())
                ++local[2];
        }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local, totals);
    EXPECT_EQ(totals[0], std::max(0, comm->getSize() - int(cells)));
    if (cells == 2 && !controls.ale)
        EXPECT_EQ(totals[1], 0); // Stationary control used for the symmetric CG operator.
    else
    {
        EXPECT_GT(totals[1], 0);
        if (comm->getSize() > 1)
            EXPECT_GT(totals[2], 0);
    }
    if (controls.ale)
        for (size_t c = 0; c < multi.mesh->num_owned_cells(); ++c)
            EXPECT_GT(
                trial.transport_snapshot().new_control_volumes[c], trial.transport_snapshot().old_control_volumes[c]);
}

TEST(QbmmMixedCarrierTransport, NativeSharedAndSuppliedSlipOnFixedAndPlanarGeometry)
{
    for (const bool ale : {false, true})
    {
        check_mixed_carriers({.ale = ale});
        // Only the middle cell has slip, so both end walls have zero normal drift.
        check_mixed_carriers({.velocity = Velocity::SuppliedPhaseSlip, .width = 2, .ale = ale}, 3);
    }
}

TEST(QbmmMixedCarrierTransport, KineticNodeSlipAndLinearLimitersUseOnlyUnboundColumns)
{
    check_mixed_carriers({.family = FamilyTransport::KineticEuler, .velocity = Velocity::NodeDependentSlip});
    for (const bool ale : {false, true})
        for (const auto limiter : {SlopeLimiter::RealizabilityOnly, SlopeLimiter::TvdMc})
            check_mixed_carriers({.family = FamilyTransport::RealizableLinearSSPRK2, .limiter = limiter, .ale = ale});
}

TEST(QbmmMixedCarrierTransport, KrylovPoliciesAndWidthsMatchScalarTransport)
{
    check_mixed_carriers({.backend = LinearSolverBackend::Cg}, 2);
    for (const auto backend : {LinearSolverBackend::Gmres, LinearSolverBackend::BiCGStab})
    {
        check_mixed_carriers({.backend = backend, .policy = MultiRhsExecutionPolicy::BatchedKrylov, .width = 2});
        check_mixed_carriers({.backend = backend, .policy = MultiRhsExecutionPolicy::ColumnSerial});
        check_mixed_carriers({.backend = backend, .width = 1});
    }
}

TEST(QbmmMixedCarrierTransport, EmptyOwnersParticipateInPlanarTransportAndControlValidation)
{
    check_mixed_carriers({.ale = true}, 2);
    Fixture f({.cells = 2}, MixedCarrierCase{});
    const auto comm = f.mesh->owned_cell_map()->getComm();
    const bool different = comm->getRank() == comm->getSize() - 1;
    for (int control = 0; control < 3; ++control)
    {
        auto options = f.options;
        if (different && control == 0)
            options.use_multi_rhs_transport = false;
        if (different && control == 1)
            options.linear.max_batch_width = 1;
        if (different && control == 2)
            options.linear.multi_rhs_policy = MultiRhsExecutionPolicy::ColumnSerial;
        if (comm->getSize() > 1)
            EXPECT_THROW((Flow(f.mesh, f.eos, options)), std::exception);
        else
            EXPECT_NO_THROW((Flow(f.mesh, f.eos, options)));
    }
}

TEST(QbmmMixedCarrierTransport, ScalarAndBatchedBothContinueTheirPortableCheckpoint)
{
    const FixtureOptions setup{.gas_reference = -6000};
    for (const bool batched : {false, true})
    {
        SCOPED_TRACE(batched);
        Fixture full(setup, MixedCarrierCase{}, batched), resumed(setup, MixedCarrierCase{}, batched);
        Fixture switched(setup, MixedCarrierCase{}, !batched);
        const auto first = full.flow->preview(dt);
        full.flow->commit(first);
        const auto image = full.flow->serialize_qbmm_restart();
        resumed.flow->restore_qbmm_restart(image);
        switched.flow->restore_qbmm_restart(image);
        const auto second = full.flow->preview(dt);
        const auto replay = resumed.flow->preview(dt);
        expect_carrier_trial(replay, second);
        const auto switched_replay = switched.flow->preview(dt);
        expect_carrier_trial(switched_replay, second);
        EXPECT_EQ(switched_replay.diagnostics().transport_multi_rhs.operator_groups, batched ? 0u : 2u);
        EXPECT_EQ(replay.diagnostics().transport_multi_rhs.operator_groups, batched ? 2u : 0u);
        full.flow->commit(second);
        resumed.flow->commit(replay);
        switched.flow->commit(switched_replay);
        EXPECT_EQ(full.flow->state().revision, 2u);
        EXPECT_EQ(resumed.flow->state().revision, full.flow->state().revision);
        EXPECT_EQ(switched.flow->state().revision, full.flow->state().revision);
    }
}

TEST(QbmmMixedCarrierTransport, UnrepresentableUnboundCountColumnRollsBackOnEveryRank)
{
    Fixture f({}, MixedCarrierCase{});
    const auto checkpoint = f.flow->checkpoint();
    const auto reference = f.flow->preview(dt);
    auto state = f.flow->state();
    bool has_invalid_column = false;
    for (size_t c = 0; c < state.cells.size(); ++c)
        if (f.global_cell(static_cast<LO>(c)) == 0)
        {
            // Finite extensive count, but N/V overflows in just one unbound column on one owner.
            state.cells[c].phase_numbers[2] = std::numeric_limits<double>::max();
            has_invalid_column = true;
        }
    f.flow->initialize(state);
    const auto accepted = f.flow->serialize_qbmm_restart();
    std::string message;
    try
    {
        (void) f.flow->preview(dt);
    }
    catch (const std::exception& error)
    {
        message = error.what();
    }
    if (has_invalid_column)
        EXPECT_EQ(message, "Population number concentration is not representable.");
    else
        EXPECT_EQ(message, "Accepted low-Mach batched transport concentration failed on another rank.");
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), accepted);
    f.flow->restore(checkpoint);
    expect_carrier_trial(f.flow->preview(dt), reference);
}

TEST(QbmmMixedCarrierTransport, OneUnboundSpeciesFailureRetainsAcceptedStateAndValidTrial)
{
    Fixture f({}, MixedCarrierCase{});
    const auto retained = f.flow->preview(dt);
    const auto accepted = f.flow->serialize_qbmm_restart();
    std::vector<Flow::CellSources> sources(f.mesh->num_owned_cells());
    for (size_t c = 0; c < sources.size(); ++c)
        if (f.global_cell(static_cast<LO>(c)) == 0)
            sources[c].species_moles = {{0., -1.}, {0., 0.}, {0., 0.}};
    EXPECT_THROW((void) f.flow->preview(dt, sources), std::exception);
    EXPECT_EQ(f.flow->serialize_qbmm_restart(), accepted);
    ASSERT_NO_THROW(f.flow->commit(retained));
    EXPECT_EQ(f.flow->state().revision, 1u);
}
} // namespace
