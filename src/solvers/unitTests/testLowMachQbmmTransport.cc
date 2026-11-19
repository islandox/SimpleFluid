/** Native common-map QBMM transport and owner-oriented receipt regressions. */
#include "equations/CollectiveValidation.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/details/LowMachQbmmState.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <string>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
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

struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::vector<std::shared_ptr<const R::MixtureThermodynamics>> eos;
    Flow::Options options;
    std::shared_ptr<Flow> flow;
    unsigned cells;

    explicit Fixture(FixtureOptions input = {}) : cells(input.cells)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{
                cells == 4 ? ArrReal{0, input.x_split, 1} : ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
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
        flow = std::make_shared<Flow>(mesh, eos, options);
        flow->initialize(state);
    }

    size_t global_cell(LO cell) const
    {
        const auto center = mesh->cell_centroid(cell);
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
} // namespace
