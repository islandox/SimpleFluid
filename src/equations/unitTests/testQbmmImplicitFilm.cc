#include "equations/QbmmLawInputs.hh"
#include "equations/QbmmPopulationLawAdapters.hh"
#include "equations/SpeciesBubbleProcesses.hh"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <limits>
#include <numbers>

namespace
{
namespace B = SimpleFluid::BubbleGasPhysics;
namespace R = SimpleFluid::Reacting;
using Phase = R::SpeciesPhase;

struct Binding
{
    enum class Role
    {
        DispersedGas
    } role;
    std::string identity;
};

constexpr R::SpeciesId species_a = 41, species_b = 7, species_no_liquid = 102;
constexpr double liquid_volume = .5, bulk_volume = 7, dt = .25;
constexpr std::array<R::QbmmAtom, 2> two_nodes{{{4, 2}, {2, 8}}};
constexpr std::array<double, 2> two_diameters{.25, .5};

struct QbmmImplicitFilmTest : testing::Test
{
    const std::shared_ptr<const R::SpeciesCatalog> catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{species_a, "A", .125, {{"A", 1}}, {Phase::Dispersed, Phase::Dissolved}},
            {species_b, "B", .25, {{"B", 1}}, {Phase::Dispersed, Phase::Dissolved}},
            {species_no_liquid, "C", .5, {{"C", 1}}, {Phase::Dispersed}}});
    const R::QbmmPopulationSchema small = family("small", 1, 0, 0);
    const R::QbmmPopulationSchema upper = family("upper", 1, 0, 0);
    const R::QbmmPopulationSchema other = family("other", 0, 1, 0);

    R::QbmmPopulationSchema family(const char* name, double a, double b, double c) const
    {
        return {catalog, catalog->fingerprint(), name, Binding{Binding::Role::DispersedGas, name}, .25,
            {{species_a, a}, {species_b, b}, {species_no_liquid, c}}};
    }
};

R::QbmmLawInputs snapshot(const R::QbmmPopulationSchema& family, std::span<const R::QbmmAtom> atoms,
    std::array<double, 2> diameters, std::optional<std::vector<double>> concentrations,
    std::optional<double> volume = liquid_volume, std::optional<double> duration = dt,
    std::optional<std::vector<double>> dissolved_moles = {})
{
    const auto quadrature = R::invert_qbmm_moments(R::qbmm_atom_moments(atoms));
    std::optional<R::QbmmGeometryView> geometry;
    if (!atoms.empty())
    {
        geometry.emplace();
        geometry->active_atoms = quadrature.active_atoms;
        geometry->diameter_m = diameters;
    }
    return {quadrature, geometry, family.mass_scale_kg(), bulk_volume, 100000, 300, duration, atoms.empty() ? 0. : .2,
        {}, {}, std::move(concentrations), 300, {}, family.fingerprint(), family.catalog().fingerprint(),
        family.population(), {}, {}, {}, volume, std::move(dissolved_moles)};
}

R::QbmmFilmNodeValues law(double alpha, R::QbmmAtom atom, double diameter, double equilibrium_moles)
{
    // Manufacture the weight independently in terms of the projected circular diameter.
    return {alpha * liquid_volume / (dt * std::numbers::pi * atom.count * diameter * diameter),
        equilibrium_moles / liquid_volume};
}

double signed_amount(const R::QbmmLawInputs& input, const R::QbmmEventOperation& operation, std::size_t species)
{
    const auto& resize = std::get<R::QbmmResize>(operation);
    EXPECT_EQ(std::get<R::QbmmPhaseExchange>(resize.transfer.counterpart).phase, Phase::Dissolved);
    EXPECT_EQ(resize.count, input.quadrature.atoms[resize.donor].count);
    EXPECT_GT(resize.mass_over_scale, 0);
    for (std::size_t s = 0; s < resize.transfer.moles.size(); ++s)
        if (s != species)
            EXPECT_EQ(resize.transfer.moles[s], 0);
    const double amount = resize.transfer.moles.at(species);
    return resize.mass_over_scale > input.quadrature.atoms[resize.donor].mass_over_scale ? amount : -amount;
}

R::QbmmPopulationAdvance advance(const R::QbmmPopulationSchema& family, std::span<const R::QbmmAtom> atoms,
    double dissolved, const std::vector<R::QbmmEventOperation>& operations)
{
    const auto moments = R::qbmm_atom_moments(atoms);
    R::CellSpeciesState stock(family.catalog());
    stock.set_amount(family.catalog(), species_a, Phase::Dissolved, dissolved);
    stock.dispersedPopulations = {{family.population(), family.species_moles(moments[1])}};
    const auto& moles = stock.dispersedPopulations.front().amounts;
    const auto shape = R::qbmm_migrate_mass_pivots(family, moments[0], moles, atoms);
    const R::QbmmCellView pre{
        family.population(), family.catalog().fingerprint(), moles, moments[0], shape, family.fingerprint()};
    R::QbmmEventBatch batch{family.fingerprint(), 0, dt, {}};
    for (const auto& operation : operations)
        batch.events.push_back({"implicit-film", 1, operation});
    return R::advance_qbmm_population(family, stock, pre, batch);
}

TEST_F(QbmmImplicitFilmTest, TwoNodesResolveStiffAbsorptionWithoutOvershootAndPreserveMoments)
{
    constexpr double dissolved = 8, equilibrium = 2;
    const auto input = snapshot(small, two_nodes, two_diameters, std::vector<double>{16, 0, 0});
    for (const double alpha : {20., 0x1p80})
    {
        SCOPED_TRACE(alpha);
        const std::array values{law(alpha, two_nodes[0], two_diameters[0], equilibrium),
            law(alpha, two_nodes[1], two_diameters[1], equilibrium)};
        const auto operations = R::qbmm_implicit_film_transfer(small, input, species_a, dissolved, values);
        ASSERT_EQ(operations.size(), 2u);
        const auto result = advance(small, two_nodes, dissolved, operations);
        ASSERT_EQ(result.status, R::QbmmEventStatus::Accepted) << result.diagnostic;
        const auto& endpoint = result.endpoint.value();
        EXPECT_EQ(endpoint.count, 6);
        EXPECT_EQ(endpoint.receipts.size(), 2u);

        const double final_dissolved = endpoint.species.amount(*catalog, species_a, Phase::Dissolved);
        EXPECT_GE(final_dissolved, equilibrium);
        const double expected_offset = (dissolved - equilibrium) / (1 + 2 * alpha);
        EXPECT_NEAR(final_dissolved - equilibrium, expected_offset, 32 * std::numeric_limits<double>::epsilon());
        const double expected_transfer = alpha * expected_offset;
        const double final_gas = endpoint.species.amount(*catalog, species_a, Phase::Dispersed, small.population());
        EXPECT_NEAR(final_gas, 48 + 2 * expected_transfer, 2e-14);
        EXPECT_NEAR(final_gas + final_dissolved, 56, 2e-14);
        for (const auto& operation : operations)
            EXPECT_NEAR(signed_amount(input, operation, 0), expected_transfer, 5e-15);

        const double mass_first = 2 + .5 * expected_transfer / 4;
        const double mass_second = 8 + .5 * expected_transfer / 2;
        const R::QbmmCellView final{small.population(), catalog->fingerprint(),
            endpoint.species.dispersedPopulations.front().amounts, endpoint.count, endpoint.shape, small.fingerprint()};
        const auto moments = R::qbmm_cell_moments(small, final);
        for (std::size_t k = 0; k < moments.size(); ++k)
        {
            const double expected = 4 * std::pow(mass_first, k) + 2 * std::pow(mass_second, k);
            EXPECT_NEAR(moments[k], expected, expected * 2e-14);
        }
    }
}

TEST_F(QbmmImplicitFilmTest, NodeFilmsCanTransferInOppositeDirectionsAgainstTheSameEndpoint)
{
    const auto input = snapshot(small, two_nodes, two_diameters, std::vector<double>{22, 0, 0}, liquid_volume, dt,
        std::vector<double>{11, 0, 0});
    const std::array values{law(2, two_nodes[0], two_diameters[0], 2), law(3, two_nodes[1], two_diameters[1], 7)};
    const auto operations = R::qbmm_implicit_film_transfer(small, input, species_a, 11, values);
    ASSERT_EQ(operations.size(), 2u);
    const auto convenience = R::qbmm_implicit_film_transfer(small, input, species_a, values);
    ASSERT_EQ(convenience.size(), 2u);
    for (std::size_t i = 0; i < operations.size(); ++i)
        EXPECT_EQ(signed_amount(input, operations[i], 0), signed_amount(input, convenience[i], 0));
    // D'=(11 + 2*2 + 3*7)/6=6. The first node gains 8 mol and the second returns 3 mol.
    EXPECT_NEAR(signed_amount(input, operations[0], 0), 8, 4e-15);
    EXPECT_NEAR(signed_amount(input, operations[1], 0), -3, 4e-15);
    const auto result = advance(small, two_nodes, 11, operations);
    ASSERT_EQ(result.status, R::QbmmEventStatus::Accepted) << result.diagnostic;
    EXPECT_EQ(result.endpoint->count, 6);
    EXPECT_NEAR(result.endpoint->species.amount(*catalog, species_a, Phase::Dissolved), 6, 4e-15);
}

TEST_F(QbmmImplicitFilmTest, FrozenConvenienceUsesExactDissolvedStockAtSaturation)
{
    constexpr double dissolved = .01, volume = .29;
    const std::array atoms{R::QbmmAtom{1, dissolved / 8}};
    const double concentration = dissolved / volume;
    // Reconstructing from the independently rounded concentration would overdraw the original donor.
    ASSERT_GT(static_cast<double>(static_cast<long double>(volume) * concentration), dissolved);
    const auto input = snapshot(small, atoms, {.25, 0}, std::vector<double>{concentration, 0, 0}, volume, dt,
        std::vector<double>{dissolved, 0, 0});
    const std::array values{R::QbmmFilmNodeValues{0x1p80, 0}};
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(small, input, species_a, std::nextafter(dissolved, 1.), values),
        std::invalid_argument);
    const auto operations = R::qbmm_implicit_film_transfer(small, input, species_a, values);
    ASSERT_EQ(operations.size(), 1u);
    EXPECT_EQ(signed_amount(input, operations.front(), 0), dissolved);
    const auto result = advance(small, atoms, dissolved, operations);
    ASSERT_EQ(result.status, R::QbmmEventStatus::Accepted) << result.diagnostic;
    EXPECT_EQ(result.endpoint->species.amount(*catalog, species_a, Phase::Dissolved), 0);
}

TEST_F(QbmmImplicitFilmTest, StiffNonreferenceFilmRetainsTransferWhenEndpointRoundsNearItsEquilibrium)
{
    for (const auto [count, coefficient] :
        std::array{std::pair{693., 0x1.e1a235bab7461p+68}, std::pair{2541., 0x1.8a105a817eadbp+49}})
    {
        const long double alpha = coefficient * (std::numbers::pi_v<long double> * count);
        const double dissolved = static_cast<double>(alpha);
        // With equilibria 0 and 1 mol the upper driving difference is
        // (D - alpha - 1)/(1+2*alpha), even when the rounded endpoint is 1.
        const long double upper_transfer =
            ((static_cast<long double>(dissolved) - alpha) - 1) * (alpha / (1 + 2 * alpha));
        const double initial_mass = static_cast<double>(2 * std::max(1.L, std::abs(upper_transfer)) / count);
        const std::array atoms{R::QbmmAtom{count, initial_mass}};
        const std::vector<double> concentrations{dissolved, 0, 0};
        const auto first = snapshot(small, atoms, {1, 0}, concentrations, 1., 1.);
        const auto second = snapshot(upper, atoms, {1, 0}, concentrations, 1., 1.);
        const std::array left{R::QbmmFilmNodeValues{coefficient, 0}};
        const std::array right{R::QbmmFilmNodeValues{coefficient, 1}};
        const auto operations = R::qbmm_implicit_film_transfer(
            std::array{R::QbmmImplicitFilmFamily{small, first, species_a, dissolved, left},
                R::QbmmImplicitFilmFamily{upper, second, species_a, dissolved, right}});
        ASSERT_EQ(operations[0].size(), 1u);
        ASSERT_EQ(operations[1].size(), 1u);
        EXPECT_NEAR(signed_amount(second, operations[1][0], 0), upper_transfer, std::abs(upper_transfer) * 5e-14L);
    }
}

TEST_F(QbmmImplicitFilmTest, StiffCloseEquilibriaRetainTheirConcentrationDifference)
{
    constexpr double volume = .37, dissolved = .37;
    const double coefficient = std::ldexp(volume / std::numbers::pi, 52);
    const double equilibrium_upper = std::nextafter(1., 2.);
    const std::array atoms{R::QbmmAtom{1, .25}};
    const std::vector<double> concentrations{1, 0, 0};
    const auto first = snapshot(small, atoms, {1, 0}, concentrations, volume, 1.);
    const auto second = snapshot(upper, atoms, {1, 0}, concentrations, volume, 1.);
    const std::array left{R::QbmmFilmNodeValues{coefficient, 1}};
    const std::array right{R::QbmmFilmNodeValues{coefficient, equilibrium_upper}};
    const auto operations =
        R::qbmm_implicit_film_transfer(std::array{R::QbmmImplicitFilmFamily{small, first, species_a, dissolved, left},
            R::QbmmImplicitFilmFamily{upper, second, species_a, dissolved, right}});
    ASSERT_EQ(operations[0].size(), 1u);
    ASSERT_EQ(operations[1].size(), 1u);

    const long double alpha = coefficient * std::numbers::pi_v<long double> / volume;
    const long double weighted_gap = alpha * volume * (equilibrium_upper - 1.);
    const long double denominator = 1 + 2 * alpha;
    EXPECT_NEAR(signed_amount(first, operations[0][0], 0), (alpha / denominator) * weighted_gap, 3e-15);
    EXPECT_NEAR(signed_amount(second, operations[1][0], 0), -(1 + alpha) / denominator * weighted_gap, 3e-15);
}

TEST_F(QbmmImplicitFilmTest, JointMonodisperseFilmsMatchUpstreamAndRequestTheSameBitsInEitherOrder)
{
    const std::array small_atoms{R::QbmmAtom{4, 2}};
    const std::array upper_atoms{R::QbmmAtom{2, 8}};
    const std::vector<double> concentrations{22, 0, 0};
    const auto first = snapshot(small, small_atoms, {.25, 0}, concentrations);
    const auto second = snapshot(upper, upper_atoms, {.5, 0}, concentrations);
    R::CellSpeciesState stock(*catalog);
    stock.dispersedPopulations = {{small.population(), {16, 0, 0}}, {upper.population(), {32, 0, 0}}};
    stock.set_amount(*catalog, species_a, Phase::Dissolved, 11);
    const R::ReactionEnvironment environment{0, 300, 100000, liquid_volume, bulk_volume};
    const R::ReactionContext chemistry{*catalog, stock, environment};
    const B::MultispeciesPopulationContext upstream_context{
        chemistry, {{small.population(), 4, .125, {}}, {upper.population(), 2, .25, {}}}};
    const auto upstream_law = [](R::QbmmFilmNodeValues value)
    {
        return B::SpeciesFilmLaw{species_a, [value](const auto&, const auto&) { return value.coefficient_m_per_s; },
            [value](const auto&, const auto&) { return value.equilibrium_concentration_mol_per_m3; }};
    };
    for (const double alpha : {20., 0x1p80})
    {
        SCOPED_TRACE(alpha);
        const double target = alpha == 20 ? 1.875 : 2;
        const std::array left{law(alpha, small_atoms[0], .25, 2)};
        const std::array right{law(alpha, upper_atoms[0], .5, target)};
        const R::QbmmImplicitFilmFamily small_film{small, first, species_a, 11, left};
        const R::QbmmImplicitFilmFamily upper_film{upper, second, species_a, 11, right};
        const auto joint = R::qbmm_implicit_film_transfer(std::array{small_film, upper_film});
        const auto reversed = R::qbmm_implicit_film_transfer(std::array{upper_film, small_film});
        const auto upstream = B::make_implicit_film_transfer_process({{small.population(), {upstream_law(left[0])}},
            {upper.population(), {upstream_law(right[0])}}})(upstream_context, dt);
        EXPECT_TRUE(upstream.numberEvents.empty());
        ASSERT_EQ(upstream.transfers.size(), 2u);
        for (std::size_t i = 0; i < joint.size(); ++i)
        {
            ASSERT_EQ(joint[i].size(), 1u);
            ASSERT_EQ(reversed[1 - i].size(), 1u);
            const auto& inputs = i == 0 ? first : second;
            const double mol = signed_amount(inputs, joint[i][0], 0);
            const double reversed_mol = signed_amount(inputs, reversed[1 - i][0], 0);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(mol), std::bit_cast<std::uint64_t>(reversed_mol));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(std::get<R::QbmmResize>(joint[i][0]).mass_over_scale),
                std::bit_cast<std::uint64_t>(std::get<R::QbmmResize>(reversed[1 - i][0]).mass_over_scale));
            EXPECT_EQ(upstream.transfers[i].receiverPopulation, inputs.population);
            EXPECT_NEAR(mol, upstream.transfers[i].moles, 5e-15);
        }
    }
}

TEST_F(QbmmImplicitFilmTest, DistinctSpeciesHaveIndependentPoolsAndAbsentFamilyNeedsNoLiquid)
{
    const std::array atoms{R::QbmmAtom{4, 2}};
    const std::vector<double> concentrations{22, 14, 0};
    const auto first = snapshot(small, atoms, {.25, 0}, concentrations);
    const auto second = snapshot(other, atoms, {.5, 0}, concentrations);
    const auto absent = snapshot(upper, {}, {}, {}, {});
    const std::array left{law(2, atoms[0], .25, 2)};
    const std::array right{law(3, atoms[0], .5, 3)};
    const std::array groups{R::QbmmImplicitFilmFamily{small, first, species_a, 11, left},
        R::QbmmImplicitFilmFamily{upper, absent, species_a, 11, {}},
        R::QbmmImplicitFilmFamily{other, second, species_b, 7, right}};
    const auto operations = R::qbmm_implicit_film_transfer(groups);
    ASSERT_EQ(operations[0].size(), 1u);
    EXPECT_TRUE(operations[1].empty());
    ASSERT_EQ(operations[2].size(), 1u);
    EXPECT_NEAR(signed_amount(first, operations[0][0], 0), 6, 3e-15);
    EXPECT_NEAR(signed_amount(second, operations[2][0], 1), 3, 3e-15);
    EXPECT_TRUE(R::qbmm_implicit_film_transfer(upper, absent, species_a, 0, {}).empty());
    EXPECT_TRUE(R::qbmm_implicit_film_transfer(upper, absent, species_a, {}).empty());
    EXPECT_TRUE(R::qbmm_implicit_film_transfer(std::span<const R::QbmmImplicitFilmFamily>{}).empty());
}

TEST_F(QbmmImplicitFilmTest, FilmDoesNotBorrowSameStageIncomingDissolvedMaterial)
{
    const auto input = snapshot(small, two_nodes, two_diameters, std::vector<double>{2, 0, 0});
    const std::array values{law(1, two_nodes[0], two_diameters[0], 0), law(1, two_nodes[1], two_diameters[1], 6)};
    const auto operations = R::qbmm_implicit_film_transfer(small, input, species_a, 1, values);
    ASSERT_EQ(operations.size(), 2u);
    EXPECT_GT(signed_amount(input, operations[0], 0), 1);
    EXPECT_LT(signed_amount(input, operations[1], 0), 0);
    const auto result = advance(small, two_nodes, 1, operations);
    EXPECT_EQ(result.status, R::QbmmEventStatus::SpeciesDonorOverdraw) << result.diagnostic;
    EXPECT_FALSE(result.endpoint);
}

TEST_F(QbmmImplicitFilmTest, InvalidFilmsRejectBeforeProducingEvents)
{
    const std::array atoms{R::QbmmAtom{4, 2}};
    const std::vector<double> concentrations{22, 0, 0};
    const auto input = snapshot(small, atoms, {.25, 0}, concentrations);
    const std::array values{law(2, atoms[0], .25, 2)};
    const auto mixed = family("mixed", .5, .5, 0);
    const auto no_dissolved = family("no_dissolved", 0, 0, 1);
    const auto mixed_input = snapshot(mixed, atoms, {.25, 0}, concentrations);
    const auto non_dissolved_input = snapshot(no_dissolved, atoms, {.25, 0}, concentrations);
    EXPECT_THROW(
        (void) R::qbmm_implicit_film_transfer(mixed, mixed_input, species_a, 11, values), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(no_dissolved, non_dissolved_input, species_no_liquid, 0, values),
        std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(small, input, species_b, 0, values), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(upper, input, species_a, 11, values), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(small, input, species_a, 11, {}), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(small, input, species_a, 10, values), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(small, input, species_a, values), std::invalid_argument);
    const R::QbmmImplicitFilmFamily group{small, input, species_a, 11, values};
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(std::array{group, group}), std::invalid_argument);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto volume : {std::optional<double>{}, std::optional<double>{0}, std::optional<double>{nan}})
    {
        const auto invalid = snapshot(small, atoms, {.25, 0}, concentrations, volume);
        EXPECT_THROW(
            (void) R::qbmm_implicit_film_transfer(small, invalid, species_a, 11, values), std::invalid_argument);
    }
    for (const auto invalid_value : {R::QbmmFilmNodeValues{-1, 1}, R::QbmmFilmNodeValues{nan, 1},
             R::QbmmFilmNodeValues{1, -1}, R::QbmmFilmNodeValues{1, nan}})
    {
        const std::array invalid{invalid_value};
        EXPECT_THROW(
            (void) R::qbmm_implicit_film_transfer(small, input, species_a, 11, invalid), std::invalid_argument);
    }
    const auto bad_geometry = snapshot(small, atoms, {0, 0}, concentrations);
    EXPECT_THROW(
        (void) R::qbmm_implicit_film_transfer(small, bad_geometry, species_a, 11, values), std::invalid_argument);
    const std::array no_transfer{law(0, atoms[0], .25, 2)};
    EXPECT_TRUE(R::qbmm_implicit_film_transfer(small, input, species_a, 11, no_transfer).empty());

    // With a saturated weight, transfer rounds to the exact original 16 mol at the first equilibrium.
    for (const double equilibrium : {27., 28.})
    {
        const std::array depletion{law(0x1p80, atoms[0], .25, equilibrium)};
        EXPECT_THROW(
            (void) R::qbmm_implicit_film_transfer(small, input, species_a, 11, depletion), std::invalid_argument);
    }
    const std::array too_small{law(1e-18, atoms[0], .25, 2)};
    EXPECT_THROW((void) R::qbmm_implicit_film_transfer(small, input, species_a, 11, too_small), std::invalid_argument);
}

TEST_F(QbmmImplicitFilmTest, FrozenBuilderSuppliesVolumeOnlyWithFiniteCataloguedLiquidConcentrations)
{
    const std::array atoms{R::QbmmAtom{4, 2}};
    const std::array moles{16., 0., 0.};
    const auto q = R::invert_qbmm_moments(R::qbmm_atom_moments(atoms));
    R::QbmmFrozenLawState frozen{
        q, {}, moles, bulk_volume, 1., 100000, 300, dt, .2, std::vector<double>{22, 0, 0}, 300, {}};
    EXPECT_FALSE(R::make_qbmm_law_inputs(small, frozen).carrier_liquid_volume_m3);
    frozen.carrier_liquid_volume_m3 = liquid_volume;
    EXPECT_EQ(R::make_qbmm_law_inputs(small, frozen).carrier_liquid_volume_m3, liquid_volume);
    frozen.carrier_liquid_concentration_mol_per_m3.reset();
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
    frozen.carrier_liquid_concentration_mol_per_m3 = std::vector<double>{22};
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
    frozen.carrier_liquid_concentration_mol_per_m3 = std::vector<double>{22, -1, 0};
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
    frozen.carrier_liquid_concentration_mol_per_m3 = std::vector<double>{22, 0, 0};
    frozen.carrier_liquid_dissolved_moles = std::vector<double>{11, 0, 0};
    EXPECT_EQ(
        R::make_qbmm_law_inputs(small, frozen).carrier_liquid_dissolved_moles, frozen.carrier_liquid_dissolved_moles);
    frozen.carrier_liquid_volume_m3.reset();
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
    frozen.carrier_liquid_volume_m3 = liquid_volume;
    frozen.carrier_liquid_dissolved_moles = std::vector<double>{11};
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
    frozen.carrier_liquid_dissolved_moles = std::vector<double>{11, -1, 0};
    EXPECT_THROW((void) R::make_qbmm_law_inputs(small, frozen), std::invalid_argument);
}
} // namespace
