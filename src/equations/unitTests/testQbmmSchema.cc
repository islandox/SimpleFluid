/** @file testQbmmSchema.cc
 * @brief Fixed-composition material authority, initialization and physical views for the local QBMM kernel.
 */
#include "equations/QbmmPopulationSchema.hh"
#include "equations/ReactingThermodynamics.hh"
#include "solvers/LowMachMixtureSolver.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace
{
namespace R = SimpleFluid::Reacting;
using Role = R::LowMachMixtureSolver::PhaseRole;
using Binding = R::LowMachMixtureSolver::PhaseRoleDescriptor;

constexpr R::SpeciesId A = 31, B = 8;
constexpr double scale = 1e-6;
using Moles = std::array<double, 2>;
constexpr Moles base_moles{7.0 / 20000, 21.0 / 40000};
constexpr R::QbmmShapeState base_shape{50, 194};

std::shared_ptr<const R::SpeciesCatalog> make_catalog(std::uint64_t revision = 1)
{
    const std::vector phases{R::SpeciesPhase::Dispersed, R::SpeciesPhase::Liquid};
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{A, "A", 0.01, {{"X", 1}}, phases}, {B, "B", 0.02, {{"X", 2}}, phases}},
        revision);
}

R::QbmmPopulationSchema family(const std::shared_ptr<const R::SpeciesCatalog>& catalog, double reference = scale,
    R::QbmmClosureVersions versions = {})
{
    return {catalog, catalog->fingerprint(), "bubbles", Binding{Role::DispersedGas, "bubbles"}, reference,
        {{A, 0.25}, {B, 0.75}}, versions};
}

R::QbmmCellView view(
    const R::QbmmPopulationSchema& schema, const Moles& moles, const double& count, const R::QbmmShapeState& shape)
{
    return {schema.population(), schema.catalog().fingerprint(), moles, count, shape, schema.fingerprint()};
}

TEST(QbmmSchemaTest, ChecksCatalogOrderAllowedSpeciesPhysicalPhaseAndVersionedIdentity)
{
    const auto catalog = make_catalog();
    const auto schema = family(catalog);
    EXPECT_EQ(schema.fingerprint(), family(make_catalog()).fingerprint());
    EXPECT_NE(schema.fingerprint(), family(make_catalog(2)).fingerprint());
    EXPECT_NE(schema.fingerprint(), family(catalog, 2 * scale).fingerprint());
    R::QbmmClosureVersions versions{};
    ++versions.source;
    EXPECT_NE(schema.fingerprint(), family(catalog, scale, versions).fingerprint());
    versions = {};
    versions.source_identity = "independent-law";
    EXPECT_NE(schema.fingerprint(), family(catalog, scale, versions).fingerprint());

    const auto make = [&](std::string name, Binding binding, std::vector<R::QbmmMassFraction> fractions)
    { return R::QbmmPopulationSchema(catalog, catalog->fingerprint(), name, binding, scale, fractions); };
    const Binding gas{Role::DispersedGas, "bubbles"};
    const std::vector<R::QbmmMassFraction> fractions{{A, 0.25}, {B, 0.75}};
    EXPECT_NE(schema.fingerprint(), make("bubbles", gas, {{A, 1}, {B, 0}}).fingerprint());
    EXPECT_NE(schema.fingerprint(), make("other", {Role::DispersedGas, "other"}, fractions).fingerprint());
    EXPECT_THROW(make("", {Role::DispersedGas, ""}, fractions), std::invalid_argument);
    EXPECT_THROW(make("other", gas, fractions), std::invalid_argument);
    for (Role role : {Role::BulkLiquid, Role::BulkGas})
        EXPECT_THROW(make("bubbles", (Binding{role, "bubbles"}), fractions), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{B, 0.75}, {A, 0.25}}), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{A, 0.25}, {A, 0.75}}), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{A, 0.25}, {999, 0.75}}), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{A, 1}}), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{A, 0.1}, {B, 0.2}}), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{A, -0.1}, {B, 1.1}}), std::invalid_argument);
    EXPECT_THROW(make("bubbles", gas, {{A, 1}, {B, std::numeric_limits<double>::quiet_NaN()}}), std::invalid_argument);
    for (const double bad : {0., -1., std::numeric_limits<double>::infinity()})
        EXPECT_THROW(family(catalog, bad), std::invalid_argument);
    EXPECT_THROW(R::QbmmPopulationSchema(catalog, make_catalog(2)->fingerprint(), "bubbles", gas, scale, fractions),
        std::invalid_argument);
    for (auto revision : {&R::QbmmClosureVersions::inversion, &R::QbmmClosureVersions::thermodynamics,
             &R::QbmmClosureVersions::transport, &R::QbmmClosureVersions::source})
    {
        versions = {};
        versions.*revision = 0;
        EXPECT_THROW(family(catalog, scale, versions), std::invalid_argument);
    }

    auto species = catalog->species();
    species[1].allowedPhases = {R::SpeciesPhase::Liquid};
    const auto restricted = std::make_shared<const R::SpeciesCatalog>(species);
    EXPECT_THROW(family(restricted), std::invalid_argument);
    EXPECT_NO_THROW(
        R::QbmmPopulationSchema(restricted, restricted->fingerprint(), "bubbles", gas, scale, {{A, 1}, {B, 0}}));
}

TEST(QbmmSchemaTest, CanonicalMolesAndCountAreLiveOwnersAndSelectiveTransferFailsAtEveryStockScale)
{
    const auto schema = family(make_catalog());
    Moles moles = base_moles;
    double count = 5;
    R::QbmmShapeState shape = base_shape;
    const auto cell = view(schema, moles, count, shape);
    const auto base = R::qbmm_cell_moments(schema, cell);
    const R::QbmmMomentBlock expected{5, 14, 50, 194};
    for (unsigned k = 0; k < 4; ++k)
        EXPECT_DOUBLE_EQ(base[k], expected[k]);
    EXPECT_DOUBLE_EQ(schema.mass_scale_kg() * base[1], 7.0 / 500000);
    const auto reconstructed = schema.species_moles(14);
    ASSERT_EQ(reconstructed.size(), 2u);
    for (unsigned i = 0; i < 2; ++i)
        EXPECT_DOUBLE_EQ(reconstructed[i], base_moles[i]);
    const auto q = R::invert_qbmm_moments(base);
    ASSERT_EQ(q.active_atoms, 2u);
    Moles from_atoms{};
    for (const auto& atom : q.atoms)
    {
        from_atoms[0] += atom.count * atom.mass_over_scale * scale * 0.25 / 0.01;
        from_atoms[1] += atom.count * atom.mass_over_scale * scale * 0.75 / 0.02;
    }
    for (unsigned i = 0; i < 2; ++i)
        EXPECT_NEAR(from_atoms[i], base_moles[i], 2e-13 * base_moles[i]);
    for (const int exponent : {-700, 0, 600})
    {
        count = std::ldexp(5., exponent);
        shape = {std::ldexp(50., exponent), std::ldexp(194., exponent)};
        moles = {std::ldexp(base_moles[0], exponent), std::ldexp(base_moles[1], exponent)};
        const auto scaled = R::qbmm_cell_moments(schema, cell);
        for (unsigned k = 0; k < 4; ++k)
            EXPECT_DOUBLE_EQ(scaled[k], std::ldexp(expected[k], exponent));
        moles[0] += std::ldexp(1e-6, exponent);
        EXPECT_THROW((void) R::qbmm_cell_moments(schema, cell), std::invalid_argument);
    }
    moles = {};
    count = 0;
    shape = {};
    EXPECT_EQ(R::qbmm_cell_moments(schema, cell), (R::QbmmMomentBlock{}));
    count = 1;
    EXPECT_THROW((void) R::qbmm_cell_moments(schema, cell), std::invalid_argument);
    count = 0;
    shape.H3 = std::numeric_limits<double>::denorm_min();
    EXPECT_THROW((void) R::qbmm_cell_moments(schema, cell), std::invalid_argument);
    shape = base_shape;
    moles = base_moles;
    EXPECT_THROW((void) R::qbmm_cell_moments(schema, cell), std::invalid_argument);
    count = 5;
    auto foreign = cell;
    foreign.population = "other";
    EXPECT_THROW((void) R::qbmm_cell_moments(schema, foreign), std::invalid_argument);
    foreign.population = cell.population;
    foreign.catalog_fingerprint = make_catalog(2)->fingerprint();
    EXPECT_THROW((void) R::qbmm_cell_moments(schema, foreign), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_cell_moments(family(make_catalog(), 2 * scale), cell), std::invalid_argument);
}

TEST(QbmmSchemaTest, MigratesMolarPivotsAndMomentsByMixtureMolarMassAndRequiresExplicitCountInitialization)
{
    const auto catalog = make_catalog();
    const auto schema = family(catalog);
    constexpr double molar_mass = 0.016; // 1 / (0.25 / 0.01 + 0.75 / 0.02).
    EXPECT_DOUBLE_EQ(schema.mixture_molar_mass_kg_per_mol(), molar_mass);
    const double unit = scale / molar_mass;
    const std::array pivots{R::QbmmMolarPivot{2, unit}, R::QbmmMolarPivot{3, 4 * unit}, R::QbmmMolarPivot{0, 7 * unit}};
    const R::QbmmMomentBlock molar{5, 14 * unit, 50 * unit * unit, 194 * std::pow(unit, 3)};
    const std::array atoms{R::QbmmAtom{2, 1}, R::QbmmAtom{3, 4}};
    for (const auto shape : {R::qbmm_migrate_molar_pivots(schema, 5, base_moles, pivots),
             R::qbmm_migrate_mass_pivots(schema, 5, base_moles, atoms),
             R::qbmm_convert_molar_moments(schema, 5, base_moles, molar)})
    {
        EXPECT_NEAR(shape.H2, 50, 1e-12);
        EXPECT_NEAR(shape.H3, 194, 4e-12);
    }
    const auto mono = R::qbmm_initialize_monodisperse(schema, 5, base_moles);
    EXPECT_NEAR(mono.H2, 196.0 / 5, 1e-12);
    EXPECT_NEAR(mono.H3, 2744.0 / 25, 3e-12);
    const auto doubled = family(catalog, 2 * scale);
    const auto converted = R::qbmm_rescale_moments(schema, doubled, {5, 14, 50, 194});
    EXPECT_EQ(converted, (R::QbmmMomentBlock{5, 7, 12.5, 24.25}));
    EXPECT_THROW((void) R::qbmm_migrate_molar_pivots(schema, 4, base_moles, pivots), std::invalid_argument);
    auto wrong = molar;
    wrong[1] *= 0.5;
    EXPECT_THROW((void) R::qbmm_convert_molar_moments(schema, 5, base_moles, wrong), std::invalid_argument);
    EXPECT_THROW((void) R::qbmm_initialize_monodisperse(schema, 0, base_moles), std::invalid_argument);
    const auto vacuum = R::qbmm_initialize_monodisperse(schema, 0, Moles{});
    EXPECT_EQ(vacuum.H2, 0);
    EXPECT_EQ(vacuum.H3, 0);
}

TEST(QbmmSchemaTest, MicrogramAndNanogramReferenceMassesGiveTheSamePhysicalAtoms)
{
    const auto catalog = make_catalog();
    constexpr double bubble_mass = 1e-12, count = 5;
    constexpr Moles moles{0.25 * 14 * bubble_mass / 0.01, 0.75 * 14 * bubble_mass / 0.02};
    for (const double reference : {1e-9, 1e-12})
    {
        const auto schema = family(catalog, reference);
        const double x = bubble_mass / reference;
        const R::QbmmShapeState shape{50 * x * x, 194 * x * x * x};
        const auto cell = view(schema, moles, count, shape);
        const auto h = R::qbmm_cell_moments(schema, cell);
        EXPECT_NEAR(h[1], 14 * x, 14 * x * 1e-13);
        const auto g = R::qbmm_geometry(schema, cell, 1., 0.8);
        ASSERT_EQ(g.active_atoms, 2u);
        EXPECT_NEAR(g.mass_kg, 14 * bubble_mass, 14 * bubble_mass * 1e-13);
        EXPECT_NEAR(g.mass_kg_per_bubble[0], bubble_mass, bubble_mass * 1e-13);
        EXPECT_NEAR(g.mass_kg_per_bubble[1], 4 * bubble_mass, 4 * bubble_mass * 1e-13);
        for (unsigned i = 0; i < 2; ++i)
        {
            const double physical_mass = (i == 0 ? 1. : 4.) * bubble_mass;
            const double radius = std::cbrt(3 * physical_mass * 0.8 / (4 * std::numbers::pi));
            EXPECT_NEAR(g.radius_m[i], radius, radius * 1e-13);
            EXPECT_NEAR(g.diameter_m[i], 2 * radius, radius * 2e-13);
        }
    }
}

TEST(QbmmSchemaTest, CommonPositiveMapPreservesCanonicalInventoryAndRejectsMismatchedShapeTransport)
{
    const auto schema = family(make_catalog());
    // The second predecessor is four bubbles of size 2. Apply the same two-column map
    // to the independent species, count and shape stocks before deriving H1.
    constexpr Moles second_moles{2e-4, 3e-4};
    constexpr std::array<Moles, 2> matrix{{{0.75, 1. / 3}, {0.25, 2. / 3}}};
    constexpr std::array<R::QbmmMomentBlock, 2> exact{
        {{61. / 12, 79. / 6, 257. / 6, 937. / 6}, {47. / 12, 53. / 6, 139. / 6, 419. / 6}}};
    R::QbmmMomentBlock total{};
    Moles species_total{};
    for (unsigned i = 0; i < 2; ++i)
    {
        const auto [a, b] = matrix[i];
        Moles moles{};
        for (unsigned j = 0; j < moles.size(); ++j)
        {
            moles[j] = a * base_moles[j] + b * second_moles[j];
            species_total[j] += moles[j];
        }
        const double count = a * 5 + b * 4;
        R::QbmmShapeState shape{a * 50 + b * 16, a * 194 + b * 32};
        const auto cell = view(schema, moles, count, shape);
        const auto h = R::qbmm_cell_moments(schema, cell);
        for (unsigned k = 0; k < total.size(); ++k)
        {
            EXPECT_NEAR(h[k], exact[i][k], 1e-13 * exact[i][k]);
            total[k] += h[k];
        }
        EXPECT_EQ(R::invert_qbmm_moments(h).status, R::InversionStatus::TwoNodes);
        // A different, still nonnegative conservative map sends H2 to row 1.
        shape.H2 = i == 0 ? 1 : 65;
        EXPECT_FALSE(R::invert_qbmm_moments(R::qbmm_cell_moments(schema, cell)).valid());
    }
    const R::QbmmMomentBlock old_total{9, 22, 66, 226};
    for (unsigned k = 0; k < total.size(); ++k)
        EXPECT_DOUBLE_EQ(total[k], old_total[k]);
    for (unsigned j = 0; j < species_total.size(); ++j)
        EXPECT_DOUBLE_EQ(species_total[j], base_moles[j] + second_moles[j]);
}

TEST(QbmmSchemaTest, GenericReactionChangesTotalMolesAndPreservesMassAndDeclaredElement)
{
    using P = R::SpeciesPhase;
    const auto catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{A, "A", 1, {{"X", 1}}, {P::Liquid}}, {B, "B", 2, {{"X", 2}}, {P::Liquid}}});
    const R::ReactionMechanism mechanism(catalog, {{3, "2A to B", {{A, P::Liquid, -2}, {B, P::Liquid, 1}}}});
    R::CellSpeciesState state(*catalog);
    state.set_amount(*catalog, A, P::Liquid, 2);
    R::ReactionSourceInput input;
    input.integratedExtents = {{3, 1}};
    const auto after = R::advance_reactions(mechanism, state, {0, 300, 1e5, 1., 1.}, 0.75, input);
    const double da = after.ledger.chemicalChange.amount(*catalog, A, P::Liquid);
    const double db = after.ledger.chemicalChange.amount(*catalog, B, P::Liquid);
    EXPECT_EQ(da, -2);
    EXPECT_EQ(db, 1);
    EXPECT_EQ(after.state.amount(*catalog, A, P::Liquid), 0);
    EXPECT_EQ(after.state.amount(*catalog, B, P::Liquid), 1);
    const auto& data = catalog->species();
    EXPECT_EQ(da + db, -1);
    EXPECT_EQ(data[0].molarMass * da + data[1].molarMass * db, 0);
    EXPECT_EQ(data[0].elements.at("X") * da + data[1].elements.at("X") * db, 0);
}

TEST(QbmmSchemaTest, CommonPressureEosVolumeIsIndependentOfHigherMomentsWhileAreaChanges)
{
    const auto catalog = make_catalog();
    const auto schema = family(catalog);
    const auto eos = R::make_ideal_gas_mixture(catalog, {{20, 300, 0}, {25, 300, 0}}, {250, 600, 1e4, 1e6});
    constexpr double gas_constant = 8.31446261815324, bulk_volume = 2e-3, count = 5;
    constexpr double t = 360;
    const double energy = (t - 300) * (20 * base_moles[0] + 25 * base_moles[1]);
    const R::QbmmShapeState mono{196.0 / 5, 2744.0 / 25};
    for (const double p : {1e5, 2e5})
    {
        const std::array phases{R::PhaseThermalState{eos, {base_moles[0], base_moles[1]}, p, energy, t}};
        const auto thermal = R::recover_phase_temperatures(phases, energy);
        const double expected_volume = (base_moles[0] + base_moles[1]) * gas_constant * t / p;
        ASSERT_NEAR(thermal.total_volume, expected_volume, expected_volume * 1e-11);
        const double specific_volume = thermal.total_volume / (14 * scale);
        const auto multi_cell = view(schema, base_moles, count, base_shape);
        const auto mono_cell = view(schema, base_moles, count, mono);
        const auto multi = R::qbmm_geometry(schema, multi_cell, bulk_volume, specific_volume);
        const auto single = R::qbmm_geometry(schema, mono_cell, bulk_volume, specific_volume);
        EXPECT_EQ(multi.active_atoms, 2u);
        EXPECT_EQ(single.active_atoms, 1u);
        EXPECT_DOUBLE_EQ(multi.mass_kg, single.mass_kg);
        EXPECT_DOUBLE_EQ(multi.mass_kg, 14 * scale);
        EXPECT_DOUBLE_EQ(multi.number_density, count / bulk_volume);
        EXPECT_LT(multi.area_density, single.area_density);
        const auto d1 = std::cbrt(6 * specific_volume * scale / std::numbers::pi);
        const auto d4 = std::cbrt(4.) * d1;
        const double area = std::numbers::pi * (2 * d1 * d1 + 3 * d4 * d4);
        EXPECT_NEAR(multi.area_density * bulk_volume, area, area * 1e-12);
        ASSERT_TRUE(multi.d32 && multi.min_diameter && multi.max_diameter);
        EXPECT_NEAR(*multi.d32 * area, 6 * expected_volume, expected_volume * 1e-10);
        EXPECT_NEAR(*multi.min_diameter, d1, d1 * 1e-12);
        EXPECT_NEAR(*multi.max_diameter, d4, d4 * 1e-12);
        for (unsigned k = 0; k < 4; ++k)
        {
            const double diameter_moment = (2 * std::pow(d1, k) + 3 * std::pow(d4, k)) / bulk_volume;
            EXPECT_NEAR(multi.diameter_moment_density[k], diameter_moment, diameter_moment * 1e-12);
        }
        for (const auto& geometry : {single, multi})
            EXPECT_NEAR(std::numbers::pi / 6 * bulk_volume * geometry.diameter_moment_density[3], expected_volume,
                expected_volume * 1e-11);
        EXPECT_THROW((void) R::qbmm_geometry(schema, multi_cell, bulk_volume, std::nullopt), std::invalid_argument);
        EXPECT_THROW((void) R::qbmm_geometry(schema, multi_cell, 0, specific_volume), std::invalid_argument);
    }
    const auto empty = R::qbmm_geometry(schema, view(schema, Moles{}, 0, {}), bulk_volume, std::nullopt);
    EXPECT_EQ(empty.active_atoms, 0u);
    EXPECT_EQ(empty.number_density, 0);
    EXPECT_EQ(empty.area_density, 0);
    EXPECT_EQ(empty.mass_kg, 0);
    EXPECT_FALSE(empty.d32 || empty.min_diameter || empty.max_diameter);
}
} // namespace
