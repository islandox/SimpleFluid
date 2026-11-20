/** @file testQbmm.cc
 * @brief Independent mass-coordinate inversion checks for the revision 0.2 arithmetic fixtures.
 */
#include "equations/QbmmMoments.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>

namespace
{
namespace R = SimpleFluid::Reacting;

// Forward integration is independent of both the production moment sum and inversion.
R::QbmmMomentBlock integrate(std::span<const R::QbmmAtom> atoms)
{
    R::QbmmMomentBlock result{};
    for (unsigned k = 0; k < result.size(); ++k)
    {
        long double integral = 0;
        for (const auto& [weight, mass] : atoms)
            integral += static_cast<long double>(weight) * std::pow(static_cast<long double>(mass), k);
        result[k] = static_cast<double>(integral);
    }
    return result;
}

void check_reconstruction(const R::QbmmQuadrature& q, const R::QbmmMomentBlock& input, double relative_tolerance)
{
    ASSERT_TRUE(q.valid());
    ASSERT_LE(q.active_atoms, q.atoms.size());
    const auto recovered = integrate(std::span(q.atoms).first(q.active_atoms));
    for (unsigned k = 0; k < input.size(); ++k)
    {
        EXPECT_NEAR(recovered[k], input[k], relative_tolerance * std::abs(input[k])) << "moment " << k;
        EXPECT_GE(q.reconstruction.relative[k], 0);
        EXPECT_LE(q.reconstruction.relative[k], R::qbmm_reconstruction_tolerance);
    }
}

TEST(QbmmInversionTest, ExactAbsenceAndOnePositiveAtom)
{
    const auto empty = R::invert_qbmm_moments({});
    EXPECT_EQ(empty.status, R::InversionStatus::Absent);
    EXPECT_EQ(empty.active_atoms, 0u);
    EXPECT_EQ(R::qbmm_atom_moments({}), (R::QbmmMomentBlock{}));
    const std::array atoms{R::QbmmAtom{5, 2}};
    const R::QbmmMomentBlock expected{5, 10, 20, 40};
    EXPECT_EQ(R::qbmm_atom_moments(atoms), expected);
    const auto q = R::invert_qbmm_moments(expected);
    ASSERT_EQ(q.status, R::InversionStatus::RankOne);
    ASSERT_EQ(q.active_atoms, 1u);
    EXPECT_DOUBLE_EQ(q.atoms[0].count, 5);
    EXPECT_DOUBLE_EQ(q.atoms[0].mass_over_scale, 2);
    check_reconstruction(q, expected, 0);
}

TEST(QbmmInversionTest, SubnormalThirdMomentMustRetainItsRelativeAccuracy)
{
    // In binary64, (2^-358)^3 is the smallest subnormal. The nearby mass
    // has a third moment 729/512 times as large, rounding to the same value.
    const double mass = std::ldexp(1., -358);
    const std::array exact{R::QbmmAtom{1, mass}}, lossy{R::QbmmAtom{1, 1.125 * mass}};
    const auto h = R::qbmm_atom_moments(exact);
    EXPECT_EQ(h[3], std::numeric_limits<double>::denorm_min());
    EXPECT_EQ(h[2], std::ldexp(1., -716));
    EXPECT_EQ(R::invert_qbmm_moments(h).status, R::InversionStatus::RankOne);
    EXPECT_THROW((void) R::qbmm_atom_moments(lossy), std::invalid_argument);
}

TEST(QbmmInversionTest, RoundedPositiveSingletonAtShiftedHankelBoundaryRetainsEntireCount)
{
    const std::array atoms{R::QbmmAtom{0.4, 5}};
    const R::QbmmMomentBlock h{0.4, 2, 10, 50};
    ASSERT_EQ(R::qbmm_atom_moments(atoms), h);
    const auto result = R::invert_qbmm_moments(h);
    ASSERT_EQ(result.status, R::InversionStatus::RankOne);
    EXPECT_EQ(result.atoms[0].count, 0.4);
    EXPECT_EQ(result.atoms[0].mass_over_scale, 5);
    EXPECT_EQ(result.conditioning.shifted_hankel_margin, 0);
    // An extra positive count at zero mass has no such exact singleton witness.
    const auto phantom = R::invert_qbmm_moments({std::nextafter(1., 2.), 1, 1, 1});
    EXPECT_EQ(phantom.status, R::InversionStatus::NonpositiveSupport);
    EXPECT_EQ(phantom.active_atoms, 0u);
}

TEST(QbmmInversionTest, TwoPositiveAtomsAndUnscaledJacobiRecurrence)
{
    const std::array atoms{R::QbmmAtom{2, 1}, R::QbmmAtom{3, 4}};
    const R::QbmmMomentBlock expected{5, 14, 50, 194};
    EXPECT_EQ(R::qbmm_atom_moments(atoms), expected);
    const auto q = R::invert_qbmm_moments(expected);
    ASSERT_EQ(q.status, R::InversionStatus::TwoNodes);
    ASSERT_EQ(q.active_atoms, 2u);
    EXPECT_NEAR(q.conditioning.a0, 14.0 / 5, 2e-14);
    EXPECT_NEAR(q.conditioning.b1, 54.0 / 25, 2e-14);
    EXPECT_NEAR(q.conditioning.a1, 11.0 / 5, 2e-14);
    for (unsigned i = 0; i < atoms.size(); ++i)
    {
        EXPECT_NEAR(q.atoms[i].count, atoms[i].count, 1e-13);
        EXPECT_NEAR(q.atoms[i].mass_over_scale, atoms[i].mass_over_scale, 1e-13);
    }
    check_reconstruction(q, expected, 1e-13);
}

TEST(QbmmInversionTest, RejectsInconsistentRankNegativeSupportAndNonfiniteDataAtActualStockScale)
{
    const std::array invalid{R::QbmmMomentBlock{2, 1, 1, 1}, // Positive weights on zero and one.
        R::QbmmMomentBlock{1, 2, 1, 1},                      // Material-only update to a monodisperse population.
        R::QbmmMomentBlock{1, 1, 1, 0.5}, R::QbmmMomentBlock{0, 1, 1, 1},
        R::QbmmMomentBlock{2, 1, 5, 7}, // Unit weights at -1 and 2.
        R::QbmmMomentBlock{1, 0, 0, 0}, R::QbmmMomentBlock{1, 1, 1, 1.00001}, R::QbmmMomentBlock{-1, 0, 0, 0}};
    for (const int exponent : {-800, 0, 700})
        for (auto h : invalid)
        {
            for (auto& value : h)
                value = std::ldexp(value, exponent);
            SCOPED_TRACE(::testing::PrintToString(h));
            const auto q = R::invert_qbmm_moments(h);
            EXPECT_FALSE(q.valid());
            EXPECT_EQ(q.active_atoms, 0u);
        }
    for (unsigned k = 0; k < 4; ++k)
        for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        {
            R::QbmmMomentBlock h{5, 14, 50, 194};
            h[k] = bad;
            EXPECT_EQ(R::invert_qbmm_moments(h).status, R::InversionStatus::InvalidInput);
        }
}

TEST(QbmmInversionTest, RankAndReconstructionSurviveCountAndCoordinateRescaling)
{
    for (const auto [count_exponent, coordinate_exponent] : {std::pair{-800, 0}, std::pair{700, 0}, std::pair{0, -250},
             std::pair{0, 250}, std::pair{-560, 230}, std::pair{560, -230}})
    {
        SCOPED_TRACE(::testing::Message() << count_exponent << ", " << coordinate_exponent);
        const double w = std::ldexp(1., count_exponent), x = std::ldexp(1., coordinate_exponent);
        const std::array atoms{R::QbmmAtom{2 * w, x}, R::QbmmAtom{3 * w, 4 * x}};
        const auto h = integrate(atoms);
        const auto q = R::invert_qbmm_moments(h);
        ASSERT_EQ(q.status, R::InversionStatus::TwoNodes);
        ASSERT_EQ(q.active_atoms, 2u);
        for (unsigned i = 0; i < atoms.size(); ++i)
        {
            EXPECT_NEAR(q.atoms[i].count / atoms[i].count, 1, 1e-12);
            EXPECT_NEAR(q.atoms[i].mass_over_scale / atoms[i].mass_over_scale, 1, 1e-12);
        }
        check_reconstruction(q, h, 1e-12);
    }
}

TEST(QbmmInversionTest, JacobiVarianceOutsideDoubleRangeDoesNotRejectRepresentableAtoms)
{
    // The count cancels from b1, whose magnitude scales with the square of the
    // coordinate. Choose inverse count scaling to keep all four input moments finite.
    for (const auto [count_exponent, coordinate_exponent] : {std::pair{-640, 520}, std::pair{640, -550}})
    {
        SCOPED_TRACE(coordinate_exponent);
        const double w = std::ldexp(1., count_exponent), x = std::ldexp(1., coordinate_exponent);
        const std::array atoms{R::QbmmAtom{2 * w, x}, R::QbmmAtom{3 * w, 4 * x}};
        const auto h = integrate(atoms);
        const auto q = R::invert_qbmm_moments(h);
        ASSERT_EQ(q.status, R::InversionStatus::TwoNodes);
        ASSERT_EQ(q.active_atoms, atoms.size());
        EXPECT_TRUE(std::isnan(q.conditioning.b1));
        EXPECT_NEAR(q.conditioning.a0 / x, 14. / 5, 1e-13);
        EXPECT_NEAR(q.conditioning.a1 / x, 11. / 5, 1e-13);
        for (std::size_t i = 0; i < atoms.size(); ++i)
        {
            EXPECT_NEAR(q.atoms[i].count / atoms[i].count, 1., 1e-13);
            EXPECT_NEAR(q.atoms[i].mass_over_scale / atoms[i].mass_over_scale, 1., 1e-13);
        }
        check_reconstruction(q, h, 1e-13);
    }
}

TEST(QbmmInversionTest, WideNodeAndWeightRatiosPreserveBothAtomsOrIdentifyUnresolvedSupport)
{
    const std::array atoms{R::QbmmAtom{1, 1}, R::QbmmAtom{std::ldexp(1., -32), std::ldexp(1., 18)}};
    const auto h = integrate(atoms);
    const auto q = R::invert_qbmm_moments(h);
    ASSERT_EQ(q.status, R::InversionStatus::TwoNodes);
    ASSERT_EQ(q.active_atoms, 2u);
    for (unsigned i = 0; i < atoms.size(); ++i)
    {
        EXPECT_NEAR(q.atoms[i].count / atoms[i].count, 1, 1e-10);
        EXPECT_NEAR(q.atoms[i].mass_over_scale / atoms[i].mass_over_scale, 1, 1e-10);
    }
    check_reconstruction(q, h, 1e-12);

    // Both atoms have unit weight. H2 and H3 have rounded to 1, while H1
    // still resolves a positive displacement from the zero-support boundary.
    const auto unresolved = R::invert_qbmm_moments({2, 1 + std::ldexp(1., -49), 1, 1});
    EXPECT_EQ(unresolved.status, R::InversionStatus::IllConditioned);
    EXPECT_EQ(unresolved.active_atoms, 0u);
    EXPECT_GT(unresolved.conditioning.shifted_hankel_margin, 0);
}

TEST(QbmmInversionTest, ResolvesSeparatedCloseNodesAndBoundsNearRankLoss)
{
    for (const int separation_bits : {8, 20, 24})
    {
        const double gap = std::ldexp(1., -separation_bits);
        const std::array atoms{R::QbmmAtom{1, 1 - gap}, R::QbmmAtom{1, 1 + gap}};
        const auto h = integrate(atoms);
        const auto q = R::invert_qbmm_moments(h);
        SCOPED_TRACE(separation_bits);
        if (separation_bits == 8)
        {
            ASSERT_EQ(q.status, R::InversionStatus::TwoNodes);
            for (unsigned i = 0; i != 2; ++i)
            {
                EXPECT_NEAR(q.atoms[i].mass_over_scale, atoms[i].mass_over_scale, 1e-8);
                EXPECT_NEAR(q.atoms[i].count, 1, 1e-6);
            }
        }
        if (q.valid())
        {
            EXPECT_NE(q.status, R::InversionStatus::Absent);
            check_reconstruction(q, h, R::qbmm_reconstruction_tolerance);
            for (unsigned i = 0; i < q.active_atoms; ++i)
            {
                EXPECT_GT(q.atoms[i].count, 0);
                EXPECT_GT(q.atoms[i].mass_over_scale, 0);
            }
        }
        else
        {
            EXPECT_EQ(q.status, R::InversionStatus::IllConditioned);
            EXPECT_EQ(q.active_atoms, 0u);
        }
    }
}

TEST(QbmmInversionTest, SeededWellSeparatedNodesRecoverAllWeightsCoordinatesAndMoments)
{
    std::mt19937_64 rng(20261119);
    std::uniform_real_distribution<double> low(0.5, 1.5), high(3, 5), weight(0.5, 5);
    for (unsigned sample = 0; sample < 256; ++sample)
    {
        SCOPED_TRACE(sample);
        const double x0 = low(rng), x1 = high(rng);
        const std::array atoms{R::QbmmAtom{weight(rng), x0}, R::QbmmAtom{weight(rng), x1}};
        const auto h = integrate(atoms);
        const auto q = R::invert_qbmm_moments(h);
        ASSERT_EQ(q.status, R::InversionStatus::TwoNodes);
        ASSERT_EQ(q.active_atoms, 2u);
        for (unsigned i = 0; i < atoms.size(); ++i)
        {
            EXPECT_LT(std::abs(q.atoms[i].count - atoms[i].count) / atoms[i].count, 1e-10);
            EXPECT_LT(
                std::abs(q.atoms[i].mass_over_scale - atoms[i].mass_over_scale) / atoms[i].mass_over_scale, 1e-10);
        }
        check_reconstruction(q, h, 1e-11);
    }
}
} // namespace
