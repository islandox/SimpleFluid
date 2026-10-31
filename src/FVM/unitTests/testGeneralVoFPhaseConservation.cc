/** @file testGeneralVoFPhaseConservation.cc
 * @brief Independent threshold, arithmetic-policy and phase-isolation checks.
 */
#include "FVM/details/GeneralVoF/PhasePublication.hh"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace local = SimpleFluid::VoF::general_vof_detail;
using Policy = local::PhaseConservationPolicy;
using Result = local::PhaseConservationResult;
using Summary = local::PhaseConservationSummary;
constexpr auto infinity = std::numeric_limits<double>::infinity();

TEST(GeneralVoFPhaseConservation, OrdinaryPhysicalAllowanceIncludesEquality)
{
    Summary summary{1, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ(local::check_phase_conservation<Policy::Ordinary>(summary, 1, 1), Result::Accepted);
    EXPECT_EQ(local::check_phase_conservation<Policy::Bounded>(summary, 1, 1), Result::Nonconserved);
    summary[0] = std::nextafter(1., infinity);
    EXPECT_EQ(local::check_phase_conservation<Policy::Ordinary>(summary, 1, 1), Result::Nonconserved);
}

TEST(GeneralVoFPhaseConservation, ZeroAllowanceRejectsTheSmallestRepresentableChange)
{
    Summary summary{};
    const auto check = [&]<Policy policy>()
    {
        EXPECT_EQ(local::check_phase_conservation<policy>(summary, 1, 0), Result::Accepted);
        summary[0] = std::nextafter(0., infinity);
        EXPECT_EQ(local::check_phase_conservation<policy>(summary, 1, 0), Result::Nonconserved);
        summary[0] = 0;
    };
    check.template operator()<Policy::Ordinary>();
    check.template operator()<Policy::Bounded>();
}

TEST(GeneralVoFPhaseConservation, OppositePhaseChangesAndOtherPhaseAllowanceCannotCancel)
{
    const Summary cancellation{1, -1, 0, 0, 2, 2, 0, 0};
    const Summary other_phase_allowance{0, 1, 2, 0, 0, 1, 0, 0};
    const Summary own_publication_allowance{1, 0, 2, 0, 1, 0, 0, 0};
    const auto check = [&]<Policy policy>()
    {
        EXPECT_EQ(local::check_phase_conservation<policy>(cancellation, 1, 0), Result::Nonconserved);
        EXPECT_EQ(local::check_phase_conservation<policy>(other_phase_allowance, 1, 0), Result::Nonconserved);
        EXPECT_EQ(local::check_phase_conservation<policy>(own_publication_allowance, 1, 0), Result::Accepted);
    };
    check.template operator()<Policy::Ordinary>();
    check.template operator()<Policy::Bounded>();
}

TEST(GeneralVoFPhaseConservation, OrdinaryPublicationAdditionRetainsBinary64Rounding)
{
    const auto h = std::ldexp(1., -53);
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        Summary summary{};
        summary[phase] = 1 + 4 * h;
        summary[2 + phase] = 3 * h;
        summary[6 + phase] = 1;
        // The binary64 sum 1 + 3h rounds to 1 + 4h before the reduction allowance.
        EXPECT_EQ(local::check_phase_conservation<Policy::Ordinary>(summary, 1, 1), Result::Accepted);
        EXPECT_EQ(local::check_phase_conservation<Policy::Bounded>(summary, 1, 1), Result::Nonconserved);
        summary[phase] = 1 + 148 * h;
        summary[4 + phase] = 3 + 148 * h;
        // This boundary also exercises the subsequent long-double reduction allowance.
        EXPECT_EQ(local::check_phase_conservation<Policy::Ordinary>(summary, 1, 1), Result::Accepted);
        EXPECT_EQ(local::check_phase_conservation<Policy::Bounded>(summary, 1, 1), Result::Nonconserved);
    }
}

TEST(GeneralVoFPhaseConservation, OrdinaryPhysicalProductAndPublicationSumRetainBinary64Overflow)
{
    const auto maximum = std::numeric_limits<double>::max();
    const Summary physical_overflow{0, 0, 0, 0, 0, 0, maximum, 0};
    EXPECT_EQ(local::check_phase_conservation<Policy::Ordinary>(physical_overflow, 1, 2), Result::Nonconserved);
    EXPECT_EQ(local::check_phase_conservation<Policy::Bounded>(physical_overflow, 1, 2), Result::Accepted);
    const Summary addition_overflow{0, 0, maximum, 0, 0, 0, maximum, 0};
    EXPECT_EQ(local::check_phase_conservation<Policy::Ordinary>(addition_overflow, 1, 1), Result::Nonconserved);
}

TEST(GeneralVoFPhaseConservation, NonfiniteChangesAndBoundsReject)
{
    const auto check = []<Policy policy>()
    {
        for (const auto change : {std::numeric_limits<double>::quiet_NaN(), infinity, -infinity})
        {
            const Summary summary{change, 0, 0, 0, 0, 0, 0, 0};
            EXPECT_EQ(local::check_phase_conservation<policy>(summary, 1, 0), Result::Nonconserved);
        }
        const Summary infinite_error{0, 0, infinity, 0, 0, 0, 0, 0};
        const Summary nan_magnitude{0, 0, 0, 0, std::numeric_limits<double>::quiet_NaN(), 0, 0, 0};
        EXPECT_EQ(local::check_phase_conservation<policy>(infinite_error, 1, 0), Result::Nonconserved);
        EXPECT_EQ(local::check_phase_conservation<policy>(nan_magnitude, 1, 0), Result::Nonconserved);
    };
    check.template operator()<Policy::Ordinary>();
    check.template operator()<Policy::Bounded>();
}
} // namespace
