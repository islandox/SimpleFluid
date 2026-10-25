/** @file Transport.cc
 * @brief Supported compiled General VoF arithmetic modes.
 */
#include "FVM/details/GeneralVoF/Transport.tcc"
namespace SimpleFluid::VoF
{
template GeneralVoF::Trial GeneralVoF::plan_transport_finite<ArithmeticMode::Ordinary>(
    std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
template GeneralVoF::Trial GeneralVoF::plan_transport_finite<ArithmeticMode::BoundedCertification>(
    std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
} // namespace SimpleFluid::VoF
