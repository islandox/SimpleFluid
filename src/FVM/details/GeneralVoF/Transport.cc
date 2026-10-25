/** @file Transport.cc
 * @brief Operation entry and supported compiled arithmetic modes.
 */
#include "FVM/details/GeneralVoF/Transport.tcc"
namespace SimpleFluid::VoF
{
GeneralVoF::Trial GeneralVoF::plan_transport(std::span<const GeneralVoFNodeVelocity> velocity,
    std::span<const GeneralVoFFaceVolume> volumes, double timestep) const
{
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        return Exact::GeneralVoF::plan_transport(*this, velocity, volumes, timestep);
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return plan_transport_finite<ArithmeticMode::Ordinary>(velocity, volumes, timestep);
    return plan_transport_finite<ArithmeticMode::BoundedCertification>(velocity, volumes, timestep);
}
template GeneralVoF::Trial GeneralVoF::plan_transport_finite<ArithmeticMode::Ordinary>(
    std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
template GeneralVoF::Trial GeneralVoF::plan_transport_finite<ArithmeticMode::BoundedCertification>(
    std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
} // namespace SimpleFluid::VoF
