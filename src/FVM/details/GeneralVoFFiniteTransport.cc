/** @file GeneralVoFFiniteTransport.cc
 * @brief Runtime entry dispatch and compiled finite transport kernels.
 */
#include "FVM/details/GeneralVoFFiniteTransport.tcc"
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
GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap(const Hierarchy::Plan& plan,
    std::shared_ptr<const Hierarchy> target, std::span<const GeneralVoFFiniteOwnedChannels> channels,
    GeneralVoFFiniteRemapOptions options) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return plan_remap_finite<ArithmeticMode::Ordinary>(plan, std::move(target), channels, std::move(options));
    return plan_remap_finite<ArithmeticMode::BoundedCertification>(
        plan, std::move(target), channels, std::move(options));
}
template GeneralVoF::Trial GeneralVoF::plan_transport_finite<ArithmeticMode::Ordinary>(
    std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
template GeneralVoF::Trial GeneralVoF::plan_transport_finite<ArithmeticMode::BoundedCertification>(
    std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
template GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap_finite<ArithmeticMode::Ordinary>(const Hierarchy::Plan&,
    std::shared_ptr<const Hierarchy>, std::span<const GeneralVoFFiniteOwnedChannels>,
    GeneralVoFFiniteRemapOptions) const;
template GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap_finite<ArithmeticMode::BoundedCertification>(
    const Hierarchy::Plan&, std::shared_ptr<const Hierarchy>, std::span<const GeneralVoFFiniteOwnedChannels>,
    GeneralVoFFiniteRemapOptions) const;
} // namespace SimpleFluid::VoF
