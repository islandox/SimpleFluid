/** @file Transport.cc
 * @brief Operation dispatch and trial publication validation.
 */
#include "FVM/details/GeneralVoFInternal.hh"
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
} // namespace SimpleFluid::VoF
