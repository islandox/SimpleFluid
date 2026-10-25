/** @file Remap.cc
 * @brief Supported compiled General VoF arithmetic modes.
 */
#include "FVM/details/GeneralVoF/Remap.tcc"
namespace SimpleFluid::VoF
{
template GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap_finite<ArithmeticMode::Ordinary>(const Hierarchy::Plan&,
    std::shared_ptr<const Hierarchy>, std::span<const GeneralVoFFiniteOwnedChannels>,
    GeneralVoFFiniteRemapOptions) const;
template GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap_finite<ArithmeticMode::BoundedCertification>(
    const Hierarchy::Plan&, std::shared_ptr<const Hierarchy>, std::span<const GeneralVoFFiniteOwnedChannels>,
    GeneralVoFFiniteRemapOptions) const;
} // namespace SimpleFluid::VoF
