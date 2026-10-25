/** @file Remap.cc
 * @brief Operation dispatch and trial publication validation.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
void GeneralVoF::FiniteRemapTrial::validate_observations() const
{
    if (!source_)
        throw std::invalid_argument("Finite remap has no authenticated source.");
    wire::collective(*source_->hierarchy->communicator(),
        [&]
        {
            current(*source_);
            if (!candidate_ || source_->options.arithmetic_mode == ArithmeticMode::Exact ||
                candidate_->options.arithmetic_mode == ArithmeticMode::Exact ||
                candidate_->options.arithmetic_mode != source_->options.arithmetic_mode)
                throw std::invalid_argument("Finite remap has no compatible candidate.");
            current(*candidate_);
        });
}
std::shared_ptr<const GeneralVoF> GeneralVoF::FiniteRemapTrial::candidate() const
{
    validate_observations();
    return std::shared_ptr<const GeneralVoF>(new GeneralVoF(candidate_));
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
} // namespace SimpleFluid::VoF
