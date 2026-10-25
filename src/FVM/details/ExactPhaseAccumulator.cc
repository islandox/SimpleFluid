/** @file ExactPhaseAccumulator.cc
 * @brief Compiled raw geometric phase accumulation before donor relabelling.
 */
#include "FVM/details/ExactPhaseAccumulator.hh"
#include <limits>
#include <stdexcept>
namespace SimpleFluid::VoF::detail
{
ExactPhaseAccumulator::ExactPhaseAccumulator(ExactLiveStorage& live)
    : live_(live), scope_(live, {&raw_.bulk, &raw_.gas, &raw_.carrier})
{
}
void ExactPhaseAccumulator::add(
    const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& input, int sign)
{
    if ((sign != 1 && sign != -1) || count_ == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Invalid exact raw phase group sign/count.");
    auto bulk_source = live_.pin(input.bulk);
    auto gas_source = live_.pin(input.gas);
    auto carrier_source = live_.pin(input.carrier);
    // Preserve the old per-piece complementary-label admission before using
    // linearity; certified geometric inputs never skip this validity check.
    if (live_.sum(input.gas, input.carrier).compare(input.bulk))
        throw std::invalid_argument("Exact raw phase group contains a noncomplementary piece.");
    if (sign == 1)
    {
        live_.add(raw_.bulk, input.bulk);
        live_.add(raw_.gas, input.gas);
        live_.add(raw_.carrier, input.carrier);
    }
    else
    {
        live_.subtract(raw_.bulk, input.bulk);
        live_.subtract(raw_.gas, input.gas);
        live_.subtract(raw_.carrier, input.carrier);
    }
    ++count_;
}
SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> ExactPhaseAccumulator::apply(
    const ExactPhaseLabelMap& map) const
{
    return map.apply(raw_, live_.operation_bytes());
}
} // namespace SimpleFluid::VoF::detail
