/** @file Wire.cc
 * @brief Mode-checked phase record encoding and decoding.
 */
#include "FVM/details/GeneralVoF/Wire.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
void finite_phases(wire::Writer& w, const PLIC::DynamicPhaseMeasures& p)
{
    finite_value(w, p.bulk);
    finite_value(w, p.gas);
    finite_value(w, p.carrier);
}
PLIC::DynamicPhaseMeasures finite_phases(wire::Reader& r, ArithmeticMode mode)
{
    return {finite_value(r, mode), finite_value(r, mode), finite_value(r, mode)};
}
} // namespace SimpleFluid::VoF::general_vof_detail
