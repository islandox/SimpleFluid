/** @file Diagnostics.cc
 * @brief Supported compiled physical-diagnostics modes.
 */
#include "solvers/details/GeneralVoF/Diagnostics.tcc"
namespace SimpleFluid::detail::general_vof
{
template struct Diagnostics<ArithmeticMode::Ordinary>;
template struct Diagnostics<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
