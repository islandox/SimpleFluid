/** @file PhaseStorage.cc
 * @brief Supported compiled physical-storage modes.
 */
#include "solvers/details/GeneralVoF/PhaseStorage.tcc"
namespace SimpleFluid::detail::general_vof
{
template struct PhaseStorage<ArithmeticMode::Ordinary>;
template struct PhaseStorage<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
