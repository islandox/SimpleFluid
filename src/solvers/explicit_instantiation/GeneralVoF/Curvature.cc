/** @file Curvature.cc
 * @brief Supported compiled curvature assembly modes.
 */
#include "solvers/details/GeneralVoF/Curvature.tcc"
namespace SimpleFluid::detail::general_vof
{
template struct Curvature<ArithmeticMode::Ordinary>;
template struct Curvature<ArithmeticMode::BoundedCertification>;
template struct Curvature<ArithmeticMode::Exact>;
} // namespace SimpleFluid::detail::general_vof
