/** @file Remap.cc
 * @brief Compiled general VoF Remap operation.
 */
#include "solvers/details/GeneralVoF/Remap.tcc"

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
template struct Remap<ArithmeticMode::Ordinary>;
template struct Remap<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
