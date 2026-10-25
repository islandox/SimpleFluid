/** @file Advance.cc
 * @brief Compiled general VoF Advance operation.
 */
#include "solvers/details/GeneralVoF/Advance.tcc"

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
template struct Advance<ArithmeticMode::Ordinary>;
template struct Advance<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
