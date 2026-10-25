/** @file Initialization.cc
 * @brief Compiled general VoF Initialization operation.
 */
#include "solvers/details/GeneralVoF/Initialization.tcc"

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
template struct Initialization<ArithmeticMode::Ordinary>;
template struct Initialization<ArithmeticMode::BoundedCertification>;
template struct Initialization<ArithmeticMode::Exact>;
} // namespace SimpleFluid::detail::general_vof
