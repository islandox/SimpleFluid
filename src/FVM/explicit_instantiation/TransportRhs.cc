/** @file FVM/explicit_instantiation/TransportRhs.cc
 * @brief Default-pack instantiations for basic scalar/vector transport RHS adapters.
 */
#include "FVM/details/TransportRhs.tcc"

namespace SimpleFluid::FVM::detail
{

template struct BasicScalarTransportRhs<DefaultTpetraTypes>;
template struct BasicVectorTransportRhs<DefaultTpetraTypes>;

} // namespace SimpleFluid::FVM::detail
