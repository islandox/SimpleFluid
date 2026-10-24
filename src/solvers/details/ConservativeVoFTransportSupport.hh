/** @file ConservativeVoFTransportSupport.hh
 * @brief Private collective input guard shared by transport arithmetic backends.
 */
#pragma once
#include "solvers/details/ConservativeVoFTransport.hh"
#include <Teuchos_CommHelpers.hpp>
#include <stdexcept>
namespace SimpleFluid::detail::transport_support
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
inline void require(const Pack::map_type& map, bool valid, const char* message)
{
    int local = valid, global = 0;
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error(message);
}
} // namespace SimpleFluid::detail::transport_support
