/** @file GeneralVoFProjectionSupport.ipp
 * @brief Caller-dependent collective arithmetic failure guard.
 */
#pragma once
#include <Teuchos_CommHelpers.hpp>
#include <stdexcept>

namespace SimpleFluid::detail::projection_support
{
template<class Work> void arithmetic_guard(const Teuchos::Comm<int>& comm, Work&& work, const char* message)
{
    bool accepted = true;
    try
    {
        work();
    }
    catch (...)
    {
        accepted = false;
    }
    int local = accepted, global = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error(message);
}
} // namespace SimpleFluid::detail::projection_support
