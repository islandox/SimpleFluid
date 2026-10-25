/** @file SolverImpl.ipp
 * @brief Caller-dependent collective exception propagation.
 */
#pragma once
#include <Teuchos_CommHelpers.hpp>
#include <exception>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
template<class Work> void GeneralVoFSolver::Impl::collective(Work&& work) const
{
    std::exception_ptr error;
    try
    {
        std::forward<Work>(work)();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = bool(error), global = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
    if (global)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("General native VoF rejected on another rank.");
    }
}
} // namespace SimpleFluid
