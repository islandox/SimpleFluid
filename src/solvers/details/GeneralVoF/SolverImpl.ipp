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
    detail::general_vof::Execution(*mesh).collective(std::forward<Work>(work));
}
} // namespace SimpleFluid
