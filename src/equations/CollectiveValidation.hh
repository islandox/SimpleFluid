/**
 * @file CollectiveValidation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Rank-coherent validation helpers for distributed equation state.
 * @version 0.1
 * @date 2026-07-22
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/Mesh.hh"

#include <Teuchos_CommHelpers.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace SimpleFluid::collective_detail
{

/**
 * @brief Run pure local work and propagate a failure to every rank.
 *
 * The operation must not communicate. A collective reduction follows it so
 * every rank fails before a caller enters its next import, assembly, or solve.
 * The rank where the failure originated retains its original exception; peers
 * receive a runtime error containing @p context.
 */
template<TpetraTypePack Pack, class Operation>
void collective_local_validation(
    const Mesh<Pack>& mesh,
    std::string_view context,
    Operation&& operation)
{
    std::exception_ptr local_error;
    try
    {
        std::forward<Operation>(operation)();
    }
    catch (...)
    {
        local_error = std::current_exception();
    }

    const int local_failed = local_error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_MAX,
        1,
        &local_failed,
        &any_failed);
    if (any_failed == 0)
    {
        return;
    }
    if (local_error)
    {
        std::rethrow_exception(local_error);
    }
    throw std::runtime_error(
        std::string(context) + " failed on another rank.");
}

} // namespace SimpleFluid::collective_detail
