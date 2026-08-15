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
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::collective_detail
{

/** @brief Reusable storage for allocation-free validation batches after warmup. */
struct CollectiveValidationBatchScratch
{
    std::vector<int> local_status;
    std::vector<int> global_status;
    std::vector<unsigned char> globally_active;
};

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

/**
 * @brief Run a batch of optional local operations with one failure reduction.
 *
 * The range size and order must agree on every rank. Active status is reduced
 * alongside callback failures, so a rank-local enablement mismatch fails on
 * every rank before the caller enters any field import. The returned entries
 * identify operations that were active on every rank.
 *
 * @param mesh Distributed mesh providing the communicator.
 * @param context Diagnostic context reported by peer ranks.
 * @param operations Rank-consistent ordered range of operation descriptors.
 * @param is_active Pure rank-local predicate selecting dynamic work.
 * @param operation Pure rank-local work; it must not communicate.
 * @param scratch Reusable communication and result storage.
 * @return One byte per operation, nonzero only when active on every rank.
 */
template<TpetraTypePack Pack, class Range, class ActivePredicate,
         class Operation>
const std::vector<unsigned char>& collective_local_validation_batch(
    const Mesh<Pack>& mesh,
    std::string_view context,
    Range&& operations,
    ActivePredicate&& is_active,
    Operation&& operation,
    CollectiveValidationBatchScratch& scratch)
{
    using std::begin;
    using std::end;
    const auto operation_count = static_cast<size_t>(
        std::distance(begin(operations), end(operations)));
    scratch.local_status.assign(operation_count + 1, 0);
    scratch.global_status.assign(operation_count + 1, 0);
    scratch.globally_active.assign(operation_count, 0);
    auto& local_status = scratch.local_status;
    auto& global_status = scratch.global_status;
    std::exception_ptr local_error;

    size_t index = 0;
    for (auto&& descriptor : operations)
    {
        bool active = false;
        try
        {
            active = std::invoke(is_active, descriptor);
            local_status[index] = active ? 1 : 0;
            if (active)
            {
                std::invoke(operation, descriptor);
            }
        }
        catch (...)
        {
            if (!local_error)
            {
                local_error = std::current_exception();
            }
        }
        ++index;
    }
    local_status.back() = local_error ? 1 : 0;

    const auto communicator = mesh.owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_SUM,
        static_cast<int>(local_status.size()),
        local_status.data(),
        global_status.data());

    const bool any_failed = global_status.back() != 0;
    if (any_failed)
    {
        if (local_error)
        {
            std::rethrow_exception(local_error);
        }
        throw std::runtime_error(
            std::string(context) + " failed on another rank.");
    }

    const auto rank_count = communicator->getSize();
    for (size_t operation_index = 0;
         operation_index < operation_count;
         ++operation_index)
    {
        const auto active_count = global_status[operation_index];
        if (active_count != 0 && active_count != rank_count)
        {
            throw std::invalid_argument(
                std::string(context)
                + " activation must agree on every rank.");
        }
        scratch.globally_active[operation_index] =
            active_count == rank_count ? 1 : 0;
    }
    return scratch.globally_active;
}

} // namespace SimpleFluid::collective_detail
