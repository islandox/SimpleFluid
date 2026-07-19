/**
 * @file TurbulenceCollectiveValidation.hh
 * @brief Rank-coherent validation helpers for distributed turbulence state.
 */

#pragma once

#include "geometry/Mesh.hh"

#include <Teuchos_CommHelpers.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace SimpleFluid::turbulence_detail
{

/**
 * @brief Run pure local validation and propagate failure to every rank.
 *
 * The callback must not perform communication. A collective reduction follows
 * it, allowing all ranks to fail before any caller enters its next import,
 * matrix assembly, or linear solve.
 */
template <TpetraTypePack Pack, class Operation>
void collective_local_validation(const Mesh<Pack>& mesh, std::string_view context,
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
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_failed,
                       &any_failed);
    if (any_failed == 0)
    {
        return;
    }
    if (local_error)
    {
        std::rethrow_exception(local_error);
    }
    throw std::runtime_error(std::string(context) + " failed on another rank.");
}

/** @brief Require one integral configuration value on every rank. */
template <TpetraTypePack Pack>
void require_uniform_integral(const Mesh<Pack>& mesh, int local_value, std::string_view context)
{
    int minimum = 0;
    int maximum = 0;
    const auto communicator = mesh.owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_value, &minimum);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_value, &maximum);
    if (minimum != maximum)
    {
        throw std::invalid_argument(std::string(context) + " must agree on every rank.");
    }
}

/** @brief Require one finite floating-point configuration value on every rank. */
template <TpetraTypePack Pack>
void require_uniform_real(const Mesh<Pack>& mesh, real_t local_value, std::string_view context)
{
    real_t minimum = {};
    real_t maximum = {};
    const auto communicator = mesh.owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_value, &minimum);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_value, &maximum);
    if (minimum != maximum)
    {
        throw std::invalid_argument(std::string(context) + " must agree on every rank.");
    }
}

} // namespace SimpleFluid::turbulence_detail
