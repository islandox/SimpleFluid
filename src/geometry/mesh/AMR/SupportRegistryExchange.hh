/** @file SupportRegistryExchange.hh
 * @brief Budgeted, authenticated streaming support membership exchange.
 */
#pragma once
#include "geometry/mesh/AMR/SupportRegistry.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <functional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class PooledSupportRegistry;
struct Budget;
/** Route owner-local changes to support authorities in synchronized byte pages.
 * next is called only under a collective error vote and returns false forever
 * after exhaustion. A failure may leave this registry partly changed; it must
 * belong to a discardable candidate. cursor_capacity is a checked upper bound
 * on the returned RegistryChange's support and dynamic ancestry capacities;
 * callback-internal work admits its own scoped scratch in the same arena.
 * The arena outlives this collective. */
void exchange_support_registry_stream(PooledSupportRegistry&, const std::function<bool(RegistryChange&)>& next,
    ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t page_bytes, Budget&,
    std::size_t cursor_capacity);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
