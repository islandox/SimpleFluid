/** @file VoFFailure.cc
 * @brief Single diagnostic context provider shared by VoF library boundaries.
 */
#include "FVM/details/VoFFailure.hh"
namespace SimpleFluid::detail::vof_failure
{
ContextState& context_state() noexcept
{
    static thread_local ContextState state;
    return state;
}
} // namespace SimpleFluid::detail::vof_failure
