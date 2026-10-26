/** @file Execution.ipp
 * @brief Caller-defined local work with collective failure propagation.
 */
#pragma once
namespace SimpleFluid::detail::general_vof
{
template<class Work> void Execution::collective(Work&& work, const char* stage) const
{
    vof_failure::collective<std::runtime_error>(*mesh_.owned_cell_map()->getComm(), std::forward<Work>(work), stage);
}
} // namespace SimpleFluid::detail::general_vof
