/** @file Execution.ipp
 * @brief Caller-defined local work with collective failure propagation.
 */
#pragma once
namespace SimpleFluid::detail::general_vof
{
template<class Work> void Execution::collective(Work&& work) const
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
    Teuchos::reduceAll(*mesh_.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
    if (global)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("General native VoF rejected on another rank.");
    }
}
} // namespace SimpleFluid::detail::general_vof
