/** @file GeneralVoFPhiClosureSupport.ipp
 * @brief Caller-dependent collective exception propagation.
 */
#pragma once
namespace SimpleFluid::detail::phi_support
{
template<class Work> void collective(const Teuchos::Comm<int>& comm, Work work)
{
    std::exception_ptr error;
    int local = 0, global = 0;
    try
    {
        work();
    }
    catch (const GeneralVoFPhiInfeasible&)
    {
        error = std::current_exception();
        local = 1;
    }
    catch (const PhiProposalResource&)
    {
        error = std::current_exception();
        local = 2;
    }
    catch (...)
    {
        error = std::current_exception();
        local = 3;
    }
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &global);
    if (global == 1)
    {
        if (error)
            std::rethrow_exception(error);
        throw GeneralVoFPhiInfeasible("Bounded arithmetic closure infeasible on another rank.");
    }
    if (global == 2)
    {
        if (local == 2)
            std::rethrow_exception(error);
        throw PhiProposalResource("Bounded arithmetic closure resources exhausted on another rank.");
    }
    if (global == 3)
    {
        if (local == 3)
            std::rethrow_exception(error);
        throw std::runtime_error("General VoF Phi arithmetic closure invalid on another rank.");
    }
}
} // namespace SimpleFluid::detail::phi_support
