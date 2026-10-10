#pragma once

#include <Teuchos_Comm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

namespace SimpleFluid
{
/** Identical ordered rank groups, including independently duplicated contexts. */
inline bool communicators_congruent(const Teuchos::Comm<int>& a, const Teuchos::Comm<int>& b)
{
    if (&a == &b)
        return true;
#ifdef HAVE_TEUCHOS_MPI
    const auto* ma = dynamic_cast<const Teuchos::MpiComm<int>*>(&a);
    const auto* mb = dynamic_cast<const Teuchos::MpiComm<int>*>(&b);
    if (ma && mb)
    {
        int comparison = MPI_UNEQUAL;
        return MPI_Comm_compare(*ma->getRawMpiComm(), *mb->getRawMpiComm(), &comparison) == MPI_SUCCESS &&
               (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
    }
#endif
    return a.getSize() == 1 && b.getSize() == 1;
}
} // namespace SimpleFluid
