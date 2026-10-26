/** @file Execution.cc
 * @brief Collective native imports and scalar checks shared by General VoF operations.
 */
#include "solvers/details/GeneralVoF/Execution.hh"
#include <algorithm>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
void Execution::require(bool condition, const char* message) const
{
    vof_failure::collective<std::runtime_error>(
        *mesh_.owned_cell_map()->getComm(),
        [&]
        {
            if (!condition)
                throw std::invalid_argument(message);
        },
        "solver_admission");
}
double Execution::maximum(double value) const
{
    double result = 0;
    Teuchos::reduceAll(*mesh_.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &value, &result);
    return result;
}
std::vector<double> Execution::import_owned(
    std::span<const double> owned, const Teuchos::RCP<const Pack::map_type>& target) const
{
    require(owned.size() == mesh_.num_owned_cells(), "General VoF owned import dimensions differ.");
    Pack::vector_type from(mesh_.owned_cell_map()), to(target);
    {
        auto values = from.getDataNonConst();
        std::copy(owned.begin(), owned.end(), values.begin());
    }
    Pack::import_type plan(mesh_.owned_cell_map(), target);
    to.doImport(from, plan, Tpetra::INSERT);
    const auto values = to.getData();
    return {values.begin(), values.end()};
}
} // namespace SimpleFluid::detail::general_vof
