#include "solvers/LowMachMixtureSolver.hh"

#include "equations/CollectiveValidation.hh"

#include <stdexcept>

namespace SimpleFluid::Reacting
{
void LowMachMixtureSolver::initialize_qbmm(
    State legacy, const QbmmInitialization& initialization, const ControllerLease* controller)
{
    require_controller(controller);
    supply_qbmm_shape(legacy, initialization);
    initialize(std::move(legacy), controller);
}

void LowMachMixtureSolver::supply_qbmm_shape(State& legacy, const QbmmInitialization& initialization) const
{
    const auto& families = options().qbmm_families;
    collective_detail::collective_local_validation(*mesh(), "Explicit QBMM initialization of missing shape",
        [&]
        {
            if (families.empty() || !legacy.qbmm_families.empty() ||
                initialization.cells.size() != legacy.cells.size() || legacy.cells.size() != mesh()->num_owned_cells())
                throw std::invalid_argument("QBMM migration needs a count-only owned state and explicit cell choices.");
            legacy.qbmm_families.resize(families.size());
            for (std::size_t p = 0; p < families.size(); ++p)
                if (families[p])
                    legacy.qbmm_families[p] = qbmm_family_stamp(*families[p]);
            for (std::size_t c = 0; c < legacy.cells.size(); ++c)
            {
                auto& cell = legacy.cells[c];
                if (!cell.qbmm_shape.empty() || !cell.qbmm_kinetic_closure.empty() ||
                    cell.phases.size() != families.size() || cell.phase_numbers.size() != families.size() ||
                    initialization.cells[c].size() != families.size())
                    throw std::invalid_argument("QBMM migration requires a count-only cell in physical phase order.");
                cell.qbmm_shape.resize(families.size());
                for (std::size_t p = 0; p < families.size(); ++p)
                {
                    const auto& choice = initialization.cells[c][p];
                    if (choice.has_value() != bool(families[p]))
                        throw std::invalid_argument("QBMM migration requires a choice for every enabled phase only.");
                    if (!choice)
                        continue;
                    const auto& phase = cell.phases[p];
                    const auto& family = *families[p];
                    if (!cell.phase_numbers[p] || !phase.thermodynamics ||
                        phase.thermodynamics->catalog().fingerprint() != family.catalog().fingerprint())
                        throw std::invalid_argument(
                            "Count-only migration needs an engaged count and matching material catalog.");
                    const double count = *cell.phase_numbers[p];
                    if (const auto* atoms = std::get_if<std::vector<QbmmAtom>>(&*choice))
                        cell.qbmm_shape[p] = qbmm_migrate_mass_pivots(family, count, phase.moles, *atoms);
                    else
                        cell.qbmm_shape[p] = qbmm_initialize_monodisperse(family, count, phase.moles);
                }
            }
        });
}
} // namespace SimpleFluid::Reacting
