/** @file ReactingMixtureCoupler.hh
 * @brief External transactional reaction controller for conservative low-Mach mixture flow.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "equations/ReactionMechanism.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace SimpleFluid::Reacting
{
/** One physical energy/transport phase mapped to distinct chemical compartments.
 * Liquid and Dissolved species share one physical liquid phase. For liquid,
 * species_phases explicitly chooses the inventory for each catalog entry.
 * Gas phases use phase for every species and population for a dispersed bin.
 */
struct MixturePhaseBinding
{
    SpeciesPhase phase = SpeciesPhase::Liquid;
    std::string population;
    std::vector<SpeciesPhase> species_phases;
};

struct CoupledReactionContext
{
    ReactionContext reaction;
    const LowMachMixtureSolver::CellState& thermodynamic_trial; ///< Trial T/p, with actual transported composition.
    std::size_t owned_cell = 0;
    const LowMachMixtureSolver::TransportCellInventory& transported; ///< Pre-source moles/U/work-volume packet.
};

struct ReactingMixtureOptions
{
    std::shared_ptr<const ReactionMechanism> mechanism;
    std::vector<MixturePhaseBinding> phases;
    std::string source_identity;
    std::uint64_t source_revision = 1; ///< Immutable source law revision, identical on every rank.
    std::function<ReactionSourceInput(const CoupledReactionContext&, double)> source;
    /** Optional distribution of external plus chemical heat over physical phases [J].
     * The ledger permits explicit zero-sum allocation for cross-phase reactions.
     * Separate-temperature cross-phase reactions require this callback.
     * The returned sum must equal the supplied amount; material transfer energy
     * is assigned separately to its declared recipient compartment.
     */
    std::function<std::vector<double>(double, const SpeciesSourceLedger&, const LowMachMixtureSolver::CellState&)>
        partition_heat;
    std::size_t liquid_phase = 0;
    std::size_t heat_phase = 0;
    int maximum_correctors = 20; ///< Maximum evaluations in the shared transport/source fixed point.
    double source_relative_tolerance = 1e-8;
    ReactionAdvanceOptions integration;
};

class SIMPLEFLUID_SOLVERS_EXPORT ReactingMixtureCoupler
{
public:
    struct Diagnostics
    {
        int correctors = 0;
        std::vector<double> reaction_extents; ///< MPI signed totals, mol/reaction for this accepted interval.
        std::vector<double> forward_reaction_extents, reverse_reaction_extents; ///< Nonnegative gross accepted extents.
        double chemical_heat = 0, external_heat = 0, transfer_energy = 0, external_mass = 0;
        double net_thermal_energy = 0; ///< MPI total of compensated per-cell heat plus transfer energy, J.
        LowMachMixtureSolver::Diagnostics flow;
    };
    class Trial
    {
    public:
        [[nodiscard]] const LowMachMixtureSolver::State& state() const;
        [[nodiscard]] const Diagnostics& diagnostics() const;
        [[nodiscard]] const std::vector<SpeciesSourceLedger>& local_sources() const;
        [[nodiscard]] const LowMachMixtureSolver::TransportSnapshot& transport_snapshot() const;

    private:
        friend class ReactingMixtureCoupler;
        struct SIMPLEFLUID_SOLVERS_LOCAL Data;
        std::shared_ptr<const Data> d;
    };
    class Checkpoint
    {
        friend class ReactingMixtureCoupler;
        const ReactingMixtureCoupler* owner = nullptr;
        LowMachMixtureSolver::Checkpoint flow;
        Diagnostics diagnostics;
    };

    /** Retains shared ownership and exclusively leases advancement of an initialized solver.
     * Source callbacks are deterministic, MPI-free and have no external mutable history.
     * Each source evaluation consumes the current conservative transport snapshot.
     * Commit and restore never evaluate a source; ledgers belong to its exact final receipt.
     * All methods except const queries are collective on the solver mesh communicator.
     */
    ReactingMixtureCoupler(std::shared_ptr<LowMachMixtureSolver> solver, ReactingMixtureOptions options);
    ~ReactingMixtureCoupler();
    ReactingMixtureCoupler(const ReactingMixtureCoupler&) = delete;
    ReactingMixtureCoupler& operator=(const ReactingMixtureCoupler&) = delete;
    [[nodiscard]] Trial preview(double time_step);
    void commit(const Trial& trial);
    Diagnostics advance(double time_step);
    [[nodiscard]] Checkpoint checkpoint() const;
    void restore(const Checkpoint& checkpoint);
    [[nodiscard]] const std::shared_ptr<LowMachMixtureSolver>& solver() const noexcept;
    [[nodiscard]] const Diagnostics& diagnostics() const noexcept;

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Implementation;
    std::unique_ptr<Implementation> d;
};
} // namespace SimpleFluid::Reacting
