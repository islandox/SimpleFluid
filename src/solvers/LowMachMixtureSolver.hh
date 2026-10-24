/** @file LowMachMixtureSolver.hh
 * @brief Native conservative multicomponent low-Mach flow with supplied caloric mixture EOS.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "equations/LowMachPressureEquation.hh"
#include "equations/ReactingThermodynamics.hh"
#include <cstdint>
#include <memory>
#include <vector>

namespace SimpleFluid::Reacting
{
class SIMPLEFLUID_SOLVERS_EXPORT LowMachMixtureSolver
{
public:
    using Pack = DefaultTpetraTypes;
    using Mesh = MeshHandle<Pack>;
    using Traits = MeshFieldTraits<Pack, Mesh>;
    using Vector = Traits::vector_cell_type::vec_type;
    struct Options
    {
        PhaseThermalMode thermal_mode = PhaseThermalMode::Equilibrium;
        double minimum_pressure = 0, maximum_pressure = 0; ///< Explicit uniform-p0 bracket, Pa.
        double volume_relative_tolerance = 2e-9, conservation_relative_tolerance = 2e-10;
        double continuity_tolerance = 1e-10;
        int maximum_correctors = 80, maximum_pressure_iterations = 100;
        LinearSolverOptions linear;
        std::function<double(std::span<const PhaseThermalState>, std::span<const double>)> mixture_dynamic_viscosity;
    };
    struct CellState
    {
        std::vector<PhaseThermalState> phases; ///< One liquid (including dissolved), then independently owned gas bins.
        std::vector<Vector> drift_velocity; ///< Prescribed phase velocity relative to the shared continuous velocity.
        Vector velocity{};
        double mechanical_pressure = 0;
    };
    struct State
    {
        std::vector<CellState> cells; ///< Owned cells only, mesh order.
        double thermodynamic_pressure = 0, time = 0;
        std::uint64_t revision = 0;
    };
    struct CellSources
    {
        std::vector<std::vector<double>> species_moles; ///< Integrated signed mol per phase/species; empty means zero.
        std::vector<double> internal_energy;            ///< Integrated J per phase, excludes solver-owned p*dV work.
        Vector momentum{};                              ///< Integrated external physical kg m/s.
    };
    struct Diagnostics
    {
        int correctors = 0;
        double maximum_occupancy_error = 0, maximum_continuity_residual = 0;
        double mass_error = 0, internal_energy_error = 0, pressure_work = 0;
        double caloric_energy_scale = 0; ///< Sum m*T*de/dT plus reference-free source/work/transport scales, J.
        double internal_energy_roundoff_bound = 0; ///< Explicit energy-operand cancellation bound, J.
        double empty_phase_energy_roundoff = 0;    ///< Bounded operand cancellation; never physical heat.
        Vector momentum_error{};
        std::vector<double> species_errors;
        LinearSolveSummary transport_linear, momentum_linear;
    };
    class Checkpoint
    {
        friend class LowMachMixtureSolver;
        State state;
        const LowMachMixtureSolver* owner = nullptr;
        std::uint64_t geometry_epoch = 0;
        std::vector<double> face_flux;
        Diagnostics diagnostics;
    };
    class ControllerLease
    {
        friend class LowMachMixtureSolver;
        LowMachMixtureSolver* owner = nullptr;
        std::uint64_t id = 0;
        SIMPLEFLUID_SOLVERS_LOCAL ControllerLease(LowMachMixtureSolver* solver, std::uint64_t value) : owner(solver), id(value) {}

    public:
        ControllerLease(const ControllerLease&) = delete;
        ControllerLease& operator=(const ControllerLease&) = delete;
        ControllerLease(ControllerLease&& other) noexcept;
        ControllerLease& operator=(ControllerLease&& other) noexcept;
        ~ControllerLease();
    };
    class Trial
    {
        friend class LowMachMixtureSolver;
        State candidate;
        Diagnostics statistics;
        std::vector<double> face_flux;
        const LowMachMixtureSolver* owner = nullptr;
        std::uint64_t nonce = 0, base_revision = 0, geometry_epoch = 0;

    public:
        [[nodiscard]] const State& state() const noexcept { return candidate; }
        [[nodiscard]] const Diagnostics& diagnostics() const noexcept { return statistics; }
        [[nodiscard]] std::span<const double> owned_volume_flux() const noexcept { return face_flux; }
    };
    /** One external controller may stage/commit while retaining shared ownership of this solver. */
    [[nodiscard]] ControllerLease acquire_controller();
    [[nodiscard]] Trial preview(double time_step, const std::vector<CellSources>& integrated_sources = {},
        const ControllerLease* controller = nullptr);
    void commit(const Trial& trial, const ControllerLease* controller = nullptr);
    LowMachMixtureSolver(SP<const Mesh> mesh,
        std::vector<std::shared_ptr<const MixtureThermodynamics>> phase_thermodynamics, Options options);
    ~LowMachMixtureSolver();
    LowMachMixtureSolver(const LowMachMixtureSolver&) = delete;
    LowMachMixtureSolver& operator=(const LowMachMixtureSolver&) = delete;
    void initialize(State state, const ControllerLease* controller = nullptr);
    /** Closed, fixed orthogonal mapped Cartesian/cylindrical meshes. No implicit chemistry ownership.
     * Every corrector starts from the same accepted state; only passing physical gates publishes once.
     * Face species and phase energy use the exact same implicit upwind carrier-plus-prescribed-slip flux.
     * Pressure work uses the EOS phase-volume trial; external/chemical transfer packets must exclude it.
     * Physical energy tolerance uses caloric scales independent of an additive mass-energy reference.
     * Operand roundoff is reported separately and must consume at most one quarter of that accuracy budget.
     */
    Diagnostics advance(double time_step, const std::vector<CellSources>& integrated_sources = {});
    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] const SP<const Mesh>& mesh() const noexcept;
    [[nodiscard]] const Options& options() const noexcept;
    [[nodiscard]] const State& state() const noexcept;
    [[nodiscard]] const Diagnostics& diagnostics() const noexcept;
    [[nodiscard]] Checkpoint checkpoint() const;
    void restore(const Checkpoint& checkpoint, const ControllerLease* controller = nullptr);
    [[nodiscard]] const Traits::scalar_face_type& volume_flux() const noexcept;
    [[nodiscard]] const Traits::scalar_cell_type& density() const noexcept;
    [[nodiscard]] const Traits::vector_cell_type& velocity() const noexcept;

private:
    SIMPLEFLUID_SOLVERS_LOCAL void require_controller(const ControllerLease* controller) const;
    SIMPLEFLUID_SOLVERS_LOCAL void release_controller(std::uint64_t id) noexcept;
    struct SIMPLEFLUID_SOLVERS_LOCAL Implementation;
    std::unique_ptr<Implementation> d;
};
} // namespace SimpleFluid::Reacting
