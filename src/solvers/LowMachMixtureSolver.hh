/** @file LowMachMixtureSolver.hh
 * @brief Native conservative multicomponent low-Mach flow with supplied caloric mixture EOS.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "equations/LowMachPressureEquation.hh"
#include "equations/ReactingThermodynamics.hh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
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
        double source_relative_tolerance = 1e-8, thermal_relative_tolerance = 2e-9;
        int maximum_caloric_seed_evaluations = 64;
        std::shared_ptr<const CaloricPressureDomainProvider> caloric_domain_provider;
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
    /** A transport inventory is before local sources, Fourier heat and pressure work.
     * advected_material_volume is the exact work/continuity packet. It is NOT
     * an EOS donor phase volume and must never be a concentration denominator.
     */
    struct TransportPhaseInventory
    {
        std::vector<double> moles;
        double internal_energy = 0, advected_material_volume = 0;
        std::optional<double> population_number; ///< Absent until an explicit dispersed-number role is implemented.
        bool operator==(const TransportPhaseInventory&) const = default;
    };
    struct TransportCellInventory
    {
        std::vector<TransportPhaseInventory> phases;
        bool operator==(const TransportCellInventory&) const = default;
    };
    struct SourceStageMetadata
    {
        std::string provider_identity;
        std::uint64_t provider_revision = 0, invocation = 0;
        size_t corrector = 0; ///< Zero-based transport corrector; distinct from a pressure-root evaluation.
        bool operator==(const SourceStageMetadata&) const = default;
    };
    struct TransportSnapshot
    {
        std::vector<TransportCellInventory> cells; ///< Owned-cell order; owns all arrays.
        std::vector<double> old_control_volumes, new_control_volumes;
        std::uint64_t accepted_revision = 0, accepted_generation = 0;
        std::uint64_t old_geometry_epoch = 0, new_geometry_epoch = 0, schema_fingerprint = 0;
        double accepted_time = 0, time_step = 0;
        SourceStageMetadata stage;
        bool operator==(const TransportSnapshot&) const = default;
    };
    struct SourceStageContext
    {
        const TransportSnapshot& transport;
        const State& thermodynamic_trial; ///< Rates use this T/p together with actual stage composition.
        const Mesh& mesh; ///< Borrowed read-only mesh for collective halo/face stages; epochs are in transport.
    };
    struct SourceStageResult
    {
        std::vector<CellSources> sources;
        /** Nonnegative integrated gross events in the provider's immutable schema/order.
         * Empty outer vector disables this convergence channel. Otherwise there is
         * one fixed-size event vector per owned cell; dimensions cannot change
         * between correctors. Empty owners still participate in all collectives.
         */
        std::vector<std::vector<double>> gross_events;
    };
    using SourceStageCallback = std::function<SourceStageResult(const SourceStageContext&)>;
    /** Immutable metadata and shared callback lifetime, independent of any controller object.
     * This is a CONTROLLER-INTERNAL collective mesh stage, not a scalar user law.
     * User chemistry, EOS, partial-h, equilibrium and heat-law callbacks remain
     * MPI-free. The internal provider evaluates each local stage inside a failure
     * fence BEFORE its next halo exchange/reduction, including empty owners.
     * It is deterministic and never advances accepted/external physical history.
     * A temporary per-preview event receipt is allowed.
     */
    struct SourceStageProvider
    {
        const std::string identity;
        const std::uint64_t revision, schema_fingerprint;
        const SourceStageCallback evaluate;
        const double relative_tolerance = 0; ///< Zero uses Options::source_relative_tolerance.
        const int maximum_evaluations = 0;   ///< Zero uses Options::maximum_correctors; an override must be >=2.
    };
    /** Exact owned-face packet used by the final conservative transport stage.
     * carrier_volume_flux is the STAGE-USED absolute carrier flux, before the last pressure correction.
     * phase_volume_flux contains absolute carrier-plus-phase-slip fluxes, m3/s.
     * material_fraction contains that phase's actual upwind EOS fraction; sums may exceed one in counterflow.
     * Trial::owned_volume_flux() instead contains the FINAL corrected absolute carrier flux.
     */
    struct PhaseFaceTransport
    {
        double carrier_volume_flux = 0;
        std::vector<double> phase_volume_flux, material_fraction;
        bool operator==(const PhaseFaceTransport&) const = default;
    };
    struct Diagnostics
    {
        int correctors = 0, source_evaluations = 0;
        double maximum_source_residual = 0, maximum_transport_residual = 0, maximum_thermal_residual = 0;
        double maximum_event_residual = 0;
        ///< Residuals above are normalized by their declared gates; acceptance requires <=1.
        double maximum_occupancy_error = 0, maximum_continuity_residual = 0;
        double mass_error = 0, internal_energy_error = 0, pressure_work = 0;
        double caloric_energy_scale = 0; ///< Sum m*T*de/dT plus reference-free source/work/transport scales, J.
        double internal_energy_roundoff_bound = 0; ///< Explicit energy-operand cancellation bound, J.
        double empty_phase_energy_roundoff = 0;    ///< Bounded operand cancellation; never physical heat.
        Vector momentum_error{};
        Vector pressure_impulse{},
            external_impulse{}; ///< Global integrated mechanical-pressure/external momentum, kg m/s.
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
        SIMPLEFLUID_SOLVERS_LOCAL ControllerLease(LowMachMixtureSolver* solver, std::uint64_t value)
            : owner(solver), id(value)
        {
        }

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
        TransportSnapshot transport;
        std::vector<CellSources> sources;
        std::vector<std::vector<double>> gross_events;
        std::shared_ptr<const SourceStageProvider> provider;
        std::vector<double> face_flux;
        std::vector<PhaseFaceTransport> phase_transport;
        const LowMachMixtureSolver* owner = nullptr;
        std::uint64_t nonce = 0, base_revision = 0, geometry_epoch = 0;

    public:
        [[nodiscard]] const TransportSnapshot& transport_snapshot() const noexcept { return transport; }
        [[nodiscard]] std::span<const CellSources> integrated_sources() const noexcept { return sources; }
        [[nodiscard]] std::span<const std::vector<double>> integrated_gross_events() const noexcept
        {
            return gross_events;
        }
        [[nodiscard]] const SourceStageMetadata& source_stage() const noexcept { return transport.stage; }
        [[nodiscard]] const State& state() const noexcept { return candidate; }
        [[nodiscard]] const Diagnostics& diagnostics() const noexcept { return statistics; }
        /** Final corrected absolute carrier flux, m3/s, in this Trial's owned-face map/geometry epoch. */
        [[nodiscard]] std::span<const double> owned_volume_flux() const noexcept { return face_flux; }
        [[nodiscard]] std::span<const PhaseFaceTransport> owned_phase_transport() const noexcept
        {
            return phase_transport;
        }
    };
    /** One external controller may stage/commit while retaining shared ownership of this solver. */
    [[nodiscard]] ControllerLease acquire_controller();
    [[nodiscard]] Trial preview(double time_step, const std::vector<CellSources>& integrated_sources = {},
        const ControllerLease* controller = nullptr);
    /** Invoke a pure collective source stage after EACH conservative transport corrector.
     * At least two source evaluations and source/event/transport/thermal fixed-point
     * convergence are required. No callback runs during commit or restore.
     */
    [[nodiscard]] Trial preview_with_sources(double time_step, std::shared_ptr<const SourceStageProvider> provider,
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
     * Mechanical correctors include the actual fixed-packet EOS volume susceptibility. Uniform gauge shifts
     * move into p0 without changing absolute pressure; nonlinear physical continuity/thermal gates still apply.
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
    /** Accepted absolute carrier flux. Continuity uses the distinct phase-weighted material flux
     * sum(alpha_upwind*(carrier+phase_slip)), with exact packets retained by each Trial.
     */
    [[nodiscard]] const Traits::scalar_face_type& volume_flux() const noexcept;
    [[nodiscard]] const Traits::scalar_cell_type& density() const noexcept;
    [[nodiscard]] const Traits::vector_cell_type& velocity() const noexcept;

private:
    SIMPLEFLUID_SOLVERS_LOCAL Trial preview_impl(double time_step, const std::vector<CellSources>& fixed_sources,
        std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller);
    SIMPLEFLUID_SOLVERS_LOCAL void require_controller(const ControllerLease* controller) const;
    SIMPLEFLUID_SOLVERS_LOCAL void release_controller(std::uint64_t id) noexcept;
    struct SIMPLEFLUID_SOLVERS_LOCAL Implementation;
    std::unique_ptr<Implementation> d;
};
} // namespace SimpleFluid::Reacting
