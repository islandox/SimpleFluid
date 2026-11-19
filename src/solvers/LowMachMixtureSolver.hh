/** @file LowMachMixtureSolver.hh
 * @brief Native conservative multicomponent low-Mach flow with supplied caloric mixture EOS.
 */
#pragma once
#include "equations/LowMachPressureEquation.hh"
#include "equations/PressureConditionedThermal.hh"
#include "equations/QbmmPopulationEvents.hh"
#include "equations/QbmmPopulationSchema.hh"
#include "equations/ReactingThermodynamics.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "solvers/QbmmModelOptions.hh"
#include "utils/export_helper.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid::Reacting
{
class ReactingMixtureCoupler;
namespace low_mach_faces
{
class SIMPLEFLUID_SOLVERS_LOCAL LowMachFaceTransportAdapter;
struct ComposedSourceStage;
} // namespace low_mach_faces

class SIMPLEFLUID_SOLVERS_EXPORT LowMachMixtureSolver
{
    struct SIMPLEFLUID_SOLVERS_LOCAL ALEGeometryCertificate;
    struct SIMPLEFLUID_SOLVERS_LOCAL QbmmPublication;
    friend class ReactingMixtureCoupler;

public:
    using Pack = DefaultTpetraTypes;
    using Mesh = MeshHandle<Pack>;
    using PlanarMotion = SimpleFluid::PlanarALEMeshMotion<Pack>;
    using Traits = MeshFieldTraits<Pack, Mesh>;
    using Vector = Traits::vector_cell_type::vec_type;
    enum class PhaseRole
    {
        BulkLiquid,
        BulkGas,
        DispersedGas
    };
    struct PhaseRoleDescriptor
    {
        PhaseRole role = PhaseRole::BulkGas;
        std::string identity; ///< Required unique identity for an explicitly owned dispersed population.
        bool operator==(const PhaseRoleDescriptor&) const = default;
    };
    struct QbmmFamilyStamp
    {
        std::uint64_t catalog_fingerprint = 0, family_fingerprint = 0;
        double mass_scale_kg = 0;
        bool operator==(const QbmmFamilyStamp&) const = default;
    };
    [[nodiscard]] static QbmmFamilyStamp qbmm_family_stamp(const QbmmPopulationSchema& schema) noexcept
    {
        return {schema.catalog().fingerprint(), schema.fingerprint(), schema.mass_scale_kg()};
    }
    using QbmmFamilyLayout = std::vector<std::optional<QbmmFamilyStamp>>;
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
        bool use_multi_rhs_transport = true;
        std::function<double(std::span<const PhaseThermalState>, std::span<const double>)> mixture_dynamic_viscosity;
        std::vector<PhaseRoleDescriptor> phase_roles; ///< Constructor-copied immutable schema, aligned with EOS phases.
        std::vector<std::shared_ptr<const QbmmPopulationSchema>>
            qbmm_families; ///< Empty disables QBMM. Otherwise EOS-indexed, null for unbound phases.
        QbmmModelOptions qbmm_model;
        std::vector<std::shared_ptr<const QbmmNodeSlipLaw>>
            qbmm_node_slip_laws; ///< Kinetic-only, EOS-indexed and nonnull precisely for bound families.
    };
    struct QbmmKineticClosure
    {
        std::size_t active_atoms = 0;
        std::array<Vector, 2> node_velocities{}; ///< Relative m/s in the endpoint quadrature's active order.
        Vector drift_momentum{};                 ///< Extensive sum(count * mass * slip), kg m/s.
        bool operator==(const QbmmKineticClosure&) const = default;
    };
    struct CellState
    {
        std::vector<PhaseThermalState> phases; ///< One liquid (including dissolved), then independently owned gas bins.
        std::vector<Vector>
            drift_velocity; ///< Supplied m/s relative to the carrier, retained unchanged by advancement.
        Vector velocity{};  ///< Carrier velocity, m/s; not generally the barycentric velocity.
        double mechanical_pressure = 0;
        std::vector<std::optional<double>>
            phase_numbers; ///< Engaged extensive finite N only for explicit dispersed roles.
        std::vector<std::optional<QbmmShapeState>> qbmm_shape; ///< Extensive H2,H3 only for enabled families.
        std::vector<std::optional<QbmmKineticClosure>>
            qbmm_kinetic_closure; ///< Solver-owned derived endpoint, in physical phase order for kinetic runs.
        std::vector<double> qbmm_kinetic_eos_volumes; ///< Physical EOS volumes used for the retained endpoint law.
    };
    struct State
    {
        std::vector<CellState> cells; ///< Owned cells only, mesh order.
        double thermodynamic_pressure = 0, time = 0;
        std::uint64_t revision = 0;
        QbmmFamilyLayout qbmm_families; ///< Also required on empty owners.
    };
    struct CellSources
    {
        std::vector<std::vector<double>> species_moles; ///< Integrated signed mol per phase/species; empty means zero.
        std::vector<double> internal_energy;            ///< Integrated J per phase, excludes solver-owned p*dV work.
        Vector momentum{};                              ///< Integrated external physical kg m/s.
        std::vector<double>
            population_number_changes; ///< Integrated signed number per physical phase; empty means zero.
        std::vector<std::optional<QbmmShapeState>>
            qbmm_shape_changes; ///< Signed integrated shape packet; empty means zero.
    };
    /** A transport inventory is before local sources, Fourier heat and pressure work.
     * advected_material_volume is the exact work/continuity packet. It is NOT
     * an EOS donor phase volume and must never be a concentration denominator.
     */
    struct TransportPhaseInventory
    {
        std::vector<double> moles;
        double internal_energy = 0, advected_material_volume = 0;
        std::optional<double> population_number;  ///< Actual pre-source transported N for explicit dispersed roles.
        std::optional<QbmmShapeState> qbmm_shape; ///< Pre-source shape in the snapshot's family layout.
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
        QbmmFamilyLayout qbmm_families;
        bool operator==(const TransportSnapshot&) const = default;
    };
    struct SourceStageContext
    {
        const TransportSnapshot& transport;
        const State& thermodynamic_trial; ///< Rates use this T/p together with actual stage composition.
        const Mesh& mesh; ///< Borrowed read-only mesh for collective halo/face stages; epochs are in transport.
    };
    struct QbmmSourceReceipt
    {
        static constexpr size_t gross_channels = 18;
        std::vector<QbmmEventReceipt> events;
        double final_count = 0;
        std::vector<double> final_moles;
        QbmmShapeState final_shape;
        std::uint64_t family_fingerprint = 0;
        std::array<long double, 2> event_shape_change{}; ///< Q1 signed H2,H3 before canonical endpoint rounding.
    };
    using QbmmSourceCells = std::vector<std::vector<std::optional<QbmmSourceReceipt>>>;
    struct FourierFaceResolution
    {
        Pack::global_ordinal_type geometry_identity;
        double energy;             ///< Exact signed energy received by this owned cell, J.
        double decrease, increase; ///< Distances to the temperature-operand interval endpoints, J.
    };
    class FourierGrossResolution
    {
        friend class low_mach_faces::LowMachFaceTransportAdapter;
        friend struct low_mach_faces::ComposedSourceStage;
        size_t channel_;                           ///< Adjacent exact positive and negative Fourier heat totals.
        std::vector<FourierFaceResolution> faces_; ///< Sorted by canonical geometry identity.
        FourierGrossResolution(size_t channel, std::vector<FourierFaceResolution> faces)
            : channel_(channel), faces_(std::move(faces))
        {
        }

    public:
        [[nodiscard]] size_t positive_channel() const noexcept { return channel_; }
        [[nodiscard]] std::span<const FourierFaceResolution> faces() const noexcept { return faces_; }
    };
    using FourierGrossResolutions = std::vector<std::vector<FourierGrossResolution>>;
    struct SourceStageResult
    {
        std::vector<CellSources> sources;
        /** Nonnegative integrated gross events in the provider's immutable schema/order.
         * Empty outer vector disables this convergence channel. Otherwise there is
         * one fixed-size event vector per owned cell; dimensions cannot change
         * between correctors. Empty owners still participate in all collectives.
         */
        std::vector<std::vector<double>> gross_events;
        std::shared_ptr<const PressureConditionedThermalBatch>
            thermal_receipt;        ///< Frozen events, raw energy; no rates inside pressure roots.
        QbmmSourceCells qbmm_cells; ///< Owned-cell and physical-phase order; engaged for every bound family.
        FourierGrossResolutions fourier_gross_resolution; ///< Sparse owned rows from the canonical Fourier adapter.
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
        /** Uniform m/s2 for balanced pressure reconstruction only. A nonzero value asserts that evaluate()
         * includes the matching integrated physical impulse M*g*dt in each cell's momentum packet, using
         * the current trial mixture mass, alongside any other external impulses. Direct providers must
         * satisfy this contract too: metadata alone supplies no force and must not be used without that impulse.
         * Absent and exactly zero values use the same pressure reconstruction; their metadata remain distinct.
         * Presence and all components must agree across ranks, including empty owners.
         */
        const std::optional<Vector> body_acceleration = std::nullopt;
    };
    /** Actual interval-averaged atom transfers with owner-oriented signs, including for the backward direction. */
    struct QbmmDirectionalFlux
    {
        std::vector<double> species_molar_flux;
        QbmmMomentBlock moment_flux{};
        double mass_flux = 0, internal_energy_flux = 0, material_volume_flux = 0;
        Vector carrier_momentum_flux{}, drift_momentum_flux{}; ///< Frozen donor trace contributions.
        double integrated_count = 0, integrated_mass = 0;      ///< Actual count and kg over the full interval.
        bool operator==(const QbmmDirectionalFlux&) const = default;
    };
    struct QbmmKineticFaceFlux
    {
        QbmmDirectionalFlux forward, backward;
        double theta = 0, slip_volume_flux = 0; ///< Frozen pressure coefficients over the actual substeps.
        double pressure_rounding_flux = 0;      ///< Representation correction in m3/s at the stage carrier flux.
        bool operator==(const QbmmKineticFaceFlux&) const = default;
    };
    struct QbmmFaceFlux
    {
        std::vector<double> species_molar_flux; ///< Owner-oriented mol/s, ordered catalog.
        double population_number_flux = 0;      ///< Owner-oriented number/s.
        QbmmShapeState shape_flux;              ///< Owner-oriented H2/s,H3/s.
        double internal_energy_flux = 0;        ///< Owner-oriented J/s; native phase energy, counted once.
        double material_volume_flux = 0;        ///< Exact owner-oriented m3/s consumed by native work and continuity.
        double derived_mass_flux = 0;           ///< Species rates weighted by molar mass and rounded to double, kg/s.
        Vector derived_drift_momentum_flux{};   ///< Donor drift-weighted mass rate rounded to double, kg m/s2.
        std::optional<QbmmKineticFaceFlux> kinetic;
        bool operator==(const QbmmFaceFlux&) const = default;
    };
    /** Exact owned-face packet used by the final conservative transport stage.
     * carrier_volume_flux is the STAGE-USED absolute carrier flux, before the last pressure correction.
     * phase_volume_flux contains absolute carrier-plus-phase-slip fluxes, m3/s, for ordinary/native phases.
     * Kinetic slots use only the directional receipts in qbmm_flux and have zero phase_volume_flux
     * and relative_phase_volume_flux entries.
     * material_fraction contains that phase's actual upwind EOS fraction; sums may exceed one in counterflow.
     * Trial::owned_volume_flux() instead contains the FINAL corrected absolute carrier flux.
     */
    struct PhaseFaceTransport
    {
        double carrier_volume_flux = 0;
        std::vector<double> phase_volume_flux, material_fraction;
        double mesh_volume_flux = 0;                        ///< Canonical swept rate; zero for fixed geometry.
        std::vector<double> relative_phase_volume_flux;     ///< Actual phase advection rate; no extra alpha.
        std::vector<std::optional<QbmmFaceFlux>> qbmm_flux; ///< EOS-indexed; engaged for bound families only.
        double momentum_mass_flux = 0; ///< Exact owner-oriented kg/s consumed by native momentum convection.
        Vector drift_momentum_flux{};  ///< Exact owner-oriented kg m/s2 consumed by native drift-momentum divergence.
        double forward_mass_flux = 0, backward_mass_flux = 0; ///< Directional magnitudes across all physical phases.
        Vector momentum_owner_velocity{},
            momentum_neighbor_velocity{};        ///< Kinetic implicit carrier solve, before projection.
        Vector implicit_carrier_momentum_flux{}; ///< Owner-oriented carrier momentum flux from that solve.
        double symmetric_mass_flux = 0;          ///< Counterflow coefficient in the carrier convection solve, kg/s.
        double theta = 0, slip_volume_flux = 0;  ///< Time-averaged physical material coefficients.
        double pressure_rounding_flux = 0; ///< Difference between direct material flux and represented theta*phi + S.
        double material_volume_flux = 0;   ///< All-phase direct volume rate at the stage carrier flux, m3/s.
        bool operator==(const PhaseFaceTransport&) const = default;
    };
    struct Diagnostics
    {
        int correctors = 0, source_evaluations = 0;
        double maximum_source_residual = 0, maximum_transport_residual = 0, maximum_thermal_residual = 0;
        double maximum_event_residual = 0;
        double maximum_shape_source_residual = 0; ///< Signed H2,H3 source convergence, normalized physical gate.
        double maximum_conditioned_thermal_residual =
            0; ///< Actual evaluated energy + raw plan convergence, normalized physical gate.
        double conditioned_raw_energy_roundoff_bound =
            0; ///< Additional raw channel uncertainty included in energy gate, J.
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
        std::vector<double> population_number_errors; ///< Per-phase global deltaN-sourceN, zero for bulk roles.
        double maximum_shape_transport_residual = 0;  ///< Per-H2/H3 physical face balance, normalized gate.
        double maximum_population_transport_residual =
            0; ///< Local count balance normalized by its physical relative gate.
        double maximum_initial_transport_residual = 0; ///< Before componentwise repair, normalized local gate.
        double maximum_species_transport_residual = 0, maximum_energy_transport_residual = 0;
        ///< Final componentwise stock/face residuals; each must be <=1 independently of global Krylov convergence.
        int transport_consistency_refinements = 0; ///< Scalar and common-family balance sweeps in the final corrector.
        ///< Every sweep spends one iteration of the corresponding transport budget.
        LinearSolveSummary transport_linear, momentum_linear;
        ///< transport_linear counts Krylov iterations only; final residuals describe the actually returned fields.
        ///< The separate consistency-refinement count completes the work accounting.
        LinearSolveStatistics pressure_linear;     ///< Final corrector, including its actual global RHS norm.
        double pressure_linear_rhs_norm_floor = 0; ///< Pressure-only true-residual normalization, m3/s.
        struct TransportMultiRhsDiagnostics
        {
            std::size_t operator_groups = 0;
            std::size_t transported_columns = 0;
            std::size_t active_columns = 0;
            std::size_t matrix_assemblies = 0;
            std::size_t preconditioner_builds = 0;
            std::size_t batched_krylov_solves = 0;
            std::size_t column_serial_solves = 0;
            std::size_t single_rhs_solves = 0;
            std::size_t operator_apply_calls = 0;
            std::size_t operator_column_applies = 0;
            std::size_t max_batch_width = 0;
            int batch_krylov_iterations = 0;
            int column_serial_iterations = 0;
            std::size_t halo_cell_imports = 0;
            std::size_t halo_face_imports = 0;
            MultiRhsExecutionMode execution_mode = MultiRhsExecutionMode::SingleRhs;
            std::vector<int> column_consistency_sweeps = {};
            double pack_unpack_seconds = 0;
            double matrix_assembly_seconds = 0;
            double preconditioner_setup_seconds = 0;
            double linear_solve_seconds = 0;
            double halo_exchange_seconds = 0;
            double consistency_repair_seconds = 0;
        };
        TransportMultiRhsDiagnostics transport_multi_rhs = {};
        std::optional<double> maximum_fourier_gross_comparison_uncertainty_j;
        ///< Fourier operand comparison bound in joules, retained by portable v5 when known.
        int kinetic_substeps = 0; ///< Actual globally synchronized substeps in the final corrector.
        double maximum_kinetic_cfl = 0;
    };
    enum class QbmmStateKind
    {
        Accepted,
        Trial
    };
    enum class QbmmStateOrigin
    {
        Initialized,
        Corrector,
        Checkpoint,
        PortableRestart,
        CountOnlyMigration
    };
    enum class QbmmReceiptHistory
    {
        NoInterval,
        Available,
        UnavailableAfterPortableRestart,
        UnavailableAfterCountOnlyMigration
    };
    struct QbmmReportProvenance
    {
        QbmmStateKind kind = QbmmStateKind::Accepted;
        QbmmStateOrigin origin = QbmmStateOrigin::Initialized;
        std::uint64_t revision = 0, generation = 0, geometry_epoch = 0, catalog_fingerprint = 0;
        double time = 0, thermodynamic_pressure_pa = 0;
        std::optional<std::uint64_t> preview_id;          ///< Unique selection within this solver, for trials only.
        std::optional<std::uint64_t> archived_generation; ///< Generation of the decoded portable state, if any.
        std::optional<std::uint64_t> controller_configuration_signature; ///< Used or required by the retained state.
    };
    struct QbmmCellFamilyReport
    {
        std::vector<double> species_moles; ///< Catalog order, mol/cell.
        QbmmMomentBlock moments{};         ///< Extensive H0..H3 in family coordinates. H1 is derived from species.
        std::array<long double, 4> mass_moment_density_si{}; ///< Sum(N*m^k)/Vcell, kg^k/m3 (bulk), k=0..3.
        std::vector<double> bulk_concentration_mol_per_m3;
        std::optional<std::vector<double>> phase_concentration_mol_per_m3;
        double phase_fraction = 0, phase_mass_kg = 0, phase_volume_m3 = 0, bulk_mass_density_kg_per_m3 = 0;
        double absolute_pressure_pa = 0, phase_internal_energy_j = 0;
        std::optional<double> phase_density_kg_per_m3, temperature_K; ///< Disengaged for exact absence.
        QbmmQuadrature quadrature;
        std::optional<QbmmGeometryView> geometry; ///< Engaged for supported inversion, including exact absence.
        bool closure_evaluated = false, closure_validated = false; ///< Physical validation requires application data.
        std::vector<QbmmApplicabilityEvaluation> applicability;    ///< Supplied ranges; distinct from realizability.
        std::optional<QbmmKineticClosure> kinetic_closure;
        std::optional<Vector> mean_node_slip_m_per_s; ///< D/M for an occupied kinetic family.
        std::array<std::optional<QbmmApplicabilityEvaluation>, 2> node_slip_applicability;
    };
    struct QbmmCellReport
    {
        Pack::global_ordinal_type geometry_cell_id = 0;
        double bulk_volume_m3 = 0, mixture_density_kg_per_m3 = 0, gas_volume_fraction = 0;
        double mechanical_pressure_pa = 0;
        Vector carrier_velocity{}, barycentric_velocity{};         ///< Cell vectors, m/s.
        Vector mass_weighted_drift{};                              ///< sum(M_phase * drift_phase) / sum(M_phase), m/s.
        std::optional<double> temperature_K;                       ///< Shared temperature of the occupied phases.
        std::vector<std::optional<QbmmCellFamilyReport>> families; ///< EOS order.
    };
    struct QbmmFamilyReport
    {
        static constexpr size_t status_count = static_cast<size_t>(InversionStatus::ReconstructionFailure) + 1;
        std::string population;
        QbmmFamilyStamp stamp;
        std::array<std::uint64_t, status_count> inversion_counts{}; ///< Index by InversionStatus.
        std::optional<long double> minimum_hankel_margin, minimum_shifted_hankel_margin;
        MomentResiduals maximum_reconstruction;    ///< Per-moment maxima over owned cells on all ranks.
        std::vector<QbmmLawMetadata> law_metadata; ///< Immutable contracts in family schema order.
        // Per-law cell totals: within all ranges, outside/missing input, absent; reduced over all ranks.
        std::vector<std::array<std::uint64_t, 3>> applicability_counts;
        std::vector<std::uint64_t> applicability_gate_failures;
    };
    struct QbmmKineticLawReceipt
    {
        Pack::global_ordinal_type geometry_cell_id = 0;
        std::size_t phase = 0, substep = 0, atom = 0; ///< Physical phase, interval substep, and local active atom.
        double duration = 0, mass_kg_per_bubble = 0;
        Vector slip_velocity{};
        QbmmApplicabilityEvaluation applicability;
    };
    struct QbmmIntervalReport
    {
        std::uint64_t preview_id = 0, transported_from_revision = 0, transported_from_generation = 0;
        std::uint64_t old_geometry_epoch = 0, new_geometry_epoch = 0, catalog_fingerprint = 0;
        double transported_from_time = 0, time_step = 0;
        SourceStageMetadata source_stage;
        QbmmSourceCells local_receipts;
        std::vector<std::vector<double>> local_gross_events; ///< Exact final per-cell corrector packets.
        int kinetic_substeps = 0;
        double maximum_kinetic_cfl = 0;
        std::vector<PhaseFaceTransport> local_face_transport; ///< Final corrector's kinetic owned-face evidence.
        std::vector<QbmmKineticLawReceipt> local_kinetic_law_receipts;
    };
    struct QbmmConvergenceReport
    {
        int correctors = 0, source_evaluations = 0;
        double maximum_event_residual = 0; ///< Gross source convergence, normalized by its physical gate.
        std::optional<double> maximum_shape_source_residual, maximum_shape_transport_residual;
        ///< Shape residuals are disengaged when omitted by the source archive.
        std::optional<double> maximum_fourier_gross_comparison_uncertainty_j;
        ///< Fourier operand comparison bound in joules, disengaged when unavailable.
    };
    enum class QbmmMaterialBalanceBasis
    {
        AppliedNativeSources, ///< Native fixed source packets are prescribed external material.
        DeclaredReservoirs,   ///< Reacting chemistry and internal transfers conserve mass and elements.
        ArchivedNativeDefect  ///< Legacy archive provides native mass/energy and prescribed species defects.
    };
    struct QbmmBalances
    {
        QbmmMaterialBalanceBasis material_basis = QbmmMaterialBalanceBasis::AppliedNativeSources;
        double mass_defect_kg = 0, internal_energy_defect_j = 0, pressure_work_j = 0, energy_roundoff_bound_j = 0;
        std::vector<double> species_defects_mol;           ///< Native change minus the applied source, catalog order.
        std::vector<double> external_species_moles;        ///< Catalog order, signed. Empty for legacy archives.
        std::map<std::string, double> element_defects_mol; ///< Moles of atoms, change minus external input.
    };
    struct QbmmReport
    {
        QbmmReportProvenance provenance;
        std::vector<SpeciesId> species_ids;                    ///< Native catalog order for moles and concentrations.
        std::vector<QbmmCellReport> cells;                     ///< Locally owned cells in mesh order.
        std::vector<std::optional<QbmmFamilyReport>> families; ///< Collective diagnostics in EOS order.
        QbmmReceiptHistory receipt_history = QbmmReceiptHistory::NoInterval;
        std::shared_ptr<const QbmmIntervalReport> interval;
        std::optional<QbmmConvergenceReport> convergence;
        std::optional<QbmmBalances> balances;
        QbmmModelOptions::FamilyTransport family_transport = QbmmModelOptions::FamilyTransport::NativeImplicit;
        int kinetic_substeps = 0; ///< Last interval, including when portable history omits the face receipts.
        double maximum_kinetic_cfl = 0;
    };
    class Checkpoint
    {
        friend class LowMachMixtureSolver;
        friend class ReactingMixtureCoupler;
        State state;
        std::uint64_t generation = 0;
        const LowMachMixtureSolver* owner = nullptr;
        std::uint64_t geometry_epoch = 0, logical_geometry = 0;
        std::shared_ptr<const ALEGeometryCertificate> ale_geometry;
        std::vector<double> face_flux;
        Diagnostics diagnostics;
        std::shared_ptr<const QbmmPublication> qbmm_publication;
        std::optional<std::uint64_t> qbmm_controller_signature;
        std::optional<std::array<double, 3>> count_only_body_acceleration;
        bool qbmm_legacy_default_only = false;
    };
    class ControllerLease
    {
        friend class LowMachMixtureSolver;
        LowMachMixtureSolver* owner = nullptr;
        std::uint64_t id = 0;
        bool reacting_source_authority = false;
        SIMPLEFLUID_SOLVERS_LOCAL ControllerLease(
            LowMachMixtureSolver* solver, std::uint64_t value, bool reacting_source_authority)
            : owner(solver), id(value), reacting_source_authority(reacting_source_authority)
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
        std::uint64_t preview_id = 0;
        std::vector<std::vector<std::optional<InversionStatus>>> qbmm_validation;
        Diagnostics statistics;
        TransportSnapshot transport;
        std::vector<CellSources> sources;
        std::vector<std::vector<double>> gross_events;
        QbmmSourceCells qbmm_cells;
        std::optional<QbmmBalances> qbmm_balances;
        std::shared_ptr<const SourceStageProvider> provider;
        std::shared_ptr<const PressureConditionedThermalBatch> thermal_receipt;
        std::vector<PressureConditionedThermalEvaluation> thermal_evaluations;
        std::vector<double> face_flux;
        std::vector<PhaseFaceTransport> phase_transport;
        std::vector<QbmmKineticLawReceipt> kinetic_law_receipts;
        const LowMachMixtureSolver* owner = nullptr;
        std::uint64_t nonce = 0, base_revision = 0, geometry_epoch = 0, logical_geometry = 0;
        std::shared_ptr<const ALEGeometryCertificate> ale_geometry;

    public:
        [[nodiscard]] const TransportSnapshot& transport_snapshot() const noexcept { return transport; }
        /** Exact provider-returned static packets. A conditioned trial must be replayed with its thermal receipt;
         * final applied phase energies are retained in final_thermal_evaluations().
         */
        [[nodiscard]] std::span<const CellSources> integrated_sources() const noexcept { return sources; }
        [[nodiscard]] std::span<const std::vector<double>> integrated_gross_events() const noexcept
        {
            return gross_events;
        }
        [[nodiscard]] const QbmmSourceCells& qbmm_source_cells() const noexcept { return qbmm_cells; }
        [[nodiscard]] const std::shared_ptr<const PressureConditionedThermalBatch>&
        conditioned_thermal_receipt() const noexcept
        {
            return thermal_receipt;
        }
        [[nodiscard]] std::span<const PressureConditionedThermalEvaluation> final_thermal_evaluations() const noexcept
        {
            return thermal_evaluations;
        }
        [[nodiscard]] const SourceStageMetadata& source_stage() const noexcept { return transport.stage; }
        [[nodiscard]] const auto& qbmm_inversion_status() const noexcept { return qbmm_validation; }
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
    /** Prescribed closed planar ALE. Preview owns begin/rollback and returns
     * an immutable candidate on restored accepted geometry; explicit commit requires live motion authority.
     */
    [[nodiscard]] Trial preview_ale(double time_step, PlanarMotion& motion, double target_surface,
        const std::vector<CellSources>& integrated_sources = {}, const ControllerLease* controller = nullptr);
    [[nodiscard]] Trial preview_ale_with_sources(double time_step, PlanarMotion& motion, double target_surface,
        std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller = nullptr);
    /** Explicit live motion authority; fixed commit rejects moving candidates. */
    void commit_ale(const Trial& trial, PlanarMotion& motion, const ControllerLease* controller = nullptr);
    LowMachMixtureSolver(SP<const Mesh> mesh,
        std::vector<std::shared_ptr<const MixtureThermodynamics>> phase_thermodynamics, Options options);
    ~LowMachMixtureSolver();
    LowMachMixtureSolver(const LowMachMixtureSolver&) = delete;
    LowMachMixtureSolver& operator=(const LowMachMixtureSolver&) = delete;
    struct QbmmMonodisperse
    {
    };
    using QbmmInitializationChoice = std::variant<QbmmMonodisperse, std::vector<QbmmAtom>>;
    struct QbmmInitialization
    {
        std::vector<std::vector<std::optional<QbmmInitializationChoice>>> cells;
    };
    /** Explicitly initialize missing shape. Supply one choice per bound family per owned cell.
     * External atoms use extensive counts and mass divided by that family's reference mass.
     */
    void initialize_qbmm(
        State legacy, const QbmmInitialization& initialization, const ControllerLease* controller = nullptr);
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
    /** Collective, read-only QBMM output on accepted geometry or a retained trial's geometry. A trial must be
     * selected uniformly and still belong to this solver's accepted generation. Local EOS evaluation or
     * unrepresentable physical diagnostics can cause a collective report failure.
     */
    [[nodiscard]] QbmmReport qbmm_report() const;
    [[nodiscard]] QbmmReport qbmm_report(const Trial& trial) const;
    /** Collective, partition-local checkpoint encoding for the fixed-mesh QBMM path. */
    [[nodiscard]] std::vector<std::byte> serialize_qbmm_restart() const;
    void restore_qbmm_restart(std::span<const std::byte> payload, const ControllerLease* controller = nullptr);
    /** Collective portable restart on prescribed planar geometry. The default stamp describes direct numeric targets.
     * Coupled callers supply the immutable identity and revision of their target law.
     */
    [[nodiscard]] std::vector<std::byte> serialize_qbmm_restart(const PlanarMotion& motion,
        std::string_view target_identity = "SimpleFluid.explicit-planar-target",
        std::uint64_t target_revision = 1) const;
    /** Restore with matching original reference geometry, motion options, target stamp, maps and configuration.
     * The coupled path requires its live controller lease. Restore does not call a target or source law.
     */
    void restore_qbmm_restart(std::span<const std::byte> payload, PlanarMotion& motion,
        std::string_view target_identity = "SimpleFluid.explicit-planar-target", std::uint64_t target_revision = 1,
        const ControllerLease* controller = nullptr);
    /** Migrate an owner-bound count-only checkpoint on the same live fixed mesh.
     * Preserve the checkpoint's accepted flux and diagnostics while initializing its missing shape.
     * A checkpoint with supplied acceleration requires a receiving controller with the same acceleration.
     */
    void restore_qbmm_checkpoint(const LowMachMixtureSolver& previous, const Checkpoint& saved,
        const QbmmInitialization& initialization, const ControllerLease* controller = nullptr);
    [[nodiscard]] Checkpoint checkpoint() const;
    void restore(const Checkpoint& checkpoint, const ControllerLease* controller = nullptr);
    [[nodiscard]] Checkpoint checkpoint(const PlanarMotion& motion) const;
    void restore(const Checkpoint& checkpoint, PlanarMotion& motion, const ControllerLease* controller = nullptr);
    /** Accepted absolute carrier flux. Continuity uses the distinct phase-weighted material flux
     * sum(alpha_upwind*(carrier+phase_slip)), with exact packets retained by each Trial.
     */
    [[nodiscard]] const Traits::scalar_face_type& volume_flux() const noexcept;
    [[nodiscard]] const Traits::scalar_cell_type& density() const noexcept;
    [[nodiscard]] const Traits::vector_cell_type& velocity() const noexcept;

private:
    enum class AdvancementOperation
    {
        Preview,
        PreviewWithSources,
        PreviewWithQbmmSources,
        PreviewPlanar,
        PreviewPlanarWithSources,
        PreviewPlanarWithQbmmSources,
        Commit,
        CommitPlanar,
        ControllerPreview,
        ControllerCommit
    };
    SIMPLEFLUID_SOLVERS_LOCAL static void require_uniform_advancement(const Mesh& mesh, AdvancementOperation operation);
    struct SIMPLEFLUID_SOLVERS_LOCAL ControllerRestartDiagnostics;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL Checkpoint checkpoint_state() const;
    [[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::vector<std::byte> serialize_qbmm_restart_impl(
        const PlanarMotion* motion, std::string_view target_identity, std::uint64_t target_revision,
        const ControllerRestartDiagnostics* controller_diagnostics = nullptr) const;
    SIMPLEFLUID_SOLVERS_LOCAL void restore_qbmm_restart_impl(std::span<const std::byte> payload, PlanarMotion* motion,
        std::string_view target_identity, std::uint64_t target_revision, const ControllerLease* controller,
        const ControllerRestartDiagnostics* controller_diagnostics = nullptr);
    SIMPLEFLUID_SOLVERS_LOCAL void supply_qbmm_shape(State& legacy, const QbmmInitialization& initialization) const;
    SIMPLEFLUID_SOLVERS_LOCAL Trial preview_with_qbmm_sources(
        double time_step, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller);
    SIMPLEFLUID_SOLVERS_LOCAL Trial preview_ale_with_qbmm_sources(double time_step, PlanarMotion& motion,
        double target_surface, std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller);
    SIMPLEFLUID_SOLVERS_LOCAL Trial preview_impl(double time_step, const std::vector<CellSources>& fixed_sources,
        std::shared_ptr<const SourceStageProvider> provider, const ControllerLease* controller,
        const FVM::ALEControlVolumeState* ale = nullptr, bool coherent_qbmm = false);
    SIMPLEFLUID_SOLVERS_LOCAL Trial preview_planar_impl(double time_step, PlanarMotion& motion, double target_surface,
        const std::vector<CellSources>& fixed_sources, std::shared_ptr<const SourceStageProvider> provider,
        const ControllerLease* controller, bool coherent_qbmm = false);
    SIMPLEFLUID_SOLVERS_LOCAL std::shared_ptr<ALEGeometryCertificate> capture_accepted_geometry(
        const PlanarMotion& motion) const;
    SIMPLEFLUID_SOLVERS_LOCAL void restore_accepted_after_motion(
        PlanarMotion& motion, std::span<const double> absolute_flux);
    SIMPLEFLUID_SOLVERS_LOCAL void rollback_failed_planar_restore(PlanarMotion& motion);
    SIMPLEFLUID_SOLVERS_LOCAL ControllerLease acquire_controller_for(std::string_view signature,
        bool legacy_compatible = true, std::optional<Vector> body_acceleration = {},
        const PlanarMotion* motion = nullptr, const ControllerRestartDiagnostics* restart = nullptr);
    SIMPLEFLUID_SOLVERS_LOCAL void require_controller(const ControllerLease* controller) const;
    SIMPLEFLUID_SOLVERS_LOCAL void release_controller(std::uint64_t id) noexcept;
    SIMPLEFLUID_SOLVERS_LOCAL QbmmReport qbmm_report_impl(const Trial* trial) const;
    SIMPLEFLUID_SOLVERS_LOCAL void finalize_qbmm_balances(
        Trial& trial, std::optional<std::span<const SpeciesSourceLedger>> chemistry = std::nullopt) const;
    SIMPLEFLUID_SOLVERS_LOCAL static std::shared_ptr<const QbmmIntervalReport> qbmm_interval(const Trial& trial);
    SIMPLEFLUID_SOLVERS_LOCAL std::shared_ptr<const QbmmPublication> prepare_qbmm_publication(const Trial& trial) const;
    struct SIMPLEFLUID_SOLVERS_LOCAL Implementation;
    std::unique_ptr<Implementation> d;
};
} // namespace SimpleFluid::Reacting
