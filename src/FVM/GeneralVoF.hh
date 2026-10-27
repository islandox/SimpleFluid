/** @file GeneralVoF.hh
 * @brief Source-pinned unsplit phase transport on authority-certified native cells.
 */
#pragma once
#include "FVM/GeneralVoFSurface.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/ExperimentalPhaseInventory.hh"
#include "FVM/PhaseVolumes.hh"
#include "geometry/GeometryStatisticsSnapshot.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/SweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "utils/ArithmeticMode.hh"
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
namespace SimpleFluid::VoF
{
namespace Exact
{
struct GeneralVoF;
}
namespace general_vof_detail
{
struct LocalTransportControls;
}
enum class GeneralVoFPhaseGeometry
{
    Infer,
    PureGas,
    PureCarrier
};
/** One coherent Eulerian backend per step. SweptPartition is the primary
 * default; the algebraic face backends remain explicit alternatives. */
enum class GeneralVoFTransportBackend { SweptPartition, DonorCell, GeometricFace };
struct GeneralVoFDirectedStream
{
    std::uint64_t face = 0, donor = 0, recipient = 0;
    int donor_owner = -1, recipient_owner = -1, direction = 1;
    long double volume = 0, gas = 0, liquid = 0, mass = 0;
    long double gas_low = 0, gas_high = 0, liquid_low = 0, liquid_high = 0, theta = 1;
    bool lower_order = false;
};
struct GeneralVoFTransportDiagnostics
{
    std::size_t streams = 0, limited_streams = 0, fallback_streams = 0, plane_measurements = 0;
    long double maximum_outgoing_courant = 0, maximum_continuity_defect = 0, maximum_occupancy_defect = 0;
    long double limited_volume = 0, fallback_volume = 0;
    std::size_t indexed_cells = 0, index_queries = 0, index_node_visits = 0, donor_candidates = 0;
    std::size_t geometric_faces = 0, algebraic_faces = 0, swept_pieces = 0, plane_requests = 0;
    /** Ordinary interface-local scalar geometry; swept_pieces counts immutable
     * source tetrahedra, not individually measured closed solids. */
    std::size_t aggregate_groups = 0, aggregate_witnesses = 0, cancelled_boundaries = 0, aggregate_retries = 0;
    std::size_t limited_faces = 0, gross_counterstreams = 0, quality_limited_faces = 0;
    long double minimum_quality_theta = 1;
    long double minimum_theta = 1;
};
struct GeneralVoFPreparedTransport;
struct GeneralVoFFaceInterface
{
    std::uint64_t face = 0, first_cell = 0, second_cell = 0;
    int first_owner = -1, second_owner = -1;
    PLIC::GeometryPoint centroid{}, normal{};
    double area = 0;
};
struct GeneralVoFOptions
{
    /** Rank-local application allocation bound, checked at each construction /
     * communication phase. This is not an RSS or source-plus-candidate peak.
     * Native ghost transport also obeys the hierarchy's own rank-local budget.
     * All other options below describe shared algorithm semantics and must
     * agree collectively, including on empty owners. */
    std::size_t byte_budget = 64 * 1024 * 1024, maximum_stencil_cells = 128, maximum_halo_cells = 65536;
    unsigned support_rings = 2;
    GeneralVoFTransportBackend transport_backend = GeneralVoFTransportBackend::SweptPartition;
    PLIC::PolyhedralReconstructionOptions reconstruction;
    SimpleFluid::PLIC::SweepLimits sweep;
    ArithmeticMode arithmetic_mode = ArithmeticMode::Ordinary;
};
enum class GeneralVoFFiniteRemapCarrier
{
    BulkVolume,
    GasVolume,
    CarrierVolume,
    MixtureMass
};
struct GeneralVoFFiniteRemapChannel
{
    std::string name;
    GeneralVoFFiniteRemapCarrier carrier = GeneralVoFFiniteRemapCarrier::BulkVolume;
    bool nonnegative = true;
};
struct GeneralVoFFiniteOwnedChannels
{
    Meshes::AMR::AdaptiveCellKey key;
    std::vector<PLIC::DynamicGeometryValue> amounts;
};
struct GeneralVoFFiniteRemapRoute
{
    Meshes::AMR::AdaptiveCellKey source_key, target_key;
    std::uint64_t source_gid = 0, target_gid = 0;
    int source_owner = -1, target_owner = -1;
    PLIC::DynamicPhaseMeasures amount;
    /** Signed redistribution derived from source fit deviations and raw
     * phase/bulk shares; historical allowance remains a separate bound. */
    PLIC::DynamicGeometryValue fit_allocation, inherited_fit_allocation;
};
struct GeneralVoFFiniteCommonCell
{
    Meshes::AMR::AdaptiveCellKey source_key, target_key;
    std::uint64_t source_gid = 0, target_gid = 0;
    int target_owner = -1;
    PLIC::DynamicConvexGeometryPtr geometry;
};
struct GeneralVoFFiniteRemapOptions
{
    std::array<double, 2> phase_density{1, 1};
    std::vector<GeneralVoFFiniteRemapChannel> channels;
    std::size_t byte_budget = 64 * 1024 * 1024;
    double maximum_relative_target_defect = 1e-12, maximum_relative_target_fit = 1e-12;
};
struct GeneralVoFNodeVelocity
{
    std::uint64_t node = 0;
    SimpleFluid::PLIC::GeometryPoint velocity{};
};
struct GeneralVoFNodeTrajectory
{
    std::uint64_t node = 0;
    SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact> trajectory;
};
struct GeneralVoFFaceVolume
{
    std::uint64_t face = 0;
    double integrated_volume = 0; ///< Canonical first-cell outward signed m^3 when exact control is absent.
    /** Optional authoritative rational Phi for the explicit exact route only.
     * Finite rejects this payload; its bounded control is separate. In the
     * exact route the double above is diagnostic only.
     * Projection/face-arithmetic repair remains the caller's separately gated
     * physical responsibility, never a hidden source correction here. */
    std::optional<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> exact_integrated_volume;
    /** Finite numerical authority, separate from the explicit exact control. */
    std::optional<PLIC::DynamicGeometryValue> bounded_integrated_volume;
};
/** Immutable owned controls, collectively mode-checked and pinned to their
 * accepted source. Empty node arrays support face-only transport; empty owners
 * remain valid typed states. Full catalog agreement is checked when planning. */
class GeneralVoFTransportControls
{
public:
    [[nodiscard]] std::span<const GeneralVoFNodeVelocity> nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::span<const GeneralVoFFaceVolume> faces() const noexcept { return faces_; }
    [[nodiscard]] double timestep() const noexcept { return timestep_; }

private:
    friend class GeneralVoF;
    std::shared_ptr<const void> source_;
    std::uint64_t source_control_ = 0;
    ArithmeticMode mode_ = ArithmeticMode::Ordinary;
    double timestep_ = 0;
    std::vector<GeneralVoFNodeVelocity> nodes_;
    std::vector<GeneralVoFFaceVolume> faces_;
};
template<ArithmeticMode Mode> class GeneralVoFControlView
{
public:
    static constexpr ArithmeticMode arithmetic_mode = Mode;
    [[nodiscard]] std::span<const GeneralVoFNodeVelocity> nodes() const noexcept { return controls_->nodes(); }
    [[nodiscard]] std::span<const GeneralVoFFaceVolume> faces() const noexcept { return controls_->faces(); }
    [[nodiscard]] double timestep() const noexcept { return controls_->timestep(); }

private:
    friend class GeneralVoF;
    explicit GeneralVoFControlView(std::shared_ptr<const GeneralVoFTransportControls> controls)
        : controls_(std::move(controls))
    {
    }
    std::shared_ptr<const GeneralVoFTransportControls> controls_;
};
/** Diagnostic intervals only. There is no implicit scalar conversion; a known
 * nonzero interval may contain0 below output range. Never use these to build
 * physical fluxes, budgets or matrix coefficients: use exact measures. */
struct GeneralVoFPhaseEnclosure
{
    SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> bulk, gas, carrier;
};
/** One signed simplex/donor contribution; opposite directions never net away.
 * These are geometric donor contributions, not necessarily incident upwind cells. */
struct GeneralVoFFaceContribution
{
    std::uint64_t face = 0, donor = 0;
    int donor_owner = -1, direction = 1;
    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> exact;
    GeneralVoFPhaseEnclosure diagnostic;
};
/** Transaction-local geometric donor-to-recipient measure, including self atoms.
 * It is seeded from finite accepted phases and may support one-step density
 * weighting/binding. It is never a persistent accepted inventory. */
struct GeneralVoFTransfer
{
    std::uint64_t donor = 0, recipient = 0;
    int donor_owner = -1, recipient_owner = -1;
    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> exact;
    GeneralVoFPhaseEnclosure diagnostic;
};
/** One-step certificate lifted from the current finite accepted state.
 * Unrounded remainders are not carried into the next accepted state. */
struct GeneralVoFDonorBudget
{
    std::uint64_t donor = 0;
    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> exact_old, exact_exported, exact_retained;
    GeneralVoFPhaseEnclosure old_diagnostic, exported_diagnostic, retained_diagnostic;
    /** Exact cumulative absolute gas relabeling; carrier correction is its
     * opposite. Signed-face multiplicities are counted before cancellation. */
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> signed_repair_absolute, signed_repair_bound;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> route_repair_absolute, route_repair_bound;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> full_cell_fit_discrepancy;
};
/** Fixed-storage numerical records used exclusively by the Finite route.
 * A canonical owned face contributes one signed donor group; receivers consume
 * the same record with opposite incidence. No rational value is manufactured.
 */
struct GeneralVoFFiniteFaceContribution
{
    std::uint64_t face = 0, donor = 0;
    int donor_owner = -1, direction = 1;
    PLIC::DynamicPhaseMeasures amount;
};
struct GeneralVoFFiniteTransfer
{
    std::uint64_t donor = 0, recipient = 0;
    int donor_owner = -1, recipient_owner = -1;
    PLIC::DynamicPhaseMeasures amount;
    PhaseRoundoff inherited_error;
};
struct GeneralVoFFiniteDonorBudget
{
    std::uint64_t donor = 0;
    PLIC::DynamicPhaseMeasures old, exported, retained;
};
/** Source-resolved signed update coefficient, combined before taking absolute
 * history weights. Negative coefficients are not negative accepted inventories. */
struct GeneralVoFSignedReceipt
{
    std::uint64_t donor = 0, recipient = 0;
    int donor_owner = -1, recipient_owner = -1;
    PLIC::DynamicPhaseMeasures amount;
    PhaseRoundoff inherited_error;
};
/** old = diagonal + positive_remote - negative_remote, independently for bulk
 * and each phase. Positive remote coefficients are bounded by old inventory. */
struct GeneralVoFSignedDonorBudget
{
    std::uint64_t donor = 0;
    PLIC::DynamicPhaseMeasures old, positive_remote, negative_remote, diagonal;
};
struct GeneralVoFFiniteNodeTrajectory
{
    std::uint64_t node = 0;
    PLIC::DynamicPoint point, departure;
};
/** Actual nominal face-chain quantities. The allowance is allocated from the
 * frozen incident cells, never inferred from the observed construction defect. */
struct GeneralVoFFaceChainDiagnostic
{
    std::uint64_t face = 0;
    PLIC::DynamicGeometryValue control, constructed, signed_difference;
    long double allowed_difference = 0;
};
/** Owned-cell closure using the original cell allowance, including the signed
 * geometry/control differences of every incident canonical subface. */
struct GeneralVoFCellClosureDiagnostic
{
    Meshes::AMR::AdaptiveCellKey key;
    std::uint64_t cell = 0;
    std::size_t canonical_face_count = 0;
    long double quota_scale = 0;
    PLIC::DynamicGeometryValue canonical_residual, signed_geometry_difference, combined_residual;
    long double absolute_geometry_difference = 0, allowed_closure = 0;
    /** Signed phase-label cut closure, separate from the physical face chain.
     * combined_residual = canonical_residual + signed_geometry_difference + this value. */
    long double signed_phase_cut_closure = 0;
};
/** The transport family is independent of arithmetic and candidate measurement.
 * GeometricFace measures a geometric candidate, then uses algebraic face volume. */
enum class GeneralVoFTransportFamily
{
    AlgebraicFaceVolume,
    GeometricSweptTetrahedra,
    InterfaceLocalSwept
};
/** Immutable, source-pinned phase ledger. Signed physical face contributions
 * and positive donor/recipient support remain separate; swept donors need not
 * be adjacent to their face. streams is populated only for algebraic transport.
 * Owned capacities and histories follow the source owned_order. */
struct GeneralVoFPreparedTransport
{
    std::shared_ptr<const void> source_identity, candidate_identity, mesh_identity, topology_identity;
    std::uint64_t source_control = 0, transaction = 0, mesh_generation = 0, mesh_topology_generation = 0, time_stage = 0;
    double timestep = 0;
    std::array<double, 2> density{1, 1};
    ArithmeticMode arithmetic_mode = ArithmeticMode::Ordinary;
    GeneralVoFTransportBackend backend = GeneralVoFTransportBackend::DonorCell;
    GeneralVoFTransportFamily family = GeneralVoFTransportFamily::AlgebraicFaceVolume;
    std::vector<GeneralVoFDirectedStream> streams;
    /** Physical signed coefficients: direction multiplies amount, never clip
     * or reinterpret these records as positive adjacent-cell streams. */
    std::vector<GeneralVoFFiniteFaceContribution> signed_face_contributions;
    /** Positive physical transfers for legacy/homogeneous paths. Empty for
     * InterfaceLocalSwept; its signed receipts and numerical support are below. */
    std::vector<GeneralVoFFiniteTransfer> positive_transfers;
    std::vector<GeneralVoFFiniteDonorBudget> donor_budgets;
    std::vector<GeneralVoFSignedReceipt> signed_receipts;
    std::vector<GeneralVoFSignedDonorBudget> signed_donor_budgets;
    /** Donor-cell numerical support only; never physical signed receipts. */
    std::vector<GeneralVoFFiniteTransfer> numerical_support_transfers;
    std::vector<GeneralVoFFiniteDonorBudget> numerical_support_budgets;
    std::vector<GeneralVoFFiniteNodeTrajectory> node_trajectories;
    std::vector<PhaseRoundoff> phase_roundoff, old_phase_history, new_phase_history;
    std::vector<long double> old_mass, new_mass, old_mass_error, new_mass_error;
    GeneralVoFTransportDiagnostics diagnostics;
    std::vector<GeneralVoFFaceChainDiagnostic> face_chain_diagnostics;
    std::vector<GeneralVoFCellClosureDiagnostic> cell_closure_diagnostics;
    /** Solver-attached immutable coefficient preparation, shared by component
     * solves. Traces, histories and convergence results are never cached here. */
    std::shared_ptr<const void> momentum_preparation;
    std::size_t momentum_builds = 0, momentum_reuses = 0;
    /** Checks the complete retained ledger allocation before copying it. */
    SIMPLEFLUID_FVM_EXPORT void validate_byte_budget(std::size_t) const;

private:
    friend class GeneralVoF;
    // The actual family is factory-authenticated, independently of public
    // backend configuration and of whether this rank retains any streams.
    std::shared_ptr<const void> authenticated_source_, authenticated_candidate_;
    std::uint64_t authenticated_transaction_ = 0, authenticated_source_control_ = 0;
    std::uint64_t authenticated_time_stage_ = 0, authenticated_mesh_generation_ = 0,
                  authenticated_mesh_topology_generation_ = 0;
    double authenticated_timestep_ = 0;
    std::array<double, 2> authenticated_density_{1, 1};
    GeneralVoFTransportFamily authenticated_family_ = GeneralVoFTransportFamily::GeometricSweptTetrahedra;
};
/** Finite immutable prepared ledger, retaining both source and candidate after
 * the originating Trial is destroyed. Exact keeps its separate trial authority. */
template<ArithmeticMode Mode> class GeneralVoFPreparedTrialView
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);

public:
    static constexpr ArithmeticMode arithmetic_mode = Mode;
    [[nodiscard]] const GeneralVoFPreparedTransport& ledger() const noexcept { return *prepared_; }
    [[nodiscard]] const std::shared_ptr<const GeneralVoFPreparedTransport>& shared_ledger() const noexcept
    {
        return prepared_;
    }

private:
    friend class GeneralVoF;
    explicit GeneralVoFPreparedTrialView(std::shared_ptr<const GeneralVoFPreparedTransport> prepared)
        : prepared_(std::move(prepared))
    {
    }
    std::shared_ptr<const GeneralVoFPreparedTransport> prepared_;
};
struct GeneralVoFSupportCell
{
    std::uint64_t cell = 0, revision = 0;
    int owner = -1;
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry;
    SimpleFluid::PLIC::GeometryPoint centroid{};
    Meshes::AMR::AdaptiveCellKey key;
    PLIC::DynamicConvexGeometryPtr finite_geometry;
};
/** Owned refinement requests, collected without preparing an expensive fit. */
struct GeneralVoFResolutionRequest
{
    std::uint64_t cell = 0;
    Meshes::AMR::AdaptiveCellKey key;
    PLIC::PolyhedralResolutionAssessment assessment;
};
struct GeneralVoFReconstructionPatch
{
    std::uint64_t cell = 0;
    int owner = -1;
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry;
    PLIC::PolyhedralInterface interface;
    PLIC::DynamicConvexGeometryPtr finite_geometry;
};
class SIMPLEFLUID_FVM_EXPORT GeneralVoF
{
public:
    /** This library image's calling-thread counters. Call every observed
     * image's bridge and deduplicate snapshots by their separate domain IDs. */
    static void enable_geometry_statistics(bool enabled);
    static void reset_geometry_statistics();
    [[nodiscard]] static Meshes::GeometryStatisticsSnapshot geometry_statistics();
    struct Data;
    using Hierarchy = Meshes::AMR::DistributedAdaptiveHierarchy;
    using Key = Meshes::AMR::AdaptiveCellKey;
    /** Input phases follow input_keys; each local owner must name every owned
     * leaf exactly once. Full ghosts are owner-authenticated sparse imports.
     * With omitted declarations, a fresh exactly absent phase proves pure
     * geometry. Transport retains that hint only while no positive minority
     * geometry arrives; independent trace inventory carried by same-phase pure
     * geometry does not itself assert a new interface. Nonzero fresh-input
     * minorities remain Infer regardless of magnitude; explicit declarations
     * (including Infer) and adopted capsule geometry remain authoritative. */
    GeneralVoF(std::shared_ptr<const Hierarchy>, std::vector<Key> input_keys, std::vector<double> gas,
        std::vector<double> carrier, GeneralVoFOptions = {}, std::vector<GeneralVoFPhaseGeometry> declarations = {});
    [[nodiscard]] const std::shared_ptr<const MeshHandle<>>& mesh() const noexcept;
    [[nodiscard]] const std::shared_ptr<const Hierarchy>& hierarchy() const noexcept;
    [[nodiscard]] const std::vector<Key>& owned_order() const noexcept;
    [[nodiscard]] const std::vector<double>& gas_volumes() const noexcept;
    [[nodiscard]] const std::vector<double>& carrier_volumes() const noexcept;
    /** Finite-mode authority; in ExperimentalExact these are rounded diagnostic
     * views only. The policy is explicit and never inferred from phase values. */
    [[nodiscard]] const std::vector<PhaseVolumes>& phase_volumes() const noexcept;
    [[nodiscard]] PhaseInventoryPolicy inventory_policy() const noexcept;
    [[nodiscard]] ArithmeticMode arithmetic_mode() const noexcept;
    /** Accepted geometry declarations in owned_order, without fitting or export. */
    [[nodiscard]] std::vector<GeneralVoFPhaseGeometry> phase_geometry_declarations() const;
    /** Exact-mode authority only. Finite modes use phase_volumes(); requesting
     * this view never performs an implicit rational lift of finite storage. */
    [[nodiscard]] std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>>
    inventory_measures() const;
    /** Explicit experimental authority; rejects Finite mode. Unqualified for
     * finite-duration physical flow, performance and storage scalability. */
    [[nodiscard]] const std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>>&
    experimental_phases() const;
    /** Latest publication arithmetic uncertainty, never a physical source. */
    [[nodiscard]] const std::vector<PhaseRoundoff>& phase_roundoff_bounds() const noexcept;
    /** Independent accumulated finite arithmetic uncertainty across accepted
     * steps, distinct from this publication's conversion error. */
    [[nodiscard]] const std::vector<PhaseRoundoff>& accumulated_phase_errors() const noexcept;
    /** Exact-mode complementary geometric fitting labels in owned_order.
     * Finite callers use geometric_gas_fractions() for a diagnostic view. */
    [[nodiscard]] std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>>
    geometric_phases() const;
    /** Geometry-only gas/V enclosure view after an exact [0,1] check. Rounding
     * to0/1 is permitted while an extensive minority remains positive. This is
     * never phase-presence, transport or mass authority; use phase_volumes(). */
    [[nodiscard]] std::vector<double> geometric_gas_fractions() const;
    [[nodiscard]] const GeneralVoFOptions& options() const noexcept;
    /** Finite current provenance seeded by input admission and enlarged only
     * by explicitly derived publication/remap arithmetic uncertainty. */
    [[nodiscard]] const std::vector<PhaseError>& initial_occupancy_errors() const noexcept;
    /** Collective read-only admission preflight. The returned requests are
     * locally owned; an external controller may refine/reinitialize and recheck.
     * plan_transport rejects atomically when requests remain unavailable. */
    [[nodiscard]] std::vector<GeneralVoFResolutionRequest> resolution_requests() const;
    [[nodiscard]] std::vector<PLIC::PolyhedralInterface> reconstruct() const;
    [[nodiscard]] std::vector<GeneralVoFReconstructionPatch> reconstruction_patches() const;
    /** Explicit opposite-pure interfaces on bounded halo canonical fine subfaces.
     * Ordinary arithmetic, independent of backend. Deduplicate replicas by face ID. */
    [[nodiscard]] std::vector<GeneralVoFFaceInterface> face_interfaces() const;
    /** Collective, detached capillary geometry. Never changes phase inventories,
     * histories or transport declarations. Incomplete support requests resolution. */
    [[nodiscard]] GeneralVoFSurfaceCatalog capillary_surface(GeneralVoFSurfaceOptions = {}) const;
    [[nodiscard]] const std::vector<Meshes::AMR::AdaptiveNode>& nodes() const noexcept;
    [[nodiscard]] const std::vector<Meshes::AMR::AdaptiveTile>& tiles() const noexcept;
    /** Retained tile ordinals incident to a canonical node, in tiles() order.
     * The immutable incidence storage is shared by accepted transport states. */
    [[nodiscard]] const std::vector<std::size_t>& node_tiles(std::uint64_t node) const;
    [[nodiscard]] std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>>
    owned_geometry() const;
    template<ArithmeticMode Mode>
    [[nodiscard]] std::vector<PLIC::ConvexGeometryPtr<Mode>> owned_finite_geometry() const;
    /** Immutable original node constructions for a retained canonical face. */
    template<ArithmeticMode Mode>
    [[nodiscard]] std::vector<PLIC::Point<Mode>> finite_face_points(std::uint64_t face) const;
    [[nodiscard]] std::vector<std::uint64_t> node_cells(std::uint64_t node) const;
    [[nodiscard]] std::vector<std::uint64_t> cell_support(std::uint64_t owned_cell) const;
    [[nodiscard]] std::vector<GeneralVoFSupportCell> support_cells() const;
    /** Local immutable publication tokens; these allocate no snapshot buffers.
     * Callers must still validate native/map observations collectively. */
    [[nodiscard]] std::shared_ptr<const void> accepted_identity() const noexcept;
    [[nodiscard]] std::uint64_t accepted_control() const noexcept;
    [[nodiscard]] bool accepted_unchanged() const noexcept;
    /** Collective source and native-publication authentication. */
    void validate_prepared_transport_source(const GeneralVoFPreparedTransport&) const;
    /** Collective on the opaque transfer's source communicator. Authenticate
     * this actual adopted publication without exporting/reconstructing capsules. */
    void validate_transfer_target(const ExactPhaseTransferResult&) const;
    [[nodiscard]] ExactPhaseTransferSource export_transfer_source(bool prepare_geometry = false) const;
    [[nodiscard]] std::shared_ptr<const GeneralVoF> adopt_transfer(const ExactPhaseTransferResult&) const;
    /** Source-pinned bounded handoff. No exact capsule can be adopted through
     * this seam. The native hierarchy remains the immutable mesh authority. */
    class FiniteRemapTrial
    {
    public:
        [[nodiscard]] std::shared_ptr<const GeneralVoF> candidate() const;
        [[nodiscard]] const std::vector<GeneralVoFFiniteOwnedChannels>& owned_channels() const noexcept
        {
            return channels_;
        }
        [[nodiscard]] bool ownership_only() const noexcept { return ownership_only_; }
        [[nodiscard]] const std::vector<GeneralVoFFiniteRemapRoute>& owned_routes() const noexcept { return routes_; }
        /** Source-owned immutable common-refinement pieces, authenticated by
         * this trial's source/target observations and bounded cover checks. */
        [[nodiscard]] const std::vector<GeneralVoFFiniteCommonCell>& source_common_cells() const noexcept
        {
            return common_cells_;
        }
        /** Collective on the pinned source communicator. */
        void validate_observations() const;

    private:
        friend GeneralVoF;
        friend struct Exact::GeneralVoF;
        std::shared_ptr<const Data> source_, candidate_;
        std::vector<GeneralVoFFiniteOwnedChannels> channels_;
        std::vector<GeneralVoFFiniteRemapRoute> routes_;
        std::vector<GeneralVoFFiniteCommonCell> common_cells_;
        bool ownership_only_ = false;
    };
    [[nodiscard]] FiniteRemapTrial plan_remap(const Hierarchy::Plan&,
        std::shared_ptr<const Hierarchy> actual_bound_target,
        std::span<const GeneralVoFFiniteOwnedChannels> owned_channels = {}, GeneralVoFFiniteRemapOptions = {}) const;

    class Trial
    {
    public:
        /** Explicit exact-route records. Empty for Finite; canonical face views
         * are below and family-specific ledgers are bound by prepare_transport. */
        [[nodiscard]] const std::vector<GeneralVoFFaceContribution>& face_contributions() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFTransfer>& transfers() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFDonorBudget>& donor_budgets() const noexcept;
        /** This trial's once-rounded phase error; identity trials report zero. */
        [[nodiscard]] const std::vector<PhaseRoundoff>& phase_roundoff_bounds() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFNodeTrajectory>& node_trajectories() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteFaceContribution>& finite_face_contributions() const noexcept;
        /** Legacy/homogeneous positive records. InterfaceLocalSwept leaves these
         * empty; use its prepared signed receipts/budgets and numerical support. */
        [[nodiscard]] const std::vector<GeneralVoFFiniteTransfer>& finite_transfers() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteDonorBudget>& finite_donor_budgets() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteNodeTrajectory>& finite_node_trajectories() const noexcept;
        /** All locally retained canonical tiles; MPI shared-face replicas repeat.
         * Deduplicate by canonical face ID before forming a global diagnostic sum. */
        [[nodiscard]] const std::vector<GeneralVoFFaceChainDiagnostic>& finite_face_chain_diagnostics() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFCellClosureDiagnostic>&
        finite_cell_closure_diagnostics() const noexcept;
        [[nodiscard]] std::shared_ptr<const GeneralVoF> candidate() const;
        /** Optional prebound compatibility ledger for face transport. Swept
         * callers use prepare_transport()/prepare_transport_view() uniformly;
         * detached preparation does not mutate this trial or cache context. */
        [[nodiscard]] const std::shared_ptr<const GeneralVoFPreparedTransport>& prepared_transport() const noexcept
        {
            return prepared_;
        }
        /** Collective detached finite preparation. Preserves both transport
         * families without reinterpreting the physical signed ledger. Exact
         * trials retain their separate authority and reject this finite seam. */
        [[nodiscard]] std::shared_ptr<const GeneralVoFPreparedTransport> prepare_transport(
            std::array<double, 2> density = {1, 1}, std::uint64_t time_stage = 0) const;
        template<ArithmeticMode Mode>
        [[nodiscard]] GeneralVoFPreparedTrialView<Mode> prepare_transport_view(
            std::array<double, 2> density = {1, 1}, std::uint64_t time_stage = 0) const
        {
            return GeneralVoFPreparedTrialView<Mode>(prepare_transport_checked(Mode, density, time_stage));
        }
        template<ArithmeticMode Mode>
        void validate_prepared_transport(const GeneralVoFPreparedTrialView<Mode>& view, std::array<double, 2> density,
            std::uint64_t time_stage) const
        {
            validate_prepared_transport_mode(Mode, &view.shared_ledger());
            validate_prepared_transport(view.ledger(), density, time_stage);
        }
        /** Collective transaction and caller-context authentication. */
        void validate_prepared_transport(
            const GeneralVoFPreparedTransport&, std::array<double, 2> density, std::uint64_t time_stage) const;

    private:
        friend GeneralVoF;
        friend struct Exact::GeneralVoF;
        SIMPLEFLUID_FVM_LOCAL void validate_retained_algebraic_payload() const;
        void validate_prepared_transport_mode(ArithmeticMode,
            const std::shared_ptr<const GeneralVoFPreparedTransport>* retained = nullptr) const;
        [[nodiscard]] std::shared_ptr<const GeneralVoFPreparedTransport> prepare_transport_checked(
            ArithmeticMode, std::array<double, 2>, std::uint64_t) const;
        std::shared_ptr<const Data> source_, candidate_;
        std::uint64_t transaction_ = 0;
        double timestep_ = 0;
        std::shared_ptr<const GeneralVoFPreparedTransport> prepared_;
        bool homogeneous_algebraic_ = false, interface_local_ = false;
        std::vector<GeneralVoFSignedReceipt> signed_receipts_;
        std::vector<GeneralVoFSignedDonorBudget> signed_budgets_;
        std::vector<GeneralVoFFiniteTransfer> numerical_support_transfers_;
        std::vector<GeneralVoFFiniteDonorBudget> numerical_support_budgets_;
        std::vector<GeneralVoFDirectedStream> unbound_streams_;
        GeneralVoFTransportDiagnostics unbound_diagnostics_;
        std::vector<GeneralVoFFaceContribution> faces_;
        std::vector<GeneralVoFTransfer> transfers_;
        std::vector<GeneralVoFDonorBudget> budgets_;
        std::vector<PhaseRoundoff> roundoff_;
        std::vector<GeneralVoFNodeTrajectory> trajectories_;
        std::vector<GeneralVoFFiniteFaceContribution> finite_faces_;
        std::vector<GeneralVoFFiniteTransfer> finite_transfers_;
        std::vector<GeneralVoFFiniteDonorBudget> finite_budgets_;
        std::vector<GeneralVoFFiniteNodeTrajectory> finite_trajectories_;
        std::vector<GeneralVoFFaceChainDiagnostic> finite_face_chain_;
        std::vector<GeneralVoFCellClosureDiagnostic> finite_cell_closure_;
    };
    /** All locally retained canonical nodes/faces must be supplied exactly once.
     * Shared authority verifies identical represented velocity/Phi controls.
     * A closed impermeable boundary projects trajectories onto the complete
     * authenticated incident wall catalog. Ordinary uses nominal FP geometry
     * and ledgers. BoundedCertification adds explicit enclosures; Exact alone
     * selects the rational construction route. Modes never fall back. Shared
     * controls authenticate the complete catalog and departure before transport.
     * Source-free occupancy requires canonical face closure within its original
     * physical tolerance (exact zero in Exact). Ordinary conservation uses the
     * configured source/recipient phase scale and independent rounding allowance.
     * Ordinary face geometry errors have owner-issued shares of the existing
     * cell allowance; canonical residual plus signed geometry defects must still
     * satisfy that original cell allowance without a second budget claim.
     * Before binary64 publication, each certified finite recipient phase must
     * independently admit its current-step geometry/arithmetic enclosure against
     * reconstruction.volume_tolerance times that phase's positive lower amount.
     * Donor/bulk amounts and historical error cannot supply this allowance;
     * final narrowing and accumulated temporal history remain separate evidence.
     * Exhaustion is an arithmetic rejection, not a spatial-resolution diagnosis.
     * No cell/source correction is invented. Mixed finite interfaces
     * require neighborhood resolution admission before expensive fitting;
     * unavailable refinement rejects with a request and preserves accepted state.
     * Unsupported periodic images reject until image-qualified sweeps exist. */
    [[nodiscard]] Trial plan_transport(
        std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double timestep) const;
    template<ArithmeticMode Mode>
    [[nodiscard]] GeneralVoFControlView<Mode> transport_controls(std::span<const GeneralVoFNodeVelocity> nodes,
        std::span<const GeneralVoFFaceVolume> faces, double timestep) const
    {
        return GeneralVoFControlView<Mode>(make_transport_controls(Mode, nodes, faces, timestep));
    }
    template<ArithmeticMode Mode> [[nodiscard]] Trial plan_transport(const GeneralVoFControlView<Mode>& view) const
    {
        validate_transport_controls(view.controls_, Mode, false);
        return plan_transport_impl(view.nodes(), view.faces(), view.timestep());
    }
    template<ArithmeticMode Mode>
    [[nodiscard]] Trial plan_face_transport(const GeneralVoFControlView<Mode>& view,
        std::array<double, 2> density = {1, 1}, std::uint64_t time_stage = 0) const
    {
        validate_transport_controls(view.controls_, Mode, true);
        return plan_face_transport(view.faces(), view.timestep(), density, time_stage);
    }
    template<ArithmeticMode Mode>
    [[nodiscard]] Trial plan_homogeneous_transport(const GeneralVoFControlView<Mode>& view) const
    {
        validate_transport_controls(view.controls_, Mode, true);
        return plan_homogeneous_transport(view.faces(), view.timestep());
    }
    /** Collective whole-domain admission for Ordinary SweptPartition's
     * interface-free algebraic specialization. Explicit pure declarations and
     * same-phase nonperiodic adjacency are required, including retained ghosts. */
    [[nodiscard]] bool homogeneous_transport_eligible() const;
    /** Explicit fixed-mesh homogeneous model with authoritative canonical face
     * volumes. Rechecks collective eligibility and face controls, without
     * requesting or fabricating node trajectories. Density and time-stage are
     * bound later by Trial::prepare_transport. Exact zero flux preserves the
     * source publication identity and phase histories. */
    [[nodiscard]] Trial plan_homogeneous_transport(std::span<const GeneralVoFFaceVolume>, double timestep) const;
    /** Fixed-mesh source-free Ordinary transport. Canonical integrated volumes
     * are authoritative under a constant normal face velocity model. Closed
     * boundaries only: nonzero exterior flux and periodic images reject.
     * No swept solids, recipient overlays, or complete donor partitions.
     * Densities and time-stage identity are collective controls. */
    [[nodiscard]] Trial plan_face_transport(std::span<const GeneralVoFFaceVolume>, double timestep,
        std::array<double, 2> density = {1, 1}, std::uint64_t time_stage = 0) const;
    [[nodiscard]] std::shared_ptr<const GeneralVoF> apply(const Trial&) const;

private:
    friend struct Exact::GeneralVoF;
    [[nodiscard]] std::shared_ptr<const GeneralVoFTransportControls> make_transport_controls(
        ArithmeticMode, std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
    void validate_transport_controls(
        const std::shared_ptr<const GeneralVoFTransportControls>&, ArithmeticMode, bool face_only) const;
    [[nodiscard]] Trial plan_transport_impl(
        std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
    SIMPLEFLUID_FVM_LOCAL Trial plan_face_transport_impl(std::span<const GeneralVoFFaceVolume>, double,
        std::array<double, 2>, std::uint64_t, bool homogeneous, std::uint64_t transaction_id = 0,
        bool numerical_baseline = false) const;
    SIMPLEFLUID_FVM_LOCAL Trial plan_interface_transport(
        Trial, const general_vof_detail::LocalTransportControls&) const;
    template<ArithmeticMode Mode>
    SIMPLEFLUID_FVM_LOCAL Trial plan_transport_finite(
        std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double) const;
    template<ArithmeticMode Mode>
    SIMPLEFLUID_FVM_LOCAL FiniteRemapTrial plan_remap_finite(const Hierarchy::Plan&, std::shared_ptr<const Hierarchy>,
        std::span<const GeneralVoFFiniteOwnedChannels>, GeneralVoFFiniteRemapOptions) const;
    GeneralVoF(std::shared_ptr<const Hierarchy>, std::vector<Key>, std::vector<double>, std::vector<double>,
        GeneralVoFOptions, std::vector<GeneralVoFPhaseGeometry>, const ExactPhaseTransferResult*);
    template<ArithmeticMode Mode>
    SIMPLEFLUID_FVM_LOCAL void construct_mode(std::shared_ptr<const Hierarchy>, std::vector<Key>, std::vector<double>,
        std::vector<double>, GeneralVoFOptions, std::vector<GeneralVoFPhaseGeometry>, const ExactPhaseTransferResult*);
    explicit GeneralVoF(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::VoF
