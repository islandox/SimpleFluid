/** @file GeneralVoF.hh
 * @brief Source-pinned unsplit phase transport on authority-certified native cells.
 */
#pragma once
#include "geometry/GeometryStatisticsSnapshot.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/ExperimentalPhaseInventory.hh"
#include "FVM/PhaseVolumes.hh"
#include "FVM/PolyhedralReconstruction.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/BoundedConvexGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include <array>
#include <memory>
#include <string>
#include <optional>
#include <span>
#include <vector>
namespace SimpleFluid::VoF
{
enum class GeneralVoFPhaseGeometry
{
    Infer,
    PureGas,
    PureCarrier
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
    PolyhedralReconstructionOptions reconstruction;
    Meshes::ExactSweepLimits sweep;
    PhaseInventoryPolicy inventory_policy = PhaseInventoryPolicy::Finite;
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
    std::vector<Meshes::BoundedGeometryValue> amounts;
};
struct GeneralVoFFiniteRemapRoute
{
    Meshes::AMR::AdaptiveCellKey source_key, target_key;
    std::uint64_t source_gid = 0, target_gid = 0;
    int source_owner = -1, target_owner = -1;
    Meshes::BoundedPhaseMeasures amount;
};
struct GeneralVoFFiniteCommonCell
{
    Meshes::AMR::AdaptiveCellKey source_key, target_key;
    std::uint64_t source_gid = 0, target_gid = 0;
    int target_owner = -1;
    Meshes::BoundedConvexGeometryPtr geometry;
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
    Meshes::GeometryPoint velocity{};
};
struct GeneralVoFNodeTrajectory
{
    std::uint64_t node = 0;
    Meshes::ExactNodeTrajectory trajectory;
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
    std::optional<Meshes::ExactGeometryMeasure> exact_integrated_volume;
    /** Finite numerical authority, separate from the explicit exact control. */
    std::optional<Meshes::BoundedGeometryValue> bounded_integrated_volume;
};
/** Diagnostic intervals only. There is no implicit scalar conversion; a known
 * nonzero interval may contain0 below output range. Never use these to build
 * physical fluxes, budgets or matrix coefficients: use exact measures. */
struct GeneralVoFPhaseEnclosure
{
    Meshes::CertifiedGeometryValue bulk, gas, carrier;
};
/** One signed simplex/donor contribution; opposite directions never net away.
 * These are geometric donor contributions, not necessarily incident upwind cells. */
struct GeneralVoFFaceContribution
{
    std::uint64_t face = 0, donor = 0;
    int donor_owner = -1, direction = 1;
    Meshes::ExactPhaseMeasures exact;
    GeneralVoFPhaseEnclosure diagnostic;
};
/** Transaction-local geometric donor-to-recipient measure, including self atoms.
 * It is seeded from finite accepted phases and may support one-step density
 * weighting/binding. It is never a persistent accepted inventory. */
struct GeneralVoFTransfer
{
    std::uint64_t donor = 0, recipient = 0;
    int donor_owner = -1, recipient_owner = -1;
    Meshes::ExactPhaseMeasures exact;
    GeneralVoFPhaseEnclosure diagnostic;
};
/** One-step certificate lifted from the current finite accepted state.
 * Unrounded remainders are not carried into the next accepted state. */
struct GeneralVoFDonorBudget
{
    std::uint64_t donor = 0;
    Meshes::ExactPhaseMeasures exact_old, exact_exported, exact_retained;
    GeneralVoFPhaseEnclosure old_diagnostic, exported_diagnostic, retained_diagnostic;
    /** Exact cumulative absolute gas relabeling; carrier correction is its
     * opposite. Signed-face multiplicities are counted before cancellation. */
    Meshes::ExactGeometryMeasure signed_repair_absolute, signed_repair_bound;
    Meshes::ExactGeometryMeasure route_repair_absolute, route_repair_bound;
    Meshes::ExactGeometryMeasure full_cell_fit_discrepancy;
};
/** Fixed-storage numerical records used exclusively by the Finite route.
 * A canonical owned face contributes one signed donor group; receivers consume
 * the same record with opposite incidence. No rational value is manufactured.
 */
struct GeneralVoFFiniteFaceContribution
{
    std::uint64_t face = 0, donor = 0;
    int donor_owner = -1, direction = 1;
    Meshes::BoundedPhaseMeasures amount;
};
struct GeneralVoFFiniteTransfer
{
    std::uint64_t donor = 0, recipient = 0;
    int donor_owner = -1, recipient_owner = -1;
    Meshes::BoundedPhaseMeasures amount;
    PhaseRoundoff inherited_error;
};
struct GeneralVoFFiniteDonorBudget
{
    std::uint64_t donor = 0;
    Meshes::BoundedPhaseMeasures old, exported, retained;
};
struct GeneralVoFFiniteNodeTrajectory
{
    std::uint64_t node = 0;
    Meshes::BoundedPoint point, departure;
};
struct GeneralVoFSupportCell
{
    std::uint64_t cell = 0, revision = 0;
    int owner = -1;
    Meshes::ConvexGeometryPtr geometry;
    Meshes::GeometryPoint centroid{};
    Meshes::AMR::AdaptiveCellKey key;
    Meshes::BoundedConvexGeometryPtr finite_geometry;
};
/** Owned refinement requests, collected without preparing an expensive fit. */
struct GeneralVoFResolutionRequest
{
    std::uint64_t cell = 0;
    Meshes::AMR::AdaptiveCellKey key;
    PolyhedralResolutionAssessment assessment;
};
struct GeneralVoFReconstructionPatch
{
    std::uint64_t cell = 0;
    int owner = -1;
    Meshes::ConvexGeometryPtr geometry;
    PolyhedralInterface interface;
    Meshes::BoundedConvexGeometryPtr finite_geometry;
};
class SIMPLEFLUID_FVM_EXPORT GeneralVoF
{
public:
    /** This library image's calling-thread counters. Call every observed
     * image's bridge and deduplicate snapshots by their separate domain IDs. */
    static void enable_exact_geometry_statistics(bool enabled);
    static void reset_geometry_statistics();
    [[nodiscard]] static Meshes::GeometryStatisticsSnapshot geometry_statistics();
    struct Data;
    using Hierarchy = Meshes::AMR::DistributedAdaptiveHierarchy;
    using Key = Meshes::AMR::AdaptiveCellKey;
    /** Input phases follow input_keys; each local owner must name every owned
     * leaf exactly once. Full ghosts are owner-authenticated sparse imports.
     * With omitted declarations, a fresh exactly absent phase proves pure
     * geometry, retained under the existing fixed consistency bound. Nonzero
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
    /** Current authority: fresh finite lifts by default, retained exact values
     * only after explicit ExperimentalExact opt-in. No cross-mode cache. */
    [[nodiscard]] std::vector<Meshes::ExactPhaseMeasures> inventory_measures() const;
    /** Explicit experimental authority; rejects Finite mode. Unqualified for
     * finite-duration physical flow, performance and storage scalability. */
    [[nodiscard]] const std::vector<Meshes::ExactPhaseMeasures>& experimental_phases() const;
    /** Latest publication arithmetic uncertainty, never a physical source. */
    [[nodiscard]] const std::vector<PhaseRoundoff>& phase_roundoff_bounds() const noexcept;
    /** Independent accumulated finite arithmetic uncertainty across accepted
     * steps, distinct from this publication's conversion error. */
    [[nodiscard]] const std::vector<PhaseRoundoff>& accumulated_phase_errors() const noexcept;
    /** Complementary geometric fitting labels in owned_order. The physical
     * minority is lifted from finite storage and the other label is actualV−minority;
     * physical inventories and signed occupancy provenance remain unchanged. */
    [[nodiscard]] std::vector<Meshes::ExactPhaseMeasures> geometric_phases() const;
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
    [[nodiscard]] std::vector<PolyhedralInterface> reconstruct() const;
    [[nodiscard]] std::vector<GeneralVoFReconstructionPatch> reconstruction_patches() const;
    [[nodiscard]] const std::vector<Meshes::AMR::AdaptiveNode>& nodes() const noexcept;
    [[nodiscard]] const std::vector<Meshes::AMR::AdaptiveTile>& tiles() const noexcept;
    /** Retained tile ordinals incident to a canonical node, in tiles() order.
     * The immutable incidence storage is shared by accepted transport states. */
    [[nodiscard]] const std::vector<std::size_t>& node_tiles(std::uint64_t node) const;
    [[nodiscard]] std::vector<Meshes::ConvexGeometryPtr> owned_geometry() const;
    [[nodiscard]] std::vector<Meshes::BoundedConvexGeometryPtr> owned_finite_geometry() const;
    /** Immutable original node constructions for a retained canonical face. */
    [[nodiscard]] std::vector<Meshes::BoundedPoint> finite_face_points(std::uint64_t face) const;
    [[nodiscard]] std::vector<std::uint64_t> node_cells(std::uint64_t node) const;
    [[nodiscard]] std::vector<std::uint64_t> cell_support(std::uint64_t owned_cell) const;
    [[nodiscard]] std::vector<GeneralVoFSupportCell> support_cells() const;
    /** Local immutable publication tokens; these allocate no snapshot buffers.
     * Callers must still validate native/map observations collectively. */
    [[nodiscard]] std::shared_ptr<const void> accepted_identity() const noexcept;
    [[nodiscard]] std::uint64_t accepted_control() const noexcept;
    [[nodiscard]] bool accepted_unchanged() const noexcept;
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
        std::shared_ptr<const Data> source_, candidate_;
        std::vector<GeneralVoFFiniteOwnedChannels> channels_;
        std::vector<GeneralVoFFiniteRemapRoute> routes_;
        std::vector<GeneralVoFFiniteCommonCell> common_cells_;
        bool ownership_only_ = false;
    };
    [[nodiscard]] FiniteRemapTrial plan_remap_finite(const Hierarchy::Plan&,
        std::shared_ptr<const Hierarchy> actual_bound_target,
        std::span<const GeneralVoFFiniteOwnedChannels> owned_channels = {}, GeneralVoFFiniteRemapOptions = {}) const;

    class Trial
    {
    public:
        /** Explicit exact-route records. Empty for Finite; its authoritative
         * numerical records are available through the finite_* accessors. */
        [[nodiscard]] const std::vector<GeneralVoFFaceContribution>& face_contributions() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFTransfer>& transfers() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFDonorBudget>& donor_budgets() const noexcept;
        /** This trial's once-rounded phase error; identity trials report zero. */
        [[nodiscard]] const std::vector<PhaseRoundoff>& phase_roundoff_bounds() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFNodeTrajectory>& node_trajectories() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteFaceContribution>& finite_face_contributions() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteTransfer>& finite_transfers() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteDonorBudget>& finite_donor_budgets() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFFiniteNodeTrajectory>& finite_node_trajectories() const noexcept;
        [[nodiscard]] std::shared_ptr<const GeneralVoF> candidate() const;

    private:
        friend GeneralVoF;
        std::shared_ptr<const Data> source_, candidate_;
        std::uint64_t transaction_ = 0;
        std::vector<GeneralVoFFaceContribution> faces_;
        std::vector<GeneralVoFTransfer> transfers_;
        std::vector<GeneralVoFDonorBudget> budgets_;
        std::vector<PhaseRoundoff> roundoff_;
        std::vector<GeneralVoFNodeTrajectory> trajectories_;
        std::vector<GeneralVoFFiniteFaceContribution> finite_faces_;
        std::vector<GeneralVoFFiniteTransfer> finite_transfers_;
        std::vector<GeneralVoFFiniteDonorBudget> finite_budgets_;
        std::vector<GeneralVoFFiniteNodeTrajectory> finite_trajectories_;
    };
    /** All locally retained canonical nodes/faces must be supplied exactly once.
     * Shared authority verifies identical represented velocity/Phi controls.
     * A closed impermeable boundary projects trajectories onto the complete
     * authenticated incident wall catalog. Finite uses sealed source supports,
     * bounded FP trajectories/face chains and independently bounded phase
     * ledgers; ExperimentalExact retains the exact construction route. Shared
     * controls authenticate the complete catalog and departure before transport.
     * Source-free occupancy requires canonical face closure within the derived
     * arithmetic bound and unchanged physical target (exact zero in the exact
     * route). No cell/source correction is invented. Mixed finite interfaces
     * require neighborhood resolution admission before expensive fitting;
     * unavailable refinement rejects with a request and preserves accepted state.
     * Unsupported periodic images reject until image-qualified sweeps exist. */
    [[nodiscard]] Trial plan_transport(
        std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double timestep) const;
    [[nodiscard]] std::shared_ptr<const GeneralVoF> apply(const Trial&) const;

private:
    GeneralVoF(std::shared_ptr<const Hierarchy>, std::vector<Key>, std::vector<double>, std::vector<double>,
        GeneralVoFOptions, std::vector<GeneralVoFPhaseGeometry>, const ExactPhaseTransferResult*);
    explicit GeneralVoF(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    [[nodiscard]] Trial plan_transport_finite(std::span<const GeneralVoFNodeVelocity>,
        std::span<const GeneralVoFFaceVolume>, double timestep) const;
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::VoF
