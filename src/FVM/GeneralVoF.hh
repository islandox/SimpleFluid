/** @file GeneralVoF.hh
 * @brief Source-pinned unsplit phase transport on authority-certified native cells.
 */
#pragma once
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/PhaseVolumes.hh"
#include "FVM/PolyhedralReconstruction.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include <memory>
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
    /** Optional authoritative rational Phi. When present the double above is
     * diagnostic only; no re-rounding may destroy exact occupancy closure.
     * Projection/face-arithmetic repair remains the caller's separately gated
     * physical responsibility, never a hidden source correction here. */
    std::optional<Meshes::ExactGeometryMeasure> exact_integrated_volume;
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
struct GeneralVoFSupportCell
{
    std::uint64_t cell = 0, revision = 0;
    int owner = -1;
    Meshes::ConvexGeometryPtr geometry;
    Meshes::GeometryPoint centroid{};
    Meshes::AMR::AdaptiveCellKey key;
};
struct GeneralVoFReconstructionPatch
{
    std::uint64_t cell = 0;
    int owner = -1;
    Meshes::ConvexGeometryPtr geometry;
    PolyhedralInterface interface;
};
class SIMPLEFLUID_FVM_EXPORT GeneralVoF
{
public:
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
    /** Authoritative independent finite binary64 phase volumes in owned order. */
    [[nodiscard]] const std::vector<PhaseVolumes>& phase_volumes() const noexcept;
    /** Latest publication arithmetic uncertainty, never a physical source. */
    [[nodiscard]] const std::vector<PhaseRoundoff>& phase_roundoff_bounds() const noexcept;
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
    [[nodiscard]] std::vector<PolyhedralInterface> reconstruct() const;
    [[nodiscard]] std::vector<GeneralVoFReconstructionPatch> reconstruction_patches() const;
    [[nodiscard]] const std::vector<Meshes::AMR::AdaptiveNode>& nodes() const noexcept;
    [[nodiscard]] const std::vector<Meshes::AMR::AdaptiveTile>& tiles() const noexcept;
    [[nodiscard]] std::vector<Meshes::ConvexGeometryPtr> owned_geometry() const;
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
    class Trial
    {
    public:
        [[nodiscard]] const std::vector<GeneralVoFFaceContribution>& face_contributions() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFTransfer>& transfers() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFDonorBudget>& donor_budgets() const noexcept;
        /** This trial's once-rounded phase error; identity trials report zero. */
        [[nodiscard]] const std::vector<PhaseRoundoff>& phase_roundoff_bounds() const noexcept;
        [[nodiscard]] const std::vector<GeneralVoFNodeTrajectory>& node_trajectories() const noexcept;
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
    };
    /** All locally retained canonical nodes/faces must be supplied exactly once.
     * Shared authority verifies identical represented velocity/Phi controls.
     * A closed impermeable boundary uses exact tangent trajectories projected
     * from the represented velocity onto the complete authenticated incident
     * exterior-wall planes. The full wall catalog and final exact departure
     * are shared controls; no caller can omit a wall or override departure.
     * Source-free occupancy requires exact zero cell sum of canonical Phi;
     * nonzero divergence rejects, never inventing a cell/source correction.
     * Unsupported periodic images reject until image-qualified sweeps exist. */
    [[nodiscard]] Trial plan_transport(
        std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double timestep) const;
    [[nodiscard]] std::shared_ptr<const GeneralVoF> apply(const Trial&) const;

private:
    GeneralVoF(std::shared_ptr<const Hierarchy>, std::vector<Key>, std::vector<double>, std::vector<double>,
        GeneralVoFOptions, std::vector<GeneralVoFPhaseGeometry>, const ExactPhaseTransferResult*);
    explicit GeneralVoF(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::VoF
