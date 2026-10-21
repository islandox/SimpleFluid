/** @file AdaptiveVoF.hh
 * @brief Conservative geometric VoF on a distributed dyadic Cartesian hierarchy.
 */
#pragma once

#include "FVM/GeometricVoF.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace SimpleFluid::VoF
{

/** Extensive amounts are distributed with their named carrier during remapping. */
enum class InventoryCarrier
{
    BulkVolume,
    GasVolume,
    LiquidVolume,
    MixtureMass
};

struct ConservedQuantity
{
    std::string name;
    InventoryCarrier carrier = InventoryCarrier::BulkVolume;
    std::vector<double> amount; ///< Extensive amount per owned cell, not per unit volume.
    bool nonnegative = true;
};

struct AdaptiveRemapResult
{
    std::vector<double> fraction;
    std::vector<ConservedQuantity> inventories;
};

struct AdaptiveAdvection
{
    std::vector<double> fraction; ///< Owned cells.
    /** Overlap faces, owner-outward signed m3; historical liquid_volume denotes the nonresolved 1-f carrier. */
    std::vector<double> gas_volume, liquid_volume;
    double maximum_volume_divergence = 0.0;      ///< Global max |dt div u|.
    double maximum_volume_source_residual = 0.0; ///< Global max |dt div u-(DeltaG+DeltaC)/V|.
};

/** Signed physical phase-volume changes over this step, in m3 per owned cell. */
struct PhaseVolumeIncrement
{
    std::vector<double> gas;
    std::vector<double> carrier; ///< The entire nonresolved 1-f volume, including dispersed void.
};

/** @brief Side of a resolved interface used for cell sampling and source deposition. */
enum class InterfaceSide
{
    Gas,
    Carrier
};

/**
 * @brief Uniquely owned resolved-interface geometry and its adjacent global cells.
 *
 * A PLIC patch belongs to its owned mixed cell and uses that GID on both sides.
 * A pure gas/carrier mesh-face patch belongs to the canonical face owner; the
 * carrier cell can therefore be remote. Centroids use the native geometry's
 * physical coordinates; film geometry retains the correct periodic image.
 * Records must originate from this exact immutable hierarchy and remain intact.
 */
struct InterfacePatch
{
    enum class Kind
    {
        Plic,
        MeshFace
    };
    Kind kind = Kind::Plic;
    std::uint64_t gas_cell = 0;
    std::uint64_t carrier_cell = 0;
    Vector centroid{};
    Vector normal{}; ///< Unit normal directed from resolved gas into carrier.
    double area = 0.0;

private:
    friend class AdaptiveVoF;
    std::shared_ptr<const SimpleFluid::Meshes::AMR::CartesianAMR> source_hierarchy_;
    std::uint64_t source_entity_ = 0;
    int owner_rank_ = -1;
    Vector lower_{}, upper_{}, plane_normal_{}; ///< Carrier-cell local box and gas-side plane.
    double plane_offset_ = 0.0;
};

struct RefinementOptions
{
    double minimum_resolved_diameter = 0.01; ///< Metres; initial resolution target, not qualification.
    unsigned cells_per_diameter = 32;
    unsigned stencil_cells = 4;         ///< Minimum protected reconstruction neighborhood.
    double forecast_time = 0.0;         ///< Seconds; adds a max-speed motion buffer.
    double curvature_resolution = 0.25; ///< Refine when |kappa| h exceeds this; zero disables.
    double wake_threshold = 1.0;        ///< Threshold for supplied dimensionless wake indicator.
};

struct RefinementHints
{
    std::vector<double> wake_indicator;            ///< Optional owned-cell dimensionless nonnegative values.
    std::vector<SimpleFluid::Meshes::AMR::AMRBox> promotion_regions; ///< Refine/protect to maximum level.
};

struct RefinementPlan
{
    RefinementPlan() = default;
    std::shared_ptr<const SimpleFluid::Meshes::AMR::CartesianAMR>
        source_hierarchy; ///< Guards leaf ordinals against topology replacement.
    SimpleFluid::Meshes::AMR::HierarchyObservation
        source_observation;                   ///< Exact topology, partition and geometry observation.
    std::shared_ptr<const void> source_state; ///< Optional retained solver publication; independent of geometry.
    std::optional<std::uint64_t> source_state_revision; ///< Solver control revision when state-dependent.
    std::vector<std::size_t> refine;
    std::vector<SimpleFluid::Meshes::AMR::AMRBlock> coarsen;
    bool resolution_limited = false; ///< A protected interface is at max_level above the diameter target.
};

/** Rank-local reconstruction metadata payload, excluding allocator/map/importer overhead.
 * Payload bytes are capacity-based estimates, not process RSS; shared instances
 * report the same shared payload rather than additive ownership.
 */
struct AdaptiveGeometryStorage
{
    std::size_t patches = 0, virtual_cells = 0, direct_cells = 0, explicit_intersections = 0;
    std::size_t sample_payload_bytes = 0;
};

/**
 * Distributed leaf-cell geometry, with native canonical coarse/fine subfaces.
 *
 * Inputs and returned cell arrays contain owned cells in native map order.
 * Face arrays contain native overlap faces in canonical owner orientation.
 * Numerical field operations are collective on the native mesh communicator;
 * film_length is explicitly a local geometric query. Only compact
 * hierarchy metadata is replicated. Temporary reconstruction halos use sparse
 * Tpetra imports. No finest-level volume mesh or global solution copy is made.
 * The topology is immutable; construct a new instance and remap to regrid.
 * Instances constructed from the same hierarchy share immutable reconstruction
 * metadata and import plans collectively, while each retains private exchange
 * workspaces. Fraction-dependent reconstruction results remain call-local.
 */
class SIMPLEFLUID_FVM_EXPORT AdaptiveVoF
{
public:
    using Hierarchy = SimpleFluid::Meshes::AMR::CartesianAMR;
    using Handle = MeshHandle<>;
    explicit AdaptiveVoF(std::shared_ptr<const Hierarchy> hierarchy);
    ~AdaptiveVoF();
    AdaptiveVoF(AdaptiveVoF&&) noexcept;
    AdaptiveVoF& operator=(AdaptiveVoF&&) noexcept;
    AdaptiveVoF(const AdaptiveVoF&) = delete;
    AdaptiveVoF& operator=(const AdaptiveVoF&) = delete;

    [[nodiscard]] const std::shared_ptr<const Hierarchy>& hierarchy() const noexcept;
    [[nodiscard]] const SP<const Handle>& mesh() const noexcept;
    [[nodiscard]] AdaptiveGeometryStorage geometry_storage() const noexcept;
    [[nodiscard]] std::vector<double> exchange_cells(const std::vector<double>& owned) const;
    /** Batched native import with instance-private mutable work vectors. */
    [[nodiscard]] std::vector<Vector> exchange_cell_vectors(const std::vector<Vector>& owned) const;
    [[nodiscard]] std::vector<Vector> exchange_face_vectors(const std::vector<Vector>& overlap) const;
    [[nodiscard]] std::vector<double> exchange_faces(const std::vector<double>& overlap) const;
    [[nodiscard]] std::vector<InterfacePlane> reconstruct(const std::vector<double>& fraction) const;
    [[nodiscard]] std::vector<double> curvature(const std::vector<double>& fraction) const;
    /** Shared timestep, CFL <= 0.5 and |dt div u| <= 1e-10; no level subcycling. */
    [[nodiscard]] AdaptiveAdvection advect(const std::vector<double>& fraction,
        const std::vector<double>& face_velocity, double dt, std::size_t first_axis = 0) const;
    /**
     * Physical-source counterpart: retain the frozen-WY directional correction,
     * then apply DeltaG-chi*(DeltaG+DeltaC). Projected face-volume divergence
     * must match the supplied total source, and closed global compatibility and
     * every intermediate/final phase bound remain mandatory. No source clipping
     * or mean subtraction is performed. Nonzero increments require dt>0.
     */
    [[nodiscard]] AdaptiveAdvection advect(const std::vector<double>& fraction,
        const std::vector<double>& face_velocity, double dt, const PhaseVolumeIncrement& source,
        std::size_t first_axis = 0) const;
    /** Collective PLIC and exact pure gas/carrier face patches, owned exactly once. */
    [[nodiscard]] std::vector<InterfacePatch> interface_patches(const std::vector<double>& fraction) const;
    /** Collective overlap-face carrier aperture: intersection of both incident PLIC traces; walls are closed. */
    [[nodiscard]] std::vector<double> carrier_face_aperture(const std::vector<double>& fraction) const;
    /** Shared gas traces; physical boundary faces return their incident gas trace for attachment detection. */
    [[nodiscard]] std::vector<double> gas_face_aperture(const std::vector<double>& fraction) const;
    /**
     * Local carrier-volume average of min(normal distance to interface, delta),
     * in metres. Integrates the actual cut-cell geometry; delta must be positive.
     * The normalized geometric factor does not include dispersed/aqueous fractions.
     */
    [[nodiscard]] double film_length(const InterfacePatch& patch, double delta) const;
    /** Collective sparse deposition of signed patch amounts onto the selected cell side. */
    [[nodiscard]] std::vector<double> scatter_patch_amounts(
        const std::vector<InterfacePatch>& patches, const std::vector<double>& amounts, InterfaceSide side) const;
    /** Collective sparse sampling of an owned-cell scalar at each patch's selected cell side. */
    [[nodiscard]] std::vector<double> sample_patch_values(
        const std::vector<InterfacePatch>& patches, const std::vector<double>& owned_values, InterfaceSide side) const;
    /** PLIC intersection for phase volumes; carrier-weighted conservative extensives. */
    [[nodiscard]] AdaptiveRemapResult remap(const AdaptiveVoF& target, const std::vector<double>& fraction,
        const std::vector<ConservedQuantity>& inventories = {}, double liquid_density = 1.0,
        double gas_density = 1.0) const;
    /** Protect all interfaces and their stencils; coarsen only uniform pure-phase sibling groups. */
    [[nodiscard]] RefinementPlan refinement_plan(const std::vector<double>& fraction,
        const std::vector<double>& face_velocity, RefinementOptions options = {}, RefinementHints hints = {}) const;

private:
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::vector<double> phase_face_aperture(
        const std::vector<double>& fraction, bool gas) const;
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL AdaptiveAdvection advect_with_sources(const std::vector<double>& fraction,
        const std::vector<double>& face_velocity, double dt, const PhaseVolumeIncrement* source,
        std::size_t first_axis) const;
    SIMPLEFLUID_FVM_LOCAL void validate_patches(const std::vector<InterfacePatch>& patches, InterfaceSide side) const;
    struct SIMPLEFLUID_FVM_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace SimpleFluid::VoF
