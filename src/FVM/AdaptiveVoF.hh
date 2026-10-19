/** @file AdaptiveVoF.hh
 * @brief Conservative geometric VoF on a distributed dyadic Cartesian hierarchy.
 */
#pragma once

#include "FVM/GeometricVoF.hh"
#include "geometry/CartesianAMR.hh"
#include "geometry/MeshHandle.hh"

#include <memory>
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
    std::vector<double> fraction;                  ///< Owned cells.
    std::vector<double> gas_volume, liquid_volume; ///< Overlap faces, owner-outward signed m3.
    double maximum_volume_divergence = 0.0;        ///< Global max |dt div u|.
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
    std::vector<Meshes::AMRBox> promotion_regions; ///< Refine/protect to maximum level.
};

struct RefinementPlan
{
    RefinementPlan() = default;
    std::shared_ptr<const Meshes::CartesianAMR>
        source_hierarchy; ///< Guards leaf ordinals against topology replacement.
    std::vector<std::size_t> refine;
    std::vector<Meshes::AMRBlock> coarsen;
    bool resolution_limited = false; ///< A protected interface is at max_level above the diameter target.
};

/**
 * Distributed leaf-cell geometry, with native canonical coarse/fine subfaces.
 *
 * Inputs and returned cell arrays contain owned cells in native map order.
 * Face arrays contain native overlap faces in canonical owner orientation.
 * All operations are collective on the native mesh communicator. Only compact
 * hierarchy metadata is replicated. Temporary reconstruction halos use sparse
 * Tpetra imports. No finest-level volume mesh or global solution copy is made.
 * The topology is immutable; construct a new instance and remap to regrid.
 */
class SIMPLEFLUID_FVM_EXPORT AdaptiveVoF
{
public:
    using Hierarchy = Meshes::CartesianAMR;
    using Handle = MeshHandle<>;
    explicit AdaptiveVoF(std::shared_ptr<const Hierarchy> hierarchy);
    ~AdaptiveVoF();
    AdaptiveVoF(AdaptiveVoF&&) noexcept;
    AdaptiveVoF& operator=(AdaptiveVoF&&) noexcept;
    AdaptiveVoF(const AdaptiveVoF&) = delete;
    AdaptiveVoF& operator=(const AdaptiveVoF&) = delete;

    [[nodiscard]] const std::shared_ptr<const Hierarchy>& hierarchy() const noexcept;
    [[nodiscard]] const SP<const Handle>& mesh() const noexcept;
    [[nodiscard]] std::vector<double> exchange_cells(const std::vector<double>& owned) const;
    [[nodiscard]] std::vector<double> exchange_faces(const std::vector<double>& overlap) const;
    [[nodiscard]] std::vector<InterfacePlane> reconstruct(const std::vector<double>& fraction) const;
    [[nodiscard]] std::vector<double> curvature(const std::vector<double>& fraction) const;
    /** Shared timestep, CFL <= 0.5 and |dt div u| <= 1e-10; no level subcycling. */
    [[nodiscard]] AdaptiveAdvection advect(const std::vector<double>& fraction,
        const std::vector<double>& face_velocity, double dt, std::size_t first_axis = 0) const;
    /** PLIC intersection for phase volumes; carrier-weighted conservative extensives. */
    [[nodiscard]] AdaptiveRemapResult remap(const AdaptiveVoF& target, const std::vector<double>& fraction,
        const std::vector<ConservedQuantity>& inventories = {}, double liquid_density = 1.0,
        double gas_density = 1.0) const;
    /** Protect all interfaces and their stencils; coarsen only uniform pure-phase sibling groups. */
    [[nodiscard]] RefinementPlan refinement_plan(const std::vector<double>& fraction,
        const std::vector<double>& face_velocity, RefinementOptions options = {}, RefinementHints hints = {}) const;

private:
    struct SIMPLEFLUID_FVM_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace SimpleFluid::VoF
