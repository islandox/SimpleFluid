/** @file ConservativeAMRRemap.hh
 * @brief Sparse conservative extensive transfer on certified ancestry overlaps.
 */
#pragma once
#include "FVM/AdaptiveVoF.hh"
#include "geometry/AdaptiveOverlap.hh"

namespace SimpleFluid::VoF
{
/** Geometry supplied by the accepted reconstruction, separate from physical inventories. */
enum class AMRPhaseGeometry
{
    Plic,
    PureGas,
    PureCarrier
};

/** Physical phase volumes, independently retained even when a fraction rounds pure.
 * Arrays follow source owned-cell order. Plic is the default declaration and
 * requires a finite nonzero plane in every physically mixed cell. An explicit
 * PureGas/PureCarrier declaration represents accepted rounded-pure geometry;
 * only its empty minority within the physical consistency bound may use bulk
 * overlap weights. An omitted/zero mixed-cell plane is never that declaration.
 * Volumes are m3; plane coordinates are physical metres. Pure physical cells
 * ignore the plane. Empty geometry_declaration selects Plic for every cell.
 */
struct AMRRemapPhases
{
    std::vector<double> gas_volume, carrier_volume;
    std::vector<Meshes::GeometryHalfspace> gas_halfspace;
    std::vector<AMRPhaseGeometry> geometry_declaration;
};

struct AMRRemapOutput
{
    std::vector<double> gas_volume, carrier_volume; ///< Target owned cells, m3.
    std::vector<ConservedQuantity> inventories;
};

/** Overflow-safe signed and L1 budgets, including both retained phases.
 * Physical quantities equal ldexp(scaled value, binary_exponent). The exponent
 * is selected separately for each budget, preserving microscopic/subnormal-only
 * budgets without a physical norm floor. Values remain scaled so diagnostics do
 * not require an unrepresentable physical double total. MPI reductions use only
 * supported integer/double types; acceptance includes their arithmetic bound.
 */
struct AMRRemapReport
{
    std::vector<int> binary_exponent;
    std::vector<double> scaled_before, scaled_after, scaled_absolute_before, scaled_absolute_after;
};

/** Certified geometry plus sparse native donor imports and reverse coverage sums.
 * Construction and apply are collective on the default Tpetra communicator.
 * Endpoint mesh identities, epochs, owned maps and exact plan observations are
 * retained. Only target-owned records and required donor field values are used
 * during application; all source/target certificate catalog cells are exactly
 * matched to their native owners, including nonactive remote siblings. Owner
 * region/material/typed ancestry bindings are authenticated by exact byte
 * exchange; this is not the future Stage 6 global ancestry directory. The
 * supplied geometry catalog is never searched all-pairs. Native shell bootstrap
 * exchange pads to the global maximum per-cell payload width; variable-length
 * packing for heterogeneous large shells remains a Stage 6 scalability gate.
 * The existing optimized Cartesian remap is a separate numerical backend.
 * No solver fields, fractions, output ghosts or accepted states are published.
 */
class SIMPLEFLUID_FVM_EXPORT ConservativeAMRRemap
{
public:
    using Handle = MeshHandle<>;
    ConservativeAMRRemap(
        SP<const Handle> source, SP<const Handle> target, std::shared_ptr<const Meshes::AdaptiveOverlapPlan> overlaps);
    ~ConservativeAMRRemap();
    ConservativeAMRRemap(ConservativeAMRRemap&&) noexcept;
    ConservativeAMRRemap& operator=(ConservativeAMRRemap&&) noexcept;
    ConservativeAMRRemap(const ConservativeAMRRemap&) = delete;
    ConservativeAMRRemap& operator=(const ConservativeAMRRemap&) = delete;

    /** Atomically replace output only after geometry, physical carrier, positivity
     * and signed L1 conservation gates pass. Amounts are extensive, never densities.
     * MixtureMass uses the supplied positive phase densities in kg/m3.
     * A positive phase within its geometry-scaled consistency bound may use bulk
     * overlap weights when its PLIC remnant is unresolvable; paired inventories
     * always use the same carrier weights. Material discrepancies reject.
     */
    [[nodiscard]] AMRRemapReport apply(const AMRRemapPhases& phases, const std::vector<ConservedQuantity>& inventories,
        AMRRemapOutput& output, double liquid_density = 1.0, double gas_density = 1.0) const;

private:
    struct SIMPLEFLUID_FVM_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::VoF
