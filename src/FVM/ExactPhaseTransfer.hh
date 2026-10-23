/** @file ExactPhaseTransfer.hh
 * @brief Exact-geometry handoff of finite phase inventories across native AMR transactions.
 */
#pragma once
#include "FVM/ExperimentalPhaseInventory.hh"
#include "FVM/PhaseVolumes.hh"
#include "FVM/PolyhedralReconstruction.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include <optional>
#include <string>
namespace SimpleFluid::VoF
{
class GeneralVoF;
enum class ExactTransferCarrier
{
    BulkVolume,
    GasVolume,
    CarrierVolume,
    MixtureMass
};
struct ExactTransferChannel
{
    std::string name;
    ExactTransferCarrier carrier = ExactTransferCarrier::BulkVolume;
    bool nonnegative = true;
    auto operator<=>(const ExactTransferChannel&) const = default;
};
/** Bounded transaction-local products from current selected phase authority.
 * Not an accepted physical-state remainder or a physical-step history. */
struct ExactTransferPhaseMass
{
    Meshes::ExactGeometryMeasure gas, carrier;
};
/** Provenance remains distinct: physical phase inputs are never normalized to
 * erase numerical occupancy or PLIC fitting differences. */
using ExactTransferError = PhaseError;
struct ExactPhaseTransferCapsule
{
    Meshes::AMR::AdaptiveCellKey key;
    std::uint64_t revision = 0;
    PhaseInventoryPolicy policy = PhaseInventoryPolicy::Finite;
    /** Present only for explicit ExperimentalExact authority. Finite views and
     * finite diagnostic bounds remain distinct from this cold payload. */
    std::shared_ptr<const ExperimentalPhaseInventory> experimental_exact;
    PhaseVolumes phases;
    PhaseRoundoff roundoff;
    /** Independent accumulated finite arithmetic history. Explicit reference
     * remaps transfer it with phase carriers; ExperimentalExact leaves it empty. */
    PhaseRoundoff accumulated_roundoff;
    PolyhedralInterface geometry;
    bool geometry_ready = false;
    ExactTransferError error;
    std::optional<ExactTransferPhaseMass> phase_mass;
    /** Transaction-local weighted products, never adopted as accepted physical
     * phase/history state. Rebuild from the current selected authority and
     * physical fields per remap; these are distinct from the optional payload. */
    std::vector<Meshes::ExactGeometryMeasure> amounts;
};
struct ExactTransferOwnedChannels
{
    Meshes::AMR::AdaptiveCellKey key;
    std::vector<Meshes::ExactGeometryMeasure> amounts;
    std::optional<ExactTransferPhaseMass> phase_mass;
};
struct ExactPhaseTransferOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024;
    /** Derived defect bound relative to each actual target bulk volume; no
     * absolute carrier floor. Exhaustion rejects before state publication. */
    double maximum_relative_target_defect = 1e-12;
    double maximum_relative_target_fit = 1e-12;
};
/** Only an accepted GeneralVoF snapshot creates the immutable phase source.
 * Flow clients may append extensive channels: e.g. physical momentum uses
 * MixtureMass, pressure-volume history uses BulkVolume. Values are not primitive
 * cell averages. A MixtureMass channel requires explicit phase mass inventories;
 * no mass is inferred from an interface fraction or source velocity. */
class SIMPLEFLUID_FVM_EXPORT ExactPhaseTransferSource
{
public:
    using Hierarchy = Meshes::AMR::DistributedAdaptiveHierarchy;
    struct Data;
    [[nodiscard]] const std::shared_ptr<const Hierarchy>& hierarchy() const noexcept;
    [[nodiscard]] const std::shared_ptr<const void>& identity() const noexcept;
    [[nodiscard]] std::uint64_t control() const noexcept;
    [[nodiscard]] PhaseInventoryPolicy inventory_policy() const noexcept;
    [[nodiscard]] const std::vector<ExactPhaseTransferCapsule>& owned() const noexcept;
    [[nodiscard]] const std::vector<ExactTransferChannel>& channels() const noexcept;
    /** Collective. Exact keyed inventory inputs must cover local owners once. */
    [[nodiscard]] ExactPhaseTransferSource with_channels(std::vector<ExactTransferChannel>,
        std::vector<ExactTransferOwnedChannels>, ExactPhaseTransferOptions = {}) const;

private:
    friend class GeneralVoF;
    friend class ExactPhaseTransfer;
    ExactPhaseTransferSource(std::shared_ptr<const Hierarchy>, std::shared_ptr<const void> identity,
        std::uint64_t control, std::vector<ExactPhaseTransferCapsule>, PhaseInventoryPolicy);
    explicit ExactPhaseTransferSource(std::shared_ptr<const Data>);
    std::shared_ptr<const Data> data_;
};
/** Each step names its ACTUAL native-bound candidate. A following plan may
 * pin this candidate's map/geometry observation; equivalent rebuilt views are
 * never substituted by the transfer utility. */
struct ExactTransferPlanStep
{
    ExactPhaseTransferSource::Hierarchy::Plan plan;
    std::shared_ptr<const ExactPhaseTransferSource::Hierarchy> candidate;
};
/** Opaque validated handoff. GeneralVoF adoption compares source identity,
 * policy, control and observed native maps, then installs the selected finite
 * or explicitly experimental authority. The finite path has no cold payload.
 * Mixed topology outputs require reconstruction on the target geometry. */
class SIMPLEFLUID_FVM_EXPORT ExactPhaseTransferResult
{
public:
    struct Data;
    [[nodiscard]] const std::shared_ptr<const ExactPhaseTransferSource::Hierarchy>& source_hierarchy() const noexcept;
    [[nodiscard]] const std::shared_ptr<const ExactPhaseTransferSource::Hierarchy>& target_hierarchy() const noexcept;
    [[nodiscard]] const std::shared_ptr<const void>& source_identity() const noexcept;
    [[nodiscard]] std::uint64_t source_control() const noexcept;
    [[nodiscard]] std::uint64_t transaction() const noexcept;
    [[nodiscard]] bool ownership_only() const noexcept;
    [[nodiscard]] PhaseInventoryPolicy inventory_policy() const noexcept;
    [[nodiscard]] const std::vector<ExactPhaseTransferCapsule>& owned() const noexcept;
    [[nodiscard]] const std::vector<ExactTransferChannel>& channels() const noexcept;
    /** Collective source/target geometry and map staleness preflight. */
    void validate_observations() const;

private:
    friend class ExactPhaseTransfer;
    explicit ExactPhaseTransferResult(std::shared_ptr<const Data>);
    std::shared_ptr<const Data> data_;
};
class SIMPLEFLUID_FVM_EXPORT ExactPhaseTransfer
{
public:
    /** Target is the actual native-bound candidate of the supplied source plan.
     * Geometric source/target coverage is exact. Physical phases publish once
     * in binary64 with derived conversion bounds; Source/Result phase/error
     * fields retain no rational remainder in the default Finite policy. An
     * explicit ExperimentalExact handoff preserves its separate cold authority;
     * policies are pinned and cannot cross on adoption. Optional weighted products
     * remain transaction-local until consumers publish finite u/p histories. */
    [[nodiscard]] static ExactPhaseTransferResult prepare(const ExactPhaseTransferSource&,
        const ExactPhaseTransferSource::Hierarchy::Plan&,
        std::shared_ptr<const ExactPhaseTransferSource::Hierarchy> target, ExactPhaseTransferOptions = {});
    /** Authenticate at most eight immutable hierarchy plans in order, then
     * transfer DIRECTLY from the original accepted source to the final actual
     * native-bound target. No intermediate physical ledger is generated.
     * Original/final overlaps must still satisfy the existing one-generation
     * per ancestry factor contract. Every intermediate observation is retained
     * and revalidated. Chain length and all transaction IDs agree collectively.
     */
    [[nodiscard]] static ExactPhaseTransferResult prepare(
        const ExactPhaseTransferSource&, std::span<const ExactTransferPlanStep> chain, ExactPhaseTransferOptions = {});
};
} // namespace SimpleFluid::VoF
