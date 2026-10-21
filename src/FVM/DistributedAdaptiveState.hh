/** @file DistributedAdaptiveState.hh
 * @brief Atomic native publication of named conservative/history snapshots.
 */
#pragma once
#include "FVM/ConservativeAMRRemap.hh"
#include "fields/FieldStored.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/FrozenNative.hh"
#include <optional>
namespace SimpleFluid::VoF
{
/** A caller-declared accepted history, not a time-advance instruction. Amounts
 * and physical carriers follow the explicit input key order. Pressure and other
 * intensive solver histories are not reconstructed by this conservative API. */
struct AdaptiveInventoryPair
{
    std::string number, moles;
};
struct AdaptiveHistoryFrame
{
    std::string name;
    AMRRemapPhases phases;
    std::vector<ConservedQuantity> inventories;
    std::vector<AdaptiveInventoryPair> pairs;
    double carrier_density = 1, gas_density = 1;
    bool reconstruction_ready = true;
};
struct AdaptiveHistoryReconstruction
{
    std::string name;
    std::vector<Meshes::GeometryHalfspace> gas_halfspace;
    std::vector<AMRPhaseGeometry> geometry_declaration;
};
/** Per-rank application payload limit, independent of planner/transport limits.
 * Each frame allocation, field allocation and packed phase must fit this bound;
 * simultaneous planner/provider/runtime storage is reported separately. */
struct AdaptiveStateOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024;
    std::vector<SimpleFluid::Meshes::AMR::AdaptiveCellKey> additional_full_ghosts;
};
/** Component capacities/payload counts and explicit estimates, not an additive
 * whole-process total. Native/planner components may share immutable recipes;
 * source/candidate plans and runtime communication buffers are separate. */
struct AdaptiveStateStorage
{
    Meshes::MeshStorageReport native;
    SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy::Storage planner;
    SimpleFluid::Meshes::AMR::FrozenNativeModel::Storage frozen_planner;
    std::size_t frame_capacity_bytes = 0, schema_bytes = 0, field_scalar_payload_bytes = 0;
    std::size_t state_metadata_estimate_bytes = 0, native_registry_estimate_bytes = 0;
    std::size_t owned_cells = 0, full_ghost_cells = 0, outer_metric_cells = 0;
    /** Public map element payload counts; excludes Tpetra hash tables/control
     * blocks. Importer routing is an estimate, not an allocator measurement. */
    std::size_t map_gid_payload_bytes = 0, importer_routing_estimate_bytes = 0;
    /** Whole-process samples, normalized to bytes; includes runtime/allocator
     * overhead and is not a component allocation measurement. */
    std::optional<std::size_t> resident_bytes, process_peak_resident_bytes;
};
class SIMPLEFLUID_FVM_EXPORT DistributedAdaptiveState
{
public:
    struct Data;
    using Key = SimpleFluid::Meshes::AMR::AdaptiveCellKey;
    using Hierarchy = SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy;
    using Field = ScalarCellFieldStored<>;
    /** Collective. The exact named schema is agreed before field construction;
     * input arrays are reordered from input_keys into the actual native map. */
    DistributedAdaptiveState(std::shared_ptr<const Hierarchy>, std::vector<Key> input_keys,
        std::vector<AdaptiveHistoryFrame> histories, AdaptiveStateOptions options = {});
    /** Ownership-only accepted native copies preserve caller-declared history
     * flags. They do not certify PLIC or enable any topology-changing subset. */
    DistributedAdaptiveState(std::shared_ptr<const SimpleFluid::Meshes::AMR::FrozenNativeModel>,
        std::vector<Key> input_keys, std::vector<AdaptiveHistoryFrame> histories, AdaptiveStateOptions options = {});
    [[nodiscard]] const std::shared_ptr<const Hierarchy>& hierarchy() const noexcept;
    [[nodiscard]] const std::shared_ptr<const SimpleFluid::Meshes::AMR::FrozenNativeModel>&
    frozen_model() const noexcept;
    [[nodiscard]] const std::shared_ptr<const MeshHandle<>>& mesh() const noexcept;
    [[nodiscard]] const std::vector<Key>& owned_order() const noexcept;
    [[nodiscard]] const std::vector<AdaptiveHistoryFrame>& histories() const noexcept;
    [[nodiscard]] std::shared_ptr<const Field> field(const std::string& history, const std::string& quantity) const;
    [[nodiscard]] AdaptiveStateStorage storage() const;
    class Plan
    {
    public:
        [[nodiscard]] std::shared_ptr<const DistributedAdaptiveState> candidate() const;
        /** Accepted local native+state application bytes, excluding self,
         * directory/halo/MPI traffic. Rejected proposed cost remains in balance(). */
        [[nodiscard]] std::uint64_t migration_wire_bytes() const noexcept { return migration_wire_bytes_; }
        /** Actual local native migration payload plus explicit application
         * framing. Rejected prepared proposals exchange no native records. */
        [[nodiscard]] std::uint64_t native_application_bytes_exchanged() const noexcept
        {
            return native_application_bytes_exchanged_;
        }
        /** Only this transaction's state-exchange scratch; not native/halo/MPI peak.
         * Source/candidate field/planner/provider snapshots coexist separately. */
        [[nodiscard]] std::size_t state_exchange_peak_bytes() const noexcept { return state_exchange_peak_bytes_; }
        /** Native migration packet-buffer phase, excluding decode/maps/halo/MPI.
         * Empty when the native implementation reports no measured phase.
         * Never a whole-process peak, and not additive with the separate
         * state-exchange phase. */
        [[nodiscard]] std::optional<std::size_t> native_packet_phase_peak_bytes() const noexcept
        {
            return native_packet_phase_peak_bytes_;
        }
        [[nodiscard]] const std::optional<SimpleFluid::Meshes::AMR::AdaptiveBalancePlan>& balance() const noexcept
        {
            return balance_;
        }
        [[nodiscard]] const std::vector<AMRRemapReport>& conservation_reports() const noexcept { return conservation_; }

    private:
        friend DistributedAdaptiveState;
        std::shared_ptr<const Data> source_, candidate_;
        std::uint64_t migration_wire_bytes_ = 0, native_application_bytes_exchanged_ = 0;
        std::size_t state_exchange_peak_bytes_ = 0;
        std::optional<std::size_t> native_packet_phase_peak_bytes_;
        std::optional<SimpleFluid::Meshes::AMR::AdaptiveBalancePlan> balance_;
        std::vector<AMRRemapReport> conservation_;
    };
    /** Ownership-only packet copies preserve represented amounts, phases and
     * reconstruction coefficients bit for bit, including signed zero/remnants. */
    [[nodiscard]] Plan plan_migration(std::span<const int> destinations) const;
    /** Requalifies candidate cost with actual per-peer native shared dictionaries,
     * schema, state/history payload and framing before accepting the proposal. */
    [[nodiscard]] Plan plan_migration(const SimpleFluid::Meshes::AMR::AdaptiveBalancePlan&) const;
    /** Exact complete bounded family closures; deeper unsupported coverage
     * rejects rather than importing an implicit global descendant catalog. */
    [[nodiscard]] Plan plan_adaptation(std::span<const SimpleFluid::Meshes::AMR::AdaptiveMark>,
        std::span<const SimpleFluid::Meshes::AMR::AdaptiveBytes> coarsen = {},
        std::span<const Key> target_additional_full_ghosts = {}) const;
    [[nodiscard]] std::shared_ptr<const DistributedAdaptiveState> apply(const Plan&) const;
    /** Supplies only reconstruction geometry for already-published physical
     * carriers. It does not advance histories or alter any conserved amount. */
    [[nodiscard]] std::shared_ptr<const DistributedAdaptiveState> with_reconstruction(
        std::span<const AdaptiveHistoryReconstruction>) const;

private:
    explicit DistributedAdaptiveState(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    [[nodiscard]] Plan migrate(std::span<const int>, const SimpleFluid::Meshes::AMR::AdaptiveBalancePlan*) const;
    [[nodiscard]] Plan migrate_frozen(std::span<const int>, const SimpleFluid::Meshes::AMR::AdaptiveBalancePlan*) const;
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::VoF
