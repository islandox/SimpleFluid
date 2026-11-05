#pragma once
#include "CartesianSourceLineRanges.hh"
#include "PooledLineExchange.hh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Only the actual Session's completed point-publication boundary may bind a
 * point-pool namespace to its exact current immutable snapshot. This is a
 * supplemental namespace capability, never authority to invent query output. */
class PooledLineCompletionSnapshotAdmission
{
public:
    PooledLineCompletionSnapshotAdmission(const PooledLineCompletionSnapshotAdmission&) = default;
    PooledLineCompletionSnapshotAdmission(PooledLineCompletionSnapshotAdmission&&) = default;
    ~PooledLineCompletionSnapshotAdmission() = default;

private:
    PooledLineCompletionSnapshotAdmission(
        std::shared_ptr<const ContactSourcePointSnapshot> snapshot, const IncidencePools& pools)
        : snapshot_(std::move(snapshot)), point_pool_(&pools.points)
    {
    }
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot_;
    const void* point_pool_;
    friend class DistributedAdaptiveIncidenceSession;
    friend class CertifiedIncidenceResultView;
};

/** Immutable view of one already certified real-kernel result. Construction
 * checks saved global slots AND their complete canonical identities. The source
 * namespace check is supplemental: it cannot certify a query by itself. */
class CertifiedIncidenceResultView
{
public:
    [[nodiscard]] static std::optional<CertifiedIncidenceResultView> certify(
        std::shared_ptr<const CartesianSourceLineRangeRecipe>, std::size_t query,
        std::span<const std::size_t> expected_slots, const ContactSourcePointSnapshot& current);
    /** Retire only cached DATA after full ordered equality to a receiver-issued
     * result and the original captured request. Failure leaves completion intact.
     * The caller releases completion only after publishing the returned view. */
    [[nodiscard]] static std::optional<CertifiedIncidenceResultView> certify_completed(const PooledLineCompletion&,
        std::size_t line, const PooledLineSubscriber& original, const PooledLineCompletionSnapshotAdmission&,
        std::span<const std::size_t> expected_slots, const IncidencePools&, ContactHaloContext, unsigned kernel_layer,
        Budget&, std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>);
    [[nodiscard]] const PooledLineSubscriber* original_request() const noexcept;
    [[nodiscard]] std::shared_ptr<const PooledLineCompletion::Proof> completion_proof() const noexcept;
    [[nodiscard]] bool imports_owner(const void*) const noexcept;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t operator[](std::size_t) const;
    [[nodiscard]] ContactSourcePoint value(std::size_t) const;
    [[nodiscard]] const std::shared_ptr<const CartesianSourceLineRangeRecipe>& recipe() const noexcept;
    [[nodiscard]] const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot() const noexcept;
    [[nodiscard]] std::size_t query_ordinal() const noexcept;
    void retain_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    CertifiedIncidenceResultView(std::shared_ptr<const CartesianSourceLineRangeRecipe>, std::size_t);
    struct RemoteValue;
    explicit CertifiedIncidenceResultView(std::shared_ptr<const RemoteValue>);
    std::shared_ptr<const CartesianSourceLineRangeRecipe> recipe_;
    std::shared_ptr<const RemoteValue> remote_;
    std::size_t query_ = 0;
};

/** One immutable edge-view array, shared by the saved edge cache and loop
 * concatenation. Construction admits all foreign dependencies first. */
class CertifiedIncidenceResultParts
{
public:
    [[nodiscard]] static CertifiedIncidenceResultParts create(std::span<const CertifiedIncidenceResultView>, Budget&,
        std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>);
    [[nodiscard]] std::span<const CertifiedIncidenceResultView> parts() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t lease_capacity() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t known_payload_bytes() const noexcept;
    [[nodiscard]] const void* storage_identity() const noexcept;
    [[nodiscard]] bool imports_owner(const void*) const noexcept;
    void retain_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    struct Storage;
    explicit CertifiedIncidenceResultParts(std::shared_ptr<const Storage>);
    std::shared_ptr<const Storage> storage_;
};

/** Exact compression of saved ordered loop outputs, never geometry authority.
 * An ineligible candidate leaves the caller's explicit original untouched.
 * The shared descriptor has no Layer/snapshot-history/arena back edge. */
class CertifiedIncidenceLoopView
{
public:
    enum class Kind : std::uint8_t
    {
        Quad,
        Concatenation
    };
    [[nodiscard]] static std::optional<CertifiedIncidenceLoopView> certify_quad(
        std::shared_ptr<const ContactSourcePointSnapshot>, std::span<const std::size_t> expected_slots, Budget&,
        std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena);
    /** Keep edge order and omit exactly its last endpoint. No reversal, sorting,
     * interval estimate, periodic translation, or output-vector cache is used. */
    [[nodiscard]] static std::optional<CertifiedIncidenceLoopView> certify_concatenation(
        std::span<const CertifiedIncidenceResultView> ordered_edges, std::span<const std::size_t> expected_slots,
        const ContactSourcePointSnapshot& current, Budget&, std::shared_ptr<Budget> keeper,
        std::shared_ptr<CartesianAuthorityArena> arena);
    [[nodiscard]] static std::optional<CertifiedIncidenceLoopView> certify_concatenation(
        const CertifiedIncidenceResultParts&, std::span<const std::size_t> expected_slots,
        const ContactSourcePointSnapshot& current, Budget&, std::shared_ptr<Budget>,
        std::shared_ptr<CartesianAuthorityArena>);
    /** A checked slice of the SAME immutable parts owner. Periodic users select
     * physical image zero instead of concatenating its inverse-image query. */
    [[nodiscard]] static std::optional<CertifiedIncidenceLoopView> certify_concatenation(
        const CertifiedIncidenceResultParts&, std::size_t first_part, std::size_t part_count,
        std::span<const std::size_t> expected_slots, const ContactSourcePointSnapshot& current, Budget&,
        std::shared_ptr<Budget>, std::shared_ptr<CartesianAuthorityArena>);
    [[nodiscard]] std::span<const CertifiedIncidenceResultView> parts() const noexcept;
    /** Share the descriptor after exact ordered slot and FULL identity replay. */
    [[nodiscard]] bool exact_replay(std::span<const std::size_t>, const ContactSourcePointSnapshot&) const;
    [[nodiscard]] const CertifiedIncidenceResultParts* result_parts() const noexcept;
    [[nodiscard]] Kind kind() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t operator[](std::size_t) const;
    [[nodiscard]] ContactSourcePoint value(std::size_t) const;
    [[nodiscard]] std::size_t part_count() const noexcept;
    [[nodiscard]] std::size_t part_capacity() const noexcept;
    [[nodiscard]] std::size_t lease_capacity() const noexcept;
    /** Admission charge includes the explicit control allowance; payload bytes
     * report actual C++ objects/capacities, excluding unknown allocator/control
     * overhead. Neither measure includes dependency bodies. */
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t known_payload_bytes() const noexcept;
    [[nodiscard]] const Budget* allocation_budget() const noexcept;
    [[nodiscard]] bool has_arena_keeper() const noexcept;
    [[nodiscard]] const void* storage_identity() const noexcept;
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept;
    [[nodiscard]] bool imports_owner(const void*) const noexcept;
    void retain_dependencies(cartesian_detail::CreditConstruction&,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>&, cartesian_detail::Allocation&) const;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    struct Storage;
    explicit CertifiedIncidenceLoopView(std::shared_ptr<const Storage>);
    static CertifiedIncidenceLoopView publish(std::shared_ptr<const ContactSourcePointSnapshot>, std::uint64_t,
        std::optional<CertifiedIncidenceResultParts>, std::size_t, Budget&, std::shared_ptr<Budget>,
        std::shared_ptr<CartesianAuthorityArena>, std::size_t first_part = 0, std::size_t part_count = 0);
    std::shared_ptr<const Storage> storage_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
