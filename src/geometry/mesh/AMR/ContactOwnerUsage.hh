/** @file ContactOwnerUsage.hh
 * @brief Persistent physical-owner multiplicities for contact page deltas.
 */
#pragma once
#include "ContactGroupAllocation.hh"
#include <cstdint>
#include <functional>
#include <optional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactOwnerUsageEditor;

/** Counts physical allocations whose owners are pinned by the caller's actual
 * row/fan/fact pages. Numeric keys never establish ownership or extend a body's
 * lifetime. An index must not escape those owners. Roles distinguish separate
 * allocations, not separate consumers of the same allocation.
 *
 * The index admits its metadata only. Callers use zero/nonzero transitions to
 * admit or retire bodies, and must separately reconcile incidence-shared
 * credits. Forking requires a fresh candidate arena and imports cached metadata
 * bytes in constant work. Mutation copies only balanced search paths. */
class ContactOwnerUsage
{
    struct Node;
    using Pointer = std::shared_ptr<const Node>;

public:
    struct Key
    {
        const void* allocation = nullptr;
        std::uint32_t role = 0;
        bool operator==(const Key&) const = default;
    };
    struct Entry
    {
        std::size_t references = 0, bytes = 0;
        const Budget* origin = nullptr;
    };
    struct Transition
    {
        std::size_t before = 0, after = 0, bytes = 0;
        const Budget* origin = nullptr;
        [[nodiscard]] bool inserted() const noexcept { return !before && after; }
        [[nodiscard]] bool retired() const noexcept { return before && !after; }
    };
    struct Work
    {
        std::size_t visited_nodes = 0, created_nodes = 0;
    };
    struct Page
    {
        const void* identity;
        const Budget* origin;
        std::size_t bytes; ///< Admitted bytes, including the control-block allowance.
        bool inherited;
        std::size_t object_bytes; ///< Actual object payload; shared-control overhead excluded.
    };

    explicit ContactOwnerUsage(std::shared_ptr<Budget>);
    ContactOwnerUsage(const ContactOwnerUsage&, std::shared_ptr<Budget>);
    ContactOwnerUsage(ContactOwnerUsage&&) noexcept = default;
    ContactOwnerUsage(const ContactOwnerUsage&) = delete;
    ContactOwnerUsage& operator=(const ContactOwnerUsage&) = delete;
    ~ContactOwnerUsage();
    [[nodiscard]] ContactOwnerUsage fork(std::shared_ptr<Budget> arena) const
    {
        return ContactOwnerUsage(*this, std::move(arena));
    }
    [[nodiscard]] Transition retain(Key, std::size_t bytes, const Budget* origin);
    [[nodiscard]] Transition release(Key);
    [[nodiscard]] std::optional<Entry> find(Key) const noexcept;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] std::size_t distinct_owners() const noexcept;
    [[nodiscard]] std::size_t overlap_bytes() const noexcept;
    [[nodiscard]] Work last_edit_work() const noexcept { return work_; }
    /** End the accepted/candidate overlap after the owning operation commits.
     * Subsequent edits retire replaced inherited metadata immediately. */
    void finish_overlap() noexcept;
    /** Diagnostic census only; neither fork nor mutation invokes these scans. */
    void visit_entries(const std::function<void(Key, Entry)>&) const;
    void visit_pages(const std::function<void(Page)>&) const;

private:
    friend struct ContactOwnerUsageEditor;
    [[nodiscard]] std::size_t inherited_bytes(const Pointer&) const noexcept;
    void publish(Pointer, Work);

    ContactGroupAllocation imported_;
    Pointer root_;
    std::uint64_t editor_;
    Work work_;
    bool keep_overlap_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
