/** @file AdaptiveLeafBodyUsage.hh
 * @brief Persistent physical allocation usage for immutable AMR leaf pages.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"

#include <map>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct AdaptiveLeafBodyBuilder;
enum class AdaptiveLeafBodyRole { Leaf, Family };

/** Count each physical body once across active leaf pages and saved family
 * parents. The hierarchy owns this index; leaf pages and family capsules pin
 * the actual bodies independently. */
class AdaptiveLeafBodyUsage
{
    struct Node;
    using NodePtr = std::shared_ptr<const Node>;

public:
    AdaptiveLeafBodyUsage() = default;
    [[nodiscard]] const PLIC::details::OwnerManifestPair& owners() const noexcept { return owners_; }
    [[nodiscard]] const PLIC::details::NativeOwnerReceipts& paid_receipts() const noexcept { return paid_; }
    /** Detached control for exactly one immutable leaf/family membership. */
    [[nodiscard]] const auto& membership_control() const noexcept { return membership_; }
    [[nodiscard]] const PLIC::details::GeometryNativeAdministration& administration() const noexcept { return administration_; }
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t leaf_payload_bytes() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] std::size_t projected_bytes() const;
    /** Exact canonical graph/control lookup; no diagnostic owner enumeration. */
    [[nodiscard]] bool same_membership(const AdaptiveLeafBodyUsage&) const noexcept;
    void project_administration(PLIC::details::GeometryOwnerGraphEditor&, bool retain) const;
    void visit_role_delta(const AdaptiveLeafBodyUsage& previous, AdaptiveLeafBodyRole,
        PLIC::details::NativePaidExtentGroup::MemberSink) const;
    [[nodiscard]] std::optional<AdaptiveLeafBodyRole> physical_role(
        std::uint8_t kind, const std::shared_ptr<const void>&, std::size_t bytes) const;
    /** Borrow exact index object payload, excluding control/admission allowance. */
    void visit_storage(const std::function<void(const void*, std::size_t)>&) const;

private:
    friend struct AdaptiveLeafBodyBuilder;
    friend class AdaptiveLeafBodyDelta;
    explicit AdaptiveLeafBodyUsage(NodePtr root) : root_(std::move(root)) {}
    NodePtr root_;
    PLIC::details::OwnerManifestPair owners_;
    PLIC::details::NativeOwnerReceipts paid_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> membership_;
    PLIC::details::GeometryNativeAdministration administration_;
};

/** Page-local additions and retirements accumulated during a private edit.
 * The same arena admits delta metadata, per-page census and new index nodes. */
class AdaptiveLeafBodyDelta
{
public:
    struct Result
    {
        AdaptiveLeafBodyUsage usage;
        std::size_t new_payload_bytes = 0, reused_payload_bytes = 0;
    };

    explicit AdaptiveLeafBodyDelta(std::shared_ptr<Budget> arena);
    ~AdaptiveLeafBodyDelta();
    AdaptiveLeafBodyDelta(const AdaptiveLeafBodyDelta&) = delete;
    AdaptiveLeafBodyDelta& operator=(const AdaptiveLeafBodyDelta&) = delete;

    void record(std::span<const AdaptiveLeaf>, bool inserted);
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charged_; }
    /** New payload is relative to the accepted owner union if supplied. */
    [[nodiscard]] Result apply(const AdaptiveLeafBodyUsage&, AdaptiveLeafBodyRole = AdaptiveLeafBodyRole::Leaf,
        const AdaptiveLeafBodyUsage* accepted = nullptr) const;

private:
    friend struct AdaptiveLeafBodyBuilder;
    friend class AdaptiveLeafBodyUsage;
    struct Key
    {
        std::uint8_t kind;
        const void* address;
        [[nodiscard]] bool operator<(const Key&) const noexcept;
    };
    struct Counts
    {
        std::size_t payload, inserted = 0, retired = 0;
        std::weak_ptr<const void> control;
    };

    std::shared_ptr<Budget> arena_;
    std::map<Key, Counts> changes_;
    std::vector<std::pair<std::span<const AdaptiveLeaf>, bool>> projections_;
    std::size_t charged_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
