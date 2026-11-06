/** @file AdaptiveLeafBodyUsage.hh
 * @brief Persistent physical allocation usage for immutable AMR leaf pages.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"

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
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t leaf_payload_bytes() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    friend struct AdaptiveLeafBodyBuilder;
    friend class AdaptiveLeafBodyDelta;
    explicit AdaptiveLeafBodyUsage(NodePtr root) : root_(std::move(root)) {}
    NodePtr root_;
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
    };

    std::shared_ptr<Budget> arena_;
    std::map<Key, Counts> changes_;
    std::size_t charged_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
