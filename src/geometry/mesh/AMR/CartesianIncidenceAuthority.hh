/** @file CartesianIncidenceAuthority.hh
 * @brief Branch-local complete-group witnesses issued by real incidence kernels.
 */
#pragma once
#include "geometry/mesh/AMR/CartesianIncidenceSource.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include <memory>
#include <span>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class CartesianGroupAuthority;
class CartesianCompleteSupport;
class CartesianCompleteLine;
class CartesianReplayCertificate;
using CartesianLineQueryEmit = CartesianBorrowedCallback<std::size_t, const PLIC::OrdinaryEdgeQueryScratch&>;
namespace cartesian_detail
{
struct AuthorityState;
struct ArenaCredits;
struct SupportSeal;
struct LineSeal;
struct LoopSeal;
} // namespace cartesian_detail
using CartesianAuthorityArena = cartesian_detail::ArenaCredits;

/** An immutable explicit fallback occurrence inventory, including exact original
 * packets and its independent revision/loop generation. It is never procedural. */
class CartesianExplicitSupportSource
{
public:
    static std::shared_ptr<const CartesianExplicitSupportSource> make(const AdaptiveLeaf&, int owner,
        std::uint64_t version, std::uint64_t loop_generation, Budget&, std::shared_ptr<Budget> keeper = {},
        ContactHaloContext context = {});
    ~CartesianExplicitSupportSource();
    const AdaptiveBytes& identity() const noexcept;
    const ContactHaloContext& context() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    CartesianExplicitSupportSource() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
};
/** Immutable explicit subscriptions from the existing fallback protocol. Full
 * endpoints, key and supplied original identity are bound to both generations. */
class CartesianExplicitLineSource
{
public:
    static std::shared_ptr<const CartesianExplicitLineSource> make(std::span<const char> original_identity,
        std::span<const char> line, std::span<const LineSubscription>, int owner, std::uint64_t revision,
        std::uint64_t loop_generation, Budget&, std::shared_ptr<Budget> keeper = {}, std::uint64_t arrival = UINT64_MAX,
        ContactHaloContext context = {});
    ~CartesianExplicitLineSource();
    const AdaptiveBytes& identity() const noexcept;
    const ContactHaloContext& context() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    CartesianExplicitLineSource() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
};
/** Opaque authenticated completion. Only the authority's actual contributor
 * enumeration can issue it. Membership arrays are bounded construction scratch. */
class CartesianCompletionReceipt
{
public:
    ~CartesianCompletionReceipt();
    std::size_t member_count() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    CartesianCompletionReceipt() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
};
class CartesianCompletionBatch
{
public:
    ~CartesianCompletionBatch();
    std::span<const std::shared_ptr<const CartesianCompletionReceipt>> receipts() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    CartesianCompletionBatch() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
};
/** A compact handle to a REAL support output. It carries the exact immutable
 * contributor closure, support generation, emission, layer and loop generation.
 * No caller can construct a handle from an arbitrary polygon. */
class CartesianSealedTile
{
public:
    std::size_t emission() const noexcept { return emission_; }
    unsigned layer() const noexcept { return layer_; }
    std::uint64_t support_generation() const noexcept;
    std::uint64_t loop_generation() const noexcept;
    bool stores_explicit_loop() const noexcept;

private:
    std::shared_ptr<const cartesian_detail::SupportSeal> support_;
    std::shared_ptr<const cartesian_detail::LoopSeal> loop_;
    std::size_t emission_ = 0;
    unsigned layer_ = 0;
    CartesianSealedTile() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
    friend class CartesianCompleteSupport;
    friend class CartesianReplayCertificate;
    friend struct cartesian_detail::AuthorityState;
};
struct CartesianTileUse
{
    CartesianSealedTile tile;
    int resident = -1;
    std::uint64_t arrival = 0; ///< Authenticated legacy subscription-arrival order.
};
using CartesianSealedTileEmit = CartesianBorrowedCallback<const CartesianSealedTile&, const PrivateTileRecipe&>;

/** Sealed output token, issued only after authenticated completeness and an
 * actual make_support_recipes execution. It retains no member/output arrays. */
class CartesianCompleteSupport
{
public:
    std::size_t tile_count() const noexcept;
    void visit_tiles(Budget&, const CartesianSealedTileEmit&) const;

private:
    std::shared_ptr<const cartesian_detail::SupportSeal> seal_;
    CartesianCompleteSupport() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
    friend class CartesianReplayCertificate;
};
/** Sealed output token, issued only after real query_complete_line execution.
 * Complete subscriptions are regenerated from controlled tile/explicit snapshots. */
class CartesianCompleteLine
{
public:
    std::size_t subscription_count() const noexcept;
    void visit_queries(Budget&, const CartesianLineQueryEmit&) const;

private:
    std::shared_ptr<const cartesian_detail::LineSeal> seal_;
    CartesianCompleteLine() = default;
    friend class CartesianGroupAuthority;
    friend struct cartesian_detail::AuthorityState;
    friend class CartesianReplayCertificate;
};

/** Mutable authority state belongs to one transaction branch. Replacements are
 * atomic, invalidate controlled generations, and include explicit contributors.
 * Fork shares immutable contributor/recipe snapshots, copies private version maps
 * and mints local bindings after exact snapshot/context/version equality. Only
 * authenticated ancestor witnesses can rebind; unrelated handles still reject. */
class CartesianGroupAuthority
{
public:
    /** One explicit arena per shared Budget domain. A default make creates a
     * dedicated domain; independent authorities sharing a Budget must pass the
     * same arena. Foreign-budget forks/outputs require an explicit arena. */
    static std::shared_ptr<CartesianAuthorityArena> make_arena(Budget&, std::shared_ptr<Budget> keeper = {});
    static std::unique_ptr<CartesianGroupAuthority> make(ContactHaloContext, Budget&,
        std::shared_ptr<Budget> keeper = {}, std::size_t tile_budget = AdaptiveClosureOptions{}.tile_budget,
        std::shared_ptr<CartesianAuthorityArena> arena = {});
    ~CartesianGroupAuthority();
    std::unique_ptr<CartesianGroupAuthority> fork(
        Budget&, std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    /** Controlled cache reuse for a transaction/round advance. Geometry epoch,
     * topology and partition agreement must remain exact; new geometry uses a
     * fresh current inventory. Sources are never silently retagged. */
    std::unique_ptr<CartesianGroupAuthority> fork(ContactHaloContext next, Budget&, std::shared_ptr<Budget> keeper = {},
        std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    std::shared_ptr<const CartesianCompleteSupport> adopt(const CartesianCompleteSupport&, Budget&,
        std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    std::shared_ptr<const CartesianCompleteLine> adopt(const CartesianCompleteLine&, Budget&,
        std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    std::span<const std::shared_ptr<const CartesianIncidenceSource>> procedural_sources() const noexcept;
    void replace_support_contributors(std::span<const std::shared_ptr<const CartesianIncidenceSource>>,
        std::span<const std::shared_ptr<const CartesianExplicitSupportSource>>);
    void replace_line_contributors(std::span<const CartesianTileUse>,
        std::span<const std::shared_ptr<const CartesianExplicitLineSource>>, unsigned layer = 0);
    std::shared_ptr<const CartesianCompletionBatch> support_completions(std::span<const char> group, Budget&,
        std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    std::shared_ptr<const CartesianCompletionBatch> line_completions(std::span<const char> group, unsigned layer,
        Budget&, std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    std::shared_ptr<const CartesianCompleteSupport> complete_support(std::span<const char> group,
        std::span<const std::shared_ptr<const CartesianCompletionReceipt>>, Budget&,
        std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    std::shared_ptr<const CartesianCompleteLine> complete_line(std::span<const char> group, unsigned layer,
        std::span<const std::shared_ptr<const CartesianCompletionReceipt>>, Budget&,
        std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;
    /** Every edge reply must come from a current sealed complete-line result;
     * this executes the unchanged with_edge_vertices enrichment in legacy order.
     * Returned nontrivial loop is an explicit immutable exception, never a proof
     * of an arithmetic neighbor recipe. Layer 63 cannot be advanced. */
    struct EdgeReply
    {
        std::shared_ptr<const CartesianCompleteLine> line;
        std::size_t subscription = 0;
    };
    CartesianSealedTile advance_tile(const CartesianSealedTile&, std::span<const EdgeReply>, unsigned authority_ranks,
        Budget&, std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> arena = {}) const;

private:
    std::shared_ptr<cartesian_detail::AuthorityState> state_;
    CartesianGroupAuthority() = default;
    friend class CartesianReplayCertificate;
};

/** Only sealed results can issue certificates. A matching record holds compact
 * recipe selectors and a controlled witness; a mismatch holds actual identities
 * as an explicit exception. Raw manifests or caller output bytes cannot certify.
 * Replay requires the current authority branch and validates its witness. */
class CartesianReplayCertificate
{
public:
    static std::shared_ptr<const CartesianReplayCertificate> from_support(const CartesianCompleteSupport&,
        const CartesianReplayRecipe&, Budget&, std::shared_ptr<Budget> keeper = {},
        std::shared_ptr<CartesianAuthorityArena> arena = {});
    static std::shared_ptr<const CartesianReplayCertificate> from_line(const CartesianCompleteLine&,
        std::size_t subscription, const CartesianReplayRecipe&, Budget&, std::shared_ptr<Budget> keeper = {},
        std::shared_ptr<CartesianAuthorityArena> arena = {});
    ~CartesianReplayCertificate();
    bool implicit() const noexcept;
    void replay(const CartesianGroupAuthority&, Budget&, const CartesianIdentityEmit&) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    CartesianReplayCertificate() = default;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
