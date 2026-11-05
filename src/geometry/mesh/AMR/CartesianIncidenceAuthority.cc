/** @file CartesianIncidenceAuthority.cc
 * @brief Controlled immutable contributors, real complete kernels and branch witnesses.
 */
#include "geometry/mesh/AMR/CartesianIncidenceAuthority.hh"
#include "geometry/mesh/AMR/CartesianAuthorityCredits.hh"
#include "geometry/mesh/AMR/CartesianIncidenceBudget.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace cartesian_detail
{
constexpr std::size_t owner_bytes = 128;
struct Owned
{
    Budget* budget = nullptr;
    std::size_t charge = 0;
    std::shared_ptr<Budget> keeper;
    std::vector<std::shared_ptr<AuthorityOwnerCredit>> leases;
    ~Owned()
    {
        if (budget)
            budget->used -= charge;
    }
    void publish(Allocation& allocation, std::shared_ptr<Budget> owner)
    {
        budget = &allocation.budget;
        keeper = std::move(owner);
        charge = allocation.bytes;
        allocation.bytes = 0;
    }
};
struct BranchOrigin : Owned
{
    ContactHaloContext context;
};
/** Cached reconstruction restores caller controls/flags. A callback executes in
 * that caller environment and its resulting flags survive the reconstruction. */
struct Evaluation
{
    std::fenv_t caller{};
    explicit Evaluation(const std::fenv_t& captured)
    {
        if (std::fegetenv(&caller))
            throw std::invalid_argument("Cannot capture Cartesian caller environment.");
        if (std::fesetenv(&captured))
        {
            (void) std::fesetenv(&caller);
            throw std::invalid_argument("Cannot install Cartesian certified evaluation environment.");
        }
    }
    ~Evaluation() { (void) std::fesetenv(&caller); }
    template<class F> void callback(F&& function)
    {
        std::fenv_t active{};
        if (std::fegetenv(&active) || std::fesetenv(&caller))
            throw std::invalid_argument("Cannot restore Cartesian callback environment.");
        auto restore = [&]
        {
            (void) std::fegetenv(&caller);
            (void) std::fesetenv(&active);
        };
        try
        {
            function();
        }
        catch (...)
        {
            restore();
            throw;
        }
        restore();
    }
};
struct ExplicitFace
{
    AdaptiveBytes group, packet;
    std::size_t vertices = 0, whole_vertices = 0;
};
struct ExplicitSupportData : Owned
{
    ContactHaloContext context;
    AdaptiveBytes identity;
    AdaptiveCellKey cell;
    int owner = -1;
    std::uint64_t revision = 0, loop = 0;
    std::vector<ExplicitFace> faces;
};
struct ExplicitLineData : Owned
{
    ContactHaloContext context;
    AdaptiveBytes identity, line;
    std::vector<LineSubscription> subscriptions;
    int owner = -1;
    std::uint64_t revision = 0, loop = 0, arrival = UINT64_MAX;
};
struct SupportRegistry : Owned
{
    std::vector<std::shared_ptr<const CartesianIncidenceSource>> sources;
    std::vector<std::shared_ptr<const CartesianExplicitSupportSource>> explicit_sources;
};
struct LineRegistry : Owned
{
    std::vector<CartesianTileUse> tiles;
    std::vector<std::shared_ptr<const CartesianExplicitLineSource>> explicit_sources;
};
struct AuthorityState;
struct SupportSeal : Owned
{
    std::weak_ptr<AuthorityState> authority;
    std::shared_ptr<const BranchOrigin> origin;
    std::shared_ptr<const SupportRegistry> registry;
    AdaptiveBytes group, domain;
    std::uint64_t generation = 0;
    std::size_t tile_count = 0, tile_budget = AdaptiveClosureOptions{}.tile_budget;
    std::fenv_t environment{};
    CartesianEvaluationEnvironment controls;
};
struct LineSeal : Owned
{
    std::weak_ptr<AuthorityState> authority;
    std::shared_ptr<const BranchOrigin> origin;
    std::shared_ptr<const LineRegistry> registry;
    AdaptiveBytes group;
    std::uint64_t generation = 0;
    unsigned layer = 0;
    std::size_t subscriptions = 0;
    std::array<std::uint64_t, 64> layer_generations{};
    std::fenv_t environment{};
    CartesianEvaluationEnvironment controls;
};
struct LoopSeal : Owned
{
    std::vector<LinePoint> points;
    std::shared_ptr<const LoopSeal> previous;
    bool identity = false;
    std::vector<std::shared_ptr<const LineSeal>> lines;
    std::uint64_t generation = 0;
};
struct ReceiptData : Owned
{
    std::weak_ptr<AuthorityState> authority;
    std::shared_ptr<const BranchOrigin> origin;
    AdaptiveBytes group;
    std::uint64_t generation = 0;
    unsigned layer = 0;
    std::size_t contributor = 0;
    bool line = false;
    CartesianEvaluationEnvironment controls;
    std::vector<AdaptiveBytes> members;
};
void capture(std::fenv_t& environment)
{
    if (std::fegetenv(&environment))
        throw std::invalid_argument("Cannot capture Cartesian complete-group environment.");
}
void copy_membership(AdaptiveBytes& result, std::span<const char> source, std::size_t ordinal, Allocation& allocation)
{
    Writer writer;
    allocation.reserve(writer.data, add(source.size(), 24));
    writer.bytes(source);
    writer.word(ordinal);
    writer.word(0x434f4e5452494232ULL);
    result = std::move(writer.data);
}
std::size_t decode_bound(const ExplicitFace& face)
{
    // Decode validates/re-encodes the complete packet, creates its support,
    // reconstructs one polygon, and translates both native nesting images.
    auto size = product(face.packet.size(), 20);
    size =
        add(size, product(4, PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(face.vertices)));
    if (face.whole_vertices)
        size = add(size,
            product(4, PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(face.whole_vertices)));
    return size;
}
/** Pairwise convex clipping keeps at most n+m vertices at every clipping stage.
 * Admit every vector growth history, pair output, validation set, reversed/
 * translated face and duplicate-audit encoding before entering legacy code. */
std::size_t overlay_bound(std::span<const SupportOccurrence> occurrences)
{
    std::size_t vertices = 0, encoded = 0;
    for (const auto& entry : occurrences)
    {
        vertices = std::max(vertices, entry.face->points().size());
        encoded = std::max(encoded, entry.face->encoded_incidence_size());
    }
    const auto polygon = product(2, vertices);
    auto per_pair = product(8, PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(polygon));
    per_pair = add(per_pair, product(add(vertices, 4), growth(polygon, 3 * sizeof(long double))));
    per_pair = add(per_pair, product(16, encoded));
    per_pair = add(per_pair, product(polygon, 256));
    return product(product(occurrences.size(), occurrences.size()), per_pair);
}
struct SupportInput
{
    SupportOccurrence occurrence;
    std::size_t ordinal = 0;
};
using MemberEmit = CartesianBorrowedCallback<std::size_t, std::span<const char>>;
struct ForkBinding
{
    std::shared_ptr<const BranchOrigin> origin;
    std::shared_ptr<const SupportRegistry> support;
    std::uint64_t support_generation = 0;
    std::array<std::shared_ptr<const LineRegistry>, 64> lines;
    std::array<std::uint64_t, 64> line_generations{};
};
struct AuthorityState : Owned, std::enable_shared_from_this<AuthorityState>
{
    ContactHaloContext context;
    std::shared_ptr<ArenaCredits> credits;
    std::size_t tile_budget = AdaptiveClosureOptions{}.tile_budget;
    std::shared_ptr<const BranchOrigin> origin;
    std::vector<ForkBinding> inherited;
    std::uint64_t support_generation = 1, loop_generation = 0;
    std::array<std::uint64_t, 64> line_generations{};
    std::shared_ptr<const SupportRegistry> support;
    std::array<std::shared_ptr<const LineRegistry>, 64> lines;
    std::optional<CartesianEvaluationEnvironment> support_controls;
    std::array<std::optional<CartesianEvaluationEnvironment>, 64> line_controls;
    AuthorityState() { line_generations.fill(1); }
    bool alive = true;
    bool inherited_context = false;
    void invalidate_lines()
    {
        for (auto generation : line_generations)
            if (generation == UINT64_MAX)
                throw std::invalid_argument("Cartesian line generation exhausted.");
        for (auto& generation : line_generations)
            ++generation;
        for (auto& registry : lines)
            registry.reset();
    }
    std::shared_ptr<ArenaCredits> arena_for(
        Budget& target, const std::shared_ptr<Budget>& keeper, std::shared_ptr<ArenaCredits> arena = {}) const
    {
        if (!arena && &target == budget)
            arena = credits;
        if (!arena || arena->budget != &target)
            throw std::invalid_argument("Cartesian borrowed output requires its explicit budget arena.");
        check_keeper(target, keeper);
        if (arena->keeper && arena->keeper.get() != &target)
            throw std::invalid_argument("Cartesian credit arena keeper belongs to a different budget.");
        if (&target != budget && (!keeper || !arena->keeper))
            throw std::invalid_argument("Foreign Cartesian credit arena requires its lifetime keeper.");
        return arena;
    }
    void credit(CreditConstruction&, const std::shared_ptr<const CartesianIncidenceSource>&, Owned&, Allocation&) const;
    void credit(
        CreditConstruction&, const std::shared_ptr<const CartesianExplicitSupportSource>&, Owned&, Allocation&) const;
    void credit(
        CreditConstruction&, const std::shared_ptr<const CartesianExplicitLineSource>&, Owned&, Allocation&) const;
    void credit(CreditConstruction&, const std::shared_ptr<const SupportRegistry>&, Owned&, Allocation&) const;
    void credit(CreditConstruction&, const std::shared_ptr<const LineRegistry>&, Owned&, Allocation&) const;
    void credit(CreditConstruction&, const std::shared_ptr<const SupportSeal>&, Owned&, Allocation&) const;
    void credit(CreditConstruction&, const std::shared_ptr<const LineSeal>&, Owned&, Allocation&) const;
    void credit(CreditConstruction&, const std::shared_ptr<const LoopSeal>&, Owned&, Allocation&) const;
    template<class T>
    bool credit_owner(CreditConstruction& census, const std::shared_ptr<const T>& owner, Owned& destination,
        Allocation& allocation) const
    {
        if (!owner ||
            !census.retain(owner, owner->budget, owner->charge, bool(owner->keeper), destination.leases, allocation))
            return false;
        for (const auto& lease : owner->leases)
            census.retain(std::shared_ptr<const AuthorityOwnerCredit>(lease), lease->budget, lease->metadata,
                bool(lease->keeper), destination.leases, allocation);
        return true;
    }
    void prune_bindings()
    {
        for (auto& binding : inherited)
            for (unsigned layer = 0; layer < 64; ++layer)
            {
                bool current = binding.lines[layer] == lines[layer];
                for (unsigned dependency = 0; current && dependency <= layer; ++dependency)
                    current = binding.line_generations[dependency] == line_generations[dependency];
                if (!current)
                    binding.lines[layer].reset();
            }
        std::erase_if(inherited,
            [&](const auto& binding)
            {
                const bool support_witness =
                    binding.support && binding.support == support && binding.support_generation == support_generation;
                return binding.origin.use_count() == 1 ||
                       (!support_witness &&
                           std::ranges::none_of(binding.lines, [](const auto& line) { return bool(line); }));
            });
    }
    void refresh(bool line_query, unsigned layer = 0);
    bool bound(const SupportSeal& seal) const
    {
        if (seal.origin == origin)
            return true;
        return std::ranges::any_of(inherited,
            [&](const auto& binding)
            {
                return binding.origin == seal.origin && binding.support == seal.registry &&
                       binding.support_generation == seal.generation;
            });
    }
    bool bound(const LineSeal& seal) const
    {
        if (seal.origin == origin)
            return true;
        return std::ranges::any_of(inherited,
            [&](const auto& binding)
            {
                if (binding.origin != seal.origin || binding.lines[seal.layer] != seal.registry)
                    return false;
                for (unsigned layer = 0; layer <= seal.layer; ++layer)
                    if (binding.line_generations[layer] != seal.layer_generations[layer])
                        return false;
                return true;
            });
    }
    void check(const SupportSeal& seal) const
    {
        if (!alive || !bound(seal) || seal.generation != support_generation || seal.registry != support)
            throw std::invalid_argument("Stale or foreign sealed Cartesian support witness.");
    }
    void check(const LineSeal& seal) const
    {
        if (!alive || seal.layer >= 64 || !bound(seal) || seal.generation != line_generations[seal.layer] ||
            seal.registry != lines[seal.layer])
            throw std::invalid_argument("Stale or foreign sealed Cartesian line witness.");
        for (unsigned layer = 0; layer <= seal.layer; ++layer)
            if (seal.layer_generations[layer] != line_generations[layer])
                throw std::invalid_argument("Stale Cartesian layered line dependency.");
    }
    void check(const CartesianSealedTile& tile) const
    {
        if (!tile.support_ || tile.layer_ >= 64)
            throw std::invalid_argument("Incomplete Cartesian sealed tile.");
        check(*tile.support_);
        if (tile.loop_)
            for (const auto& dependency : tile.loop_->lines)
                check(*dependency);
    }
    std::size_t support_contributors() const
    {
        return support ? support->sources.size() + support->explicit_sources.size() : 0;
    }
    std::size_t line_contributors(unsigned layer) const
    {
        return lines[layer] ? lines[layer]->tiles.size() + lines[layer]->explicit_sources.size() : 0;
    }
    void support_members(std::span<const char> group, Budget& budget, const MemberEmit& emit,
        const CartesianBorrowedCallback<const SupportInput&>& occurrence = {}) const;
    void support_run(const SupportSeal&, Budget&, const CartesianBorrowedCallback<const PrivateTileRecipe&>&) const;
    void tile_run(
        const CartesianSealedTile&, Budget&, const CartesianBorrowedCallback<const PrivateTileRecipe&>&) const;
    void line_members(std::span<const char> group, unsigned layer, Budget&, const MemberEmit&,
        const CartesianBorrowedCallback<const LineSubscription&>& = {}) const;
    void line_run(const LineSeal&, Budget&, const CartesianLineQueryEmit&) const;
    void authenticate(bool lines, std::span<const char> group, unsigned layer,
        std::span<const std::shared_ptr<const CartesianCompletionReceipt>> received, Budget&) const;
};
} // namespace cartesian_detail
using namespace cartesian_detail;
struct CartesianExplicitSupportSource::Impl : ExplicitSupportData
{
};
struct CartesianExplicitLineSource::Impl : ExplicitLineData
{
};
struct CartesianCompletionReceipt::Impl : ReceiptData
{
};
struct CartesianCompletionBatch::Impl : Owned
{
    std::vector<std::shared_ptr<const CartesianCompletionReceipt>> receipts;
};
void AuthorityState::credit(CreditConstruction& census, const std::shared_ptr<const CartesianIncidenceSource>& source,
    Owned& target, Allocation& memory) const
{
    if (source)
        census.retain(source, source->allocation_budget(), source->metadata_bytes(), source->has_arena_keeper(),
            target.leases, memory);
}
void AuthorityState::credit(CreditConstruction& census,
    const std::shared_ptr<const CartesianExplicitSupportSource>& source, Owned& target, Allocation& memory) const
{
    if (source)
        credit_owner(census, source->impl_, target, memory);
}
void AuthorityState::credit(CreditConstruction& census,
    const std::shared_ptr<const CartesianExplicitLineSource>& source, Owned& target, Allocation& memory) const
{
    if (source)
        credit_owner(census, source->impl_, target, memory);
}
void AuthorityState::credit(CreditConstruction& census, const std::shared_ptr<const SupportRegistry>& registry,
    Owned& target, Allocation& memory) const
{
    if (!credit_owner(census, registry, target, memory))
        return;
    for (const auto& source : registry->sources)
        credit(census, source, target, memory);
    for (const auto& source : registry->explicit_sources)
        credit(census, source, target, memory);
}
void AuthorityState::credit(CreditConstruction& census, const std::shared_ptr<const LineRegistry>& registry,
    Owned& target, Allocation& memory) const
{
    if (!credit_owner(census, registry, target, memory))
        return;
    for (const auto& use : registry->tiles)
    {
        credit(census, use.tile.support_, target, memory);
        credit(census, use.tile.loop_, target, memory);
    }
    for (const auto& source : registry->explicit_sources)
        credit(census, source, target, memory);
}
void AuthorityState::credit(
    CreditConstruction& census, const std::shared_ptr<const SupportSeal>& seal, Owned& target, Allocation& memory) const
{
    if (!credit_owner(census, seal, target, memory))
        return;
    credit_owner(census, seal->origin, target, memory);
    credit(census, seal->registry, target, memory);
}
void AuthorityState::credit(
    CreditConstruction& census, const std::shared_ptr<const LineSeal>& seal, Owned& target, Allocation& memory) const
{
    if (!credit_owner(census, seal, target, memory))
        return;
    credit_owner(census, seal->origin, target, memory);
    credit(census, seal->registry, target, memory);
}
void AuthorityState::credit(
    CreditConstruction& census, const std::shared_ptr<const LoopSeal>& loop, Owned& target, Allocation& memory) const
{
    if (!credit_owner(census, loop, target, memory))
        return;
    credit(census, loop->previous, target, memory);
    for (const auto& line : loop->lines)
        credit(census, line, target, memory);
}
void AuthorityState::refresh(bool line_query, unsigned layer)
{
    if (layer >= 64)
        throw std::invalid_argument("Cartesian query layer exceeds 64-layer contract.");
    if ((!line_query && !support) || (line_query && !lines[layer]))
        throw std::invalid_argument("Missing Cartesian complete-group contributor registry.");
    auto& controls = line_query ? line_controls[layer] : support_controls;
    const auto current = cartesian_evaluation_environment();
    const bool source_change = !line_query && support &&
                               std::ranges::any_of(support->sources,
                                   [&](const auto& source) { return source->evaluation_environment() != current; });
    if ((controls && *controls != current) || source_change)
    {
        if (line_query)
        {
            if (line_generations[layer] == UINT64_MAX)
                throw std::invalid_argument("Cartesian generation exhausted.");
            ++line_generations[layer];
            prune_bindings();
        }
        else
        {
            if (support_generation == UINT64_MAX)
                throw std::invalid_argument("Cartesian generation exhausted.");
            Allocation memory(*budget);
            memory.retain(sizeof(SupportRegistry) + owner_bytes);
            auto refreshed = std::make_shared<SupportRegistry>();
            memory.reserve(refreshed->sources, support->sources.size());
            memory.reserve(refreshed->explicit_sources, support->explicit_sources.size());
            for (const auto& source : support->sources)
            {
                auto next = source->evaluation_environment() != current ? source->reevaluated(*budget, keeper) : source;
                refreshed->sources.push_back(std::move(next));
            }
            for (const auto& source : support->explicit_sources)
            {
                refreshed->explicit_sources.push_back(source);
            }
            // Admission failure leaves both registry and all generation maps intact.
            for (auto generation : line_generations)
                if (generation == UINT64_MAX)
                    throw std::invalid_argument("Cartesian line generation exhausted.");
            CreditConstruction census(credits);
            for (const auto& source : refreshed->sources)
                credit(census, source, *refreshed, memory);
            for (const auto& source : refreshed->explicit_sources)
                credit(census, source, *refreshed, memory);
            refreshed->publish(memory, keeper);
            census.committed = true;
            invalidate_lines();
            support = std::move(refreshed);
            ++support_generation;
            inherited.clear();
            leases.clear();
            credits->reap();
        }
    }
    controls = current;
}
CartesianExplicitSupportSource::~CartesianExplicitSupportSource() = default;
CartesianExplicitLineSource::~CartesianExplicitLineSource() = default;
CartesianCompletionReceipt::~CartesianCompletionReceipt() = default;
CartesianCompletionBatch::~CartesianCompletionBatch() = default;
const AdaptiveBytes& CartesianExplicitSupportSource::identity() const noexcept
{
    return impl_->identity;
}
const ContactHaloContext& CartesianExplicitSupportSource::context() const noexcept
{
    return impl_->context;
}
const AdaptiveBytes& CartesianExplicitLineSource::identity() const noexcept
{
    return impl_->identity;
}
const ContactHaloContext& CartesianExplicitLineSource::context() const noexcept
{
    return impl_->context;
}
std::size_t CartesianCompletionReceipt::member_count() const noexcept
{
    return impl_->members.size();
}
std::span<const std::shared_ptr<const CartesianCompletionReceipt>> CartesianCompletionBatch::receipts() const noexcept
{
    return impl_->receipts;
}
std::shared_ptr<const CartesianExplicitSupportSource> CartesianExplicitSupportSource::make(const AdaptiveLeaf& leaf,
    int owner, std::uint64_t version, std::uint64_t loop, Budget& budget, std::shared_ptr<Budget> keeper,
    ContactHaloContext context)
{
    check_keeper(budget, keeper);
    if (owner < 0 || leaf.domain.empty() || leaf.surfaces.empty())
        throw std::invalid_argument("Incomplete explicit Cartesian support contributor.");
    Allocation retained(budget);
    retained.retain(sizeof(CartesianExplicitSupportSource) + sizeof(Impl) + 2 * owner_bytes + key_capacity(leaf.key));
    auto result = std::shared_ptr<CartesianExplicitSupportSource>(new CartesianExplicitSupportSource);
    auto body = std::make_shared<Impl>();
    body->cell = leaf.key;
    body->owner = owner;
    body->revision = version;
    body->loop = loop;
    body->context = context;
    retained.reserve(body->faces, leaf.surfaces.size());
    std::size_t identity_size = 88;
    for (std::size_t f = 0; f < leaf.surfaces.size(); ++f)
    {
        Allocation encoding(budget);
        auto packet = surface_packet(leaf, f, encoding);
        body->faces.emplace_back();
        auto& face = body->faces.back();
        retained.copy(face.packet, packet);
        retained.copy(face.group, leaf.surfaces[f].key);
        face.vertices = leaf.surfaces[f].geometry->points().size();
        if (leaf.surfaces[f].native_whole_geometry != leaf.surfaces[f].geometry &&
            leaf.surfaces[f].native_whole_geometry)
            face.whole_vertices = leaf.surfaces[f].native_whole_geometry->points().size();
        identity_size = add(identity_size, add(face.packet.size(), 8));
    }
    Writer identity;
    retained.reserve(identity.data, identity_size);
    identity.word(0x4558505355505033ULL);
    for (auto value : {context.epoch, context.topology, context.partition, context.round, context.transaction})
        identity.word(value);
    identity.word(owner);
    identity.word(version);
    identity.word(loop);
    identity.word(leaf.revision);
    identity.word(body->faces.size());
    for (const auto& face : body->faces)
        identity.bytes(face.packet);
    body->identity = std::move(identity.data);
    body->publish(retained, std::move(keeper));
    result->impl_ = std::move(body);
    return result;
}
std::shared_ptr<const CartesianExplicitLineSource> CartesianExplicitLineSource::make(std::span<const char> original,
    std::span<const char> line, std::span<const LineSubscription> subscriptions, int owner, std::uint64_t revision,
    std::uint64_t loop, Budget& budget, std::shared_ptr<Budget> keeper, std::uint64_t arrival,
    ContactHaloContext context)
{
    check_keeper(budget, keeper);
    if (original.empty() || line.empty() || owner < 0)
        throw std::invalid_argument("Incomplete explicit Cartesian line contributor.");
    Allocation retained(budget);
    retained.retain(sizeof(CartesianExplicitLineSource) + sizeof(Impl) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianExplicitLineSource>(new CartesianExplicitLineSource);
    auto body = std::make_shared<Impl>();
    body->owner = owner;
    body->revision = revision;
    body->loop = loop;
    body->arrival = arrival;
    body->context = context;
    retained.copy(body->line, line);
    retained.reserve(body->subscriptions, subscriptions.size());
    body->subscriptions.assign(subscriptions.begin(), subscriptions.end());
    std::size_t identity_size = add(add(original.size(), line.size()), 104);
    for (const auto& sub : subscriptions)
        identity_size =
            add(identity_size, add(add(PLIC::canonical_key_size(sub.first), PLIC::canonical_key_size(sub.last)), 16));
    Writer identity;
    retained.reserve(identity.data, identity_size);
    identity.word(0x4558504c494e4533ULL);
    for (auto value : {context.epoch, context.topology, context.partition, context.round, context.transaction})
        identity.word(value);
    identity.word(owner);
    identity.word(revision);
    identity.word(loop);
    identity.word(arrival);
    identity.bytes(original);
    identity.bytes(line);
    identity.word(subscriptions.size());
    for (const auto& sub : subscriptions)
    {
        Allocation encoding(budget);
        identity.bytes(identity_key(sub.first, encoding));
        identity.bytes(identity_key(sub.last, encoding));
    }
    if (identity.data.size() != identity_size)
        throw std::logic_error("Explicit line identity count differs.");
    body->identity = std::move(identity.data);
    body->publish(retained, std::move(keeper));
    result->impl_ = std::move(body);
    return result;
}
void AuthorityState::support_members(std::span<const char> group, Budget& budget, const MemberEmit& emit,
    const CartesianBorrowedCallback<const SupportInput&>& consume) const
{
    if (!support || group.empty())
        throw std::invalid_argument("Incomplete Cartesian support registry/query.");
    std::size_t contributor = 0;
    for (const auto& source : support->sources)
    {
        source->visit_surfaces(
            budget,
            [&](const auto& surface)
            {
                emit(contributor, surface.member);
                if (consume)
                {
                    Allocation decoding(budget);
                    ExplicitFace description;
                    description.vertices = surface.surface.geometry->points().size();
                    // Count packet and all decoder/transformation growth before allocation.
                    decoding.retain(add(product(surface.packet.size(), 20),
                        product(4, PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(
                                       description.vertices))));
                    const auto occurrence =
                        decode_support_surface(surface.packet, group, source->owner(), decoding.bytes);
                    consume({occurrence, surface.ordinal});
                }
            },
            group);
        ++contributor;
    }
    for (const auto& source : support->explicit_sources)
    {
        const auto& data = *source->impl_;
        for (std::size_t f = 0; f < data.faces.size(); ++f)
        {
            const auto& face = data.faces[f];
            if (!std::ranges::equal(group, face.group))
                continue;
            Allocation encoding(budget);
            AdaptiveBytes member;
            copy_membership(member, data.identity, f, encoding);
            emit(contributor, member);
            if (consume)
            {
                Allocation decoding(budget);
                decoding.retain(decode_bound(face));
                const auto occurrence = decode_support_surface(face.packet, group, data.owner, decoding.bytes);
                consume({occurrence, f});
            }
        }
        ++contributor;
    }
}
void AuthorityState::support_run(
    const SupportSeal& seal, Budget& budget, const CartesianBorrowedCallback<const PrivateTileRecipe&>& emit) const
{
    check(seal);
    Allocation input_memory(budget), work(budget);
    std::vector<SupportInput> input;
    // Inputs escaping the enumeration callback receive their own admitted
    // decoder/geometry owners before the callback's transient decoder releases.
    support_members(
        seal.group, budget, [](auto, auto) {},
        [&](const SupportInput& entry)
        {
            input_memory.append(input);
            input_memory.retain(add(key_capacity(entry.occurrence.cell),
                add(entry.occurrence.domain.capacity(), entry.occurrence.boundary.capacity())));
            IdentityBudget ledger;
            const auto before = budget.used;
            try
            {
                ledger.face(budget, *entry.occurrence.face);
                if (entry.occurrence.whole_geometry != entry.occurrence.face && entry.occurrence.whole_geometry)
                    ledger.face(budget, *entry.occurrence.whole_geometry);
            }
            catch (...)
            {
                input_memory.bytes = add(input_memory.bytes, budget.used - before);
                throw;
            }
            input_memory.bytes = add(input_memory.bytes, budget.used - before);
            input.push_back(entry);
        });
    std::sort(input.begin(), input.end(),
        [](const auto& a, const auto& b)
        {
            return std::tie(a.occurrence.rank, a.occurrence.cell, a.ordinal) <
                   std::tie(b.occurrence.rank, b.occurrence.cell, b.ordinal);
        });
    std::vector<SupportOccurrence> occurrences;
    input_memory.reserve(occurrences, input.size());
    for (auto& entry : input)
        occurrences.push_back(std::move(entry.occurrence));
    work.retain(overlay_bound(occurrences));
    auto charge = [&](std::size_t bytes) { work.retain(bytes); };
    auto charge_face = [&](const SupportFace& face)
    {
        // Pairwise output/validation allocations were preflighted above; the
        // legacy callback still receives its separate retained accounting.
        (void) face;
    };
    auto publish = [&](const PrivateTileRecipe& value) { emit(value); };
    make_support_recipes(seal.group, occurrences, budget.limit, seal.tile_budget, charge, charge_face, publish);
}
void AuthorityState::tile_run(const CartesianSealedTile& tile, Budget& budget,
    const CartesianBorrowedCallback<const PrivateTileRecipe&>& emit) const
{
    check(tile);
    Evaluation evaluation(tile.support_->environment);
    std::size_t index = 0;
    auto output = [&](const PrivateTileRecipe& recipe)
    {
        if (index++ != tile.emission_)
            return;
        Allocation points_memory(budget);
        const LoopSeal* loop = tile.loop_.get();
        while (loop && loop->identity)
            loop = loop->previous.get();
        std::vector<LinePoint> intrinsic;
        if (!loop)
        {
            points_memory.retain(growth(recipe.face->points().size(), sizeof(LinePoint)));
            intrinsic = recipe.face->vertex_identities();
        }
        const std::span<const LinePoint> points = loop ? std::span(loop->points) : std::span(intrinsic);
        const auto frame = PLIC::capture_incidence_frame(*recipe.face);
        evaluation.callback(
            [&]
            {
                Allocation construction(budget);
                construction.retain(
                    PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(points.size()));
                auto physical = recipe;
                physical.face = PLIC::restore_incidence_loop(frame, points);
                emit(physical);
            });
    };
    support_run(*tile.support_, budget, output);
    if (index != tile.support_->tile_count || tile.emission_ >= index)
        throw std::invalid_argument("Sealed Cartesian tile emission changed.");
}
void AuthorityState::line_members(std::span<const char> group, unsigned layer, Budget& budget, const MemberEmit& emit,
    const CartesianBorrowedCallback<const LineSubscription&>& consume) const
{
    if (layer >= 64)
        throw std::invalid_argument("Invalid Cartesian line layer.");
    const auto& line = lines[layer];
    if (!line || group.empty())
        throw std::invalid_argument("Incomplete Cartesian line registry/query.");
    struct Record
    {
        int owner;
        std::uint64_t arrival;
        std::size_t contributor, edge;
        AdaptiveBytes member;
        LineSubscription sub;
    };
    Allocation memory(budget);
    std::vector<Record> ordered;
    for (std::size_t i = 0; i < line->tiles.size(); ++i)
    {
        const auto& use = line->tiles[i];
        check(use.tile);
        if (use.tile.layer_ != layer)
            continue;
        Allocation encoding(budget);
        std::size_t identity_size = add(use.tile.support_->group.size(), 72);
        for (const auto& source : use.tile.support_->registry->sources)
            identity_size = add(identity_size, add(source->identity().size(), 8));
        for (const auto& source : use.tile.support_->registry->explicit_sources)
            identity_size = add(identity_size, add(source->identity().size(), 8));
        Writer member;
        encoding.reserve(member.data, identity_size);
        member.word(0x5345414c54494c32ULL);
        member.bytes(use.tile.support_->group);
        member.word(use.tile.support_->generation);
        member.word(use.tile.emission_);
        member.word(layer);
        member.word(use.tile.loop_generation());
        member.word(use.resident);
        member.word(use.arrival);
        member.word(use.tile.support_->registry->sources.size() + use.tile.support_->registry->explicit_sources.size());
        for (const auto& source : use.tile.support_->registry->sources)
            member.bytes(source->identity());
        for (const auto& source : use.tile.support_->registry->explicit_sources)
            member.bytes(source->identity());
        auto tile_output = [&](const PrivateTileRecipe& tile)
        {
            visit_cartesian_tile_edges(
                use.tile.support_->domain, member.data, tile, budget,
                [&](const auto& edge)
                {
                    memory.append(ordered);
                    ordered.push_back({use.resident, use.arrival, i,
                        edge.image * tile.face->points().size() + edge.edge, {}, {edge.first, edge.last}});
                    memory.copy(ordered.back().member, edge.member);
                },
                group);
        };
        tile_run(use.tile, budget, tile_output);
    }
    for (std::size_t i = 0; i < line->explicit_sources.size(); ++i)
    {
        const auto& data = *line->explicit_sources[i]->impl_;
        if (!std::ranges::equal(group, data.line))
            continue;
        for (std::size_t j = 0; j < data.subscriptions.size(); ++j)
        {
            memory.append(ordered);
            ordered.push_back(
                {data.owner, data.arrival == UINT64_MAX ? std::uint64_t(line->tiles.size() + i) : data.arrival,
                    line->tiles.size() + i, j, {}, data.subscriptions[j]});
            copy_membership(ordered.back().member, data.identity, j, memory);
        }
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const auto& a, const auto& b)
        {
            return std::tie(a.owner, a.arrival, a.edge, a.contributor) <
                   std::tie(b.owner, b.arrival, b.edge, b.contributor);
        });
    for (const auto& record : ordered)
    {
        emit(record.contributor, record.member);
        if (consume)
            consume(record.sub);
    }
}
void AuthorityState::line_run(const LineSeal& seal, Budget& budget, const CartesianLineQueryEmit& emit) const
{
    check(seal);
    Allocation input_memory(budget), work(budget);
    std::vector<LineSubscription> input;
    line_members(
        seal.group, seal.layer, budget, [](auto, auto) {},
        [&](const auto& subscription)
        {
            input_memory.append(input);
            input.push_back(subscription);
        });
    auto charge = [&](std::size_t bytes) { work.retain(bytes); };
    // The existing empty-line path constructs one empty ordered index even
    // when its per-point envelope is zero. Its implementation is one std::map
    // plus the shared control block; admit that fixed owner before entry.
    work.retain(sizeof(PLIC::EdgePointIndex<ArithmeticMode::Ordinary>) +
                sizeof(std::map<std::array<long double, 3>, LinePoint>) + 2 * sizeof(void*));
    query_complete_line(input, charge, emit);
}
void AuthorityState::authenticate(bool lines, std::span<const char> group, unsigned layer,
    std::span<const std::shared_ptr<const CartesianCompletionReceipt>> received, Budget& budget) const
{
    if (lines && layer >= line_generations.size())
        throw std::invalid_argument("Cartesian query layer exceeds 64-layer contract.");
    const auto count = lines ? line_contributors(layer) : support_contributors();
    if (received.size() != count)
        throw std::invalid_argument("Missing complete Cartesian contributor receipt.");
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!received[i] || !received[i]->impl_)
            throw std::invalid_argument("Missing sealed Cartesian contributor receipt.");
        const auto& receipt = *received[i]->impl_;
        if (receipt.authority.lock().get() != this ||
            receipt.generation != (lines ? line_generations[layer] : support_generation) || receipt.contributor != i ||
            receipt.line != lines || receipt.layer != layer || receipt.controls != cartesian_evaluation_environment() ||
            !std::ranges::equal(receipt.group, group))
            throw std::invalid_argument("Foreign, duplicate or stale Cartesian completion receipt.");
    }
    Allocation positions_memory(budget);
    std::vector<std::size_t> positions;
    positions_memory.reserve(positions, count);
    positions.resize(count, 0);
    auto compare = [&](std::size_t contributor, std::span<const char> member)
    {
        auto& position = positions[contributor];
        const auto& expected = received[contributor]->impl_->members;
        if (position >= expected.size() || !std::ranges::equal(member, expected[position++]))
            throw std::invalid_argument("Changed full Cartesian contributor membership.");
    };
    if (lines)
        line_members(group, layer, budget, compare);
    else
        support_members(group, budget, compare);
    for (std::size_t i = 0; i < count; ++i)
        if (positions[i] != received[i]->impl_->members.size())
            throw std::invalid_argument("Incomplete Cartesian contributor membership, including empty completion.");
}
std::shared_ptr<CartesianAuthorityArena> CartesianGroupAuthority::make_arena(
    Budget& budget, std::shared_ptr<Budget> keeper)
{
    check_keeper(budget, keeper);
    Allocation memory(budget);
    memory.retain(sizeof(ArenaCredits) + owner_bytes);
    auto arena = std::make_shared<ArenaCredits>(budget, std::move(keeper));
    arena->charge = memory.bytes;
    memory.bytes = 0;
    return arena;
}
std::unique_ptr<CartesianGroupAuthority> CartesianGroupAuthority::make(ContactHaloContext context, Budget& budget,
    std::shared_ptr<Budget> keeper, std::size_t tile_budget, std::shared_ptr<CartesianAuthorityArena> arena)
{
    check_keeper(budget, keeper);
    if (!arena)
        arena = make_arena(budget, keeper);
    if (arena->budget != &budget || (arena->keeper && arena->keeper.get() != &budget))
        throw std::invalid_argument("Cartesian authority arena belongs to a different budget.");
    if (!keeper)
        keeper = arena->keeper;
    Allocation memory(budget);
    // State body and weak control are separate. The branch cookie owns the
    // control allowance until every sealed output/receipt weak handle expires.
    memory.retain(sizeof(CartesianGroupAuthority) + sizeof(AuthorityState));
    Allocation origin_memory(budget);
    origin_memory.retain(sizeof(BranchOrigin) + 2 * owner_bytes);
    auto result = std::unique_ptr<CartesianGroupAuthority>(new CartesianGroupAuthority);
    auto state = std::shared_ptr<AuthorityState>(new AuthorityState);
    state->credits = std::move(arena);
    state->tile_budget = tile_budget;
    state->context = context;
    auto origin = std::make_shared<BranchOrigin>();
    origin->context = context;
    origin->publish(origin_memory, keeper);
    state->origin = std::move(origin);
    state->publish(memory, std::move(keeper));
    result->state_ = std::move(state);
    return result;
}
CartesianGroupAuthority::~CartesianGroupAuthority()
{
    if (state_)
        state_->alive = false;
}
std::unique_ptr<CartesianGroupAuthority> CartesianGroupAuthority::fork(
    Budget& budget, std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    return fork(state_->context, budget, std::move(keeper), std::move(arena));
}
std::unique_ptr<CartesianGroupAuthority> CartesianGroupAuthority::fork(ContactHaloContext next, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    if (!state_->alive)
        throw std::invalid_argument("Retired Cartesian fork authority.");
    check_keeper(budget, keeper);
    if (&budget != state_->budget && !state_->keeper)
        throw std::invalid_argument("Cross-arena authority fork needs the source arena lifetime keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    arena = state_->arena_for(budget, keeper, std::move(arena));
    const auto& old = state_->context;
    if (next.epoch != old.epoch || next.topology != old.topology || next.partition != old.partition ||
        (next != old &&
            (next.transaction < old.transaction || (next.transaction == old.transaction && next.round <= old.round))))
        throw std::invalid_argument("Invalid geometry or transaction agreement for authenticated source fork.");
    auto result = make(next, budget, keeper, state_->tile_budget, arena);
    Allocation bindings(budget);
    CreditConstruction census(arena);
    auto& target = *result->state_;
    target.inherited_context = state_->inherited_context || next != state_->context;
    state_->credit(census, state_->support, target, bindings);
    for (const auto& line : state_->lines)
        state_->credit(census, line, target, bindings);
    for (const auto& binding : state_->inherited)
        if (binding.support == state_->support && binding.support_generation == state_->support_generation &&
            binding.origin.use_count() > 1)
        {
            state_->credit_owner(census, binding.origin, target, bindings);
            state_->credit(census, binding.support, target, bindings);
            for (const auto& line : binding.lines)
                state_->credit(census, line, target, bindings);
            bindings.append(target.inherited);
            target.inherited.push_back(binding);
        }
    if (state_->origin.use_count() > 1)
    {
        state_->credit_owner(census, state_->origin, target, bindings);
        bindings.append(target.inherited);
        target.inherited.emplace_back();
        auto& binding = target.inherited.back();
        binding.origin = state_->origin;
        binding.support = state_->support;
        binding.support_generation = state_->support_generation;
        binding.lines = state_->lines;
        binding.line_generations = state_->line_generations;
    }
    target.support = state_->support;
    target.lines = state_->lines;
    target.support_generation = state_->support_generation;
    target.line_generations = state_->line_generations;
    target.loop_generation = state_->loop_generation;
    target.support_controls = state_->support_controls;
    target.line_controls = state_->line_controls;
    target.charge = add(target.charge, bindings.bytes);
    bindings.bytes = 0;
    census.committed = true;
    return result;
}
std::shared_ptr<const CartesianCompleteSupport> CartesianGroupAuthority::adopt(const CartesianCompleteSupport& old,
    Budget& budget, std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    state_->check(*old.seal_);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena support adoption needs its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation memory(budget);
    memory.retain(sizeof(CartesianCompleteSupport) + sizeof(SupportSeal) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianCompleteSupport>(new CartesianCompleteSupport);
    auto seal = std::make_shared<SupportSeal>();
    seal->authority = state_;
    seal->origin = state_->origin;
    seal->registry = old.seal_->registry;
    state_->credit_owner(census, seal->origin, *seal, memory);
    state_->credit(census, seal->registry, *seal, memory);
    seal->generation = state_->support_generation;
    seal->tile_count = old.seal_->tile_count;
    seal->tile_budget = old.seal_->tile_budget;
    seal->environment = old.seal_->environment;
    seal->controls = old.seal_->controls;
    memory.copy(seal->group, old.seal_->group);
    memory.copy(seal->domain, old.seal_->domain);
    seal->publish(memory, std::move(keeper));
    census.committed = true;
    result->seal_ = std::move(seal);
    return result;
}
std::shared_ptr<const CartesianCompleteLine> CartesianGroupAuthority::adopt(const CartesianCompleteLine& old,
    Budget& budget, std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    state_->check(*old.seal_);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena line adoption needs its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation memory(budget);
    memory.retain(sizeof(CartesianCompleteLine) + sizeof(LineSeal) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianCompleteLine>(new CartesianCompleteLine);
    auto seal = std::make_shared<LineSeal>();
    seal->authority = state_;
    seal->origin = state_->origin;
    seal->registry = old.seal_->registry;
    state_->credit_owner(census, seal->origin, *seal, memory);
    state_->credit(census, seal->registry, *seal, memory);
    seal->layer = old.seal_->layer;
    seal->generation = state_->line_generations[seal->layer];
    seal->layer_generations = state_->line_generations;
    seal->subscriptions = old.seal_->subscriptions;
    seal->environment = old.seal_->environment;
    seal->controls = old.seal_->controls;
    memory.copy(seal->group, old.seal_->group);
    seal->publish(memory, std::move(keeper));
    census.committed = true;
    result->seal_ = std::move(seal);
    return result;
}
std::span<const std::shared_ptr<const CartesianIncidenceSource>>
CartesianGroupAuthority::procedural_sources() const noexcept
{
    return state_->support ? std::span(state_->support->sources)
                           : std::span<const std::shared_ptr<const CartesianIncidenceSource>>{};
}
void CartesianGroupAuthority::replace_support_contributors(
    std::span<const std::shared_ptr<const CartesianIncidenceSource>> sources,
    std::span<const std::shared_ptr<const CartesianExplicitSupportSource>> explicit_sources)
{
    if (!state_->alive || state_->support_generation == UINT64_MAX)
        throw std::invalid_argument("Cartesian authority generation exhausted.");
    Allocation memory(*state_->budget);
    memory.retain(sizeof(SupportRegistry) + owner_bytes);
    auto next = std::make_shared<SupportRegistry>();
    memory.reserve(next->sources, sources.size());
    memory.reserve(next->explicit_sources, explicit_sources.size());
    for (const auto& source : sources)
    {
        if (!source || std::ranges::find(next->sources, source) != next->sources.end())
            throw std::invalid_argument("Null or duplicate Cartesian procedural contributor.");
        if (source->context() != state_->context &&
            !(state_->inherited_context && state_->support &&
                std::ranges::find(state_->support->sources, source) != state_->support->sources.end()))
            throw std::invalid_argument("Stale Cartesian procedural contributor context.");
        if (source->allocation_budget() != state_->budget)
        {
            if (!source->has_arena_keeper())
                throw std::invalid_argument("Cross-arena Cartesian source has no lifetime keeper.");
        }
        next->sources.push_back(source);
    }
    for (const auto& source : explicit_sources)
    {
        if (!source || std::ranges::find(next->explicit_sources, source) != next->explicit_sources.end())
            throw std::invalid_argument("Null or duplicate Cartesian explicit contributor.");
        if (source->context() != state_->context && !(state_->inherited_context && state_->support &&
                                                        std::ranges::find(state_->support->explicit_sources, source) !=
                                                            state_->support->explicit_sources.end()))
            throw std::invalid_argument("Stale explicit support contributor context.");
        if (source->impl_->budget != state_->budget)
        {
            if (!source->impl_->keeper)
                throw std::invalid_argument("Cross-arena explicit contributor has no lifetime keeper.");
        }
        next->explicit_sources.push_back(source);
    }
    CreditConstruction census(state_->credits);
    for (const auto& source : next->sources)
        state_->credit(census, source, *next, memory);
    for (const auto& source : next->explicit_sources)
        state_->credit(census, source, *next, memory);
    next->publish(memory, state_->keeper);
    state_->invalidate_lines();
    state_->support = std::move(next);
    ++state_->support_generation;
    state_->inherited.clear();
    state_->leases.clear();
    census.committed = true;
    state_->credits->reap();
}
void CartesianGroupAuthority::replace_line_contributors(std::span<const CartesianTileUse> tiles,
    std::span<const std::shared_ptr<const CartesianExplicitLineSource>> explicit_sources, unsigned requested_layer)
{
    const auto layer = tiles.empty() ? requested_layer : tiles[0].tile.layer_;
    if (layer >= 64 || !state_->alive || state_->line_generations[layer] == UINT64_MAX)
        throw std::invalid_argument("Cartesian line authority generation exhausted.");
    Allocation memory(*state_->budget);
    memory.retain(sizeof(LineRegistry) + owner_bytes);
    auto next = std::make_shared<LineRegistry>();
    memory.reserve(next->tiles, tiles.size());
    memory.reserve(next->explicit_sources, explicit_sources.size());
    for (const auto& tile : tiles)
    {
        state_->check(tile.tile);
        if (tile.tile.layer_ != layer || tile.resident < 0)
            throw std::invalid_argument("Invalid Cartesian tile subscriber rank.");
        next->tiles.push_back(tile);
    }
    for (const auto& source : explicit_sources)
    {
        if (!source || std::ranges::find(next->explicit_sources, source) != next->explicit_sources.end())
            throw std::invalid_argument("Null or duplicate Cartesian explicit line contributor.");
        if (source->context() != state_->context && !(state_->inherited_context && state_->lines[layer] &&
                                                        std::ranges::find(state_->lines[layer]->explicit_sources,
                                                            source) != state_->lines[layer]->explicit_sources.end()))
            throw std::invalid_argument("Stale explicit line contributor context.");
        if (source->impl_->budget != state_->budget)
        {
            if (!source->impl_->keeper)
                throw std::invalid_argument("Cross-arena explicit line source has no lifetime keeper.");
        }
        next->explicit_sources.push_back(source);
    }
    CreditConstruction census(state_->credits);
    for (const auto& use : next->tiles)
    {
        state_->credit(census, use.tile.support_, *next, memory);
        state_->credit(census, use.tile.loop_, *next, memory);
    }
    for (const auto& source : next->explicit_sources)
        state_->credit(census, source, *next, memory);
    next->publish(memory, state_->keeper);
    state_->lines[layer] = std::move(next);
    ++state_->line_generations[layer];
    state_->prune_bindings();
    state_->leases.clear();
    census.committed = true;
    state_->credits->reap();
}
std::shared_ptr<const CartesianCompletionBatch> CartesianGroupAuthority::support_completions(
    std::span<const char> group, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena complete-group output requires its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    if (group.empty() || !state_->support)
        throw std::invalid_argument("Missing Cartesian support contributor registry.");
    state_->refresh(false);
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation batch_memory(budget);
    batch_memory.retain(sizeof(CartesianCompletionBatch) + sizeof(CartesianCompletionBatch::Impl) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianCompletionBatch>(new CartesianCompletionBatch);
    auto batch = std::make_shared<CartesianCompletionBatch::Impl>();
    batch_memory.reserve(batch->receipts, state_->support_contributors());
    for (std::size_t i = 0; i < state_->support_contributors(); ++i)
    {
        Allocation memory(budget);
        memory.retain(sizeof(CartesianCompletionReceipt) + sizeof(CartesianCompletionReceipt::Impl) + 2 * owner_bytes);
        auto receipt = std::shared_ptr<CartesianCompletionReceipt>(new CartesianCompletionReceipt);
        auto data = std::make_shared<CartesianCompletionReceipt::Impl>();
        data->authority = state_;
        data->origin = state_->origin;
        census.next_snapshot();
        state_->credit_owner(census, data->origin, *data, memory);
        data->generation = state_->support_generation;
        data->contributor = i;
        data->controls = cartesian_evaluation_environment();
        memory.copy(data->group, group);
        state_->support_members(group, budget,
            [&](std::size_t contributor, auto member)
            {
                if (contributor != i)
                    return;
                memory.append(data->members);
                data->members.emplace_back();
                memory.copy(data->members.back(), member);
            });
        data->publish(memory, keeper);
        receipt->impl_ = std::move(data);
        batch->receipts.push_back(std::move(receipt));
    }
    batch->publish(batch_memory, std::move(keeper));
    census.committed = true;
    result->impl_ = std::move(batch);
    return result;
}
std::shared_ptr<const CartesianCompletionBatch> CartesianGroupAuthority::line_completions(std::span<const char> group,
    unsigned layer, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena complete-group output requires its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    if (group.empty() || layer >= 64 || !state_->lines[layer])
        throw std::invalid_argument("Missing Cartesian line contributor registry.");
    state_->refresh(true, layer);
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation batch_memory(budget);
    batch_memory.retain(sizeof(CartesianCompletionBatch) + sizeof(CartesianCompletionBatch::Impl) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianCompletionBatch>(new CartesianCompletionBatch);
    auto batch = std::make_shared<CartesianCompletionBatch::Impl>();
    batch_memory.reserve(batch->receipts, state_->line_contributors(layer));
    for (std::size_t i = 0; i < state_->line_contributors(layer); ++i)
    {
        Allocation memory(budget);
        memory.retain(sizeof(CartesianCompletionReceipt) + sizeof(CartesianCompletionReceipt::Impl) + 2 * owner_bytes);
        auto receipt = std::shared_ptr<CartesianCompletionReceipt>(new CartesianCompletionReceipt);
        auto data = std::make_shared<CartesianCompletionReceipt::Impl>();
        data->authority = state_;
        data->origin = state_->origin;
        census.next_snapshot();
        state_->credit_owner(census, data->origin, *data, memory);
        data->generation = state_->line_generations[layer];
        data->contributor = i;
        data->controls = cartesian_evaluation_environment();
        data->line = true;
        data->layer = layer;
        memory.copy(data->group, group);
        state_->line_members(group, layer, budget,
            [&](std::size_t contributor, auto member)
            {
                if (contributor != i)
                    return;
                memory.append(data->members);
                data->members.emplace_back();
                memory.copy(data->members.back(), member);
            });
        data->publish(memory, keeper);
        receipt->impl_ = std::move(data);
        batch->receipts.push_back(std::move(receipt));
    }
    batch->publish(batch_memory, std::move(keeper));
    census.committed = true;
    result->impl_ = std::move(batch);
    return result;
}
std::shared_ptr<const CartesianCompleteSupport> CartesianGroupAuthority::complete_support(std::span<const char> group,
    std::span<const std::shared_ptr<const CartesianCompletionReceipt>> received, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena complete-group output requires its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    state_->refresh(false);
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation memory(budget);
    memory.retain(sizeof(CartesianCompleteSupport) + sizeof(SupportSeal) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianCompleteSupport>(new CartesianCompleteSupport);
    auto seal = std::make_shared<SupportSeal>();
    seal->authority = state_;
    seal->origin = state_->origin;
    seal->registry = state_->support;
    state_->credit_owner(census, seal->origin, *seal, memory);
    state_->credit(census, seal->registry, *seal, memory);
    state_->authenticate(false, group, 0, received, budget);
    seal->tile_budget = state_->tile_budget;
    seal->generation = state_->support_generation;
    seal->controls = cartesian_evaluation_environment();
    capture(seal->environment);
    memory.copy(seal->group, group);
    state_->support_members(
        group, budget, [](auto, auto) {},
        [&](const SupportInput& input)
        {
            if (seal->domain.empty())
                memory.copy(seal->domain, input.occurrence.domain);
        });
    auto count = [&](const PrivateTileRecipe&) { ++seal->tile_count; };
    state_->support_run(*seal, budget, count);
    seal->publish(memory, std::move(keeper));
    census.committed = true;
    result->seal_ = std::move(seal);
    return result;
}
std::shared_ptr<const CartesianCompleteLine> CartesianGroupAuthority::complete_line(std::span<const char> group,
    unsigned layer, std::span<const std::shared_ptr<const CartesianCompletionReceipt>> received, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena complete-group output requires its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    state_->refresh(true, layer);
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation memory(budget);
    memory.retain(sizeof(CartesianCompleteLine) + sizeof(LineSeal) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianCompleteLine>(new CartesianCompleteLine);
    auto seal = std::make_shared<LineSeal>();
    seal->authority = state_;
    seal->origin = state_->origin;
    seal->registry = state_->lines[layer];
    state_->credit_owner(census, seal->origin, *seal, memory);
    state_->credit(census, seal->registry, *seal, memory);
    state_->authenticate(true, group, layer, received, budget);
    seal->generation = state_->line_generations[layer];
    seal->layer_generations = state_->line_generations;
    seal->layer = layer;
    seal->controls = cartesian_evaluation_environment();
    capture(seal->environment);
    memory.copy(seal->group, group);
    state_->line_run(*seal, budget, [&](auto, const auto&) { ++seal->subscriptions; });
    seal->publish(memory, std::move(keeper));
    census.committed = true;
    result->seal_ = std::move(seal);
    return result;
}
std::size_t CartesianCompleteSupport::tile_count() const noexcept
{
    return seal_->tile_count;
}
std::size_t CartesianCompleteLine::subscription_count() const noexcept
{
    return seal_->subscriptions;
}
std::uint64_t CartesianSealedTile::support_generation() const noexcept
{
    return support_->generation;
}
std::uint64_t CartesianSealedTile::loop_generation() const noexcept
{
    return loop_ ? loop_->generation : 0;
}
bool CartesianSealedTile::stores_explicit_loop() const noexcept
{
    const auto* loop = loop_.get();
    while (loop && loop->identity)
        loop = loop->previous.get();
    return loop != nullptr;
}
void CartesianCompleteSupport::visit_tiles(Budget& budget, const CartesianSealedTileEmit& emit) const
{
    const auto state = seal_->authority.lock();
    if (!state)
        throw std::invalid_argument("Retired Cartesian support authority.");
    state->check(*seal_);
    Evaluation evaluation(seal_->environment);
    std::size_t count = 0;
    auto publish = [&](const PrivateTileRecipe& output)
    {
        CartesianSealedTile tile;
        tile.support_ = seal_;
        tile.emission_ = count++;
        evaluation.callback([&] { emit(tile, output); });
    };
    state->support_run(*seal_, budget, publish);
    if (count != seal_->tile_count)
        throw std::invalid_argument("Changed sealed Cartesian support output.");
}
void CartesianCompleteLine::visit_queries(Budget& budget, const CartesianLineQueryEmit& emit) const
{
    const auto state = seal_->authority.lock();
    if (!state)
        throw std::invalid_argument("Retired Cartesian line authority.");
    state->check(*seal_);
    Evaluation evaluation(seal_->environment);
    state->line_run(*seal_, budget,
        [&](std::size_t index, const auto& output) { evaluation.callback([&] { emit(index, output); }); });
}
struct CartesianReplayCertificate::Impl : Owned
{
    std::shared_ptr<const SupportSeal> support;
    std::shared_ptr<const LineSeal> line;
    std::size_t subscription = 0;
    CartesianReplayRecipe recipe;
    std::vector<AdaptiveBytes> exceptions;
    bool implicit = false;
};
namespace
{
template<class Certificate>
void build_certificate(Certificate& certificate, const CartesianReplayRecipe& recipe, Budget& budget,
    Allocation& retained, const std::fenv_t& environment,
    const CartesianBorrowedCallback<const CartesianIdentityEmit&>& actual)
{
    Allocation actual_memory(budget);
    std::vector<AdaptiveBytes> output;
    {
        Evaluation evaluation(environment);
        actual(
            [&](std::span<const char> identity)
            {
                actual_memory.append(output);
                output.emplace_back();
                actual_memory.copy(output.back(), identity);
            });
        std::size_t position = 0;
        bool matches = true;
        replay_cartesian_recipe(recipe, budget,
            [&](auto identity)
            {
                if (position >= output.size() || !std::ranges::equal(identity, output[position]))
                    matches = false;
                ++position;
            });
        certificate.implicit = matches && position == output.size();
    }
    if (!certificate.implicit)
    {
        certificate.exceptions = std::move(output);
        retained.bytes = add(retained.bytes, actual_memory.bytes);
        actual_memory.bytes = 0;
        return;
    }
    retained.reserve(certificate.recipe.sources, recipe.sources.size());
    certificate.recipe.sources.assign(recipe.sources.begin(), recipe.sources.end());
    retained.reserve(certificate.recipe.selectors, recipe.selectors.size());
    for (const auto& selector : recipe.selectors)
    {
        certificate.recipe.selectors.emplace_back();
        auto& target = certificate.recipe.selectors.back();
        target.source = selector.source;
        target.range = selector.range;
        target.cell = selector.cell;
        target.face = selector.face;
        target.output = selector.output;
        target.first = selector.first;
        target.count = selector.count;
        target.reverse = selector.reverse;
        target.authority_image = selector.authority_image;
        retained.copy(target.support, selector.support);
    }
}
void check_recipe_sources(const CartesianReplayRecipe& recipe, const std::shared_ptr<const SupportRegistry>& registry)
{
    for (const auto& source : recipe.sources)
        if (!source || std::ranges::find(registry->sources, source) == registry->sources.end())
            throw std::invalid_argument("Unregistered procedural source in Cartesian replay recipe.");
}
} // namespace
std::shared_ptr<const CartesianReplayCertificate> CartesianReplayCertificate::from_support(
    const CartesianCompleteSupport& complete, const CartesianReplayRecipe& recipe, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
{
    check_keeper(budget, keeper);
    const auto state = complete.seal_->authority.lock();
    if (!state)
        throw std::invalid_argument("Retired Cartesian support authority.");
    state->check(*complete.seal_);
    check_recipe_sources(recipe, complete.seal_->registry);
    if (&budget != state->budget && !keeper)
        throw std::invalid_argument("Cross-arena Cartesian certificate requires its own keeper.");
    if (!keeper && &budget == state->budget)
        keeper = state->keeper;
    CreditConstruction census(state->arena_for(budget, keeper, std::move(arena)));
    Allocation retained(budget);
    retained.retain(sizeof(CartesianReplayCertificate) + sizeof(Impl) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianReplayCertificate>(new CartesianReplayCertificate);
    auto body = std::make_shared<Impl>();
    body->support = complete.seal_;
    state->credit(census, body->support, *body, retained);
    build_certificate(*body, recipe, budget, retained, complete.seal_->environment,
        [&](const auto& emit)
        {
            auto tile = [&](const PrivateTileRecipe& output)
            {
                Allocation encoding(budget);
                const auto key = face_key(output.face, encoding);
                emit(key);
            };
            state->support_run(*complete.seal_, budget, tile);
        });
    for (const auto& source : body->recipe.sources)
        state->credit(census, source, *body, retained);
    body->publish(retained, std::move(keeper));
    census.committed = true;
    result->impl_ = std::move(body);
    return result;
}
std::shared_ptr<const CartesianReplayCertificate> CartesianReplayCertificate::from_line(
    const CartesianCompleteLine& complete, std::size_t subscription, const CartesianReplayRecipe& recipe,
    Budget& budget, std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
{
    check_keeper(budget, keeper);
    const auto state = complete.seal_->authority.lock();
    if (!state || subscription >= complete.seal_->subscriptions)
        throw std::invalid_argument("Unknown sealed Cartesian line subscription.");
    state->check(*complete.seal_);
    for (const auto& source : recipe.sources)
    {
        bool registered = false;
        for (const auto& tile : complete.seal_->registry->tiles)
            registered = registered || std::ranges::find(tile.tile.support_->registry->sources, source) !=
                                           tile.tile.support_->registry->sources.end();
        if (!registered)
            throw std::invalid_argument("Unregistered source in complete Cartesian line recipe.");
    }
    if (&budget != state->budget && !keeper)
        throw std::invalid_argument("Cross-arena Cartesian certificate requires its own keeper.");
    if (!keeper && &budget == state->budget)
        keeper = state->keeper;
    CreditConstruction census(state->arena_for(budget, keeper, std::move(arena)));
    Allocation retained(budget);
    retained.retain(sizeof(CartesianReplayCertificate) + sizeof(Impl) + 2 * owner_bytes);
    auto result = std::shared_ptr<CartesianReplayCertificate>(new CartesianReplayCertificate);
    auto body = std::make_shared<Impl>();
    body->line = complete.seal_;
    state->credit(census, body->line, *body, retained);
    body->subscription = subscription;
    build_certificate(*body, recipe, budget, retained, complete.seal_->environment,
        [&](const auto& emit)
        {
            state->line_run(*complete.seal_, budget,
                [&](std::size_t index, const auto& points)
                {
                    if (index != subscription)
                        return;
                    for (std::size_t i = 0; i < points.size(); ++i)
                    {
                        Allocation encoding(budget);
                        const auto key = identity_key(points.point(i), encoding);
                        emit(key);
                    }
                });
        });
    for (const auto& source : body->recipe.sources)
        state->credit(census, source, *body, retained);
    body->publish(retained, std::move(keeper));
    census.committed = true;
    result->impl_ = std::move(body);
    return result;
}
CartesianReplayCertificate::~CartesianReplayCertificate() = default;
bool CartesianReplayCertificate::implicit() const noexcept
{
    return impl_->implicit;
}
void CartesianReplayCertificate::replay(
    const CartesianGroupAuthority& authority, Budget& budget, const CartesianIdentityEmit& emit) const
{
    if (impl_->support)
        authority.state_->check(*impl_->support);
    else
        authority.state_->check(*impl_->line);
    if (!impl_->implicit)
    {
        for (const auto& value : impl_->exceptions)
            emit(value);
        return;
    }
    Allocation scratch(budget);
    std::vector<AdaptiveBytes> output;
    {
        Evaluation environment(impl_->support ? impl_->support->environment : impl_->line->environment);
        replay_cartesian_recipe(impl_->recipe, budget,
            [&](auto value)
            {
                scratch.append(output);
                output.emplace_back();
                scratch.copy(output.back(), value);
            });
    }
    for (const auto& value : output)
        emit(value);
}
CartesianSealedTile CartesianGroupAuthority::advance_tile(const CartesianSealedTile& tile,
    std::span<const EdgeReply> replies, unsigned authority_ranks, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena) const
{
    check_keeper(budget, keeper);
    if (&budget != state_->budget && !keeper)
        throw std::invalid_argument("Cross-arena tile output requires its own keeper.");
    if (!keeper && &budget == state_->budget)
        keeper = state_->keeper;
    state_->check(tile);
    if (!authority_ranks || tile.layer_ >= 63)
        throw std::invalid_argument("Invalid Cartesian tile layer/authority count.");
    struct Reply
    {
        AdaptiveBytes line;
        unsigned image = 0;
        std::size_t edge = 0;
        std::vector<LinePoint> points;
    };
    CreditConstruction census(state_->arena_for(budget, keeper, std::move(arena)));
    Allocation memory(budget), retained(budget);
    retained.retain(sizeof(LoopSeal) + owner_bytes);
    auto loop = std::make_shared<LoopSeal>();
    state_->credit(census, tile.support_, *loop, retained);
    for (const auto& reply : replies)
    {
        if (!reply.line)
            throw std::invalid_argument("Missing sealed Cartesian edge reply.");
        state_->credit(census, reply.line->seal_, *loop, retained);
    }
    std::vector<Reply> answers;
    // The controlled line registry contains exact tile subscriber identities.
    // Locate each query there; no raw endpoint/output constructor is exposed.
    for (const auto& reply : replies)
    {
        if (!reply.line)
            throw std::invalid_argument("Missing sealed Cartesian edge reply.");
        const auto& seal = *reply.line->seal_;
        state_->check(seal);
        if (seal.layer != tile.layer_ || reply.subscription >= seal.subscriptions)
            throw std::invalid_argument("Stale Cartesian tile edge layer.");
        std::size_t index = 0, contributor = 0;
        std::optional<LineSubscription> subscription;
        state_->line_members(
            seal.group, seal.layer, budget,
            [&](std::size_t source, auto)
            {
                if (index == reply.subscription)
                    contributor = source;
            },
            [&](const auto& value)
            {
                if (index++ == reply.subscription)
                    subscription = value;
            });
        if (contributor >= seal.registry->tiles.size())
            throw std::invalid_argument("Explicit query cannot impersonate a sealed tile edge reply.");
        const auto& original = seal.registry->tiles[contributor].tile;
        if (original.support_->registry != tile.support_->registry ||
            original.support_->group != tile.support_->group ||
            original.support_->generation != tile.support_->generation || original.emission_ != tile.emission_ ||
            original.loop_ != tile.loop_ || original.layer_ != tile.layer_)
            throw std::invalid_argument("Foreign tile source in sealed Cartesian edge reply.");
        bool found = false;
        state_->tile_run(tile, budget,
            [&](const PrivateTileRecipe& current)
            {
                std::array<char, 1> member{'s'};
                visit_cartesian_tile_edges(
                    tile.support_->domain, member, current, budget,
                    [&](const auto& edge)
                    {
                        if (edge.key != seal.group || edge.first != subscription->first ||
                            edge.last != subscription->last)
                            return;
                        for (const auto& old : answers)
                            if (old.image == edge.image && old.edge == edge.edge)
                                throw std::invalid_argument("Duplicate complete Cartesian edge reply.");
                        memory.append(answers);
                        answers.emplace_back();
                        auto& output = answers.back();
                        memory.copy(output.line, edge.key);
                        output.image = edge.image;
                        output.edge = edge.edge;
                        found = true;
                        reply.line->visit_queries(budget,
                            [&](std::size_t q, const auto& points)
                            {
                                if (q != reply.subscription)
                                    return;
                                memory.reserve(output.points, points.size());
                                for (std::size_t p = 0; p < points.size(); ++p)
                                    output.points.push_back(edge.image
                                                                ? PLIC::point_translated<ArithmeticMode::Ordinary>(
                                                                      points.point(p), current.second_image)
                                                                : points.point(p));
                            });
                    },
                    seal.group);
            });
        if (!found)
            throw std::invalid_argument("Foreign sealed Cartesian tile edge reply.");
    }
    CartesianSealedTile result = tile;
    state_->tile_run(tile, budget,
        [&](const PrivateTileRecipe& current)
        {
            const auto expected = current.face->points().size() * (current.second_image.is_zero() ? 1 : 2);
            if (answers.size() != expected)
                throw std::invalid_argument("Missing complete Cartesian tile edge reply.");
            std::sort(answers.begin(), answers.end(),
                [&](const auto& a, const auto& b)
                {
                    const auto aa = adaptive_route_hash(a.line) % authority_ranks;
                    const auto ba = adaptive_route_hash(b.line) % authority_ranks;
                    return std::tie(aa, a.line, a.image, a.edge) < std::tie(ba, b.line, b.image, b.edge);
                });
            std::size_t count = 0;
            for (const auto& answer : answers)
                count = add(count, answer.points.size());
            Allocation work(budget);
            std::vector<LinePoint> points;
            work.reserve(points, count);
            for (const auto& answer : answers)
                points.insert(points.end(), answer.points.begin(), answer.points.end());
            work.retain(add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(
                                add(current.face->points().size(), count)),
                product(add(current.face->points().size(), count), 256)));
            const auto enriched = current.face->with_edge_vertices(points);
            work.retain(growth(enriched->points().size(), sizeof(LinePoint)));
            work.retain(growth(current.face->points().size(), sizeof(LinePoint)));
            const auto vertices = enriched->vertex_identities();
            const auto input = current.face->vertex_identities();
            if (vertices == input)
            {
                loop->identity = true;
                loop->previous = tile.loop_;
            }
            else
            {
                retained.reserve(loop->points, vertices.size());
                loop->points.assign(vertices.begin(), vertices.end());
            }
        });
    retained.reserve(loop->lines, replies.size());
    for (const auto& reply : replies)
        loop->lines.push_back(reply.line->seal_);
    if (state_->loop_generation == UINT64_MAX)
        throw std::invalid_argument("Cartesian loop generation exhausted.");
    state_->credit(census, tile.support_, *loop, retained);
    state_->credit(census, loop->previous, *loop, retained);
    for (const auto& line : loop->lines)
        state_->credit(census, line, *loop, retained);
    loop->generation = ++state_->loop_generation;
    loop->publish(retained, std::move(keeper));
    census.committed = true;
    result.loop_ = std::move(loop);
    result.layer_++;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
