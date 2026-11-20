#include "CartesianSourceLineRanges.hh"
#include "CartesianSourceManifests.hh"
#include "NativeContactFunding.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Allocation = cartesian_detail::Allocation;
using cartesian_detail::add;
using cartesian_detail::product;
constexpr std::size_t control = 128;
constexpr auto absent = std::numeric_limits<std::size_t>::max();
/** Captured in the owning shared_ptr's deleter. Deleter storage survives until
 * final weak-control deletion, so its immutable charge/keeper cannot disappear
 * when the geometry body alone expires. The token itself has no weak owner. */
using contact_funding_detail::ControlCredit;
using contact_funding_detail::make_controlled;
constexpr auto control_bytes = ControlCredit::allowance();
template<class T>
void retain_owner(cartesian_detail::CreditConstruction& credits, const std::shared_ptr<T>& owner, const Budget* origin,
    std::size_t bytes, bool keeper, std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases,
    Allocation& memory)
{
    const std::shared_ptr<const std::remove_const_t<T>> immutable = owner;
    credits.retain(immutable, origin, bytes, keeper, leases, memory);
}
struct WordRange
{
    std::uint64_t first, last;
};
struct AxisEntry
{
    std::uint64_t coordinate;
    std::size_t component;
};
struct AxisMap
{
    std::array<std::uint64_t, 8> kind;
    std::vector<AxisEntry> entries;
};
struct ParsedSupport
{
    std::size_t offset;
    std::uint64_t normal, level;
};
template<class Index, class Value> Index checked_index(Value value)
{
    if (!std::in_range<Index>(value))
        throw std::invalid_argument("Source line range index exceeds its platform or storage width.");
    return static_cast<Index>(value);
}
template<class Index> struct ComponentSelector
{
    std::array<Index, 3> selectors{};
    std::uint8_t map_bits = 0;
};
std::optional<ParsedSupport> parse_support(std::span<const char> key, const CartesianSourceManifestSet& manifests)
{
    if (key.size() < 64)
        return {};
    Reader reader{key};
    if (reader.word() != 0x5346434152543631ULL)
        return {};
    const auto domain = reader.bytes_view();
    if (key.size() != add(64, domain.size()) || reader.word() != manifests.grid().region_id)
        return {};
    bool source_domain = false;
    for (std::size_t owner = 0; owner < manifests.source_count(); ++owner)
        if (manifests.source(owner).cell_count() && std::ranges::equal(domain, manifests.source(owner).domain()))
        {
            source_domain = true;
            break;
        }
    if (!source_domain)
        return {};
    const auto normal = reader.word(), level = reader.word();
    if (normal >= 3 || level >= 63)
        return {};
    return ParsedSupport{reader.offset, normal, level};
}
std::uint64_t word(const IncidenceSupportKey& key, std::size_t offset)
{
    Reader reader{key.bytes(), offset};
    return reader.word();
}
bool key_equal(const CartesianSupportRangeKeyView& a, const CartesianSupportRangeKeyView& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
            return false;
    return true;
}
bool key_before(const CartesianSupportRangeKeyView& a, const CartesianSupportRangeKeyView& b)
{
    const auto size = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < size; ++i)
        if (a[i] != b[i])
            return a[i] < b[i];
    return a.size() < b.size();
}
bool full_point_equal(const ContactSourcePoint& a, const ContactSourcePoint& b)
{
    using Access = PLIC::details::GeometryIdentityAccess;
    const auto bytes = PLIC::canonical_key_size(a);
    if (bytes != PLIC::canonical_key_size(b))
        return false;
    for (std::size_t offset = 0; offset < bytes; offset += 8)
        if (Access::canonical_word_at(a, offset) != Access::canonical_word_at(b, offset))
            return false;
    return true;
}
enum class Membership : std::uint8_t
{
    Contiguous,
    InlineMask,
    Mask,
    Ranges
};
/** Published descriptors have no owning subobjects. The representation width
 * is selected once for the whole recipe, after every actual index and sum is
 * checked. Coordinates and identity words retain their original precision. */
template<class Index> struct Family
{
    Index prototype = 0, offset = 0, count = 0;
    Index membership_offset = 0, membership_count = 0, result_offset = 0, result_count = 0;
    std::array<Index, 6> selectors{};
    std::uint64_t lower = 0, upper = 0, inline_mask = 0;
    std::uint64_t emission = 0, edge = 0, generation = 0;
    int resident = -1, authority = -1;
    std::uint8_t image = 0, map_bits = 0;
    Membership membership = Membership::Contiguous;
    PooledLineSubscriber::ResultState result_state = PooledLineSubscriber::ResultState::ForwardEndpoints;

    template<class Other> static Family copy(const Family<Other>& from)
    {
        Family f;
        // The complete recipe preflight, including cumulative flat offsets,
        // precedes every narrowing conversion and final allocation.
        f.prototype = checked_index<Index>(from.prototype);
        f.offset = checked_index<Index>(from.offset);
        f.count = checked_index<Index>(from.count);
        f.membership_offset = checked_index<Index>(from.membership_offset);
        f.membership_count = checked_index<Index>(from.membership_count);
        f.result_offset = checked_index<Index>(from.result_offset);
        f.result_count = checked_index<Index>(from.result_count);
        for (unsigned i = 0; i != 6; ++i)
            f.selectors[i] = checked_index<Index>(from.selectors[i]);
        f.lower = from.lower;
        f.upper = from.upper;
        f.inline_mask = from.inline_mask;
        f.emission = from.emission;
        f.edge = from.edge;
        f.generation = from.generation;
        f.resident = from.resident;
        f.authority = from.authority;
        f.image = from.image;
        f.map_bits = from.map_bits;
        f.membership = from.membership;
        f.result_state = from.result_state;
        return f;
    }
    [[nodiscard]] ComponentSelector<std::size_t> endpoint(unsigned first) const
    {
        return {{{checked_index<std::size_t>(selectors[first]), checked_index<std::size_t>(selectors[first + 1]),
                    checked_index<std::size_t>(selectors[first + 2])}},
            std::uint8_t((map_bits >> first) & 7)};
    }
    template<class Emit>
    void visit_coordinates(std::span<const std::uint64_t> masks, std::span<const WordRange> ranges, Emit&& emit) const
    {
        if (membership == Membership::Contiguous)
        {
            for (auto n = lower;; ++n)
            {
                emit(n);
                if (n == upper)
                    break;
            }
            return;
        }
        if (membership == Membership::Ranges)
        {
            for (const auto& range : ranges.subspan(
                     checked_index<std::size_t>(membership_offset), checked_index<std::size_t>(membership_count)))
                for (auto n = range.first;; ++n)
                {
                    emit(n);
                    if (n == range.last)
                        break;
                }
            return;
        }
        const auto words = membership == Membership::InlineMask
                               ? std::span<const std::uint64_t>(&inline_mask, 1)
                               : masks.subspan(checked_index<std::size_t>(membership_offset),
                                     checked_index<std::size_t>(membership_count));
        for (std::size_t i = 0; i < words.size(); ++i)
        {
            auto bits = words[i];
            while (bits)
            {
                const auto bit = std::countr_zero(bits);
                emit(lower + std::uint64_t(i) * 64 + unsigned(bit));
                bits &= bits - 1;
            }
        }
    }
    /** Ordered by actual encoded char bytes, including127/128 and255/256.
     * Ordinary short intervals avoid radix work. No ordinal permutation lives
     * in the recipe. */
    std::uint64_t coordinate_at(
        std::size_t index, std::span<const std::uint64_t> masks, std::span<const WordRange> ranges) const
    {
        if (index >= count)
            throw std::out_of_range("Invalid source line range coordinate ordinal.");
        const bool simple = (lower >> 8) == (upper >> 8) && ((lower & 128) == (upper & 128));
        if (simple)
        {
            std::uint64_t result = 0;
            std::size_t position = 0;
            visit_coordinates(masks, ranges,
                [&](auto value)
                {
                    if (position++ == index)
                        result = value;
                });
            return result;
        }
        std::uint64_t prefix = 0, prefix_mask = 0;
        for (unsigned shift = 0; shift < 64; shift += 8)
        {
            std::array<std::size_t, 256> counts{};
            visit_coordinates(masks, ranges,
                [&](auto value)
                {
                    if ((value & prefix_mask) == prefix)
                        ++counts[static_cast<unsigned char>(value >> shift)];
                });
            for (unsigned order = 0; order < 256; ++order)
            {
                const auto byte = std::numeric_limits<char>::is_signed ? ((order + 128) & 255) : order;
                if (index < counts[byte])
                {
                    prefix |= std::uint64_t(byte) << shift;
                    prefix_mask |= std::uint64_t(255) << shift;
                    break;
                }
                index -= counts[byte];
            }
        }
        return prefix;
    }
};
using WideFamily = Family<std::uint64_t>;
using FamilyView = Family<std::size_t>;
using CompactFamily = Family<std::uint32_t>;
using WideSelector = ComponentSelector<std::uint64_t>;
using SelectorView = ComponentSelector<std::size_t>;
using CompactSelector = ComponentSelector<std::uint32_t>;
struct PrototypeEntry
{
    std::size_t offset, size;
};
/** Mutable per-line draft retains the two certification columns. These are
 * never copied into a published recipe and are reused only by its builder. */
struct FamilyDraft
{
    FamilyView value;
    std::array<std::size_t, 6> fixed{}, maps{absent, absent, absent, absent, absent, absent};
};
bool constants_equal(const PooledLineSubscriber& a, const PooledLineSubscriber& b)
{
    return std::tie(a.resident, a.support_authority, a.emission, a.image, a.edge, a.source_generation) ==
           std::tie(b.resident, b.support_authority, b.emission, b.image, b.edge, b.source_generation);
}
bool family_before(const FamilyView& a, std::uint64_t ak, std::span<const char> ap, const FamilyView& b,
    std::uint64_t bk, std::span<const char> bp)
{
    if (a.resident != b.resident)
        return a.resident < b.resident;
    if (a.authority != b.authority)
        return a.authority < b.authority;
    const CartesianSupportRangeKeyView av{ap, a.offset, ak}, bv{bp, b.offset, bk};
    if (!key_equal(av, bv))
        return key_before(av, bv);
    return std::tie(a.emission, a.image, a.edge) < std::tie(b.emission, b.image, b.edge);
}

} // namespace

struct CartesianSourceLineRangeCatalog::Impl
{
    const ControlCredit* control_credit = nullptr; // owned by the body's custom deleter
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0;
    std::shared_ptr<const CartesianSourceManifestSet> manifests;
    std::shared_ptr<const ContactSourcePointSnapshot> points;
    ContactHaloContext context;
    CartesianEvaluationEnvironment evaluation;
    std::uint64_t generation = 0;
    unsigned layer = 0;
    bool frozen = false;
    std::vector<AxisMap> maps;
    std::vector<PrototypeEntry> prototypes;
    AdaptiveBytes prototype_bytes;
    // Requests and unique entries expose actual complete-value duplication;
    // there is no hash-only match and no retained mutable lookup directory.
    std::size_t prototype_requests = 0;
    [[nodiscard]] std::span<const char> prototype(std::size_t index) const
    {
        const auto& entry = prototypes.at(index);
        return std::span<const char>(prototype_bytes).subspan(entry.offset, entry.size);
    }
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
    ~Impl()
    {
        decltype(maps){}.swap(maps);
        decltype(prototypes){}.swap(prototypes);
        decltype(prototype_bytes){}.swap(prototype_bytes);
        manifests.reset();
        points.reset();
        decltype(leases){}.swap(leases);
        if (budget)
            budget->used -= charge;
    }
};
struct CartesianSourceLineRangeRecipe::Impl
{
    const ControlCredit* control_credit = nullptr; // owned by the body's custom deleter
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0, count = 0, line_slot = 0;
    unsigned certified_layer = 0;
    AdaptiveBytes line;
    std::shared_ptr<const CartesianSourceLineRangeCatalog> catalog;
    // Exactly one descriptor/selector alternative is populated per recipe.
    std::vector<CompactFamily> compact;
    std::vector<WideFamily> wide;
    std::vector<CompactSelector> compact_results;
    std::vector<WideSelector> wide_results;
    std::vector<std::uint64_t> masks;
    std::vector<WordRange> ranges;
    [[nodiscard]] std::size_t family_count() const noexcept { return compact.empty() ? wide.size() : compact.size(); }
    [[nodiscard]] FamilyView family(std::size_t index) const
    {
        return compact.empty() ? FamilyView::copy(wide.at(index)) : FamilyView::copy(compact.at(index));
    }
    [[nodiscard]] SelectorView result(std::size_t index) const
    {
        SelectorView selected;
        if (compact.empty())
        {
            const auto& from = wide_results.at(index);
            for (unsigned i = 0; i != 3; ++i)
                selected.selectors[i] = checked_index<std::size_t>(from.selectors[i]);
            selected.map_bits = from.map_bits;
        }
        else
        {
            const auto& from = compact_results.at(index);
            for (unsigned i = 0; i != 3; ++i)
                selected.selectors[i] = checked_index<std::size_t>(from.selectors[i]);
            selected.map_bits = from.map_bits;
        }
        return selected;
    }
    [[nodiscard]] std::uint64_t coordinate(const FamilyView& f, std::size_t index) const
    {
        return f.coordinate_at(index, masks, ranges);
    }
    ~Impl()
    {
        decltype(line){}.swap(line);
        decltype(compact){}.swap(compact);
        decltype(wide){}.swap(wide);
        decltype(compact_results){}.swap(compact_results);
        decltype(wide_results){}.swap(wide_results);
        decltype(masks){}.swap(masks);
        decltype(ranges){}.swap(ranges);
        catalog.reset();
        if (budget)
            budget->used -= charge;
    }
};
struct CartesianSourceLineRangeBuilder::Impl
{
    Allocation memory;
    std::shared_ptr<CartesianSourceLineRangeCatalog> catalog;
    // Invocation-local charged workspace, reused across the whole closed pass.
    // Declaration order destroys every buffer before the allocation lease.
    std::vector<std::array<std::size_t, 6>> components;
    std::vector<ParsedSupport> parsed;
    std::vector<std::size_t> result_offsets, members, best;
    std::vector<std::array<std::size_t, 3>> result_components;
    std::vector<std::uint8_t> consumed;
    std::vector<FamilyDraft> drafts;
    std::vector<std::uint64_t> masks;
    std::vector<WordRange> ranges;
    std::vector<SelectorView> results;
    explicit Impl(Budget& budget) : memory(budget) {}
};
CartesianSourceLineRangeCatalog::CartesianSourceLineRangeCatalog() = default;
CartesianSourceLineRangeCatalog::~CartesianSourceLineRangeCatalog() = default;
NativeContactOwnerFunding CartesianSourceLineRangeCatalog::native_funding(
    const std::shared_ptr<const CartesianSourceLineRangeCatalog>& owner, std::size_t bytes) const
{
    return NativeContactFundingAccess::split_named(owner, bytes,
        [](const CartesianSourceLineRangeCatalog& value) { return value.impl_->charge; },
        [](const CartesianSourceLineRangeCatalog& value) { return value.impl_->control_credit; });
}
std::size_t CartesianSourceLineRangeCatalog::metadata_bytes() const noexcept
{
    return impl_->charge + control_bytes;
}
const Budget* CartesianSourceLineRangeCatalog::allocation_budget() const noexcept
{
    return impl_->budget;
}
bool CartesianSourceLineRangeCatalog::has_arena_keeper() const noexcept
{
    return bool(impl_->keeper);
}
std::size_t CartesianSourceLineRangeCatalog::map_count() const noexcept
{
    return impl_->maps.size();
}
std::size_t CartesianSourceLineRangeCatalog::mapped_components() const noexcept
{
    std::size_t result = 0;
    for (const auto& map : impl_->maps)
        result += map.entries.size();
    return result;
}
std::size_t CartesianSourceLineRangeCatalog::prototype_count() const noexcept
{
    return impl_->prototypes.size();
}
std::size_t CartesianSourceLineRangeCatalog::prototype_request_count() const noexcept
{
    return impl_->prototype_requests;
}
std::pair<const void*, std::size_t> CartesianSourceLineRangeCatalog::borrowed_control_payload() const noexcept
{
    return {impl_->control_credit, sizeof(ControlCredit)};
}
std::pair<const void*, std::size_t> CartesianSourceLineRangeRecipe::borrowed_control_payload() const noexcept
{
    return {impl_->control_credit, sizeof(ControlCredit)};
}
void CartesianSourceLineRangeCatalog::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(visitor, this, ContactPayloadRole::CartesianLineCatalogObject, sizeof(*this), true);
    if (!impl_)
        return;
    const auto& c = *impl_;
    contact_payload_object(visitor, &c, ContactPayloadRole::CartesianLineCatalogImpl, sizeof(c));
    if (c.control_credit)
        contact_payload_object(
            visitor, c.control_credit, ContactPayloadRole::CartesianControlCreditObject, sizeof(ControlCredit), true);
    contact_payload_buffer(visitor, &c, ContactPayloadRole::CartesianLineCatalogMaps, c.maps);
    contact_payload_buffer(visitor, &c, ContactPayloadRole::CartesianLinePrototypeEntries, c.prototypes);
    contact_payload_buffer(visitor, &c, ContactPayloadRole::CartesianLinePrototype, c.prototype_bytes);
    for (const auto& map : c.maps)
        contact_payload_buffer(visitor, &c, ContactPayloadRole::CartesianLineAxisEntries, map.entries);
    contact_payload_buffer(visitor, &c, ContactPayloadRole::CartesianLineCatalogLeases, c.leases);
    for (const auto& lease : c.leases)
        if (lease)
            lease->visit_borrowed_payload(visitor);
    if (c.points)
        c.points->visit_borrowed_payload(visitor);
    if (c.manifests)
        c.manifests->visit_borrowed_payload(visitor);
}
void CartesianSourceLineRangeRecipe::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(visitor, this, ContactPayloadRole::CartesianLineRecipeObject, sizeof(*this), true);
    if (!impl_)
        return;
    const auto& r = *impl_;
    contact_payload_object(visitor, &r, ContactPayloadRole::CartesianLineRecipeImpl, sizeof(r));
    if (r.control_credit)
        contact_payload_object(
            visitor, r.control_credit, ContactPayloadRole::CartesianControlCreditObject, sizeof(ControlCredit), true);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineBytes, r.line);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineFamilies, r.compact);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineFamilies, r.wide);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineMask, r.masks);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineRanges, r.ranges);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineResultPoints, r.compact_results);
    contact_payload_buffer(visitor, &r, ContactPayloadRole::CartesianLineResultPoints, r.wide_results);
    if (r.catalog)
        r.catalog->visit_borrowed_payload(visitor);
}
CartesianSourceLineRangeRecipe::CartesianSourceLineRangeRecipe() = default;
CartesianSourceLineRangeRecipe::~CartesianSourceLineRangeRecipe() = default;
std::size_t CartesianSourceLineRangeRecipe::size() const noexcept
{
    return impl_->count;
}
unsigned CartesianSourceLineRangeRecipe::certified_layer() const noexcept
{
    return impl_->certified_layer;
}
std::size_t CartesianSourceLineRangeRecipe::family_count() const noexcept
{
    return impl_->family_count();
}
std::size_t CartesianSourceLineRangeRecipe::index_width_bytes() const noexcept
{
    return impl_->compact.empty() ? sizeof(std::uint64_t) : sizeof(std::uint32_t);
}
std::size_t CartesianSourceLineRangeRecipe::nonforward_count() const noexcept
{
    std::size_t count = 0;
    for (std::size_t i = 0; i != impl_->family_count(); ++i)
    {
        const auto family = impl_->family(i);
        if (family.result_state == PooledLineSubscriber::ResultState::Explicit)
            count += family.count;
    }
    return count;
}
std::size_t CartesianSourceLineRangeRecipe::result_selector_count() const noexcept
{
    return impl_->compact_results.size() + impl_->wide_results.size();
}
NativeContactOwnerFunding CartesianSourceLineRangeRecipe::native_funding(
    const std::shared_ptr<const CartesianSourceLineRangeRecipe>& owner, std::size_t bytes) const
{
    return NativeContactFundingAccess::split_named(owner, bytes,
        [](const CartesianSourceLineRangeRecipe& value) { return value.impl_->charge; },
        [](const CartesianSourceLineRangeRecipe& value) { return value.impl_->control_credit; });
}
std::size_t CartesianSourceLineRangeRecipe::metadata_bytes() const noexcept
{
    return impl_->charge + control_bytes;
}
std::size_t CartesianSourceLineRangeRecipe::control_metadata_bytes() noexcept
{
    return control_bytes;
}
const Budget* CartesianSourceLineRangeRecipe::allocation_budget() const noexcept
{
    return impl_->budget;
}
bool CartesianSourceLineRangeRecipe::has_arena_keeper() const noexcept
{
    return bool(impl_->keeper);
}
const std::shared_ptr<const CartesianSourceLineRangeCatalog>& CartesianSourceLineRangeRecipe::catalog() const noexcept
{
    return impl_->catalog;
}

CartesianSourceLineRangeBuilder::CartesianSourceLineRangeBuilder(
    const CartesianLineRangeAdmission& admission, Budget& budget, std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (!arena || arena->budget != &budget || !arena->keeper || !admission.manifests_ || !admission.points_ ||
        !admission.generation_ || admission.layer_ >= 64 || admission.points_->context() != admission.context_ ||
        admission.points_->certified_layer() != admission.layer_ ||
        admission.manifests_->context() != admission.context_ ||
        admission.manifests_->generation() != admission.generation_ || !admission.manifests_->owner_count())
        throw std::invalid_argument("Incomplete closed source line range admission.");
    Allocation construction(budget);
    construction.retain(sizeof(Impl));
    impl_ = std::make_unique<Impl>(budget);
    impl_->memory.bytes = std::exchange(construction.bytes, 0);
    Allocation memory(budget);
    memory.retain(sizeof(CartesianSourceLineRangeCatalog) + sizeof(CartesianSourceLineRangeCatalog::Impl));
    auto [catalog_owner, catalog_control] =
        make_controlled(budget, arena->keeper, [] { return new CartesianSourceLineRangeCatalog; });
    impl_->catalog = std::move(catalog_owner);
    auto& catalog = *impl_->catalog;
    catalog.impl_ = std::make_unique<CartesianSourceLineRangeCatalog::Impl>();
    auto& c = *catalog.impl_;
    c.control_credit = catalog_control.get();
    c.manifests = admission.manifests_;
    c.points = admission.points_;
    c.context = admission.context_;
    c.evaluation = admission.points_->evaluation_environment();
    c.generation = admission.generation_;
    c.layer = admission.layer_;
    cartesian_detail::CreditConstruction credits(arena);
    credits.retain(c.points, c.points->allocation_budget(), c.points->metadata_bytes(), c.points->has_arena_keeper(),
        c.leases, memory);
    if (!arena->prepaid(c.points, c.points->allocation_budget(), c.points->metadata_bytes(), false))
        c.points->visit_dependencies<CartesianDependencyVisit::Physical>(
            [&](const auto& owner, auto origin, auto bytes, auto keeper)
            { retain_owner(credits, owner, origin, bytes, keeper, c.leases, memory); });
    credits.retain(c.manifests, c.manifests->allocation_budget(), c.manifests->metadata_bytes(),
        c.manifests->has_arena_keeper(), c.leases, memory);
    if (!c.manifests->local_dependencies_admitted_in(*arena) &&
        !arena->prepaid(c.manifests, c.manifests->allocation_budget(), c.manifests->metadata_bytes(), false))
        c.manifests->visit_dependencies<CartesianDependencyVisit::Physical>(
            [&](const auto& owner, auto origin, auto bytes, auto keeper)
            { retain_owner(credits, owner, origin, bytes, keeper, c.leases, memory); });
    c.budget = &budget;
    c.keeper = arena->keeper;
    c.charge = std::exchange(memory.bytes, 0);
    credits.committed = true;
}
CartesianSourceLineRangeBuilder::~CartesianSourceLineRangeBuilder() = default;
void CartesianSourceLineRangeBuilder::freeze()
{
    auto& c = *impl_->catalog->impl_;
    if (c.frozen)
        return;
    Allocation memory(*c.budget);
    memory.bytes = std::exchange(c.charge, 0);
    struct RestoreCharge
    {
        Allocation& memory;
        std::size_t& charge;
        ~RestoreCharge() { charge = std::exchange(memory.bytes, 0); }
    } restore{memory, c.charge};
    const auto exact = [&]<class T>(std::vector<T>& buffer)
    {
        if (buffer.size() == buffer.capacity())
            return;
        const auto old = product(buffer.capacity(), sizeof(T));
        {
            std::vector<T> replacement;
            memory.reserve(replacement, buffer.size());
            replacement.assign(buffer.begin(), buffer.end());
            buffer.swap(replacement);
        }
        // Only released live storage is subtracted. Budget cumulative remains
        // monotone and includes both arrays throughout the copying overlap.
        memory.bytes -= old;
        memory.budget.used -= old;
    };
    exact(c.prototypes);
    exact(c.prototype_bytes);
    c.frozen = true;
}

std::shared_ptr<const CartesianSourceLineRangeRecipe> CartesianSourceLineRangeBuilder::make_recipe(std::size_t line,
    std::span<const PooledLineSubscriber> values, const IncidencePools& pools, unsigned certified_layer)
{
    auto& c = *impl_->catalog->impl_;
    if (c.frozen)
        throw std::invalid_argument("Source line range catalog is already immutable.");
    // Saved output from an older explicit/mixed generation may lie beyond the
    // current fixed point. Its original exception remains authoritative; this
    // current admission cannot rebind it. Empty real completions retain no
    // subscriber/result arrays and need no additional source recipe.
    if (certified_layer > c.layer || values.empty() || pools.points.source() != c.points)
        return {};
    auto& scratch = impl_->memory;
    auto& components = impl_->components;
    components.clear();
    auto& parsed = impl_->parsed;
    parsed.clear();
    auto& result_offsets = impl_->result_offsets;
    result_offsets.clear();
    auto& result_components = impl_->result_components;
    result_components.clear();
    auto& consumed = impl_->consumed;
    consumed.clear();
    auto& drafts = impl_->drafts;
    drafts.clear();
    auto& masks = impl_->masks;
    masks.clear();
    auto& ranges = impl_->ranges;
    ranges.clear();
    auto& results = impl_->results;
    results.clear();
    scratch.reserve(components, values.size());
    scratch.reserve(parsed, values.size());
    scratch.reserve(consumed, values.size());
    const bool explicit_results = std::ranges::any_of(
        values, [](const auto& value) { return value.result_state == PooledLineSubscriber::ResultState::Explicit; });
    if (explicit_results)
        scratch.reserve(result_offsets, values.size());
    for (const auto& value : values)
    {
        if (!value.native_source || !value.source_generation || !value.has_result())
            return {};
        const auto support_key = pools.supports.get(value.support);
        const auto support = parse_support(support_key.bytes(), *c.manifests);
        const auto first = c.points->component_indices({value.first}), last = c.points->component_indices({value.last});
        if (!support || !first || !last || !c.points->find_components(*first) || !c.points->find_components(*last))
            return {};
        parsed.push_back(*support);
        components.push_back({(*first)[0], (*first)[1], (*first)[2], (*last)[0], (*last)[1], (*last)[2]});
        consumed.push_back(0);
        if (explicit_results)
            result_offsets.push_back(result_components.size());
        if (value.result_state == PooledLineSubscriber::ResultState::Explicit)
            for (std::size_t output = 0; output < value.result_size(); ++output)
            {
                const auto selected = c.points->component_indices({value.result_at(output)});
                if (!selected || !c.points->find_components(*selected))
                    return {};
                scratch.append(result_components);
                result_components.push_back(*selected);
            }
    }
    Allocation memory(*c.budget);
    memory.retain(sizeof(CartesianSourceLineRangeRecipe) + sizeof(CartesianSourceLineRangeRecipe::Impl));
    auto [result, recipe_control] =
        make_controlled(*c.budget, c.keeper, [] { return new CartesianSourceLineRangeRecipe; });
    result->impl_ = std::make_unique<CartesianSourceLineRangeRecipe::Impl>();
    auto& r = *result->impl_;
    r.control_credit = recipe_control.get();
    r.catalog = impl_->catalog;
    r.count = values.size();
    r.line_slot = line;
    r.certified_layer = certified_layer;
    memory.copy(r.line, pools.lines.get(line));
    auto& members = impl_->members;
    auto& best = impl_->best;
    scratch.reserve(members, values.size());
    scratch.reserve(best, values.size());
    for (std::size_t begin = 0; begin < values.size(); ++begin)
    {
        if (consumed[begin])
            continue;
        const auto prototype = pools.supports.get(values[begin].support);
        std::size_t best_field = 0;
        best.clear();
        // Exhaust every original integer key field. Both endpoint orientations
        // are represented independently by the six actual component columns.
        for (std::size_t field = 0; field < 3; ++field)
        {
            members.clear();
            const auto offset = parsed[begin].offset + field * 8;
            for (std::size_t i = begin; i < values.size(); ++i)
            {
                if (consumed[i] || !constants_equal(values[begin], values[i]) ||
                    values[begin].result_state != values[i].result_state ||
                    values[begin].result_size() != values[i].result_size())
                    continue;
                const auto key = pools.supports.get(values[i].support);
                if (key.size() != prototype.size())
                    continue;
                bool same = true;
                for (std::size_t byte = 0; byte < key.size(); ++byte)
                    if ((byte < offset || byte >= offset + 8) && key[byte] != prototype[byte])
                    {
                        same = false;
                        break;
                    }
                if (same)
                    members.push_back(i);
            }
            if (members.size() > best.size())
            {
                best.assign(members.begin(), members.end());
                best_field = field;
            }
        }
        scratch.append(drafts);
        drafts.emplace_back();
        auto& draft = drafts.back();
        auto& f = draft.value;
        // Intern only a complete byte-exact prototype. The replaced word is
        // deliberately not normalized: every catalog entry is an actual key.
        std::size_t prototype_index = 0;
        for (; prototype_index != c.prototypes.size(); ++prototype_index)
            if (std::ranges::equal(c.prototype(prototype_index), prototype))
                break;
        if (prototype_index == c.prototypes.size())
        {
            const auto byte_end = add(c.prototype_bytes.size(), prototype.size());
            Allocation catalog_memory(*c.budget);
            catalog_memory.bytes = std::exchange(c.charge, 0);
            struct RestorePrototypeCharge
            {
                Allocation& memory;
                std::size_t& charge;
                ~RestorePrototypeCharge() { charge = std::exchange(memory.bytes, 0); }
            } restore{catalog_memory, c.charge};
            catalog_memory.append(c.prototypes);
            const auto capacity = c.prototype_bytes.capacity();
            if (byte_end > capacity)
                catalog_memory.reserve(
                    c.prototype_bytes, std::max(byte_end, capacity ? product(capacity, 2) : byte_end));
            c.prototypes.push_back({c.prototype_bytes.size(), prototype.size()});
            c.prototype_bytes.insert(c.prototype_bytes.end(), prototype.begin(), prototype.end());
        }
        c.prototype_requests = add(c.prototype_requests, 1);
        f.prototype = prototype_index;
        f.offset = parsed[begin].offset + best_field * 8;
        f.count = best.size();
        f.resident = values[begin].resident;
        f.authority = values[begin].support_authority;
        f.emission = values[begin].emission;
        f.image = values[begin].image;
        f.edge = values[begin].edge;
        f.generation = values[begin].source_generation;
        f.result_state = values[begin].result_state;
        std::sort(best.begin(), best.end(),
            [&](auto a, auto b)
            {
                return word(pools.supports.get(values[a].support), f.offset) <
                       word(pools.supports.get(values[b].support), f.offset);
            });
        f.lower = word(pools.supports.get(values[best.front()].support), f.offset);
        f.upper = word(pools.supports.get(values[best.back()].support), f.offset);
        std::size_t range_count = 0;
        std::uint64_t previous = 0;
        for (auto i : best)
        {
            const auto k = word(pools.supports.get(values[i].support), f.offset);
            if (range_count && k == previous)
                throw std::invalid_argument("Duplicate source-coordinate occurrence in a line family.");
            if (!range_count || previous == std::numeric_limits<std::uint64_t>::max() || k != previous + 1)
                ++range_count;
            previous = k;
            consumed[i] = 1;
        }
        const auto distance = f.upper - f.lower;
        const auto words = distance / 64 + 1;
        if (range_count == 1)
            f.membership = Membership::Contiguous;
        else if (words == 1)
        {
            f.membership = Membership::InlineMask;
            for (auto i : best)
                f.inline_mask |= std::uint64_t(1) << (word(pools.supports.get(values[i].support), f.offset) - f.lower);
        }
        else if (words <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) &&
                 words * sizeof(std::uint64_t) < product(range_count, sizeof(WordRange)))
        {
            f.membership = Membership::Mask;
            f.membership_offset = masks.size();
            f.membership_count = std::size_t(words);
            const auto end = add(masks.size(), f.membership_count);
            scratch.reserve(masks, end);
            masks.resize(end);
            for (auto i : best)
            {
                const auto delta = word(pools.supports.get(values[i].support), f.offset) - f.lower;
                masks[f.membership_offset + std::size_t(delta / 64)] |= std::uint64_t(1) << (delta % 64);
            }
        }
        else
        {
            f.membership = Membership::Ranges;
            f.membership_offset = ranges.size();
            f.membership_count = range_count;
            scratch.reserve(ranges, add(ranges.size(), range_count));
            for (auto i : best)
            {
                const auto k = word(pools.supports.get(values[i].support), f.offset);
                if (ranges.size() == f.membership_offset ||
                    ranges.back().last == std::numeric_limits<std::uint64_t>::max() || k != ranges.back().last + 1)
                    ranges.push_back({k, k});
                else
                    ranges.back().last = k;
            }
        }
        const auto certify_column = [&](std::size_t column, auto component) -> std::pair<std::size_t, std::size_t>
        {
            const auto fixed = component(best.front());
            if (std::ranges::all_of(best, [&](auto i) { return component(i) == fixed; }))
                return {fixed, absent};
            const std::array<std::uint64_t, 8> kind{parsed[begin].normal, parsed[begin].level, best_field, f.image,
                f.emission, f.edge, column, f.generation};
            std::size_t chosen = absent;
            for (std::size_t map = 0; map < c.maps.size(); ++map)
            {
                if (c.maps[map].kind != kind)
                    continue;
                bool matches = true;
                for (auto i : best)
                {
                    const auto k = word(pools.supports.get(values[i].support), f.offset);
                    const auto& entries = c.maps[map].entries;
                    const auto at = std::lower_bound(entries.begin(), entries.end(), k,
                        [](const auto& entry, auto value) { return entry.coordinate < value; });
                    if (at != entries.end() && at->coordinate == k && at->component != component(i))
                    {
                        matches = false;
                        break;
                    }
                }
                if (matches)
                {
                    chosen = map;
                    break;
                }
            }
            Allocation maps(*c.budget);
            maps.bytes = std::exchange(c.charge, 0);
            struct RestoreCharge
            {
                Allocation& allocation;
                std::size_t& charge;
                ~RestoreCharge() { charge = std::exchange(allocation.bytes, 0); }
            } restore{maps, c.charge};
            if (chosen == absent)
            {
                maps.append(c.maps);
                chosen = c.maps.size();
                c.maps.push_back({kind, {}});
            }
            auto& entries = c.maps[chosen].entries;
            for (auto i : best)
            {
                const auto k = word(pools.supports.get(values[i].support), f.offset);
                const auto at = std::lower_bound(entries.begin(), entries.end(), k,
                    [](const auto& entry, auto value) { return entry.coordinate < value; });
                if (at == entries.end() || at->coordinate != k)
                {
                    const auto offset = at - entries.begin();
                    maps.append(entries);
                    entries.insert(entries.begin() + offset, {k, component(i)});
                }
            }
            return {fixed, chosen};
        };
        for (std::size_t column = 0; column < 6; ++column)
        {
            std::tie(draft.fixed[column], draft.maps[column]) =
                certify_column(column, [&](auto i) { return components[i][column]; });
            const bool mapped = draft.maps[column] != absent;
            f.selectors[column] = mapped ? draft.maps[column] : draft.fixed[column];
            if (mapped)
                f.map_bits |= std::uint8_t(1u << column);
        }
        if (f.result_state == PooledLineSubscriber::ResultState::Explicit)
        {
            f.result_offset = results.size();
            f.result_count = values[begin].result_size();
            scratch.reserve(results, add(results.size(), f.result_count));
            for (std::size_t output = 0; output < f.result_count; ++output)
            {
                SelectorView selected;
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    const auto [fixed, map] = certify_column(add(6, add(product(output, 3), axis)),
                        [&](auto i) { return result_components[result_offsets[i] + output][axis]; });
                    selected.selectors[axis] = map == absent ? fixed : map;
                    if (map != absent)
                        selected.map_bits |= std::uint8_t(1u << axis);
                }
                results.push_back(selected);
            }
        }
    }
    // Check all actual values and cumulative ends before allocating either
    // published alternative. No sentinel consumes a selector value or bit.
    bool compact = true;
#ifdef SIMPLEFLUID_CARTESIAN_LINE_FORCE_WIDE
    compact = false;
#endif
    const auto fits = [&](std::size_t value)
    { compact = compact && cartesian_line_detail::compact_indices_fit({checked_index<std::uint64_t>(value)}); };
    fits(drafts.size());
    fits(masks.size());
    fits(ranges.size());
    fits(results.size());
    for (const auto& draft : drafts)
    {
        const auto& f = draft.value;
        fits(f.prototype);
        fits(f.offset);
        fits(f.count);
        fits(f.membership_offset);
        fits(f.membership_count);
        fits(add(f.membership_offset, f.membership_count));
        fits(f.result_offset);
        fits(f.result_count);
        fits(add(f.result_offset, f.result_count));
        for (auto selector : f.selectors)
            fits(selector);
    }
    for (const auto& selected : results)
        for (auto selector : selected.selectors)
            fits(selector);
    // Allocation::reserve admits old/new overlap and rejects capacity excess.
    // Every retained family/selector/membership array has exact final capacity.
    if (compact)
    {
        memory.reserve(r.compact, drafts.size());
        memory.reserve(r.compact_results, results.size());
        for (const auto& draft : drafts)
            r.compact.push_back(CompactFamily::copy(draft.value));
        for (const auto& selected : results)
            r.compact_results.push_back({{{std::uint32_t(selected.selectors[0]), std::uint32_t(selected.selectors[1]),
                                             std::uint32_t(selected.selectors[2])}},
                selected.map_bits});
    }
    else
    {
        memory.reserve(r.wide, drafts.size());
        memory.reserve(r.wide_results, results.size());
        for (const auto& draft : drafts)
            r.wide.push_back(WideFamily::copy(draft.value));
        for (const auto& selected : results)
            r.wide_results.push_back({{{checked_index<std::uint64_t>(selected.selectors[0]),
                                          checked_index<std::uint64_t>(selected.selectors[1]),
                                          checked_index<std::uint64_t>(selected.selectors[2])}},
                selected.map_bits});
    }
    memory.reserve(r.masks, masks.size());
    r.masks.assign(masks.begin(), masks.end());
    memory.reserve(r.ranges, ranges.size());
    r.ranges.assign(ranges.begin(), ranges.end());
    r.budget = c.budget;
    r.keeper = c.keeper;
    r.charge = std::exchange(memory.bytes, 0);
    return result;
}

std::pair<std::size_t, std::uint64_t> CartesianSourceLineRangeRecipe::locate(
    std::size_t index, const IncidencePools& pools) const
{
    if (pools.lines.get(impl_->line_slot) != impl_->line)
        throw std::invalid_argument("Source result has a foreign full line identity.");
    return locate_captured(index);
}
std::pair<std::size_t, std::uint64_t> CartesianSourceLineRangeRecipe::locate_captured(std::size_t index) const
{
    const auto& r = *impl_;
    const auto& c = *r.catalog->impl_;
    if (!c.frozen || index >= r.count)
        throw std::invalid_argument("Invalid immutable source line range replay binding.");
    // Locate by FULL original comparator. Counting predecessors avoids a stored
    // subscriber permutation and uses only bounded stack scalars at each read.
    std::size_t selected = absent;
    std::uint64_t coordinate = 0;
    for (std::size_t fi = 0; fi != r.family_count(); ++fi)
    {
        const auto f = r.family(fi);
        std::size_t lower = 0, upper = f.count;
        while (lower < upper)
        {
            const auto middle = lower + (upper - lower) / 2;
            const auto k = r.coordinate(f, middle);
            std::size_t rank = middle;
            for (std::size_t oi = 0; oi != r.family_count(); ++oi)
            {
                if (oi == fi)
                    continue;
                const auto other = r.family(oi);
                std::size_t a = 0, b = other.count;
                while (a < b)
                {
                    const auto m = a + (b - a) / 2;
                    if (family_before(other, r.coordinate(other, m), c.prototype(other.prototype), f, k,
                            c.prototype(f.prototype)))
                        a = m + 1;
                    else
                        b = m;
                }
                rank = add(rank, a);
            }
            if (rank == index)
            {
                selected = fi;
                coordinate = k;
                break;
            }
            if (rank < index)
                lower = middle + 1;
            else
                upper = middle;
        }
        if (selected != absent)
            break;
    }
    if (selected == absent)
        throw std::invalid_argument("Source line range ordinal has no exact ordered record.");
    return {selected, coordinate};
}
std::size_t CartesianSourceLineRangeRecipe::resolve_captured_point(
    const std::array<std::size_t, 3>& selectors, std::uint8_t map_bits, std::uint64_t coordinate) const
{
    const auto& c = *impl_->catalog->impl_;
    auto components = selectors;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (map_bits & (1u << axis))
        {
            const auto& entries = c.maps.at(selectors[axis]).entries;
            const auto at = std::lower_bound(entries.begin(), entries.end(), coordinate,
                [](const auto& entry, auto value) { return entry.coordinate < value; });
            if (at == entries.end() || at->coordinate != coordinate)
                throw std::invalid_argument("Source line range lacks an actual component mapping.");
            components[axis] = at->component;
        }
    const auto point = c.points->find_components(components);
    if (!point)
        throw std::invalid_argument("Source line range lost its full point identity.");
    return *point;
}
std::size_t CartesianSourceLineRangeRecipe::resolve_point(const std::array<std::size_t, 3>& selectors,
    std::uint8_t map_bits, std::uint64_t coordinate, const IncidencePools& pools) const
{
    const auto& c = *impl_->catalog->impl_;
    const auto point = resolve_captured_point(selectors, map_bits, coordinate);
    const auto slot = pools.points.source() == c.points ? std::optional<std::size_t>(point)
                                                        : pools.points.find(c.points->value({point}));
    if (!slot)
        throw std::invalid_argument("Source line range point is absent in destination pools.");
    return *slot;
}
PooledLineSubscriber CartesianSourceLineRangeRecipe::at(std::size_t index, const IncidencePools& pools) const
{
    const auto [family, coordinate] = locate(index, pools);
    const auto f = impl_->family(family);
    const auto support = pools.supports.find_borrowed(
        CartesianSupportRangeKeyView{impl_->catalog->impl_->prototype(f.prototype), f.offset, coordinate});
    if (!support)
        throw std::invalid_argument("Source line range lost its full support identity.");
    PooledLineSubscriber value;
    value.support = *support;
    const auto first = f.endpoint(0), last = f.endpoint(3);
    value.first = resolve_point(first.selectors, first.map_bits, coordinate, pools);
    value.last = resolve_point(last.selectors, last.map_bits, coordinate, pools);
    value.resident = f.resident;
    value.support_authority = f.authority;
    value.emission = f.emission;
    value.image = f.image;
    value.edge = f.edge;
    value.source_generation = f.generation;
    value.native_source = true;
    value.result_state = f.result_state;
    return value;
}
const std::shared_ptr<const ContactSourcePointSnapshot>&
CartesianSourceLineRangeRecipe::captured_point_snapshot() const noexcept
{
    return impl_->catalog->impl_->points;
}
std::size_t CartesianSourceLineRangeRecipe::captured_result_size(std::size_t index) const
{
    const auto [family, coordinate] = locate_captured(index);
    (void) coordinate;
    const auto f = impl_->family(family);
    return f.result_state == PooledLineSubscriber::ResultState::Explicit ? f.result_count : 2;
}
std::size_t CartesianSourceLineRangeRecipe::captured_result_at(std::size_t index, std::size_t output) const
{
    const auto [family, coordinate] = locate_captured(index);
    const auto f = impl_->family(family);
    if (f.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        if (output >= f.result_count)
            throw std::out_of_range("Invalid captured source result ordinal.");
        const auto selected = impl_->result(add(f.result_offset, output));
        return resolve_captured_point(selected.selectors, selected.map_bits, coordinate);
    }
    if (output >= 2)
        throw std::out_of_range("Invalid captured source result ordinal.");
    const auto selected = f.endpoint(output ? 3u : 0u);
    return resolve_captured_point(selected.selectors, selected.map_bits, coordinate);
}
std::size_t CartesianSourceLineRangeRecipe::result_size(std::size_t index, const IncidencePools& pools) const
{
    const auto [family, coordinate] = locate(index, pools);
    (void) coordinate;
    const auto f = impl_->family(family);
    return f.result_state == PooledLineSubscriber::ResultState::Explicit ? f.result_count : 2;
}
std::size_t CartesianSourceLineRangeRecipe::result_at(
    std::size_t index, std::size_t output, const IncidencePools& pools) const
{
    const auto [family, coordinate] = locate(index, pools);
    const auto f = impl_->family(family);
    if (f.result_state == PooledLineSubscriber::ResultState::Explicit)
    {
        if (output >= f.result_count)
            throw std::out_of_range("Invalid source result ordinal.");
        const auto selected = impl_->result(add(f.result_offset, output));
        return resolve_point(selected.selectors, selected.map_bits, coordinate, pools);
    }
    if (output >= 2)
        throw std::out_of_range("Invalid source forward-result point ordinal.");
    const auto selected = f.endpoint(output ? 3u : 0u);
    return resolve_point(selected.selectors, selected.map_bits, coordinate, pools);
}

bool CartesianSourceLineRangeRecipe::exact_replay(
    std::span<const PooledLineSubscriber> values, const IncidencePools& pools) const
{
    std::fenv_t caller;
    if (std::feholdexcept(&caller))
        throw std::invalid_argument("Cannot isolate source line range identity replay.");
    struct RestoreEnvironment
    {
        const std::fenv_t& caller;
        ~RestoreEnvironment() { (void) std::fesetenv(&caller); }
    } environment{caller};
    if (!impl_->catalog->impl_->frozen || pools.lines.get(impl_->line_slot) != impl_->line || values.size() != size())
        return false;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const auto& actual = values[i];
        const auto replayed = at(i, pools);
        if (!constants_equal(actual, replayed) || actual.native_source != replayed.native_source ||
            actual.result_state != replayed.result_state || actual.result_size() != result_size(i, pools) ||
            pools.supports.get(actual.support) != pools.supports.get(replayed.support) ||
            !full_point_equal(pools.points.get(actual.first), pools.points.get(replayed.first)) ||
            !full_point_equal(pools.points.get(actual.last), pools.points.get(replayed.last)))
            return false;
        for (std::size_t j = 0; j < actual.result_size(); ++j)
            if (!full_point_equal(pools.points.get(actual.result_at(j)), pools.points.get(result_at(i, j, pools))))
                return false;
    }
    return true;
}
void CartesianSourceLineRangeRecipe::retain_dependencies(cartesian_detail::CreditConstruction& credits,
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases, Allocation& memory) const
{
    const auto& catalog = impl_->catalog;
    const auto& c = *catalog->impl_;
    if (!credits.descend(catalog, c.budget, catalog->metadata_bytes(), bool(c.keeper), leases, memory))
        return;
    const auto visit = [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
    {
        using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
        const std::shared_ptr<const std::remove_const_t<Element>> stable = owner;
        return credits.descend(stable, origin, bytes, keeper, leases, memory);
    };
    if (credits.descend(c.points, c.points->allocation_budget(), c.points->metadata_bytes(),
            c.points->has_arena_keeper(), leases, memory))
    {
        if (credits.observer)
            c.points->visit_dependencies<CartesianDependencyVisit::Observed>(visit);
        else
            c.points->visit_dependencies<CartesianDependencyVisit::Physical>(visit);
    }
    if (credits.descend(c.manifests, c.manifests->allocation_budget(), c.manifests->metadata_bytes(),
            c.manifests->has_arena_keeper(), leases, memory))
    {
        if (credits.observer)
            c.manifests->visit_dependencies<CartesianDependencyVisit::Observed>(visit);
        else
            c.manifests->visit_dependencies<CartesianDependencyVisit::Physical>(visit);
    }
    for (const auto& lease : c.leases)
        retain_owner(credits, lease, lease->budget, lease->metadata, bool(lease->keeper), leases, memory);
}
bool CartesianSourceLineRangeRecipe::imports_owner(const void* identity) const noexcept
{
    const auto& c = *impl_->catalog->impl_;
    bool found = false;
    const auto inspect = [&](const auto& owner, const Budget*, std::size_t, bool)
    { found = found || owner.get() == identity; };
    c.points->visit_dependencies(inspect);
    c.manifests->visit_dependencies(inspect);
    for (const auto& lease : c.leases)
        found = found || lease.get() == identity;
    return found;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
