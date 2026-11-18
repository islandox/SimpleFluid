#include "geometry/PLIC/details/NativeGeometryByteWriter.hh"
#include "geometry/PLIC/details/NativePlanarCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "AdaptiveSubfaceSnapshot.hh"
#include "CertifiedIncidenceViews.hh"
#include "ContactPointPageView.hh"
#include "IncidencePhysicalOwners.hh"
#include "NativePublicationOwnerGraph.hh"
#include "IncidencePools.hh"
#include "PersistentOrderedPages.hh"
#include "TileBodyPages.hh"
#include "TileRecipeCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <tuple>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t page_records = 64, control_bytes = 128;
constexpr auto absent = std::numeric_limits<std::size_t>::max();
constexpr auto no_native_face = std::numeric_limits<std::uint64_t>::max();
using CellView = IdentityPool<AdaptiveCellKey>::PageView;
using BytesView = IdentityPool<AdaptiveBytes>::PageView;
using PlaneView = IdentityPool<IncidencePools::Plane>::PageView;
using NormalView = IdentityPool<IncidencePools::Normal>::PageView;
using ImageView = IdentityPool<IncidencePools::Image>::PageView;
using contact_funding_detail::make_controlled;

std::size_t checked_sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Subface admission size overflows.");
    return a + b;
}
std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Subface admission size overflows.");
    return a * b;
}
struct PageCharge
{
    std::shared_ptr<Budget> arena;
    std::size_t bytes = 0;
    explicit PageCharge(std::shared_ptr<Budget> owner) : arena(std::move(owner)) {}
    ~PageCharge() { arena->used -= bytes; }
    PageCharge(const PageCharge&) = delete;
    PageCharge& operator=(const PageCharge&) = delete;
    void add(std::size_t value)
    {
        const auto next = checked_sum(bytes, value);
        arena->add(value);
        bytes = next;
    }
    template<class T> void append(std::vector<T>& values)
    {
        const auto old = arena->used;
        try
        {
            arena->append_capacity(values);
        }
        catch (...)
        {
            bytes += arena->used - old;
            throw;
        }
        bytes += arena->used - old;
    }
};
template<class Owner>
void visit_owner(
    const std::shared_ptr<Owner>& owner, const Budget* origin, std::size_t bytes, const SubfaceOwnerVisit& visit)
{
    if (owner)
        visit(std::static_pointer_cast<const void>(owner), origin, bytes);
}
template<class View>
void visit_view(const std::shared_ptr<View>& view, const SubfaceOwnerVisit& visit, bool include_directory)
{
    if (!view)
        return;
    if (include_directory)
        visit_owner(view, view->origin_arena(), view->metadata_bytes(), visit);
    view->visit_credit_owners(
        [&](const auto& credit, std::size_t bytes) { visit_owner(credit, view->origin_arena(), bytes, visit); });
}

struct Member
{
    std::uint64_t key = 0, revision = 0, surface = 0, native_face = no_native_face;
    int owner = -1;
    bool copied = false;
    bool operator==(const Member&) const = default;
};
struct Group
{
    std::uint64_t support, domain, generation;
    std::size_t boundary;
    bool operator==(const Group&) const = default;
};
struct Provenance
{
    std::size_t group, first, second, shift;
    std::uint64_t emission;
    AdaptiveBytes authority_identity;
};
} // namespace

struct SubfaceProvenancePage
{
    PageCharge charge;
    std::shared_ptr<CellView> keys;
    std::shared_ptr<BytesView> supports, domains;
    std::vector<AdaptiveCellKey> foreign_keys;
    std::vector<Member> members;
    std::vector<Group> groups;
    std::vector<Provenance> records;
    std::vector<AdaptiveBytes> boundaries;
    std::vector<IncidencePools::Image> shifts;

    SubfaceProvenancePage(IncidencePools& pools, std::shared_ptr<Budget> arena)
        : charge(std::move(arena)), keys(pools.cells.page_view()), supports(pools.supports.page_view()),
          domains(pools.domains.page_view())
    {
    }
    [[nodiscard]] std::size_t charged_bytes() const { return checked_sum(charge.bytes, control_bytes); }
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const SubfaceProvenancePage>& owner, std::size_t bytes) const
    {
        return NativeContactFundingAccess::split_named<control_bytes>(owner, bytes,
            [](const SubfaceProvenancePage& value) { return value.charge.bytes; },
            [](const SubfaceProvenancePage& value) -> const auto& { return value.charge.arena; });
    }
    static bool project_body(const std::shared_ptr<const SubfaceProvenancePage>& owner,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
    {
        return owner && NativeContactFundingAccess::project(graph, owner, owner->charged_bytes(),
            owner->native_funding(owner, owner->charged_bytes()), retain);
    }
    const AdaptiveCellKey& key(const Member& member) const
    {
        return member.copied ? foreign_keys.at(member.key) : keys->get({member.key});
    }
    void freeze()
    {
        keys->freeze();
        supports->freeze();
        domains->freeze();
    }
    [[nodiscard]] std::size_t exclusive_bytes() const
    {
        return checked_sum(charged_bytes(),
            checked_sum(keys->metadata_bytes(), checked_sum(supports->metadata_bytes(), domains->metadata_bytes())));
    }
    void observe_shared_owners(IncidencePhysicalOwners& owners) const
    {
        owners.observe_pool_credits(keys);
        owners.observe_pool_credits(supports);
        owners.observe_pool_credits(domains);
    }
    void visit_dependencies(const SubfaceOwnerVisit& visit, bool include_directory) const
    {
        visit_view(keys, visit, include_directory);
        visit_view(supports, visit, include_directory);
        visit_view(domains, visit, include_directory);
    }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visit) const
    {
        contact_payload_object(visit, this, ContactPayloadRole::SubfaceProvenancePageObject, sizeof(*this), true);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceProvenanceRecords, records);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceMembers, members);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceGroups, groups);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceForeignKeys, foreign_keys);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceBoundaryNames, boundaries);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceShifts, shifts);
        for (const auto& record : records)
            contact_payload_buffer(
                visit, this, ContactPayloadRole::SubfaceAuthorityKeyBytes, record.authority_identity);
        for (const auto& key : foreign_keys)
            IdentityPoolBorrowedPayload<AdaptiveCellKey>::visit(key, visit);
        for (const auto& boundary : boundaries)
            contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceBoundaryBytes, boundary);
        keys->visit_borrowed_payload(visit);
        supports->visit_borrowed_payload(visit);
        domains->visit_borrowed_payload(visit);
    }
};

struct SubfaceProvenanceBuilder::Impl
{
    PageCharge charge;
    IncidencePools& pools;
    int resident;
    std::vector<std::shared_ptr<SubfaceProvenancePage>> pages;
    bool frozen = false;
    Impl(IncidencePools& source, std::shared_ptr<Budget> arena, int owner)
        : charge(std::move(arena)), pools(source), resident(owner)
    {
    }
};
SubfaceProvenanceBuilder::SubfaceProvenanceBuilder(IncidencePools& pools, std::shared_ptr<Budget> arena, int resident)
{
    if (!arena || arena.get() != pools.budget || resident < 0)
        throw std::invalid_argument("Unbound subface provenance arena.");
    PageCharge memory(arena);
    memory.add(sizeof(Impl));
    impl_ = std::make_unique<Impl>(pools, std::move(arena), resident);
    impl_->charge.bytes = std::exchange(memory.bytes, 0);
}
SubfaceProvenanceBuilder::~SubfaceProvenanceBuilder() = default;
SubfaceProvenanceRef SubfaceProvenanceBuilder::append(const ReceivedTileRecipe& recipe, std::uint64_t generation,
    const std::function<const AdaptiveLeaf&(const AdaptiveCellKey&)>& lookup)
{
    auto& state = *impl_;
    if (state.frozen || !recipe.face ||
        (recipe.first.owner != state.resident && (!recipe.second || recipe.second->owner != state.resident)))
        throw std::invalid_argument("Unauthenticated subface provenance.");
    if (state.pages.empty() || state.pages.back()->records.size() == page_records)
    {
        state.charge.append(state.pages);
        PageCharge memory(state.charge.arena);
        memory.add(sizeof(SubfaceProvenancePage));
        auto page = make_controlled<control_bytes>(*state.charge.arena, state.charge.arena,
            [&] { return new SubfaceProvenancePage(state.pools, state.charge.arena); }).first;
        page->charge.bytes = std::exchange(memory.bytes, 0);
        state.pages.push_back(std::move(page));
    }
    auto& page = *state.pages.back();
    const auto intern = [&]<class T>(std::vector<T>& values, const T& value)
    {
        const auto found = std::find(values.begin(), values.end(), value);
        if (found != values.end())
            return std::size_t(found - values.begin());
        page.charge.append(values);
        values.push_back(value);
        return values.size() - 1;
    };
    std::size_t boundary = absent;
    auto member = [&](const RegistryMember& incoming) -> std::pair<std::size_t, IncidencePools::Image>
    {
        Member value{0, incoming.revision, incoming.surface, no_native_face, incoming.owner, false};
        if (const auto slot = state.pools.cells.find(incoming.cell))
            value.key = page.keys->capture(state.pools.cells, *slot).value;
        else
        {
            if (incoming.owner == state.resident)
                throw std::invalid_argument("Subface is missing a resident cell identity.");
            const auto found = std::find(page.foreign_keys.begin(), page.foreign_keys.end(), incoming.cell);
            value.copied = true;
            if (found != page.foreign_keys.end())
                value.key = found - page.foreign_keys.begin();
            else
            {
                page.charge.append(page.foreign_keys);
                page.charge.add(key_capacity(incoming.cell));
                value.key = page.foreign_keys.size();
                page.foreign_keys.push_back(incoming.cell);
            }
        }
        IncidencePools::Image shift{};
        if (incoming.owner == state.resident)
        {
            const auto& leaf = lookup(incoming.cell);
            if (leaf.key != incoming.cell || leaf.revision != incoming.revision || leaf.domain != recipe.domain ||
                incoming.surface >= leaf.surfaces.size())
                throw std::invalid_argument("Subface source locator changed after authentication.");
            const auto& surface = leaf.surfaces[incoming.surface];
            if (surface.key != recipe.support || surface.exterior == bool(recipe.second))
                throw std::invalid_argument("Subface source support or exterior role differs.");
            value.native_face = surface.native_face;
            shift = surface.authority_image;
            if (surface.exterior)
            {
                const auto found = std::find_if(page.boundaries.begin(), page.boundaries.end(),
                    [&](const auto& value) { return std::ranges::equal(value, surface.boundary); });
                if (found != page.boundaries.end())
                    boundary = found - page.boundaries.begin();
                else
                {
                    page.charge.append(page.boundaries);
                    page.charge.add(surface.boundary.size());
                    boundary = page.boundaries.size();
                    page.boundaries.emplace_back(surface.boundary.begin(), surface.boundary.end());
                }
            }
        }
        return {intern(page.members, value), shift};
    };
    const auto [first, local_first_shift] = member(recipe.first);
    std::size_t second = absent;
    auto first_shift = local_first_shift;
    if (recipe.second)
    {
        const auto [index, local_second_shift] = member(*recipe.second);
        second = index;
        if (recipe.first.owner != state.resident)
            first_shift = PLIC::displacement_difference<ordinary>(local_second_shift, recipe.second_image);
    }
    const auto shift = first_shift.is_zero() ? absent : intern(page.shifts, first_shift);
    const auto support = state.pools.supports.find(recipe.support), domain = state.pools.domains.find(recipe.domain);
    if (!support || !domain)
        throw std::invalid_argument("Subface group is missing its pooled support or domain.");
    const auto group =
        intern(page.groups, Group{page.supports->capture(state.pools.supports, *support).value,
                                page.domains->capture(state.pools.domains, *domain).value, generation, boundary});
    // Capture the original authority identity at the delivery seam while the
    // already validated intrinsic face is available. Warm native numbering
    // borrows these bytes and never reconstructs that face merely for its key.
    PageCharge identity_scratch(state.charge.arena);
    const auto encoded = recipe.face->encoded_incidence_size();
    identity_scratch.add(checked_product(encoded, 6));
    identity_scratch.add(recipe.face->encoding_scratch_bytes());
    identity_scratch.add(
        checked_product(PLIC::PlanarFace<ordinary>::construction_capacity_bound(recipe.face->points().size()), 2));
    const auto authority = first_shift.is_zero() ? recipe.face : recipe.face->translated(first_shift);
    const auto reverse = authority->reversed();
    const auto forward_key = PLIC::canonical_face_key(*authority), reverse_key = PLIC::canonical_face_key(*reverse);
    const auto& physical_key = std::min(forward_key, reverse_key);
    const auto identity_bytes = checked_sum(16, checked_sum(recipe.support.size(), physical_key.size()));
    page.charge.add(identity_bytes);
    Writer identity;
    identity.data.reserve(identity_bytes);
    if (identity.data.capacity() != identity_bytes)
        throw std::invalid_argument("Native authority key capacity differs from admission.");
    identity.bytes(recipe.support);
    identity.bytes(physical_key);
    page.charge.append(page.records);
    page.records.push_back({group, first, second, shift, recipe.emission, std::move(identity.data)});
    return {state.pages.back(), page.records.size() - 1};
}
void SubfaceProvenanceBuilder::freeze()
{
    for (const auto& page : impl_->pages)
        page->freeze();
    impl_->frozen = true;
}

namespace
{
struct SharedRingControlCredit
{
    std::shared_ptr<Budget> keeper;
    std::size_t bytes = 0;
    static constexpr std::size_t allowance() noexcept { return control_bytes; }
    ~SharedRingControlCredit() { if (keeper) keeper->used -= bytes; }
};
struct SharedRingDelete
{
    std::shared_ptr<SharedRingControlCredit> control;
    void operator()(SharedSubfaceRing* ring) const noexcept { delete ring; }
};
#if defined(__GLIBCXX__)
static_assert(sizeof(std::_Sp_counted_ptr_inplace<SharedRingControlCredit, std::allocator<void>,
                  __gnu_cxx::__default_lock_policy>) +
    sizeof(std::_Sp_counted_deleter<SharedSubfaceRing*, SharedRingDelete, std::allocator<void>,
        __gnu_cxx::__default_lock_policy>) <= control_bytes,
    "Shared ring control requests exceed the existing unchanged control allowance.");
#endif
} // namespace
std::shared_ptr<const SharedSubfaceRing> SharedSubfaceRing::transfer(
    std::vector<std::size_t>& slots, Budget& budget, std::shared_ptr<Budget> arena, std::size_t& former_charge)
{
    const auto bytes = checked_product(slots.capacity(), sizeof(slots[0]));
    if (slots.empty() || !arena || arena.get() != &budget || bytes > former_charge)
        throw std::invalid_argument("Invalid subface ring ownership transfer.");
    PageCharge memory(arena);
    memory.add(sizeof(SharedSubfaceRing) + control_bytes);
    auto control = std::make_shared<SharedRingControlCredit>();
    control->keeper = arena;
    control->bytes = control_bytes;
    memory.bytes -= control_bytes; // Exact precharged weak-control obligation.
    auto* raw = new SharedSubfaceRing(std::move(arena));
    raw->charge_ = std::exchange(memory.bytes, 0) + control_bytes;
    auto result = std::shared_ptr<SharedSubfaceRing>(raw, SharedRingDelete{std::move(control)});
    result->self_ = result;
    result->slots_.swap(slots);
    former_charge -= bytes;
    result->charge_ += bytes;
    return result;
}
NativeContactOwnerFunding SharedSubfaceRing::native_funding(
    const std::shared_ptr<const SharedSubfaceRing>& actual, std::size_t bytes) const
{
    const auto* deleter = std::get_deleter<SharedRingDelete>(actual);
    const std::shared_ptr<const SharedRingControlCredit> group = deleter ? deleter->control : nullptr;
    return NativeContactFundingAccess::split(actual, bytes, group, control_bytes,
        [](const SharedSubfaceRing& ring) { return ring.charge_ - control_bytes; },
        [](const auto& ring, const auto& token)
        {
            const auto* actual_deleter = std::get_deleter<SharedRingDelete>(ring);
            return actual_deleter && actual_deleter->control == token && token->bytes == control_bytes &&
                token->keeper == ring->arena_ && !ring->self_.owner_before(ring) &&
                !ring.owner_before(ring->self_);
        });
}
void SharedSubfaceRing::project_owners(const std::shared_ptr<const SharedSubfaceRing>& ring,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
{
    if (ring)
        NativeContactFundingAccess::project(graph, ring, ring->charged_bytes(),
            ring->native_funding(ring, ring->charged_bytes()), retain);
}
SharedSubfaceRing::~SharedSubfaceRing()
{
    decltype(slots_){}.swap(slots_);
    if (charge_ < control_bytes) std::terminate();
    arena_->used -= charge_ - control_bytes;
}

void SharedSubfaceRing::visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visit) const
{
    contact_payload_object(visit, this, ContactPayloadRole::SubfaceRingObject, sizeof(*this), true);
    contact_payload_buffer(visit, this, ContactPayloadRole::SubfaceRingSlots, slots_);
    if (const auto actual = self_.lock())
        if (const auto* deleter = std::get_deleter<SharedRingDelete>(actual); deleter && deleter->control)
            contact_payload_object(visit, deleter->control.get(), ContactPayloadRole::CartesianControlCreditObject,
                sizeof(SharedRingControlCredit), true);
}

namespace
{
struct Piece
{
    SubfaceProvenanceRef provenance;
    TileBodyHandle body;
    std::shared_ptr<const SharedSubfaceRing> explicit_loop;
    std::optional<CertifiedIncidenceLoopView> certified;
    std::uint64_t plane, normal, image;
};
using PointPageRange = ContactPointPageView::CapturedPage;
} // namespace
struct SubfacePiecePage
{
    PageCharge charge;
    std::shared_ptr<PlaneView> planes;
    std::shared_ptr<NormalView> normals;
    std::shared_ptr<ImageView> images;
    std::shared_ptr<ContactPointPageView> physical;
    std::shared_ptr<const ContactSourcePointSnapshot> source;
    std::vector<PointPageRange> points;
    std::vector<Piece> records;
    std::fenv_t environment{};
    bool captured_environment = false;

    SubfacePiecePage(IncidencePools& pools, std::shared_ptr<Budget> arena)
        : charge(std::move(arena)), planes(pools.planes.page_view()), normals(pools.normals.page_view()),
          images(pools.images.page_view()), source(pools.points.source()),
          captured_environment(std::fegetenv(&environment) == 0)
    {
    }
    [[nodiscard]] std::size_t charged_bytes() const { return checked_sum(charge.bytes, control_bytes); }
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const SubfacePiecePage>& owner, std::size_t bytes) const
    {
        return NativeContactFundingAccess::split_named<control_bytes>(owner, bytes,
            [](const SubfacePiecePage& value) { return value.charge.bytes; },
            [](const SubfacePiecePage& value) -> const auto& { return value.charge.arena; });
    }
    static bool project_body(const std::shared_ptr<const SubfacePiecePage>& owner,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
    {
        return owner && NativeContactFundingAccess::project(graph, owner, owner->charged_bytes(),
            owner->native_funding(owner, owner->charged_bytes()), retain);
    }
    void capture(IncidencePointDirectory<>& directory, std::size_t slot)
    {
        const auto prefix = source ? source->size() : 0;
        if (slot < prefix)
            return; // Certified prefix already owns its exact immutable namespace.
        slot -= prefix;
        const auto after = std::upper_bound(points.begin(), points.end(), slot,
            [](std::size_t value, const auto& entry) { return value < entry.first_slot; });
        if (after != points.begin() && slot - std::prev(after)->first_slot < std::prev(after)->count)
            return;
        const auto offset = after - points.begin();
        if (!physical)
            physical = directory.physical_pool().page_view();
        const auto range = physical->capture_page(directory.physical_pool(), slot);
        charge.append(points);
        points.insert(points.begin() + offset, range);
    }
    ContactPointIdentity point(std::size_t slot) const
    {
        const auto prefix = source ? source->size() : 0;
        if (slot < prefix)
            return source->value({slot});
        slot -= prefix;
        const auto after = std::upper_bound(points.begin(), points.end(), slot,
            [](std::size_t value, const auto& entry) { return value < entry.first_slot; });
        if (after == points.begin() || slot - std::prev(after)->first_slot >= std::prev(after)->count)
            throw std::out_of_range("Uncaptured canonical subface point slot.");
        const auto& range = *std::prev(after);
        return physical->get({range.first_reference.value + slot - range.first_slot});
    }
    [[nodiscard]] std::size_t exclusive_bytes() const
    {
        return checked_sum(
            charged_bytes(), checked_sum(planes->metadata_bytes(),
                              checked_sum(normals->metadata_bytes(),
                                  checked_sum(images->metadata_bytes(), physical ? physical->metadata_bytes() : 0))));
    }
    template<class Visit> void visit_provenance_pages(const Visit& visit) const
    {
        std::array<const SubfaceProvenancePage*, page_records> seen{};
        std::size_t count = 0;
        for (const auto& record : records)
        {
            const auto& page = record.provenance.page;
            if (std::find(seen.begin(), seen.begin() + count, page.get()) != seen.begin() + count)
                continue;
            seen[count++] = page.get();
            visit(page);
        }
    }
    void observe_shared_owners(IncidencePhysicalOwners& owners) const
    {
        using Role = IncidencePhysicalOwners::Role;
        owners.observe_pool_credits(planes);
        owners.observe_pool_credits(normals);
        owners.observe_pool_credits(images);
        owners.observe_point_credits(physical);
        owners.observe_snapshot(source);
        for (const auto& record : records)
        {
            if (record.explicit_loop)
                owners.observe(Role::SubfaceRing, record.explicit_loop, record.explicit_loop->charged_bytes(),
                    record.explicit_loop->origin());
            record.body.observe_physical_owners(owners);
            if (record.certified)
                owners.observe_proof(*record.certified);
        }
    }
    static void project_certified(const CertifiedIncidenceLoopView& proof,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
    {
        Budget budget{std::numeric_limits<std::size_t>::max()};
        struct Observation
        {
            PLIC::details::GeometryOwnerGraphEditor& graph;
            bool retain;
            NativeContactOwnerFunding funding;
        } context{graph, retain, {}};
        cartesian_detail::CreditConstruction observer(budget, &context,
            [](void* raw, const std::shared_ptr<const void>& owner, const Budget*, std::size_t bytes,
                bool, bool, std::uint64_t)
            {
                auto& value = *static_cast<Observation*>(raw);
                const auto funding = std::exchange(value.funding, {});
                return NativeContactFundingAccess::project(value.graph, owner, bytes, funding, value.retain);
            },
            [](void* raw, const NativeContactOwnerFunding& funding)
            { static_cast<Observation*>(raw)->funding = funding; });
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> no_leases;
        cartesian_detail::Allocation no_storage(budget);
        proof.retain_dependencies(observer, no_leases, no_storage);
    }
    void visit_dependencies(const SubfaceOwnerVisit& visit, bool include_directory) const
    {
        visit_view(planes, visit, include_directory);
        visit_view(normals, visit, include_directory);
        visit_view(images, visit, include_directory);
        visit_view(physical, visit, include_directory);
        for (const auto& record : records)
        {
            if (record.explicit_loop)
                visit_owner(
                    record.explicit_loop, record.explicit_loop->origin(), record.explicit_loop->charged_bytes(), visit);
            visit_owner(record.body.body_storage_owner(), record.body.credit_origin(), record.body.body_bytes(), visit);
            visit_owner(record.body.credit_owner(), record.body.credit_origin(),
                record.body.credit_bytes() - record.body.body_bytes(), visit);
        }
    }
    void retain_source_dependencies(cartesian_detail::CreditConstruction& credits,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases,
        cartesian_detail::Allocation& memory) const
    {
        if (source)
        {
            credits.retain(source, source->allocation_budget(), source->metadata_bytes(), source->has_arena_keeper(),
                leases, memory);
            source->visit_dependencies<CartesianDependencyVisit::Physical>(
                [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keeper)
                {
                    using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
                    const std::shared_ptr<const std::remove_const_t<Element>> stable = owner;
                    credits.retain(stable, origin, bytes, keeper, leases, memory);
                });
        }
        // A page can keep a certified record after its corresponding live
        // tile and layer are gone. Import every physically held descriptor.
        for (const auto& record : records)
            if (record.certified)
                record.certified->retain_dependencies(credits, leases, memory);
    }
    [[nodiscard]] bool imports_source_owner(const void* identity) const noexcept
    {
        bool found = source.get() == identity;
        if (source && !found)
            source->visit_dependencies(
                [&](const auto& owner, const Budget*, std::size_t, bool) { found |= owner.get() == identity; });
        return found || std::ranges::any_of(records, [&](const auto& record)
                            { return record.certified && record.certified->imports_owner(identity); });
    }
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>& visit) const
    {
        contact_payload_object(visit, this, ContactPayloadRole::SubfacePiecePageObject, sizeof(*this), true);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfacePieceRecords, records);
        contact_payload_buffer(visit, this, ContactPayloadRole::SubfacePointRanges, points);
        planes->visit_borrowed_payload(visit);
        normals->visit_borrowed_payload(visit);
        images->visit_borrowed_payload(visit);
        if (physical)
            physical->visit_borrowed_payload(visit);
        if (source)
            source->visit_borrowed_payload(visit);
        for (const auto& record : records)
        {
            record.body.visit_borrowed_payload(visit);
            if (record.explicit_loop)
                record.explicit_loop->visit_borrowed_payload(visit);
            if (record.certified)
                record.certified->visit_borrowed_payload(visit);
        }
    }
};
namespace
{
void project_provenance(const std::shared_ptr<const SubfaceProvenancePage>& page,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
{
    if (!SubfaceProvenancePage::project_body(page, graph, retain)) return;
    CellView::project(page->keys, graph, retain);
    BytesView::project(page->supports, graph, retain);
    BytesView::project(page->domains, graph, retain);
    for (const auto& shift : page->shifts) graph.displacement_identity(shift, retain);
}
void project_snapshot(const std::shared_ptr<const ContactSourcePointSnapshot>& source,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
{
    ContactSourcePointSnapshot::project(source, graph, retain);
}
void project_piece_page(const std::shared_ptr<const SubfacePiecePage>& page,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
{
    if (!SubfacePiecePage::project_body(page, graph, retain)) return;
    PlaneView::project(page->planes, graph, retain);
    NormalView::project(page->normals, graph, retain);
    ImageView::project(page->images, graph, retain);
    ContactPointPageView::project(page->physical, graph, retain);
    project_snapshot(page->source, graph, retain);
    for (const auto& record : page->records)
    {
        // Each record owns its provenance field even when several slots share
        // a page. The page's direct children are first/last gated exactly once.
        project_provenance(record.provenance.page, graph, retain);
        SharedSubfaceRing::project_owners(record.explicit_loop, graph, retain);
        record.body.project_owners(graph, retain);
        if (record.certified)
            SubfacePiecePage::project_certified(*record.certified, graph, retain);
    }
}
} // namespace
void CanonicalSubfaceView::project_owners(PLIC::details::GeometryOwnerGraphEditor& graph, bool retain) const
{
    project_piece_page(page_, graph, retain);
}
PLIC::details::NativeWeakOwnerObservation CanonicalSubfaceView::weak_page_observation() const
{
    if (!page_) return {};
    return page_->native_funding(page_, page_->charged_bytes()).body.physical_observation();
}
struct SubfacePieceBuilder::Impl
{
    PageCharge charge;
    IncidencePools& pools;
    std::vector<std::shared_ptr<SubfacePiecePage>> pages;
    bool frozen = false;
    Impl(IncidencePools& source, std::shared_ptr<Budget> arena) : charge(std::move(arena)), pools(source) {}
};
SubfacePieceBuilder::SubfacePieceBuilder(IncidencePools& pools, std::shared_ptr<Budget> arena)
{
    if (!arena || arena.get() != pools.budget)
        throw std::invalid_argument("Unbound canonical subface arena.");
    PageCharge memory(arena);
    memory.add(sizeof(Impl));
    impl_ = std::make_unique<Impl>(pools, std::move(arena));
    impl_->charge.bytes = std::exchange(memory.bytes, 0);
}
SubfacePieceBuilder::~SubfacePieceBuilder() = default;
CanonicalSubfaceView SubfacePieceBuilder::append(const SubfaceProvenanceRef& provenance, const TileBodyHandle& body,
    std::shared_ptr<const SharedSubfaceRing> explicit_loop, const CertifiedIncidenceLoopView* certified)
{
    auto& state = *impl_;
    if (state.frozen || !provenance || !body || (explicit_loop && certified))
        throw std::invalid_argument("Invalid completed canonical subface.");
    if (state.pages.empty() || state.pages.back()->records.size() == page_records)
    {
        state.charge.append(state.pages);
        PageCharge memory(state.charge.arena);
        memory.add(sizeof(SubfacePiecePage));
        auto page = make_controlled<control_bytes>(*state.charge.arena, state.charge.arena,
            [&] { return new SubfacePiecePage(state.pools, state.charge.arena); }).first;
        page->charge.bytes = std::exchange(memory.bytes, 0);
        state.pages.push_back(std::move(page));
    }
    auto& page = *state.pages.back();
    for (const auto slot : body.intrinsic_loop())
        page.capture(state.pools.points, slot);
    if (explicit_loop)
        for (const auto slot : explicit_loop->slots())
            page.capture(state.pools.points, slot);
    const auto tile = body.scalars();
    page.charge.append(page.records);
    page.records.push_back(
        {provenance, body, std::move(explicit_loop), certified ? std::optional(*certified) : std::nullopt,
            page.planes->capture(state.pools.planes, tile.plane).value,
            page.normals->capture(state.pools.normals, tile.normal_slot).value,
            page.images->capture(state.pools.images, tile.image_slot).value});
    return {state.pages.back(), page.records.size() - 1};
}
void SubfacePieceBuilder::freeze()
{
    for (const auto& page : impl_->pages)
    {
        page->planes->freeze();
        page->normals->freeze();
        page->images->freeze();
        if (page->physical)
            page->physical->freeze();
    }
    impl_->frozen = true;
}

const AdaptiveCellKey& CanonicalSubfaceIncident::key() const
{
    return page_->key(page_->members.at(slot_));
}
int CanonicalSubfaceIncident::owner() const
{
    return page_->members.at(slot_).owner;
}
std::uint64_t CanonicalSubfaceIncident::revision() const
{
    return page_->members.at(slot_).revision;
}
std::uint64_t CanonicalSubfaceIncident::source_surface() const
{
    return page_->members.at(slot_).surface;
}
std::optional<std::uint64_t> CanonicalSubfaceIncident::native_face() const
{
    const auto value = page_->members.at(slot_).native_face;
    return value == no_native_face ? std::nullopt : std::optional(value);
}
namespace
{
const Piece& piece(const std::shared_ptr<const SubfacePiecePage>& page, std::size_t slot)
{
    if (!page)
        throw std::invalid_argument("Empty canonical subface view.");
    return page->records.at(slot);
}
const Provenance& provenance(const Piece& record)
{
    return record.provenance.page->records.at(record.provenance.slot);
}
const Group& group(const Piece& record)
{
    return record.provenance.page->groups.at(provenance(record).group);
}
} // namespace
CanonicalSubfaceIncident CanonicalSubfaceView::first() const
{
    const auto& record = piece(page_, slot_);
    return {record.provenance.page, provenance(record).first};
}
std::optional<CanonicalSubfaceIncident> CanonicalSubfaceView::second() const
{
    const auto& record = piece(page_, slot_);
    const auto slot = provenance(record).second;
    if (slot == absent)
        return std::nullopt;
    return CanonicalSubfaceIncident(record.provenance.page, slot);
}
std::span<const char> CanonicalSubfaceView::identity() const
{
    return provenance(piece(page_, slot_)).authority_identity;
}
std::span<const char> CanonicalSubfaceView::support() const
{
    const auto& record = piece(page_, slot_);
    return record.provenance.page->supports->get({group(record).support});
}
std::span<const char> CanonicalSubfaceView::domain() const
{
    const auto& record = piece(page_, slot_);
    return record.provenance.page->domains->get({group(record).domain});
}
std::string_view CanonicalSubfaceView::boundary() const
{
    const auto& record = piece(page_, slot_);
    const auto boundary = group(record).boundary;
    if (boundary == absent)
        return {};
    const auto& value = record.provenance.page->boundaries.at(boundary);
    return {value.data(), value.size()};
}
std::uint64_t CanonicalSubfaceView::generation() const
{
    return group(piece(page_, slot_)).generation;
}
std::uint64_t CanonicalSubfaceView::emission() const
{
    return provenance(piece(page_, slot_)).emission;
}
PLIC::DisplacementIdentity<ordinary> CanonicalSubfaceView::second_image() const
{
    const auto& record = piece(page_, slot_);
    return page_->images->get({record.image});
}
std::size_t CanonicalSubfaceView::point_count() const
{
    const auto& record = piece(page_, slot_);
    if (record.certified)
        return record.certified->size();
    if (record.explicit_loop)
        return record.explicit_loop->slots().size();
    return record.body.intrinsic_loop().size();
}
PLIC::PointIdentity<ordinary> CanonicalSubfaceView::point(std::size_t slot) const
{
    const auto& record = piece(page_, slot_);
    if (slot >= point_count())
        throw std::out_of_range("Canonical subface loop point is out of range.");
    if (record.certified)
        return record.certified->value(slot);
    return page_->point(
        record.explicit_loop ? record.explicit_loop->slots()[slot] : record.body.intrinsic_loop()[slot]);
}
MaterializedCanonicalSubface CanonicalSubfaceView::materialize(Budget& budget, IdentityBudget& identities) const
{
    struct RestoreEnvironment
    {
        std::fenv_t previous{};
        bool saved;
        explicit RestoreEnvironment(const SubfacePiecePage& page) : saved(std::fegetenv(&previous) == 0)
        {
            if (!page.captured_environment || !saved || std::fesetenv(&page.environment) != 0)
            {
                if (saved)
                    (void) std::fesetenv(&previous);
                throw std::invalid_argument("Canonical subface construction controls are unavailable.");
            }
        }
        ~RestoreEnvironment() { (void) std::fesetenv(&previous); }
    };
    using Face = PLIC::PlanarFace<ordinary>;
    const auto& record = piece(page_, slot_);
    const RestoreEnvironment environment(*page_);
    const auto intrinsic = record.body.intrinsic_loop();
    const auto count = point_count();
    struct Scratch
    {
        Budget& budget;
        std::size_t bytes = 0;
        ~Scratch() { budget.used -= bytes; }
        void add(std::size_t value)
        {
            const auto next = checked_sum(bytes, value);
            budget.add(value);
            bytes = next;
        }
    } scratch{budget};
    scratch.add(checked_product(std::max(intrinsic.size(), count), sizeof(ContactPointIdentity)));
    scratch.add(checked_product(Face::construction_capacity_bound(intrinsic.size()), 3));
    scratch.add(Face::construction_capacity_bound(count));
    std::vector<ContactPointIdentity> points;
    points.reserve(std::max(intrinsic.size(), count));
    for (const auto slot : intrinsic)
        points.push_back(page_->point(slot));
    const PLIC::OrdinaryIncidenceFrame frame{
        page_->planes->get({record.plane}), page_->normals->get({record.normal}).values};
    const auto initial = PLIC::restore_incidence_loop(frame, points);
    const auto shift = provenance(record).shift;
    const auto authority = shift == absent ? initial : initial->translated(record.provenance.page->shifts.at(shift));
    const auto reversed = authority->reversed();
    const auto encoded = authority->encoded_incidence_size();
    // Match the qualified authority-key envelope. Both directional keys
    // coexist with the vector growth of each encoding, including ancestry.
    scratch.add(checked_product(encoded, 6));
    scratch.add(authority->encoding_scratch_bytes());
    const auto forward = PLIC::canonical_face_key(*authority), backward = PLIC::canonical_face_key(*reversed);
    Writer identity;
    const auto& canonical = std::min(forward, backward);
    const auto identity_bytes = checked_sum(16, checked_sum(support().size(), canonical.size()));
    const auto output_bytes = checked_sum(identity_bytes, sizeof(Face));
    scratch.add(output_bytes);
    identity.data.reserve(identity_bytes);
    if (identity.data.capacity() != identity_bytes)
        throw std::invalid_argument("Canonical subface identity capacity differs from admission.");
    identity.bytes(support());
    identity.bytes(canonical);
    auto geometry = initial;
    if (record.explicit_loop || record.certified)
    {
        points.clear();
        for (std::size_t slot = 0; slot < count; ++slot)
            points.push_back(point(slot));
        geometry = PLIC::restore_incidence_loop(frame, points);
    }
    identities.face(budget, *geometry);
    scratch.bytes -= output_bytes;
    return {std::move(geometry), std::move(identity.data)};
}

NativeMaterializedCanonicalSubface CanonicalSubfaceView::materialize_native(
    Budget& budget, IdentityBudget& identities, const geometry_detail::NativeConstructionAdmission& admission,
    bool detached_support) const
{
    struct RestoreEnvironment
    {
        std::fenv_t previous{};
        explicit RestoreEnvironment(const SubfacePiecePage& page)
        {
            const bool saved = std::fegetenv(&previous) == 0;
            if (!page.captured_environment || !saved || std::fesetenv(&page.environment) != 0)
            {
                if (saved)
                    (void) std::fesetenv(&previous);
                throw std::invalid_argument("Canonical subface construction controls are unavailable.");
            }
        }
        ~RestoreEnvironment() { (void) std::fesetenv(&previous); }
    };
    using Face = PLIC::PlanarFace<ordinary>;
    using Access = PLIC::details::NativePlanarOwnerAccess;
    const auto& record = piece(page_, slot_);
    const RestoreEnvironment environment(*page_);
    const auto intrinsic = record.body.intrinsic_loop();
    const auto count = point_count();
    struct Scratch
    {
        Budget& budget;
        std::size_t bytes = 0;
        ~Scratch() { budget.used -= bytes; }
        void add(std::size_t value)
        {
            const auto next = checked_sum(bytes, value);
            budget.add(value);
            bytes = next;
        }
    } logical{budget};
    logical.add(checked_product(std::max(intrinsic.size(), count), sizeof(ContactPointIdentity)));
    logical.add(checked_product(Face::construction_capacity_bound(intrinsic.size()), 3));
    logical.add(Face::construction_capacity_bound(count));
    geometry_detail::NativeConstructorVector<Face::WidePoint> points(admission.allocator<Face::WidePoint>());
    points.reserve(std::max(intrinsic.size(), count));
    for (const auto slot : intrinsic)
        points.push_back(PLIC::details::GeometryIdentityAccess::point(page_->point(slot)));
    PLIC::OrdinaryIncidenceFrame frame{
        page_->planes->get({record.plane}), page_->normals->get({record.normal}).values};
    if (detached_support)
    {
        const auto storage = geometry_detail::native_constructor_storage(admission);
        std::vector<char> key;
        storage.owner->reserve(key, PLIC::canonical_key_size(frame.support));
        PLIC::details::NativeGeometryByteWriter writer{&key, &admission};
        writer.native_identity(frame.support);
        auto retained = PLIC::details::GeometryIdentityAccess::decode_support_admitted(key, budget.limit, admission);
        if (retained != frame.support)
            throw std::invalid_argument("Detached native face support differs from the actual canonical support.");
        frame.support = std::move(retained);
    }
    const auto initial = Access::retained(points, frame.normal, frame.support, admission);
    const auto shift = provenance(record).shift;
    const auto authority =
        shift == absent ? initial : Access::translated(*initial, record.provenance.page->shifts.at(shift), admission);
    const auto reversed = Access::reversed(*authority, admission);
    const auto encoded = authority->encoded_incidence_size();
    logical.add(checked_product(encoded, 6));
    logical.add(authority->encoding_scratch_bytes());
    const auto scratch = geometry_detail::native_constructor_storage(admission);
    const auto forward = PLIC::details::NativePlanarCodecAccess::canonical(*authority, scratch);
    const auto backward = PLIC::details::NativePlanarCodecAccess::canonical(*reversed, scratch);
    const auto& canonical = std::min(forward, backward);
    const auto identity_bytes = checked_sum(16, checked_sum(support().size(), canonical.size()));
    const auto output_bytes = checked_sum(identity_bytes, sizeof(Face));
    logical.add(output_bytes);
    // The already authenticated page owns this exact key. Verify its full
    // original construction bytes without creating another owning key buffer.
    Reader key{identity()};
    if (!std::ranges::equal(key.bytes_view(), support()) || !std::ranges::equal(key.bytes_view(), canonical))
        throw std::invalid_argument("Captured original subface key differs from full materializer.");
    key.done();
    auto geometry = initial;
    if (record.explicit_loop || record.certified)
    {
        points.clear();
        for (std::size_t slot = 0; slot < count; ++slot)
            points.push_back(PLIC::details::GeometryIdentityAccess::point(point(slot)));
        geometry = Access::retained(points, frame.normal, frame.support, admission);
    }
    identities.face(budget, *geometry);
    logical.bytes -= output_bytes; // Preserve the original numerical Budget contract.
    return {std::move(geometry), identity()};
}

namespace subface_index_detail
{
template<class Row, class Less, bool FixedLeafCapacity>
using OrderedPages = persistent_ordered_detail::OrderedPages<Row, Less, FixedLeafCapacity, SubfaceIndexWork>;
std::strong_ordering compare_piece(std::span<const char> a_support, std::uint64_t a_emission,
    std::span<const char> b_support, std::uint64_t b_emission)
{
    const auto support =
        std::lexicographical_compare_three_way(a_support.begin(), a_support.end(), b_support.begin(), b_support.end());
    return support == 0 ? a_emission <=> b_emission : support;
}
struct PieceLess
{
    bool operator()(const CanonicalSubfaceView& a, const CanonicalSubfaceView& b) const
    {
        return compare_piece(a.support(), a.emission(), b.support(), b.emission()) < 0;
    }
};
struct FaceRow
{
    const SubfacePiecePage* page = nullptr;
    std::uint8_t code = 0; // twice the page record plus its local side
    [[nodiscard]] const Piece& piece_record() const { return page->records[code / 2]; }
    [[nodiscard]] const Member& member() const
    {
        const auto& record = piece_record();
        const auto& proof = provenance(record);
        return record.provenance.page->members[code % 2 ? proof.second : proof.first];
    }
    [[nodiscard]] const AdaptiveCellKey& key() const { return piece_record().provenance.page->key(member()); }
    [[nodiscard]] std::uint64_t surface() const { return member().surface; }
    [[nodiscard]] std::span<const char> support() const
    {
        const auto& record = piece_record();
        return record.provenance.page->supports->get({group(record).support});
    }
    [[nodiscard]] std::uint64_t emission() const { return provenance(piece_record()).emission; }
};
struct FaceLess
{
    bool operator()(const FaceRow& a, const FaceRow& b) const
    {
        if (const auto key = a.key() <=> b.key(); key != 0)
            return key < 0;
        if (a.surface() != b.surface())
            return a.surface() < b.surface();
        const auto source = compare_piece(a.support(), a.emission(), b.support(), b.emission());
        return source == 0 ? a.code % 2 < b.code % 2 : source < 0;
    }
};
template<class Page> struct OwnerRow
{
    std::shared_ptr<const Page> page;
    std::size_t references = 0;
};
struct OwnerLess
{
    template<class Page> bool operator()(const OwnerRow<Page>& a, const OwnerRow<Page>& b) const noexcept
    {
        return std::less<const Page*>{}(a.page.get(), b.page.get());
    }
};
using PieceIndex = OrderedPages<CanonicalSubfaceView, PieceLess, true>;
using FaceIndex = OrderedPages<FaceRow, FaceLess, true>;
using PieceOwner = OwnerRow<SubfacePiecePage>;
using ProvenanceOwner = OwnerRow<SubfaceProvenancePage>;
using OwnerIndex = OrderedPages<PieceOwner, OwnerLess, true>;
using ProvenanceIndex = OrderedPages<ProvenanceOwner, OwnerLess, true>;
template<class Row> void reserve(std::vector<Row>& rows, std::size_t count, PageCharge& charge)
{
    charge.add(checked_product(count, sizeof(Row)));
    rows.reserve(count);
    if (rows.capacity() != count)
        throw std::invalid_argument("Subface query scratch capacity differs from admission.");
}
template<class Owner> std::span<const Owner> unique_owners(std::vector<Owner>& owners)
{
    std::sort(owners.begin(), owners.end(), OwnerLess{});
    std::size_t unique = 0;
    for (std::size_t i = 0; i < owners.size(); ++i)
        if (unique && owners[unique - 1].page == owners[i].page)
            owners[unique - 1].references += owners[i].references;
        else if (unique++ != i)
            owners[unique - 1] = std::move(owners[i]);
    return std::span(owners).first(unique);
}
} // namespace subface_index_detail
struct SubfaceIndexRoot
{
    PageCharge charge;
    subface_index_detail::PieceIndex::Ptr pieces;
    subface_index_detail::FaceIndex::Ptr faces;
    subface_index_detail::OwnerIndex::Ptr owners;
    subface_index_detail::ProvenanceIndex::Ptr provenance_owners;
    explicit SubfaceIndexRoot(std::shared_ptr<Budget> arena) : charge(std::move(arena)) {}
    static std::shared_ptr<SubfaceIndexRoot> create(const std::shared_ptr<Budget>& arena)
    {
        PageCharge memory(arena);
        memory.add(sizeof(SubfaceIndexRoot));
        auto result = make_controlled<control_bytes>(*arena, arena, [&] { return new SubfaceIndexRoot(arena); }).first;
        result->charge.bytes = std::exchange(memory.bytes, 0);
        return result;
    }
    [[nodiscard]] std::size_t charged_bytes() const { return checked_sum(charge.bytes, control_bytes); }
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const SubfaceIndexRoot>& owner, std::size_t bytes) const
    {
        return NativeContactFundingAccess::split_named<control_bytes>(owner, bytes,
            [](const SubfaceIndexRoot& value) { return value.charge.bytes; },
            [](const SubfaceIndexRoot& value) -> const auto& { return value.charge.arena; });
    }
    static bool project_body(const std::shared_ptr<const SubfaceIndexRoot>& owner,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
    {
        return owner && NativeContactFundingAccess::project(graph, owner, owner->charged_bytes(),
            owner->native_funding(owner, owner->charged_bytes()), retain);
    }
    [[nodiscard]] std::size_t index_bytes() const
    {
        auto result = charged_bytes();
        if (pieces)
            result = checked_sum(result, pieces->subtree_bytes);
        if (faces)
            result = checked_sum(result, faces->subtree_bytes);
        if (owners)
            result = checked_sum(result, owners->subtree_bytes);
        if (provenance_owners)
            result = checked_sum(result, provenance_owners->subtree_bytes);
        return result;
    }
    template<class Visit> bool visit_piece_pages(const Visit& visit) const
    {
        return subface_index_detail::OwnerIndex::visit_from(
            owners, [](const auto&) { return false; }, [&](const auto& row) { return visit(row.page); });
    }
    template<class Visit> void visit_provenance_pages(const Visit& visit) const
    {
        subface_index_detail::ProvenanceIndex::visit_from(
            provenance_owners, [](const auto&) { return false; },
            [&](const auto& row)
            {
                visit(row.page);
                return true;
            });
    }
};
namespace
{
std::shared_ptr<const geometry_detail::NativeAllocationState> subface_index_admission(
    const std::shared_ptr<Budget>& arena, bool admitted)
{
    if (!admitted) return {};
    const geometry_detail::NativeConstructionAdmission domain(
        geometry_detail::NativeConstructionCounterAccess::create<Budget>(
            arena->limit, 0, NativeSourceOwnerStore::mirror(arena)));
    return domain.state();
}
} // namespace
std::size_t AdaptiveSubfaceSnapshot::charged_bytes() const noexcept { return charge_ + control_bytes; }
NativeContactOwnerFunding AdaptiveSubfaceSnapshot::native_funding(
    const std::shared_ptr<const AdaptiveSubfaceSnapshot>& owner, std::size_t bytes) const
{
    return NativeContactFundingAccess::split_named<control_bytes>(owner, bytes,
        [](const AdaptiveSubfaceSnapshot& value) { return value.charge_; },
        [](const AdaptiveSubfaceSnapshot& value) -> const auto& { return value.arena_; });
}
PLIC::details::NativeWeakOwnerObservation AdaptiveSubfaceSnapshot::weak_observation(
    const std::shared_ptr<const AdaptiveSubfaceSnapshot>& snapshot)
{
    if (!snapshot) return {};
    return snapshot->native_funding(snapshot, snapshot->charged_bytes()).body.physical_observation();
}
void AdaptiveSubfaceSnapshot::project_legacy_scope(
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, const Budget* foreign_to) const
{
    using namespace subface_index_detail;
    using Funding = NativeContactFundingAccess;
    const auto& root = indexes_;
    const auto project = [&](const auto& owner, const Budget* origin, std::size_t bytes)
    {
        return Funding::project_reimported(graph, owner, Funding::capture(owner, bytes, origin), retain);
    };
    if (root->charge.arena.get() != foreign_to && !project(root, root->charge.arena.get(), root->charged_bytes()))
        return;
    const auto pages = [&]<class Index>(std::type_identity<Index> tag, const auto& self,
                           const typename Index::Ptr& node, const auto& row) -> void
    {
        if (!node || !graph.external(node, 0, retain)) return;
        const auto* origin = node->charge.arena.get();
        if (foreign_to != origin) project(node, origin, node->charge.bytes);
        for (const auto& child : node->children) self(tag, self, child, row);
        for (const auto& value : node->rows) row(value);
    };
    pages(std::type_identity<PieceIndex>{}, pages, root->pieces, [](const auto&) {});
    pages(std::type_identity<FaceIndex>{}, pages, root->faces, [](const auto&) {});
    pages(std::type_identity<OwnerIndex>{}, pages, root->owners, [&](const auto& row)
    {
        const auto& piece = row.page;
        if (piece->charge.arena.get() == foreign_to || !project(piece, piece->charge.arena.get(), piece->charged_bytes()))
            return;
        PlaneView::project_reimported(piece->planes, graph, retain);
        NormalView::project_reimported(piece->normals, graph, retain);
        ImageView::project_reimported(piece->images, graph, retain);
        ContactPointPageView::project_reimported(piece->physical, graph, retain);
    });
    pages(std::type_identity<ProvenanceIndex>{}, pages, root->provenance_owners, [&](const auto& row)
    {
        const auto& provenance = row.page;
        if (provenance->charge.arena.get() == foreign_to ||
            !project(provenance, provenance->charge.arena.get(), provenance->charged_bytes()))
            return;
        CellView::project_reimported(provenance->keys, graph, retain);
        BytesView::project_reimported(provenance->supports, graph, retain);
        BytesView::project_reimported(provenance->domains, graph, retain);
    });
}
void AdaptiveSubfaceSnapshot::seal_legacy_scope(const AdaptiveSubfaceSnapshot* previous,
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& native)
{
    if (!native_index_) return;
    const auto build = [&](const auto& prior, const Budget* foreign_to)
    {
        if (previous && previous->indexes_ == indexes_) return prior;
        std::size_t used = 0;
        PLIC::details::GeometryOwnerGraphEditor graph(prior, 0, used, arena_->limit, nullptr, nullptr, native);
        project_legacy_scope(graph, true, foreign_to);
        if (previous) previous->project_legacy_scope(graph, false, foreign_to);
        return graph.finish();
    };
    if (previous)
    {
        legacy_ = build(previous->legacy_, nullptr);
        const auto& prior = previous->origin() == origin() ? previous->inherited_legacy_ : previous->legacy_;
        inherited_legacy_ = build(prior, origin());
    }
    else legacy_ = build(legacy_, nullptr);
    if (legacy_.payload_bytes() != checked_sum(index_bytes_, exclusive_payload_bytes_) ||
        inherited_legacy_.payload_bytes() != checked_sum(foreign_index_bytes_, foreign_exclusive_payload_bytes_))
        throw std::logic_error("Native subface members differ from their original exact legacy admission scope.");
}
AdaptiveSubfaceSnapshot::~AdaptiveSubfaceSnapshot()
{
    indexes_.reset();
    arena_->used -= charge_;
}
std::shared_ptr<const AdaptiveSubfaceSnapshot> AdaptiveSubfaceSnapshot::seal(ContactHaloContext context,
    GeometryConstructionControls controls, int resident, std::span<const CanonicalSubfaceView> pieces, Budget& budget,
    std::shared_ptr<Budget> arena, ContactPhysicalResidency* residency, cartesian_detail::ArenaCredits* credits,
    bool native_index)
{
    using namespace subface_index_detail;
    if (!arena || arena.get() != &budget || resident < 0 || pieces.size() > std::numeric_limits<std::size_t>::max() / 2)
        throw std::invalid_argument("Invalid canonical subface snapshot publication.");
    std::size_t selectors = 0;
    for (const auto& view : pieces)
    {
        const auto second = view.second();
        const bool local_first = view.first().owner() == resident, local_second = second && second->owner() == resident;
        if (!local_first && !local_second)
            throw std::invalid_argument("Canonical subface lacks a resident source locator.");
        selectors += local_first + local_second;
    }
    PageCharge memory(arena);
    memory.add(sizeof(AdaptiveSubfaceSnapshot));
    auto result = make_controlled<control_bytes>(budget, arena,
        [&] { return new AdaptiveSubfaceSnapshot(context, controls, resident, arena); }).first;
    result->charge_ = std::exchange(memory.bytes, 0);
    result->native_index_ = native_index;
    auto indexes = SubfaceIndexRoot::create(arena);
    const auto native = subface_index_admission(arena, native_index);
    {
        PageCharge scratch(arena);
        std::vector<CanonicalSubfaceView> ordered;
        reserve(ordered, pieces.size(), scratch);
        ordered.assign(pieces.begin(), pieces.end());
        std::sort(ordered.begin(), ordered.end(), PieceLess{});
        indexes->pieces = PieceIndex(arena, nullptr, {}, native).build(ordered);
    }
    {
        PageCharge scratch(arena);
        std::vector<FaceRow> faces;
        reserve(faces, selectors, scratch);
        for (const auto& view : pieces)
        {
            const auto second = view.second();
            if (view.first().owner() == resident)
                faces.push_back({view.page_.get(), std::uint8_t(2 * view.slot_)});
            if (second && second->owner() == resident)
                faces.push_back({view.page_.get(), std::uint8_t(2 * view.slot_ + 1)});
        }
        std::sort(faces.begin(), faces.end(), FaceLess{});
        indexes->faces = FaceIndex(arena, nullptr, {}, native).build(faces);
    }
    {
        PageCharge scratch(arena);
        std::vector<PieceOwner> owners;
        for (const auto& view : pieces)
            if (owners.empty() || owners.back().page != view.page_)
            {
                scratch.append(owners);
                owners.push_back({view.page_, 1});
            }
            else
                ++owners.back().references;
        const auto counted_pieces = unique_owners(owners);
        indexes->owners = OwnerIndex(arena, nullptr, {}, native).build(counted_pieces);
        const auto account = [&](const auto& page)
        {
            const auto bytes = page->exclusive_bytes();
            result->exclusive_payload_bytes_ = checked_sum(result->exclusive_payload_bytes_, bytes);
            if (page->charge.arena != arena)
                result->foreign_exclusive_payload_bytes_ = checked_sum(result->foreign_exclusive_payload_bytes_, bytes);
            if (residency)
            {
                IncidencePhysicalOwners owners(*residency, credits, budget, true);
                page->observe_shared_owners(owners);
            }
        };
        std::vector<ProvenanceOwner> provenance;
        for (const auto& owner : counted_pieces)
        {
            account(owner.page);
            owner.page->visit_provenance_pages(
                [&](const auto& page)
                {
                    scratch.append(provenance);
                    provenance.push_back({page, 1});
                });
        }
        const auto counted_provenance = unique_owners(provenance);
        indexes->provenance_owners = ProvenanceIndex(arena, nullptr, {}, native).build(counted_provenance);
        for (const auto& owner : counted_provenance)
            account(owner.page);
    }
    result->index_bytes_ = indexes->index_bytes();
    result->indexes_ = std::move(indexes);
    result->size_ = pieces.size();
    result->seal_legacy_scope(nullptr, native);
    return result;
}
std::shared_ptr<const AdaptiveSubfaceSnapshot> AdaptiveSubfaceSnapshot::derive(
    const std::shared_ptr<const AdaptiveSubfaceSnapshot>& previous, ContactHaloContext context,
    GeometryConstructionControls controls, int resident, std::span<const CanonicalSubfaceChange> edits, Budget& budget,
    std::shared_ptr<Budget> arena, ContactPhysicalResidency* residency, cartesian_detail::ArenaCredits* credits,
    bool native_index)
{
    using namespace subface_index_detail;
    if (!previous || !arena || arena.get() != &budget || resident != previous->resident_)
        throw std::invalid_argument("Invalid canonical subface successor binding.");
    if (previous->size() && controls != previous->controls())
        controls.supported = false;
    PageCharge memory(arena);
    memory.add(sizeof(AdaptiveSubfaceSnapshot));
    auto result = make_controlled<control_bytes>(budget, arena,
        [&] { return new AdaptiveSubfaceSnapshot(context, controls, resident, arena); }).first;
    result->charge_ = std::exchange(memory.bytes, 0);
    result->native_index_ = native_index && previous->native_index_;
    const auto native = subface_index_admission(arena, result->native_index_);
    // A new incidence fork enters a fresh arena. Repeated edits in that
    // arena inherit the previous exact foreign allowance.
    result->foreign_index_bytes_ =
        previous->origin() == arena.get() ? previous->foreign_index_bytes_ : previous->index_bytes_;
    result->exclusive_payload_bytes_ = previous->exclusive_payload_bytes_;
    result->foreign_exclusive_payload_bytes_ = previous->origin() == arena.get()
                                                   ? previous->foreign_exclusive_payload_bytes_
                                                   : previous->exclusive_payload_bytes_;
    if (edits.empty())
    {
        result->size_ = previous->size_;
        result->index_bytes_ = previous->index_bytes_;
        result->indexes_ = previous->indexes_;
        result->seal_legacy_scope(previous.get(), native);
        return result;
    }
    auto indexes = SubfaceIndexRoot::create(arena);
    indexes->pieces = previous->indexes_->pieces;
    indexes->faces = previous->indexes_->faces;
    indexes->owners = previous->indexes_->owners;
    indexes->provenance_owners = previous->indexes_->provenance_owners;
    const PieceIndex pieces(arena, &result->work_, {}, native);
    const FaceIndex faces(arena, &result->work_, {}, native);
    const OwnerIndex owners(arena, &result->work_, {}, native);
    const ProvenanceIndex provenance_owners(arena, &result->work_, {}, native);
    if (previous->indexes_->charge.arena != arena)
        result->work_.retired_foreign_index_bytes = previous->indexes_->charged_bytes();
    const PieceLess piece_less;
    const FaceLess face_less;
    const OwnerLess owner_less;
    const auto adjust = [&]<class Index, class Page>(const Index& index, typename Index::Ptr& root,
                            const std::shared_ptr<const Page>& page, bool insert)
    {
        OwnerRow<Page> key{page, 0};
        const auto* found = Index::lower_bound(root,
            [&](const auto& row)
            {
                ++result->work_.owner_probes;
                return owner_less(row, key);
            });
        const bool present = found && found->page == page;
        if (!present && !insert)
            throw std::invalid_argument("Canonical subface edit lost its exact accepted page owner.");
        const auto count = present ? found->references : 0;
        key.references = insert ? checked_sum(count, 1) : count - 1;
        root = index.edit(root, key, key.references ? &key : nullptr, present);
        if (present == bool(key.references))
            return false;
        const auto bytes = page->exclusive_bytes();
        const auto account = [&](std::size_t& value)
        {
            if (insert)
                value = checked_sum(value, bytes);
            else
            {
                if (bytes > value)
                    throw std::logic_error("Canonical subface owner retirement exceeds inherited admission.");
                value -= bytes;
            }
        };
        account(result->exclusive_payload_bytes_);
        if (page->charge.arena != arena)
            account(result->foreign_exclusive_payload_bytes_);
        if (residency)
        {
            IncidencePhysicalOwners owners(*residency, credits, budget, insert);
            page->observe_shared_owners(owners);
        }
        return true;
    };
    const auto adjust_owner = [&](const auto& page, bool insert)
    {
        if (adjust(owners, indexes->owners, page, insert))
            page->visit_provenance_pages(
                [&](const auto& source) { adjust(provenance_owners, indexes->provenance_owners, source, insert); });
    };
    const auto local_face = [&](const auto& view, bool side) -> std::optional<FaceRow>
    {
        if (!view)
            return std::nullopt;
        if (side)
        {
            const auto second = view.second();
            if (!second || second->owner() != resident)
                return std::nullopt;
        }
        else if (view.first().owner() != resident)
            return std::nullopt;
        return FaceRow{view.page_.get(), std::uint8_t(2 * view.slot_ + side)};
    };
    for (const auto& [before, after] : edits)
    {
        if (!before && !after)
            throw std::invalid_argument("Empty canonical subface selector edit.");
        if (after && !local_face(after, false) && !local_face(after, true))
            throw std::invalid_argument("New canonical subface lacks a resident source locator.");
        if (before)
        {
            const auto* existing = PieceIndex::lower_bound(
                indexes->pieces, [&](const auto& candidate) { return piece_less(candidate, before); });
            if (!existing || existing->page_ != before.page_ || existing->slot_ != before.slot_)
                throw std::invalid_argument("Canonical subface edit lost its exact accepted piece.");
        }
        const bool replace_piece = before && after && !piece_less(before, after) && !piece_less(after, before);
        if (before)
            indexes->pieces = pieces.edit(indexes->pieces, before, replace_piece ? &after : nullptr, true);
        if (after && !replace_piece)
            indexes->pieces = pieces.edit(indexes->pieces, after, &after, false);
        if (before.page_ != after.page_)
        {
            if (after)
                adjust_owner(after.page_, true);
            if (before)
                adjust_owner(before.page_, false);
        }
        for (const bool side : {false, true})
        {
            const auto old = local_face(before, side), next = local_face(after, side);
            const bool replace = old && next && !face_less(*old, *next) && !face_less(*next, *old);
            if (old)
                indexes->faces = faces.edit(indexes->faces, *old, replace ? &*next : nullptr, true);
            if (next && !replace)
                indexes->faces = faces.edit(indexes->faces, *next, &*next, false);
        }
    }
    result->size_ = indexes->pieces ? indexes->pieces->size : 0;
    result->index_bytes_ = indexes->index_bytes();
    if (result->work_.retired_foreign_index_bytes > result->foreign_index_bytes_)
        throw std::logic_error("Canonical subface index retirement exceeds inherited admission.");
    result->foreign_index_bytes_ -= result->work_.retired_foreign_index_bytes;
    result->indexes_ = std::move(indexes);
    result->seal_legacy_scope(previous.get(), native);
    return result;
}
void AdaptiveSubfaceSnapshot::project_owners(const std::shared_ptr<const AdaptiveSubfaceSnapshot>& snapshot,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
{
    if (!snapshot || !NativeContactFundingAccess::project(graph, snapshot, snapshot->charged_bytes(),
            snapshot->native_funding(snapshot, snapshot->charged_bytes()), retain))
        return;
    const auto side_indexes = [&]
    {
        project_contact_side_index(snapshot->legacy_, graph, retain);
        project_contact_side_index(snapshot->inherited_legacy_, graph, retain);
    };
    const auto indexes = [&]
    {
        const auto& root = snapshot->indexes_;
        if (!SubfaceIndexRoot::project_body(root, graph, retain)) return;
        NativePublicationOwnerGraph::Context context(graph, retain, false);
        const auto index = [&]<class Index>(std::type_identity<Index> tag, const auto& self, const typename Index::Ptr& page,
                               const auto& rows) -> void
        {
            if (!page) return;
            if (page->charge.native)
            {
                if (!Index::visit_native_node(page, context.sink)) return;
            }
            else
            {
                if (snapshot->native_index_)
                    throw std::invalid_argument("Native contact snapshot has a legacy ordered index allocation.");
                if (!graph.external(page, page->charge.bytes, retain)) return;
                graph.external(page->charge.arena, sizeof(Budget) + 4 * sizeof(void*), retain);
            }
            for (const auto& child : page->children) self(tag, self, child, rows);
            for (const auto& row : page->rows) rows(row);
        };
        // Face rows borrow their piece page and add no owning edge.
        using namespace subface_index_detail;
        index(std::type_identity<PieceIndex>{}, index, root->pieces,
            [&](const CanonicalSubfaceView& view) { view.project_owners(graph, retain); });
        index(std::type_identity<FaceIndex>{}, index, root->faces, [](const auto&) {});
        index(std::type_identity<OwnerIndex>{}, index, root->owners,
            [&](const auto& row) { project_piece_page(row.page, graph, retain); });
        index(std::type_identity<ProvenanceIndex>{}, index, root->provenance_owners,
            [&](const auto& row) { project_provenance(row.page, graph, retain); });
    };
    if (!retain) side_indexes();
    indexes();
    if (retain) side_indexes();
}
void AdaptiveSubfaceSnapshot::visit_pieces(const CanonicalSubfaceEmit& emit) const
{
    using subface_index_detail::PieceIndex;
    PieceIndex::visit_from(
        indexes_->pieces, [](const auto&) { return false; },
        [&](const auto& view)
        {
            emit(view);
            return true;
        });
}
void AdaptiveSubfaceSnapshot::visit_face_pieces(
    const AdaptiveCellKey& key, std::uint64_t surface, const CanonicalSubfaceEmit& emit) const
{
    using namespace subface_index_detail;
    const auto before = [&](const FaceRow& row)
    { return std::tuple<const AdaptiveCellKey&, std::uint64_t>(row.key(), row.surface()) < std::tie(key, surface); };
    FaceIndex::visit_from(indexes_->faces, before,
        [&](const FaceRow& row)
        {
            if (row.key() != key || row.surface() != surface)
                return false;
            const auto* piece = PieceIndex::lower_bound(indexes_->pieces,
                [&](const auto& candidate)
                {
                    return compare_piece(candidate.support(), candidate.emission(), row.support(), row.emission()) < 0;
                });
            if (!piece || piece->page_.get() != row.page || piece->slot_ != row.code / 2)
                throw std::logic_error("Canonical face selector lost its owning piece page.");
            auto view = *piece;
            view.side_ = row.code % 2;
            emit(view);
            return true;
        });
}
void AdaptiveSubfaceSnapshot::visit_owners(const SubfaceOwnerVisit& visit) const
{
    using namespace subface_index_detail;
    visit_owner(indexes_, indexes_->charge.arena.get(), indexes_->charged_bytes(), visit);
    const auto index_owner = [&](const auto& page)
    { visit_owner(page, page->charge.arena.get(), page->charge.bytes, visit); };
    PieceIndex::visit_pages(indexes_->pieces, index_owner);
    FaceIndex::visit_pages(indexes_->faces, index_owner);
    OwnerIndex::visit_pages(indexes_->owners, index_owner);
    ProvenanceIndex::visit_pages(indexes_->provenance_owners, index_owner);
    indexes_->visit_piece_pages(
        [&](const auto& page)
        {
            visit_owner(page, page->charge.arena.get(), page->charged_bytes(), visit);
            page->visit_dependencies(visit, true);
            return true;
        });
    indexes_->visit_provenance_pages(
        [&](const auto& page)
        {
            visit_owner(page, page->charge.arena.get(), page->charged_bytes(), visit);
            page->visit_dependencies(visit, true);
        });
}
void AdaptiveSubfaceSnapshot::visit_payload_owners(const SubfaceOwnerVisit& visit) const
{
    indexes_->visit_piece_pages(
        [&](const auto& page)
        {
            page->visit_dependencies(visit, false);
            return true;
        });
    indexes_->visit_provenance_pages([&](const auto& page) { page->visit_dependencies(visit, false); });
}
void AdaptiveSubfaceSnapshot::retain_source_dependencies(cartesian_detail::CreditConstruction& credits,
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases,
    cartesian_detail::Allocation& memory) const
{
    indexes_->visit_piece_pages(
        [&](const auto& page)
        {
            page->retain_source_dependencies(credits, leases, memory);
            return true;
        });
}
bool AdaptiveSubfaceSnapshot::imports_source_owner(const void* identity) const noexcept
{
    return !indexes_->visit_piece_pages([&](const auto& page) { return !page->imports_source_owner(identity); });
}
void AdaptiveSubfaceSnapshot::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visit) const
{
    using namespace subface_index_detail;
    contact_payload_object(visit, this, ContactPayloadRole::SubfaceSnapshotObject, sizeof(*this), true);
    contact_payload_object(visit, indexes_.get(), ContactPayloadRole::SubfaceIndexRootObject, sizeof(*indexes_), true);
    const auto index_page = [&](const auto& page, ContactPayloadRole role)
    {
        contact_payload_object(visit, page.get(), ContactPayloadRole::SubfaceIndexPageObject, sizeof(*page), true);
        contact_payload_buffer(visit, page.get(), role, page->rows);
        contact_payload_buffer(visit, page.get(), ContactPayloadRole::SubfaceIndexChildren, page->children);
    };
    PieceIndex::visit_pages(
        indexes_->pieces, [&](const auto& page) { index_page(page, ContactPayloadRole::SubfacePieces); });
    FaceIndex::visit_pages(
        indexes_->faces, [&](const auto& page) { index_page(page, ContactPayloadRole::SubfaceSelectors); });
    OwnerIndex::visit_pages(
        indexes_->owners, [&](const auto& page) { index_page(page, ContactPayloadRole::SubfacePieceOwnerIndexRows); });
    ProvenanceIndex::visit_pages(indexes_->provenance_owners,
        [&](const auto& page) { index_page(page, ContactPayloadRole::SubfaceProvenanceOwnerIndexRows); });
    indexes_->visit_piece_pages(
        [&](const auto& page)
        {
            page->visit_borrowed_payload(visit);
            return true;
        });
    indexes_->visit_provenance_pages([&](const auto& page) { page->visit_borrowed_payload(visit); });
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
