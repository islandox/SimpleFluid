/** @file ContactPhysicalResidency.hh
 * @brief Shared physical contact and subface admission across one halo generation.
 */
#pragma once
#include "ContactOwnerUsage.hh"
#include "DistributedAdaptiveContactHalo.hh"
#include "NativeContactFunding.hh"
#include "NativeSourceOwnerStore.hh"
#include <algorithm>
#include <array>
#include <map>
#include <span>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** The live halo and incidence session borrow one editor. Immutable contact
 * and subface views pin their own bodies, independently of this editor. */
class ContactPhysicalResidency
{
public:
    enum class Role : std::uint32_t
    {
        FanBlock,
        RowGroups,
        GroupDomain,
        PointView,
        FallbackPoints,
        CreditMetadata,
        PointBody,
        SubfacePage,
        SubfaceRing,
        ProvenancePage,
        Layer,
        LayerHistory = Layer,
        TileBody,
        TileSourceContext,
        LineBucket,
        SourceDependency,
        TraceInventory,
        TraceSupport,
        RankCovers
    };
    explicit ContactPhysicalResidency(std::shared_ptr<Budget> arena, bool maintain_legacy_scope = false)
        : imported_(std::move(arena)), retired_metadata_(imported_.keeper()), usage_(imported_.keeper()),
          native_(geometry_detail::NativeConstructionCounterAccess::create<Budget>(
              imported_.arena().limit, 0, NativeSourceOwnerStore::mirror(imported_.keeper()))),
          maintain_legacy_scope_(maintain_legacy_scope)
    {
    }
    ContactPhysicalResidency(const ContactPhysicalResidency& source, std::shared_ptr<Budget> arena)
        : imported_(std::move(arena)), retired_metadata_(imported_.keeper()), usage_(source.usage_, imported_.keeper()),
          unique_(source.unique_), funded_point_scope_(source.funded_point_scope_),
          native_(geometry_detail::NativeConstructionCounterAccess::create<Budget>(
              imported_.arena().limit, 0, NativeSourceOwnerStore::mirror(imported_.keeper()))),
          owners_(source.owners_), paid_(source.paid_), administration_(source.administration_),
          legacy_members_(source.legacy_members_), inherited_members_(source.legacy_members_),
          point_members_(source.point_members_), inherited_point_members_(source.point_members_),
          missing_source_owners_(source.missing_source_owners_),
          foreign_source_owners_(source.foreign_source_owners_ + source.current_source_owners_),
          funded_point_bytes_(source.funded_point_bytes_), keep_overlap_(true),
          maintain_legacy_scope_(source.maintain_legacy_scope_)
    {
        imported_.add(source.usage_.payload_bytes() - funded_point_bytes_ + unique_);
    }
    using Transition = ContactOwnerUsage::Transition;
    using PointScope = std::uint64_t;
    template<class Owner>
    Transition retain(Role role, const std::shared_ptr<Owner>& owner, std::size_t bytes, const Budget* origin,
        std::size_t* prepaid = nullptr, PointScope scope = 0,
        const NativeContactOwnerFunding* supplied_funding = nullptr)
    {
        if (legacy_transferred_) throw std::logic_error("Sealed contact finance cannot be edited.");
        if (!owner)
            throw std::invalid_argument("Contact owner allocation identity is null.");
        const std::shared_ptr<const void> empty;
        if (!owner.owner_before(empty) && !empty.owner_before(owner))
            throw std::invalid_argument("Contact owner control is empty.");
        const ContactOwnerUsage::Key key{owner.get(), std::uint32_t(role)};
        const auto active = usage_.find(key);
        if (active && !usage_.authentic(key, owner))
            throw std::invalid_argument("Contact allocation address has conflicting owners.");
        const bool covered = covered_point_owner(role, scope);
        const bool new_foreign = !covered && origin != &imported_.arena() && !active;
        auto retired = new_foreign ? retired_.find(key) : retired_.end();
        if (retired != retired_.end())
        {
            const bool same_owner =
                !retired->second.control.owner_before(owner) && !owner.owner_before(retired->second.control);
            if (!same_owner)
            {
                // The former owner expired and the allocator reused its
                // address. Its weak control authenticates a distinct body.
                if (!retired->second.control.expired())
                    throw std::invalid_argument("Contact allocation address has conflicting owners.");
                imported_.release(retired->second.bytes);
                retired_payload_ -= retired->second.bytes;
                retired_.erase(retired);
                retired_metadata_.release(node_bytes);
                retired = retired_.end();
            }
            else if (retired->second.bytes != bytes || retired->second.origin != origin)
                throw std::invalid_argument("Reintroduced contact allocation disagrees with its source credit.");
        }
        const std::shared_ptr<const std::remove_const_t<Owner>> stable = owner;
        const auto funding =
            supplied_funding ? *supplied_funding : NativeContactFundingAccess::capture(stable, bytes, origin);
        if (funding.original_bytes && (funding.original_bytes != bytes ||
                (funding.index_owner ? (!funding.index_owner->matches(stable) ||
                                           funding.index_owner->state()->mirror.owner.get() != origin) :
                     (!funding.body.valid() || funding.body.origin_address() != origin ||
                         !funding.body.matches(PLIC::details::GeometryOwnerGraphEditor::external_kind(), stable,
                             funding.body.bytes())))))
            throw std::invalid_argument("Contact funding does not match the actual canonical physical control.");
        const bool new_charge = new_foreign && retired == retired_.end();
        const bool adopt = new_charge && prepaid && *prepaid;
        if (adopt && (*prepaid != bytes || bytes > imported_.arena().used ||
                         bytes > std::numeric_limits<std::size_t>::max() - imported_.bytes()))
            throw std::invalid_argument("Contact owner prepaid bytes disagree with its live allocation.");
        if (new_charge && !adopt)
            imported_.add(bytes);
        auto next_owners = [&]
        {
            try
            {
                return stage_owner(role, owner, bytes, true, funding, covered, origin != &imported_.arena());
            }
            catch (...)
            {
                if (new_charge && !adopt)
                    imported_.release(bytes);
                throw;
            }
        }();
        Transition transition;
        try
        {
            transition = usage_.retain(key, bytes, origin, owner);
        }
        catch (...)
        {
            if (new_charge && !adopt)
                imported_.release(bytes);
            throw;
        }
        owners_ = std::move(next_owners.owners);
        paid_ = std::move(next_owners.paid);
        administration_ = std::move(next_owners.administration);
        legacy_members_ = std::move(next_owners.legacy_members);
        inherited_members_ = std::move(next_owners.inherited_members);
        point_members_ = std::move(next_owners.point_members);
        inherited_point_members_ = std::move(next_owners.inherited_point_members);
        if (adopt)
        {
            imported_.adopt_precharged(bytes);
            *prepaid = 0;
        }
        observe_edit();
        if (transition.inserted() && source_role(role)) ++source_owner_count(funding);
        if (retired != retired_.end())
        {
            retired_payload_ -= retired->second.bytes;
            retired_.erase(retired);
            retired_metadata_.release(node_bytes);
        }
        if (transition.inserted() && covered)
        {
            funded_point_bytes_ += bytes;
            if (origin == &imported_.arena())
                funded_point_local_ += bytes;
        }
        if (transition.inserted() && !covered && role == Role::CreditMetadata && scope && origin == &imported_.arena())
        {
            if (scope == point_scope_)
                scoped_point_credit_bytes_ += bytes;
            if (fixed_pool_scope(scope))
                scoped_fixed_pool_credit_bytes_ += bytes;
            if (scope == source_credit_scope_)
                scoped_source_credit_bytes_ += bytes;
        }
        if (transition.inserted() && role == Role::TileSourceContext && origin == &imported_.arena())
            exported_tile_source_bytes_ += bytes;
        return transition;
    }
    template<class Owner> Transition release(Role role, const std::shared_ptr<Owner>& owner, PointScope scope = 0,
        const NativeContactOwnerFunding* supplied_funding = nullptr)
    {
        if (legacy_transferred_) throw std::logic_error("Sealed contact finance cannot be edited.");
        const ContactOwnerUsage::Key key{owner.get(), std::uint32_t(role)};
        const auto before = usage_.find(key);
        if (!before)
            throw std::logic_error("Retired contact allocation is absent.");
        if (!usage_.authentic(key, owner))
            throw std::invalid_argument("Contact allocation address has conflicting owners.");
        const std::shared_ptr<const std::remove_const_t<Owner>> stable = owner;
        const auto funding = supplied_funding ? *supplied_funding :
            NativeContactFundingAccess::capture(stable, before->bytes, before->origin);
        const bool covered = covered_point_owner(role, scope);
        auto next_owners = stage_owner(role, owner, before->bytes, false, funding, covered,
            before->origin != &imported_.arena());
        const bool retiring_foreign = !covered && before->references == 1 && before->origin != &imported_.arena();
        if (retiring_foreign && keep_overlap_)
        {
            retired_metadata_.add(node_bytes);
            try
            {
                const auto inserted = retired_.emplace(key, *before);
                if (!inserted.second)
                    throw std::logic_error("Retired contact allocation already has an overlap credit.");
            }
            catch (...)
            {
                retired_metadata_.release(node_bytes);
                throw;
            }
        }
        Transition transition;
        try
        {
            transition = usage_.release(key, owner);
        }
        catch (...)
        {
            if (retiring_foreign && keep_overlap_)
            {
                retired_.erase(key);
                retired_metadata_.release(node_bytes);
            }
            throw;
        }
        owners_ = std::move(next_owners.owners);
        paid_ = std::move(next_owners.paid);
        administration_ = std::move(next_owners.administration);
        legacy_members_ = std::move(next_owners.legacy_members);
        inherited_members_ = std::move(next_owners.inherited_members);
        point_members_ = std::move(next_owners.point_members);
        inherited_point_members_ = std::move(next_owners.inherited_point_members);
        observe_edit();
        if (transition.retired() && source_role(role))
        {
            auto& count = source_owner_count(funding);
            if (!count) std::terminate();
            --count;
        }
        if (retiring_foreign)
        {
            if (keep_overlap_)
                retired_payload_ += before->bytes;
            else
                imported_.release(before->bytes);
        }
        if (transition.retired() && covered)
        {
            funded_point_bytes_ -= before->bytes;
            if (before->origin == &imported_.arena())
                funded_point_local_ -= before->bytes;
        }
        if (transition.retired() && !covered && role == Role::CreditMetadata && scope &&
            before->origin == &imported_.arena())
        {
            if (scope == point_scope_)
                scoped_point_credit_bytes_ -= before->bytes;
            if (fixed_pool_scope(scope))
                scoped_fixed_pool_credit_bytes_ -= before->bytes;
            if (scope == source_credit_scope_)
                scoped_source_credit_bytes_ -= before->bytes;
        }
        if (transition.retired() && role == Role::TileSourceContext && before->origin == &imported_.arena())
            exported_tile_source_bytes_ -= before->bytes;
        return transition;
    }
    [[nodiscard]] bool admits(
        Role role, const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes) const noexcept
    {
        const auto entry = admitted_owner(role, owner);
        return entry && entry->origin == origin && entry->bytes == bytes;
    }
    [[nodiscard]] bool admits(Role role, const std::shared_ptr<const void>& owner, std::size_t bytes) const noexcept
    {
        const auto entry = admitted_owner(role, owner);
        return entry && entry->bytes == bytes;
    }
    /** Bind an independently admitted persistent physical point pool. Its
     * scope and exact credit controls survive a fork. */
    void bind_funded_point_scope(PointScope scope)
    {
        if (!scope || (funded_point_scope_ && funded_point_scope_ != scope))
            throw std::invalid_argument("Contact physical pool scope replacement requires detach.");
        funded_point_scope_ = scope;
    }
    [[nodiscard]] PointScope funded_point_scope() const noexcept { return funded_point_scope_; }
    /** Transfer the remaining foreign alias floor before the old pool releases
     * it. The caller disowns exactly the returned credit from that pool. */
    [[nodiscard]] std::size_t detach_funded_point_scope(PointScope scope)
    {
        if (!scope || scope != funded_point_scope_)
            throw std::invalid_argument("Contact physical pool detach has the wrong scope.");
        const auto transfer = funded_point_bytes_ - funded_point_local_;
        imported_.adopt_precharged(transfer);
        funded_point_scope_ = 0;
        funded_point_bytes_ = funded_point_local_ = 0;
        point_members_ = {};
        inherited_point_members_ = {};
        return transfer;
    }
    void begin_source_credit_scope(PointScope scope) noexcept
    {
        if (source_credit_scope_ != scope)
        {
            source_credit_scope_ = scope;
            scoped_source_credit_bytes_ = 0;
        }
    }
    [[nodiscard]] std::size_t scoped_source_credit_bytes() const noexcept { return scoped_source_credit_bytes_; }
    void begin_point_scope(PointScope scope) noexcept
    {
        if (point_scope_ != scope)
        {
            point_scope_ = scope;
            scoped_point_credit_bytes_ = 0;
        }
    }
    [[nodiscard]] std::size_t scoped_point_credit_bytes() const noexcept { return scoped_point_credit_bytes_; }

    void set_fixed_pool_scopes(std::span<const PointScope> scopes)
    {
        if (scopes.size() > fixed_scopes_.size())
            throw std::invalid_argument("Contact incidence pool scope count overflows.");
        std::array<PointScope, 7> next{};
        std::ranges::copy(scopes, next.begin());
        if (next != fixed_scopes_)
        {
            fixed_scopes_ = next;
            scoped_fixed_pool_credit_bytes_ = 0;
        }
    }
    [[nodiscard]] std::size_t scoped_fixed_pool_credit_bytes() const noexcept
    {
        return scoped_fixed_pool_credit_bytes_;
    }

    /** Unique row/record buffers have exactly one active semantic slot. */
    void retain_unique(std::size_t bytes)
    {
        if (bytes > std::numeric_limits<std::size_t>::max() - unique_)
            throw std::invalid_argument("Unique contact allocation census overflows.");
        unique_ += bytes;
        ++work_;
    }
    void release_unique(std::size_t bytes, const Budget* origin)
    {
        if (bytes > unique_)
            throw std::logic_error("Unique contact allocation census underflows.");
        unique_ -= bytes;
        ++work_;
        if (origin != &imported_.arena())
        {
            if (keep_overlap_)
                retired_payload_ += bytes;
            else
                imported_.release(bytes);
        }
    }
    /** Observe exact view objects and their physical credit metadata. The
     * incidence point pool separately admits the point bodies themselves. */
    template<class View> void observe_point_view(const std::shared_ptr<View>& view, bool added)
    {
        if (!view || !view->metadata_bytes())
            return;
        const auto before = usage_.find({view.get(), std::uint32_t(Role::PointView)});
        if (added)
            retain(Role::PointView, view, view->metadata_bytes(), view->origin_arena());
        else
            release(Role::PointView, view);
        // A batch namespace may serve many rows and subface pages. Walk its
        // dependencies only when that exact namespace enters/leaves the union.
        if ((added && !before) || (!added && before->references == 1))
            view->visit_credit_owners(
                [&](const auto& credit, std::size_t bytes) { observe_point_credit(credit, bytes, added); });
    }
    template<class Credit>
    void observe_point_credit(const std::shared_ptr<Credit>& credit, std::size_t bytes, bool added)
    {
        if (!bytes)
            return;
        if constexpr (requires {
                          credit->persistent_owner;
                          credit->body_owner();
                          credit->body_bytes();
                          credit->body_scope();
                      })
        {
            if (credit->persistent_owner)
            {
                NativeContactOwnerFunding funding;
                if constexpr (requires { credit->native_body_funding(credit); })
                    funding = credit->native_body_funding(credit);
                if (added)
                    retain(Role::PointBody, credit->body_owner(), credit->body_bytes(), credit->body_origin(), nullptr,
                        credit->body_scope(), &funding);
                else
                    release(Role::PointBody, credit->body_owner(), credit->body_scope(), &funding);
            }
        }
        if (added)
            retain(Role::CreditMetadata, credit, bytes, credit->budget, nullptr, scope_of(*credit));
        else
            release(Role::CreditMetadata, credit, scope_of(*credit));
    }
    [[nodiscard]] std::size_t exported_tile_source_bytes() const noexcept { return exported_tile_source_bytes_; }
    [[nodiscard]] std::size_t payload_bytes() const noexcept { return usage_.payload_bytes() + unique_; }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return retired_payload_ + retired_metadata_.bytes() + usage_.overlap_bytes();
    }
    [[nodiscard]] std::size_t work() const noexcept { return work_; }
    [[nodiscard]] const ContactOwnerUsage& usage() const noexcept { return usage_; }
    [[nodiscard]] const PLIC::details::GeometryNativeAdministration& source_administration() const noexcept
    {
        return administration_;
    }
    [[nodiscard]] const PLIC::details::geometry_demand::GeometryOwnerManifest& source_owners() const noexcept
    {
        return owners_;
    }
    [[nodiscard]] const PLIC::details::NativeOwnerReceipts& source_paid_receipts() const noexcept { return paid_; }
    [[nodiscard]] std::size_t unfunded_source_owners() const noexcept { return missing_source_owners_; }
    [[nodiscard]] std::size_t foreign_source_owners() const noexcept { return foreign_source_owners_; }
    [[nodiscard]] bool maintains_legacy_scope() const noexcept { return maintain_legacy_scope_; }
    [[nodiscard]] std::size_t inherited_point_floor_bytes() const noexcept
    { return inherited_point_members_.payload_bytes(); }
    [[nodiscard]] const auto& legacy_members() const noexcept { return legacy_members_; }
    [[nodiscard]] const auto& inherited_members() const noexcept { return inherited_members_; }
    [[nodiscard]] NativeContactSideIndexes source_side_indexes() const
    { return {legacy_members_, inherited_members_, point_members_, inherited_point_members_}; }
    void transfer_inherited_members(ContactExternalResidencyLease& target, std::size_t point_floor = 0)
    {
        const auto bytes = inherited_members_.payload_bytes();
        const auto& arena = target.arena_identity();
        if (!maintain_legacy_scope_ || legacy_transferred_ || arena.get() != imported_.keeper().get() ||
            arena.owner_before(imported_.keeper()) || imported_.keeper().owner_before(arena) ||
            point_floor != inherited_point_floor_bytes() || point_floor > bytes ||
            bytes - point_floor > imported_.bytes())
            throw std::invalid_argument("Inherited contact body floor lacks its exact original arena precharge.");
        target.adopt_precharged(bytes);
        imported_.disown_precharged(bytes - point_floor);
        legacy_transferred_ = true;
    }
    void finish_overlap() noexcept
    {
        imported_.release(retired_payload_);
        retired_payload_ = 0;
        retired_.clear();
        retired_metadata_.release(retired_metadata_.bytes());
        keep_overlap_ = false;
        usage_.finish_overlap();
    }

private:
    struct StagedOwnerGraph
    {
        PLIC::details::geometry_demand::GeometryOwnerManifest owners;
        PLIC::details::NativeOwnerReceipts paid;
        PLIC::details::GeometryNativeAdministration administration;
        PLIC::details::geometry_demand::GeometryOwnerManifest legacy_members, inherited_members;
        PLIC::details::geometry_demand::GeometryOwnerManifest point_members, inherited_point_members;
    };
    static bool source_role(Role role) noexcept
    {
        switch (role)
        {
            case Role::CreditMetadata:
            case Role::PointBody:
            case Role::SubfacePage:
            case Role::SubfaceRing:
            case Role::ProvenancePage:
            case Role::TileBody:
            case Role::TileSourceContext:
            case Role::SourceDependency:
                return true;
            default:
                return false;
        }
    }
    [[nodiscard]] std::size_t& source_owner_count(const NativeContactOwnerFunding& funding) noexcept
    {
        if (!funding.original_bytes) return missing_source_owners_;
        const auto origin = funding.index_owner ? funding.index_owner->state()->mirror.owner : funding.body.origin();
        const auto same_origin = [&](const std::shared_ptr<const void>& owner)
        {
            const auto& arena = imported_.keeper();
            return owner.get() == arena.get() && !owner.owner_before(arena) && !arena.owner_before(owner);
        };
        const auto local_observation = [&](const PLIC::details::NativeWeakOwnerObservation& observation)
        {
            if (const auto* contact = observation.contact_control_group()) return same_origin(contact->keeper);
            if (const auto* native = observation.native_envelope()) return same_origin((*native)->state()->mirror.owner);
            return true;
        };
        const bool current = same_origin(origin) && local_observation(funding.body.physical_observation()) &&
            local_observation(funding.body.financial_observation()) && local_observation(funding.related_control);
        const bool local_dependency = !funding.dependent_fixed || same_origin(funding.dependent_fixed->state()->mirror.owner);
        return current && local_dependency ? current_source_owners_ : foreign_source_owners_;
    }
    template<class Owner>
    StagedOwnerGraph stage_owner(Role role, const std::shared_ptr<Owner>& owner, std::size_t bytes, bool retain,
        const NativeContactOwnerFunding& funding, bool point_funded, bool foreign)
    {
        // Fan/row/trace/cover and line registry caches retain their original
        // arena admission. They are private to H and do not fund native output.
        if (!source_role(role))
            return {owners_, paid_, administration_, legacy_members_, inherited_members_, point_members_,
                inherited_point_members_};
        std::size_t cursor = 0;
        PLIC::details::GeometryOwnerGraphEditor graph(
            owners_, 0, cursor, imported_.arena().limit, nullptr, nullptr, native_.state(), &paid_);
        NativeContactFundingAccess::project(graph, owner, bytes, funding, retain);
        auto next = graph.finish();
        PLIC::details::GeometryOwnerUnionWork work;
        auto administration =
            administration_.updated(NativeSourceOwnerStore::single(owners_), NativeSourceOwnerStore::single(next), {},
                {}, cursor, imported_.arena().limit, native_.state(), work, &paid_, &graph.paid_receipts());
        auto legacy = legacy_members_, inherited = inherited_members_;
        auto point = point_members_, inherited_point = inherited_point_members_;
        if (maintain_legacy_scope_ && funding.original_bytes)
        {
            const auto edit = [&](const auto& root)
            {
                std::size_t payload = 0;
                PLIC::details::GeometryOwnerGraphEditor scope(
                    root, 0, payload, imported_.arena().limit, nullptr, nullptr, native_.state());
                NativeContactFundingAccess::project_reimported(scope, owner, funding, retain);
                return scope.finish();
            };
            legacy = edit(legacy);
            if (foreign) inherited = edit(inherited);
            if (point_funded)
            {
                point = edit(point);
                if (foreign) inherited_point = edit(inherited_point);
            }
        }
        return {std::move(next), graph.paid_receipts(), std::move(administration), std::move(legacy),
            std::move(inherited), std::move(point), std::move(inherited_point)};
    }

    [[nodiscard]] std::optional<ContactOwnerUsage::Entry> admitted_owner(
        Role role, const std::shared_ptr<const void>& owner) const noexcept
    {
        const ContactOwnerUsage::Key key{owner.get(), std::uint32_t(role)};
        if (const auto active = usage_.find(key))
            return usage_.authentic(key, owner) ? active : std::nullopt;
        if (const auto old = retired_.find(key); old != retired_.end())
            if (!old->second.control.owner_before(owner) && !owner.owner_before(old->second.control))
                return old->second;
        return {};
    }
    [[nodiscard]] bool covered_point_owner(Role role, PointScope scope) const noexcept
    {
        return scope && scope == funded_point_scope_ && (role == Role::PointBody || role == Role::CreditMetadata);
    }
    [[nodiscard]] bool fixed_pool_scope(PointScope scope) const noexcept
    {
        return std::ranges::find(fixed_scopes_, scope) != fixed_scopes_.end();
    }
    template<class Credit> static PointScope scope_of(const Credit& credit) noexcept
    {
        if constexpr (requires { credit.pool_scope; })
            return credit.pool_scope;
        if constexpr (requires { credit.arena_scope; })
            return credit.arena_scope;
        return 0;
    }
    struct Less
    {
        bool operator()(ContactOwnerUsage::Key a, ContactOwnerUsage::Key b) const noexcept
        {
            if (a.allocation != b.allocation)
                return std::less<const void*>{}(a.allocation, b.allocation);
            return a.role < b.role;
        }
    };
    using Retired = std::map<ContactOwnerUsage::Key, ContactOwnerUsage::Entry, Less>;
    static constexpr auto node_bytes = sizeof(Retired::value_type) + 4 * sizeof(void*);
    void observe_edit() noexcept { work_ += usage_.last_edit_work().visited_nodes; }
    ContactGroupAllocation imported_, retired_metadata_;
    ContactOwnerUsage usage_;
    geometry_detail::NativeConstructionAdmission native_;
    PLIC::details::geometry_demand::GeometryOwnerManifest owners_;
    PLIC::details::NativeOwnerReceipts paid_;
    PLIC::details::GeometryNativeAdministration administration_;
    PLIC::details::geometry_demand::GeometryOwnerManifest legacy_members_, inherited_members_;
    PLIC::details::geometry_demand::GeometryOwnerManifest point_members_, inherited_point_members_;
    std::size_t missing_source_owners_ = 0, foreign_source_owners_ = 0, current_source_owners_ = 0;
    Retired retired_;
    std::size_t unique_ = 0, retired_payload_ = 0, work_ = 0;
    PointScope source_credit_scope_ = 0;
    std::size_t scoped_source_credit_bytes_ = 0;
    std::size_t exported_tile_source_bytes_ = 0;
    PointScope point_scope_ = 0;
    std::size_t scoped_point_credit_bytes_ = 0, scoped_fixed_pool_credit_bytes_ = 0;
    std::array<PointScope, 7> fixed_scopes_{};
    PointScope funded_point_scope_ = 0;
    std::size_t funded_point_bytes_ = 0, funded_point_local_ = 0;
    bool keep_overlap_ = false;
    bool maintain_legacy_scope_ = false, legacy_transferred_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
