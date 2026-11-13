
/** @file ProductFamilyElectionState.hh
 * @brief Persistent local factor contributors and sharded rank proposals.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveFamilyProofFragment.hh"
#include "geometry/mesh/AMR/ContactGroupAllocation.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"

#include <map>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Published state is const. A private edit imports the accepted physical
 * union and copies bounded paths. Complete family bodies live at their elected
 * owner. A shard stores the authenticated record and shared dictionary
 * selectors for each rank proposal, and reuses the proof on max-only edits. */
class ProductFamilyElectionState
{
    using Charge = persistent_ordered_detail::PageCharge;

public:
    struct Proof
    {
        Charge charge;
        AdaptiveFamilyProofFragment fragment;
        explicit Proof(const std::shared_ptr<Budget>& arena) : charge(arena) {}
        [[nodiscard]] AdaptiveFamilyProofView view() const noexcept { return fragment.view(); }
    };
    struct Proposal
    {
        Charge charge;
        const int rank;
        const AdaptiveCellKey anchor;
        const std::uint64_t anchor_revision, maximum;
        const std::shared_ptr<const Proof> proof;
        Proposal(const std::shared_ptr<Budget>& arena, int peer, const AdaptiveCellKey& leaf, std::uint64_t revision,
            std::uint64_t max, std::shared_ptr<const Proof> witness)
            : charge(arena), rank(peer), anchor(leaf), anchor_revision(revision), maximum(max),
              proof(std::move(witness))
        {
        }
    };
    using ProposalPtr = std::shared_ptr<const Proposal>;
    struct Winner
    {
        const Proposal* proposal;
        std::uint64_t maximum;
    };
    struct Local
    {
        const AdaptiveCellKey& anchor;
        std::uint64_t anchor_revision, maximum;
    };
    struct Work
    {
        std::size_t pages = 0, page_bytes = 0, contributor_edits = 0, proposal_edits = 0;
    };

private:
    struct Member
    {
        Charge charge;
        const AdaptiveBytes family;
        const AdaptiveCellKey leaf;
        const std::uint64_t revision;
        Member(const std::shared_ptr<Budget>& arena, const AdaptiveBytes& key, const AdaptiveCellKey& cell,
            std::uint64_t generation)
            : charge(arena), family(key), leaf(cell), revision(generation)
        {
        }
    };
    using MemberPtr = std::shared_ptr<const Member>;
    struct MemberLess
    {
        bool operator()(const MemberPtr& a, const MemberPtr& b) const noexcept
        {
            return a->family == b->family ? a->leaf < b->leaf : a->family < b->family;
        }
    };
    struct RevisionLess
    {
        bool operator()(const MemberPtr& a, const MemberPtr& b) const noexcept
        {
            if (a->family != b->family)
                return a->family < b->family;
            return a->revision == b->revision ? a->leaf < b->leaf : a->revision > b->revision;
        }
    };
    struct Group
    {
        Charge charge;
        const AdaptiveBytes family;
        std::vector<ProposalPtr> proposals;
        const Proposal* winner = nullptr;
        std::uint64_t maximum = 0;
        std::size_t bytes = 0, local = 0;
        Group(const std::shared_ptr<Budget>& arena, const AdaptiveBytes& key) : charge(arena), family(key) {}
    };
    using GroupPtr = std::shared_ptr<const Group>;
    struct GroupLess
    {
        bool operator()(const GroupPtr& a, const GroupPtr& b) const noexcept { return a->family < b->family; }
    };
    struct Part
    {
        Charge charge;
        const AdaptiveBytes value;
        Part(const std::shared_ptr<Budget>& arena, const AdaptiveBytes& bytes) : charge(arena), value(bytes) {}
    };
    struct PartEntry
    {
        std::shared_ptr<const Part> owner;
        std::size_t references = 0;
    };
    struct PartLess
    {
        bool operator()(const PartEntry& a, const PartEntry& b) const noexcept
        {
            return a.owner->value < b.owner->value;
        }
    };
    struct ElectedPartEntry
    {
        AdaptiveFamilyProofPart owner;
        std::size_t references = 0;
    };
    struct ElectedPartLess
    {
        bool operator()(const ElectedPartEntry& a, const ElectedPartEntry& b) const noexcept
        {
            return *a.owner < *b.owner;
        }
    };
    struct Weights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t total = 0, local = 0;
        };
        static void add_row(Summary& sum, const MemberPtr& member, const Budget* arena)
        {
            sum.total = add(sum.total, member->charge.bytes);
            if (member->charge.arena.get() == arena)
                sum.local = add(sum.local, member->charge.bytes);
        }
        static void add_row(Summary& sum, const GroupPtr& group, const Budget* arena)
        {
            sum.total = add(sum.total, group->bytes);
            if (group->charge.arena.get() == arena)
                sum.local = add(sum.local, group->local);
        }
        static void add_row(Summary& sum, const PartEntry& part, const Budget* arena)
        {
            sum.total = add(sum.total, part.owner->charge.bytes);
            if (part.owner->charge.arena.get() == arena)
                sum.local = add(sum.local, part.owner->charge.bytes);
        }
        static void add_child(Summary& sum, const Summary& child, bool same_arena)
        {
            sum.total = add(sum.total, child.total);
            if (same_arena)
                sum.local = add(sum.local, child.local);
        }
    };
    using OrderedWork = persistent_ordered_detail::OrderedPageWork;
    using Members = persistent_ordered_detail::OrderedPages<MemberPtr, MemberLess, false, OrderedWork, Weights>;
    using Revisions = persistent_ordered_detail::OrderedPages<MemberPtr, RevisionLess>;
    using Groups = persistent_ordered_detail::OrderedPages<GroupPtr, GroupLess, false, OrderedWork, Weights>;
    using Parts = persistent_ordered_detail::OrderedPages<PartEntry, PartLess, false, OrderedWork, Weights>;
    using ElectedParts = persistent_ordered_detail::OrderedPages<ElectedPartEntry, ElectedPartLess>;

public:
    explicit ProductFamilyElectionState(
        std::shared_ptr<Budget> arena, const ProductFamilyElectionState* source = nullptr)
        : shell_(arena), imported_(std::move(arena))
    {
        if (source)
        {
            if (imported_.keeper() == source->imported_.keeper())
                throw std::invalid_argument("Product election requires a fresh candidate arena.");
            imported_.add(source->storage_bytes());
            members_ = source->members_;
            revisions_ = source->revisions_;
            groups_ = source->groups_;
            parts_ = source->parts_;
            elected_parts_ = source->elected_parts_;
        }
    }

    [[nodiscard]] static std::shared_ptr<ProductFamilyElectionState> create(
        const std::shared_ptr<Budget>& arena, const ProductFamilyElectionState* source = nullptr)
    {
        ContactGroupAllocation header_credit(arena);
        header_credit.add(shell_bytes());
        auto value = std::make_shared<ProductFamilyElectionState>(arena, source);
        value->shell_.transfer_from(header_credit, header_credit.bytes());
        return value;
    }
    [[nodiscard]] static constexpr std::size_t shell_bytes() noexcept { return header<ProductFamilyElectionState>(); }

    [[nodiscard]] const std::shared_ptr<Budget>& keeper() const noexcept { return imported_.keeper(); }

    void change_member(const AdaptiveBytes& family, const AdaptiveCellKey& leaf, std::uint64_t revision, bool insert)
    {
        const auto* previous = Members::lower_bound(members_,
            [&](const MemberPtr& row) { return row->family == family ? row->leaf < leaf : row->family < family; });
        const bool exists = previous && (*previous)->family == family && (*previous)->leaf == leaf;
        if (exists == insert || (exists && (*previous)->revision != revision))
            throw std::invalid_argument("Product contribution differs from its accepted revision.");
        MemberPtr row;
        if (insert)
        {
            Charge memory(keeper());
            memory.add(add(header<Member>(), add(family.capacity(), key_bytes(leaf))));
            auto next = std::make_shared<Member>(keeper(), family, leaf, revision);
            transfer(memory, next->charge, add(header<Member>(), add(next->family.capacity(), key_bytes(next->leaf))));
            row = std::move(next);
        }
        else
            row = *previous;
        OrderedWork primary, secondary;
        auto members = Members(keeper(), &primary).edit(members_, row, insert ? &row : nullptr, !insert);
        auto revisions = Revisions(keeper(), &secondary).edit(revisions_, row, insert ? &row : nullptr, !insert);
        observe(primary);
        observe(secondary);
        members_ = std::move(members);
        revisions_ = std::move(revisions);
        ++work_.contributor_edits;
    }

    [[nodiscard]] std::optional<Local> local(const AdaptiveBytes& key) const
    {
        const auto precedes = [&](const MemberPtr& row) { return row->family < key; };
        const auto* first = Members::lower_bound(members_, precedes);
        if (!first || (*first)->family != key)
            return {};
        const auto* largest = Revisions::lower_bound(revisions_, precedes);
        if (!largest || (*largest)->family != key)
            throw std::logic_error("Product contribution selectors disagree.");
        return Local{(*first)->leaf, (*first)->revision, (*largest)->revision};
    }

    [[nodiscard]] std::shared_ptr<const Proof> proof(AdaptiveFamilyProofView value)
    {
        using persistent_ordered_detail::checked_product;
        if (value.record.empty())
            throw std::invalid_argument("Missing Product proof record.");
        Charge memory(keeper());
        memory.add(add(header<Proof>(),
            add(value.record.size(), checked_product(value.parts.size(), sizeof(AdaptiveFamilyProofPart)))));
        auto result = std::make_shared<Proof>(keeper());
        result->fragment.record.assign(value.record.begin(), value.record.end());
        result->fragment.parts.reserve(value.parts.size());
        if (result->fragment.record.capacity() != value.record.size() ||
            result->fragment.parts.capacity() != value.parts.size())
            throw std::invalid_argument("Product proof row capacity differs from admission.");
        for (const auto& part : value.parts)
        {
            if (!part)
                throw std::invalid_argument("Missing Product proof dictionary part.");
            result->fragment.parts.push_back(retain_part(*part));
        }
        transfer(memory, result->charge, memory.bytes);
        return result;
    }
    [[nodiscard]] ProposalPtr proposal(int rank, const AdaptiveCellKey& anchor, std::uint64_t revision,
        std::uint64_t max, std::shared_ptr<const Proof> proof) const
    {
        if (rank < 0 || revision > max || !proof)
            throw std::invalid_argument("Invalid Product rank proposal.");
        Charge memory(keeper());
        memory.add(add(header<Proposal>(), key_bytes(anchor)));
        auto result = std::make_shared<Proposal>(keeper(), rank, anchor, revision, max, std::move(proof));
        transfer(memory, result->charge, add(header<Proposal>(), key_bytes(result->anchor)));
        return result;
    }
    [[nodiscard]] std::span<const ProposalPtr> proposals(const AdaptiveBytes& key) const
    {
        const auto* row = group(key);
        return row ? std::span((*row)->proposals) : std::span<const ProposalPtr>{};
    }
    [[nodiscard]] Winner winner(const AdaptiveBytes& key) const
    {
        const auto* row = group(key);
        return row ? Winner{(*row)->winner, (*row)->maximum} : Winner{nullptr, 0};
    }

    /** The caller authenticates each changed witness and every rank receipt
     * before publishing the complete, rank-ordered proposals of this key. */
    void set_proposals(const AdaptiveBytes& key, std::span<const ProposalPtr> rows)
    {
        const auto* previous = group(key);
        if (rows.empty() && !previous)
            return;
        GroupPtr replacement;
        if (!rows.empty())
        {
            Charge memory(keeper());
            memory.add(add(header<Group>(),
                add(key.capacity(), persistent_ordered_detail::checked_product(rows.size(), sizeof(ProposalPtr)))));
            auto next = std::make_shared<Group>(keeper(), key);
            next->proposals.reserve(rows.size());
            if (next->proposals.capacity() != rows.size())
                throw std::invalid_argument("Product proposal row capacity differs from admission.");
            int rank = -1;
            for (const auto& proposal : rows)
            {
                if (!proposal || proposal->rank <= rank)
                    throw std::invalid_argument("Product rank proposals are not strictly ordered.");
                rank = proposal->rank;
                if (!next->winner || proposal->anchor < next->winner->anchor)
                    next->winner = proposal.get();
                next->maximum = std::max(next->maximum, proposal->maximum);
                next->proposals.push_back(proposal);
                accumulate(*next, proposal->charge);
                accumulate(*next, proposal->proof->charge);
            }
            const auto bytes = add(header<Group>(),
                add(next->family.capacity(),
                    persistent_ordered_detail::checked_product(next->proposals.capacity(), sizeof(ProposalPtr))));
            transfer(memory, next->charge, bytes);
            accumulate(*next, next->charge);
            replacement = std::move(next);
        }
        const auto& selector = previous ? *previous : replacement;
        OrderedWork work;
        auto root =
            Groups(keeper(), &work).edit(groups_, selector, replacement ? &replacement : nullptr, previous != nullptr);
        observe(work);
        if (previous)
        {
            for (const auto& proposal : (*previous)->proposals)
            {
                const auto next = std::lower_bound(rows.begin(), rows.end(), proposal->rank,
                    [](const ProposalPtr& row, int peer) { return row->rank < peer; });
                if (next == rows.end() || (*next)->rank != proposal->rank || (*next)->proof != proposal->proof)
                    for (const auto& part : proposal->proof->fragment.parts)
                        retire_part(part);
            }
        }
        groups_ = std::move(root);
        ++work_.proposal_edits;
    }

    /** The elected dictionary shares the actual family body owners.
     * AdaptiveLeafBodyUsage admits their payload; this root admits only its
     * metadata. Neither a wave decoder nor an old state owns the entries. */
    [[nodiscard]] AdaptiveFamilyProofPart elected_part(std::span<const char> bytes) const
    {
        const auto* entry = find_elected_part(bytes);
        return entry ? entry->owner : nullptr;
    }

    void update_elected_part(const AdaptiveFamilyProofPart& part, std::size_t retired, std::size_t inserted)
    {
        if (!part)
            throw std::invalid_argument("Missing elected Product dictionary part.");
        const auto* previous = find_elected_part(*part);
        if (previous && (previous->owner.get() != part.get() || previous->owner.owner_before(part) ||
                            part.owner_before(previous->owner)))
            throw std::invalid_argument("Elected Product dictionary part differs from its actual owner.");
        const auto before = previous ? previous->references : 0;
        if (retired > before)
            throw std::logic_error("Product elected part retirement exceeds its accepted references.");
        const auto after = add(before - retired, inserted);
        if (after == before)
            return;
        const ElectedPartEntry next{part, after};
        OrderedWork work;
        elected_parts_ =
            ElectedParts(keeper(), &work).edit(elected_parts_, next, after ? &next : nullptr, previous != nullptr);
        observe(work);
    }

    [[nodiscard]] std::size_t storage_bytes() const noexcept
    {
        return total(members_) + total(groups_) + total(parts_) + (revisions_ ? revisions_->subtree_bytes : 0) +
               (elected_parts_ ? elected_parts_->subtree_bytes : 0);
    }
    [[nodiscard]] std::size_t inherited_bytes() const noexcept
    {
        return storage_bytes() - current(members_) - current(groups_) - current(parts_) -
               (revisions_ && revisions_->charge.arena == keeper() ? revisions_->same_arena_bytes : 0) -
               (elected_parts_ && elected_parts_->charge.arena == keeper() ? elected_parts_->same_arena_bytes : 0);
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return imported_.bytes() - inherited_bytes(); }
    void finish_overlap() noexcept { imported_.release(overlap_bytes()); }
    [[nodiscard]] const Work& work() const noexcept { return work_; }
    [[nodiscard]] std::size_t dictionary_size() const noexcept { return parts_ ? parts_->size : 0; }
    [[nodiscard]] std::size_t dictionary_bytes() const noexcept { return parts_ ? parts_->weights.total : 0; }
    [[nodiscard]] std::size_t elected_dictionary_size() const noexcept
    {
        return elected_parts_ ? elected_parts_->size : 0;
    }

private:
    [[nodiscard]] const ElectedPartEntry* find_elected_part(std::span<const char> bytes) const
    {
        const auto* entry = ElectedParts::lower_bound(elected_parts_, [&](const auto& row)
            { return std::lexicographical_compare(row.owner->begin(), row.owner->end(), bytes.begin(), bytes.end()); });
        return entry && std::ranges::equal(*entry->owner, bytes) ? entry : nullptr;
    }

    [[nodiscard]] const PartEntry* find_part(const AdaptiveBytes& bytes) const
    {
        const auto* row = Parts::lower_bound(parts_, [&](const auto& entry) { return entry.owner->value < bytes; });
        return row && row->owner->value == bytes ? row : nullptr;
    }
    [[nodiscard]] AdaptiveFamilyProofPart retain_part(const AdaptiveBytes& bytes)
    {
        const auto* previous = find_part(bytes);
        PartEntry next;
        if (previous)
        {
            next = *previous;
            next.references = add(next.references, 1);
        }
        else
        {
            Charge admission(keeper());
            admission.add(add(header<Part>(), bytes.capacity()));
            auto value = std::make_shared<Part>(keeper(), bytes);
            transfer(admission, value->charge, add(header<Part>(), value->value.capacity()));
            next = {std::move(value), 1};
        }
        OrderedWork work;
        parts_ = Parts(keeper(), &work).edit(parts_, next, &next, previous != nullptr);
        observe(work);
        return AdaptiveFamilyProofPart(next.owner, &next.owner->value);
    }
    void retire_part(const AdaptiveFamilyProofPart& part)
    {
        const auto* previous = find_part(*part);
        if (!previous || !previous->references || part.get() != &previous->owner->value ||
            part.owner_before(previous->owner) || previous->owner.owner_before(part))
            throw std::logic_error("Product proof does not own its accepted dictionary part.");
        auto next = *previous;
        --next.references;
        OrderedWork work;
        parts_ = Parts(keeper(), &work).edit(parts_, next, next.references ? &next : nullptr, true);
        observe(work);
    }

    [[nodiscard]] const GroupPtr* group(const AdaptiveBytes& key) const
    {
        const auto* row = Groups::lower_bound(groups_, [&](const GroupPtr& value) { return value->family < key; });
        return row && (*row)->family == key ? row : nullptr;
    }
    template<class T> static constexpr std::size_t header() noexcept
    {
        return sizeof(T) + persistent_ordered_detail::control_bytes;
    }
    static std::size_t add(std::size_t a, std::size_t b) { return persistent_ordered_detail::checked_sum(a, b); }
    static std::size_t key_bytes(const AdaptiveCellKey& key)
    {
        return std::visit(
            [](const auto& ancestry) -> std::size_t
            {
                using persistent_ordered_detail::checked_product;
                if constexpr (requires { ancestry.path; })
                    return checked_product(ancestry.path.capacity(), sizeof(std::uint64_t));
                else if constexpr (requires { ancestry.base.path; })
                    return checked_product(
                        add(ancestry.base.path.capacity(), ancestry.axial.path.capacity()), sizeof(std::uint64_t));
                else
                    return 0;
            },
            key.ancestry);
    }
    static void transfer(Charge& scratch, Charge& target, std::size_t bytes)
    {
        if (bytes > scratch.bytes)
            throw std::invalid_argument("Product election row exceeds preadmission.");
        scratch.bytes -= bytes;
        target.bytes = bytes;
    }
    static void accumulate(Group& group, const Charge& item)
    {
        group.bytes = add(group.bytes, item.bytes);
        if (item.arena == group.charge.arena)
            group.local = add(group.local, item.bytes);
    }
    template<class Root> static std::size_t total(const Root& root) noexcept
    {
        return root ? root->subtree_bytes + root->weights.total : 0;
    }
    template<class Root> std::size_t current(const Root& root) const noexcept
    {
        return root && root->charge.arena == keeper() ? root->same_arena_bytes + root->weights.local : 0;
    }
    void observe(const OrderedWork& work)
    {
        work_.pages = add(work_.pages, work.copied_pages);
        work_.page_bytes = add(work_.page_bytes, work.copied_bytes);
    }

    ContactGroupAllocation shell_, imported_;
    Members::Ptr members_;
    Revisions::Ptr revisions_;
    Groups::Ptr groups_;
    Parts::Ptr parts_;
    ElectedParts::Ptr elected_parts_;
    Work work_;
};
/** Visit only byte owners retained by a decoded family. A descriptor view is
 * a value inside its descriptor, so its transport selector is not retained. */
template<class Visit> void visit_elected_product_parts(const AdaptiveLeaf& parent, const Visit& visit)
{
    if (parent.native_recipe)
        for (const auto& part : parent.native_recipe->parts)
            visit(part);
    visit(parent.join_recipe);
    if (parent.factor_recipe)
        for (const auto& part : {parent.factor_recipe->root, parent.factor_recipe->base, parent.factor_recipe->axis})
            visit(part);
}

/** Candidate-only owner changes. Count actual controls and apply net
 * references, so identical parts in a fresh witness do not copy any path. */
class ProductElectedPartDelta
{
    struct Change
    {
        AdaptiveFamilyProofPart owner;
        std::size_t retired = 0, inserted = 0;
    };
    using Changes = std::map<const void*, Change>;
    ContactGroupAllocation memory_;
    Changes changes_;

public:
    explicit ProductElectedPartDelta(std::shared_ptr<Budget> arena) : memory_(std::move(arena)) {}

    void record(const AdaptiveLeaf& parent, bool inserted)
    {
        visit_elected_product_parts(parent,
            [&](const AdaptiveFamilyProofPart& part)
            {
                if (!part)
                    return;
                auto found = changes_.find(part.get());
                if (found == changes_.end())
                {
                    memory_.add(sizeof(Changes::value_type) + 4 * sizeof(void*));
                    found = changes_.try_emplace(part.get(), Change{part}).first;
                }
                auto& value = found->second;
                if (part.owner_before(value.owner) || value.owner.owner_before(part))
                    throw std::invalid_argument("Conflicting Product elected part controls.");
                auto& count = inserted ? value.inserted : value.retired;
                count = persistent_ordered_detail::checked_sum(count, 1);
            });
    }

    void apply(ProductFamilyElectionState& candidate) const
    {
        for (const auto& [address, change] : changes_)
            if (change.retired != change.inserted)
                candidate.update_elected_part(change.owner, change.retired, change.inserted);
    }
};

} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
