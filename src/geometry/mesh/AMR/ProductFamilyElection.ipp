/** @file ProductFamilyElection.ipp
 * @brief Changed Product contributor receipts and sharded family election.
 *
 * Included in DistributedAdaptiveHierarchy.cc after FamilyJournal, so the
 * bounded packet, provider-proof, and retention helpers are shared.
 */
#include "ProductFamilyFragmentPackets.ipp"

struct ProductFamilyResult
{
    AdaptiveBytes key;
    std::optional<AdaptiveFamily> family;
    bool changes_parts;
};

void collect_product_families(DistributedAdaptiveHierarchy::Data& state,
    const DistributedAdaptiveHierarchy::Data* source, std::span<const AdaptiveCellKey> dirty, std::uint64_t transaction,
    ProductFamilyElectionState& edit, ClosureScratch& retained, std::vector<ProductFamilyResult>& result)
{
    ClosureScratch work{retained.budget};
    const auto* accepted = source ? source->product_election.get() : nullptr;
    const auto ranks = state.comm->getSize(), rank = state.comm->getRank();
    const ProductProofContext context{state.topology, state.partition, transaction};
    const auto remaining = [](ClosureScratch& memory)
    {
        const auto cap = memory.budget.limit - memory.budget.used;
        if (!cap)
            throw std::invalid_argument("Product family workspace lease exhausted.");
        memory.add(cap);
        return cap;
    };
    ProductWaveDictionary proposal_parts, result_parts;
    std::set<const void*> delivered_parts;
    const auto decode = [&](AdaptiveFamilyProofView proof, ClosureScratch& memory)
    {
        const auto cap = remaining(memory);
        auto family = decode_family_proof_fragment(proof, cap);
        const auto bytes = fetch_retention({}, std::span(&family, 1), cap, false, nullptr, true);
        memory.drop(cap - bytes);
        return family;
    };
    const auto capture = [&](const AdaptiveFamily& family)
    {
        const auto cap = remaining(work);
        auto proof = encode_family_proof_fragment(family, cap);
        Budget observed{cap};
        observed.add(proof.record.capacity());
        observed.add(
            persistent_ordered_detail::checked_product(proof.parts.capacity(), sizeof(AdaptiveFamilyProofPart)));
        work.drop(cap - observed.used);
        // Own the complete byte value once per wave. In particular an
        // aliased native descriptor view must not pin every recipe part.
        for (auto& part : proof.parts)
            part = proposal_parts.retain(*part, work);
        return proof;
    };
    const auto find_active = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf*
    {
        if (!source)
        {
            const auto at = state.owned().find(key);
            return at == state.owned().end() ? nullptr : &at->second;
        }
        return state.layout()->find(key);
    };

    struct ProposalDelta
    {
        AdaptiveBytes family, anchor;
        AdaptiveFamilyProofFragment witness;
        std::uint64_t revision = 0, maximum = 0;
        int rank;
        std::size_t receipt = 0;
    };
    std::vector<ProposalDelta> updates;
    std::vector<OwnerPacket> outgoing;
    const auto charge_key = [](ClosureScratch& memory, const AdaptiveCellKey& key) { charge_encoded_key(memory, key); };
    const auto record_member = [&](const AdaptiveLeaf& leaf, bool insert, auto& affected)
    {
        if (leaf.provider != AdaptiveProviderKind::Product)
            return;
        ClosureScratch temporary{work.budget};
        const auto cap = remaining(temporary);
        auto selectors = state.kernel->ancestry_family_selectors(leaf);
        Budget observed{cap};
        observed.add(
            persistent_ordered_detail::checked_product(selectors.capacity(), sizeof(AdaptiveProductFamilySelector)));
        for (const auto& selector : selectors)
            observed.add(fetch_key_bytes(selector.parent));
        temporary.drop(cap - observed.used);
        for (const auto& selector : selectors)
        {
            // adaptive_family_key adds a split suffix to the full parent key.
            ClosureScratch key_memory{work.budget};
            charge_key(key_memory, selector.parent);
            const auto key = adaptive_family_key(selector.parent, selector.split);
            edit.change_member(key, leaf.key, leaf.revision, insert);
            if (!affected.contains(key))
            {
                work.add(sizeof(AdaptiveBytes) + 4 * sizeof(void*) + key.size());
                affected.insert(key);
            }
        }
    };

    collective(*state.comm,
        [&]
        {
            std::set<AdaptiveBytes> affected;
            if (source)
                for (const auto& key : dirty)
                {
                    if (const auto* leaf = source->layout()->find(key))
                        record_member(*leaf, false, affected);
                    if (const auto* leaf = find_active(key))
                        record_member(*leaf, true, affected);
                }
            else
                state.visit_active([&](const AdaptiveLeaf& leaf) { record_member(leaf, true, affected); });

            for (const auto& key : affected)
            {
                const auto prior = accepted ? accepted->local(key) : std::nullopt;
                const auto next = edit.local(key);
                if (!prior && !next)
                    continue;
                const bool same_anchor =
                    prior && next && prior->anchor == next->anchor && prior->anchor_revision == next->anchor_revision;
                const bool fresh_witness =
                    next && (!same_anchor || std::binary_search(dirty.begin(), dirty.end(), next->anchor));
                if (same_anchor && !fresh_witness && prior->maximum == next->maximum)
                    continue;
                ProposalDelta delta;
                delta.rank = rank;
                work.add(key.size());
                delta.family = key;
                if (next)
                {
                    charge_key(work, next->anchor);
                    delta.anchor = encode_adaptive_key(next->anchor);
                    delta.revision = next->anchor_revision;
                    delta.maximum = next->maximum;
                }
                if (fresh_witness)
                {
                    const auto* leaf = find_active(next->anchor);
                    if (!leaf || leaf->revision != next->anchor_revision ||
                        leaf->provider != AdaptiveProviderKind::Product)
                        throw std::invalid_argument("Product contributor is not its actual local anchor.");
                    ClosureScratch witness{work.budget};
                    const auto cap = remaining(witness);
                    const auto selectors = state.kernel->ancestry_family_selectors(*leaf);
                    std::optional<AdaptiveFamily> capsule;
                    for (const auto& selector : selectors)
                        if (adaptive_family_key(selector.parent, selector.split) == key)
                        {
                            capsule.emplace(state.kernel->ancestry_family(*leaf, selector));
                            break;
                        }
                    if (!capsule || capsule->key != key || capsule->anchor != delta.anchor ||
                        capsule->parent.provider != AdaptiveProviderKind::Product)
                        throw std::invalid_argument("Product provider did not prove the selected family.");
                    const auto bytes = fetch_retention({}, std::span(&*capsule, 1), cap, false);
                    Budget observed{cap};
                    observed.add(bytes);
                    observed.add(persistent_ordered_detail::checked_product(
                        selectors.capacity(), sizeof(AdaptiveProductFamilySelector)));
                    for (const auto& selector : selectors)
                        observed.add(fetch_key_bytes(selector.parent));
                    witness.drop(cap - observed.used);
                    // The wire proof is tied to the actual anchor's revision.
                    // The rank maximum is a separate compact receipt.
                    capsule->revision = state.topology;
                    capsule->parent.revision = next->anchor_revision;
                    delta.witness = capture(*capsule);
                }
                work.append(updates);
                updates.push_back(std::move(delta));
            }
            ClosureScratch frame_memory{work.budget};
            std::vector<ProductProofFrame> frames;
            for (const auto& delta : updates)
            {
                const auto peer = authority(delta.family, ranks);
                if (peer == rank)
                    continue;
                frame_memory.append(frames);
                frames.push_back(
                    {peer, delta.family, delta.anchor, delta.revision, delta.maximum, delta.witness.view()});
            }
            encode_product_frames(frames, 1, context, work, outgoing);
            std::erase_if(updates, [&](const auto& delta) { return authority(delta.family, ranks) != rank; });
        });
    auto incoming = closure_exchange(outgoing, state.comm, work);
    std::vector<AdaptiveBytes> anchor_keys;
    collective(*state.comm,
        [&]
        {
            for (const auto& packet : incoming.packets)
                decode_product_frames(packet, 1, context, proposal_parts, work,
                    [&](const ProductProofFrame& row)
                    {
                        using persistent_ordered_detail::checked_product;
                        using persistent_ordered_detail::checked_sum;
                        const auto bytes = checked_sum(checked_sum(row.family.size(), row.anchor.size()),
                            checked_sum(row.proof.record.size(),
                                checked_product(row.proof.parts.size(), sizeof(AdaptiveFamilyProofPart))));
                        work.add(bytes);
                        work.append(updates);
                        updates.push_back({AdaptiveBytes(row.family.begin(), row.family.end()),
                            AdaptiveBytes(row.anchor.begin(), row.anchor.end()),
                            {AdaptiveBytes(row.proof.record.begin(), row.proof.record.end()),
                                {row.proof.parts.begin(), row.proof.parts.end()}},
                            row.revision, row.maximum, row.peer});
                    });
            std::sort(updates.begin(), updates.end(), [](const auto& a, const auto& b)
                { return a.family == b.family ? a.rank < b.rank : a.family < b.family; });
            for (std::size_t i = 0; i < updates.size(); ++i)
            {
                auto& update = updates[i];
                const bool duplicate =
                    i && updates[i - 1].family == update.family && updates[i - 1].rank == update.rank;
                validate_product_proposal_route(
                    {update.rank, update.family, update.anchor, update.revision, update.maximum, update.witness.view()},
                    rank, ranks, duplicate);
                if (!update.anchor.empty())
                {
                    update.receipt = anchor_keys.size();
                    work.append(anchor_keys);
                    work.add(update.anchor.size());
                    anchor_keys.push_back(update.anchor);
                }
            }
        });
    std::size_t lookup_cap = 0;
    collective(*state.comm, [&] { lookup_cap = remaining(work); });
    const auto owners = state.cells->lookup(anchor_keys, true, lookup_cap);
    collective(*state.comm,
        [&]
        {
            if (owners.size() != anchor_keys.size())
                throw std::logic_error("Product owner receipt count differs from its requests.");
            Budget observed{lookup_cap};
            observed.add(persistent_ordered_detail::checked_product(owners.capacity(), sizeof(AdaptiveOwner)));
            for (const auto& owner : owners)
                observed.add(owner.key.capacity());
            work.drop(lookup_cap - observed.used);
        });

    const auto equivalent = [&](const AdaptiveFamily& a, const AdaptiveFamily& b, ClosureScratch& memory)
    {
        if (a.key != b.key || a.split != b.split || a.parent.key != b.parent.key ||
            a.parent.provider != b.parent.provider || a.parent.domain != b.parent.domain ||
            a.parent.material != b.parent.material || a.children.size() != b.children.size())
            throw std::invalid_argument("Product factor authorities disagree on saved parent metadata.");
        // Full child identities decide equality independently of provider order.
        ClosureScratch comparison{memory.budget};
        comparison.add(persistent_ordered_detail::checked_product(2 * sizeof(AdaptiveCellKey), a.children.size()));
        for (const auto& child : a.children)
            comparison.add(fetch_key_bytes(child));
        for (const auto& child : b.children)
            comparison.add(fetch_key_bytes(child));
        auto first = a.children, second = b.children;
        std::sort(first.begin(), first.end());
        std::sort(second.begin(), second.end());
        if (first != second || !state.kernel->geometry(a.parent)->equivalent(*state.kernel->geometry(b.parent)))
            throw std::invalid_argument("Product factor authorities disagree on saved exact parent.");
    };
    const auto keep_delivered_part = [&](const AdaptiveFamilyProofPart& part)
    {
        if (!part || delivered_parts.contains(part.get()))
            return;
        work.add(sizeof(const void*) + 4 * sizeof(void*));
        delivered_parts.insert(part.get());
        if (const auto previous = accepted ? accepted->elected_part(*part) : nullptr)
        {
            if (previous.get() != part.get() || previous.owner_before(part) || part.owner_before(previous))
                throw std::logic_error("Product delivery bypassed its accepted part owner.");
            return;
        }
        const auto bytes = ProductWaveDictionary::body_bytes(*part);
        if (bytes > work.held)
            throw std::logic_error("Product delivered part lacks its wave admission.");
        work.held -= bytes;
        retained.held += bytes;
    };
    const auto deliver = [&](const ProductProofFrame& row)
    {
        if (row.peer != authority(row.family, ranks) || row.family.empty() || row.revision > row.maximum)
            throw std::invalid_argument("Foreign elected Product family authority.");
        retained.append(result);
        retained.add(row.family.size());
        AdaptiveBytes key(row.family.begin(), row.family.end());
        std::optional<AdaptiveFamily> family;
        if (!row.anchor.empty())
        {
            ClosureScratch scratch{work.budget};
            scratch.add(persistent_ordered_detail::checked_product(row.anchor.size(), 2));
            const auto anchor = decode_adaptive_key(row.anchor);
            const auto* leaf = find_active(anchor);
            if (!leaf || leaf->revision != row.revision || leaf->provider != AdaptiveProviderKind::Product)
                throw std::invalid_argument("Elected Product family lacks its actual local anchor.");
            if (row.proof.record.empty())
            {
                const auto prior = source ? source->family_layout->find(key) : std::nullopt;
                if (!prior || prior->parent.provider != AdaptiveProviderKind::Product ||
                    !std::ranges::equal(prior->anchor, row.anchor))
                    throw std::invalid_argument("Product revision receipt lacks its accepted exact witness.");
                charge_candidate_copy(retained, *prior);
                family.emplace(prior->materialize());
            }
            else
            {
                std::vector<AdaptiveFamilyProofPart> parts;
                scratch.add(persistent_ordered_detail::checked_product(
                    row.proof.parts.size(), sizeof(AdaptiveFamilyProofPart)));
                parts.reserve(row.proof.parts.size());
                if (parts.capacity() != row.proof.parts.size())
                    throw std::invalid_argument("Product delivered selector capacity differs from admission.");
                for (const auto& part : row.proof.parts)
                {
                    auto actual = accepted ? accepted->elected_part(*part) : nullptr;
                    parts.push_back(actual ? std::move(actual) : result_parts.retain(*part, work));
                }
                family.emplace(decode({row.proof.record, parts}, retained));
                if (family->key != key || !std::ranges::equal(family->anchor, row.anchor) ||
                    family->parent.revision != row.revision || family->parent.provider != AdaptiveProviderKind::Product)
                    throw std::invalid_argument("Elected Product proof differs from its exact receipt.");
                visit_elected_product_parts(family->parent, keep_delivered_part);
            }
            family->revision = state.topology;
            family->parent.revision = row.maximum;
        }
        else if (!row.proof.record.empty() || row.revision || row.maximum)
            throw std::invalid_argument("Invalid Product winner tombstone.");
        const bool changes_parts = row.anchor.empty() || !row.proof.record.empty();
        result.push_back({std::move(key), std::move(family), changes_parts});
    };

    struct WinnerDelta
    {
        int peer;
        const AdaptiveBytes* key;
        AdaptiveBytes anchor;
        std::uint64_t revision, maximum;
        std::shared_ptr<const ProductFamilyElectionState::Proof> proof;
    };
    std::vector<WinnerDelta> winners;
    std::vector<OwnerPacket>().swap(outgoing);
    collective(*state.comm,
        [&]
        {
            using ProposalPtr = ProductFamilyElectionState::ProposalPtr;
            for (std::size_t first = 0; first < updates.size();)
            {
                std::size_t last = first + 1;
                const auto& key = updates[first].family;
                while (last < updates.size() && updates[last].family == key)
                    ++last;
                const auto previous = accepted ? accepted->winner(key) : ProductFamilyElectionState::Winner{nullptr, 0};
                const auto old =
                    accepted ? accepted->proposals(key) : std::span<const ProductFamilyElectionState::ProposalPtr>{};
                ClosureScratch family_memory{work.budget};
                std::vector<ProposalPtr> next;
                const auto room = persistent_ordered_detail::checked_sum(old.size(), last - first);
                family_memory.add(persistent_ordered_detail::checked_product(room, sizeof(ProposalPtr)));
                next.reserve(room);
                if (next.capacity() != room)
                    throw std::invalid_argument("Product delta proposal capacity differs from admission.");
                std::optional<AdaptiveFamily> witness;
                // Any surviving prior proof authenticates the same accepted
                // exact body. If none survives, compare all newly supplied proofs.
                const bool has_fresh = std::any_of(updates.begin() + first, updates.begin() + last,
                    [](const auto& delta) { return !delta.witness.record.empty(); });
                if (has_fresh)
                    for (const auto& row : old)
                    {
                        const auto changed = std::lower_bound(updates.begin() + first, updates.begin() + last,
                            row->rank, [](const auto& delta, int peer) { return delta.rank < peer; });
                        if (changed == updates.begin() + last || changed->rank != row->rank ||
                            (!changed->anchor.empty() && changed->witness.record.empty()))
                        {
                            witness.emplace(decode(row->proof->view(), family_memory));
                            break;
                        }
                    }
                std::size_t prior = 0;
                for (std::size_t i = first; i < last; ++i)
                {
                    auto& change = updates[i];
                    while (prior < old.size() && old[prior]->rank < change.rank)
                        next.push_back(old[prior++]);
                    const auto* prior_rank =
                        prior < old.size() && old[prior]->rank == change.rank ? old[prior++].get() : nullptr;
                    if (change.anchor.empty())
                    {
                        if (!prior_rank)
                            throw std::invalid_argument("Product proposal tombstone lacks an accepted rank.");
                        continue;
                    }
                    validate_product_anchor_receipt(
                        change.anchor, change.rank, change.revision, owners[change.receipt]);
                    family_memory.add(persistent_ordered_detail::checked_product(change.anchor.size(), 2));
                    const auto anchor = decode_adaptive_key(change.anchor);
                    std::shared_ptr<const ProductFamilyElectionState::Proof> proof;
                    if (change.witness.record.empty())
                    {
                        if (!prior_rank || prior_rank->anchor != anchor ||
                            prior_rank->anchor_revision != change.revision)
                            throw std::invalid_argument("Product proposal lacks its accepted exact witness.");
                        proof = prior_rank->proof;
                    }
                    else
                    {
                        ClosureScratch decoded{family_memory.budget};
                        auto family = decode(change.witness.view(), decoded);
                        if (family.key != key || family.anchor != change.anchor || family.revision != state.topology ||
                            family.parent.revision != change.revision ||
                            family.parent.provider != AdaptiveProviderKind::Product)
                            throw std::invalid_argument("Product witness differs from its proposal.");
                        validate_family(*state.kernel, family);
                        if (witness)
                            equivalent(*witness, family, family_memory);
                        else
                        {
                            family_memory.held += decoded.held;
                            decoded.held = 0;
                            witness.emplace(std::move(family));
                        }
                        proof = edit.proof(change.witness.view());
                    }
                    next.push_back(
                        edit.proposal(change.rank, anchor, change.revision, change.maximum, std::move(proof)));
                }
                while (prior < old.size())
                    next.push_back(old[prior++]);
                edit.set_proposals(key, next);
                const auto elected = edit.winner(key);
                const auto send = [&](int peer, const ProductFamilyElectionState::Proposal* proposal,
                                      std::uint64_t maximum, bool fresh)
                {
                    WinnerDelta delta{peer, &key, {}, 0, maximum, fresh ? proposal->proof : nullptr};
                    if (proposal)
                    {
                        charge_key(work, proposal->anchor);
                        delta.anchor = encode_adaptive_key(proposal->anchor);
                        delta.revision = proposal->anchor_revision;
                    }
                    if (peer == rank)
                        deliver({rank, key, delta.anchor, delta.revision, maximum,
                            delta.proof ? delta.proof->view() : AdaptiveFamilyProofView{}});
                    else
                    {
                        work.append(winners);
                        winners.push_back(std::move(delta));
                    }
                };
                if (previous.proposal && (!elected.proposal || previous.proposal->rank != elected.proposal->rank))
                    send(previous.proposal->rank, nullptr, 0, false);
                if (elected.proposal)
                {
                    const auto fresh = !previous.proposal || previous.proposal->rank != elected.proposal->rank ||
                                       previous.proposal->proof != elected.proposal->proof;
                    if (fresh || previous.maximum != elected.maximum)
                        send(elected.proposal->rank, elected.proposal, elected.maximum, fresh);
                }
                first = last;
            }
            ClosureScratch frame_memory{work.budget};
            std::vector<ProductProofFrame> frames;
            for (const auto& row : winners)
            {
                frame_memory.append(frames);
                frames.push_back({row.peer, *row.key, row.anchor, row.revision, row.maximum,
                    row.proof ? row.proof->view() : AdaptiveFamilyProofView{}});
            }
            encode_product_frames(frames, 2, context, work, outgoing);
        });
    auto elected = closure_exchange(outgoing, state.comm, work);
    collective(*state.comm,
        [&]
        {
            for (const auto& packet : elected.packets)
                decode_product_frames(packet, 2, context, result_parts, work, deliver, accepted);
            std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
            if (std::adjacent_find(result.begin(), result.end(),
                    [](const auto& a, const auto& b) { return a.key == b.key; }) != result.end())
                throw std::invalid_argument("Duplicate elected Product family result.");
            if (source)
            {
                ProductElectedPartDelta parts(edit.keeper());
                for (const auto& row : result)
                {
                    if (!row.changes_parts)
                        continue;
                    if (const auto previous = source->family_layout->find(row.key))
                        parts.record(previous->parent, false);
                    if (row.family)
                        parts.record(row.family->parent, true);
                }
                parts.apply(edit);
            }
        });
}
