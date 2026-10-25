/** @file DistributedAdaptiveHierarchy.cc
 * @brief Distributed complete-family transactions with immutable local state.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <Tpetra_Map.hpp>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
using namespace adaptive_detail;
struct DistributedAdaptiveHierarchy::Data
{
    std::map<AdaptiveCellKey, AdaptiveLeaf> owned;
    std::map<AdaptiveBytes, AdaptiveFamily> families;
    std::shared_ptr<const AdaptiveProviderKernel> kernel;
    std::shared_ptr<const DistributedAncestryDirectory> cells, family_directory;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    DistributedAdaptiveOptions options;
    std::shared_ptr<const void> token = std::make_shared<const char>(0), topology_token = token,
                                partition_token = token;
    std::shared_ptr<const CompositePartition<>> native;
    Teuchos::RCP<const DefaultTpetraTypes::map_type> owned_map;
    std::shared_ptr<const void> map_token;
    std::uint64_t native_epoch = 0;
    std::uint64_t topology = 0, partition = 0, cell_begin = 0, global_cells = 0;
};
namespace
{
std::uint64_t transaction(const Teuchos::Comm<int>& comm)
{
    static std::atomic<std::uint64_t> counter{0};
    std::uint64_t id = 0;
    if (comm.getRank() == 0)
    {
        auto previous = counter.load();
        while (previous != std::numeric_limits<std::uint64_t>::max())
            if (counter.compare_exchange_weak(previous, previous + 1))
            {
                id = previous + 1;
                break;
            }
    }
    Teuchos::broadcast(comm, 0, 1, &id);
    if (!id)
        throw std::overflow_error("Distributed transaction identity exhausted.");
    return id;
}
void agree(const Teuchos::Comm<int>& comm, std::span<const std::uint64_t> descriptor)
{
    std::vector<std::uint64_t> root;
    collective(comm, [&] { root.assign(descriptor.begin(), descriptor.end()); });
    Teuchos::broadcast(comm, 0, int(root.size()), root.data());
    collective(comm,
        [&]
        {
            if (!std::equal(root.begin(), root.end(), descriptor.begin(), descriptor.end()))
                throw std::invalid_argument("Distributed semantic descriptors differ.");
        });
}
void intern_state(DistributedAdaptiveHierarchy::Data& state)
{
    std::vector<AdaptiveLeaf> leaves;
    std::vector<AdaptiveFamily> families;
    for (const auto& [key, leaf] : state.owned)
        leaves.push_back(leaf);
    for (const auto& [key, family] : state.families)
        families.push_back(family);
    auto packet = encode_adaptive_batch(leaves, families, state.options.byte_budget);
    auto shared = decode_adaptive_batch(packet, state.options.byte_budget);
    state.owned.clear();
    state.families.clear();
    for (auto& leaf : shared.leaves)
        state.owned.emplace(leaf.key, std::move(leaf));
    for (auto& family : shared.families)
        state.families.emplace(family.key, std::move(family));
}
bool descendant(const AdaptiveCellKey& parent, const AdaptiveCellKey& child)
{
    if (parent.region != child.region || parent.ancestry.index() != child.ancestry.index())
        return false;
    auto prefix = [](const auto& a, const auto& b)
    { return a.path.size() <= b.path.size() && std::equal(a.path.begin(), a.path.end(), b.path.begin()); };
    return std::visit(
        [&](const auto& p)
        {
            using T = std::decay_t<decltype(p)>;
            const auto& c = std::get<T>(child.ancestry);
            if constexpr (std::is_same_v<T, CartesianCellAncestry>)
            {
                if (c.level < p.level || c.level - p.level >= 64)
                    return false;
                for (unsigned a = 0; a < 3; ++a)
                    if ((c.index[a] >> (c.level - p.level)) != p.index[a])
                        return false;
                return true;
            }
            else if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                return p.root == c.root && prefix(p, c);
            else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                return p.base.root == c.base.root && p.base.root_template == c.base.root_template &&
                       p.axial.root == c.axial.root && prefix(p.base, c.base) && prefix(p.axial, c.axial);
            else
                return p == c;
        },
        parent.ancestry);
}
void validate_family(const AdaptiveProviderKernel& kernel, const AdaptiveFamily& family)
{
    kernel.validate(family.parent);
    if (family.key != adaptive_family_key(family.parent.key, family.split) || family.anchor.empty() ||
        !descendant(family.parent.key, decode_adaptive_key(family.anchor)))
        throw std::invalid_argument("Family key/anchor is not actual typed ancestry.");
    auto expected = kernel.split(family.parent, family.split).family.children, actual = family.children;
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    if (expected != actual || std::adjacent_find(actual.begin(), actual.end()) != actual.end())
        throw std::invalid_argument("Family capsule does not name complete immediate provider children.");
}
void refresh_product_families(DistributedAdaptiveHierarchy::Data& state)
{
    std::vector<OwnerPacket> packets;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            std::erase_if(state.families,
                [](const auto& entry) { return entry.second.parent.provider == AdaptiveProviderKind::Product; });
            std::map<AdaptiveBytes, AdaptiveFamily> local;
            for (const auto& [key, leaf] : state.owned)
                if (leaf.provider == AdaptiveProviderKind::Product)
                    for (auto family : state.kernel->ancestry_families(leaf))
                    {
                        family.revision = state.topology;
                        family.anchor = encode_adaptive_key(key);
                        auto [it, inserted] = local.emplace(family.key, family);
                        if (!inserted)
                        {
                            const auto revision = std::max(it->second.parent.revision, family.parent.revision);
                            if (decode_adaptive_key(family.anchor) < decode_adaptive_key(it->second.anchor))
                                it->second = std::move(family);
                            it->second.parent.revision = revision;
                        }
                        else
                            budget.add(sizeof(AdaptiveFamily) + family.key.size());
                    }
            std::map<int, std::vector<AdaptiveFamily>> groups;
            for (auto& [key, family] : local)
                groups[authority(key, state.comm->getSize())].push_back(std::move(family));
            for (const auto& [rank, families] : groups)
            {
                auto bytes = encode_adaptive_batch({}, families, state.options.byte_budget);
                budget.packet(bytes.size());
                packets.push_back({rank, std::move(bytes)});
            }
        });
    auto incoming = exchange_owner_packets(packets, state.comm, state.options.byte_budget);
    struct Candidate
    {
        AdaptiveFamily family;
        int owner;
    };
    std::vector<Candidate> candidates;
    std::vector<AdaptiveBytes> anchors;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            for (const auto& packet : incoming.packets)
            {
                budget.add(packet.bytes.size());
                auto batch = decode_adaptive_batch(packet.bytes, state.options.byte_budget);
                if (!batch.leaves.empty())
                    throw std::invalid_argument("Unexpected leaf in factor authority packet.");
                for (auto& family : batch.families)
                {
                    budget.add(sizeof(Candidate));
                    validate_family(*state.kernel, family);
                    if (authority(family.key, state.comm->getSize()) != state.comm->getRank())
                        throw std::invalid_argument("Misrouted product factor family.");
                    anchors.push_back(family.anchor);
                    candidates.push_back({std::move(family), packet.peer});
                }
            }
        });
    auto owners = state.cells->lookup(anchors);
    packets.clear();
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            std::map<AdaptiveBytes, Candidate> elected;
            for (std::size_t i = 0; i < candidates.size(); ++i)
            {
                auto candidate = std::move(candidates[i]);
                if (owners[i].owner != candidate.owner)
                    throw std::invalid_argument("Product family candidate lacks actual anchor ownership.");
                auto [it, inserted] = elected.emplace(candidate.family.key, candidate);
                if (!inserted)
                {
                    auto& old = it->second;
                    const auto& a = old.family.parent;
                    const auto& b = candidate.family.parent;
                    auto ac = old.family.children, bc = candidate.family.children;
                    std::sort(ac.begin(), ac.end());
                    std::sort(bc.begin(), bc.end());
                    if (ac != bc || a.domain != b.domain || a.material != b.material || a.provider != b.provider ||
                        !state.kernel->geometry(a)->equivalent(*state.kernel->geometry(b)))
                        throw std::invalid_argument("Product factor authorities disagree on saved exact parent.");
                    const auto revision = std::max(a.revision, b.revision);
                    if (decode_adaptive_key(candidate.family.anchor) < decode_adaptive_key(old.family.anchor))
                        old = std::move(candidate);
                    old.family.parent.revision = revision;
                }
            }
            std::map<int, std::vector<AdaptiveFamily>> groups;
            for (auto& [key, candidate] : elected)
                groups[candidate.owner].push_back(std::move(candidate.family));
            for (const auto& [rank, families] : groups)
            {
                auto bytes = encode_adaptive_batch({}, families, state.options.byte_budget);
                budget.packet(bytes.size());
                packets.push_back({rank, std::move(bytes)});
            }
        });
    auto elected = exchange_owner_packets(packets, state.comm, state.options.byte_budget);
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            for (const auto& packet : elected.packets)
            {
                budget.add(packet.bytes.size());
                auto batch = decode_adaptive_batch(packet.bytes, state.options.byte_budget);
                if (!batch.leaves.empty())
                    throw std::invalid_argument("Unexpected elected leaf packet.");
                for (auto& family : batch.families)
                    if (packet.peer != authority(family.key, state.comm->getSize()) ||
                        !state.owned.contains(decode_adaptive_key(family.anchor)) ||
                        !state.families.emplace(family.key, std::move(family)).second)
                        throw std::invalid_argument("Invalid elected product factor authority.");
            }
        });
}
void register_state(DistributedAdaptiveHierarchy::Data& state)
{
    const std::uint64_t local = state.owned.size();
    std::uint64_t prefix = 0;
    collective(*state.comm,
        [&]
        {
            if (local > std::numeric_limits<std::uint64_t>::max() / std::uint64_t(state.comm->getSize()))
                throw std::overflow_error("Global adaptive cell count exceeds safe prefix bound.");
        });
    Teuchos::scan(*state.comm, Teuchos::REDUCE_SUM, 1, &local, &prefix);
    Teuchos::reduceAll(*state.comm, Teuchos::REDUCE_SUM, 1, &local, &state.global_cells);
    state.cell_begin = prefix - local;
    std::vector<AdaptiveOwner> cells, families;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            std::uint64_t ordinal = 0;
            for (const auto& [key, leaf] : state.owned)
            {
                auto bytes = encode_adaptive_key(key);
                budget.add(sizeof(AdaptiveOwner));
                budget.add(bytes.size());
                cells.push_back({std::move(bytes), state.comm->getRank(), leaf.revision, state.cell_begin + ordinal++});
            }
        });
    state.cells = std::make_shared<const DistributedAncestryDirectory>(cells, state.comm, state.options.byte_budget);
    std::vector<AdaptiveBytes> partner_keys;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            for (const auto& [key, leaf] : state.owned)
                for (const auto& partner : state.kernel->matching_partners(leaf))
                {
                    auto bytes = encode_adaptive_key(partner);
                    budget.add(bytes.size());
                    partner_keys.push_back(std::move(bytes));
                }
        });
    (void) state.cells->lookup(partner_keys);
    refresh_product_families(state);
    collective(*state.comm, [&] { intern_state(state); });
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            for (const auto& [key, family] : state.families)
            {
                validate_family(*state.kernel, family);
                if (!state.owned.contains(decode_adaptive_key(family.anchor)))
                    throw std::invalid_argument("Saved parent authority lost actual descendant owner.");
                budget.add(sizeof(AdaptiveOwner));
                budget.add(key.size());
                families.push_back({key, state.comm->getRank(), family.revision});
            }
        });
    state.family_directory =
        std::make_shared<const DistributedAncestryDirectory>(families, state.comm, state.options.byte_budget);
}
std::vector<AdaptiveMark> closure_marks(const DistributedAdaptiveHierarchy::Data& state)
{
    std::vector<AdaptiveLeaf> owned;
    collective(*state.comm,
        [&]
        {
            Budget budget{state.options.byte_budget};
            for (const auto& [key, leaf] : state.owned)
            {
                budget.add(sizeof(AdaptiveLeaf) + encode_adaptive_key(key).size());
                owned.push_back(leaf);
            }
        });
    AdaptiveClosureOptions options;
    options.byte_budget = state.options.byte_budget;
    options.contact_budget = state.options.maximum_tiles;
    options.tile_budget = state.options.maximum_tiles;
    options.maximum_span_ratio = state.options.maximum_span_ratio;
    return required_adaptive_refinements(owned, *state.kernel, state.comm, options);
}
std::optional<AdaptiveCellKey> original_cover(
    const DistributedAdaptiveHierarchy::Data& source, const AdaptiveCellKey& key)
{
    if (source.owned.contains(key))
        return key;
    AdaptiveCellKey ancestor = key;
    if (auto* p = std::get_if<CartesianCellAncestry>(&ancestor.ancestry))
    {
        while (p->level)
        {
            *p = p->parent();
            if (source.owned.contains(ancestor))
                return ancestor;
        }
    }
    else if (auto* p = std::get_if<ConvexCellAncestry>(&ancestor.ancestry))
    {
        while (!p->path.empty())
        {
            p->path.pop_back();
            if (source.owned.contains(ancestor))
                return ancestor;
        }
    }
    else if (auto* p = std::get_if<ProductCellAncestry>(&ancestor.ancestry))
    {
        for (;;)
        {
            auto probe = ancestor;
            auto& axial = std::get<ProductCellAncestry>(probe.ancestry).axial;
            for (;;)
            {
                if (source.owned.contains(probe))
                    return probe;
                if (axial.path.empty())
                    break;
                axial.path.pop_back();
            }
            if (p->base.path.empty())
                break;
            p->base.path.pop_back();
        }
    }
    return {};
}
bool repeated_factor(const AdaptiveCellKey& origin, const AdaptiveCellKey& current, AdaptiveSplit split)
{
    return std::visit(
        [&](const auto& old)
        {
            using T = std::decay_t<decltype(old)>;
            const auto& now = std::get<T>(current.ancestry);
            if constexpr (std::is_same_v<T, CartesianCellAncestry>)
                return now.level > old.level;
            else if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                return now.path.size() > old.path.size();
            else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                return (split != AdaptiveSplit::Axial && now.base.path.size() > old.base.path.size()) ||
                       (split != AdaptiveSplit::Base && now.axial.path.size() > old.axial.path.size());
            else
                return true;
        },
        origin.ancestry);
}
void close_candidate(DistributedAdaptiveHierarchy::Data& candidate, const DistributedAdaptiveHierarchy::Data& source)
{
    for (unsigned round = 0; round < candidate.options.maximum_closure_rounds; ++round)
    {
        auto needed = closure_marks(candidate);
        std::uint64_t local = needed.size(), global = 0;
        Teuchos::reduceAll(*candidate.comm, Teuchos::REDUCE_SUM, 1, &local, &global);
        if (!global)
            return;
        std::vector<AdaptiveBytes> keys;
        std::vector<AdaptiveSplit> kinds;
        collective(*candidate.comm,
            [&]
            {
                Budget budget{candidate.options.byte_budget};
                for (const auto& mark : needed)
                    for (const auto& member : candidate.kernel->refinement_group(mark.key, mark.split))
                    {
                        auto key = encode_adaptive_key(member);
                        budget.add(key.size() + sizeof(AdaptiveSplit));
                        keys.push_back(std::move(key));
                        kinds.push_back(mark.split);
                    }
            });
        auto owners = candidate.cells->lookup(keys);
        std::vector<OwnerPacket> packets;
        collective(*candidate.comm,
            [&]
            {
                Budget budget{candidate.options.byte_budget};
                for (std::size_t i = 0; i < owners.size(); ++i)
                {
                    Writer w;
                    w.bytes(owners[i].key);
                    w.word(std::uint64_t(kinds[i]));
                    w.word(owners[i].revision);
                    budget.packet(w.data.size());
                    packets.push_back({owners[i].owner, std::move(w.data)});
                }
            });
        auto requests = exchange_owner_packets(packets, candidate.comm, candidate.options.byte_budget);
        collective(*candidate.comm,
            [&]
            {
                Budget budget{candidate.options.byte_budget};
                std::map<AdaptiveCellKey, AdaptiveSplit> marks;
                for (const auto& packet : requests.packets)
                {
                    Reader r{packet.bytes};
                    auto key = decode_adaptive_key(r.bytes());
                    auto split = r.word(), revision = r.word();
                    r.done();
                    auto it = candidate.owned.find(key);
                    if (split > 2 || it == candidate.owned.end() || it->second.revision != revision)
                        throw std::invalid_argument("Stale distributed closure group.");
                    auto [mark, inserted] = marks.emplace(key, AdaptiveSplit(split));
                    if (!inserted && mark->second != AdaptiveSplit(split))
                    {
                        if (it->second.provider != AdaptiveProviderKind::Product)
                            throw std::invalid_argument("Incompatible native closure factors.");
                        mark->second = AdaptiveSplit::Isotropic;
                    }
                }
                for (const auto& [key, split] : marks)
                {
                    const auto origin = original_cover(source, key);
                    if (!origin || repeated_factor(*origin, key, split))
                        throw std::invalid_argument(
                            "Closure would require multiple source generations or undo coarsening.");
                    auto result = candidate.kernel->split(candidate.owned.at(key), split);
                    candidate.owned.erase(key);
                    for (auto& [family_key, family] : candidate.families)
                        if (family.anchor == encode_adaptive_key(key))
                            family.anchor = result.family.anchor;
                    for (auto& child : result.children)
                    {
                        budget.add(sizeof(AdaptiveLeaf) + encode_adaptive_key(child.key).size());
                        if (!candidate.owned.emplace(child.key, std::move(child)).second)
                            throw std::invalid_argument("Closure child already active.");
                    }
                    if (!candidate.families.emplace(result.family.key, std::move(result.family)).second)
                        throw std::invalid_argument("Closure family already retained.");
                }
                if (candidate.owned.size() > candidate.options.maximum_owned)
                    throw std::invalid_argument("Closure local leaf budget exhausted.");
                std::vector<AdaptiveLeaf> leaves;
                for (const auto& [key, leaf] : candidate.owned)
                    leaves.push_back(leaf);
                auto repacked = candidate.kernel->repack(leaves);
                candidate.owned.clear();
                for (auto& leaf : repacked)
                {
                    candidate.kernel->validate(leaf);
                    candidate.owned.emplace(leaf.key, std::move(leaf));
                }
            });
        register_state(candidate);
    }
    throw std::invalid_argument("Distributed closure round budget exhausted.");
}
} // namespace
DistributedAdaptiveHierarchy::DistributedAdaptiveHierarchy(std::vector<AdaptiveLeaf> owned,
    std::vector<AdaptiveFamily> families, std::shared_ptr<const AdaptiveProviderKernel> kernel,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, DistributedAdaptiveOptions options)
{
    if (comm.is_null())
        throw std::invalid_argument("Distributed hierarchy needs communicator.");
    std::shared_ptr<Data> state;
    collective(*comm,
        [&]
        {
            state = std::make_shared<Data>();
            state->comm = comm;
            state->options = options;
            state->kernel = std::move(kernel);
        });
    collective(*comm,
        [&]
        {
            if (!state->kernel || !options.byte_budget || !options.maximum_owned || !options.maximum_tiles ||
                !options.maximum_closure_rounds || !std::isfinite(options.maximum_span_ratio) ||
                options.maximum_span_ratio < 1 || owned.size() > options.maximum_owned)
                throw std::invalid_argument("Invalid distributed hierarchy limits.");
            Budget budget{options.byte_budget};
            for (auto& leaf : owned)
            {
                state->kernel->validate(leaf);
                budget.add(encode_adaptive_key(leaf.key).size() + leaf.domain.size() + leaf.parent_family.size());
                budget.add(sizeof(AdaptiveLeaf));
                if (!state->owned.emplace(leaf.key, std::move(leaf)).second)
                    throw std::invalid_argument("Duplicate owned adaptive leaf.");
            }
            for (auto& family : families)
            {
                if (!state->owned.contains(decode_adaptive_key(family.anchor)))
                    throw std::invalid_argument("Family capsule must reside with actual anchor descendant.");
                state->kernel->validate(family.parent);
                budget.add(family.key.size() + family.anchor.size() + family.children.size() * sizeof(AdaptiveCellKey));
                budget.add(sizeof(AdaptiveFamily));
                if (!state->families.emplace(family.key, std::move(family)).second)
                    throw std::invalid_argument("Duplicate retained family capsule.");
            }
        });
    const std::array<std::uint64_t, 6> description{options.byte_budget, options.maximum_owned, options.maximum_tiles,
        options.maximum_closure_rounds, std::bit_cast<std::uint64_t>(options.maximum_span_ratio), 1};
    agree(*comm, description);
    AdaptiveBytes kernel_descriptor;
    collective(*comm,
        [&]
        {
            kernel_descriptor = state->kernel->descriptor();
            if (kernel_descriptor.size() > options.byte_budget)
                throw std::invalid_argument("Kernel semantic descriptor exceeds budget.");
        });
    std::uint64_t descriptor_size = kernel_descriptor.size(), root_size = descriptor_size;
    Teuchos::broadcast(*comm, 0, 1, &root_size);
    collective(*comm,
        [&]
        {
            if (root_size != descriptor_size || root_size > std::size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Kernel semantic descriptors have different lengths.");
        });
    AdaptiveBytes root_descriptor;
    collective(*comm, [&] { root_descriptor = kernel_descriptor; });
    Teuchos::broadcast(*comm, 0, int(root_size), root_descriptor.data());
    collective(*comm,
        [&]
        {
            if (root_descriptor != kernel_descriptor)
                throw std::invalid_argument("Kernel semantic descriptors differ.");
        });
    register_state(*state);
    const auto initial_requirements = closure_marks(*state);
    collective(*comm,
        [&]
        {
            if (!initial_requirements.empty())
                throw std::invalid_argument("Initial distributed hierarchy contact grading is inadmissible.");
        });
    data_ = std::move(state);
}
const std::map<AdaptiveCellKey, AdaptiveLeaf>& DistributedAdaptiveHierarchy::owned() const noexcept
{
    return data_->owned;
}
const std::map<AdaptiveBytes, AdaptiveFamily>& DistributedAdaptiveHierarchy::families() const noexcept
{
    return data_->families;
}
std::uint64_t DistributedAdaptiveHierarchy::topology_generation() const noexcept
{
    return data_->topology;
}
std::uint64_t DistributedAdaptiveHierarchy::partition_generation() const noexcept
{
    return data_->partition;
}
const std::shared_ptr<const void>& DistributedAdaptiveHierarchy::identity() const noexcept
{
    return data_->token;
}
const Teuchos::RCP<const Teuchos::Comm<int>>& DistributedAdaptiveHierarchy::communicator() const noexcept
{
    return data_->comm;
}
const DistributedAdaptiveOptions& DistributedAdaptiveHierarchy::options() const noexcept
{
    return data_->options;
}
std::uint64_t DistributedAdaptiveHierarchy::global_cells() const noexcept
{
    return data_->global_cells;
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> DistributedAdaptiveHierarchy::geometry(const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return data_->kernel->geometry(leaf);
}
AdaptiveCellMetrics DistributedAdaptiveHierarchy::metrics(const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return data_->kernel->metrics(leaf);
}
std::vector<AdaptiveSurface> DistributedAdaptiveHierarchy::surfaces(const AdaptiveLeaf& leaf) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Distributed resident geometry moved.");
    return data_->kernel->surfaces(leaf);
}
bool DistributedAdaptiveHierarchy::geometry_unchanged() const noexcept
{
    if (!data_->native)
        return true;
    try
    {
        return data_->native->geometry_unchanged() &&
               data_->native->geometry()->geometry_epoch() == data_->native_epoch;
    }
    catch (...)
    {
        return false;
    }
}
HierarchyObservation DistributedAdaptiveHierarchy::observation() const noexcept
{
    std::uint64_t epoch = data_->native_epoch;
    try
    {
        if (data_->native)
            epoch = data_->native->geometry()->geometry_epoch();
    }
    catch (...)
    {
        epoch = std::numeric_limits<std::uint64_t>::max();
    }
    std::shared_ptr<const void> ownership = data_->map_token ? data_->map_token
                                            : data_->native  ? std::shared_ptr<const void>(data_->native)
                                                             : data_->partition_token;
    return {data_->token, data_->topology_token, ownership, data_->topology, data_->partition, epoch,
        data_->native ? data_->native->geometry().get() : nullptr, data_->comm.getRawPtr()};
}
const std::shared_ptr<const CompositePartition<>>& DistributedAdaptiveHierarchy::native_partition() const noexcept
{
    return data_->native;
}
std::shared_ptr<const DistributedAdaptiveHierarchy> DistributedAdaptiveHierarchy::with_partition(
    std::shared_ptr<const CompositePartition<>> partition, Teuchos::RCP<const DefaultTpetraTypes::map_type> map) const
{
    std::shared_ptr<Data> bound;
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || !partition || !partition->geometry_unchanged() ||
                partition->plan().communicator().getRawPtr() != data_->comm.getRawPtr() ||
                !partition->geometry()->sparse_native() || partition->geometry()->num_cells() != data_->global_cells ||
                partition->plan().owned_cells().size() != data_->owned.size())
                throw std::invalid_argument("Foreign, incomplete or moved distributed native binding.");
            if (map.is_null() || map->getComm().getRawPtr() != data_->comm.getRawPtr() ||
                map->getGlobalNumElements() != data_->global_cells ||
                map->getLocalNumElements() != data_->owned.size() || map->getIndexBase() != 0)
                throw std::invalid_argument("Foreign or incomplete actual distributed owned map.");
            std::uint64_t id = data_->cell_begin;
            std::size_t ordinal = 0;
            for (const auto& [key, leaf] : data_->owned)
            {
                if (partition->plan().owned_cells()[ordinal] != id ||
                    map->getGlobalElement(ordinal) != DefaultTpetraTypes::global_ordinal_type(id) ||
                    partition->geometry()->sparse_cell_owner(id) != data_->comm->getRank())
                    throw std::invalid_argument("Native binding does not own actual typed cell GID.");
                const auto [region, local_cell] = partition->geometry()->native_cell(id);
                (void) local_cell;
                const auto family = partition->geometry()->region_layout(region).family;
                if ((leaf.provider == AdaptiveProviderKind::Cartesian && family != RegionLayout::Family::Rectilinear) ||
                    (leaf.provider == AdaptiveProviderKind::Product && family != RegionLayout::Family::Extruded))
                    throw std::invalid_argument("Procedural adaptive family was flattened in native binding.");
                if (!partition->geometry()->certified_sparse_cell_geometry(id)->equivalent(
                        *data_->kernel->geometry(leaf)))
                    throw std::invalid_argument("Native binding differs from actual typed cell geometry.");
                ++id;
                ++ordinal;
            }
            bound = std::make_shared<Data>(*data_);
            bound->native = std::move(partition);
            bound->owned_map = map;
            auto holder = std::make_shared<const Teuchos::RCP<const DefaultTpetraTypes::map_type>>(map);
            bound->map_token = std::shared_ptr<const void>(holder, map.getRawPtr());
            bound->native_epoch = bound->native->geometry()->geometry_epoch();
        });
    const auto& plan = bound->native->plan();
    agree(*data_->comm, std::array<std::uint64_t, 4>{
                            plan.collective_identity(), plan.global_cells(), plan.global_faces(), plan.global_nodes()});
    return std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(std::move(bound)));
}
namespace
{
std::size_t fetch_key_bytes(const AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& p) -> std::size_t
        {
            if constexpr (requires { p.path; })
                return p.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (requires { p.base.path; })
                return (p.base.path.capacity() + p.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
void fetch_copy_charge(Budget& budget, const AdaptiveLeaf& leaf)
{
    budget.add(sizeof(AdaptiveLeaf));
    budget.add(fetch_key_bytes(leaf.key));
    budget.add(leaf.domain.capacity());
    budget.add(leaf.parent_family.capacity());
    if (leaf.surfaces.capacity() > budget.limit / sizeof(AdaptiveSurface))
        throw std::invalid_argument("Needed leaf surface count exceeds lease.");
    budget.add(leaf.surfaces.capacity() * sizeof(AdaptiveSurface));
    for (const auto& face : leaf.surfaces)
    {
        budget.add(face.key.capacity());
        budget.add(face.boundary.capacity());
        budget.add(face.authority_image.capacity());
    }
}
void fetch_copy_charge(Budget& budget, const AdaptiveFamily& family)
{
    budget.add(sizeof(AdaptiveFamily));
    budget.add(family.key.capacity());
    budget.add(family.anchor.capacity());
    if (family.children.capacity() > budget.limit / sizeof(AdaptiveCellKey))
        throw std::invalid_argument("Needed family child count exceeds lease.");
    budget.add(family.children.capacity() * sizeof(AdaptiveCellKey));
    for (const auto& key : family.children)
        budget.add(fetch_key_bytes(key));
    fetch_copy_charge(budget, family.parent);
}
// Per-message decoded retention admission. Shared procedural dictionaries are
// counted once, while separately decoded cell/face certificates are counted per
// occurrence. The sender reserves growth slack; the receiver checks capacities.
std::size_t fetch_retention(std::span<const AdaptiveLeaf> leaves, std::span<const AdaptiveFamily> families,
    std::size_t limit, bool reserve_growth)
{
    Budget bytes{limit};
    std::set<const void*> objects;
    auto charge = [&](std::size_t n)
    {
        if (reserve_growth)
        {
            if (n > limit / 4)
                throw std::invalid_argument("Needed decoded record exceeds retention lease.");
            n *= 4;
        }
        bytes.add(n);
    };
    auto unique = [&](const void* p)
    {
        if (!p || objects.contains(p))
            return false;
        bytes.add(8 * sizeof(void*));
        objects.insert(p);
        return true;
    };
    auto part = [&](const std::shared_ptr<const AdaptiveBytes>& p)
    {
        if (p && unique(p.get()))
            charge(sizeof(AdaptiveBytes) + p->capacity());
    };
    auto leaf = [&](const AdaptiveLeaf& value)
    {
        Budget metadata{limit};
        fetch_copy_charge(metadata, value);
        charge(metadata.used);
        if (value.native_recipe && unique(value.native_recipe.get()))
        {
            charge(sizeof(NativeFragmentDescriptor) + value.native_recipe->view.capacity() +
                   value.native_recipe->parts.capacity() * sizeof(std::shared_ptr<const AdaptiveBytes>));
            for (const auto& p : value.native_recipe->parts)
                part(p);
        }
        part(value.join_recipe);
        if (value.factor_recipe && unique(value.factor_recipe.get()))
        {
            charge(sizeof(AdaptiveFactorRecipe));
            part(value.factor_recipe->root);
            part(value.factor_recipe->base);
            part(value.factor_recipe->axis);
        }
        if (value.geometry)
            charge(value.geometry->storage_bytes());
        for (const auto& face : value.surfaces)
        {
            if (face.geometry)
                charge(face.geometry->storage_bytes());
            if (face.native_whole_geometry)
                charge(face.native_whole_geometry->storage_bytes());
        }
    };
    for (const auto& value : leaves)
        leaf(value);
    for (const auto& value : families)
    {
        Budget metadata{limit};
        fetch_copy_charge(metadata, value);
        charge(metadata.used);
        leaf(value.parent);
    }
    return bytes.used;
}
AdaptiveBatch fetch_records(const DistributedAdaptiveHierarchy::Data& state, std::span<const AdaptiveBytes> keys,
    bool family, std::size_t cap, bool leased)
{
    const auto& directory = family ? state.family_directory : state.cells;
    const auto packet_cap = leased ? cap / 8 : cap;
    const auto exchange_cap = leased ? cap / 2 : cap;
    const auto batch_cap = leased ? cap / 4 : cap;
    const auto owner_cap = leased ? std::optional<std::size_t>(cap / 4) : std::nullopt;
    collective(*state.comm,
        [&]
        {
            if (leased && packet_cap < 256)
                throw std::invalid_argument("Needed-record lease is too small.");
        });
    auto owners = directory->lookup(keys, true, owner_cap);
    const auto check_packets = [&](const auto& packets)
    {
        if (!leased)
            return;
        Budget live{cap / 4};
        if (packets.capacity() > (cap / 4) / sizeof(OwnerPacket))
            throw std::invalid_argument("Needed packet metadata exceeds transient lease.");
        live.add(packets.capacity() * sizeof(OwnerPacket));
        for (const auto& packet : packets)
            live.add(packet.bytes.capacity());
    };
    std::vector<OwnerPacket> packets;
    collective(*state.comm,
        [&]
        {
            Budget budget{packet_cap};
            std::map<int, Writer> groups;
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                budget.add(keys[i].size() + 24);
                auto& w = groups[owners[i].owner];
                w.word(i);
                w.bytes(keys[i]);
                w.word(owners[i].revision);
            }
            for (auto& [rank, w] : groups)
            {
                budget.packet(w.data.size());
                packets.push_back({rank, std::move(w.data)});
            }
        });
    collective(*state.comm, [&] { check_packets(packets); });
    auto incoming = exchange_owner_packets(packets, state.comm, exchange_cap);
    packets.clear();
    collective(*state.comm,
        [&]
        {
            Budget budget{leased ? cap / 4 : cap};
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                Writer order;
                std::vector<AdaptiveLeaf> leaves;
                std::vector<AdaptiveFamily> families;
                if (leased)
                {
                    // Count framed requests without allocating key copies. Reserve
                    // the outer record array before any immutable source copy.
                    Reader scan{packet.bytes};
                    std::size_t count = 0;
                    while (scan.offset < scan.data.size())
                    {
                        (void) scan.word();
                        const auto length = scan.word();
                        if (length > scan.data.size() - scan.offset)
                            throw std::invalid_argument("Invalid needed-record request framing.");
                        scan.offset += length;
                        (void) scan.word();
                        ++count;
                    }
                    const auto record_bytes = family ? sizeof(AdaptiveFamily) : sizeof(AdaptiveLeaf);
                    if (count > budget.limit / (2 * record_bytes))
                        throw std::invalid_argument("Needed source-record vector exceeds lease.");
                    budget.add(2 * count * record_bytes);
                    if (family)
                    {
                        families.reserve(count);
                        if (families.capacity() > 2 * count)
                            throw std::invalid_argument("Needed family vector capacity exceeds admission.");
                    }
                    else
                    {
                        leaves.reserve(count);
                        if (leaves.capacity() > 2 * count)
                            throw std::invalid_argument("Needed leaf vector capacity exceeds admission.");
                    }
                }
                while (r.offset < r.data.size())
                {
                    auto index = r.word();
                    auto key = r.bytes();
                    auto revision = r.word();
                    budget.add(key.size() + 16);
                    order.word(index);
                    if (family)
                    {
                        auto it = state.families.find(key);
                        if (it == state.families.end() || it->second.revision != revision)
                            throw std::invalid_argument("Missing or stale needed family capsule.");
                        if (leased)
                            fetch_copy_charge(budget, it->second);
                        families.push_back(it->second);
                    }
                    else
                    {
                        auto typed = decode_adaptive_key(key);
                        auto it = state.owned.find(typed);
                        if (it == state.owned.end() || it->second.revision != revision)
                            throw std::invalid_argument("Missing or stale needed owned leaf.");
                        if (leased)
                            fetch_copy_charge(budget, it->second);
                        leaves.push_back(it->second);
                    }
                }
                if (!family && !leased)
                {
                    std::map<AdaptiveCellKey, AdaptiveLeaf> unique;
                    for (const auto& leaf : leaves)
                        unique.emplace(leaf.key, leaf);
                    std::vector<AdaptiveLeaf> selected;
                    for (const auto& [key, leaf] : unique)
                        selected.push_back(leaf);
                    auto packed = state.kernel->repack(selected);
                    unique.clear();
                    for (auto& leaf : packed)
                        unique.emplace(leaf.key, std::move(leaf));
                    for (auto& leaf : leaves)
                        leaf = unique.at(leaf.key);
                }
                const auto retention = leased ? fetch_retention(leaves, families, batch_cap, true) : 0;
                if (leased)
                {
                    // The owner table, received requests, copies/order, prior replies
                    // and codec arena coexist until this reply has been constructed.
                    Budget live{cap};
                    live.add(owners.capacity() * sizeof(AdaptiveOwner));
                    for (const auto& owner : owners)
                        live.add(owner.key.capacity());
                    live.add(incoming.packets.capacity() * sizeof(OwnerPacket));
                    for (const auto& input : incoming.packets)
                        live.add(input.bytes.capacity());
                    live.add(budget.used);
                    live.add(order.data.capacity());
                    live.add(packets.capacity() * sizeof(OwnerPacket));
                    for (const auto& output : packets)
                        live.add(output.bytes.capacity());
                    live.add(2 * packet_cap); // encoder vector growth, not only logical wire size
                    live.add(cap / 4);        // pending reply allocation
                }
                const auto encoded_batch = encode_adaptive_batch(leaves, families, packet_cap);
                Writer reply;
                if (leased)
                {
                    if (encoded_batch.capacity() > 2 * packet_cap)
                        throw std::invalid_argument("Needed-record encoded capacity exceeds its growth lease.");
                    Budget live{cap / 4};
                    if (packets.capacity() > live.limit / sizeof(OwnerPacket))
                        throw std::invalid_argument("Needed reply metadata exceeds lease.");
                    live.add(packets.capacity() * sizeof(OwnerPacket));
                    for (const auto& packet : packets)
                        live.add(packet.bytes.capacity());
                    live.add(24);
                    live.add(order.data.size());
                    live.add(encoded_batch.size());
                    reply.data.reserve(24 + order.data.size() + encoded_batch.size());
                }
                reply.bytes(order.data);
                if (leased)
                    reply.word(retention);
                reply.bytes(encoded_batch);
                budget.packet(reply.data.size());
                packets.push_back({packet.peer, std::move(reply.data)});
            }
        });
    if (leased)
        incoming = {};
    collective(*state.comm, [&] { check_packets(packets); });
    auto replies = exchange_owner_packets(packets, state.comm, exchange_cap);
    if (leased)
        std::vector<OwnerPacket>().swap(packets);
    AdaptiveBatch result;
    collective(*state.comm,
        [&]
        {
            Budget budget{leased ? cap / 4 : cap};
            if (leased)
            {
                if (keys.size() > budget.limit / (sizeof(AdaptiveLeaf) + sizeof(AdaptiveFamily) + 1))
                    throw std::invalid_argument("Needed result metadata exceeds retention lease.");
                budget.add(keys.size()); // packed seen flags, conservatively one byte per key
            }
            std::vector<bool> seen(keys.size());
            if (family)
            {
                budget.add(keys.size() * sizeof(AdaptiveFamily));
                result.families.resize(keys.size());
            }
            else
            {
                budget.add(keys.size() * sizeof(AdaptiveLeaf));
                result.leaves.resize(keys.size());
            }
            std::size_t remaining_retention = batch_cap;
            for (const auto& packet : replies.packets)
            {
                budget.add(packet.bytes.size());
                Reader r{packet.bytes};
                auto order = r.bytes();
                const auto retention = leased ? r.word() : batch_cap;
                if (leased && retention > remaining_retention)
                    throw std::invalid_argument("Cumulative needed-record retention exceeds lease.");
                // One half of this reservation is the bounded codec arena;
                // the other half is decoded record retention. Thus previous
                // replies and this decode share one cumulative pool.
                auto batch = decode_adaptive_batch(r.bytes(), leased ? std::min(batch_cap, retention / 2) : batch_cap);
                if (leased)
                {
                    if (fetch_retention(batch.leaves, batch.families, retention / 2, false) > retention / 2)
                        throw std::invalid_argument("Needed-record decoded capacities exceed admission.");
                    remaining_retention -= retention;
                }
                r.done();
                Reader indices{order};
                if ((family && !batch.leaves.empty()) || (!family && !batch.families.empty()))
                    throw std::invalid_argument("Needed-record owner returned wrong payload type.");
                const auto count = family ? batch.families.size() : batch.leaves.size();
                if (count != order.size() / 8 || order.size() % 8)
                    throw std::invalid_argument("Needed-record owner count differs from ordered request.");
                for (std::size_t i = 0; i < count; ++i)
                {
                    auto index = indices.word();
                    if (index >= keys.size() || seen[index] || packet.peer != owners[index].owner)
                        throw std::invalid_argument("Foreign, repeated or misordered needed-record reply.");
                    seen[index] = true;
                    if (family)
                    {
                        auto& value = batch.families[i];
                        if (value.key != keys[index] || value.revision != owners[index].revision)
                            throw std::invalid_argument("Needed family changed exact identity or revision.");
                        validate_family(*state.kernel, value);
                        result.families[index] = std::move(value);
                    }
                    else
                    {
                        auto& value = batch.leaves[i];
                        if (encode_adaptive_key(value.key) != keys[index] || value.revision != owners[index].revision)
                            throw std::invalid_argument("Needed leaf changed exact identity or revision.");
                        state.kernel->validate(value);
                        result.leaves[index] = std::move(value);
                    }
                }
            }
            for (bool present : seen)
                if (!present)
                    throw std::invalid_argument("Missing needed-record owner reply.");
        });
    return result;
}
} // namespace
std::vector<AdaptiveOwner> DistributedAdaptiveHierarchy::lookup_cells(
    std::span<const AdaptiveCellKey> keys, bool require_all, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    std::vector<AdaptiveBytes> encoded;
    collective(*data_->comm,
        [&]
        {
            Budget budget{transient_cap ? cap / 8 : cap};
            if (transient_cap)
            {
                if (keys.size() > (cap / 8) / sizeof(AdaptiveBytes))
                    throw std::invalid_argument("Needed key metadata exceeds transient lease.");
                budget.add(keys.size() * sizeof(AdaptiveBytes));
                encoded.reserve(keys.size());
            }
            for (const auto& key : keys)
            {
                if (transient_cap)
                {
                    const auto dynamic = fetch_key_bytes(key);
                    if (budget.limit < 256 || dynamic > budget.limit / 2 - 128)
                        throw std::invalid_argument("Needed key encoding exceeds transient lease.");
                    budget.add(2 * (128 + dynamic)); // framing plus vector growth, BEFORE encoding
                }
                auto bytes = encode_adaptive_key(key);
                if (!transient_cap)
                    budget.add(bytes.size());
                encoded.push_back(std::move(bytes));
            }
        });
    return data_->cells->lookup(
        encoded, require_all, transient_cap ? std::optional<std::size_t>(cap - cap / 8) : std::nullopt);
}
std::vector<AdaptiveOwner> DistributedAdaptiveHierarchy::lookup_families(
    std::span<const AdaptiveBytes> keys, bool require_all, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    return data_->family_directory->lookup(keys, require_all, transient_cap);
}
AdaptiveBytes DistributedAdaptiveHierarchy::configuration_descriptor() const
{
    return data_->kernel->descriptor();
}
std::vector<AdaptiveLeaf> DistributedAdaptiveHierarchy::fetch_leaves(
    std::span<const AdaptiveCellKey> keys, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    std::vector<AdaptiveBytes> encoded;
    collective(*data_->comm,
        [&]
        {
            Budget budget{transient_cap ? cap / 8 : cap};
            if (transient_cap)
            {
                if (keys.size() > (cap / 8) / sizeof(AdaptiveBytes))
                    throw std::invalid_argument("Needed key metadata exceeds transient lease.");
                budget.add(keys.size() * sizeof(AdaptiveBytes));
                encoded.reserve(keys.size());
            }
            for (const auto& key : keys)
            {
                if (transient_cap)
                {
                    const auto dynamic = fetch_key_bytes(key);
                    if (budget.limit < 256 || dynamic > budget.limit / 2 - 128)
                        throw std::invalid_argument("Needed key encoding exceeds transient lease.");
                    budget.add(2 * (128 + dynamic)); // framing plus vector growth, BEFORE encoding
                }
                auto bytes = encode_adaptive_key(key);
                if (!transient_cap)
                    budget.add(bytes.size());
                encoded.push_back(std::move(bytes));
            }
        });
    return fetch_records(*data_, encoded, false, transient_cap ? cap - cap / 8 : cap, transient_cap.has_value()).leaves;
}
std::vector<AdaptiveFamily> DistributedAdaptiveHierarchy::fetch_families(
    std::span<const AdaptiveBytes> keys, std::optional<std::size_t> transient_cap) const
{
    const auto cap = transient_cap.value_or(data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (!cap || cap > data_->options.byte_budget)
                throw std::invalid_argument("Invalid hierarchy transient lease.");
        });

    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    return fetch_records(*data_, keys, true, cap, transient_cap.has_value()).families;
}
DistributedAdaptiveHierarchy::Storage DistributedAdaptiveHierarchy::storage() const
{
    Storage result;
    result.owned_leaves = data_->owned.size();
    result.family_capsules = data_->families.size();
    result.directory_entries = data_->cells->local_entries() + data_->family_directory->local_entries();
    result.directory_bytes = data_->cells->storage_bytes() + data_->family_directory->storage_bytes();
    result.owned_bytes = sizeof(Data);
    std::set<const void*> parts, objects, geometry;
    auto key_dynamic = [](const AdaptiveCellKey& key)
    {
        return std::visit(
            [](const auto& p) -> std::size_t
            {
                using T = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                    return p.path.capacity() * sizeof(std::uint64_t);
                else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                    return (p.base.path.capacity() + p.axial.path.capacity()) * sizeof(std::uint64_t);
                else
                    return 0;
            },
            key.ancestry);
    };
    auto part = [&](const std::shared_ptr<const AdaptiveBytes>& value)
    {
        if (value && parts.insert(value.get()).second)
            result.procedural_bytes += sizeof(AdaptiveBytes) + value->capacity();
    };
    auto leaf = [&](const AdaptiveLeaf& value, std::size_t& bytes)
    {
        bytes += key_dynamic(value.key) + value.domain.capacity() + value.parent_family.capacity() +
                 value.surfaces.capacity() * sizeof(AdaptiveSurface);
        if (value.native_recipe && objects.insert(value.native_recipe.get()).second)
        {
            result.procedural_bytes +=
                sizeof(NativeFragmentDescriptor) + value.native_recipe->view.capacity() +
                value.native_recipe->parts.capacity() * sizeof(std::shared_ptr<const AdaptiveBytes>);
            for (const auto& p : value.native_recipe->parts)
                part(p);
        }
        part(value.join_recipe);
        if (value.factor_recipe && objects.insert(value.factor_recipe.get()).second)
        {
            result.procedural_bytes += sizeof(AdaptiveFactorRecipe);
            part(value.factor_recipe->root);
            part(value.factor_recipe->base);
            part(value.factor_recipe->axis);
        }
        if (value.geometry && geometry.insert(value.geometry.get()).second)
            bytes += value.geometry->storage_bytes();
        for (const auto& face : value.surfaces)
        {
            bytes += face.key.capacity() + face.boundary.capacity() + face.authority_image.capacity();
            if (face.geometry && geometry.insert(face.geometry.get()).second)
                bytes += face.geometry->storage_bytes();
        }
    };
    for (const auto& [key, value] : data_->owned)
    {
        result.owned_bytes +=
            sizeof(std::pair<const AdaptiveCellKey, AdaptiveLeaf>) + 4 * sizeof(void*) + key_dynamic(key);
        leaf(value, result.owned_bytes);
    }
    for (const auto& [key, value] : data_->families)
    {
        result.ancestor_bytes += sizeof(std::pair<const AdaptiveBytes, AdaptiveFamily>) + 4 * sizeof(void*) +
                                 key.capacity() + value.key.capacity() + value.anchor.capacity() +
                                 value.children.capacity() * sizeof(AdaptiveCellKey);
        for (const auto& child : value.children)
            result.ancestor_bytes += key_dynamic(child);
        leaf(value.parent, result.ancestor_bytes);
    }
    if (data_->native)
    {
        result.native_bytes = data_->native->geometry()->storage_report().measured_bytes();
        const auto& plan = data_->native->plan();
        result.native_map_entries = plan.cell_ids().size() + plan.face_ids().size() + plan.node_ids().size();
        result.native_bytes +=
            plan.cell_ids().capacity() * sizeof(std::uint64_t) + plan.cell_owner_ranks().capacity() * sizeof(int) +
            plan.face_ids().capacity() * sizeof(std::uint64_t) + plan.node_ids().capacity() * sizeof(std::uint64_t);
    }
    return result;
}
DistributedAdaptiveHierarchy::Plan DistributedAdaptiveHierarchy::plan_adaptation(
    std::span<const AdaptiveMark> refine, std::span<const AdaptiveBytes> coarsen) const
{
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    std::map<AdaptiveCellKey, AdaptiveSplit> marks;
    std::vector<AdaptiveBytes> family_keys;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            if (data_->topology == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Distributed topology generation exhausted.");
            for (const auto& mark : refine)
            {
                budget.add(sizeof(mark) + encode_adaptive_key(mark.key).size());
                if (!data_->owned.contains(mark.key) || !marks.emplace(mark.key, mark.split).second)
                    throw std::invalid_argument("Refinement mark is repeated or not locally owned.");
            }
            std::set<AdaptiveBytes> unique;
            for (const auto& key : coarsen)
            {
                budget.add(key.size());
                if (unique.insert(key).second)
                    family_keys.push_back(key);
            }
        });
    std::vector<AdaptiveBytes> group_keys;
    std::vector<AdaptiveSplit> group_splits;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& [key, split] : marks)
                for (const auto& member : data_->kernel->refinement_group(key, split))
                {
                    auto bytes = encode_adaptive_key(member);
                    budget.add(bytes.size() + sizeof(AdaptiveSplit));
                    group_keys.push_back(std::move(bytes));
                    group_splits.push_back(split);
                }
        });
    auto group_owners = data_->cells->lookup(group_keys);
    std::vector<OwnerPacket> group_requests;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (std::size_t i = 0; i < group_owners.size(); ++i)
            {
                Writer w;
                w.bytes(group_owners[i].key);
                w.word(std::uint64_t(group_splits[i]));
                w.word(group_owners[i].revision);
                budget.packet(w.data.size());
                group_requests.push_back({group_owners[i].owner, std::move(w.data)});
            }
        });
    auto group_incoming = exchange_owner_packets(group_requests, data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            marks.clear();
            for (const auto& packet : group_incoming.packets)
            {
                Reader r{packet.bytes};
                auto key = decode_adaptive_key(r.bytes());
                auto split = r.word(), revision = r.word();
                r.done();
                auto leaf = data_->owned.find(key);
                if (split > 2 || leaf == data_->owned.end() || leaf->second.revision != revision)
                    throw std::invalid_argument("Invalid distributed native refinement group.");
                auto [it, inserted] = marks.emplace(key, AdaptiveSplit(split));
                if (!inserted && it->second != AdaptiveSplit(split))
                    throw std::invalid_argument("Conflicting distributed factor marks.");
            }
        });
    const auto plan_id = transaction(*data_->comm);
    std::uint64_t local_changes = refine.size() + coarsen.size(), global_changes = 0;
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &local_changes, &global_changes);
    if (!global_changes)
    {
        Plan plan;
        plan.source_ = data_->token;
        plan.observation_ = observation();
        plan.topology_ = data_->topology;
        plan.partition_ = data_->partition;
        plan.transaction_ = plan_id;
        plan.candidate_ = std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(data_));
        return plan;
    }
    auto owners = data_->family_directory->lookup(family_keys);
    std::vector<OwnerPacket> requests;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& owner : owners)
            {
                Writer w;
                w.bytes(owner.key);
                w.word(owner.revision);
                budget.packet(w.data.size());
                requests.push_back({owner.owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, data_->comm, data_->options.byte_budget);
    std::set<AdaptiveBytes> requested;
    collective(*data_->comm,
        [&]
        {
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto key = r.bytes();
                auto revision = r.word();
                r.done();
                auto it = data_->families.find(key);
                if (it == data_->families.end() || it->second.revision != revision)
                    throw std::invalid_argument("Stale family authority request.");
                requested.insert(std::move(key));
            }
        });
    std::vector<AdaptiveBytes> grouped_families;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            std::set<AdaptiveBytes> unique;
            for (const auto& family : requested)
                for (auto key : data_->kernel->coarsening_group(data_->families.at(family)))
                {
                    budget.add(key.size());
                    if (unique.insert(key).second)
                        grouped_families.push_back(std::move(key));
                }
        });
    auto grouped_owners = data_->family_directory->lookup(grouped_families);
    requests.clear();
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& owner : grouped_owners)
            {
                Writer w;
                w.bytes(owner.key);
                w.word(owner.revision);
                budget.packet(w.data.size());
                requests.push_back({owner.owner, std::move(w.data)});
            }
        });
    incoming = exchange_owner_packets(requests, data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            requested.clear();
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto key = r.bytes();
                auto revision = r.word();
                r.done();
                auto it = data_->families.find(key);
                if (it == data_->families.end() || it->second.revision != revision)
                    throw std::invalid_argument("Invalid native coarsening group authority.");
                requested.insert(std::move(key));
            }
        });
    std::vector<AdaptiveBytes> child_keys;
    std::vector<AdaptiveBytes> child_families;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& key : requested)
                for (const auto& child : data_->families.at(key).children)
                {
                    auto encoded = encode_adaptive_key(child);
                    budget.add(encoded.size() + key.size());
                    child_keys.push_back(std::move(encoded));
                    child_families.push_back(key);
                }
        });
    owners = data_->cells->lookup(child_keys);
    requests.clear();
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (std::size_t i = 0; i < owners.size(); ++i)
            {
                Writer w;
                w.bytes(child_families[i]);
                w.bytes(owners[i].key);
                w.word(owners[i].revision);
                budget.packet(w.data.size());
                requests.push_back({owners[i].owner, std::move(w.data)});
            }
        });
    incoming = exchange_owner_packets(requests, data_->comm, data_->options.byte_budget);
    std::vector<OwnerPacket> replies;
    std::set<AdaptiveCellKey> retiring;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& packet : incoming.packets)
            {
                Reader r{packet.bytes};
                auto family = r.bytes();
                auto key = decode_adaptive_key(r.bytes());
                auto revision = r.word();
                r.done();
                auto it = data_->owned.find(key);
                if (it == data_->owned.end() || it->second.revision != revision || marks.contains(key) ||
                    !retiring.insert(key).second)
                    throw std::invalid_argument("Incomplete, conflicting or stale source sibling family.");
                Writer w;
                w.bytes(family);
                w.bytes(encode_adaptive_leaf(it->second));
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto children = exchange_owner_packets(replies, data_->comm, data_->options.byte_budget);
    std::map<AdaptiveBytes, std::vector<AdaptiveLeaf>> families;
    std::shared_ptr<Data> next;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& packet : children.packets)
            {
                budget.add(packet.bytes.size());
                Reader r{packet.bytes};
                auto family = r.bytes();
                if (!requested.contains(family))
                    throw std::invalid_argument("Foreign family child response.");
                auto leaf = decode_adaptive_leaf(r.bytes());
                r.done();
                families[family].push_back(std::move(leaf));
            }
            next = std::make_shared<Data>(*data_);
            next->token = std::make_shared<const char>(0);
            next->topology_token = next->token;
            next->partition_token = next->token;
            next->native.reset();
            next->owned_map = Teuchos::null;
            next->map_token.reset();
            next->native_epoch = 0;
            ++next->topology;
            if (next->partition == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Distributed partition generation exhausted.");
            ++next->partition;
            for (const auto& key : retiring)
                next->owned.erase(key);
            for (const auto& key : requested)
            {
                auto parent = data_->kernel->coarsen(data_->families.at(key), families[key]);
                for (auto& [ancestor_key, capsule] : next->families)
                    if (std::find(data_->families.at(key).children.begin(), data_->families.at(key).children.end(),
                            decode_adaptive_key(capsule.anchor)) != data_->families.at(key).children.end())
                        capsule.anchor = encode_adaptive_key(parent.key);
                if (!next->owned.emplace(parent.key, std::move(parent)).second)
                    throw std::invalid_argument("Coarsened parent already active.");
                next->families.erase(key);
            }
            for (const auto& [key, kind] : marks)
            {
                auto result = data_->kernel->split(data_->owned.at(key), kind);
                next->owned.erase(key);
                for (auto& [ancestor_key, capsule] : next->families)
                    if (capsule.anchor == encode_adaptive_key(key))
                        capsule.anchor = result.family.anchor;
                for (auto& leaf : result.children)
                {
                    budget.add(sizeof(AdaptiveLeaf) + encode_adaptive_key(leaf.key).size());
                    data_->kernel->validate(leaf);
                    if (!next->owned.emplace(leaf.key, std::move(leaf)).second)
                        throw std::invalid_argument("Refined child already active.");
                }
                if (!next->families.emplace(result.family.key, std::move(result.family)).second)
                    throw std::invalid_argument("Refined family already retained.");
            }
            std::vector<AdaptiveLeaf> records;
            for (const auto& [key, leaf] : next->owned)
                records.push_back(leaf);
            auto packed = data_->kernel->repack(records);
            next->owned.clear();
            for (auto& leaf : packed)
            {
                budget.add(sizeof(AdaptiveLeaf) + encode_adaptive_key(leaf.key).size());
                data_->kernel->validate(leaf);
                if (!next->owned.emplace(leaf.key, std::move(leaf)).second)
                    throw std::invalid_argument("Duplicate repacked adaptive leaf.");
            }
            if (next->owned.size() > data_->options.maximum_owned)
                throw std::invalid_argument("Distributed local leaf budget exhausted.");
        });
    register_state(*next);
    close_candidate(*next, *data_);
    Plan plan;
    plan.source_ = data_->token;
    plan.observation_ = observation();
    plan.topology_ = data_->topology;
    plan.partition_ = data_->partition;
    plan.transaction_ = plan_id;
    plan.candidate_ =
        std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(std::move(next)));
    return plan;
}
DistributedAdaptiveHierarchy::Plan DistributedAdaptiveHierarchy::plan_migration(std::span<const int> destinations) const
{
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    NativeMigrationDiagnostics diagnostics;
    const auto id = transaction(*data_->comm);
    std::vector<OwnerPacket> packets;
    std::uint64_t local_changes = 0, changes = 0;
    collective(*data_->comm,
        [&]
        {
            if (destinations.size() != data_->owned.size() ||
                data_->partition == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Invalid ownership migration layout or exhausted generation.");
            Budget budget{data_->options.byte_budget};
            std::map<AdaptiveBytes, int> routes;
            struct Group
            {
                std::vector<AdaptiveLeaf> leaves;
                std::vector<AdaptiveFamily> families;
            };
            std::map<int, Group> groups;
            std::size_t i = 0;
            for (const auto& [key, leaf] : data_->owned)
            {
                const auto rank = destinations[i++];
                if (rank < 0 || rank >= data_->comm->getSize())
                    throw std::invalid_argument("Invalid migration destination.");
                local_changes += rank != data_->comm->getRank();
                budget.add(sizeof(AdaptiveLeaf));
                routes.emplace(encode_adaptive_key(key), rank);
                groups[rank].leaves.push_back(leaf);
            }
            for (const auto& [key, family] : data_->families)
            {
                auto route = routes.find(family.anchor);
                if (route == routes.end())
                    throw std::invalid_argument("Detached family authority anchor.");
                budget.add(sizeof(AdaptiveFamily));
                groups[route->second].families.push_back(family);
            }
            for (const auto& [rank, group] : groups)
            {
                auto bytes = encode_adaptive_batch(group.leaves, group.families, data_->options.byte_budget);
                budget.packet(bytes.size());
                packets.push_back({rank, std::move(bytes)});
            }
        });
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &local_changes, &changes);
    if (!changes)
    {
        Plan plan;
        plan.source_ = data_->token;
        plan.observation_ = observation();
        plan.topology_ = data_->topology;
        plan.partition_ = data_->partition;
        plan.transaction_ = id;
        plan.candidate_ = std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(data_));
        return plan;
    }
    collective(*data_->comm,
        [&]
        {
            std::set<int> peers;
            for (const auto& packet : packets)
            {
                diagnostics.peak_native_packet_buffers += packet.bytes.capacity();
                if (packet.peer == data_->comm->getRank())
                    diagnostics.self_payload_bytes += packet.bytes.size();
                else
                {
                    diagnostics.remote_sent_payload_bytes += packet.bytes.size();
                    ++diagnostics.remote_sent_packets;
                    peers.insert(packet.peer);
                }
            }
            diagnostics.peak_native_packet_buffers += packets.capacity() * sizeof(OwnerPacket);
            diagnostics.remote_sent_peers = peers.size();
        });
    auto incoming = exchange_owner_packets(packets, data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            std::set<int> peers;
            for (const auto& packet : incoming.packets)
                if (packet.peer != data_->comm->getRank())
                {
                    diagnostics.remote_received_payload_bytes += packet.bytes.size();
                    ++diagnostics.remote_received_packets;
                    peers.insert(packet.peer);
                }
            diagnostics.remote_received_peers = peers.size();
            if (incoming.stats.peak_scratch_bytes >
                std::numeric_limits<std::size_t>::max() - diagnostics.peak_native_packet_buffers)
                throw std::overflow_error("Native packet peak counter overflow.");
            diagnostics.peak_native_packet_buffers += incoming.stats.peak_scratch_bytes;
        });
    std::shared_ptr<Data> next;
    collective(*data_->comm,
        [&]
        {
            next = std::make_shared<Data>(*data_);
            next->owned.clear();
            next->families.clear();
            next->token = std::make_shared<const char>(0);
            next->partition_token = next->token;
            next->native.reset();
            next->owned_map = Teuchos::null;
            next->map_token.reset();
            next->native_epoch = 0;
            ++next->partition;
            Budget budget{data_->options.byte_budget};
            for (const auto& packet : incoming.packets)
            {
                budget.add(packet.bytes.size());
                auto batch = decode_adaptive_batch(packet.bytes, data_->options.byte_budget);
                for (auto& leaf : batch.leaves)
                {
                    data_->kernel->validate(leaf);
                    if (!next->owned.emplace(leaf.key, std::move(leaf)).second)
                        throw std::invalid_argument("Duplicate migrated leaf owner.");
                }
                for (auto& family : batch.families)
                {
                    validate_family(*data_->kernel, family);
                    if (!next->families.emplace(family.key, std::move(family)).second)
                        throw std::invalid_argument("Duplicate migrated family authority.");
                }
            }
            if (next->owned.size() > data_->options.maximum_owned)
                throw std::invalid_argument("Migrated local leaf budget exhausted.");
            std::vector<AdaptiveLeaf> records;
            for (const auto& [key, leaf] : next->owned)
                records.push_back(leaf);
            auto packed = data_->kernel->repack(records);
            next->owned.clear();
            for (auto& leaf : packed)
                if (!next->owned.emplace(leaf.key, std::move(leaf)).second)
                    throw std::invalid_argument("Duplicate repacked migration leaf.");
            for (const auto& [key, family] : next->families)
                if (!next->owned.contains(decode_adaptive_key(family.anchor)))
                    throw std::invalid_argument("Migrated capsule lost actual anchor.");
        });
    register_state(*next);
    Plan plan;
    plan.native_migration_ = diagnostics;
    plan.source_ = data_->token;
    plan.observation_ = observation();
    plan.topology_ = data_->topology;
    plan.partition_ = data_->partition;
    plan.transaction_ = id;
    plan.candidate_ =
        std::shared_ptr<const DistributedAdaptiveHierarchy>(new DistributedAdaptiveHierarchy(std::move(next)));
    return plan;
}
DistributedAdaptiveHierarchy::Plan DistributedAdaptiveHierarchy::plan_migration(
    const AdaptiveBalancePlan& proposal) const
{
    collective(*data_->comm,
        [&]
        {
            if (proposal.source.identity != data_->token || proposal.source.topology != data_->topology ||
                proposal.source.partition != data_->partition || proposal.keys.size() != data_->owned.size() ||
                proposal.revisions.size() != data_->owned.size() ||
                proposal.localdestinations.size() != data_->owned.size())
                throw std::invalid_argument("Foreign, stale or incomplete measured ownership proposal.");
            if (proposal.accepted && !proposal.proposed.wire_bytes_exact)
                throw std::invalid_argument("Measured ownership proposal lacks actual packed stream qualification.");
            std::size_t i = 0;
            for (const auto& [key, leaf] : data_->owned)
            {
                if (proposal.keys[i] != key || proposal.revisions[i] != leaf.revision)
                    throw std::invalid_argument("Measured ownership proposal changed local key/revision order.");
                ++i;
            }
        });
    return plan_migration(proposal.localdestinations);
}
std::shared_ptr<const DistributedAdaptiveHierarchy> DistributedAdaptiveHierarchy::apply(const Plan& plan) const
{
    agree(*data_->comm, std::array<std::uint64_t, 1>{plan.transaction_});
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged() || plan.observation_ != observation() || plan.source_ != data_->token ||
                plan.topology_ != data_->topology || plan.partition_ != data_->partition || !plan.candidate_)
                throw std::invalid_argument("Foreign or stale distributed adaptation plan.");
        });
    return plan.candidate_;
}
std::shared_ptr<const DistributedAdaptiveHierarchy> DistributedAdaptiveHierarchy::apply(
    const Plan& plan, std::shared_ptr<const DistributedAdaptiveHierarchy> candidate) const
{
    (void) apply(plan);
    collective(*data_->comm,
        [&]
        {
            if (!candidate || candidate->identity() != plan.candidate_->identity() ||
                candidate->topology_generation() != plan.candidate_->topology_generation() ||
                candidate->partition_generation() != plan.candidate_->partition_generation() ||
                !candidate->native_partition() || !candidate->geometry_unchanged())
                throw std::invalid_argument("Native-bound candidate does not belong to this source-pinned plan.");
        });
    return candidate;
}
AdaptiveNativeDraft DistributedAdaptiveHierarchy::native_draft() const
{
    collective(*data_->comm,
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Distributed source resident geometry moved.");
        });
    AdaptiveNativeDraft result;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& [key, leaf] : data_->owned)
            {
                auto transient = leaf;
                transient.geometry = data_->kernel->geometry(leaf);
                transient.surfaces = data_->kernel->surfaces(leaf);
                budget.add(encode_adaptive_leaf(transient).size());
                result.owned.push_back(std::move(transient));
            }
        });
    result.global_cells = data_->global_cells;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            std::uint64_t gid = data_->cell_begin;
            for (const auto& [key, leaf] : data_->owned)
            {
                budget.add(sizeof(AdaptiveNativeCellBinding) + encode_adaptive_key(key).size());
                const auto values = data_->kernel->metrics(leaf);
                AdaptiveNativeCellBinding binding;
                binding.key = key;
                binding.gid = gid++;
                binding.material = leaf.material;
                binding.native_cell = leaf.native_cell;
                binding.owner = data_->comm->getRank();
                binding.provider = leaf.provider;
                binding.full = true;
                binding.volume = values.volume;
                binding.centroid = values.centroid;
                binding.native_recipe = leaf.native_recipe;
                result.cell_bindings.push_back(std::move(binding));
            }
        });
    AdaptiveAuthorityStats authority_stats;
    result.tiles = decide_adaptive_tiles(result.owned, data_->comm, data_->options.byte_budget,
        data_->options.maximum_tiles, &authority_stats, &result.nodes);
    result.global_faces = authority_stats.global_faces;
    result.global_nodes = authority_stats.global_nodes;
    collective(*data_->comm,
        [&]
        {
            std::map<AdaptiveCellKey, std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>> incidence;
            for (const auto& tile : result.tiles)
            {
                incidence[tile.first].push_back(tile.geometry);
                if (tile.second)
                    incidence[*tile.second].push_back(tile.second_image_identity.empty()
                                                          ? tile.geometry->reversed()
                                                          : tile.geometry->reversed()->translated_packet(
                                                                PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                                                    {}, tile.second_image_identity)));
            }
            for (const auto& leaf : result.owned)
            {
                if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(incidence.at(leaf.key))
                        ->equivalent(*leaf.geometry))
                    throw std::invalid_argument("Authority tiles do not form complete exact local shell.");
            }
        });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
