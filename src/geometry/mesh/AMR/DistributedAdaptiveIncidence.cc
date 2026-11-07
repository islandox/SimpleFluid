/** @file DistributedAdaptiveIncidence.cc
 * @brief Incremental complete-support and layered exact-line incidence. */
#include "DistributedAdaptiveIncidence.hh"
#include "AdaptiveSubfaceSnapshot.hh"
#include "CertifiedIncidenceViews.hh"
#include "PersistentTileIndex.hh"
#include "PersistentSlotValues.hh"
#include "DeferredOwnerCharges.hh"
#include "IncidenceOwnerRows.hh"
#include "IncidencePools.hh"
#include "IncidencePhysicalOwners.hh"
#include "IncidenceTileDelivery.hh"
#include "OwnedSupportMembership.hh"
#include "PooledLineExchange.hh"
#include "PooledSupportRegistry.hh"
#include "PooledTileCodec.hh"
#include "PrivateTileRecipe.hh"
#include "SupportGroupFetch.hh"
#include "SupportRegistryExchange.hh"
#include "SupportSurfaceCodec.hh"
#include "TileBodyPages.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t owner_bytes = 128;
using Slot = std::size_t;
using TileId = std::pair<Slot, std::uint64_t>;
bool same_saved_point(const IncidencePools::Point& first, const IncidencePools::Point& last)
{
    using Access = PLIC::details::GeometryIdentityAccess;
    const auto bytes = PLIC::canonical_key_size(first);
    if (bytes != PLIC::canonical_key_size(last))
        return false;
    for (std::size_t offset = 0; offset < bytes; offset += 8)
        if (Access::canonical_word_at(first, offset) != Access::canonical_word_at(last, offset))
            return false;
    return true;
}

struct Lease
{
    Budget* budget;
    std::size_t bytes = 0;
    explicit Lease(Budget& b) : budget(&b) {}
    ~Lease() { budget->used -= bytes; }
    void add(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::invalid_argument("Incidence allocation size overflow.");
        budget->add(n);
        bytes += n;
    }
    template<class T> void append(std::vector<T>& values)
    {
        const auto before = budget->used;
        budget->append_capacity(values);
        bytes += budget->used - before;
    }
};
/** finish_rows transfers this buffer's admission to its caller. Until public
 * return, destroy its handles/buffer before returning that live credit. Point
 * compaction may replace the vector, so observe its current capacity on unwind. */
struct ReturnedRowsCredit
{
    Budget& budget;
    std::vector<IncidenceRowDelivery>* rows;
    ~ReturnedRowsCredit()
    {
        if (rows)
        {
            const auto bytes = rows->capacity() * sizeof(IncidenceRowDelivery);
            std::vector<IncidenceRowDelivery>().swap(*rows);
            budget.used -= bytes;
        }
    }
    void transfer() noexcept { rows = nullptr; }
};
static_assert(std::is_nothrow_move_constructible_v<std::vector<IncidenceRowDelivery>>);
static_assert(std::is_nothrow_move_constructible_v<IncidenceDelta>);
struct EdgeState
{
    Slot line = 0;
    std::vector<Slot> result;
};
struct Layer
{
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0;
    std::vector<Slot> input, output;
    std::shared_ptr<const SharedSubfaceRing> final_output;
    std::vector<EdgeState> edges;
    ~Layer()
    {
        std::vector<Slot>().swap(input);
        std::vector<Slot>().swap(output);
        std::vector<EdgeState>().swap(edges);
        if (budget)
            budget->used -= charge;
    }
};
/** Only source-protocol factories create this subtype. The session protocol is
 * sticky across forks; ordinary Layers retain their exact original layout. */
struct SavedLayer : Layer
{
    std::optional<CertifiedIncidenceLoopView> saved_input, saved_output;
    std::optional<CertifiedIncidenceResultParts> saved_edges;
    std::vector<Slot> endpoint_facts;
    std::vector<PooledLineCompletionView> completions;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
    std::uint64_t source_generation = 0;
    unsigned kernel_layer = 0;
    bool native_member = false;
    ~SavedLayer()
    {
        decltype(endpoint_facts){}.swap(endpoint_facts);
        decltype(completions){}.swap(completions);
        decltype(leases){}.swap(leases);
    }
    void retain_dependencies(cartesian_detail::CreditConstruction& credits,
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& out,
        cartesian_detail::Allocation& memory) const
    {
        if (saved_input)
            saved_input->retain_dependencies(credits, out, memory);
        if (saved_output)
            saved_output->retain_dependencies(credits, out, memory);
        if (saved_edges)
            saved_edges->retain_dependencies(credits, out, memory);
        for (const auto& completion : completions)
            if (completion)
            {
                if (credits.descend(completion, completion->allocation_budget(), completion->metadata_bytes(),
                        completion->has_arena_keeper(), out, memory))
                    completion->retain_dependencies(credits, out, memory);
            }
        for (const auto& lease : leases)
            credits.retain(std::shared_ptr<const cartesian_detail::AuthorityOwnerCredit>(lease), lease->budget,
                lease->metadata, bool(lease->keeper), out, memory);
    }
};
TileBodyLoopView saved_loop(const CertifiedIncidenceLoopView& view)
{
    return TileBodyLoopView(&view, view.size(),
        [](const void* p, std::size_t i) { return (*static_cast<const CertifiedIncidenceLoopView*>(p))[i]; });
}
TileBodyLoopView input_loop(const Layer& layer, bool source)
{
    if (source && static_cast<const SavedLayer&>(layer).saved_input)
        return saved_loop(*static_cast<const SavedLayer&>(layer).saved_input);
    return layer.input;
}
TileBodyLoopView output_loop(const Layer& layer, bool source)
{
    if (source && static_cast<const SavedLayer&>(layer).saved_output)
        return saved_loop(*static_cast<const SavedLayer&>(layer).saved_output);
    return layer.final_output ? TileBodyLoopView(layer.final_output->slots()) : TileBodyLoopView(layer.output);
}
TileBodyLoopView edge_result(const Layer& layer, std::size_t edge, bool source)
{
    const auto& explicit_result = layer.edges.at(edge).result;
    if (source && explicit_result.empty())
    {
        const auto& saved = static_cast<const SavedLayer&>(layer);
        if (edge < saved.completions.size() && saved.completions[edge])
            return saved.completions[edge]->result();
        if (saved.saved_edges)
        {
            const auto& part = saved.saved_edges->parts()[edge];
            return TileBodyLoopView(&part, part.size(),
                [](const void* p, std::size_t i) { return (*static_cast<const CertifiedIncidenceResultView*>(p))[i]; });
        }
    }
    return explicit_result;
}
struct LayerHistory
{
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0;
    std::vector<std::pair<unsigned, std::shared_ptr<Layer>>> layers;
    ~LayerHistory()
    {
        decltype(layers)().swap(layers);
        if (budget)
            budget->used -= charge;
    }
};
struct TileEntry
{
    TileBodyHandle body;
    std::shared_ptr<LayerHistory> history;
    bool initialized = false;
    std::uint64_t support_generation = 0;
    bool native_source = false;
    SubfaceProvenanceRef provenance;
    CanonicalSubfaceView final_piece;
    [[nodiscard]] TileBodyScalars tile() const { return body.scalars(); }
    [[nodiscard]] const auto& layers() const
    {
        static const decltype(LayerHistory::layers) empty;
        return history ? history->layers : empty;
    }
};
struct LayerWork
{
    TileBodyLoopView input;
    // A span-backed input may outlive replacement/pruning of its history.
    // Keep its actual Layer owner through this reconstruction iteration.
    std::shared_ptr<const Layer> input_owner;
    // Line slots, supplied first endpoints, source-only supplied last endpoints,
    // then receipt bit words. Identity regions belong to distinct typed pools; the tail
    // is bookkeeping and never establishes an identity.
    std::vector<Slot> lines_and_receipts;
    std::vector<PooledLineCompletionView> completed_edges;
    std::shared_ptr<Layer> body;
    bool require_all = false;
};
using TileMap = PersistentTileIndex<TileEntry>;
using ChangedTileMap = std::map<TileId, std::optional<TileEntry>>;
using LayerMap = std::map<TileId, LayerWork>;
using ExpectedSupportMap = std::map<Slot, std::vector<RegistryMember>>;
constexpr std::size_t changed_map_node = sizeof(ChangedTileMap::value_type) + 4 * sizeof(void*);
constexpr std::size_t layer_map_node = sizeof(LayerMap::value_type) + 4 * sizeof(void*);
constexpr std::size_t expected_support_node = sizeof(ExpectedSupportMap::value_type) + 4 * sizeof(void*);
constexpr std::size_t tile_set_node = sizeof(TileId) + 4 * sizeof(void*);
constexpr std::size_t slot_set_node = sizeof(Slot) + 4 * sizeof(void*);
constexpr std::size_t pointer_set_node = sizeof(const void*) + 4 * sizeof(void*);
std::shared_ptr<Layer> at_layer_owner(const TileEntry& state, unsigned layer)
{
    const auto& layers = state.layers();
    const auto at = std::upper_bound(
        layers.begin(), layers.end(), layer, [](unsigned value, const auto& entry) { return value < entry.first; });
    return at == layers.begin() ? std::shared_ptr<Layer>{} : std::prev(at)->second;
}
const Layer* at_layer(const TileEntry& state, unsigned layer)
{
    return at_layer_owner(state, layer).get();
}
TileBodyLoopView final_loop(const TileEntry& state, unsigned layer, bool source)
{
    const auto* value = at_layer(state, layer);
    return value ? output_loop(*value, source) : state.body.intrinsic_loop();
}
std::size_t layer_bytes(const Layer& layer, bool source)
{
    std::size_t result = sizeof(Layer) + owner_bytes +
                         (layer.input.capacity() + layer.output.capacity()) * sizeof(Slot) +
                         layer.edges.capacity() * sizeof(EdgeState);
    for (const auto& edge : layer.edges)
        result += edge.result.capacity() * sizeof(Slot);
    if (source)
    {
        const auto& saved = static_cast<const SavedLayer&>(layer);
        result += sizeof(SavedLayer) - sizeof(Layer) + saved.endpoint_facts.capacity() * sizeof(Slot) +
                  saved.completions.capacity() * sizeof(PooledLineCompletionView) +
                  saved.leases.capacity() * sizeof(decltype(saved.leases)::value_type);
    }
    return result;
}
std::size_t history_bytes(const LayerHistory& history)
{
    return sizeof(LayerHistory) + owner_bytes +
           history.layers.capacity() * sizeof(decltype(history.layers)::value_type);
}
/** One provider recipe at a time. The provider's existing internal geometry
 * construction cap remains its boundary; all caller-owned output is admitted. */
class MaterializedLookup
{
    const AdaptiveProviderKernel& kernel_;
    const OwnedLeafLookup& lookup_;
    Budget& budget_;
    std::optional<AdaptiveLeaf> value_;
    std::unique_ptr<IdentityBudget> identities_;
    std::shared_ptr<const CartesianIncidenceSource> source_;
    std::unique_ptr<const CartesianNativeIncidenceCell> native_;
    std::size_t charge_ = 0;

public:
    MaterializedLookup(const AdaptiveProviderKernel& k, const OwnedLeafLookup& l, Budget& b)
        : kernel_(k), lookup_(l), budget_(b)
    {
    }
    ~MaterializedLookup() { clear(); }
    void bind_source(std::shared_ptr<const CartesianIncidenceSource> source) { source_ = std::move(source); }
    const CartesianNativeIncidenceCell* native_proof() const noexcept { return native_.get(); }
    void clear()
    {
        native_.reset();
        value_.reset();
        identities_.reset();
        budget_.used -= charge_;
        charge_ = 0;
    }
    const AdaptiveLeaf& get(const AdaptiveCellKey& key)
    {
        const auto& leaf = lookup_(key);
        if (leaf.key != key)
            throw std::invalid_argument("Foreign incidence source lookup.");
        if (native_ && native_->leaf().key == key && native_->leaf().revision == leaf.revision)
            return native_->leaf();
        if (value_ && value_->key == key && value_->revision == leaf.revision)
            return *value_;
        clear();
        if (source_ && CartesianIncidenceSource::eligible(leaf) && leaf.domain == source_->domain() &&
            key.region == source_->region())
            if (const auto* cell = std::get_if<CartesianCellAncestry>(&key.ancestry);
                cell && source_->contains(*cell) && source_->revision(*cell) == leaf.revision)
            {
                native_ = CartesianIncidenceSource::evaluate_cell(source_, *cell, budget_);
                return native_->leaf();
            }
        const auto available = budget_.limit - budget_.used;
        budget_.add(available);
        try
        {
            value_ = leaf;
            // Retained leaf surfaces are provider facts, not necessarily the
            // current authority supports. Mixed routes rewrite their keys and
            // destinations only in this transient kernel-generated view.
            value_->geometry.reset();
            value_->surfaces = kernel_.surfaces(leaf);
            Budget actual{available};
            actual.add(sizeof(AdaptiveLeaf) + key_capacity(value_->key) + value_->domain.capacity() +
                       value_->parent_family.capacity() + value_->surfaces.capacity() * sizeof(AdaptiveSurface));
            identities_ = std::make_unique<IdentityBudget>();
            if (value_->geometry)
                identities_->retain(actual, value_->geometry);
            for (const auto& surface : value_->surfaces)
            {
                actual.add(surface.key.capacity() + surface.boundary.capacity());
                if (surface.geometry)
                    identities_->face(actual, *surface.geometry);
                if (surface.native_whole_geometry && surface.native_whole_geometry != surface.geometry)
                    identities_->face(actual, *surface.native_whole_geometry);
            }
            charge_ = actual.used;
            budget_.used -= available - charge_;
        }
        catch (...)
        {
            value_.reset();
            identities_.reset();
            budget_.used -= available;
            throw;
        }
        return *value_;
    }
};
} // namespace

struct DistributedAdaptiveIncidenceSession::Impl
{
    Lease lease;
    DeferredOwnerCharges borrowed;
    const AdaptiveProviderKernel* kernel;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    AdaptiveClosureOptions options;
    ContactHaloContext context;
    Budget* budget;
    IdentityBudget* identities;
    std::shared_ptr<Budget> keeper;
    bool external_physical_residency;
    std::shared_ptr<ContactPhysicalResidency> physical_residency;
    std::shared_ptr<CartesianAuthorityArena> point_arena;
    std::shared_ptr<const CartesianSourceManifestSet> source_manifests;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> layer_source_leases;
    IncidencePools pools;
    TileBodyPages tile_bodies;
    OwnedSupportMembership owned;
    PooledSupportRegistry supports;
    PooledLineRegistry lines;
    IncidenceOwnerRows owner_rows;
    TileMap tiles;
    std::shared_ptr<const AdaptiveSubfaceSnapshot> subfaces;
    std::size_t subface_foreign_bytes = 0, tile_index_foreign_bytes = 0;
    std::size_t retired_inherited_metadata_bytes = 0;
    PersistentSlotValues<std::size_t> authority_tile_counts;
    PersistentSlotValues<std::uint64_t> resident_generations;
    ContactHaloStatistics statistics;
    unsigned last_layer = 0;
    std::uint64_t point_generation = 0;
    std::size_t periodic_tiles = 0, active_tile_histories = 0;
    std::size_t authority_tiles = 0;
    std::size_t observed_work_capacity_peak = 0, observed_layer_capacity_peak = 0;
    bool ready = false;
    bool native_source_protocol = false;
    bool native_inventory_complete = false;
    bool inherited_overlap_active = false;
    const char* diagnostic_phase = "construction";

    static std::shared_ptr<ContactPhysicalResidency> make_residency(Lease& lease,
        const std::shared_ptr<Budget>& arena, std::shared_ptr<ContactPhysicalResidency> supplied,
        const ContactPhysicalResidency* previous = nullptr)
    {
        if (supplied || !arena)
            return supplied;
        lease.add(sizeof(ContactPhysicalResidency) + owner_bytes);
        return previous ? std::make_shared<ContactPhysicalResidency>(*previous, arena)
                        : std::make_shared<ContactPhysicalResidency>(arena);
    }
    std::shared_ptr<CartesianAuthorityArena> make_point_arena()
    {
        auto arena = CartesianGroupAuthority::make_arena(*budget, keeper);
        IncidencePhysicalOwners::bind_policy(arena, physical_residency);
        return arena;
    }
    void bind_physical_pool_scopes()
    {
        if (!physical_residency)
            return;
        const std::array fixed{pools.cells.scope_id(), pools.domains.scope_id(), pools.supports.scope_id(),
            pools.planes.scope_id(), pools.normals.scope_id(), pools.images.scope_id(),
            IncidencePhysicalOwners::tile_credit_scope};
        physical_residency->set_fixed_pool_scopes(fixed);
        const auto& point_pool = pools.points.physical_pool();
        physical_residency->begin_point_scope(point_pool.scope_id());
        if (point_pool.persistent())
            physical_residency->bind_funded_point_scope(point_pool.scope_id());
    }
    Impl(const AdaptiveProviderKernel& k, Teuchos::RCP<const Teuchos::Comm<int>> c, AdaptiveClosureOptions o,
        ContactHaloContext ctx, Budget& b, IdentityBudget& i, std::shared_ptr<Budget> arena,
        std::shared_ptr<ContactPhysicalResidency> physical)
        : lease(b), borrowed(b), kernel(&k), comm(std::move(c)), options(o), context(ctx), budget(&b), identities(&i),
          keeper(std::move(arena)), external_physical_residency(bool(physical)),
          physical_residency(make_residency(lease, keeper, std::move(physical))),
          pools(b, i, keeper, physical_residency ? IdentityPoolStorage::Persistent : IdentityPoolStorage::Flat),
          tile_bodies(b, keeper), owned(b, pools, keeper, physical_residency.get()),
          supports(b, pools.supports, pools.cells, physical_residency ? keeper : nullptr), lines(b, pools),
          owner_rows(b, i, pools, k, comm->getRank(), keeper), tiles(b, keeper),
          authority_tile_counts(b, keeper), resident_generations(b, keeper)
    {
        lease.add(sizeof(Impl));
        bind_physical_pool_scopes();
    }
    Impl(const Impl& source, Budget& b, IdentityBudget& i, std::shared_ptr<Budget> arena,
        std::size_t* source_overlap, std::shared_ptr<ContactPhysicalResidency> physical)
        : lease(b), borrowed(b), kernel(source.kernel), comm(source.comm), options(source.options),
          context(source.context), budget(&b), identities(&i), keeper(std::move(arena)),
          external_physical_residency(bool(physical)),
          physical_residency(source.physical_residency
                                 ? make_residency(lease, keeper, std::move(physical), source.physical_residency.get())
                                 : nullptr),
          point_arena((source.native_source_protocol || source.pools.points.source() || source.owned.source())
                          ? make_point_arena() : nullptr),
          source_manifests(source.source_manifests), pools(source.pools, b, i, keeper, point_arena),
          tile_bodies(b, keeper), owned(source.owned, b, pools, point_arena, keeper, physical_residency.get()),
          supports(source.supports, b, pools.supports, pools.cells, keeper), lines(source.lines, b, pools),
          owner_rows(source.owner_rows, b, i, pools, keeper), tiles(b, keeper), subfaces(source.subfaces),
          authority_tile_counts(source.authority_tile_counts, b, keeper),
          resident_generations(source.resident_generations, b, keeper),
          statistics(source.statistics),
          last_layer(source.last_layer), point_generation(source.point_generation),
          periodic_tiles(source.periodic_tiles), active_tile_histories(source.active_tile_histories),
          authority_tiles(source.authority_tiles), ready(source.ready),
          native_source_protocol(source.native_source_protocol),
          native_inventory_complete(source.native_inventory_complete), inherited_overlap_active(true)
    {
        lease.add(sizeof(Impl));
        bind_physical_pool_scopes();
        if (physical_residency && source_overlap && source.physical_residency)
        {
                const auto point = source.physical_residency->scoped_point_credit_bytes();
                const auto fixed = source.physical_residency->scoped_fixed_pool_credit_bytes();
                if (fixed > std::numeric_limits<std::size_t>::max() - point)
                    throw std::invalid_argument("Incidence shared pool metadata exceeds its capacity.");
                const auto paid = point + fixed;
                if (paid > *source_overlap || paid > b.used)
                    throw std::invalid_argument("Shared point-page metadata exceeds incidence source overlap.");
                *source_overlap -= paid;
                b.used -= paid;
        }
        Lease scratch(b);
        std::set<const Layer*> layers;
        std::set<const LayerHistory*> histories;
        if (point_arena && !physical_residency)
        {
            cartesian_detail::CreditConstruction credits(point_arena);
            cartesian_detail::Allocation imported(b);
            // Live layers and detached subfaces enter one physical-owner
            // admission, even if either side is the sole remaining holder.
            if (native_source_protocol && active_tile_histories)
                for (std::size_t n = 0; n < source.tiles.size(); ++n)
                    for (const auto& [ordinal, body] : source.tiles.value_at(n).layers())
                        if (body)
                            static_cast<const SavedLayer&>(*body).retain_dependencies(
                                credits, layer_source_leases, imported);
            if (subfaces)
                subfaces->retain_source_dependencies(credits, layer_source_leases, imported);
            lease.bytes += std::exchange(imported.bytes, 0);
            credits.committed = true;
        }
        tiles.fork_from(source.tiles);
        tile_index_foreign_bytes = tiles.inherited_metadata_bytes();
        lease.add(tile_index_foreign_bytes);
        if (!physical_residency && (active_tile_histories || !subfaces))
            for (std::size_t n = 0; n < source.tiles.size(); ++n)
            {
                const auto& tile = source.tiles.value_at(n);
                if (tile.history && !histories.contains(tile.history.get()))
                {
                    scratch.add(pointer_set_node);
                    histories.insert(tile.history.get());
                    lease.add(history_bytes(*tile.history));
                }
                for (const auto& [layer, body] : tile.layers())
                    if (body && !layers.contains(body.get()))
                    {
                        scratch.add(pointer_set_node);
                        layers.insert(body.get());
                        lease.add(layer_bytes(*body, native_source_protocol));
                        if (body->final_output)
                            admit_subface_owner(body->final_output, body->final_output->origin(),
                                body->final_output->charged_bytes());
                    }
                if (!subfaces)
                {
                    admit_subface_owner(tile.body.body_storage_owner(), tile.body.credit_origin(), tile.body.body_bytes());
                    admit_subface_owner(tile.body.credit_owner(), tile.body.credit_origin(),
                        tile.body.credit_bytes() - tile.body.body_bytes());
                }
            }
        if (subfaces)
        {
            admit_subface_owner(subfaces, subfaces->origin(), subfaces->charged_bytes());
            lease.add(subfaces->index_bytes());
            lease.add(subfaces->exclusive_payload_bytes());
            subface_foreign_bytes = subfaces->index_bytes() + subfaces->exclusive_payload_bytes();
            if (!physical_residency)
                subfaces->visit_payload_owners(
                    [&](const auto& owner, const Budget* origin, std::size_t bytes)
                    {
                        const bool current_pool_credit =
                            origin == source.budget && source.pools.points.physical_pool().owns_page_credit(owner.get());
                        admit_subface_owner(owner, origin, bytes, current_pool_credit ? source_overlap : nullptr);
                    });
        }
    }
    void retire_inherited_metadata(std::size_t bytes) noexcept
    {
        // Retired candidate paths are still pinned by the accepted source.
        // Their imported charge lasts until the enclosing operation finishes.
        if (inherited_overlap_active)
            retired_inherited_metadata_bytes += bytes;
        else
        {
            lease.bytes -= bytes;
            budget->used -= bytes;
        }
    }
    void finish_inherited_overlap() noexcept
    {
        lease.bytes -= retired_inherited_metadata_bytes;
        budget->used -= retired_inherited_metadata_bytes;
        retired_inherited_metadata_bytes = 0;
        inherited_overlap_active = false;
        authority_tile_counts.finish_overlap();
        resident_generations.finish_overlap();
        owner_rows.finish_inherited_overlap();
        owned.finish_inherited_overlap();
        supports.finish_overlap();
        pools.finish_overlap();
        if (physical_residency && !external_physical_residency)
            physical_residency->finish_overlap();
    }
    void sync_tile_index_charge()
    {
        const auto inherited = tiles.inherited_metadata_bytes();
        if (inherited > tile_index_foreign_bytes)
            throw std::logic_error("Incidence tile edit introduced foreign metadata pages.");
        const auto retired = tile_index_foreign_bytes - inherited;
        retire_inherited_metadata(retired);
        tile_index_foreign_bytes = inherited;
    }
    void observe_history(const std::shared_ptr<LayerHistory>& history, bool added)
    {
        if (!physical_residency || !history)
            return;
        using Role = IncidencePhysicalOwners::Role;
        IncidencePhysicalOwners owners(*physical_residency, point_arena.get(), *budget, added);
        if (!owners.observe(Role::LayerHistory, history, history->charge, history->budget))
            return;
        for (const auto& [ordinal, layer] : history->layers)
        {
            if (!layer || !owners.observe(Role::Layer, layer, layer->charge, layer->budget))
                continue;
            if (layer->final_output)
                owners.observe(Role::SubfaceRing, layer->final_output, layer->final_output->charged_bytes(),
                    layer->final_output->origin());
            if (native_source_protocol)
                owners.observe_proof(static_cast<const SavedLayer&>(*layer));
        }
    }
    template<class Edit> void update_tile(TileId id, const Edit& edit)
    {
        const auto before = tiles.at(id).history;
        tiles.update(id, edit);
        const auto& after = tiles.at(id).history;
        if (after != before)
        {
            observe_history(after, true);
            observe_history(before, false);
        }
        if (after && !before)
            ++active_tile_histories;
        else if (before && !after)
            --active_tile_histories;
        sync_tile_index_charge();
    }
    void admit_subface_owner(const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes,
        std::size_t* source_overlap = nullptr)
    {
        if (origin == budget || borrowed.contains(owner))
            return;
        if (source_overlap)
        {
            if (bytes > *source_overlap || !borrowed.adopt(owner, bytes))
                throw std::invalid_argument("Subface point-page credit exceeds source overlap admission.");
            *source_overlap -= bytes;
            return;
        }
        lease.add(bytes);
        if (borrowed.adopt(owner, bytes))
            lease.bytes -= bytes;
    }
    bool exports_source_credit(const void* identity) const noexcept
    {
        bool found = false;
        const auto inspect = [&](const auto& owner, const Budget*, std::size_t, bool)
        { found = found || owner.get() == identity; };
        if (!physical_residency && pools.points.source())
            pools.points.source()->visit_dependencies(inspect);
        if (!physical_residency && owned.inventory())
            owned.inventory()->visit_dependencies(inspect);
        if (physical_residency)
            found = found || bool(physical_residency->usage().find({identity,
                std::uint32_t(ContactPhysicalResidency::Role::CreditMetadata)}));
        else if (native_source_protocol)
            for (std::size_t n = 0; n < tiles.size(); ++n)
                for (const auto& [ordinal, body] : tiles.value_at(n).layers())
                    if (body)
                    {
                        const auto& saved = static_cast<const SavedLayer&>(*body);
                        found = found || (saved.saved_input && saved.saved_input->imports_owner(identity)) ||
                                (saved.saved_output && saved.saved_output->imports_owner(identity)) ||
                                (saved.saved_edges && saved.saved_edges->imports_owner(identity));
                        for (const auto& completion : saved.completions)
                            if (completion)
                                found = found || completion.get() == identity || completion->imports_owner(identity);
                        for (const auto& lease : saved.leases)
                            found = found || lease.get() == identity;
                    }
        // Session-private import tokens are not themselves re-imported by a
        // no-edit descendant. Their current-arena metadata stays in accepted
        // overlap unless an immutable SavedLayer/descriptor actually exports it.
        return found || lines.imports_source_owner(identity) || tile_bodies.imports_source_owner(identity) ||
               (!physical_residency && subfaces && subfaces->imports_source_owner(identity));
    }
    std::size_t source_arena_overlap_metadata() const noexcept
    {
        auto bytes = point_arena ? point_arena->charge : 0;
        if (point_arena)
            for (const auto& [owner, credit] : point_arena->entries)
                if (!exports_source_credit(credit.get()))
                    bytes += credit->metadata;
        return bytes;
    }
    void census_saved_layers()
    {
        if (!native_source_protocol)
        {
            // The default explicit path gains no admission or retained header.
            statistics.retained_explicit_layer_input_slots = 0;
            statistics.retained_explicit_layer_output_slots = 0;
            statistics.retained_explicit_layer_edge_result_slots = 0;
            statistics.retained_completion_owners = 0;
            statistics.retained_expanded_completion_result_slots = 0;
            for (std::size_t n = 0; n < tiles.size(); ++n)
            {
                const auto& history = tiles.value_at(n).layers();
                for (std::size_t ordinal = 0; ordinal < history.size(); ++ordinal)
                {
                    const auto& body = history[ordinal].second;
                    if (!body)
                        continue;
                    bool seen = false;
                    for (std::size_t prior = 0; prior < ordinal; ++prior)
                        seen = seen || history[prior].second == body;
                    if (seen)
                        continue;
                    statistics.retained_explicit_layer_input_slots += body->input.capacity();
                    statistics.retained_explicit_layer_output_slots += body->final_output ? body->final_output->capacity() : body->output.capacity();
                    for (const auto& edge : body->edges)
                        statistics.retained_explicit_layer_edge_result_slots += edge.result.capacity();
                }
            }
            return;
        }
        // Observations only. This admitted census runs during collective
        // mutation; noexcept statistics reads never allocate or prove identity.
        Lease scratch(*budget);
        std::set<const void*> retained, active, descriptors;
        const auto first = [&](auto& seen, const void* owner)
        {
            if (seen.contains(owner))
                return false;
            scratch.add(pointer_set_node);
            seen.insert(owner);
            return true;
        };
        auto& result = statistics;
        result.active_saved_layers = result.retained_saved_layers = 0;
        result.retained_explicit_layer_input_slots = result.retained_explicit_layer_output_slots = 0;
        result.retained_explicit_layer_edge_result_slots = 0;
        result.retained_certified_layer_input_slots = result.retained_certified_layer_output_slots = 0;
        result.retained_certified_layer_edge_result_slots = result.retained_certified_layer_descriptor_bytes = 0;
        result.retained_layer_endpoint_identity_slots = 0;
        result.retained_completion_owners = result.retained_expanded_completion_result_slots = 0;
        const auto count_parts = [&](const CertifiedIncidenceResultParts* parts)
        {
            if (parts && first(descriptors, parts->storage_identity()))
                result.retained_certified_layer_descriptor_bytes += parts->known_payload_bytes();
        };
        const auto count_loop = [&](const auto& view)
        {
            if (!view || !first(descriptors, view->storage_identity()))
                return;
            result.retained_certified_layer_descriptor_bytes += view->known_payload_bytes();
            count_parts(view->result_parts());
        };
        for (std::size_t n = 0; n < tiles.size(); ++n)
        {
            const auto& tile = tiles.value_at(n);
            if (native_source_protocol)
                if (const auto* body = at_layer(tile, last_layer); body && first(active, body))
                    ++result.active_saved_layers;
            for (const auto& [ordinal, body] : tile.layers())
            {
                if (!body || !first(retained, body.get()))
                    continue;
                const auto& layer = *body;
                result.retained_explicit_layer_input_slots += layer.input.capacity();
                result.retained_explicit_layer_output_slots += layer.final_output ? layer.final_output->capacity() : layer.output.capacity();
                for (const auto& edge : layer.edges)
                    result.retained_explicit_layer_edge_result_slots += edge.result.capacity();
                if (!native_source_protocol)
                    continue;
                ++result.retained_saved_layers;
                const auto& saved = static_cast<const SavedLayer&>(layer);
                result.retained_certified_layer_input_slots += saved.saved_input ? saved.saved_input->size() : 0;
                result.retained_certified_layer_output_slots += saved.saved_output ? saved.saved_output->size() : 0;
                if (saved.saved_edges)
                    for (std::size_t e = 0; e < saved.edges.size(); ++e)
                        if (saved.edges[e].result.empty() && (e >= saved.completions.size() || !saved.completions[e]))
                            result.retained_certified_layer_edge_result_slots += saved.saved_edges->parts()[e].size();
                result.retained_layer_endpoint_identity_slots += saved.endpoint_facts.capacity();
                result.retained_certified_layer_descriptor_bytes +=
                    sizeof(SavedLayer) + saved.completions.capacity() * sizeof(PooledLineCompletionView) +
                    saved.leases.capacity() * sizeof(decltype(saved.leases)::value_type);
                for (const auto& completion : saved.completions)
                    if (completion && first(descriptors, completion.get()))
                    {
                        ++result.retained_completion_owners;
                        result.retained_expanded_completion_result_slots += completion->result_capacity();
                        result.retained_explicit_layer_edge_result_slots += completion->result_capacity();
                        result.retained_certified_layer_descriptor_bytes += completion->known_payload_bytes();
                    }
                count_loop(saved.saved_input);
                count_loop(saved.saved_output);
                count_parts(saved.saved_edges ? &*saved.saved_edges : nullptr);
            }
        }
    }
    bool any(bool local) const
    {
        int value = local, result = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &value, &result);
        return result;
    }
    void diagnostic(const char* event) const noexcept
    {
        if (!std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            return;
        std::size_t tile_owners = tile_bodies.retained_bytes(), layer_owners = 0, layers = 0, edge_replies = 0;
        for (std::size_t tile_index = 0; tile_index < tiles.size(); ++tile_index)
        {
            const auto& tile = tiles.value_at(tile_index);
            tile_owners += tile.history ? history_bytes(*tile.history) : 0;
            for (const auto& [layer, value] : tile.layers())
                if (value)
                {
                    ++layers;
                    layer_owners += layer_bytes(*value, native_source_protocol);
                    for (std::size_t e = 0; e < value->edges.size(); ++e)
                        edge_replies += edge_result(*value, e, native_source_protocol).size();
                }
        }
        std::fprintf(stderr,
            "ADAPTIVE_INCIDENCE rank=%d event=%s phase=%s used=%zu peak=%zu cumulative=%zu "
            "tiles=%zu tile_owner_bytes=%zu layers=%zu layer_owner_bytes=%zu edge_reply_points=%zu "
            "cell_identities=%zu point_identities=%zu line_identities=%zu dirty_supports=%zu dirty_lines=%zu\n",
            comm->getRank(), event, diagnostic_phase, budget->used, budget->peak, budget->cumulative, tiles.size(),
            tile_owners, layers, layer_owners, edge_replies, pools.cells.size(), pools.points.size(),
            pools.lines.size(), statistics.dirty_supports, statistics.dirty_lines);
        std::fprintf(stderr,
            "ADAPTIVE_INCIDENCE_STORAGE rank=%d phase=%s pool_cells=%zu pool_domains=%zu pool_supports=%zu "
            "pool_lines=%zu pool_points=%zu pool_planes=%zu pool_normals=%zu pool_images=%zu owned_membership=%zu "
            "support_registry=%zu "
            "line_registry=%zu owner_rows=%zu session_metadata_and_shared=%zu\n",
            comm->getRank(), diagnostic_phase, pools.cells.retained_bytes(), pools.domains.retained_bytes(),
            pools.supports.retained_bytes(), pools.lines.retained_bytes(), pools.points.retained_bytes(),
            pools.planes.retained_bytes(), pools.normals.retained_bytes(), pools.images.retained_bytes(),
            owned.retained_bytes(), supports.retained_bytes(), lines.retained_bytes(), owner_rows.metadata_bytes(),
            lease.bytes + borrowed.retained_bytes() + tile_bodies.retained_bytes() + tiles.metadata_bytes());
        const auto bodies = tile_bodies.statistics();
        const auto line_storage = lines.active_storage();
        std::fprintf(stderr,
            "ADAPTIVE_INCIDENCE_NATIVE_LINES rank=%d phase=%s active_lines=%zu explicit_records=%zu "
            "run_records=%zu runs=%zu explicit_bytes=%zu run_bytes=%zu source_records=%zu source_families=%zu "
            "source_bytes=%zu source_catalog_bytes=%zu source_maps=%zu source_mapped_components=%zu\n",
            comm->getRank(), diagnostic_phase, line_storage.lines, line_storage.explicit_records,
            line_storage.run_records, line_storage.runs, line_storage.explicit_bytes, line_storage.run_bytes,
            line_storage.source_records, line_storage.source_families, line_storage.source_bytes,
            line_storage.source_catalog_bytes, line_storage.source_maps, line_storage.source_mapped_components);
        std::fprintf(stderr,
            "ADAPTIVE_INCIDENCE_TILE_BODIES rank=%d phase=%s pages=%zu wide_pages=%zu occupied_records=%zu "
            "live_records=%zu record_capacity=%zu ring_size=%zu ring_capacity=%zu ring_bytes=%zu "
            "directory_metadata_bytes=%zu page_body_bytes=%zu page_credit_overhead_bytes=%zu "
            "deferred_pages=%zu deferred_body_bytes=%zu deferred_metadata_bytes=%zu source_pages=%zu "
            "source_records=%zu\n",
            comm->getRank(), diagnostic_phase, bodies.pages, bodies.wide_pages, bodies.occupied_records,
            bodies.live_records, bodies.record_capacity, bodies.ring_size, bodies.ring_capacity, bodies.ring_bytes,
            bodies.metadata_bytes, bodies.page_body_bytes, bodies.page_credit_overhead_bytes, bodies.deferred_pages,
            bodies.deferred_body_bytes, bodies.deferred_metadata_bytes, bodies.source_pages, bodies.source_records);
    }
    void phase(const char* name) noexcept
    {
        diagnostic_phase = name;
        diagnostic("begin");
    }
    void diagnostic_work(
        const char* event, const LayerMap& working, const std::vector<PooledLineEdit>& buffered) noexcept
    {
        if (!std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            return;
        std::size_t pending = working.size() * layer_map_node + buffered.capacity() * sizeof(PooledLineEdit);
        std::size_t bodies = 0, layers = 0;
        for (const auto& [id, work] : working)
        {
            pending += work.lines_and_receipts.capacity() * sizeof(Slot) +
                       work.completed_edges.capacity() * sizeof(PooledLineCompletionView);
            if (work.body)
            {
                ++layers;
                bodies += layer_bytes(*work.body, native_source_protocol) - owner_bytes;
            }
        }
        observed_work_capacity_peak = std::max(observed_work_capacity_peak, pending + bodies);
        observed_layer_capacity_peak = std::max(observed_layer_capacity_peak, bodies);
        std::fprintf(stderr,
            "ADAPTIVE_INCIDENCE_WORK rank=%d event=%s phase=%s tiles=%zu pending_buffer_and_index_bytes=%zu "
            "explicit_layers=%zu layer_object_and_buffer_bytes=%zu shared_control_allowance=%zu "
            "observed_work_capacity_peak=%zu observed_layer_capacity_peak=%zu arena_used=%zu arena_peak=%zu\n",
            comm->getRank(), event, diagnostic_phase, working.size(), pending, layers, bodies, layers * owner_bytes,
            observed_work_capacity_peak, observed_layer_capacity_peak, budget->used, budget->peak);
    }
    void reap_deferred() noexcept
    {
        borrowed.reap();
        owned.reap_deferred();
        supports.reap_deferred();
        lines.reap_deferred();
    }
    void defer_layer(const std::shared_ptr<Layer>& value)
    {
        if (!physical_residency && value && value->budget != budget)
        {
            const auto bytes = layer_bytes(*value, native_source_protocol);
            if (borrowed.adopt(value, bytes))
                lease.bytes -= bytes;
        }
    }
    void defer_history(const std::shared_ptr<LayerHistory>& value)
    {
        if (physical_residency || !value)
            return;
        if (value->budget != budget)
        {
            const auto bytes = history_bytes(*value);
            if (borrowed.adopt(value, bytes))
                lease.bytes -= bytes;
        }
        for (const auto& [layer, body] : value->layers)
            defer_layer(body);
    }
    std::size_t page_bytes() const
    {
        std::uint64_t available = budget->limit - budget->used, common = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &available, &common);
        return std::min<std::size_t>(256 * 1024, common / 16);
    }
    TileEntry new_tile(const ReceivedTileRecipe& recipe, SubfaceProvenanceBuilder* provenance,
        const OwnedLeafLookup& lookup)
    {
        // The transient vector remains admitted during page insertion. Destroy
        // it before its scoped charge retires; published pages own no vectors
        // or shared control blocks per tile.
        Lease scratch(*budget);
        scratch.add(3 * recipe.face->points().size() * sizeof(IncidencePools::Point) + sizeof(PooledTile) +
                    recipe.face->points().size() * sizeof(Slot));
        auto draft = freeze_tile(recipe, comm->getRank(), pools);
        if (draft.intrinsic_loop.capacity() != recipe.face->points().size())
            throw std::invalid_argument("Transient tile ring capacity differs from admission.");
        const bool native = native_source_protocol && source_manifests &&
                            !source_manifests->has_explicit_support(recipe.support) &&
                            source_manifests->native_member(recipe.first, recipe.domain) &&
                            (!recipe.second || source_manifests->native_member(*recipe.second, recipe.domain));
        const auto generation = resident_generations.at(draft.support);
        auto captured = provenance ? provenance->append(recipe, generation, lookup) : SubfaceProvenanceRef{};
        return {tile_bodies.append(draft), {}, false, generation, native, std::move(captured), {}};
    }
    LayerHistory& edit_history(TileEntry& entry)
    {
        if (entry.history && entry.history.use_count() == 1 && entry.history->budget == budget)
            return *entry.history;
        const auto count = entry.layers().size();
        const auto bytes =
            sizeof(LayerHistory) + owner_bytes + count * sizeof(decltype(LayerHistory::layers)::value_type);
        budget->add(bytes);
        std::shared_ptr<LayerHistory> value;
        try
        {
            value = std::make_shared<LayerHistory>();
        }
        catch (...)
        {
            budget->used -= bytes;
            throw;
        }
        value->budget = budget;
        value->keeper = keeper;
        value->charge = bytes;
        value->layers.reserve(count);
        if (value->layers.capacity() != count)
            throw std::invalid_argument("Layer history capacity differs from admission.");
        value->layers.assign(entry.layers().begin(), entry.layers().end());
        defer_history(entry.history);
        entry.history = std::move(value);
        return *entry.history;
    }
    std::shared_ptr<Layer> allocate_layer(std::size_t bytes)
    {
        budget->add(bytes);
        std::shared_ptr<Layer> layer;
        try
        {
            layer = native_source_protocol ? std::shared_ptr<Layer>(std::make_shared<SavedLayer>())
                                           : std::make_shared<Layer>();
        }
        catch (...)
        {
            budget->used -= bytes;
            throw;
        }
        layer->budget = budget;
        layer->keeper = keeper;
        layer->charge = bytes;
        return layer;
    }
    std::shared_ptr<Layer> new_layer(const LayerWork& work)
    {
        const auto input = work.input;
        std::optional<CertifiedIncidenceLoopView> inherited;
        if (native_source_protocol && work.input_owner)
        {
            const auto& previous = static_cast<const SavedLayer&>(*work.input_owner);
            if (previous.saved_output && input == output_loop(previous, true))
                inherited = previous.saved_output;
            else if (previous.saved_input && input == input_loop(previous, true))
                inherited = previous.saved_input;
        }
        cartesian_detail::Allocation imported(*budget);
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
        std::optional<cartesian_detail::CreditConstruction> credits;
        if (inherited)
        {
            if (!point_arena)
                point_arena = make_point_arena();
            credits.emplace(point_arena);
            inherited->retain_dependencies(*credits, leases, imported);
        }
        const auto object = native_source_protocol ? sizeof(SavedLayer) : sizeof(Layer);
        auto layer = allocate_layer(object + owner_bytes);
        if (native_source_protocol)
        {
            auto& saved = static_cast<SavedLayer&>(*layer);
            saved.saved_input = std::move(inherited);
            saved.leases.swap(leases);
            layer->charge += std::exchange(imported.bytes, 0);
            if (credits)
                credits->committed = true;
        }
        if (!native_source_protocol || !static_cast<SavedLayer&>(*layer).saved_input)
        {
            reserve_owned(layer->input, input.size(), layer->charge);
            layer->input.assign(input.begin(), input.end());
        }
        return layer;
    }
    std::shared_ptr<Layer> clone_layer(const Layer& source)
    {
        cartesian_detail::Allocation imported(*budget);
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
        std::optional<cartesian_detail::CreditConstruction> credits;
        if (native_source_protocol)
        {
            if (!point_arena)
                point_arena = make_point_arena();
            credits.emplace(point_arena);
            static_cast<const SavedLayer&>(source).retain_dependencies(*credits, leases, imported);
        }
        auto layer = allocate_layer(layer_bytes(source, native_source_protocol));
        layer->input = source.input;
        layer->output = source.output;
        layer->final_output = source.final_output;
        layer->edges = source.edges;
        if (native_source_protocol)
        {
            auto& saved = static_cast<SavedLayer&>(*layer);
            const auto& old = static_cast<const SavedLayer&>(source);
            saved.saved_input = old.saved_input;
            saved.saved_output = old.saved_output;
            saved.saved_edges = old.saved_edges;
            saved.endpoint_facts = old.endpoint_facts;
            saved.completions = old.completions;
            saved.source_generation = old.source_generation;
            saved.kernel_layer = old.kernel_layer;
            saved.native_member = old.native_member;
            // Original leases are dependencies, not copied duplicate capacity.
            const auto old_lease_bytes = old.leases.capacity() * sizeof(decltype(old.leases)::value_type);
            layer->charge -= old_lease_bytes;
            budget->used -= old_lease_bytes;
            saved.leases.swap(leases);
            layer->charge += std::exchange(imported.bytes, 0);
            credits->committed = true;
        }
        return layer;
    }
    template<class T> void reserve_owned(std::vector<T>& target, std::size_t count, std::size_t& charge)
    {
        if (count <= target.capacity())
            return;
        const auto old = target.capacity() * sizeof(T);
        budget->add(count * sizeof(T));
        charge += count * sizeof(T);
        target.reserve(count);
        if (target.capacity() != count)
            throw std::invalid_argument("Incidence capacity differs from admission.");
        budget->used -= old;
        charge -= old;
    }
    void put_layer(TileEntry& state, unsigned layer, std::shared_ptr<Layer> body, bool replay)
    {
        auto& history = edit_history(state);
        auto& layers = history.layers;
        if (replay && layer < 63)
        {
            auto next = std::lower_bound(
                layers.begin(), layers.end(), layer + 1, [](const auto& value, unsigned n) { return value.first < n; });
            if (next == layers.end() || next->first != layer + 1)
            {
                auto old = std::upper_bound(
                    layers.begin(), layers.end(), layer, [](unsigned n, const auto& value) { return n < value.first; });
                auto previous = old == layers.begin() ? std::shared_ptr<Layer>{} : std::prev(old)->second;
                const auto offset = next - layers.begin();
                reserve_owned(layers, layers.size() + 1, history.charge);
                layers.insert(layers.begin() + offset, {layer + 1, std::move(previous)});
            }
        }
        auto position = std::lower_bound(
            layers.begin(), layers.end(), layer, [](const auto& value, unsigned n) { return value.first < n; });
        if (position != layers.end() && position->first == layer)
        {
            defer_layer(position->second);
            position->second = std::move(body);
        }
        else
        {
            const auto offset = position - layers.begin();
            reserve_owned(layers, layers.size() + 1, history.charge);
            layers.insert(layers.begin() + offset, {layer, std::move(body)});
        }
    }
    const auto& tile_image(const TileBodyScalars& tile) const { return pools.images.get(tile.image_slot); }
    int tile_authority(const TileBodyScalars& tile) const
    {
        return authority(pools.supports.get(tile.support), comm->getSize());
    }
    template<class Emit> void visit_edges(const TileEntry& state, TileBodyLoopView loop, Emit&& emit)
    {
        const auto tile = state.tile();
        Lease scratch(*budget);
        scratch.add(2 * PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(loop.size()) +
                    loop.size() *
                        (3 * sizeof(IncidencePools::Point) + 3 * sizeof(PLIC::LineIdentity<ArithmeticMode::Ordinary>)));
        const auto face = restore_tile(tile, loop, pools);
        for (unsigned image = 0; image < (tile_image(tile).is_zero() ? 1U : 2U); ++image)
        {
            // Preserve translated-loop validation before any line registration;
            // finite image rounding can collapse a valid source polygon.
            const auto physical =
                image ? face->translated(PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile_image(tile)))
                      : face;
            const auto points = physical->vertex_identities();
            const auto edge_lines = physical->edge_support_identities();
            if (points.size() != edge_lines.size())
                throw std::invalid_argument("Invalid tile edge incidence.");
            for (std::size_t edge = 0; edge < points.size(); ++edge)
            {
                Lease key_scratch(*budget);
                const auto key_size =
                    16 + pools.domains.get(tile.domain).size() + PLIC::canonical_key_size(edge_lines[edge]);
                key_scratch.add(4 * key_size);
                Writer key;
                key.data.reserve(key_size);
                key.bytes(pools.domains.get(tile.domain));
                key.bytes(PLIC::canonical_key(edge_lines[edge]));
                PooledLineSubscriber sub;
                sub.support = tile.support;
                sub.resident = comm->getRank();
                sub.support_authority = tile_authority(tile);
                sub.emission = tile.emission;
                sub.image = image;
                sub.edge = edge;
                sub.source_generation = state.support_generation;
                sub.native_source = state.native_source;
                sub.first = pools.points.intern(points[edge]);
                sub.last = pools.points.intern(points[(edge + 1) % points.size()]);
                emit(pools.lines.intern(std::move(key.data)), sub);
            }
        }
    }
    std::size_t edge_count(const TileBodyScalars& tile, std::size_t points) const
    {
        if (points < 3 || points > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Invalid retained incidence loop size.");
        return points * (tile_image(tile).is_zero() ? 1 : 2);
    }
    LayerWork make_work(const TileBodyScalars& tile, TileBodyLoopView input, bool require_all, Lease& scratch,
        std::shared_ptr<const Layer> input_owner = {})
    {
        const auto count = edge_count(tile, input.size());
        constexpr auto bits = std::numeric_limits<Slot>::digits;
        const auto words = count / bits + (count % bits != 0);
        const auto regions = native_source_protocol ? 3U : 2U;
        if (count > (std::numeric_limits<std::size_t>::max() - words) / regions ||
            regions * count + words > budget->limit / sizeof(Slot))
            throw std::invalid_argument("Incidence receipt capacity overflow.");
        scratch.add((regions * count + words) * sizeof(Slot));
        LayerWork result;
        result.input = std::move(input);
        result.input_owner = std::move(input_owner);
        result.require_all = require_all;
        result.lines_and_receipts.resize(regions * count + words, 0);
        if (result.lines_and_receipts.capacity() != regions * count + words)
            throw std::invalid_argument("Incidence receipt capacity differs from admission.");
        return result;
    }
    void register_edge(const TileBodyScalars& tile, LayerWork& work, Slot line, const PooledLineSubscriber& sub)
    {
        const auto count = edge_count(tile, work.input.size());
        const auto edge = sub.image * work.input.size() + sub.edge;
        work.lines_and_receipts.at(edge) = line;
        work.lines_and_receipts.at(count + edge) = sub.first;
        if (native_source_protocol)
            work.lines_and_receipts.at(2 * count + edge) = sub.last;
    }
    std::pair<Slot, Slot> physical_endpoints(const TileBodyScalars& tile, const LayerWork& work, std::size_t edge) const
    {
        const auto count = edge_count(tile, work.input.size());
        const auto next = edge - edge % work.input.size() + (edge + 1) % work.input.size();
        return {work.lines_and_receipts[count + edge],
            work.lines_and_receipts[native_source_protocol ? 2 * count + edge : count + next]};
    }
    bool endpoint_reply(
        const TileBodyScalars& tile, const LayerWork& work, std::size_t edge, std::span<const Slot> result) const
    {
        if (result.size() != 2)
            return false;
        const auto [first, last] = physical_endpoints(tile, work, edge);
        // A closed_segment endpoint-only answer is ordered from the supplied
        // first endpoint to the last. Other orders remain explicit replies.
        return result[0] == first && result[1] == last;
    }
    void materialize_endpoints(const TileBodyScalars& tile, LayerWork& work, Lease& scratch)
    {
        auto body = new_layer(work);
        reserve_owned(body->output, work.input.size(), body->charge);
        body->output.assign(work.input.begin(), work.input.end());
        const auto count = edge_count(tile, work.input.size());
        reserve_owned(body->edges, count, body->charge);
        if (native_source_protocol && !work.completed_edges.empty())
        {
            auto& saved = static_cast<SavedLayer&>(*body);
            saved.completions.swap(work.completed_edges);
            const auto bytes = saved.completions.capacity() * sizeof(PooledLineCompletionView);
            saved.charge += bytes;
            scratch.bytes -= bytes;
        }
        for (std::size_t edge = 0; edge < count; ++edge)
        {
            body->edges.push_back({work.lines_and_receipts[edge], {}});
            if (native_source_protocol)
            {
                const auto& saved = static_cast<const SavedLayer&>(*body);
                if (edge < saved.completions.size() && saved.completions[edge])
                    continue;
            }
            const auto [first, last] = physical_endpoints(tile, work, edge);
            auto& result = body->edges.back().result;
            reserve_owned(result, 2, body->charge);
            result.push_back(first);
            result.push_back(last);
        }
        work.body = std::move(body);
    }
    void receive_edge(const TileBodyScalars& tile, LayerWork& work, Slot line, const PooledLineSubscriber& sub)
    {
        if (sub.edge >= work.input.size())
            throw std::invalid_argument("Invalid incidence edge ordinal.");
        const auto count = edge_count(tile, work.input.size());
        const auto edge = sub.image * work.input.size() + sub.edge;
        if (edge >= count || work.lines_and_receipts[edge] != line)
            throw std::invalid_argument("Mismatched incremental edge reply.");
        // Original v2 replies do not contain request first/last fields. Their
        // member/result envelope is authenticated by the exchange; the exact
        // original endpoints remain in work's admitted request facts and are
        // compared to the sealed query when the saved view is constructed.
        constexpr auto bits = std::numeric_limits<Slot>::digits;
        auto& word = work.lines_and_receipts[(native_source_protocol ? 3 : 2) * count + edge / bits];
        const auto bit = Slot(1) << (edge % bits);
        if (word & bit)
            throw std::invalid_argument("Duplicate incremental edge reply.");
        word |= bit;
    }
    void certify_receipts(const TileBodyScalars& tile, const LayerWork& work) const
    {
        if (!work.require_all)
            return;
        constexpr auto bits = std::numeric_limits<Slot>::digits;
        const auto count = edge_count(tile, work.input.size());
        for (std::size_t edge = 0; edge < count; ++edge)
            if (!(work.lines_and_receipts[(native_source_protocol ? 3 : 2) * count + edge / bits] &
                    (Slot(1) << (edge % bits))))
                throw std::invalid_argument("Missing complete edge reply.");
    }
    template<class LineAt> std::vector<Slot> legacy_edge_order(std::size_t count, LineAt line_at, Lease& scratch) const
    {
        scratch.add(count * sizeof(Slot));
        std::vector<Slot> order;
        order.reserve(count);
        for (std::size_t edge = 0; edge < count; ++edge)
            order.push_back(edge);
        std::sort(order.begin(), order.end(),
            [&](Slot a, Slot b)
            {
                const auto& first = pools.lines.get(line_at(a));
                const auto& last = pools.lines.get(line_at(b));
                const auto first_authority = authority(first, comm->getSize());
                const auto last_authority = authority(last, comm->getSize());
                if (first_authority != last_authority)
                    return first_authority < last_authority;
                if (first != last)
                    return first < last;
                return a < b; // original physical-image/edge subscription ordinal
            });
        return order;
    }
    void finish_endpoint_loop(const TileEntry& state, LayerWork& work)
    {
        const auto tile = state.tile();
        const auto& input = work.input;
        const auto count = edge_count(tile, input.size());
        const auto candidates = 2 * count;
        Lease scratch(*budget);
        scratch.add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(input.size()) +
                    PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(input.size() + candidates) +
                    (3 * input.size() + 4 * candidates) * sizeof(IncidencePools::Point) + candidates * 128);
        const auto order = legacy_edge_order(count, [&](Slot edge) { return work.lines_and_receipts[edge]; }, scratch);
        const auto face = restore_tile(tile, input, pools);
        std::vector<IncidencePools::Point> points;
        points.reserve(candidates);
        for (const auto edge : order)
        {
            const auto [first, last] = physical_endpoints(tile, work, edge);
            for (const auto point : {first, last})
                points.push_back(edge >= input.size() ? PLIC::point_translated<ArithmeticMode::Ordinary>(
                                                            pools.points.get(point), tile_image(tile))
                                                      : pools.points.get(point));
        }
        // Use the exact physical replies in the original return order, including
        // finite periodic inverse translation. Endpoint round trips may add a
        // vertex; the full native enrichment and retained-support checks remain.
        const auto enriched = face->with_edge_vertices(points);
        const auto vertices = enriched->vertex_identities();
        const bool unchanged = vertices.size() == input.size() &&
                               std::equal(vertices.begin(), vertices.end(), input.begin(),
                                   [&](const auto& point, Slot slot) { return point == pools.points.get(slot); });
        if (unchanged && (!work.require_all || input == state.body.intrinsic_loop()))
            return;
        auto value = new_layer(work);
        reserve_owned(value->output, vertices.size(), value->charge);
        for (const auto& point : vertices)
            value->output.push_back(pools.points.intern(point));
        // Empty edges encode only an ordered, certified endpoint-query baseline.
        work.body = std::move(value);
    }
    void rebuild_output(const TileBodyScalars& tile, Layer& layer)
    {
        const auto input = input_loop(layer, native_source_protocol);
        std::size_t count = 0;
        for (std::size_t e = 0; e < layer.edges.size(); ++e)
            count += edge_result(layer, e, native_source_protocol).size();
        Lease scratch(*budget);
        scratch.add(PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(input.size()) +
                    PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(input.size() + count) +
                    (3 * input.size() + 4 * count) * sizeof(IncidencePools::Point) + count * 128);
        const auto face = restore_tile(tile, input, pools);
        std::vector<IncidencePools::Point> points;
        points.reserve(count);
        const auto order =
            legacy_edge_order(layer.edges.size(), [&](Slot edge) { return layer.edges[edge].line; }, scratch);
        for (const auto e : order)
        {
            const bool image = e >= input.size();
            for (const auto point : edge_result(layer, e, native_source_protocol))
                points.push_back(
                    image ? PLIC::point_translated<ArithmeticMode::Ordinary>(pools.points.get(point), tile_image(tile))
                          : pools.points.get(point));
        }
        const auto enriched = face->with_edge_vertices(points);
        const auto vertices = enriched->vertex_identities();
        if (native_source_protocol)
            static_cast<SavedLayer&>(layer).saved_output.reset();
        layer.final_output.reset();
        reserve_owned(layer.output, vertices.size(), layer.charge);
        layer.output.clear();
        for (const auto& vertex : vertices)
            layer.output.push_back(pools.points.intern(vertex));
    }
    bool intrinsic_fixed_layer(const TileEntry& state, const Layer& layer) const
    {
        const auto input = input_loop(layer, native_source_protocol);
        if (input != state.body.intrinsic_loop() || output_loop(layer, native_source_protocol) != input)
            return false;
        const auto size = input.size();
        if (layer.edges.empty())
            return true;
        if (layer.edges.size() != size * (tile_image(state.tile()).is_zero() ? 1 : 2))
            return false;
        for (std::size_t edge = 0; edge < layer.edges.size(); ++edge)
        {
            const auto& result = edge_result(layer, edge, native_source_protocol);
            if (result.size() != 2)
                return false;
            auto first = pools.points.get(input[edge % size]);
            auto last = pools.points.get(input[(edge + 1) % size]);
            if (edge >= size)
            {
                const auto image =
                    PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile_image(state.tile()));
                first = PLIC::point_translated<ArithmeticMode::Ordinary>(first, image);
                last = PLIC::point_translated<ArithmeticMode::Ordinary>(last, image);
            }
            const auto& a = pools.points.get(result[0]);
            const auto& b = pools.points.get(result[1]);
            if (a != first || b != last)
                return false;
        }
        return true;
    }

    using ChangedTiles = ChangedTileMap;
    void remember(ChangedTiles& before, TileId id, Lease& scratch)
    {
        if (before.contains(id))
            return;
        scratch.add(changed_map_node);
        const auto found = tiles.find(id);
        before.emplace(id, found ? std::optional<TileEntry>(*found) : std::optional<TileEntry>{});
    }
    void replace_group(Slot support, std::span<const ReceivedTileRecipe> recipes, ChangedTiles& before, Lease& scratch,
        TileMap::Batch& batch, SubfaceProvenanceBuilder* provenance, const OwnedLeafLookup& lookup)
    {
        batch.replace_support(support);
        for (auto i = tiles.lower_bound({support, 0}); i < tiles.size() && tiles.key_at(i).first == support; ++i)
        {
            const auto id = tiles.key_at(i);
            const auto& tile = tiles.value_at(i);
            remember(before, id, scratch);
            defer_history(tile.history);
            observe_history(tile.history, false);
            active_tile_histories -= bool(tile.history);
            periodic_tiles -= !tile_image(tile.tile()).is_zero();
        }
        for (const auto& recipe : recipes)
        {
            auto state = new_tile(recipe, provenance, lookup);
            periodic_tiles += !recipe.second_image.is_zero();
            const TileId id{support, recipe.emission};
            remember(before, id, scratch);
            batch.insert(id, std::move(state));
        }
    }
    void fetch_supports(
        const OwnedLeafLookup& lookup, ChangedTiles& changed, Lease& scratch, TileMap::Batch& tile_batch)
    {
        std::optional<SubfaceProvenanceBuilder> provenance;
        collective(*comm,
            [&]
            {
                if (keeper)
                    provenance.emplace(pools, keeper, comm->getRank());
            });
        struct Source
        {
            Slot support;
            AdaptiveBytes domain;
            std::vector<RegistryMember> members;
            std::vector<PrivateTileRecipe> recipes;
        };
        std::vector<Source> sources;
        std::map<Slot, std::vector<RegistryMember>> expected;
        auto page = std::make_unique<Lease>(*budget);
        auto page_identities = std::make_unique<IdentityBudget>();
        fetch_dirty_support_groups(
            supports, lookup, context, comm, *budget, *identities, 256,
            [&](std::span<const char> key)
            {
                Lease key_scratch(*budget);
                key_scratch.add(key.size());
                AdaptiveBytes bytes(key.begin(), key.end());
                const auto slot = pools.supports.find(bytes);
                if (!slot)
                    throw std::invalid_argument("Unrequested owner support fetch.");
                if (expected.contains(*slot))
                    return;
                page->add(expected_support_node);
                auto& members = expected[*slot];
                owned.visit_support_members(*slot,
                    [&](Slot cell, std::uint64_t revision, std::size_t ordinal, const auto& member)
                    {
                        page->append(members);
                        page->add(key_capacity(pools.cells.get(cell)));
                        members.push_back(
                            {pools.cells.get(cell), revision, ordinal, member.encoded_size, comm->getRank()});
                    });
                if (members.empty())
                    throw std::invalid_argument("Foreign support fetch recipient.");
                std::sort(members.begin(), members.end(), [](const auto& a, const auto& b)
                    { return std::tie(a.owner, a.cell, a.surface) < std::tie(b.owner, b.cell, b.surface); });
            },
            [&](std::span<const char> key, std::span<const RegistryMember> members,
                std::span<const SupportOccurrence> occurrences)
            {
                ++statistics.dirty_supports;
                statistics.dirty_support_members += members.size();
                Lease key_scratch(*budget);
                key_scratch.add(key.size());
                const auto support_slot = *pools.supports.find(AdaptiveBytes(key.begin(), key.end()));
                const auto* previous = authority_tile_counts.find(support_slot);
                authority_tiles -= previous ? *previous : 0;
                if (members.empty())
                {
                    authority_tile_counts.set(support_slot, 0);
                    return;
                }
                std::size_t count = 0;
                page->append(sources);
                page->add(occurrences.front().domain.size() + members.size() * sizeof(RegistryMember));
                Source source;
                source.support = support_slot;
                source.domain = occurrences.front().domain;
                source.members.reserve(members.size());
                for (const auto& member : members)
                {
                    page->add(key_capacity(member.cell));
                    source.members.push_back(member);
                }
                auto charge = [&](std::size_t n) { page->add(n); };
                auto face_charge = [&](const SupportFace& face)
                {
                    const auto before_bytes = budget->used;
                    page_identities->face(*budget, *face);
                    page->bytes += budget->used - before_bytes;
                };
                auto emit = [&](PrivateTileRecipe recipe)
                {
                    if (authority_tiles >= options.tile_budget)
                        throw std::invalid_argument("Authority tile budget exhausted.");
                    ++authority_tiles;
                    ++count;
                    page->append(source.recipes);
                    face_charge(recipe.face);
                    source.recipes.push_back(std::move(recipe));
                };
                make_support_recipes(
                    key, occurrences, budget->limit - budget->used, options.tile_budget, charge, face_charge, emit);
                authority_tile_counts.set(support_slot, count);
                sources.push_back(std::move(source));
            },
            [&]
            {
                Lease descriptors(*budget);
                std::vector<IncidenceDeliverySource> outgoing;
                std::vector<IncidenceDeliveryExpected> incoming;
                collective(*comm,
                    [&]
                    {
                        descriptors.add(sources.size() * sizeof(IncidenceDeliverySource) +
                                        expected.size() * sizeof(IncidenceDeliveryExpected));
                        outgoing.reserve(sources.size());
                        for (const auto& source : sources)
                            outgoing.push_back({pools.supports.get(source.support), source.domain, source.members,
                                source.recipes, supports.generation(source.support)});
                        incoming.reserve(expected.size());
                        for (const auto& [support, members] : expected)
                        {
                            const auto generation = resident_generations.find(support);
                            if (generation && *generation == std::numeric_limits<std::uint64_t>::max())
                                throw std::invalid_argument("Resident support generation exhausted.");
                            incoming.push_back({pools.supports.get(support),
                                generation ? *generation + 1 : 0, members});
                        }
                    });
                deliver_incidence_tiles(outgoing, incoming, context, comm, *budget, *identities, lookup,
                    [&](std::span<const char> key, std::uint64_t generation,
                        std::span<const ReceivedTileRecipe> recipes)
                    {
                        Lease key_scratch(*budget);
                        key_scratch.add(key.size());
                        const auto support = pools.supports.find(AdaptiveBytes(key.begin(), key.end()));
                        if (!support)
                            throw std::invalid_argument("Foreign delivered support.");
                        resident_generations.set(*support, generation);
                        replace_group(*support, recipes, changed, scratch, tile_batch,
                            provenance ? &*provenance : nullptr, lookup);
                    });
                collective(*comm,
                    [&]
                    {
                        std::vector<Source>().swap(sources);
                        decltype(expected)().swap(expected);
                        page_identities.reset();
                        page.reset();
                        page = std::make_unique<Lease>(*budget);
                        page_identities = std::make_unique<IdentityBudget>();
                    });
            });
        collective(*comm,
            [&]
            {
                if (provenance)
                    provenance->freeze();
                tile_batch.publish(options.tile_budget);
                sync_tile_index_charge();
            });
        supports.clear_dirty();
    }

    void replay(ChangedTiles& changed, Lease& scratch, bool bootstrap)
    {
        Lease frontier(*budget);
        std::set<TileId> replacements;
        std::set<TileId> pending;
        collective(*comm,
            [&]
            {
                tile_bodies.freeze();
                lines.begin_replay(bootstrap);
                for (const auto& [id, state] : changed)
                {
                    frontier.add(tile_set_node);
                    pending.insert(id);
                    if (!bootstrap)
                    {
                        frontier.add(tile_set_node);
                        replacements.insert(id);
                    }
                }
            });
        const unsigned previous_last = last_layer;
        const bool periodic = any(periodic_tiles != 0);
        for (unsigned layer = 0; layer < 64; ++layer)
        {
            phase("edge_staging");
            Lease iteration(*budget);
            std::vector<PooledLineEdit> buffered;
            LayerMap working;
            struct WorkObservation
            {
                Impl& state;
                const LayerMap& working;
                const std::vector<PooledLineEdit>& buffered;
                ~WorkObservation() { state.diagnostic_work("scope_exit", working, buffered); }
            } work_observation{*this, working, buffered};
            auto current_id = pending.begin();
            unsigned edit_phase = 0;
            std::size_t cursor = 0;
            const auto next_edit = [&](PooledLineEdit& value)
            {
                for (;;)
                {
                    if (cursor < buffered.size())
                    {
                        value = buffered[cursor++];
                        return true;
                    }
                    const auto old_bytes = buffered.capacity() * sizeof(PooledLineEdit);
                    std::vector<PooledLineEdit>().swap(buffered);
                    iteration.bytes -= old_bytes;
                    budget->used -= old_bytes;
                    cursor = 0;
                    if (current_id == pending.end())
                    {
                        if (++edit_phase == 2)
                            return false;
                        current_id = pending.begin();
                    }
                    if (current_id == pending.end())
                        continue;
                    const auto id = *current_id++;
                    const auto current = tiles.find(id);
                    if (edit_phase == 0)
                    {
                        const auto original = changed.find(id);
                        const TileEntry* old_state = nullptr;
                        if (bootstrap)
                            old_state = layer && current != nullptr ? current : nullptr;
                        else if (replacements.contains(id))
                            old_state = original == changed.end() ? nullptr
                                        : original->second        ? &*original->second
                                                                  : nullptr;
                        else if (current != nullptr)
                            old_state = current;
                        const auto* old_layer = old_state ? at_layer(*old_state, layer) : nullptr;
                        if (old_layer || (old_state && old_state->initialized))
                            visit_edges(*old_state,
                                old_layer ? input_loop(*old_layer, native_source_protocol)
                                          : old_state->body.intrinsic_loop(),
                                [&](Slot line, const auto& sub)
                                {
                                    iteration.append(buffered);
                                    buffered.push_back({line, sub, true});
                                });
                        continue;
                    }
                    if (current == nullptr)
                        continue;
                    remember(changed, id, scratch);
                    auto previous_owner = layer ? at_layer_owner(*current, layer - 1) : std::shared_ptr<Layer>{};
                    const auto* previous_layer = previous_owner.get();
                    const auto input = previous_layer ? output_loop(*previous_layer, native_source_protocol)
                                                      : current->body.intrinsic_loop();
                    auto next = make_work(current->tile(), input, true, iteration, std::move(previous_owner));
                    visit_edges(*current, input,
                        [&](Slot line, const auto& sub)
                        {
                            register_edge(current->tile(), next, line, sub);
                            iteration.append(buffered);
                            buffered.push_back({line, sub, false});
                        });
                    iteration.add(layer_map_node);
                    working.emplace(id, std::move(next));
                }
            };
            const auto prepare_reply_work = [&](Slot line, const PooledLineSubscriber& sub,
                                                bool existing_only = false) -> LayerWork&
            {
                const TileId id{sub.support, sub.emission};
                const auto current = tiles.find(id);
                if (current == nullptr || tile_authority(current->tile()) != sub.support_authority ||
                    sub.resident != comm->getRank())
                    throw std::invalid_argument("Foreign incremental edge reply.");
                if (native_source_protocol && (sub.source_generation != current->support_generation ||
                                                  sub.native_source != current->native_source))
                    throw std::invalid_argument("Stale native support generation in edge reply.");
                auto work = working.find(id);
                if (work == working.end())
                {
                    if (existing_only)
                        throw std::invalid_argument("Cached completion has no original materialized request work.");
                    auto previous_owner = at_layer_owner(*current, layer);
                    const auto* previous = previous_owner.get();
                    if (!previous && !current->initialized)
                        throw std::invalid_argument("Missing cached edge subscription.");
                    const auto input =
                        previous ? input_loop(*previous, native_source_protocol) : current->body.intrinsic_loop();
                    auto next = make_work(current->tile(), input, false, iteration, std::move(previous_owner));
                    const auto count = edge_count(current->tile(), input.size());
                    bool captured = native_source_protocol && previous && previous->edges.size() == count;
                    if (captured)
                    {
                        const auto& saved = static_cast<const SavedLayer&>(*previous);
                        for (std::size_t edge = 0; edge < count && captured; ++edge)
                        {
                            PooledLineSubscriber original;
                            if (edge < saved.completions.size() && saved.completions[edge])
                                original = saved.completions[edge]->proof()->original_request();
                            else if (saved.saved_edges && saved.edges[edge].result.empty())
                            {
                                const auto& view = saved.saved_edges->parts()[edge];
                                if (const auto* request = view.original_request())
                                    original = *request;
                                else
                                    original = view.recipe()->at(view.query_ordinal(), pools);
                            }
                            else if (saved.endpoint_facts.size() == 2 * count)
                            {
                                original.support = current->tile().support;
                                original.resident = comm->getRank();
                                original.support_authority = tile_authority(current->tile());
                                original.emission = current->tile().emission;
                                original.image = edge / input.size();
                                original.edge = edge % input.size();
                                original.source_generation = saved.source_generation;
                                original.native_source = saved.native_member;
                                original.first = saved.endpoint_facts[2 * edge];
                                original.last = saved.endpoint_facts[2 * edge + 1];
                            }
                            else
                            {
                                captured = false;
                                break;
                            }
                            if (original.support != current->tile().support || original.resident != comm->getRank() ||
                                original.support_authority != tile_authority(current->tile()) ||
                                original.emission != current->tile().emission ||
                                original.image != edge / input.size() || original.edge != edge % input.size() ||
                                original.source_generation != current->support_generation ||
                                original.native_source != current->native_source)
                                throw std::invalid_argument("Changed original saved query member.");
                            register_edge(current->tile(), next, previous->edges[edge].line, original);
                        }
                    }
                    // This is the original cached-reply reconstruction path,
                    // used once only when no complete saved original exists.
                    if (!captured)
                        visit_edges(*current, input, [&](Slot expected, const auto& edge)
                            { register_edge(current->tile(), next, expected, edge); });
                    if (previous && !previous->edges.empty())
                    {
                        if (previous->edges.size() != count)
                            throw std::invalid_argument("Invalid cached incidence edge count.");
                        for (std::size_t edge = 0; edge < count; ++edge)
                            if (next.lines_and_receipts[edge] != previous->edges[edge].line)
                                throw std::invalid_argument("Changed cached incidence line identity.");
                        next.body = clone_layer(*previous);
                    }
                    iteration.add(layer_map_node);
                    work = working.emplace(id, std::move(next)).first;
                }
                if (sub.edge >= work->second.input.size() || sub.image > 1)
                    throw std::invalid_argument("Invalid original edge request ordinal.");
                const auto edge = sub.image * work->second.input.size() + sub.edge;
                const auto count = edge_count(current->tile(), work->second.input.size());
                if (edge >= count || work->second.lines_and_receipts[edge] != line)
                    throw std::invalid_argument("Changed original edge request line.");
                return work->second;
            };
            const auto resolve_original =
                [&](Slot line, const PooledLineSubscriber& decoded) -> std::optional<PooledLineSubscriber>
            {
                auto& work = prepare_reply_work(line, decoded);
                const auto& tile = tiles.at(TileId{decoded.support, decoded.emission});
                const auto edge = decoded.image * work.input.size() + decoded.edge;
                const auto [first, last] = physical_endpoints(tile.tile(), work, edge);
                PooledLineSubscriber original;
                original.support = tile.tile().support;
                original.resident = comm->getRank();
                original.support_authority = tile_authority(tile.tile());
                original.emission = tile.tile().emission;
                original.image = decoded.image;
                original.edge = decoded.edge;
                original.source_generation = tile.support_generation;
                original.native_source = tile.native_source;
                original.first = first;
                original.last = last;
                return original;
            };
            const auto legacy_reply = [&](Slot line, const PooledLineSubscriber& sub)
            {
                ++statistics.dirty_subscriptions;
                auto& value = prepare_reply_work(line, sub);
                const auto& current = tiles.at(TileId{sub.support, sub.emission});
                receive_edge(current.tile(), value, line, sub);
                const auto edge = sub.image * value.input.size() + sub.edge;
                if (!value.body && endpoint_reply(current.tile(), value, edge, sub.result))
                    return;
                if (!value.body)
                    materialize_endpoints(current.tile(), value, iteration);
                auto& result = value.body->edges[edge].result;
                reserve_owned(result, sub.result.size(), value.body->charge);
                result.assign(sub.result.begin(), sub.result.end());
                if (native_source_protocol)
                {
                    auto& saved = static_cast<SavedLayer&>(*value.body);
                    if (edge < saved.completions.size())
                        saved.completions[edge].reset();
                }
            };
            const auto store_completion = [&](LayerWork& work, std::size_t edge, PooledLineCompletion&& completion)
            {
                auto& saved = static_cast<SavedLayer&>(*work.body);
                const auto& original = completion.proof()->original_request();
                const auto& current = tiles.at(TileId{original.support, original.emission});
                const auto count = edge_count(current.tile(), work.input.size());
                const auto [first, last] = physical_endpoints(current.tile(), work, edge);
                if (original.first != first || original.last != last ||
                    !same_saved_point(completion.proof()->endpoint(0), pools.points.get(first)) ||
                    !same_saved_point(completion.proof()->endpoint(1), pools.points.get(last)))
                    throw std::invalid_argument("Completed edge proof differs from its original supplied endpoints.");
                reserve_owned(saved.completions, count, saved.charge);
                saved.completions.resize(count);
                auto handle = std::move(completion).share_immutable();
                release_saved_vector(saved.edges[edge].result, saved);
                saved.completions[edge] = std::move(handle);
            };
            const auto completed_reply = [&](Slot line, PooledLineCompletion&& completion)
            {
                const auto& original = completion.proof()->original_request();
                ++statistics.dirty_subscriptions;
                auto& value = prepare_reply_work(line, original);
                const auto& current = tiles.at(TileId{original.support, original.emission});
                receive_edge(current.tile(), value, line, original);
                const auto edge = original.image * value.input.size() + original.edge;
                const auto count = edge_count(current.tile(), value.input.size());
                if (!value.body)
                {
                    if (value.completed_edges.empty())
                    {
                        iteration.add(count * sizeof(PooledLineCompletionView));
                        value.completed_edges.resize(count);
                        if (value.completed_edges.capacity() != count)
                            throw std::invalid_argument("Pending completion capacity differs from admission.");
                    }
                    value.completed_edges[edge] = std::move(completion).share_immutable();
                    if (endpoint_reply(current.tile(), value, edge, value.completed_edges[edge]->result()))
                        return;
                    materialize_endpoints(current.tile(), value, iteration);
                }
                else
                    store_completion(value, edge, std::move(completion));
            };
            const auto consume_endpoints = [&](Slot line, const PooledLineBorrowedCompletion& completion)
            {
                if (completion.context() != context || completion.layer() != layer ||
                    completion.kernel_witness_layer() != layer || completion.point_pool_identity() != &pools.points ||
                    !std::ranges::equal(completion.line_key(), pools.lines.get(line)))
                    throw std::invalid_argument("Stale borrowed edge-completion attestation.");
                const auto& original = completion.original_request();
                auto& value = prepare_reply_work(line, original, true);
                const auto& current = tiles.at(TileId{original.support, original.emission});
                const auto edge = original.image * value.input.size() + original.edge;
                const auto [first, last] = physical_endpoints(current.tile(), value, edge);
                if (!original.native_source || original.has_result() || original.result.capacity() ||
                    original.first != first || original.last != last ||
                    !std::ranges::equal(completion.support_key(), pools.supports.get(current.tile().support)) ||
                    !same_saved_point(completion.endpoint(0), pools.points.get(first)) ||
                    !same_saved_point(completion.endpoint(1), pools.points.get(last)))
                    throw std::invalid_argument("Borrowed edge reply differs from its original supplied endpoints.");
                // This exact authenticated result needs no retained receipt while
                // the tile remains endpoint-only. A later changed edge first
                // materializes the captured ordered baselines, then the existing
                // cached-read phase obtains genuine proofs before compact sealing.
                if (value.body || !endpoint_reply(current.tile(), value, edge, completion.result()))
                    return false;
                receive_edge(current.tile(), value, line, original);
                ++statistics.dirty_subscriptions;
                return true;
            };
            PooledLineOriginalRequest original_request;
            PooledLineCompletionReceive completion_receive;
            PooledLineCompletionConsume completion_consume;
            if (native_source_protocol && keeper)
            {
                original_request = [&resolve_original](Slot line, const auto& decoded)
                { return resolve_original(line, decoded); };
                completion_receive = [&completed_reply](Slot line, PooledLineCompletion&& completion)
                { completed_reply(line, std::move(completion)); };
                completion_consume = [&consume_endpoints](Slot line, const PooledLineBorrowedCompletion& completion)
                { return consume_endpoints(line, completion); };
                if (!point_arena)
                    point_arena = make_point_arena();
            }
            const auto page = page_bytes();
            phase("line_exchange");
            exchange_pooled_line_edits(
                lines, pools, next_edit, layer, context, comm, *budget, page,
                [&legacy_reply](Slot line, const auto& sub) { legacy_reply(line, sub); }, {}, native_source_protocol,
                {}, {}, original_request, completion_receive, keeper, point_arena, completion_consume);
            if (completion_receive)
            {
                auto cached_work = working.begin();
                std::size_t cached_edge = 0;
                const auto needs_completion = [](const SavedLayer& saved, std::size_t edge)
                {
                    const bool completed = edge < saved.completions.size() && saved.completions[edge];
                    const bool compact = saved.saved_edges && saved.edges[edge].result.empty();
                    return !completed && !compact;
                };
                const auto next_cached = [&](PooledLineCompletionRequest& request)
                {
                    while (cached_work != working.end())
                    {
                        auto& work = cached_work->second;
                        const auto& current = tiles.at(cached_work->first);
                        if (!work.body || !current.native_source)
                        {
                            ++cached_work;
                            cached_edge = 0;
                            continue;
                        }
                        const auto& saved = static_cast<const SavedLayer&>(*work.body);
                        while (cached_edge < saved.edges.size())
                        {
                            const auto edge = cached_edge++;
                            if (!needs_completion(saved, edge))
                                continue;
                            const auto [first, last] = physical_endpoints(current.tile(), work, edge);
                            PooledLineCompletionRequest next;
                            next.line = saved.edges[edge].line;
                            next.original.support = current.tile().support;
                            next.original.resident = comm->getRank();
                            next.original.support_authority = tile_authority(current.tile());
                            next.original.emission = current.tile().emission;
                            next.original.image = edge / work.input.size();
                            next.original.edge = edge % work.input.size();
                            next.original.source_generation = current.support_generation;
                            next.original.native_source = current.native_source;
                            next.original.first = first;
                            next.original.last = last;
                            request = std::move(next);
                            ++statistics.cached_completion_requests;
                            if (authority(pools.lines.get(request.line), comm->getSize()) != comm->getRank())
                                ++statistics.cached_completion_remote_requests;
                            return true;
                        }
                        ++cached_work;
                        cached_edge = 0;
                    }
                    return false;
                };
                const auto cached_completion = [&](Slot line, PooledLineCompletion&& completion)
                {
                    const auto& proof = *completion.proof();
                    if (!proof.is_cached_read() || proof.layer() != layer || proof.context() != context)
                        throw std::invalid_argument("Stale cached edge-completion attestation.");
                    const auto& original = proof.original_request();
                    auto& work = prepare_reply_work(line, original, true);
                    if (!work.body)
                        throw std::invalid_argument("Cached edge completion has no materialized original baseline.");
                    const auto edge = original.image * work.input.size() + original.edge;
                    const auto& saved = static_cast<const SavedLayer&>(*work.body);
                    if (!needs_completion(saved, edge))
                        throw std::invalid_argument("Repeated cached edge completion.");
                    const auto baseline = edge_result(saved, edge, true);
                    const auto actual = completion.result();
                    if (baseline.size() != actual.size())
                        throw std::invalid_argument("Cached edge completion differs from its original stored result.");
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        if (actual[i] != baseline[i] ||
                            !same_saved_point(pools.points.get(actual[i]), pools.points.get(baseline[i])))
                            throw std::invalid_argument("Cached edge completion changed an original ordered identity.");
                    // The independently authenticated cached-read result replaces
                    // the original baseline only after exact full equality. No
                    // changed-reply receipt bit or dirty count is replayed.
                    store_completion(work, edge, std::move(completion));
                    ++statistics.cached_completion_received;
                };
                phase("cached_line_completion");
                const auto cached_page = page_bytes();
                fetch_pooled_line_completions(
                    lines, pools, [&next_cached](PooledLineCompletionRequest& request) { return next_cached(request); },
                    layer, context, comm, *budget, cached_page,
                    [&cached_completion](Slot line, PooledLineCompletion&& completion)
                    { cached_completion(line, std::move(completion)); }, keeper, point_arena);
                collective(*comm,
                    [&]
                    {
                        for (const auto& [id, work] : working)
                        {
                            if (!work.body || !tiles.at(id).native_source)
                                continue;
                            const auto& saved = static_cast<const SavedLayer&>(*work.body);
                            for (std::size_t edge = 0; edge < saved.edges.size(); ++edge)
                                if (needs_completion(saved, edge))
                                    throw std::invalid_argument("Missing authenticated cached edge completion.");
                        }
                    });
            }
            diagnostic_work("complete_replies", working, buffered);
            bool size_changed = false;
            phase("edge_reconstruction");
            std::set<TileId> following;
            collective(*comm,
                [&]
                {
                    for (auto& [id, work] : working)
                    {
                        const auto& current = tiles.at(id);
                        certify_receipts(current.tile(), work);
                        if (work.body)
                        {
                            for (std::size_t e = 0; e < work.body->edges.size(); ++e)
                                if (edge_result(*work.body, e, native_source_protocol).empty())
                                    throw std::invalid_argument("Missing cached edge reply.");
                            rebuild_output(current.tile(), *work.body);
                            size_changed |= output_loop(*work.body, native_source_protocol).size() !=
                                            input_loop(*work.body, native_source_protocol).size();
                        }
                        else
                        {
                            finish_endpoint_loop(current, work);
                            if (!work.body && !work.require_all)
                            {
                                ++statistics.recomputed_tiles;
                                continue;
                            }
                        }
                        if (native_source_protocol && work.body)
                        {
                            auto& saved = static_cast<SavedLayer&>(*work.body);
                            saved.kernel_layer = layer;
                            saved.source_generation = current.support_generation;
                            saved.native_member = current.native_source;
                            const auto count = work.body->edges.size();
                            reserve_owned(saved.endpoint_facts, 2 * count, saved.charge);
                            saved.endpoint_facts.clear();
                            for (std::size_t e = 0; e < count; ++e)
                            {
                                const auto [first, last] = physical_endpoints(current.tile(), work, e);
                                saved.endpoint_facts.push_back(first);
                                saved.endpoint_facts.push_back(last);
                            }
                        }
                        ++statistics.recomputed_tiles;
                        const auto output = work.body ? output_loop(*work.body, native_source_protocol) : work.input;
                        size_changed |= output.size() != work.input.size();
                        const auto* previous = at_layer(current, layer);
                        const bool changed_output =
                            previous ? output != output_loop(*previous, native_source_protocol)
                                     : (!current.initialized || output != current.body.intrinsic_loop());
                        remember(changed, id, scratch);
                        update_tile(id,
                            [&](TileEntry& candidate)
                            {
                                if (work.body || !candidate.layers().empty())
                                    put_layer(candidate, layer, work.body, !bootstrap);
                                candidate.initialized = true;
                            });
                        if (changed_output)
                        {
                            iteration.add(tile_set_node);
                            following.insert(id);
                        }
                    }
                    // New and retired tiles remain explicit changes through the
                    // old depth, even when their new geometry has already settled.
                    for (const auto id : replacements)
                        if (!following.contains(id))
                        {
                            iteration.add(tile_set_node);
                            following.insert(id);
                        }
                });
            diagnostic_work("complete_reconstruction", working, buffered);
            const bool global_changed = any(size_changed);
            if ((!periodic || !global_changed) && layer >= previous_last)
            {
                last_layer = layer;
                collective(*comm,
                    [&]
                    {
                        for (const auto& [id, old] : changed)
                        {
                            const auto found = tiles.find(id);
                            if (found == nullptr)
                                continue;
                            const auto& entry = *found;
                            if (!entry.history)
                                continue;
                            const auto& previous = entry.layers();
                            const auto retained_end = std::upper_bound(previous.begin(), previous.end(), layer,
                                [](unsigned n, const auto& value) { return n < value.first; });
                            const bool intrinsic = std::all_of(previous.begin(), retained_end, [&](const auto& value)
                                { return !value.second || intrinsic_fixed_layer(entry, *value.second); });
                            if (!intrinsic && retained_end == previous.end())
                                continue;
                            update_tile(id, [&](TileEntry& candidate)
                                {
                                    if (intrinsic)
                                    {
                                        defer_history(candidate.history);
                                        candidate.history.reset();
                                    }
                                    else
                                    {
                                        auto& history = edit_history(candidate).layers;
                                        const auto future = std::upper_bound(history.begin(), history.end(), layer,
                                            [](unsigned n, const auto& value) { return n < value.first; });
                                        for (auto old = future; old != history.end(); ++old)
                                            defer_layer(old->second);
                                        history.erase(future, history.end());
                                    }
                                });
                        }
                        lines.finish_replay(layer);
                        statistics.dirty_lines = lines.processed_lines();
                    });
                return;
            }
            if (layer == 63)
                throw std::invalid_argument("Periodic edge-incidence propagation exhausted its bounded rounds.");
            collective(*comm,
                [&]
                {
                    const auto old_pending = pending.size() * tile_set_node;
                    pending.clear();
                    frontier.bytes -= old_pending;
                    budget->used -= old_pending;
                    frontier.add(following.size() * tile_set_node);
                    pending.insert(following.begin(), following.end());
                });
        }
    }
    template<class T> void release_saved_vector(std::vector<T>& values, Layer& layer)
    {
        const auto bytes = values.capacity() * sizeof(T);
        std::vector<T>().swap(values);
        layer.charge -= bytes;
        budget->used -= bytes;
    }
    void seal_saved_layer(
        const TileEntry& tile, SavedLayer& saved, const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot)
    {
        if (!saved.saved_input && !saved.input.empty())
            if (auto view =
                    CertifiedIncidenceLoopView::certify_quad(snapshot, saved.input, *budget, keeper, point_arena))
            {
                saved.saved_input = std::move(view);
                release_saved_vector(saved.input, saved);
            }
        if (!saved.edges.empty() && saved.endpoint_facts.size() == 2 * saved.edges.size())
        {
            cartesian_detail::Allocation scratch(*budget);
            std::vector<CertifiedIncidenceResultView> candidates;
            scratch.reserve(candidates, saved.edges.size());
            bool complete = true;
            const auto scalars = tile.tile();
            const auto input = input_loop(saved, true);
            for (std::size_t e = 0; e < saved.edges.size(); ++e)
            {
                // Unchanged cached results keep their original sealed member,
                // generation and actual kernel-layer witness independently.
                if (saved.edges[e].result.empty() && saved.saved_edges &&
                    (e >= saved.completions.size() || !saved.completions[e]))
                {
                    candidates.push_back(saved.saved_edges->parts()[e]);
                    continue;
                }
                PooledLineSubscriber member;
                member.resident = comm->getRank();
                member.support_authority = tile_authority(scalars);
                member.support = scalars.support;
                member.emission = scalars.emission;
                member.image = e / input.size();
                member.edge = e % input.size();
                member.source_generation = saved.source_generation;
                member.native_source = saved.native_member;
                member.first = saved.endpoint_facts[2 * e];
                member.last = saved.endpoint_facts[2 * e + 1];
                const auto completion =
                    e < saved.completions.size() ? saved.completions[e] : PooledLineCompletionView{};
                const auto witness_layer =
                    completion ? completion->proof()->kernel_witness_layer() : saved.kernel_layer;
                if (completion)
                    member = completion->proof()->original_request();
                const auto capture =
                    lines.capture_saved_source_query(saved.edges[e].line, saved.kernel_layer, witness_layer, member);
                const auto expected = completion ? completion->result() : std::span<const Slot>(saved.edges[e].result);
                std::optional<CertifiedIncidenceResultView> view;
                if (capture)
                    view =
                        CertifiedIncidenceResultView::certify(capture->recipe, capture->ordinal, expected, *snapshot);
                if (!view && completion)
                    view = CertifiedIncidenceResultView::certify_completed(*completion, saved.edges[e].line, member,
                        DistributedAdaptiveIncidenceSession::source_result_admission(snapshot, pools), expected, pools,
                        completion->proof()->context(), completion->proof()->layer(), *budget, keeper, point_arena);
                if (!view)
                {
                    complete = false;
                    break;
                }
                candidates.push_back(std::move(*view));
            }
            if (complete)
            {
                auto parts = CertifiedIncidenceResultParts::create(candidates, *budget, keeper, point_arena);
                saved.saved_edges = std::move(parts);
                for (auto& edge : saved.edges)
                    release_saved_vector(edge.result, saved);
                release_saved_vector(saved.endpoint_facts, saved);
                release_saved_vector(saved.completions, saved);
            }
        }
        if (!saved.saved_output && !saved.output.empty())
        {
            if (saved.saved_input && saved.saved_input->exact_replay(saved.output, *snapshot))
                saved.saved_output = saved.saved_input;
            else if (saved.saved_edges)
                saved.saved_output = CertifiedIncidenceLoopView::certify_concatenation(*saved.saved_edges, 0,
                    input_loop(saved, true).size(), saved.output, *snapshot, *budget, keeper, point_arena);
            if (!saved.saved_output)
                saved.saved_output =
                    CertifiedIncidenceLoopView::certify_quad(snapshot, saved.output, *budget, keeper, point_arena);
            if (saved.saved_output)
                release_saved_vector(saved.output, saved);
        }
    }
    void seal_saved_layers(const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot,
        std::optional<std::span<const TileId>> touched)
    {
        if (!native_source_protocol || !snapshot)
            return;
        Lease scratch(*budget);
        std::map<const Layer*, std::shared_ptr<Layer>> rewritten;
        using LoopKey = std::pair<std::size_t, std::uint64_t>;
        using KnownLoops = std::multimap<LoopKey, CertifiedIncidenceLoopView>;
        KnownLoops known_loops;
        std::set<const void*> known_descriptors;
        const auto loop_key = [](TileBodyLoopView loop)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto slot : loop)
            {
                hash ^= static_cast<std::uint64_t>(slot);
                hash *= 1099511628211ULL;
            }
            return LoopKey{loop.size(), hash};
        };
        const auto reuse_loop = [&](SavedLayer& saved, auto& explicit_loop, auto& view)
        {
            if (view || explicit_loop.empty())
                return;
            const auto candidates = known_loops.equal_range(loop_key(TileBodyLoopView(explicit_loop)));
            for (auto at = candidates.first; at != candidates.second; ++at)
                if (at->second.exact_replay(explicit_loop, *snapshot))
                {
                    view = at->second;
                    release_saved_vector(explicit_loop, saved);
                    return;
                }
        };
        const auto remember_loop = [&](const auto& view)
        {
            if (!view)
                return;
            if (known_descriptors.contains(view->storage_identity()))
                return;
            scratch.add(pointer_set_node + sizeof(KnownLoops::value_type) + 4 * sizeof(void*));
            known_descriptors.insert(view->storage_identity());
            // Integer hashes route only; reuse_loop always replays FULL facts.
            known_loops.emplace(loop_key(saved_loop(*view)), *view);
        };
        const auto seal = [&](TileId id)
        {
            const auto* current = tiles.find(id);
            if (!current || !current->history)
                return;
            update_tile(id, [&](TileEntry& tile)
                {
                    // Work only on candidate-owned histories. Old escaped sessions and
                    // undo histories keep their original vectors and descriptors.
                    auto& history = edit_history(tile);
                    for (auto& [ordinal, body] : history.layers)
                    {
                        if (!body)
                            continue;
                        if (auto at = rewritten.find(body.get()); at != rewritten.end())
                        {
                            body = at->second;
                            continue;
                        }
                        const auto* original = body.get();
                        if (body->budget != budget || body.use_count() > 1)
                        {
                            auto replacement = clone_layer(*body);
                            defer_layer(body);
                            body = std::move(replacement);
                        }
                        auto& saved = static_cast<SavedLayer&>(*body);
                        reuse_loop(saved, saved.input, saved.saved_input);
                        reuse_loop(saved, saved.output, saved.saved_output);
                        seal_saved_layer(tile, saved, snapshot);
                        remember_loop(saved.saved_input);
                        remember_loop(saved.saved_output);
                        scratch.add(sizeof(decltype(rewritten)::value_type) + 4 * sizeof(void*));
                        rewritten.emplace(original, body);
                    }
                });
        };
        if (touched)
            for (const auto id : *touched)
                seal(id);
        else
            for (std::size_t n = 0; n < tiles.size(); ++n)
                seal(tiles.key_at(n));
    }
    void compact_point_values(std::vector<IncidenceRowDelivery>& rows,
        std::optional<std::span<const TileId>> touched = std::nullopt)
    {
        census_saved_layers();
        if (!native_source_protocol || !keeper)
            return;
        if (pools.points.source() && pools.points.physical_pool().size() == 0 && rows.empty())
        {
            const auto storage = lines.active_storage();
            if (!storage.retained_explicit_records && !storage.retained_run_records)
            {
                seal_saved_layers(pools.points.source(), touched);
                census_saved_layers();
                return;
            }
        }
        const auto grid = trusted_cartesian_incidence_grid(*kernel);
        if (!grid || !source_manifests || !source_manifests->generation() ||
            source_manifests->generation() <= point_generation)
            throw std::invalid_argument("Incomplete native point-source generation.");
        const auto next_generation = source_manifests->generation();
        if (!point_arena)
            point_arena = make_point_arena();
        const auto admission = DistributedAdaptiveIncidenceSession::source_point_admission(*grid, context,
            comm->getRank(), next_generation, last_layer, pools.points.size(), owned.source(), source_manifests,
            pools.points.source());
        const bool recapture = bool(pools.points.source());
        const auto capture_spent = budget->cumulative;
        const auto previous_points = pools.points.source();
        auto snapshot = pools.points.prepare_source(admission, point_arena);
        const auto retire_physical_suffix = [&](auto& pool)
        {
            if (physical_residency && pool.persistent())
                pool.transfer_inherited_page_bytes(physical_residency->detach_funded_point_scope(pool.scope_id()));
        };
        if (!snapshot || !pools.points.publish_source(snapshot, retire_physical_suffix))
            return;
        if (physical_residency)
        {
            bind_physical_pool_scopes();
            IncidencePhysicalOwners added(*physical_residency, point_arena.get(), *budget, true);
            added.observe_snapshot(snapshot);
            IncidencePhysicalOwners retired(*physical_residency, point_arena.get(), *budget, false);
            retired.observe_snapshot(previous_points);
        }
        if (recapture)
        {
            ++statistics.source_point_recaptures;
            statistics.source_point_recaptured_records += snapshot->size();
            statistics.source_point_recapture_admitted_bytes += budget->cumulative - capture_spent;
        }
        if (snapshot->size() <= std::numeric_limits<std::uint32_t>::max())
        {
            // finish_rows created these objects privately in this bootstrap;
            // no row has been returned or bound into a halo. Reusing buffers
            // preserves their exact capacity and original deleter credit.
            auto view = ContactSourcePointView::create(snapshot, *budget, keeper, point_arena);
            for (const auto& original : rows)
            {
                auto& row = *const_cast<ContactIncidenceRow*>(original.row.get());
                if (!row.points.bind_private_initial_source(pools.points, view))
                    throw std::invalid_argument("Unexpected initial source-row construction kind.");
            }
            view->freeze();
            if (native_inventory_complete)
                (void) tile_bodies.certify_private_source_loops(snapshot, point_arena);
            lines.certify_source_ranges(DistributedAdaptiveIncidenceSession::source_line_admission(
                                            source_manifests, snapshot, context, next_generation, last_layer),
                point_arena);
            seal_saved_layers(snapshot, touched);
            census_saved_layers();
            point_generation = next_generation;
            return;
        }
        cartesian_detail::Allocation vector_memory(*budget);
        std::vector<IncidenceRowDelivery> replacement;
        vector_memory.reserve(replacement, rows.size());
        ContactPointBatch batch(pools.points);
        for (const auto& delivery : rows)
        {
            const auto& original = delivery.row;
            cartesian_detail::Allocation memory(*budget);
            memory.retain(
                sizeof(ContactIncidenceRow) + owner_bytes + key_capacity(original->key) + original->domain.size());
            ContactIncidenceRow row;
            row.key = original->key;
            row.domain = original->domain;
            row.revision = original->revision;
            row.incidence_generation = original->incidence_generation;
            const auto before = budget->used;
            try
            {
                row.points = ContactPointArray::pooled(batch, original->points.size(), *budget);
                for (const auto point : original->points)
                {
                    const auto physical = pools.points.find(point.physical),
                               normalized = pools.points.find(point.normalized);
                    if (!physical || !normalized)
                        throw std::invalid_argument("Source row lost a complete original point identity.");
                    row.points.push_reference(
                        batch.capture_source(*physical), batch.capture_source(*normalized), *budget);
                }
            }
            catch (...)
            {
                memory.bytes = cartesian_detail::add(memory.bytes, budget->used - before);
                throw;
            }
            memory.bytes = cartesian_detail::add(memory.bytes, budget->used - before);
            auto destroy = [arena = budget, keeper = keeper, charge = memory.bytes](const ContactIncidenceRow* value)
            {
                (void) keeper;
                delete value;
                arena->used -= charge;
            };
            std::unique_ptr<const ContactIncidenceRow, decltype(destroy)> pending(
                new ContactIncidenceRow(std::move(row)), std::move(destroy));
            memory.bytes = 0;
            replacement.push_back({IncidenceRowHandle(std::move(pending)), delivery.leaf});
        }
        batch.freeze();
        if (native_inventory_complete)
            (void) tile_bodies.certify_private_source_loops(snapshot, point_arena);
        lines.certify_source_ranges(DistributedAdaptiveIncidenceSession::source_line_admission(
                                        source_manifests, snapshot, context, next_generation, last_layer),
            point_arena);
        const auto previous = rows.capacity() * sizeof(IncidenceRowDelivery);
        rows.swap(replacement);
        {
            std::vector<IncidenceRowDelivery> retired;
            retired.swap(replacement);
        }
        budget->used -= previous;
        vector_memory.bytes = 0;
        seal_saved_layers(snapshot, touched);
        census_saved_layers();
        point_generation = next_generation;
    }
    struct DirtySubface
    {
        TileId id;
        CanonicalSubfaceView before;
    };
    void seal_subface(TileEntry& tile, SubfacePieceBuilder& builder)
    {
        if (tile.final_piece)
            return;
        auto layer = at_layer_owner(tile, last_layer);
        if (layer && !layer->output.empty())
        {
            if (layer->budget != budget || layer.use_count() > 2 || tile.history.use_count() > 1)
            {
                auto& history = edit_history(tile);
                const auto found = std::upper_bound(history.layers.begin(), history.layers.end(), last_layer,
                    [](unsigned value, const auto& entry) { return value < entry.first; });
                auto& target = std::prev(found)->second;
                auto replacement = clone_layer(*layer);
                defer_layer(target);
                target = replacement;
                layer = std::move(replacement);
            }
            layer->final_output = SharedSubfaceRing::transfer(layer->output, *budget, keeper, layer->charge);
        }
        const CertifiedIncidenceLoopView* certified = nullptr;
        if (layer && native_source_protocol)
        {
            const auto& saved = static_cast<const SavedLayer&>(*layer);
            if (saved.saved_output)
                certified = &*saved.saved_output;
        }
        tile.final_piece = builder.append(tile.provenance, tile.body, layer ? layer->final_output : nullptr, certified);
    }
    void publish_subfaces(std::shared_ptr<const AdaptiveSubfaceSnapshot> next)
    {
        const auto foreign = next->foreign_index_bytes() + next->foreign_exclusive_payload_bytes();
        if (foreign > subface_foreign_bytes)
            throw std::logic_error("Canonical subface update added unadmitted foreign pages.");
        const auto released = subface_foreign_bytes - foreign;
        subfaces = std::move(next);
        subface_foreign_bytes = foreign;
        retire_inherited_metadata(released);
    }
    void finish_subfaces(std::span<const DirtySubface> dirty = {})
    {
        if (!keeper)
            return;
        const auto controls = geometry_construction_controls();
        if (subfaces && dirty.empty())
        {
            publish_subfaces(AdaptiveSubfaceSnapshot::derive(
                subfaces, context, controls, comm->getRank(), {}, *budget, keeper,
                physical_residency.get(), point_arena.get()));
            return;
        }
        SubfacePieceBuilder builder(pools, keeper);
        Lease scratch(*budget);
        if (subfaces)
        {
            std::vector<CanonicalSubfaceChange> changes;
            scratch.add(dirty.size() * sizeof(CanonicalSubfaceChange));
            changes.reserve(dirty.size());
            if (changes.capacity() != dirty.size())
                throw std::invalid_argument("Canonical subface change capacity differs from admission.");
            for (const auto& item : dirty)
            {
                CanonicalSubfaceView after;
                if (tiles.find(item.id))
                    update_tile(item.id, [&](TileEntry& tile)
                        {
                            seal_subface(tile, builder);
                            after = tile.final_piece;
                        });
                changes.push_back({item.before, std::move(after)});
            }
            builder.freeze();
            publish_subfaces(AdaptiveSubfaceSnapshot::derive(
                subfaces, context, controls, comm->getRank(), changes, *budget, keeper,
                physical_residency.get(), point_arena.get()));
            return;
        }
        std::vector<CanonicalSubfaceView> views;
        scratch.add(tiles.size() * sizeof(CanonicalSubfaceView));
        views.reserve(tiles.size());
        if (views.capacity() != tiles.size())
            throw std::invalid_argument("Canonical subface scratch capacity differs from admission.");
        for (std::size_t index = 0; index < tiles.size(); ++index)
        {
            const auto id = tiles.key_at(index);
            update_tile(id, [&](TileEntry& tile)
                {
                    seal_subface(tile, builder);
                    views.push_back(tile.final_piece);
                });
        }
        builder.freeze();
        publish_subfaces(AdaptiveSubfaceSnapshot::seal(context, controls, comm->getRank(), views, *budget, keeper,
            physical_residency.get(), point_arena.get()));
    }
    std::vector<IncidenceRowDelivery> finish_rows(ChangedTiles& changed, std::span<const IncidenceCellVersion> retired,
        std::span<const AdaptiveCellKey> inserted, const OwnedLeafLookup& lookup, unsigned old_last, Lease& scratch,
        bool keep_identity, std::vector<DirtySubface>* dirty = nullptr, std::vector<TileId>* touched = nullptr)
    {
        collective(*comm,
            [&]
            {
                for (const auto& [id, old] : changed)
                    if (old)
                        owner_rows.remove(old->tile(), final_loop(*old, old_last, native_source_protocol));
                for (const auto& old : retired)
                    owner_rows.erase(*pools.cells.find(old.key));
                for (const auto& key : inserted)
                {
                    std::uint64_t generation = 0;
                    for (const auto& old : retired)
                        if (old.key == key)
                            generation = old.incidence_generation;
                    owner_rows.insert(*pools.cells.find(key), generation);
                }
                for (const auto& [id, old] : changed)
                {
                    auto current = tiles.find(id);
                    if (current)
                    {
                        if (current->final_piece && old && at_layer(*old, old_last) != at_layer(*current, last_layer))
                        {
                            update_tile(id, [](TileEntry& entry) { entry.final_piece = {}; });
                            current = tiles.find(id);
                        }
                        owner_rows.add(current->tile(), final_loop(*current, last_layer, native_source_protocol));
                        if (touched && native_source_protocol)
                        {
                            scratch.append(*touched);
                            touched->push_back(id);
                        }
                    }
                    if (keeper && dirty && (current ? !current->final_piece : old && old->final_piece))
                    {
                        scratch.append(*dirty);
                        dirty->push_back({id, old ? old->final_piece : CanonicalSubfaceView{}});
                    }
                }
            });
        collective(*comm,
            [&]
            {
                // Neither old tile bodies nor the change index is consulted
                // after contribution replacement. Retire them before row
                // allocation, then release only physically expired floors.
                const auto staged = changed.size() * changed_map_node;
                if (staged > scratch.bytes)
                    throw std::invalid_argument("Incidence change-index admission disagrees.");
                changed.clear();
                scratch.bytes -= staged;
                budget->used -= staged;
                (void) tile_bodies.reap();
                reap_deferred();
            });
        Lease result_memory(*budget);
        std::vector<IncidenceRowDelivery> result;
        collective(*comm,
            [&]
            {
                Lease patch_memory(*budget);
                auto patches = owner_rows.finish(lookup, keep_identity);
                patch_memory.bytes = patches.capacity() * sizeof(IncidenceOwnerRows::Patch);
                result_memory.add(patches.size() * sizeof(IncidenceRowDelivery));
                result.reserve(patches.size());
                for (auto& patch : patches)
                    result.push_back({std::move(patch.row), patch.leaf});
            });
        result_memory.bytes = 0;
        return result;
    }
};

ContactSourcePointAdmission DistributedAdaptiveIncidenceSession::source_point_admission(AMRGrid grid,
    ContactHaloContext context, int owner, std::uint64_t generation, unsigned layer, std::size_t count,
    std::shared_ptr<const CartesianIncidenceSource> source, std::shared_ptr<const CartesianSourceManifestSet> manifests,
    std::shared_ptr<const ContactSourcePointSnapshot> previous)
{
    return ContactSourcePointAdmission(std::move(grid), context, owner, generation, layer, count, std::move(source),
        std::move(manifests), std::move(previous));
}
CartesianSourceManifestAdmission DistributedAdaptiveIncidenceSession::source_manifest_admission(AMRGrid grid,
    ContactHaloContext context, std::uint64_t generation, std::size_t cells,
    std::shared_ptr<const CartesianSourceInventory> inventory, const OwnedSupportMembership* owned)
{
    return CartesianSourceManifestAdmission(std::move(grid), context, generation, cells, std::move(inventory), owned);
}
PooledLineCompletionSnapshotAdmission DistributedAdaptiveIncidenceSession::source_result_admission(
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot, const IncidencePools& pools)
{
    if (!snapshot || snapshot != pools.points.source() || !snapshot->slot_namespace() || !snapshot->generation() ||
        snapshot->size() != pools.points.size())
        throw std::invalid_argument("Resident result retirement requires the actual complete point publication.");
    return PooledLineCompletionSnapshotAdmission(std::move(snapshot), pools);
}
CartesianLineRangeAdmission DistributedAdaptiveIncidenceSession::source_line_admission(
    std::shared_ptr<const CartesianSourceManifestSet> manifests,
    std::shared_ptr<const ContactSourcePointSnapshot> points, ContactHaloContext context, std::uint64_t generation,
    unsigned layer)
{
    return CartesianLineRangeAdmission(std::move(manifests), std::move(points), context, generation, layer);
}

DistributedAdaptiveIncidenceSession::DistributedAdaptiveIncidenceSession(const AdaptiveProviderKernel& kernel,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveClosureOptions options, ContactHaloContext context,
    Budget& budget, IdentityBudget& identities, std::shared_ptr<Budget> keeper,
    std::shared_ptr<ContactPhysicalResidency> physical)
{
    if (comm.is_null())
        throw std::invalid_argument("Incremental incidence requires a communicator.");
    if ((keeper && keeper.get() != &budget) || (physical && !keeper))
        throw std::invalid_argument("Foreign or unbound incidence arena keeper.");
    impl_ = std::make_unique<Impl>(kernel, std::move(comm), options, context, budget, identities,
        std::move(keeper), std::move(physical));
}
DistributedAdaptiveIncidenceSession::DistributedAdaptiveIncidenceSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}
DistributedAdaptiveIncidenceSession::~DistributedAdaptiveIncidenceSession() = default;
DistributedAdaptiveIncidenceSession::DistributedAdaptiveIncidenceSession(
    DistributedAdaptiveIncidenceSession&&) noexcept = default;
DistributedAdaptiveIncidenceSession& DistributedAdaptiveIncidenceSession::operator=(
    DistributedAdaptiveIncidenceSession&&) noexcept = default;
std::unique_ptr<DistributedAdaptiveIncidenceSession> DistributedAdaptiveIncidenceSession::fork(
    Budget& budget, IdentityBudget& identities, std::shared_ptr<Budget> keeper, std::size_t* source_overlap,
    std::shared_ptr<ContactPhysicalResidency> physical) const
{
    if ((keeper && keeper.get() != &budget) || (physical && !keeper))
        throw std::invalid_argument("Foreign or unbound incidence fork arena keeper.");
    std::unique_ptr<Impl> candidate;
    collective(*impl_->comm,
        [&]
        {
            if (!impl_->ready)
                throw std::invalid_argument("Cannot fork incomplete incidence state.");
            if (&budget == impl_->budget)
                throw std::invalid_argument("Incidence fork requires a distinct arena.");
            if (impl_->external_physical_residency != bool(physical))
                throw std::invalid_argument("Incidence fork needs its exact shared residency policy.");
            (void) impl_->tile_bodies.reap();
            impl_->reap_deferred();
            candidate = std::make_unique<Impl>(*impl_, budget, identities, std::move(keeper), source_overlap, std::move(physical));
        });
    return std::unique_ptr<DistributedAdaptiveIncidenceSession>(
        new DistributedAdaptiveIncidenceSession(std::move(candidate)));
}
std::vector<IncidenceRowDelivery> DistributedAdaptiveIncidenceSession::bootstrap(
    std::span<const AdaptiveLeaf> leaves, const OwnedLeafLookup& raw_lookup, bool keep_identity)
{
    auto& state = *impl_;
    struct Observation
    {
        Impl& state;
        ~Observation() { state.diagnostic("return"); }
    } observation{state};
    MaterializedLookup materialized(*state.kernel, raw_lookup, *state.budget);
    OwnedLeafLookup lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf& { return materialized.get(key); };
    state.reap_deferred();
    struct ReapGuard
    {
        Impl& state;
        ~ReapGuard() { state.reap_deferred(); }
    } reaper{state};
    Lease scratch(*state.budget);
    Impl::ChangedTiles changed;
    TileMap::Batch tile_batch(state.tiles);
    state.phase("owned_membership");
    collective(*state.comm,
        [&]
        {
            if (state.ready)
                throw std::invalid_argument("Repeated incidence bootstrap.");
            (void) state.tile_bodies.reap();
            state.owner_rows.begin_delta();
            const auto grid = trusted_cartesian_incidence_grid(*state.kernel);
            if (grid && !leaves.empty() && state.keeper)
            {
                const auto& domain = leaves.front().domain;
                if (std::ranges::all_of(leaves, [&](const auto& leaf)
                        { return CartesianIncidenceSource::eligible(leaf) && leaf.domain == domain; }))
                {
                    std::fenv_t caller{};
                    if (!std::fegetenv(&caller))
                    {
                        try
                        {
                            auto source = CartesianIncidenceSource::make(*grid, leaves, domain, state.comm->getRank(),
                                1, state.context, *state.budget, state.keeper);
                            if (std::fesetenv(&caller))
                                throw std::invalid_argument("Cannot restore source-admission caller environment.");
                            materialized.bind_source(std::move(source));
                        }
                        catch (const CartesianSourceAuthenticationFailure&)
                        {
                            if (std::fesetenv(&caller))
                                throw std::invalid_argument("Cannot restore explicit source fallback environment.");
                        }
                        catch (...)
                        {
                            (void) std::fesetenv(&caller);
                            throw;
                        }
                    }
                }
            }
            for (const auto& raw : leaves)
            {
                const auto& leaf = lookup(raw.key);
                const auto slot =
                    materialized.native_proof() ? state.owned.add(*materialized.native_proof()) : state.owned.add(leaf);
                state.owner_rows.insert(slot);
                ++state.statistics.bootstrap_owned_visits;
            }
        });
    int local_source = state.keeper && trusted_cartesian_incidence_grid(*state.kernel) &&
                       (leaves.empty() || state.owned.has_complete_construction_proof());
    int complete_source = 0;
    Teuchos::reduceAll(*state.comm, Teuchos::REDUCE_MIN, 1, &local_source, &complete_source);
    state.native_source_protocol = complete_source;
    state.native_inventory_complete = complete_source;
    std::shared_ptr<const CartesianSourceInventory> initial_inventory;
    if (state.native_source_protocol)
    {
        const auto grid = trusted_cartesian_incidence_grid(*state.kernel);
        collective(*state.comm,
            [&]
            {
                if (!grid)
                    throw std::invalid_argument("Native source closure lost its trusted provider capability.");
                if (!state.point_arena)
                    state.point_arena = state.make_point_arena();
                std::array<CartesianSourceInventory::CapturedSource, 1> captured{};
                std::size_t count = 0;
                if (state.owned.construction_source())
                    captured[count++] = {
                        state.owned.construction_source(), state.owned.construction_evaluation_state()};
                initial_inventory = CartesianSourceInventory::make(state.context, state.comm->getRank(), 1,
                    std::span(captured).first(count), *state.budget, state.keeper, state.point_arena);
            });
        const auto admission =
            source_manifest_admission(*grid, state.context, 1, leaves.size(), initial_inventory, &state.owned);
        state.source_manifests =
            CartesianSourceManifestSet::exchange(admission, state.comm, *state.budget, state.keeper, state.point_arena);
    }
    std::size_t cell = 0, surface = 0;
    const auto next = [&](RegistryChange& output)
    {
        while (cell < leaves.size())
        {
            const auto slot = state.owned.require(leaves[cell].key, leaves[cell].revision);
            if (surface < state.owned.member_count(slot))
            {
                output = state.owned.change(slot, surface++, state.comm->getRank(), false);
                return true;
            }
            ++cell;
            surface = 0;
        }
        return false;
    };
    const auto page = state.page_bytes();
    state.phase("support_registry");
    std::size_t cursor_capacity = 0;
    collective(*state.comm, [&] { cursor_capacity = state.owned.maximum_change_capacity(); });
    exchange_support_registry_stream(
        state.supports, next, state.context, state.comm, page, *state.budget, state.owned.maximum_change_capacity());
    materialized.clear();
    state.phase("support_fetch");
    state.fetch_supports(lookup, changed, scratch, tile_batch);
    materialized.clear();
    if (state.native_source_protocol)
        collective(*state.comm,
            [&]
            {
                // The closed owner inventory is independent of the authority's
                // request list. A whole omitted support cannot hide by simply
                // never requesting its members or delivering a group receipt.
                for (const auto& leaf : leaves)
                {
                    const auto cell = state.owned.require(leaf.key, leaf.revision);
                    state.owned.visit_members(cell,
                        [&](std::size_t, const auto& member)
                        {
                            if (!state.resident_generations.contains(member.support))
                                throw std::invalid_argument(
                                    "Missing complete native support receipt from owner inventory.");
                        });
                }
            });
    state.phase("line_replay");
    state.replay(changed, scratch, true);
    state.phase("owner_rows");
    auto rows = state.finish_rows(changed, {}, {}, raw_lookup, 0, scratch, keep_identity);
    ReturnedRowsCredit rows_credit{*state.budget, &rows};
    state.phase("source_membership_certification");
    collective(*state.comm,
        [&]
        {
            if (initial_inventory)
                state.owned.publish_inventory(initial_inventory, state.point_arena);
            else
                (void) state.owned.publish_constructed_source();
        });
    state.phase("point_source_certification");
    collective(*state.comm,
        [&]
        {
            state.compact_point_values(rows);
            state.finish_subfaces();
        });
    state.ready = true;
    state.phase("ready");
    rows_credit.transfer();
    return rows;
}
IncidenceDelta DistributedAdaptiveIncidenceSession::apply_delta(std::span<const IncidenceCellVersion> retired,
    std::span<const AdaptiveCellKey> inserted, ContactHaloContext next, const OwnedLeafLookup& raw_lookup, bool keep_identity)
{
    auto& state = *impl_;
    struct Observation
    {
        Impl& state;
        ~Observation() { state.diagnostic("return"); }
    } observation{state};
    state.phase("delta_membership");
    MaterializedLookup materialized(*state.kernel, raw_lookup, *state.budget);
    OwnedLeafLookup lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf& { return materialized.get(key); };
    state.reap_deferred();
    struct ReapGuard
    {
        Impl& state;
        ~ReapGuard() { state.reap_deferred(); }
    } reaper{state};
    Lease scratch(*state.budget);
    Impl::ChangedTiles changed;
    TileMap::Batch tile_batch(state.tiles);
    std::set<Slot> retired_supports;
    cartesian_detail::Allocation source_delta_memory(*state.budget);
    std::vector<CartesianSourceInventory::Retirement> source_retired;
    std::shared_ptr<const CartesianSourceInventory> next_inventory;
    std::shared_ptr<const CartesianIncidenceSource> inserted_source;
    std::fenv_t inserted_environment{};
    collective(*state.comm,
        [&]
        {
            if (!state.ready || next.transaction < state.context.transaction ||
                (next.transaction == state.context.transaction && next.round <= state.context.round))
                throw std::invalid_argument("Invalid incidence delta context.");
            (void) state.tile_bodies.reap();
            state.owner_rows.begin_delta();
            if (state.owned.inventory() &&
                state.owned.inventory()->version() == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Native inventory version exhausted.");
            if (state.owned.inventory())
                source_delta_memory.reserve(source_retired, retired.size());
            for (const auto& old : retired)
            {
                const auto slot = state.owned.require(old.key, old.leaf_revision);
                if (state.owned.inventory() && state.owned.inventory()->contains(old.key))
                    source_retired.push_back({old.key, old.leaf_revision});
                state.owned.visit_members(slot,
                    [&](std::size_t, const auto& member)
                    {
                        if (!retired_supports.contains(member.support))
                        {
                            scratch.add(slot_set_node);
                            retired_supports.insert(member.support);
                        }
                    });
                ++state.statistics.delta_owned_visits;
            }
            state.context = next;
        });
    const unsigned old_last = state.last_layer;
    std::size_t cell = 0, surface = 0;
    const auto retire_next = [&](RegistryChange& output)
    {
        while (cell < retired.size())
        {
            const auto slot = state.owned.require(retired[cell].key, retired[cell].leaf_revision);
            if (surface < state.owned.member_count(slot))
            {
                output = state.owned.change(slot, surface++, state.comm->getRank(), true);
                return true;
            }
            ++cell;
            surface = 0;
        }
        return false;
    };
    auto page = state.page_bytes();
    std::size_t cursor_capacity = 0;
    collective(*state.comm, [&] { cursor_capacity = state.owned.maximum_change_capacity(); });
    exchange_support_registry_stream(
        state.supports, retire_next, state.context, state.comm, page, *state.budget, cursor_capacity);
    collective(*state.comm,
        [&]
        {
            for (const auto& old : retired)
                state.owned.erase(state.owned.require(old.key, old.leaf_revision));
            if (state.owned.inventory() && !inserted.empty())
            {
                const auto grid = trusted_cartesian_incidence_grid(*state.kernel);
                const auto& first = raw_lookup(inserted.front());
                bool eligible = bool(grid) && CartesianIncidenceSource::eligible(first);
                for (const auto& key : inserted)
                {
                    const auto& raw = raw_lookup(key);
                    eligible = eligible && raw.key == key && CartesianIncidenceSource::eligible(raw) &&
                               raw.domain == first.domain;
                }
                if (eligible)
                {
                    cartesian_detail::Allocation metadata(*state.budget);
                    std::vector<AdaptiveLeaf> native;
                    metadata.reserve(native, inserted.size());
                    for (const auto& key : inserted)
                    {
                        const auto& raw = raw_lookup(key);
                        metadata.retain(cartesian_detail::add(raw.domain.size(), raw.parent_family.size()));
                        native.push_back(raw);
                    }
                    if (std::fegetenv(&inserted_environment))
                        throw std::invalid_argument("Cannot capture inserted native source evaluation.");
                    try
                    {
                        inserted_source =
                            CartesianIncidenceSource::make(*grid, native, first.domain, state.comm->getRank(),
                                state.owned.inventory()->version() + 1, state.context, *state.budget, state.keeper);
                    }
                    catch (const CartesianSourceAuthenticationFailure&)
                    {
                        if (std::fesetenv(&inserted_environment))
                            throw std::invalid_argument(
                                "Cannot restore explicit inserted-source fallback environment.");
                    }
                    catch (...)
                    {
                        (void) std::fesetenv(&inserted_environment);
                        throw;
                    }
                    if (std::fesetenv(&inserted_environment))
                        throw std::invalid_argument("Cannot restore inserted native source caller environment.");
                    if (inserted_source)
                        materialized.bind_source(inserted_source);
                }
            }
            for (const auto& key : inserted)
            {
                const auto& leaf = lookup(key);
                if (materialized.native_proof())
                    (void) state.owned.add(*materialized.native_proof());
                else
                    (void) state.owned.add(leaf);
                ++state.statistics.delta_owned_visits;
            }
            for (const auto support : retired_supports)
                if (!state.owned.has_support(support))
                    state.replace_group(support, {}, changed, scratch, tile_batch, nullptr, lookup);
        });
    materialized.clear();
    cell = surface = 0;
    const auto insert_next = [&](RegistryChange& output)
    {
        while (cell < inserted.size())
        {
            const auto slot = *state.pools.cells.find(inserted[cell]);
            if (surface < state.owned.member_count(slot))
            {
                output = state.owned.change(slot, surface++, state.comm->getRank(), false);
                return true;
            }
            ++cell;
            surface = 0;
        }
        return false;
    };
    page = state.page_bytes();
    collective(*state.comm, [&] { cursor_capacity = state.owned.maximum_change_capacity(); });
    exchange_support_registry_stream(
        state.supports, insert_next, state.context, state.comm, page, *state.budget, cursor_capacity);
    if (state.owned.inventory())
    {
        collective(*state.comm,
            [&]
            {
                if (state.owned.inventory()->version() == std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument("Native inventory version exhausted.");
                std::array<CartesianSourceInventory::CapturedSource, 1> added{};
                std::size_t count = 0;
                if (inserted_source)
                    added[count++] = {inserted_source, inserted_environment};
                next_inventory = state.owned.inventory()->derive(state.context, state.owned.inventory()->version() + 1,
                    source_retired, std::span(added).first(count), *state.budget, state.keeper, state.point_arena);
            });
        const auto grid = trusted_cartesian_incidence_grid(*state.kernel);
        const auto admission = source_manifest_admission(*grid, state.context, next_inventory->version(),
            next_inventory->cell_count(), next_inventory, &state.owned);
        state.source_manifests =
            CartesianSourceManifestSet::exchange(admission, state.comm, *state.budget, state.keeper, state.point_arena);
        int local_complete =
            state.owned.explicit_cell_count() == inserted.size() && (inserted.empty() || bool(inserted_source));
        int complete = 0;
        Teuchos::reduceAll(*state.comm, Teuchos::REDUCE_MIN, 1, &local_complete, &complete);
        state.native_inventory_complete = complete;
    }
    state.fetch_supports(lookup, changed, scratch, tile_batch);
    materialized.clear();
    state.replay(changed, scratch, false);
    // This lease precedes result: copied retired keys and both vector buffers
    // are physically destroyed before any failed return admission is released.
    Lease retired_memory(*state.budget);
    IncidenceDelta result;
    std::vector<Impl::DirtySubface> dirty_subfaces;
    std::vector<TileId> touched;
    result.changed = state.finish_rows(
        changed, retired, inserted, raw_lookup, old_last, scratch, keep_identity, &dirty_subfaces, &touched);
    ReturnedRowsCredit changed_credit{*state.budget, &result.changed};
    collective(*state.comm,
        [&]
        {
            if (next_inventory)
                state.owned.publish_inventory(next_inventory, state.point_arena);
            state.compact_point_values(result.changed, std::span<const TileId>(touched));
            state.finish_subfaces(dirty_subfaces);
        });
    collective(*state.comm,
        [&]
        {
            retired_memory.add(retired.size() * sizeof(IncidenceCellVersion));
            result.retired.reserve(retired.size());
            for (const auto& old : retired)
            {
                retired_memory.add(key_capacity(old.key));
                result.retired.push_back(old);
            }
        });
    changed_credit.transfer();
    retired_memory.bytes = 0;
    return result;
}
const std::shared_ptr<const AdaptiveSubfaceSnapshot>& DistributedAdaptiveIncidenceSession::subface_snapshot() const noexcept
{
    return impl_->subfaces;
}
std::size_t DistributedAdaptiveIncidenceSession::inherited_overlap_metadata_bytes() const noexcept
{
    const auto& state = *impl_;
    const auto owned_registry = state.physical_residency && !state.external_physical_residency
                                    ? state.physical_residency->overlap_bytes() : 0;
    return state.retired_inherited_metadata_bytes + state.authority_tile_counts.overlap_bytes() +
           state.resident_generations.overlap_bytes() + state.owner_rows.inherited_overlap_metadata_bytes() +
           state.pools.overlap_bytes() + state.supports.overlap_bytes() +
           state.owned.inherited_overlap_metadata_bytes() + owned_registry;
}
void DistributedAdaptiveIncidenceSession::finish_inherited_overlap() noexcept
{
    impl_->finish_inherited_overlap();
}
const ContactHaloStatistics& DistributedAdaptiveIncidenceSession::statistics() const noexcept
{
    auto& state = *impl_;
    const auto lines = state.lines.active_storage();
    const auto tiles = state.tile_bodies.statistics();
    state.statistics.source_owner_manifests = state.source_manifests ? state.source_manifests->owner_count() : 0;
    state.statistics.source_membership_cells = state.owned.source_cell_count();
    state.statistics.source_coordinate_points = state.pools.points.source_size();
    state.statistics.explicit_coordinate_points = state.pools.points.physical_pool().size();
    state.statistics.source_tile_records = tiles.source_records;
    state.statistics.explicit_tile_ring_slots = tiles.ring_size;
    state.statistics.source_line_records = lines.source_records;
    state.statistics.source_line_nonforward_records = lines.source_nonforward_records;
    state.statistics.source_line_families = lines.source_families;
    state.statistics.source_axis_maps = lines.source_maps;
    state.statistics.source_result_selectors = lines.source_result_selectors;
    state.statistics.explicit_line_records = lines.explicit_records;
    state.statistics.stride_line_records = lines.run_records;
    state.statistics.retained_explicit_line_records = lines.retained_explicit_records;
    state.statistics.retained_stride_line_records = lines.retained_run_records;
    state.statistics.retained_source_line_records = lines.retained_source_records;
    return impl_->statistics;
}
std::size_t DistributedAdaptiveIncidenceSession::metadata_bytes() const noexcept
{
    const auto& state = *impl_;
    impl_->reap_deferred();
    // Persistent pool and owner-row roots and LayerHistory owners import their cached
    // allowance on fork, including the source paths held through overlap.
    // TileBodyPages still has separate per-arena credit objects.
    const auto source_page_credits = state.tile_bodies.statistics().page_credit_overhead_bytes;
    const auto source_arena_metadata = state.source_arena_overlap_metadata();
    return sizeof(Impl) + state.pools.source_only_metadata_bytes() +
           state.owned.source_overlap_metadata_bytes() + state.supports.source_only_metadata_bytes() +
           state.lines.metadata_bytes() +
           state.owner_rows.source_overlap_metadata_bytes() +
           state.borrowed.metadata_bytes() +
           state.tile_bodies.metadata_bytes() + source_page_credits + source_arena_metadata +
           state.layer_source_leases.capacity() * sizeof(decltype(state.layer_source_leases)::value_type);
}
bool DistributedAdaptiveIncidenceSession::owns_point_page_credit(const void* identity) const noexcept
{
    return impl_->pools.points.owns_page_credit(identity) ||
           (impl_->point_arena && impl_->point_arena->owns_credit(identity) && !impl_->exports_source_credit(identity));
}
bool DistributedAdaptiveIncidenceSession::admits_source_metadata_owner(
    const std::shared_ptr<const void>& owner, std::size_t bytes) const noexcept
{
    return (impl_->physical_residency &&
               impl_->physical_residency->admits(ContactPhysicalResidency::Role::CreditMetadata, owner, bytes)) ||
           impl_->borrowed.admits(owner, bytes) ||
           (impl_->point_arena && impl_->point_arena->admits_owner(owner, bytes));
}
void DistributedAdaptiveIncidenceSession::for_each_borrowed_source_admission(
    const std::function<void(const void*, std::size_t)>& visitor) const
{
    impl_->borrowed.visit_live_owners(visitor);
    if (impl_->physical_residency)
        impl_->physical_residency->usage().visit_entries([&](auto key, const auto& entry)
            {
                if (entry.origin != impl_->budget)
                    visitor(key.allocation, entry.bytes);
            });
    if (impl_->point_arena)
        for (const auto& [owner, credit] : impl_->point_arena->entries)
            if (!owner.expired() && credit->charged_payload)
                visitor(credit->owner_identity, credit->payload);
}
void DistributedAdaptiveIncidenceSession::for_each_borrowed_overlap_credit(
    const std::function<void(const void*, std::size_t, bool)>& visitor) const
{
    if (impl_->point_arena)
        for (const auto& [owner, credit] : impl_->point_arena->entries)
            visitor(credit.get(), credit->metadata, impl_->exports_source_credit(credit.get()));
}
std::size_t DistributedAdaptiveIncidenceSession::source_arena_overlap_metadata_bytes() const noexcept
{
    return impl_->source_arena_overlap_metadata();
}
void DistributedAdaptiveIncidenceSession::for_each_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor, bool include_object) const
{
    if (include_object)
        contact_payload_object(visitor, this, ContactPayloadRole::IncidenceSessionObject, sizeof(*this));
    const auto& state = *impl_;
    contact_payload_object(visitor, impl_.get(), ContactPayloadRole::IncidenceSessionImpl, sizeof(Impl));
    state.pools.cells.visit_borrowed_payload(visitor);
    state.pools.domains.visit_borrowed_payload(visitor);
    state.pools.supports.visit_borrowed_payload(visitor);
    state.pools.lines.visit_borrowed_payload(visitor);
    state.pools.points.visit_borrowed_payload(visitor);
    state.pools.planes.visit_borrowed_payload(visitor);
    state.pools.normals.visit_borrowed_payload(visitor);
    state.pools.images.visit_borrowed_payload(visitor);
    state.owned.visit_borrowed_payload(visitor);
    state.supports.visit_borrowed_payload(visitor);
    state.owner_rows.visit_borrowed_payload(visitor);
    state.tiles.visit_borrowed_payload(visitor);
    if (state.subfaces)
        state.subfaces->visit_borrowed_payload(visitor);
    state.borrowed.visit_borrowed_payload(visitor);
    contact_payload_buffer(
        visitor, impl_.get(), ContactPayloadRole::IncidenceSessionLayerSourceLeases, state.layer_source_leases);
    for (const auto& lease : state.layer_source_leases)
        lease->visit_borrowed_payload(visitor);
    state.authority_tile_counts.visit_borrowed_payload(visitor, ContactPayloadRole::IncidenceAuthorityCountNodes);
    state.resident_generations.visit_borrowed_payload(visitor, ContactPayloadRole::IncidenceResidentGenerationNodes);
    for (std::size_t i = 0; i != state.tiles.size(); ++i)
    {
        const auto& tile = state.tiles.value_at(i);
        if (!tile.history)
            continue;
        contact_payload_object(
            visitor, tile.history.get(), ContactPayloadRole::IncidenceLayerHistoryObject, sizeof(LayerHistory), true);
        contact_payload_buffer(
            visitor, tile.history.get(), ContactPayloadRole::IncidenceLayerHistoryEntries, tile.history->layers);
        for (const auto& entry : tile.history->layers)
            if (entry.second)
            {
                const auto& layer = *entry.second;
                contact_payload_object(visitor, &layer, ContactPayloadRole::IncidenceLayerObject,
                    state.native_source_protocol ? sizeof(SavedLayer) : sizeof(Layer), true);
                contact_payload_buffer(visitor, &layer, ContactPayloadRole::IncidenceLayerInput, layer.input);
                contact_payload_buffer(visitor, &layer, ContactPayloadRole::IncidenceLayerOutput, layer.output);
                if (layer.final_output)
                    layer.final_output->visit_borrowed_payload(visitor);
                contact_payload_buffer(visitor, &layer, ContactPayloadRole::IncidenceLayerEdges, layer.edges);
                for (const auto& edge : layer.edges)
                    contact_payload_buffer(visitor, &layer, ContactPayloadRole::IncidenceLayerEdgeResults, edge.result);
                if (state.native_source_protocol)
                {
                    const auto& saved = static_cast<const SavedLayer&>(layer);
                    contact_payload_buffer(
                        visitor, &saved, ContactPayloadRole::IncidenceLayerEndpointFacts, saved.endpoint_facts);
                    contact_payload_buffer(visitor, &saved, ContactPayloadRole::IncidenceLayerLeases, saved.leases);
                    contact_payload_buffer(
                        visitor, &saved, ContactPayloadRole::IncidenceLayerCompletions, saved.completions);
                    for (const auto& completion : saved.completions)
                        if (completion)
                            completion->visit_borrowed_payload(visitor);
                    if (saved.saved_input)
                        saved.saved_input->visit_borrowed_payload(visitor);
                    if (saved.saved_output)
                        saved.saved_output->visit_borrowed_payload(visitor);
                    if (saved.saved_edges)
                        saved.saved_edges->visit_borrowed_payload(visitor);
                    for (const auto& lease : saved.leases)
                        lease->visit_borrowed_payload(visitor);
                }
            }
    }
    state.lines.visit_borrowed_payload(visitor);
    state.tile_bodies.visit_borrowed_payload(visitor);
    if (state.source_manifests)
        state.source_manifests->visit_borrowed_payload(visitor);
    if (state.point_arena)
        state.point_arena->visit_borrowed_payload(visitor);
}
void DistributedAdaptiveIncidenceSession::for_each_borrowed_geometry_ledger(
    const std::function<void(const PLIC::GeometryAllocationLedger&)>& visitor) const
{
    visitor(impl_->identities->borrowed_geometry_ledger());
}
void DistributedAdaptiveIncidenceSession::for_each_borrowed_plane_identity(
    const std::function<void(const PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>&)>& visitor) const
{
    for (std::size_t slot = 0; slot != impl_->pools.planes.size(); ++slot)
        visitor(impl_->pools.planes.get(slot));
}
cartesian_detail::ArenaCredits* DistributedAdaptiveIncidenceSession::source_credit_arena() const noexcept
{
    return impl_->point_arena.get();
}
const IncidencePointDirectory<>& DistributedAdaptiveIncidenceSession::point_pool() const noexcept
{
    return impl_->pools.points;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
