/** @file DistributedAdaptiveIncidence.cc
 * @brief Incremental complete-support and layered exact-line incidence. */
#include "DistributedAdaptiveIncidence.hh"
#include "CompactTileIndex.hh"
#include "DeferredOwnerCharges.hh"
#include "IncidenceOwnerRows.hh"
#include "IncidencePools.hh"
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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t owner_bytes = 128;
using Slot = std::size_t;
using TileId = std::pair<Slot, std::uint64_t>;
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
    // Line slots, then physical ring-vertex slots, then receipt bit words.
    // The two identity regions belong to their distinct typed pools; the tail
    // is bookkeeping and never establishes an identity.
    std::vector<Slot> lines_and_receipts;
    std::shared_ptr<Layer> body;
    bool require_all = false;
};
using TileMap = CompactTileIndex<TileEntry>;
using ChangedTileMap = std::map<TileId, std::optional<TileEntry>>;
using LayerMap = std::map<TileId, LayerWork>;
using ExpectedSupportMap = std::map<Slot, std::vector<RegistryMember>>;
constexpr std::size_t changed_map_node = sizeof(ChangedTileMap::value_type) + 4 * sizeof(void*);
constexpr std::size_t layer_map_node = sizeof(LayerMap::value_type) + 4 * sizeof(void*);
constexpr std::size_t expected_support_node = sizeof(ExpectedSupportMap::value_type) + 4 * sizeof(void*);
constexpr std::size_t tile_set_node = sizeof(TileId) + 4 * sizeof(void*);
constexpr std::size_t slot_set_node = sizeof(Slot) + 4 * sizeof(void*);
constexpr std::size_t pointer_set_node = sizeof(const void*) + 4 * sizeof(void*);
constexpr std::size_t slot_count_node = sizeof(std::pair<const Slot, std::size_t>) + 4 * sizeof(void*);
constexpr std::size_t slot_generation_node = sizeof(std::pair<const Slot, std::uint64_t>) + 4 * sizeof(void*);
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
TileBodyLoopView final_loop(const TileEntry& state, unsigned layer)
{
    const auto* value = at_layer(state, layer);
    return value ? TileBodyLoopView(value->output) : state.body.intrinsic_loop();
}
std::size_t layer_bytes(const Layer& layer)
{
    std::size_t result = sizeof(Layer) + owner_bytes +
                         (layer.input.capacity() + layer.output.capacity()) * sizeof(Slot) +
                         layer.edges.capacity() * sizeof(EdgeState);
    for (const auto& edge : layer.edges)
        result += edge.result.capacity() * sizeof(Slot);
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
    std::size_t charge_ = 0;

public:
    MaterializedLookup(const AdaptiveProviderKernel& k, const OwnedLeafLookup& l, Budget& b)
        : kernel_(k), lookup_(l), budget_(b)
    {
    }
    ~MaterializedLookup() { clear(); }
    void clear()
    {
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
        if (value_ && value_->key == key && value_->revision == leaf.revision)
            return *value_;
        clear();
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
    IncidencePools pools;
    TileBodyPages tile_bodies;
    OwnedSupportMembership owned;
    PooledSupportRegistry supports;
    PooledLineRegistry lines;
    IncidenceOwnerRows owner_rows;
    TileMap tiles;
    std::map<Slot, std::size_t> authority_tile_counts;
    std::map<Slot, std::uint64_t> resident_generations;
    ContactHaloStatistics statistics;
    unsigned last_layer = 0;
    std::size_t periodic_tiles = 0;
    std::size_t authority_tiles = 0;
    std::size_t observed_work_capacity_peak = 0, observed_layer_capacity_peak = 0;
    bool ready = false;
    const char* diagnostic_phase = "construction";

    Impl(const AdaptiveProviderKernel& k, Teuchos::RCP<const Teuchos::Comm<int>> c, AdaptiveClosureOptions o,
        ContactHaloContext ctx, Budget& b, IdentityBudget& i, std::shared_ptr<Budget> arena)
        : lease(b), borrowed(b), kernel(&k), comm(std::move(c)), options(o), context(ctx), budget(&b), identities(&i),
          keeper(std::move(arena)), pools(b, i, keeper), tile_bodies(b, keeper), owned(b, pools),
          supports(b, pools.supports, pools.cells), lines(b, pools),
          owner_rows(b, i, pools, k, comm->getRank(), keeper), tiles(b, keeper)
    {
        lease.add(sizeof(Impl));
    }
    Impl(const Impl& source, Budget& b, IdentityBudget& i, std::shared_ptr<Budget> arena)
        : lease(b), borrowed(b), kernel(source.kernel), comm(source.comm), options(source.options),
          context(source.context), budget(&b), identities(&i), keeper(std::move(arena)),
          pools(source.pools, b, i, keeper), tile_bodies(source.tile_bodies, b, keeper), owned(source.owned, b, pools),
          supports(source.supports, b, pools.supports, pools.cells), lines(source.lines, b, pools),
          owner_rows(source.owner_rows, b, i, pools, keeper), tiles(b, keeper), statistics(source.statistics),
          last_layer(source.last_layer), periodic_tiles(source.periodic_tiles), authority_tiles(source.authority_tiles),
          ready(source.ready)
    {
        lease.add(sizeof(Impl));
        Lease scratch(b);
        std::set<const Layer*> layers;
        std::set<const LayerHistory*> histories;
        tiles.clone_from(source.tiles,
            [&](const TileEntry& tile)
            {
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
                        lease.add(layer_bytes(*body));
                    }
                return TileEntry{tile_bodies.adopt(tile.body), tile.history, tile.initialized};
            });
        lease.add(source.authority_tile_counts.size() * slot_count_node +
                  source.resident_generations.size() * slot_generation_node);
        authority_tile_counts = source.authority_tile_counts;
        resident_generations = source.resident_generations;
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
                    layer_owners += layer_bytes(*value);
                    for (const auto& edge : value->edges)
                        edge_replies += edge.result.size();
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
        std::fprintf(stderr,
            "ADAPTIVE_INCIDENCE_TILE_BODIES rank=%d phase=%s pages=%zu wide_pages=%zu occupied_records=%zu "
            "live_records=%zu record_capacity=%zu ring_size=%zu ring_capacity=%zu ring_bytes=%zu "
            "directory_metadata_bytes=%zu page_body_bytes=%zu page_credit_overhead_bytes=%zu "
            "deferred_pages=%zu deferred_body_bytes=%zu deferred_metadata_bytes=%zu\n",
            comm->getRank(), diagnostic_phase, bodies.pages, bodies.wide_pages, bodies.occupied_records,
            bodies.live_records, bodies.record_capacity, bodies.ring_size, bodies.ring_capacity, bodies.ring_bytes,
            bodies.metadata_bytes, bodies.page_body_bytes, bodies.page_credit_overhead_bytes, bodies.deferred_pages,
            bodies.deferred_body_bytes, bodies.deferred_metadata_bytes);
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
            pending += work.lines_and_receipts.capacity() * sizeof(Slot);
            if (work.body)
            {
                ++layers;
                bodies += layer_bytes(*work.body) - owner_bytes;
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
        if (value && value->budget != budget)
        {
            const auto bytes = layer_bytes(*value);
            if (borrowed.adopt(value, bytes))
                lease.bytes -= bytes;
        }
    }
    void defer_history(const std::shared_ptr<LayerHistory>& value)
    {
        if (!value)
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
    TileEntry new_tile(const ReceivedTileRecipe& recipe)
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
        return {tile_bodies.append(draft), {}, false};
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
    std::shared_ptr<Layer> new_layer(TileBodyLoopView input)
    {
        const auto bytes = sizeof(Layer) + owner_bytes + input.size() * sizeof(Slot);
        budget->add(bytes);
        auto layer = std::make_shared<Layer>();
        layer->budget = budget;
        layer->keeper = keeper;
        layer->charge = bytes;
        layer->input.assign(input.begin(), input.end());
        return layer;
    }
    std::shared_ptr<Layer> clone_layer(const Layer& source)
    {
        const auto bytes = layer_bytes(source);
        budget->add(bytes);
        auto layer = std::make_shared<Layer>();
        layer->budget = budget;
        layer->keeper = keeper;
        layer->charge = bytes;
        layer->input = source.input;
        layer->output = source.output;
        layer->edges = source.edges;
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
    template<class Emit> void visit_edges(const TileBodyScalars& tile, TileBodyLoopView loop, Emit&& emit)
    {
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
        if (count > (std::numeric_limits<std::size_t>::max() - words) / 2 ||
            2 * count + words > budget->limit / sizeof(Slot))
            throw std::invalid_argument("Incidence receipt capacity overflow.");
        scratch.add((2 * count + words) * sizeof(Slot));
        LayerWork result;
        result.input = std::move(input);
        result.input_owner = std::move(input_owner);
        result.require_all = require_all;
        result.lines_and_receipts.resize(2 * count + words, 0);
        if (result.lines_and_receipts.capacity() != 2 * count + words)
            throw std::invalid_argument("Incidence receipt capacity differs from admission.");
        return result;
    }
    void register_edge(const TileBodyScalars& tile, LayerWork& work, Slot line, const PooledLineSubscriber& sub)
    {
        const auto count = edge_count(tile, work.input.size());
        const auto edge = sub.image * work.input.size() + sub.edge;
        work.lines_and_receipts.at(edge) = line;
        work.lines_and_receipts.at(count + edge) = sub.first;
    }
    std::pair<Slot, Slot> physical_endpoints(const TileBodyScalars& tile, const LayerWork& work, std::size_t edge) const
    {
        const auto count = edge_count(tile, work.input.size());
        const auto next = edge - edge % work.input.size() + (edge + 1) % work.input.size();
        return {work.lines_and_receipts[count + edge], work.lines_and_receipts[count + next]};
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
    void materialize_endpoints(const TileBodyScalars& tile, LayerWork& work)
    {
        auto body = new_layer(work.input);
        reserve_owned(body->output, work.input.size(), body->charge);
        body->output.assign(work.input.begin(), work.input.end());
        const auto count = edge_count(tile, work.input.size());
        reserve_owned(body->edges, count, body->charge);
        for (std::size_t edge = 0; edge < count; ++edge)
        {
            body->edges.push_back({work.lines_and_receipts[edge], {}});
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
        constexpr auto bits = std::numeric_limits<Slot>::digits;
        auto& word = work.lines_and_receipts[2 * count + edge / bits];
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
            if (!(work.lines_and_receipts[2 * count + edge / bits] & (Slot(1) << (edge % bits))))
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
        auto value = new_layer(input);
        reserve_owned(value->output, vertices.size(), value->charge);
        for (const auto& point : vertices)
            value->output.push_back(pools.points.intern(point));
        // Empty edges encode only an ordered, certified endpoint-query baseline.
        work.body = std::move(value);
    }
    void rebuild_output(const TileBodyScalars& tile, Layer& layer)
    {
        std::size_t count = 0;
        for (const auto& edge : layer.edges)
            count += edge.result.size();
        Lease scratch(*budget);
        scratch.add(
            PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(layer.input.size()) +
            PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(layer.input.size() + count) +
            (3 * layer.input.size() + 4 * count) * sizeof(IncidencePools::Point) + count * 128);
        const auto face = restore_tile(tile, layer.input, pools);
        std::vector<IncidencePools::Point> points;
        points.reserve(count);
        const auto order =
            legacy_edge_order(layer.edges.size(), [&](Slot edge) { return layer.edges[edge].line; }, scratch);
        for (const auto e : order)
        {
            const bool image = e >= layer.input.size();
            for (const auto point : layer.edges[e].result)
                points.push_back(
                    image ? PLIC::point_translated<ArithmeticMode::Ordinary>(pools.points.get(point), tile_image(tile))
                          : pools.points.get(point));
        }
        const auto enriched = face->with_edge_vertices(points);
        const auto vertices = enriched->vertex_identities();
        reserve_owned(layer.output, vertices.size(), layer.charge);
        layer.output.clear();
        for (const auto& vertex : vertices)
            layer.output.push_back(pools.points.intern(vertex));
    }
    bool intrinsic_fixed_layer(const TileEntry& state, const Layer& layer) const
    {
        if (layer.input != state.body.intrinsic_loop() || layer.output != layer.input)
            return false;
        const auto size = layer.input.size();
        if (layer.edges.empty())
            return true;
        if (layer.edges.size() != size * (tile_image(state.tile()).is_zero() ? 1 : 2))
            return false;
        for (std::size_t edge = 0; edge < layer.edges.size(); ++edge)
        {
            const auto& result = layer.edges[edge].result;
            if (result.size() != 2)
                return false;
            auto first = pools.points.get(layer.input[edge % size]);
            auto last = pools.points.get(layer.input[(edge + 1) % size]);
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
        TileMap::Batch& batch)
    {
        batch.replace_support(support);
        for (auto i = tiles.lower_bound({support, 0}); i < tiles.size() && tiles.key_at(i).first == support; ++i)
        {
            const auto id = tiles.key_at(i);
            const auto& tile = tiles.value_at(i);
            remember(before, id, scratch);
            defer_history(tile.history);
            periodic_tiles -= !tile_image(tile.tile()).is_zero();
        }
        for (const auto& recipe : recipes)
        {
            auto state = new_tile(recipe);
            periodic_tiles += !recipe.second_image.is_zero();
            const TileId id{support, recipe.emission};
            remember(before, id, scratch);
            batch.insert(id, std::move(state));
        }
    }
    void fetch_supports(
        const OwnedLeafLookup& lookup, ChangedTiles& changed, Lease& scratch, TileMap::Batch& tile_batch)
    {
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
                const auto* cells = owned.support_cells(*slot);
                if (!cells)
                    throw std::invalid_argument("Foreign support fetch recipient.");
                for (const auto cell : *cells)
                    for (std::size_t ordinal = 0; ordinal < owned.at(cell).members.size(); ++ordinal)
                    {
                        const auto& member = owned.at(cell).members[ordinal];
                        if (member.support == *slot)
                        {
                            page->append(members);
                            page->add(key_capacity(pools.cells.get(cell)));
                            members.push_back({pools.cells.get(cell), owned.at(cell).revision, ordinal,
                                member.encoded_size, comm->getRank()});
                        }
                    }
            },
            [&](std::span<const char> key, std::span<const RegistryMember> members,
                std::span<const SupportOccurrence> occurrences)
            {
                ++statistics.dirty_supports;
                statistics.dirty_support_members += members.size();
                Lease key_scratch(*budget);
                key_scratch.add(key.size());
                const auto support_slot = *pools.supports.find(AdaptiveBytes(key.begin(), key.end()));
                auto count = authority_tile_counts.find(support_slot);
                if (count == authority_tile_counts.end())
                {
                    lease.add(slot_count_node);
                    count = authority_tile_counts.emplace(support_slot, 0).first;
                }
                authority_tiles -= count->second;
                count->second = 0;
                if (members.empty())
                    return;
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
                    ++count->second;
                    page->append(source.recipes);
                    face_charge(recipe.face);
                    source.recipes.push_back(std::move(recipe));
                };
                make_support_recipes(
                    key, occurrences, budget->limit - budget->used, options.tile_budget, charge, face_charge, emit);
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
                            if (generation != resident_generations.end() &&
                                generation->second == std::numeric_limits<std::uint64_t>::max())
                                throw std::invalid_argument("Resident support generation exhausted.");
                            incoming.push_back({pools.supports.get(support),
                                generation == resident_generations.end() ? 0 : generation->second + 1, members});
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
                        if (!resident_generations.contains(*support))
                            lease.add(slot_generation_node);
                        resident_generations[*support] = generation;
                        replace_group(*support, recipes, changed, scratch, tile_batch);
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
        collective(*comm, [&] { tile_batch.publish(options.tile_budget); });
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
                            visit_edges(old_state->tile(),
                                old_layer ? TileBodyLoopView(old_layer->input) : old_state->body.intrinsic_loop(),
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
                    const auto input =
                        previous_layer ? TileBodyLoopView(previous_layer->output) : current->body.intrinsic_loop();
                    auto next = make_work(current->tile(), input, true, iteration, std::move(previous_owner));
                    visit_edges(current->tile(), input,
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
            const auto page = page_bytes();
            phase("line_exchange");
            exchange_pooled_line_edits(lines, pools, next_edit, layer, context, comm, *budget, page,
                [&](Slot line, const PooledLineSubscriber& sub)
                {
                    ++statistics.dirty_subscriptions;
                    const TileId id{sub.support, sub.emission};
                    const auto current = tiles.find(id);
                    if (current == nullptr || tile_authority(current->tile()) != sub.support_authority)
                        throw std::invalid_argument("Foreign incremental edge reply.");
                    auto work = working.find(id);
                    if (work == working.end())
                    {
                        auto previous_owner = at_layer_owner(*current, layer);
                        const auto* previous = previous_owner.get();
                        if (!previous && !current->initialized)
                            throw std::invalid_argument("Missing cached edge subscription.");
                        const auto input =
                            previous ? TileBodyLoopView(previous->input) : current->body.intrinsic_loop();
                        auto next = make_work(current->tile(), input, false, iteration, std::move(previous_owner));
                        const auto count = edge_count(current->tile(), input.size());
                        visit_edges(current->tile(), input, [&](Slot expected, const auto& edge)
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
                    auto& value = work->second;
                    receive_edge(current->tile(), value, line, sub);
                    const auto edge = sub.image * value.input.size() + sub.edge;
                    if (!value.body && endpoint_reply(current->tile(), value, edge, sub.result))
                        return;
                    if (!value.body)
                        materialize_endpoints(current->tile(), value);
                    auto& result = value.body->edges[edge].result;
                    reserve_owned(result, sub.result.size(), value.body->charge);
                    result.assign(sub.result.begin(), sub.result.end());
                });
            diagnostic_work("complete_replies", working, buffered);
            bool size_changed = false;
            phase("edge_reconstruction");
            std::set<TileId> following;
            collective(*comm,
                [&]
                {
                    for (auto& [id, work] : working)
                    {
                        auto& current = tiles.at(id);
                        certify_receipts(current.tile(), work);
                        if (work.body)
                        {
                            for (const auto& edge : work.body->edges)
                                if (edge.result.empty())
                                    throw std::invalid_argument("Missing cached edge reply.");
                            rebuild_output(current.tile(), *work.body);
                            size_changed |= work.body->output.size() != work.body->input.size();
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
                        ++statistics.recomputed_tiles;
                        const auto output = work.body ? TileBodyLoopView(work.body->output) : work.input;
                        size_changed |= output.size() != work.input.size();
                        const auto* previous = at_layer(current, layer);
                        const bool changed_output =
                            previous ? output != previous->output
                                     : (!current.initialized || output != current.body.intrinsic_loop());
                        remember(changed, id, scratch);
                        if (work.body || !current.layers().empty())
                            put_layer(current, layer, work.body, !bootstrap);
                        current.initialized = true;
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
                            auto& entry = (*found);
                            if (!entry.history)
                                continue;
                            const auto& previous = entry.layers();
                            const auto retained_end = std::upper_bound(previous.begin(), previous.end(), layer,
                                [](unsigned n, const auto& value) { return n < value.first; });
                            const bool intrinsic = std::all_of(previous.begin(), retained_end, [&](const auto& value)
                                { return !value.second || intrinsic_fixed_layer(entry, *value.second); });
                            if (intrinsic)
                            {
                                defer_history(entry.history);
                                entry.history.reset();
                            }
                            else if (retained_end != previous.end())
                            {
                                auto& history = edit_history(entry).layers;
                                const auto future = std::upper_bound(history.begin(), history.end(), layer,
                                    [](unsigned n, const auto& value) { return n < value.first; });
                                for (auto old = future; old != history.end(); ++old)
                                    defer_layer(old->second);
                                history.erase(future, history.end());
                            }
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
    std::vector<IncidenceRowHandle> finish_rows(ChangedTiles& changed, std::span<const IncidenceCellVersion> retired,
        std::span<const AdaptiveCellKey> inserted, const OwnedLeafLookup& lookup, unsigned old_last, Lease& scratch)
    {
        collective(*comm,
            [&]
            {
                for (const auto& [id, old] : changed)
                    if (old)
                        owner_rows.remove(old->tile(), final_loop(*old, old_last));
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
                    if (const auto current = tiles.find(id); current != nullptr)
                        owner_rows.add(current->tile(), final_loop(*current, last_layer));
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
        std::vector<IncidenceRowHandle> result;
        collective(*comm,
            [&]
            {
                auto patches = owner_rows.finish(lookup);
                budget->add(patches.size() * sizeof(IncidenceRowHandle));
                result.reserve(patches.size());
                for (auto& patch : patches)
                    result.push_back(std::move(patch.row));
                const auto bytes = patches.capacity() * sizeof(IncidenceOwnerRows::Patch);
                std::vector<IncidenceOwnerRows::Patch>().swap(patches);
                budget->used -= bytes;
            });
        return result;
    }
};

DistributedAdaptiveIncidenceSession::DistributedAdaptiveIncidenceSession(const AdaptiveProviderKernel& kernel,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveClosureOptions options, ContactHaloContext context,
    Budget& budget, IdentityBudget& identities, std::shared_ptr<Budget> keeper)
{
    if (comm.is_null())
        throw std::invalid_argument("Incremental incidence requires a communicator.");
    if (keeper && keeper.get() != &budget)
        throw std::invalid_argument("Foreign incidence arena keeper.");
    impl_ = std::make_unique<Impl>(kernel, std::move(comm), options, context, budget, identities, std::move(keeper));
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
    Budget& budget, IdentityBudget& identities, std::shared_ptr<Budget> keeper) const
{
    if (keeper && keeper.get() != &budget)
        throw std::invalid_argument("Foreign incidence fork arena keeper.");
    std::unique_ptr<Impl> candidate;
    collective(*impl_->comm,
        [&]
        {
            if (!impl_->ready)
                throw std::invalid_argument("Cannot fork incomplete incidence state.");
            if (&budget == impl_->budget)
                throw std::invalid_argument("Incidence fork requires a distinct arena.");
            (void) impl_->tile_bodies.reap();
            impl_->reap_deferred();
            candidate = std::make_unique<Impl>(*impl_, budget, identities, std::move(keeper));
        });
    return std::unique_ptr<DistributedAdaptiveIncidenceSession>(
        new DistributedAdaptiveIncidenceSession(std::move(candidate)));
}
std::vector<IncidenceRowHandle> DistributedAdaptiveIncidenceSession::bootstrap(
    std::span<const AdaptiveLeaf> leaves, const OwnedLeafLookup& raw_lookup)
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
            for (const auto& raw : leaves)
            {
                const auto& leaf = lookup(raw.key);
                const auto slot = state.owned.add(leaf);
                state.owner_rows.insert(slot);
                ++state.statistics.bootstrap_owned_visits;
            }
        });
    std::size_t cell = 0, surface = 0;
    const auto next = [&](RegistryChange& output)
    {
        while (cell < leaves.size())
        {
            const auto slot = state.owned.require(leaves[cell].key, leaves[cell].revision);
            if (surface < state.owned.at(slot).members.size())
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
    exchange_support_registry_stream(state.supports, next, state.context, state.comm, page, *state.budget);
    materialized.clear();
    state.phase("support_fetch");
    state.fetch_supports(lookup, changed, scratch, tile_batch);
    materialized.clear();
    state.phase("line_replay");
    state.replay(changed, scratch, true);
    state.phase("owner_rows");
    auto rows = state.finish_rows(changed, {}, {}, raw_lookup, 0, scratch);
    state.ready = true;
    state.phase("ready");
    return rows;
}
IncidenceDelta DistributedAdaptiveIncidenceSession::apply_delta(std::span<const IncidenceCellVersion> retired,
    std::span<const AdaptiveCellKey> inserted, ContactHaloContext next, const OwnedLeafLookup& raw_lookup)
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
    collective(*state.comm,
        [&]
        {
            if (!state.ready || next.transaction < state.context.transaction ||
                (next.transaction == state.context.transaction && next.round <= state.context.round))
                throw std::invalid_argument("Invalid incidence delta context.");
            (void) state.tile_bodies.reap();
            state.owner_rows.begin_delta();
            for (const auto& old : retired)
            {
                const auto slot = state.owned.require(old.key, old.leaf_revision);
                for (const auto& member : state.owned.at(slot).members)
                    if (!retired_supports.contains(member.support))
                    {
                        scratch.add(slot_set_node);
                        retired_supports.insert(member.support);
                    }
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
            if (surface < state.owned.at(slot).members.size())
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
    exchange_support_registry_stream(state.supports, retire_next, state.context, state.comm, page, *state.budget);
    collective(*state.comm,
        [&]
        {
            for (const auto& old : retired)
                state.owned.erase(state.owned.require(old.key, old.leaf_revision));
            for (const auto& key : inserted)
            {
                (void) state.owned.add(lookup(key));
                ++state.statistics.delta_owned_visits;
            }
            for (const auto support : retired_supports)
                if (!state.owned.support_cells(support))
                    state.replace_group(support, {}, changed, scratch, tile_batch);
        });
    materialized.clear();
    cell = surface = 0;
    const auto insert_next = [&](RegistryChange& output)
    {
        while (cell < inserted.size())
        {
            const auto slot = *state.pools.cells.find(inserted[cell]);
            if (surface < state.owned.at(slot).members.size())
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
    exchange_support_registry_stream(state.supports, insert_next, state.context, state.comm, page, *state.budget);
    state.fetch_supports(lookup, changed, scratch, tile_batch);
    materialized.clear();
    state.replay(changed, scratch, false);
    IncidenceDelta result;
    result.changed = state.finish_rows(changed, retired, inserted, raw_lookup, old_last, scratch);
    collective(*state.comm,
        [&]
        {
            state.budget->add(retired.size() * sizeof(IncidenceCellVersion));
            result.retired.reserve(retired.size());
            for (const auto& old : retired)
            {
                state.budget->add(key_capacity(old.key));
                result.retired.push_back(old);
            }
        });
    return result;
}
const ContactHaloStatistics& DistributedAdaptiveIncidenceSession::statistics() const noexcept
{
    return impl_->statistics;
}
std::size_t DistributedAdaptiveIncidenceSession::metadata_bytes() const noexcept
{
    const auto& state = *impl_;
    impl_->reap_deferred();
    // Forked page credits are distinct allocations in the two arenas. The
    // immutable LayerHistory objects/vectors are shared and already admitted
    // by the candidate constructor, so they do not enter source-only overlap.
    const auto source_page_credits = state.tile_bodies.statistics().page_credit_overhead_bytes;
    return sizeof(Impl) + state.pools.metadata_bytes() + state.pools.page_credit_overhead_bytes() +
           state.owned.metadata_bytes() + state.supports.metadata_bytes() + state.lines.metadata_bytes() +
           state.owner_rows.metadata_bytes() + state.tiles.metadata_bytes() +
           state.authority_tile_counts.size() * slot_count_node +
           state.resident_generations.size() * slot_generation_node + state.borrowed.metadata_bytes() +
           state.tile_bodies.metadata_bytes() + source_page_credits;
}
bool DistributedAdaptiveIncidenceSession::owns_point_page_credit(const void* identity) const noexcept
{
    return impl_->pools.points.owns_page_credit(identity);
}
const IdentityPool<ContactPointIdentity>& DistributedAdaptiveIncidenceSession::point_pool() const noexcept
{
    return impl_->pools.points;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
