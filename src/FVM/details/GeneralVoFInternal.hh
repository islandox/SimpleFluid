/** @file GeneralVoFInternal.hh
 * @brief Private accepted-state layout and shared transport support declarations.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/details/DistributedAMROverlap.hh"
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/ExactPhaseAccumulator.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "FVM/details/ExperimentalPhaseBackend.hh"
#include "FVM/details/PhaseArithmetic.hh"
#include "FVM/details/VoFFailure.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/swept_volume/FiniteVoFTransport.hh"
#include "utils/CompensatedSum.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
namespace SimpleFluid::VoF
{
namespace Exact
{
struct SIMPLEFLUID_FVM_LOCAL GeneralVoF
{
    using State = VoF::GeneralVoF;
    using Data = State::Data;
    using Trial = State::Trial;
    using Key = State::Key;
    static Trial plan_transport(
        const State&, std::span<const GeneralVoFNodeVelocity>, std::span<const GeneralVoFFaceVolume>, double);
    static void validate_transfer_target(const State&, const ExactPhaseTransferResult&);
    static ExactPhaseTransferSource export_transfer_source(const State&, bool);
    static std::shared_ptr<const State> adopt_transfer(const State&, const ExactPhaseTransferResult&);
};
} // namespace Exact
namespace general_vof_detail
{
namespace AMR = Meshes::AMR;
namespace wire
{
using namespace AMR::adaptive_detail;
struct Budget : AMR::adaptive_detail::Budget
{
    explicit Budget(std::size_t limit) : AMR::adaptive_detail::Budget{limit} {}
    void add(std::size_t bytes)
    {
        if (bytes > limit - std::min(used, limit))
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "General VoF payload byte budget exhausted.");
        used += bytes;
    }
    void packet(std::size_t payload)
    {
        add(sizeof(OwnerPacket));
        add(payload);
    }
};
template<class Work> void collective(const Teuchos::Comm<int>& comm, Work&& action)
{
    ::SimpleFluid::detail::vof_failure::collective(comm, std::forward<Work>(action));
}
} // namespace wire
using Key = AMR::AdaptiveCellKey;
/** Canonical encoded size for construction preflight, without encoding storage. */
inline std::size_t topology_key_wire_bytes(const Key& key)
{
    wire::Budget bytes{std::numeric_limits<std::size_t>::max()};
    bytes.add(5 * sizeof(std::uint64_t));
    const auto path = [&](const auto& entries)
    {
        if (entries.size() > bytes.limit / sizeof(std::uint64_t))
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "General VoF key ancestry byte size overflows.");
        bytes.add(entries.size() * sizeof(std::uint64_t));
    };
    std::visit(
        [&](const auto& ancestry)
        {
            using T = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<T, AMR::CartesianCellAncestry>)
                bytes.add(4 * sizeof(std::uint64_t));
            else if constexpr (std::is_same_v<T, AMR::ConvexCellAncestry>)
            {
                bytes.add(2 * sizeof(std::uint64_t));
                path(ancestry.path);
            }
            else if constexpr (std::is_same_v<T, AMR::ProductCellAncestry>)
            {
                bytes.add(5 * sizeof(std::uint64_t));
                path(ancestry.base.path);
                path(ancestry.axial.path);
            }
            else
                bytes.add(2 * sizeof(std::uint64_t));
        },
        key.ancestry);
    return bytes.used;
}
/** Sorted immutable-key rows borrow one stable topology-owned intern pool. */
class KeySet
{
    using Storage = std::vector<const Key*>;

public:
    class Iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Key;
        using difference_type = std::ptrdiff_t;
        using pointer = const Key*;
        using reference = const Key&;
        Iterator() = default;
        explicit Iterator(Storage::const_iterator position) : position_(position) {}
        reference operator*() const { return **position_; }
        pointer operator->() const { return *position_; }
        Iterator& operator++()
        {
            ++position_;
            return *this;
        }
        Iterator operator++(int)
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const Iterator&) const = default;

    private:
        Storage::const_iterator position_;
    };
    [[nodiscard]] Iterator begin() const { return Iterator(keys_.begin()); }
    [[nodiscard]] Iterator end() const { return Iterator(keys_.end()); }
    [[nodiscard]] std::size_t size() const { return keys_.size(); }
    [[nodiscard]] bool empty() const { return keys_.empty(); }
    [[nodiscard]] std::size_t storage_bytes() const { return keys_.capacity() * sizeof(const Key*); }
    [[nodiscard]] bool contains(const Key& key) const
    {
        const auto found = lower_bound(key);
        return found != keys_.end() && **found == key;
    }
    std::pair<Iterator, bool> insert(const Key& key, std::set<Key>& pool, wire::Budget* budget = nullptr)
    {
        const auto position = lower_bound(key);
        if (position != keys_.end() && **position == key)
            return {Iterator(position), false};
        const auto index = position - keys_.begin();
        const auto pooled = pool.find(key);
        const auto capacity = keys_.capacity();
        if (keys_.size() == capacity && capacity > keys_.max_size() / 2)
            throw std::invalid_argument("General VoF key row capacity overflows.");
        const auto next = keys_.size() == capacity ? (capacity ? 2 * capacity : 1) : capacity;
        if (budget)
        {
            // Encoding bounds ancestry vector storage; retain the existing
            // independent encoded-key allowance as well as the pool allocation.
            if (pooled == pool.end())
            {
                const auto bytes = topology_key_wire_bytes(key);
                budget->add(sizeof(Key) + 4 * sizeof(void*));
                budget->add(bytes);
                budget->add(bytes);
            }
            // Charge both buffers during growth, while reserve retains the old one.
            if (next != capacity)
                budget->add(next * sizeof(const Key*));
        }
        const auto canonical = pooled == pool.end() ? pool.insert(key).first : pooled;
        if (next != capacity)
            keys_.reserve(next);
        const auto inserted = keys_.insert(keys_.begin() + index, &*canonical);
        return {Iterator(inserted), true};
    }

private:
    [[nodiscard]] Storage::const_iterator lower_bound(const Key& key) const
    {
        return std::lower_bound(
            keys_.begin(), keys_.end(), key, [](const Key* stored, const Key& sought) { return *stored < sought; });
    }
    Storage keys_;
};
using Measure = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
namespace fp = finite_detail;
struct LocalSpatialCache;
struct Cell
{
    Key key;
    std::uint64_t gid = 0, revision = 0;
    int owner = -1;
    SimpleFluid::PLIC::GeometryPoint centroid{};
    double native_volume = 0;
    std::size_t canonical_face_count = 0;
    long double ordinary_quota_scale = 0;
    PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary> phase;
    PLIC::DynamicConvexGeometryPtr finite_geometry;
    PhaseError occupancy;
    PhaseRoundoff roundoff, finite_history;
    std::shared_ptr<const ExperimentalPhaseInventory> experimental_exact;
    GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
    std::optional<PLIC::PolyhedralInterface> retained_interface;
    bool inherited = false;
    // Authenticated independently of the optional owner-only retained plane.
    bool requires_interface = false;
};
} // namespace general_vof_detail
using namespace general_vof_detail;
struct GeneralVoF::Data
{
    /** Immutable identity metadata from the already admitted native draft.
     * This catalog authorizes dependency lookups; it does not enlarge the
     * caller-visible owned-incident face or node controls. */
    using FaceAuthority = AMR::AdaptiveTile;
    struct Wall
    {
        std::variant<std::uint64_t, AMR::AdaptiveBytes> authority;
        SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Ordinary> geometry;
        PLIC::DynamicPoint finite_normal;
        PLIC::DynamicFaceConstraint finite_constraint;
    };
    struct Topology
    {
        enum class SurfaceTileOwner
        {
            OwnedTiles,
            ExtraSurfaceTiles
        };
        struct SurfaceTileReference
        {
            SurfaceTileOwner owner;
            std::size_t index;
        };
        AMR::AdaptiveNativePublication native;
        AMR::AdaptiveNativeDraft draft;
        std::vector<AMR::AdaptiveTile> tiles, extra_surface_tiles;
        // Draft order, borrowing owned-incident tiles and owning only the
        // additional full-halo support. Indices survive either owner's growth.
        std::vector<SurfaceTileReference> surface_tile_references, frontier_tile_references;
        [[nodiscard]] const AMR::AdaptiveTile& surface_tile(const SurfaceTileReference& reference) const
        {
            switch (reference.owner)
            {
                case SurfaceTileOwner::OwnedTiles:
                    if (reference.index < tiles.size())
                        return tiles[reference.index];
                    break;
                case SurfaceTileOwner::ExtraSurfaceTiles:
                    if (reference.index < extra_surface_tiles.size())
                        return extra_surface_tiles[reference.index];
                    break;
            }
            throw std::invalid_argument("General VoF surface tile owner or index is invalid.");
        }
        void validate_surface_tiles() const
        {
            for (const auto& reference : surface_tile_references)
                (void) surface_tile(reference);
            for (const auto& reference : frontier_tile_references)
                (void) surface_tile(reference);
            for (const auto& [id, reference] : construction_faces)
                (void) construction_face(id);
            for (const auto& [node, walls] : node_walls)
                for (const auto& wall : walls)
                    if (const auto* id = std::get_if<std::uint64_t>(&wall.authority))
                    {
                        const auto& face = construction_face(*id);
                        if (!face.exterior || face.geometry != wall.geometry)
                            throw std::invalid_argument("General VoF wall authority differs from its native face.");
                    }
        }
        [[nodiscard]] const FaceAuthority& construction_face(std::uint64_t id) const
        {
            const auto& tile = surface_tile(construction_faces.at(id));
            if (tile.face_gid != id)
                throw std::invalid_argument("General VoF construction face identity differs from its catalog key.");
            return tile;
        }
        [[nodiscard]] const AMR::AdaptiveBytes& wall_identity(const Wall& wall) const
        {
            if (const auto* id = std::get_if<std::uint64_t>(&wall.authority))
                return construction_face(*id).identity;
            return std::get<AMR::AdaptiveBytes>(wall.authority);
        }
        class SurfaceTiles
        {
        public:
            struct Iterator
            {
                const Topology* topology;
                const std::vector<SurfaceTileReference>* references;
                std::size_t index;
                const AMR::AdaptiveTile& operator*() const { return topology->surface_tile(references->at(index)); }
                Iterator& operator++()
                {
                    ++index;
                    return *this;
                }
                bool operator==(const Iterator&) const = default;
            };
            SurfaceTiles(const Topology& topology, const std::vector<SurfaceTileReference>& references)
                : topology_(&topology), references_(&references)
            {
            }
            [[nodiscard]] Iterator begin() const { return {topology_, references_, 0}; }
            [[nodiscard]] Iterator end() const { return {topology_, references_, size()}; }
            [[nodiscard]] std::size_t size() const { return references_->size(); }
            [[nodiscard]] bool empty() const { return size() == 0; }

        private:
            const Topology* topology_;
            const std::vector<SurfaceTileReference>* references_;
        };
        [[nodiscard]] SurfaceTiles surface_tiles() const { return SurfaceTiles(*this, surface_tile_references); }
        // Full-shell boundary evidence survives moving draft into the native publication.
        [[nodiscard]] SurfaceTiles frontier_tiles() const { return SurfaceTiles(*this, frontier_tile_references); }
        std::vector<AMR::AdaptiveNode> nodes;
        std::map<std::uint64_t, PLIC::DynamicPoint> finite_nodes;
        std::map<std::uint64_t, SurfaceTileReference> construction_faces;
        std::map<std::uint64_t, AMR::AdaptiveBytes> construction_nodes;
        std::map<std::uint64_t, Key> by_gid;
        // Declared before the borrowed rows so it also outlives their destruction.
        std::set<Key> graph_keys;
        std::map<Key, KeySet> adjacency;
        std::map<std::uint64_t, KeySet> node_fans;
        std::map<std::uint64_t, std::vector<Wall>> node_walls;
        std::vector<Key> order;
        std::map<Key, std::vector<std::size_t>> cell_tiles;
        std::map<std::uint64_t, std::vector<std::size_t>> node_tiles;
        // Geometry-only cache shared by accepted transport publications. A cache
        // never retains this topology or a phase-bearing Data/Cell object.
        mutable std::mutex spatial_cache_mutex;
        mutable std::shared_ptr<const LocalSpatialCache> spatial_cache;
    };
    std::shared_ptr<const Topology> topology;
    std::shared_ptr<const Hierarchy> hierarchy;
    AMR::HierarchyObservation observation;
    std::shared_ptr<const Hierarchy> supplied_hierarchy;
    AMR::HierarchyObservation supplied_observation;
    GeneralVoFOptions options;
    GeneralVoFPhaseCleanupState phase_cleanup;
    std::vector<GeneralVoFPhaseCleanupDelta> latest_phase_cleanup;
    std::uint64_t control = 0;
    std::weak_ptr<const void> adopted_source;
    std::uint64_t adopted_source_control = 0, adopted_transaction = 0, adopted_target_control = 0;
    std::map<Key, Cell> cells;
    std::vector<double> gas, carrier;
    std::vector<PhaseVolumes> phases;
    std::vector<PhaseRoundoff> roundoff, phase_history;
    std::vector<PhaseError> occupancy;
    std::shared_ptr<const std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>>>
        experimental_catalog;
};
namespace general_vof_detail
{
SIMPLEFLUID_FVM_LOCAL PhaseInventoryPolicy inventory_policy_for(ArithmeticMode mode);
SIMPLEFLUID_FVM_LOCAL PLIC::GeometryLimits geometry_limits(ArithmeticMode mode, std::size_t bytes = 64 * 1024 * 1024);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL long double phase_allowance(PLIC::DynamicGeometryValue source, double relative);
SIMPLEFLUID_FVM_LOCAL SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phase_measures(
    const Cell& cell);
SIMPLEFLUID_FVM_LOCAL Measure finite_bound(double value);
template<ArithmeticMode Mode> SIMPLEFLUID_FVM_LOCAL void occupancy(Cell& cell);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL void validate_cell(const Cell& cell, const GeneralVoFOptions& options);
SIMPLEFLUID_FVM_LOCAL void real(wire::Writer& w, double value);
SIMPLEFLUID_FVM_LOCAL double real(wire::Reader& r);
SIMPLEFLUID_FVM_LOCAL void finite_real(wire::Writer& w, long double value);
SIMPLEFLUID_FVM_LOCAL long double finite_real(wire::Reader& r);
SIMPLEFLUID_FVM_LOCAL void finite_value(wire::Writer& w, PLIC::DynamicGeometryValue v);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicGeometryValue finite_value(
    wire::Reader& r, std::optional<ArithmeticMode> expected = {});
SIMPLEFLUID_FVM_LOCAL void plane(wire::Writer& w, const PLIC::PolyhedralInterface& p);
SIMPLEFLUID_FVM_LOCAL PLIC::PolyhedralInterface plane(wire::Reader& r, std::optional<ArithmeticMode> mode = {});
SIMPLEFLUID_FVM_LOCAL SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phases(
    const SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>& fragment,
    const PLIC::PolyhedralInterface& p);
SIMPLEFLUID_FVM_LOCAL GeneralVoFPhaseEnclosure diagnostic(
    const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& measures);
SIMPLEFLUID_FVM_LOCAL double output(const Measure& measure);
SIMPLEFLUID_FVM_LOCAL SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> geometric_labels(
    const Cell& cell);
SIMPLEFLUID_FVM_LOCAL void refresh_experimental_catalog(GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL void current(const GeneralVoF::Data& data);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL void retained_budget(const GeneralVoF::Data& data,
    std::optional<std::size_t> proposed_spatial_bytes = {}, std::size_t extra_retained_bytes = 0,
    std::size_t* remaining_bytes = nullptr);
// Every incident owner registers its actual cell once. The node authority
// authenticates those registrations and returns the complete canonical fan;
// geometric coincidence and whichever ghost rows happen to be resident play no
// part in defining the interpolation stencil.
SIMPLEFLUID_FVM_LOCAL void complete_node_fans(GeneralVoF::Data& data, GeneralVoF::Data::Topology& topology);
SIMPLEFLUID_FVM_LOCAL std::uint64_t transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm);
SIMPLEFLUID_FVM_LOCAL void agree_transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm, std::uint64_t id);
SIMPLEFLUID_FVM_LOCAL void agree_options(const GeneralVoFOptions& options, Teuchos::RCP<const Teuchos::Comm<int>> comm);
template<ArithmeticMode Mode> SIMPLEFLUID_FVM_LOCAL void import_phases(GeneralVoF::Data& data);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL PLIC::PolyhedralPhaseSample<Mode> reconstruction_sample(const Cell& cell);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL std::vector<PLIC::PolyhedralPhaseSample<Mode>> reconstruction_stencil(
    const GeneralVoF::Data& data, const Key& key, bool complete_support = false, std::size_t* fitting_cells = nullptr);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL PLIC::PolyhedralInterface reconstruct_cell(const GeneralVoF::Data& data, const Key& key);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data);
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> import_planes(const GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> import_requested_planes(const GeneralVoF::Data& data,
    const std::set<Key>& requested, const PLIC::GeometryPacketContext& context, geometry::swept_volume::Work& work,
    GeneralVoFOwnerPlaneRequestDiagnostics& diagnostics, GeneralVoFWorkSample* balance_work = nullptr);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> import_planes(const GeneralVoF::Data& data);
} // namespace general_vof_detail
} // namespace SimpleFluid::VoF
