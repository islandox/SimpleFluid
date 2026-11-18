/** @file AdaptiveGeometrySnapshot.cc
 * @brief Mesh-owned immutable sparse native and fallback source construction.
 */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.tcc"
#include "geometry/PLIC/details/CellCatalogDemandAccess.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotInternal.hh"
#include "geometry/mesh/AMR/AdaptivePreparedCartesianPages.tcc"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
{
PLIC::GeometryLimits geometry_limits(ArithmeticMode mode, std::size_t bytes)
{
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = mode;
    limits.byte_budget = bytes;
    return limits;
}
} // namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail
{
GeometryConstructionControls construction_controls()
{
    GeometryConstructionControls result;
    result.rounding = std::fegetround();
    if (result.rounding == -1)
        return result;
#if defined(__GLIBC__) && defined(__x86_64__)
    std::fenv_t environment{};
    if (std::fegetenv(&environment) != 0)
        return result;
    result.x87_control = environment.__control_word;
#if defined(__x86_64__)
    // MXCSR bits 0..5 are sticky exception status. Retain all control bits,
    // including rounding, exception masks, FTZ and DAZ.
    result.simd_control = environment.__mxcsr & ~std::uint32_t(0x3f);
#endif
    result.supported = true;
#endif
    return result;
}
std::size_t checked_sum(std::size_t first, std::size_t second)
{
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Native geometry storage sum overflows.");
    return first + second;
}
std::size_t checked_product(std::size_t count, std::size_t width)
{
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Native geometry storage count product overflows.");
    return count * width;
}
void charge_items(adaptive_detail::Budget& budget, std::size_t count, std::size_t width)
{
    if (budget.used > budget.limit || (width && count > (budget.limit - budget.used) / width))
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Native geometry storage exceeds its remaining byte budget.");
    budget.add(checked_product(count, width));
}
namespace
{
struct RecipeReader
{
    std::span<const char> bytes;
    template<class T> T take()
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (bytes.size() < sizeof(T))
            throw std::invalid_argument("Truncated native Cartesian geometry recipe.");
        T value;
        std::memcpy(&value, bytes.data(), sizeof(T));
        bytes = bytes.subspan(sizeof(T));
        return value;
    }
    void string()
    {
        const auto count = take<std::uint64_t>();
        if (count > bytes.size())
            throw std::invalid_argument("Native Cartesian recipe string exceeds its payload.");
        bytes = bytes.subspan(static_cast<std::size_t>(count));
    }
};
} // namespace
RecipeAxes::RecipeAxes(const Meshes::NativeFragmentDescriptor& recipe)
{
    if (recipe.kind != Meshes::NativeFragmentDescriptor::Kind::Cartesian || recipe.parts.size() != 4 ||
        std::any_of(recipe.parts.begin(), recipe.parts.end(), [](const auto& part) { return !part; }))
        throw std::invalid_argument("Native Cartesian geometry recipe has invalid parts.");
    RecipeReader topology{*recipe.parts[0]}, view{recipe.view};
    view.string();
    if (!view.bytes.empty())
        throw std::invalid_argument("Native Cartesian recipe view has trailing metadata.");
    std::uint64_t cells = 1;
    for (auto& count : counts)
    {
        count = topology.take<std::uint64_t>();
        if (!count || count >= std::numeric_limits<unsigned>::max() ||
            count > std::numeric_limits<std::uint64_t>::max() / cells)
            throw std::invalid_argument("Native Cartesian recipe extent overflows.");
        cells *= count;
    }
    for (unsigned axis = 0; axis < 3; ++axis)
        if (topology.take<std::uint8_t>() != 0)
            throw std::invalid_argument("Native Cartesian local recipe has periodic topology.");
    for (unsigned face = 0; face < 6; ++face)
        topology.string();
    if (!topology.bytes.empty())
        throw std::invalid_argument("Native Cartesian topology has trailing bytes.");
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        RecipeReader coordinates{*recipe.parts[axis + 1]};
        const auto count = coordinates.take<std::uint64_t>();
        if (count != counts[axis] + 1 || count > coordinates.bytes.size() / sizeof(double) ||
            count * sizeof(double) != coordinates.bytes.size())
            throw std::invalid_argument("Native Cartesian axis differs from its topology.");
        values[axis] = coordinates.bytes;
        double previous = 0;
        for (std::uint64_t i = 0; i < count; ++i)
        {
            const auto value = coordinates.take<double>();
            if (!std::isfinite(value) || (i && !(value > previous)))
                throw std::invalid_argument("Native Cartesian source axis is not finite and increasing.");
            previous = value;
        }
    }
}
double RecipeAxes::value(unsigned axis, std::uint64_t index) const
{
    if (index > counts[axis])
        throw std::invalid_argument("Native Cartesian source ordinal exceeds its original recipe.");
    RecipeReader reader{values[axis].subspan(static_cast<std::size_t>(index) * sizeof(double))};
    return reader.take<double>();
}
std::array<std::uint64_t, 3> RecipeAxes::location(std::uint64_t ordinal) const
{
    std::array<std::uint64_t, 3> location{};
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        location[axis] = ordinal % counts[axis];
        ordinal /= counts[axis];
    }
    if (ordinal)
        throw std::invalid_argument("Native Cartesian cell ordinal exceeds recipe extent.");
    return location;
}
std::array<std::uint64_t, 3> RecipeAxes::location(const AdaptiveLeaf& leaf) const
{
    const auto* ancestry = std::get_if<CartesianCellAncestry>(&leaf.key.ancestry);
    if (!ancestry)
        throw std::invalid_argument("Native Cartesian geometry lacks Cartesian ancestry.");
    const auto indices = location(leaf.native_cell);
    for (unsigned axis = 0; axis < 3; ++axis)
        if (indices[axis] != ancestry->index[axis] % counts[axis])
            throw std::invalid_argument("Native Cartesian ordinal differs from persistent typed ancestry.");
    return indices;
}
AdaptiveCellMetrics RecipeAxes::metrics(const AdaptiveLeaf& leaf) const
{
    const auto indices = location(leaf);
    PLIC::GeometryPoint lower{}, upper{}, center{};
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        lower[axis] = value(axis, indices[axis]);
        upper[axis] = value(axis, indices[axis] + 1);
        center[axis] = std::midpoint(lower[axis], upper[axis]);
    }
    return {(upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]), center};
}
bool eligible(const DistributedAdaptiveHierarchy& hierarchy, const AdaptiveLeaf& leaf, ArithmeticMode mode)
{
    // These records come only from the hierarchy's validated owned/full native
    // leaf boundary. The public provider enum alone is never an authority.
    return hierarchy.has_native_cartesian_geometry() && mode == ArithmeticMode::Ordinary && !leaf.geometry &&
           leaf.native_recipe && leaf.native_recipe->kind == Meshes::NativeFragmentDescriptor::Kind::Cartesian &&
           leaf.provider == AdaptiveProviderKind::Cartesian &&
           std::holds_alternative<CartesianCellAncestry>(leaf.key.ancestry);
}
bool same_native_recipe(const AdaptiveLeaf& first, const AdaptiveLeaf& second)
{
    if (first.key != second.key || first.revision != second.revision || first.material != second.material ||
        first.provider != second.provider || first.domain != second.domain || first.native_cell != second.native_cell ||
        !first.native_recipe || !second.native_recipe)
        return false;
    const auto& a = *first.native_recipe;
    const auto& b = *second.native_recipe;
    if (a.kind != b.kind || a.view != b.view || a.parts.size() != b.parts.size())
        return false;
    for (std::size_t index = 0; index < a.parts.size(); ++index)
        if (!a.parts[index] || !b.parts[index] || *a.parts[index] != *b.parts[index])
            return false;
    return true;
}
const AdaptiveLeaf* exact_owned_cartesian_source(
    const DistributedAdaptiveHierarchy& hierarchy, const AdaptiveLeaf& leaf)
{
    const auto* owned = hierarchy.leaf_layout()->find(leaf.key);
    if (!owned || (owned != &leaf && (owned->geometry != leaf.geometry || !same_native_recipe(*owned, leaf) ||
                                         !GeometrySnapshotBinding::same_auxiliary_recipes(*owned, leaf))))
        return nullptr;
    return owned;
}
} // namespace SimpleFluid::Meshes::AMR::geometry_snapshot_detail

namespace SimpleFluid::Meshes::AMR
{
GeometryConstructionControls geometry_construction_controls()
{
    return geometry_snapshot_detail::construction_controls();
}
namespace
{

bool requires_surface_binding(const AdaptiveLeaf& leaf)
{
    return leaf.join_recipe && leaf.geometry && leaf.provider == AdaptiveProviderKind::Polyhedral;
}
bool same_surface(const AdaptiveSurface& first, const AdaptiveSurface& second)
{
    return first.key == second.key && first.geometry == second.geometry && first.orientation == second.orientation &&
           first.exterior == second.exterior && first.boundary == second.boundary &&
           first.native_face == second.native_face && first.authority_image == second.authority_image &&
           first.require_native_whole_tile == second.require_native_whole_tile &&
           first.native_whole_geometry == second.native_whole_geometry;
}
bool same_surface_binding(const AdaptiveLeaf& first, const AdaptiveLeaf& second)
{
    if (!requires_surface_binding(first) && !requires_surface_binding(second))
        return true;
    return first.surfaces.size() == second.surfaces.size() &&
           std::equal(first.surfaces.begin(), first.surfaces.end(), second.surfaces.begin(), same_surface);
}
bool same_capsule(const std::shared_ptr<const AdaptiveBytes>& first, const std::shared_ptr<const AdaptiveBytes>& second)
{
    return first == second || (first && second && *first == *second);
}
bool same_factor_recipe(
    const std::shared_ptr<const AdaptiveFactorRecipe>& first, const std::shared_ptr<const AdaptiveFactorRecipe>& second)
{
    return first == second || (first && second && same_capsule(first->root, second->root) &&
                                  same_capsule(first->base, second->base) && same_capsule(first->axis, second->axis));
}
} // namespace
bool GeometrySnapshotBinding::same_auxiliary_capsules(const AdaptiveLeaf& first, const AdaptiveLeaf& second)
{
    return same_capsule(first.join_recipe, second.join_recipe) &&
           same_factor_recipe(first.factor_recipe, second.factor_recipe);
}
bool GeometrySnapshotBinding::same_auxiliary_recipes(const AdaptiveLeaf& first, const AdaptiveLeaf& second)
{
    return same_auxiliary_capsules(first, second) && same_surface_binding(first, second);
}
#include "geometry/mesh/AMR/AdaptiveGeometryBindings.tcc"

std::size_t AdaptiveGeometrySnapshot::NativeMetrics::size() const noexcept
{
    return snapshot_->purpose() == GeometrySnapshotPurpose::State ? snapshot_->bindings_->size() : 0;
}
AdaptiveCellMetrics AdaptiveGeometrySnapshot::NativeMetrics::at(std::size_t slot) const
{
    if (slot >= size())
        throw std::out_of_range("Invalid native geometry metric slot.");
    return snapshot_->bindings_->metric(slot);
}
GeometryEvaluationPolicy AdaptiveGeometrySnapshot::evaluation_policy(std::size_t slot) const
{
    return bindings_->policy(slot);
}
std::optional<std::size_t> AdaptiveGeometrySnapshot::find(const AdaptiveLeaf& leaf) const
{
    return bindings_->find(leaf);
}
bool AdaptiveGeometrySnapshot::prepare_cartesian_page(const DistributedAdaptiveHierarchy& hierarchy,
    const AdaptiveLeaf& requested, std::size_t& used, std::size_t operation_limit, std::size_t optional_headroom,
    std::uint64_t block_first, PLIC::GeometryAllocationLedger& owners,
    geometry_snapshot_detail::OptionalOwnerReservation publication)
{
    using namespace geometry_snapshot_detail;
    using Access = PLIC::details::NativeCellGeometrySourceAccess;
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    constexpr auto page_size = prepared_cartesian_page_size;
    if (mode_ != ordinary || purpose_ != GeometrySnapshotPurpose::State || !stable_controls_ ||
        construction_controls() != controls_ || !matches_layout(hierarchy.leaf_layout()) ||
        !eligible(hierarchy, requested, ordinary))
        return false;
    // Optional own-index receipts are actual new allocations, so their
    // reservation never needs an imported-index compatibility census.
    const auto* anchor = exact_owned_cartesian_source(hierarchy, requested);
    if (!anchor || anchor->native_recipe != requested.native_recipe ||
        anchor->native_recipe.owner_before(requested.native_recipe) ||
        requested.native_recipe.owner_before(anchor->native_recipe))
        return false;
    // The demand binding already pins this exact recipe owner. An equal
    // descriptor with a distinct owner remains eligible for demand geometry.
    const auto anchor_slot = find(*anchor);
    if (!anchor_slot || !PLIC::CellGeometryView<ordinary>(ordinary_, *anchor_slot).is_procedural())
        return false;
    const auto first = block_first == ~std::uint64_t{0} ? requested.native_cell / page_size * page_size : block_first;
    if (prepared_cartesian_ && prepared_cartesian_->find(anchor->native_recipe, first))
        return false;
    const auto pool_bound = prepared_cartesian_
                                ? prepared_cartesian_->insertion_byte_bound()
                                : geometry_snapshot_detail::PreparedCartesianPages{}.insertion_byte_bound();
    const auto graph_edges = prepared_cartesian_
                                 ? prepared_cartesian_->insertion_dependency_edge_bound()
                                 : geometry_snapshot_detail::PreparedCartesianPages{}.insertion_dependency_edge_bound();
    const auto graph_owners =
        prepared_cartesian_ ? prepared_cartesian_->insertion_dependency_owner_bound()
                            : geometry_snapshot_detail::PreparedCartesianPages{}.insertion_dependency_owner_bound();
    const auto enough_room = [&](std::size_t cells, std::size_t points)
    {
        auto available = std::min(optional_headroom, operation_limit - std::min(used, operation_limit));
        const auto new_owners = PLIC::details::geometry_demand::checked_sum(graph_owners, points);
        const auto frontier = PLIC::details::geometry_demand::checked_sum(
            payload_owners_[0].metadata_owner_count_bound(),
            PreparedOwnerManifest::metadata_owner_count_bound(
                PLIC::details::geometry_demand::checked_sum(payload_owners_[0].owner_count(), new_owners),
                payload_owners_[0].typed_owner_count()));
        const auto reserved = publication.bytes ? publication.bytes(publication.context, new_owners, frontier) : 0;
        if (reserved > available)
            return false;
        available -= reserved;
        // The optional prepared index creates ordinary procedural sources.
        // Its own owner edit cannot introduce typed dependencies.
        const auto graph_bound = PLIC::details::GeometryOwnerGraphEditor::edit_byte_bound(
            payload_owners_[0], graph_edges + points, graph_owners + points);
        return pool_bound <= available && graph_bound <= available - pool_bound &&
               Access::regular_page_byte_bound(cells, points) <= available - pool_bound - graph_bound;
    };
    if (!enough_room(2, 6))
        return false;

    // Optional work is isolated from the caller's sticky floating-point flags.
    struct RestoreEnvironment
    {
        std::fenv_t environment;
        bool captured = std::fegetenv(&environment) == 0;
        ~RestoreEnvironment()
        {
            if (captured)
                (void) std::fesetenv(&environment);
        }
    } environment;
    if (!environment.captured)
        return false;
    const RecipeAxes axes(*anchor->native_recipe);
    const auto& origin = std::get<CartesianCellAncestry>(anchor->key.ancestry);
    const auto anchor_location = axes.location(*anchor);
    const auto cell_count = axes.counts[0] * axes.counts[1] * axes.counts[2];
    if (first >= cell_count)
        return false;
    struct PreparedRow
    {
        std::array<std::uint64_t, 3> location;
        AdaptiveCartesianGeometryBounds bounds;
    };
    std::array<PreparedRow, page_size> rows;
    std::size_t count = 0;
    std::uint64_t mask = 0;
    for (std::size_t offset = 0; offset < std::min<std::uint64_t>(page_size, cell_count - first); ++offset)
    {
        const auto ordinal = first + offset;
        const auto location = axes.location(ordinal);
        auto ancestry = origin;
        for (std::size_t axis = 0; axis < 3; ++axis)
            ancestry.index[axis] = origin.index[axis] - anchor_location[axis] + location[axis];
        const auto* owned = hierarchy.leaf_layout()->find({anchor->key.region, ancestry});
        if (!owned || owned->native_recipe != anchor->native_recipe || owned->native_cell != ordinal ||
            !eligible(hierarchy, *owned, ordinary))
            continue;
        const auto bounds = hierarchy.native_cartesian_geometry_bounds(*owned);
        if (construction_controls() != controls_)
            return false;
        if (!bounds ||
            !Access::state_compatible(bounds->lower, bounds->upper, geometry_limits(ordinary, optional_headroom)))
            continue;
        rows[count++] = {location, *bounds};
        mask |= std::uint64_t(1) << offset;
    }
    if (count < 2 || construction_controls() != controls_)
        return false;

    struct Coordinate
    {
        std::uint64_t index;
        double value;
    };
    std::array<std::array<Coordinate, 2 * page_size>, 3> coordinates;
    std::array<std::size_t, 3> sizes{};
    std::size_t points = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        auto values = std::span(coordinates[axis]).first(2 * count);
        for (std::size_t row = 0; row < count; ++row)
        {
            const auto& item = rows[row];
            values[2 * row] = {item.location[axis], item.bounds.lower[axis]};
            values[2 * row + 1] = {item.location[axis] + 1, item.bounds.upper[axis]};
        }
        const auto less = [](const auto& a, const auto& b) { return a.index < b.index; };
        std::sort(values.begin(), values.end(), less);
        for (std::size_t i = 1; i < values.size(); ++i)
            if (values[i - 1].index == values[i].index &&
                std::bit_cast<std::uint64_t>(values[i - 1].value) != std::bit_cast<std::uint64_t>(values[i].value))
                return false;
        sizes[axis] =
            std::unique(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.index == b.index; }) -
            values.begin();
        const auto distinct = values.first(sizes[axis]);
        for (std::size_t i = 1; i < distinct.size(); ++i)
            if (!(distinct[i].value > distinct[i - 1].value))
                return false;
        points += sizes[axis];
    }
    if (!enough_room(count, points))
        return false;

    adaptive_detail::Budget budget{operation_limit, used};
    struct Spent
    {
        std::size_t& used;
        const adaptive_detail::Budget& budget;
        ~Spent() { used = budget.used; }
    } spent{used, budget};
    PLIC::CartesianSourceDescription description;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        reserve_checked(description.axes[axis], sizes[axis], budget);
        for (const auto& point : std::span(coordinates[axis]).first(sizes[axis]))
            description.axes[axis].push_back(point.value);
    }
    reserve_checked(description.cells, count, budget);
    for (const auto& row : std::span(rows).first(count))
    {
        PLIC::CartesianSourceCellSlot cell;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto values = std::span(coordinates[axis]).first(sizes[axis]);
            const auto slot = [&](std::uint64_t index)
            {
                return std::uint32_t(std::lower_bound(values.begin(), values.end(), index,
                                         [](const auto& point, auto wanted) { return point.index < wanted; }) -
                                     values.begin());
            };
            cell.lower[axis] = slot(row.location[axis]);
            cell.upper[axis] = slot(row.location[axis] + 1);
        }
        description.cells.push_back(cell);
    }
    auto source = Access::make(std::move(description), geometry_limits(ordinary, budget.limit - budget.used),
        PLIC::details::NativeCellGeometryMoments::PrimedState);
    auto replacement = PreparedCartesianPages::appended(prepared_cartesian_ ? &*prepared_cartesian_ : nullptr,
        {anchor->native_recipe, std::move(source), first, mask}, budget);
    PLIC::details::GeometryOwnerGraphWork work;
    PLIC::details::GeometryOwnerGraphEditor editor(payload_owners_[0], 0, budget.used, budget.limit, &owners, &work);
    replacement.visit_dependencies(editor, true);
    if (prepared_cartesian_)
        prepared_cartesian_->visit_dependencies(editor, false);
    auto manifest = editor.finish();
    prepared_cartesian_ = std::move(replacement);
    payload_owners_[0] = std::move(manifest);
    record_owner_work(work);
    return true;
}
PLIC::details::OwnerManifestPair adaptive_detail::GeometrySnapshotAccess::owner_pair(
    const AdaptiveGeometrySnapshot& snapshot)
{
    return PLIC::details::OwnerManifestPair::sealed(snapshot.payload_owners_, snapshot.cross_payload_bytes_);
}
PLIC::details::GeometryOwnerRootBinding adaptive_detail::GeometrySnapshotAccess::owner_binding(
    const std::shared_ptr<const AdaptiveGeometrySnapshot>& snapshot)
{
    if (!snapshot)
        return {};
    return {snapshot, sizeof(*snapshot), owner_pair(*snapshot)};
}
geometry_snapshot_detail::PreparedOwnerManifest adaptive_detail::GeometrySnapshotAccess::owner_manifest(
    const AdaptiveGeometrySnapshot& snapshot, adaptive_detail::Budget& budget, bool bounded)
{
    return snapshot.owner_manifest(budget, bounded);
}
void adaptive_detail::GeometrySnapshotAccess::owner_manifests(const AdaptiveGeometrySnapshot& snapshot,
    adaptive_detail::Budget& budget, geometry_snapshot_detail::PreparedOwnerManifest& ordinary,
    geometry_snapshot_detail::PreparedOwnerManifest& bounded, bool payload_prepaid)
{
    snapshot.owner_manifests(budget, ordinary, bounded, payload_prepaid);
}

void adaptive_detail::GeometrySnapshotAccess::visit_cartesian_payload_edges(const AdaptiveGeometrySnapshot& snapshot,
    const std::function<void(const void*, const void*, std::size_t)>& visitor,
    const std::function<void(const void*, const void*, std::size_t)>& buffers,
    const std::function<void(const void*, const AdaptiveCellKey&)>& page_keys)
{
    if (snapshot.mode_ != ArithmeticMode::Ordinary || snapshot.purpose_ != GeometrySnapshotPurpose::OriginalSources ||
        snapshot.finite_ordinary_ || snapshot.bounded_ || snapshot.prepared_cartesian_)
        throw std::logic_error("Cartesian edge probe requires an Ordinary Raw native root.");
    std::set<const void*> expanded;
    const auto edge = [&](const void* parent, const void* owner, std::size_t bytes)
    {
        if (!owner)
            return false;
        visitor(parent, owner, bytes);
        return expanded.insert(owner).second;
    };
    if (edge(nullptr, snapshot.bindings_.get(), snapshot.bindings_->exclusive_bytes()))
        snapshot.bindings_->visit_cartesian_edges(edge, buffers, page_keys);
    if (!snapshot.ordinary_ || !edge(nullptr, snapshot.ordinary_.get(), snapshot.ordinary_->storage_bytes()))
        return;
    const auto source = [&](const void* parent, const PLIC::CartesianCellSource& value)
    {
        using Access = PLIC::details::NativeCellGeometrySourceAccess;
        const auto bytes = Access::exclusive_bytes(value);
        if (!edge(parent, &value, bytes[0] + bytes[1] + bytes[2]))
            return;
        const auto base = Access::base_owner(value);
        const auto base_bytes = Access::base_bytes(value);
        if (!edge(&value, base.get(), base_bytes[0] + base_bytes[1] + base_bytes[2]))
            return;
        for (const auto& support : Access::retained_supports(value))
        {
            const void* identity_parent = base.get();
            PLIC::visit_identity_allocations(support,
                [&](const PLIC::GeometryIdentityAllocation& allocation)
                {
                    const bool descend = edge(identity_parent, allocation.owner.get(), allocation.payload_bytes);
                    identity_parent = allocation.owner.get();
                    return descend;
                });
        }
    };
    struct Context
    {
        const decltype(edge)& emit_edge;
        const decltype(source)& emit_source;
        const void* catalog;
    } context{edge, source, snapshot.ordinary_.get()};
    PLIC::details::CellCatalogDemandAccess::visit_borrowed_dependencies(*snapshot.ordinary_,
        {&context,
            [](void* raw, const void* owner, const void* parent, std::size_t bytes)
            {
                auto& value = *static_cast<Context*>(raw);
                return value.emit_edge(parent ? parent : value.catalog, owner, bytes);
            }},
        {&context, [](void* raw, const void* parent, PLIC::details::geometry_demand::SourceKind kind, const void* owner)
            {
                if (kind != PLIC::details::geometry_demand::SourceKind::Cartesian)
                    throw std::logic_error("Cartesian edge probe encountered an explicit shell page.");
                static_cast<Context*>(raw)->emit_source(parent, *static_cast<const PLIC::CartesianCellSource*>(owner));
            }});
}

adaptive_detail::GeometrySnapshotOwnerStatistics adaptive_detail::GeometrySnapshotAccess::owner_statistics(
    const AdaptiveGeometrySnapshot& snapshot)
{
    const auto& work = snapshot.owner_work_;
    GeometrySnapshotOwnerStatistics result{
        work[0], work[1], work[2], work[3], work[4], work[5], work[6], work[7], work[8]};
    for (const auto& manifest : snapshot.payload_owners_)
    {
        result.stored_metadata_payload_bytes += manifest.metadata_payload_bytes();
        result.stored_metadata_admission_bytes += manifest.metadata_admission_bytes();
    }
    return result;
}
std::shared_ptr<const void> adaptive_detail::GeometrySnapshotAccess::binding_owner(
    const AdaptiveGeometrySnapshot& snapshot)
{
    return snapshot.bindings_;
}
void adaptive_detail::GeometrySnapshotAccess::visit_owner_manifest_metadata(
    const AdaptiveGeometrySnapshot& snapshot, PLIC::details::geometry_demand::MetadataSink metadata)
{
    for (const auto& manifest : snapshot.payload_owners_)
        manifest.visit_dependencies(metadata, {nullptr, [](void*, const auto&) {}});
}
void AdaptiveGeometrySnapshot::record_owner_work(const PLIC::details::GeometryOwnerGraphWork& work)
{
    owner_work_[0] += work.dependency_entries;
    owner_work_[1] += work.edge_visits;
    owner_work_[2] += work.first_transitions;
    owner_work_[3] += work.last_transitions;
    owner_work_[4] += work.index.copied_pages;
    owner_work_[5] += work.index.copied_bytes;
    owner_work_[6] += work.index.visited_pages;
    owner_work_[8] += work.lookup.visited_pages;
}
void AdaptiveGeometrySnapshot::visit_payload_dependencies(PLIC::details::GeometryOwnerGraphEditor& ordinary,
    PLIC::details::GeometryOwnerGraphEditor& bounded, bool retain) const
{
    if (ordinary.external(bindings_, bindings_->exclusive_bytes(), retain, true))
        bindings_->visit_dependencies(ordinary, retain);
    ordinary.catalog(ordinary_, retain);
    ordinary.catalog(finite_ordinary_, retain);
    bounded.catalog(bounded_, retain);
    if (prepared_cartesian_)
        prepared_cartesian_->visit_dependencies(ordinary, retain);
}
void AdaptiveGeometrySnapshot::update_owner_manifests(const AdaptiveGeometrySnapshot* previous,
    PLIC::details::GeometryOwnerGraphEditor& ordinary, PLIC::details::GeometryOwnerGraphEditor& bounded)
{
    // Catalog roots were admitted immediately after each construction, before
    // the next independent catalog can consume the remaining operation budget.
    if (ordinary.external(bindings_, bindings_->exclusive_bytes(), true, true))
        bindings_->visit_dependencies(ordinary, true);
    if (prepared_cartesian_)
        prepared_cartesian_->visit_dependencies(ordinary, true);
    if (previous)
        previous->visit_payload_dependencies(ordinary, bounded, false);
    auto ordinary_manifest = ordinary.finish();
    auto bounded_manifest = bounded.finish();
    payload_owners_ = {std::move(ordinary_manifest), std::move(bounded_manifest)};
}
void AdaptiveGeometrySnapshot::owner_manifests(adaptive_detail::Budget& budget,
    geometry_snapshot_detail::PreparedOwnerManifest& ordinary, geometry_snapshot_detail::PreparedOwnerManifest& bounded,
    bool payload_prepaid) const
{
    (void) payload_prepaid;
    (void) budget;
    ordinary = payload_owners_[0];
    bounded = payload_owners_[1];
}

geometry_snapshot_detail::PreparedOwnerManifest AdaptiveGeometrySnapshot::owner_manifest(
    adaptive_detail::Budget& budget, bool bounded) const
{
    geometry_snapshot_detail::PreparedOwnerManifest ordinary, bounded_owners;
    owner_manifests(budget, ordinary, bounded_owners);
    return bounded ? bounded_owners : ordinary;
}

AdaptiveGeometrySnapshot::PreparedSourceSlot AdaptiveGeometrySnapshot::prepared_slot(const AdaptiveLeaf& leaf) const
{
    if (!prepared_cartesian_ || !leaf.native_recipe)
        return {};
    const auto first = leaf.native_cell / geometry_snapshot_detail::prepared_cartesian_page_size *
                       geometry_snapshot_detail::prepared_cartesian_page_size;
    const auto* page = prepared_cartesian_->find(leaf.native_recipe, first);
    if (!page)
        return {};
    const auto bit = std::uint64_t(1) << (leaf.native_cell - first);
    if (!(page->rows & bit))
        return {};
    // A page stores only the cells set in its mask, in native order, so the
    // source slot is the rank below this cell.
    const auto slot = std::size_t(std::popcount(page->rows & (bit - 1)));
    if (slot >= page->source->size())
        return {};
    return {page->source, slot, true};
}

bool AdaptiveGeometrySnapshot::can_rebind_prepared_slots(
    std::size_t count, std::size_t available, geometry_snapshot_detail::OptionalOwnerReservation publication) const
{
    using Access = PLIC::details::CellCatalogDemandAccess;
    using Manifest = geometry_snapshot_detail::PreparedOwnerManifest;
    namespace G = PLIC::details::geometry_demand;
    if (!count)
        return true;
    if (count > 4)
        return false;
    std::size_t bytes = geometry_snapshot_detail::checked_product(count, sizeof(G::SourceSlot)), owners = 0;
    for (const auto& catalog : {ordinary_, finite_ordinary_})
        if (catalog)
        {
            const auto bound = Access::rebound_bound(*catalog, count);
            bytes = G::checked_sum(bytes, bound.bytes);
            owners = G::checked_sum(owners, bound.owners);
        }
    // Existing source DAGs may be deep. Their entire cached incoming-edge
    // count bounds the retired frontier without walking any unrelated owner.
    const auto edges = G::checked_sum(payload_owners_[0].physical_edge_count(),
        G::checked_sum(4, geometry_snapshot_detail::checked_product(owners, G::leaf_capacity)));
    const auto keys = G::checked_sum(payload_owners_[0].owner_count(), owners);
    bytes = G::checked_sum(
        bytes, PLIC::details::GeometryOwnerGraphEditor::edit_byte_bound(payload_owners_[0], edges, keys));
    bytes = G::checked_sum(bytes, PLIC::details::identity_ledger_bookkeeping_bound(owners));
    // Both selected slots are procedural. Existing typed pages still travel
    // with the manifest when its enclosing cache reserves an import.
    const auto frontier = G::checked_sum(payload_owners_[0].metadata_owner_count_bound(),
        Manifest::metadata_owner_count_bound(keys, payload_owners_[0].typed_owner_count()));
    const auto reserved = publication.bytes ? publication.bytes(publication.context, owners, frontier) : 0;
    return reserved <= available && bytes <= available - reserved;
}

void AdaptiveGeometrySnapshot::rebind_prepared_slots(std::span<const std::size_t> ordinals,
    std::span<const PreparedSourceSlot> physical, PLIC::GeometryAllocationLedger& ordinary_owners, std::size_t& used,
    std::size_t limit, PLIC::details::CellCatalogDemandWork* demand_work)
{
    using Access = PLIC::details::CellCatalogDemandAccess;
    namespace G = PLIC::details::geometry_demand;
    if (ordinals.empty())
        return;
    std::vector<G::SourceSlot> slots;
    const auto slot_bytes = geometry_snapshot_detail::checked_product(physical.size(), sizeof(G::SourceSlot));
    if (used > limit || slot_bytes > limit - used)
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Prepared slot scratch exceeds admission.");
    used += slot_bytes;
    slots.reserve(physical.size());
    const auto extra =
        geometry_snapshot_detail::checked_product(slots.capacity() - physical.size(), sizeof(G::SourceSlot));
    if (extra > limit - used)
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Prepared slot capacity exceeds admission.");
    used += extra;
    for (const auto& value : physical)
        slots.push_back({value.source, value.row, value.valid ? G::SourceKind::Cartesian : G::SourceKind::Explicit});
    const auto remaining = [&] { return limit - std::min(used, limit); };
    PLIC::details::GeometryOwnerGraphWork work;
    PLIC::details::GeometryOwnerGraphEditor editor(payload_owners_[0], 0, used, limit, &ordinary_owners, &work);
    struct Counter
    {
        std::size_t& used;
        std::size_t limit;
    } counter{used, limit};
    const G::Admission construction{&counter, [](void* raw, std::size_t bytes)
        {
            auto& value = *static_cast<Counter*>(raw);
            if (value.used > value.limit || bytes > value.limit - value.used)
                throw PLIC::GeometryError(
                    PLIC::GeometryErrorCode::ResourceLimit, "Prepared catalog construction receipt exceeds admission.");
            value.used += bytes;
        }};
    auto ordinary = ordinary_, finite = finite_ordinary_;
    if (ordinary)
    {
        ordinary = Access::rebound_precharged(ordinary, ordinals, slots,
            geometry_snapshot_detail::geometry_limits(ArithmeticMode::Ordinary, remaining()), construction, demand_work);
        editor.catalog(ordinary, true, true);
    }
    if (finite)
    {
        finite = Access::rebound_precharged(finite, ordinals, slots,
            geometry_snapshot_detail::geometry_limits(ArithmeticMode::Ordinary, remaining()), construction, demand_work);
        editor.catalog(finite, true, true);
    }
    editor.catalog(ordinary_, false);
    editor.catalog(finite_ordinary_, false);
    auto manifest = editor.finish();
    ordinary_ = std::move(ordinary);
    finite_ordinary_ = std::move(finite);
    payload_owners_[0] = std::move(manifest);
    record_owner_work(work);
}

void AdaptiveGeometrySnapshot::retain_prepared_cartesian_sources(
    PLIC::GeometryAllocationLedger& owners, std::size_t& used, std::size_t limit) const
{
    if (!prepared_cartesian_)
        return;
    // The immutable index value itself is inside this snapshot header.

    prepared_cartesian_->visit_metadata(
        [&](const auto& owner, std::size_t bytes) { owners.retain_external_owner(owner, bytes, used, limit); });
    prepared_cartesian_->visit_pages([&](const auto& page) { owners.retain(page.source, used, limit); });
}
void AdaptiveGeometrySnapshot::visit_prepared_cartesian_sources(
    const std::function<void(const PLIC::CartesianCellSource&)>& visitor) const
{
    if (prepared_cartesian_)
        prepared_cartesian_->visit_pages([&](const auto& page) { visitor(*page.source); });
}
std::size_t AdaptiveGeometrySnapshot::metadata_bytes() const noexcept
{
    return sizeof(*this) + bindings_->metadata_bytes() +
           (prepared_cartesian_ ? prepared_cartesian_->metadata_bytes() : 0);
}
void AdaptiveGeometrySnapshot::visit_external_owners(const std::function<void(const void*, std::size_t)>& visitor) const
{
    visitor(this, sizeof(*this));
    visitor(bindings_.get(), bindings_->exclusive_bytes());
    if (prepared_cartesian_)
        prepared_cartesian_->visit_metadata([&](const auto& owner, std::size_t bytes) { visitor(owner.get(), bytes); });
    bindings_->visit_children([&](const auto& owner, std::size_t bytes) { visitor(owner.get(), bytes); });
}
void adaptive_detail::GeometrySnapshotAccess::visit_binding_sources(
    const AdaptiveGeometrySnapshot& snapshot, const std::function<void(const void*)>& visitor)
{
    snapshot.bindings_->visit_source_descriptors(visitor);
}
void AdaptiveGeometrySnapshot::visit_authoritative_geometries(
    const std::function<void(const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&)>& visitor) const
{
    bindings_->visit_certificates(
        [&](const auto& certificate)
        {
            if (certificate.geometry)
                visitor(*certificate.geometry);
        });
}
void AdaptiveGeometrySnapshot::visit_authoritative_surfaces(
    const std::function<void(const GeometrySnapshotSurface&)>& visitor) const
{
    bindings_->visit_certificates(
        [&](const auto& certificate)
        {
            for (const auto& value : certificate.surfaces)
                visitor(value);
        });
}
void AdaptiveGeometrySnapshot::retain(PLIC::GeometryAllocationLedger& ordinary_owners,
    PLIC::GeometryAllocationLedger& bounded_owners, std::size_t& used, std::size_t limit) const
{
    ordinary_owners.retain_external_owner(shared_from_this(), sizeof(*this), used, limit);
    ordinary_owners.retain_external_owner(bindings_, bindings_->exclusive_bytes(), used, limit);
    bindings_->visit_children([&](const auto& owner, std::size_t bytes)
        { ordinary_owners.retain_external_owner(owner, bytes, used, limit); });
    bindings_->visit_certificates(
        [&](const auto& certificate)
        {
            if (certificate.geometry)
                ordinary_owners.retain(certificate.geometry, used, limit);
            for (const auto& value : certificate.surfaces)
            {
                if (value.surface.geometry)
                    ordinary_owners.retain_planar_face(value.surface.geometry, used, limit);
                if (value.surface.native_whole_geometry)
                    ordinary_owners.retain_planar_face(value.surface.native_whole_geometry, used, limit);
            }
        });
    // Ordinary displacement identities are inline in AdaptiveSurface.
    static_assert(!PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>::shared_storage);
    retain_prepared_cartesian_sources(ordinary_owners, used, limit);
    ordinary_owners.retain(ordinary_, used, limit);
    if (finite_ordinary_)
        ordinary_owners.retain(finite_ordinary_, used, limit);
    if (bounded_)
        bounded_owners.retain(bounded_, used, limit);
    // Index administration follows the manifest's arithmetic channel. This
    // lets a previous import credit its own index exactly once in that ledger.
    struct MetadataContext
    {
        PLIC::GeometryAllocationLedger* ledger;
        std::uint8_t channel;
        std::size_t& used;
        std::size_t limit;
    } metadata{&ordinary_owners, 0, used, limit};
    for (unsigned channel = 0; channel < payload_owners_.size(); ++channel)
    {
        metadata.ledger = channel ? &bounded_owners : &ordinary_owners;
        metadata.channel = std::uint8_t(channel);
        payload_owners_[channel].visit_dependencies(
            {&metadata,
                [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
                {
                    auto& value = *static_cast<MetadataContext*>(raw);
                    PLIC::details::GeometryAllocationOwnerAccess::retain_manifest_metadata(
                        *value.ledger, value.channel, owner, payload, admitted, value.used, value.limit);
                    return true;
                }},
            {nullptr, [](void*, const auto&) {}});
    }
}
std::size_t AdaptiveGeometrySnapshot::retained_bytes() const
{
    PLIC::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t bytes = 0;
    retain(ordinary_owners, bounded_owners, bytes, std::numeric_limits<std::size_t>::max());
    return bytes;
}
std::shared_ptr<AdaptiveGeometrySnapshot> AdaptiveGeometrySnapshot::build(const DistributedAdaptiveHierarchy& hierarchy,
    std::span<const AdaptiveLeaf* const> leaves, ArithmeticMode mode, GeometrySnapshotPurpose purpose,
    std::size_t& used, std::size_t limit, PLIC::GeometryAllocationLedger& ordinary_owners,
    PLIC::GeometryAllocationLedger& bounded_owners, std::shared_ptr<const AdaptiveGeometrySnapshot> previous,
    std::span<const geometry_snapshot_detail::PublicationSource> publication,
    PLIC::details::CellCatalogDemandWork* demand_work)
{
    struct RejectOperationLedgers
    {
        PLIC::GeometryAllocationLedger& ordinary;
        PLIC::GeometryAllocationLedger& bounded;
        int exceptions = std::uncaught_exceptions();
        ~RejectOperationLedgers() noexcept
        {
            if (std::uncaught_exceptions() > exceptions)
            {
                PLIC::details::GeometryAllocationOwnerAccess::poison(ordinary);
                PLIC::details::GeometryAllocationOwnerAccess::poison(bounded);
            }
        }
    } reject_ledgers{ordinary_owners, bounded_owners};
    PLIC::details::GeometryOwnerAdmissionPair paired_admission(ordinary_owners, bounded_owners);
    adaptive_detail::Budget budget{limit, used};
    struct SpentCounter
    {
        std::size_t& published;
        adaptive_detail::Budget& budget;
        ~SpentCounter() { published = std::max(published, budget.used); }
    } spent{used, budget};
    if (!publication.empty())
    {
        if (publication.size() != leaves.size())
            throw std::invalid_argument("Publication snapshot lacks its complete selected source provenance.");
        for (std::size_t slot = 0; slot < leaves.size(); ++slot)
            if (!leaves[slot] || publication[slot].leaf != leaves[slot] || !publication[slot].binding ||
                !publication[slot].binding->full || publication[slot].binding->key != leaves[slot]->key)
                throw std::invalid_argument("Publication snapshot source is not its authenticated full row.");
    }
    const auto controls = geometry_construction_controls();
    if (previous && (previous->mode_ != mode || !previous->stable_controls_ || !controls.supported ||
                        previous->controls_ != controls))
        previous.reset();
    // Only the exact same layout can extend the selected set with older rows.
    // Across layouts reuse is restricted to exact matches of requested cells.
    const bool extend = previous && previous->matches_layout(hierarchy.leaf_layout());
    const bool sparse = extend && previous->purpose_ == purpose && publication.empty();
    std::vector<std::optional<std::size_t>> previous_slots;
    if (sparse)
    {
        geometry_snapshot_detail::reserve_checked(previous_slots, leaves.size(), budget);
        for (const auto* leaf : leaves)
        {
            if (!leaf)
                throw std::invalid_argument("Geometry snapshot lacks an authenticated source leaf.");
            previous_slots.push_back(previous->bindings_->find_key(leaf->key));
        }
    }
    std::vector<std::unique_ptr<AdaptiveLeaf>> trailing;
    std::vector<const AdaptiveLeaf*> expanded;
    geometry_snapshot_detail::reserve_checked(expanded, leaves.size(), budget);
    expanded.insert(expanded.end(), leaves.begin(), leaves.end());
    if (std::any_of(leaves.begin(), leaves.end(), [](const auto* leaf) { return !leaf; }))
        throw std::invalid_argument("Geometry snapshot lacks an authenticated source leaf.");
    {
        std::vector<const AdaptiveLeaf*> requested_order;
        geometry_snapshot_detail::reserve_checked(requested_order, leaves.size(), budget);
        requested_order.assign(leaves.begin(), leaves.end());
        std::sort(requested_order.begin(), requested_order.end(),
            [](const auto* a, const auto* b) { return a->key < b->key; });
        if (std::adjacent_find(requested_order.begin(), requested_order.end(),
                [](const auto* a, const auto* b) { return a->key == b->key; }) != requested_order.end())
            throw std::invalid_argument("Geometry snapshot contains a duplicate persistent cell key.");
    }
    if (extend && !sparse)
    {
        std::vector<const AdaptiveLeaf*> ordered;
        geometry_snapshot_detail::reserve_checked(ordered, leaves.size(), budget);
        ordered.assign(leaves.begin(), leaves.end());
        const auto less = [](const AdaptiveLeaf* first, const AdaptiveLeaf* second)
        { return first->key < second->key; };
        std::sort(ordered.begin(), ordered.end(), less);
        // Use the prior binding keys to count precisely without constructing a row.
        const auto remaining = previous->bindings_->unselected(ordered);
        if (remaining)
        {
            geometry_snapshot_detail::reserve_checked(expanded, leaves.size() + remaining, budget);
            previous->bindings_->append_unselected(ordered, *hierarchy.leaf_layout(), trailing, expanded, budget);
        }
    }
    AdaptiveGeometrySnapshot result;
    geometry_snapshot_detail::BuildFacts facts;
    switch (mode)
    {
        case ArithmeticMode::Ordinary:
            result = geometry_snapshot_detail::Builder::build<ArithmeticMode::Ordinary>(hierarchy, expanded, budget,
                previous.get(), purpose, ordinary_owners, bounded_owners, publication, facts, true);
            break;
        case ArithmeticMode::BoundedCertification:
            result = geometry_snapshot_detail::Builder::build<ArithmeticMode::BoundedCertification>(hierarchy, expanded,
                budget, previous.get(), purpose, ordinary_owners, bounded_owners, publication, facts, true);
            break;
        case ArithmeticMode::Exact:
            result = geometry_snapshot_detail::Builder::build<ArithmeticMode::Exact>(hierarchy, expanded, budget,
                previous.get(), purpose, ordinary_owners, bounded_owners, publication, facts, true);
            break;
        default:
            throw std::invalid_argument("Unknown geometry snapshot arithmetic mode.");
    }
    result.source_layout_ = hierarchy.leaf_layout();
    if (extend && purpose == GeometrySnapshotPurpose::State && previous->purpose_ == purpose)
        result.prepared_cartesian_ = previous->prepared_cartesian_;
    using CatalogAccess = PLIC::details::CellCatalogDemandAccess;
    const auto limits = [&](ArithmeticMode arithmetic)
    { return geometry_snapshot_detail::geometry_limits(arithmetic, budget.limit - budget.used); };
    PLIC::details::GeometryOwnerGraphWork ordinary_work, bounded_work;
    PLIC::details::GeometryOwnerGraphEditor ordinary_graph(
        previous ? previous->payload_owners_[0] : geometry_snapshot_detail::PreparedOwnerManifest{}, 0, budget.used,
        budget.limit, &ordinary_owners, &ordinary_work);
    PLIC::details::GeometryOwnerGraphEditor bounded_graph(
        previous ? previous->payload_owners_[1] : geometry_snapshot_detail::PreparedOwnerManifest{}, 1, budget.used,
        budget.limit, &bounded_owners, &bounded_work);
    std::size_t cross_payload = previous ? previous->cross_payload_bytes_ : 0;
    ordinary_graph.pair_with(bounded_graph, cross_payload);
    const auto retain_catalog = [&]<ArithmeticMode M>(
                                    std::shared_ptr<const PLIC::CellGeometryCatalog<M>> catalog, auto&)
    {
        if constexpr (M == ArithmeticMode::BoundedCertification)
            bounded_graph.catalog(catalog, true, true);
        else
            ordinary_graph.catalog(catalog, true, true);
        return catalog;
    };
    namespace G = PLIC::details::geometry_demand;
    const G::Admission construction{
        &budget, [](void* raw, std::size_t bytes) { static_cast<adaptive_detail::Budget*>(raw)->add(bytes); }};
    const G::SourceSink ordinary_created{&ordinary_graph,
        [](void* raw, G::SourceKind kind, const std::shared_ptr<const void>& owner)
        {
            if (kind != G::SourceKind::Explicit)
                throw std::logic_error("Catalog constructor issued a foreign source receipt.");
            static_cast<PLIC::details::GeometryOwnerGraphEditor*>(raw)->prepay_catalog_page<ArithmeticMode::Ordinary>(
                std::static_pointer_cast<const PLIC::CellGeometryCatalog<ArithmeticMode::Ordinary>::ExplicitPage>(
                    owner));
        }};
    const G::SourceSink bounded_created{&bounded_graph,
        [](void* raw, G::SourceKind kind, const std::shared_ptr<const void>& owner)
        {
            if (kind != G::SourceKind::Explicit)
                throw std::logic_error("Catalog constructor issued a foreign source receipt.");
            static_cast<PLIC::details::GeometryOwnerGraphEditor*>(raw)
                ->prepay_catalog_page<ArithmeticMode::BoundedCertification>(std::static_pointer_cast<
                    const PLIC::CellGeometryCatalog<ArithmeticMode::BoundedCertification>::ExplicitPage>(owner));
        }};
    if (sparse)
    {
        const auto requested_ordinary = result.ordinary_;
        result.ordinary_ =
            retain_catalog(CatalogAccess::extended_precharged(previous->ordinary_, requested_ordinary, previous_slots,
                               limits(ArithmeticMode::Ordinary), demand_work, construction, ordinary_created),
                ordinary_owners);
        if (previous->finite_ordinary_ || result.finite_ordinary_)
            result.finite_ordinary_ = retain_catalog(
                CatalogAccess::extended_precharged(
                    previous->finite_ordinary_ ? previous->finite_ordinary_ : previous->ordinary_,
                    result.finite_ordinary_ ? result.finite_ordinary_ : requested_ordinary, previous_slots,
                    limits(ArithmeticMode::Ordinary), demand_work, construction, ordinary_created),
                ordinary_owners);
        if (result.bounded_)
            result.bounded_ = retain_catalog(
                CatalogAccess::extended_precharged(previous->bounded_, result.bounded_, previous_slots,
                    limits(ArithmeticMode::BoundedCertification), demand_work, construction, bounded_created),
                bounded_owners);
        result.bindings_ = geometry_snapshot_detail::Bindings::extended(
            *previous->bindings_, leaves, facts, *hierarchy.leaf_layout(), budget);
    }
    else
    {
        result.ordinary_ = retain_catalog(CatalogAccess::initialize_precharged(result.ordinary_,
                                              limits(ArithmeticMode::Ordinary), demand_work, construction),
            ordinary_owners);
        if (result.finite_ordinary_)
            result.finite_ordinary_ = retain_catalog(CatalogAccess::initialize_precharged(result.finite_ordinary_,
                                                         limits(ArithmeticMode::Ordinary), demand_work, construction),
                ordinary_owners);
        if (result.bounded_)
            result.bounded_ = retain_catalog(CatalogAccess::initialize_precharged(result.bounded_,
                                                 limits(ArithmeticMode::BoundedCertification), demand_work, construction),
                bounded_owners);
        const auto same_raw_bindings = [&]
        {
            if (!publication.empty() || !extend || mode != ArithmeticMode::Ordinary ||
                purpose != GeometrySnapshotPurpose::OriginalSources ||
                previous->purpose_ != GeometrySnapshotPurpose::State || !result.stable_controls_ ||
                !previous->stable_controls_ || result.controls_ != previous->controls_ ||
                previous->bindings_->size() != expanded.size() || result.ordinary_->size() != expanded.size())
                return false;
            for (std::size_t slot = 0; slot < expanded.size(); ++slot)
            {
                const auto old_slot = previous->find(*expanded[slot]);
                if (!old_slot || *old_slot != slot ||
                    !PLIC::CellGeometryView<ArithmeticMode::Ordinary>(result.ordinary_, slot).is_procedural() ||
                    !PLIC::CellGeometryView<ArithmeticMode::Ordinary>(previous->ordinary_, slot).is_procedural())
                    return false;
                const auto old_policy = previous->evaluation_policy(slot);
                const auto new_policy = facts.policy[slot];
                if (old_policy.original_integrals || old_policy.ordinary_integrals || old_policy.bounded_integrals ||
                    new_policy.original_integrals || new_policy.ordinary_integrals || new_policy.bounded_integrals)
                    return false;
            }
            return true;
        };
        // Purpose changes the evaluation contexts, not authenticated key and
        // recipe bindings. Raw catalogs remain independent deferred contexts;
        // their public metric range is empty regardless of shared metadata.
        if (same_raw_bindings())
            result.bindings_ = previous->bindings_;
        else
            result.bindings_ = geometry_snapshot_detail::Bindings::build(expanded, facts,
                previous && previous->purpose_ == purpose ? previous->bindings_.get() : nullptr, false, budget);
    }
    result.update_owner_manifests(previous.get(), ordinary_graph, bounded_graph);
    result.cross_payload_bytes_ = cross_payload;
    result.record_owner_work(ordinary_work);
    result.record_owner_work(bounded_work);
    budget.add(sizeof(AdaptiveGeometrySnapshot));
    auto snapshot = std::make_shared<AdaptiveGeometrySnapshot>(std::move(result));
    PLIC::details::GeometryAllocationOwnerAccess::admit_prepaid(
        ordinary_owners, snapshot, sizeof(*snapshot), budget.used, limit);
    return snapshot;
}
} // namespace SimpleFluid::Meshes::AMR
