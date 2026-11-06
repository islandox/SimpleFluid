/** @file AdaptiveGeometrySnapshot.cc
 * @brief Mesh-owned immutable sparse native and fallback source construction.
 */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.tcc"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotInternal.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
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
std::array<std::uint64_t, 3> RecipeAxes::location(const AdaptiveLeaf& leaf) const
{
    const auto* ancestry = std::get_if<CartesianCellAncestry>(&leaf.key.ancestry);
    if (!ancestry)
        throw std::invalid_argument("Native Cartesian geometry lacks Cartesian ancestry.");
    std::array<std::uint64_t, 3> location{};
    auto ordinal = leaf.native_cell;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        location[axis] = ordinal % counts[axis];
        ordinal /= counts[axis];
        if (location[axis] != ancestry->index[axis] % counts[axis])
            throw std::invalid_argument("Native Cartesian ordinal differs from persistent typed ancestry.");
    }
    if (ordinal)
        throw std::invalid_argument("Native Cartesian cell ordinal exceeds recipe extent.");
    return location;
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
std::size_t AdaptiveGeometrySnapshot::metadata_bytes() const noexcept
{
    return sizeof(*this) + bindings_->metadata_bytes();
}
void AdaptiveGeometrySnapshot::visit_external_owners(const std::function<void(const void*, std::size_t)>& visitor) const
{
    visitor(this, sizeof(*this));
    visitor(bindings_.get(), bindings_->exclusive_bytes());
    bindings_->visit_children([&](const auto& owner, std::size_t bytes) { visitor(owner.get(), bytes); });
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
    bindings_->visit_children(
        [&](const auto& owner, std::size_t bytes) { ordinary_owners.retain_external_owner(owner, bytes, used, limit); });
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
    ordinary_owners.retain(ordinary_, used, limit);
    if (finite_ordinary_)
        ordinary_owners.retain(finite_ordinary_, used, limit);
    if (bounded_)
        bounded_owners.retain(bounded_, used, limit);
}
std::size_t AdaptiveGeometrySnapshot::retained_bytes() const
{
    PLIC::GeometryAllocationLedger ordinary_owners, bounded_owners;
    std::size_t bytes = 0;
    retain(ordinary_owners, bounded_owners, bytes, std::numeric_limits<std::size_t>::max());
    return bytes;
}
std::shared_ptr<const AdaptiveGeometrySnapshot> AdaptiveGeometrySnapshot::build(
    const DistributedAdaptiveHierarchy& hierarchy, std::span<const AdaptiveLeaf* const> leaves, ArithmeticMode mode,
    GeometrySnapshotPurpose purpose, std::size_t& used, std::size_t limit,
    PLIC::GeometryAllocationLedger& ordinary_owners, PLIC::GeometryAllocationLedger& bounded_owners,
    std::shared_ptr<const AdaptiveGeometrySnapshot> previous,
    std::span<const geometry_snapshot_detail::PublicationSource> publication)
{
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
    std::vector<AdaptiveLeaf> trailing;
    std::vector<const AdaptiveLeaf*> expanded;
    geometry_snapshot_detail::reserve_checked(expanded, leaves.size(), budget);
    expanded.insert(expanded.end(), leaves.begin(), leaves.end());
    if (std::any_of(leaves.begin(), leaves.end(), [](const auto* leaf) { return !leaf; }))
        throw std::invalid_argument("Geometry snapshot lacks an authenticated source leaf.");
    if (extend)
    {
        std::vector<const AdaptiveLeaf*> ordered;
        geometry_snapshot_detail::reserve_checked(ordered, leaves.size(), budget);
        ordered.assign(leaves.begin(), leaves.end());
        const auto less = [](const AdaptiveLeaf* first, const AdaptiveLeaf* second) { return first->key < second->key; };
        std::sort(ordered.begin(), ordered.end(), less);
        // Use the prior binding keys to count precisely without constructing a row.
        const auto remaining = previous->bindings_->unselected(ordered);
        if (remaining)
        {
            geometry_snapshot_detail::reserve_checked(trailing, remaining, budget);
            geometry_snapshot_detail::reserve_checked(expanded, leaves.size() + remaining, budget);
            previous->bindings_->append_unselected(ordered, trailing, budget);
            for (const auto& leaf : trailing)
                expanded.push_back(&leaf);
        }
    }
    AdaptiveGeometrySnapshot result;
    geometry_snapshot_detail::BuildFacts facts;
    switch (mode)
    {
        case ArithmeticMode::Ordinary:
            result = geometry_snapshot_detail::Builder::build<ArithmeticMode::Ordinary>(
                hierarchy, expanded, budget, previous.get(), purpose, ordinary_owners, bounded_owners, publication, facts);
            break;
        case ArithmeticMode::BoundedCertification:
            result = geometry_snapshot_detail::Builder::build<ArithmeticMode::BoundedCertification>(
                hierarchy, expanded, budget, previous.get(), purpose, ordinary_owners, bounded_owners, publication, facts);
            break;
        case ArithmeticMode::Exact:
            result = geometry_snapshot_detail::Builder::build<ArithmeticMode::Exact>(
                hierarchy, expanded, budget, previous.get(), purpose, ordinary_owners, bounded_owners, publication, facts);
            break;
        default:
            throw std::invalid_argument("Unknown geometry snapshot arithmetic mode.");
    }
    result.source_layout_ = hierarchy.leaf_layout();
    result.bindings_ = geometry_snapshot_detail::Bindings::build(expanded, facts,
        previous && previous->purpose_ == purpose ? previous->bindings_.get() : nullptr, budget);
    budget.add(sizeof(AdaptiveGeometrySnapshot));
    auto snapshot = std::make_shared<const AdaptiveGeometrySnapshot>(std::move(result));
    // Bindings and source roots join the same caller ledger used throughout
    // construction. Existing payload owners are deduplicated; new caller ledger
    // metadata is admitted normally without resetting the cumulative counter.
    snapshot->retain(ordinary_owners, bounded_owners, budget.used, limit);
    return snapshot;
}
} // namespace SimpleFluid::Meshes::AMR
