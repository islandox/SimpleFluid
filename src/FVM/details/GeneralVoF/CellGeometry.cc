/** @file CellGeometry.cc
 * @brief Compiled sparse recipe decoding, catalog construction and existing-owner forwarding.
 */
#include "FVM/details/GeneralVoF/CellCatalog.tcc"
#include "FVM/details/GeneralVoF/CellCatalogInternal.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace SimpleFluid::VoF::general_vof_detail
{
PLIC::GeometryLimits geometry_limits(ArithmeticMode mode, std::size_t bytes)
{
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = mode;
    limits.byte_budget = bytes;
    return limits;
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> existing_native_geometry(
    const GeneralVoF::Data& data, const Cell& cell)
{
    return PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(native_cell_geometry(data, cell));
}
template<ArithmeticMode Mode>
PLIC::ConvexGeometryPtr<Mode> existing_cell_geometry(const GeneralVoF::Data& data, const Cell& cell)
    requires(Mode != ArithmeticMode::Exact)
{
    return PLIC::details::NativeCellGeometrySourceAccess::existing_geometry(cell_geometry<Mode>(data, cell));
}
template PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> existing_cell_geometry<ArithmeticMode::Ordinary>(
    const GeneralVoF::Data&, const Cell&);
template PLIC::ConvexGeometryPtr<ArithmeticMode::BoundedCertification>
existing_cell_geometry<ArithmeticMode::BoundedCertification>(const GeneralVoF::Data&, const Cell&);
} // namespace SimpleFluid::VoF::general_vof_detail

namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
{
CatalogConstructionControls construction_controls()
{
    CatalogConstructionControls result;
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
void charge_items(wire::Budget& budget, std::size_t count, std::size_t width)
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
std::array<std::uint64_t, 3> RecipeAxes::location(const AMR::AdaptiveLeaf& leaf) const
{
    const auto* ancestry = std::get_if<AMR::CartesianCellAncestry>(&leaf.key.ancestry);
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
AMR::AdaptiveCellMetrics RecipeAxes::metrics(const AMR::AdaptiveLeaf& leaf) const
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
bool eligible(const GeneralVoF::Hierarchy& hierarchy, const AMR::AdaptiveLeaf& leaf, ArithmeticMode mode)
{
    // These records come only from the hierarchy's validated owned/full native
    // leaf boundary. The public provider enum alone is never an authority.
    return hierarchy.has_native_cartesian_geometry() && mode == ArithmeticMode::Ordinary && !leaf.geometry &&
           leaf.native_recipe && leaf.native_recipe->kind == Meshes::NativeFragmentDescriptor::Kind::Cartesian &&
           leaf.provider == AMR::AdaptiveProviderKind::Cartesian &&
           std::holds_alternative<AMR::CartesianCellAncestry>(leaf.key.ancestry);
}
bool same_native_recipe(const AMR::AdaptiveLeaf& first, const AMR::AdaptiveLeaf& second)
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
bool same_native_representation(const PLIC::CellGeometryView<ArithmeticMode::Ordinary>& first,
    const PLIC::CellGeometryView<ArithmeticMode::Ordinary>& second)
{
    if (!first || !second || !first.is_procedural() || !second.is_procedural() || first.origin() != second.origin() ||
        first.vertex_count() != second.vertex_count() || first.num_faces() != second.num_faces() ||
        first.volume().value != second.volume().value)
        return false;
    for (std::size_t vertex = 0; vertex < first.vertex_count(); ++vertex)
        if (first.corner(vertex) != second.corner(vertex))
            return false;
    for (std::size_t face = 0; face < first.num_faces(); ++face)
    {
        if (first.source_support(face) != second.source_support(face))
            return false;
        const auto a = PLIC::details::NativeCellGeometrySourceAccess::original_face(first, face);
        const auto b = PLIC::details::NativeCellGeometrySourceAccess::original_face(second, face);
        if (a.normal != b.normal || a.anchor != b.anchor || a.frame_origin != b.frame_origin ||
            a.frame_exponent != b.frame_exponent || a.orientation != b.orientation || a.offset != b.offset)
            return false;
    }
    return true;
}
struct NativeMetrics::Impl
{
    const GeneralVoF::Hierarchy& hierarchy;
    ArithmeticMode mode;
    wire::Budget& budget;
    std::map<const Meshes::NativeFragmentDescriptor*, RecipeAxes> recipes;
};
NativeMetrics::NativeMetrics(const GeneralVoF::Hierarchy& hierarchy, ArithmeticMode mode, wire::Budget& budget)
{
    charge_items(budget, 1, sizeof(Impl));
    impl_ = std::make_unique<Impl>(Impl{hierarchy, mode, budget, {}});
}
NativeMetrics::~NativeMetrics() = default;
AMR::AdaptiveCellMetrics NativeMetrics::operator()(const AMR::AdaptiveLeaf& leaf)
{
    if (!eligible(impl_->hierarchy, leaf, impl_->mode))
        return impl_->hierarchy.metrics(leaf);
    auto found = impl_->recipes.find(leaf.native_recipe.get());
    if (found == impl_->recipes.end())
    {
        charge_items(impl_->budget, 1, sizeof(typename decltype(impl_->recipes)::value_type) + 4 * sizeof(void*));
        found = impl_->recipes.emplace(leaf.native_recipe.get(), RecipeAxes(*leaf.native_recipe)).first;
    }
    return found->second.metrics(leaf);
}
} // namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
{
void reserve_leaf_references(std::vector<const AMR::AdaptiveLeaf*>& values, std::size_t count, wire::Budget& budget)
{
    reserve_checked(values, count, budget);
}
void reserve_indices(std::vector<std::size_t>& values, std::size_t count, wire::Budget& budget)
{
    reserve_checked(values, count, budget);
}
template CatalogBuild build<ArithmeticMode::Ordinary>(const GeneralVoF::Hierarchy&,
    std::span<const AMR::AdaptiveLeaf* const>, wire::Budget&, const GeneralVoF::Data*, CatalogPurpose,
    PLIC::GeometryAllocationLedger*, const CatalogReuseAuthority*, const void*, const AMR::AdaptiveNativeDraft*,
    const AMR::HierarchyObservation*);
template CatalogBuild build<ArithmeticMode::BoundedCertification>(const GeneralVoF::Hierarchy&,
    std::span<const AMR::AdaptiveLeaf* const>, wire::Budget&, const GeneralVoF::Data*, CatalogPurpose,
    PLIC::GeometryAllocationLedger*, const CatalogReuseAuthority*, const void*, const AMR::AdaptiveNativeDraft*,
    const AMR::HierarchyObservation*);
template CatalogBuild build<ArithmeticMode::Exact>(const GeneralVoF::Hierarchy&,
    std::span<const AMR::AdaptiveLeaf* const>, wire::Budget&, const GeneralVoF::Data*, CatalogPurpose,
    PLIC::GeometryAllocationLedger*, const CatalogReuseAuthority*, const void*, const AMR::AdaptiveNativeDraft*,
    const AMR::HierarchyObservation*);
} // namespace SimpleFluid::VoF::general_vof_detail::cell_catalog_detail
