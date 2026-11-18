/** @file NativeFragmentDecodePlan.hh
 * @brief Span-only constructor counts tied to exact descriptor controls.
 */
#pragma once
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include <array>
#include <span>
namespace SimpleFluid::Meshes
{
/** Counts are construction inputs, not geometry certification. Cartesian and
 * split constructed-Product schemas are scanned without allocation. Legacy
 * native schemas require their separate admitted storage planners; they do not
 * enter this planner through an estimated byte multiplier.
 */
class SIMPLEFLUID_LOCAL NativeFragmentDecodePlan
{
public:
    struct Counts
    {
        std::size_t nodes = 0, cells = 0, loop_incidence = 0, maximum_loop = 0;
        std::size_t boundary_edges = 0, boundary_bytes = 0;
        std::size_t point_recipes = 0, point_parents = 0, active_points = 0;
        std::size_t axial_recipes = 0, axial_parents = 0, selected_axial = 0;
        std::size_t aliases = 0, alias_bytes = 0, name_bytes = 0;
        std::uint64_t first_layer = 0, layers = 0;
    };
    [[nodiscard]] static NativeFragmentDecodePlan scan(const std::shared_ptr<const NativeFragmentDescriptor>&);
    [[nodiscard]] const Counts& counts() const noexcept { return counts_; }
    [[nodiscard]] bool matches(const std::shared_ptr<const NativeFragmentDescriptor>&) const noexcept;
    [[nodiscard]] const std::shared_ptr<const NativeFragmentDescriptor>& descriptor() const noexcept { return recipe_; }

private:
    std::shared_ptr<const NativeFragmentDescriptor> recipe_;
    NativeFragmentDescriptor::Kind kind_ = NativeFragmentDescriptor::Kind::Cartesian;
    std::array<std::shared_ptr<const NativeFragmentBytes>, 4> parts_;
    std::array<std::span<const char>, 4> spans_;
    std::span<const char> view_;
    Counts counts_;
};
} // namespace SimpleFluid::Meshes
