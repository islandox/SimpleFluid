#pragma once
#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
namespace SimpleFluid::PLIC::details
{
struct ExtrudedPreparedRecipes;
class SIMPLEFLUID_LOCAL ExtrudedConstructionAccess
{
public:
    using Construction = ExtrudedConstruction<ArithmeticMode::Ordinary>;
    /** Borrow the original wide Product factors without creating a cell shell or rounding coordinates. */
    struct CellFactors
    {
        std::span<const std::array<long double, 3>> base_points;
        std::span<const std::size_t> base_selection;
        std::span<const PlaneSupportIdentity<ArithmeticMode::Ordinary>> side_supports;
        std::array<long double, 2> axial;
    };
    [[nodiscard]] static CellFactors cell_factors(const Construction&, std::size_t);
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> face(
        const Construction&, std::size_t, const geometry_detail::NativeConstructionAdmission&);
    static std::shared_ptr<const Construction> decode(std::shared_ptr<const Meshes::ExtrudedTopology>,
        std::span<const char>, const geometry_detail::NativeConstructionAdmission&);
    static std::shared_ptr<const Construction> decode_parts(std::shared_ptr<const Meshes::ExtrudedTopology>,
        std::span<const char> base, std::span<const char> values, std::span<const char> selection,
        std::span<const char> view, const geometry_detail::NativeConstructionAdmission&,
        std::shared_ptr<const Construction> base_anchor = {}, std::shared_ptr<const Construction> values_anchor = {},
        std::shared_ptr<const Construction> selection_anchor = {});
    static std::shared_ptr<const Construction> make(std::shared_ptr<const Meshes::ExtrudedTopology>,
        std::span<const PointRecipe>, std::span<const size_t> active, std::span<const ScalarRecipe>,
        std::span<const size_t> zactive, const geometry_detail::NativeConstructionAdmission&);
    static std::shared_ptr<const Construction> shared_view(std::shared_ptr<const Meshes::ExtrudedTopology>,
        std::shared_ptr<const Construction> base, std::shared_ptr<const Construction> axis, size_t first, size_t layers,
        const geometry_detail::NativeConstructionAdmission&);
    static std::shared_ptr<const Construction> shared_axis_selection(std::shared_ptr<const Meshes::ExtrudedTopology>,
        std::shared_ptr<const Construction> base, std::shared_ptr<const Construction> values,
        std::span<const size_t> active, const geometry_detail::NativeConstructionAdmission&);

private:
    static std::shared_ptr<const Construction> make_owned(std::shared_ptr<const Meshes::ExtrudedTopology>,
        ExtrudedPreparedRecipes&&, const geometry_detail::NativeConstructionAdmission&,
        std::shared_ptr<const Construction> base_anchor = {}, std::shared_ptr<const Construction> values_anchor = {},
        std::shared_ptr<const Construction> selection_anchor = {});
};
} // namespace SimpleFluid::PLIC::details
