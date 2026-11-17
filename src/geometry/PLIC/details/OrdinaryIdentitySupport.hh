/** @file OrdinaryIdentitySupport.hh
 * @brief Private native source-support recipes and bounded import.
 */
#pragma once
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include <array>
#include <vector>
namespace SimpleFluid::PLIC::details
{
enum class OrdinarySupportKind : std::uint8_t
{
    Axis = 0,
    Source = 1,
    Input = 2,
    Plane = 3,
    Translation = 4,
    RetainedExact = 5,
    ExtrusionDag = 6
};
struct OrdinarySupportRecipe
{
    bool original = true;
    OrdinaryIdentityPoint root{};
    std::vector<std::size_t> parents;
};
[[nodiscard]] GeometryIdentityBackingPtr import_ordinary_support_backing(
    std::span<const char> checked_key, std::size_t byte_budget);
struct OrdinarySupportImportProfile
{
    std::size_t retained = 0, scratch = 0, nodes = 0, depth = 0;
};
[[nodiscard]] OrdinarySupportImportProfile ordinary_support_import_cpp_profile(std::span<const char>);
[[nodiscard]] GeometryIdentityBackingPtr import_ordinary_support_backing_admitted(
    std::span<const char>, std::size_t, const geometry_detail::NativeConstructionAdmission&);
[[nodiscard]] OrdinarySupportKind ordinary_support_kind(const PlaneSupportIdentity<ArithmeticMode::Ordinary>&);
[[nodiscard]] std::strong_ordering compare_ordinary_support_native(
    const GeometryIdentityBackingPtr&, const GeometryIdentityBackingPtr&);
} // namespace SimpleFluid::PLIC::details
