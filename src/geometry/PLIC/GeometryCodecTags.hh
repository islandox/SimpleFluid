/** @file GeometryCodecTags.hh
 * @brief Compact category tags; numeric payload widths are independent.
 */
#pragma once
#include <cstdint>
namespace SimpleFluid::PLIC
{
using std::uint8_t;

enum class GeometryPacketFamily : uint8_t
{
    Geometry = 1
};
/** Arithmetic format identifiers, independent of storage size and padding. */
enum class GeometryNumericProfile : uint8_t
{
    ExactRational = 0,
    Binary64 = 1,
    X87Binary80 = 2,
    Binary128 = 3,
    Unsupported = 0xff
};
enum class GeometryTag : uint8_t
{
    OrdinaryPoint = 0x01,
    OrdinaryLine = 0x02,
    OrdinaryDisplacement = 0x03,
    OrdinaryAxisSupport = 0x10,
    OrdinarySourceSupport = 0x11,
    OrdinaryInputSupport = 0x12,
    OrdinaryPlaneSupport = 0x13,
    OrdinaryTranslatedSupport = 0x14,
    OrdinaryExtrusionDagSupport = 0x15,
    ExactPoint = 0x21,
    ExactLine = 0x22,
    ExactPlaneSupport = 0x23,
    ExactDisplacement = 0x24,
    OrdinaryPlanarFace = 0x30,
    ExactPlanarFace = 0x31,
    OrdinaryCell = 0x32,
    ExactCell = 0x33,
    OrdinaryComplex = 0x34,
    ExactComplex = 0x35,
    ExtrudedConstruction = 0x40,
    ExtrusionBase = 0x41,
    AxisDescriptor = 0x42,
    AxisValues = 0x43,
    ProductView = 0x44,
    FiniteGeometry = 0x50,
    ConstructionRegistry = 0x51,
    OrdinaryMeasure = 0x60,
    ExactMeasure = 0x61
};
enum class SupportRecipeKind : uint8_t
{
    Original = 0,
    Derived = 1
};
enum class AuthorityKeyTag : uint8_t
{
    OriginalSupport = 0xA1
};
inline constexpr std::uint64_t authority_key_length_mask = (std::uint64_t(1) << 56) - 1;
[[nodiscard]] constexpr std::uint64_t original_support_key_header(std::uint64_t length) noexcept
{
    return (std::uint64_t(AuthorityKeyTag::OriginalSupport) << 56) | length;
}
} // namespace SimpleFluid::PLIC
