/** @file PlaneOrientation.hh
 * @brief Private orientation provenance word shared by interface wire formats.
 */
#pragma once
#include "geometry/PLIC/PolyhedralReconstruction.hh"

namespace SimpleFluid::VoF::plane_wire_detail
{
inline PLIC::PolyhedralOrientationMethod orientation_method(std::uint64_t value)
{
    if (value > std::uint64_t(PLIC::PolyhedralOrientationMethod::AngularVolumeFit))
        throw std::invalid_argument("Invalid PLIC interface orientation method.");
    return PLIC::PolyhedralOrientationMethod(value);
}
template<class Writer> void orientation(Writer& writer, PLIC::PolyhedralOrientationMethod method)
{
    const auto value = std::uint64_t(method);
    (void) orientation_method(value);
    writer.word(value);
}
template<class Reader> PLIC::PolyhedralOrientationMethod orientation(Reader& reader)
{
    return orientation_method(reader.word());
}
} // namespace SimpleFluid::VoF::plane_wire_detail
