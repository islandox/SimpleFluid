#include "geometry/PLIC/details/NativeConstructionWireAccess.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
namespace SimpleFluid::PLIC::details
{
double NativeConstructionWireAccess::ordinary_scalar(std::span<const char> packet, std::size_t& cursor)
{
    if (cursor > packet.size())
        throw std::invalid_argument("Native constructor scalar cursor exceeds its packet.");
    ordinary_native::Reader reader{packet};
    reader.cursor = cursor;
    const auto result = GeometryValue<ArithmeticMode::Ordinary>{reader.real()}.as_double();
    cursor = reader.cursor;
    return result;
}
} // namespace SimpleFluid::PLIC::details
