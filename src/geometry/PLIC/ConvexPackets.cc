#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/ConvexPackets.tcc"
#include "geometry/PLIC/details/ordinary/ConvexPackets.tcc"

namespace SimpleFluid::PLIC
{
const void* geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::PLIC
