#include "geometry/PLIC/details/GeometryKernels.hh"

namespace SimpleFluid::PLIC
{
const void* geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::PLIC

#include "geometry/PLIC/details/bounded/ConvexPackets.tcc"
#include "geometry/PLIC/details/ordinary/ConvexPackets.tcc"
