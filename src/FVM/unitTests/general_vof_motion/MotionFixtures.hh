/** Native cell extraction and prescribed sheared-motion fixture. */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "ReferenceOracle.hh"
#include <memory>
#include <vector>

namespace SimpleFluid::VoF::test::motion
{
struct ReferenceCell
{
    Polygon xy;
    long double height = 0;
    long double volume() const { return area(xy) * height; }
    long double phase(long double time, bool gas) const { return clipped_area(xy, time, gas) * height; }
};
ReferenceCell reference_cell(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& shape);
std::shared_ptr<const GeneralVoF> motion_fixture(
    GeneralVoFTransportBackend backend = GeneralVoFTransportBackend::SweptPartition);
long double streamfunction(XY p);
std::vector<GeneralVoFNodeVelocity> velocities(const GeneralVoF& source);
std::vector<GeneralVoFFaceVolume> controls(const GeneralVoF& source);
} // namespace SimpleFluid::VoF::test::motion
