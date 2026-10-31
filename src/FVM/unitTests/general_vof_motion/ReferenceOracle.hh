/** Independent test-local polygon oracle; no production PLIC cuts or measures. */
#pragma once
#include <array>
#include <vector>

namespace SimpleFluid::VoF::test::motion
{
inline constexpr long double shear = .25L, speed = 1.L / 16, dt = 1.L / 64;
using XY = std::array<long double, 2>;
using Polygon = std::vector<XY>;
long double cross(XY a, XY b, XY c);
Polygon hull(Polygon p);
long double area(const Polygon& p);
long double clipped_area(const Polygon& p, long double time, bool gas);
Polygon intersection(Polygon p, const Polygon& clip);
} // namespace SimpleFluid::VoF::test::motion
