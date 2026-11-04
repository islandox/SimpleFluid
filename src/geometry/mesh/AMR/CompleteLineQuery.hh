/** @file CompleteLineQuery.hh
 * @brief One complete line's ordered endpoint-query kernel.
 */
#pragma once
#include "geometry/PLIC/PlanarGeometry.hh"
#include <functional>
#include <span>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using LinePoint = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
struct LineSubscription
{
    LinePoint first, last;
};
using LineQueryEmit = std::function<void(std::size_t, const PLIC::OrdinaryEdgeQueryScratch&)>;
using LineCharge = std::function<void(std::size_t)>;

/** Subscriptions must already be in legacy request-arrival order, including
 * duplicate resident tile and periodic-image subscriptions. Every result is
 * borrowed until the next emit returns. Caller owns the scoped admission. */
void query_complete_line(std::span<const LineSubscription>, const LineCharge&, const LineQueryEmit&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
