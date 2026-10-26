/** @file PhaseAllocation.hh
 * @brief Physical phase labels and uncertainty shared by transport and remap.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
using PhaseValue = PLIC::DynamicGeometryValue;
using PhaseValues = PLIC::DynamicPhaseMeasures;
template<ArithmeticMode Mode> struct SIMPLEFLUID_FVM_LOCAL PhaseSum
{
    fp::SumFor<Mode> bulk, gas, carrier;
    PhaseSum() = default;
    explicit PhaseSum(ArithmeticMode mode) : bulk(mode), gas(mode), carrier(mode) {}
    void add(const PhaseValues& phases, int sign = 1);
    PhaseValues value() const { return {bulk.value(), gas.value(), carrier.value()}; }
};
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL PhaseValues phase_inventory(const Cell&, const PLIC::ConvexCellGeometry<Mode>&);
template<ArithmeticMode Mode> struct SIMPLEFLUID_FVM_LOCAL PhaseLabel
{
    PhaseValue gas_factor, carrier_factor, gas_history_factor, carrier_history_factor;
    PhaseValues raw_source;
    PhaseValue gas_deviation, carrier_deviation;
    bool gas_bulk = false, carrier_bulk = false;
    PhaseValue allocation(const PhaseValues& raw_piece) const;
    /** Aggregate scalar admission has its own metric contract and supplies both
     * phases independently. This overload never constructs a cut-cell shell. */
    PhaseValues apply(const PhaseValues& raw_piece) const;
    PhaseValues history(const PhaseValues& raw_piece) const;
    PhaseValues apply(const PLIC::ConvexCellGeometry<Mode>& geometry, const PLIC::PolyhedralInterface& plane,
        std::array<bool, 2>* geometric_presence = nullptr, PhaseValue* fit_allocation = nullptr) const;
    PhaseValues history(const PLIC::ConvexCellGeometry<Mode>& geometry, const PLIC::PolyhedralInterface& plane) const;
};
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL PhaseLabel<Mode> phase_label(
    const Cell&, const PLIC::ConvexCellGeometry<Mode>&, const PLIC::PolyhedralInterface&, double tolerance);
struct AllocationProvenance
{
    long double admitted_bound = 0, fit_allocation_bound = 0, current_allocation_bound = 0;
};
template<ArithmeticMode Mode>
SIMPLEFLUID_FVM_LOCAL AllocationProvenance allocation_provenance(PhaseValue discrepancy, PhaseValue allocation,
    PhaseValue inherited, PhaseValue inherited_fit, PhaseValue closure, long double publication_bound);
} // namespace SimpleFluid::VoF::general_vof_detail
