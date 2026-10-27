/** @file PhaseAllocation.tcc
 * @brief Compiled physical phase allocation and provenance arithmetic.
 */
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
using SimpleFluid::geometry::swept_volume::runtime_phases;
using SimpleFluid::geometry::swept_volume::runtime_value;
template<ArithmeticMode Mode> void PhaseSum<Mode>::add(const PhaseValues& p, int sign)
{
    using Math = fp::Arithmetic<Mode>;
    bulk.add(sign > 0 ? p.bulk : Math::negate(p.bulk));
    gas.add(sign > 0 ? p.gas : Math::negate(p.gas));
    carrier.add(sign > 0 ? p.carrier : Math::negate(p.carrier));
}

template<ArithmeticMode Mode> PhaseValue PhaseLabel<Mode>::allocation(const PhaseValues& raw_piece) const
{
    using Math = fp::Arithmetic<Mode>;
    // Uniform pure-geometry labels have the same bulk share for both
    // inventories; their centered redistribution is structurally zero.
    if (gas_bulk && carrier_bulk)
        return Math::make_value(raw_source.bulk.arithmetic_mode, 0);
    const auto bulk_share = Math::divide(raw_piece.bulk, raw_source.bulk);
    const auto gas_share = Math::divide(raw_piece.gas, raw_source.gas);
    const auto carrier_share = Math::divide(raw_piece.carrier, raw_source.carrier);
    return Math::add(Math::multiply(gas_deviation, Math::subtract(gas_share, bulk_share)),
        Math::multiply(carrier_deviation, Math::subtract(carrier_share, bulk_share)));
}

template<ArithmeticMode Mode> PhaseValues PhaseLabel<Mode>::apply(const PhaseValues& raw) const
{
    using Math = fp::Arithmetic<Mode>;
    return {raw.bulk, Math::multiply(gas_bulk ? raw.bulk : raw.gas, gas_factor),
        Math::multiply(carrier_bulk ? raw.bulk : raw.carrier, carrier_factor)};
}

template<ArithmeticMode Mode> PhaseValues PhaseLabel<Mode>::history(const PhaseValues& raw) const
{
    using Math = fp::Arithmetic<Mode>;
    return {Math::make_value(raw.bulk.arithmetic_mode, 0),
        Math::multiply(gas_bulk ? raw.bulk : raw.gas, gas_history_factor),
        Math::multiply(carrier_bulk ? raw.bulk : raw.carrier, carrier_history_factor)};
}

template<ArithmeticMode Mode>
PhaseValues PhaseLabel<Mode>::apply(const PLIC::ConvexCellGeometry<Mode>& geometry,
    const PLIC::PolyhedralInterface& plane, std::array<bool, 2>* geometric_presence, PhaseValue* fit_allocation) const
{
    using Math = fp::Arithmetic<Mode>;
    const auto bulk = runtime_value<Mode>(geometry.volume());
    const auto raw =
        (plane.pure_gas || plane.pure_carrier)
            ? PhaseValues{bulk, Math::make_value(bulk.arithmetic_mode, 0), Math::make_value(bulk.arithmetic_mode, 0)}
            : runtime_phases<Mode>(geometry.phases(plane.gas));
    if (geometric_presence)
    {
        // Only exact geometric absence retains a pure hint. An unresolved
        // raw interval cannot prove absence, even if a label factor is zero.
        (*geometric_presence)[0] =
            plane.pure_gas || (!plane.pure_carrier && (raw.gas.value != 0 || raw.gas.absolute_error != 0));
        (*geometric_presence)[1] =
            plane.pure_carrier || (!plane.pure_gas && (raw.carrier.value != 0 || raw.carrier.absolute_error != 0));
    }
    if (fit_allocation)
        *fit_allocation = allocation(raw);
    return {bulk, Math::multiply(gas_bulk ? bulk : raw.gas, gas_factor),
        Math::multiply(carrier_bulk ? bulk : raw.carrier, carrier_factor)};
}

template<ArithmeticMode Mode>
PhaseValues PhaseLabel<Mode>::history(
    const PLIC::ConvexCellGeometry<Mode>& geometry, const PLIC::PolyhedralInterface& plane) const
{
    using Math = fp::Arithmetic<Mode>;
    const auto bulk = runtime_value<Mode>(geometry.volume());
    const auto raw =
        (plane.pure_gas || plane.pure_carrier)
            ? PhaseValues{bulk, Math::make_value(bulk.arithmetic_mode, 0), Math::make_value(bulk.arithmetic_mode, 0)}
            : runtime_phases<Mode>(geometry.phases(plane.gas));
    return {Math::make_value(bulk.arithmetic_mode, 0), Math::multiply(gas_bulk ? bulk : raw.gas, gas_history_factor),
        Math::multiply(carrier_bulk ? bulk : raw.carrier, carrier_history_factor)};
}

template<ArithmeticMode Mode>
PhaseValues phase_inventory(const Cell& cell, const PLIC::ConvexCellGeometry<Mode>& geometry)
{
    using Math = fp::Arithmetic<Mode>;
    constexpr auto mode = Mode;
    return {runtime_value<Mode>(geometry.volume()), Math::make_value(mode, cell.phase.gas_volume),
        Math::make_value(mode, cell.phase.carrier_volume)};
}

template<ArithmeticMode Mode>
PhaseLabel<Mode> phase_label(const Cell& cell, const PLIC::ConvexCellGeometry<Mode>& geometry,
    const PLIC::PolyhedralInterface& plane, double tolerance)
{
    using Math = fp::Arithmetic<Mode>;
    const auto inventory = phase_inventory<Mode>(cell, geometry);
    PhaseLabel<Mode> result;
    result.raw_source.bulk = inventory.bulk;
    if (plane.pure_gas || plane.pure_carrier)
    {
        result.gas_bulk = result.carrier_bulk = true;
        result.gas_factor = Math::divide(inventory.gas, inventory.bulk);
        result.carrier_factor = Math::divide(inventory.carrier, inventory.bulk);
        result.gas_history_factor =
            Math::divide(Math::make_value(inventory.bulk.arithmetic_mode, cell.finite_history.gas), inventory.bulk);
        result.carrier_history_factor =
            Math::divide(Math::make_value(inventory.bulk.arithmetic_mode, cell.finite_history.carrier), inventory.bulk);
        result.raw_source.gas = result.raw_source.carrier = Math::make_value(inventory.bulk.arithmetic_mode, 0);
        result.gas_deviation = result.carrier_deviation = Math::make_value(inventory.bulk.arithmetic_mode, 0);
        return result;
    }
    const auto raw = runtime_phases<Mode>(geometry.phases(plane.gas));
    result.raw_source = raw;
    result.gas_deviation = Math::subtract(inventory.gas, raw.gas);
    result.carrier_deviation = Math::subtract(inventory.carrier, raw.carrier);
    const auto check = [&](PhaseValue geometric, PhaseValue physical)
    {
        if (physical.value == 0)
            return Math::make_value(physical.arithmetic_mode, 0);
        const auto defect = Math::subtract(geometric, physical);
        const auto bound = phase_allowance<Mode>(physical, tolerance);
        if (fp::bound_add(std::abs(defect.value), defect.absolute_error) >
            fp::bound_add(bound, physical.absolute_error))
            throw std::invalid_argument("Finite VoF phase fit exceeds its independent phase budget.");
        return Math::divide(physical, geometric);
    };
    // The minority fit determines the plane; the dominant label additionally
    // includes the already admitted physical/native occupancy discrepancy.
    result.gas_factor = Math::divide(inventory.gas, raw.gas);
    result.carrier_factor = Math::divide(inventory.carrier, raw.carrier);
    result.gas_history_factor =
        Math::divide(Math::make_value(inventory.bulk.arithmetic_mode, cell.finite_history.gas), raw.gas);
    result.carrier_history_factor =
        Math::divide(Math::make_value(inventory.bulk.arithmetic_mode, cell.finite_history.carrier), raw.carrier);
    if (inventory.gas.value <= inventory.carrier.value)
        result.gas_factor = check(raw.gas, inventory.gas);
    else
        result.carrier_factor = check(raw.carrier, inventory.carrier);
    return result;
}

template<ArithmeticMode Mode>
AllocationProvenance allocation_provenance(PhaseValue discrepancy, PhaseValue allocation, PhaseValue inherited,
    PhaseValue inherited_fit, PhaseValue closure, long double publication_bound)
{
    using Math = fp::Arithmetic<Mode>;
    // The old fitting allowance is already inside inherited occupancy. Its
    // separate diagnostic must never be added to the admitted bound again.
    const auto unexplained = Math::subtract(discrepancy, allocation);
    const auto uniform = fp::bound_add(Math::upper(inherited), std::abs(closure.value));
    const auto arithmetic = [&]
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return publication_bound;
        else
            return unexplained.absolute_error;
    }();
    const auto unexplained_bound = fp::bound_add(uniform, arithmetic);
    if (std::abs(unexplained.value) > unexplained_bound)
        throw std::invalid_argument("Finite VoF physical occupancy is not explained by its signed fitting provenance.");
    AllocationProvenance result;
    result.current_allocation_bound = fp::bound_add(std::abs(allocation.value), allocation.absolute_error);
    result.admitted_bound = fp::bound_add(unexplained_bound, std::abs(allocation.value));
    result.fit_allocation_bound =
        fp::bound_add(Math::upper(inherited_fit), fp::bound_add(result.current_allocation_bound, publication_bound));
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail
