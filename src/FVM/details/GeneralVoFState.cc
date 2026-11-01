/** @file GeneralVoFState.cc
 * @brief Accepted-state validation, inventory diagnostics and publication checks.
 */
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
PhaseInventoryPolicy inventory_policy_for(ArithmeticMode mode)
{
    return mode == ArithmeticMode::Exact ? PhaseInventoryPolicy::ExperimentalExact : PhaseInventoryPolicy::Finite;
}
PLIC::GeometryLimits geometry_limits(ArithmeticMode mode, std::size_t bytes)
{
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = mode;
    limits.byte_budget = bytes;
    return limits;
}
template<ArithmeticMode Mode> long double phase_allowance(PLIC::DynamicGeometryValue source, double relative)
{
    using Math = fp::Arithmetic<Mode>;
    return Math::lower(Math::multiply(Math::make_value(Math::lower(source)), Math::make_value(relative)));
}
SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phase_measures(const Cell& cell)
{
    if (cell.finite_geometry)
        throw std::logic_error("Exact phase measures require an explicitly selected exact cell state.");
    if (cell.experimental_exact)
        return cell.experimental_exact->phases;
    // The selected exact route lifts finite accepted inventories for this trial.
    return {Measure::volume(*cell.phase.geometry), Measure::from_value(cell.phase.gas_volume),
        Measure::from_value(cell.phase.carrier_volume)};
}
Measure finite_bound(double value)
{
    return Measure::from_value(value);
}
namespace
{
template<ArithmeticMode Mode>
PLIC::DynamicGeometryValue finite_cell_defect(const Cell& cell, PLIC::DynamicGeometryValue actual)
{
    using Math = fp::Arithmetic<Mode>;
    fp::SumFor<Mode> defect_sum(actual.arithmetic_mode);
    defect_sum.add(Math::make_value(cell.phase.gas_volume));
    defect_sum.add(Math::make_value(cell.phase.carrier_volume));
    defect_sum.add(Math::negate(actual));
    return defect_sum.value();
}
} // namespace
template<ArithmeticMode Mode> void occupancy(Cell& cell)
{
    if constexpr (Mode != ArithmeticMode::Exact)
    {
        using namespace finite_detail;
        const auto actual = SimpleFluid::geometry::swept_volume::runtime_value<Mode>(
            PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume());
        const auto conversion =
            fp::Arithmetic<Mode>::subtract(fp::Arithmetic<Mode>::make_value(cell.native_volume), actual);
        const auto defect = finite_cell_defect<Mode>(cell, actual);
        const auto input = static_cast<long double>(cell.native_volume) * 8 * std::numeric_limits<double>::epsilon();
        if (!(cell.native_volume > 0) || !std::isfinite(cell.native_volume) ||
            std::abs(conversion.value) > input + conversion.absolute_error ||
            std::abs(defect.value - conversion.value) > input + defect.absolute_error + conversion.absolute_error)
            throw std::invalid_argument("General VoF finite phase/native-volume input exceeds its arithmetic bound.");
        cell.occupancy.difference = static_cast<double>(defect.value);
        cell.occupancy.native_conversion = static_cast<double>(conversion.value);
        cell.occupancy.phase_input_bound = upper_double(upward(input + defect.absolute_error));
        cell.occupancy.admitted_bound = upper_double(
            upward(std::abs(conversion.value) + input + conversion.absolute_error + defect.absolute_error));
        return;
    }
    if (!std::isfinite(cell.native_volume) || !(cell.native_volume > 0))
        throw std::invalid_argument("General VoF native metric is invalid.");
    const auto lifted = phase_measures(cell);
    const auto native = Measure::from_value(cell.native_volume);
    const auto defect = lifted.gas + lifted.carrier - lifted.bulk;
    const auto conversion = native - lifted.bulk;
    const auto absolute = [](const Measure& value) { return value.sign() < 0 ? value.scaled(-1) : value; };
    const auto input = native.scaled(8 * std::numeric_limits<double>::epsilon());
    if (absolute(conversion).compare(input) > 0 || absolute(defect - conversion).compare(input) > 0)
        throw std::invalid_argument("General VoF finite phase/native-volume input error exceeds its bound.");
    const auto diagnostic = detail::round_phase_residual(defect);
    const auto native_diagnostic = detail::round_phase_residual(conversion);
    const auto diagnostic_error =
        finite_bound(diagnostic.absolute_error) + finite_bound(native_diagnostic.absolute_error);
    cell.occupancy.difference = diagnostic.value;
    cell.occupancy.native_conversion = native_diagnostic.value;
    cell.occupancy.phase_input_bound = detail::phase_upper_bound(input + diagnostic_error);
    cell.occupancy.admitted_bound = detail::phase_upper_bound(absolute(conversion) + input + diagnostic_error);
}
template<ArithmeticMode Mode> void validate_cell(const Cell& cell, const GeneralVoFOptions& options)
{
    constexpr bool experimental = Mode == ArithmeticMode::Exact;
    if (cell.retained_interface)
        validate_plane_metadata(*cell.retained_interface, options.reconstruction, Mode);
    if (options.arithmetic_mode != Mode)
        throw std::invalid_argument("General VoF state arithmetic mode differs from its selected operation.");
    if (experimental != bool(cell.experimental_exact))
        throw std::invalid_argument("General VoF inventory policy/payload differs.");
    if (experimental)
    {
        detail::validate_experimental(
            *cell.experimental_exact, *cell.phase.geometry, cell.native_volume, options.byte_budget);
        const auto view = detail::experimental_view(*cell.experimental_exact, cell.native_volume);
        if (view.phases.gas != cell.phase.gas_volume || view.phases.carrier != cell.phase.carrier_volume)
            throw std::invalid_argument("Experimental exact diagnostic phase view differs.");
    }
    if constexpr (!experimental)
    {
        using namespace finite_detail;
        if (!cell.finite_geometry || cell.finite_geometry->arithmetic_mode() != options.arithmetic_mode)
            throw std::invalid_argument("Finite General VoF is missing bounded geometry authority.");
        const auto actual = SimpleFluid::geometry::swept_volume::runtime_value<Mode>(
            PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume());
        const auto defect = finite_cell_defect<Mode>(cell, actual);
        const auto& e = cell.occupancy;
        bool valid = std::isfinite(cell.phase.gas_volume) && cell.phase.gas_volume >= 0 &&
                     std::isfinite(cell.phase.carrier_volume) && cell.phase.carrier_volume >= 0 &&
                     std::isfinite(e.difference) && std::isfinite(e.native_conversion) &&
                     std::abs(defect.value) <= e.admitted_bound + defect.absolute_error;
        for (double bound :
            {e.admitted_bound, e.phase_input_bound, e.fit_bound, e.transferred_bound, e.fit_allocation_bound,
                cell.roundoff.gas, cell.roundoff.carrier, cell.finite_history.gas, cell.finite_history.carrier})
            valid = valid && std::isfinite(bound) && bound >= 0;
        if (!valid || std::uint64_t(cell.declaration) > 2)
            throw std::invalid_argument("General VoF finite occupancy provenance is invalid.");
        PLIC::PolyhedralReconstruction<Mode>::validate(reconstruction_sample<Mode>(cell), options.reconstruction);
        if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
        {
            const auto minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas ? cell.phase.carrier_volume
                                                                                       : cell.phase.gas_volume;
            const auto bound = cell.inherited
                                   ? std::max(e.admitted_bound, e.fit_bound)
                                   : actual.absolute_error + 8 * std::numeric_limits<double>::epsilon() * actual.value;
            if (minority > bound)
                throw std::invalid_argument("General VoF pure declaration contradicts finite inventory.");
        }
        return;
    }
    // Explicit exact-policy diagnostic views retain their exact provenance.
    const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> lifted{
        Measure::volume(*cell.phase.geometry), Measure::from_value(cell.phase.gas_volume),
        Measure::from_value(cell.phase.carrier_volume)};
    const auto defect = lifted.gas + lifted.carrier - lifted.bulk;
    const auto absolute = defect.sign() < 0 ? defect.scaled(-1) : defect;
    const auto& e = cell.occupancy;
    bool finite = std::isfinite(e.difference) && std::isfinite(e.native_conversion) && std::isfinite(e.fit_residual);
    for (double bound : {e.admitted_bound, e.phase_input_bound, e.fit_bound, e.transferred_bound,
             e.fit_allocation_bound, cell.roundoff.gas, cell.roundoff.carrier})
        finite = finite && std::isfinite(bound) && bound >= 0;
    if (!finite || !std::isfinite(cell.phase.gas_volume) || !std::isfinite(cell.phase.carrier_volume) ||
        cell.phase.gas_volume < 0 || cell.phase.carrier_volume < 0 ||
        detail::phase_signed_value(defect) != e.difference || absolute.compare(finite_bound(e.admitted_bound)) > 0 ||
        detail::phase_signed_value(Measure::from_value(cell.native_volume) - lifted.bulk) != e.native_conversion)
        throw std::invalid_argument("General VoF finite occupancy provenance is invalid.");
    // Physical occupancy was checked independently above. Geometry fitting
    // always uses complementary labels, not a fresh input tolerance on an
    // accepted inventory carrying explicit publication uncertainty.
    PLIC::PolyhedralReconstruction<ArithmeticMode::Exact>::validate(
        reconstruction_sample<ArithmeticMode::Exact>(cell), options.reconstruction);
    if (std::uint64_t(cell.declaration) > 2)
        throw std::invalid_argument("General VoF invalid phase geometry declaration.");
    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
    {
        const double minority =
            cell.declaration == GeneralVoFPhaseGeometry::PureGas ? cell.phase.carrier_volume : cell.phase.gas_volume;
        const auto metric = cell.phase.geometry->moments().volume;
        const double inherited_bound = std::max(e.admitted_bound, e.fit_bound);
        if (experimental)
        {
            const auto& x = *cell.experimental_exact;
            const auto actual = cell.declaration == GeneralVoFPhaseGeometry::PureGas ? x.phases.carrier : x.phases.gas;
            const auto bound =
                x.error.admitted_bound.compare(x.error.fit_bound) >= 0 ? x.error.admitted_bound : x.error.fit_bound;
            if (actual.compare(bound) > 0)
                throw std::invalid_argument("Experimental pure declaration contradicts exact inventory.");
        }
        if ((!experimental && cell.inherited && minority > inherited_bound) ||
            (!experimental && !cell.inherited &&
                minority >
                    metric.binary64_absolute_error() + 8 * std::numeric_limits<double>::epsilon() * metric.value))
            throw std::invalid_argument("General VoF pure declaration contradicts finite inventory.");
    }
}
SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phases(
    const SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>& fragment,
    const PLIC::PolyhedralInterface& p)
{
    if (p.pure_gas)
        return {fragment.volume(), fragment.volume(), {}};
    if (p.pure_carrier)
        return {fragment.volume(), {}, fragment.volume()};
    return fragment.phases(p.gas);
}
GeneralVoFPhaseEnclosure diagnostic(
    const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& measures)
{
    return {measures.bulk.enclosure(), measures.gas.enclosure(), measures.carrier.enclosure()};
}
double output(const Measure& measure)
{
    if (measure.sign() < 0)
        throw std::invalid_argument("General VoF phase measure is negative.");
    return measure.sign() ? measure.certified().as_double() : 0;
}
void refresh_experimental_catalog(GeneralVoF::Data& data)
{
    if (data.options.arithmetic_mode != ArithmeticMode::Exact)
    {
        data.experimental_catalog.reset();
        return;
    }
    wire::Budget budget{data.options.byte_budget};
    if (data.topology->order.size() >
        budget.limit / sizeof(SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>))
        throw std::invalid_argument("Experimental owned inventory catalog exceeds budget.");
    budget.add(
        data.topology->order.size() * sizeof(SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>));
    auto catalog =
        std::make_shared<std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>>>();
    catalog->reserve(data.topology->order.size());
    for (const auto& key : data.topology->order)
    {
        const auto& cell = data.cells.at(key);
        if (!cell.experimental_exact)
            throw std::invalid_argument("Missing experimental owned inventory.");
        catalog->push_back(cell.experimental_exact->phases);
    }
    data.experimental_catalog = std::move(catalog);
}
void current(const GeneralVoF::Data& data)
{
    if (!data.hierarchy->geometry_unchanged() || data.hierarchy->observation() != data.observation ||
        !data.supplied_hierarchy->geometry_unchanged() ||
        data.supplied_hierarchy->observation() != data.supplied_observation)
        throw std::invalid_argument("General VoF source geometry/control is stale.");
}
template void occupancy<ArithmeticMode::Ordinary>(Cell&);
template void validate_cell<ArithmeticMode::Ordinary>(const Cell&, const GeneralVoFOptions&);
template void occupancy<ArithmeticMode::BoundedCertification>(Cell&);
template void validate_cell<ArithmeticMode::BoundedCertification>(const Cell&, const GeneralVoFOptions&);
template void occupancy<ArithmeticMode::Exact>(Cell&);
template void validate_cell<ArithmeticMode::Exact>(const Cell&, const GeneralVoFOptions&);
template long double phase_allowance<ArithmeticMode::Ordinary>(PLIC::DynamicGeometryValue, double);
template long double phase_allowance<ArithmeticMode::BoundedCertification>(PLIC::DynamicGeometryValue, double);
} // namespace SimpleFluid::VoF::general_vof_detail
