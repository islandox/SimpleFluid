/** @file Reconstruction.cc
 * @brief Phase fitting labels, reconstruction support and observed owner reconstruction.
 */
#include "FVM/details/GeneralVoF/Reconstruction.hh"
#include <chrono>
namespace SimpleFluid::VoF::general_vof_detail
{
SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> geometric_labels(const Cell& cell)
{
    const bool gas = phase_measures(cell).gas.compare(phase_measures(cell).carrier) <= 0;
    const auto minority = gas ? phase_measures(cell).gas : phase_measures(cell).carrier;
    const auto dominant = phase_measures(cell).bulk - minority;
    if (minority.sign() < 0 || dominant.sign() < 0)
        throw std::invalid_argument("General VoF physical minority has no complementary geometric representation.");
    // This is a geometric fitting label only. Authoritative independent phase
    // inventories and their inherited occupancy error are never overwritten.
    return {phase_measures(cell).bulk, gas ? minority : dominant, gas ? dominant : minority};
}
template<ArithmeticMode Mode> PLIC::PolyhedralPhaseSample<Mode> reconstruction_sample(const Cell& cell)
{
    const bool observed_mixed =
        cell.declaration == GeneralVoFPhaseGeometry::Infer &&
        (!cell.retained_interface || (!cell.retained_interface->pure_gas && !cell.retained_interface->pure_carrier));
    if constexpr (Mode == ArithmeticMode::Exact)
    {
        const auto labels = geometric_labels(cell);
        return {PLIC::ConvexCellGeometry<Mode>::from_geometry(*cell.phase.geometry),
            detail::round_phase_value(labels.gas).value, detail::round_phase_value(labels.carrier).value, cell.gid,
            observed_mixed};
    }
    else
    {
        const auto geometry = PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry);
        const auto bulk = geometry->volume();
        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
        const auto minority = gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
        const long double dominant = bulk.value - minority;
        if (dominant < 0)
            throw std::invalid_argument("Finite VoF minority exceeds actual cell geometry.");
        return {geometry, gas ? minority : static_cast<double>(dominant),
            gas ? static_cast<double>(dominant) : minority, cell.gid, observed_mixed};
    }
}
template<ArithmeticMode Mode>
std::vector<PLIC::PolyhedralPhaseSample<Mode>> reconstruction_stencil(
    const GeneralVoF::Data& data, const Key& key, bool complete_support, std::size_t* fitting_cells)
{
    std::set<Key> visited{key}, front{key};
    std::vector<PLIC::PolyhedralPhaseSample<Mode>> result;
    std::vector<std::size_t> ring_ends;
    ring_ends.reserve(std::min(std::size_t(data.options.support_rings), data.options.maximum_stencil_cells));
    for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& center : front)
        {
            const auto found = data.topology->adjacency.find(center);
            if (found == data.topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
            {
                if (!data.cells.contains(neighbor))
                {
                    if constexpr (Mode != ArithmeticMode::Exact)
                        throw std::invalid_argument("General VoF complete reconstruction ring is missing a halo cell.");
                    else
                        continue;
                }
                if (visited.insert(neighbor).second)
                {
                    if (result.size() == data.options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF reconstruction stencil budget exhausted.");
                    if constexpr (Mode != ArithmeticMode::Exact)
                        if (result.size() == data.options.reconstruction.maximum_stencil_cells)
                            throw std::invalid_argument("Finite PLIC physical stencil budget exhausted.");
                    result.push_back(reconstruction_sample<Mode>(data.cells.at(neighbor)));
                    next.insert(neighbor);
                }
            }
        }
        front = std::move(next);
        ring_ends.push_back(result.size());
        if (front.empty())
            break;
    }
    // Gather the complete configured context before selecting a fitting ring.
    // The full span supplies phase availability; only a prefix supplies local
    // direction and tangent support. Selection never clips or inspects fit RMS.
    std::size_t selected = result.size();
    if constexpr (Mode != ArithmeticMode::Exact)
    {
        const auto target = reconstruction_sample<Mode>(data.cells.at(key));
        if (complete_support)
            (void) PLIC::PolyhedralReconstruction<Mode>::assess_stencil_geometry(
                target, result, data.options.reconstruction);
        else
            for (const auto count : ring_ends)
            {
                const std::span<const PLIC::PolyhedralPhaseSample<Mode>> prefix(result.data(), count);
                const auto geometry = PLIC::PolyhedralReconstruction<Mode>::assess_stencil_geometry(
                    target, prefix, data.options.reconstruction);
                if (!geometry.admitted())
                    continue;
                const auto support = PLIC::PolyhedralReconstruction<Mode>::assess_transport_support(
                    target, result, count, data.options.reconstruction);
                using Reason = PLIC::PolyhedralResolutionAssessment::Reason;
                if (support.reason != Reason::RankDeficientAngularSupport &&
                    support.reason != Reason::IllConditionedAngularSupport)
                {
                    // NoDirectionalSignal deliberately keeps this first local
                    // ring and its rejection. A larger ring cannot mask it.
                    selected = count;
                    break;
                }
            }
    }
    if (fitting_cells)
        *fitting_cells = selected;
    else if (!complete_support)
        result.resize(selected);
    return result;
}
namespace
{
struct OwnerReconstructionObservation
{
    GeneralVoFOwnerPlaneRequestDiagnostics* diagnostics;
    GeneralVoFWorkCell* cell;
    GeneralVoFOwnerPlaneRequestDiagnostics previous{};
    std::chrono::steady_clock::time_point started{};
    PLIC::GeometryStatistics before{};
    explicit OwnerReconstructionObservation(GeneralVoFOwnerPlaneRequestDiagnostics* value, GeneralVoFWorkCell* row)
        : diagnostics(value), cell(row)
    {
        if (diagnostics)
        {
            if (cell)
                previous = *diagnostics;
            ++diagnostics->owner_reconstruction_calls;
            before = PLIC::geometry_statistics();
            started = std::chrono::steady_clock::now();
        }
    }
    ~OwnerReconstructionObservation() noexcept
    {
        if (diagnostics)
        {
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            diagnostics->reconstruction_seconds += elapsed;
            const auto after = PLIC::geometry_statistics();
            diagnostics->geometry_volume_evaluations += after.volume_evaluations - before.volume_evaluations;
            diagnostics->geometry_intersections += after.intersections - before.intersections;
            if (cell)
            {
                cell->reconstruction.seconds += elapsed;
                cell->reconstruction.floating_operations += after.floating_operations - before.floating_operations;
                cell->reconstruction.volume_evaluations += after.volume_evaluations - before.volume_evaluations;
                cell->reconstruction.intersections += after.intersections - before.intersections;
                cell->reconstruction.constructions += after.constructions - before.constructions;
                cell->calls += diagnostics->owner_reconstruction_calls - previous.owner_reconstruction_calls;
                cell->mixed_fits += diagnostics->mixed_cell_fits - previous.mixed_cell_fits;
                cell->retained_returns += diagnostics->retained_interface_returns - previous.retained_interface_returns;
                cell->pure_returns += diagnostics->pure_declaration_returns - previous.pure_declaration_returns;
                cell->zero_phase_returns += diagnostics->zero_phase_offset_returns - previous.zero_phase_offset_returns;
            }
        }
    }
};
template<ArithmeticMode Mode>
PLIC::PolyhedralInterface reconstruct_cell_impl(const GeneralVoF::Data& data, const Key& key,
    GeneralVoFOwnerPlaneRequestDiagnostics* diagnostics, GeneralVoFWorkCell* row = nullptr)
{
    OwnerReconstructionObservation observation(diagnostics, row);
    const auto& cell = data.cells.at(key);
    ::SimpleFluid::detail::vof_failure::Scope scope("general_vof_reconstruct_cell");
    scope.record.cell = cell.gid;
    scope.record.mark(::SimpleFluid::detail::vof_failure::Field::Cell);
    scope.record.cell_owner = cell.owner;
    scope.record.mark(::SimpleFluid::detail::vof_failure::Field::CellOwner);
    scope.record.arithmetic_mode = Mode;
    scope.record.mark(::SimpleFluid::detail::vof_failure::Field::Arithmetic);
    validate_cell<Mode>(cell, data.options);
    if (cell.retained_interface)
    {
        if constexpr (Mode == ArithmeticMode::Exact)
        {
            if (!cell.retained_interface->finite_geometry)
            {
                if (diagnostics)
                    ++diagnostics->retained_interface_returns;
                return *cell.retained_interface;
            }
        }
        else
        {
            if (!cell.retained_interface->finite_geometry)
                throw std::invalid_argument("Finite VoF cannot adopt an exact-policy interface capsule.");
            if (diagnostics)
                ++diagnostics->retained_interface_returns;
            return *cell.retained_interface;
        }
    }
    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
    {
        PLIC::PolyhedralInterface result;
        result.pure_gas = cell.declaration == GeneralVoFPhaseGeometry::PureGas;
        result.pure_carrier = !result.pure_gas;
        const auto minority = result.pure_gas ? cell.phase.carrier_volume : cell.phase.gas_volume;
        result.target_volume_residual = minority;
        if constexpr (Mode == ArithmeticMode::Exact)
        {
            const auto difference =
                (result.pure_gas ? phase_measures(cell).carrier : phase_measures(cell).gas).scaled(-1);
            result.target_volume_difference = difference.enclosure();
        }
        else
        {
            result.finite_geometry = true;
            result.finite_target_volume_difference =
                fp::Arithmetic<Mode>::make_value(-static_cast<long double>(minority));
        }
        if (diagnostics)
            ++diagnostics->pure_declaration_returns;
        return result;
    }
    if (cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0)
    {
        auto result = PLIC::PolyhedralReconstruction<Mode>::solve_offset(
            reconstruction_sample<Mode>(cell), {1, 0, 0}, data.options.reconstruction);
        if (diagnostics)
            ++diagnostics->zero_phase_offset_returns;
        return result;
    }
    if (diagnostics)
        ++diagnostics->mixed_cell_fits;
    const auto target = reconstruction_sample<Mode>(cell);
    std::size_t fitting_cells = 0;
    const auto stencil = reconstruction_stencil<Mode>(data, key, false, &fitting_cells);
    if constexpr (Mode != ArithmeticMode::Exact)
    {
        const auto support = PLIC::PolyhedralReconstruction<Mode>::assess_transport_support(
            target, stencil, fitting_cells, data.options.reconstruction);
        if (!support.admitted())
            throw PLIC::PolyhedralResolutionRequired(support);
        auto result = PLIC::PolyhedralReconstruction<Mode>::reconstruct(
            target, stencil, fitting_cells, data.options.reconstruction);
        result.resolution = support;
        return result;
    }
    else
        return PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, data.options.reconstruction);
}
} // namespace
template<ArithmeticMode Mode> PLIC::PolyhedralInterface reconstruct_cell(const GeneralVoF::Data& data, const Key& key)
{
    return reconstruct_cell_impl<Mode>(data, key, nullptr);
}
template<ArithmeticMode Mode> std::map<Key, PLIC::PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data)
{
    std::map<Key, PLIC::PolyhedralInterface> result;
    for (const auto& key : data.topology->order)
        result.emplace(key, reconstruct_cell<Mode>(data, key));
    return result;
}
std::map<Key, PLIC::PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data)
{
    if (data.options.arithmetic_mode == ArithmeticMode::Exact)
        return reconstruct_local<ArithmeticMode::Exact>(data);
    if (data.options.arithmetic_mode == ArithmeticMode::Ordinary)
        return reconstruct_local<ArithmeticMode::Ordinary>(data);
    return reconstruct_local<ArithmeticMode::BoundedCertification>(data);
}
PLIC::PolyhedralInterface reconstruct_observed_ordinary_cell(const GeneralVoF::Data& data, const Key& key,
    GeneralVoFOwnerPlaneRequestDiagnostics* diagnostics, GeneralVoFWorkCell* row)
{
    return reconstruct_cell_impl<ArithmeticMode::Ordinary>(data, key, diagnostics, row);
}
template PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary> reconstruction_sample<ArithmeticMode::Ordinary>(
    const Cell&);
template std::vector<PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary>>
reconstruction_stencil<ArithmeticMode::Ordinary>(const GeneralVoF::Data&, const Key&, bool, std::size_t*);
template PLIC::PolyhedralInterface reconstruct_cell<ArithmeticMode::Ordinary>(const GeneralVoF::Data&, const Key&);
template std::map<Key, PLIC::PolyhedralInterface> reconstruct_local<ArithmeticMode::Ordinary>(const GeneralVoF::Data&);
template PLIC::PolyhedralPhaseSample<ArithmeticMode::BoundedCertification>
reconstruction_sample<ArithmeticMode::BoundedCertification>(const Cell&);
template std::vector<PLIC::PolyhedralPhaseSample<ArithmeticMode::BoundedCertification>>
reconstruction_stencil<ArithmeticMode::BoundedCertification>(const GeneralVoF::Data&, const Key&, bool, std::size_t*);
template PLIC::PolyhedralInterface reconstruct_cell<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&, const Key&);
template std::map<Key, PLIC::PolyhedralInterface> reconstruct_local<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&);
template PLIC::PolyhedralPhaseSample<ArithmeticMode::Exact> reconstruction_sample<ArithmeticMode::Exact>(const Cell&);
template std::vector<PLIC::PolyhedralPhaseSample<ArithmeticMode::Exact>> reconstruction_stencil<ArithmeticMode::Exact>(
    const GeneralVoF::Data&, const Key&, bool, std::size_t*);
template PLIC::PolyhedralInterface reconstruct_cell<ArithmeticMode::Exact>(const GeneralVoF::Data&, const Key&);
template std::map<Key, PLIC::PolyhedralInterface> reconstruct_local<ArithmeticMode::Exact>(const GeneralVoF::Data&);
} // namespace SimpleFluid::VoF::general_vof_detail
