/** @file GeneralVoFState.cc
 * @brief Shared accepted-state validation, codecs, reconstruction and halo support.
 */
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.hh"
#include "FVM/details/GeneralVoF/WorkCapture.hh"
#include "FVM/details/GeneralVoF/PlaneOrientation.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "FVM/details/GeneralVoF/PhaseCleanup.hh"
#include <chrono>
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
        const auto actual =
            SimpleFluid::geometry::swept_volume::runtime_value<Mode>(PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume());
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
        cell.occupancy.admitted_bound = upper_double(upward(std::abs(conversion.value) + input +
            conversion.absolute_error + defect.absolute_error));
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
        const auto actual =
            SimpleFluid::geometry::swept_volume::runtime_value<Mode>(PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume());
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
            const auto minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas ?
                cell.phase.carrier_volume : cell.phase.gas_volume;
            const auto bound = cell.inherited ? std::max(e.admitted_bound, e.fit_bound) :
                actual.absolute_error + 8 * std::numeric_limits<double>::epsilon() * actual.value;
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
void real(wire::Writer& w, double value)
{
    w.word(std::bit_cast<std::uint64_t>(value));
}
double real(wire::Reader& r)
{
    return std::bit_cast<double>(r.word());
}
void finite_real(wire::Writer& w, long double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Finite VoF wire value is nonfinite.");
    int exponent = 0;
    const auto mantissa = std::frexp(value, &exponent);
    const auto high = static_cast<double>(mantissa);
    const auto low = static_cast<double>(mantissa - high);
    if (static_cast<long double>(high) + low != mantissa)
        throw std::invalid_argument("Finite VoF wire significand exceeds its fixed representation.");
    real(w, high);
    real(w, low);
    w.word(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
}
long double finite_real(wire::Reader& r)
{
    const auto high = real(r), low = real(r);
    const auto exponent = std::bit_cast<std::int64_t>(r.word());
    if (!std::isfinite(high) || !std::isfinite(low) || exponent < std::numeric_limits<int>::min() ||
        exponent > std::numeric_limits<int>::max())
        throw std::invalid_argument("Finite VoF wire scalar is malformed.");
    const auto result = std::scalbn(static_cast<long double>(high) + low, int(exponent));
    if (!std::isfinite(result) || ((high != 0 || low != 0) && result == 0))
        throw std::invalid_argument("Finite VoF wire scalar exceeds qualified range.");
    return result;
}
void finite_value(wire::Writer& w, PLIC::DynamicGeometryValue v)
{
    finite_detail::validate(v);
    w.word(std::uint64_t(v.arithmetic_mode));
    finite_real(w, v.value);
    finite_real(w, v.absolute_error);
}
PLIC::DynamicGeometryValue finite_value(wire::Reader& r, std::optional<ArithmeticMode> expected)
{
    const auto mode = r.word();
    if (mode > std::uint64_t(ArithmeticMode::BoundedCertification))
        throw std::invalid_argument("Finite VoF wire arithmetic mode is invalid.");
    PLIC::DynamicGeometryValue v{finite_real(r), finite_real(r), ArithmeticMode(mode)};
    finite_detail::validate(v);
    if ((v.arithmetic_mode == ArithmeticMode::Ordinary && v.absolute_error != 0) ||
        (expected && v.arithmetic_mode != *expected))
        throw std::invalid_argument("Finite VoF wire scalar arithmetic mode differs from its state.");
    return v;
}
void plane(wire::Writer& w, const PLIC::PolyhedralInterface& p)
{
    for (auto v : p.gas.normal)
        real(w, v);
    for (auto v : p.gas.anchor)
        real(w, v);
    real(w, p.gas.offset);
    w.word(p.pure_gas);
    w.word(p.pure_carrier);
    real(w, p.stencil_volume_residual);
    plane_wire_detail::boolean(w, p.stencil_volume_residual_measured);
    real(w, p.stencil_condition);
    real(w, p.orientation_step);
    w.word(p.fit_iterations);
    plane_wire_detail::orientation(w, p.orientation_method);
    w.word(p.fit_converged);
    finite_real(w, p.target_volume_numerical_error_estimate);
    w.word(p.finite_geometry);
    if (p.finite_geometry)
        finite_value(w, p.finite_target_volume_difference);
}
PLIC::PolyhedralInterface plane(wire::Reader& r, std::optional<ArithmeticMode> mode)
{
    PLIC::PolyhedralInterface p;
    for (auto& v : p.gas.normal)
        v = real(r);
    for (auto& v : p.gas.anchor)
        v = real(r);
    p.gas.offset = real(r);
    p.pure_gas = plane_wire_detail::boolean(r);
    p.pure_carrier = plane_wire_detail::boolean(r);
    p.stencil_volume_residual = real(r);
    p.stencil_volume_residual_measured = plane_wire_detail::boolean(r);
    p.stencil_condition = real(r);
    p.orientation_step = real(r);
    p.fit_iterations = r.word();
    p.orientation_method = plane_wire_detail::orientation(r);
    p.fit_converged = plane_wire_detail::boolean(r);
    p.target_volume_numerical_error_estimate = finite_real(r);
    p.finite_geometry = plane_wire_detail::boolean(r);
    if (p.finite_geometry)
        p.finite_target_volume_difference = finite_value(r, mode);
    return p;
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
template<ArithmeticMode Mode>
void retained_budget(const GeneralVoF::Data& data, std::optional<std::size_t> proposed_spatial_bytes,
    std::size_t extra_retained_bytes, std::size_t* remaining_bytes)
{
    wire::Budget budget{data.options.byte_budget};
    budget.add(extra_retained_bytes);
    std::set<const void*> geometry;
    budget.add(data.latest_phase_cleanup.capacity() * sizeof(GeneralVoFPhaseCleanupDelta));
    for (const auto& delta : data.latest_phase_cleanup)
        phase_cleanup_key_storage(delta.key, budget);
    budget.add(sizeof(data) + sizeof(GeneralVoF::Data::Topology) + data.topology->order.capacity() * sizeof(Key) +
               (data.gas.capacity() + data.carrier.capacity()) * sizeof(double) +
               data.phases.capacity() * sizeof(PhaseVolumes) +
               (data.roundoff.capacity() + data.phase_history.capacity()) * sizeof(PhaseRoundoff) +
               data.occupancy.capacity() * sizeof(PhaseError));
    if (proposed_spatial_bytes)
        budget.add(*proposed_spatial_bytes);
    else
    {
        std::lock_guard lock(data.topology->spatial_cache_mutex);
        if (data.topology->spatial_cache)
            budget.add(data.topology->spatial_cache->bytes);
    }
    std::set<const void*> exact_payloads;
    const auto charge_exact = [&](const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>& value)
    {
        if (exact_payloads.insert(value.storage_identity()).second)
            budget.add(value.storage_bytes());
    };
    if (data.experimental_catalog)
        budget.add(sizeof(*data.experimental_catalog) +
                   data.experimental_catalog->capacity() *
                       sizeof(SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>));
    for (const auto& [key, cell] : data.cells)
    {
        budget.add(sizeof(Cell) + sizeof(Key) + 8 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
        if (geometry.insert(cell.phase.geometry.get()).second)
            budget.add(cell.phase.geometry->storage_bytes());
        if constexpr (Mode != ArithmeticMode::Exact)
            if (cell.finite_geometry && geometry.insert(cell.finite_geometry.get()).second)
                budget.add(cell.finite_geometry->storage_bytes<Mode>());
        if (cell.experimental_exact)
        {
            budget.add(sizeof(ExperimentalPhaseInventory));
            const auto& x = *cell.experimental_exact;
            for (const auto* q :
                {&x.phases.bulk, &x.phases.gas, &x.phases.carrier, &x.error.difference, &x.error.admitted_bound,
                    &x.error.native_conversion, &x.error.phase_input_bound, &x.error.fit_residual, &x.error.fit_bound,
                    &x.error.transferred_bound, &x.error.fit_allocation_bound})
                charge_exact(*q);
        }
    }
    for (const auto& node : data.topology->nodes)
        budget.add(sizeof(node) + node.identity.capacity() + node.coordinate_identity.capacity());
    for (const auto& [id, point] : data.topology->finite_nodes)
        budget.add(sizeof(id) + 4 * sizeof(void*) + PLIC::dynamic_point_storage_bytes(point));
    for (const auto& [id, identity] : data.topology->construction_nodes)
        budget.add(sizeof(id) + sizeof(identity) + 4 * sizeof(void*) + identity.capacity());
    for (const auto& [id, face] : data.topology->construction_faces)
        budget.add(sizeof(id) + sizeof(face) + 4 * sizeof(void*) + face.domain.capacity() + face.surface.capacity() +
                   face.identity.capacity() + face.node_gids.capacity() * sizeof(std::uint64_t) +
                   AMR::encode_adaptive_key(face.first).size() +
                   (face.second ? AMR::encode_adaptive_key(*face.second).size() : 0));
    for (const auto& tile : data.topology->tiles)
    {
        budget.add(sizeof(tile) + tile.node_gids.capacity() * sizeof(std::uint64_t));
        if (geometry.insert(tile.geometry.get()).second)
            budget.add(tile.geometry->storage_bytes());
    }
    for (const auto& tile : data.topology->surface_tiles)
    {
        budget.add(sizeof(tile) + tile.node_gids.capacity() * sizeof(std::uint64_t) + tile.domain.capacity() +
            tile.surface.capacity() + tile.identity.capacity() + tile.boundary.capacity() +
            AMR::encode_adaptive_key(tile.first).size() + (tile.second ? AMR::encode_adaptive_key(*tile.second).size() : 0));
        if (geometry.insert(tile.geometry.get()).second) budget.add(tile.geometry->storage_bytes());
    }
    budget.add((data.topology->frontier_tiles.capacity() - data.topology->frontier_tiles.size()) *
               sizeof(AMR::AdaptiveTile));
    for (const auto& tile : data.topology->frontier_tiles)
    {
        budget.add(sizeof(tile) +
                   (tile.node_gids.capacity() + tile.second_node_gids.capacity()) * sizeof(std::uint64_t) +
                   tile.domain.capacity() + tile.surface.capacity() + tile.identity.capacity() +
                   tile.boundary.capacity() + tile.second_image_identity.capacity() +
                   AMR::encode_adaptive_key(tile.first).size() +
                   (tile.second ? AMR::encode_adaptive_key(*tile.second).size() : 0));
        if (geometry.insert(tile.geometry.get()).second)
            budget.add(tile.geometry->storage_bytes());
    }
    auto graph = [&](const auto& entries)
    {
        for (const auto& [key, values] : entries)
        {
            budget.add(sizeof(key) + sizeof(values) + 8 * sizeof(void*));
            for (const auto& value : values)
                budget.add(sizeof(value) + 4 * sizeof(void*) + AMR::encode_adaptive_key(value).size());
        }
    };
    graph(data.topology->adjacency);
    graph(data.topology->node_fans);
    for (const auto& [gid, key] : data.topology->by_gid)
        budget.add(sizeof(gid) + sizeof(Key) + 4 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
    const auto incidence = [&](const auto& entries)
    {
        for (const auto& [key, indices] : entries)
        {
            budget.add(sizeof(key) + sizeof(indices) + 4 * sizeof(void*) +
                       indices.capacity() * sizeof(std::size_t));
            if constexpr (std::is_same_v<std::decay_t<decltype(key)>, Key>)
                budget.add(AMR::encode_adaptive_key(key).size());
        }
    };
    incidence(data.topology->cell_tiles);
    incidence(data.topology->node_tiles);
    for (const auto& [node, walls] : data.topology->node_walls)
    {
        budget.add(sizeof(node) + sizeof(walls) + walls.capacity() * sizeof(GeneralVoF::Data::Wall));
        for (const auto& wall : walls)
        {
            budget.add(wall.identity.capacity());
            if (geometry.insert(wall.geometry.get()).second)
                budget.add(wall.geometry->storage_bytes());
        }
    }
    if (remaining_bytes)
        *remaining_bytes = budget.limit - budget.used;
}
// Every incident owner registers its actual cell once. The node authority
// authenticates those registrations and returns the complete canonical fan;
// geometric coincidence and whichever ghost rows happen to be resident play no
// part in defining the interpolation stencil.
void complete_node_fans(GeneralVoF::Data& data, GeneralVoF::Data::Topology& topology)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::vector<OwnerPacket> requests;
    std::map<std::uint64_t, const AMR::AdaptiveNode*> native_nodes;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::uint64_t, std::set<Key>> local;
            for (const auto& node : topology.draft.nodes)
            {
                budget.add(sizeof(decltype(native_nodes)::value_type) + 4 * sizeof(void*));
                native_nodes.emplace(node.gid, &node);
            }
            for (const auto& tile : topology.draft.tiles)
                for (auto node : tile.node_gids)
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && data.hierarchy->owned().contains(*key))
                        {
                            budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 8 * sizeof(void*));
                            local[node].insert(*key);
                        }
            for (const auto& [node, keys] : local)
            {
                wire::Writer w;
                const auto& identity = native_nodes.at(node)->identity;
                budget.packet(32 + identity.size());
                w.word(node);
                w.bytes(identity);
                w.word(keys.size());
                for (const auto& key : keys)
                {
                    const auto bytes = AMR::encode_adaptive_key(key);
                    budget.add(16 + bytes.size());
                    w.bytes(bytes);
                    w.word(data.hierarchy->owned().at(key).revision);
                }
                requests.push_back({int(node % std::uint64_t(comm->getSize())), std::move(w.data)});
                topology.node_fans.emplace(node, std::set<Key>{});
            }
        });
    auto received = exchange_owner_packets(requests, comm, limit);
    struct Registration
    {
        std::uint64_t node, revision;
        Key key;
        int peer;
    };
    std::vector<Registration> registrations;
    std::vector<Key> keys;
    std::map<std::uint64_t, std::set<int>> subscribers;
    std::map<std::uint64_t, AMR::AdaptiveBytes> identities;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                const auto node = r.word();
                auto identity = r.bytes();
                budget.add(identity.size() + sizeof(identity) + 8 * sizeof(void*));
                if (node % std::uint64_t(comm->getSize()) != std::uint64_t(comm->getRank()) ||
                    !subscribers[node].insert(packet.peer).second)
                    throw std::invalid_argument("General VoF duplicate or misrouted node fan registration.");
                const auto [it, fresh] = identities.emplace(node, identity);
                if (!fresh && it->second != identity)
                    throw std::invalid_argument("General VoF node fan identity differs.");
                const auto count = r.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF node fan budget exhausted.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto bytes = r.bytes();
                    budget.add(bytes.size() + sizeof(Registration) + sizeof(Key));
                    auto key = AMR::decode_adaptive_key(bytes);
                    const auto revision = r.word();
                    registrations.push_back({node, revision, key, packet.peer});
                    keys.push_back(std::move(key));
                }
                r.done();
            }
        });
    const auto owners = data.hierarchy->lookup_cells(keys);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::uint64_t, std::set<Key>> fans;
            for (std::size_t i = 0; i < registrations.size(); ++i)
            {
                const auto& entry = registrations[i];
                if (owners[i].owner != entry.peer || owners[i].revision != entry.revision ||
                    !fans[entry.node].insert(entry.key).second)
                    throw std::invalid_argument("General VoF unauthenticated node fan cell.");
                if (fans[entry.node].size() > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF complete node fan budget exhausted.");
            }
            for (const auto& [node, fan] : fans)
                for (auto peer : subscribers.at(node))
                {
                    wire::Writer w;
                    budget.packet(24 + identities.at(node).size());
                    w.word(node);
                    w.bytes(identities.at(node));
                    w.word(fan.size());
                    for (const auto& key : fan)
                    {
                        const auto bytes = AMR::encode_adaptive_key(key);
                        budget.add(8 + bytes.size());
                        w.bytes(bytes);
                    }
                    replies.push_back({peer, std::move(w.data)});
                }
        });
    auto returned = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(native_nodes.size() * (sizeof(decltype(native_nodes)::value_type) + 4 * sizeof(void*)));
            std::set<std::uint64_t> seen;
            for (const auto& packet : returned.packets)
            {
                wire::Reader r{packet.bytes};
                const auto node = r.word();
                const auto identity = r.bytes();
                if (!topology.node_fans.contains(node) || !seen.insert(node).second ||
                    packet.peer != int(node % std::uint64_t(comm->getSize())))
                    throw std::invalid_argument("General VoF unexpected node fan reply.");
                const auto native = native_nodes.find(node);
                if (native == native_nodes.end() || native->second->identity != identity)
                    throw std::invalid_argument("General VoF node fan reply identity differs.");
                const auto count = r.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF node fan reply budget exhausted.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    const auto bytes = r.bytes();
                    budget.add(bytes.size() + sizeof(Key) + 8 * sizeof(void*));
                    if (!topology.node_fans.at(node).insert(AMR::decode_adaptive_key(bytes)).second)
                        throw std::invalid_argument("General VoF duplicate node fan member.");
                }
                r.done();
            }
            if (seen.size() != topology.node_fans.size())
                throw std::invalid_argument("General VoF incomplete canonical node fan.");
        });
}
std::uint64_t transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::uint64_t id = 0;
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                static std::uint64_t next = 0;
                if (next == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("General VoF transaction identity exhausted.");
                id = ++next;
            }
        });
    Teuchos::broadcast(*comm, 0, 1, &id);
    return id;
}
void agree_transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm, std::uint64_t id)
{
    std::uint64_t low = 0, high = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &id, &low);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &id, &high);
    wire::collective(*comm,
        [&]
        {
            if (!id || low != high)
                throw std::invalid_argument("General VoF mixed collective trial controls.");
        });
}
void agree_options(const GeneralVoFOptions& options, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{options.byte_budget};
            if (std::uint64_t(options.arithmetic_mode) > std::uint64_t(ArithmeticMode::Exact))
                throw std::invalid_argument("Unknown General VoF arithmetic mode.");
            if (std::uint64_t(options.transport_backend) > std::uint64_t(GeneralVoFTransportBackend::GeometricFace) ||
                (options.transport_backend != GeneralVoFTransportBackend::SweptPartition &&
                    options.arithmetic_mode != ArithmeticMode::Ordinary))
                throw std::invalid_argument("Face transport currently supports Ordinary arithmetic only.");
            const auto ratio = options.reconstruction.minimum_optimized_phase_length_ratio;
            if (!std::isfinite(ratio) || ratio < 0 || ratio > 1)
                throw std::invalid_argument("General VoF optimized phase length ratio must be finite in [0,1].");
            if (!options.maximum_face_geometry_operations)
                throw std::invalid_argument("General VoF face geometry operation limit must be positive.");
            validate_phase_cleanup_options(options);
            budget.packet(26 * 8);
            wire::Writer w;
            w.word(7); // Includes optional source-pinned balance work capture.
            for (auto value : {std::uint64_t(options.maximum_stencil_cells), std::uint64_t(options.maximum_halo_cells),
                     std::uint64_t(options.support_rings), std::uint64_t(options.arithmetic_mode),
                     std::uint64_t(options.transport_backend), std::uint64_t(options.reconstruction.measure_only),
                     std::uint64_t(options.reconstruction.maximum_offset_iterations),
                     std::uint64_t(options.reconstruction.maximum_fit_iterations),
                     std::uint64_t(options.reconstruction.maximum_stencil_cells),
                     std::uint64_t(options.reconstruction.maximum_clip_evaluations),
                     std::uint64_t(options.sweep.maximum_simplices), std::uint64_t(options.sweep.maximum_atoms),
                     std::uint64_t(options.sweep.maximum_pair_tests),
                     std::uint64_t(options.maximum_face_geometry_operations),
                     std::uint64_t(options.record_balance_work)})
                w.word(value);
            real(w, options.phase_cleanup.relative_cutoff);
            for (double value : options.phase_cleanup.maximum_absolute_volume)
                real(w, value);
            real(w, options.reconstruction.volume_tolerance);
            real(w, options.reconstruction.fit_tolerance);
            real(w, options.reconstruction.maximum_condition);
            real(w, options.reconstruction.orientation_tolerance);
            real(w, options.reconstruction.maximum_model_residual);
            real(w, options.reconstruction.minimum_phase_support_fraction);
            real(w, options.reconstruction.minimum_optimized_phase_length_ratio);
            packets.push_back({0, std::move(w.data)});
        });
    const auto received = exchange_owner_packets(packets, comm, options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                if (received.packets.size() != std::size_t(comm->getSize()))
                    throw std::invalid_argument("General VoF missing algorithm controls.");
                for (const auto& packet : received.packets)
                    if (packet.bytes != received.packets.front().bytes)
                        throw std::invalid_argument("General VoF algorithm controls disagree across ranks.");
            }
        });
}
template<ArithmeticMode Mode> void import_phases(GeneralVoF::Data& data)
{
    auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != comm->getRank())
                {
                    const auto encoded = AMR::encode_adaptive_key(key);
                    budget.packet(encoded.size() + 16);
                    wire::Writer w;
                    w.bytes(encoded);
                    w.word(cell.revision);
                    requests.push_back({cell.owner, std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto bytes = r.bytes();
                auto key = AMR::decode_adaptive_key(bytes);
                auto revision = r.word();
                r.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != comm->getRank() || cell.revision != revision)
                    throw std::invalid_argument("General VoF stale phase request.");
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                w.word(Mode == ArithmeticMode::Ordinary ? 4 : 3);
                w.word(std::uint64_t(data.options.arithmetic_mode));
                w.word(cell.canonical_face_count);
                finite_real(w, cell.ordinary_quota_scale);
                real(w, cell.phase.gas_volume);
                real(w, cell.phase.carrier_volume);
                w.word(std::uint64_t(cell.declaration));
                w.word(cell.inherited);
                if constexpr (Mode == ArithmeticMode::Ordinary)
                    w.word(cell.retained_interface && !cell.retained_interface->pure_gas &&
                           !cell.retained_interface->pure_carrier);
                real(w, cell.roundoff.gas);
                real(w, cell.roundoff.carrier);
                real(w, cell.finite_history.gas);
                real(w, cell.finite_history.carrier);
                for (const auto* value :
                    {&cell.occupancy.difference, &cell.occupancy.admitted_bound, &cell.occupancy.native_conversion,
                        &cell.occupancy.phase_input_bound, &cell.occupancy.fit_residual, &cell.occupancy.fit_bound,
                        &cell.occupancy.transferred_bound, &cell.occupancy.fit_allocation_bound})
                    real(w, *value);
                if (data.options.arithmetic_mode == ArithmeticMode::Exact)
                {
                    if (!cell.experimental_exact)
                        throw std::invalid_argument("Missing exact owner phase state.");
                    const auto encoded = detail::encode_experimental(*cell.experimental_exact, limit);
                    budget.add(8 + encoded.size());
                    w.bytes(encoded);
                }
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !seen.insert(key).second)
                    throw std::invalid_argument("General VoF unauthenticated phase reply.");
                if (r.word() != (Mode == ArithmeticMode::Ordinary ? 4 : 3) ||
                    r.word() != std::uint64_t(data.options.arithmetic_mode))
                    throw std::invalid_argument("General VoF ghost inventory policy differs.");
                const auto count = r.word();
                const auto quota = finite_real(r);
                if (!count || count > limit / sizeof(std::size_t) || !std::isfinite(quota) || quota < 0 ||
                    (data.options.arithmetic_mode == ArithmeticMode::Ordinary && !(quota > 0)))
                    throw std::invalid_argument("General VoF owner face quota is malformed.");
                cell.canonical_face_count = static_cast<std::size_t>(count);
                cell.ordinary_quota_scale = quota;
                cell.phase.gas_volume = real(r);
                cell.phase.carrier_volume = real(r);
                cell.declaration = GeneralVoFPhaseGeometry(r.word());
                cell.inherited = r.word();
                if constexpr (Mode == ArithmeticMode::Ordinary)
                {
                    const auto requires_interface = r.word();
                    if (requires_interface > 1)
                        throw std::invalid_argument("General VoF malformed owner interface requirement.");
                    cell.requires_interface = requires_interface;
                }
                cell.roundoff.gas = real(r);
                cell.roundoff.carrier = real(r);
                cell.finite_history.gas = real(r);
                cell.finite_history.carrier = real(r);
                for (auto* value :
                    {&cell.occupancy.difference, &cell.occupancy.admitted_bound, &cell.occupancy.native_conversion,
                        &cell.occupancy.phase_input_bound, &cell.occupancy.fit_residual, &cell.occupancy.fit_bound,
                        &cell.occupancy.transferred_bound, &cell.occupancy.fit_allocation_bound})
                    *value = real(r);
                if (data.options.arithmetic_mode == ArithmeticMode::Exact)
                    cell.experimental_exact = detail::decode_experimental(r.bytes(), limit);
                else
                    cell.experimental_exact.reset();
                r.done();
                validate_cell<Mode>(cell, data.options);
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete phase replies.");
        });
}
template<ArithmeticMode Mode> PLIC::PolyhedralPhaseSample<Mode> reconstruction_sample(const Cell& cell)
{
    const bool observed_mixed = cell.declaration == GeneralVoFPhaseGeometry::Infer &&
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
        return {
            geometry, gas ? minority : static_cast<double>(dominant), gas ? static_cast<double>(dominant) : minority,
            cell.gid, observed_mixed};
    }
}
template<ArithmeticMode Mode>
std::vector<PLIC::PolyhedralPhaseSample<Mode>> reconstruction_stencil(
    const GeneralVoF::Data& data, const Key& key, bool complete_support)
{
    std::set<Key> visited{key}, front{key};
    std::vector<PLIC::PolyhedralPhaseSample<Mode>> result;
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
                    result.push_back(reconstruction_sample<Mode>(data.cells.at(neighbor)));
                    next.insert(neighbor);
                }
            }
        }
        front = std::move(next);
        // Select a complete canonical-face ring before measuring any plane
        // residual. Keep the separately retained full halo for other consumers.
        if constexpr (Mode != ArithmeticMode::Exact)
        {
            const auto geometry = PLIC::PolyhedralReconstruction<Mode>::assess_stencil_geometry(
                reconstruction_sample<Mode>(data.cells.at(key)), result, data.options.reconstruction);
            if (!complete_support && geometry.admitted())
            {
                const auto support = PLIC::PolyhedralReconstruction<Mode>::assess_transport_support(
                    reconstruction_sample<Mode>(data.cells.at(key)), result, data.options.reconstruction);
                using Reason = PLIC::PolyhedralResolutionAssessment::Reason;
                if (support.reason != Reason::RankDeficientAngularSupport &&
                    support.reason != Reason::IllConditionedAngularSupport)
                    break;
            }
        }
    }
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
PLIC::PolyhedralInterface reconstruct_cell_impl(
    const GeneralVoF::Data& data, const Key& key, GeneralVoFOwnerPlaneRequestDiagnostics* diagnostics,
    GeneralVoFWorkCell* row = nullptr)
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
    const auto stencil = reconstruction_stencil<Mode>(data, key);
    if constexpr (Mode != ArithmeticMode::Exact)
    {
        const auto support = PLIC::PolyhedralReconstruction<Mode>::assess_transport_support(
            target, stencil, data.options.reconstruction);
        if (!support.admitted())
            throw PLIC::PolyhedralResolutionRequired(support);
        auto result = PLIC::PolyhedralReconstruction<Mode>::reconstruct(target, stencil, data.options.reconstruction);
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
template<ArithmeticMode Mode> std::map<Key, PLIC::PolyhedralInterface> import_planes(const GeneralVoF::Data& data)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::map<Key, PLIC::PolyhedralInterface> result;
    wire::collective(*comm,
        [&]
        {
            current(data);
            result = reconstruct_local<Mode>(data);
        });
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != comm->getRank())
                {
                    auto bytes = AMR::encode_adaptive_key(key);
                    budget.packet(bytes.size() + 16);
                    wire::Writer w;
                    w.bytes(bytes);
                    w.word(cell.revision);
                    requests.push_back({cell.owner, std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto bytes = r.bytes();
                auto key = AMR::decode_adaptive_key(bytes);
                auto revision = r.word();
                r.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != comm->getRank() || cell.revision != revision)
                    throw std::invalid_argument("General VoF stale plane request.");
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                plane(w, result.at(key));
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                const auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !seen.insert(key).second)
                    throw std::invalid_argument("General VoF unauthenticated plane reply.");
                auto fitted = plane(r, data.options.arithmetic_mode == ArithmeticMode::Exact ?
                    std::optional<ArithmeticMode>{} : data.options.arithmetic_mode);
                if constexpr (Mode != ArithmeticMode::Exact)
                {
                    if (!fitted.finite_geometry)
                        throw std::invalid_argument("Finite VoF imported an exact interface authority.");
                    if (!fitted.pure_gas && !fitted.pure_carrier)
                    {
                        const auto amounts = SimpleFluid::geometry::swept_volume::runtime_phases<Mode>(
                            PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->phases(fitted.gas));
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        fitted.finite_target_volume_difference =
                            fp::Arithmetic<Mode>::subtract(gas ? amounts.gas : amounts.carrier,
                                {gas ? cell.phase.gas_volume : cell.phase.carrier_volume, 0});
                        fitted.target_volume_residual =
                            static_cast<double>(fitted.finite_target_volume_difference.value);
                    }
                }
                else if (!fitted.pure_gas && !fitted.pure_carrier)
                {
                    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                    const auto measures =
                        SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(
                            *cell.phase.geometry)
                            ->phases(fitted.gas);
                    const auto residual = (gas ? measures.gas : measures.carrier) -
                                          Measure::from_value(gas ? cell.phase.gas_volume : cell.phase.carrier_volume);
                    fitted.target_volume_difference = residual.enclosure();
                    fitted.target_volume_residual = double(fitted.target_volume_difference.value);
                }
                result.emplace(key, std::move(fitted));
                r.done();
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete plane replies.");
        });
    return result;
}

std::map<Key, PLIC::PolyhedralInterface> import_requested_planes(const GeneralVoF::Data& data,
    const std::set<Key>& requested, const PLIC::GeometryPacketContext& context, geometry::swept_volume::Work& work,
    GeneralVoFOwnerPlaneRequestDiagnostics& diagnostics, GeneralVoFWorkSample* balance_work)
{
    constexpr auto Mode = ArithmeticMode::Ordinary;
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    PlaneImportStorage storage(work, diagnostics);
    const auto reconstruct = [&](const Key& key)
    {
        auto* row = balance_cell(balance_work, key);
        auto fitted = reconstruct_cell_impl<Mode>(data, key, &diagnostics, row);
        if (row)
        {
            row->mixed = !fitted.pure_gas && !fitted.pure_carrier;
            row->fit_iterations = fitted.fit_iterations;
            row->request_payload_bytes = owner_plane_key_wire_bytes(key) + 48;
            // Same codec as a real reply; this calibration is outside reconstruction timing.
            storage.charge(216);
            {
                wire::Writer writer;
                writer.data.reserve(216);
                if (writer.data.capacity() > 216)
                    storage.charge(writer.data.capacity() - 216);
                plane(writer, fitted);
                if (writer.data.size() > 216)
                    throw std::invalid_argument("General VoF plane exceeds its capture envelope.");
                row->reply_payload_bytes = row->request_payload_bytes + writer.data.size();
                if (writer.data.capacity() > 216)
                    storage.release(writer.data.capacity() - 216);
            }
            storage.release(216);
        }
        return fitted;
    };
    std::map<Key, PLIC::PolyhedralInterface> result;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (context.publication[0] != data.control ||
                context.publication[1] != data.hierarchy->topology_generation() ||
                context.publication[2] != data.hierarchy->partition_generation())
                throw std::invalid_argument("General VoF sparse plane source context is stale.");
            for (const auto& key : requested)
                if (data.cells.at(key).owner == comm->getRank())
                {
                    ++diagnostics.local_owned_requests;
                    result.emplace(key, reconstruct(key));
                }
        });
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::size_t count = 0;
            for (const auto& key : requested)
                count += data.cells.at(key).owner != comm->getRank();
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(OwnerPacket))
                throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Sparse plane requests overflow.");
            storage.charge(count * sizeof(OwnerPacket));
            requests.reserve(count);
            if (requests.capacity() > count)
                storage.charge((requests.capacity() - count) * sizeof(OwnerPacket));
            for (const auto& key : requested)
                if (const auto& cell = data.cells.at(key); cell.owner != comm->getRank())
                {
                    const auto wire_bytes = owner_plane_key_wire_bytes(key);
                    if (wire_bytes > (std::numeric_limits<std::size_t>::max() - 48) / 3)
                        throw PLIC::GeometryError(
                            PLIC::GeometryErrorCode::ResourceLimit, "Sparse plane key overflows.");
                    const auto scratch = 3 * wire_bytes;
                    storage.charge(scratch);
                    {
                        auto bytes = AMR::encode_adaptive_key(key);
                        if (bytes.size() != wire_bytes)
                            throw std::invalid_argument("Sparse plane key differs from its wire envelope.");
                        const auto payload = wire_bytes + 48;
                        budget.packet(payload);
                        storage.charge(payload);
                        wire::Writer w;
                        w.data.reserve(payload);
                        if (w.data.capacity() > payload)
                            storage.charge(w.data.capacity() - payload);
                        w.bytes(bytes);
                        w.word(cell.revision);
                        for (auto word : context.publication)
                            w.word(word);
                        requests.push_back({cell.owner, std::move(w.data)});
                    }
                    storage.release(scratch);
                }
        });
    const auto remote_requests = requests.size();
    auto incoming = exchange_owner_packets(requests, comm, storage.remaining_bytes());
    wire::collective(*comm,
        [&]
        {
            storage.observe_exchange_peak(incoming.stats.peak_scratch_bytes);
            const auto request_bytes = owner_plane_packet_storage_bytes(requests) +
                                       (requests.capacity() - requests.size()) * sizeof(OwnerPacket);
            std::vector<OwnerPacket>().swap(requests);
            storage.release(request_bytes);
            storage.charge(owner_plane_packet_storage_bytes(incoming.packets) +
                           (incoming.packets.capacity() - incoming.packets.size()) * sizeof(OwnerPacket));
        });
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            replies = owner_plane_replies(
                result, incoming.packets, context, comm->getRank(), comm->getSize(), storage, diagnostics,
                [&](const Key& key, std::uint64_t revision, int)
                {
                    const auto& cell = data.cells.at(key);
                    if (cell.owner != comm->getRank() || cell.revision != revision)
                        throw std::invalid_argument("General VoF stale plane request.");
                },
                [&](const Key& key) { return reconstruct(key); },
                [&](wire::Writer& writer, const PLIC::PolyhedralInterface& fitted)
                {
                    plane(writer, fitted);
                    budget.packet(writer.data.size());
                });
            const auto incoming_bytes = owner_plane_packet_storage_bytes(incoming.packets) +
                                        (incoming.packets.capacity() - incoming.packets.size()) * sizeof(OwnerPacket);
            std::vector<OwnerPacket>().swap(incoming.packets);
            storage.release(incoming_bytes);
        });
    auto received = exchange_owner_packets(replies, comm, storage.remaining_bytes());
    wire::collective(*comm,
        [&]
        {
            storage.observe_exchange_peak(received.stats.peak_scratch_bytes);
            // The peer-only map is already gone. Keep replies charged until
            // exchange and their destruction; account the received owners too.
            storage.charge(owner_plane_packet_storage_bytes(received.packets) +
                           (received.packets.capacity() - received.packets.size()) * sizeof(OwnerPacket));
            std::size_t key_scratch = 0;
            for (const auto& packet : received.packets)
            {
                if (packet.bytes.size() > (std::numeric_limits<std::size_t>::max() - sizeof(Key)) / 3)
                    throw PLIC::GeometryError(
                        PLIC::GeometryErrorCode::ResourceLimit, "Sparse plane reply key overflows.");
                key_scratch = std::max(key_scratch, sizeof(Key) + 3 * packet.bytes.size());
            }
            storage.charge(key_scratch);
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                const auto size = r.word();
                if (size > r.data.size() - r.offset)
                    throw std::invalid_argument("Invalid sparse plane reply key length.");
                auto key = AMR::decode_adaptive_key(r.data.subspan(r.offset, static_cast<std::size_t>(size)));
                r.offset += static_cast<std::size_t>(size);
                auto revision = r.word();
                for (auto word : context.publication)
                    if (r.word() != word)
                        throw std::invalid_argument("General VoF sparse plane context mismatch.");
                const auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !requested.contains(key) || seen.contains(key))
                    throw std::invalid_argument("General VoF unauthenticated plane reply.");
                storage.charge(
                    sizeof(decltype(seen)::value_type) + 4 * sizeof(void*) + owner_plane_key_storage_bytes(key));
                seen.insert(key);
                auto fitted =
                    plane(r, data.options.arithmetic_mode == ArithmeticMode::Exact ? std::optional<ArithmeticMode>{}
                                                                                   : data.options.arithmetic_mode);
                GeneralVoFWorkReceiver* receiver = nullptr;
                if (balance_work)
                    for (auto& row : balance_work->receivers)
                        if (row.donor.key == key)
                        {
                            receiver = &row;
                            row.mixed = !fitted.pure_gas && !fitted.pure_carrier;
                            row.request_payload_bytes = owner_plane_key_wire_bytes(key) + 48;
                            row.reply_payload_bytes = packet.bytes.size();
                            break;
                        }
                if constexpr (Mode != ArithmeticMode::Exact)
                {
                    if (!fitted.finite_geometry)
                        throw std::invalid_argument("Finite VoF imported an exact interface authority.");
                    if (!fitted.pure_gas && !fitted.pure_carrier)
                    {
                        WorkObservation receiver_observation(receiver ? &receiver->remeasurement : nullptr);
                        const auto amounts = SimpleFluid::geometry::swept_volume::runtime_phases<Mode>(
                            PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->phases(fitted.gas));
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        fitted.finite_target_volume_difference =
                            fp::Arithmetic<Mode>::subtract(gas ? amounts.gas : amounts.carrier,
                                {gas ? cell.phase.gas_volume : cell.phase.carrier_volume, 0});
                        fitted.target_volume_residual =
                            static_cast<double>(fitted.finite_target_volume_difference.value);
                    }
                }
                result.emplace(key, std::move(fitted));
                r.done();
            }
            storage.release(key_scratch);
            if (seen.size() != remote_requests)
                throw std::invalid_argument("General VoF incomplete plane replies.");
        });
    if (balance_work)
        balance_work->fully_deduplicated = diagnostics.cache_admission_denials == 0;
    return result;
}

std::map<Key, PLIC::PolyhedralInterface> import_planes(const GeneralVoF::Data& data)
{
    if (data.options.arithmetic_mode == ArithmeticMode::Exact)
        return import_planes<ArithmeticMode::Exact>(data);
    if (data.options.arithmetic_mode == ArithmeticMode::Ordinary)
        return import_planes<ArithmeticMode::Ordinary>(data);
    return import_planes<ArithmeticMode::BoundedCertification>(data);
}
template void occupancy<ArithmeticMode::Ordinary>(Cell&);
template void validate_cell<ArithmeticMode::Ordinary>(const Cell&, const GeneralVoFOptions&);
template std::map<Key, PLIC::PolyhedralInterface> import_planes<ArithmeticMode::Ordinary>(const GeneralVoF::Data&);
template PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary> reconstruction_sample<ArithmeticMode::Ordinary>(
    const Cell&);
template std::vector<PLIC::PolyhedralPhaseSample<ArithmeticMode::Ordinary>>
reconstruction_stencil<ArithmeticMode::Ordinary>(const GeneralVoF::Data&, const Key&, bool);
template PLIC::PolyhedralInterface reconstruct_cell<ArithmeticMode::Ordinary>(const GeneralVoF::Data&, const Key&);
template std::map<Key, PLIC::PolyhedralInterface> reconstruct_local<ArithmeticMode::Ordinary>(const GeneralVoF::Data&);
template void occupancy<ArithmeticMode::BoundedCertification>(Cell&);
template void validate_cell<ArithmeticMode::BoundedCertification>(const Cell&, const GeneralVoFOptions&);
template std::map<Key, PLIC::PolyhedralInterface> import_planes<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&);
template PLIC::PolyhedralPhaseSample<ArithmeticMode::BoundedCertification>
reconstruction_sample<ArithmeticMode::BoundedCertification>(const Cell&);
template std::vector<PLIC::PolyhedralPhaseSample<ArithmeticMode::BoundedCertification>>
reconstruction_stencil<ArithmeticMode::BoundedCertification>(const GeneralVoF::Data&, const Key&, bool);
template PLIC::PolyhedralInterface reconstruct_cell<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&, const Key&);
template std::map<Key, PLIC::PolyhedralInterface> reconstruct_local<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&);
template PLIC::PolyhedralPhaseSample<ArithmeticMode::Exact> reconstruction_sample<ArithmeticMode::Exact>(const Cell&);
template std::vector<PLIC::PolyhedralPhaseSample<ArithmeticMode::Exact>> reconstruction_stencil<ArithmeticMode::Exact>(
    const GeneralVoF::Data&, const Key&, bool);
template PLIC::PolyhedralInterface reconstruct_cell<ArithmeticMode::Exact>(const GeneralVoF::Data&, const Key&);
template std::map<Key, PLIC::PolyhedralInterface> reconstruct_local<ArithmeticMode::Exact>(const GeneralVoF::Data&);
template void occupancy<ArithmeticMode::Exact>(Cell&);
template void validate_cell<ArithmeticMode::Exact>(const Cell&, const GeneralVoFOptions&);
template std::map<Key, PLIC::PolyhedralInterface> import_planes<ArithmeticMode::Exact>(const GeneralVoF::Data&);
template void import_phases<ArithmeticMode::Ordinary>(GeneralVoF::Data&);
template long double phase_allowance<ArithmeticMode::Ordinary>(PLIC::DynamicGeometryValue, double);
template void import_phases<ArithmeticMode::BoundedCertification>(GeneralVoF::Data&);
template long double phase_allowance<ArithmeticMode::BoundedCertification>(PLIC::DynamicGeometryValue, double);
template void import_phases<ArithmeticMode::Exact>(GeneralVoF::Data&);
template void retained_budget<ArithmeticMode::Ordinary>(
    const GeneralVoF::Data&, std::optional<std::size_t>, std::size_t, std::size_t*);
template void retained_budget<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&, std::optional<std::size_t>, std::size_t, std::size_t*);
template void retained_budget<ArithmeticMode::Exact>(
    const GeneralVoF::Data&, std::optional<std::size_t>, std::size_t, std::size_t*);
} // namespace SimpleFluid::VoF::general_vof_detail
