/** @file GeneralVoFState.cc
 * @brief Shared accepted-state validation, codecs, reconstruction and halo support.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
PhaseInventoryPolicy inventory_policy_for(ArithmeticMode mode)
{
    return mode == ArithmeticMode::Exact ? PhaseInventoryPolicy::ExperimentalExact : PhaseInventoryPolicy::Finite;
}
Meshes::BoundedGeometryLimits geometry_limits(ArithmeticMode mode, std::size_t bytes)
{
    Meshes::BoundedGeometryLimits limits;
    limits.arithmetic_mode = mode;
    limits.byte_budget = bytes;
    return limits;
}
long double phase_allowance(Meshes::BoundedGeometryValue source, double relative)
{
    const auto evaluate = [&]<ArithmeticMode Mode>()
    {
        using Math = fp::Arithmetic<Mode>;
        return Math::lower(Math::multiply(Math::make_value(Math::lower(source)), Math::make_value(relative)));
    };
    if (source.arithmetic_mode == ArithmeticMode::Ordinary)
        return evaluate.template operator()<ArithmeticMode::Ordinary>();
    return evaluate.template operator()<ArithmeticMode::BoundedCertification>();
}
Meshes::ExactPhaseMeasures phase_measures(const Cell& cell)
{
    if (cell.experimental_exact)
        return cell.experimental_exact->phases;
    // Default policy: one-step lift only, no rational accepted history.
    return {Measure::volume(*cell.phase.geometry), Measure::from_value(cell.phase.gas_volume),
        Measure::from_value(cell.phase.carrier_volume)};
}
Measure finite_bound(double value)
{
    return Measure::from_value(value);
}
namespace
{
template<ArithmeticMode Mode> Meshes::BoundedGeometryValue finite_cell_defect(const Cell& cell,
    Meshes::BoundedGeometryValue actual)
{
    using Math = fp::Arithmetic<Mode>;
    fp::SumFor<Mode> defect_sum(actual.arithmetic_mode);
    defect_sum.add(Math::make_value(cell.phase.gas_volume));
    defect_sum.add(Math::make_value(cell.phase.carrier_volume));
    defect_sum.add(Math::negate(actual));
    return defect_sum.value();
}
Meshes::BoundedGeometryValue finite_cell_defect(const Cell& cell, Meshes::BoundedGeometryValue actual)
{
    if (actual.arithmetic_mode == ArithmeticMode::Ordinary)
        return finite_cell_defect<ArithmeticMode::Ordinary>(cell, actual);
    return finite_cell_defect<ArithmeticMode::BoundedCertification>(cell, actual);
}
Meshes::BoundedGeometryValue finite_native_conversion(const Cell& cell, Meshes::BoundedGeometryValue actual)
{
    const auto evaluate = [&]<ArithmeticMode Mode>()
    {
        using Math = fp::Arithmetic<Mode>;
        return Math::subtract(Math::make_value(cell.native_volume), actual);
    };
    if (actual.arithmetic_mode == ArithmeticMode::Ordinary)
        return evaluate.template operator()<ArithmeticMode::Ordinary>();
    return evaluate.template operator()<ArithmeticMode::BoundedCertification>();
}
} // namespace
void occupancy(Cell& cell)
{
    if (cell.finite_geometry)
    {
        using namespace finite_detail;
        const auto actual = cell.finite_geometry->volume();
        const auto conversion = finite_native_conversion(cell, actual);
        const auto defect = finite_cell_defect(cell, actual);
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
void validate_cell(const Cell& cell, const GeneralVoFOptions& options)
{
    const bool experimental = options.arithmetic_mode == ArithmeticMode::Exact;
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
    if (!experimental)
    {
        using namespace finite_detail;
        if (!cell.finite_geometry || cell.finite_geometry->arithmetic_mode() != options.arithmetic_mode)
            throw std::invalid_argument("Finite General VoF is missing bounded geometry authority.");
        const auto actual = cell.finite_geometry->volume();
        const auto defect = finite_cell_defect(cell, actual);
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
        PolyhedralReconstruction::validate(finite_sample(cell), options.reconstruction);
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
    const Meshes::ExactPhaseMeasures lifted{Measure::volume(*cell.phase.geometry),
        Measure::from_value(cell.phase.gas_volume), Measure::from_value(cell.phase.carrier_volume)};
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
    Exact::PolyhedralReconstruction::validate(geometric_sample(cell), options.reconstruction);
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
void finite_value(wire::Writer& w, Meshes::BoundedGeometryValue v)
{
    finite_detail::validate(v);
    w.word(std::uint64_t(v.arithmetic_mode));
    finite_real(w, v.value);
    finite_real(w, v.absolute_error);
}
Meshes::BoundedGeometryValue finite_value(wire::Reader& r, std::optional<ArithmeticMode> expected)
{
    const auto mode = r.word();
    if (mode > std::uint64_t(ArithmeticMode::BoundedCertification))
        throw std::invalid_argument("Finite VoF wire arithmetic mode is invalid.");
    Meshes::BoundedGeometryValue v{finite_real(r), finite_real(r), ArithmeticMode(mode)};
    finite_detail::validate(v);
    if ((v.arithmetic_mode == ArithmeticMode::Ordinary && v.absolute_error != 0) ||
        (expected && v.arithmetic_mode != *expected))
        throw std::invalid_argument("Finite VoF wire scalar arithmetic mode differs from its state.");
    return v;
}
void plane(wire::Writer& w, const PolyhedralInterface& p)
{
    for (auto v : p.gas.normal)
        real(w, v);
    for (auto v : p.gas.anchor)
        real(w, v);
    real(w, p.gas.offset);
    w.word(p.pure_gas);
    w.word(p.pure_carrier);
    real(w, p.stencil_volume_residual);
    real(w, p.stencil_condition);
    real(w, p.orientation_step);
    w.word(p.fit_iterations);
    w.word(p.fit_converged);
    w.word(p.finite_geometry);
    if (p.finite_geometry)
        finite_value(w, p.finite_target_volume_difference);
}
PolyhedralInterface plane(wire::Reader& r, std::optional<ArithmeticMode> mode)
{
    PolyhedralInterface p;
    for (auto& v : p.gas.normal)
        v = real(r);
    for (auto& v : p.gas.anchor)
        v = real(r);
    p.gas.offset = real(r);
    p.pure_gas = r.word();
    p.pure_carrier = r.word();
    p.stencil_volume_residual = real(r);
    p.stencil_condition = real(r);
    p.orientation_step = real(r);
    p.fit_iterations = r.word();
    p.fit_converged = r.word();
    p.finite_geometry = r.word();
    if (p.finite_geometry)
        p.finite_target_volume_difference = finite_value(r, mode);
    return p;
}
Meshes::ExactPhaseMeasures phases(const Meshes::ExactConvexFragment& fragment, const PolyhedralInterface& p)
{
    if (p.pure_gas)
        return {fragment.volume(), fragment.volume(), {}};
    if (p.pure_carrier)
        return {fragment.volume(), {}, fragment.volume()};
    return fragment.phases(p.gas);
}
GeneralVoFPhaseEnclosure diagnostic(const Meshes::ExactPhaseMeasures& measures)
{
    return {measures.bulk.enclosure(), measures.gas.enclosure(), measures.carrier.enclosure()};
}
double output(const Measure& measure)
{
    if (measure.sign() < 0)
        throw std::invalid_argument("General VoF phase measure is negative.");
    return measure.sign() ? measure.certified().as_double() : 0;
}
Meshes::ExactPhaseMeasures geometric_labels(const Cell& cell)
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
PolyhedralPhaseSample geometric_sample(const Cell& cell)
{
    const auto labels = geometric_labels(cell);
    return {cell.phase.geometry, detail::round_phase_value(labels.gas).value,
        detail::round_phase_value(labels.carrier).value};
}
void refresh_experimental_catalog(GeneralVoF::Data& data)
{
    if (data.options.arithmetic_mode != ArithmeticMode::Exact)
    {
        data.experimental_catalog.reset();
        return;
    }
    wire::Budget budget{data.options.byte_budget};
    if (data.topology->order.size() > budget.limit / sizeof(Meshes::ExactPhaseMeasures))
        throw std::invalid_argument("Experimental owned inventory catalog exceeds budget.");
    budget.add(data.topology->order.size() * sizeof(Meshes::ExactPhaseMeasures));
    auto catalog = std::make_shared<std::vector<Meshes::ExactPhaseMeasures>>();
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
void retained_budget(const GeneralVoF::Data& data)
{
    wire::Budget budget{data.options.byte_budget};
    std::set<const void*> geometry;
    budget.add(sizeof(data) + sizeof(GeneralVoF::Data::Topology) + data.topology->order.capacity() * sizeof(Key) +
               (data.gas.capacity() + data.carrier.capacity()) * sizeof(double) +
               data.phases.capacity() * sizeof(PhaseVolumes) +
               (data.roundoff.capacity() + data.phase_history.capacity()) * sizeof(PhaseRoundoff) +
               data.occupancy.capacity() * sizeof(PhaseError));
    std::set<const void*> exact_payloads;
    const auto charge_exact = [&](const Meshes::ExactGeometryMeasure& value)
    {
        if (exact_payloads.insert(value.storage_identity()).second)
            budget.add(value.storage_bytes());
    };
    if (data.experimental_catalog)
        budget.add(sizeof(*data.experimental_catalog) +
                   data.experimental_catalog->capacity() * sizeof(Meshes::ExactPhaseMeasures));
    for (const auto& [key, cell] : data.cells)
    {
        budget.add(sizeof(Cell) + sizeof(Key) + 8 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
        if (geometry.insert(cell.phase.geometry.get()).second)
            budget.add(cell.phase.geometry->storage_bytes());
        if (cell.finite_geometry && geometry.insert(cell.finite_geometry.get()).second)
            budget.add(cell.finite_geometry->storage_bytes());
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
        budget.add(sizeof(node) + node.identity.capacity() + node.exact_identity.capacity());
    for (const auto& [id, point] : data.topology->finite_nodes)
        budget.add(sizeof(id) + 4 * sizeof(void*) + Meshes::bounded_point_storage_bytes(point));
    for (const auto& tile : data.topology->tiles)
    {
        budget.add(sizeof(tile) + tile.node_gids.capacity() * sizeof(std::uint64_t));
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
            budget.packet(18 * 8);
            wire::Writer w;
            w.word(2); // Full arithmetic-mode protocol, distinct from legacy inventory policy.
            for (auto value : {std::uint64_t(options.maximum_stencil_cells), std::uint64_t(options.maximum_halo_cells),
                     std::uint64_t(options.support_rings), std::uint64_t(options.arithmetic_mode),
                     std::uint64_t(options.reconstruction.maximum_offset_iterations),
                     std::uint64_t(options.reconstruction.maximum_fit_iterations),
                     std::uint64_t(options.reconstruction.maximum_stencil_cells),
                     std::uint64_t(options.reconstruction.maximum_clip_evaluations),
                     std::uint64_t(options.sweep.maximum_simplices), std::uint64_t(options.sweep.maximum_atoms),
                     std::uint64_t(options.sweep.maximum_pair_tests)})
                w.word(value);
            real(w, options.reconstruction.volume_tolerance);
            real(w, options.reconstruction.fit_tolerance);
            real(w, options.reconstruction.maximum_condition);
            real(w, options.reconstruction.orientation_tolerance);
            real(w, options.reconstruction.maximum_model_residual);
            real(w, options.reconstruction.minimum_phase_support_fraction);
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
void import_phases(GeneralVoF::Data& data)
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
                w.word(3);
                w.word(std::uint64_t(data.options.arithmetic_mode));
                w.word(cell.canonical_face_count);
                finite_real(w, cell.ordinary_quota_scale);
                real(w, cell.phase.gas_volume);
                real(w, cell.phase.carrier_volume);
                w.word(std::uint64_t(cell.declaration));
                w.word(cell.inherited);
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
                if (r.word() != 3 || r.word() != std::uint64_t(data.options.arithmetic_mode))
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
                validate_cell(cell, data.options);
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete phase replies.");
        });
}
FinitePolyhedralPhaseSample finite_sample(const Cell& cell)
{
    const auto bulk = cell.finite_geometry->volume();
    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
    const auto minority = gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
    const long double dominant = bulk.value - minority;
    if (dominant < 0)
        throw std::invalid_argument("Finite VoF minority exceeds actual cell geometry.");
    return {cell.finite_geometry, gas ? minority : static_cast<double>(dominant),
        gas ? static_cast<double>(dominant) : minority};
}
std::vector<FinitePolyhedralPhaseSample> finite_stencil(const GeneralVoF::Data& data, const Key& key)
{
    std::set<Key> visited{key}, front{key};
    std::vector<FinitePolyhedralPhaseSample> result;
    for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& center : front)
        {
            const auto found = data.topology->adjacency.find(center);
            if (found == data.topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
                if (data.cells.contains(neighbor) && visited.insert(neighbor).second)
                {
                    if (result.size() == data.options.maximum_stencil_cells)
                        throw std::invalid_argument("Finite VoF reconstruction stencil budget exhausted.");
                    result.push_back(finite_sample(data.cells.at(neighbor)));
                    next.insert(neighbor);
                }
        }
        front = std::move(next);
    }
    return result;
}
PolyhedralInterface reconstruct_finite_cell(const GeneralVoF::Data& data, const Key& key)
{
    const auto& cell = data.cells.at(key);
    validate_cell(cell, data.options);
    if (cell.retained_interface)
    {
        if (!cell.retained_interface->finite_geometry)
            throw std::invalid_argument("Finite VoF cannot adopt an exact-policy interface capsule.");
        return *cell.retained_interface;
    }
    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
    {
        PolyhedralInterface result;
        result.finite_geometry = true;
        result.pure_gas = cell.declaration == GeneralVoFPhaseGeometry::PureGas;
        result.pure_carrier = !result.pure_gas;
        const auto minority = result.pure_gas ? cell.phase.carrier_volume : cell.phase.gas_volume;
        result.target_volume_residual = minority;
        result.finite_target_volume_difference = fp::make_value(data.options.arithmetic_mode,
            -static_cast<long double>(minority));
        return result;
    }
    if (cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0)
        return PolyhedralReconstruction::solve_offset(
            finite_sample(cell), {1, 0, 0}, data.options.reconstruction);
    return PolyhedralReconstruction::reconstruct(
        finite_sample(cell), finite_stencil(data, key), data.options.reconstruction);
}
std::map<Key, PolyhedralInterface> reconstruct_finite_local(const GeneralVoF::Data& data)
{
    std::map<Key, PolyhedralInterface> result;
    for (const auto& key : data.topology->order)
        result.emplace(key, reconstruct_finite_cell(data, key));
    return result;
}
std::map<Key, PolyhedralInterface> reconstruct_exact_local(const GeneralVoF::Data& data)
{
    std::map<Key, PolyhedralInterface> result;
    for (const auto& key : data.topology->order)
    {
        const auto& target = data.cells.at(key);
        validate_cell(target, data.options);
        if (target.retained_interface && !target.retained_interface->finite_geometry)
        {
            result.emplace(key, *target.retained_interface);
            continue;
        }
        if (target.declaration != GeneralVoFPhaseGeometry::Infer)
        {
            const bool gas = target.declaration == GeneralVoFPhaseGeometry::PureGas;
            const auto minority = gas ? target.phase.carrier_volume : target.phase.gas_volume;
            PolyhedralInterface p;
            p.pure_gas = gas;
            p.pure_carrier = !gas;
            const auto difference = (gas ? phase_measures(target).carrier : phase_measures(target).gas).scaled(-1);
            p.target_volume_difference = difference.enclosure();
            p.target_volume_residual = minority;
            result.emplace(key, p);
            continue;
        }
        if (target.phase.gas_volume == 0 || target.phase.carrier_volume == 0)
        {
            result.emplace(key, Exact::PolyhedralReconstruction::solve_offset(
                                    geometric_sample(target), {1, 0, 0}, data.options.reconstruction));
            continue;
        }
        std::set<Key> visited{key}, front{key};
        std::vector<PolyhedralPhaseSample> stencil;
        for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
        {
            std::set<Key> next;
            for (const auto& center : front)
            {
                const auto found = data.topology->adjacency.find(center);
                if (found == data.topology->adjacency.end())
                    continue;
                for (const auto& neighbor : found->second)
                    if (data.cells.contains(neighbor) && visited.insert(neighbor).second)
                    {
                        if (stencil.size() == data.options.maximum_stencil_cells)
                            throw std::invalid_argument("General VoF stencil budget exhausted.");
                        stencil.push_back(geometric_sample(data.cells.at(neighbor)));
                        next.insert(neighbor);
                    }
            }
            front = std::move(next);
        }
        result.emplace(key, Exact::PolyhedralReconstruction::reconstruct(
                                geometric_sample(target), stencil, data.options.reconstruction));
    }
    return result;
}
std::map<Key, PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data)
{
    return data.options.arithmetic_mode != ArithmeticMode::Exact ?
        reconstruct_finite_local(data) : reconstruct_exact_local(data);
}
std::map<Key, PolyhedralInterface> import_planes(const GeneralVoF::Data& data)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::map<Key, PolyhedralInterface> result;
    wire::collective(*comm,
        [&]
        {
            current(data);
            result = reconstruct_local(data);
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
                budget.packet(bytes.size() + 128);
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                plane(w, result.at(key));
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
                if (data.options.arithmetic_mode != ArithmeticMode::Exact)
                {
                    if (!fitted.finite_geometry)
                        throw std::invalid_argument("Finite VoF imported an exact interface authority.");
                    if (!fitted.pure_gas && !fitted.pure_carrier)
                    {
                        const auto amounts = cell.finite_geometry->phases(fitted.gas);
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        fitted.finite_target_volume_difference = finite_detail::subtract(
                            gas ? amounts.gas : amounts.carrier,
                            {gas ? cell.phase.gas_volume : cell.phase.carrier_volume, 0});
                        fitted.target_volume_residual =
                            static_cast<double>(fitted.finite_target_volume_difference.value);
                    }
                }
                else if (!fitted.pure_gas && !fitted.pure_carrier)
                {
                    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                    const auto measures =
                        Meshes::ExactConvexFragment::from_geometry(*cell.phase.geometry)->phases(fitted.gas);
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
}
