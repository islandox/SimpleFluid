/** @file Remap.tcc
 * @brief Conservative AMR remap; source and target evidence remain pinned.
 */
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
#include "FVM/details/GeneralVoF/Wire.hh"
namespace SimpleFluid::VoF
{
using fp::runtime_phases;
using fp::runtime_value;
namespace
{
template<ArithmeticMode Mode>
PhaseValue finite_remap_weighted(
    const PhaseValues& p, GeneralVoFFiniteRemapCarrier carrier, const std::array<double, 2>& density)
{
    using Math = fp::Arithmetic<Mode>;
    switch (carrier)
    {
        case GeneralVoFFiniteRemapCarrier::BulkVolume:
            return p.bulk;
        case GeneralVoFFiniteRemapCarrier::GasVolume:
            return p.gas;
        case GeneralVoFFiniteRemapCarrier::CarrierVolume:
            return p.carrier;
        case GeneralVoFFiniteRemapCarrier::MixtureMass:
            return Math::add(Math::multiply(p.gas, {density[0], 0}), Math::multiply(p.carrier, {density[1], 0}));
    }
    throw std::invalid_argument("Unknown finite remap channel carrier.");
}
void finite_remap_error(wire::Writer& w, const PhaseError& e)
{
    for (double v : {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound, e.fit_residual,
             e.fit_bound, e.transferred_bound, e.fit_allocation_bound})
        real(w, v);
}
PhaseError finite_remap_error(wire::Reader& r)
{
    PhaseError e;
    e.difference = real(r);
    e.admitted_bound = real(r);
    e.native_conversion = real(r);
    e.phase_input_bound = real(r);
    e.fit_residual = real(r);
    e.fit_bound = real(r);
    e.transferred_bound = real(r);
    e.fit_allocation_bound = real(r);
    return e;
}
// Local stage. The remap coordinator agrees the returned manifest collectively.
template<ArithmeticMode Mode>
void prepare_remap_schema(const GeneralVoF::Data& data, const GeneralVoF::Hierarchy::Plan& plan,
    const std::shared_ptr<const GeneralVoF::Hierarchy>& target, std::span<const GeneralVoFFiniteOwnedChannels> channels,
    const GeneralVoFFiniteRemapOptions& options, std::map<Key, PLIC::ConvexGeometryPtr<Mode>>& cell_geometry,
    std::map<Key, std::vector<PhaseValue>>& source_channels, std::vector<char>& manifest)
{
    using Math = fp::Arithmetic<Mode>;
    const auto comm = data.hierarchy->communicator();
    const auto limit = std::min(data.options.byte_budget, options.byte_budget);
    current(data);
    if (cell_geometry.empty())
    {
        wire::Budget geometry_budget{limit};
        for (const auto& [key, cell] : data.cells)
        {
            geometry_budget.add(sizeof(typename std::remove_reference_t<decltype(cell_geometry)>::value_type) +
                                4 * sizeof(void*) + sizeof(PLIC::ConvexCellGeometry<Mode>) +
                                AMR::encode_adaptive_key(key).size());
            cell_geometry.emplace(key, PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry));
        }
    }
    if (data.options.arithmetic_mode == ArithmeticMode::Exact || !target ||
        target->communicator().getRawPtr() != comm.getRawPtr() || !target->native_partition() || !limit ||
        !plan.collective_identity() || !std::isfinite(options.maximum_relative_target_defect) ||
        !(options.maximum_relative_target_defect > 0) || !std::isfinite(options.maximum_relative_target_fit) ||
        !(options.maximum_relative_target_fit > 0))
        throw std::invalid_argument("Finite remap requires compatible source, actual native target and options.");
    wire::Budget budget{limit};
    wire::Writer w;
    w.word(data.control);
    w.word(std::uint64_t(data.options.arithmetic_mode));
    w.word(plan.collective_identity());
    w.word(options.channels.size());
    for (double density : options.phase_density)
    {
        if (!std::isfinite(density) || !(density > 0))
            throw std::invalid_argument("Finite remap phase density must be finite and positive.");
        real(w, density);
    }
    real(w, options.maximum_relative_target_defect);
    real(w, options.maximum_relative_target_fit);
    std::set<std::string> names;
    for (const auto& channel : options.channels)
    {
        if (channel.name.empty() || !names.insert(channel.name).second ||
            static_cast<unsigned>(channel.carrier) > static_cast<unsigned>(GeneralVoFFiniteRemapCarrier::MixtureMass))
            throw std::invalid_argument("Finite remap channel schema is malformed.");
        budget.add(channel.name.size() + sizeof(GeneralVoFFiniteRemapChannel));
        w.bytes(std::vector<char>(channel.name.begin(), channel.name.end()));
        w.word(static_cast<unsigned>(channel.carrier));
        w.word(channel.nonnegative);
    }
    budget.packet(w.data.size());
    manifest = std::move(w.data);
    if (options.channels.empty() && channels.empty())
        for (const auto& key : data.topology->order)
            source_channels.emplace(key, std::vector<PhaseValue>{});
    else
        for (const auto& item : channels)
        {
            if (!data.hierarchy->owned().contains(item.key) || item.amounts.size() != options.channels.size() ||
                !source_channels.emplace(item.key, item.amounts).second)
                throw std::invalid_argument("Finite remap source channels have duplicate, foreign or missing keys.");
            budget.add(sizeof(GeneralVoFFiniteOwnedChannels) + item.amounts.size() * sizeof(PhaseValue));
            for (std::size_t i = 0; i < item.amounts.size(); ++i)
            {
                const auto input = item.amounts[i];
                fp::validate(input);
                if constexpr (Mode == ArithmeticMode::BoundedCertification)
                    if (input.arithmetic_mode != data.options.arithmetic_mode)
                        throw std::invalid_argument("Finite remap source channel arithmetic mode differs.");
                const auto v = Math::make_value(data.options.arithmetic_mode, input.value, input.absolute_error);
                source_channels.at(item.key)[i] = v;
                if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0 ||
                    (options.channels[i].nonnegative && (Math::lower(v) < 0 || v.value < 0)))
                    throw std::invalid_argument("Finite remap source channel is nonfinite or unresolved nonnegative.");
            }
        }
    if (source_channels.size() != data.topology->order.size())
        throw std::invalid_argument("Finite remap channels do not cover every source owner.");
}
} // namespace

template<ArithmeticMode Mode>
GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap_finite(const Hierarchy::Plan& plan,
    std::shared_ptr<const Hierarchy> target, std::span<const GeneralVoFFiniteOwnedChannels> channels,
    GeneralVoFFiniteRemapOptions options) const
{
    using Math = fp::Arithmetic<Mode>;
    const auto& data = *data_;
    std::map<Key, PLIC::ConvexGeometryPtr<Mode>> cell_geometry;
    const auto comm = data.hierarchy->communicator();
    const auto limit = std::min(data.options.byte_budget, options.byte_budget);
    std::map<Key, std::vector<PhaseValue>> source_channels;
    std::vector<char> manifest;
    wire::collective(*comm,
        [&]
        {
            prepare_remap_schema<Mode>(data, plan, target, channels, options, cell_geometry, source_channels, manifest);
        });
    std::uint64_t manifest_size = manifest.size();
    Teuchos::broadcast(*comm, 0, 1, &manifest_size);
    std::vector<char> agreed;
    wire::collective(*comm,
        [&]
        {
            if (manifest_size > limit || manifest_size > std::uint64_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Finite remap collective schema exceeds its payload cap.");
            agreed = comm->getRank() == 0 ? manifest : std::vector<char>(std::size_t(manifest_size));
        });
    Teuchos::broadcast(*comm, 0, int(manifest_size), agreed.data());
    wire::collective(*comm,
        [&]
        {
            if (agreed != manifest)
                throw std::invalid_argument("Finite remap schema, source publication or plan differs across ranks.");
        });
    target = data.hierarchy->apply(plan, std::move(target));
    const auto target_observation = target->observation();
    const bool ownership = target->topology_generation() == data.hierarchy->topology_generation();
    const auto catalog = detail::build_adaptive_overlap_catalog(*data.hierarchy, *target, limit);
    std::map<std::uint64_t, const AMR::AdaptiveCellBinding*> from, to;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& cell : catalog.source_cells)
            {
                budget.add(sizeof(typename decltype(from)::value_type) + 4 * sizeof(void*));
                if (!from.emplace(cell.gid, &cell).second)
                    throw std::invalid_argument("Finite remap source catalog repeats a global ID.");
            }
            for (const auto& cell : catalog.target_cells)
            {
                budget.add(sizeof(typename decltype(to)::value_type) + 4 * sizeof(void*));
                if (!to.emplace(cell.gid, &cell).second)
                    throw std::invalid_argument("Finite remap target catalog repeats a global ID.");
            }
        });
    struct Relation
    {
        Key source, target;
        std::uint64_t source_gid, source_revision, target_gid, target_revision;
        AMR::AdaptiveAncestryRelation::Kind kind;
        int target_owner;
        PLIC::ConvexGeometryPtr<Mode> overlap;
    };
    std::vector<Relation> requests;
    std::vector<Key> source_keys;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& relation : catalog.relations)
            {
                const auto& source_cell = *from.at(relation.source_gid);
                const auto& target_cell = *to.at(relation.target_gid);
                if (!target->owned().contains(target_cell.key))
                    continue;
                budget.add(sizeof(Relation) + 2 * sizeof(Key));
                source_keys.push_back(source_cell.key);
                requests.push_back({source_cell.key, target_cell.key, source_cell.gid, 0, target_cell.gid,
                    target->owned().at(target_cell.key).revision, relation.kind, comm->getRank(), {}});
            }
        });
    const auto locations = data.hierarchy->lookup_cells(source_keys);
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t i = 0; i < requests.size(); ++i)
            {
                auto& request = requests[i];
                request.source_revision = locations[i].revision;
                if (request.source_gid != locations[i].address)
                    throw std::invalid_argument("Finite remap source catalog is stale.");
                wire::Writer w;
                w.bytes(AMR::encode_adaptive_key(request.source));
                w.word(request.source_gid);
                w.word(request.source_revision);
                w.bytes(AMR::encode_adaptive_key(request.target));
                w.word(request.target_gid);
                w.word(request.target_revision);
                w.word(static_cast<unsigned>(request.kind));
                budget.packet(w.data.size());
                packets.push_back({locations[i].owner, std::move(w.data)});
            }
        });
    const auto incoming = exchange_owner_packets(packets, comm, limit);
    std::vector<Relation> columns;
    std::vector<Key> target_keys;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::set<std::pair<Key, Key>> seen;
            std::map<std::pair<Key, Key>, AMR::AdaptiveAncestryRelation::Kind> admitted_relations;
            for (const auto& relation : catalog.relations)
                admitted_relations.emplace(
                    std::pair{from.at(relation.source_gid)->key, to.at(relation.target_gid)->key}, relation.kind);
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                Relation column;
                column.source = AMR::decode_adaptive_key(r.bytes());
                column.source_gid = r.word();
                column.source_revision = r.word();
                column.target = AMR::decode_adaptive_key(r.bytes());
                column.target_gid = r.word();
                column.target_revision = r.word();
                const auto kind = r.word();
                r.done();
                if (!data.hierarchy->owned().contains(column.source) ||
                    !seen.emplace(column.source, column.target).second ||
                    kind > static_cast<unsigned>(AMR::AdaptiveAncestryRelation::Kind::ProductIntersection))
                    throw std::invalid_argument(
                        "Finite remap received duplicate, foreign or malformed overlap request.");
                column.kind = static_cast<AMR::AdaptiveAncestryRelation::Kind>(kind);
                const auto authenticated = admitted_relations.find({column.source, column.target});
                if (authenticated == admitted_relations.end() || authenticated->second != column.kind)
                    throw std::invalid_argument("Finite remap request does not match authenticated native ancestry.");
                const auto& cell = data.cells.at(column.source);
                if (cell.gid != column.source_gid || cell.revision != column.source_revision)
                    throw std::invalid_argument("Finite remap overlap source revision differs.");
                column.target_owner = packet.peer;
                budget.add(sizeof(Relation) + sizeof(Key));
                target_keys.push_back(column.target);
                columns.push_back(std::move(column));
            }
        });
    const auto target_locations = target->lookup_cells(target_keys);
    const auto target_leaves = target->fetch_leaves(target_keys, limit);
    std::map<Key, PLIC::ConvexGeometryPtr<Mode>> imported_targets;
    std::map<Key, std::vector<std::size_t>> by_source;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t i = 0; i < columns.size(); ++i)
            {
                auto& column = columns[i];
                const auto& location = target_locations[i];
                const auto& leaf = target_leaves[i];
                const auto& source_leaf = data.hierarchy->owned().at(column.source);
                if (location.owner != column.target_owner || location.address != column.target_gid ||
                    location.revision != column.target_revision || leaf.key != column.target ||
                    leaf.revision != column.target_revision || leaf.material != source_leaf.material ||
                    leaf.key.region != source_leaf.key.region)
                    throw std::invalid_argument("Finite remap target owner/material/revision differs.");
                if (!imported_targets.contains(column.target))
                {
                    PLIC::GeometryLimits limits;
                    limits.arithmetic_mode = data.options.arithmetic_mode;
                    limits.byte_budget = limit;
                    auto geometry = PLIC::ConvexCellGeometry<Mode>::from_geometry(*target->geometry(leaf), limits);
                    budget.add(geometry->storage_bytes());
                    imported_targets.emplace(column.target, std::move(geometry));
                }
                const auto& source_geometry = cell_geometry.at(column.source);
                const auto& target_geometry = imported_targets.at(column.target);
                // Native plan ancestry authenticates exact containment for immediate
                // refinement/coarsening. Numerical partition checks below still
                // establish that the supplied whole pieces form a complete cover.
                if (column.kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                    column.kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening)
                    column.overlap = source_geometry;
                else if (column.kind == AMR::AdaptiveAncestryRelation::Kind::Refinement)
                    column.overlap = target_geometry;
                else
                    column.overlap = source_geometry->intersect(*target_geometry);
                if (!column.overlap)
                    throw std::invalid_argument("Finite remap ancestry overlap is geometrically empty.");
                budget.add(sizeof(std::size_t) + column.overlap->storage_bytes());
                by_source[column.source].push_back(i);
            }
        });
    struct Output
    {
        Relation relation;
        PhaseValues phases, history;
        std::vector<PhaseValue> amounts;
        PhaseValue inherited_occupancy, fit_allocation, inherited_fit_allocation, inherited_pure_fit;
        PhaseValue inherited_phase_input, inherited_transferred;
        std::array<bool, 2> geometry_presence{};
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
        std::optional<PLIC::PolyhedralInterface> retained_interface;
    };
    std::vector<Output> outgoing;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& key : data.topology->order)
            {
                const auto& source = data.cells.at(key);
                const auto found = by_source.find(key);
                if (found == by_source.end() || found->second.empty())
                    throw std::invalid_argument("Finite remap has an uncovered source owner.");
                const auto& indices = found->second;
                std::vector<PLIC::ConvexGeometryPtr<Mode>> cover;
                for (auto i : indices)
                    cover.push_back(columns[i].overlap);
                cell_geometry.at(source.key)->certify_fragment_partition(cover, options.maximum_relative_target_defect);
                const bool whole = indices.size() == 1 &&
                                   (columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                                       columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening);
                PLIC::PolyhedralInterface fitted;
                PhaseLabel<Mode> labels;
                if (!whole)
                {
                    fitted = reconstruct_cell<Mode>(data, key);
                    labels = phase_label<Mode>(
                        source, *cell_geometry.at(key), fitted, data.options.reconstruction.volume_tolerance);
                }
                const auto old = phase_inventory<Mode>(source, *cell_geometry.at(source.key));
                // Carrier existence is a source contract even on ownership-only
                // copies; the whole-cell fast path cannot legitimize an unsupported
                // nonzero channel merely because it needs no geometric weighting.
                for (std::size_t c = 0; c < options.channels.size(); ++c)
                {
                    const auto carrier =
                        finite_remap_weighted<Mode>(old, options.channels[c].carrier, options.phase_density);
                    const auto amount = source_channels.at(key)[c];
                    if (carrier.value == 0 && carrier.absolute_error == 0)
                    {
                        if (amount.value != 0 || amount.absolute_error != 0)
                            throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                    }
                    else if (!(Math::lower(carrier) > 0))
                        throw std::invalid_argument("Finite remap source channel carrier is unresolved.");
                }
                PhaseSum<Mode> total(data.options.arithmetic_mode);
                std::vector<fp::SumFor<Mode>> channel_totals(options.channels.size());
                for (auto i : indices)
                {
                    const auto& column = columns[i];
                    Output output;
                    output.fit_allocation = Math::make_value(data.options.arithmetic_mode, 0);
                    output.relation = column;
                    if (whole)
                    {
                        output.phases = old;
                        output.geometry_presence = {source.declaration != GeneralVoFPhaseGeometry::PureCarrier,
                            source.declaration != GeneralVoFPhaseGeometry::PureGas};
                    }
                    else
                        output.phases =
                            labels.apply(*column.overlap, fitted, &output.geometry_presence, &output.fit_allocation);
                    output.history =
                        whole ? PhaseValues{Math::make_value(data.options.arithmetic_mode, 0),
                                    Math::make_value(data.options.arithmetic_mode, source.finite_history.gas),
                                    Math::make_value(data.options.arithmetic_mode, source.finite_history.carrier)}
                              : labels.history(*column.overlap, fitted);
                    output.declaration = source.declaration;
                    output.original_error = source.occupancy;
                    output.original_roundoff = source.roundoff;
                    if (ownership)
                        output.retained_interface = source.retained_interface;
                    const auto bulk_share = whole ? Math::make_value(data.options.arithmetic_mode, 1)
                                                  : Math::divide(output.phases.bulk, old.bulk);
                    output.inherited_occupancy = Math::multiply({source.occupancy.admitted_bound, 0}, bulk_share);
                    output.inherited_fit_allocation =
                        Math::multiply({source.occupancy.fit_allocation_bound, 0}, bulk_share);
                    output.inherited_pure_fit = Math::multiply({source.occupancy.fit_bound, 0}, bulk_share);
                    output.inherited_phase_input = Math::multiply({source.occupancy.phase_input_bound, 0}, bulk_share);
                    output.inherited_transferred = Math::multiply({source.occupancy.transferred_bound, 0}, bulk_share);
                    for (std::size_t c = 0; c < options.channels.size(); ++c)
                    {
                        const auto& input = source_channels.at(key)[c];
                        auto amount = Math::make_value(data.options.arithmetic_mode, 0);
                        if (whole)
                            amount = input;
                        else
                        {
                            const auto denominator =
                                finite_remap_weighted<Mode>(old, options.channels[c].carrier, options.phase_density);
                            const auto numerator = finite_remap_weighted<Mode>(
                                output.phases, options.channels[c].carrier, options.phase_density);
                            if (denominator.value == 0 && denominator.absolute_error == 0)
                            {
                                if (input.value != 0 || input.absolute_error != 0 || numerator.value != 0 ||
                                    numerator.absolute_error != 0)
                                    throw std::invalid_argument(
                                        "Finite remap nonzero channel has no physical carrier.");
                            }
                            else
                                amount = Math::multiply(input, Math::divide(numerator, denominator));
                        }
                        output.amounts.push_back(amount);
                        channel_totals[c].add(amount);
                    }
                    total.add(output.phases);
                    budget.add(sizeof(Output) + output.amounts.size() * sizeof(PhaseValue));
                    outgoing.push_back(std::move(output));
                }
                const auto delivered = total.value();
                for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
                {
                    const auto residual = Math::subtract(pair.second, pair.first);
                    const auto bound = phase_allowance<Mode>(pair.first, options.maximum_relative_target_fit);
                    if (!Math::closure_within(residual, bound))
                        throw std::invalid_argument(
                            "Finite remap source phase closure exceeds its unchanged target-fit bound.");
                }
                for (std::size_t c = 0; c < options.channels.size(); ++c)
                {
                    const auto residual = Math::subtract(channel_totals[c].value(), source_channels.at(key)[c]);
                    const auto allowed =
                        options.maximum_relative_target_fit * std::abs(source_channels.at(key)[c].value);
                    const auto nonconservative = [&]
                    {
                        if constexpr (Mode == ArithmeticMode::Ordinary)
                            return !Math::closure_within(residual, allowed);
                        else
                            return std::abs(residual.value) > residual.absolute_error;
                    }();
                    if (nonconservative)
                        throw std::invalid_argument("Finite remap weighted channel column is not conservative.");
                }
            }
        });
    packets.clear();
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& output : outgoing)
            {
                const auto& column = output.relation;
                wire::Writer w;
                w.bytes(AMR::encode_adaptive_key(column.source));
                w.word(column.source_gid);
                w.word(column.source_revision);
                w.bytes(AMR::encode_adaptive_key(column.target));
                w.word(column.target_gid);
                w.word(column.target_revision);
                finite_phases(w, output.phases);
                finite_phases(w, output.history);
                finite_value(w, output.inherited_occupancy);
                finite_value(w, output.fit_allocation);
                finite_value(w, output.inherited_fit_allocation);
                finite_value(w, output.inherited_pure_fit);
                finite_value(w, output.inherited_phase_input);
                finite_value(w, output.inherited_transferred);
                w.word(output.geometry_presence[0]);
                w.word(output.geometry_presence[1]);
                w.word(static_cast<unsigned>(output.declaration));
                w.word(output.amounts.size());
                for (auto amount : output.amounts)
                    finite_value(w, amount);
                if (ownership)
                {
                    finite_remap_error(w, output.original_error);
                    real(w, output.original_roundoff.gas);
                    real(w, output.original_roundoff.carrier);
                    w.word(bool(output.retained_interface));
                    if (output.retained_interface)
                        plane(w, *output.retained_interface);
                }
                budget.packet(w.data.size());
                packets.push_back({column.target_owner, std::move(w.data)});
            }
        });
    const auto responses = exchange_owner_packets(packets, comm, limit);
    struct Row
    {
        PhaseSum<Mode> phases, history;
        fp::SumFor<Mode> occupancy, fit_allocation, inherited_fit_allocation, pure_fit, phase_input, transferred;
        std::array<bool, 2> geometry_presence{};
        std::vector<fp::SumFor<Mode>> channels;
        std::optional<GeneralVoFPhaseGeometry> declaration;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        std::optional<PLIC::PolyhedralInterface> retained_interface;
        std::size_t pieces = 0;
    };
    std::map<Key, Row> rows;
    FiniteRemapTrial result;
    result.source_ = data_;
    result.ownership_only_ = ownership;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& column : columns)
            {
                budget.add(sizeof(GeneralVoFFiniteCommonCell) + column.overlap->storage_bytes());
                result.common_cells_.push_back({column.source, column.target, column.source_gid, column.target_gid,
                    column.target_owner, column.overlap->dynamic_geometry()});
            }
        });
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::pair<Key, Key>, std::size_t> expected;
            for (std::size_t i = 0; i < requests.size(); ++i)
                if (!expected.emplace(std::pair{requests[i].source, requests[i].target}, i).second)
                    throw std::invalid_argument("Finite remap ancestry repeats a source/target pair.");
            std::set<std::pair<Key, Key>> seen;
            for (const auto& packet : responses.packets)
            {
                wire::Reader r{packet.bytes};
                const auto sk = AMR::decode_adaptive_key(r.bytes());
                const auto sgid = r.word(), revision = r.word();
                const auto tk = AMR::decode_adaptive_key(r.bytes());
                const auto tgid = r.word(), target_revision = r.word();
                const auto item = expected.find({sk, tk});
                if (item == expected.end() || !seen.emplace(sk, tk).second)
                    throw std::invalid_argument("Finite remap output is duplicate or not an authenticated overlap.");
                const auto& request = requests[item->second];
                if (packet.peer != locations[item->second].owner || sgid != request.source_gid ||
                    revision != request.source_revision || tgid != request.target_gid ||
                    target_revision != request.target_revision)
                    throw std::invalid_argument("Finite remap result source/target ownership or revision differs.");
                const auto phases = finite_phases(r, data.options.arithmetic_mode),
                           history = finite_phases(r, data.options.arithmetic_mode);
                const auto inherited_occupancy = finite_value(r, data.options.arithmetic_mode);
                const auto allocation = finite_value(r, data.options.arithmetic_mode),
                           inherited_fit = finite_value(r, data.options.arithmetic_mode),
                           pure_fit = finite_value(r, data.options.arithmetic_mode);
                const auto phase_input = finite_value(r, data.options.arithmetic_mode),
                           transferred = finite_value(r, data.options.arithmetic_mode);
                const auto gas_geometry = r.word(), carrier_geometry = r.word();
                const auto declaration = r.word(), count = r.word();
                if (gas_geometry > 1 || carrier_geometry > 1 ||
                    declaration > static_cast<unsigned>(GeneralVoFPhaseGeometry::PureCarrier) ||
                    count != options.channels.size())
                    throw std::invalid_argument("Finite remap result schema is malformed.");
                for (auto value : {phases.bulk, phases.gas, phases.carrier, history.gas, history.carrier,
                         inherited_occupancy, inherited_fit, pure_fit, phase_input, transferred})
                    if (value.value < 0 || Math::lower(value) < 0)
                        throw std::invalid_argument("Finite remap output has unresolved nonnegative amounts.");
                auto& row = rows[tk];
                if (row.pieces++ == 0)
                {
                    row.channels.resize(options.channels.size());
                    row.declaration = static_cast<GeneralVoFPhaseGeometry>(declaration);
                }
                else if (*row.declaration != static_cast<GeneralVoFPhaseGeometry>(declaration))
                    row.declaration = GeneralVoFPhaseGeometry::Infer;
                row.phases.add(phases);
                row.history.add(history);
                row.occupancy.add(inherited_occupancy);
                row.fit_allocation.add(allocation);
                row.inherited_fit_allocation.add(inherited_fit);
                row.pure_fit.add(pure_fit);
                row.phase_input.add(phase_input);
                row.transferred.add(transferred);
                row.geometry_presence[0] = row.geometry_presence[0] || gas_geometry;
                row.geometry_presence[1] = row.geometry_presence[1] || carrier_geometry;
                for (auto& channel : row.channels)
                    channel.add(finite_value(r, data.options.arithmetic_mode));
                if (ownership)
                {
                    if (row.pieces != 1 || sk != tk)
                        throw std::invalid_argument("Finite ownership migration changes cell ancestry.");
                    row.original_error = finite_remap_error(r);
                    row.original_roundoff = {real(r), real(r)};
                    const auto retained = r.word();
                    if (retained > 1)
                        throw std::invalid_argument("Finite migration interface tag is malformed.");
                    if (retained)
                    {
                        row.retained_interface = plane(r, data.options.arithmetic_mode);
                        if (!row.retained_interface->finite_geometry)
                            throw std::invalid_argument("Finite migration cannot acquire an exact interface.");
                    }
                }
                r.done();
                budget.add(
                    sizeof(Row) + sizeof(GeneralVoFFiniteRemapRoute) + row.channels.size() * sizeof(fp::SumFor<Mode>));
                result.routes_.push_back(
                    {sk, tk, sgid, tgid, packet.peer, comm->getRank(), phases, allocation, inherited_fit});
            }
            if (seen.size() != expected.size() || rows.size() != target->owned().size())
                throw std::invalid_argument("Finite remap outputs do not cover every target owner.");
        });
    // Independently validate recipient geometry before physical publication.
    // Sparse ancestry selects candidates; a net volume sum cannot certify cover.
    std::map<Key, PLIC::ConvexGeometryPtr<Mode>> local_source_geometry;
    std::map<Key, std::vector<PLIC::ConvexGeometryPtr<Mode>>> target_covers;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& request : requests)
            {
                if (!local_source_geometry.contains(request.source))
                {
                    const auto existing = data.cells.find(request.source);
                    auto geometry =
                        existing != data.cells.end()
                            ? cell_geometry.at(request.source)
                            : PLIC::ConvexCellGeometry<Mode>::from_geometry(*from.at(request.source_gid)->geometry,
                                  geometry_limits(data.options.arithmetic_mode, limit));
                    budget.add(geometry->storage_bytes());
                    local_source_geometry.emplace(request.source, std::move(geometry));
                }
                if (!imported_targets.contains(request.target))
                {
                    auto geometry = PLIC::ConvexCellGeometry<Mode>::from_geometry(
                        *to.at(request.target_gid)->geometry, geometry_limits(data.options.arithmetic_mode, limit));
                    budget.add(geometry->storage_bytes());
                    imported_targets.emplace(request.target, std::move(geometry));
                }
                const auto& source_geometry = local_source_geometry.at(request.source);
                const auto& target_geometry = imported_targets.at(request.target);
                auto overlap = source_geometry;
                if (request.kind == AMR::AdaptiveAncestryRelation::Kind::Refinement)
                    overlap = target_geometry;
                else if (request.kind == AMR::AdaptiveAncestryRelation::Kind::ProductIntersection)
                    overlap = source_geometry->intersect(*target_geometry);
                if (!overlap)
                    throw std::invalid_argument("Finite remap recipient overlap is empty.");
                target_covers[request.target].push_back(std::move(overlap));
            }
            for (const auto& [key, leaf] : target->owned())
                imported_targets.at(key)->certify_fragment_partition(
                    target_covers.at(key), options.maximum_relative_target_defect);
        });
    // Build only geometry/maps in a private pure placeholder. Actual remapped
    // inventories and their inherited error evidence are installed together.
    std::vector<Key> order;
    std::vector<double> zero, native_volume;
    wire::collective(*comm,
        [&]
        {
            for (const auto& [key, leaf] : target->owned())
            {
                order.push_back(key);
                zero.push_back(0);
                native_volume.push_back(target->metrics(leaf).volume);
            }
        });
    const auto placeholder = std::make_shared<const GeneralVoF>(target, order, zero, native_volume, data.options);
    auto candidate = std::make_shared<Data>(*placeholder->data_);
    wire::collective(*comm,
        [&]
        {
            candidate->gas.clear();
            candidate->carrier.clear();
            candidate->phases.clear();
            candidate->roundoff.clear();
            candidate->phase_history.clear();
            candidate->occupancy.clear();
            wire::Budget budget{limit};
            for (const auto& key : candidate->topology->order)
            {
                auto& cell = candidate->cells.at(key);
                auto& row = rows.at(key);
                const auto totals = row.phases.value(), history = row.history.value();
                const auto actual =
                    runtime_value<Mode>(PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume());
                const auto closure = Math::subtract(totals.bulk, actual);
                const auto defect_bound = phase_allowance<Mode>(actual, options.maximum_relative_target_defect);
                if (!Math::closure_within(closure, defect_bound))
                    throw std::invalid_argument("Finite remap recipient coverage exceeds its geometric error budget.");
                if (!ownership)
                {
                    Math::admit_receipt(totals.gas, data.options.reconstruction.volume_tolerance, "gas");
                    Math::admit_receipt(totals.carrier, data.options.reconstruction.volume_tolerance, "carrier");
                }
                const auto gas = Math::publish(totals.gas), carrier = Math::publish(totals.carrier);
                cell.phase.gas_volume = gas.first;
                cell.phase.carrier_volume = carrier.first;
                cell.roundoff = ownership ? row.original_roundoff : PhaseRoundoff{gas.second, carrier.second};
                cell.finite_history = {
                    fp::upper_double(Math::upper(history.gas)), fp::upper_double(Math::upper(history.carrier))};
                if (!ownership)
                {
                    cell.finite_history.gas = fp::upper_double(fp::upward(cell.finite_history.gas + gas.second));
                    cell.finite_history.carrier =
                        fp::upper_double(fp::upward(cell.finite_history.carrier + carrier.second));
                }
                cell.declaration = row.declaration.value_or(GeneralVoFPhaseGeometry::Infer);
                cell.retained_interface =
                    ownership ? row.retained_interface : std::optional<PLIC::PolyhedralInterface>{};
                cell.inherited = true;
                if (ownership)
                    cell.occupancy = row.original_error;
                else
                {
                    const auto discrepancy = Math::subtract(
                        Math::add(Math::make_value(data.options.arithmetic_mode, gas.first, gas.second),
                            Math::make_value(data.options.arithmetic_mode, carrier.first, carrier.second)),
                        actual);
                    const auto inherited = row.occupancy.value();
                    const auto allocation = row.fit_allocation.value();
                    // For source deviations deltaG=G-g and deltaC=C-c and raw
                    // shares sg,sc,sb, the fitting redistribution is
                    // A=deltaG*(sg-sb)+deltaC*(sc-sb). The uniform source occupancy
                    // already travels with sb; A is independent signed producer
                    // evidence, never an allowance manufactured from this D.
                    const auto provenance = allocation_provenance<Mode>(discrepancy, allocation, inherited,
                        row.inherited_fit_allocation.value(), closure, fp::bound_add(gas.second, carrier.second));
                    const auto fit_bound = phase_allowance<Mode>(actual, options.maximum_relative_target_fit);
                    if (provenance.current_allocation_bound > fit_bound || provenance.admitted_bound > defect_bound)
                        throw std::invalid_argument(
                            "Finite remap physical occupancy exceeds its original target-defect bound.");
                    cell.occupancy.difference = static_cast<double>(discrepancy.value);
                    cell.occupancy.admitted_bound = fp::upper_double(provenance.admitted_bound);
                    cell.occupancy.phase_input_bound = fp::upper_double(
                        fp::bound_add(Math::upper(row.phase_input.value()), fp::bound_add(gas.second, carrier.second)));
                    cell.occupancy.transferred_bound =
                        fp::upper_double(fp::bound_add(Math::upper(row.transferred.value()),
                            fp::bound_add(totals.gas.absolute_error, totals.carrier.absolute_error)));
                    // Previous allocation is already part of inherited occupancy.
                    // Preserve it separately without counting it twice in admission.
                    cell.occupancy.fit_allocation_bound = fp::upper_double(provenance.fit_allocation_bound);
                    if ((cell.declaration == GeneralVoFPhaseGeometry::PureGas && row.geometry_presence[1]) ||
                        (cell.declaration == GeneralVoFPhaseGeometry::PureCarrier && row.geometry_presence[0]))
                        cell.declaration = GeneralVoFPhaseGeometry::Infer;
                    // This is inherited permission for an authenticated surviving
                    // pure geometry, not an occupancy or current arithmetic budget.
                    cell.occupancy.fit_bound = cell.declaration == GeneralVoFPhaseGeometry::Infer
                                                   ? 0
                                                   : fp::upper_double(Math::upper(row.pure_fit.value()));
                }
                validate_cell<Mode>(cell, data.options);
                candidate->gas.push_back(gas.first);
                candidate->carrier.push_back(carrier.first);
                candidate->phases.push_back({cell.native_volume, gas.first, carrier.first});
                candidate->roundoff.push_back(cell.roundoff);
                candidate->phase_history.push_back(cell.finite_history);
                candidate->occupancy.push_back(cell.occupancy);
                GeneralVoFFiniteOwnedChannels output;
                output.key = key;
                for (std::size_t i = 0; i < row.channels.size(); ++i)
                {
                    const auto value = row.channels[i].value();
                    if (options.channels[i].nonnegative && Math::lower(value) < 0)
                        throw std::invalid_argument("Finite remap nonnegative channel output is unresolved.");
                    output.amounts.push_back(value);
                }
                budget.add(sizeof(output) + output.amounts.size() * sizeof(PhaseValue));
                result.channels_.push_back(std::move(output));
            }
            current(data);
            if (!target->geometry_unchanged() || target->observation() != target_observation)
                throw std::invalid_argument("Finite remap target observation changed during preparation.");
        });
    import_phases<Mode>(*candidate);
    wire::collective(*comm, [&] { retained_budget<Mode>(*candidate); });
    result.candidate_ = std::move(candidate);
    result.validate_observations();
    return result;
}
} // namespace SimpleFluid::VoF
