/** @file GeneralVoFFiniteRemap.inc
 * @brief Owner-local bounded phase/channel handoff across authenticated AMR plans.
 * Included after GeneralVoFFinite.inc inside SimpleFluid::VoF.
 */
namespace
{
FValue finite_remap_weighted(const FPhases& p, GeneralVoFFiniteRemapCarrier carrier,
    const std::array<double, 2>& density)
{
    switch (carrier)
    {
    case GeneralVoFFiniteRemapCarrier::BulkVolume: return p.bulk;
    case GeneralVoFFiniteRemapCarrier::GasVolume: return p.gas;
    case GeneralVoFFiniteRemapCarrier::CarrierVolume: return p.carrier;
    case GeneralVoFFiniteRemapCarrier::MixtureMass:
        return fp::add(fp::multiply(p.gas, {density[0], 0}), fp::multiply(p.carrier, {density[1], 0}));
    }
    throw std::invalid_argument("Unknown finite remap channel carrier.");
}
void finite_remap_error(wire::Writer& w, const PhaseError& e)
{
    for (double v : {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
             e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound})
        real(w, v);
}
PhaseError finite_remap_error(wire::Reader& r)
{
    PhaseError e;
    e.difference = real(r); e.admitted_bound = real(r); e.native_conversion = real(r);
    e.phase_input_bound = real(r); e.fit_residual = real(r); e.fit_bound = real(r);
    e.transferred_bound = real(r); e.fit_allocation_bound = real(r);
    return e;
}
} // namespace

void GeneralVoF::FiniteRemapTrial::validate_observations() const
{
    if (!source_)
        throw std::invalid_argument("Finite remap has no authenticated source.");
    wire::collective(*source_->hierarchy->communicator(), [&]
    {
        current(*source_);
        if (!candidate_ || source_->options.inventory_policy != PhaseInventoryPolicy::Finite ||
            candidate_->options.inventory_policy != PhaseInventoryPolicy::Finite)
            throw std::invalid_argument("Finite remap has no compatible candidate.");
        current(*candidate_);
    });
}
std::shared_ptr<const GeneralVoF> GeneralVoF::FiniteRemapTrial::candidate() const
{
    validate_observations();
    return std::shared_ptr<const GeneralVoF>(new GeneralVoF(candidate_));
}
GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap_finite(const Hierarchy::Plan& plan,
    std::shared_ptr<const Hierarchy> target, std::span<const GeneralVoFFiniteOwnedChannels> channels,
    GeneralVoFFiniteRemapOptions options) const
{
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const auto limit = std::min(data.options.byte_budget, options.byte_budget);
    std::map<Key, std::vector<FValue>> source_channels;
    std::vector<char> manifest;
    wire::collective(*comm, [&]
    {
        current(data);
        if (data.options.inventory_policy != PhaseInventoryPolicy::Finite || !target ||
            target->communicator().getRawPtr() != comm.getRawPtr() || !target->native_partition() || !limit ||
            !plan.collective_identity() || !std::isfinite(options.maximum_relative_target_defect) ||
            !(options.maximum_relative_target_defect > 0) || !std::isfinite(options.maximum_relative_target_fit) ||
            !(options.maximum_relative_target_fit > 0))
            throw std::invalid_argument("Finite remap requires compatible source, actual native target and options.");
        wire::Budget budget{limit};
        wire::Writer w;
        w.word(data.control);
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
                source_channels.emplace(key, std::vector<FValue>{});
        else
            for (const auto& item : channels)
            {
                if (!data.hierarchy->owned().contains(item.key) || item.amounts.size() != options.channels.size() ||
                    !source_channels.emplace(item.key, item.amounts).second)
                    throw std::invalid_argument("Finite remap source channels have duplicate, foreign or missing keys.");
                budget.add(sizeof(GeneralVoFFiniteOwnedChannels) + item.amounts.size() * sizeof(FValue));
                for (std::size_t i = 0; i < item.amounts.size(); ++i)
                {
                    const auto v = item.amounts[i];
                    if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0 ||
                        (options.channels[i].nonnegative && (v.lower() < 0 || v.value < 0)))
                        throw std::invalid_argument("Finite remap source channel is nonfinite or unresolved nonnegative.");
                }
            }
        if (source_channels.size() != data.topology->order.size())
            throw std::invalid_argument("Finite remap channels do not cover every source owner.");
    });
    std::uint64_t manifest_size = manifest.size();
    Teuchos::broadcast(*comm, 0, 1, &manifest_size);
    std::vector<char> agreed;
    wire::collective(*comm, [&]
    {
        if (manifest_size > limit || manifest_size > std::uint64_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Finite remap collective schema exceeds its payload cap.");
        agreed = comm->getRank() == 0 ? manifest : std::vector<char>(std::size_t(manifest_size));
    });
    Teuchos::broadcast(*comm, 0, int(manifest_size), agreed.data());
    wire::collective(*comm, [&]
    {
        if (agreed != manifest)
            throw std::invalid_argument("Finite remap schema, source publication or plan differs across ranks.");
    });
    target = data.hierarchy->apply(plan, std::move(target));
    const auto target_observation = target->observation();
    const bool ownership = target->topology_generation() == data.hierarchy->topology_generation();
    const auto catalog = detail::build_adaptive_overlap_catalog(*data.hierarchy, *target, limit);
    std::map<std::uint64_t, const AMR::AdaptiveCellBinding*> from, to;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& cell : catalog.source_cells)
        {
            budget.add(sizeof(decltype(from)::value_type) + 4 * sizeof(void*));
            if (!from.emplace(cell.gid, &cell).second)
                throw std::invalid_argument("Finite remap source catalog repeats a global ID.");
        }
        for (const auto& cell : catalog.target_cells)
        {
            budget.add(sizeof(decltype(to)::value_type) + 4 * sizeof(void*));
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
        Meshes::BoundedConvexGeometryPtr overlap;
    };
    std::vector<Relation> requests;
    std::vector<Key> source_keys;
    wire::collective(*comm, [&]
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
    wire::collective(*comm, [&]
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
            w.word(request.source_gid); w.word(request.source_revision);
            w.bytes(AMR::encode_adaptive_key(request.target));
            w.word(request.target_gid); w.word(request.target_revision);
            w.word(static_cast<unsigned>(request.kind));
            budget.packet(w.data.size());
            packets.push_back({locations[i].owner, std::move(w.data)});
        }
    });
    const auto incoming = exchange_owner_packets(packets, comm, limit);
    std::vector<Relation> columns;
    std::vector<Key> target_keys;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::set<std::pair<Key, Key>> seen;
        std::map<std::pair<Key, Key>, AMR::AdaptiveAncestryRelation::Kind> admitted_relations;
        for (const auto& relation : catalog.relations)
            admitted_relations.emplace(std::pair{from.at(relation.source_gid)->key,
                to.at(relation.target_gid)->key}, relation.kind);
        for (const auto& packet : incoming.packets)
        {
            wire::Reader r{packet.bytes};
            Relation column;
            column.source = AMR::decode_adaptive_key(r.bytes());
            column.source_gid = r.word(); column.source_revision = r.word();
            column.target = AMR::decode_adaptive_key(r.bytes());
            column.target_gid = r.word(); column.target_revision = r.word();
            const auto kind = r.word();
            r.done();
            if (!data.hierarchy->owned().contains(column.source) ||
                !seen.emplace(column.source, column.target).second ||
                kind > static_cast<unsigned>(AMR::AdaptiveAncestryRelation::Kind::ProductIntersection))
                throw std::invalid_argument("Finite remap received duplicate, foreign or malformed overlap request.");
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
    std::map<Key, Meshes::BoundedConvexGeometryPtr> imported_targets;
    std::map<Key, std::vector<std::size_t>> by_source;
    wire::collective(*comm, [&]
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
                Meshes::BoundedGeometryLimits limits;
                limits.byte_budget = limit;
                auto geometry = Meshes::BoundedConvexGeometry::from_geometry(*target->geometry(leaf), limits);
                budget.add(geometry->storage_bytes());
                imported_targets.emplace(column.target, std::move(geometry));
            }
            const auto& source_geometry = data.cells.at(column.source).finite_geometry;
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
        FPhases phases, history;
        std::vector<FValue> amounts;
        FValue inherited_occupancy;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
        std::optional<PolyhedralInterface> retained_interface;
    };
    std::vector<Output> outgoing;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& key : data.topology->order)
        {
            const auto& source = data.cells.at(key);
            const auto found = by_source.find(key);
            if (found == by_source.end() || found->second.empty())
                throw std::invalid_argument("Finite remap has an uncovered source owner.");
            const auto& indices = found->second;
            std::vector<Meshes::BoundedConvexGeometryPtr> cover;
            for (auto i : indices)
                cover.push_back(columns[i].overlap);
            source.finite_geometry->certify_partition(cover, options.maximum_relative_target_defect);
            const bool whole = indices.size() == 1 &&
                (columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                 columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening);
            PolyhedralInterface fitted;
            FLabel labels;
            if (!whole)
            {
                fitted = reconstruct_finite_cell(data, key);
                labels = finite_label(source, fitted, data.options.reconstruction.volume_tolerance);
            }
            const auto old = finite_inventory(source);
            // Carrier existence is a source contract even on ownership-only
            // copies; the whole-cell fast path cannot legitimize an unsupported
            // nonzero channel merely because it needs no geometric weighting.
            for (std::size_t c = 0; c < options.channels.size(); ++c)
            {
                const auto carrier = finite_remap_weighted(old, options.channels[c].carrier, options.phase_density);
                const auto amount = source_channels.at(key)[c];
                if (carrier.value == 0 && carrier.absolute_error == 0)
                {
                    if (amount.value != 0 || amount.absolute_error != 0)
                        throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                }
                else if (!(carrier.lower() > 0))
                    throw std::invalid_argument("Finite remap source channel carrier is unresolved.");
            }
            FPhaseSum total;
            std::vector<fp::Sum> channel_totals(options.channels.size());
            for (auto i : indices)
            {
                const auto& column = columns[i];
                Output output;
                output.relation = column;
                output.phases = whole ? old : labels.apply(*column.overlap, fitted);
                output.history = whole ? FPhases{{}, {source.finite_history.gas, 0},
                    {source.finite_history.carrier, 0}} : labels.history(*column.overlap, fitted);
                output.declaration = source.declaration;
                output.original_error = source.occupancy;
                output.original_roundoff = source.roundoff;
                if (ownership)
                    output.retained_interface = source.retained_interface;
                const auto bulk_share = whole ? FValue{1, 0} : fp::divide(output.phases.bulk, old.bulk);
                output.inherited_occupancy = fp::multiply({source.occupancy.admitted_bound, 0}, bulk_share);
                for (std::size_t c = 0; c < options.channels.size(); ++c)
                {
                    const auto& input = source_channels.at(key)[c];
                    FValue amount;
                    if (whole)
                        amount = input;
                    else
                    {
                        const auto denominator = finite_remap_weighted(old, options.channels[c].carrier, options.phase_density);
                        const auto numerator = finite_remap_weighted(output.phases, options.channels[c].carrier, options.phase_density);
                        if (denominator.value == 0 && denominator.absolute_error == 0)
                        {
                            if (input.value != 0 || input.absolute_error != 0 ||
                                numerator.value != 0 || numerator.absolute_error != 0)
                                throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                        }
                        else
                            amount = fp::multiply(input, fp::divide(numerator, denominator));
                    }
                    output.amounts.push_back(amount);
                    channel_totals[c].add(amount);
                }
                total.add(output.phases);
                budget.add(sizeof(Output) + output.amounts.size() * sizeof(FValue));
                outgoing.push_back(std::move(output));
            }
            const auto delivered = total.value();
            for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
            {
                const auto residual = fp::subtract(pair.second, pair.first);
                if (std::abs(residual.value) > residual.absolute_error ||
                    residual.absolute_error > std::abs(pair.first.value) * options.maximum_relative_target_fit)
                    throw std::invalid_argument("Finite remap source phase closure exceeds its unchanged target-fit bound.");
            }
            for (std::size_t c = 0; c < options.channels.size(); ++c)
            {
                const auto residual = fp::subtract(channel_totals[c].value(), source_channels.at(key)[c]);
                if (std::abs(residual.value) > residual.absolute_error)
                    throw std::invalid_argument("Finite remap weighted channel column is not conservative.");
            }
        }
    });
    packets.clear();
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& output : outgoing)
        {
            const auto& column = output.relation;
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(column.source));
            w.word(column.source_gid); w.word(column.source_revision);
            w.bytes(AMR::encode_adaptive_key(column.target));
            w.word(column.target_gid); w.word(column.target_revision);
            finite_phases(w, output.phases);
            finite_phases(w, output.history);
            finite_value(w, output.inherited_occupancy);
            w.word(static_cast<unsigned>(output.declaration));
            w.word(output.amounts.size());
            for (auto amount : output.amounts)
                finite_value(w, amount);
            if (ownership)
            {
                finite_remap_error(w, output.original_error);
                real(w, output.original_roundoff.gas); real(w, output.original_roundoff.carrier);
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
        FPhaseSum phases, history;
        fp::Sum occupancy;
        std::vector<fp::Sum> channels;
        std::optional<GeneralVoFPhaseGeometry> declaration;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        std::optional<PolyhedralInterface> retained_interface;
        std::size_t pieces = 0;
    };
    std::map<Key, Row> rows;
    FiniteRemapTrial result;
    result.source_ = data_;
    result.ownership_only_ = ownership;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& column : columns)
        {
            budget.add(sizeof(GeneralVoFFiniteCommonCell) + column.overlap->storage_bytes());
            result.common_cells_.push_back({column.source, column.target, column.source_gid,
                column.target_gid, column.target_owner, column.overlap});
        }
    });
    wire::collective(*comm, [&]
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
                revision != request.source_revision || tgid != request.target_gid || target_revision != request.target_revision)
                throw std::invalid_argument("Finite remap result source/target ownership or revision differs.");
            const auto phases = finite_phases(r), history = finite_phases(r);
            const auto inherited_occupancy = finite_value(r);
            const auto declaration = r.word(), count = r.word();
            if (declaration > static_cast<unsigned>(GeneralVoFPhaseGeometry::PureCarrier) || count != options.channels.size())
                throw std::invalid_argument("Finite remap result schema is malformed.");
            for (auto value : {phases.bulk, phases.gas, phases.carrier, history.gas, history.carrier, inherited_occupancy})
                if (value.value < 0 || value.lower() < 0)
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
            for (auto& channel : row.channels)
                channel.add(finite_value(r));
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
                    row.retained_interface = plane(r);
                    if (!row.retained_interface->finite_geometry)
                        throw std::invalid_argument("Finite migration cannot acquire an exact interface.");
                }
            }
            r.done();
            budget.add(sizeof(Row) + sizeof(GeneralVoFFiniteRemapRoute) + row.channels.size() * sizeof(fp::Sum));
            result.routes_.push_back({sk, tk, sgid, tgid, packet.peer, comm->getRank(), phases});
        }
        if (seen.size() != expected.size() || rows.size() != target->owned().size())
            throw std::invalid_argument("Finite remap outputs do not cover every target owner.");
    });
    // Independently validate recipient geometry before physical publication.
    // Sparse ancestry selects candidates; a net volume sum cannot certify cover.
    std::map<Key, Meshes::BoundedConvexGeometryPtr> local_source_geometry;
    std::map<Key, std::vector<Meshes::BoundedConvexGeometryPtr>> target_covers;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& request : requests)
        {
            if (!local_source_geometry.contains(request.source))
            {
                const auto existing = data.cells.find(request.source);
                auto geometry = existing != data.cells.end() ? existing->second.finite_geometry :
                    Meshes::BoundedConvexGeometry::from_geometry(*from.at(request.source_gid)->geometry);
                budget.add(geometry->storage_bytes());
                local_source_geometry.emplace(request.source, std::move(geometry));
            }
            if (!imported_targets.contains(request.target))
            {
                auto geometry = Meshes::BoundedConvexGeometry::from_geometry(*to.at(request.target_gid)->geometry);
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
            imported_targets.at(key)->certify_partition(target_covers.at(key), options.maximum_relative_target_defect);
    });
    // Build only geometry/maps in a private pure placeholder. Actual remapped
    // inventories and their inherited error evidence are installed together.
    std::vector<Key> order;
    std::vector<double> zero, native_volume;
    wire::collective(*comm, [&]
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
    wire::collective(*comm, [&]
    {
        candidate->gas.clear(); candidate->carrier.clear(); candidate->phases.clear();
        candidate->roundoff.clear(); candidate->phase_history.clear(); candidate->occupancy.clear();
        wire::Budget budget{limit};
        for (const auto& key : candidate->topology->order)
        {
            auto& cell = candidate->cells.at(key);
            auto& row = rows.at(key);
            const auto totals = row.phases.value(), history = row.history.value();
            const auto actual = cell.finite_geometry->volume();
            const auto closure = fp::subtract(totals.bulk, actual);
            if (std::abs(closure.value) > closure.absolute_error ||
                closure.absolute_error > actual.value * options.maximum_relative_target_defect)
                throw std::invalid_argument("Finite remap recipient coverage exceeds its geometric error budget.");
            const auto gas = fp::publish(totals.gas), carrier = fp::publish(totals.carrier);
            cell.phase.gas_volume = gas.first;
            cell.phase.carrier_volume = carrier.first;
            cell.roundoff = ownership ? row.original_roundoff : PhaseRoundoff{gas.second, carrier.second};
            cell.finite_history = {fp::upper_double(history.gas.upper()), fp::upper_double(history.carrier.upper())};
            if (!ownership)
            {
                cell.finite_history.gas = fp::upper_double(fp::upward(cell.finite_history.gas + gas.second));
                cell.finite_history.carrier = fp::upper_double(fp::upward(cell.finite_history.carrier + carrier.second));
            }
            cell.declaration = row.declaration.value_or(GeneralVoFPhaseGeometry::Infer);
            cell.retained_interface = ownership ? row.retained_interface : std::optional<PolyhedralInterface>{};
            cell.inherited = true;
            if (ownership)
                cell.occupancy = row.original_error;
            else
            {
                const auto discrepancy = fp::subtract(fp::add({gas.first, gas.second}, {carrier.first, carrier.second}), actual);
                const auto inherited = row.occupancy.value();
                const auto admitted = fp::upward(inherited.upper() + discrepancy.absolute_error + std::abs(closure.value));
                if (std::abs(discrepancy.value) > admitted ||
                    admitted > actual.value * options.maximum_relative_target_defect)
                    throw std::invalid_argument("Finite remap physical occupancy exceeds its original target-defect bound.");
                cell.occupancy.difference = static_cast<double>(discrepancy.value);
                cell.occupancy.admitted_bound = fp::upper_double(admitted);
                cell.occupancy.phase_input_bound = fp::upper_double(fp::upward(gas.second + carrier.second));
                cell.occupancy.transferred_bound = fp::upper_double(fp::upward(totals.gas.absolute_error + totals.carrier.absolute_error));
            }
            validate_cell(cell, data.options);
            candidate->gas.push_back(gas.first); candidate->carrier.push_back(carrier.first);
            candidate->phases.push_back({cell.native_volume, gas.first, carrier.first});
            candidate->roundoff.push_back(cell.roundoff); candidate->phase_history.push_back(cell.finite_history);
            candidate->occupancy.push_back(cell.occupancy);
            GeneralVoFFiniteOwnedChannels output;
            output.key = key;
            for (std::size_t i = 0; i < row.channels.size(); ++i)
            {
                const auto value = row.channels[i].value();
                if (options.channels[i].nonnegative && value.lower() < 0)
                    throw std::invalid_argument("Finite remap nonnegative channel output is unresolved.");
                output.amounts.push_back(value);
            }
            budget.add(sizeof(output) + output.amounts.size() * sizeof(FValue));
            result.channels_.push_back(std::move(output));
        }
        current(data);
        if (!target->geometry_unchanged() || target->observation() != target_observation)
            throw std::invalid_argument("Finite remap target observation changed during preparation.");
    });
    import_phases(*candidate);
    wire::collective(*comm, [&] { retained_budget(*candidate); });
    result.candidate_ = std::move(candidate);
    result.validate_observations();
    return result;
}
