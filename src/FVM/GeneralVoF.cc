/** @file GeneralVoF.cc
 * @brief Accepted phase state construction, queries and reconstruction.
 */
#include "FVM/details/FiniteGeometryAccess.hh"
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
GeneralVoF::GeneralVoF(std::shared_ptr<const Hierarchy> hierarchy, std::vector<Key> order, std::vector<double> gas,
    std::vector<double> carrier, GeneralVoFOptions options, std::vector<GeneralVoFPhaseGeometry> declarations)
    : GeneralVoF(std::move(hierarchy), std::move(order), std::move(gas), std::move(carrier), options,
          std::move(declarations), nullptr)
{
}
GeneralVoF::GeneralVoF(std::shared_ptr<const Hierarchy> hierarchy, std::vector<Key> order, std::vector<double> gas,
    std::vector<double> carrier, GeneralVoFOptions options, std::vector<GeneralVoFPhaseGeometry> declarations,
    const ExactPhaseTransferResult* transfer)
{
    if (options.transport_backend == GeneralVoFTransportBackend::GeometricFace)
        options.reconstruction.measure_only = true;
    if (options.arithmetic_mode == ArithmeticMode::Exact)
        construct_mode<ArithmeticMode::Exact>(std::move(hierarchy), std::move(order), std::move(gas),
            std::move(carrier), options, std::move(declarations), transfer);
    else if (options.arithmetic_mode == ArithmeticMode::Ordinary)
        construct_mode<ArithmeticMode::Ordinary>(std::move(hierarchy), std::move(order), std::move(gas),
            std::move(carrier), options, std::move(declarations), transfer);
    else
        construct_mode<ArithmeticMode::BoundedCertification>(std::move(hierarchy), std::move(order), std::move(gas),
            std::move(carrier), options, std::move(declarations), transfer);
}
template<ArithmeticMode Mode>
void GeneralVoF::construct_mode(std::shared_ptr<const Hierarchy> hierarchy, std::vector<Key> order,
    std::vector<double> gas, std::vector<double> carrier, GeneralVoFOptions options,
    std::vector<GeneralVoFPhaseGeometry> declarations, const ExactPhaseTransferResult* transfer)
{
    const auto* capsules = transfer ? &transfer->owned() : nullptr;
    auto comm = hierarchy ? hierarchy->communicator() : Tpetra::getDefaultComm();
    std::shared_ptr<Data> data;
    std::shared_ptr<Data::Topology> topology;
    wire::collective(*comm,
        [&]
        {
            if (!hierarchy || !options.byte_budget || !options.maximum_stencil_cells || !options.maximum_halo_cells ||
                !options.support_rings || options.support_rings > 8 || order.size() != gas.size() ||
                order.size() != carrier.size() || order.size() != hierarchy->owned().size() ||
                (!declarations.empty() && declarations.size() != order.size()) ||
                (capsules && capsules->size() != order.size()))
                throw std::invalid_argument("Invalid General VoF source arrays/options.");
            wire::Budget budget{options.byte_budget};
            budget.add(order.size() * (sizeof(Cell) + sizeof(Key) + 2 * sizeof(double)));
            data = std::make_shared<Data>();
            topology = std::make_shared<Data::Topology>();
            data->topology = topology;
            data->hierarchy = hierarchy;
            data->observation = hierarchy->observation();
            data->supplied_hierarchy = hierarchy;
            data->supplied_observation = data->observation;
            data->options = options;
            current(*data);
            std::set<Key> seen;
            for (std::size_t i = 0; i < order.size(); ++i)
            {
                if (!hierarchy->owned().contains(order[i]) || !seen.insert(order[i]).second)
                    throw std::invalid_argument("General VoF input ownership differs.");
                const auto& leaf = hierarchy->owned().at(order[i]);
                Cell cell;
                cell.key = order[i];
                cell.owner = comm->getRank();
                cell.revision = leaf.revision;
                cell.phase = {hierarchy->geometry(leaf), gas[i], carrier[i]};
                cell.native_volume = hierarchy->metrics(leaf).volume;
                if constexpr (Mode != ArithmeticMode::Exact)
                    cell.finite_geometry = PLIC::ConvexCellGeometry<Mode>::from_geometry(
                        *cell.phase.geometry, geometry_limits(Mode, options.byte_budget))
                                               ->dynamic_geometry();
                if (capsules)
                {
                    cell.inherited = true;
                    const auto& capsule = capsules->at(i);
                    if (capsule.key != cell.key || capsule.revision != cell.revision ||
                        capsule.phases.bulk != cell.native_volume)
                        throw std::invalid_argument("General VoF exact adoption geometry/ownership differs.");
                    cell.phase.gas_volume = capsule.phases.gas;
                    cell.phase.carrier_volume = capsule.phases.carrier;
                    cell.roundoff = capsule.roundoff;
                    cell.finite_history = capsule.accumulated_roundoff;
                    const auto& e = capsule.error;
                    cell.occupancy = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
                        e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
                    if (capsule.geometry_ready && options.arithmetic_mode == ArithmeticMode::Exact)
                        cell.retained_interface = capsule.geometry;
                }
                else
                    occupancy<Mode>(cell);
                if constexpr (Mode == ArithmeticMode::Exact)
                {
                    if (capsules)
                    {
                        const auto& capsule = capsules->at(i);
                        if (capsule.policy != inventory_policy_for(options.arithmetic_mode) ||
                            !capsule.experimental_exact)
                            throw std::invalid_argument("Experimental adoption policy/payload differs.");
                        cell.experimental_exact = capsule.experimental_exact;
                    }
                    else
                        cell.experimental_exact = detail::experimental_initial(*cell.phase.geometry, cell.native_volume,
                            cell.phase.gas_volume, cell.phase.carrier_volume, options.byte_budget);
                }
                else if (capsules &&
                         (capsules->at(i).policy != PhaseInventoryPolicy::Finite || capsules->at(i).experimental_exact))
                    throw std::invalid_argument("Finite adoption cannot acquire experimental history.");
                if (!declarations.empty())
                    cell.declaration = declarations[i];
                else if (!capsules)
                {
                    // Exact absence proves pure geometry at fresh bootstrap.
                    // Persist that evidence through the existing fixed-bound
                    // declaration policy; never classify a nonzero minority by
                    // magnitude or replace an explicit/adopted declaration.
                    if (cell.phase.carrier_volume == 0 && cell.phase.gas_volume > 0)
                        cell.declaration = GeneralVoFPhaseGeometry::PureGas;
                    else if (cell.phase.gas_volume == 0 && cell.phase.carrier_volume > 0)
                        cell.declaration = GeneralVoFPhaseGeometry::PureCarrier;
                }
                // Validates independent occupancy without changing represented phases.
                validate_cell<Mode>(cell, options);
                budget.add(cell.phase.geometry->storage_bytes());
                data->cells.emplace(order[i], std::move(cell));
            }
        });
    agree_options(options, comm);
    data->control = transaction(comm);
    if (transfer)
    {
        data->adopted_source = transfer->source_identity();
        data->adopted_source_control = transfer->source_control();
        data->adopted_transaction = transfer->transaction();
        data->adopted_target_control = data->control;
    }
    topology->draft = AMR::fetch_adaptive_native_draft(*hierarchy);
    complete_node_fans(*data, *topology);
    for (unsigned ring = 1; ring <= options.support_rings; ++ring)
    {
        std::vector<Key> additional;
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{options.byte_budget};
                std::map<Key, std::set<Key>> graph;
                for (const auto& tile : topology->draft.tiles)
                    if (tile.second)
                    {
                        budget.add(4 * (sizeof(Key) + 8 * sizeof(void*) + AMR::encode_adaptive_key(tile.first).size() +
                                           AMR::encode_adaptive_key(*tile.second).size()));
                        graph[tile.first].insert(*tile.second);
                        graph[*tile.second].insert(tile.first);
                    }
                std::set<Key> seen, front;
                for (const auto& [key, leaf] : hierarchy->owned())
                {
                    seen.insert(key);
                    front.insert(key);
                }
                for (const auto& [node, fan] : topology->node_fans)
                    for (const auto& key : fan)
                    {
                        seen.insert(key);
                        front.insert(key);
                    }
                for (unsigned depth = 0; depth <= ring; ++depth)
                {
                    std::set<Key> next;
                    for (const auto& key : front)
                        for (const auto& neighbor : graph[key])
                            if (seen.insert(neighbor).second)
                            {
                                if (seen.size() - hierarchy->owned().size() > options.maximum_halo_cells)
                                    throw std::invalid_argument("General VoF halo cell budget exhausted.");
                                budget.add(sizeof(Key) + AMR::encode_adaptive_key(neighbor).size());
                                next.insert(neighbor);
                            }
                    front = std::move(next);
                }
                for (const auto& key : seen)
                    if (!hierarchy->owned().contains(key))
                        additional.push_back(key);
            });
        topology->draft = AMR::fetch_adaptive_native_draft(*hierarchy, additional);
    }
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{options.byte_budget};
            budget.add(topology->draft.cell_bindings.size() * (sizeof(Cell) + sizeof(Key)));
            std::map<Key, const AMR::AdaptiveLeaf*> leaves;
            for (const auto& leaf : topology->draft.owned)
                leaves.emplace(leaf.key, &leaf);
            for (const auto& leaf : topology->draft.ghosts)
                leaves.emplace(leaf.key, &leaf);
            for (const auto& binding : topology->draft.cell_bindings)
                if (binding.full)
                {
                    auto found = data->cells.find(binding.key);
                    const auto& retained_leaf = *leaves.at(binding.key);
                    const auto retained_geometry =
                        retained_leaf.geometry ? retained_leaf.geometry : hierarchy->geometry(retained_leaf);
                    if (found != data->cells.end() && (!found->second.phase.geometry->equivalent(*retained_geometry) ||
                                                          found->second.native_volume != binding.volume))
                        throw std::invalid_argument("General VoF retained native geometry/metric differs from source.");
                    if (found == data->cells.end())
                    {
                        const auto& leaf = retained_leaf;
                        Cell cell;
                        cell.key = binding.key;
                        cell.owner = binding.owner;
                        cell.revision = leaf.revision;
                        cell.phase.geometry = retained_geometry;
                        if constexpr (Mode != ArithmeticMode::Exact)
                            cell.finite_geometry = PLIC::ConvexCellGeometry<Mode>::from_geometry(
                                *retained_geometry, geometry_limits(Mode, options.byte_budget))
                                                       ->dynamic_geometry();
                        budget.add(cell.phase.geometry->storage_bytes());
                        found = data->cells.emplace(binding.key, std::move(cell)).first;
                    }
                    found->second.gid = binding.gid;
                    found->second.centroid = binding.centroid;
                    found->second.native_volume = binding.volume;
                    if (!topology->by_gid.emplace(binding.gid, binding.key).second)
                        throw std::invalid_argument("General VoF duplicate native cell GID.");
                }
            for (const auto& tile : topology->draft.tiles)
            {
                if (!tile.second_image_identity.empty())
                    throw std::invalid_argument("General VoF periodic image sweeps are not qualified.");
                if constexpr (Mode == ArithmeticMode::Ordinary)
                {
                    const bool first_full = data->cells.contains(tile.first);
                    const bool second_full = tile.second && data->cells.contains(*tile.second);
                    if (!first_full && !second_full)
                        throw std::invalid_argument("General VoF native tile has no full retained endpoint.");
                    if (first_full != second_full)
                    {
                        if (topology->frontier_tiles.size() == topology->frontier_tiles.capacity())
                        {
                            const auto capacity = topology->frontier_tiles.capacity();
                            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                                throw std::invalid_argument("General VoF frontier capacity overflows.");
                            const auto next = capacity ? 2 * capacity : 1;
                            if (next - capacity > (budget.limit - budget.used) / sizeof(AMR::AdaptiveTile))
                                throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                                    "General VoF frontier exceeds its byte budget.");
                            budget.add((next - capacity) * sizeof(AMR::AdaptiveTile));
                            topology->frontier_tiles.reserve(next);
                        }
                        budget.add(
                            (tile.node_gids.size() + tile.second_node_gids.size()) * sizeof(std::uint64_t) +
                            tile.domain.size() + tile.surface.size() + tile.identity.size() + tile.boundary.size() +
                            tile.second_image_identity.size() + AMR::encode_adaptive_key(tile.first).size() +
                            (tile.second ? AMR::encode_adaptive_key(*tile.second).size() : 0));
                        topology->frontier_tiles.push_back(tile);
                    }
                }
                if (tile.second)
                {
                    topology->adjacency[tile.first].insert(*tile.second);
                    topology->adjacency[*tile.second].insert(tile.first);
                }
                if (hierarchy->owned().contains(tile.first) ||
                    (tile.second && hierarchy->owned().contains(*tile.second)))
                    topology->tiles.push_back(tile);
                if (options.transport_backend != GeneralVoFTransportBackend::SweptPartition && tile.second &&
                    data->cells.contains(tile.first) && data->cells.contains(*tile.second))
                {
                    budget.add(sizeof(AMR::AdaptiveTile) + tile.node_gids.size() * sizeof(std::uint64_t) +
                        tile.domain.size() + tile.surface.size() + tile.identity.size() + tile.boundary.size() +
                        AMR::encode_adaptive_key(tile.first).size() + AMR::encode_adaptive_key(*tile.second).size());
                    topology->surface_tiles.push_back(tile);
                }
                if constexpr (Mode != ArithmeticMode::Exact)
                {
                    budget.add(sizeof(Data::FaceAuthority) + sizeof(std::uint64_t) + 4 * sizeof(void*) +
                               tile.domain.size() + tile.surface.size() + tile.identity.size() +
                               tile.node_gids.size() * sizeof(std::uint64_t) +
                               AMR::encode_adaptive_key(tile.first).size() +
                               (tile.second ? AMR::encode_adaptive_key(*tile.second).size() : 0));
                    if (!topology->construction_faces
                            .emplace(tile.face_gid, Data::FaceAuthority{tile.domain, tile.surface, tile.identity,
                                                        tile.node_gids, tile.first, tile.second, tile.exterior})
                            .second)
                        throw std::invalid_argument("Finite VoF construction face identity is duplicated.");
                }
            }
            // Preserve canonical tile order in every incidence row. Transport
            // sums and node-wall packets use this same order after publication.
            const auto append_index = [&](auto& entries, const auto& key, std::size_t index)
            {
                if (!entries.contains(key))
                {
                    budget.add(sizeof(key) + sizeof(std::vector<std::size_t>) + 4 * sizeof(void*));
                    if constexpr (std::is_same_v<std::decay_t<decltype(key)>, Key>)
                        budget.add(AMR::encode_adaptive_key(key).size());
                }
                auto& indices = entries[key];
                if (!indices.empty() && indices.back() == index)
                    return;
                if (indices.size() == indices.capacity())
                {
                    const auto capacity = indices.capacity() ? 2 * indices.capacity() : 1;
                    budget.add((capacity - indices.capacity()) * sizeof(std::size_t));
                    indices.reserve(capacity);
                }
                indices.push_back(index);
            };
            for (std::size_t index = 0; index < topology->tiles.size(); ++index)
            {
                const auto& tile = topology->tiles[index];
                append_index(topology->cell_tiles, tile.first, index);
                if (tile.second)
                    append_index(topology->cell_tiles, *tile.second, index);
                for (auto node : tile.node_gids)
                    append_index(topology->node_tiles, node, index);
            }
            for (const auto& [key, leaf] : hierarchy->owned())
            {
                auto& cell = data->cells.at(key);
                cell.canonical_face_count = topology->cell_tiles.at(key).size();
                if (!cell.canonical_face_count)
                    throw std::invalid_argument("General VoF owned cell has no canonical boundary tiles.");
                if constexpr (Mode == ArithmeticMode::Ordinary)
                {
                    const auto volume = std::min(static_cast<long double>(cell.native_volume),
                        PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume().value);
                    cell.ordinary_quota_scale = std::nextafter(
                        volume / static_cast<long double>(cell.canonical_face_count), 0.L);
                    if (!(cell.ordinary_quota_scale > 0) || !std::isfinite(cell.ordinary_quota_scale))
                        throw std::invalid_argument("General VoF ordinary face quota has invalid capacity.");
                }
            }
            std::set<std::uint64_t> control_nodes;
            for (const auto& tile : topology->tiles)
                control_nodes.insert(tile.node_gids.begin(), tile.node_gids.end());
            for (const auto& node : topology->draft.nodes)
                if (control_nodes.contains(node.gid))
                    topology->nodes.push_back(node);
            std::map<std::uint64_t, std::set<Key>> node_source_cells;
            std::map<AMR::AdaptiveBytes, const AMR::AdaptiveTile*> wall_tiles;
            if constexpr (Mode != ArithmeticMode::Exact)
                for (const auto& tile : topology->draft.tiles)
                {
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && data->cells.contains(*key))
                            for (auto node : tile.node_gids)
                                if (node_source_cells[node].insert(*key).second)
                                    budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 8 * sizeof(void*));
                    if (tile.exterior)
                        wall_tiles.try_emplace(tile.identity, &tile);
                }
            if constexpr (Mode != ArithmeticMode::Exact)
                for (const auto& node : topology->draft.nodes)
                {
                    const auto fan = topology->node_fans.find(node.gid);
                    std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>>
                        incident_sources;
                    if (fan != topology->node_fans.end() && !fan->second.empty())
                        for (const auto& key : fan->second)
                            incident_sources.push_back(data->cells.at(key).phase.geometry);
                    else if (const auto sources = node_source_cells.find(node.gid); sources != node_source_cells.end())
                        for (const auto& key : sources->second)
                            incident_sources.push_back(data->cells.at(key).phase.geometry);
                    if (incident_sources.empty())
                        throw std::invalid_argument("Finite VoF canonical node lacks an authenticated source fan.");
                    auto p = PLIC::source_point<Mode>(incident_sources, node.coordinate_identity).dynamic();
                    topology->finite_nodes.emplace(node.gid, std::move(p));
                    budget.add(
                        sizeof(std::uint64_t) + sizeof(AMR::AdaptiveBytes) + 4 * sizeof(void*) + node.identity.size());
                    if (!topology->construction_nodes.emplace(node.gid, node.identity).second)
                        throw std::invalid_argument("Finite VoF construction node identity is duplicated.");
                }
            const auto node_sources = [&](std::uint64_t node) -> const std::set<Key>&
            {
                if (const auto complete = topology->node_fans.find(node); complete != topology->node_fans.end())
                    return complete->second;
                return node_source_cells.at(node);
            };
            std::map<std::uint64_t,
                std::map<AMR::AdaptiveBytes, SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Ordinary>>>
                walls;
            std::map<std::uint64_t, std::set<Key>> incident;
            for (const auto& tile : topology->draft.tiles)
                for (auto node : tile.node_gids)
                    if (control_nodes.contains(node) || Mode != ArithmeticMode::Exact)
                    {
                        const auto& fan = node_sources(node);
                        for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                            if (key && fan.contains(*key) && incident[node].insert(*key).second)
                                budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 4 * sizeof(void*));
                        if (tile.exterior)
                        {
                            if (!fan.contains(tile.first))
                                throw std::invalid_argument("General VoF wall is outside authenticated node fan.");
                            auto& catalog = walls[node];
                            const auto found = catalog.find(tile.identity);
                            if (found == catalog.end())
                            {
                                budget.add(sizeof(Data::Wall) + tile.identity.size() + 4 * sizeof(void*));
                                catalog.emplace(tile.identity, tile.geometry);
                            }
                            else if (found->second->encode_incidence() != tile.geometry->encode_incidence())
                                throw std::invalid_argument("General VoF canonical wall certificates differ.");
                        }
                    }
            const auto& wall_nodes = [&]() -> const std::vector<AMR::AdaptiveNode>&
            {
                if constexpr (Mode == ArithmeticMode::Exact)
                    return topology->nodes;
                else
                    return topology->draft.nodes;
            }();
            for (const auto& node : wall_nodes)
            {
                if (control_nodes.contains(node.gid) && incident[node.gid] != topology->node_fans.at(node.gid))
                    throw std::invalid_argument(
                        "General VoF wall catalog lacks a complete authenticated incident fan.");
                for (const auto& key : node_sources(node.gid))
                    if (!leaves.contains(key))
                        throw std::invalid_argument("General VoF wall catalog requires full incident native shells.");
                auto& retained = topology->node_walls[node.gid];
                for (const auto& [identity, geometry] : walls[node.gid])
                {
                    PLIC::DynamicPoint normal;
                    normal.arithmetic_mode = options.arithmetic_mode;
                    PLIC::DynamicFaceConstraint constraint;
                    if constexpr (Mode != ArithmeticMode::Exact)
                    {
                        const auto found_face = wall_tiles.find(identity);
                        if (found_face == wall_tiles.end() || found_face->second->node_gids.size() < 3)
                            throw std::invalid_argument("Finite VoF wall lacks its authenticated native loop.");
                        const auto* face = found_face->second;
                        const auto& source_cell = data->cells.at(face->first);
                        constraint.geometry = source_cell.finite_geometry;
                        bool matched = false;
                        for (std::size_t index = 0; index < source_cell.phase.geometry->num_faces(); ++index)
                            if (source_cell.phase.geometry->face_geometry(index)->contains(*face->geometry))
                            {
                                constraint.face = index;
                                matched = true;
                                break;
                            }
                        if (!matched)
                            throw std::invalid_argument("Finite VoF source wall has no authenticated cell support.");
                        const auto& a = topology->finite_nodes.at(face->node_gids[0]);
                        using Math = finite_detail::Arithmetic<Mode>;
                        for (std::size_t i = 1; i + 1 < face->node_gids.size(); ++i)
                            normal = Math::add(
                                normal, Math::cross(Math::subtract(topology->finite_nodes.at(face->node_gids[i]), a),
                                            Math::subtract(topology->finite_nodes.at(face->node_gids[i + 1]), a)));
                    }
                    retained.push_back({identity, geometry, normal, constraint});
                }
            }
            for (const auto& [key, leaf] : hierarchy->owned())
            {
                topology->order.push_back(key);
                data->gas.push_back(data->cells.at(key).phase.gas_volume);
                data->carrier.push_back(data->cells.at(key).phase.carrier_volume);
            }
        });
    import_phases<Mode>(*data);
    AMR::AdaptiveBytes schema;
    wire::collective(*comm,
        [&]
        {
            schema = hierarchy->configuration_descriptor();
            if (schema.size() > options.byte_budget)
                throw std::invalid_argument("General VoF native schema budget exhausted.");
        });
    topology->native = AMR::assemble_adaptive_native(std::move(topology->draft), schema, comm, options.byte_budget);
    data->hierarchy = hierarchy->with_partition(topology->native.partition, topology->native.mesh->owned_cell_map());
    data->observation = data->hierarchy->observation();
    wire::collective(*comm,
        [&]
        {
            topology->order.clear();
            data->gas.clear();
            data->carrier.clear();
            data->phases.clear();
            data->roundoff.clear();
            data->phase_history.clear();
            data->occupancy.clear();
            for (std::size_t c = 0; c < topology->native.mesh->num_owned_cells(); ++c)
            {
                const auto key = topology->by_gid.at(topology->native.mesh->cell_global_id(int(c)));
                topology->order.push_back(key);
                data->gas.push_back(data->cells.at(key).phase.gas_volume);
                data->carrier.push_back(data->cells.at(key).phase.carrier_volume);
                data->phases.push_back({data->cells.at(key).native_volume, data->cells.at(key).phase.gas_volume,
                    data->cells.at(key).phase.carrier_volume});
                data->roundoff.push_back(data->cells.at(key).roundoff);
                data->phase_history.push_back(data->cells.at(key).finite_history);
                data->occupancy.push_back(data->cells.at(key).occupancy);
            }
            refresh_experimental_catalog(*data);
        });
    wire::collective(*comm,
        [&]
        {
            retained_budget<Mode>(*data);
            current(*data);
        });
    data_ = std::move(data);
}
std::shared_ptr<const void> GeneralVoF::accepted_identity() const noexcept
{
    return data_;
}
std::uint64_t GeneralVoF::accepted_control() const noexcept
{
    return data_->control;
}
bool GeneralVoF::accepted_unchanged() const noexcept
{
    return data_->hierarchy->geometry_unchanged() && data_->hierarchy->observation() == data_->observation &&
           data_->supplied_hierarchy->geometry_unchanged() &&
           data_->supplied_hierarchy->observation() == data_->supplied_observation;
}

const std::shared_ptr<const MeshHandle<>>& GeneralVoF::mesh() const noexcept
{
    return data_->topology->native.mesh;
}
const std::shared_ptr<const GeneralVoF::Hierarchy>& GeneralVoF::hierarchy() const noexcept
{
    return data_->hierarchy;
}
const std::vector<GeneralVoF::Key>& GeneralVoF::owned_order() const noexcept
{
    return data_->topology->order;
}
const std::vector<double>& GeneralVoF::gas_volumes() const noexcept
{
    return data_->gas;
}
const std::vector<double>& GeneralVoF::carrier_volumes() const noexcept
{
    return data_->carrier;
}
const std::vector<PhaseVolumes>& GeneralVoF::phase_volumes() const noexcept
{
    return data_->phases;
}
PhaseInventoryPolicy GeneralVoF::inventory_policy() const noexcept
{
    return inventory_policy_for(data_->options.arithmetic_mode);
}
ArithmeticMode GeneralVoF::arithmetic_mode() const noexcept
{
    return data_->options.arithmetic_mode;
}
std::vector<GeneralVoFPhaseGeometry> GeneralVoF::phase_geometry_declarations() const
{
    std::vector<GeneralVoFPhaseGeometry> declarations;
    declarations.reserve(data_->topology->order.size());
    for (const auto& key : data_->topology->order)
        declarations.push_back(data_->cells.at(key).declaration);
    return declarations;
}
std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> GeneralVoF::inventory_measures() const
{
    if (arithmetic_mode() != ArithmeticMode::Exact)
        throw std::invalid_argument(
            "Exact inventory view requires explicit Exact mode; use phase_volumes for finite storage.");
    return experimental_phases();
}
const std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>>&
GeneralVoF::experimental_phases() const
{
    if (inventory_policy() != PhaseInventoryPolicy::ExperimentalExact || !data_->experimental_catalog)
        throw std::invalid_argument("Exact inventory access requires explicit ExperimentalExact policy.");
    return *data_->experimental_catalog;
}
const std::vector<PhaseRoundoff>& GeneralVoF::phase_roundoff_bounds() const noexcept
{
    return data_->roundoff;
}
const std::vector<PhaseRoundoff>& GeneralVoF::accumulated_phase_errors() const noexcept
{
    return data_->phase_history;
}
std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> GeneralVoF::geometric_phases() const
{
    if (arithmetic_mode() != ArithmeticMode::Exact)
        throw std::invalid_argument("Exact geometric phase view requires explicit Exact arithmetic mode.");
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                validate_cell<ArithmeticMode::Exact>(cell, data_->options);
                auto labels = geometric_labels(cell);
                budget.add(sizeof(labels) + labels.bulk.storage_bytes() + labels.gas.storage_bytes() +
                           labels.carrier.storage_bytes());
                result.push_back(std::move(labels));
            }
        });
    return result;
}
std::vector<double> GeneralVoF::geometric_gas_fractions() const
{
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    std::vector<double> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            wire::Budget budget{data_->options.byte_budget};
            if (inventory_policy() == PhaseInventoryPolicy::Finite)
            {
                const auto evaluate = [&]<ArithmeticMode Mode>()
                {
                    for (const auto& key : data_->topology->order)
                    {
                        const auto& cell = data_->cells.at(key);
                        validate_cell<Mode>(cell, data_->options);
                        const auto bulk = PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->volume();
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        const auto minority = gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
                        const auto fraction = static_cast<long double>(minority) / bulk.value;
                        const auto value = gas ? fraction : 1 - fraction;
                        if (!(bulk.value > 0) || !std::isfinite(value) || value < 0 || value > 1)
                            throw std::invalid_argument("Finite VoF geometric fraction has no bounded view.");
                        result.push_back(static_cast<double>(value));
                    }
                };
                if (arithmetic_mode() == ArithmeticMode::Ordinary)
                    evaluate.template operator()<ArithmeticMode::Ordinary>();
                else
                    evaluate.template operator()<ArithmeticMode::BoundedCertification>();
                return;
            }
            const auto one = Measure::from_value(1);
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                validate_cell<ArithmeticMode::Exact>(cell, data_->options);
                const auto labels = geometric_labels(cell);
                if (labels.bulk.sign() <= 0)
                    throw std::invalid_argument("General VoF geometric fraction has nonpositive actual volume.");
                const auto fraction = labels.gas.divided_by(labels.bulk, data_->options.byte_budget);
                if (fraction.sign() < 0 || fraction.compare(one) > 0)
                    throw std::invalid_argument("General VoF geometric gas fraction is outside [0,1].");
                budget.add(sizeof(double) + fraction.storage_bytes());
                const double view = double(fraction.enclosure().value);
                if (!std::isfinite(view) || view < 0 || view > 1)
                    throw std::invalid_argument("General VoF geometric fraction has no finite bounded view.");
                result.push_back(view);
            }
        });
    return result;
}
const GeneralVoFOptions& GeneralVoF::options() const noexcept
{
    return data_->options;
}
const std::vector<PhaseError>& GeneralVoF::initial_occupancy_errors() const noexcept
{
    return data_->occupancy;
}
std::vector<GeneralVoFResolutionRequest> GeneralVoF::resolution_requests() const
{
    std::vector<GeneralVoFResolutionRequest> result;
    wire::collective(*data_->hierarchy->communicator(), [&]
    {
        current(*data_);
        if (inventory_policy() != PhaseInventoryPolicy::Finite)
            return;
        const auto evaluate = [&]<ArithmeticMode Mode>()
        {
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer || cell.phase.gas_volume == 0 ||
                    cell.phase.carrier_volume == 0)
                    continue;
                auto assessment =
                    PLIC::PolyhedralReconstruction<Mode>::assess_resolution(reconstruction_sample<Mode>(cell),
                        reconstruction_stencil<Mode>(*data_, key), data_->options.reconstruction);
                if (!assessment.admitted())
                    result.push_back({cell.gid, key, assessment});
            }
        };
        if (arithmetic_mode() == ArithmeticMode::Ordinary)
            evaluate.template operator()<ArithmeticMode::Ordinary>();
        else
            evaluate.template operator()<ArithmeticMode::BoundedCertification>();
    });
    return result;
}
std::vector<PLIC::PolyhedralInterface> GeneralVoF::reconstruct() const
{
    std::vector<PLIC::PolyhedralInterface> result;
    auto comm = data_->hierarchy->communicator();
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            auto planes = reconstruct_local(*data_);
            for (const auto& key : data_->topology->order)
                result.push_back(planes.at(key));
        });
    return result;
}
std::vector<GeneralVoFReconstructionPatch> GeneralVoF::reconstruction_patches() const
{
    const auto planes = import_planes(*data_);
    std::vector<GeneralVoFReconstructionPatch> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            for (const auto& [key, cell] : data_->cells)
                result.push_back({cell.gid, cell.owner, cell.phase.geometry, planes.at(key), cell.finite_geometry});
        });
    return result;
}
const std::vector<AMR::AdaptiveNode>& GeneralVoF::nodes() const noexcept
{
    return data_->topology->nodes;
}
const std::vector<AMR::AdaptiveTile>& GeneralVoF::tiles() const noexcept
{
    return data_->topology->tiles;
}
const std::vector<std::size_t>& GeneralVoF::node_tiles(std::uint64_t node) const
{
    return data_->topology->node_tiles.at(node);
}
std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>>
GeneralVoF::owned_geometry() const
{
    std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> result;
    for (const auto& key : data_->topology->order)
        result.push_back(data_->cells.at(key).phase.geometry);
    return result;
}
template<ArithmeticMode Mode> std::vector<PLIC::ConvexGeometryPtr<Mode>> GeneralVoF::owned_finite_geometry() const
{
    if (arithmetic_mode() != Mode)
        throw std::invalid_argument("General VoF geometry mode differs from its selected operation.");
    std::vector<PLIC::ConvexGeometryPtr<Mode>> result;
    result.reserve(data_->topology->order.size());
    for (const auto& key : data_->topology->order)
        result.push_back(PLIC::ConvexCellGeometry<Mode>::from_dynamic(data_->cells.at(key).finite_geometry));
    return result;
}
template<ArithmeticMode Mode> std::vector<PLIC::Point<Mode>> GeneralVoF::finite_face_points(std::uint64_t face) const
{
    if (arithmetic_mode() != Mode)
        throw std::invalid_argument("General VoF face mode differs from its selected operation.");
    const auto tile = std::find_if(data_->topology->tiles.begin(), data_->topology->tiles.end(),
        [&](const auto& tile) { return tile.face_gid == face; });
    if (tile == data_->topology->tiles.end())
        throw std::invalid_argument("Finite canonical face identity is not retained.");
    std::vector<PLIC::Point<Mode>> points;
    points.reserve(tile->node_gids.size());
    for (auto node : tile->node_gids)
        points.emplace_back(data_->topology->finite_nodes.at(node));
    return points;
}
template std::vector<PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>>
GeneralVoF::owned_finite_geometry<ArithmeticMode::Ordinary>() const;
template std::vector<PLIC::Point<ArithmeticMode::Ordinary>> GeneralVoF::finite_face_points<ArithmeticMode::Ordinary>(
    std::uint64_t) const;
template std::vector<PLIC::ConvexGeometryPtr<ArithmeticMode::BoundedCertification>>
GeneralVoF::owned_finite_geometry<ArithmeticMode::BoundedCertification>() const;
template std::vector<PLIC::Point<ArithmeticMode::BoundedCertification>>
GeneralVoF::finite_face_points<ArithmeticMode::BoundedCertification>(std::uint64_t) const;
const std::vector<GeneralVoFFaceChainDiagnostic>& GeneralVoF::Trial::finite_face_chain_diagnostics() const noexcept
{
    return finite_face_chain_;
}
const std::vector<GeneralVoFCellClosureDiagnostic>& GeneralVoF::Trial::finite_cell_closure_diagnostics() const noexcept
{
    return finite_cell_closure_;
}
const std::vector<GeneralVoFFiniteFaceContribution>& GeneralVoF::Trial::finite_face_contributions() const noexcept
{
    return finite_faces_;
}
const std::vector<GeneralVoFFiniteTransfer>& GeneralVoF::Trial::finite_transfers() const noexcept
{
    return finite_transfers_;
}
const std::vector<GeneralVoFFiniteDonorBudget>& GeneralVoF::Trial::finite_donor_budgets() const noexcept
{
    return finite_budgets_;
}
const std::vector<GeneralVoFFiniteNodeTrajectory>& GeneralVoF::Trial::finite_node_trajectories() const noexcept
{
    return finite_trajectories_;
}
std::vector<std::uint64_t> GeneralVoF::node_cells(std::uint64_t node) const
{
    std::set<std::uint64_t> cells;
    const auto& fan = data_->topology->node_fans.at(node);
    std::set<Key> seen = fan, front = fan;
    for (const auto& key : fan)
        cells.insert(data_->cells.at(key).gid);
    for (unsigned ring = 0; ring < data_->options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& key : front)
        {
            auto found = data_->topology->adjacency.find(key);
            if (found == data_->topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
            {
                if (!data_->cells.contains(neighbor))
                    throw std::invalid_argument("General VoF canonical node support is incomplete.");
                if (seen.insert(neighbor).second)
                {
                    if (seen.size() > data_->options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF node support budget exhausted.");
                    next.insert(neighbor);
                    cells.insert(data_->cells.at(neighbor).gid);
                }
            }
        }
        front = std::move(next);
    }
    return {cells.begin(), cells.end()};
}
std::vector<std::uint64_t> GeneralVoF::cell_support(std::uint64_t gid) const
{
    const auto key = data_->topology->by_gid.at(gid);
    if (!data_->hierarchy->owned().contains(key))
        throw std::invalid_argument("General VoF cell support requires an owned seed.");
    std::set<Key> seen{key}, front{key};
    for (unsigned ring = 0; ring < data_->options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& center : front)
        {
            const auto found = data_->topology->adjacency.find(center);
            if (found == data_->topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
                if (data_->cells.contains(neighbor) && seen.insert(neighbor).second)
                {
                    if (seen.size() > data_->options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF cell support budget exhausted.");
                    next.insert(neighbor);
                }
        }
        front = std::move(next);
    }
    std::vector<std::uint64_t> result;
    for (const auto& member : seen)
        result.push_back(data_->cells.at(member).gid);
    return result;
}
std::vector<GeneralVoFSupportCell> GeneralVoF::support_cells() const
{
    std::vector<GeneralVoFSupportCell> result;
    for (const auto& [key, cell] : data_->cells)
        result.push_back(
            {cell.gid, cell.revision, cell.owner, cell.phase.geometry, cell.centroid, key, cell.finite_geometry});
    return result;
}
const std::vector<GeneralVoFDonorBudget>& GeneralVoF::Trial::donor_budgets() const noexcept
{
    return budgets_;
}
const std::vector<PhaseRoundoff>& GeneralVoF::Trial::phase_roundoff_bounds() const noexcept
{
    return roundoff_;
}

const std::vector<GeneralVoFNodeTrajectory>& GeneralVoF::Trial::node_trajectories() const noexcept
{
    return trajectories_;
}
const std::vector<GeneralVoFFaceContribution>& GeneralVoF::Trial::face_contributions() const noexcept
{
    return faces_;
}
const std::vector<GeneralVoFTransfer>& GeneralVoF::Trial::transfers() const noexcept
{
    return transfers_;
}
std::shared_ptr<const GeneralVoF> GeneralVoF::Trial::candidate() const
{
    if (!source_ || !candidate_)
        throw std::invalid_argument("General VoF trial has no candidate.");
    agree_transaction(source_->hierarchy->communicator(), transaction_);
    std::shared_ptr<const GeneralVoF> result;
    wire::collective(*source_->hierarchy->communicator(),
        [&]
        {
            current(*source_);
            if (candidate_->options.arithmetic_mode != source_->options.arithmetic_mode)
                throw std::invalid_argument("General VoF trial candidate inventory policy differs.");
            result = std::shared_ptr<const GeneralVoF>(new GeneralVoF(candidate_));
        });
    return result;
}
std::shared_ptr<const GeneralVoF> GeneralVoF::apply(const Trial& trial) const
{
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            if (trial.source_ != data_ || !trial.candidate_ ||
                trial.candidate_->options.arithmetic_mode != data_->options.arithmetic_mode)
                throw std::invalid_argument("General VoF trial belongs to another source.");
        });
    return trial.candidate();
}
} // namespace SimpleFluid::VoF

namespace SimpleFluid::VoF
{
void GeneralVoF::enable_geometry_statistics(bool enabled)
{
    SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(enabled);
}
void GeneralVoF::reset_geometry_statistics()
{
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    PLIC::reset_geometry_statistics();
}
Meshes::GeometryStatisticsSnapshot GeneralVoF::geometry_statistics()
{
    return {SimpleFluid::PLIC::backend_statistics_domain<SimpleFluid::ArithmeticMode::Exact>(),
        PLIC::geometry_statistics_domain(), SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>(),
        PLIC::geometry_statistics()};
}
}
