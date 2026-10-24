/** @file GeneralVoF.cc
 * @brief Accepted phase state construction, queries and reconstruction.
 */
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
                if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                    cell.finite_geometry = Meshes::BoundedConvexGeometry::from_geometry(*cell.phase.geometry);
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
                    if (capsule.geometry_ready && options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                        cell.retained_interface = capsule.geometry;
                }
                else
                    occupancy(cell);
                if (options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                {
                    if (capsules)
                    {
                        const auto& capsule = capsules->at(i);
                        if (capsule.policy != options.inventory_policy || !capsule.experimental_exact)
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
                validate_cell(cell, options);
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
                        if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                            cell.finite_geometry = Meshes::BoundedConvexGeometry::from_geometry(*retained_geometry);
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
                if (!tile.exact_second_image.empty())
                    throw std::invalid_argument("General VoF periodic image sweeps are not qualified.");
                if (tile.second)
                {
                    topology->adjacency[tile.first].insert(*tile.second);
                    topology->adjacency[*tile.second].insert(tile.first);
                }
                if (hierarchy->owned().contains(tile.first) ||
                    (tile.second && hierarchy->owned().contains(*tile.second)))
                    topology->tiles.push_back(tile);
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
            std::set<std::uint64_t> control_nodes;
            for (const auto& tile : topology->tiles)
                control_nodes.insert(tile.node_gids.begin(), tile.node_gids.end());
            for (const auto& node : topology->draft.nodes)
                if (control_nodes.contains(node.gid))
                    topology->nodes.push_back(node);
            std::map<std::uint64_t, std::set<Key>> node_source_cells;
            std::map<AMR::AdaptiveBytes, const AMR::AdaptiveTile*> wall_tiles;
            if (options.inventory_policy == PhaseInventoryPolicy::Finite)
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
            if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                for (const auto& node : topology->draft.nodes)
                {
                    const auto fan = topology->node_fans.find(node.gid);
                    std::vector<Meshes::ConvexGeometryPtr> incident_sources;
                    if (fan != topology->node_fans.end() && !fan->second.empty())
                        for (const auto& key : fan->second)
                            incident_sources.push_back(data->cells.at(key).phase.geometry);
                    else if (const auto sources = node_source_cells.find(node.gid); sources != node_source_cells.end())
                        for (const auto& key : sources->second)
                            incident_sources.push_back(data->cells.at(key).phase.geometry);
                    if (incident_sources.empty())
                        throw std::invalid_argument("Finite VoF canonical node lacks an authenticated source fan.");
                    auto p = Meshes::bounded_source_point(incident_sources, node.exact_identity);
                    topology->finite_nodes.emplace(node.gid, std::move(p));
                }
            std::map<std::uint64_t, std::map<AMR::AdaptiveBytes, Meshes::ExactPlanarFacePtr>> walls;
            std::map<std::uint64_t, std::set<Key>> incident;
            for (const auto& tile : topology->draft.tiles)
                for (auto node : tile.node_gids)
                    if (control_nodes.contains(node))
                    {
                        const auto& fan = topology->node_fans.at(node);
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
            for (const auto& node : topology->nodes)
            {
                if (incident[node.gid] != topology->node_fans.at(node.gid))
                    throw std::invalid_argument(
                        "General VoF wall catalog lacks a complete authenticated incident fan.");
                for (const auto& key : topology->node_fans.at(node.gid))
                    if (!leaves.contains(key))
                        throw std::invalid_argument("General VoF wall catalog requires full incident native shells.");
                auto& retained = topology->node_walls[node.gid];
                for (const auto& [identity, geometry] : walls[node.gid])
                {
                    Meshes::BoundedPoint normal;
                    Meshes::BoundedFaceConstraint constraint;
                    if (options.inventory_policy == PhaseInventoryPolicy::Finite)
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
                        for (std::size_t i = 1; i + 1 < face->node_gids.size(); ++i)
                            normal = finite_detail::add(normal, finite_detail::cross(
                                finite_detail::subtract(topology->finite_nodes.at(face->node_gids[i]), a),
                                finite_detail::subtract(topology->finite_nodes.at(face->node_gids[i + 1]), a)));
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
    import_phases(*data);
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
            retained_budget(*data);
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
    return data_->options.inventory_policy;
}
std::vector<Meshes::ExactPhaseMeasures> GeneralVoF::inventory_measures() const
{
    if (inventory_policy() == PhaseInventoryPolicy::ExperimentalExact)
        return experimental_phases();
    std::vector<Meshes::ExactPhaseMeasures> result;
    result.reserve(data_->phases.size());
    for (const auto& value : data_->phases)
        result.push_back(
            {Measure::from_value(value.bulk), Measure::from_value(value.gas), Measure::from_value(value.carrier)});
    return result;
}
const std::vector<Meshes::ExactPhaseMeasures>& GeneralVoF::experimental_phases() const
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
std::vector<Meshes::ExactPhaseMeasures> GeneralVoF::geometric_phases() const
{
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    std::vector<Meshes::ExactPhaseMeasures> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                validate_cell(cell, data_->options);
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
                for (const auto& key : data_->topology->order)
                {
                    const auto& cell = data_->cells.at(key);
                    validate_cell(cell, data_->options);
                    const auto bulk = cell.finite_geometry->volume();
                    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                    const auto minority = gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
                    const auto fraction = static_cast<long double>(minority) / bulk.value;
                    const auto value = gas ? fraction : 1 - fraction;
                    if (!(bulk.value > 0) || !std::isfinite(value) || value < 0 || value > 1)
                        throw std::invalid_argument("Finite VoF geometric fraction has no bounded view.");
                    result.push_back(static_cast<double>(value));
                }
                return;
            }
            const auto one = Measure::from_value(1);
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                validate_cell(cell, data_->options);
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
        for (const auto& key : data_->topology->order)
        {
            const auto& cell = data_->cells.at(key);
            if (cell.declaration != GeneralVoFPhaseGeometry::Infer ||
                cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0)
                continue;
            auto assessment = PolyhedralReconstruction::assess_resolution(
                finite_sample(cell), finite_stencil(*data_, key), data_->options.reconstruction);
            if (!assessment.admitted())
                result.push_back({cell.gid, key, assessment});
        }
    });
    return result;
}
std::vector<PolyhedralInterface> GeneralVoF::reconstruct() const
{
    std::vector<PolyhedralInterface> result;
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
std::vector<Meshes::ConvexGeometryPtr> GeneralVoF::owned_geometry() const
{
    std::vector<Meshes::ConvexGeometryPtr> result;
    for (const auto& key : data_->topology->order)
        result.push_back(data_->cells.at(key).phase.geometry);
    return result;
}
std::vector<Meshes::BoundedPoint> GeneralVoF::finite_face_points(std::uint64_t face) const
{
    current(*data_);
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        throw std::invalid_argument("Finite canonical face points require Finite inventory policy.");
    const auto tile = std::find_if(data_->topology->tiles.begin(), data_->topology->tiles.end(),
        [&](const auto& tile) { return tile.face_gid == face; });
    if (tile == data_->topology->tiles.end())
        throw std::invalid_argument("Finite canonical face identity is not retained.");
    std::vector<Meshes::BoundedPoint> points;
    for (auto node : tile->node_gids)
        points.push_back(data_->topology->finite_nodes.at(node));
    return points;
}
std::vector<Meshes::BoundedConvexGeometryPtr> GeneralVoF::owned_finite_geometry() const
{
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        throw std::invalid_argument("Finite geometry access requires Finite inventory policy.");
    std::vector<Meshes::BoundedConvexGeometryPtr> result;
    for (const auto& key : data_->topology->order)
        result.push_back(data_->cells.at(key).finite_geometry);
    return result;
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
        result.push_back({cell.gid, cell.revision, cell.owner, cell.phase.geometry, cell.centroid, key, cell.finite_geometry});
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
            if (candidate_->options.inventory_policy != source_->options.inventory_policy)
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
                trial.candidate_->options.inventory_policy != data_->options.inventory_policy)
                throw std::invalid_argument("General VoF trial belongs to another source.");
        });
    return trial.candidate();
}
} // namespace SimpleFluid::VoF

namespace SimpleFluid::VoF
{
void GeneralVoF::enable_exact_geometry_statistics(bool enabled)
{
    Meshes::enable_exact_geometry_statistics(enabled);
}
void GeneralVoF::reset_geometry_statistics()
{
    Meshes::reset_exact_geometry_statistics();
    Meshes::reset_bounded_geometry_statistics();
}
Meshes::GeometryStatisticsSnapshot GeneralVoF::geometry_statistics()
{
    return {Meshes::exact_geometry_statistics_domain(), Meshes::bounded_geometry_statistics_domain(),
        Meshes::exact_geometry_statistics(), Meshes::bounded_geometry_statistics()};
}
}
