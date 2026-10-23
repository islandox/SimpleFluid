/** @file GeneralVoFFiniteRemap.inc
 * @brief Fixed-clock finite channel recovery and ownership-only canonical face handoff.
 * Included only inside GeneralVoFSolver::Impl.
 */
    double handoff_finite_owned_faces(const State& old, Impl& target, State& state) const
    {
        namespace fp = VoF::finite_detail;
        const auto comm = mesh->owned_cell_map()->getComm();
        const auto identity = [](const Meshes::AMR::AdaptiveTile& tile)
        {
            std::vector<char> result;
            for (const auto* key : {&tile.domain, &tile.surface, &tile.identity})
            {
                append_word(result, key->size());
                result.insert(result.end(), key->begin(), key->end());
            }
            return result;
        };
        const auto authority = [&](const std::vector<char>& key)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (unsigned char c : key) hash = (hash ^ c) * 1099511628211ULL;
            return int(hash % comm->getSize()); // Routing only; full identity is compared.
        };
        const auto bytes = [](std::vector<char>& out, const std::vector<char>& value)
        {
            append_word(out, value.size());
            out.insert(out.end(), value.begin(), value.end());
        };
        const auto scalar = [](std::vector<char>& out, long double value)
        {
            int exponent = 0;
            const auto mantissa = std::frexp(value, &exponent);
            const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
            if (!std::isfinite(value) || static_cast<long double>(high) + low != mantissa)
                throw std::runtime_error("Finite face handoff scalar exceeds fixed precision.");
            append_word(out, std::bit_cast<std::uint64_t>(high));
            append_word(out, std::bit_cast<std::uint64_t>(low));
            append_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
        };
        const auto word = [](const std::vector<char>& in, std::size_t& offset)
        {
            const auto value = read_word(in, offset);
            offset += 8;
            return value;
        };
        const auto read_bytes = [&](const std::vector<char>& in, std::size_t& offset)
        {
            const auto count = word(in, offset);
            if (count > in.size() - std::min(offset, in.size()))
                throw std::runtime_error("Finite face handoff identity length is malformed.");
            std::vector<char> result(in.begin() + offset, in.begin() + offset + count);
            offset += count;
            return result;
        };
        const auto read_scalar = [&](const std::vector<char>& in, std::size_t& offset)
        {
            const auto high = std::bit_cast<double>(word(in, offset)), low = std::bit_cast<double>(word(in, offset));
            const auto exponent = std::bit_cast<std::int64_t>(word(in, offset));
            if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
                throw std::runtime_error("Finite face handoff exponent is malformed.");
            const auto value = std::scalbn(static_cast<long double>(high) + low, int(exponent));
            if (!std::isfinite(value)) throw std::runtime_error("Finite face handoff scalar is nonfinite.");
            return value;
        };
        std::map<GO, const Meshes::AMR::AdaptiveTile*> target_tiles;
        std::vector<OwnerPacket> packets;
        collective([&]
        {
            if (old.finite_projection.face_rate.size() != mesh->num_owned_faces() ||
                old.finite_projection.face_radius.size() != mesh->num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks source rate authority.");
            std::size_t used = 0;
            for (const auto& tile : old.phases->tiles())
            {
                const auto face = mesh->owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0) continue;
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_word(packet.bytes, 0);
                append_word(packet.bytes, old.phases->accepted_control());
                bytes(packet.bytes, key);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                bytes(packet.bytes, tile.second ? Meshes::AMR::encode_adaptive_key(*tile.second) : std::vector<char>{});
                scalar(packet.bytes, old.finite_projection.face_rate[face]);
                scalar(packet.bytes, old.finite_projection.face_radius[face]);
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > options.byte_budget / 2) throw std::runtime_error("Finite face source directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
            for (const auto& tile : state.phases->tiles())
            {
                const auto face = target.mesh->owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0) continue;
                target_tiles.emplace(GO(tile.face_gid), &tile);
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_word(packet.bytes, 1);
                append_word(packet.bytes, old.phases->accepted_control());
                bytes(packet.bytes, key);
                append_word(packet.bytes, tile.face_gid);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > options.byte_budget / 2) throw std::runtime_error("Finite face target directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
        });
        const auto directory = exchange_owner_packets(packets, comm, options.byte_budget / 2);
        struct SourceFace {std::vector<char> first, second; long double rate, radius;};
        struct Request {std::vector<char> identity, first; std::uint64_t face; int owner;};
        std::map<std::vector<char>, SourceFace> sources;
        std::vector<Request> requests;
        packets.clear();
        collective([&]
        {
            for (const auto& packet : directory.packets)
            {
                std::size_t offset = 0;
                const auto type = word(packet.bytes, offset), control = word(packet.bytes, offset);
                auto key = read_bytes(packet.bytes, offset);
                if (control != old.phases->accepted_control() || authority(key) != comm->getRank())
                    throw std::runtime_error("Finite face directory has stale or foreign source authority.");
                if (type == 0)
                {
                    auto first = read_bytes(packet.bytes, offset), second = read_bytes(packet.bytes, offset);
                    const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                    if (radius < 0 || !sources.emplace(key, SourceFace{first, second, rate, radius}).second)
                        throw std::runtime_error("Finite canonical source face is duplicated.");
                }
                else if (type == 1)
                {
                    const auto face = word(packet.bytes, offset);
                    requests.push_back({key, read_bytes(packet.bytes, offset), face, packet.peer});
                }
                else throw std::runtime_error("Finite face directory type is malformed.");
                if (offset != packet.bytes.size()) throw std::runtime_error("Finite face directory has trailing payload.");
            }
            for (const auto& request : requests)
            {
                const auto found = sources.find(request.identity);
                if (found == sources.end())
                    throw std::runtime_error("Finite adaptive face handoff requires a bounded surface-overlap certificate.");
                const auto& source = found->second;
                const auto sign = request.first == source.first ? 1 : request.first == source.second ? -1 : 0;
                if (!sign) throw std::runtime_error("Finite face handoff incident-cell orientation is foreign.");
                OwnerPacket reply;
                reply.peer = request.owner;
                append_word(reply.bytes, request.face);
                bytes(reply.bytes, request.identity);
                scalar(reply.bytes, sign * source.rate);
                scalar(reply.bytes, source.radius);
                packets.push_back(std::move(reply));
            }
        });
        const auto result = exchange_owner_packets(packets, comm, options.byte_budget / 2);
        target.collective([&]
        {
            state.finite_projection.byte_budget = options.byte_budget;
            state.finite_projection.face_rate.resize(target.mesh->num_owned_faces());
            state.finite_projection.face_radius.resize(target.mesh->num_owned_faces());
            state.finite_projection.cell_radius.resize(target.volume.size());
            state.face_velocity.resize(target.mesh->num_faces());
            std::set<GO> seen;
            for (const auto& packet : result.packets)
            {
                std::size_t offset = 0;
                const auto gid = GO(word(packet.bytes, offset));
                const auto key = read_bytes(packet.bytes, offset);
                const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                const auto local = target.mesh->owned_face_map()->getLocalElement(gid);
                if (local < 0 || !seen.insert(gid).second || !target_tiles.contains(gid) ||
                    identity(*target_tiles.at(gid)) != key || packet.peer != authority(key) ||
                    offset != packet.bytes.size() || radius < 0)
                    throw std::runtime_error("Finite ownership face handoff has foreign or duplicate target.");
                if (target.mesh->is_geometry_exterior_face(local) && rate != 0)
                    throw std::runtime_error("Finite ownership handoff changes a closed wall.");
                const auto velocity = fp::divide({rate, radius}, {target.mesh->face_area(local), 0});
                state.face_velocity[local] = static_cast<double>(velocity.value);
                if (!std::isfinite(state.face_velocity[local]))
                    throw std::runtime_error("Finite ownership face velocity diagnostic overflows.");
                state.finite_projection.face_rate[local] = rate;
                state.finite_projection.face_radius[local] = radius;
            }
            if (seen.size() != target.mesh->num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks target face coverage.");
        });
        const auto rates = target.finite_rates(state);
        double local_divergence = 0;
        target.collective([&]
        {
            const auto lease = target.mesh->acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                fp::Sum net;
                for (const auto face : target.mesh->faces(LO(c)))
                    net.add(target.mesh->owner_cell(face) == LO(c) ? rates[face] : fp::negate(rates[face]));
                const auto residual = net.value();
                if (std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite ownership handoff loses canonical source-free closure.");
                state.finite_projection.cell_radius[c] = residual.absolute_error;
                local_divergence = std::max(local_divergence, fp::upper_double(
                    fp::bound_add(std::abs(residual.value), residual.absolute_error) / target.volume[c]));
            }
        });
        state.face_velocity = target.flow.faces(state.face_velocity);
        const auto divergence = target.maximum(local_divergence);
        target.require(divergence <= options.flow.divergence_tolerance,
            "Finite fixed-clock face handoff exceeds the unchanged physical divergence target.");
        return divergence;
    }
    std::unique_ptr<Impl> remap_finite(bool migration, std::span<const int> destinations,
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
    {
        namespace fp = VoF::finite_detail;
        const auto& old = *accepted;
        const auto hierarchy = old.phases->hierarchy();
        std::vector<int> ordered;
        if (migration)
        {
            require(destinations.size() == old.phases->owned_order().size(), "Finite migration destination shape mismatch.");
            collective([&]
            {
                std::map<VoF::GeneralVoF::Key, int> requested;
                for (std::size_t i = 0; i < destinations.size(); ++i) requested.emplace(old.phases->owned_order()[i], destinations[i]);
                for (const auto& [key, leaf] : hierarchy->owned()) ordered.push_back(requested.at(key));
            });
        }
        const auto plan = migration ? hierarchy->plan_migration(ordered) : hierarchy->plan_adaptation(refine, coarsen);
        const auto raw_target = hierarchy->apply(plan);
        const auto native = Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
        const auto bound = raw_target->with_partition(native.partition, native.mesh->owned_cell_map());
        const auto target = hierarchy->apply(plan, bound);
        VoF::GeneralVoFFiniteRemapOptions remap_options;
        remap_options.byte_budget = options.byte_budget;
        remap_options.phase_density = {material.gas_density, material.liquid_density};
        remap_options.channels = {{"momentum_x", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
            {"momentum_y", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
            {"momentum_z", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
            {"pressure_volume", VoF::GeneralVoFFiniteRemapCarrier::BulkVolume, false}};
        std::vector<VoF::GeneralVoFFiniteOwnedChannels> channels;
        collective([&]
        {
            const auto geometry = old.phases->owned_finite_geometry();
            for (std::size_t c = 0; c < old.velocity.size(); ++c)
            {
                VoF::GeneralVoFFiniteOwnedChannels item;
                item.key = old.phases->owned_order()[c];
                for (unsigned a = 0; a < 3; ++a)
                    item.amounts.push_back(fp::multiply({old.finite_mass[c], old.finite_mass_error[c]}, {old.velocity[c][a], 0}));
                item.amounts.push_back(fp::multiply(geometry[c]->volume(), {old.pressure[c], 0}));
                channels.push_back(std::move(item));
            }
        });
        const auto trial = old.phases->plan_remap_finite(plan, target, channels, remap_options);
        trial.validate_observations();
        const auto phases = trial.candidate();
        auto candidate = std::make_unique<Impl>(phases, material, options);
        auto state = std::make_shared<State>();
        state->phases = phases;
        candidate->phase_storage_finite(*state);
        state->velocity.resize(candidate->volume.size());
        state->pressure.resize(candidate->volume.size());
        std::map<VoF::GeneralVoF::Key, const VoF::GeneralVoFFiniteOwnedChannels*> result;
        for (const auto& item : trial.owned_channels()) result.emplace(item.key, &item);
        candidate->collective([&]
        {
            const auto geometry = phases->owned_finite_geometry();
            for (std::size_t c = 0; c < candidate->volume.size(); ++c)
            {
                const auto& item = *result.at(phases->owned_order()[c]);
                if (item.amounts.size() != 4) throw std::runtime_error("Finite remap channel schema differs.");
                for (unsigned a = 0; a < 3; ++a)
                    state->velocity[c][a] = fp::divide(item.amounts[a],
                        {state->finite_mass[c], state->finite_mass_error[c]}).as_double();
                state->pressure[c] = fp::divide(item.amounts[3], geometry[c]->volume()).as_double();
            }
        });
        candidate->surface_curvature(*state);
        const auto handoff_divergence = trial.ownership_only() ? handoff_finite_owned_faces(old, *candidate, *state) :
            handoff_finite_adapted_faces(old, *candidate, *state, trial);
        state->diagnostics = old.diagnostics;
        state->diagnostics.time = old.diagnostics.time;
        state->momentum_transport = old.momentum_transport;
        state->diagnostics.maximum_divergence = handoff_divergence;
        candidate->budget_finite(*state);
        // Recovering a finite velocity must preserve the original physical
        // phase-mass-weighted momentum gate independently of channel transfer.
        for (unsigned a = 0; a < 3; ++a)
        {
            fp::Sum change, reference;
            for (std::size_t c = 0; c < old.velocity.size(); ++c)
            {
                const auto amount = fp::multiply({old.finite_mass[c], old.finite_mass_error[c]}, {old.velocity[c][a], 0});
                change.add(fp::negate(amount));
                reference.add({std::abs(amount.value), amount.absolute_error});
            }
            for (std::size_t c = 0; c < state->velocity.size(); ++c)
            {
                const auto amount = fp::multiply({state->finite_mass[c], state->finite_mass_error[c]}, {state->velocity[c][a], 0});
                change.add(amount);
                reference.add({std::abs(amount.value), amount.absolute_error});
            }
            const auto residual = change.value(), l1 = reference.value();
            int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
            if (l1.value != 0) (void) std::frexp(l1.value + l1.absolute_error, &local_exponent);
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
            const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
            const auto center = residual.value / scale;
            const auto narrowed_center = static_cast<double>(center);
            const auto lower_reference = std::max(0.L, l1.value - l1.absolute_error) / scale;
            auto narrowed_reference = static_cast<double>(lower_reference);
            if (static_cast<long double>(narrowed_reference) > lower_reference)
                narrowed_reference = std::nextafter(narrowed_reference, 0.);
            double local[4] = {narrowed_center,
                fp::upper_double(fp::bound_add(residual.absolute_error / scale, std::abs(center - narrowed_center))),
                narrowed_reference, fp::upper_double(std::abs(center))}, global[4]{};
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
            const auto u = (8.L * mesh->owned_cell_map()->getComm()->getSize() + 16) * std::numeric_limits<double>::epsilon();
            const auto error = (global[1] + u * (global[3] + global[1])) / (1 - u);
            require(std::abs(global[0]) + error <= options.momentum_relative_tolerance * global[2] * (1 - u),
                "Finite fixed-clock remap changed physical phase-mass-weighted momentum.");
        }
        trial.validate_observations();
        candidate->accepted = std::move(state);
        return candidate;
    }
