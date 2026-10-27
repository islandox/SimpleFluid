/** @file AdaptedFaceTransfer.tcc
 * @brief Compiled transfer kernel with explicit source/target execution boundaries.
 */
#include "FVM/ExactFaceTransfer.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/GeneralVoF/AdaptedFaceTransfer.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode>
FaceTransferResult transfer_adapted_faces(const FaceTransferSource& source_input, const FaceTransferTarget& target,
    FaceTransferControls controls, const VoF::GeneralVoF::FiniteRemapTrial& trial)
{
    using LO = DefaultTpetraTypes::local_ordinal_type;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    const Execution source_execution(source_input.mesh), target_execution(target.mesh);
    FaceTransferResult transfer;
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    source_execution.require(target.arithmetic_mode == source_input.arithmetic_mode &&
                                 target.phases.arithmetic_mode() == source_input.arithmetic_mode,
        "General VoF adapted handoff changed arithmetic mode.");
    namespace fp = VoF::finite_detail;
    using Point = PLIC::Point<Mode>;
    using PA = SimpleFluid::geometry::swept_volume::PointArithmetic<Mode>;
    using Value = PLIC::DynamicGeometryValue;
    using Key = VoF::GeneralVoF::Key;
    using Loop = std::vector<Point>;
    using Support = PLIC::PlaneAuthority;
    const auto comm = source_input.mesh.owned_cell_map()->getComm();
    const auto& common = trial.source_common_cells();
    std::vector<PLIC::ConvexGeometryPtr<Mode>> common_geometry;
    std::size_t common_geometry_bytes = 0;
    source_execution.collective(
        [&]
        {
            constexpr auto per_cell = sizeof(PLIC::ConvexGeometryPtr<Mode>) + sizeof(PLIC::ConvexCellGeometry<Mode>);
            if (common.size() > controls.byte_budget / per_cell)
                throw std::runtime_error("Finite common-cell typed geometry storage cap exhausted.");
            common_geometry_bytes = common.size() * per_cell;
            common_geometry.reserve(common.size());
            for (const auto& cell : common)
                common_geometry.push_back(PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.geometry));
        });
    std::vector<Key> requested;
    std::map<Key, std::vector<std::size_t>> columns;
    for (std::size_t i = 0; i < common.size(); ++i)
    {
        columns[common[i].source_key].push_back(i);
        requested.push_back(common[i].target_key);
    }
    const auto draft = Meshes::AMR::fetch_adaptive_native_draft(*target.phases.hierarchy(), requested);
    std::map<Key, SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> native_sources;
    std::map<std::uint64_t, std::set<Key>> native_fans;
    source_execution.collective(
        [&]
        {
            for (const auto* leaves : {&draft.owned, &draft.ghosts})
                for (const auto& leaf : *leaves)
                    native_sources.emplace(leaf.key, target.phases.hierarchy()->geometry(leaf));
            for (const auto& tile : draft.tiles)
                for (auto node : tile.node_gids)
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && native_sources.contains(*key))
                            native_fans[node].insert(*key);
        });
    std::map<std::uint64_t, Point> native_points;
    source_execution.collective(
        [&]
        {
            for (const auto& node : draft.nodes)
            {
                const auto fan = native_fans.find(node.gid);
                if (fan == native_fans.end() || fan->second.empty())
                    continue;
                std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> cells;
                for (const auto& key : fan->second)
                    cells.push_back(native_sources.at(key));
                native_points.emplace(node.gid, PLIC::source_point<Mode>(cells, node.coordinate_identity));
            }
        });
    std::map<std::uint64_t, Loop> target_loops;
    std::map<Key, std::vector<const Meshes::AMR::AdaptiveTile*>> target_faces;
    source_execution.collective(
        [&]
        {
            std::set<Key> required(requested.begin(), requested.end());
            for (const auto& tile : draft.tiles)
                if (required.contains(tile.first))
                {
                    Loop loop;
                    for (auto node : tile.node_gids)
                        loop.push_back(native_points.at(node));
                    target_loops.emplace(tile.face_gid, std::move(loop));
                    target_faces[tile.first].push_back(&tile);
                }
        });
    const auto rates =
        face_rates<Mode>(source_input.mesh, source_input.phases, source_input.projection, source_input.flow);
    std::map<Key, std::vector<const Meshes::AMR::AdaptiveTile*>> source_faces;
    std::map<std::uint64_t, Loop> source_loops;
    source_execution.collective(
        [&]
        {
            for (const auto& tile : source_input.phases.tiles())
            {
                if (columns.contains(tile.first))
                    source_faces[tile.first].push_back(&tile);
                if (tile.second && columns.contains(*tile.second))
                    source_faces[*tile.second].push_back(&tile);
                source_loops.emplace(tile.face_gid, source_input.phases.finite_face_points<Mode>(tile.face_gid));
            }
        });
    PLIC::GeometryLimits geometry_limits;
    geometry_limits.byte_budget = controls.byte_budget / 2;
    geometry_limits.arithmetic_mode = source_input.arithmetic_mode;
    std::size_t pairs = 0, patches = 0, retained_bytes = common_geometry_bytes;
    const auto charge = [&](std::size_t bytes)
    {
        if (bytes > controls.byte_budget - std::min(retained_bytes, controls.byte_budget))
            throw std::runtime_error("Finite common-face handoff storage cap exhausted.");
        retained_bytes += bytes;
    };
    const auto shared_support = [](const Loop& first, const Loop& second) -> std::optional<Support>
    {
        const auto a = PLIC::common_supports<Mode>(first), b = PLIC::common_supports<Mode>(second);
        for (const auto& key : a)
            if (std::find(b.begin(), b.end(), key) != b.end())
                return key;
        return {};
    };
    const auto overlap = [&](const Loop& first, const Loop& second)
    {
        if (++pairs > VoF::ExactFaceTransferOptions{}.maximum_pair_tests)
            throw std::runtime_error("Finite common-face pair cap exhausted.");
        const auto support = shared_support(first, second);
        if (!support)
            return PLIC::PlanarFaceOverlap<Mode>{};
        auto result = PLIC::planar_face_overlap<Mode>(first, second, *support, geometry_limits);
        if (!result.overlap_vertices.empty())
        {
            if (++patches > VoF::ExactFaceTransferOptions{}.maximum_patches)
                throw std::runtime_error("Finite common-face patch cap exhausted.");
            charge(sizeof(result) + result.overlap_vertices.capacity() * sizeof(Point));
            for (const auto& point : result.overlap_vertices)
                charge(PLIC::point_storage_bytes<Mode>(point) - sizeof(Point));
        }
        return result;
    };
    const auto nonempty = [](const PLIC::PlanarFaceOverlap<Mode>& value)
    {
        if (value.overlap_projected_area.value == 0 &&
            SimpleFluid::geometry::swept_volume::runtime_value<Mode>(value.overlap_projected_area).absolute_error == 0)
            return false;
        if (!(value.overlap_projected_area.value >
                SimpleFluid::geometry::swept_volume::runtime_value<Mode>(value.overlap_projected_area).absolute_error))
            throw std::runtime_error("Finite common-face overlap positivity is unresolved.");
        return true;
    };
    const auto certify_cover = [&](const Loop& whole, const std::vector<Loop>& pieces)
    {
        VoF::finite_detail::SumFor<Mode> fraction(source_input.arithmetic_mode);
        for (std::size_t i = 0; i < pieces.size(); ++i)
        {
            const auto part = overlap(whole, pieces[i]);
            if (!nonempty(part))
                throw std::runtime_error("Finite common-face cover contains empty piece.");
            fraction.add(part.orientation > 0 ? SimpleFluid::geometry::swept_volume::runtime_value<Mode>(part.overlap_over_first)
                                              : A::negate(SimpleFluid::geometry::swept_volume::runtime_value<Mode>(part.overlap_over_first)));
            for (std::size_t j = 0; j < i; ++j)
            {
                const auto collision = overlap(pieces[i], pieces[j]);
                if (collision.overlap_projected_area.value != 0 ||
                    SimpleFluid::geometry::swept_volume::runtime_value<Mode>(collision.overlap_projected_area).absolute_error != 0)
                    throw std::runtime_error("Finite common-face cover has overlapping or unresolved interiors.");
            }
        }
        const auto difference = A::subtract(fraction.value(), A::make_value(1, 0));
        if (bounded_mode
                ? (std::abs(difference.value) > difference.absolute_error ||
                      difference.absolute_error > VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                : std::abs(difference.value) > VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
            throw std::runtime_error("Finite common-face cover has a hole or exhausted arithmetic budget.");
    };
    struct Part
    {
        Loop loop;
        Value rate;
        std::optional<std::size_t> edge;
        int sign = 1;
    };
    struct Facet
    {
        std::size_t cell;
        Loop loop;
        std::vector<Part> parts;
        bool inherited = false;
    };
    struct Internal
    {
        std::size_t first, second;
        Value rate;
    };
    struct Output
    {
        Key source, target;
        std::uint64_t source_revision, face, proof;
        int target_owner;
        Value rate, fraction;
    };
    std::vector<Output> outputs;
    source_execution.collective(
        [&]
        {
            for (const auto& [source_key, unordered_indices] : columns)
            {
                auto indices = unordered_indices;
                std::sort(indices.begin(), indices.end(),
                    [&](auto a, auto b) { return common[a].target_key < common[b].target_key; });
                const auto source_row =
                    source_input.mesh.owned_cell_map()->getLocalElement(GO(common[indices.front()].source_gid));
                if (source_row < 0)
                    throw std::runtime_error("Finite common-face source is not locally owned.");
                const auto velocity = source_input.velocity[source_row];
                std::vector<Facet> facets;
                std::vector<std::vector<std::size_t>> cell_facets(indices.size());
                std::map<std::uint64_t, std::vector<Loop>> old_cover;
                for (std::size_t c = 0; c < indices.size(); ++c)
                {
                    const auto& geometry = *common_geometry[indices[c]];
                    const auto points = geometry.points();
                    for (const auto& face : geometry.faces())
                    {
                        Facet facet{c, {}, {}};
                        for (auto vertex : face)
                            facet.loop.push_back(points[vertex]);
                        for (const auto* tile : source_faces.at(source_key))
                        {
                            const auto part = overlap(facet.loop, source_loops.at(tile->face_gid));
                            if (!nonempty(part))
                                continue;
                            const auto local =
                                source_input.mesh.overlap_face_map()->getLocalElement(GO(tile->face_gid));
                            if (local < 0)
                                throw std::runtime_error("Finite source face rate is absent.");
                            const auto rate =
                                A::multiply(rates[local], SimpleFluid::geometry::swept_volume::runtime_value<Mode>(part.overlap_over_second));
                            facet.parts.push_back({part.overlap_vertices, rate, {}, 1});
                            old_cover[tile->face_gid].push_back(part.overlap_vertices);
                        }
                        if (!facet.parts.empty())
                        {
                            std::vector<Loop> cover;
                            for (const auto& part : facet.parts)
                                cover.push_back(part.loop);
                            certify_cover(facet.loop, cover);
                            facet.inherited = true;
                        }
                        charge(sizeof(Facet) + facet.loop.size() * sizeof(Point));
                        if (++patches > VoF::ExactFaceTransferOptions{}.maximum_patches)
                            throw std::runtime_error("Finite common-face patch cap exhausted.");
                        cell_facets[c].push_back(facets.size());
                        facets.push_back(std::move(facet));
                    }
                }
                for (const auto* tile : source_faces.at(source_key))
                {
                    const auto cover = old_cover.find(tile->face_gid);
                    if (cover == old_cover.end())
                        throw std::runtime_error("Finite old source face is uncovered.");
                    certify_cover(source_loops.at(tile->face_gid), cover->second);
                }
                std::vector<Internal> internal;
                for (std::size_t f = 0; f < facets.size(); ++f)
                {
                    if (facets[f].inherited)
                        continue;
                    for (std::size_t g = f + 1; g < facets.size(); ++g)
                    {
                        if (facets[g].inherited || facets[g].cell == facets[f].cell)
                            continue;
                        const auto piece = overlap(facets[f].loop, facets[g].loop);
                        if (!nonempty(piece))
                            continue;
                        if (piece.orientation != -1)
                            throw std::runtime_error("Finite interior common faces have equal orientation.");
                        Point area;
                        const auto& loop = piece.overlap_vertices;
                        for (std::size_t i = 1; i + 1 < loop.size(); ++i)
                            area = PA::add(
                                area, PA::cross(PA::subtract(loop[i], loop[0]), PA::subtract(loop[i + 1], loop[0])));
                        const auto rate = A::divide(PA::dot(area, Point(A::point(velocity))), A::make_value(2, 0));
                        const auto index = internal.size();
                        internal.push_back({facets[f].cell, facets[g].cell, rate});
                        facets[f].parts.push_back({loop, A::make_value(0), index, 1});
                        auto reversed = loop;
                        std::reverse(reversed.begin(), reversed.end());
                        facets[g].parts.push_back({std::move(reversed), A::make_value(0), index, -1});
                    }
                }
                for (const auto& face : facets)
                {
                    std::vector<Loop> pieces;
                    for (const auto& part : face.parts)
                        pieces.push_back(part.loop);
                    certify_cover(face.loop, pieces);
                }
                const auto divergences = [&]
                {
                    std::vector<VoF::finite_detail::SumFor<Mode>> sums(
                        indices.size(), VoF::finite_detail::SumFor<Mode>(source_input.arithmetic_mode));
                    for (const auto& face : facets)
                        for (const auto& part : face.parts)
                            sums[face.cell].add(part.edge ? (part.sign > 0 ? internal[*part.edge].rate
                                                                           : A::negate(internal[*part.edge].rate))
                                                          : part.rate);
                    std::vector<Value> result;
                    for (const auto& sum : sums)
                        result.push_back(sum.value());
                    return result;
                };
                std::vector<std::vector<std::size_t>> adjacent(indices.size());
                for (std::size_t e = 0; e < internal.size(); ++e)
                {
                    adjacent[internal[e].first].push_back(e);
                    adjacent[internal[e].second].push_back(e);
                }
                const auto none = std::numeric_limits<std::size_t>::max();
                std::vector<std::size_t> parent(indices.size(), none), parent_edge(indices.size(), none), order{0};
                parent[0] = 0;
                for (std::size_t i = 0; i < order.size(); ++i)
                    for (auto edge : adjacent[order[i]])
                    {
                        const auto other =
                            internal[edge].first == order[i] ? internal[edge].second : internal[edge].first;
                        if (parent[other] == none)
                        {
                            parent[other] = order[i];
                            parent_edge[other] = edge;
                            order.push_back(other);
                        }
                    }
                if (order.size() != indices.size())
                    throw std::runtime_error("Finite common refinement is not face connected.");
                auto defect = divergences();
                for (std::size_t i = order.size(); i-- > 1;)
                {
                    const auto child = order[i], edge = parent_edge[child];
                    internal[edge].rate = A::add(
                        internal[edge].rate, internal[edge].first == child ? A::negate(defect[child]) : defect[child]);
                    defect[parent[child]] = A::add(defect[parent[child]], defect[child]);
                }
                const auto remaining = divergences();
                for (std::size_t cell = 0; cell < remaining.size(); ++cell)
                {
                    const auto& residual = remaining[cell];
                    const auto allowed = bounded_mode ? residual.absolute_error
                                                      : static_cast<long double>(controls.divergence_tolerance) *
                                                            common_geometry[indices[cell]]->volume().value;
                    if (std::abs(residual.value) > allowed)
                        throw std::runtime_error("Finite new common-face extension fails source-free closure.");
                }
                std::uint64_t proof = 0;
                for (const auto& face : facets)
                    for (const auto& part : face.parts)
                    {
                        const auto& cell = common[indices[face.cell]];
                        const auto rate = part.edge ? (part.sign > 0 ? internal[*part.edge].rate
                                                                     : A::negate(internal[*part.edge].rate))
                                                    : part.rate;
                        for (const auto* tile : target_faces.at(cell.target_key))
                        {
                            const auto piece = overlap(part.loop, target_loops.at(tile->face_gid));
                            if (!nonempty(piece))
                                continue;
                            if (piece.orientation != 1)
                                throw std::runtime_error("Finite target boundary has inconsistent orientation.");
                            outputs.push_back({source_key, cell.target_key,
                                source_input.phases.hierarchy()->owned().at(source_key).revision, tile->face_gid,
                                proof++, tile->first_owner,
                                A::multiply(rate, SimpleFluid::geometry::swept_volume::runtime_value<Mode>(piece.overlap_over_first)),
                                SimpleFluid::geometry::swept_volume::runtime_value<Mode>(piece.overlap_over_second)});
                        }
                    }
            }
        });
    // The producer has independently proved every common-cell facet cover,
    // inherited source-face cover and opposite internal incidence. Together
    // with the opaque phase trial's donor/recipient cover, these immutable
    // proofs compose into a target boundary partition. The receiver checks
    // the bounded area closure and authenticates each proof exactly once.
    const auto put_scalar = [](std::vector<char>& out, long double value)
    {
        int exponent = 0;
        const auto m = std::frexp(value, &exponent);
        const auto hi = static_cast<double>(m), lo = static_cast<double>(m - hi);
        if (!std::isfinite(value) || static_cast<long double>(hi) + lo != m)
            throw std::runtime_error("Finite adapted face scalar exceeds wire precision.");
        append_face_word(out, std::bit_cast<std::uint64_t>(hi));
        append_face_word(out, std::bit_cast<std::uint64_t>(lo));
        append_face_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
    };
    const auto put_bytes = [](std::vector<char>& out, const std::vector<char>& key)
    {
        append_face_word(out, key.size());
        out.insert(out.end(), key.begin(), key.end());
    };
    std::vector<OwnerPacket> packets;
    source_execution.collective(
        [&]
        {
            std::size_t used = 0;
            for (const auto& item : outputs)
            {
                OwnerPacket packet;
                packet.peer = item.target_owner;
                append_face_word(packet.bytes, source_input.phases.accepted_control());
                append_face_word(packet.bytes, target.phases.accepted_control());
                put_bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(item.source));
                append_face_word(packet.bytes, item.source_revision);
                put_bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(item.target));
                append_face_word(packet.bytes, item.face);
                append_face_word(packet.bytes, item.proof);
                for (const auto value : {item.rate, item.fraction})
                {
                    put_scalar(packet.bytes, value.value);
                    put_scalar(packet.bytes, value.absolute_error);
                }
                used += sizeof(OwnerPacket) + packet.bytes.size();
                if (used > controls.byte_budget / 2)
                    throw std::runtime_error("Finite adapted face packet cap exhausted.");
                packets.push_back(std::move(packet));
            }
        });
    const auto received = exchange_owner_packets(packets, comm, controls.byte_budget / 2);
    struct Incoming
    {
        Key source, target;
        std::uint64_t revision, face, proof;
        int owner;
        Value rate, fraction;
    };
    std::vector<Incoming> incoming;
    std::vector<Key> donor_keys;
    target_execution.collective(
        [&]
        {
            for (const auto& packet : received.packets)
            {
                std::size_t offset = 0;
                const auto word = [&]()
                {
                    const auto value = read_face_word(packet.bytes, offset);
                    offset += 8;
                    return value;
                };
                const auto blob = [&]()
                {
                    const auto count = word();
                    if (count > packet.bytes.size() - std::min(offset, packet.bytes.size()))
                        throw std::runtime_error("Finite adapted face identity is malformed.");
                    std::vector<char> value(packet.bytes.begin() + offset, packet.bytes.begin() + offset + count);
                    offset += count;
                    return value;
                };
                const auto scalar = [&]()
                {
                    const auto hi = std::bit_cast<double>(word()), lo = std::bit_cast<double>(word());
                    const auto exponent = std::bit_cast<std::int64_t>(word());
                    if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
                        throw std::runtime_error("Finite adapted face exponent is malformed.");
                    const auto value = std::scalbn(static_cast<long double>(hi) + lo, int(exponent));
                    if (!std::isfinite(value))
                        throw std::runtime_error("Finite adapted face scalar is nonfinite.");
                    return value;
                };
                if (word() != source_input.phases.accepted_control() || word() != target.phases.accepted_control())
                    throw std::runtime_error("Finite adapted face proof publication is stale.");
                Incoming item;
                item.source = Meshes::AMR::decode_adaptive_key(blob());
                item.revision = word();
                item.target = Meshes::AMR::decode_adaptive_key(blob());
                item.face = word();
                item.proof = word();
                const auto rate_value = scalar(), rate_error = scalar();
                const auto fraction_value = scalar(), fraction_error = scalar();
                item.rate = A::make_value(rate_value, rate_error);
                item.fraction = A::make_value(fraction_value, fraction_error);
                item.owner = packet.peer;
                if (offset != packet.bytes.size() || item.rate.absolute_error < 0 || item.fraction.absolute_error < 0 ||
                    item.fraction.value < (bounded_mode ? item.fraction.absolute_error : 0))
                    throw std::runtime_error("Finite adapted face proof has malformed bounded values.");
                donor_keys.push_back(item.source);
                incoming.push_back(std::move(item));
            }
        });
    const auto owners = source_input.phases.hierarchy()->lookup_cells(donor_keys);
    target_execution.collective(
        [&]
        {
            std::map<GO, const Meshes::AMR::AdaptiveTile*> actual;
            for (const auto& tile : target.phases.tiles())
                if (target.mesh.owned_face_map()->getLocalElement(GO(tile.face_gid)) >= 0)
                    actual.emplace(GO(tile.face_gid), &tile);
            std::vector<VoF::finite_detail::SumFor<Mode>> flux(
                target.mesh.num_owned_faces(), VoF::finite_detail::SumFor<Mode>(source_input.arithmetic_mode)),
                coverage(flux.size(), VoF::finite_detail::SumFor<Mode>(source_input.arithmetic_mode));
            std::set<std::tuple<Key, std::uint64_t, std::uint64_t>> seen;
            for (std::size_t i = 0; i < incoming.size(); ++i)
            {
                const auto& item = incoming[i];
                const auto local = target.mesh.owned_face_map()->getLocalElement(GO(item.face));
                if (owners[i].owner != item.owner || owners[i].revision != item.revision || local < 0 ||
                    actual.at(GO(item.face))->first != item.target ||
                    !seen.emplace(item.source, item.face, item.proof).second)
                    throw std::runtime_error(
                        "Finite adapted face proof has foreign or duplicate source/target authority.");
                flux[local].add(item.rate);
                coverage[local].add(item.fraction);
            }
            transfer.projection.byte_budget = controls.byte_budget;
            transfer.projection.arithmetic_mode = source_input.arithmetic_mode;
            transfer.projection.face_rate.resize(flux.size());
            transfer.projection.face_radius.resize(flux.size());
            transfer.projection.cell_radius.resize(target.volume.size());
            transfer.face_velocity.resize(target.mesh.num_faces());
            for (std::size_t f = 0; f < flux.size(); ++f)
            {
                const auto cover = A::subtract(coverage[f].value(), A::make_value(1, 0));
                if (bounded_mode
                        ? (std::abs(cover.value) > cover.absolute_error ||
                              cover.absolute_error > VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                        : std::abs(cover.value) > VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                    throw std::runtime_error("Finite adapted target face has incomplete bounded coverage.");
                const auto rate = flux[f].value();
                if (target.mesh.is_geometry_exterior_face(LO(f)) && (rate.value != 0 || rate.absolute_error != 0))
                    throw std::runtime_error("Finite adapted handoff changes a physical impermeable wall.");
                transfer.projection.face_rate[f] = rate.value;
                transfer.projection.face_radius[f] = rate.absolute_error;
                const auto diagnostic = A::divide(rate, A::make_value(target.mesh.face_area(LO(f)), 0));
                transfer.face_velocity[f] = static_cast<double>(diagnostic.value);
                if (!std::isfinite(transfer.face_velocity[f]))
                    throw std::runtime_error("Finite adapted face velocity diagnostic overflows.");
            }
        });
    const auto final_rates = face_rates<Mode>(target.mesh, target.phases, transfer.projection, target.flow);
    double local_divergence = 0;
    target_execution.collective(
        [&]
        {
            const auto lease = target.mesh.acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                VoF::finite_detail::SumFor<Mode> net(source_input.arithmetic_mode);
                for (auto face : target.mesh.faces(LO(c)))
                    net.add(target.mesh.owner_cell(face) == LO(c) ? final_rates[face] : A::negate(final_rates[face]));
                const auto residual = net.value();
                if (bounded_mode && std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite adapted handoff fails canonical target closure.");
                transfer.projection.cell_radius[c] = residual.absolute_error;
                const auto divergence_bound = A::divide(
                    A::make_value(bounded_mode ? fp::bound_add(std::abs(residual.value), residual.absolute_error)
                                               : std::abs(residual.value),
                        0),
                    A::make_value(target.volume[c], 0));
                local_divergence = std::max(local_divergence, fp::upper_double(A::upper(divergence_bound)));
            }
        });
    const auto divergence = target_execution.maximum(local_divergence);
    target_execution.require(divergence <= controls.divergence_tolerance,
        "Finite adapted handoff exceeds the unchanged physical divergence target.");
    transfer.face_velocity = target.flow.faces(transfer.face_velocity);
    trial.validate_observations();
    transfer.divergence = divergence;
    return transfer;
}
} // namespace SimpleFluid::detail::general_vof
