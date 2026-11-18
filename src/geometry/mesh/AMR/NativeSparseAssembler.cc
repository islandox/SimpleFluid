#include "geometry/mesh/AMR/NativeSparseAssembler.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include "geometry/details/NativeMeshHandleAccess.hh"
#include <algorithm>
#include <bit>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Admission = geometry_detail::NativeConstructionAdmission;
using ID = std::uint64_t;
template<class Pages, class Emit> void visit_rows(const typename Pages::Ptr& root, const Emit& emit)
{
    if (!root)
        return;
    if (root->leaf())
        for (const auto& row : root->rows)
            emit(row);
    else
        for (const auto& c : root->children)
            visit_rows<Pages>(c, emit);
}
std::shared_ptr<Budget> scalar_arena(const Admission& a)
{
    return std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(a.state()->counter_owner));
}
struct SourceRoute
{
    std::shared_ptr<const NativeCellSourceProof> source;
    std::shared_ptr<const NativeCellIncidenceProof> incidence;
    ID gid = 0;
    int owner = -1;
};
const SourceRoute* find_source(std::span<const SourceRoute> rows, const AdaptiveCellKey& key)
{
    const auto at = std::lower_bound(
        rows.begin(), rows.end(), key, [](const auto& row, const auto& key) { return row.source->source().key < key; });
    return at == rows.end() || at->source->source().key != key ? nullptr : &*at;
}
std::size_t region_ordinal(const NativeSparseFragments& fragments, const NativeFragmentRecord* record)
{
    const auto at = std::lower_bound(
        fragments.route.begin(), fragments.route.end(), record, std::less<const NativeFragmentRecord*>{});
    if (at == fragments.route.end() || *at != record)
        throw std::invalid_argument("Source fragment is absent from current full subset.");
    return at - fragments.route.begin();
}
std::size_t remaining(const Admission& a)
{
    const auto c = a.counters();
    return c.limit - c.used;
}
} // namespace
std::shared_ptr<NativePublicationViewData> NativeSparseAssembler::assemble(NativePublicationSession& session,
    const std::shared_ptr<NativePublicationNumbering>& numbering, std::span<const NativeRemoteSource> full,
    std::span<const NativeRemoteIncidence> remote_incidence, std::span<const NativeRemoteSource> outer)
{
    const auto& a = session.admission();
    const auto& input = session.inputs();
    if (!numbering || full.size() != remote_incidence.size())
        throw std::invalid_argument("Incomplete private sparse assembler input.");
    geometry_detail::NativeConstructorVector<SourceRoute> sources(a.allocator<SourceRoute>());
    geometry_detail::NativeConstructorVector<NativeSparseSealedCell> seals(a.allocator<NativeSparseSealedCell>());
    std::shared_ptr<NativeSparseFragments> fragments;
    std::shared_ptr<NativeSparseNumericIndex> numeric;
    collective(*input.comm,
        [&]
        {
            auto fragment_owner = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSparseFragments>(a, a);
            fragments = fragment_owner.owner;
            fragments->allocation = fragment_owner.envelope;
            if (numbering->fragments)
                fragments->root = numbering->fragments->root;
            auto numeric_owner = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeSparseNumericIndex>(a, a);
            numeric = numeric_owner.owner;
            numeric->allocation = numeric_owner.envelope;
        });
    NativeSparseFragments::Pages fragment_editor(scalar_arena(a), nullptr, {}, a.state());
    NativePublicationNumbering::RemotePages remote_editor(scalar_arena(a), nullptr, {}, a.state());
    NativePublicationNumbering::BoundaryPages boundary_editor(scalar_arena(a), nullptr, {}, a.state());
    collective(*input.comm,
        [&]
        {
            sources.reserve(input.layout->size() + full.size());
            std::size_t ordinal = 0;
            for (const auto& leaf : *input.layout)
            {
                const auto proof = session.source(leaf);
                const auto row = session.indexes()->find(leaf.key);
                if (!row || !row->incidence)
                    throw std::invalid_argument("Owned native source has no complete current incidence.");
                sources.push_back({proof, row->incidence, input.cell_begin + ordinal++, input.comm->getRank()});
            }
            for (std::size_t i = 0; i < full.size(); ++i)
            {
                if (!full[i].source || !remote_incidence[i].proof ||
                    &remote_incidence[i].proof->source() != full[i].source.get())
                    throw std::invalid_argument("Remote native incidence/source controls differ.");
                sources.push_back({full[i].source, remote_incidence[i].proof, full[i].gid, full[i].owner});
            }
            std::sort(sources.begin(), sources.end(),
                [](const auto& a, const auto& b) { return a.source->source().key < b.source->source().key; });
            for (std::size_t i = 1; i < sources.size(); ++i)
                if (sources[i - 1].source->source().key == sources[i].source->source().key)
                    throw std::invalid_argument("Duplicate current full native source.");
            fragments->route.reserve(sources.size());
            for (const auto& source : sources)
            {
                if (!source.source->fragment())
                    throw std::invalid_argument("Full source has no detached immutable native fragment.");
                fragments->route.push_back(source.source->fragment().get());
                NativeSparseFragments::Row selector{source.source->fragment()};
                const auto at = NativeSparseFragments::Pages::lower_bound(fragments->root, [&](const auto& row)
                    { return std::less<const void*>{}(row.fragment.get(), selector.fragment.get()); });
                if (!at || at->fragment.get() != selector.fragment.get())
                    fragments->root = fragment_editor.edit(fragments->root, selector, &selector, false);
            }
            std::sort(fragments->route.begin(), fragments->route.end(), std::less<const NativeFragmentRecord*>{});
            fragments->route.erase(
                std::unique(fragments->route.begin(), fragments->route.end()), fragments->route.end());
            if (numbering->fragments)
                visit_rows<NativeSparseFragments::Pages>(numbering->fragments->root,
                    [&](const auto& row)
                    {
                        if (!std::binary_search(fragments->route.begin(), fragments->route.end(), row.fragment.get(),
                                std::less<const NativeFragmentRecord*>{}))
                            fragments->root = fragment_editor.edit(fragments->root, row, nullptr, true);
                    });
            numeric->cells.reserve(sources.size() + outer.size());
            seals.reserve(sources.size());
            numeric->native_cells.reserve(sources.size());
            for (const auto& source : sources)
            {
                const auto region = region_ordinal(*fragments, source.source->fragment().get());
                const auto& metrics = source.source->metrics();
                numeric->cells.push_back({source.gid, source.owner, region, source.source->native_cell(), true,
                    metrics.volume, metrics.centroid});
                numeric->native_cells.push_back({{region, source.source->native_cell()}, source.gid});
                seals.push_back({source.gid, region, session.seal(source.source, source.incidence)});
            }
            for (const auto& metric : outer)
            {
                if (!metric.source || find_source(sources, metric.source->source().key))
                    throw std::invalid_argument("Outer metric source overlaps full native selection.");
                const auto& metrics = metric.source->metrics();
                numeric->cells.push_back({metric.gid, metric.owner, 0, 0, false, metrics.volume, metrics.centroid});
            }
            std::sort(numeric->cells.begin(), numeric->cells.end(),
                [](const auto& a, const auto& b) { return a.gid < b.gid; });
            std::sort(numeric->native_cells.begin(), numeric->native_cells.end());
            std::sort(seals.begin(), seals.end(), [](const auto& a, const auto& b) { return a.gid < b.gid; });
        });
    // Boundary strings are detached admitted owners, created only for newly
    // observed schema names. Every current numeric row borrows those owners.
    geometry_detail::NativeConstructorStorageOwner schema_storage;
    collective(*input.comm, [&] { schema_storage = geometry_detail::native_constructor_storage(a); });
    std::vector<OwnerPacket> schema_packets;
    collective(*input.comm,
        [&]
        {
            geometry_detail::NativeConstructorVector<std::string_view> names(a.allocator<std::string_view>());
            numbering->visit_current_pieces(*input.subfaces,
                [&](const CanonicalSubfaceView& view, const NativeNumberingPieceRoute&, std::size_t)
                {
                    if (!view.boundary().empty())
                        names.push_back(view.boundary());
                });
            std::sort(names.begin(), names.end());
            names.erase(std::unique(names.begin(), names.end()), names.end());
            std::size_t bytes = sizeof(ID);
            for (const auto name : names)
            {
                if (name.size() > SIZE_MAX - bytes - sizeof(ID))
                    throw std::invalid_argument("Native boundary schema extent overflows.");
                bytes += name.size() + sizeof(ID);
            }
            schema_storage.owner->reserve(schema_packets, input.comm->getSize());
            for (int rank = 0; rank < input.comm->getSize(); ++rank)
            {
                Writer w;
                schema_storage.owner->reserve(w.data, bytes);
                w.word(names.size());
                for (const auto name : names)
                    w.bytes({name.data(), name.size()});
                schema_packets.push_back({rank, std::move(w.data)});
            }
        });
    auto global_schema = exchange_adaptive_packets(schema_packets, input.comm, remaining(a));
    geometry_detail::NativeConstructorVector<std::string_view> names(a.allocator<std::string_view>());
    collective(*input.comm,
        [&]
        {
            for (const auto& packet : global_schema.packets)
            {
                Reader r{packet.bytes};
                const auto n = r.word();
                if (n > packet.bytes.size() / sizeof(ID))
                    throw std::invalid_argument("Invalid native boundary schema count.");
                for (ID i = 0; i < n; ++i)
                {
                    const auto name = r.bytes_view();
                    if (name.empty())
                        throw std::invalid_argument("Empty native boundary schema name.");
                    names.emplace_back(name.data(), name.size());
                }
                r.done();
            }
            std::sort(names.begin(), names.end());
            names.erase(std::unique(names.begin(), names.end()), names.end());
            if (names.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Native boundary schema exceeds ID range.");
            numeric->boundaries.reserve(names.size());
            for (std::size_t i = 0; i < names.size(); ++i)
            {
                const auto at = NativePublicationNumbering::BoundaryPages::lower_bound(
                    numbering->boundary_root, [&](const auto& row) { return row.record->name < names[i]; });
                std::shared_ptr<const NativeNumberingBoundary> owner;
                if (at && at->record->name == names[i])
                    owner = at->record;
                else
                {
                    auto created = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeNumberingBoundary>(a);
                    created.owner->allocation = created.envelope;
                    created.owner->metadata = geometry_detail::native_constructor_storage(a);
                    created.owner->metadata.owner->string(created.owner->name, names[i]);
                    owner = std::move(created.owner);
                    NativeNumberingBoundaryRow row{owner};
                    numbering->boundary_root = boundary_editor.edit(numbering->boundary_root, row, &row, false);
                }
                numeric->boundaries.push_back({static_cast<int>(i), &owner->name});
            }
            // Schema names are stable for a built-in source. Retire an absent name
            // explicitly rather than retaining a predecessor schema controller.
            const auto old = numbering->boundary_root;
            visit_rows<NativePublicationNumbering::BoundaryPages>(old,
                [&](const auto& row)
                {
                    if (!std::binary_search(names.begin(), names.end(), std::string_view(row.record->name)))
                        numbering->boundary_root = boundary_editor.edit(numbering->boundary_root, row, nullptr, true);
                });
        });
    auto boundary = [&](std::string_view name)
    {
        if (name.empty())
            return MultiRegionMesh::invalid_boundary_id;
        const auto at = std::lower_bound(numeric->boundaries.begin(), numeric->boundaries.end(), name,
            [](const auto& row, std::string_view name) { return *row.second < name; });
        if (at == numeric->boundaries.end() || *at->second != name)
            throw std::invalid_argument("Native face label is outside global schema.");
        return at->first;
    };
    auto add_node = [&](ID gid, PLIC::PointIdentity<ArithmeticMode::Ordinary> identity)
    { numeric->nodes.push_back({gid, {}, PLIC::point_display<ArithmeticMode::Ordinary>(identity)}); };
    auto add_alias = [&](const NativeFragmentRecord& fragment, ID native, ID gid)
    {
        if (native != UINT64_MAX)
            numeric->native_nodes.push_back({{region_ordinal(*fragments, &fragment), native}, gid});
    };
    auto logical = [&](RegionFace address, ID gid, int sign) { numeric->flux_signs.push_back({address, gid, sign}); };
    auto loop = [&](std::span<const ID> nodes)
    {
        const auto begin = numeric->loops.size();
        numeric->loops.insert(numeric->loops.end(), nodes.begin(), nodes.end());
        return begin;
    };
    collective(*input.comm,
        [&]
        {
            for (const auto& row : numbering->points)
                add_node(row.gid, row.record->point);
            const auto numbered_nodes = std::span<const ID>(numbering->loops);
            numbering->visit_current_pieces(*input.subfaces,
                [&](const CanonicalSubfaceView& view, const NativeNumberingPieceRoute& row, std::size_t)
                {
                    const auto& record = *row.record;
                    NativeSparseFaceRoute face;
                    face.gid = row.face_gid;
                    face.first = row.first_gid;
                    face.second = row.second_gid;
                    face.physical = &record.physical;
                    face.native = {region_ordinal(*fragments, record.first_fragment.get()), record.first_native_face};
                    if (record.second_fragment)
                        face.second_native =
                            RegionFace{region_ordinal(*fragments, record.second_fragment.get()), record.second_native_face};
                    const auto first = numbered_nodes.subspan(row.nodes_begin, row.nodes_size),
                               second = numbered_nodes.subspan(row.second_begin, row.second_size);
                    if (record.physical.override_geometry)
                    {
                        face.loop_begin = loop(first);
                        face.loop_size = first.size();
                    }
                    if (!record.physical.second_image.is_zero())
                    {
                        face.second_loop_begin = loop(second);
                        face.second_loop_size = second.size();
                    }
                    face.boundary = boundary(view.boundary());
                    logical(face.native, face.gid, record.first_sign);
                    if (face.second_native && *face.second_native != face.native)
                        logical(*face.second_native, face.gid, record.second_sign);
                    for (std::size_t i = 0; i < record.first_native_nodes.size(); ++i)
                        add_alias(*record.first_fragment, record.first_native_nodes[i], first[i]);
                    if (record.second_fragment)
                        for (std::size_t i = 0; i < record.second_native_nodes.size(); ++i)
                            add_alias(*record.second_fragment, record.second_native_nodes[i], second[i]);
                    numeric->faces.push_back(face);
                });
        });
    // Remote physical adapters are keyed by exact current incidence proof
    // control and slot. Unchanged tickets retain these actual records.
    geometry_detail::NativeConstructorVector<const NativeNumberingRemotePiece*> current_remote(
        a.allocator<const NativeNumberingRemotePiece*>());
    collective(*input.comm,
        [&]
        {
            for (std::size_t g = 0; g < full.size(); ++g)
            {
                const auto& reply = remote_incidence[g];
                const auto tiles = reply.proof->remote_tiles();
                if (tiles.size() != reply.numeric.size())
                    throw std::invalid_argument("Remote rich/numeric piece extents differ.");
                for (std::size_t slot = 0; slot < tiles.size(); ++slot)
                {
                    const auto& tile = tiles[slot];
                    const auto& route = reply.numeric[slot];
                    auto current_gid = [&](const AdaptiveCellKey& key) -> ID
                    {
                        if (const auto source = find_source(sources, key))
                            return source->gid;
                        const auto at = std::lower_bound(outer.begin(), outer.end(), key,
                            [](const auto& row, const auto& key) { return row.source->source().key < key; });
                        if (at == outer.end() || at->source->source().key != key)
                            throw std::invalid_argument("Remote incidence has no exact current outer membership.");
                        return at->gid;
                    };
                    if (route.first_gid != current_gid(tile.first) ||
                        (tile.second ? route.second_gid != current_gid(*tile.second)
                                     : route.second_gid != MultiRegionMesh::invalid_cell_id()))
                        throw std::invalid_argument(
                            "Remote incidence scalar cell route differs from current owner membership.");
                    const auto at = NativePublicationNumbering::RemotePages::lower_bound(numbering->remote_root,
                        [&](const auto& row)
                        {
                            if (row.record->ticket != reply.proof->ticket())
                                return row.record->ticket < reply.proof->ticket();
                            return row.record->slot < slot;
                        });
                    std::shared_ptr<const NativeNumberingRemotePiece> retained;
                    if (at && at->record->ticket == reply.proof->ticket() && at->record->slot == slot)
                        retained = at->record;
                    else
                    {
                        auto made =
                            geometry_detail::NativeAllocationEnvelopeAccess::make<NativeNumberingRemotePiece>(a, a);
                        auto& record = *made.owner;
                        record.ticket = reply.proof->ticket();
                        record.source_ticket = full[g].source->ticket();
                        record.physical_record = reply.proof->remote_record();
                        record.fragment = full[g].source->fragment();
                        record.slot = slot;
                        record.allocation = made.envelope;
                        record.native_is_second = tile.first != full[g].source->source().key;
                        if (record.native_is_second && (!tile.second || *tile.second != full[g].source->source().key))
                            throw std::invalid_argument("Remote native piece does not name its sealed source.");
                        record.native_face = reply.proof->remote_native_face(slot);
                        record.physical.second_image = tile.second_image_identity;
                        record.physical.override_geometry = tile.geometry;
                        record.physical.cut_incidence = reply.proof->remote_cut_incidence(slot);
                        record.sign = reply.proof->remote_native_sign(slot);
                        const auto points = reply.proof->remote_points(slot);
                        record.native_nodes.reserve(points.size());
                        const auto& fragment = record.fragment->region();
                        std::visit(
                            [&](const auto& r)
                            {
                                const auto native = r.topology().face_nodes(record.native_face);
                                for (const auto& identity : points)
                                {
                                    auto physical = identity;
                                    if (record.native_is_second)
                                        physical = PLIC::point_translated<ArithmeticMode::Ordinary>(
                                            physical, PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                                          {}, tile.second_image_identity));
                                    const auto display = PLIC::point_display<ArithmeticMode::Ordinary>(physical);
                                    ID matched = UINT64_MAX;
                                    for (const auto node : native)
                                    {
                                        const auto p = r.geometry().node_coordinates(node);
                                        if (p.x == display[0] && p.y == display[1] && p.z == display[2])
                                        {
                                            if (matched != UINT64_MAX)
                                                throw std::invalid_argument("Ambiguous remote native point alias.");
                                            matched = node;
                                        }
                                    }
                                    record.native_nodes.push_back(matched);
                                }
                            },
                            fragment);
                        retained = std::move(made.owner);
                        NativeNumberingRemoteRow row{retained};
                        numbering->remote_root = remote_editor.edit(numbering->remote_root, row, &row, false);
                    }
                    current_remote.push_back(retained.get());
                    const auto first_points = reply.proof->remote_points(slot);
                    if (first_points.size() != route.first_nodes.size() ||
                        (!route.second_nodes.empty() && route.second_nodes.size() != first_points.size()))
                        throw std::invalid_argument("Remote numeric node loop differs from exact incidence facts.");
                    for (std::size_t i = 0; i < first_points.size(); ++i)
                    {
                        add_node(route.first_nodes[i], first_points[i]);
                        if (!route.second_nodes.empty())
                            add_node(
                                route.second_nodes[i], PLIC::point_translated<ArithmeticMode::Ordinary>(first_points[i],
                                                           PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                                               {}, tile.second_image_identity)));
                    }
                    NativeSparseFaceRoute face;
                    face.gid = route.face_gid;
                    face.first = route.first_gid;
                    face.second = route.second_gid;
                    face.physical = &retained->physical;
                    face.native_is_second = retained->native_is_second;
                    face.native = {region_ordinal(*fragments, retained->fragment.get()), retained->native_face};
                    face.boundary = boundary(tile.boundary);
                    if (retained->physical.override_geometry)
                    {
                        face.loop_begin = loop(route.first_nodes);
                        face.loop_size = route.first_nodes.size();
                    }
                    if (!tile.second_image_identity.is_zero())
                    {
                        face.second_loop_begin = loop(route.second_nodes);
                        face.second_loop_size = route.second_nodes.size();
                    }
                    logical(face.native, face.gid, retained->sign);
                    const auto& physical_nodes = retained->native_is_second ? route.second_nodes : route.first_nodes;
                    for (std::size_t i = 0; i < retained->native_nodes.size(); ++i)
                        add_alias(*retained->fragment, retained->native_nodes[i], physical_nodes.at(i));
                    numeric->faces.push_back(face);
                }
            }
            std::sort(current_remote.begin(), current_remote.end(), std::less<const NativeNumberingRemotePiece*>{});
            const auto old = numbering->remote_root;
            visit_rows<NativePublicationNumbering::RemotePages>(old,
                [&](const auto& row)
                {
                    if (!std::binary_search(current_remote.begin(), current_remote.end(), row.record.get(),
                            std::less<const NativeNumberingRemotePiece*>{}))
                        numbering->remote_root = remote_editor.edit(numbering->remote_root, row, nullptr, true);
                });
            // Merge duplicate subscriber routes after checking every scalar fact.
            // These buffers contain no typed keys or physical geometry owners.
            std::sort(numeric->nodes.begin(), numeric->nodes.end(),
                [](const auto& a, const auto& b) { return a.gid < b.gid; });
            for (std::size_t i = 1; i < numeric->nodes.size(); ++i)
                if (numeric->nodes[i - 1].gid == numeric->nodes[i].gid &&
                    numeric->nodes[i - 1].introduced_coordinate != numeric->nodes[i].introduced_coordinate)
                    throw std::invalid_argument("Canonical node subscribers disagree on exact display facts.");
            numeric->nodes.erase(std::unique(numeric->nodes.begin(), numeric->nodes.end(),
                                     [](const auto& a, const auto& b) { return a.gid == b.gid; }),
                numeric->nodes.end());
            std::sort(numeric->native_nodes.begin(), numeric->native_nodes.end());
            for (std::size_t i = 1; i < numeric->native_nodes.size(); ++i)
                if (numeric->native_nodes[i - 1].first == numeric->native_nodes[i].first &&
                    numeric->native_nodes[i - 1].second != numeric->native_nodes[i].second)
                    throw std::invalid_argument("Native node alias crosses canonical identities.");
            numeric->native_nodes.erase(
                std::unique(numeric->native_nodes.begin(), numeric->native_nodes.end()), numeric->native_nodes.end());
            for (const auto& alias : numeric->native_nodes)
            {
                const auto at = std::lower_bound(numeric->nodes.begin(), numeric->nodes.end(), alias.second,
                    [](const auto& row, ID gid) { return row.gid < gid; });
                if (at == numeric->nodes.end() || at->gid != alias.second)
                    throw std::invalid_argument("Native alias lacks current canonical node.");
                if (!at->native)
                {
                    at->native = alias.first;
                    at->introduced_coordinate = {};
                }
            }
            std::sort(numeric->faces.begin(), numeric->faces.end(),
                [](const auto& a, const auto& b) { return a.gid < b.gid; });
            auto same_loop = [&](std::size_t a, std::size_t an, std::size_t b, std::size_t bn)
            {
                return an == bn && std::equal(numeric->loops.begin() + a, numeric->loops.begin() + a + an,
                                       numeric->loops.begin() + b);
            };
            std::size_t merged = 0;
            for (std::size_t i = 0; i < numeric->faces.size();)
            {
                const auto& first = numeric->faces[i];
                auto end = i + 1;
                while (end < numeric->faces.size() && numeric->faces[end].gid == first.gid)
                    ++end;
                std::array<std::optional<RegionFace>, 2> native;
                const auto retain_address = [&](bool second, RegionFace address)
                {
                    auto& exact = native[second];
                    if (exact && *exact != address)
                        throw std::invalid_argument("Native face subscribers disagree on a canonical side address.");
                    exact = address;
                };
                const auto* selected = &first;
                const NativeSparseFaceRoute* explicit_loop = nullptr;
                for (auto j = i; j < end; ++j)
                {
                    const auto& row = numeric->faces[j];
                    if (first.first != row.first || first.second != row.second || first.boundary != row.boundary ||
                        first.physical->second_image != row.physical->second_image ||
                        !same_loop(first.second_loop_begin, first.second_loop_size, row.second_loop_begin,
                            row.second_loop_size))
                        throw std::invalid_argument("Native face subscribers disagree on current routes.");
                    if (row.loop_size)
                    {
                        if (explicit_loop && !same_loop(explicit_loop->loop_begin, explicit_loop->loop_size,
                                                 row.loop_begin, row.loop_size))
                            throw std::invalid_argument("Native face subscribers disagree on the ordered first loop.");
                        explicit_loop = &row;
                    }
                    retain_address(row.native_is_second, row.native);
                    if (row.second_native)
                        retain_address(!row.native_is_second, *row.second_native);
                    // Prefer the canonical first source. For that same source,
                    // keep its intrinsic whole-face representation when present.
                    if ((selected->native_is_second && !row.native_is_second) ||
                        (selected->native_is_second == row.native_is_second &&
                            selected->physical->override_geometry && !row.physical->override_geometry))
                        selected = &row;
                }
                auto face = *selected;
                face.native_is_second = !native[0];
                face.native = native[face.native_is_second].value();
                face.second_native = native[!face.native_is_second];
                numeric->faces[merged++] = face;
                i = end;
            }
            numeric->faces.resize(merged);
            std::sort(numeric->flux_signs.begin(), numeric->flux_signs.end(),
                [](const auto& a, const auto& b) { return std::pair{a.native, a.gid} < std::pair{b.native, b.gid}; });
            for (std::size_t i = 1; i < numeric->flux_signs.size(); ++i)
                if (numeric->flux_signs[i - 1].native == numeric->flux_signs[i].native &&
                    numeric->flux_signs[i - 1].gid == numeric->flux_signs[i].gid &&
                    numeric->flux_signs[i - 1].sign != numeric->flux_signs[i].sign)
                    throw std::invalid_argument("Native owner-oriented sign subscribers disagree.");
            numeric->flux_signs.erase(
                std::unique(numeric->flux_signs.begin(), numeric->flux_signs.end(),
                    [](const auto& a, const auto& b) { return a.native == b.native && a.gid == b.gid; }),
                numeric->flux_signs.end());
            for (std::size_t i = 0; i < numeric->flux_signs.size();)
            {
                auto end = i + 1;
                while (end < numeric->flux_signs.size() &&
                       numeric->flux_signs[end].native == numeric->flux_signs[i].native)
                    ++end;
                const auto begin = numeric->loops.size();
                for (auto j = i; j < end; ++j)
                    numeric->loops.push_back(numeric->flux_signs[j].gid);
                numeric->logical_faces.push_back({numeric->flux_signs[i].native, begin, end - i});
                i = end;
            }
            numbering->fragments = fragments;
            numeric->physical_owners = numbering;
        });
    std::shared_ptr<MultiRegionMesh> geometry;
    collective(*input.comm,
        [&]
        {
            geometry = NativeSparseAccess::make(
                session, numbering, fragments, numeric, seals, numbering->global_faces, numbering->global_nodes);
        });
    geometry_detail::NativeConstructorVector<ID> owned(a.allocator<ID>()), ghosts(a.allocator<ID>());
    collective(*input.comm,
        [&]
        {
            owned.reserve(input.layout->size());
            ghosts.reserve(full.size());
            for (std::size_t i = 0; i < input.layout->size(); ++i)
                owned.push_back(input.cell_begin + i);
            for (const auto& row : full)
                ghosts.push_back(row.gid);
        });
    const auto partition = geometry_detail::NativePartitionAccess::make(session, geometry, owned, ghosts);
    const auto handle = geometry_detail::NativeMeshHandleAccess::make(partition, a);
    std::shared_ptr<NativePublicationViewData> data;
    collective(*input.comm,
        [&]
        {
            const PipelinePhaseScope phase(PipelinePhase::NativePublication);
            auto view = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationViewData>(a);
            data = std::move(view.owner);
            data->allocation = std::move(view.envelope);
            auto payload = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationPayload>(a);
            data->payload = std::move(payload.owner);
            data->payload->allocation = std::move(payload.envelope);
            data->payload->numbering = numbering;
            data->payload->numeric = numeric;
            data->payload->fragments = fragments;
            data->payload->owned = session.indexes()->owned_keys;
            data->payload->remote = session.indexes()->remote;
            data->payload->cell_begin = input.cell_begin;
            data->payload->partition = partition.partition;
            data->payload->mesh = handle;
            auto storage = geometry_detail::native_constructor_storage(a);
            storage.owner->reserve(data->payload->keys, numeric->cells.size());
            for (const auto& cell : numeric->cells)
            {
                if (cell.gid >= input.cell_begin && cell.gid - input.cell_begin < input.layout->size())
                    data->payload->keys.push_back({cell.gid, static_cast<std::size_t>(cell.gid - input.cell_begin), false});
                else
                {
                    const SourceRoute* source = nullptr;
                    for (const auto& row : sources)
                        if (row.gid == cell.gid)
                        {
                            source = &row;
                            break;
                        }
                    const AdaptiveCellKey* key = source ? &source->source->source().key : nullptr;
                    if (!key)
                        for (const auto& row : outer)
                            if (row.gid == cell.gid)
                            {
                                key = &row.source->source().key;
                                break;
                            }
                    if (!key)
                        throw std::invalid_argument("Numeric cell has no current borrowed key source.");
                    const auto ordinal = data->payload->remote.ordinal_of(*key);
                    if (!ordinal)
                        throw std::invalid_argument("Current remote key catalog lacks numeric route.");
                    data->payload->keys.push_back({cell.gid, *ordinal, true});
                }
            }
            // The output graph supplies exact unique physical categories during
            // first/last edits. Numeric routes are separately measured capacities.
            data->payload->summary.scalar_numeric = numeric->storage_bytes();
            // Keep the admitted concrete key-route buffer's receipt alive with the
            // current view. It is never a controller or prior publication capture.
            data->payload->route_storage = std::move(storage);
        });
    return data;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
