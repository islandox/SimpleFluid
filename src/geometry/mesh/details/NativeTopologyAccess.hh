#pragma once
#include "geometry/mesh/SemiStructMeshTopo.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "geometry/VoFWireProtocol.hh"
#include <unordered_set>
#include <limits>

namespace SimpleFluid::Meshes::details
{
class NativeTopologyAccess
{
public:
    static SemiStructMeshTopo make(unsigned nodes, const Arr<Arr<unsigned>>& cells, unsigned layers,
        const Arr<SemiStructMeshTopo::BoundaryEdge>& boundaries,
        const geometry_detail::NativeConstructionAdmission& admission, bool periodic = false)
    {
        geometry_detail::NativeConstructorStorage::require_profile();
        if (!nodes || cells.empty() || !layers)
            throw std::invalid_argument("Semi-structured topology requires nonempty nodes, cells and layers.");
        if (cells.size() > std::numeric_limits<unsigned>::max())
            throw std::overflow_error("Semi-structured base-cell count exceeds its ID type.");
        auto storage = geometry_detail::native_constructor_storage(admission);
        auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<SemiStructMeshTopo::BaseData>(
            admission, std::move(storage));
        SemiStructMeshTopo result(SemiStructMeshTopo::AdmittedStorageTag{}, std::move(tracked.owner));
        result.d_base->allocation_envelope = std::move(tracked.envelope);
        result.build_base_topology(nodes, cells, boundaries);
        if (result.d_base->side_faces.size() > std::numeric_limits<unsigned>::max())
            throw std::overflow_error("Semi-structured side-face count exceeds its ID type.");
        result.d_indexer = SemiStructuredIndexer(static_cast<unsigned>(cells.size()),
            static_cast<unsigned>(result.d_base->side_faces.size()), nodes, layers, periodic);
        result.initialize_face_adjacency();
        result.initialize_cell_adjacency();
        result.initialize_boundary_batches();
        return result;
    }
    /** Rebuild only immutable connectivity from original integer ordinals,
     * preserving first-edge ownership and boundary IDs. No metric arithmetic. */
    static SemiStructMeshTopo detach(const SemiStructMeshTopo& source,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        if (source.d_base->allocation_envelope)
        {
            const auto& envelope = source.d_base->allocation_envelope;
            if (!envelope->authentic_owner(envelope) || !envelope->matches(source.d_base))
                throw std::invalid_argument("Foreign immutable base allocation provenance.");
            return source;
        }
        if (source.d_base->storage.owner)
            throw std::invalid_argument("Incomplete immutable base allocation provenance.");
        auto scratch = geometry_detail::native_constructor_storage(admission);
        Arr<Arr<unsigned>> loops;
        scratch.owner->reserve(loops, source.indexer().num_cells_per_layer);
        loops.resize(source.indexer().num_cells_per_layer);
        for (unsigned c = 0; c < loops.size(); ++c)
        {
            scratch.owner->reserve(loops[c], source.cell_side_faces(c).size());
            for (const auto f : source.cell_side_faces(c))
            {
                const auto& edge = source.side_face(f);
                loops[c].push_back(edge.nodes[edge.owner == c ? 0 : 1]);
            }
        }
        Arr<SemiStructMeshTopo::BoundaryEdge> boundaries;
        size_t count = 0;
        for (const auto& edge : source.side_faces()) count += edge.boundary_id >= 2;
        scratch.owner->reserve(boundaries, count);
        for (const auto& edge : source.side_faces())
            if (edge.boundary_id >= 2)
            {
                SemiStructMeshTopo::BoundaryEdge boundary;
                boundary.batch_name.clear();
                boundary.node0 = edge.nodes[0]; boundary.node1 = edge.nodes[1];
                scratch.owner->string(boundary.batch_name, source.boundary_batch_name(edge.boundary_id));
                boundaries.push_back(std::move(boundary));
            }
        return make(source.indexer().num_nodes_per_layer, loops, source.indexer().num_layers, boundaries,
            admission, source.indexer().axial_periodic);
    }
    using AliasView = std::pair<std::string_view, std::string_view>;
    static std::shared_ptr<const ExtrudedTopology> topology(SemiStructMeshTopo base,
        std::span<const AliasView> aliases, const geometry_detail::NativeConstructionAdmission& admission)
    {
        if (base.indexer().axial_periodic || !base.indexer().num_layers || !base.indexer().num_cells_per_layer)
            throw std::invalid_argument("Extrusion templates require nonempty nonperiodic axial layers.");
        auto storage = geometry_detail::native_constructor_storage(admission);
        Arr<std::pair<int, std::string>> resolved;
        storage.owner->reserve(resolved, aliases.size());
        using Seen = std::unordered_set<std::string_view, std::hash<std::string_view>, std::equal_to<std::string_view>,
            geometry_detail::NativeStorageAllocator<std::string_view>>;
        Seen names(0, std::hash<std::string_view>{}, std::equal_to<std::string_view>{},
            geometry_detail::NativeStorageAllocator<std::string_view>(storage.owner));
        size_t matched = 0;
        // Use the retained map's iteration order, just like the legacy constructor.
        for (const auto& [id, original] : base.d_base->boundary_names)
        {
            const AliasView* selected = nullptr;
            for (const auto& alias : aliases)
                if (alias.first == original)
                {
                    if (selected) throw std::invalid_argument("Repeated source boundary alias.");
                    selected = &alias;
                }
            const std::string_view name = selected ? selected->second : std::string_view(original);
            if (name.empty() || !names.insert(name).second)
                throw std::invalid_argument("Extruded boundary aliases require unique nonempty final names.");
            if (selected)
            {
                ++matched;
                std::string owned;
                storage.owner->string(owned, name);
                resolved.emplace_back(id, std::move(owned));
            }
        }
        if (matched != aliases.size())
            throw std::invalid_argument("Extruded boundary alias refers to an unknown source name.");
        auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<ExtrudedTopology>(
            admission, ExtrudedTopology::AdmittedStorageTag{}, std::move(base), std::move(storage), std::move(resolved));
        tracked.owner->d_allocation_envelope = std::move(tracked.envelope);
        return std::move(tracked.owner);
    }
    static std::shared_ptr<const ExtrudedTopology> with_layers(const ExtrudedTopology& source, unsigned layers,
        std::span<const AliasView> aliases, const geometry_detail::NativeConstructionAdmission& admission)
    {
        auto scratch = geometry_detail::native_constructor_storage(admission);
        std::vector<AliasView> combined;
        // At most one retained alias for each boundary. Avoid capacity growth.
        scratch.owner->reserve(combined, source.d_topology.d_base->boundary_names.size());
        for (const auto& [id, original] : source.d_topology.d_base->boundary_names)
        {
            const std::string_view current = source.boundary_batch_name(id);
            std::string_view selected = current;
            bool overridden = false;
            for (const auto& alias : aliases)
                if (alias.first == original)
                {
                    if (overridden) throw std::invalid_argument("Repeated source boundary alias.");
                    overridden = true; selected = alias.second;
                }
            if (overridden || selected != original) combined.emplace_back(original, selected);
        }
        for (const auto& alias : aliases)
        {
            bool found = false;
            for (const auto& [id, name] : source.d_topology.d_base->boundary_names) found |= name == alias.first;
            if (!found) throw std::invalid_argument("Extruded boundary alias refers to an unknown source name.");
        }
        return topology(detach(source.d_topology, admission).with_layers(layers), combined, admission);
    }
    static std::shared_ptr<const ExtrudedGeometry> geometry(std::shared_ptr<const ExtrudedTopology> topology,
        std::span<const MeshUtils::Vec3> nodes, std::span<const real_t> z,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        auto storage = geometry_detail::native_constructor_storage(admission);
        Arr<MeshUtils::Vec3> owned_nodes;
        ArrReal owned_z;
        storage.owner->copy(owned_nodes, nodes);
        storage.owner->copy(owned_z, z);
        auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<ExtrudedGeometry>(admission,
            ExtrudedGeometry::AdmittedStorageTag{}, std::move(topology), std::move(storage),
            std::move(owned_nodes), std::move(owned_z));
        tracked.owner->d_allocation_envelope = std::move(tracked.envelope);
        return std::move(tracked.owner);
    }
    static std::shared_ptr<const ExtrudedGeometry> geometry(std::shared_ptr<const ExtrudedTopology> topology,
        std::shared_ptr<const PLIC::ExtrudedConstruction<ArithmeticMode::Ordinary>> construction,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        auto storage = geometry_detail::native_constructor_storage(admission);
        auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<ExtrudedGeometry>(admission,
            ExtrudedGeometry::AdmittedStorageTag{}, std::move(topology), std::move(storage), std::move(construction));
        tracked.owner->d_allocation_envelope = std::move(tracked.envelope);
        return std::move(tracked.owner);
    }
    static std::shared_ptr<const ExtrudedTopology> decode_base(std::span<const char> packet, unsigned layers,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        struct Reader
        {
            std::span<const char> bytes;
            size_t cursor = 0;
            uint8_t byte()
            {
                if (cursor == bytes.size()) throw std::invalid_argument("Truncated shared topology packet.");
                return static_cast<uint8_t>(bytes[cursor++]);
            }
            uint64_t word()
            {
                if (bytes.size() - cursor < 8) throw std::invalid_argument("Truncated shared topology packet.");
                uint64_t result = 0;
                for (unsigned i = 0; i < 8; ++i) result |= uint64_t(byte()) << (8 * i);
                return result;
            }
            size_t count()
            {
                const auto value = word();
                if (value > 65536 || value > bytes.size() - cursor)
                    throw std::invalid_argument("Excessive shared topology packet count.");
                return static_cast<size_t>(value);
            }
        } reader{packet};
        packet_codec::read_header(reader, packet_codec::Family::ExtrudedTopologyBase, uint8_t{2},
            packet_codec::Record::Value);
        const auto nodes = reader.word(), cells = reader.count();
        if (nodes > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Shared base node extent overflow.");
        auto scratch = geometry_detail::native_constructor_storage(admission);
        Arr<Arr<unsigned>> loops;
        scratch.owner->reserve(loops, cells); loops.resize(cells);
        for (auto& loop : loops)
        {
            const auto count = reader.count();
            scratch.owner->reserve(loop, count);
            for (size_t i = 0; i < count; ++i)
            {
                const auto id = reader.word();
                if (id >= nodes) throw std::invalid_argument("Invalid shared base node ordinal.");
                loop.push_back(static_cast<unsigned>(id));
            }
        }
        Arr<SemiStructMeshTopo::BoundaryEdge> boundaries;
        const auto count = reader.count();
        scratch.owner->reserve(boundaries, count);
        for (size_t i = 0; i < count; ++i)
        {
            const auto a = reader.word(), b = reader.word(), length = reader.word();
            if (a >= nodes || b >= nodes || length > packet.size() - reader.cursor)
                throw std::invalid_argument("Invalid shared base boundary record.");
            SemiStructMeshTopo::BoundaryEdge edge;
            edge.node0 = static_cast<unsigned>(a); edge.node1 = static_cast<unsigned>(b);
            edge.batch_name.clear();
            scratch.owner->string(edge.batch_name, {packet.data() + reader.cursor, static_cast<size_t>(length)});
            reader.cursor += length;
            boundaries.push_back(std::move(edge));
        }
        if (reader.cursor != packet.size()) throw std::invalid_argument("Trailing shared topology packet bytes.");
        return topology(make(static_cast<unsigned>(nodes), loops, layers, boundaries, admission), {}, admission);
    }

    /** Canonical comparison writes directly into an existing descriptor span;
     * no encode vector, boundary list, label copy, or hash allocation. */
    static bool matches_base(const ExtrudedTopology& topology, std::span<const char> packet) noexcept
    {
        struct Compare
        {
            std::span<const char> bytes;
            size_t cursor = 0;
            bool equal = true;
            void byte(uint8_t value) noexcept
            {
                if (cursor >= bytes.size() || static_cast<uint8_t>(bytes[cursor]) != value) equal = false;
                ++cursor;
            }
            void word(uint64_t value) noexcept
            { for (unsigned i = 0; i < 8; ++i) byte(static_cast<uint8_t>(value >> (8 * i))); }
        } writer{packet};
        emit_base(topology, writer);
        return writer.equal && writer.cursor == packet.size();
    }
    template<class Writer> static void emit_base(const ExtrudedTopology& topology, Writer& writer)
    {
        packet_codec::write_header(writer, packet_codec::Family::ExtrudedTopologyBase, uint8_t{2},
            packet_codec::Record::Value);
        writer.word(topology.indexer().num_nodes_per_layer);
        writer.word(topology.indexer().num_cells_per_layer);
        for (size_t c = 0; c < topology.indexer().num_cells_per_layer; ++c)
        {
            const auto nodes = topology.base_cell_nodes(c);
            writer.word(nodes.size()); for (const auto n : nodes) writer.word(n);
        }
        size_t count = 0;
        for (const auto& edge : topology.d_topology.side_faces())
            count += edge.neighbor == SemiStructMeshTopo::invalid_ordinal;
        writer.word(count);
        for (const auto& edge : topology.d_topology.side_faces())
            if (edge.neighbor == SemiStructMeshTopo::invalid_ordinal)
            {
                writer.word(edge.nodes[0]); writer.word(edge.nodes[1]);
                const auto& label = topology.d_topology.boundary_batch_name(edge.boundary_id);
                writer.word(label.size()); for (const auto c : label) writer.byte(static_cast<uint8_t>(c));
            }
    }
    static std::vector<char> encode_base(const ExtrudedTopology& topology,
        const std::shared_ptr<geometry_detail::NativeConstructorStorage>& storage)
    {
        struct Count
        {
            size_t size = 0;
            void byte(uint8_t) { if (size == std::numeric_limits<size_t>::max()) throw std::bad_alloc(); ++size; }
            void word(uint64_t value) { for (unsigned i = 0; i < 8; ++i) byte(static_cast<uint8_t>(value >> (8 * i))); }
        } count;
        emit_base(topology, count);
        std::vector<char> bytes;
        storage->reserve(bytes, count.size);
        struct Writer
        {
            std::vector<char>& bytes;
            void byte(uint8_t value) { bytes.push_back(static_cast<char>(value)); }
            void word(uint64_t value) { for (unsigned i = 0; i < 8; ++i) byte(static_cast<uint8_t>(value >> (8 * i))); }
        } writer{bytes};
        emit_base(topology, writer);
        return bytes;
    }
    template<class Writer> static void emit_construction_topology(const ExtrudedTopology& topology, Writer& writer)
    {
        const auto& index = topology.indexer();
        writer.word(index.num_nodes_per_layer); writer.word(index.num_layers); writer.word(index.num_cells_per_layer);
        for (size_t c = 0; c < index.num_cells_per_layer; ++c)
        {
            const auto nodes = topology.base_cell_nodes(c); writer.word(nodes.size());
            for (const auto node : nodes) writer.word(node);
        }
        const auto& batches = topology.d_topology.d_base->boundary_base_faces;
        writer.word(batches.size());
        for (const auto& [id, batch] : batches)
        {
            const auto& name = topology.boundary_batch_name(id); writer.word(name.size());
            for (const unsigned char ch : name) writer.word(ch);
        }
        for (const auto& edge : topology.d_topology.side_faces())
        {
            writer.word(edge.nodes[0]); writer.word(edge.nodes[1]); writer.word(edge.owner);
            writer.word(edge.neighbor); writer.word(edge.boundary_id);
        }
    }

};
} // namespace SimpleFluid::Meshes::details
