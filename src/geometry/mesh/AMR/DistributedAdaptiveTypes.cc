/** @file DistributedAdaptiveTypes.cc
 * @brief Versioned bounded lossless ancestry and owner-record encoding.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/NativeAdaptiveLeafCodec.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/PLIC/details/NativeGeometryByteWriter.hh"
#include "geometry/PLIC/details/GeometryPacketHeader.hh"
#include "geometry/mesh/NativeDescriptorCodecAccess.hh"

#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/VoFWireProtocol.hh"
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/mesh/AMR/AdaptivePipelineAudit.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNativeRetention.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
struct Writer
{
    AdaptiveBytes data;
    void byte(uint8_t value) { data.push_back(static_cast<char>(value)); }
    void word(std::uint64_t n)
    {
        for (unsigned i = 0; i < 8; ++i)
            data.push_back(char(n >> (8 * i)));
    }
    void bytes(std::span<const char> value)
    {
        word(value.size());
        data.insert(data.end(), value.begin(), value.end());
    }
    void path(const std::vector<std::uint64_t>& path)
    {
        word(path.size());
        for (auto n : path)
            word(n);
    }
};
struct SizeWriter
{
    std::size_t size = 0;
    void add(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() - size)
            throw std::invalid_argument("Adaptive encoded length overflows.");
        size += n;
    }
    void byte(uint8_t) { add(1); }
    void word(std::uint64_t) { add(8); }
    void bytes(std::span<const char> value)
    {
        word(value.size());
        add(value.size());
    }
    void sized_bytes(std::size_t n)
    {
        word(n);
        add(n);
    }
    void path(const std::vector<std::uint64_t>& path)
    {
        word(path.size());
        for (auto n : path)
            word(n);
    }
};
struct AdaptiveRecordReader
{
    std::span<const char> data;
    std::size_t position = 0;
    const geometry_detail::NativeConstructorStorageOwner* native_storage = nullptr;
    uint8_t byte()
    {
        if (position == data.size())
            throw std::invalid_argument("Truncated adaptive tag.");
        return static_cast<uint8_t>(data[position++]);
    }
    std::uint64_t word()
    {
        if (data.size() - position < 8)
            throw std::invalid_argument("Truncated adaptive packet.");
        std::uint64_t result = 0;
        for (unsigned i = 0; i < 8; ++i)
            result |= std::uint64_t(static_cast<unsigned char>(data[position++])) << (8 * i);
        return result;
    }
    std::span<const char> bytes_view()
    {
        const auto count = word();
        if (count > data.size() - position)
            throw std::invalid_argument("Adaptive packet length exceeds payload.");
        const auto result = data.subspan(position, static_cast<std::size_t>(count));
        position += static_cast<std::size_t>(count);
        return result;
    }
    AdaptiveBytes bytes()
    {
        auto n = word();
        if (n > data.size() - position)
            throw std::invalid_argument("Adaptive packet length exceeds payload.");
        AdaptiveBytes value(data.begin() + position, data.begin() + position + n);
        position += n;
        return value;
    }
    std::vector<std::uint64_t> path()
    {
        auto n = word();
        if (n > (data.size() - position) / 8)
            throw std::invalid_argument("Invalid adaptive ancestry depth.");
        std::vector<std::uint64_t> result;
        if (native_storage)
            native_storage->owner->reserve(result, static_cast<std::size_t>(n));
        else
            result.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i)
            result.push_back(word());
        return result;
    }
    void done()
    {
        if (position != data.size())
            throw std::invalid_argument("Trailing adaptive packet bytes.");
    }
};
template<class PacketWriter> void base(PacketWriter& w, const BaseCellKey& key)
{
    w.word(key.root);
    w.word(key.root_template);
    w.path(key.path);
}
BaseCellKey base(AdaptiveRecordReader& r)
{
    BaseCellKey key;
    key.root = r.word();
    key.root_template = r.word();
    key.path = r.path();
    return key;
}
template<class PacketWriter> void axial(PacketWriter& w, const AxialIntervalKey& key)
{
    w.word(key.root);
    w.path(key.path);
}
AxialIntervalKey axial(AdaptiveRecordReader& r)
{
    AxialIntervalKey key;
    key.root = r.word();
    key.path = r.path();
    return key;
}
// Canonical ancestry keys retain their original encoding and routing hashes.
// This namespace prefix is not a decodable transport-record header.
constexpr std::uint64_t canonical_ancestry_prefix = 0x5346414450543631ULL;
template<class PacketWriter> void canonical_key_header(PacketWriter& w)
{
    w.word(canonical_ancestry_prefix);
    w.word(1);
    w.word(0);
}
void canonical_key_header(AdaptiveRecordReader& r)
{
    if (r.word() != canonical_ancestry_prefix || r.word() != 1 || r.word() != 0)
        throw std::invalid_argument("Invalid canonical adaptive key version/type.");
}
enum class RecordKind : uint8_t { Leaf = 1, Family = 2 };
constexpr uint8_t record_version = 5;
} // namespace
namespace
{
template<class PacketWriter> void emit_adaptive_key(PacketWriter& w, const AdaptiveCellKey& key)
{
    canonical_key_header(w);
    w.word(key.region);
    w.word(key.ancestry.index());
    std::visit(
        [&](const auto& value)
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, CartesianCellAncestry>)
            {
                w.word(value.level);
                for (auto n : value.index)
                    w.word(n);
            }
            else if constexpr (std::is_same_v<T, ConvexCellAncestry>)
            {
                w.word(value.root);
                w.path(value.path);
            }
            else if constexpr (std::is_same_v<T, ProductCellAncestry>)
            {
                base(w, value.base);
                axial(w, value.axial);
            }
            else
            {
                w.word(value.provider_root);
                w.word(value.native_cell);
            }
        },
        key.ancestry);
}
} // namespace
namespace
{
// No-throw counting writer. The shared emit traversal (not a duplicate
// encoding) decides every field, so the counted length cannot drift.
struct KeySizeWriter
{
    std::size_t size = 0;
    void byte(uint8_t) noexcept { ++size; }
    void word(std::uint64_t) noexcept { size += 8; }
    void path(const std::vector<std::uint64_t>& path) noexcept { size += 8 + 8 * path.size(); }
};
} // namespace
std::size_t encoded_adaptive_key_size(const AdaptiveCellKey& key) noexcept
{
    KeySizeWriter w;
    emit_adaptive_key(w, key);
    return w.size;
}
AdaptiveBytes encode_adaptive_key(const AdaptiveCellKey& key)
{
    Writer w;
    const auto size = encoded_adaptive_key_size(key);
    w.data.reserve(size);
    // The caller admitted exactly `size` before this allocation; require the
    // materialized capacity to match the admitted amount.
    if (w.data.capacity() != size)
        throw std::logic_error("Adaptive key capacity differs from its admitted size.");
    emit_adaptive_key(w, key);
    if (w.data.size() != size)
        throw std::logic_error("Adaptive key encoded length differs from its counted length.");
    return std::move(w.data);
}

namespace
{
AdaptiveCellKey read_adaptive_key(AdaptiveRecordReader& r)
{
    canonical_key_header(r);
    AdaptiveCellKey key;
    key.region = r.word();
    const auto tag = r.word();
    if (tag == 0)
    {
        CartesianCellAncestry value;
        auto level = r.word();
        if (level > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Cartesian level exceeds its represented unsigned type.");
        value.level = level;
        for (auto& n : value.index)
            n = r.word();
        key.ancestry = value;
    }
    else if (tag == 1)
    {
        ConvexCellAncestry value;
        value.root = r.word();
        value.path = r.path();
        key.ancestry = std::move(value);
    }
    else if (tag == 2)
    {
        ProductCellAncestry value;
        value.base = base(r);
        value.axial = axial(r);
        key.ancestry = std::move(value);
    }
    else if (tag == 3)
    {
        StaticCellAncestry value;
        value.provider_root = r.word();
        value.native_cell = r.word();
        key.ancestry = value;
    }
    else
        throw std::invalid_argument("Unknown adaptive ancestry tag.");
    r.done();
    return key;
}
} // namespace
AdaptiveCellKey decode_adaptive_key(std::span<const char> packet)
{
    AdaptiveRecordReader r{packet};
    return read_adaptive_key(r);
}
void adaptive_detail::NativeAdaptiveKeyCodec::write(adaptive_detail::Writer& output, const AdaptiveCellKey& key)
{
    const auto size = encoded_adaptive_key_size(key);
    if (output.data.capacity() - output.data.size() < 8 || size > output.data.capacity() - output.data.size() - 8)
        throw std::invalid_argument("Native framed key destination was not preadmitted.");
    output.word(size);
    write_payload(output, key);
}
void adaptive_detail::NativeAdaptiveKeyCodec::write_payload(adaptive_detail::Writer& output, const AdaptiveCellKey& key)
{
    const auto size = encoded_adaptive_key_size(key);
    if (size > output.data.capacity() - output.data.size())
        throw std::invalid_argument("Native key packet destination was not preadmitted.");
    struct DirectWriter
    {
        adaptive_detail::Writer& output;
        void word(std::uint64_t value) { output.word(value); }
        void path(const std::vector<std::uint64_t>& values)
        {
            word(values.size());
            for (const auto value : values)
                word(value);
        }
    };
    DirectWriter writer{output};
    emit_adaptive_key(writer, key);
}
AdaptiveBytes adaptive_detail::NativeAdaptiveKeyCodec::encode(
    const AdaptiveCellKey& key, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!storage.owner)
        throw std::invalid_argument("Native key encoding requires its concrete storage owner.");
    adaptive_detail::Writer writer;
    storage.owner->reserve(writer.data, encoded_adaptive_key_size(key));
    write_payload(writer, key);
    return std::move(writer.data);
}
AdaptiveCellKey adaptive_detail::NativeAdaptiveKeyCodec::copy(
    const AdaptiveCellKey& key, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!storage.owner)
        throw std::invalid_argument("Native key copy requires its real storage owner.");
    AdaptiveCellKey result;
    result.region = key.region;
    std::visit(
        [&](const auto& source)
        {
            using T = std::decay_t<decltype(source)>;
            T value;
            if constexpr (std::is_same_v<T, ConvexCellAncestry>)
            {
                value.root = source.root;
                storage.owner->copy(value.path, std::span(source.path));
            }
            else if constexpr (std::is_same_v<T, ProductCellAncestry>)
            {
                value.base.root = source.base.root;
                value.base.root_template = source.base.root_template;
                value.axial.root = source.axial.root;
                storage.owner->copy(value.base.path, std::span(source.base.path));
                storage.owner->copy(value.axial.path, std::span(source.axial.path));
            }
            else
                value = source;
            result.ancestry = std::move(value);
        },
        key.ancestry);
    return result;
}
AdaptiveCellKey adaptive_detail::NativeAdaptiveKeyCodec::decode(
    std::span<const char> packet, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!storage.owner)
        throw std::invalid_argument("Native key decode requires its real storage owner.");
    AdaptiveRecordReader reader{packet, 0, &storage};
    return read_adaptive_key(reader); // Same tag/range/trailing-byte validation as the public decoder.
}
std::uint64_t adaptive_route_hash(std::span<const char> bytes) noexcept
{
    std::uint64_t result = 14695981039346656037ULL;
    for (auto c : bytes)
    {
        result ^= static_cast<unsigned char>(c);
        result *= 1099511628211ULL;
    }
    return result;
}
double AdaptiveWork::total() const
{
    if (!std::isfinite(migration) || migration < 0)
        throw std::invalid_argument("Invalid adaptive migration penalty.");
    double sum = 0;
    for (auto value : {reconstruction, interface, population})
    {
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("Invalid measured adaptive work.");
        sum += value;
    }
    if (!std::isfinite(sum) || sum <= 0)
        throw std::invalid_argument("Adaptive work must be finite and positive.");
    return sum;
}
namespace
{
struct NativeLeafSizeWriter : SizeWriter
{
    const geometry_detail::NativeConstructionAdmission& admission;
    explicit NativeLeafSizeWriter(const geometry_detail::NativeConstructionAdmission& value) : admission(value) {}
    void key(const AdaptiveCellKey& value) { sized_bytes(encoded_adaptive_key_size(value)); }
    void cell(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& value)
    {
        sized_bytes(value ? PLIC::details::NativeCellCodecAccess::size(*value, admission) : 0);
    }
    void face(const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& value, bool incidence)
    {
        sized_bytes(PLIC::details::NativePlanarCodecAccess::size(*value, incidence, admission));
    }
    void identity(const PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>& value)
    {
        sized_bytes(PLIC::encoded_identity_size(value));
    }
    void descriptor(const std::shared_ptr<const NativeFragmentDescriptor>& value)
    {
        sized_bytes(value ? encoded_native_fragment_descriptor_size(*value) : 0);
    }
};
struct NativeLeafWriter
{
    adaptive_detail::Writer& output;
    const geometry_detail::NativeConstructionAdmission& admission;
    void byte(std::uint8_t value) { output.byte(value); }
    void word(std::uint64_t value) { output.word(value); }
    void bytes(std::span<const char> value) { output.bytes(value); }
    void key(const AdaptiveCellKey& value) { adaptive_detail::NativeAdaptiveKeyCodec::write(output, value); }
    void cell(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& value)
    {
        if (!value)
        {
            word(0);
            return;
        }
        word(PLIC::details::NativeCellCodecAccess::size(*value, admission));
        PLIC::details::NativeCellCodecAccess::append(output.data, *value, admission);
    }
    void face(const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& value, bool incidence)
    {
        word(PLIC::details::NativePlanarCodecAccess::size(*value, incidence, admission));
        PLIC::details::NativePlanarCodecAccess::append(output.data, *value, incidence, admission);
    }
    void identity(const PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>& value)
    {
        const auto size = PLIC::encoded_identity_size(value);
        word(size);
        const auto header = PLIC::details::geometry_packet_header(PLIC::GeometryPacketKind::DisplacementIdentity,
            ArithmeticMode::Ordinary, size - PLIC::geometry_packet_header_bytes);
        output.data.insert(output.data.end(), header.begin(), header.end());
        PLIC::details::NativeGeometryByteWriter destination{&output.data, &admission};
        destination.native_identity(value);
    }
    void descriptor(const std::shared_ptr<const NativeFragmentDescriptor>& value)
    {
        word(value ? encoded_native_fragment_descriptor_size(*value) : 0);
        if (value)
            NativeDescriptorCodecAccess::append(output.data, *value);
    }
};
template<class PacketWriter>
void emit_adaptive_leaf(PacketWriter& w, const AdaptiveLeaf& leaf, std::size_t scratch_budget)
{
    if (leaf.domain.empty())
        throw std::invalid_argument("Adaptive connected-domain provenance is required.");
    if (!adaptive_detail::adaptive_surface_count_valid(leaf.geometry, leaf.surfaces.size()) ||
        (!leaf.geometry && !leaf.native_recipe))
        throw std::invalid_argument("Incomplete adaptive leaf representation.");
    (void) leaf.work.total();
    packet_codec::write_header(w, packet_codec::Family::AdaptiveRecords, record_version, RecordKind::Leaf);
    if constexpr (requires { w.key(leaf.key); })
        w.key(leaf.key);
    else if constexpr (std::is_same_v<PacketWriter, Writer>)
        w.bytes(encode_adaptive_key(leaf.key));
    else
    {
        SizeWriter key;
        emit_adaptive_key(key, leaf.key);
        w.sized_bytes(key.size);
    }
    w.bytes(leaf.domain);
    w.word(leaf.material);
    w.word(leaf.revision);
    w.word(std::uint64_t(leaf.provider));
    if constexpr (requires { w.cell(leaf.geometry); })
        w.cell(leaf.geometry);
    else if constexpr (std::is_same_v<PacketWriter, Writer>)
        w.bytes(leaf.geometry ? leaf.geometry->encode_certificate() : AdaptiveBytes{});
    else
        w.sized_bytes(leaf.geometry ? leaf.geometry->encoded_certificate_size(scratch_budget) : 0);
    w.word(leaf.surfaces.size());
    for (const auto& s : leaf.surfaces)
    {
        if (s.key.empty() || !s.geometry || (s.orientation != 1 && s.orientation != -1) ||
            (!s.exterior && !s.boundary.empty()) || (s.require_native_whole_tile && !s.native_whole_geometry))
            throw std::invalid_argument("Invalid adaptive surface.");
        w.bytes(s.key);
        if constexpr (requires { w.face(s.geometry, false); })
            w.face(s.geometry, false);
        else if constexpr (std::is_same_v<PacketWriter, Writer>)
            w.bytes(s.geometry->encode());
        else
            w.sized_bytes(s.geometry->encoded_size(scratch_budget));
        w.word(s.orientation == 1);
        w.word(s.exterior);
        w.bytes(std::span(s.boundary));
        w.word(s.native_face);
        if constexpr (requires { w.identity(s.authority_image); })
            w.identity(s.authority_image);
        else if constexpr (std::is_same_v<PacketWriter, Writer>)
            w.bytes(PLIC::encode_identity(s.authority_image));
        else
            w.sized_bytes(PLIC::encoded_identity_size(s.authority_image));
        w.word(s.require_native_whole_tile);
        w.word(bool(s.native_whole_geometry));
        if (s.native_whole_geometry)
        {
            if constexpr (requires { w.face(s.native_whole_geometry, true); })
                w.face(s.native_whole_geometry, true);
            else if constexpr (std::is_same_v<PacketWriter, Writer>)
                w.bytes(s.native_whole_geometry->encode_incidence());
            else
                w.sized_bytes(s.native_whole_geometry->encoded_incidence_size());
        }
    }
    if constexpr (requires { w.descriptor(leaf.native_recipe); })
        w.descriptor(leaf.native_recipe);
    else if constexpr (std::is_same_v<PacketWriter, Writer>)
        w.bytes(leaf.native_recipe ? encode_native_fragment_descriptor(*leaf.native_recipe) : AdaptiveBytes{});
    else
        w.sized_bytes(leaf.native_recipe ? encoded_native_fragment_descriptor_size(*leaf.native_recipe) : 0);
    w.word(leaf.native_cell);
    for (const auto& part :
        leaf.factor_recipe ? std::array{leaf.factor_recipe->root, leaf.factor_recipe->base, leaf.factor_recipe->axis}
                           : std::array<std::shared_ptr<const AdaptiveBytes>, 3>{})
        if (part)
            w.bytes(*part);
        else
            w.bytes(std::span<const char>{});
    if (leaf.join_recipe)
        w.bytes(*leaf.join_recipe);
    else
        w.bytes(std::span<const char>{});
    w.bytes(leaf.parent_family);
    for (double n : {leaf.work.reconstruction, leaf.work.interface, leaf.work.population, leaf.work.migration})
        w.word(std::bit_cast<std::uint64_t>(n));
}
} // namespace
std::size_t adaptive_detail::NativeAdaptiveLeafCodec::size(
    const AdaptiveLeaf& leaf, const geometry_detail::NativeConstructionAdmission& admission)
{
    NativeLeafSizeWriter writer(admission);
    emit_adaptive_leaf(writer, leaf, PLIC::geometry_packet_default_budget);
    return writer.size;
}
void adaptive_detail::NativeAdaptiveLeafCodec::write(adaptive_detail::Writer& output, const AdaptiveLeaf& leaf,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto extent = size(leaf, admission);
    if (output.data.capacity() - output.data.size() < 8 || extent > output.data.capacity() - output.data.size() - 8)
        throw std::invalid_argument("Native framed leaf destination was not preadmitted.");
    output.word(extent);
    write_payload(output, leaf, admission);
}
void adaptive_detail::NativeAdaptiveLeafCodec::write_payload(adaptive_detail::Writer& output, const AdaptiveLeaf& leaf,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    const auto extent = size(leaf, admission);
    if (extent > output.data.capacity() - output.data.size())
        throw std::invalid_argument("Native leaf destination was not preadmitted.");
    NativeLeafWriter writer{output, admission};
    const auto initial = output.data.size();
    emit_adaptive_leaf(writer, leaf, PLIC::geometry_packet_default_budget);
    if (output.data.size() - initial != extent)
        throw std::invalid_argument("Native leaf length differs from its admitted count.");
    record_pipeline_work(PipelineWork::WireEncodedLeaves);
}
adaptive_detail::NativeDecodedLeaf adaptive_detail::NativeAdaptiveLeafCodec::decode(std::span<const char> packet,
    const geometry_detail::NativeConstructionAdmission& admission, const AdaptiveLeaf* retained_parts)
{
    constexpr auto M = ArithmeticMode::Ordinary;
    using namespace adaptive_detail;
    namespace O = geometry_detail;
    NativeDecodedLeaf result{O::native_constructor_storage(admission), {}};
    auto& leaf = result.value;
    AdaptiveRecordReader reader{packet};
    packet_codec::read_header(reader, packet_codec::Family::AdaptiveRecords, record_version, RecordKind::Leaf);
    leaf.key = NativeAdaptiveKeyCodec::decode(reader.bytes_view(), result.storage);
    result.storage.owner->copy(leaf.domain, reader.bytes_view());
    leaf.material = reader.word();
    leaf.revision = reader.word();
    const auto kind = reader.word();
    if (kind > 3)
        throw std::invalid_argument("Invalid adaptive provider kind.");
    leaf.provider = AdaptiveProviderKind(kind);
    const auto geometry = reader.bytes_view();
    if (!geometry.empty())
    {
        record_pipeline_work(PipelineWork::PhysicalShellReconstruction);
        leaf.geometry =
            PLIC::details::NativeCellCodecAccess::decode(geometry, PLIC::geometry_packet_default_budget, admission);
    }
    const auto count = reader.word();
    if (!adaptive_surface_count_valid(leaf.geometry, count) || count > (packet.size() - reader.position) / 32 ||
        count > PLIC::geometry_packet_default_budget / sizeof(AdaptiveSurface))
        throw std::invalid_argument("Invalid adaptive surface count.");
    result.storage.owner->reserve(leaf.surfaces, count);
    for (std::size_t i = 0; i < count; ++i)
    {
        AdaptiveSurface surface;
        result.storage.owner->copy(surface.key, reader.bytes_view());
        surface.geometry = PLIC::details::NativePlanarOwnerAccess::decode(
            reader.bytes_view(), PLIC::geometry_packet_default_budget, admission);
        const auto orientation = reader.word(), exterior = reader.word();
        if (orientation > 1 || exterior > 1)
            throw std::invalid_argument("Invalid adaptive incidence.");
        surface.orientation = orientation ? 1 : -1;
        surface.exterior = exterior;
        const auto name = reader.bytes_view();
        result.storage.owner->string(surface.boundary, std::string_view(name.data(), name.size()));
        surface.native_face = reader.word();
        surface.authority_image = PLIC::decode_identity<PLIC::DisplacementIdentity<M>>(reader.bytes_view());
        const auto whole = reader.word(), proof = reader.word();
        if (whole > 1 || proof > 1)
            throw std::invalid_argument("Invalid native face nesting proof flags.");
        surface.require_native_whole_tile = whole;
        if (proof)
            surface.native_whole_geometry = PLIC::details::NativePlanarOwnerAccess::decode(
                reader.bytes_view(), PLIC::geometry_packet_default_budget, admission);
        if (whole && !surface.native_whole_geometry)
            throw std::invalid_argument("Missing original native face nesting proof.");
        leaf.surfaces.push_back(std::move(surface));
    }
    const auto descriptor = reader.bytes_view();
    if (!descriptor.empty())
        leaf.native_recipe = NativeDescriptorCodecAccess::decode(
            descriptor, admission, retained_parts ? retained_parts->native_recipe : nullptr);
    leaf.native_cell = reader.word();
    const auto copy_bytes =
        [&](std::span<const char> input,
            const std::shared_ptr<const AdaptiveBytes>& previous) -> std::shared_ptr<const AdaptiveBytes>
    {
        if (input.empty())
            return {};
        if (previous && previous->size() == input.size() && std::equal(input.begin(), input.end(), previous->begin()))
            return previous; // Physical immutable part reuse only; never source authority.
        auto output = O::NativeObjectOwnerAccess<AdaptiveBytes>::make(admission);
        O::NativeObjectOwnerAccess<AdaptiveBytes>::storage(output.owner).owner->copy(*output.owner, input);
        return std::move(output.owner);
    };
    std::array<std::span<const char>, 3> factors;
    for (auto& factor : factors)
        factor = reader.bytes_view();
    if (!factors[0].empty() || !factors[1].empty() || !factors[2].empty())
    {
        auto factor = O::NativeObjectOwnerAccess<AdaptiveFactorRecipe>::make_without_buffers(admission);
        const auto previous = retained_parts ? retained_parts->factor_recipe : nullptr;
        factor.owner->root = copy_bytes(factors[0], previous ? previous->root : nullptr);
        factor.owner->base = copy_bytes(factors[1], previous ? previous->base : nullptr);
        factor.owner->axis = copy_bytes(factors[2], previous ? previous->axis : nullptr);
        leaf.factor_recipe = std::move(factor.owner);
    }
    leaf.join_recipe = copy_bytes(reader.bytes_view(), retained_parts ? retained_parts->join_recipe : nullptr);
    result.storage.owner->copy(leaf.parent_family, reader.bytes_view());
    leaf.work = {std::bit_cast<double>(reader.word()), std::bit_cast<double>(reader.word()),
        std::bit_cast<double>(reader.word()), std::bit_cast<double>(reader.word())};
    reader.done();
    {
        const auto scratch = O::native_constructor_storage(admission);
        adaptive_detail::Writer canonical;
        scratch.owner->reserve(canonical.data, size(leaf, admission));
        write_payload(canonical, leaf, admission);
        if (canonical.data.size() != packet.size() ||
            !std::equal(canonical.data.begin(), canonical.data.end(), packet.begin()))
            throw std::invalid_argument("Noncanonical adaptive leaf.");
    }
    if (leaf.geometry)
    {
        O::NativeConstructorVector<SupportFace> faces(admission.allocator<SupportFace>()),
            wholes(admission.allocator<SupportFace>());
        faces.reserve(leaf.surfaces.size());
        bool has_whole = false;
        for (const auto& surface : leaf.surfaces)
            has_whole |= surface.require_native_whole_tile;
        if (has_whole)
            wholes.reserve(leaf.surfaces.size());
        for (const auto& surface : leaf.surfaces)
        {
            if (!surface.geometry || (surface.require_native_whole_tile && !surface.native_whole_geometry))
                throw std::invalid_argument("Missing adaptive surface or whole-face certificate.");
            faces.push_back(surface.geometry);
            if (has_whole)
                wholes.push_back(surface.require_native_whole_tile ? surface.native_whole_geometry : nullptr);
        }
        certify_adaptive_physical_boundary_native(
            leaf.geometry, faces, PLIC::geometry_packet_default_budget, admission, wholes);
    }
    record_pipeline_work(PipelineWork::WireDecodedLeaves);
    return result;
}
AdaptiveBytes encode_adaptive_leaf(const AdaptiveLeaf& leaf)
{
    const adaptive_detail::PipelinePhaseScope phase(adaptive_detail::PipelinePhase::Transport);
    adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireEncodedLeaves);
    Writer w;
    emit_adaptive_leaf(w, leaf, PLIC::geometry_packet_default_budget);
    return std::move(w.data);
}
std::size_t adaptive_detail::adaptive_leaf_encoding_scratch_bytes(const AdaptiveLeaf& leaf)
{
    if (!PLIC::geometry_encoding_capacity_profile_supported())
        throw std::invalid_argument("Adaptive encoding capacity policy is not qualified for this standard library.");
    std::size_t bytes = leaf.geometry ? leaf.geometry->certificate_encoding_scratch_bytes() : 0;
    for (const auto& surface : leaf.surfaces)
        if (surface.geometry)
            bytes = std::max(bytes, surface.geometry->encoding_scratch_bytes());
    return bytes;
}
std::size_t adaptive_detail::validate_adaptive_leaf_encoding(const AdaptiveLeaf& leaf, std::size_t scratch_budget)
{
    if (adaptive_leaf_encoding_scratch_bytes(leaf) > scratch_budget)
        throw std::invalid_argument("Adaptive encoding scratch exceeds its byte budget.");
    SizeWriter w;
    emit_adaptive_leaf(w, leaf, scratch_budget);
    return w.size;
}

bool adaptive_detail::adaptive_surface_count_valid(
    const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& geometry, std::uint64_t count) noexcept
{
    return count <= PLIC::GeometryLimits{}.maximum_faces && (geometry ? count >= geometry->num_faces() : count == 0);
}

void adaptive_detail::validate_adaptive_leaf_boundary(const AdaptiveLeaf& leaf, std::size_t byte_budget)
{
    if (!adaptive_surface_count_valid(leaf.geometry, leaf.surfaces.size()))
        throw std::invalid_argument("Invalid adaptive surface proof count.");
    if (!leaf.geometry)
        return;
    Budget admission{byte_budget};
    if (leaf.surfaces.capacity() > byte_budget / sizeof(AdaptiveSurface))
        throw std::invalid_argument("Adaptive surface proof inputs exceed the byte budget.");
    admission.add(leaf.surfaces.capacity() * sizeof(AdaptiveSurface));
    bool has_whole = false;
    for (const auto& surface : leaf.surfaces)
    {
        if (!surface.geometry || (surface.require_native_whole_tile && !surface.native_whole_geometry))
            throw std::invalid_argument("Missing adaptive surface or whole-face certificate.");
        admission.add(surface.key.capacity());
        admission.add(surface.boundary.capacity());
        has_whole |= surface.require_native_whole_tile;
    }
    const auto count = leaf.surfaces.size();
    const auto row_bytes = sizeof(SupportFace) * (has_whole ? 2 : 1);
    if (count > (admission.limit - admission.used) / row_bytes)
        throw std::invalid_argument("Adaptive surface proof rows exceed the byte budget.");
    admission.add(count * row_bytes);
    std::vector<SupportFace> faces, wholes;
    faces.reserve(count);
    if (has_whole)
        wholes.reserve(count);
    if (faces.capacity() != count || (has_whole && wholes.capacity() != count))
        throw std::invalid_argument("Adaptive surface proof row capacity differs from admission.");
    for (const auto& surface : leaf.surfaces)
    {
        faces.push_back(surface.geometry);
        if (has_whole)
            wholes.push_back(surface.require_native_whole_tile ? surface.native_whole_geometry : nullptr);
    }
    if (has_whole || count != leaf.geometry->num_faces())
        certify_adaptive_physical_boundary(leaf.geometry, faces, admission.limit - admission.used, nullptr, wholes);
    else if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces)->equivalent(*leaf.geometry))
        throw std::invalid_argument("Adaptive surface shell differs from leaf certificate.");
}

AdaptiveLeaf decode_adaptive_leaf(std::span<const char> packet)
{
    const adaptive_detail::PipelinePhaseScope phase(adaptive_detail::PipelinePhase::Transport);
    adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireDecodedLeaves);
    AdaptiveRecordReader r{packet};
    packet_codec::read_header(r, packet_codec::Family::AdaptiveRecords, record_version, RecordKind::Leaf);
    AdaptiveLeaf leaf;
    leaf.key = decode_adaptive_key(r.bytes());
    leaf.domain = r.bytes();
    leaf.material = r.word();
    leaf.revision = r.word();
    auto kind = r.word();
    if (kind > 3)
        throw std::invalid_argument("Invalid adaptive provider kind.");
    leaf.provider = AdaptiveProviderKind(kind);
    auto geometry = r.bytes();
    if (!geometry.empty())
        leaf.geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(geometry);
    auto n = r.word();
    if (!adaptive_detail::adaptive_surface_count_valid(leaf.geometry, n) || n > (packet.size() - r.position) / 32 ||
        n > PLIC::geometry_packet_default_budget / sizeof(AdaptiveSurface))
        throw std::invalid_argument("Invalid adaptive surface count.");
    leaf.surfaces.reserve(n);
    if (leaf.surfaces.capacity() != n)
        throw std::invalid_argument("Adaptive surface row capacity differs from admission.");
    for (std::uint64_t i = 0; i < n; ++i)
    {
        AdaptiveSurface s;
        s.key = r.bytes();
        s.geometry = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
        auto orientation = r.word(), exterior = r.word();
        if (orientation > 1 || exterior > 1)
            throw std::invalid_argument("Invalid adaptive incidence.");
        s.orientation = orientation ? 1 : -1;
        s.exterior = exterior;
        auto name = r.bytes();
        s.boundary.assign(name.begin(), name.end());
        s.native_face = r.word();
        s.authority_image = PLIC::decode_identity<PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>>(r.bytes());
        auto whole = r.word(), proof = r.word();
        if (whole > 1 || proof > 1)
            throw std::invalid_argument("Invalid native face nesting proof flags.");
        s.require_native_whole_tile = whole;
        if (proof)
            s.native_whole_geometry = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
        if (whole && !s.native_whole_geometry)
            throw std::invalid_argument("Missing original native face nesting proof.");
        leaf.surfaces.push_back(std::move(s));
    }
    auto recipe = r.bytes();
    if (!recipe.empty())
    {
        NativeFragmentCache cache;
        leaf.native_recipe = decode_native_fragment_descriptor(recipe, cache);
    }
    leaf.native_cell = r.word();
    auto factor = std::make_shared<AdaptiveFactorRecipe>();
    for (auto* part : {&factor->root, &factor->base, &factor->axis})
    {
        auto bytes = r.bytes();
        if (!bytes.empty())
            *part = std::make_shared<const AdaptiveBytes>(std::move(bytes));
    }
    if (factor->root || factor->base || factor->axis)
        leaf.factor_recipe = std::move(factor);
    auto join = r.bytes();
    if (!join.empty())
        leaf.join_recipe = std::make_shared<const AdaptiveBytes>(std::move(join));
    leaf.parent_family = r.bytes();
    leaf.work = {std::bit_cast<double>(r.word()), std::bit_cast<double>(r.word()), std::bit_cast<double>(r.word()),
        std::bit_cast<double>(r.word())};
    r.done();
    if (!std::ranges::equal(encode_adaptive_leaf(leaf), packet))
        throw std::invalid_argument("Noncanonical adaptive leaf.");
    adaptive_detail::validate_adaptive_leaf_boundary(leaf, PLIC::geometry_packet_default_budget);
    return leaf;
}
AdaptiveBytes encode_adaptive_family(const AdaptiveFamily& family)
{
    const adaptive_detail::PipelinePhaseScope phase(adaptive_detail::PipelinePhase::Transport);
    adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireEncodedFamilies);
    if (family.key.empty() || family.children.empty())
        throw std::invalid_argument("Invalid adaptive family.");
    Writer w;
    packet_codec::write_header(w, packet_codec::Family::AdaptiveRecords, record_version, RecordKind::Family);
    w.bytes(family.key);
    w.bytes(encode_adaptive_leaf(family.parent));
    w.word(std::uint64_t(family.split));
    w.word(family.revision);
    w.bytes(family.anchor);
    w.word(family.children.size());
    std::set<AdaptiveCellKey> unique;
    for (const auto& child : family.children)
    {
        if (!unique.insert(child).second)
            throw std::invalid_argument("Duplicate family child.");
        w.bytes(encode_adaptive_key(child));
    }
    return std::move(w.data);
}
AdaptiveFamily decode_adaptive_family(std::span<const char> packet)
{
    const adaptive_detail::PipelinePhaseScope phase(adaptive_detail::PipelinePhase::Transport);
    adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireDecodedFamilies);
    AdaptiveRecordReader r{packet};
    packet_codec::read_header(r, packet_codec::Family::AdaptiveRecords, record_version, RecordKind::Family);
    AdaptiveFamily family;
    family.key = r.bytes();
    family.parent = decode_adaptive_leaf(r.bytes());
    auto split = r.word();
    if (split > 2)
        throw std::invalid_argument("Invalid family split.");
    family.split = AdaptiveSplit(split);
    family.revision = r.word();
    family.anchor = r.bytes();
    auto n = r.word();
    if (n > (packet.size() - r.position) / 8)
        throw std::invalid_argument("Invalid family size.");
    for (std::uint64_t i = 0; i < n; ++i)
        family.children.push_back(decode_adaptive_key(r.bytes()));
    r.done();
    (void) encode_adaptive_family(family);
    return family;
}
namespace adaptive_detail
{
struct CompactNativeOwnedBudget::Impl
{
    enum class Kind
    {
        Face,
        FacePayload,
        NativeRecipe,
        Bytes,
        Factor
    };
    struct Less
    {
        bool operator()(
            const std::pair<Kind, const void*>& first, const std::pair<Kind, const void*>& second) const noexcept
        {
            return first.first != second.first ? first.first < second.first
                                               : std::less<const void*>{}(first.second, second.second);
        }
    };
    Budget budget;
    PLIC::GeometryAllocationLedger retained_geometry;
    std::size_t graph_charged = 0;
    std::set<std::pair<Kind, const void*>, Less> objects;
    CompactNativeOwnedStats stats;
    const std::vector<AdaptiveLeaf>* destination = nullptr;
    std::size_t destination_size = 0, destination_capacity = 0;
    bool failed = false;
    explicit Impl(std::size_t limit) : budget{limit} { budget.add(sizeof(Impl)); }
    void destination_admission(const std::vector<AdaptiveLeaf>& values)
    {
        if (!destination)
        {
            if (!values.empty())
                throw std::invalid_argument("Compact native ledger needs an initially empty destination.");
            capacity(values.capacity(), sizeof(AdaptiveLeaf));
            destination = &values;
            destination_capacity = values.capacity();
        }
        else if (destination != &values || destination_size != values.size() ||
                 destination_capacity != values.capacity())
            throw std::invalid_argument("Compact native destination changed outside its admission ledger.");
    }
    void capacity(std::size_t count, std::size_t width)
    {
        if (width && count > std::numeric_limits<std::size_t>::max() / width)
            throw std::invalid_argument("Compact native retained capacity overflows.");
        budget.add(count * width);
    }
    bool unique(Kind kind, const void* pointer)
    {
        if (!pointer || objects.contains({kind, pointer}))
            return false;
        // Same conservative ordered-node estimate used by AMR admission.
        budget.add(sizeof(typename decltype(objects)::value_type) + 4 * sizeof(void*));
        objects.emplace(kind, pointer);
        return true;
    }
    template<class Value> void geometry_owner(const Value& value)
    {
        const auto before = graph_charged;
        const auto available = budget.limit - budget.used;
        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto limit = available > maximum - graph_charged ? maximum : graph_charged + available;
        try
        {
            retained_geometry.retain(value, graph_charged, limit);
        }
        catch (...)
        {
            budget.add(graph_charged - before);
            throw;
        }
        budget.add(graph_charged - before);
    }
    void bytes_owner(const std::shared_ptr<const AdaptiveBytes>& value)
    {
        if (value && unique(Kind::Bytes, value.get()))
        {
            budget.add(sizeof(AdaptiveBytes));
            budget.add(value->capacity());
        }
    }
    void face_owner(const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& face)
    {
        if (!face)
            return;
        if (unique(Kind::Face, face.get()))
            budget.add(sizeof(*face));
        if (unique(Kind::FacePayload, face->storage_identity()))
            budget.add(face->exclusive_storage_bytes());
        geometry_owner(PLIC::planar_support<ArithmeticMode::Ordinary>(*face));
    }
    void leaf_metadata(const AdaptiveLeaf& leaf)
    {
        std::visit(
            [&](const auto& ancestry)
            {
                if constexpr (requires { ancestry.path; })
                    capacity(ancestry.path.capacity(), sizeof(std::uint64_t));
                else if constexpr (requires { ancestry.base.path; })
                {
                    capacity(ancestry.base.path.capacity(), sizeof(std::uint64_t));
                    capacity(ancestry.axial.path.capacity(), sizeof(std::uint64_t));
                }
            },
            leaf.key.ancestry);
        budget.add(leaf.domain.capacity());
        budget.add(leaf.parent_family.capacity());
        capacity(leaf.surfaces.capacity(), sizeof(AdaptiveSurface));
        for (const auto& surface : leaf.surfaces)
        {
            budget.add(surface.key.capacity());
            // Counting SSO capacity too is conservative and avoids relying on
            // a particular standard-library string allocation threshold.
            budget.add(surface.boundary.capacity());
            // Ordinary displacement identity is inline in AdaptiveSurface.
            face_owner(surface.geometry);
            face_owner(surface.native_whole_geometry);
        }
        if (leaf.native_recipe && unique(Kind::NativeRecipe, leaf.native_recipe.get()))
        {
            budget.add(sizeof(NativeFragmentDescriptor));
            budget.add(leaf.native_recipe->view.capacity());
            capacity(leaf.native_recipe->parts.capacity(), sizeof(std::shared_ptr<const AdaptiveBytes>));
            for (const auto& part : leaf.native_recipe->parts)
                bytes_owner(part);
        }
        bytes_owner(leaf.join_recipe);
        if (leaf.factor_recipe && unique(Kind::Factor, leaf.factor_recipe.get()))
        {
            budget.add(sizeof(AdaptiveFactorRecipe));
            bytes_owner(leaf.factor_recipe->root);
            bytes_owner(leaf.factor_recipe->base);
            bytes_owner(leaf.factor_recipe->axis);
        }
    }
};
CompactNativeOwnedBudget::CompactNativeOwnedBudget(std::size_t limit)
{
    if (limit < sizeof(Impl))
        throw std::invalid_argument("Compact native ledger exceeds its byte budget.");
    impl_ = std::make_unique<Impl>(limit);
}
CompactNativeOwnedBudget::~CompactNativeOwnedBudget() = default;
void CompactNativeOwnedBudget::append(
    std::vector<AdaptiveLeaf>& destination, AdaptiveLeaf transient, bool discard_geometry)
{
    auto& state = *impl_;
    if (state.failed)
        throw std::invalid_argument("Compact native admission cannot resume a rejected transaction.");
    try
    {
        state.destination_admission(destination);
        state.leaf_metadata(transient);
        if (!discard_geometry && transient.geometry)
            state.geometry_owner(transient.geometry);
        std::size_t transient_bytes = 0;
        {
            // Only the current disposable full cell enters this local ledger.
            // Destroy all its pins before destroying the actual owner below.
            PLIC::GeometryAllocationLedger scratch_geometry;
            if (discard_geometry && transient.geometry)
            {
                scratch_geometry.retain(transient.geometry, transient_bytes, state.budget.limit - state.budget.used);
                state.budget.add(transient_bytes);
                state.stats.transient_geometry_peak_bytes =
                    std::max(state.stats.transient_geometry_peak_bytes, transient_bytes);
            }
            const auto scratch = adaptive_leaf_encoding_scratch_bytes(transient);
            state.budget.add(scratch);
            state.stats.encoding_scratch_peak_bytes = std::max(state.stats.encoding_scratch_peak_bytes, scratch);
            const auto serialized = validate_adaptive_leaf_encoding(transient, scratch);
            if (serialized > std::numeric_limits<std::size_t>::max() - state.stats.serialized_bytes)
                throw std::invalid_argument("Compact native cumulative serialized bytes overflow.");
            state.stats.serialized_bytes += serialized;
            // validate_adaptive_leaf_encoding has returned and released every
            // conversion/canonicalization buffer before its lease retires.
            state.budget.used -= scratch;
        }
        if (discard_geometry)
            transient.geometry.reset();
        state.budget.used -= transient_bytes;
        state.budget.append_capacity(destination);
        destination.push_back(std::move(transient));
        state.destination_size = destination.size();
        state.destination_capacity = destination.capacity();
        ++state.stats.leaves;
    }
    catch (...)
    {
        state.failed = true;
        throw;
    }
}
bool CompactNativeOwnedBudget::append_bounded(
    std::vector<AdaptiveLeaf>& destination, const AdaptiveLeaf& source, const AdaptiveProviderKernel& kernel)
{
    auto& state = *impl_;
    if (state.failed)
        throw std::invalid_argument("Compact native admission cannot resume a rejected transaction.");
    try
    {
        const auto bound = kernel.compact_native_materialization_bound(source);
        if (!bound)
            return false;
        state.destination_admission(destination);
        state.budget.add(bound);
        state.stats.materialization_peak_bytes = std::max(state.stats.materialization_peak_bytes, bound);
        record_pipeline_work(PipelineWork::NativeMaterializationRequest);
        auto transient = kernel.compact_native_materialize(source);
        // Keep the construction envelope leased while append admits the
        // returned graph and allocates its tracking metadata. This deliberately
        // double-counts one bounded record rather than exposing an uncharged
        // live graph during the transfer to persistent admission.
        append(destination, std::move(transient), true);
        state.budget.used -= bound;
        return true;
    }
    catch (...)
    {
        state.failed = true;
        throw;
    }
}
CompactNativeOwnedStats CompactNativeOwnedBudget::statistics() const noexcept
{
    auto result = impl_->stats;
    result.retained_bytes = impl_->budget.used;
    result.live_peak_bytes = impl_->budget.peak;
    result.cumulative_bytes = impl_->budget.cumulative;
    return result;
}
} // namespace adaptive_detail
} // namespace SimpleFluid::Meshes::AMR
