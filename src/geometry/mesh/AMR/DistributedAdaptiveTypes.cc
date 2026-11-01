/** @file DistributedAdaptiveTypes.cc
 * @brief Versioned bounded lossless ancestry and owner-record encoding.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNativeRetention.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/VoFWireProtocol.hh"
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
struct Reader
{
    std::span<const char> data;
    std::size_t position = 0;
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
BaseCellKey base(Reader& r)
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
AxialIntervalKey axial(Reader& r)
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
void canonical_key_header(Reader& r)
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
AdaptiveBytes encode_adaptive_key(const AdaptiveCellKey& key)
{
    Writer w;
    emit_adaptive_key(w, key);
    return std::move(w.data);
}

AdaptiveCellKey decode_adaptive_key(std::span<const char> packet)
{
    Reader r{packet};
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
template<class PacketWriter>
void emit_adaptive_leaf(PacketWriter& w, const AdaptiveLeaf& leaf, std::size_t scratch_budget)
{
    if (leaf.domain.empty())
        throw std::invalid_argument("Adaptive connected-domain provenance is required.");
    if ((leaf.geometry && leaf.surfaces.size() != leaf.geometry->num_faces()) ||
        (!leaf.geometry && (!leaf.surfaces.empty() || !leaf.native_recipe)))
        throw std::invalid_argument("Incomplete adaptive leaf representation.");
    (void) leaf.work.total();
    packet_codec::write_header(w, packet_codec::Family::AdaptiveRecords, record_version, RecordKind::Leaf);
    if constexpr (std::is_same_v<PacketWriter, Writer>)
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
    if constexpr (std::is_same_v<PacketWriter, Writer>)
        w.bytes(leaf.geometry ? leaf.geometry->encode_certificate() : AdaptiveBytes{});
    else
        w.sized_bytes(leaf.geometry ? leaf.geometry->encoded_certificate_size(scratch_budget) : 0);
    w.word(leaf.surfaces.size());
    for (const auto& s : leaf.surfaces)
    {
        if (s.key.empty() || !s.geometry || (s.orientation != 1 && s.orientation != -1) ||
            (!s.exterior && !s.boundary.empty()))
            throw std::invalid_argument("Invalid adaptive surface.");
        w.bytes(s.key);
        if constexpr (std::is_same_v<PacketWriter, Writer>)
            w.bytes(s.geometry->encode());
        else
            w.sized_bytes(s.geometry->encoded_size(scratch_budget));
        w.word(s.orientation == 1);
        w.word(s.exterior);
        w.bytes(std::span(s.boundary));
        w.word(s.native_face);
        if constexpr (std::is_same_v<PacketWriter, Writer>)
            w.bytes(PLIC::encode_identity(s.authority_image));
        else
            w.sized_bytes(PLIC::encoded_identity_size(s.authority_image));
        w.word(s.require_native_whole_tile);
        w.word(bool(s.native_whole_geometry));
        if (s.native_whole_geometry)
        {
            if constexpr (std::is_same_v<PacketWriter, Writer>)
                w.bytes(s.native_whole_geometry->encode_incidence());
            else
                w.sized_bytes(s.native_whole_geometry->encoded_incidence_size());
        }
    }
    if constexpr (std::is_same_v<PacketWriter, Writer>)
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
AdaptiveBytes encode_adaptive_leaf(const AdaptiveLeaf& leaf)
{
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

AdaptiveLeaf decode_adaptive_leaf(std::span<const char> packet)
{
    Reader r{packet};
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
    if (n != (leaf.geometry ? leaf.geometry->num_faces() : 0) || n > (packet.size() - r.position) / 32)
        throw std::invalid_argument("Invalid adaptive surface count.");
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
    auto canonical = encode_adaptive_leaf(leaf);
    if (canonical != AdaptiveBytes(packet.begin(), packet.end()))
        throw std::invalid_argument("Noncanonical adaptive leaf.");
    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
    for (const auto& s : leaf.surfaces)
        faces.push_back(s.geometry);
    if (leaf.geometry && !PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces)->equivalent(*leaf.geometry))
        throw std::invalid_argument("Adaptive surface shell differs from leaf certificate.");
    return leaf;
}
AdaptiveBytes encode_adaptive_family(const AdaptiveFamily& family)
{
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
    Reader r{packet};
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
