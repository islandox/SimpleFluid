/** @file DistributedAdaptivePackets.cc
 * @brief Versioned shared-part packets for owner migration and family exchanges.
 */
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryPacketHeader.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/PLIC/details/NativeGeometryByteWriter.hh"
#include "geometry/PLIC/details/NativePlanarCodecAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/mesh/AMR/AdaptiveFamilyProofFragment.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using adaptive_detail::Reader;
using Cell = PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>;
using Face = PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>;
using Displacement = PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>;
using Admission = geometry_detail::NativeConstructionAdmission;
static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big);
struct Writer
{
    AdaptiveBytes data;
    std::size_t limit = std::numeric_limits<std::size_t>::max();
    void room(std::size_t n) const
    {
        if (n > limit - data.size())
            throw std::invalid_argument("Adaptive batch wire budget exhausted.");
    }
    void word(std::uint64_t n)
    {
        room(8);
        for (unsigned i = 0; i < 8; ++i)
            data.push_back(char(n >> (8 * i)));
    }
    void bytes(std::span<const char> value)
    {
        room(8);
        if (value.size() > limit - data.size() - 8)
            throw std::invalid_argument("Adaptive batch wire budget exhausted.");
        word(value.size());
        data.insert(data.end(), value.begin(), value.end());
    }
    void key(const AdaptiveCellKey& value) { bytes(encode_adaptive_key(value)); }
    void cell(const Cell& value) { bytes(value ? value->encode_certificate(limit) : AdaptiveBytes{}); }
    void face(const Face& value) { bytes(value ? value->encode_incidence(limit) : AdaptiveBytes{}); }
    void image(const Displacement& value) { bytes(PLIC::encode_identity(value)); }
};
void write_batch_header(AdaptiveBytes& data, std::uint64_t dictionary_count)
{
    using adaptive_detail::BatchHeader;
    data.resize(sizeof(BatchHeader), 0);
    const auto family = adaptive_detail::PacketFamily::AdaptiveBatch;
    const auto version = adaptive_detail::batch_version;
    if constexpr (std::endian::native == std::endian::big)
        dictionary_count = std::byteswap(dictionary_count);
    std::memcpy(data.data() + offsetof(BatchHeader, family), &family, sizeof(family));
    std::memcpy(data.data() + offsetof(BatchHeader, version), &version, sizeof(version));
    std::memcpy(data.data() + offsetof(BatchHeader, dictionary_count), &dictionary_count, sizeof(dictionary_count));
}
std::uint64_t read_batch_header(Reader& reader)
{
    using adaptive_detail::BatchHeader;
    if (reader.data.size() < sizeof(BatchHeader))
        throw std::invalid_argument("Truncated adaptive batch header.");
    const auto* bytes = reader.data.data();
    if (static_cast<std::uint8_t>(bytes[offsetof(BatchHeader, family)]) !=
            static_cast<std::uint8_t>(adaptive_detail::PacketFamily::AdaptiveBatch) ||
        static_cast<std::uint8_t>(bytes[offsetof(BatchHeader, version)]) != adaptive_detail::batch_version)
        throw std::invalid_argument("Unsupported adaptive batch packet family or version.");
    for (std::size_t i = 0; i < sizeof(BatchHeader); ++i)
    {
        const auto in_field = [i](std::size_t offset, std::size_t width) { return i >= offset && i - offset < width; };
        const bool field = in_field(offsetof(BatchHeader, family), sizeof(BatchHeader::family)) ||
                           in_field(offsetof(BatchHeader, version), sizeof(BatchHeader::version)) ||
                           in_field(offsetof(BatchHeader, dictionary_count), sizeof(BatchHeader::dictionary_count));
        if (!field && bytes[i] != 0)
            throw std::invalid_argument("Nonzero adaptive batch header padding.");
    }
    std::uint64_t count;
    std::memcpy(&count, bytes + offsetof(BatchHeader, dictionary_count), sizeof(count));
    if constexpr (std::endian::native == std::endian::big)
        count = std::byteswap(count);
    reader.offset = sizeof(BatchHeader);
    return count;
}
using Part = std::shared_ptr<const AdaptiveBytes>;
Part native_view(const std::shared_ptr<const NativeFragmentDescriptor>& recipe)
{
    // Alias immutable descriptor storage; dictionary equality is full bytes,
    // and no per-record copy of a potentially large run view is allocated.
    return recipe ? Part(recipe, &recipe->view) : nullptr;
}
struct PartLess
{
    bool operator()(const Part& a, const Part& b) const { return *a < *b; }
};
template<class Index, class Parts> struct BatchDictionary
{
    Index index;
    Parts parts;
    std::size_t budget, charged = 0;
    bool emit_payload = true;
    void add(const Part& part)
    {
        if (part && !index.contains(part))
        {
            constexpr auto metadata = sizeof(Part) + sizeof(std::pair<const Part, std::uint64_t>) + 4 * sizeof(void*);
            const auto payload = emit_payload ? part->size() : 0;
            if (metadata > budget - charged || payload > budget - charged - metadata)
                throw std::invalid_argument("Adaptive batch dictionary budget exhausted.");
            charged += metadata + payload;
            index.emplace(part, parts.size() + 1);
            parts.push_back(part);
        }
    }
    void add(const AdaptiveLeaf& leaf)
    {
        add(leaf.join_recipe);
        if (leaf.native_recipe)
        {
            add(native_view(leaf.native_recipe));
            for (const auto& part : leaf.native_recipe->parts)
            {
                if (!part)
                    throw std::invalid_argument("Null native batch recipe part.");
                add(part);
            }
        }
        if (leaf.factor_recipe)
            for (const auto& part : {leaf.factor_recipe->root, leaf.factor_recipe->base, leaf.factor_recipe->axis})
                add(part);
    }
    std::uint64_t id(const Part& part) const { return part ? index.at(part) : 0; }
};
using Dictionary = BatchDictionary<std::map<Part, std::uint64_t, PartLess>, std::vector<Part>>;
using NativeDictionaryIndex = std::map<Part, std::uint64_t, PartLess,
    geometry_detail::NativeConstructionAllocator<std::pair<const Part, std::uint64_t>>>;
using NativeDictionary = BatchDictionary<NativeDictionaryIndex, geometry_detail::NativeConstructorVector<Part>>;

void validate_frame(const AdaptiveLeaf& leaf)
{
    if (leaf.domain.empty() || std::uint64_t(leaf.provider) > 3 ||
        !adaptive_detail::adaptive_surface_count_valid(leaf.geometry, leaf.surfaces.size()) || (!leaf.geometry && !leaf.native_recipe))
        throw std::invalid_argument("Incomplete adaptive batch leaf.");
    (void) leaf.work.total();
    for (const auto& s : leaf.surfaces)
    {
        if (s.key.empty() || !s.geometry || (s.orientation != 1 && s.orientation != -1) ||
            (!s.exterior && !s.boundary.empty()))
            throw std::invalid_argument("Invalid adaptive batch surface.");
        if (s.require_native_whole_tile && !s.native_whole_geometry)
            throw std::invalid_argument("Missing native whole-face certificate.");
    }
    if (leaf.native_recipe && std::uint64_t(leaf.native_recipe->kind) > 2)
        throw std::invalid_argument("Invalid native batch recipe kind.");
}
void validate(const AdaptiveLeaf& leaf, std::size_t byte_budget, const Admission* admission = nullptr)
{
    validate_frame(leaf);
    if (admission && leaf.geometry)
    {
        using adaptive_detail::SupportFace;
        adaptive_detail::Budget inputs{byte_budget};
        inputs.add(leaf.surfaces.capacity() * sizeof(AdaptiveSurface));
        bool has_whole = false;
        for (const auto& surface : leaf.surfaces)
        {
            inputs.add(surface.key.capacity());
            inputs.add(surface.boundary.capacity());
            has_whole |= surface.require_native_whole_tile;
        }
        inputs.add(leaf.surfaces.size() * sizeof(SupportFace) * (has_whole ? 2 : 1));
        geometry_detail::NativeConstructorVector<SupportFace> faces(admission->allocator<SupportFace>()),
            wholes(admission->allocator<SupportFace>());
        faces.reserve(leaf.surfaces.size());
        if (has_whole)
            wholes.reserve(leaf.surfaces.size());
        for (const auto& surface : leaf.surfaces)
        {
            faces.push_back(surface.geometry);
            if (has_whole)
                wholes.push_back(surface.require_native_whole_tile ? surface.native_whole_geometry : nullptr);
        }
        adaptive_detail::certify_adaptive_physical_boundary_native(
            leaf.geometry, faces, inputs.limit - inputs.used, *admission, wholes);
    }
    else
        adaptive_detail::validate_adaptive_leaf_boundary(leaf, byte_budget);
}
void write_leaf(auto& w, const AdaptiveLeaf& leaf, const auto& dict)
{
    w.key(leaf.key);
    w.bytes(leaf.domain);
    w.word(leaf.material);
    w.word(leaf.revision);
    w.word(std::uint64_t(leaf.provider));
    w.cell(leaf.geometry);
    w.word(leaf.surfaces.size());
    for (const auto& s : leaf.surfaces)
    {
        w.bytes(s.key);
        w.face(s.geometry);
        w.image(s.authority_image);
        w.word(s.require_native_whole_tile);
        w.face(s.native_whole_geometry);
        w.word(s.native_face);
        w.word(s.orientation == 1);
        w.word(s.exterior);
        w.bytes(std::span(s.boundary));
    }
    w.word(bool(leaf.native_recipe));
    if (leaf.native_recipe)
    {
        w.word(std::uint64_t(leaf.native_recipe->kind));
        w.word(leaf.native_recipe->parts.size());
        for (const auto& part : leaf.native_recipe->parts)
            w.word(dict.id(part));
        w.word(dict.id(native_view(leaf.native_recipe)));
    }
    w.word(leaf.native_cell);
    w.word(bool(leaf.factor_recipe));
    if (leaf.factor_recipe)
        for (const auto& part : {leaf.factor_recipe->root, leaf.factor_recipe->base, leaf.factor_recipe->axis})
            w.word(dict.id(part));
    w.word(dict.id(leaf.join_recipe));
    w.bytes(leaf.parent_family);
    for (double value : {leaf.work.reconstruction, leaf.work.interface, leaf.work.population, leaf.work.migration})
        w.word(std::bit_cast<std::uint64_t>(value));
}
/** Public batch rows own standard containers. Keep their profiled C++ requests
 * in the same admission as concrete geometry until control returns to the
 * caller's decoded-residency calculation. */
class DecodeMetadata
{
public:
    explicit DecodeMetadata(const geometry_detail::NativeConstructionAdmission* admission)
        : state_(admission ? admission->state() : nullptr)
    {
    }
    ~DecodeMetadata()
    {
        if (state_ && bytes_)
            state_->finish_profile(bytes_, false);
    }
    DecodeMetadata(const DecodeMetadata&) = delete;
    DecodeMetadata& operator=(const DecodeMetadata&) = delete;

    void add(std::size_t bytes)
    {
        if (state_ && bytes)
        {
            state_->reserve(bytes, false);
            bytes_ += bytes;
        }
    }
    void append(std::size_t count, std::size_t extent)
    {
        if (count > std::numeric_limits<std::size_t>::max() / extent)
            throw std::invalid_argument("Adaptive batch metadata extent overflows.");
        add(count * extent);
    }
    template<class T> void reserve(std::vector<T>& values, std::size_t count)
    {
        append(count, sizeof(T));
        values.reserve(count);
        if (values.capacity() != count)
            throw std::invalid_argument("Adaptive batch row capacity differs from admission.");
    }
    template<class T>
    void node()
    {
#if defined(__GLIBCXX__)
        add(sizeof(std::_Rb_tree_node<T>));
#else
        if (state_)
            throw std::invalid_argument("Unsupported native adaptive batch metadata profile.");
#endif
    }
    AdaptiveBytes bytes(Reader& r)
    {
        const auto value = r.bytes_view();
        add(value.size());
        return {value.begin(), value.end()};
    }
    AdaptiveCellKey key(Reader& r)
    {
        const auto value = r.bytes_view();
        // A canonical key's only allocations are its two possible ancestry
        // path arrays, each smaller than its encoded frame.
        add(value.size());
        return decode_adaptive_key(value);
    }
    void string(std::string& result, Reader& r)
    {
        const auto value = r.bytes_view();
        if (value.size() > result.capacity())
            add(std::max(value.size(), 2 * result.capacity()) + 1);
        result.assign(value.data(), value.size());
    }

private:
    std::shared_ptr<const geometry_detail::NativeAllocationState> state_;
    std::size_t bytes_ = 0;
};

struct Decoder
{
    std::size_t byte_budget;
    const geometry_detail::NativeConstructionAdmission* admission;
    DecodeMetadata metadata;
    explicit Decoder(std::size_t limit, const geometry_detail::NativeConstructionAdmission* native = nullptr)
        : byte_budget(limit), admission(native), metadata(native)
    {
    }
    std::vector<Part> parts;
    std::map<AdaptiveBytes, std::shared_ptr<const NativeFragmentDescriptor>> recipes;
    std::map<std::array<std::uint64_t, 3>, std::shared_ptr<const AdaptiveFactorRecipe>> factors;
    Part part(std::uint64_t id) const
    {
        if (id > parts.size())
            throw std::invalid_argument("Adaptive batch part reference is out of range.");
        return id ? parts[id - 1] : nullptr;
    }
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> planar(std::span<const char> encoded) const
    {
        return admission ? PLIC::details::NativePlanarOwnerAccess::decode(encoded, byte_budget, *admission)
                         : PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(encoded, byte_budget);
    }
    AdaptiveLeaf leaf(Reader& r)
    {
        AdaptiveLeaf leaf;
        leaf.key = metadata.key(r);
        leaf.domain = metadata.bytes(r);
        leaf.material = r.word();
        leaf.revision = r.word();
        const auto provider = r.word();
        if (provider > 3)
            throw std::invalid_argument("Invalid adaptive batch provider.");
        leaf.provider = AdaptiveProviderKind(provider);
        const auto geometry = r.bytes_view();
        if (!geometry.empty())
            leaf.geometry = admission
                ? PLIC::details::NativeCellCodecAccess::decode(geometry, byte_budget, *admission)
                : PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(geometry, byte_budget);
        const auto count = r.word();
        if (!adaptive_detail::adaptive_surface_count_valid(leaf.geometry, count) || count > (r.data.size() - r.offset) / 32 ||
            count > byte_budget / sizeof(AdaptiveSurface))
            throw std::invalid_argument("Invalid adaptive batch surface count.");
        metadata.reserve(leaf.surfaces, count);
        for (std::uint64_t i = 0; i < count; ++i)
        {
            AdaptiveSurface s;
            s.key = metadata.bytes(r);
            s.geometry = planar(r.bytes_view());
            s.authority_image =
                PLIC::decode_identity<PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>>(r.bytes_view(), byte_budget);
            const auto native_whole = r.word();
            if (native_whole > 1)
                throw std::invalid_argument("Invalid native whole-face declaration.");
            s.require_native_whole_tile = native_whole;
            const auto whole_geometry = r.bytes_view();
            if (!whole_geometry.empty())
                s.native_whole_geometry = planar(whole_geometry);
            if (s.require_native_whole_tile && !s.native_whole_geometry)
                throw std::invalid_argument("Missing native whole-face certificate.");
            s.native_face = r.word();
            const auto orientation = r.word(), exterior = r.word();
            if (orientation > 1 || exterior > 1)
                throw std::invalid_argument("Invalid adaptive batch incidence.");
            s.orientation = orientation ? 1 : -1;
            s.exterior = exterior;
            metadata.string(s.boundary, r);
            leaf.surfaces.push_back(std::move(s));
        }
        const auto native = r.word();
        if (native > 1)
            throw std::invalid_argument("Invalid adaptive batch native flag.");
        if (native)
        {
            const auto kind = r.word(), n = r.word();
            if (kind > 2 || n > (r.data.size() - r.offset) / 8)
                throw std::invalid_argument("Invalid native batch descriptor.");
            Writer key;
            if (n > std::numeric_limits<std::size_t>::max() / 8 - 3)
                throw std::invalid_argument("Native batch descriptor key extent overflows.");
            metadata.reserve(key.data, (n + 3) * 8);
            key.word(kind);
            key.word(n);
            for (std::uint64_t i = 0; i < n; ++i)
            {
                auto id = r.word();
                if (!part(id))
                    throw std::invalid_argument("Null native batch descriptor part.");
                key.word(id);
            }
            const auto view_id = r.word();
            auto view = part(view_id);
            if (!view)
                throw std::invalid_argument("Null native batch descriptor view.");
            key.word(view_id);
            if (const auto existing = recipes.find(key.data); existing != recipes.end())
                leaf.native_recipe = existing->second;
            else
            {
                using Access = geometry_detail::NativeObjectOwnerAccess<NativeFragmentDescriptor>;
                auto recipe = admission ? Access::make(*admission).owner : std::make_shared<NativeFragmentDescriptor>();
                recipe->kind = NativeFragmentDescriptor::Kind(kind);
                Reader ids{key.data, 16};
                if (admission)
                {
                    const auto& storage = Access::storage(recipe);
                    storage.owner->reserve(recipe->parts, n);
                    storage.owner->copy(recipe->view, std::span<const char>(*view));
                }
                else
                {
                    recipe->parts.reserve(n);
                    recipe->view = *view;
                }
                for (std::uint64_t i = 0; i < n; ++i)
                    recipe->parts.push_back(part(ids.word()));
                metadata.node<decltype(recipes)::value_type>();
                leaf.native_recipe = recipes.emplace(std::move(key.data), std::move(recipe)).first->second;
            }
        }
        leaf.native_cell = r.word();
        const auto factor = r.word();
        if (factor > 1)
            throw std::invalid_argument("Invalid adaptive batch factor flag.");
        if (factor)
        {
            std::array<std::uint64_t, 3> ids{r.word(), r.word(), r.word()};
            if (const auto existing = factors.find(ids); existing != factors.end())
                leaf.factor_recipe = existing->second;
            else
            {
                using Access = geometry_detail::NativeObjectOwnerAccess<AdaptiveFactorRecipe>;
                auto result = admission ? Access::make_without_buffers(*admission).owner
                                        : std::make_shared<AdaptiveFactorRecipe>();
                result->root = part(ids[0]);
                result->base = part(ids[1]);
                result->axis = part(ids[2]);
                metadata.node<decltype(factors)::value_type>();
                leaf.factor_recipe = factors.emplace(ids, std::move(result)).first->second;
            }
        }
        leaf.join_recipe = part(r.word());
        leaf.parent_family = metadata.bytes(r);
        leaf.work = {std::bit_cast<double>(r.word()), std::bit_cast<double>(r.word()), std::bit_cast<double>(r.word()),
            std::bit_cast<double>(r.word())};
        validate(leaf, byte_budget, admission);
        return leaf;
    }
};
void family_valid(const AdaptiveFamily& family, DecodeMetadata* metadata = nullptr)
{
    if (family.key.empty() || family.children.empty() || std::uint64_t(family.split) > 2)
        throw std::invalid_argument("Invalid adaptive batch family.");
    std::set<AdaptiveCellKey> unique;
    for (const auto& child : family.children)
    {
        if (metadata)
        {
            metadata->node<AdaptiveCellKey>();
            metadata->add(encoded_adaptive_key_size(child));
        }
        if (!unique.insert(child).second)
            throw std::invalid_argument("Duplicate adaptive batch family child.");
    }
}
void write_family(auto& w, const AdaptiveFamily& family, const auto& dict)
{
    w.bytes(family.key);
    write_leaf(w, family.parent, dict);
    w.word(std::uint64_t(family.split));
    w.word(family.revision);
    w.bytes(family.anchor);
    w.word(family.children.size());
    for (const auto& child : family.children)
        w.key(child);
}
AdaptiveFamily read_family(Reader& r, Decoder& decoder)
{
    AdaptiveFamily family;
    family.key = decoder.metadata.bytes(r);
    family.parent = decoder.leaf(r);
    const auto split = r.word();
    if (split > 2)
        throw std::invalid_argument("Invalid adaptive batch family split.");
    family.split = AdaptiveSplit(split);
    family.revision = r.word();
    family.anchor = decoder.metadata.bytes(r);
    const auto n = r.word();
    if (n > (r.data.size() - r.offset) / 8)
        throw std::invalid_argument("Invalid adaptive batch child count.");
    decoder.metadata.reserve(family.children, n);
    for (std::uint64_t j = 0; j < n; ++j)
        family.children.push_back(decoder.metadata.key(r));
    family_valid(family, decoder.admission ? &decoder.metadata : nullptr);
    return family;
}

struct NativeBatchCount
{
    const Admission& admission;
    adaptive_detail::Budget extent;
    void word(std::uint64_t) { extent.add(8); }
    void framed(std::size_t size)
    {
        word(size);
        extent.add(size);
    }
    void bytes(std::span<const char> value) { framed(value.size()); }
    void key(const AdaptiveCellKey& value) { framed(encoded_adaptive_key_size(value)); }
    void cell(const Cell& value)
    {
        framed(value ? PLIC::details::NativeCellCodecAccess::size(*value, admission) : 0);
    }
    void face(const Face& value)
    {
        framed(value ? PLIC::details::NativePlanarCodecAccess::size(*value, true, admission) : 0);
    }
    void image(const Displacement& value) { framed(PLIC::encoded_identity_size(value)); }
};
struct NativeBatchWriter
{
    adaptive_detail::Writer output;
    const Admission& admission;
    void room(std::size_t size) const
    {
        if (size > output.data.capacity() - output.data.size())
            throw std::invalid_argument("Native batch field exceeds its admitted buffer.");
    }
    void word(std::uint64_t value)
    {
        room(8);
        output.word(value);
    }
    void bytes(std::span<const char> value)
    {
        word(value.size());
        room(value.size());
        output.data.insert(output.data.end(), value.begin(), value.end());
    }
    void key(const AdaptiveCellKey& value) { adaptive_detail::NativeAdaptiveKeyCodec::write(output, value); }
    void cell(const Cell& value)
    {
        word(value ? PLIC::details::NativeCellCodecAccess::size(*value, admission) : 0);
        if (value)
            PLIC::details::NativeCellCodecAccess::append(output.data, *value, admission);
    }
    void face(const Face& value)
    {
        word(value ? PLIC::details::NativePlanarCodecAccess::size(*value, true, admission) : 0);
        if (value)
            PLIC::details::NativePlanarCodecAccess::append(output.data, *value, true, admission);
    }
    void image(const Displacement& value)
    {
        const auto size = PLIC::encoded_identity_size(value);
        word(size);
        room(size);
        const auto header = PLIC::details::geometry_packet_header(PLIC::GeometryPacketKind::DisplacementIdentity,
            ArithmeticMode::Ordinary, size - PLIC::geometry_packet_header_bytes);
        output.data.insert(output.data.end(), header.begin(), header.end());
        PLIC::details::NativeGeometryByteWriter{&output.data, &admission}.native_identity(value);
    }
};

void write_records(auto& writer, const auto& dictionary, std::span<const AdaptiveLeaf> leaves,
    std::span<const AdaptiveFamily> families)
{
    for (const auto& part : dictionary.parts)
        writer.bytes(*part);
    writer.word(leaves.size());
    writer.word(families.size());
    for (const auto& leaf : leaves)
        write_leaf(writer, leaf, dictionary);
    for (const auto& family : families)
        write_family(writer, family, dictionary);
}
} // namespace
AdaptiveBytes encode_adaptive_batch(
    std::span<const AdaptiveLeaf> leaves, std::span<const AdaptiveFamily> families, std::size_t budget)
{
    const adaptive_detail::PipelinePhaseScope phase(adaptive_detail::PipelinePhase::Transport);
    Dictionary dict{{}, {}, budget};
    for (const auto& leaf : leaves)
        dict.add(leaf);
    for (const auto& family : families)
        dict.add(family.parent);
    Writer w{{}, budget};
    w.room(sizeof(adaptive_detail::BatchHeader));
    write_batch_header(w.data, dict.parts.size());
    for (const auto& part : dict.parts)
        w.bytes(*part);
    w.word(leaves.size());
    w.word(families.size());
    for (const auto& leaf : leaves)
    {
        adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireEncodedLeaves);
        validate(leaf, budget);
        write_leaf(w, leaf, dict);
    }
    for (const auto& family : families)
    {
        adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireEncodedFamilies);
        family_valid(family);
        validate(family.parent, budget);
        write_family(w, family, dict);
    }
    if (w.data.size() > budget)
        throw std::invalid_argument("Adaptive batch wire budget exhausted.");
    return std::move(w.data);
}
AdaptiveBytes adaptive_detail::encode_adaptive_batch_admitted(std::span<const AdaptiveLeaf> leaves,
    std::span<const AdaptiveFamily> families, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    const PipelinePhaseScope phase(PipelinePhase::Transport);
    if (!storage.owner)
        throw std::invalid_argument("Native adaptive batch requires its concrete packet storage.");
    const auto& admission = storage.owner->admission();
    const auto limit = admission.counters().limit;
    NativeDictionary dictionary{NativeDictionaryIndex(PartLess{}, admission.allocator<NativeDictionaryIndex::value_type>()),
        geometry_detail::NativeConstructorVector<Part>(admission.allocator<Part>()), limit};
    for (const auto& leaf : leaves)
    {
        validate_frame(leaf);
        dictionary.add(leaf);
    }
    for (const auto& family : families)
    {
        DecodeMetadata scratch(&admission);
        family_valid(family, &scratch);
        validate_frame(family.parent);
        dictionary.add(family.parent);
    }
    NativeBatchCount counted{admission, Budget{limit}};
    counted.extent.add(sizeof(BatchHeader));
    write_records(counted, dictionary, leaves, families);
    NativeBatchWriter written{{}, admission};
    storage.owner->reserve(written.output.data, counted.extent.used);
    write_batch_header(written.output.data, dictionary.parts.size());
    write_records(written, dictionary, leaves, families);
    if (written.output.data.size() != counted.extent.used)
        throw std::invalid_argument("Native adaptive batch emission differs from its exact count.");
    record_pipeline_work(PipelineWork::WireEncodedLeaves, leaves.size());
    record_pipeline_work(PipelineWork::WireEncodedFamilies, families.size());
    return std::move(written.output.data);
}
static AdaptiveBatch decode_batch(std::span<const char> bytes, std::size_t budget,
    const geometry_detail::NativeConstructionAdmission* admission, void* context = nullptr,
    adaptive_detail::AdmittedBatchVisit inspect = nullptr)
{
    const adaptive_detail::PipelinePhaseScope phase(adaptive_detail::PipelinePhase::Transport);
    if (bytes.size() > budget)
        throw std::invalid_argument("Adaptive batch wire budget exhausted.");
    Reader r{bytes};
    const auto count = read_batch_header(r);
    Decoder decoder(budget, admission);
    AdaptiveBatch batch;
    constexpr auto part_metadata = sizeof(Part) + sizeof(AdaptiveBytes) + 6 * sizeof(void*);
    if (count > (r.data.size() - r.offset) / 8 || count > budget / part_metadata)
        throw std::invalid_argument("Invalid adaptive batch dictionary size/budget.");
    std::size_t dictionary_charge = count * part_metadata;
    std::set<Part, PartLess> unique;
    decoder.metadata.reserve(decoder.parts, count);
    for (std::uint64_t i = 0; i < count; ++i)
    {
        // Inspect the length before Reader allocates the copied payload.
        const auto old_offset = r.offset;
        const auto length = r.word();
        if (length > budget - dictionary_charge)
            throw std::invalid_argument("Adaptive batch dictionary budget exhausted.");
        dictionary_charge += static_cast<std::size_t>(length);
        r.offset = old_offset;
        const auto value = r.bytes_view();
        batch.shared_part_bytes += value.size();
        Part part;
        if (admission)
        {
            using Access = geometry_detail::NativeObjectOwnerAccess<AdaptiveBytes>;
            auto bytes = Access::make(*admission);
            Access::storage(bytes.owner).owner->copy(*bytes.owner, value);
            part = std::move(bytes.owner);
        }
        else
            part = std::make_shared<const AdaptiveBytes>(value.begin(), value.end());
        decoder.metadata.node<Part>();
        if (!unique.insert(part).second)
            throw std::invalid_argument("Duplicate adaptive batch dictionary entry.");
        decoder.parts.push_back(std::move(part));
    }
    const auto leaves = r.word(), families = r.word();
    if (leaves > (r.data.size() - r.offset) / 8 || families > (r.data.size() - r.offset) / 8 ||
        leaves > budget / sizeof(AdaptiveLeaf) || families > budget / sizeof(AdaptiveFamily))
        throw std::invalid_argument("Invalid adaptive batch record count.");
    decoder.metadata.reserve(batch.leaves, leaves);
    decoder.metadata.reserve(batch.families, families);
    for (std::uint64_t i = 0; i < leaves; ++i)
    {
        adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireDecodedLeaves);
        batch.leaves.push_back(decoder.leaf(r));
    }
    for (std::uint64_t i = 0; i < families; ++i)
    {
        adaptive_detail::record_pipeline_work(adaptive_detail::PipelineWork::WireDecodedFamilies);
        batch.families.push_back(read_family(r, decoder));
    }
    r.done();
    if (inspect)
        inspect(context, batch);
    return batch;
}
AdaptiveBatch decode_adaptive_batch(std::span<const char> bytes, std::size_t budget)
{
    return decode_batch(bytes, budget, nullptr);
}
AdaptiveBatch adaptive_detail::decode_adaptive_batch_admitted_geometry(std::span<const char> bytes, std::size_t budget,
    const geometry_detail::NativeConstructionAdmission& admission, void* context, AdmittedBatchVisit inspect)
{
    return decode_batch(bytes, budget, &admission, context, inspect);
}
adaptive_detail::AdaptiveFamilyProofFragment adaptive_detail::encode_family_proof_fragment(
    const AdaptiveFamily& family, std::size_t budget)
{
    Dictionary dict{{}, {}, budget, 0, false};
    dict.add(family.parent);
    ::SimpleFluid::Meshes::AMR::Writer writer{{}, budget};
    family_valid(family);
    validate(family.parent, budget);
    write_family(writer, family, dict);
    return {std::move(writer.data), std::move(dict.parts)};
}

AdaptiveFamily adaptive_detail::decode_family_proof_fragment(AdaptiveFamilyProofView proof, std::size_t budget)
{
    if (proof.record.empty() || proof.record.size() > budget ||
        proof.parts.size() > budget / sizeof(AdaptiveFamilyProofPart))
        throw std::invalid_argument("Product proof fragment exceeds its decode lease.");
    Reader reader{proof.record};
    Decoder decoder(budget);
    decoder.parts.assign(proof.parts.begin(), proof.parts.end());
    if (decoder.parts.capacity() != proof.parts.size())
        throw std::invalid_argument("Product proof selector capacity differs from admission.");
    auto family = read_family(reader, decoder);
    reader.done();
    return family;
}
} // namespace SimpleFluid::Meshes::AMR
