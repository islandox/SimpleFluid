/** @file DistributedAdaptivePackets.cc
 * @brief Versioned shared-part packets for owner migration and family exchanges.
 */
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <bit>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using adaptive_detail::Reader;
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
};
using Part = std::shared_ptr<const AdaptiveBytes>;
Part native_view(const std::shared_ptr<const NativeFragmentDescriptor>& recipe)
{
    // Alias immutable descriptor storage; dictionary equality is full bytes,
    // and no per-record copy of a potentially large run view is allocated.
    return recipe ? Part(recipe, &recipe->view) : nullptr;
}
constexpr std::uint64_t magic = 0x5346414442410004ULL;
struct PartLess
{
    bool operator()(const Part& a, const Part& b) const { return *a < *b; }
};
struct Dictionary
{
    std::map<Part, std::uint64_t, PartLess> index;
    std::vector<Part> parts;
    std::size_t budget, charged = 0;
    void add(const Part& part)
    {
        if (part && !index.contains(part))
        {
            constexpr auto metadata = sizeof(Part) + sizeof(std::pair<const Part, std::uint64_t>) + 4 * sizeof(void*);
            if (metadata > budget - charged || part->size() > budget - charged - metadata)
                throw std::invalid_argument("Adaptive batch dictionary budget exhausted.");
            charged += metadata + part->size();
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
void validate(const AdaptiveLeaf& leaf)
{
    if (leaf.domain.empty() || std::uint64_t(leaf.provider) > 3 ||
        (leaf.geometry && leaf.surfaces.size() != leaf.geometry->num_faces()) ||
        (!leaf.geometry && (!leaf.surfaces.empty() || !leaf.native_recipe)))
        throw std::invalid_argument("Incomplete adaptive batch leaf.");
    (void) leaf.work.total();
    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
    for (const auto& s : leaf.surfaces)
    {
        if (s.key.empty() || !s.geometry || (s.orientation != 1 && s.orientation != -1) ||
            (!s.exterior && !s.boundary.empty()))
            throw std::invalid_argument("Invalid adaptive batch surface.");
        if (s.require_native_whole_tile && !s.native_whole_geometry)
            throw std::invalid_argument("Missing native whole-face certificate.");
        faces.push_back(s.geometry);
    }
    if (leaf.geometry && !PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces)->equivalent(*leaf.geometry))
        throw std::invalid_argument("Adaptive batch surface shell differs from cell.");
    if (leaf.native_recipe && std::uint64_t(leaf.native_recipe->kind) > 2)
        throw std::invalid_argument("Invalid native batch recipe kind.");
}
void write_leaf(Writer& w, const AdaptiveLeaf& leaf, const Dictionary& dict)
{
    validate(leaf);
    w.bytes(encode_adaptive_key(leaf.key));
    w.bytes(leaf.domain);
    w.word(leaf.material);
    w.word(leaf.revision);
    w.word(std::uint64_t(leaf.provider));
    w.bytes(leaf.geometry ? leaf.geometry->encode_certificate() : AdaptiveBytes{});
    w.word(leaf.surfaces.size());
    for (const auto& s : leaf.surfaces)
    {
        w.bytes(s.key);
        w.bytes(s.geometry->encode_incidence());
        w.bytes(PLIC::encode_identity(s.authority_image));
        w.word(s.require_native_whole_tile);
        w.bytes(s.native_whole_geometry ? s.native_whole_geometry->encode_incidence() : AdaptiveBytes{});
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
struct Decoder
{
    std::vector<Part> parts;
    std::map<AdaptiveBytes, std::shared_ptr<const NativeFragmentDescriptor>> recipes;
    std::map<std::array<std::uint64_t, 3>, std::shared_ptr<const AdaptiveFactorRecipe>> factors;
    Part part(std::uint64_t id) const
    {
        if (id > parts.size())
            throw std::invalid_argument("Adaptive batch part reference is out of range.");
        return id ? parts[id - 1] : nullptr;
    }
    AdaptiveLeaf leaf(Reader& r)
    {
        AdaptiveLeaf leaf;
        leaf.key = decode_adaptive_key(r.bytes());
        leaf.domain = r.bytes();
        leaf.material = r.word();
        leaf.revision = r.word();
        const auto provider = r.word();
        if (provider > 3)
            throw std::invalid_argument("Invalid adaptive batch provider.");
        leaf.provider = AdaptiveProviderKind(provider);
        auto geometry = r.bytes();
        if (!geometry.empty())
            leaf.geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(geometry);
        const auto count = r.word();
        if (count != (leaf.geometry ? leaf.geometry->num_faces() : 0) || count > (r.data.size() - r.offset) / 32)
            throw std::invalid_argument("Invalid adaptive batch surface count.");
        leaf.surfaces.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i)
        {
            AdaptiveSurface s;
            s.key = r.bytes();
            s.geometry = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
            s.authority_image = PLIC::decode_identity<PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>>(r.bytes());
            const auto native_whole = r.word();
            if (native_whole > 1)
                throw std::invalid_argument("Invalid native whole-face declaration.");
            s.require_native_whole_tile = native_whole;
            auto whole_geometry = r.bytes();
            if (!whole_geometry.empty())
                s.native_whole_geometry = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(whole_geometry);
            if (s.require_native_whole_tile && !s.native_whole_geometry)
                throw std::invalid_argument("Missing native whole-face certificate.");
            s.native_face = r.word();
            const auto orientation = r.word(), exterior = r.word();
            if (orientation > 1 || exterior > 1)
                throw std::invalid_argument("Invalid adaptive batch incidence.");
            s.orientation = orientation ? 1 : -1;
            s.exterior = exterior;
            auto name = r.bytes();
            s.boundary.assign(name.begin(), name.end());
            leaf.surfaces.push_back(std::move(s));
        }
        const auto native = r.word();
        if (native > 1)
            throw std::invalid_argument("Invalid adaptive batch native flag.");
        if (native)
        {
            Writer key;
            auto recipe = std::make_shared<NativeFragmentDescriptor>();
            const auto kind = r.word(), n = r.word();
            if (kind > 2 || n > (r.data.size() - r.offset) / 8)
                throw std::invalid_argument("Invalid native batch descriptor.");
            recipe->kind = NativeFragmentDescriptor::Kind(kind);
            key.word(kind);
            key.word(n);
            for (std::uint64_t i = 0; i < n; ++i)
            {
                auto id = r.word();
                auto p = part(id);
                if (!p)
                    throw std::invalid_argument("Null native batch descriptor part.");
                key.word(id);
                recipe->parts.push_back(std::move(p));
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
                recipe->view = *view;
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
            auto result = std::make_shared<AdaptiveFactorRecipe>();
            result->root = part(ids[0]);
            result->base = part(ids[1]);
            result->axis = part(ids[2]);
            leaf.factor_recipe = factors.emplace(ids, std::move(result)).first->second;
        }
        leaf.join_recipe = part(r.word());
        leaf.parent_family = r.bytes();
        leaf.work = {std::bit_cast<double>(r.word()), std::bit_cast<double>(r.word()), std::bit_cast<double>(r.word()),
            std::bit_cast<double>(r.word())};
        validate(leaf);
        return leaf;
    }
};
void family_valid(const AdaptiveFamily& family)
{
    if (family.key.empty() || family.children.empty() || std::uint64_t(family.split) > 2)
        throw std::invalid_argument("Invalid adaptive batch family.");
    std::set<AdaptiveCellKey> unique;
    for (const auto& child : family.children)
        if (!unique.insert(child).second)
            throw std::invalid_argument("Duplicate adaptive batch family child.");
}
} // namespace
AdaptiveBytes encode_adaptive_batch(
    std::span<const AdaptiveLeaf> leaves, std::span<const AdaptiveFamily> families, std::size_t budget)
{
    Dictionary dict{{}, {}, budget};
    for (const auto& leaf : leaves)
        dict.add(leaf);
    for (const auto& family : families)
        dict.add(family.parent);
    Writer w{{}, budget};
    w.word(magic);
    w.word(dict.parts.size());
    for (const auto& part : dict.parts)
        w.bytes(*part);
    w.word(leaves.size());
    w.word(families.size());
    for (const auto& leaf : leaves)
        write_leaf(w, leaf, dict);
    for (const auto& family : families)
    {
        family_valid(family);
        w.bytes(family.key);
        write_leaf(w, family.parent, dict);
        w.word(std::uint64_t(family.split));
        w.word(family.revision);
        w.bytes(family.anchor);
        w.word(family.children.size());
        for (const auto& child : family.children)
            w.bytes(encode_adaptive_key(child));
    }
    if (w.data.size() > budget)
        throw std::invalid_argument("Adaptive batch wire budget exhausted.");
    return std::move(w.data);
}
AdaptiveBatch decode_adaptive_batch(std::span<const char> bytes, std::size_t budget)
{
    if (bytes.size() > budget)
        throw std::invalid_argument("Adaptive batch wire budget exhausted.");
    Reader r{bytes};
    if (r.word() != magic)
        throw std::invalid_argument("Unsupported adaptive batch packet version.");
    Decoder decoder;
    AdaptiveBatch batch;
    auto count = r.word();
    constexpr auto part_metadata = sizeof(Part) + sizeof(AdaptiveBytes) + 6 * sizeof(void*);
    if (count > (r.data.size() - r.offset) / 8 || count > budget / part_metadata)
        throw std::invalid_argument("Invalid adaptive batch dictionary size/budget.");
    std::size_t dictionary_charge = count * part_metadata;
    std::set<Part, PartLess> unique;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        // Inspect the length before Reader allocates the copied payload.
        const auto old_offset = r.offset;
        const auto length = r.word();
        if (length > budget - dictionary_charge)
            throw std::invalid_argument("Adaptive batch dictionary budget exhausted.");
        dictionary_charge += static_cast<std::size_t>(length);
        r.offset = old_offset;
        auto value = r.bytes();
        batch.shared_part_bytes += value.size();
        auto part = std::make_shared<const AdaptiveBytes>(std::move(value));
        if (!unique.insert(part).second)
            throw std::invalid_argument("Duplicate adaptive batch dictionary entry.");
        decoder.parts.push_back(std::move(part));
    }
    const auto leaves = r.word(), families = r.word();
    if (leaves > (r.data.size() - r.offset) / 8 || families > (r.data.size() - r.offset) / 8 ||
        leaves > budget / sizeof(AdaptiveLeaf) || families > budget / sizeof(AdaptiveFamily))
        throw std::invalid_argument("Invalid adaptive batch record count.");
    for (std::uint64_t i = 0; i < leaves; ++i)
        batch.leaves.push_back(decoder.leaf(r));
    for (std::uint64_t i = 0; i < families; ++i)
    {
        AdaptiveFamily family;
        family.key = r.bytes();
        family.parent = decoder.leaf(r);
        const auto split = r.word();
        if (split > 2)
            throw std::invalid_argument("Invalid adaptive batch family split.");
        family.split = AdaptiveSplit(split);
        family.revision = r.word();
        family.anchor = r.bytes();
        const auto n = r.word();
        if (n > (r.data.size() - r.offset) / 8)
            throw std::invalid_argument("Invalid adaptive batch child count.");
        for (std::uint64_t j = 0; j < n; ++j)
            family.children.push_back(decode_adaptive_key(r.bytes()));
        family_valid(family);
        batch.families.push_back(std::move(family));
    }
    r.done();
    return batch;
}
} // namespace SimpleFluid::Meshes::AMR
