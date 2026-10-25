/** @file DistributedAdaptiveTypes.cc
 * @brief Versioned bounded lossless ancestry and owner-record encoding.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
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
struct Reader
{
    std::span<const char> data;
    std::size_t position = 0;
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
void base(Writer& w, const BaseCellKey& key)
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
void axial(Writer& w, const AxialIntervalKey& key)
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
void header(Writer& w, std::uint64_t kind)
{
    w.word(0x5346414450543631ULL);
    w.word(1);
    w.word(kind);
}
void header(Reader& r, std::uint64_t kind)
{
    if (r.word() != 0x5346414450543631ULL || r.word() != 1 || r.word() != kind)
        throw std::invalid_argument("Invalid adaptive packet version/type.");
}
} // namespace
AdaptiveBytes encode_adaptive_key(const AdaptiveCellKey& key)
{
    Writer w;
    header(w, 0);
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
    return std::move(w.data);
}
AdaptiveCellKey decode_adaptive_key(std::span<const char> packet)
{
    Reader r{packet};
    header(r, 0);
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
AdaptiveBytes encode_adaptive_leaf(const AdaptiveLeaf& leaf)
{
    if (leaf.domain.empty())
        throw std::invalid_argument("Adaptive connected-domain provenance is required.");
    if ((leaf.geometry && leaf.surfaces.size() != leaf.geometry->num_faces()) ||
        (!leaf.geometry && (!leaf.surfaces.empty() || !leaf.native_recipe)))
        throw std::invalid_argument("Incomplete adaptive leaf representation.");
    (void) leaf.work.total();
    Writer w;
    header(w, 1);
    w.bytes(encode_adaptive_key(leaf.key));
    w.bytes(leaf.domain);
    w.word(leaf.material);
    w.word(leaf.revision);
    w.word(std::uint64_t(leaf.provider));
    w.bytes(leaf.geometry ? leaf.geometry->encode_certificate() : AdaptiveBytes{});
    w.word(leaf.surfaces.size());
    for (const auto& s : leaf.surfaces)
    {
        if (s.key.empty() || !s.geometry || (s.orientation != 1 && s.orientation != -1) ||
            (!s.exterior && !s.boundary.empty()))
            throw std::invalid_argument("Invalid adaptive surface.");
        w.bytes(s.key);
        w.bytes(s.geometry->encode());
        w.word(s.orientation == 1);
        w.word(s.exterior);
        w.bytes(std::span(s.boundary));
        w.word(s.native_face);
        w.bytes(s.authority_image);
        w.word(s.require_native_whole_tile);
        w.word(bool(s.native_whole_geometry));
        if (s.native_whole_geometry)
            w.bytes(s.native_whole_geometry->encode_incidence());
    }
    w.bytes(leaf.native_recipe ? encode_native_fragment_descriptor(*leaf.native_recipe) : AdaptiveBytes{});
    w.word(leaf.native_cell);
    for (const auto& part :
        leaf.factor_recipe ? std::array{leaf.factor_recipe->root, leaf.factor_recipe->base, leaf.factor_recipe->axis}
                           : std::array<std::shared_ptr<const AdaptiveBytes>, 3>{})
        w.bytes(part ? *part : AdaptiveBytes{});
    w.bytes(leaf.join_recipe ? *leaf.join_recipe : AdaptiveBytes{});
    w.bytes(leaf.parent_family);
    for (double n : {leaf.work.reconstruction, leaf.work.interface, leaf.work.population, leaf.work.migration})
        w.word(std::bit_cast<std::uint64_t>(n));
    return std::move(w.data);
}
AdaptiveLeaf decode_adaptive_leaf(std::span<const char> packet)
{
    Reader r{packet};
    header(r, 1);
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
        s.authority_image = r.bytes();
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
    header(w, 2);
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
    header(r, 2);
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
} // namespace SimpleFluid::Meshes::AMR
