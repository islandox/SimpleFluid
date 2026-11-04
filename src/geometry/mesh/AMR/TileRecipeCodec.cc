#include "TileRecipeCodec.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
void member(Writer& w, const RegistryMember& m)
{
    w.bytes(encode_adaptive_key(m.cell));
    w.word(m.revision);
    w.word(m.surface);
    w.word(m.owner);
}
RegistryMember member(Reader& r, int ranks)
{
    RegistryMember result;
    result.cell = decode_adaptive_key(r.bytes_view());
    result.revision = r.word();
    result.surface = r.word();
    const auto owner = r.word();
    if (owner >= std::uint64_t(ranks))
        throw std::invalid_argument("Invalid tile recipe owner.");
    result.owner = int(owner);
    return result;
}
} // namespace
std::size_t tile_recipe_encoded_size(std::span<const char> support, std::span<const char> domain,
    std::span<const RegistryMember> members, const PrivateTileRecipe& recipe)
{
    if (!recipe.face || recipe.first_member >= members.size() ||
        (recipe.second_member && *recipe.second_member >= members.size()))
        throw std::invalid_argument("Invalid private tile recipe.");
    std::size_t size = 88;
    auto add = [&](std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() - size)
            throw std::invalid_argument("Tile recipe encoded size overflow.");
        size += n;
    };
    add(support.size());
    add(domain.size());
    add(encode_adaptive_key(members[recipe.first_member].cell).size());
    if (recipe.second_member)
    {
        add(32);
        add(encode_adaptive_key(members[*recipe.second_member].cell).size());
    }
    add(PLIC::encoded_identity_size(recipe.second_image));
    add(recipe.face->encoded_incidence_size());
    return size;
}
AdaptiveBytes encode_tile_recipe(std::span<const char> support, std::span<const char> domain,
    std::span<const RegistryMember> members, const PrivateTileRecipe& recipe, std::uint64_t emission)
{
    if (!recipe.face || recipe.first_member >= members.size() ||
        (recipe.second_member && *recipe.second_member >= members.size()))
        throw std::invalid_argument("Invalid private tile recipe.");
    Writer w;
    const auto bytes = tile_recipe_encoded_size(support, domain, members, recipe);
    w.data.reserve(bytes);
    if (w.data.capacity() != bytes)
        throw std::invalid_argument("Tile recipe encoding capacity differs from admission.");
    w.word(1);
    w.bytes(support);
    w.bytes(domain);
    w.word(emission);
    member(w, members[recipe.first_member]);
    w.word(recipe.second_member.has_value());
    if (recipe.second_member)
        member(w, members[*recipe.second_member]);
    w.bytes(PLIC::encode_identity(recipe.second_image));
    w.bytes(recipe.face->encode_incidence());
    if (w.data.size() != bytes)
        throw std::invalid_argument("Tile recipe encoding length disagrees with admission.");
    return std::move(w.data);
}
ReceivedTileRecipe decode_tile_recipe(std::span<const char> packet, int source, int resident, int ranks,
    std::size_t budget)
{
    if (packet.size() > budget)
        throw std::invalid_argument("Tile recipe packet exceeds budget.");
    Reader r{packet};
    if (r.word() != 1)
        throw std::invalid_argument("Unsupported tile recipe version.");
    ReceivedTileRecipe result;
    result.support = r.bytes();
    result.domain = r.bytes();
    result.emission = r.word();
    result.first = member(r, ranks);
    const auto has_second = r.word();
    if (has_second > 1)
        throw std::invalid_argument("Invalid tile recipe incidence.");
    if (has_second)
        result.second = member(r, ranks);
    result.second_image = PLIC::decode_identity<PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>>(r.bytes_view(), budget);
    result.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes_view(), budget);
    r.done();
    if (result.support.empty() || result.domain.empty() || source != authority(result.support, ranks) ||
        (result.first.owner != resident && (!result.second || result.second->owner != resident)) ||
        (result.second && !(result.first.cell < result.second->cell)))
        throw std::invalid_argument("Foreign tile recipe.");
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
