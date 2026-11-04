#include "SupportSurfaceCodec.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using Displacement = PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>;
std::size_t support_surface_encoded_size(const AdaptiveLeaf& leaf, std::size_t ordinal)
{
    if (ordinal >= leaf.surfaces.size())
        throw std::invalid_argument("Unknown support surface ordinal.");
    const auto& surface = leaf.surfaces[ordinal];
    if (!surface.geometry)
        throw std::invalid_argument("Missing support geometry.");
    const bool shared_whole = surface.require_native_whole_tile && surface.native_whole_geometry &&
                              surface.native_whole_geometry == surface.geometry;
    const auto whole = surface.native_whole_geometry && !shared_whole
                           ? surface.native_whole_geometry->encoded_incidence_size()
                           : 0;
    std::size_t size = 88 + geometry_record_header_bytes;
    const auto add = [&](std::size_t value)
    {
        if (value > std::numeric_limits<std::size_t>::max() - size)
            throw std::invalid_argument("Support surface encoded size overflow.");
        size += value;
    };
    add(surface.key.size());
    add(leaf.domain.size());
    add(encode_adaptive_key(leaf.key).size());
    add(PLIC::encoded_identity_size(surface.authority_image));
    add(whole);
    add(surface.boundary.size());
    add(surface.geometry->encoded_incidence_size());
    return size;
}
AdaptiveBytes encode_support_surface(const AdaptiveLeaf& leaf, std::size_t ordinal)
{
    if (ordinal >= leaf.surfaces.size())
        throw std::invalid_argument("Unknown support surface ordinal.");
    const auto& surface = leaf.surfaces[ordinal];
    if (!surface.geometry || surface.key.empty() || (surface.orientation != 1 && surface.orientation != -1))
        throw std::invalid_argument("Incomplete support authority input.");
    if (!surface.authority_image.is_zero() &&
        (leaf.provider != AdaptiveProviderKind::Cartesian || surface.exterior ||
            PLIC::displacement_difference<ArithmeticMode::Ordinary>(surface.authority_image, {}) != surface.authority_image))
        throw std::invalid_argument("Periodic authority images require canonical Cartesian interior provenance.");
    auto face = surface.geometry->encode_incidence();
    auto key = encode_adaptive_key(leaf.key);
    const bool shared_whole = surface.require_native_whole_tile && surface.native_whole_geometry &&
                              surface.native_whole_geometry == surface.geometry;
    const auto whole = surface.native_whole_geometry && !shared_whole
                           ? surface.native_whole_geometry->encode_incidence()
                           : AdaptiveBytes{};
    if (surface.require_native_whole_tile && !shared_whole && whole.empty())
        throw std::invalid_argument("Missing actual native face nesting certificate.");
    Writer w;
    write_geometry_record_header(w, GeometryRecordKind::SurfaceRequest);
    w.bytes(surface.key);
    w.bytes(leaf.domain);
    w.bytes(key);
    w.word(surface.native_face);
    w.bytes(PLIC::encode_identity(surface.authority_image));
    w.word(shared_whole ? 2 : std::uint64_t(surface.require_native_whole_tile));
    w.bytes(whole);
    w.word(surface.orientation == 1);
    w.word(surface.exterior);
    w.bytes(std::span(surface.boundary));
    w.bytes(face);
    return std::move(w.data);
}
SupportOccurrence decode_support_surface(std::span<const char> packet, std::span<const char> expected_support,
    int owner, std::size_t byte_budget)
{
    if (packet.size() > byte_budget || owner < 0)
        throw std::invalid_argument("Invalid support surface packet.");
    Reader r{packet};
    read_geometry_record_header(r, GeometryRecordKind::SurfaceRequest);
    const auto support = r.bytes_view();
    if (!std::ranges::equal(support, expected_support))
        throw std::invalid_argument("Foreign support surface packet.");
    SupportOccurrence entry;
    entry.domain = r.bytes();
    entry.cell = decode_adaptive_key(r.bytes_view());
    entry.native_face = r.word();
    entry.image = PLIC::decode_identity<Displacement>(r.bytes_view(), byte_budget);
    const auto nesting = r.word();
    if (nesting > 2)
        throw std::invalid_argument("Invalid native face nesting policy.");
    entry.native_whole = nesting;
    const auto whole = r.bytes_view();
    if (nesting == 2 && !whole.empty())
        throw std::invalid_argument("Shared native whole-face proof has a duplicate payload.");
    if (!whole.empty())
        entry.whole_geometry = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(whole, byte_budget)->translated(entry.image);
    entry.rank = owner;
    const auto orientation = r.word(), exterior = r.word();
    if (orientation > 1 || exterior > 1)
        throw std::invalid_argument("Invalid authority support routing.");
    entry.orientation = orientation ? 1 : -1;
    entry.exterior = exterior;
    const auto boundary = r.bytes_view();
    entry.boundary.assign(boundary.begin(), boundary.end());
    entry.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes_view(), byte_budget)->translated(entry.image);
    if (nesting == 2)
        entry.whole_geometry = entry.face;
    if (entry.native_whole && !entry.whole_geometry)
        throw std::invalid_argument("Missing native whole-face authority proof.");
    if (entry.whole_geometry && !entry.whole_geometry->contains(*entry.face))
        throw std::invalid_argument("Rewritten join lies outside its actual native face.");
    r.done();
    return entry;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
