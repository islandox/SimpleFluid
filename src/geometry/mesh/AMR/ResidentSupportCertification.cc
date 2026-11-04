#include "ResidentSupportCertification.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
void certify_resident_support(std::span<const char> support, std::span<const RegistryMember> members,
    std::span<const ReceivedTileRecipe> tiles, int resident, const OwnedLeafLookup& lookup)
{
    std::map<AdaptiveCellKey, std::vector<SupportFace>> coverage;
    std::set<AdaptiveBytes> identities;
    const AdaptiveBytes* domain = nullptr;
    for (const auto& tile : tiles)
    {
        if (!std::ranges::equal(tile.support, support) || tile.domain.empty() || (domain && *domain != tile.domain))
            throw std::invalid_argument("Foreign resident support tile.");
        domain = &tile.domain;
        Writer identity;
        identity.bytes(tile.support);
        auto forward = PLIC::canonical_face_key(*tile.face);
        auto reverse = PLIC::canonical_face_key(*tile.face->reversed());
        identity.bytes(std::min(forward, reverse));
        if (!identities.insert(std::move(identity.data)).second)
            throw std::invalid_argument("Duplicate resident authority tile.");
        if (tile.first.owner == resident)
            coverage[tile.first.cell].push_back(tile.face);
        if (tile.second && tile.second->owner == resident)
            coverage[tile.second->cell].push_back(tile.face->reversed()->translated(
                PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image)));
    }
    std::set<AdaptiveCellKey> unique;
    for (const auto& member : members)
    {
        if (member.owner != resident || !unique.insert(member.cell).second)
            throw std::invalid_argument("Invalid resident support membership.");
        const auto& leaf = lookup(member.cell);
        if (leaf.key != member.cell || leaf.revision != member.revision || (domain && leaf.domain != *domain) ||
            member.surface >= leaf.surfaces.size() ||
            !std::ranges::equal(leaf.surfaces[member.surface].key, support))
            throw std::invalid_argument("Stale resident support membership.");
        const auto& surface = leaf.surfaces[member.surface];
        auto found = coverage.find(member.cell);
        if (found == coverage.end())
            throw std::invalid_argument("Missing owned authority support.");
        for (const auto& face : found->second)
            if (surface.geometry->oppositely_oriented(*face))
                throw std::invalid_argument("Authority face has inward orientation.");
        surface.geometry->certify_partition(found->second);
        coverage.erase(found);
    }
    if (!coverage.empty())
        throw std::invalid_argument("Unrequested authority support.");
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
