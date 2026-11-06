#include "PrivateTileRecipe.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <limits>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
void make_support_recipes(std::span<const char> support, std::span<const SupportOccurrence> entries,
    std::size_t byte_budget, std::size_t tile_budget, const SupportCharge& charge, const SupportChargeFace& charge_face,
    const RecipeEmit& publish, const SupportConstructionAdmission* construction)
{
    using Displacement = PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>;
    std::set<AdaptiveBytes> identities;
    std::size_t tiles = 0;
    auto emit = [&](std::size_t first, std::optional<std::size_t> second, SupportFace face)
    {
        if (tiles >= tile_budget)
            throw std::invalid_argument("Authority tile budget exhausted.");
        ++tiles;
        // Count before constructing the duplicate-identity audit and its
        // normalized/reversed keys. The native key emitter may grow a
        // vector, so include old/new capacity overlap for both encodings.
        const auto encoded = face->encoded_incidence_size();
        const auto geometry = SupportFace::element_type::construction_capacity_bound(face->points().size());
        const auto maximum = std::numeric_limits<std::size_t>::max();
        if (support.size() > maximum - 16 || encoded > maximum - 16 - support.size())
            throw std::invalid_argument("Authority face identity capacity overflow.");
        const auto identity_capacity = 16 + support.size() + encoded;
        if (identity_capacity > maximum - sizeof(AdaptiveBytes) - 4 * sizeof(void*))
            throw std::invalid_argument("Authority identity owner capacity overflow.");
        charge(identity_capacity + sizeof(AdaptiveBytes) + 4 * sizeof(void*));
        if (encoded > maximum / 6 || geometry > maximum / 3)
            throw std::invalid_argument("Authority face audit scratch overflow.");
        charge(6 * encoded);
        charge(3 * geometry);
        charge(face->encoding_scratch_bytes());
        auto first_image = entries[first].image;
        auto second_image = second ? entries[*second].image : Displacement{};
        if (second && entries[*second].cell < entries[first].cell)
        {
            std::swap(first, *second);
            std::swap(first_image, second_image);
            face = face->reversed();
        }
        Writer identity;
        identity.data.reserve(identity_capacity);
        if (identity.data.capacity() != identity_capacity)
            throw std::invalid_argument("Authority identity capacity differs from admission.");
        identity.bytes(support);
        auto forward = PLIC::canonical_face_key(*face);
        auto reverse = PLIC::canonical_face_key(*face->reversed());
        identity.bytes(std::min(forward, reverse));
        if (!identities.insert(std::move(identity.data)).second)
            throw std::invalid_argument("Duplicate canonical authority face identity.");
        auto difference = second ? PLIC::displacement_difference<ArithmeticMode::Ordinary>(second_image, first_image)
                                 : Displacement{};
        auto physical = face->translated(PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, first_image));
        publish({first, second, std::move(physical), std::move(difference)});
    };
    overlay_complete_support(entries, byte_budget, emit, charge, charge_face, construction);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
