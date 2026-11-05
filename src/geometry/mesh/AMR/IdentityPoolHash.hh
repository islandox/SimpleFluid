#pragma once
#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Lookup policies never determine identity or numerical ordering. Unlisted
 * value types retain the ordered full-value lookup without hashing. */
struct NoIdentityPoolHash
{
    static constexpr bool enabled = false;
};
template<class T> struct IdentityPoolHasher : NoIdentityPoolHash
{
};

template<> struct IdentityPoolHasher<AdaptiveCellKey>
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const AdaptiveCellKey& key) const noexcept
    {
        std::uint64_t hash = 14695981039346656037ULL;
        const auto word = [&](std::uint64_t value)
        {
            for (unsigned i = 0; i < 8; ++i)
            {
                hash ^= static_cast<unsigned char>(value >> (8 * i));
                hash *= 1099511628211ULL;
            }
        };
        word(key.region);
        word(key.ancestry.index());
        std::visit(
            [&](const auto& ancestry)
            {
                using A = std::decay_t<decltype(ancestry)>;
                if constexpr (std::is_same_v<A, CartesianCellAncestry>)
                {
                    word(ancestry.level);
                    for (const auto value : ancestry.index)
                        word(value);
                }
                else if constexpr (std::is_same_v<A, ConvexCellAncestry>)
                {
                    word(ancestry.root);
                    word(ancestry.path.size());
                    for (const auto value : ancestry.path)
                        word(value);
                }
                else if constexpr (std::is_same_v<A, ProductCellAncestry>)
                {
                    word(ancestry.base.root);
                    word(ancestry.base.root_template);
                    word(ancestry.base.path.size());
                    for (const auto value : ancestry.base.path)
                        word(value);
                    word(ancestry.axial.root);
                    word(ancestry.axial.path.size());
                    for (const auto value : ancestry.axial.path)
                        word(value);
                }
                else
                {
                    word(ancestry.provider_root);
                    word(ancestry.native_cell);
                }
            },
            key.ancestry);
        return hash;
    }
};
template<> struct IdentityPoolHasher<AdaptiveBytes>
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const AdaptiveBytes& bytes) const noexcept { return adaptive_route_hash(bytes); }
    template<class View>
        requires requires(const View& view, std::size_t i) {
            view.size();
            static_cast<unsigned char>(view[i]);
        }
    std::uint64_t operator()(const View& bytes) const noexcept
    {
        std::uint64_t value = 14695981039346656037ULL;
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            value ^= static_cast<unsigned char>(bytes[i]);
            value *= 1099511628211ULL;
        }
        return value;
    }
};
template<> struct IdentityPoolHasher<PLIC::PointIdentity<ArithmeticMode::Ordinary>>
{
    static constexpr bool enabled = true;
    std::uint64_t operator()(const PLIC::PointIdentity<ArithmeticMode::Ordinary>& point) const
    {
        // Canonical scalar emission uses floating-to-integer conversion. Keep
        // this lookup-only work from changing the caller's flags or trap mask.
        std::fenv_t saved;
        if (std::feholdexcept(&saved) != 0)
            throw std::runtime_error("Cannot isolate point lookup floating-point environment.");
        const auto hash = PLIC::canonical_key_hash(point);
        if (std::fesetenv(&saved) != 0)
            throw std::runtime_error("Cannot restore point lookup floating-point environment.");
        return hash;
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
