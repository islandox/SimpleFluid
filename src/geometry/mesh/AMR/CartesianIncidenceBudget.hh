/** @file CartesianIncidenceBudget.hh
 * @brief Checked allocation and codec preflights for Cartesian incidence preparation.
 */
#pragma once
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
namespace SimpleFluid::Meshes::AMR::adaptive_detail::cartesian_detail
{
inline std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Cartesian incidence capacity overflow.");
    return a + b;
}
inline std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Cartesian incidence capacity overflow.");
    return a * b;
}
inline void check_keeper(Budget& budget, const std::shared_ptr<Budget>& keeper)
{
    if (keeper && keeper.get() != &budget)
        throw std::invalid_argument("Cartesian incidence keeper belongs to a different budget.");
}
/** Every addition precedes its allocation; old/new vector capacities overlap.
 * The caller destroys scoped buffers before this lease. */
struct Allocation
{
    Budget& budget;
    std::size_t bytes = 0;
    explicit Allocation(Budget& b) : budget(b) {}
    ~Allocation() { budget.used -= bytes; }
    void retain(std::size_t n)
    {
        budget.add(n);
        bytes = add(bytes, n);
    }
    template<class T> void reserve(std::vector<T>& v, std::size_t n)
    {
        if (n <= v.capacity())
            return;
        if (!PLIC::geometry_encoding_capacity_profile_supported())
            throw std::invalid_argument("Unqualified Cartesian vector reservation profile.");
        const auto old = product(v.capacity(), sizeof(T));
        retain(product(n, sizeof(T)));
        v.reserve(n);
        if (v.capacity() != n)
            throw std::invalid_argument("Cartesian incidence capacity differs from admission.");
        budget.used -= old;
        bytes -= old;
    }
    template<class T> void append(std::vector<T>& v)
    {
        if (v.size() == v.capacity())
            reserve(v, v.capacity() ? product(v.capacity(), 2) : 1);
    }
    void copy(AdaptiveBytes& target, std::span<const char> input)
    {
        reserve(target, input.size());
        target.assign(input.begin(), input.end());
    }
};
/** Sum the complete push_back growth history for the qualified std::vector
 * profile, including every old/new overlap. */
inline std::size_t growth(std::size_t count, std::size_t width)
{
    if (!PLIC::geometry_encoding_capacity_profile_supported())
        throw std::invalid_argument("Unqualified Cartesian incidence vector capacity profile.");
    std::size_t capacity = 0, total = 0;
    while (capacity < count)
    {
        capacity = capacity ? product(capacity, 2) : 1;
        total = add(total, product(capacity, width));
    }
    return total;
}
/** For the qualified libstdc++/libc++ insertion profile, each growth doubles
 * capacity or jumps directly to the required size. Summed histories are <4N. */
inline std::size_t insertion_growth(std::size_t count)
{
    if (!PLIC::geometry_encoding_capacity_profile_supported())
        throw std::invalid_argument("Unqualified Cartesian incidence codec capacity profile.");
    return product(count, 4);
}
template<class Identity> AdaptiveBytes identity_key(const Identity& identity, Allocation& allocation)
{
    const auto size = PLIC::canonical_key_size(identity);
    allocation.retain(size);
    auto result = PLIC::canonical_key(identity);
    if (result.capacity() != size)
        throw std::invalid_argument("Cartesian identity encoding capacity differs from preflight.");
    return result;
}
inline AdaptiveBytes face_key(const SupportFace& face, Allocation& allocation)
{
    const auto scratch = face->encoding_scratch_bytes();
    allocation.retain(scratch);
    const auto encoded = face->encoded_size(scratch);
    const auto body = encoded - PLIC::geometry_packet_header_bytes;
    allocation.retain(add(insertion_growth(body), scratch));
    auto result = PLIC::canonical_face_key(*face);
    if (result.capacity() > insertion_growth(body))
        throw std::invalid_argument("Cartesian face key exceeded preflight.");
    return result;
}
inline AdaptiveBytes surface_packet(const AdaptiveLeaf& leaf, std::size_t ordinal, Allocation& allocation)
{
    if (ordinal >= leaf.surfaces.size())
        throw std::invalid_argument("Unknown Cartesian support ordinal.");
    const auto key_bytes = add(key_capacity(leaf.key), 128);
    // key_capacity includes arbitrary explicit ancestry payload; canonical
    // header/fixed words add at most 128 bytes. Both measuring and encoding
    // invoke the existing adaptive-key writer.
    allocation.retain(insertion_growth(key_bytes));
    const auto size = support_surface_encoded_size(leaf, ordinal);
    const auto& face = leaf.surfaces[ordinal];
    auto bound = insertion_growth(size);
    bound = add(bound, insertion_growth(key_bytes));
    bound = add(bound, product(face.geometry->encoded_incidence_size(), 5));
    bound = add(bound, product(PLIC::encoded_identity_size(face.authority_image), 2));
    if (face.native_whole_geometry && face.native_whole_geometry != face.geometry)
        bound = add(bound, product(face.native_whole_geometry->encoded_incidence_size(), 5));
    allocation.retain(bound);
    auto result = encode_support_surface(leaf, ordinal);
    if (result.size() != size || result.capacity() > insertion_growth(size))
        throw std::invalid_argument("Cartesian support packet exceeded codec preflight.");
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail::cartesian_detail
