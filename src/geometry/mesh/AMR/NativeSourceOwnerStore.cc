#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include <mutex>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
// All detached halo/native scalar updates share this allocation-free lock.
// Mutable hierarchy operations remain serialized by their existing cache token.
std::mutex native_halo_mirror_mutex;
template<bool Publication> bool precharge_halo(void* raw, std::size_t bytes) noexcept
{
    const std::lock_guard lock(native_halo_mirror_mutex);
    auto& budget = *static_cast<Budget*>(raw);
    if (budget.used > budget.limit || bytes > budget.limit - budget.used ||
        bytes > std::numeric_limits<std::size_t>::max() - budget.cumulative)
        return false;
    budget.used += bytes;
    if constexpr (Publication) budget.publication_mirror_live += bytes;
    budget.peak = std::max(budget.peak, budget.used);
    budget.cumulative += bytes;
    return true;
}
template<bool Publication> void retire_halo(void* raw, std::size_t bytes) noexcept
{
    const std::lock_guard lock(native_halo_mirror_mutex);
    auto& budget = *static_cast<Budget*>(raw);
    if (bytes > budget.used) std::terminate();
    if constexpr (Publication)
    {
        if (bytes > budget.publication_mirror_live) std::terminate();
        budget.publication_mirror_live -= bytes;
    }
    budget.used -= bytes;
}
}
geometry_detail::NativeAllocationMirror NativeSourceOwnerStore::mirror(const std::shared_ptr<Budget>& arena)
{
    return make_mirror(arena, false);
}
std::size_t NativeSourceOwnerStore::publication_mirror_bytes(const std::shared_ptr<Budget>& arena)
{
    const std::lock_guard lock(native_halo_mirror_mutex);
    if (!arena || arena->publication_mirror_live > arena->used)
        throw std::invalid_argument("Native publication mirror has an invalid live request total.");
    return arena->publication_mirror_live;
}
geometry_detail::NativeAllocationMirror NativeSourceOwnerStore::make_mirror(
    const std::shared_ptr<Budget>& arena, bool publication)
{
    if (!arena)
        throw std::invalid_argument("Native halo mirror requires its actual owned arena.");
    geometry_detail::NativeAllocationMirror result;
    result.owner = arena;
    result.owner_proof = geometry_detail::NativeMirrorOwnerProof(result.owner, sizeof(Budget) + 4 * sizeof(void*),
        geometry_detail::NativeMirrorExtentKind::ProfiledUpper, publication);
    result.try_precharge = publication ? precharge_halo<true> : precharge_halo<false>;
    result.retire_callback = publication ? retire_halo<true> : retire_halo<false>;
    return result;
}
PLIC::details::OwnerManifestPair NativeSourceOwnerStore::single(
    PLIC::details::geometry_demand::GeometryOwnerManifest ordinary)
{
    return PLIC::details::OwnerManifestPair::sealed({std::move(ordinary), {}}, 0);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
