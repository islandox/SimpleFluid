/** @file ExactFaceTransfer.hh
 * @brief Fixed-clock conservative transfer of accepted exact canonical face rates.
 */
#pragma once
#include "FVM/ExactPhaseTransfer.hh"
namespace SimpleFluid::VoF
{
class GeneralVoF;
struct ExactOwnedFaceRate
{
    std::uint64_t face = 0;
    Meshes::ExactGeometryMeasure rate;
};
struct ExactOwnedCellVelocity
{
    Meshes::AMR::AdaptiveCellKey key;
    Meshes::GeometryPoint velocity{};
};
struct ExactFaceTransferOptions
{
    /** Newly admitted records, live packet/decoder arenas, nested hierarchy
     * service leases and exact arithmetic workspaces. Metadata accounting is
     * conservative and may reject before every byte is used. Existing accepted
     * model storage and backend/allocator internals are not an RSS claim. */
    std::size_t byte_budget = 64 * 1024 * 1024;
    std::size_t maximum_common_cells = 65536, maximum_patches = 262144, maximum_pair_tests = 1048576;
};
class SIMPLEFLUID_FVM_EXPORT ExactFaceTransferResult
{
public:
    struct Data;
    [[nodiscard]] const std::vector<ExactOwnedFaceRate>& owned_rates() const noexcept;
    /** Validate actual accepted source and adopted target publications/maps. */
    void validate_observations(const GeneralVoF& source, const GeneralVoF& target) const;

private:
    friend class ExactFaceTransfer;
    explicit ExactFaceTransferResult(std::shared_ptr<const Data>);
    std::shared_ptr<const Data> data_;
};
class SIMPLEFLUID_FVM_EXPORT ExactFaceTransfer
{
public:
    /** Collective. Rates cover the actual source owned-face map once, including
     * exact-zero physical exterior rates. Seed velocities cover source owners
     * for topology changes; ownership-only transfer requires no velocity input.
     * Only NEW common-refinement internal rates are extended to close local
     * divergence. Existing boundary rates and cell momentum/pressure histories
     * are never changed. ProductIntersection is an ordinary common-refinement
     * relation, not a nested refinement shortcut. No physical projection occurs.
     */
    [[nodiscard]] static ExactFaceTransferResult prepare(const GeneralVoF& source, const GeneralVoF& adopted_target,
        const ExactPhaseTransferResult&, std::span<const ExactOwnedFaceRate>,
        std::span<const ExactOwnedCellVelocity> seed_velocity = {}, ExactFaceTransferOptions = {});
};
} // namespace SimpleFluid::VoF
