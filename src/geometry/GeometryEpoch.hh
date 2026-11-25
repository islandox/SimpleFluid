/**
 * @file GeometryEpoch.hh
 * @brief Uniform access to optional fixed-topology geometry revisions.
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Value-geometry revision state with a non-copying motion lease.
 *
 * Copying a concrete mesh copies its geometry revision but never aliases an
 * active controller lease into the independent value. Assigning a complete
 * mesh value likewise clears any lease; doing so behind an existing handle is
 * outside the controlled motion contract and is detected by that controller.
 */
struct GeometryEpochState
{
    std::uint64_t epoch = 0;
    const void* motion_owner = nullptr;

    // A transaction can restore an exact earlier geometry and its cache key.
    // The next replacement must still supersede every previously issued key.
    std::uint64_t latest_epoch() const noexcept { return std::max(epoch, d_latest_epoch); }
    void advance_epoch() noexcept { epoch = d_latest_epoch = latest_epoch() + 1; }
    void restore_epoch(std::uint64_t accepted) noexcept
    {
        d_latest_epoch = latest_epoch();
        epoch = accepted;
    }

    GeometryEpochState() = default;

    GeometryEpochState(const GeometryEpochState& other) noexcept : epoch(other.epoch) {}

    GeometryEpochState& operator=(const GeometryEpochState& other)
    {
        if (this == &other) return *this;
        const auto latest = std::max(latest_epoch(), other.epoch);
        if (latest == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Geometry revision overflow during replacement.");
        epoch = d_latest_epoch = latest + 1;
        motion_owner = nullptr;
        return *this;
    }

    GeometryEpochState(GeometryEpochState&& other) : epoch(other.epoch)
    {
        if (other.latest_epoch() == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Geometry revision overflow during move.");
        other.advance_epoch();
    }

    GeometryEpochState& operator=(GeometryEpochState&& other)
    {
        if (this == &other) return *this;
        const auto latest = std::max(latest_epoch(), other.latest_epoch());
        if (latest == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Geometry revision overflow during replacement.");
        epoch = d_latest_epoch = latest + 1;
        other.advance_epoch();
        motion_owner = nullptr;
        return *this;
    }

private:
    std::uint64_t d_latest_epoch = 0;
};

/** @brief Mesh interfaces may expose an exact geometry cache key. */
template<class MeshType>
concept GeometryEpochMesh = requires(const MeshType& mesh) {
    { mesh.geometry_epoch() } -> std::convertible_to<std::uint64_t>;
};

/**
 * @brief Return the current geometry epoch, or zero for immutable meshes.
 *
 * Mesh interfaces without a motion contract remain at epoch zero. Runtime
 * MeshHandle geometry advances explicitly.
 */
template<class MeshType> constexpr std::uint64_t mesh_geometry_epoch(const MeshType& mesh)
{
    if constexpr (GeometryEpochMesh<MeshType>)
    {
        return static_cast<std::uint64_t>(mesh.geometry_epoch());
    }
    else
    {
        static_cast<void>(mesh);
        return 0;
    }
}

} // namespace SimpleFluid
