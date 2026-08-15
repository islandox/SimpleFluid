/**
 * @file WallYPlusStatistics.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Rank-consistent wall-y-plus statistics for mesh adaptation.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/typedefs.hh"

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace SimpleFluid
{

/**
 * @brief Statistic used to control a boundary-layer first-cell height.
 *
 * `Maximum` is the conservative choice for wall-resolved SST: refinement
 * continues until every sampled wall face meets the configured target.
 * `AreaWeightedMean` is useful for patch-average wall-function comparisons.
 */
enum class WallYPlusControlStatistic : uint8_t
{
    Maximum,
    AreaWeightedMean
};

/** @brief One locally owned wall-face y-plus sample. */
struct WallYPlusSample
{
    real_t y_plus = {};
    real_t face_area = 1.0;
};

/**
 * @brief Globally reduced y-plus summary for one named boundary patch.
 *
 * Samples must be contributed only by the rank owning each boundary face.
 * Ranks with no local face on the patch contribute an empty sample range.
 */
struct WallYPlusStatistics
{
    std::string boundary_name;
    size_t global_face_count = 0;
    real_t total_face_area = {};
    real_t minimum = std::numeric_limits<real_t>::quiet_NaN();
    real_t maximum = std::numeric_limits<real_t>::quiet_NaN();
    real_t area_weighted_mean =
        std::numeric_limits<real_t>::quiet_NaN();

    bool empty() const noexcept
    {
        return global_face_count == 0;
    }

    /**
     * @brief Select the feedback measurement for a controller.
     * @throws std::logic_error If the patch has no global samples.
     */
    real_t measurement(WallYPlusControlStatistic statistic) const
    {
        if (empty())
        {
            throw std::logic_error(
                "Wall y-plus patch has no global samples: "
                + boundary_name + ".");
        }
        switch (statistic)
        {
            case WallYPlusControlStatistic::Maximum:
                return maximum;
            case WallYPlusControlStatistic::AreaWeightedMean:
                return area_weighted_mean;
        }
        throw std::logic_error("Unknown wall y-plus control statistic.");
    }
};

using WallYPlusSamplesByPatch =
    std::unordered_map<std::string, Arr<WallYPlusSample>>;

namespace wall_y_plus_detail
{

inline long long stable_patch_hash(const std::string& name) noexcept
{
    uint64_t hash = 1469598103934665603ULL;
    for (const auto character : name)
    {
        hash ^= static_cast<unsigned char>(character);
        hash *= 1099511628211ULL;
    }
    return static_cast<long long>(
        hash & static_cast<uint64_t>(
                   std::numeric_limits<long long>::max()));
}

template<class Ordinal>
void require_collectively_valid(
    const Teuchos::Comm<Ordinal>& communicator,
    bool local_valid,
    const std::string& message)
{
    const int local_failed = local_valid ? 0 : 1;
    int any_failed = 0;
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MAX, 1,
        &local_failed, &any_failed);
    if (any_failed != 0)
    {
        throw std::invalid_argument(message);
    }
}

} // namespace wall_y_plus_detail

/**
 * @brief Reduce locally owned wall-face samples into one global patch summary.
 *
 * Every rank in @p communicator must call this function in the same order and
 * pass the same @p boundary_name. A rank without a local piece of the patch
 * passes an empty sample range.
 *
 * @throws std::invalid_argument Collectively on all ranks if the patch names
 *         differ, a name is empty, or a sample is negative/non-finite.
 */
template<class Ordinal>
WallYPlusStatistics reduce_wall_y_plus_statistics(
    const Teuchos::Comm<Ordinal>& communicator,
    const std::string& boundary_name,
    std::span<const WallYPlusSample> local_samples)
{
    using wall_y_plus_detail::require_collectively_valid;

    const auto local_name_size =
        static_cast<long long>(boundary_name.size());
    const auto local_name_hash =
        wall_y_plus_detail::stable_patch_hash(boundary_name);
    long long minimum_name_size = {};
    long long maximum_name_size = {};
    long long minimum_name_hash = {};
    long long maximum_name_hash = {};
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MIN, 1,
        &local_name_size, &minimum_name_size);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MAX, 1,
        &local_name_size, &maximum_name_size);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MIN, 1,
        &local_name_hash, &minimum_name_hash);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MAX, 1,
        &local_name_hash, &maximum_name_hash);
    require_collectively_valid(
        communicator,
        !boundary_name.empty()
            && minimum_name_size == maximum_name_size
            && minimum_name_hash == maximum_name_hash,
        "Wall y-plus boundary name must be non-empty and agree on every rank.");

    bool local_valid =
        local_samples.size()
        <= static_cast<size_t>(std::numeric_limits<long long>::max());
    real_t local_area = {};
    real_t local_weighted_sum = {};
    real_t local_minimum = std::numeric_limits<real_t>::infinity();
    real_t local_maximum = -std::numeric_limits<real_t>::infinity();
    for (const auto& sample : local_samples)
    {
        if (sample.y_plus < 0.0
            || !std::isfinite(sample.y_plus)
            || !(sample.face_area > 0.0)
            || !std::isfinite(sample.face_area))
        {
            local_valid = false;
            continue;
        }
        local_area += sample.face_area;
        local_weighted_sum += sample.face_area * sample.y_plus;
        local_minimum = std::min(local_minimum, sample.y_plus);
        local_maximum = std::max(local_maximum, sample.y_plus);
        local_valid = local_valid
                   && std::isfinite(local_area)
                   && std::isfinite(local_weighted_sum);
    }
    require_collectively_valid(
        communicator, local_valid,
        "Wall y-plus samples must be non-negative and finite; "
        "face areas must be positive and finite.");

    const auto local_count =
        static_cast<long long>(local_samples.size());
    long long global_count = {};
    real_t global_area = {};
    real_t global_weighted_sum = {};
    real_t global_minimum = {};
    real_t global_maximum = {};
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_SUM, 1,
        &local_count, &global_count);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_SUM, 1,
        &local_area, &global_area);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_SUM, 1,
        &local_weighted_sum, &global_weighted_sum);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MIN, 1,
        &local_minimum, &global_minimum);
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MAX, 1,
        &local_maximum, &global_maximum);

    WallYPlusStatistics result;
    result.boundary_name = boundary_name;
    result.global_face_count = static_cast<size_t>(global_count);
    if (global_count == 0)
    {
        return result;
    }
    if (!(global_area > 0.0)
        || !std::isfinite(global_area)
        || !std::isfinite(global_weighted_sum)
        || !std::isfinite(global_minimum)
        || !std::isfinite(global_maximum))
    {
        throw std::overflow_error(
            "Global wall y-plus reduction is not finite.");
    }

    result.total_face_area = global_area;
    result.minimum = global_minimum;
    result.maximum = global_maximum;
    result.area_weighted_mean =
        global_weighted_sum / global_area;
    return result;
}

/**
 * @brief Reduce several patches in a caller-defined, rank-consistent order.
 *
 * Missing entries in @p local_samples mean that the current rank owns no face
 * on that patch. Extra map entries are ignored.
 */
template<class Ordinal>
Arr<WallYPlusStatistics> reduce_wall_y_plus_statistics(
    const Teuchos::Comm<Ordinal>& communicator,
    std::span<const std::string> boundary_names,
    const WallYPlusSamplesByPatch& local_samples)
{
    Arr<WallYPlusStatistics> result;
    result.reserve(boundary_names.size());
    for (const auto& name : boundary_names)
    {
        const auto iter = local_samples.find(name);
        const std::span<const WallYPlusSample> samples =
            iter == local_samples.end()
                ? std::span<const WallYPlusSample>{}
                : std::span<const WallYPlusSample>{iter->second};
        result.push_back(
            reduce_wall_y_plus_statistics(
                communicator, name, samples));
    }
    return result;
}

} // namespace SimpleFluid
