/**
 * @file MeshQuality.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Distributed finite-volume mesh-quality metrics and quality gate.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/MeshHandle.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Globally reduced mesh-quality diagnostics.
 *
 * Growth is the ratio of owner/neighbor normal centroid-to-face distances.
 * Non-orthogonality is the angle between the face normal and the cell-centre
 * connector. Skewness is the face-centre offset from that connector,
 * normalized by the smaller normal centroid-to-face distance. Aspect ratio is
 * the ratio of the largest and smallest normal centroid-to-face distances in
 * a cell.
 */
struct MeshQualityMetrics
{
    size_t global_cell_count = 0;
    size_t global_face_count = 0;
    real_t minimum_cell_volume =
        std::numeric_limits<real_t>::quiet_NaN();
    real_t minimum_face_area =
        std::numeric_limits<real_t>::quiet_NaN();
    real_t minimum_normal_distance =
        std::numeric_limits<real_t>::quiet_NaN();
    real_t maximum_growth_ratio = 1.0;
    real_t maximum_non_orthogonality_degrees = {};
    real_t maximum_skewness = {};
    real_t maximum_aspect_ratio = 1.0;

    std::string summary() const
    {
        std::ostringstream output;
        output << "cells=" << global_cell_count
               << ", faces=" << global_face_count
               << ", min_volume=" << minimum_cell_volume
               << ", min_area=" << minimum_face_area
               << ", min_normal_distance=" << minimum_normal_distance
               << ", max_growth=" << maximum_growth_ratio
               << ", max_non_orthogonality_deg="
               << maximum_non_orthogonality_degrees
               << ", max_skewness=" << maximum_skewness
               << ", max_aspect_ratio=" << maximum_aspect_ratio;
        return output.str();
    }
};

/**
 * @brief Configurable upper limits for the mesh-quality gate.
 *
 * Positive cell volumes, face areas, and normal distances are mandatory and
 * cannot be disabled. An empty optional disables only that upper-bound check.
 */
struct MeshQualityLimits
{
    std::optional<real_t> maximum_growth_ratio = 5.0;
    std::optional<real_t> maximum_non_orthogonality_degrees = 75.0;
    std::optional<real_t> maximum_skewness = 4.0;
    std::optional<real_t> maximum_aspect_ratio = 1.0e6;
};

/** @brief Result of applying a quality gate to one mesh. */
struct MeshQualityAssessment
{
    ArrString violations;

    bool accepted() const noexcept
    {
        return violations.empty();
    }

    std::string report(const MeshQualityMetrics& metrics) const
    {
        std::ostringstream output;
        output << (accepted() ? "Mesh quality accepted: "
                              : "Mesh quality rejected: ")
               << metrics.summary();
        if (!accepted())
        {
            output << ". Violations:";
            for (const auto& violation : violations)
            {
                output << " " << violation << ";";
            }
        }
        return output.str();
    }
};

namespace mesh_quality_detail
{

inline void validate_upper_limit(
    const std::optional<real_t>& limit,
    real_t lower_bound,
    const char* name)
{
    if (limit.has_value()
        && (!std::isfinite(*limit) || *limit < lower_bound))
    {
        throw std::invalid_argument(
            std::string(name) + " must be finite and at least "
            + std::to_string(lower_bound) + ".");
    }
}

template<class Ordinal>
void require_collectively_valid(
    const Teuchos::Comm<Ordinal>& communicator,
    bool local_valid)
{
    const int local_failed = local_valid ? 0 : 1;
    int any_failed = 0;
    Teuchos::reduceAll(
        communicator, Teuchos::REDUCE_MAX, 1,
        &local_failed, &any_failed);
    if (any_failed != 0)
    {
        throw std::runtime_error(
            "Mesh-quality evaluation encountered invalid local geometry.");
    }
}

inline real_t clamped_angle_degrees(real_t cosine)
{
    constexpr real_t radians_to_degrees =
        180.0 / std::numbers::pi_v<real_t>;
    return std::acos(std::clamp(std::abs(cosine), 0.0, 1.0))
         * radians_to_degrees;
}

} // namespace mesh_quality_detail

/**
 * @brief Evaluate global finite-volume quality metrics on a distributed mesh.
 *
 * Only owned cells and faces contribute, so partition ghosts are never
 * double-counted. All ranks either return the same metrics or throw together.
 */
template<TpetraTypePack Pack>
MeshQualityMetrics evaluate_mesh_quality(
    const MeshHandle<Pack>& mesh)
{
    using local_ordinal_type =
        typename MeshHandle<Pack>::local_ordinal_type;

    const auto communicator = mesh.owned_cell_map()->getComm();
    bool local_valid = true;
    long long local_cell_count = {};
    long long local_face_count = {};
    real_t local_minimum_volume =
        std::numeric_limits<real_t>::infinity();
    real_t local_minimum_area =
        std::numeric_limits<real_t>::infinity();
    real_t local_minimum_distance =
        std::numeric_limits<real_t>::infinity();
    real_t local_maximum_growth = 1.0;
    real_t local_maximum_non_orthogonality = {};
    real_t local_maximum_skewness = {};
    real_t local_maximum_aspect = 1.0;

    try
    {
        for (size_t local = 0; local < mesh.num_owned_cells(); ++local)
        {
            const auto cell = static_cast<local_ordinal_type>(local);
            ++local_cell_count;
            const auto volume = mesh.cell_volume(cell);
            local_valid = local_valid
                       && std::isfinite(volume)
                       && volume > 0.0;
            local_minimum_volume =
                std::min(local_minimum_volume, volume);

            real_t minimum_cell_distance =
                std::numeric_limits<real_t>::infinity();
            real_t maximum_cell_distance = {};
            const auto cell_centroid = mesh.cell_centroid(cell);
            for (const auto face : mesh.faces(cell))
            {
                const auto displacement =
                    mesh.face_centroid(face) - cell_centroid;
                const auto normal =
                    mesh.face_normal_outward(face, cell);
                const auto normal_distance =
                    std::abs(displacement.dot(normal));
                local_valid = local_valid
                           && std::isfinite(normal_distance)
                           && normal_distance > 0.0;
                local_minimum_distance =
                    std::min(local_minimum_distance, normal_distance);
                minimum_cell_distance =
                    std::min(minimum_cell_distance, normal_distance);
                maximum_cell_distance =
                    std::max(maximum_cell_distance, normal_distance);
            }
            if (!(minimum_cell_distance > 0.0)
                || !std::isfinite(minimum_cell_distance)
                || !std::isfinite(maximum_cell_distance))
            {
                local_valid = false;
            }
            else
            {
                local_maximum_aspect = std::max(
                    local_maximum_aspect,
                    maximum_cell_distance / minimum_cell_distance);
            }
        }

        for (size_t local = 0; local < mesh.num_faces(); ++local)
        {
            const auto face = static_cast<local_ordinal_type>(local);
            if (!mesh.is_owned_face(face))
            {
                continue;
            }
            ++local_face_count;
            const auto area = mesh.face_area(face);
            local_valid = local_valid
                       && std::isfinite(area)
                       && area > 0.0;
            local_minimum_area =
                std::min(local_minimum_area, area);

            const auto owner = mesh.owner_cell(face);
            if (owner == MeshHandle<Pack>::invalid_local_id())
            {
                local_valid = false;
                continue;
            }
            const auto owner_centroid = mesh.cell_centroid(owner);
            const auto face_centroid = mesh.face_centroid(face);
            const auto normal = mesh.face_normal(face);
            const auto owner_to_face = face_centroid - owner_centroid;
            const auto owner_distance =
                std::abs(owner_to_face.dot(normal));
            const auto neighbor = mesh.neighbor_cell(face);

            if (neighbor == MeshHandle<Pack>::invalid_local_id())
            {
                const auto connector_norm = owner_to_face.norm();
                local_valid = local_valid
                           && std::isfinite(connector_norm)
                           && connector_norm > 0.0
                           && std::isfinite(owner_distance)
                           && owner_distance > 0.0;
                if (connector_norm > 0.0 && owner_distance > 0.0)
                {
                    local_maximum_non_orthogonality = std::max(
                        local_maximum_non_orthogonality,
                        mesh_quality_detail::clamped_angle_degrees(
                            owner_to_face.dot(normal) / connector_norm));
                    const auto tangential_squared = std::max(
                        connector_norm * connector_norm
                            - owner_distance * owner_distance,
                        0.0);
                    local_maximum_skewness = std::max(
                        local_maximum_skewness,
                        std::sqrt(tangential_squared) / owner_distance);
                }
                continue;
            }

            const auto neighbor_centroid = mesh.cell_centroid(neighbor);
            const auto connector =
                neighbor_centroid - owner_centroid;
            const auto connector_norm = connector.norm();
            const auto neighbor_distance =
                std::abs((neighbor_centroid - face_centroid).dot(normal));
            const auto connector_normal = connector.dot(normal);
            local_valid = local_valid
                       && std::isfinite(connector_norm)
                       && connector_norm > 0.0
                       && std::isfinite(owner_distance)
                       && owner_distance > 0.0
                       && std::isfinite(neighbor_distance)
                       && neighbor_distance > 0.0
                       && std::isfinite(connector_normal)
                       && std::abs(connector_normal) > 0.0;
            if (!(connector_norm > 0.0)
                || !(owner_distance > 0.0)
                || !(neighbor_distance > 0.0)
                || !(std::abs(connector_normal) > 0.0))
            {
                continue;
            }

            local_maximum_non_orthogonality = std::max(
                local_maximum_non_orthogonality,
                mesh_quality_detail::clamped_angle_degrees(
                    connector_normal / connector_norm));
            const auto interpolation =
                owner_to_face.dot(normal) / connector_normal;
            const auto connector_intersection =
                owner_centroid + connector * interpolation;
            const auto skew_distance =
                (face_centroid - connector_intersection).norm();
            local_maximum_skewness = std::max(
                local_maximum_skewness,
                skew_distance
                    / std::min(owner_distance, neighbor_distance));
            local_maximum_growth = std::max(
                local_maximum_growth,
                std::max(owner_distance, neighbor_distance)
                    / std::min(owner_distance, neighbor_distance));
        }
    }
    catch (...)
    {
        local_valid = false;
    }

    local_valid = local_valid
               && std::isfinite(local_maximum_growth)
               && std::isfinite(local_maximum_non_orthogonality)
               && std::isfinite(local_maximum_skewness)
               && std::isfinite(local_maximum_aspect);
    mesh_quality_detail::require_collectively_valid(
        *communicator, local_valid);

    long long global_cell_count = {};
    long long global_face_count = {};
    MeshQualityMetrics result;
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_SUM, 1,
        &local_cell_count, &global_cell_count);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_SUM, 1,
        &local_face_count, &global_face_count);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1,
        &local_minimum_volume, &result.minimum_cell_volume);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1,
        &local_minimum_area, &result.minimum_face_area);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1,
        &local_minimum_distance, &result.minimum_normal_distance);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1,
        &local_maximum_growth, &result.maximum_growth_ratio);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1,
        &local_maximum_non_orthogonality,
        &result.maximum_non_orthogonality_degrees);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1,
        &local_maximum_skewness, &result.maximum_skewness);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1,
        &local_maximum_aspect, &result.maximum_aspect_ratio);
    result.global_cell_count =
        static_cast<size_t>(global_cell_count);
    result.global_face_count =
        static_cast<size_t>(global_face_count);
    return result;
}

/** @brief Validate and apply a reusable mesh-quality gate. */
class MeshQualityGate
{
public:
    explicit MeshQualityGate(MeshQualityLimits limits = {})
        : d_limits(std::move(limits))
    {
        mesh_quality_detail::validate_upper_limit(
            d_limits.maximum_growth_ratio, 1.0,
            "Maximum mesh growth ratio");
        mesh_quality_detail::validate_upper_limit(
            d_limits.maximum_non_orthogonality_degrees, 0.0,
            "Maximum mesh non-orthogonality");
        mesh_quality_detail::validate_upper_limit(
            d_limits.maximum_skewness, 0.0,
            "Maximum mesh skewness");
        mesh_quality_detail::validate_upper_limit(
            d_limits.maximum_aspect_ratio, 1.0,
            "Maximum mesh aspect ratio");
    }

    const MeshQualityLimits& limits() const noexcept
    {
        return d_limits;
    }

    MeshQualityAssessment assess(
        const MeshQualityMetrics& metrics) const
    {
        MeshQualityAssessment result;
        if (metrics.global_cell_count == 0)
        {
            result.violations.push_back("mesh has no cells");
        }
        if (metrics.global_face_count == 0)
        {
            result.violations.push_back("mesh has no faces");
        }
        if (!(metrics.minimum_cell_volume > 0.0)
            || !std::isfinite(metrics.minimum_cell_volume))
        {
            result.violations.push_back(
                "cell volumes are not positive and finite");
        }
        if (!(metrics.minimum_face_area > 0.0)
            || !std::isfinite(metrics.minimum_face_area))
        {
            result.violations.push_back(
                "face areas are not positive and finite");
        }
        if (!(metrics.minimum_normal_distance > 0.0)
            || !std::isfinite(metrics.minimum_normal_distance))
        {
            result.violations.push_back(
                "normal cell-to-face distances are not positive and finite");
        }

        check_upper_bound(
            metrics.maximum_growth_ratio,
            d_limits.maximum_growth_ratio,
            1.0, "growth ratio", result);
        check_upper_bound(
            metrics.maximum_non_orthogonality_degrees,
            d_limits.maximum_non_orthogonality_degrees,
            0.0, "non-orthogonality", result);
        check_upper_bound(
            metrics.maximum_skewness,
            d_limits.maximum_skewness,
            0.0, "skewness", result);
        check_upper_bound(
            metrics.maximum_aspect_ratio,
            d_limits.maximum_aspect_ratio,
            1.0, "aspect ratio", result);
        return result;
    }

    /**
     * @brief Throw with a metric-rich report when the mesh is unacceptable.
     */
    void require(const MeshQualityMetrics& metrics) const
    {
        const auto assessment = assess(metrics);
        if (!assessment.accepted())
        {
            throw std::runtime_error(assessment.report(metrics));
        }
    }

private:
    static void check_upper_bound(
        real_t value,
        const std::optional<real_t>& limit,
        real_t minimum,
        const char* name,
        MeshQualityAssessment& assessment)
    {
        if (!std::isfinite(value))
        {
            assessment.violations.push_back(
                std::string(name) + " is not finite");
        }
        else if (value < minimum)
        {
            assessment.violations.push_back(
                std::string(name) + " is below "
                + std::to_string(minimum));
        }
        else if (limit.has_value() && value > *limit)
        {
            assessment.violations.push_back(
                std::string(name) + " exceeds "
                + std::to_string(*limit));
        }
    }

    MeshQualityLimits d_limits;
};

} // namespace SimpleFluid
