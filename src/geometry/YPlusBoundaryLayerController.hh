/**
 * @file YPlusBoundaryLayerController.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Y-plus-driven boundary-layer mesh rebuild controller.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/BoundaryLayerMeshFactory.hh"
#include "geometry/MeshQuality.hh"
#include "geometry/WallYPlusStatistics.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Parameters for converting measured y-plus into first-cell height.
 *
 * The undamped update is
 * `h_next / h = (target_y_plus / measured_y_plus)^adaptation_exponent`.
 * Damping is applied in logarithmic space before ratio and absolute height
 * clamps. Log-space damping preserves positivity and treats refinement and
 * coarsening symmetrically.
 */
struct YPlusBoundaryLayerControllerOptions
{
    real_t target_y_plus = 1.0;
    WallYPlusControlStatistic control_statistic =
        WallYPlusControlStatistic::Maximum;
    real_t adaptation_exponent = 0.5;
    real_t damping = 1.0;
    real_t minimum_height_ratio = 0.5;
    real_t maximum_height_ratio = 2.0;
    std::optional<real_t> minimum_first_cell_height;
    std::optional<real_t> maximum_first_cell_height;
    real_t relative_tolerance = 0.1;
};

/** @brief One patch's bounded first-cell-height decision. */
struct YPlusBoundaryLayerHeightUpdate
{
    std::string boundary_name;
    real_t current_height = {};
    real_t measured_y_plus = {};
    real_t target_y_plus = {};
    real_t relative_error = {};
    real_t undamped_height_ratio = 1.0;
    real_t damped_height_ratio = 1.0;
    real_t applied_height_ratio = 1.0;
    real_t next_height = {};
    bool ratio_was_limited = false;
    bool absolute_height_was_limited = false;
    bool converged = false;
};

/** @brief Updated layer specifications and their patch-level decisions. */
struct YPlusBoundaryLayerUpdate
{
    Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec> layer_specs;
    Arr<YPlusBoundaryLayerHeightUpdate> patch_updates;
    bool converged = false;
};

/**
 * @brief Bounded feedback controller for boundary-layer first-cell heights.
 *
 * The default maximum-y-plus statistic is deliberately conservative for
 * wall-resolved SST. Select `AreaWeightedMean` explicitly for patch-average
 * high-Re wall-function matching.
 */
class YPlusBoundaryLayerController
{
public:
    explicit YPlusBoundaryLayerController(
        YPlusBoundaryLayerControllerOptions options = {})
        : d_options(std::move(options))
    {
        validate_options(d_options);
    }

    const YPlusBoundaryLayerControllerOptions& options() const noexcept
    {
        return d_options;
    }

    /**
     * @brief Compute one patch's next first-cell height.
     */
    YPlusBoundaryLayerHeightUpdate update_height(
        real_t current_height,
        const WallYPlusStatistics& statistics) const
    {
        if (!(current_height > 0.0)
            || !std::isfinite(current_height))
        {
            throw std::invalid_argument(
                "Current boundary-layer first-cell height must be "
                "positive and finite.");
        }
        const auto measured =
            statistics.measurement(d_options.control_statistic);
        if (!(measured > 0.0) || !std::isfinite(measured))
        {
            throw std::invalid_argument(
                "Wall y-plus feedback measurement must be positive "
                "and finite.");
        }

        YPlusBoundaryLayerHeightUpdate result;
        result.boundary_name = statistics.boundary_name;
        result.current_height = current_height;
        result.measured_y_plus = measured;
        result.target_y_plus = d_options.target_y_plus;
        result.relative_error =
            std::abs(measured - d_options.target_y_plus)
            / d_options.target_y_plus;
        result.converged =
            result.relative_error <= d_options.relative_tolerance;
        if (result.converged)
        {
            result.next_height = current_height;
            return result;
        }

        result.undamped_height_ratio = std::pow(
            d_options.target_y_plus / measured,
            d_options.adaptation_exponent);
        result.damped_height_ratio = std::pow(
            result.undamped_height_ratio,
            d_options.damping);
        result.applied_height_ratio = std::clamp(
            result.damped_height_ratio,
            d_options.minimum_height_ratio,
            d_options.maximum_height_ratio);
        result.ratio_was_limited =
            result.applied_height_ratio
            != result.damped_height_ratio;

        const auto unclamped_height =
            current_height * result.applied_height_ratio;
        result.next_height = unclamped_height;
        if (d_options.minimum_first_cell_height.has_value())
        {
            result.next_height = std::max(
                result.next_height,
                *d_options.minimum_first_cell_height);
        }
        if (d_options.maximum_first_cell_height.has_value())
        {
            result.next_height = std::min(
                result.next_height,
                *d_options.maximum_first_cell_height);
        }
        result.absolute_height_was_limited =
            result.next_height != unclamped_height;
        if (!(result.next_height > 0.0)
            || !std::isfinite(result.next_height))
        {
            throw std::overflow_error(
                "Y-plus boundary-layer height update is not positive "
                "and finite.");
        }
        return result;
    }

    /**
     * @brief Update every configured boundary-layer patch by name.
     *
     * Every layer specification must have exactly one global statistics entry.
     * Extra statistics are permitted because a pilot solver may report walls
     * that are not controlled by the current mesh recipe.
     */
    YPlusBoundaryLayerUpdate update_layer_specs(
        const Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec>& layer_specs,
        const Arr<WallYPlusStatistics>& statistics) const
    {
        static_cast<void>(
            BoundaryLayerMeshFactory(layer_specs));
        std::unordered_map<std::string, const WallYPlusStatistics*>
            statistics_by_name;
        statistics_by_name.reserve(statistics.size());
        for (const auto& patch : statistics)
        {
            if (!statistics_by_name.emplace(
                    patch.boundary_name, &patch).second)
            {
                throw std::invalid_argument(
                    "Duplicate wall y-plus statistics for boundary "
                    + patch.boundary_name + ".");
            }
        }

        YPlusBoundaryLayerUpdate result;
        result.layer_specs = layer_specs;
        result.patch_updates.reserve(layer_specs.size());
        result.converged = !layer_specs.empty();
        for (auto& spec : result.layer_specs)
        {
            const auto iter =
                statistics_by_name.find(spec.boundary_name);
            if (iter == statistics_by_name.end())
            {
                throw std::invalid_argument(
                    "Missing wall y-plus statistics for boundary "
                    + spec.boundary_name + ".");
            }
            auto patch_update = update_height(
                spec.first_cell_height, *iter->second);
            spec.first_cell_height = patch_update.next_height;
            result.converged =
                result.converged && patch_update.converged;
            result.patch_updates.push_back(
                std::move(patch_update));
        }
        static_cast<void>(
            BoundaryLayerMeshFactory(result.layer_specs));
        return result;
    }

private:
    static void validate_options(
        const YPlusBoundaryLayerControllerOptions& options)
    {
        const auto require_in_open_closed_unit_interval =
            [](real_t value, const char* name)
            {
                if (!(value > 0.0) || value > 1.0
                    || !std::isfinite(value))
                {
                    throw std::invalid_argument(
                        std::string(name)
                        + " must be finite and in (0, 1].");
                }
            };
        if (!(options.target_y_plus > 0.0)
            || !std::isfinite(options.target_y_plus))
        {
            throw std::invalid_argument(
                "Target wall y-plus must be positive and finite.");
        }
        require_in_open_closed_unit_interval(
            options.adaptation_exponent,
            "Y-plus adaptation exponent");
        require_in_open_closed_unit_interval(
            options.damping, "Y-plus adaptation damping");
        if (!(options.minimum_height_ratio > 0.0)
            || options.minimum_height_ratio > 1.0
            || !std::isfinite(options.minimum_height_ratio))
        {
            throw std::invalid_argument(
                "Minimum y-plus height ratio must be finite and in (0, 1].");
        }
        if (options.maximum_height_ratio < 1.0
            || !std::isfinite(options.maximum_height_ratio))
        {
            throw std::invalid_argument(
                "Maximum y-plus height ratio must be finite and at least one.");
        }
        if (options.relative_tolerance < 0.0
            || !std::isfinite(options.relative_tolerance))
        {
            throw std::invalid_argument(
                "Y-plus relative tolerance must be finite and non-negative.");
        }
        if (options.minimum_first_cell_height.has_value()
            && (!std::isfinite(*options.minimum_first_cell_height)
                || !(*options.minimum_first_cell_height > 0.0)))
        {
            throw std::invalid_argument(
                "Minimum first-cell height must be positive and finite.");
        }
        if (options.maximum_first_cell_height.has_value()
            && (!std::isfinite(*options.maximum_first_cell_height)
                || !(*options.maximum_first_cell_height > 0.0)))
        {
            throw std::invalid_argument(
                "Maximum first-cell height must be positive and finite.");
        }
        if (options.minimum_first_cell_height.has_value()
            && options.maximum_first_cell_height.has_value()
            && *options.minimum_first_cell_height
                 > *options.maximum_first_cell_height)
        {
            throw std::invalid_argument(
                "Minimum first-cell height cannot exceed the maximum.");
        }
    }

    YPlusBoundaryLayerControllerOptions d_options;
};

/** @brief Terminal state of an outer boundary-layer adaptation run. */
enum class YPlusBoundaryLayerAdaptationStatus : uint8_t
{
    Converged,
    MaximumCyclesReached,
    MeshQualityRejected
};

/** @brief Controls the outer rebuild/pilot loop. */
struct YPlusBoundaryLayerAdaptationOptions
{
    size_t maximum_cycles = 8;
    MeshQualityLimits mesh_quality_limits;
};

/** @brief Persistent, mesh-independent record for one adaptation cycle. */
struct YPlusBoundaryLayerCycleReport
{
    size_t cycle = 0;
    Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec> evaluated_layer_specs;
    MeshQualityMetrics mesh_quality;
    MeshQualityAssessment mesh_quality_assessment;
    Arr<WallYPlusStatistics> wall_y_plus_statistics;
    std::optional<YPlusBoundaryLayerUpdate> update;
};

/**
 * @brief Mesh-independent outcome of an outer adaptation run.
 *
 * `evaluated_layer_specs` produced the last mesh that actually passed through
 * the quality gate and pilot solve. It is empty if the first mesh is rejected.
 * `recommended_layer_specs` is the next controller proposal after a completed
 * pilot. On convergence, or after a later quality rejection, it is the last
 * accepted specification; it is empty when no mesh was accepted.
 */
struct YPlusBoundaryLayerAdaptationReport
{
    YPlusBoundaryLayerAdaptationStatus status =
        YPlusBoundaryLayerAdaptationStatus::MaximumCyclesReached;
    Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec>
        evaluated_layer_specs;
    Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec>
        recommended_layer_specs;
    Arr<YPlusBoundaryLayerCycleReport> cycles;

    bool converged() const noexcept
    {
        return status
            == YPlusBoundaryLayerAdaptationStatus::Converged;
    }

    std::string summary() const
    {
        std::ostringstream output;
        switch (status)
        {
            case YPlusBoundaryLayerAdaptationStatus::Converged:
                output << "Y-plus adaptation converged";
                break;
            case YPlusBoundaryLayerAdaptationStatus::MaximumCyclesReached:
                output << "Y-plus adaptation reached its cycle limit";
                break;
            case YPlusBoundaryLayerAdaptationStatus::MeshQualityRejected:
                output << "Y-plus adaptation rejected a mesh";
                break;
        }
        output << " after " << cycles.size() << " cycle(s)";
        if (!cycles.empty())
        {
            output << ". "
                   << cycles.back().mesh_quality_assessment.report(
                          cycles.back().mesh_quality);
        }
        return output.str();
    }
};

namespace y_plus_adaptation_detail
{

template<class Artifact>
decltype(auto) artifact_mesh(Artifact& artifact)
{
    if constexpr (requires { *artifact; })
    {
        return *artifact;
    }
    else
    {
        return (artifact);
    }
}

} // namespace y_plus_adaptation_detail

/**
 * @brief Outer driver that rebuilds a pristine mesh for every y-plus cycle.
 *
 * The mesh recipe receives `(layer_specs, cycle)` and returns an arbitrary
 * mesh artifact. The quality callback receives `(artifact, cycle)` and runs
 * before the pilot. The pilot callback receives `(artifact, cycle)` and
 * returns globally reduced `Arr<WallYPlusStatistics>`.
 *
 * The artifact lives only within one loop iteration and is never stored in the
 * returned report. Consequently mesh handles, fields, and solver caches cannot
 * be transferred by this API between topologically different meshes. The
 * caller should construct a new full solver from `evaluated_layer_specs` after
 * convergence.
 */
class YPlusBoundaryLayerAdaptationDriver
{
public:
    YPlusBoundaryLayerAdaptationDriver(
        YPlusBoundaryLayerController controller,
        YPlusBoundaryLayerAdaptationOptions options = {})
        : d_controller(std::move(controller)),
          d_options(std::move(options)),
          d_quality_gate(d_options.mesh_quality_limits)
    {
        if (d_options.maximum_cycles == 0)
        {
            throw std::invalid_argument(
                "Y-plus adaptation requires at least one cycle.");
        }
    }

    const YPlusBoundaryLayerController& controller() const noexcept
    {
        return d_controller;
    }

    const YPlusBoundaryLayerAdaptationOptions& options() const noexcept
    {
        return d_options;
    }

    /**
     * @brief Run with an explicit quality evaluator for a custom artifact.
     */
    template<class MeshRecipe, class QualityEvaluator, class PilotSolver>
    YPlusBoundaryLayerAdaptationReport run(
        Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec> initial_layer_specs,
        MeshRecipe&& mesh_recipe,
        QualityEvaluator&& quality_evaluator,
        PilotSolver&& pilot_solver) const
    {
        if (initial_layer_specs.empty())
        {
            throw std::invalid_argument(
                "Y-plus adaptation requires at least one boundary layer.");
        }
        static_cast<void>(
            BoundaryLayerMeshFactory(initial_layer_specs));

        YPlusBoundaryLayerAdaptationReport report;
        auto current_specs = std::move(initial_layer_specs);
        for (size_t cycle = 0;
             cycle < d_options.maximum_cycles;
             ++cycle)
        {
            auto artifact = std::invoke(
                mesh_recipe, std::as_const(current_specs), cycle);

            YPlusBoundaryLayerCycleReport cycle_report;
            cycle_report.cycle = cycle;
            cycle_report.evaluated_layer_specs = current_specs;
            cycle_report.mesh_quality = std::invoke(
                quality_evaluator, artifact, cycle);
            cycle_report.mesh_quality_assessment =
                d_quality_gate.assess(cycle_report.mesh_quality);
            if (!cycle_report.mesh_quality_assessment.accepted())
            {
                report.status =
                    YPlusBoundaryLayerAdaptationStatus::MeshQualityRejected;
                report.recommended_layer_specs =
                    report.evaluated_layer_specs;
                report.cycles.push_back(std::move(cycle_report));
                return report;
            }

            cycle_report.wall_y_plus_statistics = std::invoke(
                pilot_solver, artifact, cycle);
            cycle_report.update = d_controller.update_layer_specs(
                current_specs,
                cycle_report.wall_y_plus_statistics);
            const auto converged =
                cycle_report.update->converged;
            auto next_specs =
                cycle_report.update->layer_specs;
            report.cycles.push_back(std::move(cycle_report));
            report.evaluated_layer_specs = current_specs;
            report.recommended_layer_specs = next_specs;
            if (converged)
            {
                report.status =
                    YPlusBoundaryLayerAdaptationStatus::Converged;
                report.recommended_layer_specs = current_specs;
                return report;
            }
            current_specs = std::move(next_specs);
        }

        report.status =
            YPlusBoundaryLayerAdaptationStatus::MaximumCyclesReached;
        return report;
    }

    /**
     * @brief Run when the recipe returns a MeshHandle or pointer-like handle.
     *
     * The built-in quality evaluator dereferences pointer-like artifacts and
     * calls `evaluate_mesh_quality` before the pilot solve.
     */
    template<class MeshRecipe, class PilotSolver>
    YPlusBoundaryLayerAdaptationReport run(
        Arr<BoundaryLayerMeshFactory::BoundaryLayerSpec> initial_layer_specs,
        MeshRecipe&& mesh_recipe,
        PilotSolver&& pilot_solver) const
    {
        return run(
            std::move(initial_layer_specs),
            std::forward<MeshRecipe>(mesh_recipe),
            [](auto& artifact, size_t)
            {
                return evaluate_mesh_quality(
                    y_plus_adaptation_detail::artifact_mesh(artifact));
            },
            std::forward<PilotSolver>(pilot_solver));
    }

private:
    YPlusBoundaryLayerController d_controller;
    YPlusBoundaryLayerAdaptationOptions d_options;
    MeshQualityGate d_quality_gate;
};

} // namespace SimpleFluid
