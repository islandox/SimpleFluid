/**
 * @file VelocityProfileCsv.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Test-only centerline sampling and CSV output for CFD comparisons.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/VectorCellField.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::test
{

/**
 * @brief Sample serial cell-centered velocity profiles and write comparison CSV files.
 *
 * @tparam Pack Tpetra type pack used by the velocity field.
 */
template<TpetraTypePack Pack>
class VelocityProfileCsv
{
public:
    using field_type = VectorCellField<Pack>;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using vec_type = typename field_type::vec_type;

    /** @brief One axial coordinate and its volume-averaged velocity. */
    struct Sample
    {
        scalar_type coordinate = {};
        vec_type velocity{};
    };

    /**
     * @brief Sample the closest cell-center line parallel to an axis.
     *
     * Cells tied at the same transverse distance and axial coordinate are
     * volume averaged. This gives a symmetric centerline on even structured
     * grids where the requested line falls between two cell-center rows.
     *
     * @return Samples ordered by their axial coordinate.
     * @throws std::invalid_argument If @p axis is invalid or the mesh is not serial.
     */
    static std::vector<Sample> sample_nearest_line(
        const field_type& velocity,
        size_t axis,
        const vec_type& line_point)
    {
        if (axis >= 3)
        {
            throw std::invalid_argument(
                "VelocityProfileCsv axis must be 0, 1, or 2.");
        }
        const auto& mesh = velocity.mesh();
        if (mesh.owned_cell_map()->getComm()->getSize() != 1)
        {
            throw std::invalid_argument(
                "VelocityProfileCsv requires a serial mesh.");
        }
        if (mesh.num_owned_cells() == 0)
        {
            return {};
        }

        auto component =
            [](const vec_type& value, size_t index)
        {
            return value.component(index);
        };
        auto transverse_distance_squared =
            [&](local_ordinal_type cell_lid)
        {
            const auto center = mesh.cell_centroid(cell_lid);
            scalar_type distance_squared = {};
            for (size_t component_id = 0;
                 component_id < 3;
                 ++component_id)
            {
                if (component_id == axis)
                {
                    continue;
                }
                const auto delta =
                    component(center, component_id)
                  - component(line_point, component_id);
                distance_squared += delta * delta;
            }
            return distance_squared;
        };

        scalar_type minimum_distance_squared =
            transverse_distance_squared(local_ordinal_type{0});
        for (size_t owned = 1; owned < mesh.num_owned_cells(); ++owned)
        {
            minimum_distance_squared = std::min(
                minimum_distance_squared,
                transverse_distance_squared(
                    static_cast<local_ordinal_type>(owned)));
        }
        const auto tolerance =
            scalar_type{1.0e-12}
            * std::max(scalar_type{1}, minimum_distance_squared);

        /** @brief Candidate cell-center sample before axial aggregation. */
        struct Candidate
        {
            scalar_type coordinate = {};
            vec_type velocity{};
            scalar_type volume = {};
        };
        std::vector<Candidate> candidates;
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            if (std::abs(
                    transverse_distance_squared(cell_lid)
                  - minimum_distance_squared)
                > tolerance)
            {
                continue;
            }
            candidates.push_back({
                component(mesh.cell_centroid(cell_lid), axis),
                velocity.value(cell_lid),
                mesh.cell_volume(cell_lid)});
        }
        std::sort(
            candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right)
            {
                return left.coordinate < right.coordinate;
            });

        std::vector<Sample> samples;
        for (size_t first = 0; first < candidates.size();)
        {
            const auto coordinate = candidates[first].coordinate;
            const auto coordinate_tolerance =
                scalar_type{1.0e-12}
                * std::max(scalar_type{1}, std::abs(coordinate));
            vec_type weighted_velocity{};
            scalar_type total_volume = {};
            size_t last = first;
            while (last < candidates.size()
                   && std::abs(candidates[last].coordinate - coordinate)
                      <= coordinate_tolerance)
            {
                weighted_velocity =
                    weighted_velocity
                  + candidates[last].velocity
                    * candidates[last].volume;
                total_volume += candidates[last].volume;
                ++last;
            }
            samples.push_back({
                coordinate,
                total_volume > scalar_type{}
                  ? weighted_velocity / total_volume
                  : vec_type{}});
            first = last;
        }
        return samples;
    }

    /**
     * @brief Write ordered velocity samples using the comparison CSV schema.
     *
     * @param path Output CSV path.
     * @param samples Samples to write.
     * @throws std::runtime_error if the file cannot be opened or written.
     */
    static void write(
        const std::filesystem::path& path,
        const std::vector<Sample>& samples)
    {
        std::ofstream output(path);
        if (!output)
        {
            throw std::runtime_error(
                "VelocityProfileCsv could not open output file.");
        }
        output << "coordinate,ux,uy,uz\n";
        output << std::setprecision(17);
        for (const auto& sample : samples)
        {
            output << sample.coordinate << ','
                   << sample.velocity.x << ','
                   << sample.velocity.y << ','
                   << sample.velocity.z << '\n';
        }
        if (!output)
        {
            throw std::runtime_error(
                "VelocityProfileCsv failed while writing output.");
        }
    }

    /**
     * @brief Sample the nearest line and write it directly to CSV.
     *
     * @param path Output CSV path.
     * @param velocity Cell-centered velocity field to sample.
     * @param axis Coordinate axis parallel to the requested line.
     * @param line_point Point locating the requested line transversely.
     */
    static void write_nearest_line(
        const std::filesystem::path& path,
        const field_type& velocity,
        size_t axis,
        const vec_type& line_point)
    {
        write(path, sample_nearest_line(
            velocity, axis, line_point));
    }
};

} // namespace SimpleFluid::test
