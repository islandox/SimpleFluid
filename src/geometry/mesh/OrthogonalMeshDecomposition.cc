/** @file OrthogonalMeshDecomposition.cc */
#include "geometry/mesh/OrthogonalMeshDecomposition.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace SimpleFluid::Meshes
{
namespace
{
std::vector<size_t> balanced_cuts(const std::vector<double>& edges, size_t parts)
{
    const size_t cells = edges.size() - 1, base = cells / parts, remainder = cells % parts;
    std::vector<size_t> result(parts + 1);
    // Empty intervals trail the unit-width nonempty intervals, retaining
    // the established leading-owner convention without changing geometry.
    if (parts > cells)
    {
        for (size_t cut = 0; cut <= parts; ++cut)
            result[cut] = std::min(cut, cells);
        return result;
    }
    size_t used = 0;
    for (size_t block = 0; block < parts; ++block)
    {
        const size_t remaining = parts - block - 1;
        const bool small = remainder - used <= remaining;
        const bool large = used < remainder;
        const size_t lower = result[block] + base, upper = lower + 1;
        const long double target = static_cast<long double>(edges.front()) +
                                   (static_cast<long double>(edges.back()) - edges.front()) * (block + 1) / parts;
        const bool choose_large = large && (!small || std::abs(static_cast<long double>(edges[upper]) - target) <
                                                          std::abs(static_cast<long double>(edges[lower]) - target));
        result[block + 1] = choose_large ? upper : lower;
        used += choose_large;
    }
    return result;
}
} // namespace

OrthogonalMeshDecomposition::OrthogonalMeshDecomposition(const OrthogonalIndexer& indexer, const Edges& physical_edges,
    size_t partitions, bool allow_empty, std::optional<Shape> factors)
    : d_partitions(partitions)
{
    if (partitions == 0)
        throw std::invalid_argument("Orthogonal partition count must be positive.");
    for (size_t axis = 0; axis < 3; ++axis)
    {
        d_dimensions[axis] = indexer.num_cells_per_dim[axis];
        const auto& edges = physical_edges[axis];
        if (d_dimensions[axis] == 0 || edges.size() != d_dimensions[axis] + 1)
            throw std::invalid_argument("Orthogonal physical edges do not match cell counts.");
        for (size_t edge = 0; edge < edges.size(); ++edge)
            if (!std::isfinite(edges[edge]) || (edge && !(edges[edge] > edges[edge - 1])))
                throw std::invalid_argument("Orthogonal physical edges must be finite and increasing.");
    }
    if (factors && ((*factors)[0] == 0 || (*factors)[1] == 0 || (*factors)[2] == 0 || partitions % (*factors)[0] != 0 ||
                       partitions / (*factors)[0] % (*factors)[1] != 0 ||
                       partitions / (*factors)[0] / (*factors)[1] != (*factors)[2]))
        throw std::invalid_argument("Orthogonal partition factors must multiply to the partition count.");

    using Score = std::tuple<size_t, size_t, size_t, long double, long double, Shape>;
    std::optional<Score> best;
    const auto consider = [&](Shape shape)
    {
        Shape active{}, smallest{}, largest{};
        std::array<std::vector<size_t>, 3> cuts;
        std::array<long double, 3> minimum_width{}, maximum_width{}, widths{};
        for (size_t axis = 0; axis < 3; ++axis)
        {
            if (!allow_empty && shape[axis] > d_dimensions[axis])
                return;
            active[axis] = std::min(shape[axis], d_dimensions[axis]);
            smallest[axis] = std::max(size_t{1}, d_dimensions[axis] / shape[axis]);
            largest[axis] = d_dimensions[axis] / shape[axis] + (d_dimensions[axis] % shape[axis] != 0);
            cuts[axis] = balanced_cuts(physical_edges[axis], shape[axis]);
            minimum_width[axis] = std::numeric_limits<long double>::infinity();
            widths[axis] = static_cast<long double>(physical_edges[axis].back()) - physical_edges[axis].front();
            for (size_t block = 0; block < shape[axis]; ++block)
            {
                const auto begin = cuts[axis][block], end = cuts[axis][block + 1];
                if (begin == end)
                    continue;
                const long double width =
                    static_cast<long double>(physical_edges[axis][end]) - physical_edges[axis][begin];
                minimum_width[axis] = std::min(minimum_width[axis], width);
                maximum_width[axis] = std::max(maximum_width[axis], width);
            }
        }
        const size_t owners = active[0] * active[1] * active[2];
        const size_t maximum = largest[0] * largest[1] * largest[2];
        const size_t minimum = smallest[0] * smallest[1] * smallest[2];
        long double aspect = 1, surface = 0;
        for (size_t axis = 0; axis < 3; ++axis)
        {
            for (size_t other = 0; other < 3; ++other)
                if (axis != other)
                    aspect = std::max(aspect, maximum_width[axis] / minimum_width[other]);
            const size_t interfaces = active[axis] > 1 ? active[axis] - (indexer.periodic_dimensions[axis] ? 0 : 1) : 0;
            surface += interfaces * widths[(axis + 1) % 3] * widths[(axis + 2) % 3];
        }
        const Score score{partitions - owners, maximum, maximum - minimum, aspect, surface, shape};
        if (!best || score < *best)
        {
            best = score;
            d_factors = shape;
            d_cuts = std::move(cuts);
        }
    };
    if (factors)
        consider(*factors);
    else
        for (size_t i = 1; i <= partitions; ++i)
            if (partitions % i == 0)
                for (size_t j = 1; j <= partitions / i; ++j)
                    if (partitions / i % j == 0)
                        consider({i, j, partitions / i / j});
    if (!best)
        throw std::invalid_argument("No nonempty orthogonal tensor factorization exists for these ranks.");
}

OrthogonalMeshDecomposition::Shape OrthogonalMeshDecomposition::owned_begin(size_t rank) const
{
    if (rank >= d_partitions)
        throw std::out_of_range("Orthogonal partition rank is out of range.");
    const Shape position{rank % d_factors[0], rank / d_factors[0] % d_factors[1], rank / d_factors[0] / d_factors[1]};
    return {d_cuts[0][position[0]], d_cuts[1][position[1]], d_cuts[2][position[2]]};
}

OrthogonalMeshDecomposition::Shape OrthogonalMeshDecomposition::owned_end(size_t rank) const
{
    if (rank >= d_partitions)
        throw std::out_of_range("Orthogonal partition rank is out of range.");
    const Shape position{rank % d_factors[0], rank / d_factors[0] % d_factors[1], rank / d_factors[0] / d_factors[1]};
    return {d_cuts[0][position[0] + 1], d_cuts[1][position[1] + 1], d_cuts[2][position[2] + 1]};
}

size_t OrthogonalMeshDecomposition::num_owned_cells(size_t rank) const
{
    const auto begin = owned_begin(rank), end = owned_end(rank);
    return (end[0] - begin[0]) * (end[1] - begin[1]) * (end[2] - begin[2]);
}

size_t OrthogonalMeshDecomposition::owner(OrthogonalIndexer::CellID cell) const
{
    const Shape coordinates{cell.i, cell.j, cell.k};
    Shape position{};
    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (coordinates[axis] >= d_dimensions[axis])
            throw std::out_of_range("Orthogonal owner cell is out of range.");
        position[axis] = static_cast<size_t>(
            std::upper_bound(d_cuts[axis].begin(), d_cuts[axis].end(), coordinates[axis]) - d_cuts[axis].begin() - 1);
    }
    return position[0] + d_factors[0] * (position[1] + d_factors[1] * position[2]);
}

size_t OrthogonalMeshDecomposition::storage_bytes() const noexcept
{
    size_t bytes = sizeof(*this);
    for (const auto& cuts : d_cuts)
        bytes += cuts.capacity() * sizeof(size_t);
    return bytes;
}
} // namespace SimpleFluid::Meshes
