#pragma once

#include "fields/details/MeshTransferGeometry.hh"

#include <numeric>
#include <span>

namespace SimpleFluid::mesh_transfer_detail
{

// Cartesian boxes use XYZ; sectors and polygon prisms use enclosing RZ boxes.
// Exact overlap evaluation remains responsible for angles and curved edges.
inline std::array<double, 6> search_bounds(const Geometry& g)
{
    return g.kind == GeometryKind::Cartesian
        ? std::array<double, 6>{g[4], g[5], g[6], g[7], g[8], g[9]}
        : std::array<double, 6>{g[4], g[5], 0, 0, g[8], g[9]};
}

class OverlapCandidates
{
public:
    explicit OverlapCandidates(std::span<const Geometry> geometry) : d_geometry(geometry)
    {
        d_order.resize(geometry.size());
        std::iota(d_order.begin(), d_order.end(), size_t{0});
        d_nodes.reserve(geometry.size() / 4 + 1);
        if (!geometry.empty()) build(0, geometry.size());
    }

    void find(const Geometry& target, std::vector<size_t>& result) const
    {
        result.clear();
        if (!d_nodes.empty()) query(0, search_bounds(target), result);
        // Preserve the original rank/map donor accumulation order.
        std::sort(result.begin(), result.end());
    }

private:
    struct Node
    {
        std::array<double, 6> bounds{};
        size_t begin{}, end{}, left{}, right{};
    };

    static bool overlaps(const std::array<double, 6>& a, const std::array<double, 6>& b)
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const auto i = 2 * axis;
            if (a[i] == a[i + 1] && b[i] == b[i + 1]) continue;
            if (a[i + 1] <= b[i] || b[i + 1] <= a[i]) return false;
        }
        return true;
    }

    size_t build(size_t begin, size_t end)
    {
        Node node;
        node.begin = begin;
        node.end = end;
        node.bounds = search_bounds(d_geometry[d_order[begin]]);
        std::array<double, 6> centers{};
        for (size_t axis = 0; axis < 3; ++axis)
            centers[2 * axis] = centers[2 * axis + 1] =
                .5 * node.bounds[2 * axis] + .5 * node.bounds[2 * axis + 1];
        for (size_t i = begin + 1; i < end; ++i)
        {
            const auto bounds = search_bounds(d_geometry[d_order[i]]);
            for (size_t axis = 0; axis < 3; ++axis)
            {
                node.bounds[2 * axis] = std::min(node.bounds[2 * axis], bounds[2 * axis]);
                node.bounds[2 * axis + 1] = std::max(node.bounds[2 * axis + 1], bounds[2 * axis + 1]);
                const auto center = .5 * bounds[2 * axis] + .5 * bounds[2 * axis + 1];
                centers[2 * axis] = std::min(centers[2 * axis], center);
                centers[2 * axis + 1] = std::max(centers[2 * axis + 1], center);
            }
        }
        const auto index = d_nodes.size();
        d_nodes.push_back(node);
        if (end - begin <= 12) return index;
        size_t axis = 0;
        for (size_t candidate = 1; candidate < 3; ++candidate)
            if (centers[2 * candidate + 1] - centers[2 * candidate]
                > centers[2 * axis + 1] - centers[2 * axis]) axis = candidate;
        const auto middle = begin + (end - begin) / 2;
        std::nth_element(d_order.begin() + begin, d_order.begin() + middle, d_order.begin() + end,
            [&](size_t a, size_t b)
            {
                const auto aa = search_bounds(d_geometry[a]), bb = search_bounds(d_geometry[b]);
                const double ac = .5 * aa[2 * axis] + .5 * aa[2 * axis + 1];
                const double bc = .5 * bb[2 * axis] + .5 * bb[2 * axis + 1];
                return ac != bc ? ac < bc : a < b;
            });
        d_nodes[index].left = build(begin, middle);
        d_nodes[index].right = build(middle, end);
        return index;
    }

    void query(size_t index, const std::array<double, 6>& target, std::vector<size_t>& result) const
    {
        const auto& node = d_nodes[index];
        if (!overlaps(node.bounds, target)) return;
        if (node.left == 0)
        {
            for (size_t i = node.begin; i < node.end; ++i)
                if (overlaps(search_bounds(d_geometry[d_order[i]]), target)) result.push_back(d_order[i]);
            return;
        }
        query(node.left, target, result);
        query(node.right, target, result);
    }

    std::span<const Geometry> d_geometry;
    std::vector<size_t> d_order;
    std::vector<Node> d_nodes;
};

} // namespace SimpleFluid::mesh_transfer_detail
