/** Dyadic geometry and typed-key expectations for factory pipeline fixtures. */
#pragma once
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/MixedAMR.hh"
#include "geometry/mesh/AMR/PolyhedralAMR.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <array>
#include <map>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::test::pipeline_scaling
{
namespace A = Meshes::AMR;

struct Box
{
    std::array<double, 3> lower;
    std::array<double, 3> width;

    [[nodiscard]] double volume() const { return width[0] * width[1] * width[2]; }
    [[nodiscard]] double center(std::size_t axis) const { return lower[axis] + .5 * width[axis]; }
    [[nodiscard]] Box child(unsigned high, unsigned axes = 7) const
    {
        auto result = *this;
        for (unsigned axis = 0; axis < 3; ++axis)
            if (axes & (1u << axis))
            {
                result.width[axis] *= .5;
                if (high & (1u << axis))
                    result.lower[axis] += result.width[axis];
            }
        return result;
    }
};
struct Cell
{
    int owner;
    Box box;
};
using Cells = std::map<A::AdaptiveCellKey, Cell>;

struct Fixture
{
    A::AdaptiveBootstrap source;
    Cells roots;
    std::vector<A::AdaptiveMark> setup, edit;
    A::AdaptiveCellKey move;
};

inline A::AdaptiveCellKey product_key(std::uint64_t region, std::uint64_t base, std::uint64_t axis)
{
    return {region, A::ProductCellAncestry{{base, 1, {}}, {axis, {}}}};
}

inline A::AdaptiveCellKey convex_key(std::uint64_t region, std::uint64_t root)
{
    return {region, A::ConvexCellAncestry{root, {}}};
}

inline A::AdaptiveCellKey convex_child(A::AdaptiveCellKey key, unsigned octant)
{
    std::get<A::ConvexCellAncestry>(key.ancestry).path.push_back(octant);
    return key;
}

inline A::AdaptiveCellKey product_child(A::AdaptiveCellKey key, unsigned base, unsigned axial, A::AdaptiveSplit split)
{
    auto& product = std::get<A::ProductCellAncestry>(key.ancestry);
    if (split != A::AdaptiveSplit::Axial)
        product.base.path.push_back(base);
    if (split != A::AdaptiveSplit::Base)
        product.axial.path.push_back(axial);
    return key;
}

inline void split(Cells& cells, const A::AdaptiveMark& mark)
{
    const auto key = mark.key;
    if (const auto* cartesian = std::get_if<A::CartesianCellAncestry>(&key.ancestry))
    {
        const auto& [level, index] = *cartesian;
        // The real Cartesian adaptation unit is a 2-by-2-by-2 block.
        for (unsigned parent = 0; parent < 8; ++parent)
        {
            A::CartesianCellAncestry coarse{level, {}};
            for (unsigned axis = 0; axis < 3; ++axis)
                coarse.index[axis] = (index[axis] / 2) * 2 + ((parent >> axis) & 1u);
            const A::AdaptiveCellKey parent_key{key.region, coarse};
            const auto cell = cells.at(parent_key);
            cells.erase(parent_key);
            for (unsigned octant = 0; octant < 8; ++octant)
            {
                A::CartesianCellAncestry fine{level + 1, {}};
                for (unsigned axis = 0; axis < 3; ++axis)
                    fine.index[axis] = 2 * coarse.index[axis] + ((octant >> axis) & 1u);
                cells.emplace(A::AdaptiveCellKey{key.region, fine}, Cell{cell.owner, cell.box.child(octant)});
            }
        }
        return;
    }

    const auto cell = cells.at(key);
    cells.erase(key);
    if (std::holds_alternative<A::ConvexCellAncestry>(key.ancestry))
    {
        for (unsigned octant = 0; octant < 8; ++octant)
            cells.emplace(convex_child(key, octant), Cell{cell.owner, cell.box.child(octant)});
        return;
    }
    const unsigned bases = mark.split == A::AdaptiveSplit::Axial ? 1 : 4;
    const unsigned layers = mark.split == A::AdaptiveSplit::Base ? 1 : 2;
    const unsigned axes = (bases == 4 ? 3u : 0u) | (layers == 2 ? 4u : 0u);
    constexpr std::array quad_octants{0u, 1u, 3u, 2u};
    unsigned rotation = 0;
    for (const auto ordinal : std::get<A::ProductCellAncestry>(key.ancestry).base.path)
        rotation = (rotation + ordinal) % 4;
    for (unsigned base = 0; base < bases; ++base)
        for (unsigned layer = 0; layer < layers; ++layer)
            cells.emplace(product_child(key, base, layer, mark.split),
                Cell{cell.owner, cell.box.child(quad_octants[(rotation + base) % 4] | (layer << 2), axes)});
}

inline std::vector<std::uint64_t> select(const A::AdaptiveHierarchy& source, const Cells& cells, int rank)
{
    std::vector<std::uint64_t> result;
    for (std::uint64_t id = 0; id < cells.size(); ++id)
        if (cells.at(source.cell_key(id)).owner == rank)
            result.push_back(id);
    return result;
}

inline Fixture product(std::size_t patches, int rank, int active)
{
    Fixture out;
    A::ExtrudedAMRGrid grid;
    grid.region_id = 5;
    grid.material_id = 11;
    grid.z_edges = {0, .5, 1};
    for (unsigned patch = 0; patch < patches; ++patch)
    {
        const unsigned offset = grid.xy_nodes.size();
        for (unsigned y = 0; y < 3; ++y)
            for (unsigned x = 0; x < 3; ++x)
                grid.xy_nodes.push_back({4. * patch + .5 * x, .5 * y, 0});
        for (unsigned child = 0; child < 4; ++child)
        {
            const unsigned x = child % 2, y = child / 2;
            const unsigned first = offset + 3 * y + x;
            grid.base_cells.push_back({first, first + 1, first + 4, first + 3});
            const int owner = int((patch == 0 ? x : patch) % active);
            for (unsigned axis = 0; axis < 2; ++axis)
            {
                const auto key = product_key(grid.region_id, 4 * patch + child, axis);
                out.roots.emplace(key, Cell{owner, {{4. * patch + .5 * x, .5 * y, .5 * axis}, {.5, .5, .5}}});
                // Keep saved families on both populated ranks and across the
                // editable column. Distant patches stay at their original
                // resolution so preparing the scale fixture is bounded.
                if (patch < 3)
                    out.setup.push_back({key, A::AdaptiveSplit::Axial});
                if (patch == 0 && child == 0)
                    for (unsigned layer = 0; layer < 2; ++layer)
                        out.edit.push_back(
                            {product_child(key, 0, layer, A::AdaptiveSplit::Axial), A::AdaptiveSplit::Base});
            }
        }
    }
    const A::LocalProductAMR factory(grid);
    std::vector<A::ProductCellAncestry> selected;
    for (const auto& [key, cell] : out.roots)
        if (cell.owner == rank)
            selected.push_back(std::get<A::ProductCellAncestry>(key.ancestry));
    out.source = factory.detach_owned(selected, {'p', 'r', 'i', 's', 'm'});
    out.move = product_child(out.edit.front().key, 3, 0, A::AdaptiveSplit::Base);
    return out;
}

inline std::shared_ptr<Meshes::UnstructuredMesh> boxes(const std::vector<Box>& geometry, bool join_first = false)
{
    using U = Meshes::UnstructuredMesh;
    Arr<MeshUtils::Vec3> nodes;
    Arr<U::CellDefinition> cells;
    std::map<std::array<double, 3>, U::NodeID> ordinals;
    constexpr std::array hexahedral_order{0u, 1u, 3u, 2u, 4u, 5u, 7u, 6u};
    for (const auto& box : geometry)
    {
        U::CellDefinition cell;
        cell.type = MeshUtils::CellType::HEXAHEDRON;
        for (const auto corner : hexahedral_order)
        {
            auto point = box.lower;
            for (unsigned axis = 0; axis < 3; ++axis)
                if (corner & (1u << axis))
                    point[axis] += box.width[axis];
            const auto [it, inserted] = ordinals.emplace(point, nodes.size());
            if (inserted)
                nodes.push_back({point[0], point[1], point[2]});
            cell.node_ids.push_back(it->second);
        }
        cells.push_back(std::move(cell));
    }
    const U topology(nodes, cells);
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (std::size_t f = 0; f < topology.num_faces(); ++f)
        if (topology.is_exterior_face(f))
        {
            const bool join = join_first && topology.face_centroid(f).x == geometry.front().lower[0];
            boundaries.push_back({topology.face_nodes(f), int(join), join ? "join" : "wall"});
        }
    return std::make_shared<U>(nodes, cells, boundaries);
}

inline A::PolyhedralAMRGrid polyhedral_grid(const std::vector<Box>& geometry, bool join_first = false)
{
    A::PolyhedralAMRGrid grid;
    grid.mesh = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::native_region("polyhedral", boxes(geometry, join_first))},
        std::vector<Meshes::MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    grid.physical_regions.assign(geometry.size(), 9);
    grid.materials.assign(geometry.size(), 11);
    return grid;
}

inline Fixture polyhedral(std::size_t pairs, int rank, int active)
{
    Fixture out;
    std::vector<Box> geometry;
    for (std::size_t pair = 0; pair < pairs; ++pair)
        for (unsigned side = 0; side < 2; ++side)
        {
            const Box box{{2. * pair + side, 0, 0}, {1, 1, 1}};
            const auto key = convex_key(9, geometry.size());
            geometry.push_back(box);
            out.roots.emplace(key, Cell{int((pair == 0 ? side : pair) % active), box});
            out.setup.push_back({key, A::AdaptiveSplit::Isotropic});
        }
    const A::PolyhedralAMR factory(polyhedral_grid(geometry));
    out.source = factory.detach_owned(select(factory, out.roots, rank), {'c', 'o', 'n', 'v', 'e', 'x'});
    out.edit.push_back({convex_child(convex_key(9, 0), 0), A::AdaptiveSplit::Isotropic});
    out.move = convex_child(out.edit.front().key, 7);
    return out;
}

inline Fixture static_companion(std::size_t groups, int rank, int active)
{
    Fixture out;
    A::AMRGrid cartesian;
    cartesian.region_id = 4;
    cartesian.material_id = 11;
    cartesian.root_blocks = {groups, 1, 1};
    cartesian.block_cells = {2, 2, 2};
    cartesian.extent = {2. * groups, 2, 2};
    for (std::size_t group = 0; group < groups; ++group)
    {
        const A::AdaptiveMark mark{{4, A::CartesianCellAncestry{0, {2 * group, 0, 0}}}, A::AdaptiveSplit::Isotropic};
        (group == 0 ? out.edit : out.setup).push_back(mark);
        const int owner = int(group % active);
        for (unsigned corner = 0; corner < 8; ++corner)
        {
            const auto x = 2 * group + (corner & 1u);
            const auto y = (corner >> 1) & 1u;
            const auto z = (corner >> 2) & 1u;
            out.roots.emplace(A::AdaptiveCellKey{4, A::CartesianCellAncestry{0, {x, y, z}}},
                Cell{owner, {{double(x), double(y), double(z)}, {1, 1, 1}}});
            const Box stationary{{double(x), double(y) + 2, double(z)}, {1, 1, 1}};
            const auto ordinal = x + (2 * groups) * (y + 2 * z);
            out.roots.emplace(A::AdaptiveCellKey{8, A::StaticCellAncestry{71, ordinal}}, Cell{owner, stationary});
        }
    }
    ArrReal along;
    for (std::size_t x = 0; x <= 2 * groups; ++x)
        along.push_back(double(x));
    auto stationary =
        std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{{std::move(along), {2, 3, 4}, {0, 1, 2}}});
    A::MixedAMRGrid grid;
    grid.adaptive = {std::make_shared<const A::CartesianAMR>(cartesian)};
    grid.stationary = {{Meshes::native_region("stationary", stationary), 8, 11, 71}};
    grid.joins = {{0, 1, "ymax", "ymin"}};
    const A::MixedAMR factory(grid);
    out.source = factory.detach_owned(select(factory, out.roots, rank), {'s', 't', 'a', 't', 'i', 'c'});
    // The first native row and this physical cell keep the same ordinal at both sizes.
    out.move = {8, A::StaticCellAncestry{71, 1}};
    return out;
}

inline Fixture mixed(std::size_t distant, int rank, int active)
{
    Fixture out;
    A::ExtrudedAMRGrid prism;
    prism.region_id = 7;
    prism.material_id = 11;
    prism.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    prism.base_cells = {{0, 1, 2, 3}};
    prism.boundaries = {{1, 2, "east"}};
    prism.z_edges = {0, 1};
    const auto root = product_key(7, 0, 0);
    out.roots.emplace(root, Cell{0, {{0, 0, 0}, {1, 1, 1}}});
    out.setup.push_back({root, A::AdaptiveSplit::Axial});
    for (unsigned layer = 0; layer < 2; ++layer)
        out.edit.push_back({product_child(root, 0, layer, A::AdaptiveSplit::Axial), A::AdaptiveSplit::Base});
    out.move = product_child(out.edit.front().key, 3, 0, A::AdaptiveSplit::Base);

    std::vector<Box> geometry;
    for (std::size_t id = 0; id < distant + 2; ++id)
    {
        const Box box{{1. + id, 0, 0}, {1, 1, 1}};
        const auto key = convex_key(9, id);
        const int owner = int((id + 1) % active);
        geometry.push_back(box);
        out.roots.emplace(key, Cell{owner, box});
        out.setup.push_back({key, A::AdaptiveSplit::Isotropic});
    }
    A::MixedAMRGrid grid;
    grid.product = std::make_shared<const A::LocalProductAMR>(prism);
    grid.physical_region = prism.region_id;
    grid.adaptive = {std::make_shared<const A::PolyhedralAMR>(polyhedral_grid(geometry, true))};
    grid.joins = {{0, 1, "east", "polyhedral/join"}};
    const A::MixedAMR factory(grid);
    out.source = factory.detach_owned(select(factory, out.roots, rank), {'m', 'i', 'x', 'e', 'd'});
    return out;
}
} // namespace SimpleFluid::test::pipeline_scaling
