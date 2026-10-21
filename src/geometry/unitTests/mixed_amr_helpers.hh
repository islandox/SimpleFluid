/** @file mixed_amr_helpers.hh
 * @brief Connected mixed-product and polyhedral native test fixtures.
 */
#pragma once
#include "geometry/mesh/AMR/MixedAMR.hh"
#include <Teuchos_DefaultSerialComm.hpp>
namespace SimpleFluid::test
{
inline SimpleFluid::Meshes::AMR::ExtrudedAMRGrid mixed_product_grid(bool four = false)
{
    SimpleFluid::Meshes::AMR::ExtrudedAMRGrid g;
    g.region_id = 7;
    g.material_id = 5;
    if (!four)
    {
        g.xy_nodes = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        g.base_cells = {{0, 1, 2, 3}};
        g.boundaries = {{0, 1, "south"}, {1, 2, "east"}, {2, 3, "north"}, {3, 0, "west"}};
        g.z_edges = {0, 1};
    }
    else
    {
        g.xy_nodes = {
            {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}};
        g.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
        g.boundaries = {{0, 1, "south"}, {1, 2, "south"}, {2, 5, "east"}, {5, 8, "east"}, {8, 7, "north"},
            {7, 6, "north"}, {6, 3, "west"}, {3, 0, "west"}};
        g.z_edges = {0, .25, .5, .75, 1};
    }
    return g;
}
inline std::shared_ptr<const SimpleFluid::Meshes::AMR::PolyhedralAMR> mixed_polyhedron(
    double upper = 2, double lower = 1)
{
    using U = Meshes::UnstructuredMesh;
    Arr<MeshUtils::Vec3> nodes{{lower, 0, 0}, {upper, 0, 0}, {upper, 1, 0}, {lower, 1, 0}, {lower, 0, 1}, {upper, 0, 1},
        {upper, 1, 1}, {lower, 1, 1}};
    U::CellDefinition cell;
    cell.type = MeshUtils::CellType::HEXAHEDRON;
    cell.node_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    Arr<U::CellDefinition> cells{cell};
    U unlabelled(nodes, cells);
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (size_t f = 0; f < unlabelled.num_faces(); ++f)
    {
        const bool join = unlabelled.face_centroid(f).x == lower;
        boundaries.push_back({unlabelled.face_nodes(f), join ? 0 : 1, join ? "join" : "wall"});
    }
    auto native = std::make_shared<U>(nodes, cells, boundaries);
    SimpleFluid::Meshes::AMR::PolyhedralAMRGrid grid;
    grid.mesh = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::native_region("polyhedral", native)},
        std::vector<Meshes::MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    grid.physical_regions = {9};
    grid.materials = {5};
    return std::make_shared<SimpleFluid::Meshes::AMR::PolyhedralAMR>(grid);
}
inline SimpleFluid::Meshes::AMR::MixedAMRGrid mixed_joined_grid(double upper = 2)
{
    SimpleFluid::Meshes::AMR::MixedAMRGrid g;
    g.product = std::make_shared<SimpleFluid::Meshes::AMR::LocalProductAMR>(mixed_product_grid());
    g.physical_region = 7;
    g.adaptive = {mixed_polyhedron(upper)};
    g.joins = {{0, 1, "east", "polyhedral/join"}};
    return g;
}
inline SimpleFluid::Meshes::AMR::ProductCellAncestry first_product(const SimpleFluid::Meshes::AMR::MixedAMR& h)
{
    return std::get<SimpleFluid::Meshes::AMR::ProductCellAncestry>(h.cell_key(0).ancestry);
}
} // namespace SimpleFluid::test
