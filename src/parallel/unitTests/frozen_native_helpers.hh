/** @file frozen_native_helpers.hh
 * @brief Unsupported accepted native copy fixtures for map and named-history consumers.
 */
#pragma once
#include "geometry/mesh/AMR/FrozenNative.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "geometry/mesh/RegionProviders.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <Tpetra_Core.hpp>
namespace SimpleFluid::test
{
using namespace SimpleFluid::Meshes;
/** Deliberately aliases all source Data through observation identity, exercising
 * the lifetime hazard that independent-token native hierarchies do not expose. */
class AliasHierarchy final : public ::SimpleFluid::Meshes::AMR::AdaptiveHierarchy
{
public:
    struct Data
    {
        SP<const MultiRegionMesh> geometry, resident;
    };
    std::shared_ptr<Data> data;
    ::SimpleFluid::Meshes::AMR::HierarchyObservation snapshot;
    explicit AliasHierarchy(SP<const MultiRegionMesh> mesh, SP<const MultiRegionMesh> resident = {})
    {
        data = std::make_shared<Data>(Data{mesh, resident ? resident : mesh});
        snapshot = {data, data, data, 4, 9, data->resident->geometry_epoch(), data->resident.get(),
            Tpetra::getDefaultComm().getRawPtr()};
    }
    ::SimpleFluid::Meshes::AMR::HierarchyObservation observation() const noexcept override { return snapshot; }
    const SP<const MultiRegionMesh>& native_geometry() const noexcept override { return data->geometry; }
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey cell_key(std::uint64_t cell) const override
    {
        auto [region, native] = data->geometry->native_cell(cell);
        if (region == 2)
            return {
                42, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {native % 2, (native / 2) % 2, native / 4}}};
        return {40 + region, ::SimpleFluid::Meshes::AMR::StaticCellAncestry{region, native}};
    }
    std::uint64_t cell_material(std::uint64_t cell) const override { return 5 + cell_fragment(cell); }
    std::size_t cell_fragment(std::uint64_t cell) const override { return data->geometry->native_cell(cell).first; }
};
inline MultiRegionMesh::Region concave()
{
    using P = ExplicitRegionProvider;
    std::vector<SimpleFluid::PLIC::GeometryPoint> xy{
        {3, -3, 0}, {5, -3, 0}, {5, -2, 0}, {4, -2, 0}, {4, -1, 0}, {3, -1, 0}};
    std::vector<P::Node> nodes;
    for (std::uint64_t z = 0; z < 2; ++z)
        for (std::uint64_t i = 0; i < 6; ++i)
            nodes.push_back({6 * z + i, {xy[i][0], xy[i][1], double(z)}});
    std::vector<P::Face> faces;
    faces.push_back(
        {0, 0, MultiRegionMesh::invalid_cell_id(), {5, 4, 3, 2, 1, 0}, {3 + 5.0 / 6, -3 + 5.0 / 6, 0}, {0, 0, -3}, 0});
    faces.push_back(
        {1, 0, MultiRegionMesh::invalid_cell_id(), {6, 7, 8, 9, 10, 11}, {3 + 5.0 / 6, -3 + 5.0 / 6, 1}, {0, 0, 3}, 0});
    for (std::uint64_t i = 0; i < 6; ++i)
    {
        const auto j = (i + 1) % 6;
        const double dx = xy[j][0] - xy[i][0], dy = xy[j][1] - xy[i][1];
        faces.push_back({i + 2, 0, MultiRegionMesh::invalid_cell_id(), {i, j, j + 6, i + 6},
            {(xy[i][0] + xy[j][0]) * .5, (xy[i][1] + xy[j][1]) * .5, .5}, {dy, -dx, 0}, i == 4 ? 1 : 0});
    }
    P::Cell cell;
    cell.id = 0;
    cell.type = MeshUtils::CellType::POLYHEDRON;
    cell.centroid = {3 + 5.0 / 6, -3 + 5.0 / 6, .5};
    cell.volume = 3;
    for (std::uint64_t f = 0; f < 8; ++f)
        cell.faces.push_back(f);
    for (std::uint64_t n = 0; n < 12; ++n)
        cell.nodes.push_back(n);
    RegionLayout layout{RegionLayout::Family::Explicit, 1, 8, 12, {}};
    auto provider = std::make_shared<const P>(layout, std::vector<P::Cell>{cell}, std::move(faces), std::move(nodes),
        std::map<int, std::string>{{0, "wall"}, {1, "join"}});
    return DistributedExplicitRegion("concave", provider, provider);
}
inline SP<const MultiRegionMesh> mixed_source()
{
    auto cylinder = std::make_shared<OrthogonalCylindrial3D>(Vec3D<ArrReal>{{{3, 4}, {0, .5, 1}, {0, .5, 1}}});
    auto cartesian = cartesian_region("certified_cartesian", {{{3, 3.5, 4}, {-1, -.5, 0}, {0, .5, 1}}});
    auto curved = native_region("curved", cylinder);
    NonconformingInterface from_concave;
    from_concave.coarse_region = 0;
    from_concave.fine_region = 2;
    from_concave.coarse_boundary = 1;
    from_concave.fine_boundary = 2;
    CoarseFineFaceMapping top;
    top.coarse_face = 6;
    for (std::uint64_t face = 0; face < cartesian.layout().faces; ++face)
        if (cartesian.topology().boundary_id(face) == 2)
            top.fine_faces.push_back(face);
    from_concave.faces.push_back(std::move(top));
    NonconformingInterface from_cylinder;
    from_cylinder.coarse_region = 1;
    from_cylinder.fine_region = 2;
    from_cylinder.coarse_boundary = 2;
    from_cylinder.fine_boundary = 3;
    for (std::uint64_t coarse = 0; coarse < curved.layout().faces; ++coarse)
        if (curved.topology().boundary_id(coarse) == 2)
        {
            CoarseFineFaceMapping row;
            row.coarse_face = coarse;
            const auto z = curved.geometry().face_centroid(coarse).z;
            for (std::uint64_t fine = 0; fine < cartesian.layout().faces; ++fine)
                if (cartesian.topology().boundary_id(fine) == 3 && cartesian.geometry().face_centroid(fine).z == z)
                    row.fine_faces.push_back(fine);
            from_cylinder.faces.push_back(std::move(row));
        }
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{concave(), curved, cartesian},
        std::vector<MultiRegionMesh::Interface>{from_concave, from_cylinder},
        Teuchos::rcp(new Teuchos::SerialComm<int>()));
}

} // namespace SimpleFluid::test
