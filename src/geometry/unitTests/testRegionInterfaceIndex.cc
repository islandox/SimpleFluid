/**
 * @file testRegionInterfaceIndex.cc
 * @brief Region-local interface directory, canonical identity and compact storage regressions.
 */
#include <gtest/gtest.h>
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<MultiRegionMesh> chain(size_t regions, size_t cells_per_axis)
{
    ArrReal coordinates;
    for (size_t i = 0; i <= cells_per_axis; ++i) coordinates.push_back(real_t(i) / cells_per_axis);
    std::vector<MultiRegionMesh::Region> blocks;
    std::vector<MultiRegionMesh::Interface> interfaces;
    for (size_t r = 0; r < regions; ++r)
    {
        auto x = coordinates;
        for (auto& value : x) value += real_t(r);
        blocks.push_back(cartesian_region(std::to_string(r), {{x, coordinates, coordinates}}));
        if (r)
        {
            // Even regions retain every native face, including at nonzero
            // composite offsets; odd regions lose both opposite seam patches.
            if (r % 2) interfaces.push_back(StructuredPatchInterface{{r-1, 1}, {r, 0}});
            else interfaces.push_back(StructuredPatchInterface{{r, 0}, {r-1, 1}});
        }
    }
    return std::make_shared<MultiRegionMesh>(std::move(blocks), std::move(interfaces));
}

void check_canonical_incidence(const MultiRegionMesh& mesh)
{
    std::vector<unsigned> incidence(mesh.num_faces());
    for (size_t c = 0; c < mesh.num_cells(); ++c)
    {
        MeshUtils::Vec3 closure{};
        for (const auto face : mesh.cell_faces(c))
        {
            ++incidence[face];
            closure = closure + mesh.face_area_vector_outward(face, c);
        }
        EXPECT_NEAR(closure.norm(), 0, 1e-12);
    }
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        EXPECT_EQ(mesh.canonical_face(mesh.native_face(f)), f);
        EXPECT_EQ(incidence[f], mesh.is_exterior_face(f) ? 1U : 2U);
    }
}

// Independently mark removed native faces from the public interface descriptions;
// canonical IDs must enumerate every remaining face in region/native-ID order.
// This catches a mutually consistent but reordered rank/select implementation.
void check_reference_face_order(const MultiRegionMesh& mesh)
{
    std::vector<std::vector<bool>> removed;
    for (size_t r = 0; r < mesh.regions().size(); ++r)
        removed.emplace_back(mesh.region_layout(r).faces, false);
    for (const auto& interface : mesh.interfaces())
        if (const auto* structured = std::get_if<StructuredPatchInterface>(&interface))
        {
            const auto& patch = structured->second;
            const auto& layout = mesh.region_layout(patch.region);
            const OrthogonalIndexer indexer(static_cast<unsigned>(layout.extents[0]),
                static_cast<unsigned>(layout.extents[1]), static_cast<unsigned>(layout.extents[2]),
                layout.periodic[0], layout.periodic[1], layout.periodic[2]);
            for (size_t f = 0; f < layout.faces; ++f)
            {
                const auto face = indexer.face_id(f);
                if (face.orientation != patch.boundary / 2) continue;
                const std::array<size_t, 3> coordinate{face.i, face.j, face.k};
                if (coordinate[face.orientation] != (patch.boundary % 2 ? layout.extents[face.orientation] : 0)) continue;
                size_t tangent = 0;
                bool in_patch = true;
                for (size_t axis = 0; axis < 3; ++axis)
                    if (axis != face.orientation)
                    {
                        in_patch = in_patch && coordinate[axis] >= patch.begin[tangent]
                            && coordinate[axis] < patch.begin[tangent] + patch.extent[tangent];
                        ++tangent;
                    }
                if (in_patch) removed[patch.region][f] = true;
            }
        }
        else if (const auto* explicit_interface = std::get_if<ExplicitConformingInterface>(&interface))
            for (const auto& pair : explicit_interface->faces) removed[explicit_interface->second_region][pair.second] = true;
        else
        {
            const auto& refined = std::get<NonconformingInterface>(interface);
            for (const auto& group : refined.faces) removed[refined.coarse_region][group.coarse_face] = true;
        }
    size_t canonical = 0;
    for (size_t r = 0; r < removed.size(); ++r)
        for (size_t native = 0; native < removed[r].size(); ++native)
            if (!removed[r][native])
            {
                ASSERT_EQ(mesh.native_face(canonical), (RegionFace{r, native}));
                ASSERT_EQ(mesh.canonical_face({r, native}), canonical);
                ++canonical;
            }
    EXPECT_EQ(canonical, mesh.num_faces());
    EXPECT_THROW(mesh.native_face(canonical), std::out_of_range);
}

Vec3D<ArrReal> unit_edges(const std::array<size_t, 3>& dimensions)
{
    Vec3D<ArrReal> edges;
    for (size_t axis = 0; axis < 3; ++axis)
        for (size_t i = 0; i <= dimensions[axis]; ++i)
            edges[axis].push_back(real_t(i) / dimensions[axis]);
    return edges;
}

SP<MultiRegionMesh> full_side_star(const std::array<size_t, 3>& dimensions, unsigned removed_sides)
{
    const auto edges = unit_edges(dimensions);
    std::vector<MultiRegionMesh::Region> regions{cartesian_region("center", edges)};
    std::vector<MultiRegionMesh::Interface> interfaces;
    for (int boundary = 0; boundary < 6; ++boundary)
        if (removed_sides & (1U << boundary))
        {
            auto adjacent = edges;
            for (auto& coordinate : adjacent[boundary/2]) coordinate += boundary % 2 ? 1 : -1;
            const auto r = regions.size();
            regions.push_back(cartesian_region(std::to_string(r), adjacent));
            interfaces.push_back(StructuredPatchInterface{{r, boundary^1}, {0, boundary}});
        }
    return std::make_shared<MultiRegionMesh>(std::move(regions), std::move(interfaces));
}
}

TEST(RegionInterfaceIndexTest, FullSideSelectionMatchesReferenceForEverySideSubset)
{
    for (const std::array<size_t, 3> dimensions : {std::array<size_t, 3>{1,1,1}, {1,2,3}, {2,3,4}, {4,2,1}})
        for (unsigned mask = 0; mask < 64; ++mask)
        {
            SCOPED_TRACE(testing::Message() << "dimensions=" << dimensions[0] << ',' << dimensions[1]
                << ',' << dimensions[2] << " sides=" << mask);
            check_reference_face_order(*full_side_star(dimensions, mask));
        }
    const auto small = full_side_star({1,1,1}, 63)->storage_report();
    const auto large = full_side_star({8,9,10}, 63)->storage_report();
    EXPECT_EQ(small.indexing, large.indexing);
    EXPECT_EQ(small.interfaces, large.interfaces);
    EXPECT_EQ(large.compatibility, 0U);
}

TEST(RegionInterfaceIndexTest, PatchParametrizationsPreserveFullAndPartialFaceOrder)
{
    const std::array<size_t, 3> dimensions{3,4,5};
    const auto edges = unit_edges(dimensions);
    for (int boundary = 0; boundary < 6; ++boundary)
        for (const bool partial : {false, true})
            for (unsigned mapping = 0; mapping < 64; ++mapping)
            {
                SCOPED_TRACE(testing::Message() << "boundary=" << boundary << " partial=" << partial << " mapping=" << mapping);
                auto adjacent = edges;
                for (auto& coordinate : adjacent[boundary/2]) coordinate += boundary % 2 ? 1 : -1;
                StructuredPatchInterface seam{{1, boundary^1}, {0, boundary}};
                if (partial)
                {
                    size_t tangent = 0;
                    for (size_t axis = 0; axis < 3; ++axis)
                        if (axis != static_cast<size_t>(boundary/2))
                        {
                            seam.second.begin[tangent] = 1;
                            seam.second.extent[tangent] = dimensions[axis] - 2;
                            adjacent[axis] = ArrReal(edges[axis].begin()+1, edges[axis].end()-1);
                            ++tangent;
                        }
                }
                if (mapping & 1U) seam.first.axes = {1,0};
                if (mapping & 2U) seam.second.axes = {1,0};
                seam.first.reversed = {(mapping & 4U) != 0, (mapping & 8U) != 0};
                seam.second.reversed = {(mapping & 16U) != 0, (mapping & 32U) != 0};
                for (size_t output = 0; output < 2; ++output)
                {
                    const auto input = seam.first.axes[0] == seam.second.axes[output] ? 0U : 1U;
                    seam.permutation[output] = input;
                    seam.reversed[output] = seam.first.reversed[input] != seam.second.reversed[output];
                }
                std::vector<MultiRegionMesh::Region> regions{cartesian_region("center", edges), cartesian_region("adjacent", adjacent)};
                std::vector<MultiRegionMesh::Interface> interfaces{seam};
                if (partial && mapping == 0)
                {
                    // A full side in another orientation must still contribute
                    // to the cached partial-patch fallback's prefix counts.
                    const int other_boundary = (boundary + 2) % 6;
                    auto other_edges = edges;
                    for (auto& coordinate : other_edges[other_boundary/2]) coordinate += other_boundary % 2 ? 1 : -1;
                    regions.push_back(cartesian_region("other", other_edges));
                    interfaces.push_back(StructuredPatchInterface{{2, other_boundary^1}, {0, other_boundary}});
                }
                MultiRegionMesh mesh(std::move(regions), std::move(interfaces));
                check_reference_face_order(mesh);
            }
}

TEST(RegionInterfaceIndexTest, SelfPeriodicSelectionCoversEveryOrientationAndNativePeriodicTangents)
{
    const auto edges = unit_edges({2,3,4});
    for (size_t orientation = 0; orientation < 3; ++orientation)
        for (const bool reverse_owner : {false, true})
        {
            StructuredPatchInterface seam{{0, static_cast<int>(2*orientation + reverse_owner)},
                {0, static_cast<int>(2*orientation + !reverse_owner)}};
            const real_t distance = reverse_owner ? -1 : 1;
            seam.periodic_translation = MeshUtils::Vec3{orientation == 0 ? distance : 0,
                orientation == 1 ? distance : 0, orientation == 2 ? distance : 0};
            MultiRegionMesh mesh({cartesian_region("periodic", edges)}, {seam});
            check_reference_face_order(mesh);
        }

    StructuredPatchInterface seam{{0,5}, {1,4}};
    auto lower = edges;
    for (auto& radius : lower[0]) radius += 1;
    for (auto& theta : lower[1]) theta *= 2 * std::acos(-1.0);
    auto upper = lower;
    for (auto& z : upper[2]) z += 1;
    auto lower_native = std::make_shared<OrthogonalCylindrial3D>(lower);
    auto upper_native = std::make_shared<OrthogonalCylindrial3D>(upper);
    MultiRegionMesh tangential_periodic({native_region("lower", lower_native), native_region("upper", upper_native)}, {seam});
    check_reference_face_order(tangential_periodic);
    check_reference_face_order(*test::cylindrical_regions());
}

TEST(RegionInterfaceIndexTest, ManyRegionsRetainOnlyIncidentSidesAndDirectOrdinals)
{
    constexpr size_t regions = 17, n = 2;
    const auto mesh = chain(regions, n);
    size_t offset = 0, sides = 0;
    for (size_t r = 0; r < regions; ++r)
    {
        const size_t incident = r == 0 || r+1 == regions ? 1 : 2;
        EXPECT_EQ(mesh->region_interface_side_count(r), incident);
        sides += mesh->region_interface_side_count(r);
        EXPECT_EQ(mesh->removed_native_face_count(r), r % 2 ? incident*n*n : 0);
        const auto native_faces = mesh->region_layout(r).faces;
        if (r % 2 == 0)
            for (size_t face = 0; face < native_faces; ++face)
            {
                EXPECT_EQ(mesh->native_face(offset+face), (RegionFace{r, face}));
                EXPECT_EQ(mesh->canonical_face({r, face}), offset+face);
            }
        offset += native_faces - mesh->removed_native_face_count(r);
    }
    EXPECT_EQ(sides, 2 * (regions-1));
    EXPECT_EQ(offset, mesh->num_faces());
    check_canonical_incidence(*mesh);

    const auto finer = chain(regions, 5);
    EXPECT_EQ(mesh->storage_report().indexing, finer->storage_report().indexing);
    EXPECT_EQ(mesh->storage_report().interfaces, finer->storage_report().interfaces);
    EXPECT_EQ(finer->storage_report().compatibility, 0U);
    EXPECT_THROW(mesh->region_interface_side_count(regions), std::out_of_range);
    check_reference_face_order(*mesh);
}

TEST(RegionInterfaceIndexTest, BoundaryBucketsCoverAllOrientations)
{
    std::vector<MultiRegionMesh::Region> regions;
    std::vector<MultiRegionMesh::Interface> interfaces;
    for (size_t z = 0; z < 2; ++z)
        for (size_t y = 0; y < 2; ++y)
            for (size_t x = 0; x < 2; ++x)
            {
                const size_t r = x + 2*y + 4*z;
                regions.push_back(cartesian_region(std::to_string(r),
                    {{{real_t(x), real_t(x+1)}, {real_t(y), real_t(y+1)}, {real_t(z), real_t(z+1)}}}));
                if (x) interfaces.push_back(StructuredPatchInterface{{r-1, 1}, {r, 0}});
                if (y) interfaces.push_back(StructuredPatchInterface{{r-2, 3}, {r, 2}});
                if (z) interfaces.push_back(StructuredPatchInterface{{r-4, 5}, {r, 4}});
            }
    MultiRegionMesh mesh(std::move(regions), std::move(interfaces));
    EXPECT_EQ(mesh.num_faces(), 36U);
    for (size_t r = 0; r < 8; ++r)
    {
        EXPECT_EQ(mesh.region_interface_side_count(r), 3U);
        EXPECT_EQ(mesh.removed_native_face_count(r), (r&1) + ((r>>1)&1) + ((r>>2)&1));
    }
    for (size_t face = 0; face < mesh.num_faces(); ++face)
        if (mesh.is_boundary_face(face))
        {
            const auto native = mesh.native_face(face);
            EXPECT_TRUE(mesh.boundary_name(face).starts_with(mesh.region_name(native.region)+"/"));
        }
    check_canonical_incidence(mesh);
}

TEST(RegionInterfaceIndexTest, DisjointSubpatchesShareOneNativeBoundaryBucket)
{
    auto left = cartesian_region("left", {{{0,1}, {0,0.5,1,1.5,2}, {0,0.5,1}}});
    auto lower = cartesian_region("lower", {{{1,2}, {0,0.5,1}, {0,0.5,1}}});
    auto upper = cartesian_region("upper", {{{1,2}, {1,1.5,2}, {0,0.5,1}}});
    const StructuredPatchInterface low{{1,0}, {0,1,{0,0},{2,2}}};
    const StructuredPatchInterface high{{2,0}, {0,1,{2,0},{2,2}}};
    const StructuredPatchInterface right{{1,3}, {2,2}};
    MultiRegionMesh mesh({left, lower, upper}, {low, high, right});
    EXPECT_EQ(mesh.removed_native_face_count(0), 8U);
    EXPECT_EQ(mesh.removed_native_face_count(1), 0U);
    for (size_t i = 0; i < 2; ++i)
    {
        const auto& interface = std::get<StructuredPatchInterface>(mesh.interfaces()[i]);
        for (size_t z = 0; z < 2; ++z)
            for (size_t y = 0; y < 2; ++y)
                EXPECT_EQ(mesh.canonical_face({interface.first.region, mesh.patch_face(interface.first,{y,z})}),
                    mesh.canonical_face({0, mesh.patch_face(interface.second,{y,z})}));
    }
    check_canonical_incidence(mesh);
    check_reference_face_order(mesh);
    EXPECT_THROW((MultiRegionMesh({left, lower, upper}, {low, high, right, low})), std::invalid_argument);
}

TEST(RegionInterfaceIndexTest, IrregularDirectoryPreservesExplicitAndCoarseFineTransfer)
{
    const auto explicit_mesh = test::mixed_regions();
    const auto& explicit_interface = std::get<ExplicitConformingInterface>(explicit_mesh->interfaces()[0]);
    EXPECT_EQ(explicit_mesh->removed_native_face_count(explicit_interface.second_region), explicit_interface.faces.size());
    for (const auto& [first, second] : explicit_interface.faces)
        EXPECT_EQ(explicit_mesh->canonical_face({explicit_interface.first_region, first}),
            explicit_mesh->canonical_face({explicit_interface.second_region, second}));
    check_canonical_incidence(*explicit_mesh);
    check_reference_face_order(*explicit_mesh);

    const auto refined = test::coarse_fine_regions();
    const auto& interface = std::get<NonconformingInterface>(refined->interfaces()[0]);
    EXPECT_EQ(refined->removed_native_face_count(interface.coarse_region), interface.faces.size());
    EXPECT_EQ(refined->removed_native_face_count(interface.fine_region), 0U);
    for (const auto& group : interface.faces)
    {
        const RegionFace coarse{interface.coarse_region, group.coarse_face};
        const auto logical = refined->logical_faces(coarse);
        ASSERT_EQ(logical.size(), group.fine_faces.size());
        real_t coefficient = 0;
        for (const auto weight : refined->face_flux_weights(coarse)) coefficient += weight.coefficient;
        EXPECT_NEAR(coefficient, -1, 1e-14);
        for (size_t i = 0; i < logical.size(); ++i)
        {
            EXPECT_EQ(logical[i], refined->canonical_face({interface.fine_region, group.fine_faces[i]}));
            EXPECT_EQ(refined->native_cell(refined->neighbor_cell(logical[i])).first, interface.coarse_region);
        }
        EXPECT_THROW(refined->canonical_face(coarse), std::invalid_argument);
    }
    check_canonical_incidence(*refined);
    check_reference_face_order(*refined);
}

TEST(RegionInterfaceIndexTest, SelfPeriodicSidesPreserveAdjacentImageDistances)
{
    auto region = cartesian_region("periodic", {{{0,0.5,1,1.5,2}, {0,0.5,1}, {0,0.5,1}}});
    StructuredPatchInterface seam{{0,0}, {0,1}};
    seam.periodic_translation = MeshUtils::Vec3{2,0,0};
    MultiRegionMesh mesh({region}, {seam});
    EXPECT_EQ(mesh.region_interface_side_count(0), 2U);
    EXPECT_EQ(mesh.removed_native_face_count(0), 4U);
    const auto& interface = std::get<StructuredPatchInterface>(mesh.interfaces()[0]);
    for (size_t z = 0; z < 2; ++z)
        for (size_t y = 0; y < 2; ++y)
        {
            const auto face = mesh.canonical_face({0, mesh.patch_face(interface.first, {y,z})});
            EXPECT_EQ(face, mesh.canonical_face({0, mesh.patch_face(interface.second, {y,z})}));
            EXPECT_FALSE(mesh.is_exterior_face(face));
            const auto owner = mesh.owner_cell(face), neighbor = mesh.neighbor_cell(face);
            EXPECT_NEAR(mesh.cell_center_vector(face, owner).x, -0.5, 1e-14);
            EXPECT_NEAR(mesh.cell_to_face_distance(face, neighbor), 0.25, 1e-14);
        }
    check_canonical_incidence(mesh);
    EXPECT_THROW((MultiRegionMesh({region}, {seam, seam})), std::invalid_argument);
}

TEST(RegionDescriptorValidationTest, CartesianExtremaPreserveVolumeArithmeticChecks)
{
    // The products overflow/underflow at the first multiplication, even
    // though a reassociated mathematical product would be representable.
    for (const auto& edges : std::vector<Vec3D<ArrReal>>{
        {{{0,1e200}, {0,1e200}, {0,1e-100}}},
        {{{0,1e-200}, {0,1e-200}, {0,1e200}}},
        {{{0,1e-200,1}, {0,1e-200,1}, {0,1}}}})
    {
        auto independent = cartesian_region("independent", edges);
        auto native = std::make_shared<OrthogonalCartesian3D>(edges);
        EXPECT_THROW((MultiRegionMesh({independent}, {})), std::invalid_argument);
        EXPECT_THROW((MultiRegionMesh({native_region("native",native)}, {})), std::invalid_argument);
    }
    auto valid = cartesian_region("wide_scales", {{{0,1e-150}, {0,1e150}, {0,2}}});
    MultiRegionMesh mesh({valid}, {});
    EXPECT_NEAR(mesh.cell_volume(0), 2, 1e-14);
}

TEST(RegionDescriptorValidationTest, ExtrusionChecksEveryBaseCellAtExtremeLayerWidths)
{
    auto topology = std::make_shared<const ExtrudedTopology>(3, Arr<Arr<unsigned>>{{0,1,2}}, 2);
    for (const auto [length, thin, thick] : std::vector<std::array<real_t,3>>{
        {1e-10, 1e-310, 1}, {1e80, 1, 1e160}})
    {
        const Arr<MeshUtils::Vec3> nodes{{0,0,0}, {length,0,0}, {0,length,0}};
        const ArrReal z{0, thin, thick};
        auto independent = extruded_region("independent", topology, nodes, z);
        auto native = std::make_shared<SemiStructuredXY_Z>(nodes, Arr<Arr<unsigned>>{{0,1,2}}, z);
        EXPECT_THROW((MultiRegionMesh({independent}, {})), std::invalid_argument);
        EXPECT_THROW((MultiRegionMesh({native_region("native",native)}, {})), std::invalid_argument);
    }
    check_canonical_incidence(*test::independent_extruded_regions());
}
