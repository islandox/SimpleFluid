/**
 * @file testMultiRegionMesh.cc
 * @author islandox
 * @brief Provider, interface, canonicalization and allocation regressions.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <filesystem>
#include <fstream>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Handle = MeshHandle<>;
static_assert(!GeometryProvider<RectilinearTopology>);
static_assert(!TopologyProvider<RectilinearGeometry>);
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
}

TEST(RegionProvidersTest, SharedTopologyIndependentGeometryAndLifetime)
{
    auto topology = std::make_shared<const RectilinearTopology>(std::make_shared<const OrthoMeshTopo>(1,1,1,false,false,false,
        OrthoMeshTopo::BoundaryNames{"xmin","xmax","ymin","ymax","zmin","zmax"}));
    auto g1 = std::make_shared<const RectilinearGeometry>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto g2 = std::make_shared<const RectilinearGeometry>(Vec3D<ArrReal>{{{0,2},{0,3},{0,4}}});
    CartesianRegion a("a", topology, g1), b("b", topology, g2);
    topology.reset(); g1.reset(); g2.reset();
    EXPECT_EQ(&a.topology(), &b.topology());
    EXPECT_TRUE(std::ranges::equal(a.topology().cell_faces(0), b.topology().cell_faces(0)));
    EXPECT_DOUBLE_EQ(a.geometry().cell_volume(0), 1);
    EXPECT_DOUBLE_EQ(b.geometry().cell_volume(0), 24);
    EXPECT_DOUBLE_EQ(b.geometry().face_area(0), 12);
    EXPECT_DOUBLE_EQ(b.geometry().cell_centroid(0).z, 2);
    auto mismatch = std::make_shared<const RectilinearGeometry>(Vec3D<ArrReal>{{{0,1,2},{0,1},{0,1}}});
    EXPECT_THROW((CartesianRegion("bad", std::make_shared<const RectilinearTopology>(a.topology()), mismatch)), std::invalid_argument);
}

TEST(MultiRegionMeshTest, ImplicitConstruction)
{
    EXPECT_NO_THROW(test::two_regions());
}
TEST(MultiRegionMeshTest, CanonicalIncidenceMatchesSingleBlock)
{
    const auto mesh = test::two_regions();
    OrthogonalCartesian3D reference({{{0,0.5,1,1.5,2},{0,0.5,1},{0,0.5,1}}});
    EXPECT_EQ(mesh->num_cells(), reference.num_cells()); EXPECT_EQ(mesh->num_faces(), reference.num_faces());
    std::vector<unsigned> incidence(mesh->num_faces());
    for (size_t c = 0; c < mesh->num_cells(); ++c)
    {
        const auto faces = mesh->cell_faces(c);
        const auto nested = mesh->cell_faces((c + 1) % mesh->num_cells());
        EXPECT_EQ(faces.size(), 6U); EXPECT_EQ(nested.size(), 6U);
        MeshUtils::Vec3 closure{};
        for (const auto f : faces) { ++incidence[f]; closure = closure + mesh->face_area_vector_outward(f,c); }
        EXPECT_NEAR(closure.norm(), 0, 1e-13);
    }
    size_t seam = 0;
    for (size_t f = 0; f < mesh->num_faces(); ++f)
    {
        EXPECT_EQ(mesh->canonical_face(mesh->native_face(f)), f);
        EXPECT_EQ(incidence[f], mesh->neighbor_cell(f) == mesh->invalid_cell_id() ? 1U : 2U);
        if (std::abs(mesh->face_centroid(f).x - 1) < 1e-12 && std::abs(mesh->face_normal(f).x) > 0.5)
        {
            ++seam; EXPECT_FALSE(mesh->is_boundary_face(f));
            EXPECT_EQ(mesh->native_cell(mesh->owner_cell(f)).first, 0U);
            EXPECT_EQ(mesh->native_cell(mesh->neighbor_cell(f)).first, 1U);
        }
    }
    EXPECT_EQ(seam, 4U);
    Handle handle(std::static_pointer_cast<const MultiRegionMesh>(mesh));
    EXPECT_EQ(handle.connectivity_storage_bytes(), 0U); EXPECT_FALSE(handle.has_materialized_indexer());
    EXPECT_FALSE(handle.has_mutable_geometry());
    EXPECT_THROW(handle.visit_mutable([](auto&){}), std::logic_error);
    for (const auto& [b,batch] : handle.boundary_batches()) EXPECT_NE(handle.boundary_batch_name(b).find('/'), std::string::npos);
    handle.materialize_cell_faces();
    EXPECT_TRUE(handle.has_materialized_connectivity());
    EXPECT_TRUE(std::ranges::equal(handle.faces(0), handle.materialized_faces(0)));
}
TEST(MultiRegionMeshTest, StructuredReversalPermutationAndRoundTrip)
{
    auto base = test::two_regions(3);
    auto regions = base->regions();
    StructuredPatchInterface interface{{0,1},{1,0}};
    interface.second.axes = {1,0}; interface.second.reversed = {true,false};
    interface.permutation = {1,0}; interface.reversed = {true,false};
    MultiRegionMesh mesh(regions, {interface});
    const auto& actual = std::get<StructuredPatchInterface>(mesh.interfaces()[0]);
    for (size_t j=0;j<3;++j) for (size_t i=0;i<3;++i)
    {
        const std::array<size_t,2> p{i,j}; EXPECT_EQ(actual.inverse(actual.map(p)), p);
        EXPECT_EQ(mesh.canonical_face({0,mesh.patch_face(actual.first,p)}),
            mesh.canonical_face({1,mesh.patch_face(actual.second,actual.map(p))}));
    }
    EXPECT_THROW(actual.map({3,0}), std::out_of_range);
    interface.permutation = {0,0}; EXPECT_THROW((MultiRegionMesh(regions,{interface})), std::invalid_argument);
}
TEST(MultiRegionMeshTest, RejectsDuplicateInvalidAndNoncoincidentInterfaces)
{
    const auto base = test::two_regions();
    auto regions = base->regions(); const auto interface = base->interfaces()[0];
    EXPECT_THROW((MultiRegionMesh(regions,{interface,interface})), std::invalid_argument);
    EXPECT_THROW((MultiRegionMesh(regions,{})), std::invalid_argument);
    EXPECT_THROW((MultiRegionMesh(regions,{StructuredPatchInterface{{0,8},{1,0}}})), std::invalid_argument);
    EXPECT_THROW((MultiRegionMesh(regions,{StructuredPatchInterface{{0,1},{2,0}}})), std::out_of_range);
    regions[1] = cartesian_region("right", {{{1.1,1.6,2.1},{0,0.5,1},{0,0.5,1}}});
    EXPECT_THROW((MultiRegionMesh(regions,{interface})), std::invalid_argument);
    regions[1] = cartesian_region("right", {{{1,1.5,2},{0,0.25,0.5,0.75,1},{0,0.5,1}}});
    EXPECT_THROW((MultiRegionMesh(regions,{interface})), std::invalid_argument);
}
TEST(MultiRegionMeshTest, MixedPrismSideAndExplicitRejections)
{
    auto mesh = test::mixed_regions(); auto reference = test::explicit_reference(*mesh);
    EXPECT_EQ(mesh->num_faces(), reference->num_faces());
    const auto& valid = std::get<ExplicitConformingInterface>(mesh->interfaces()[0]);
    EXPECT_EQ(valid.faces.size(), 2U);
    auto invalid = valid; invalid.faces.pop_back();
    EXPECT_THROW((MultiRegionMesh(mesh->regions(), {invalid})), std::invalid_argument);
    invalid = valid; invalid.faces[1] = invalid.faces[0];
    EXPECT_THROW((MultiRegionMesh(mesh->regions(), {invalid})), std::invalid_argument);
    invalid = valid; invalid.faces[0].second = std::numeric_limits<uint64_t>::max();
    EXPECT_THROW((MultiRegionMesh(mesh->regions(), {invalid})), std::out_of_range);
    EXPECT_NEAR(mesh->cell_volume(4), 0.25, 1e-14);
    EXPECT_EQ(mesh->cell_type(4), MeshUtils::CellType::TRIPRISM);
}
TEST(MultiRegionMeshTest, FourBlocksShareEdgesAndOnePressureDomain)
{
    std::vector<MultiRegionMesh::Region> r;
    for (size_t y=0;y<2;++y) for (size_t x=0;x<2;++x)
        r.push_back(cartesian_region(std::to_string(r.size()), {{{real_t(x),real_t(x+1)},{real_t(y),real_t(y+1)},{0,1}}}));
    MultiRegionMesh mesh(r,{StructuredPatchInterface{{0,1},{1,0}},StructuredPatchInterface{{2,1},{3,0}},
        StructuredPatchInterface{{0,3},{2,2}},StructuredPatchInterface{{1,3},{3,2}}});
    OrthogonalCartesian3D single({{{0,1,2},{0,1,2},{0,1}}});
    EXPECT_EQ(mesh.num_faces(), single.num_faces());
    for (size_t c=0;c<4;++c) EXPECT_EQ(mesh.cell_faces(c).size(),6U);
}
TEST(MultiRegionMeshTest, NativeAliasReplacementInvalidatesStaticObservation)
{
    auto native = std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto mesh = std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("one",native)}, std::vector<MultiRegionMesh::Interface>{});
    Handle handle(mesh);
    *native = OrthogonalCartesian3D(Vec3D<ArrReal>{{{0,2},{0,1},{0,1}}});
    EXPECT_THROW(handle.geometry_epoch(), std::logic_error);
    EXPECT_THROW(handle.cell_volume(0), std::logic_error);
}
TEST(MultiRegionMeshTest, OutputIsTemporaryAndLabelsRegionCells)
{
    auto mesh = test::mixed_regions(); Handle handle(mesh);
    const auto file = std::filesystem::temp_directory_path() / "simplefluid_regions_test.vtu";
    handle.export_vtu(file.string());
    std::ifstream stream(file); const std::string text((std::istreambuf_iterator<char>(stream)), {});
    EXPECT_NE(text.find("topology_region"), std::string::npos);
    EXPECT_NE(text.find("NumberOfCells=\"6\""), std::string::npos);
    EXPECT_FALSE(handle.has_materialized_connectivity());
    std::filesystem::remove(file);
}
TEST(MultiRegionMeshTest, StructuredCorrespondenceStorageIsDescriptorSized)
{
    const auto small = test::two_regions(2), larger = test::two_regions(7);
    EXPECT_EQ(small->storage_report().interfaces, larger->storage_report().interfaces);
    EXPECT_EQ(small->storage_report().indexing, larger->storage_report().indexing);
    EXPECT_EQ(larger->storage_report().compatibility, 0U);
}
TEST(RegionProvidersTest, PolygonalExtrusionRangesAreNotLimitedToSixFaces)
{
    SemiStructuredXY_Z mesh({{0,0,0},{1,0,0},{1.5,1,0},{0.5,2,0},{-0.5,1,0}},{{0,1,2,3,4}},{0,1,3});
    auto a = mesh.cell_faces({0,0}); auto b = mesh.cell_faces({0,1});
    EXPECT_EQ(a.size(), 7U); EXPECT_EQ(b.size(), 7U);
    EXPECT_EQ(a[1], b[0]); EXPECT_DOUBLE_EQ(mesh.cell_volume({0,1}), 2 * mesh.cell_volume({0,0}));
}

TEST(MultiRegionMeshTest, EqualAreasAndCentroidsDoNotHideDifferentFaceShapes)
{
    auto left=cartesian_region("left",{{{0,1},{0,1},{0,1}}});
    auto right=cartesian_region("right",{{{1,2},{-0.5,1.5},{0.25,0.75}}});
    try
    {
        MultiRegionMesh invalid({left,right},{StructuredPatchInterface{{0,1},{1,0}}});
        FAIL()<<"Noncoincident vertices were accepted.";
    }
    catch(const std::invalid_argument& e) { EXPECT_NE(std::string(e.what()).find("face shapes"),std::string::npos); }
    auto overlap=cartesian_region("overlap",{{{0,1},{0,1},{0,1}}});
    EXPECT_THROW((MultiRegionMesh({left,overlap},{StructuredPatchInterface{{0,1},{1,1}}})),std::invalid_argument);
}
TEST(MultiRegionMeshTest, CountsSharedTopologyOnce)
{
    auto first=cartesian_region("first",{{{0,1},{0,1},{0,1}}});
    auto topology=std::make_shared<const RectilinearTopology>(first.topology());
    auto a=std::make_shared<const RectilinearGeometry>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto b=std::make_shared<const RectilinearGeometry>(Vec3D<ArrReal>{{{1,2},{0,1},{0,1}}});
    MultiRegionMesh mesh({CartesianRegion("a",topology,a),CartesianRegion("b",topology,b)},
        {StructuredPatchInterface{{0,1},{1,0}}});
    EXPECT_EQ(mesh.storage_report().topology,topology->storage_report().topology);
    // Initial/current two-point axial parameters belong to the composite.
    EXPECT_EQ(mesh.storage_report().geometry,a->storage_report().geometry+b->storage_report().geometry+4*sizeof(real_t));
}
TEST(MultiRegionMeshTest, ExactChildEpochSnapshotDetectsTheLowerEpochChanging)
{
    auto a=std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto b=std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{1,2},{0,1},{0,1}}});
    for(int i=0;i<4;++i) *b=OrthogonalCartesian3D(Vec3D<ArrReal>{{{1,2},{0,1},{0,1}}});
    auto composite=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("a",a),native_region("b",b)},
        std::vector<MultiRegionMesh::Interface>{StructuredPatchInterface{{0,1},{1,0}}});
    const auto max_before=std::max(a->geometry_epoch(),b->geometry_epoch());
    *a=OrthogonalCartesian3D(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    EXPECT_EQ(std::max(a->geometry_epoch(),b->geometry_epoch()),max_before);
    EXPECT_THROW(composite->geometry_epoch(),std::logic_error);
}
TEST(MultiRegionMeshTest, PartialPatchesUseArithmeticRankSelect)
{
    auto a=cartesian_region("a",{{{0,1},{0,1,2},{0,1,2}}});
    auto b=cartesian_region("b",{{{1,2},{1,2},{0,1,2}}});
    StructuredPatchInterface interface{{0,1,{1,0},{1,2}},{1,0}};
    MultiRegionMesh mesh({a,b},{interface});
    std::vector<unsigned> incidence(mesh.num_faces());
    for(size_t c=0;c<mesh.num_cells();++c) for(auto f:mesh.cell_faces(c)) ++incidence[f];
    for(size_t f=0;f<mesh.num_faces();++f)
    {
        EXPECT_EQ(mesh.canonical_face(mesh.native_face(f)),f);
        EXPECT_EQ(incidence[f],mesh.is_exterior_face(f)?1U:2U);
    }
}

TEST(RegionProvidersTest, RetargetedProviderAndMovedNativeCannotEvadeStaticSnapshot)
{
    using Provider=NativeRegionProvider<OrthogonalCartesian3D>;
    auto original=std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    auto replacement=std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0,2},{0,1},{0,1}}});
    auto provider=std::make_shared<Provider>(original);
    NativeIsoRegion<OrthogonalCartesian3D> region("one",provider,provider);
    *provider=Provider(replacement);
    EXPECT_THROW(region.validate_static(),std::logic_error);
    auto moved_region=native_region("moved",original);
    OrthogonalCartesian3D moved(std::move(*original));
    EXPECT_THROW(moved_region.validate_static(),std::logic_error);
}

TEST(MultiRegionMeshTest, NativeUnstructuredConstituentUsesOriginalArrays)
{
    MultiRegionMesh source({cartesian_region("source",{{{0,1},{0,1},{0,1}}})},{});
    auto native=test::explicit_reference(source);
    size_t face=0; int boundary=-1;
    for(size_t f=0;f<native->num_faces();++f)
        if(std::abs(native->face_centroid(f).x-1)<1e-12) { face=f; boundary=native->boundary_id(f); }
    auto left=native_region("explicit",native);
    EXPECT_EQ(left.topology().storage_identity(),native.get());
    EXPECT_EQ(left.geometry().storage_identity(),native.get());
    auto mesh=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{left,
        cartesian_region("cartesian",{{{1,2},{0,1},{0,1}}})},
        std::vector<MultiRegionMesh::Interface>{ExplicitConformingInterface{0,1,boundary,0,{{face,0}}}});
    EXPECT_EQ(mesh->num_faces(),11U);
    EXPECT_EQ(mesh->canonical_face({0,face}),mesh->canonical_face({1,0}));
    EXPECT_DOUBLE_EQ(mesh->cell_volume(0)+mesh->cell_volume(1),2);
    using Provider=NativeRegionProvider<UnstructuredMesh>;
    auto unrelated=std::make_shared<UnstructuredMesh>(*native);
    EXPECT_THROW((NativeIsoRegion<UnstructuredMesh>("bad",std::make_shared<const Provider>(native),
        std::make_shared<const Provider>(unrelated))),std::invalid_argument);
    *native=UnstructuredMesh(*native);
    EXPECT_THROW(mesh->geometry_epoch(),std::logic_error);
}

TEST(MultiRegionMeshTest, BoundaryNameCollisionsRequireExplicitMergePolicy)
{
    auto topology=std::make_shared<const RectilinearTopology>(std::make_shared<const OrthoMeshTopo>(1,1,1,false,false,false,
        OrthoMeshTopo::BoundaryNames{"wall","wall","wall","wall","wall","wall"}));
    auto geometry=std::make_shared<const RectilinearGeometry>(Vec3D<ArrReal>{{{0,1},{0,1},{0,1}}});
    const CartesianRegion region("region",topology,geometry);
    EXPECT_THROW((MultiRegionMesh({region},{})),std::invalid_argument);
    MultiRegionMesh merged({region},{},{},MultiRegionMesh::BoundaryNamePolicy::MergeMatchingNames);
    EXPECT_EQ(merged.num_boundary_batches(),1);
    auto faces=merged.boundary_face_batch(0);
    EXPECT_EQ(std::ranges::distance(faces),6);
    EXPECT_EQ(merged.boundary_batch_name(0),"wall");
}

TEST(MultiRegionMeshTest, NonconformingFacesAreCanonicalConservativeSubfaces)
{
    auto mesh=test::coarse_fine_regions();
    const auto& interface=std::get<NonconformingInterface>(mesh->interfaces()[0]);
    EXPECT_EQ(mesh->cell_faces(0).size(),7U);
    std::vector<unsigned> incidence(mesh->num_faces());
    for(size_t c=0;c<mesh->num_cells();++c)
    {
        MeshUtils::Vec3 closure{};
        for(auto f:mesh->cell_faces(c)) { ++incidence[f]; closure=closure+mesh->face_area_vector_outward(f,c); }
        EXPECT_NEAR(closure.norm(),0,1e-13);
    }
    for(size_t f=0;f<mesh->num_faces();++f) EXPECT_EQ(incidence[f],mesh->is_exterior_face(f)?1U:2U);
    for(const auto& group:interface.faces)
    {
        const RegionFace native{0,group.coarse_face};
        EXPECT_THROW(mesh->canonical_face(native),std::invalid_argument);
        EXPECT_EQ(mesh->logical_faces(native).size(),2U);
        real_t sum=0;
        for(const auto weight:mesh->face_flux_weights(native)) { EXPECT_DOUBLE_EQ(weight.coefficient,-0.5); sum+=weight.coefficient; }
        EXPECT_DOUBLE_EQ(sum,-1.0);
        EXPECT_DOUBLE_EQ(mesh->restrict_face_flux(native,[](uint64_t){return -1.5;}),3.0);
    }
    EXPECT_THROW(test::coarse_fine_regions(0.9),std::invalid_argument);
    auto bad=interface; bad.faces[0].fine_faces.pop_back();
    EXPECT_THROW((MultiRegionMesh(mesh->regions(),{bad})),std::invalid_argument);
    bad=interface; bad.faces[0].fine_faces[1]=bad.faces[0].fine_faces[0];
    EXPECT_THROW((MultiRegionMesh(mesh->regions(),{bad})),std::invalid_argument);
}
TEST(MultiRegionMeshTest, PlanarSubdivisionRejectsOverlapAndNonplanarity)
{
    auto rectangle=[](real_t y0,real_t y1,real_t sign)
    {
        return InterfaceFacePolygon{{0,(y0+y1)/2,0.5},{sign*(y1-y0),0,0},y1-y0,
            {{0,y0,0},{0,y1,0},{0,y1,1},{0,y0,1}}};
    };
    const auto coarse=rectangle(0,1,1);
    std::vector<InterfaceFacePolygon> fine{rectangle(0,0.5,-1),rectangle(0.5,1,-1)};
    EXPECT_NO_THROW(validate_planar_subdivision(coarse,fine,1e-12,1e-10));
    fine[1]=rectangle(0.4,0.9,-1);
    EXPECT_THROW(validate_planar_subdivision(coarse,fine,1e-12,1e-10),std::invalid_argument);
    fine[1]=rectangle(0.5,1,-1); fine[1].vertices[0].x=0.1;
    EXPECT_THROW(validate_planar_subdivision(coarse,fine,1e-12,1e-10),std::invalid_argument);
}

TEST(RegionProvidersTest, IndependentExtrusionsShareTopologyAndMatchAnalyticMetrics)
{
    auto topology=std::make_shared<const ExtrudedTopology>(3,Arr<Arr<unsigned>>{{0,1,2}},2);
    auto a=extruded_region("a",topology,{{0,0,0},{1,0,0},{0,1,0}},{0,1,3});
    auto b=extruded_region("b",topology,{{0,0,0},{2,0,0},{0,3,0}},{0,2,6});
    EXPECT_EQ(&a.topology(),&b.topology());
    EXPECT_DOUBLE_EQ(a.geometry().cell_volume(0),0.5); EXPECT_DOUBLE_EQ(b.geometry().cell_volume(0),6);
    EXPECT_DOUBLE_EQ(b.geometry().cell_volume(1),12);
    EXPECT_NEAR(b.geometry().cell_centroid(0).x,2.0/3,1e-14);
    EXPECT_DOUBLE_EQ(b.geometry().cell_centroid(1).z,4);
    EXPECT_TRUE(std::ranges::equal(a.topology().cell_faces(1),b.topology().cell_faces(1)));
    EXPECT_THROW((ExtrudedGeometry(topology,{{0,0,0},{1,0,0},{0,1,0}},{0,1})),std::invalid_argument);
    EXPECT_THROW((ExtrudedGeometry(topology,{{0,0,0},{-1,0,0},{0,1,0}},{0,1,2})),std::invalid_argument);
    const SemiStructuredXY_Z native({{0,0,0},{1,0,0},{0,1,0}},{{0,1,2}},{0,1,3});
    for(size_t f=0;f<a.layout().faces;++f)
    {
        EXPECT_NEAR(a.geometry().face_area(f),native.face_area(native.face_id(f)),1e-13);
        EXPECT_NEAR((a.geometry().face_centroid(f)-native.face_centroid(native.face_id(f))).norm(),0,1e-13);
    }
    auto composite=test::independent_extruded_regions();
    EXPECT_EQ(composite->num_cells(),4U); EXPECT_EQ(composite->num_faces(),16U);
}
TEST(MultiRegionMeshTest, TranslatedPeriodicFacesUseAdjacentImages)
{
    const auto mesh=test::periodic_regions();
    auto handle=std::make_shared<Handle>(mesh);
    size_t periodic=0;
    for(size_t f=0;f<mesh->num_faces();++f)
        if(std::abs(mesh->face_centroid(f).x)<1e-12 && std::abs(mesh->face_normal(f).x)>0.5)
        {
            ++periodic;
            const auto owner=mesh->owner_cell(f),neighbor=mesh->neighbor_cell(f);
            EXPECT_FALSE(mesh->is_boundary_face(f));
            EXPECT_NEAR(mesh->cell_to_face_distance(f,owner),0.25,1e-14);
            EXPECT_NEAR(mesh->cell_to_face_distance(f,neighbor),0.25,1e-14);
            EXPECT_NEAR(mesh->cell_center_vector(f,owner).x,-0.5,1e-14);
            EXPECT_NEAR(handle->cell_center_vector(f,neighbor).x,0.5,1e-14);
        }
    EXPECT_EQ(periodic,4U);
    for(size_t c=0;c<mesh->num_cells();++c)
    {
        const auto faces=mesh->cell_faces(c); const auto distances=mesh->face_distances(c);
        ASSERT_EQ(faces.size(),distances.size());
        for(size_t i=0;i<faces.size();++i) EXPECT_NEAR(distances[i],mesh->cell_to_face_distance(faces[i],c),1e-14);
    }
    const auto quality=evaluate_mesh_quality(*handle);
    EXPECT_GT(quality.minimum_normal_distance,0);
    EXPECT_NEAR(quality.maximum_non_orthogonality_degrees,0,1e-12);
}
TEST(MultiRegionMeshTest, CylindricalRingMatchesNativeGeometryAndIncidence)
{
    const auto mesh=test::cylindrical_regions();
    const real_t pi=std::acos(-1.0);
    OrthogonalCylindrial3D native(Vec3D<ArrReal>{{{1,1.5,2},{0,pi/4,pi/2,3*pi/4,pi,5*pi/4,3*pi/2,7*pi/4,2*pi},{0,0.5,1}}});
    EXPECT_EQ(mesh->num_cells(),native.num_cells()); EXPECT_EQ(mesh->num_faces(),native.num_faces());
    for(size_t c=0;c<mesh->num_cells();++c)
    {
        bool found=false;
        for(size_t n=0;n<native.num_cells();++n)
            if((mesh->cell_centroid(c)-native.cell_centroid(native.cell_id(n))).norm()<1e-12)
            { found=true; EXPECT_NEAR(mesh->cell_volume(c),native.cell_volume(native.cell_id(n)),1e-13); }
        EXPECT_TRUE(found);
    }
    EXPECT_EQ(mesh->vtu_topology()->cell_offsets.size(),mesh->num_cells());
}

TEST(MultiRegionMeshTest, DegenerateCylindricalOutputAndInvalidPeriodicTranslationAreRejected)
{
    const real_t pi=std::acos(-1.0);
    auto cylinder=std::make_shared<OrthogonalCylindrial3D>(Vec3D<ArrReal>{{{1,2},{0,pi,2*pi},{0,1}}});
    EXPECT_THROW((MultiRegionMesh({native_region("cylinder",cylinder)},{})),std::invalid_argument);
    auto reference=test::periodic_regions();
    auto interfaces=reference->interfaces();
    std::get<StructuredPatchInterface>(interfaces.back()).periodic_translation=MeshUtils::Vec3{1.9,0,0};
    EXPECT_THROW((MultiRegionMesh(reference->regions(),interfaces)),std::invalid_argument);
}

TEST(RegionProvidersTest, SharedMultiCellExtrusionPreservesNativeBaseNumbering)
{
    const Arr<MeshUtils::Vec3> nodes{{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
    const Arr<Arr<unsigned>> cells{{0,1,3},{1,2,3}}; const ArrReal z{0,0.2,0.7,1};
    auto topology=std::make_shared<const ExtrudedTopology>(4,cells,3);
    auto region=extruded_region("shared",topology,nodes,z);
    SemiStructuredXY_Z native(nodes,cells,z);
    for(size_t c=0;c<region.layout().cells;++c)
    {
        EXPECT_NEAR(region.geometry().cell_volume(c),native.cell_volume(native.cell_id(c)),1e-14);
        const auto actual=region.topology().cell_faces(c); const auto expected=native.cell_faces(native.cell_id(c));
        ASSERT_EQ(actual.size(),expected.size());
        for(size_t f=0;f<actual.size();++f) EXPECT_EQ(actual[f],native.face_local_id(expected[f]));
    }
    for(size_t f=0;f<region.layout().faces;++f)
        EXPECT_NEAR((region.geometry().face_area_vector(f)-native.face_area_vector(native.face_id(f))).norm(),0,1e-13);
}

TEST(MultiRegionMeshTest, FluxTransferConservesEvenAtTheGeometryMatchingTolerance)
{
    auto mesh=test::coarse_fine_regions(1.0-5e-11);
    const auto& interface=std::get<NonconformingInterface>(mesh->interfaces()[0]);
    for(const auto& group:interface.faces)
    {
        const RegionFace native{0,group.coarse_face};
        std::vector<real_t> flux(mesh->num_faces());
        for(const auto weight:mesh->face_flux_weights(native)) flux[weight.face]=3.0*weight.coefficient;
        EXPECT_NEAR(mesh->restrict_face_flux(native,[&](uint64_t face){return flux[face];}),3.0,2e-15);
    }
}

TEST(RegionProvidersTest, IndependentExtrusionRejectsSelfIntersectingBaseGeometry)
{
    auto topology=std::make_shared<const ExtrudedTopology>(4,Arr<Arr<unsigned>>{{0,1,2,3}},1);
    EXPECT_THROW((ExtrudedGeometry(topology,{{0,0,0},{2,0,0},{0,1,0},{1,1,0}},{0,1})),std::invalid_argument);
}
TEST(MultiRegionMeshTest, ExplicitConstituentsKeepStaticMotionContract)
{
    MultiRegionMesh source({cartesian_region("source",{{{0,1},{0,1},{0,1}}})},{});
    auto native=test::explicit_reference(source);
    auto geometry=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{native_region("explicit",native)},
        std::vector<MultiRegionMesh::Interface>{});
    auto mesh=std::make_shared<Handle>(geometry);
    EXPECT_FALSE(geometry->supports_axial_motion());
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh)),std::invalid_argument);
}
