#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/details/NativeCartesianAccess.hh"
#include <bit>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
namespace G = geometry_detail;
namespace D = AMR::adaptive_detail;
bool same(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
TEST(NativeCartesianAdmission, DetachedProvidersPreserveAllNativeMetricsAndReleaseOriginalTopology)
{
    if (!G::native_constructor_capacity_profile_supported())
        return; // NativeConstructorStorage tests own explicit unsupported-profile rejection.
    auto original = std::make_shared<OrthoMeshTopo>(
        2, 1, 1, false, false, false, OrthoMeshTopo::BoundaryNames{"a", "b", "c", "d", "e", "f"});
    auto source_topology = std::make_shared<const RectilinearTopology>(original);
    auto source_geometry = std::make_shared<const RectilinearGeometry>(
        Vec3D<ArrReal>{ArrReal{0, .125, 1}, ArrReal{0, 2}, ArrReal{0, 3}});
    auto source = std::make_shared<CartesianRegion>("source", source_topology, source_geometry);
    std::weak_ptr<OrthoMeshTopo> original_control = original;
    auto counter = G::NativeConstructionCounterAccess::create<D::Budget>(1024 * 1024);
    G::NativeConstructionAdmission admission(counter);
    const auto prefix = counter.owner()->used;
    auto detached = details::NativeCartesianAccess::detach(*source, admission);
    for (std::size_t cell = 0; cell < source->layout().cells; ++cell)
    {
        EXPECT_TRUE(same(source->geometry().cell_volume(cell), detached.region.geometry().cell_volume(cell)));
        const auto a = source->geometry().cell_centroid(cell), b = detached.region.geometry().cell_centroid(cell);
        EXPECT_TRUE(same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z));
    }
    for (std::size_t face = 0; face < source->layout().faces; ++face)
    {
        EXPECT_EQ(source->topology().owner_cell(face), detached.region.topology().owner_cell(face));
        EXPECT_EQ(source->topology().neighbor_cell(face), detached.region.topology().neighbor_cell(face));
        EXPECT_TRUE(same(source->geometry().face_area(face), detached.region.geometry().face_area(face)));
        const auto a = source->geometry().face_area_vector(face), b = detached.region.geometry().face_area_vector(face);
        EXPECT_TRUE(same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z));
    }
    for (std::size_t node = 0; node < source->layout().nodes; ++node)
    {
        const auto a = source->geometry().node_coordinates(node), b = detached.region.geometry().node_coordinates(node);
        EXPECT_TRUE(same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z));
    }
    *original =
        OrthoMeshTopo(2, 1, 1, false, false, false, OrthoMeshTopo::BoundaryNames{"changed", "b", "c", "d", "e", "f"});
    EXPECT_EQ(source->topology().boundary_batch_name(0), "changed");
    EXPECT_EQ(detached.region.topology().boundary_batch_name(0), "a");
    source.reset();
    source_topology.reset();
    source_geometry.reset();
    original.reset();
    EXPECT_TRUE(original_control.expired());
    EXPECT_GT(counter.owner()->used, prefix);
    EXPECT_DOUBLE_EQ(detached.region.geometry().cell_volume(0), .75);
    EXPECT_DOUBLE_EQ(detached.region.geometry().cell_volume(1), 5.25);
}
} // namespace
