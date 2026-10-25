/** @file testExactExtrudedConstruction.cc
 * @brief Independent native construction metrics, topology binding and packet oracles.
 */
#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Rational = boost::multiprecision::cpp_rational;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Fixture
{
    std::shared_ptr<const ExtrudedTopology> topology;
    std::vector<SimpleFluid::PLIC::PointRecipe> points{
        {{0.1, 0.1, 0}, {}}, {{0.2, 0.1, 0}, {}}, {{0.2, 0.2, 0}, {}}, {{0.1, 0.2, 0}, {}}, {{}, {0, 1}}, {{}, {3, 2}}};
    std::vector<std::size_t> active_points{0, 1, 2, 3, 4, 5};
    std::vector<SimpleFluid::PLIC::ScalarRecipe> axial{{0.1, {}}, {0.2, {}}, {0, {0, 1}}};
    std::vector<std::size_t> active_z{0, 2, 1};
    static std::shared_ptr<const ExtrudedTopology> make_topology(bool reverse = false)
    {
        Arr<Arr<unsigned>> cells{{0, 4, 5, 3}, {4, 1, 2, 5}};
        if (reverse)
            std::reverse(cells.begin(), cells.end());
        return std::make_shared<const ExtrudedTopology>(6, cells, 2,
            Arr<SemiStructMeshTopo::BoundaryEdge>{
                {0, 4, "bottom"}, {4, 1, "bottom"}, {1, 2, "right"}, {2, 5, "top"}, {5, 3, "top"}, {3, 0, "left"}});
    }
    Fixture() : topology(make_topology()) {}
    std::shared_ptr<const SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>> make() const
    {
        return SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::make(
            topology, points, active_points, axial, active_z);
    }
    std::shared_ptr<const SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Ordinary>>
    make_native() const
    {
        return SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Ordinary>::make(
            topology, points, active_points, axial, active_z);
    }
    SP<MultiRegionMesh> mesh(
        const std::shared_ptr<const SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Ordinary>>&
            construction) const
    {
        const auto geometry = std::make_shared<const ExtrudedGeometry>(topology, construction);
        return std::make_shared<MultiRegionMesh>(
            std::vector<MultiRegionMesh::Region>{ExtrudedRegion("constructed", topology, geometry)},
            std::vector<MultiRegionMesh::Interface>{});
    }
};
} // namespace

TEST(ExactExtrudedConstructionTest, DecimalMeansUseIndependentRationalVolumeCentroidAndFaceClosure)
{
    Fixture f;
    const auto construction = f.make();
    const Rational low(0.1), high(0.2), width = high - low;
    const double volume = (width * width * width / 4).convert_to<double>();
    for (std::size_t c = 0; c < 4; ++c)
    {
        EXPECT_EQ(construction->cell_volume(c), volume);
        const auto center = construction->cell_centroid(c);
        const Rational exact_x = c % 2 ? Rational((low + 3 * high) / 4) : Rational((3 * low + high) / 4);
        const Rational exact_z = c / 2 ? Rational((low + 3 * high) / 4) : Rational((3 * low + high) / 4);
        const double x = exact_x.convert_to<double>(), z = exact_z.convert_to<double>();
        EXPECT_EQ(center.x, x);
        EXPECT_EQ(center.y, ((low + high) / 2).convert_to<double>());
        EXPECT_EQ(center.z, z);
        MeshUtils::Vec3 area{};
        double norm = 0;
        for (const auto face : f.topology->cell_faces(c))
        {
            const auto vector = construction->face_area_vector(face);
            area = area + (f.topology->owner_cell(face) == c ? vector : vector * (-1));
            norm += construction->face_area(face);
        }
        EXPECT_LE(area.norm(), 64 * std::numeric_limits<double>::epsilon() * norm);
    }
    const auto display = construction->display_points();
    const Rational rounded_midpoint(display[4].x);
    EXPECT_NE(rounded_midpoint, (low + high) / 2);
}

TEST(ExactExtrudedConstructionTest, RecipeAndRetainedTopologyValidationRejectInvalidInputs)
{
    Fixture f;
    const auto valid = f.make();
    const auto clone = Fixture::make_topology();
    EXPECT_TRUE(valid->matches_topology(f.topology));
    EXPECT_FALSE(valid->matches_topology(clone));
    EXPECT_THROW((void) ExtrudedGeometry(clone, f.make_native()), std::invalid_argument);
    auto points = f.points;
    points[4].parents = {0, 4};
    EXPECT_THROW((void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::make(
                     f.topology, points, f.active_points, f.axial, f.active_z),
        std::invalid_argument);
    points = f.points;
    points[0].root[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::make(
                     f.topology, points, f.active_points, f.axial, f.active_z),
        std::invalid_argument);
    auto active = f.active_points;
    active.pop_back();
    EXPECT_THROW((void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::make(
                     f.topology, f.points, active, f.axial, f.active_z),
        std::invalid_argument);
    auto axial = f.axial;
    axial[2].parents = {2};
    EXPECT_THROW((void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::make(
                     f.topology, f.points, f.active_points, axial, f.active_z),
        std::invalid_argument);
}

TEST(ExactExtrudedConstructionTest, ConstructionCodecBindsFullTopologyAndRejectsMalformedPackets)
{
    Fixture f;
    const auto original = f.make();
    const auto packet = original->encode();
    const auto decoded = SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::decode(
        Fixture::make_topology(), packet);
    for (std::size_t c = 0; c < 4; ++c)
        EXPECT_TRUE(original->certified_cell_geometry(c)->equivalent(*decoded->certified_cell_geometry(c)));
    EXPECT_THROW((void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::decode(
                     Fixture::make_topology(true), packet),
        std::invalid_argument);
    auto bad = packet;
    bad.pop_back();
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::decode(f.topology, bad),
        std::invalid_argument);
    bad = packet;
    bad.push_back(0);
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::decode(f.topology, bad),
        std::invalid_argument);
    bad = packet;
    bad[0] ^= 1;
    EXPECT_THROW(
        (void) SimpleFluid::PLIC::ExtrudedConstruction<SimpleFluid::ArithmeticMode::Exact>::decode(f.topology, bad),
        std::invalid_argument);
}

TEST(ExactExtrudedConstructionTest, NativePartitionPacketPreservesAuthoritativeGeometryAndConfiguration)
{
    Fixture f;
    const auto construction = f.make_native();
    const auto mesh = f.mesh(construction);
    std::vector<std::uint64_t> visible{0, 1, 2, 3};
    const auto packet = mesh->serialize_partition(visible);
    const auto restored = MultiRegionMesh::deserialize_partition(packet, Tpetra::getDefaultComm());
    EXPECT_TRUE(restored->has_construction());
    EXPECT_EQ(mesh->configuration_signature(), restored->configuration_signature());
    for (const auto c : visible)
    {
        EXPECT_TRUE(
            SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::from_native_cell(*restored, c)
                ->equivalent(*construction->certified_cell_geometry(c)));
        EXPECT_EQ(restored->cell_volume(c), construction->cell_volume(c));
        EXPECT_THROW(
            (void) SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::native_cell_shell(
                *restored, c),
            std::invalid_argument);
        const auto certificate =
            SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::from_native_cell(*restored, c)
                ->encode_certificate();
        EXPECT_TRUE(SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::decode_certificate(
            certificate)
                ->equivalent(*construction->certified_cell_geometry(c)));
    }
}

TEST(ExactExtrudedConstructionTest, IdenticalDisplayCoordinatesCannotHideDifferentConstructionSignatures)
{
    Fixture f;
    const auto exact = f.make();
    const auto native = f.make_native();
    const auto display = exact->display_points();
    const auto node = display[4];
    f.points[4] = {{node.x, node.y, node.z}, {}};
    const auto rounded = f.make();
    const auto native_rounded = f.make_native();
    const auto other_display = rounded->display_points();
    ASSERT_EQ(display.size(), other_display.size());
    for (std::size_t p = 0; p < display.size(); ++p)
    {
        EXPECT_EQ(display[p].x, other_display[p].x);
        EXPECT_EQ(display[p].y, other_display[p].y);
        EXPECT_EQ(display[p].z, other_display[p].z);
    }
    EXPECT_NE(f.mesh(native)->configuration_signature(), f.mesh(native_rounded)->configuration_signature());
    EXPECT_FALSE(exact->certified_cell_geometry(0)->equivalent(*rounded->certified_cell_geometry(0)));
}

TEST(ExactExtrudedConstructionTest, NativePacketRejectsUnqualifiedConstructedAffineMotion)
{
    Fixture f;
    const auto mesh = f.mesh(f.make_native());
    const std::vector<std::uint64_t> visible{0, 1, 2, 3};
    auto packet = mesh->serialize_partition(visible);
    ASSERT_GE(mesh->axial_edges().size(), 2u);
    ASSERT_GE(packet.size(), sizeof(std::uint64_t) + sizeof(double));
    const auto offset = packet.size() - sizeof(std::uint64_t) - sizeof(double);
    double current = 0;
    std::memcpy(&current, packet.data() + offset, sizeof(current));
    ASSERT_EQ(current, mesh->axial_edges().back());
    // The native packet tail is current axial edges followed by the epoch.
    // Preserve finite increasing dimensions but forge a nonidentity affine map.
    current += 0.01 * (mesh->axial_edges().back() - mesh->axial_edges().front());
    std::memcpy(packet.data() + offset, &current, sizeof(current));
    EXPECT_THROW(
        (void) MultiRegionMesh::deserialize_partition(packet, Tpetra::getDefaultComm()), std::invalid_argument);
}
