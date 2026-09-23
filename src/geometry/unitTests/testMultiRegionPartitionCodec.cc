/** @file testMultiRegionPartitionCodec.cc
 * @brief Sparse native-region transfer and malformed packet rejection.
 */
#include <gtest/gtest.h>
#include "geometry/mesh/MultiRegionMesh.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <cstring>
#include <limits>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using ID = MultiRegionMesh::ID;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
auto serial_comm() -> Teuchos::RCP<const Teuchos::Comm<int>>
{ return Teuchos::rcp(new Teuchos::SerialComm<int>); }

std::shared_ptr<UnstructuredMesh> explicit_chain(size_t count)
{
    Arr<MeshUtils::Vec3> nodes;
    for (size_t x = 0; x <= count; ++x)
        for (auto yz : {std::pair{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}})
            nodes.push_back({static_cast<real_t>(x), yz.first, yz.second});
    Arr<UnstructuredMesh::CellDefinition> cells;
    for (size_t c = 0; c < count; ++c)
    {
        const auto a = 4*c, b = 4*(c+1);
        cells.push_back({MeshUtils::CellType::HEXAHEDRON, {a,b,b+1,a+1,a+3,b+3,b+2,a+2}});
    }
    return std::make_shared<UnstructuredMesh>(nodes, cells);
}

struct Records
{
    RegionLayout layout{};
    std::vector<ExplicitRegionProvider::Cell> cells;
    std::vector<ExplicitRegionProvider::Face> faces;
    std::vector<ExplicitRegionProvider::Node> nodes;
    std::map<int, std::string> boundaries;
    ExplicitRegionProvider build() const { return {layout, cells, faces, nodes, boundaries}; }
};
Records cube_records()
{
    const auto mesh = explicit_chain(1);
    Records r;
    r.layout = {RegionLayout::Family::Explicit, 1, mesh->num_faces(), mesh->num_nodes(), {}};
    r.cells.push_back({0, mesh->cell_type(0), mesh->faces(0), mesh->cell_nodes(0), mesh->cell_centroid(0), mesh->cell_volume(0)});
    for (ID face = 0; face < mesh->num_faces(); ++face)
        r.faces.push_back({face, mesh->owner_cell(face), mesh->neighbor_cell(face), mesh->face_nodes(face),
            mesh->face_centroid(face), mesh->face_area_vector(face), mesh->boundary_id(face)});
    for (ID node = 0; node < mesh->num_nodes(); ++node) r.nodes.push_back({node, mesh->node_coordinates(node)});
    return r;
}

template<class T> void overwrite(std::vector<char>& packet, size_t offset, T value)
{
    if (offset + sizeof(T) > packet.size()) throw std::logic_error("Bad codec test offset.");
    std::memcpy(packet.data() + offset, &value, sizeof(T));
}
struct Cursor
{
    const std::vector<char>& packet;
    size_t pos = 0;
    template<class T> T take()
    {
        T value;
        if (pos + sizeof(T) > packet.size()) throw std::logic_error("Bad codec test packet.");
        std::memcpy(&value, packet.data() + pos, sizeof(T)); pos += sizeof(T); return value;
    }
    void array(size_t width) { const auto count = take<uint64_t>(); pos += count * width; }
};
struct PacketOffsets
{
    size_t family{}, cells{}, extents{}, periodic{}, interface_tag{}, patch{}, cell_offsets{}, boundaries{};
};
PacketOffsets inspect_cartesian_packet(const std::vector<char>& packet)
{
    Cursor r{packet}; PacketOffsets p;
    r.take<uint64_t>(); const auto regions = r.take<uint64_t>();
    for (size_t region = 0; region < regions; ++region)
    {
        if (r.take<uint32_t>() != 0) throw std::logic_error("Codec test expected Cartesian providers.");
        r.array(1);
        const auto family = r.pos; r.take<RegionLayout::Family>();
        const auto cells = r.pos; r.pos += 3 * sizeof(uint64_t);
        const auto extents = r.pos; r.pos += 3 * sizeof(uint64_t);
        const auto periodic = r.pos; r.pos += 3;
        if (region == 0) { p.family = family; p.cells = cells; p.extents = extents; p.periodic = periodic; }
        for (size_t axis = 0; axis < 3; ++axis) r.array(sizeof(real_t));
        for (size_t boundary = 0; boundary < 6; ++boundary) r.array(1);
    }
    if (r.take<uint64_t>() != 1) throw std::logic_error("Codec test expected one interface.");
    p.interface_tag = r.pos;
    if (r.take<uint32_t>() != 0) throw std::logic_error("Codec test expected structured interface.");
    p.patch = r.pos;
    constexpr size_t patch_bytes = 5 * sizeof(uint64_t) + sizeof(int) + 2 * sizeof(unsigned) + 2;
    r.pos += 2 * patch_bytes + 2 * sizeof(unsigned) + 2;
    if (r.take<uint8_t>()) r.pos += sizeof(MeshUtils::Vec3);
    r.pos += sizeof(InterfaceTolerance);
    p.cell_offsets = r.pos;
    for (size_t field = 0; field < 4; ++field) r.array(sizeof(ID));
    p.boundaries = r.pos;
    return p;
}
} // namespace

TEST(MultiRegionPartitionCodecTest, SparseExplicitClosureRetainsNativeAdjacencyAndMetrics)
{
    const auto native = explicit_chain(8);
    MultiRegionMesh source({native_region("explicit", native)}, {}, serial_comm());
    const std::array<ID, 1> visible{0};
    const auto packet = source.serialize_partition(visible);
    const auto local = MultiRegionMesh::deserialize_partition(packet, serial_comm());
    ASSERT_TRUE(std::holds_alternative<DistributedExplicitRegion>(local->regions()[0]));
    const auto& provider = std::get<DistributedExplicitRegion>(local->regions()[0]).topology();
    EXPECT_EQ(local->num_cells(), 8U);
    EXPECT_EQ(provider.resident_cells(), 3U);
    EXPECT_EQ(provider.resident_faces(), 11U);
    EXPECT_EQ(provider.resident_nodes(), 12U);
    EXPECT_TRUE(provider.cell_faces(2).empty());
    EXPECT_DOUBLE_EQ(local->cell_centroid(2).x, 2.5);
    for (auto cell : {ID{0}, ID{1}})
    {
        EXPECT_TRUE(std::ranges::equal(local->faces(cell), source.faces(cell)));
        MeshUtils::Vec3 closure{};
        for (auto face : local->faces(cell))
        {
            EXPECT_EQ(local->owner_cell(face), source.owner_cell(face));
            EXPECT_EQ(local->neighbor_cell(face), source.neighbor_cell(face));
            EXPECT_EQ(local->face_area_vector(face), source.face_area_vector(face));
            closure = closure + local->face_area_vector_outward(face, cell);
        }
        EXPECT_NEAR(closure.norm(), 0.0, 1e-14);
    }
    const std::array<ID, 0> empty{};
    const auto none = MultiRegionMesh::deserialize_partition(source.serialize_partition(empty), serial_comm());
    EXPECT_EQ(std::get<DistributedExplicitRegion>(none->regions()[0]).topology().resident_cells(), 0U);
}

TEST(MultiRegionPartitionCodecTest, RebuildsCanonicalDirectoriesFromCompactDescriptors)
{
    MultiRegionMesh source({cartesian_region("left", {{{0,1},{0,1},{0,1}}}),
                            cartesian_region("right", {{{1,2},{0,1},{0,1}}})},
                           {StructuredPatchInterface{{0,1},{1,0}}}, serial_comm());
    const std::array<ID, 2> visible{0,1};
    const auto local = MultiRegionMesh::deserialize_partition(source.serialize_partition(visible), serial_comm());
    EXPECT_EQ(local->num_faces(), source.num_faces());
    for (ID face = 0; face < source.num_faces(); ++face)
    {
        EXPECT_EQ(local->canonical_face(local->native_face(face)), face);
        EXPECT_EQ(local->owner_cell(face), source.owner_cell(face));
        EXPECT_EQ(local->neighbor_cell(face), source.neighbor_cell(face));
        EXPECT_EQ(local->boundary_id(face), source.boundary_id(face));
        EXPECT_EQ(local->face_centroid(face), source.face_centroid(face));
    }
}

TEST(MultiRegionPartitionCodecTest, RejectsMalformedLayoutInterfaceAndCatalogBytes)
{
    MultiRegionMesh source({cartesian_region("left", {{{0,1},{0,1},{0,1}}}),
                            cartesian_region("right", {{{1,2},{0,1},{0,1}}})},
                           {StructuredPatchInterface{{0,1},{1,0}}}, serial_comm());
    const std::array<ID, 2> visible{0,1};
    const auto packet = source.serialize_partition(visible);
    const auto offsets = inspect_cartesian_packet(packet);
    const auto reject = [](const std::vector<char>& bytes) {
        EXPECT_ANY_THROW(MultiRegionMesh::deserialize_partition(bytes, serial_comm()));
    };
    auto bad = packet; overwrite(bad, 0, uint64_t{0}); reject(bad);
    for (size_t length : {size_t{0}, size_t{8}, offsets.cells, offsets.interface_tag, packet.size()-1})
        reject(std::vector<char>(packet.begin(), packet.begin() + length));
    bad = packet; bad.push_back(0); reject(bad);
    bad = packet; overwrite(bad, offsets.periodic, uint8_t{2}); reject(bad);
    bad = packet; overwrite(bad, offsets.family, RegionLayout::Family::Explicit); reject(bad);
    bad = packet; overwrite(bad, offsets.cells, uint64_t{99}); reject(bad);
    bad = packet; overwrite(bad, offsets.extents, std::numeric_limits<uint64_t>::max()); reject(bad);
    bad = packet; overwrite(bad, offsets.interface_tag, uint32_t{9}); reject(bad);
    bad = packet; overwrite(bad, offsets.patch, uint64_t{42}); reject(bad);
    bad = packet; overwrite(bad, offsets.patch + sizeof(uint64_t), int{6}); reject(bad);
    bad = packet; overwrite(bad, offsets.patch + sizeof(uint64_t) + sizeof(int) + 2*sizeof(uint64_t), uint64_t{99}); reject(bad);
    bad = packet; overwrite(bad, offsets.cell_offsets + sizeof(uint64_t), uint64_t{1}); reject(bad);
    bad = packet; overwrite(bad, offsets.boundaries, std::numeric_limits<uint64_t>::max()); reject(bad);
}

TEST(MultiRegionPartitionCodecTest, RejectsMalformedExplicitResidencyRecords)
{
    const auto valid = cube_records();
    EXPECT_NO_THROW(valid.build());
    auto bad = valid; bad.cells[0].type = static_cast<MeshUtils::CellType>(255); EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.cells[0].faces.push_back(0); EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.cells[0].nodes.clear(); EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.cells[0].type = MeshUtils::CellType::POLYHEDRON; bad.cells[0].faces.pop_back(); EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.faces[0].nodes[1] = bad.faces[0].nodes[0]; EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; std::ranges::reverse(bad.faces[0].nodes); EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.faces[0].area_vector = {1e308,1e308,1e308}; EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.faces[0].boundary = 3; EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.nodes.push_back(bad.nodes[0]); EXPECT_THROW(bad.build(), std::invalid_argument);
    bad = valid; bad.layout.cells = 2; bad.cells.push_back({1, MeshUtils::CellType::HEXAHEDRON, {}, {}, {2,0,0}, 1});
    bad.faces[0].neighbor = 1; bad.faces[0].boundary = 0; bad.boundaries.emplace(0, "bad");
    EXPECT_THROW(bad.build(), std::invalid_argument);
}
