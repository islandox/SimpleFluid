/** @file testCompactGeometryTags.cc
 * @brief Compact record bounds admit independently constructed minimal faces.
 */
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
TEST(CompactGeometryTagsTest, TriangulatedZeroAxisFaceAdmitsCompactMinimumRecords)
{
    std::vector<GeometryPoint> points;
    for (unsigned y = 0; y <= 4; ++y)
        for (unsigned x = 0; x <= 4; ++x)
            points.push_back({x * .25, y * .25, 0});
    points.insert(points.end(), {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}});
    std::vector<std::vector<std::size_t>> faces;
    for (std::size_t y = 0; y < 4; ++y)
        for (std::size_t x = 0; x < 4; ++x)
        {
            const auto a = y * 5 + x;
            faces.push_back({a, a + 6, a + 1});
            faces.push_back({a, a + 5, a + 6});
        }
    faces.insert(faces.end(), {{25, 28, 20, 15, 10, 5, 0}, {4, 9, 14, 19, 24, 27, 26}, {0, 1, 2, 3, 4, 26, 25},
                                  {28, 27, 24, 23, 22, 21, 20}, {25, 26, 27, 28}});
    using Cell = ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const auto cell = Cell::make(points, faces);
    EXPECT_EQ(cell->volume().value, 1.L);
    const auto packet = cell->encode_certificate();
    const auto envelope = decode_geometry_packet(packet, ArithmeticMode::Ordinary);
    details::ordinary_native::Reader reader{envelope.body};
    ASSERT_EQ(reader.tag(), GeometryTag::OrdinaryCell);
    (void) reader.point();
    const auto count = reader.word();
    for (std::uint64_t i = 0; i < count; ++i)
        (void) reader.point();
    const auto face_count = reader.word();
    const auto remaining = reader.bytes.size() - reader.cursor;
    ASSERT_EQ(face_count, 37u);
    // Thirty-two bottom triangles each carry a zero-location axis support.
    // The old 72-byte minimum would reject this valid compact certificate.
    EXPECT_LT(remaining, face_count * 72);
    EXPECT_GE(remaining, face_count * 65);
    const auto restored = Cell::decode_certificate(packet);
    EXPECT_EQ(restored->volume().value, 1.L);
    EXPECT_TRUE(restored->equivalent(*cell));
    EXPECT_EQ(restored->encode_certificate(), packet);
}
} // namespace
