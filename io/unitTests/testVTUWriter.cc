/**
 * @file testVTUWriter.cc
 * @brief Unit tests for VTUWriter — validates XML output, geometry, cell data, and validation.
 */

#include <gtest/gtest.h>
#include "io/VTUWriter.hh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace SimpleFluid;

// ---------------------------------------------------------------------------
// Helper: read a file into a string
// ---------------------------------------------------------------------------
std::string read_file(const std::string& filename)
{
    std::ifstream in(filename);
    EXPECT_TRUE(in.is_open()) << "Failed to open " << filename;
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

// ---------------------------------------------------------------------------
// Helper: build a minimal valid VTU file (2 tri cells, 4 points)
// ---------------------------------------------------------------------------
struct TwoTriangles
{
    VTUWriter writer;
    std::string filename = "test_vtu_two_tris.vtu";

    TwoTriangles()
    {
        writer.set_points({
            {0.0, 0.0, 0.0},   // node 0
            {1.0, 0.0, 0.0},   // node 1
            {0.0, 1.0, 0.0},   // node 2
            {1.0, 1.0, 0.0},   // node 3
        });
        writer.set_cells(
            /* connectivity */ {0, 1, 2,  1, 3, 2},
            /* offsets     */ {3, 6},
            /* cell types  */ {5, 5}  // VTK_TRIANGLE = 5
        );
    }

    ~TwoTriangles()
    {
        std::filesystem::remove(filename);
    }
};

// ===========================================================================
// Construction & basic accessors
// ===========================================================================

TEST(VTUWriter, DefaultConstructionHasZeroPointsAndCells)
{
    VTUWriter writer;
    EXPECT_EQ(writer.num_points(), 0UL);
    EXPECT_EQ(writer.num_cells(), 0UL);
}

TEST(VTUWriter, SetPointsUpdatesPointCount)
{
    VTUWriter writer;
    writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    EXPECT_EQ(writer.num_points(), 3UL);
    EXPECT_EQ(writer.num_cells(), 0UL);
}

TEST(VTUWriter, SetCellsUpdatesCellCountWithoutAffectingPoints)
{
    VTUWriter writer;
    writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    writer.set_cells(
        {0, 1, 2},    // connectivity
        {3},           // offsets
        {5}            // VTK_TRIANGLE
    );
    EXPECT_EQ(writer.num_points(), 3UL);
    EXPECT_EQ(writer.num_cells(), 1UL);
}

// ===========================================================================
// Cell data addition
// ===========================================================================

TEST(VTUWriter, AddScalarCellDataIsAccepted)
{
    TwoTriangles tt;
    tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
    // should not throw
    tt.writer.write(tt.filename);
    EXPECT_TRUE(std::filesystem::exists(tt.filename));
}

TEST(VTUWriter, AddVectorCellDataIsAccepted)
{
    TwoTriangles tt;
    tt.writer.add_vector_cell_data("velocity", {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    });
    tt.writer.write(tt.filename);
    EXPECT_TRUE(std::filesystem::exists(tt.filename));
}

TEST(VTUWriter, AddIntCellDataIsAccepted)
{
    TwoTriangles tt;
    tt.writer.add_int_cell_data("cell_id", {42, 99});
    tt.writer.write(tt.filename);
    EXPECT_TRUE(std::filesystem::exists(tt.filename));
}

TEST(VTUWriter, AddInt64CellDataIsAccepted)
{
    TwoTriangles tt;
    tt.writer.add_int64_cell_data("global_id",
                                  {static_cast<global_index_t>(1000),
                                   static_cast<global_index_t>(2000)});
    tt.writer.write(tt.filename);
    EXPECT_TRUE(std::filesystem::exists(tt.filename));
}

TEST(VTUWriter, MultipleCellDataArraysAreAccepted)
{
    TwoTriangles tt;
    tt.writer.add_scalar_cell_data("T", {300.0, 350.0});
    tt.writer.add_vector_cell_data("U", {{1, 0, 0}, {0, 1, 0}});
    tt.writer.add_int_cell_data("id", {1, 2});
    tt.writer.write(tt.filename);
    EXPECT_TRUE(std::filesystem::exists(tt.filename));
}

// ===========================================================================
// Output content validation
// ===========================================================================

TEST(VTUWriter, OutputContainsXmlDeclaration)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("<?xml version=\"1.0\"?>"), std::string::npos);
}

TEST(VTUWriter, OutputContainsUnstructuredGrid)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("UnstructuredGrid"), std::string::npos);
}

TEST(VTUWriter, OutputContainsCorrectPointAndCellCounts)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("NumberOfPoints=\"4\""), std::string::npos);
    EXPECT_NE(content.find("NumberOfCells=\"2\""), std::string::npos);
}

TEST(VTUWriter, OutputContainsPointCoordinates)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    // Should contain at least one coordinate value
    EXPECT_NE(content.find("0 0 0"), std::string::npos);
}

TEST(VTUWriter, OutputContainsConnectivity)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("Name=\"connectivity\""), std::string::npos);
}

TEST(VTUWriter, OutputContainsOffsets)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("Name=\"offsets\""), std::string::npos);
}

TEST(VTUWriter, OutputContainsTypes)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("Name=\"types\""), std::string::npos);
}

TEST(VTUWriter, ScalarCellDataAppearsInOutput)
{
    TwoTriangles tt;
    tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("Name=\"temperature\""), std::string::npos);
    EXPECT_NE(content.find("Float64"), std::string::npos);
}

TEST(VTUWriter, VectorCellDataHasThreeComponents)
{
    TwoTriangles tt;
    tt.writer.add_vector_cell_data("velocity", {{1, 0, 0}, {0, 1, 0}});
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("NumberOfComponents=\"3\""), std::string::npos);
}

TEST(VTUWriter, IntCellDataTypeIsInt32)
{
    TwoTriangles tt;
    tt.writer.add_int_cell_data("cell_id", {42, 99});
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("type=\"Int32\""), std::string::npos);
}

TEST(VTUWriter, Int64CellDataTypeIsInt64)
{
    TwoTriangles tt;
    tt.writer.add_int64_cell_data("gid", {1000, 2000});
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    EXPECT_NE(content.find("type=\"Int64\""), std::string::npos);
}

// ===========================================================================
// Validation — error paths
// ===========================================================================

TEST(VTUWriter, WriteWithoutPointsProducesZeroPointVtu)
{
    VTUWriter writer;
    writer.set_cells({0, 1, 2}, {3}, {5});
    // validate() does not check for empty points — it's valid to write
    // a VTU with zero points (e.g., an empty partition on some ranks)
    const std::string fname = "test_zero_points.vtu";
    writer.write(fname);
    EXPECT_TRUE(std::filesystem::exists(fname));
    const auto content = read_file(fname);
    EXPECT_NE(content.find("NumberOfPoints=\"0\""), std::string::npos);
    std::filesystem::remove(fname);
}

TEST(VTUWriter, MismatchedOffsetsAndTypesThrows)
{
    VTUWriter writer;
    writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    writer.set_cells(
        {0, 1, 2, 1, 3, 2},   // connectivity for 2 cells
        {3, 6},                  // 2 offsets
        {5}                      // only 1 cell type → mismatch
    );
    EXPECT_THROW(writer.write("mismatch.vtu"), std::runtime_error);
}

TEST(VTUWriter, NonMonotonicOffsetsThrows)
{
    VTUWriter writer;
    writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    writer.set_cells(
        {0, 1, 2, 1, 3, 2},
        {6, 3},   // decreasing — invalid
        {5, 5}
    );
    EXPECT_THROW(writer.write("nonmonotonic.vtu"), std::runtime_error);
}

TEST(VTUWriter, FinalOffsetMismatchThrows)
{
    VTUWriter writer;
    writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    writer.set_cells(
        {0, 1, 2},   // 3 connectivity entries
        {9},          // claims 9 entries — mismatch
        {5}
    );
    EXPECT_THROW(writer.write("bad_offset.vtu"), std::runtime_error);
}

TEST(VTUWriter, CellDataSizeMismatchThrows)
{
    TwoTriangles tt;
    tt.writer.add_scalar_cell_data("T", {300.0});  // 1 value for 2 cells
    EXPECT_THROW(tt.writer.write(tt.filename), std::runtime_error);
}

TEST(VTUWriter, WriteToUnwritablePathThrows)
{
    TwoTriangles tt;
    EXPECT_THROW(tt.writer.write("/nonexistent_dir_xyz/test.vtu"),
                 std::runtime_error);
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST(VTUWriter, ZeroPointSingleCellHexIsValid)
{
    VTUWriter writer;
    // A single hex cell at origin
    writer.set_points({
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    });
    writer.set_cells(
        {0, 1, 2, 3, 4, 5, 6, 7},
        {8},
        {12}  // VTK_HEXAHEDRON
    );
    const std::string fname = "test_single_hex.vtu";
    writer.write(fname);
    EXPECT_TRUE(std::filesystem::exists(fname));
    const auto content = read_file(fname);
    EXPECT_NE(content.find("NumberOfPoints=\"8\""), std::string::npos);
    EXPECT_NE(content.find("NumberOfCells=\"1\""), std::string::npos);
    std::filesystem::remove(fname);
}

TEST(VTUWriter, EmptyCellDataSectionHasNoDataArrays)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    // CellData section exists but contains only whitespace between tags
    auto pos = content.find("<CellData>");
    ASSERT_NE(pos, std::string::npos);
    auto end_pos = content.find("</CellData>", pos);
    ASSERT_NE(end_pos, std::string::npos);
    // The content between <CellData> and </CellData> should have no DataArray
    auto between = content.substr(pos + 10, end_pos - pos - 10);
    EXPECT_EQ(between.find("<DataArray"), std::string::npos);
}

TEST(VTUWriter, XmlSpecialCharactersInDataNameAreEscaped)
{
    TwoTriangles tt;
    tt.writer.add_scalar_cell_data("temp & pressure", {300.0, 350.0});
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);
    // The ampersand should be escaped
    EXPECT_NE(content.find("&amp;"), std::string::npos);
    // The raw & should NOT appear unescaped — but the escaped form should
    EXPECT_EQ(content.find("temp & pressure"), std::string::npos);
    // The escaped &amp; form IS what we expect in XML output
    EXPECT_NE(content.find("temp &amp; pressure"), std::string::npos);
}

} // namespace
