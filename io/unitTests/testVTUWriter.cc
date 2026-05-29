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

TEST(VTUWriter, ConstructionAndBasicAccess)
{
    VTUWriter writer;
    EXPECT_EQ(writer.num_points(), 0UL);
    EXPECT_EQ(writer.num_cells(), 0UL);

    writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    EXPECT_EQ(writer.num_points(), 3UL);

    writer.set_cells({0, 1, 2}, {3}, {5});
    EXPECT_EQ(writer.num_cells(), 1UL);
}

// ===========================================================================
// Cell data addition
// ===========================================================================

TEST(VTUWriter, CellDataAddition)
{
    // Scalar
    {
        TwoTriangles tt;
        tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Vector
    {
        TwoTriangles tt;
        tt.writer.add_vector_cell_data("velocity", {{1, 0, 0}, {0, 1, 0}});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Int
    {
        TwoTriangles tt;
        tt.writer.add_int_cell_data("cell_id", {42, 99});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Int64
    {
        TwoTriangles tt;
        tt.writer.add_int64_cell_data("global_id",
            {static_cast<global_index_t>(1000), static_cast<global_index_t>(2000)});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Multiple arrays
    {
        TwoTriangles tt;
        tt.writer.add_scalar_cell_data("T", {300.0, 350.0});
        tt.writer.add_vector_cell_data("U", {{1, 0, 0}, {0, 1, 0}});
        tt.writer.add_int_cell_data("id", {1, 2});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
}

// ===========================================================================
// Output content validation — structure
// ===========================================================================

TEST(VTUWriter, OutputStructure)
{
    TwoTriangles tt;
    tt.writer.write(tt.filename);
    const auto content = read_file(tt.filename);

    EXPECT_NE(content.find("<?xml version=\"1.0\"?>"), std::string::npos);
    EXPECT_NE(content.find("UnstructuredGrid"), std::string::npos);
    EXPECT_NE(content.find("NumberOfPoints=\"4\""), std::string::npos);
    EXPECT_NE(content.find("NumberOfCells=\"2\""), std::string::npos);
    EXPECT_NE(content.find("0 0 0"), std::string::npos);
    EXPECT_NE(content.find("Name=\"connectivity\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"offsets\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"types\""), std::string::npos);
}

// ===========================================================================
// Output content validation — cell data
// ===========================================================================

TEST(VTUWriter, OutputCellDataContent)
{
    // Scalar data
    {
        TwoTriangles tt;
        tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("Name=\"temperature\""), std::string::npos);
        EXPECT_NE(content.find("Float64"), std::string::npos);
    }
    // Vector data (3 components)
    {
        TwoTriangles tt;
        tt.writer.add_vector_cell_data("velocity", {{1, 0, 0}, {0, 1, 0}});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("NumberOfComponents=\"3\""), std::string::npos);
    }
    // Int32
    {
        TwoTriangles tt;
        tt.writer.add_int_cell_data("cell_id", {42, 99});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("type=\"Int32\""), std::string::npos);
    }
    // Int64
    {
        TwoTriangles tt;
        tt.writer.add_int64_cell_data("gid", {1000, 2000});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("type=\"Int64\""), std::string::npos);
    }
}

// ===========================================================================
// Validation — error paths
// ===========================================================================

TEST(VTUWriter, ValidationErrors)
{
    // Mismatched offsets and types
    {
        VTUWriter writer;
        writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
        writer.set_cells({0, 1, 2, 1, 3, 2}, {3, 6}, {5});
        EXPECT_THROW(writer.write("mismatch.vtu"), std::runtime_error);
    }
    // Non-monotonic offsets
    {
        VTUWriter writer;
        writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
        writer.set_cells({0, 1, 2, 1, 3, 2}, {6, 3}, {5, 5});
        EXPECT_THROW(writer.write("nonmonotonic.vtu"), std::runtime_error);
    }
    // Final offset mismatch
    {
        VTUWriter writer;
        writer.set_points({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
        writer.set_cells({0, 1, 2}, {9}, {5});
        EXPECT_THROW(writer.write("bad_offset.vtu"), std::runtime_error);
    }
    // Cell data size mismatch
    {
        TwoTriangles tt;
        tt.writer.add_scalar_cell_data("T", {300.0});
        EXPECT_THROW(tt.writer.write(tt.filename), std::runtime_error);
    }
    // Unwritable path
    {
        TwoTriangles tt;
        EXPECT_THROW(tt.writer.write("/nonexistent_dir_xyz/test.vtu"),
                     std::runtime_error);
    }
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST(VTUWriter, EdgeCases)
{
    // Write with zero points
    {
        VTUWriter writer;
        writer.set_cells({0, 1, 2}, {3}, {5});
        const std::string fname = "test_zero_points.vtu";
        writer.write(fname);
        EXPECT_TRUE(std::filesystem::exists(fname));
        const auto content = read_file(fname);
        EXPECT_NE(content.find("NumberOfPoints=\"0\""), std::string::npos);
        std::filesystem::remove(fname);
    }
    // Single hex cell
    {
        VTUWriter writer;
        writer.set_points({
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}});
        writer.set_cells({0, 1, 2, 3, 4, 5, 6, 7}, {8}, {12});
        const std::string fname = "test_single_hex.vtu";
        writer.write(fname);
        EXPECT_TRUE(std::filesystem::exists(fname));
        const auto content = read_file(fname);
        EXPECT_NE(content.find("NumberOfPoints=\"8\""), std::string::npos);
        EXPECT_NE(content.find("NumberOfCells=\"1\""), std::string::npos);
        std::filesystem::remove(fname);
    }
    // Empty CellData section
    {
        TwoTriangles tt;
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        auto pos = content.find("<CellData>");
        ASSERT_NE(pos, std::string::npos);
        auto end_pos = content.find("</CellData>", pos);
        ASSERT_NE(end_pos, std::string::npos);
        auto between = content.substr(pos + 10, end_pos - pos - 10);
        EXPECT_EQ(between.find("<DataArray"), std::string::npos);
    }
    // XML special character escaping
    {
        TwoTriangles tt;
        tt.writer.add_scalar_cell_data("temp & pressure", {300.0, 350.0});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("&amp;"), std::string::npos);
        EXPECT_EQ(content.find("temp & pressure"), std::string::npos);
        EXPECT_NE(content.find("temp &amp; pressure"), std::string::npos);
    }
}

} // namespace
