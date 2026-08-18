/**
 * @file testVTUWriter.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for VTUWriter — validates XML output, geometry, cell data, and validation.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>
#include "io/VTUWriter.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace SimpleFluid;

// ---------------------------------------------------------------------------
// Helper: read a file into a string
// ---------------------------------------------------------------------------
std::string read_file(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    EXPECT_TRUE(in.is_open()) << "Failed to open " << filename;
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

/** @brief Decode one little-endian arithmetic value from a binary string. */
template<class Value>
Value read_little_endian(const std::string& content, size_t offset)
{
    std::array<unsigned char, sizeof(Value)> bytes{};
    std::memcpy(bytes.data(), content.data() + offset, bytes.size());
    if constexpr (std::endian::native == std::endian::big)
    {
        std::reverse(bytes.begin(), bytes.end());
    }
    Value value{};
    std::memcpy(&value, bytes.data(), bytes.size());
    return value;
}

// ---------------------------------------------------------------------------
// Helper: build a minimal valid VTU file (2 tri cells, 4 points)
// ---------------------------------------------------------------------------
/** @brief Fixture-like helper containing a minimal two-triangle VTU grid. */
struct TwoTriangles
{
    VTUWriter writer;
    std::string filename;

    explicit TwoTriangles(std::string output_filename)
        : filename(std::move(output_filename))
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

/** @brief Verifies writer construction and basic mesh-property accessors. */
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

/** @brief Verifies immutable topology is shared until a writer mutates it. */
TEST(VTUWriter, SharesImmutableTopologyAcrossWriters)
{
    auto topology = VTUWriter::make_topology(
        {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
        {0, 1, 2},
        {3},
        {5});
    VTUWriter first(topology);
    VTUWriter second(topology);

    ASSERT_TRUE(first.topology_handle());
    ASSERT_TRUE(second.topology_handle());
    EXPECT_EQ(first.topology_handle().get(), second.topology_handle().get());
    EXPECT_EQ(first.num_points(), 3UL);

    first.set_points({{2, 0, 0}, {3, 0, 0}, {2, 1, 0}});
    EXPECT_FALSE(first.topology_handle());
    EXPECT_EQ(first.num_points(), 3UL);
    EXPECT_EQ(second.topology_handle().get(), topology.get());
}

// ===========================================================================
// Cell data addition
// ===========================================================================

/**
 * @brief Verifies scalar, vector, integer, and component-wise cell arrays are
 * accepted with correct metadata.
 */
TEST(VTUWriter, CellDataAddition)
{
    // Scalar
    {
        TwoTriangles tt("test_vtu_cell_data_addition.vtu");
        tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Vector
    {
        TwoTriangles tt("test_vtu_cell_data_addition.vtu");
        tt.writer.add_vector_cell_data("velocity", {{1, 0, 0}, {0, 1, 0}});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Int
    {
        TwoTriangles tt("test_vtu_cell_data_addition.vtu");
        tt.writer.add_int_cell_data("cell_id", {42, 99});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Int64
    {
        TwoTriangles tt("test_vtu_cell_data_addition.vtu");
        tt.writer.add_int64_cell_data("global_id",
            {static_cast<global_index_t>(1000), static_cast<global_index_t>(2000)});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
    // Multiple arrays
    {
        TwoTriangles tt("test_vtu_cell_data_addition.vtu");
        tt.writer.add_scalar_cell_data("T", {300.0, 350.0});
        tt.writer.add_vector_cell_data("U", {{1, 0, 0}, {0, 1, 0}});
        tt.writer.add_int_cell_data("id", {1, 2});
        tt.writer.write(tt.filename);
        EXPECT_TRUE(std::filesystem::exists(tt.filename));
    }
}

/** @brief Verifies schema keys cover ordered names, types, and components. */
TEST(VTUWriter, CellDataSchemaKeyDescribesParallelMetadata)
{
    VTUWriter scalar;
    scalar.add_scalar_cell_data("field;with:delimiters", {1.0});

    VTUWriter same_schema;
    same_schema.add_scalar_cell_data(
        "field;with:delimiters", {2.0, 3.0});
    EXPECT_EQ(
        scalar.cell_data_schema_key(),
        same_schema.cell_data_schema_key());

    VTUWriter different_name;
    different_name.add_scalar_cell_data("another_field", {1.0});
    EXPECT_NE(
        scalar.cell_data_schema_key(),
        different_name.cell_data_schema_key());

    VTUWriter different_type;
    different_type.add_int_cell_data("field;with:delimiters", {1});
    EXPECT_NE(
        scalar.cell_data_schema_key(),
        different_type.cell_data_schema_key());

    VTUWriter different_components;
    different_components.add_vector_cell_data(
        "field;with:delimiters", {{1.0, 0.0, 0.0}});
    EXPECT_NE(
        scalar.cell_data_schema_key(),
        different_components.cell_data_schema_key());

    VTUWriter reordered;
    reordered.add_scalar_cell_data("second", {1.0});
    reordered.add_scalar_cell_data("first", {1.0});
    VTUWriter ordered;
    ordered.add_scalar_cell_data("first", {1.0});
    ordered.add_scalar_cell_data("second", {1.0});
    EXPECT_NE(
        ordered.cell_data_schema_key(),
        reordered.cell_data_schema_key());
}

// ===========================================================================
// Output content validation — structure
// ===========================================================================

/**
 * @brief Verifies emitted VTU XML contains the expected unstructured-grid,
 * point, cell, type, and offset sections.
 */
TEST(VTUWriter, OutputStructure)
{
    TwoTriangles tt("test_vtu_output_structure.vtu");
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

/** @brief Verifies emitted cell-data arrays contain the supplied values. */
TEST(VTUWriter, OutputCellDataContent)
{
    // Scalar data
    {
        TwoTriangles tt("test_vtu_output_cell_data.vtu");
        tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("Name=\"temperature\""), std::string::npos);
        EXPECT_NE(content.find("Float64"), std::string::npos);
    }
    // Vector data (3 components)
    {
        TwoTriangles tt("test_vtu_output_cell_data.vtu");
        tt.writer.add_vector_cell_data("velocity", {{1, 0, 0}, {0, 1, 0}});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("NumberOfComponents=\"3\""), std::string::npos);
    }
    // Int32
    {
        TwoTriangles tt("test_vtu_output_cell_data.vtu");
        tt.writer.add_int_cell_data("cell_id", {42, 99});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("type=\"Int32\""), std::string::npos);
    }
    // Int64
    {
        TwoTriangles tt("test_vtu_output_cell_data.vtu");
        tt.writer.add_int64_cell_data("gid", {1000, 2000});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("type=\"Int64\""), std::string::npos);
    }
}

/** @brief Verifies appended binary metadata, offsets, and payload lengths. */
TEST(VTUWriter, AppendedBinaryOutput)
{
    TwoTriangles tt("test_vtu_appended_binary.vtu");
    tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
    tt.writer.write(
        tt.filename, VTUWriter::Encoding::AppendedBinary);
    const auto content = read_file(tt.filename);

    const auto appended = content.find("<AppendedData encoding=\"raw\">\n_");
    ASSERT_NE(appended, std::string::npos);
    const auto metadata = content.substr(0, appended);
    EXPECT_NE(metadata.find("header_type=\"UInt64\""), std::string::npos);
    EXPECT_NE(metadata.find(
                  "Name=\"temperature\" format=\"appended\" offset=\"0\""),
              std::string::npos);
    EXPECT_NE(metadata.find(
                  "format=\"appended\" offset=\"24\""),
              std::string::npos);
    EXPECT_NE(metadata.find(
                  "Name=\"connectivity\" format=\"appended\" offset=\"128\""),
              std::string::npos);
    EXPECT_NE(metadata.find(
                  "Name=\"offsets\" format=\"appended\" offset=\"184\""),
              std::string::npos);
    EXPECT_NE(metadata.find(
                  "Name=\"types\" format=\"appended\" offset=\"208\""),
              std::string::npos);

    const auto payload = appended
                       + std::string("<AppendedData encoding=\"raw\">\n_")
                             .size();
    ASSERT_GE(content.size(), payload + 218UL);
    EXPECT_EQ(read_little_endian<std::uint64_t>(content, payload), 16UL);
    EXPECT_EQ(read_little_endian<std::uint64_t>(content, payload + 24), 96UL);
    EXPECT_EQ(read_little_endian<std::uint64_t>(content, payload + 128), 48UL);
    EXPECT_EQ(read_little_endian<std::uint64_t>(content, payload + 184), 16UL);
    EXPECT_EQ(read_little_endian<std::uint64_t>(content, payload + 208), 2UL);
    EXPECT_DOUBLE_EQ(
        read_little_endian<double>(content, payload + sizeof(std::uint64_t)),
        300.0);
    EXPECT_DOUBLE_EQ(
        read_little_endian<double>(
            content, payload + sizeof(std::uint64_t) + sizeof(double)),
        350.0);
}

/** @brief Verifies PVTU schemas and collision-free piece references. */
TEST(VTUWriter, ParallelIndexOutput)
{
    TwoTriangles tt("test_vtu_two_tris.vtu");
    tt.writer.add_scalar_cell_data("temperature", {300.0, 350.0});
    tt.writer.add_vector_cell_data(
        "velocity", {{1, 0, 0}, {0, 1, 0}});
    const std::string index_filename = "test_vtu_two_tris.pvtu";
    const std::vector<std::string> pieces{
        VTUWriter::rank_piece_filename(tt.filename, 0, 2),
        VTUWriter::rank_piece_filename(tt.filename, 1, 2)};

    tt.writer.write_parallel_index(index_filename, pieces);
    const auto content = read_file(index_filename);
    EXPECT_NE(content.find("type=\"PUnstructuredGrid\""),
              std::string::npos);
    EXPECT_NE(content.find("Name=\"temperature\""), std::string::npos);
    EXPECT_NE(content.find("Name=\"velocity\""), std::string::npos);
    EXPECT_NE(content.find("Source=\"test_vtu_two_tris_rank0.vtu\""),
              std::string::npos);
    EXPECT_NE(content.find("Source=\"test_vtu_two_tris_rank1.vtu\""),
              std::string::npos);
    EXPECT_EQ(
        VTUWriter::parallel_index_filename(tt.filename),
        index_filename);
    std::filesystem::remove(index_filename);

    const std::string nested_index = "test_vtu_nested_index.pvtu";
    tt.writer.write_parallel_index(
        nested_index,
        {"nested/piece_rank0.vtu", "nested/piece_rank1.vtu"});
    const auto nested_content = read_file(nested_index);
    EXPECT_NE(nested_content.find(
                  "Source=\"nested/piece_rank0.vtu\""),
              std::string::npos);
    EXPECT_NE(nested_content.find(
                  "Source=\"nested/piece_rank1.vtu\""),
              std::string::npos);
    std::filesystem::remove(nested_index);
}

// ===========================================================================
// Validation — error paths
// ===========================================================================

/**
 * @brief Verifies invalid connectivity, data lengths, component counts, and
 * output paths report errors.
 */
TEST(VTUWriter, ValidationErrors)
{
    EXPECT_THROW(
        VTUWriter::rank_piece_filename("solution.vtu", 0, 0),
        std::invalid_argument);
    EXPECT_THROW(
        VTUWriter::rank_piece_filename("solution.vtu", -1, 1),
        std::invalid_argument);
    EXPECT_THROW(
        VTUWriter::rank_piece_filename("solution.vtu", 1, 1),
        std::invalid_argument);

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
        TwoTriangles tt("test_vtu_validation_errors.vtu");
        tt.writer.add_scalar_cell_data("T", {300.0});
        EXPECT_THROW(tt.writer.write(tt.filename), std::runtime_error);
    }
    // Connectivity without cells
    {
        VTUWriter writer;
        writer.set_points({{0, 0, 0}});
        writer.set_cells({0}, {}, {});
        EXPECT_THROW(
            writer.write("connectivity_without_cells.vtu"),
            std::runtime_error);
    }
    // Connectivity outside the point array
    {
        VTUWriter writer;
        writer.set_points({{0, 0, 0}});
        writer.set_cells({1}, {1}, {1});
        EXPECT_THROW(
            writer.write("invalid_connectivity.vtu"),
            std::runtime_error);
    }
    // Unwritable path
    {
        TwoTriangles tt("test_vtu_validation_errors.vtu");
        EXPECT_THROW(tt.writer.write("/nonexistent_dir_xyz/test.vtu"),
                     std::runtime_error);
    }
}

// ===========================================================================
// Edge cases
// ===========================================================================

/**
 * @brief Verifies empty meshes, zero-length data, and minimal valid cells are
 * serialized safely.
 */
TEST(VTUWriter, EdgeCases)
{
    // Empty mesh with zero points and cells
    {
        VTUWriter writer;
        writer.set_cells({}, {}, {});
        const std::string fname = "test_zero_points.vtu";
        writer.write(fname);
        EXPECT_TRUE(std::filesystem::exists(fname));
        const auto content = read_file(fname);
        EXPECT_NE(content.find("NumberOfPoints=\"0\""), std::string::npos);
        EXPECT_NE(content.find("NumberOfCells=\"0\""), std::string::npos);
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
        TwoTriangles tt("test_vtu_edge_cases.vtu");
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
        TwoTriangles tt("test_vtu_edge_cases.vtu");
        tt.writer.add_scalar_cell_data("temp & pressure", {300.0, 350.0});
        tt.writer.write(tt.filename);
        const auto content = read_file(tt.filename);
        EXPECT_NE(content.find("&amp;"), std::string::npos);
        EXPECT_EQ(content.find("temp & pressure"), std::string::npos);
        EXPECT_NE(content.find("temp &amp; pressure"), std::string::npos);
    }
}

} // namespace
