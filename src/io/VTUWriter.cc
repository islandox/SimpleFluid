/**
 * @file VTUWriter.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Concrete VTU writer implementation.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "io/VTUWriter.hh"

#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace SimpleFluid
{

namespace
{

/**
 * @brief Escape special XML characters in an attribute value.
 *
 * Replaces &, <, >, ", and ' with their XML entity equivalents.
 *
 * @param value Raw string to escape.
 * @return Escaped string safe for XML attribute insertion.
 */
std::string escape_xml_attribute(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value)
    {
        switch (ch)
        {
            case '&':
                escaped += "&amp;";
                break;
            case '<':
                escaped += "&lt;";
                break;
            case '>':
                escaped += "&gt;";
                break;
            case '"':
                escaped += "&quot;";
                break;
            case '\'':
                escaped += "&apos;";
                break;
            default:
                escaped += ch;
                break;
        }
    }

    return escaped;
}

/**
 * @brief Write space-separated scalar values to a stream.
 *
 * @param out Output stream.
 * @param values Scalar data to write.
 * @param indent Indentation string prepended to each line.
 */
void write_scalar_values(std::ostream& out,
                         const VTUWriter::ScalarData& values,
                         const std::string& indent)
{
    out << indent << "  ";
    for (size_t i = 0; i < values.size(); ++i)
    {
        out << values[i] << (i + 1 == values.size() ? "" : " ");
    }
    out << "\n";
}

/**
 * @brief Write space-separated 32-bit integer values to a stream.
 *
 * @param out Output stream.
 * @param values Integer data to write.
 * @param indent Indentation string prepended to each line.
 */
void write_int_values(std::ostream& out,
                      const VTUWriter::IntData& values,
                      const std::string& indent)
{
    out << indent << "  ";
    for (size_t i = 0; i < values.size(); ++i)
    {
        out << values[i] << (i + 1 == values.size() ? "" : " ");
    }
    out << "\n";
}

/**
 * @brief Write space-separated 64-bit integer values to a stream.
 *
 * @param out Output stream.
 * @param values Integer data to write.
 * @param indent Indentation string prepended to each line.
 */
void write_int64_values(std::ostream& out,
                        const VTUWriter::Int64Data& values,
                        const std::string& indent)
{
    out << indent << "  ";
    for (size_t i = 0; i < values.size(); ++i)
    {
        out << values[i] << (i + 1 == values.size() ? "" : " ");
    }
    out << "\n";
}

/**
 * @brief Write 3-component vector values (one per line) to a stream.
 *
 * @param out Output stream.
 * @param values Vector data to write.
 * @param indent Indentation string prepended to each line.
 */
void write_vector_values(std::ostream& out,
                         const VTUWriter::VectorData& values,
                         const std::string& indent)
{
    for (const auto& value : values)
    {
        out << indent << "  " << value.x << " " << value.y << " "
            << value.z << "\n";
    }
}

} // namespace

/**
 * @brief Get the number of entries in a data array (via variant visitation).
 *
 * @param data_array The data array to query.
 * @return Number of entries in the contained values variant.
 */
size_t VTUWriter::data_array_size(const DataArray& data_array)
{
    return std::visit(
        [](const auto& values) -> size_t { return values.size(); },
        data_array.values);
}

/**
 * @brief Write a single VTU DataArray element (cell data) to the output stream.
 *
 * @param out Output stream.
 * @param data_array Data array descriptor containing type, name, and values.
 * @param indent Indentation string for the XML elements.
 */
void VTUWriter::write_cell_data_array(std::ostream& out,
                                      const DataArray& data_array,
                                      const std::string& indent)
{
    out << indent << "<DataArray type=\"" << data_array.type
        << "\" Name=\"" << escape_xml_attribute(data_array.name) << "\"";
    if (data_array.number_of_components != 1)
    {
        out << " NumberOfComponents=\"" << data_array.number_of_components
            << "\"";
    }
    out << " format=\"ascii\">\n";

    std::visit(
        [&](const auto& values)
        {
            using values_type = std::decay_t<decltype(values)>;
            if constexpr (std::is_same_v<values_type, ScalarData>)
            {
                write_scalar_values(out, values, indent);
            }
            else if constexpr (std::is_same_v<values_type, VectorData>)
            {
                write_vector_values(out, values, indent);
            }
            else if constexpr (std::is_same_v<values_type, IntData>)
            {
                write_int_values(out, values, indent);
            }
            else
            {
                write_int64_values(out, values, indent);
            }
        },
        data_array.values);

    out << indent << "</DataArray>\n";
}

/**
 * @brief Set the mesh point coordinates for the VTU output.
 *
 * @param points Vector of 3D point coordinates.
 */
void VTUWriter::set_points(VectorData points)
{
    d_points = std::move(points);
}

/**
 * @brief Set the cell connectivity, offset, and type arrays for the VTU output.
 *
 * @param connectivity Flat array of node indices for all cells.
 * @param offsets Cumulative end-of-cell offsets into the connectivity array.
 * @param cell_types VTK cell type for each cell (e.g., 12 for HEXAHEDRON).
 */
void VTUWriter::set_cells(Int64Data connectivity,
                          Int64Data offsets,
                          UInt8Data cell_types)
{
    d_connectivity = std::move(connectivity);
    d_cell_offsets = std::move(offsets);
    d_cell_types = std::move(cell_types);
}

/**
 * @brief Add a scalar (Float64) cell-data array to the VTU output.
 *
 * @param name Name of the data array.
 * @param values Per-cell scalar values.
 */
void VTUWriter::add_scalar_cell_data(std::string name, ScalarData values)
{
    add_cell_data_array({
        std::move(name),
        "Float64",
        1,
        std::move(values)
    });
}

/**
 * @brief Add a vector (Float64, 3-component) cell-data array to the VTU output.
 *
 * @param name Name of the data array.
 * @param values Per-cell vector values.
 */
void VTUWriter::add_vector_cell_data(std::string name, VectorData values)
{
    add_cell_data_array({
        std::move(name),
        "Float64",
        3,
        std::move(values)
    });
}

/**
 * @brief Add a 32-bit integer cell-data array to the VTU output.
 *
 * @param name Name of the data array.
 * @param values Per-cell integer values.
 */
void VTUWriter::add_int_cell_data(std::string name, IntData values)
{
    add_cell_data_array({
        std::move(name),
        "Int32",
        1,
        std::move(values)
    });
}

/**
 * @brief Add a 64-bit integer cell-data array to the VTU output.
 *
 * @param name Name of the data array.
 * @param values Per-cell 64-bit integer values.
 */
void VTUWriter::add_int64_cell_data(std::string name, Int64Data values)
{
    add_cell_data_array({
        std::move(name),
        "Int64",
        1,
        std::move(values)
    });
}

/**
 * @brief Append a generic data array descriptor to the cell-data list.
 *
 * @param data_array Data array descriptor to add.
 */
void VTUWriter::add_cell_data_array(DataArray data_array)
{
    d_cell_data.push_back(std::move(data_array));
}

/**
 * @brief Validate the VTU data for internal consistency before writing.
 *
 * Checks that cell offsets and types match, offsets are monotonically
 * increasing, the final offset matches connectivity size, and cell data
 * arrays have the correct number of entries.
 *
 * @throws std::runtime_error If any consistency check fails.
 */
void VTUWriter::validate() const
{
    if (d_cell_offsets.size() != d_cell_types.size())
    {
        throw std::runtime_error(
            "VTUWriter requires one cell offset and cell type per cell.");
    }

    global_index_t previous_offset = 0;
    for (const auto offset : d_cell_offsets)
    {
        if (offset < previous_offset)
        {
            throw std::runtime_error(
                "VTUWriter cell offsets must be monotonically increasing.");
        }
        previous_offset = offset;
    }

    if (!d_cell_offsets.empty()
        && d_cell_offsets.back()
           != static_cast<global_index_t>(d_connectivity.size()))
    {
        throw std::runtime_error(
            "VTUWriter final cell offset must match connectivity size.");
    }

    for (const auto& data_array : d_cell_data)
    {
        if (data_array_size(data_array) != num_cells())
        {
            throw std::runtime_error(
                "VTUWriter cell data array '" + data_array.name
                + "' does not match the number of cells.");
        }
    }
}

/**
 * @brief Write the complete VTU XML file to disk.
 *
 * Validates the data, then writes the XML header, point coordinates,
 * connectivity, cell types, offsets, and cell-data arrays.
 *
 * @param filename Path of the output .vtu file.
 * @throws std::runtime_error If the file cannot be opened or validation fails.
 */
void VTUWriter::write(const std::string& filename) const
{
    validate();

    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Failed to open VTU output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<real_t>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << d_points.size()
        << "\" NumberOfCells=\"" << num_cells() << "\">\n";

    out << "      <PointData/>\n";
    out << "      <CellData>\n";
    for (const auto& data_array : d_cell_data)
    {
        write_cell_data_array(out, data_array, "        ");
    }
    out << "      </CellData>\n";

    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& coord : d_points)
    {
        out << "          " << coord.x << " " << coord.y << " " << coord.z
            << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t cell = 0; cell < num_cells(); ++cell)
    {
        const auto begin = cell == 0
                         ? global_index_t{0}
                         : d_cell_offsets[cell - 1];
        const auto end = d_cell_offsets[cell];

        out << "          ";
        for (auto i = begin; i < end; ++i)
        {
            out << d_connectivity[static_cast<size_t>(i)]
                << (i + 1 == end ? "" : " ");
        }
        out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (size_t cell = 0; cell < num_cells(); ++cell)
    {
        out << d_cell_offsets[cell] << (cell + 1 == num_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (size_t cell = 0; cell < num_cells(); ++cell)
    {
        out << static_cast<int>(d_cell_types[cell])
            << (cell + 1 == num_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error("Failed while writing VTU output file: " + filename);
    }
}

} // namespace SimpleFluid
