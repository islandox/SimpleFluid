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

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace SimpleFluid
{

namespace
{

static_assert(sizeof(real_t) == 8);
static_assert(sizeof(int) == 4);
static_assert(sizeof(global_index_t) == 8);
static_assert(std::numeric_limits<real_t>::is_iec559);
static_assert(
    std::endian::native == std::endian::little
    || std::endian::native == std::endian::big);
static_assert(std::is_trivially_copyable_v<VTUWriter::Vec3>);
static_assert(std::is_standard_layout_v<VTUWriter::Vec3>);
static_assert(sizeof(VTUWriter::Vec3) == 3 * sizeof(real_t));
static_assert(offsetof(VTUWriter::Vec3, x) == 0);
static_assert(offsetof(VTUWriter::Vec3, y) == sizeof(real_t));
static_assert(offsetof(VTUWriter::Vec3, z) == 2 * sizeof(real_t));

/** @brief Write one arithmetic value in the little-endian VTK byte order. */
template<class Value>
void write_little_endian(std::ostream& out, const Value& value)
{
    static_assert(std::is_arithmetic_v<Value>);
    std::array<unsigned char, sizeof(Value)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(Value));
    if constexpr (std::endian::native == std::endian::big)
    {
        std::reverse(bytes.begin(), bytes.end());
    }
    out.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
}

/** @brief Write a contiguous scalar array in little-endian order. */
template<class Value>
void write_binary_values(
    std::ostream& out,
    const std::vector<Value>& values)
{
    if constexpr (std::endian::native == std::endian::little)
    {
        if (!values.empty())
        {
            out.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(
                    values.size() * sizeof(Value)));
        }
    }
    else
    {
        for (const auto& value : values)
        {
            write_little_endian(out, value);
        }
    }
}

/** @brief Write vector values without relying on host struct padding. */
void write_binary_values(
    std::ostream& out,
    const VTUWriter::VectorData& values)
{
    if constexpr (std::endian::native == std::endian::little)
    {
        if (!values.empty())
        {
            out.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(
                    values.size() * sizeof(VTUWriter::Vec3)));
        }
    }
    else
    {
        for (const auto& value : values)
        {
            write_little_endian(out, value.x);
            write_little_endian(out, value.y);
            write_little_endian(out, value.z);
        }
    }
}

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

/** @brief Construct a writer that shares immutable mesh topology. */
VTUWriter::VTUWriter(TopologyHandle topology)
    : d_shared_topology(std::move(topology))
{
    if (!d_shared_topology)
    {
        throw std::invalid_argument(
            "VTUWriter requires a non-null topology handle.");
    }
}

/** @brief Create immutable topology suitable for reuse by many writers. */
auto VTUWriter::make_topology(
    VectorData points,
    Int64Data connectivity,
    Int64Data offsets,
    UInt8Data cell_types) -> TopologyHandle
{
    return std::make_shared<const Topology>(Topology{
        std::move(points),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types)});
}

/** @brief Return the collision-free filename for one MPI rank's piece. */
std::string VTUWriter::rank_piece_filename(
    const std::string& filename,
    int rank,
    int communicator_size)
{
    if (communicator_size <= 0)
    {
        throw std::invalid_argument(
            "VTUWriter communicator size must be positive.");
    }
    if (rank < 0 || rank >= communicator_size)
    {
        throw std::invalid_argument(
            "VTUWriter rank is outside the communicator.");
    }
    if (communicator_size == 1)
    {
        return filename;
    }

    const auto suffix = "_rank" + std::to_string(rank);
    constexpr std::string_view extension = ".vtu";
    if (filename.ends_with(extension))
    {
        return filename.substr(0, filename.size() - extension.size())
             + suffix + ".vtu";
    }
    return filename + suffix + ".vtu";
}

/** @brief Return the PVTU index filename corresponding to a VTU basename. */
std::string VTUWriter::parallel_index_filename(
    const std::string& filename)
{
    constexpr std::string_view extension = ".vtu";
    if (filename.ends_with(extension))
    {
        return filename.substr(0, filename.size() - extension.size())
             + ".pvtu";
    }
    return filename + ".pvtu";
}

/** @brief Return the active shared or writer-owned topology. */
const VTUWriter::Topology& VTUWriter::topology() const noexcept
{
    return d_shared_topology ? *d_shared_topology : d_owned_topology;
}

/** @brief Detach shared topology before a legacy mutating setter. */
VTUWriter::Topology& VTUWriter::mutable_topology()
{
    if (d_shared_topology)
    {
        d_owned_topology = *d_shared_topology;
        d_shared_topology.reset();
    }
    return d_owned_topology;
}

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

/** @brief Build an unambiguous key for the ordered CellData descriptors. */
std::string VTUWriter::cell_data_schema_key() const
{
    std::string key;
    key.append(std::to_string(d_cell_data.size()));
    key.push_back(';');
    for (const auto& data_array : d_cell_data)
    {
        key.append(std::to_string(data_array.name.size()));
        key.push_back(':');
        key.append(data_array.name);
        key.push_back(';');
        key.append(std::to_string(data_array.type.size()));
        key.push_back(':');
        key.append(data_array.type);
        key.push_back(';');
        key.append(std::to_string(data_array.number_of_components));
        key.push_back(';');
    }
    return key;
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
    mutable_topology().points = std::move(points);
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
    auto& topology_data = mutable_topology();
    topology_data.connectivity = std::move(connectivity);
    topology_data.cell_offsets = std::move(offsets);
    topology_data.cell_types = std::move(cell_types);
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
    const auto& topology_data = topology();
    if (topology_data.cell_offsets.size()
        != topology_data.cell_types.size())
    {
        throw std::runtime_error(
            "VTUWriter requires one cell offset and cell type per cell.");
    }

    global_index_t previous_offset = 0;
    for (const auto offset : topology_data.cell_offsets)
    {
        if (offset <= previous_offset)
        {
            throw std::runtime_error(
                "VTUWriter cell offsets must be strictly increasing.");
        }
        previous_offset = offset;
    }

    const auto final_offset = topology_data.cell_offsets.empty()
                            ? global_index_t{}
                            : topology_data.cell_offsets.back();
    if (final_offset
        != static_cast<global_index_t>(
            topology_data.connectivity.size()))
    {
        throw std::runtime_error(
            "VTUWriter final cell offset must match connectivity size.");
    }

    for (const auto point_index : topology_data.connectivity)
    {
        if (point_index < 0
            || static_cast<size_t>(point_index)
               >= topology_data.points.size())
        {
            throw std::runtime_error(
                "VTUWriter connectivity references an invalid point.");
        }
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

/** @brief Write the legacy human-readable ASCII representation. */
void VTUWriter::write_ascii(std::ostream& out) const
{
    const auto& topology_data = topology();
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << topology_data.points.size()
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
    for (const auto& coord : topology_data.points)
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
                         : topology_data.cell_offsets[cell - 1];
        const auto end = topology_data.cell_offsets[cell];

        out << "          ";
        for (auto i = begin; i < end; ++i)
        {
            out << topology_data.connectivity[static_cast<size_t>(i)]
                << (i + 1 == end ? "" : " ");
        }
        out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (size_t cell = 0; cell < num_cells(); ++cell)
    {
        out << topology_data.cell_offsets[cell]
            << (cell + 1 == num_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (size_t cell = 0; cell < num_cells(); ++cell)
    {
        out << static_cast<int>(topology_data.cell_types[cell])
            << (cell + 1 == num_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";
}

/** @brief Write VTK XML metadata followed by raw appended numeric blocks. */
void VTUWriter::write_appended_binary(std::ostream& out) const
{
    const auto& topology_data = topology();
    constexpr std::uint64_t header_bytes = sizeof(std::uint64_t);

    const auto checked_bytes = [](size_t count, std::uint64_t value_bytes)
    {
        if (count
            > std::numeric_limits<std::uint64_t>::max() / value_bytes)
        {
            throw std::overflow_error(
                "VTUWriter appended payload exceeds UInt64 offsets.");
        }
        return static_cast<std::uint64_t>(count) * value_bytes;
    };
    const auto data_array_bytes = [&](const DataArray& data_array)
    {
        return std::visit(
            [&](const auto& values) -> std::uint64_t
            {
                using values_type = std::decay_t<decltype(values)>;
                if constexpr (std::is_same_v<values_type, VectorData>)
                {
                    return checked_bytes(
                        values.size(), 3 * sizeof(real_t));
                }
                else
                {
                    return checked_bytes(
                        values.size(),
                        sizeof(typename values_type::value_type));
                }
            },
            data_array.values);
    };

    std::uint64_t next_offset = 0;
    auto reserve_block = [&](std::uint64_t payload_bytes)
    {
        const auto offset = next_offset;
        const auto maximum_offset =
            std::numeric_limits<std::uint64_t>::max();
        if (next_offset > maximum_offset - header_bytes
            || payload_bytes
               > maximum_offset - header_bytes - next_offset)
        {
            throw std::overflow_error(
                "VTUWriter appended offsets exceed UInt64.");
        }
        next_offset += header_bytes + payload_bytes;
        return offset;
    };

    std::vector<std::uint64_t> cell_data_offsets;
    cell_data_offsets.reserve(d_cell_data.size());
    for (const auto& data_array : d_cell_data)
    {
        cell_data_offsets.push_back(
            reserve_block(data_array_bytes(data_array)));
    }
    const auto points_bytes = checked_bytes(
        topology_data.points.size(), 3 * sizeof(real_t));
    const auto connectivity_bytes = checked_bytes(
        topology_data.connectivity.size(), sizeof(global_index_t));
    const auto offsets_bytes = checked_bytes(
        topology_data.cell_offsets.size(), sizeof(global_index_t));
    const auto types_bytes = checked_bytes(
        topology_data.cell_types.size(), sizeof(std::uint8_t));
    const auto points_offset = reserve_block(points_bytes);
    const auto connectivity_offset = reserve_block(connectivity_bytes);
    const auto offsets_offset = reserve_block(offsets_bytes);
    const auto types_offset = reserve_block(types_bytes);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
           "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << topology_data.points.size()
        << "\" NumberOfCells=\"" << num_cells() << "\">\n";
    out << "      <PointData/>\n";
    out << "      <CellData>\n";
    for (size_t array_index = 0;
         array_index < d_cell_data.size();
         ++array_index)
    {
        const auto& data_array = d_cell_data[array_index];
        out << "        <DataArray type=\"" << data_array.type
            << "\" Name=\"" << escape_xml_attribute(data_array.name)
            << "\"";
        if (data_array.number_of_components != 1)
        {
            out << " NumberOfComponents=\""
                << data_array.number_of_components << "\"";
        }
        out << " format=\"appended\" offset=\""
            << cell_data_offsets[array_index] << "\"/>\n";
    }
    out << "      </CellData>\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
           "format=\"appended\" offset=\""
        << points_offset << "\"/>\n";
    out << "      </Points>\n";
    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
           "format=\"appended\" offset=\""
        << connectivity_offset << "\"/>\n";
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" "
           "format=\"appended\" offset=\""
        << offsets_offset << "\"/>\n";
    out << "        <DataArray type=\"UInt8\" Name=\"types\" "
           "format=\"appended\" offset=\""
        << types_offset << "\"/>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "  <AppendedData encoding=\"raw\">\n_";

    const auto write_block = [&](std::uint64_t payload_bytes,
                                 const auto& values)
    {
        write_little_endian(out, payload_bytes);
        write_binary_values(out, values);
    };
    for (const auto& data_array : d_cell_data)
    {
        const auto payload_bytes = data_array_bytes(data_array);
        std::visit(
            [&](const auto& values)
            {
                write_block(payload_bytes, values);
            },
            data_array.values);
    }
    write_block(points_bytes, topology_data.points);
    write_block(connectivity_bytes, topology_data.connectivity);
    write_block(offsets_bytes, topology_data.cell_offsets);
    write_block(types_bytes, topology_data.cell_types);

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";
}

/**
 * @brief Write the complete VTU file using the requested numeric encoding.
 * @param filename Path of the output .vtu file.
 * @param encoding Human-readable ASCII or raw appended binary arrays.
 * @throws std::runtime_error If the file cannot be opened or validation fails.
 */
void VTUWriter::write(
    const std::string& filename,
    Encoding encoding) const
{
    validate();

    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error(
            "Failed to open VTU output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<real_t>::max_digits10);
    if (encoding == Encoding::AppendedBinary)
    {
        write_appended_binary(out);
    }
    else
    {
        write_ascii(out);
    }

    out.close();
    if (!out)
    {
        throw std::runtime_error(
            "Failed while writing or closing VTU output file: "
            + filename);
    }
}

/**
 * @brief Write a PVTU index referencing rank-local VTU pieces.
 *
 * The index contains the same cell-data schema as this writer. Piece paths
 * are made relative to the index directory so a complete output set remains
 * relocatable.
 */
void VTUWriter::write_parallel_index(
    const std::string& filename,
    const std::vector<std::string>& piece_filenames) const
{
    validate();
    if (piece_filenames.empty())
    {
        throw std::invalid_argument(
            "VTUWriter parallel index requires at least one piece.");
    }

    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error(
            "Failed to open PVTU output file: " + filename);
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
           "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    out << "  <PUnstructuredGrid GhostLevel=\"0\">\n";
    out << "    <PPointData/>\n";
    out << "    <PCellData>\n";
    for (const auto& data_array : d_cell_data)
    {
        out << "      <PDataArray type=\"" << data_array.type
            << "\" Name=\"" << escape_xml_attribute(data_array.name)
            << "\"";
        if (data_array.number_of_components != 1)
        {
            out << " NumberOfComponents=\""
                << data_array.number_of_components << "\"";
        }
        out << "/>\n";
    }
    out << "    </PCellData>\n";
    out << "    <PPoints>\n";
    out << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n";
    out << "    </PPoints>\n";

    const auto index_directory = std::filesystem::absolute(
        std::filesystem::path(filename)).lexically_normal().parent_path();
    for (const auto& piece_filename : piece_filenames)
    {
        const auto piece_path = std::filesystem::absolute(
            std::filesystem::path(piece_filename)).lexically_normal();
        auto source_path = piece_path.lexically_relative(index_directory);
        if (source_path.empty())
        {
            source_path = piece_path;
        }
        out << "    <Piece Source=\""
            << escape_xml_attribute(source_path.generic_string())
            << "\"/>\n";
    }
    out << "  </PUnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();
    if (!out)
    {
        throw std::runtime_error(
            "Failed while writing or closing PVTU output file: "
            + filename);
    }
}

} // namespace SimpleFluid
