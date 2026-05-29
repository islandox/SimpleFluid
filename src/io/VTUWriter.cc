/**
 * @file VTUWriter.cc
 * @brief Concrete VTU writer implementation.
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

void write_scalar_values(std::ostream& out,
                         const VTUWriter::ScalarData& values,
                         const std::string& indent)
{
    out << indent << "  ";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out << values[i] << (i + 1 == values.size() ? "" : " ");
    }
    out << "\n";
}

void write_int_values(std::ostream& out,
                      const VTUWriter::IntData& values,
                      const std::string& indent)
{
    out << indent << "  ";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out << values[i] << (i + 1 == values.size() ? "" : " ");
    }
    out << "\n";
}

void write_int64_values(std::ostream& out,
                        const VTUWriter::Int64Data& values,
                        const std::string& indent)
{
    out << indent << "  ";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out << values[i] << (i + 1 == values.size() ? "" : " ");
    }
    out << "\n";
}

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

std::size_t VTUWriter::data_array_size(const DataArray& data_array)
{
    return std::visit(
        [](const auto& values) -> std::size_t { return values.size(); },
        data_array.values);
}

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

void VTUWriter::set_points(VectorData points)
{
    d_points = std::move(points);
}

void VTUWriter::set_cells(Int64Data connectivity,
                          Int64Data offsets,
                          UInt8Data cell_types)
{
    d_connectivity = std::move(connectivity);
    d_cell_offsets = std::move(offsets);
    d_cell_types = std::move(cell_types);
}

void VTUWriter::add_scalar_cell_data(std::string name, ScalarData values)
{
    add_cell_data_array({
        std::move(name),
        "Float64",
        1,
        std::move(values)
    });
}

void VTUWriter::add_vector_cell_data(std::string name, VectorData values)
{
    add_cell_data_array({
        std::move(name),
        "Float64",
        3,
        std::move(values)
    });
}

void VTUWriter::add_int_cell_data(std::string name, IntData values)
{
    add_cell_data_array({
        std::move(name),
        "Int32",
        1,
        std::move(values)
    });
}

void VTUWriter::add_int64_cell_data(std::string name, Int64Data values)
{
    add_cell_data_array({
        std::move(name),
        "Int64",
        1,
        std::move(values)
    });
}

void VTUWriter::add_cell_data_array(DataArray data_array)
{
    d_cell_data.push_back(std::move(data_array));
}

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
    for (std::size_t cell = 0; cell < num_cells(); ++cell)
    {
        const auto begin = cell == 0
                         ? global_index_t{0}
                         : d_cell_offsets[cell - 1];
        const auto end = d_cell_offsets[cell];

        out << "          ";
        for (auto i = begin; i < end; ++i)
        {
            out << d_connectivity[static_cast<std::size_t>(i)]
                << (i + 1 == end ? "" : " ");
        }
        out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t cell = 0; cell < num_cells(); ++cell)
    {
        out << d_cell_offsets[cell] << (cell + 1 == num_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t cell = 0; cell < num_cells(); ++cell)
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
