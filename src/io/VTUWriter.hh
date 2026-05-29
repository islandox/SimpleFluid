/**
 * @file VTUWriter.hh
 * @brief Concrete VTU writer for ASCII unstructured-grid output.
 */
#pragma once

#include "dataclass/typedefs.hh"
#include "geometry/MeshUtils.hh"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <variant>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Non-template writer for VTU unstructured-grid files.
 *
 * The writer accepts plain host-side geometry and cell data arrays. Template
 * mesh and field code can gather data in their native types and delegate the
 * actual XML emission here.
 */
class VTUWriter
{
public:
    using Vec3 = MeshUtils::Vec3;
    using ScalarData = std::vector<real_t>;
    using VectorData = std::vector<Vec3>;
    using IntData = std::vector<int>;
    using Int64Data = std::vector<global_index_t>;
    using UInt8Data = std::vector<std::uint8_t>;

    void set_points(VectorData points);
    void set_cells(Int64Data connectivity,
                   Int64Data offsets,
                   UInt8Data cell_types);

    void add_scalar_cell_data(std::string name, ScalarData values);
    void add_vector_cell_data(std::string name, VectorData values);
    void add_int_cell_data(std::string name, IntData values);
    void add_int64_cell_data(std::string name, Int64Data values);

    std::size_t num_points() const noexcept { return d_points.size(); }
    std::size_t num_cells() const noexcept { return d_cell_offsets.size(); }

    void write(const std::string& filename) const;

private:
    struct DataArray
    {
        std::string name;
        std::string type;
        std::size_t number_of_components = 1;
        std::variant<ScalarData, VectorData, IntData, Int64Data> values;
    };

    void add_cell_data_array(DataArray data_array);
    void validate() const;
    static std::size_t data_array_size(const DataArray& data_array);
    static void write_cell_data_array(std::ostream& out,
                                      const DataArray& data_array,
                                      const std::string& indent);

    VectorData d_points;
    Int64Data d_connectivity;
    Int64Data d_cell_offsets;
    UInt8Data d_cell_types;
    std::vector<DataArray> d_cell_data;
};

} // namespace SimpleFluid
