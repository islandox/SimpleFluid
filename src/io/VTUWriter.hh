/**
 * @file VTUWriter.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Concrete VTU writer for ASCII or appended-binary unstructured grids.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "dataclass/typedefs.hh"
#include "geometry/MeshUtils.hh"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Non-template writer for VTU unstructured-grid files.
 *
 * The writer accepts plain host-side geometry and cell data arrays. Immutable
 * topology handles let transient solvers reuse mesh connectivity across
 * output timesteps, while appended binary encoding avoids per-value text
 * formatting for production output.
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

    /** @brief Encoding used for numeric VTU arrays. */
    enum class Encoding : std::uint8_t
    {
        Ascii = 0,
        AppendedBinary = 1
    };

    /** @brief Immutable mesh topology reusable across output timesteps. */
    struct Topology
    {
        VectorData points;
        Int64Data connectivity;
        Int64Data cell_offsets;
        UInt8Data cell_types;
    };
    using TopologyHandle = std::shared_ptr<const Topology>;

    VTUWriter() = default;
    explicit VTUWriter(TopologyHandle topology);

    static TopologyHandle make_topology(
        VectorData points,
        Int64Data connectivity,
        Int64Data offsets,
        UInt8Data cell_types);

    static std::string rank_piece_filename(
        const std::string& filename,
        int rank,
        int communicator_size);
    static std::string parallel_index_filename(
        const std::string& filename);

    void set_points(VectorData points);
    void set_cells(Int64Data connectivity,
                   Int64Data offsets,
                   UInt8Data cell_types);

    void add_scalar_cell_data(std::string name, ScalarData values);
    void add_vector_cell_data(std::string name, VectorData values);
    void add_int_cell_data(std::string name, IntData values);
    void add_int64_cell_data(std::string name, Int64Data values);

    size_t num_points() const noexcept { return topology().points.size(); }
    size_t num_cells() const noexcept
    {
        return topology().cell_offsets.size();
    }

    /**
     * @brief Return an exact key for the ordered CellData schema.
     *
     * Values and local cell counts are intentionally excluded so MPI ranks
     * can compare names, VTK types, and component counts before publishing a
     * shared PVTU index.
     */
    std::string cell_data_schema_key() const;

    /** @return Shared topology, or null when using mutable owned topology. */
    const TopologyHandle& topology_handle() const noexcept
    {
        return d_shared_topology;
    }

    void write(
        const std::string& filename,
        Encoding encoding = Encoding::Ascii) const;
    void write_parallel_index(
        const std::string& filename,
        const std::vector<std::string>& piece_filenames) const;

private:
    /** @brief Type-erased cell-data array and its VTU metadata. */
    struct DataArray
    {
        std::string name;
        std::string type;
        size_t number_of_components = 1;
        std::variant<ScalarData, VectorData, IntData, Int64Data> values;
    };

    void add_cell_data_array(DataArray data_array);
    void validate() const;
    static size_t data_array_size(const DataArray& data_array);
    static void write_cell_data_array(std::ostream& out,
                                      const DataArray& data_array,
                                      const std::string& indent);
    void write_ascii(std::ostream& out) const;
    void write_appended_binary(std::ostream& out) const;
    const Topology& topology() const noexcept;
    Topology& mutable_topology();

    Topology d_owned_topology;
    TopologyHandle d_shared_topology;
    std::vector<DataArray> d_cell_data;
};

} // namespace SimpleFluid
