/**
 * @file VectorCellField.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Three-component cell-centered vector field backed by Tpetra::MultiVector.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/CellFieldBase.hh"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Cell-centered 3D vector field stored as a Tpetra::MultiVector.
 *
 * The three MultiVector columns are ordered as x, y, z.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class VectorCellField
    : public CellFieldBase<Pack, VectorCellField<Pack>,
                           typename Pack::multi_vector_type>
{
public:
    using base_type = CellFieldBase<Pack, VectorCellField<Pack>,
                                    typename Pack::multi_vector_type>;
    using mesh_type = typename base_type::mesh_type;
    using vector_type = typename base_type::vector_type;
    using map_type = typename base_type::map_type;
    using import_type = typename base_type::import_type;
    using scalar_type = typename base_type::scalar_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;

    static constexpr size_t num_components = 3;

    explicit VectorCellField(SP<const mesh_type> mesh,
                             std::string name = std::string(),
                             bool zero_out = true);

    VectorCellField(SP<const mesh_type> mesh,
                    const vec_type& initial_value,
                    std::string name = std::string());

    void put_scalar(const vec_type& value);

    vec_type value(local_ordinal_type cell_lid) const;
    vec_type owned_value(local_ordinal_type cell_lid) const;
    vec_type local_value(local_ordinal_type cell_lid) const;

    scalar_type component_value(local_ordinal_type cell_lid,
                                size_t component) const;
    scalar_type local_component_value(local_ordinal_type cell_lid,
                                      size_t component) const;

    void set_value(local_ordinal_type cell_lid, const vec_type& value);
    void set_owned_value(local_ordinal_type cell_lid, const vec_type& value);
    void set_component_value(local_ordinal_type cell_lid,
                             size_t component,
                             const scalar_type& value);
    void set_owned_component_value(local_ordinal_type cell_lid,
                                   size_t component,
                                   const scalar_type& value);

    bool is_owned_cell(local_ordinal_type cell_lid) const;
    bool is_local_cell(local_ordinal_type cell_lid) const;

private:
    static scalar_type component(const vec_type& value, size_t index);
    static void check_component(size_t component);
};

/**
 * @brief Construct a vector cell field with three components.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Optional field name for I/O.
 * @param zero_out If true, initialize all entries to zero.
 */
template<TpetraTypePack Pack>
VectorCellField<Pack>::VectorCellField(SP<const mesh_type> mesh,
                                       std::string name,
                                       bool zero_out)
    : base_type(std::move(mesh), std::move(name), num_components, zero_out,
                "VectorCellField")
{
}

/**
 * @brief Construct a vector cell field initialized with a uniform vector value.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param initial_value Vector value to fill all owned cell entries.
 * @param name Optional field name for I/O.
 */
template<TpetraTypePack Pack>
VectorCellField<Pack>::VectorCellField(SP<const mesh_type> mesh,
                                       const vec_type& initial_value,
                                       std::string name)
    : VectorCellField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

/**
 * @brief Extract a single component from a 3D vector.
 *
 * @tparam Pack Tpetra type pack.
 * @param value The 3D vector.
 * @param index Component index (0, 1, or 2).
 * @return The scalar component value.
 */
template<TpetraTypePack Pack>
auto VectorCellField<Pack>::component(const vec_type& value,
                                      size_t index) -> scalar_type
{
    return value.component(index);
}

/**
 * @brief Validate that a component index is in range [0, num_components).
 *
 * @tparam Pack Tpetra type pack.
 * @param component Component index to validate.
 * @throws std::out_of_range if @p component is out of bounds.
 */
template<TpetraTypePack Pack>
void VectorCellField<Pack>::check_component(size_t component)
{
    if (component >= num_components)
    {
        throw std::out_of_range("VectorCellField component index is out of bounds.");
    }
}

/**
 * @brief Set all owned and overlap entries to a uniform vector value.
 *
 * @tparam Pack Tpetra type pack.
 * @param value Vector value to assign to every entry (per component).
 */
template<TpetraTypePack Pack>
void VectorCellField<Pack>::put_scalar(const vec_type& value)
{
    for (size_t component_id = 0; component_id < num_components; ++component_id)
    {
        this->d_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
        this->d_overlap_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
    }
}

/**
 * @brief Read the vector value stored at a locally owned cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @return 3D vector stored at the cell.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
auto VectorCellField<Pack>::value(local_ordinal_type cell_lid) const -> vec_type
{
    return {
        component_value(cell_lid, 0),
        component_value(cell_lid, 1),
        component_value(cell_lid, 2)
    };
}

/**
 * @brief Read the vector value stored at a locally owned cell (alias for value()).
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @return 3D vector stored at the cell.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
auto VectorCellField<Pack>::owned_value(local_ordinal_type cell_lid) const -> vec_type
{
    return value(cell_lid);
}

/**
 * @brief Read the vector value from the overlap (local) storage for a cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @return 3D vector from the overlap storage.
 * @throws std::out_of_range if @p cell_lid is out of bounds.
 */
template<TpetraTypePack Pack>
auto VectorCellField<Pack>::local_value(local_ordinal_type cell_lid) const -> vec_type
{
    return {
        local_component_value(cell_lid, 0),
        local_component_value(cell_lid, 1),
        local_component_value(cell_lid, 2)
    };
}

/**
 * @brief Read a single component value from a locally owned cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param component_id Component index (0, 1, or 2).
 * @return Scalar component value.
 * @throws std::out_of_range if @p cell_lid is out of bounds/not owned, or @p component_id is invalid.
 */
template<TpetraTypePack Pack>
auto VectorCellField<Pack>::component_value(local_ordinal_type cell_lid,
                                            size_t component_id) const
    -> scalar_type
{
    check_component(component_id);
    return this->d_data.getData(component_id)[this->owned_row_for_cell(cell_lid)];
}

/**
 * @brief Read a single component value from the overlap storage for a cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param component_id Component index (0, 1, or 2).
 * @return Scalar component value from overlap storage.
 * @throws std::out_of_range if @p cell_lid is out of bounds or @p component_id is invalid.
 */
template<TpetraTypePack Pack>
auto VectorCellField<Pack>::local_component_value(
    local_ordinal_type cell_lid,
    size_t component_id) const -> scalar_type
{
    check_component(component_id);
    return this->d_overlap_data.getData(component_id)[this->local_row_for_cell(cell_lid)];
}

/**
 * @brief Write a 3D vector to both the owned and overlap storage for a cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param value Vector value to store.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_value(local_ordinal_type cell_lid,
                                      const vec_type& value)
{
    for (size_t component_id = 0; component_id < num_components; ++component_id)
    {
        set_component_value(cell_lid, component_id, component(value, component_id));
    }
}

/**
 * @brief Write a 3D vector to the owned storage only.
 *
 * Caller must sync ghosts before reading overlap data.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param value Vector value to store.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_owned_value(local_ordinal_type cell_lid,
                                            const vec_type& value)
{
    for (size_t component_id = 0; component_id < num_components; ++component_id)
    {
        set_owned_component_value(cell_lid, component_id,
                                  component(value, component_id));
    }
}

/**
 * @brief Write a single component value to both owned and overlap storage.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param component_id Component index (0, 1, or 2).
 * @param value Scalar value to store for the given component.
 * @throws std::out_of_range if @p cell_lid is out of bounds/not owned, or @p component_id is invalid.
 */
template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_component_value(
    local_ordinal_type cell_lid,
    size_t component_id,
    const scalar_type& value)
{
    check_component(component_id);
    this->d_data.replaceLocalValue(
        this->owned_row_for_cell(cell_lid), component_id, value);
    this->d_overlap_data.replaceLocalValue(
        this->local_row_for_cell(cell_lid), component_id, value);
}

/**
 * @brief Write a single component value to the owned storage only.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param component_id Component index (0, 1, or 2).
 * @param value Scalar value to store for the given component.
 * @throws std::out_of_range if @p cell_lid is out of bounds/not owned, or @p component_id is invalid.
 */
template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_owned_component_value(
    local_ordinal_type cell_lid,
    size_t component_id,
    const scalar_type& value)
{
    check_component(component_id);
    this->d_data.replaceLocalValue(
        this->owned_row_for_cell(cell_lid), component_id, value);
}

/**
 * @brief Check whether a cell is locally owned.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell to check.
 * @return true if the cell is owned by this rank.
 * @throws std::out_of_range if @p cell_lid is out of bounds.
 */
template<TpetraTypePack Pack>
bool VectorCellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return this->d_mesh->is_owned_cell(cell_lid);
}

/**
 * @brief Check whether a cell local ID is valid (always true if in-bounds).
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell to check.
 * @return true always (every valid cell is "local" in the overlap sense).
 * @throws std::out_of_range if @p cell_lid is out of bounds.
 */
template<TpetraTypePack Pack>
bool VectorCellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return true;
}

extern template class VectorCellField<DefaultTpetraTypes>;

} // namespace SimpleFluid
