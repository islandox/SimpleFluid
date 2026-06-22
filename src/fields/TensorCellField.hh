/**
 * @file TensorCellField.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Nine-component cell-centered tensor field backed by Tpetra::MultiVector.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/CellFieldBase.hh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Cell-centered 3x3 tensor field stored as a Tpetra::MultiVector.
 *
 * Components are stored row-major. Each tensor row is represented by a Vec3.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class TensorCellField
    : public CellFieldBase<Pack, TensorCellField<Pack>,
                           typename Pack::multi_vector_type>
{
public:
    using base_type = CellFieldBase<Pack, TensorCellField<Pack>,
                                    typename Pack::multi_vector_type>;
    using mesh_type = typename base_type::mesh_type;
    using vector_type = typename base_type::vector_type;
    using map_type = typename base_type::map_type;
    using import_type = typename base_type::import_type;
    using scalar_type = typename base_type::scalar_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;
    using tensor_type = std::array<vec_type, 3>;

    static constexpr size_t num_rows = 3;
    static constexpr size_t num_columns = 3;
    static constexpr size_t num_components = num_rows * num_columns;

    explicit TensorCellField(SP<const mesh_type> mesh,
                             std::string name = std::string(),
                             bool zero_out = true);

    TensorCellField(SP<const mesh_type> mesh,
                    const tensor_type& initial_value,
                    std::string name = std::string());

    void put_scalar(const tensor_type& value);

    tensor_type value(local_ordinal_type cell_lid) const;
    tensor_type owned_value(local_ordinal_type cell_lid) const;
    tensor_type local_value(local_ordinal_type cell_lid) const;

    scalar_type component_value(local_ordinal_type cell_lid,
                                size_t component) const;
    scalar_type component_value(local_ordinal_type cell_lid,
                                size_t row,
                                size_t column) const;
    scalar_type local_component_value(local_ordinal_type cell_lid,
                                      size_t component) const;
    scalar_type local_component_value(local_ordinal_type cell_lid,
                                      size_t row,
                                      size_t column) const;

    void set_value(local_ordinal_type cell_lid, const tensor_type& value);
    void set_owned_value(local_ordinal_type cell_lid,
                         const tensor_type& value);
    void set_component_value(local_ordinal_type cell_lid,
                             size_t component,
                             const scalar_type& value);
    void set_component_value(local_ordinal_type cell_lid,
                             size_t row,
                             size_t column,
                             const scalar_type& value);
    void set_owned_component_value(local_ordinal_type cell_lid,
                                   size_t component,
                                   const scalar_type& value);
    void set_owned_component_value(local_ordinal_type cell_lid,
                                   size_t row,
                                   size_t column,
                                   const scalar_type& value);

    bool is_owned_cell(local_ordinal_type cell_lid) const;
    bool is_local_cell(local_ordinal_type cell_lid) const;

private:
    static size_t component_index(size_t row, size_t column);
    static scalar_type component(const tensor_type& value, size_t index);
    static void check_component(size_t component);
};

template<TpetraTypePack Pack>
TensorCellField<Pack>::TensorCellField(SP<const mesh_type> mesh,
                                       std::string name,
                                       bool zero_out)
    : base_type(std::move(mesh), std::move(name), num_components, zero_out,
                "TensorCellField")
{
}

template<TpetraTypePack Pack>
TensorCellField<Pack>::TensorCellField(SP<const mesh_type> mesh,
                                       const tensor_type& initial_value,
                                       std::string name)
    : TensorCellField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

template<TpetraTypePack Pack>
size_t TensorCellField<Pack>::component_index(size_t row, size_t column)
{
    if (row >= num_rows || column >= num_columns)
    {
        throw std::out_of_range(
            "TensorCellField row or column index is out of bounds.");
    }
    return row * num_columns + column;
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::component(const tensor_type& value,
                                      size_t index) -> scalar_type
{
    check_component(index);
    return value[index / num_columns].component(index % num_columns);
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::check_component(size_t component)
{
    if (component >= num_components)
    {
        throw std::out_of_range(
            "TensorCellField component index is out of bounds.");
    }
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::put_scalar(const tensor_type& value)
{
    for (size_t component_id = 0; component_id < num_components; ++component_id)
    {
        this->d_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
        this->d_overlap_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
    }
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::value(local_ordinal_type cell_lid) const
    -> tensor_type
{
    return {
        vec_type{component_value(cell_lid, 0, 0),
                 component_value(cell_lid, 0, 1),
                 component_value(cell_lid, 0, 2)},
        vec_type{component_value(cell_lid, 1, 0),
                 component_value(cell_lid, 1, 1),
                 component_value(cell_lid, 1, 2)},
        vec_type{component_value(cell_lid, 2, 0),
                 component_value(cell_lid, 2, 1),
                 component_value(cell_lid, 2, 2)}
    };
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::owned_value(local_ordinal_type cell_lid) const
    -> tensor_type
{
    return value(cell_lid);
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::local_value(local_ordinal_type cell_lid) const
    -> tensor_type
{
    return {
        vec_type{local_component_value(cell_lid, 0, 0),
                 local_component_value(cell_lid, 0, 1),
                 local_component_value(cell_lid, 0, 2)},
        vec_type{local_component_value(cell_lid, 1, 0),
                 local_component_value(cell_lid, 1, 1),
                 local_component_value(cell_lid, 1, 2)},
        vec_type{local_component_value(cell_lid, 2, 0),
                 local_component_value(cell_lid, 2, 1),
                 local_component_value(cell_lid, 2, 2)}
    };
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::component_value(
    local_ordinal_type cell_lid,
    size_t component_id) const -> scalar_type
{
    check_component(component_id);
    return this->d_data.getData(component_id)[
        this->owned_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::component_value(
    local_ordinal_type cell_lid,
    size_t row,
    size_t column) const -> scalar_type
{
    return component_value(cell_lid, component_index(row, column));
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::local_component_value(
    local_ordinal_type cell_lid,
    size_t component_id) const -> scalar_type
{
    check_component(component_id);
    return this->d_overlap_data.getData(component_id)[
        this->local_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto TensorCellField<Pack>::local_component_value(
    local_ordinal_type cell_lid,
    size_t row,
    size_t column) const -> scalar_type
{
    return local_component_value(cell_lid, component_index(row, column));
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::set_value(local_ordinal_type cell_lid,
                                      const tensor_type& value)
{
    for (size_t component_id = 0; component_id < num_components; ++component_id)
    {
        set_component_value(cell_lid, component_id,
                            component(value, component_id));
    }
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::set_owned_value(local_ordinal_type cell_lid,
                                            const tensor_type& value)
{
    for (size_t component_id = 0; component_id < num_components; ++component_id)
    {
        set_owned_component_value(cell_lid, component_id,
                                  component(value, component_id));
    }
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::set_component_value(
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

template<TpetraTypePack Pack>
void TensorCellField<Pack>::set_component_value(
    local_ordinal_type cell_lid,
    size_t row,
    size_t column,
    const scalar_type& value)
{
    set_component_value(cell_lid, component_index(row, column), value);
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::set_owned_component_value(
    local_ordinal_type cell_lid,
    size_t component_id,
    const scalar_type& value)
{
    check_component(component_id);
    this->d_data.replaceLocalValue(
        this->owned_row_for_cell(cell_lid), component_id, value);
}

template<TpetraTypePack Pack>
void TensorCellField<Pack>::set_owned_component_value(
    local_ordinal_type cell_lid,
    size_t row,
    size_t column,
    const scalar_type& value)
{
    set_owned_component_value(cell_lid, component_index(row, column), value);
}

template<TpetraTypePack Pack>
bool TensorCellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return this->d_mesh->is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
bool TensorCellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return true;
}

} // namespace SimpleFluid
