/**
 * @file VectorCellField.hh
 * @brief Three-component cell-centered vector field backed by Tpetra::MultiVector.
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

    static constexpr std::size_t num_components = 3;

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
                                std::size_t component) const;
    scalar_type local_component_value(local_ordinal_type cell_lid,
                                      std::size_t component) const;

    void set_value(local_ordinal_type cell_lid, const vec_type& value);
    void set_owned_value(local_ordinal_type cell_lid, const vec_type& value);
    void set_component_value(local_ordinal_type cell_lid,
                             std::size_t component,
                             const scalar_type& value);
    void set_owned_component_value(local_ordinal_type cell_lid,
                                   std::size_t component,
                                   const scalar_type& value);

    bool is_owned_cell(local_ordinal_type cell_lid) const;
    bool is_local_cell(local_ordinal_type cell_lid) const;

private:
    static scalar_type component(const vec_type& value, std::size_t index);
    static void check_component(std::size_t component);
};

template<TpetraTypePack Pack>
VectorCellField<Pack>::VectorCellField(SP<const mesh_type> mesh,
                                       std::string name,
                                       bool zero_out)
    : base_type(std::move(mesh), std::move(name), num_components, zero_out,
                "VectorCellField")
{
}

template<TpetraTypePack Pack>
VectorCellField<Pack>::VectorCellField(SP<const mesh_type> mesh,
                                       const vec_type& initial_value,
                                       std::string name)
    : VectorCellField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::component(const vec_type& value,
                                      std::size_t index) -> scalar_type
{
    return value.component(index);
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::check_component(std::size_t component)
{
    if (component >= num_components)
    {
        throw std::out_of_range("VectorCellField component index is out of bounds.");
    }
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::put_scalar(const vec_type& value)
{
    for (std::size_t component_id = 0; component_id < num_components; ++component_id)
    {
        this->d_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
        this->d_overlap_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
    }
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::value(local_ordinal_type cell_lid) const -> vec_type
{
    return {
        component_value(cell_lid, 0),
        component_value(cell_lid, 1),
        component_value(cell_lid, 2)
    };
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::owned_value(local_ordinal_type cell_lid) const -> vec_type
{
    return value(cell_lid);
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::local_value(local_ordinal_type cell_lid) const -> vec_type
{
    return {
        local_component_value(cell_lid, 0),
        local_component_value(cell_lid, 1),
        local_component_value(cell_lid, 2)
    };
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::component_value(local_ordinal_type cell_lid,
                                            std::size_t component_id) const
    -> scalar_type
{
    check_component(component_id);
    return this->d_data.getData(component_id)[this->owned_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::local_component_value(
    local_ordinal_type cell_lid,
    std::size_t component_id) const -> scalar_type
{
    check_component(component_id);
    return this->d_overlap_data.getData(component_id)[this->local_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_value(local_ordinal_type cell_lid,
                                      const vec_type& value)
{
    for (std::size_t component_id = 0; component_id < num_components; ++component_id)
    {
        set_component_value(cell_lid, component_id, component(value, component_id));
    }
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_owned_value(local_ordinal_type cell_lid,
                                            const vec_type& value)
{
    for (std::size_t component_id = 0; component_id < num_components; ++component_id)
    {
        set_owned_component_value(cell_lid, component_id,
                                  component(value, component_id));
    }
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_component_value(
    local_ordinal_type cell_lid,
    std::size_t component_id,
    const scalar_type& value)
{
    check_component(component_id);
    this->d_data.replaceLocalValue(
        this->owned_row_for_cell(cell_lid), component_id, value);
    this->d_overlap_data.replaceLocalValue(
        this->local_row_for_cell(cell_lid), component_id, value);
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_owned_component_value(
    local_ordinal_type cell_lid,
    std::size_t component_id,
    const scalar_type& value)
{
    check_component(component_id);
    this->d_data.replaceLocalValue(
        this->owned_row_for_cell(cell_lid), component_id, value);
}

template<TpetraTypePack Pack>
bool VectorCellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return this->d_mesh->is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
bool VectorCellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    const auto row =
        this->d_local_row_by_cell_lid[static_cast<std::size_t>(cell_lid)];
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

} // namespace SimpleFluid
