/**
 * @file CellField.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief cell-centered scalar field backed by Tpetra::Vector
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "fields/CellFieldBase.hh"

#include <cstddef>
#include <string>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Scalar field defined on the owned cells of a mesh.
 *
 * Stores values in a Tpetra vector using the mesh's owned-cell map.
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class CellField
    : public CellFieldBase<Pack, CellField<Pack>, typename Pack::vector_type>
{
public:
    using base_type = CellFieldBase<Pack, CellField<Pack>, typename Pack::vector_type>;
    using mesh_type = typename base_type::mesh_type;
    using vector_type = typename base_type::vector_type;
    using map_type = typename base_type::map_type;
    using import_type = typename base_type::import_type;
    using scalar_type = typename base_type::scalar_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using global_ordinal_type = typename base_type::global_ordinal_type;

    explicit CellField(SP<const mesh_type> mesh,
                       std::string name = std::string(),
                       bool zero_out = true);

    CellField(SP<const mesh_type> mesh,
              const scalar_type& initial_value,
              std::string name = std::string());

    vector_type& vector() noexcept { return this->d_data; }
    const vector_type& vector() const noexcept { return this->d_data; }

    void put_scalar(const scalar_type& value);

    scalar_type value(local_ordinal_type cell_lid) const;
    scalar_type owned_value(local_ordinal_type cell_lid) const;
    scalar_type local_value(local_ordinal_type cell_lid) const;
    scalar_type global_value(global_ordinal_type cell_gid) const;

    void set_value(local_ordinal_type cell_lid, const scalar_type& value);
    /**
     * @brief Update only owned storage; caller must sync ghosts before reading overlap data.
     */
    void set_owned_value(local_ordinal_type cell_lid, const scalar_type& value);
    void set_global_value(global_ordinal_type cell_gid, const scalar_type& value);

    void sum_into_value(local_ordinal_type cell_lid, const scalar_type& value);
    void sum_into_global_value(global_ordinal_type cell_gid, const scalar_type& value);

    bool is_owned_cell(local_ordinal_type cell_lid) const;
    bool is_local_cell(local_ordinal_type cell_lid) const;
    bool is_owned_global_cell(global_ordinal_type cell_gid) const;
    bool is_local_global_cell(global_ordinal_type cell_gid) const;
};

/**
 * @brief Construct a cell field over the owned cells of a mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Optional field name for I/O.
 * @param zero_out If true, initialize all entries to zero.
 */
template<TpetraTypePack Pack>
CellField<Pack>::CellField(SP<const mesh_type> mesh,
                           std::string name,
                           bool zero_out)
    : base_type(std::move(mesh), std::move(name), zero_out, "CellField")
{
}

/**
 * @brief Construct a cell field initialized with a uniform value.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param initial_value Scalar value to fill all entries.
 * @param name Optional field name for I/O.
 */
template<TpetraTypePack Pack>
CellField<Pack>::CellField(SP<const mesh_type> mesh,
                           const scalar_type& initial_value,
                           std::string name)
    : CellField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

/**
 * @brief Set all entries in both owned and overlap data vectors to a uniform value.
 *
 * @tparam Pack Tpetra type pack.
 * @param value Scalar value to assign to all entries.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::put_scalar(const scalar_type& value)
{
    this->d_data.putScalar(value);
    this->d_overlap_data.putScalar(value);
}

template<TpetraTypePack Pack>
auto CellField<Pack>::value(local_ordinal_type cell_lid) const -> scalar_type
{
    return this->d_data.getData()[this->owned_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto CellField<Pack>::owned_value(local_ordinal_type cell_lid) const -> scalar_type
{
    return value(cell_lid);
}

template<TpetraTypePack Pack>
auto CellField<Pack>::local_value(local_ordinal_type cell_lid) const -> scalar_type
{
    return this->d_overlap_data.getData()[this->local_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto CellField<Pack>::global_value(global_ordinal_type cell_gid) const -> scalar_type
{
    return this->d_data.getData()[this->owned_row_for_global_cell(cell_gid)];
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_value(local_ordinal_type cell_lid,
                                const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_cell(cell_lid), value);
    this->d_overlap_data.replaceLocalValue(this->local_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_owned_value(local_ordinal_type cell_lid,
                                      const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_global_value(global_ordinal_type cell_gid,
                                       const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_global_cell(cell_gid), value);
    this->d_overlap_data.replaceLocalValue(this->local_row_for_global_cell(cell_gid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_value(local_ordinal_type cell_lid,
                                     const scalar_type& value)
{
    this->d_data.sumIntoLocalValue(this->owned_row_for_cell(cell_lid), value);
    this->d_overlap_data.sumIntoLocalValue(this->local_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_global_value(global_ordinal_type cell_gid,
                                            const scalar_type& value)
{
    this->d_data.sumIntoLocalValue(this->owned_row_for_global_cell(cell_gid), value);
    this->d_overlap_data.sumIntoLocalValue(this->local_row_for_global_cell(cell_gid), value);
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return this->d_mesh->is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return is_local_global_cell(this->d_mesh->cell_global_id(cell_lid));
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_global_cell(global_ordinal_type cell_gid) const
{
    const auto row = this->d_data.getMap()->getLocalElement(cell_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_local_global_cell(global_ordinal_type cell_gid) const
{
    const auto row = this->d_overlap_data.getMap()->getLocalElement(cell_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

} // namespace SimpleFluid
