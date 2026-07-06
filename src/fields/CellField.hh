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
 * @brief Construct a cell field over owned cells.
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
 * @brief Construct a cell field initialized with a uniform scalar value.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param initial_value Scalar value to fill all owned cell entries.
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
 * @brief Set all owned and overlap entries to a uniform scalar value.
 *
 * @tparam Pack Tpetra type pack.
 * @param value Scalar value to assign to every entry.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::put_scalar(const scalar_type& value)
{
    this->d_data.putScalar(value);
    this->d_overlap_data.putScalar(value);
}

/**
 * @brief Read the value stored at a locally owned cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @return Stored scalar value.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::value(local_ordinal_type cell_lid) const -> scalar_type
{
    return this->d_data.getData()[this->owned_row_for_cell(cell_lid)];
}

/**
 * @brief Read the value stored at a locally owned cell (alias for value()).
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @return Stored scalar value.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::owned_value(local_ordinal_type cell_lid) const -> scalar_type
{
    return value(cell_lid);
}

/**
 * @brief Read the value from the overlap (local) storage for a given cell.
 *
 * Includes ghost cells after sync_ghosts() has been called.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @return Stored scalar value from the overlap vector.
 * @throws std::out_of_range if @p cell_lid is out of bounds.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::local_value(local_ordinal_type cell_lid) const -> scalar_type
{
    return this->d_overlap_data.getData()[this->local_row_for_cell(cell_lid)];
}

/**
 * @brief Read the value stored at a cell identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Global ID of the cell.
 * @return Stored scalar value.
 * @throws std::out_of_range if @p cell_gid is not owned by this rank.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::global_value(global_ordinal_type cell_gid) const -> scalar_type
{
    return this->d_data.getData()[this->owned_row_for_global_cell(cell_gid)];
}

/**
 * @brief Write a value to both the owned and overlap storage for a cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param value Scalar value to store.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::set_value(local_ordinal_type cell_lid,
                                const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_cell(cell_lid), value);
    this->d_overlap_data.replaceLocalValue(this->local_row_for_cell(cell_lid), value);
}

/**
 * @brief Write a value to the owned storage only.
 *
 * Caller must sync ghosts before reading overlap data.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param value Scalar value to store.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::set_owned_value(local_ordinal_type cell_lid,
                                      const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_cell(cell_lid), value);
}

/**
 * @brief Write a value to a cell identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Global ID of the cell.
 * @param value Scalar value to store.
 * @throws std::out_of_range if @p cell_gid is not owned by this rank.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::set_global_value(global_ordinal_type cell_gid,
                                       const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_global_cell(cell_gid), value);
    this->d_overlap_data.replaceLocalValue(this->local_row_for_global_cell(cell_gid), value);
}

/**
 * @brief Accumulate (sum) a value into both the owned and overlap storage.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Local ID of the cell.
 * @param value Scalar value to add.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_value(local_ordinal_type cell_lid,
                                     const scalar_type& value)
{
    this->d_data.sumIntoLocalValue(this->owned_row_for_cell(cell_lid), value);
    this->d_overlap_data.sumIntoLocalValue(this->local_row_for_cell(cell_lid), value);
}

/**
 * @brief Accumulate (sum) a value into a cell identified by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Global ID of the cell.
 * @param value Scalar value to add.
 * @throws std::out_of_range if @p cell_gid is not owned by this rank.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_global_value(global_ordinal_type cell_gid,
                                            const scalar_type& value)
{
    this->d_data.sumIntoLocalValue(this->owned_row_for_global_cell(cell_gid), value);
    this->d_overlap_data.sumIntoLocalValue(this->local_row_for_global_cell(cell_gid), value);
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
bool CellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
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
bool CellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    this->check_cell_lid(cell_lid);
    return true;
}

/**
 * @brief Check whether a global cell ID is owned by this rank.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Global ID of the cell to check.
 * @return true if the cell is owned by this rank.
 */
template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_global_cell(global_ordinal_type cell_gid) const
{
    const auto tpetra_gid = this->d_mesh->mesh_gid_to_tpetra_gid(cell_gid);
    if (tpetra_gid == invalid_id<global_ordinal_type>())
    {
        return false;
    }
    const auto row = this->d_data.getMap()->getLocalElement(tpetra_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

/**
 * @brief Check whether a global cell ID is present in the overlap (local) map.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Global ID of the cell to check.
 * @return true if the cell is in the overlap map of this rank.
 */
template<TpetraTypePack Pack>
bool CellField<Pack>::is_local_global_cell(global_ordinal_type cell_gid) const
{
    const auto tpetra_gid = this->d_mesh->mesh_gid_to_tpetra_gid(cell_gid);
    if (tpetra_gid == invalid_id<global_ordinal_type>())
    {
        return false;
    }
    const auto row = this->d_overlap_data.getMap()->getLocalElement(tpetra_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

extern template class CellField<DefaultTpetraTypes>;

} // namespace SimpleFluid
