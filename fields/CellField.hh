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

#include "geometry/Mesh.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_CombineMode.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
{
public:
    using mesh_type = Mesh<Pack>;
    using vector_type = typename Pack::vector_type;
    using map_type = typename Pack::map_type;
    using import_type = typename Pack::import_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    explicit CellField(SP<const mesh_type> mesh,
                       std::string name = std::string(),
                       bool zero_out = true);

    CellField(SP<const mesh_type> mesh,
              const scalar_type& initial_value,
              std::string name = std::string());

    const std::string& name() const noexcept { return d_name; }
    void set_name(std::string name) { d_name = std::move(name); }

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    RCP<const map_type> map() const { return d_data.getMap(); }
    RCP<const map_type> owned_map() const { return d_data.getMap(); }
    RCP<const map_type> overlap_map() const { return d_overlap_data.getMap(); }

    vector_type& data() noexcept { return d_data; }
    const vector_type& data() const noexcept { return d_data; }

    vector_type& vector() noexcept { return d_data; }
    const vector_type& vector() const noexcept { return d_data; }

    vector_type& owned_data() noexcept { return d_data; }
    const vector_type& owned_data() const noexcept { return d_data; }

    vector_type& overlap_data() noexcept { return d_overlap_data; }
    const vector_type& overlap_data() const noexcept { return d_overlap_data; }

    std::size_t num_owned_cells() const
    {
        return d_data.getMap()->getLocalNumElements();
    }

    std::size_t num_local_cells() const
    {
        return d_overlap_data.getMap()->getLocalNumElements();
    }

    void put_scalar(const scalar_type& value);
    void sync_ghosts();

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

private:
    static RCP<const map_type> require_owned_map(const SP<const mesh_type>& mesh);
    static RCP<const map_type> require_overlap_map(const SP<const mesh_type>& mesh);

    local_ordinal_type owned_row_for_cell(local_ordinal_type cell_lid) const;
    local_ordinal_type owned_row_for_global_cell(global_ordinal_type cell_gid) const;
    local_ordinal_type local_row_for_cell(local_ordinal_type cell_lid) const;
    local_ordinal_type local_row_for_global_cell(global_ordinal_type cell_gid) const;
    void check_cell_lid(local_ordinal_type cell_lid) const;
    void cache_cell_rows();

    std::string d_name;
    SP<const mesh_type> d_mesh;
    vector_type d_data;
    vector_type d_overlap_data;
    RCP<const import_type> d_owned_to_overlap_import;
    std::vector<local_ordinal_type> d_owned_row_by_cell_lid;
    std::vector<local_ordinal_type> d_local_row_by_cell_lid;
};

/**
 * @brief Construct a cell field over the owned cells of a mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Optional field name for I/O.
 * @param zero_out If true, initialize all entries to zero.
 * @throws std::invalid_argument if the mesh is null.
 * @throws std::runtime_error if the mesh does not have an owned-cell map.
 */
template<TpetraTypePack Pack>
CellField<Pack>::CellField(SP<const mesh_type> mesh,
                           std::string name,
                           bool zero_out)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_data(require_owned_map(d_mesh), zero_out),
      d_overlap_data(require_overlap_map(d_mesh), false),
      d_owned_to_overlap_import(
          Teuchos::rcp(new import_type(d_data.getMap(), d_overlap_data.getMap())))
{
    cache_cell_rows();
    if (zero_out)
    {
        sync_ghosts();
    }
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
 * @brief Validate and retrieve the owned-cell map from the mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the mesh.
 * @return RCP to the owned-cell Tpetra map.
 * @throws std::invalid_argument if the mesh is null.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::require_owned_map(const SP<const mesh_type>& mesh)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("CellField requires a non-null mesh.");
    }

    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error("CellField requires an assembled mesh with an owned-cell map.");
    }

    return map;
}

/**
 * @brief Validate and retrieve the overlap-cell map from the mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the mesh.
 * @return RCP to the overlap-cell Tpetra map.
 * @throws std::invalid_argument if the mesh is null.
 * @throws std::runtime_error if the mesh has no overlap-cell map.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::require_overlap_map(const SP<const mesh_type>& mesh)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("CellField requires a non-null mesh.");
    }

    auto map = mesh->overlap_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error("CellField requires an assembled mesh with an overlap-cell map.");
    }

    return map;
}

/**
 * @brief Validate that a cell local ID is in range.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Cell local ID to validate.
 * @throws std::out_of_range if the ID is negative or exceeds the local cell count.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::check_cell_lid(local_ordinal_type cell_lid) const
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (cell_lid < 0)
        {
            throw std::out_of_range("Cell local id cannot be negative: "
                                  + std::to_string(cell_lid));
        }
    }

    if (static_cast<std::size_t>(cell_lid) >= d_mesh->num_local_cells())
    {
        throw std::out_of_range("Cell local id is out of bounds: "
                              + std::to_string(cell_lid));
    }
}

/**
 * @brief Cache Tpetra row indices for mesh local IDs.
 *
 * The mesh-to-map relationship is fixed after assembly, so hot-path local
 * field access can avoid repeated Tpetra map lookups.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::cache_cell_rows()
{
    d_owned_row_by_cell_lid.assign(d_mesh->num_local_cells(),
                                   Teuchos::OrdinalTraits<local_ordinal_type>::invalid());
    d_local_row_by_cell_lid.assign(d_mesh->num_local_cells(),
                                   Teuchos::OrdinalTraits<local_ordinal_type>::invalid());

    for (std::size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto cell_gid = d_mesh->cell_global_id(cell_lid);
        d_local_row_by_cell_lid[cell] =
            d_overlap_data.getMap()->getLocalElement(cell_gid);
        if (d_local_row_by_cell_lid[cell]
            == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
        {
            throw std::runtime_error("CellField overlap map is missing a local cell.");
        }

        if (d_mesh->is_owned_cell(cell_lid))
        {
            d_owned_row_by_cell_lid[cell] =
                d_data.getMap()->getLocalElement(cell_gid);
            if (d_owned_row_by_cell_lid[cell]
                == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
            {
                throw std::runtime_error("CellField owned map is missing an owned cell.");
            }
        }
    }
}

/**
 * @brief Look up the owned Tpetra row index for a given cell global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Cell global ID.
 * @return Local row index in the owned data vector.
 * @throws std::out_of_range if the cell is not owned by this rank.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::owned_row_for_global_cell(global_ordinal_type cell_gid) const
    -> local_ordinal_type
{
    const auto owned_row = d_data.getMap()->getLocalElement(cell_gid);
    if (owned_row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    {
        throw std::out_of_range("Cell global id is not owned by this rank: "
                              + std::to_string(cell_gid));
    }

    return owned_row;
}

/**
 * @brief Look up the owned Tpetra row index for a given cell local ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Cell local ID.
 * @return Local row index in the owned data vector.
 * @throws std::out_of_range if the cell is not owned by this rank.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::owned_row_for_cell(local_ordinal_type cell_lid) const
    -> local_ordinal_type
{
    check_cell_lid(cell_lid);
    if (!d_mesh->is_owned_cell(cell_lid))
    {
        throw std::out_of_range("Cell local id is not owned by this rank: "
                              + std::to_string(cell_lid));
    }

    return d_owned_row_by_cell_lid[static_cast<std::size_t>(cell_lid)];
}

/**
 * @brief Look up the overlap (ghost) Tpetra row index for a given cell global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_gid Cell global ID.
 * @return Local row index in the overlap data vector.
 * @throws std::out_of_range if the cell is not local to this rank.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::local_row_for_global_cell(global_ordinal_type cell_gid) const
    -> local_ordinal_type
{
    const auto local_row = d_overlap_data.getMap()->getLocalElement(cell_gid);
    if (local_row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    {
        throw std::out_of_range("Cell global id is not local to this rank: "
                              + std::to_string(cell_gid));
    }

    return local_row;
}

/**
 * @brief Look up the overlap (ghost) Tpetra row index for a given cell local ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param cell_lid Cell local ID.
 * @return Local row index in the overlap data vector.
 */
template<TpetraTypePack Pack>
auto CellField<Pack>::local_row_for_cell(local_ordinal_type cell_lid) const
    -> local_ordinal_type
{
    check_cell_lid(cell_lid);
    const auto row = d_local_row_by_cell_lid[static_cast<std::size_t>(cell_lid)];
    if (row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    {
        throw std::out_of_range("Cell local id is not local to this rank: "
                              + std::to_string(cell_lid));
    }

    return row;
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
    d_data.putScalar(value);
    d_overlap_data.putScalar(value);
}

/**
 * @brief Synchronize owned values into the overlap (ghost) data vector.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void CellField<Pack>::sync_ghosts()
{
    d_overlap_data.doImport(d_data, *d_owned_to_overlap_import, Tpetra::INSERT);
}

template<TpetraTypePack Pack>
auto CellField<Pack>::value(local_ordinal_type cell_lid) const -> scalar_type
{
    return d_data.getData()[owned_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto CellField<Pack>::owned_value(local_ordinal_type cell_lid) const -> scalar_type
{
    return value(cell_lid);
}

template<TpetraTypePack Pack>
auto CellField<Pack>::local_value(local_ordinal_type cell_lid) const -> scalar_type
{
    return d_overlap_data.getData()[local_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto CellField<Pack>::global_value(global_ordinal_type cell_gid) const -> scalar_type
{
    return d_data.getData()[owned_row_for_global_cell(cell_gid)];
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_value(local_ordinal_type cell_lid,
                                const scalar_type& value)
{
    d_data.replaceLocalValue(owned_row_for_cell(cell_lid), value);
    d_overlap_data.replaceLocalValue(local_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_owned_value(local_ordinal_type cell_lid,
                                      const scalar_type& value)
{
    d_data.replaceLocalValue(owned_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_global_value(global_ordinal_type cell_gid,
                                       const scalar_type& value)
{
    d_data.replaceLocalValue(owned_row_for_global_cell(cell_gid), value);
    d_overlap_data.replaceLocalValue(local_row_for_global_cell(cell_gid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_value(local_ordinal_type cell_lid,
                                     const scalar_type& value)
{
    d_data.sumIntoLocalValue(owned_row_for_cell(cell_lid), value);
    d_overlap_data.sumIntoLocalValue(local_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_global_value(global_ordinal_type cell_gid,
                                            const scalar_type& value)
{
    d_data.sumIntoLocalValue(owned_row_for_global_cell(cell_gid), value);
    d_overlap_data.sumIntoLocalValue(local_row_for_global_cell(cell_gid), value);
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    check_cell_lid(cell_lid);
    return d_mesh->is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    check_cell_lid(cell_lid);
    return is_local_global_cell(d_mesh->cell_global_id(cell_lid));
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_global_cell(global_ordinal_type cell_gid) const
{
    const auto row = d_data.getMap()->getLocalElement(cell_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_local_global_cell(global_ordinal_type cell_gid) const
{
    const auto row = d_overlap_data.getMap()->getLocalElement(cell_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

} // namespace SimpleFluid
