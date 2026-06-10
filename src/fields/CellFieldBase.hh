/**
 * @file CellFieldBase.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared CRTP infrastructure for cell-centered field classes.
 * @version 0.1
 * @date 2026-06-03
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

namespace SimpleFluid
{

/**
 * @brief Shared storage, map, row lookup, and ghost-sync logic for cell fields.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
class CellFieldBase
{
public:
    using mesh_type = Mesh<Pack>;
    using vector_type = StorageVector;
    using map_type = typename Pack::map_type;
    using import_type = typename Pack::import_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    const std::string& name() const noexcept { return d_name; }
    void set_name(std::string name) { d_name = std::move(name); }

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    RCP<const map_type> map() const { return d_data.getMap(); }
    RCP<const map_type> owned_map() const { return d_data.getMap(); }
    RCP<const map_type> overlap_map() const { return d_overlap_data.getMap(); }

    vector_type& data() noexcept { return d_data; }
    const vector_type& data() const noexcept { return d_data; }

    vector_type& owned_data() noexcept { return d_data; }
    const vector_type& owned_data() const noexcept { return d_data; }

    vector_type& overlap_data() noexcept { return d_overlap_data; }
    const vector_type& overlap_data() const noexcept { return d_overlap_data; }

    size_t num_owned_cells() const
    {
        return d_data.getMap()->getLocalNumElements();
    }

    size_t num_local_cells() const
    {
        return d_overlap_data.getMap()->getLocalNumElements();
    }

    void sync_ghosts()
    {
        d_overlap_data.doImport(d_data, *d_owned_to_overlap_import, Tpetra::REPLACE);
    }

protected:
    CellFieldBase(SP<const mesh_type> mesh,
                  std::string name,
                  bool zero_out,
                  const char* class_name);

    CellFieldBase(SP<const mesh_type> mesh,
                  std::string name,
                  size_t num_components,
                  bool zero_out,
                  const char* class_name);

    static RCP<const map_type> require_owned_map(
        const SP<const mesh_type>& mesh,
        const char* class_name);

    static RCP<const map_type> require_overlap_map(
        const SP<const mesh_type>& mesh,
        const char* class_name);

    void check_cell_lid(local_ordinal_type cell_lid) const;

    void check_cell_row_invariant(const char* class_name) const;

    local_ordinal_type owned_row_for_cell(local_ordinal_type cell_lid) const;

    local_ordinal_type owned_row_for_global_cell(global_ordinal_type cell_gid) const;

    local_ordinal_type local_row_for_cell(local_ordinal_type cell_lid) const;

    local_ordinal_type local_row_for_global_cell(global_ordinal_type cell_gid) const;

    std::string d_name;
    SP<const mesh_type> d_mesh;
    vector_type d_data;
    vector_type d_overlap_data;
    RCP<const import_type> d_owned_to_overlap_import;
};

/**
 * @brief Construct a scalar cell-field base with owned and overlap storage.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Field name for I/O.
 * @param zero_out If true, initialize all entries to zero and sync ghosts.
 * @param class_name Class name used in error messages.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
CellFieldBase<Pack, Derived, StorageVector>::CellFieldBase(
    SP<const mesh_type> mesh,
    std::string name,
    bool zero_out,
    const char* class_name)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_data(require_owned_map(d_mesh, class_name), zero_out),
      d_overlap_data(require_overlap_map(d_mesh, class_name), false),
      d_owned_to_overlap_import(
          Teuchos::rcp(new import_type(d_data.getMap(), d_overlap_data.getMap())))
{
    check_cell_row_invariant(class_name);
    if (zero_out)
    {
        sync_ghosts();
    }
}

/**
 * @brief Construct a multi-component cell-field base with owned and overlap storage.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Field name for I/O.
 * @param num_components Number of components (columns in MultiVector).
 * @param zero_out If true, initialize all entries to zero and sync ghosts.
 * @param class_name Class name used in error messages.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
CellFieldBase<Pack, Derived, StorageVector>::CellFieldBase(
    SP<const mesh_type> mesh,
    std::string name,
    size_t num_components,
    bool zero_out,
    const char* class_name)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_data(require_owned_map(d_mesh, class_name), num_components, zero_out),
      d_overlap_data(require_overlap_map(d_mesh, class_name), num_components, false),
      d_owned_to_overlap_import(
          Teuchos::rcp(new import_type(d_data.getMap(), d_overlap_data.getMap())))
{
    check_cell_row_invariant(class_name);
    if (zero_out)
    {
        sync_ghosts();
    }
}

/**
 * @brief Validate and return the mesh's owned-cell map.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param mesh Shared pointer to the assembled mesh.
 * @param class_name Class name used in error messages.
 * @return RCP to the owned-cell Tpetra map.
 * @throws std::invalid_argument if @p mesh is null.
 * @throws std::runtime_error if the mesh does not have an owned-cell map.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::require_owned_map(
    const SP<const mesh_type>& mesh,
    const char* class_name)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(std::string(class_name)
                                  + " requires a non-null mesh.");
    }

    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(std::string(class_name)
                               + " requires an assembled mesh with an owned-cell map.");
    }

    return map;
}

/**
 * @brief Validate and return the mesh's overlap-cell map.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param mesh Shared pointer to the assembled mesh.
 * @param class_name Class name used in error messages.
 * @return RCP to the overlap-cell Tpetra map.
 * @throws std::invalid_argument if @p mesh is null.
 * @throws std::runtime_error if the mesh does not have an overlap-cell map.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::require_overlap_map(
    const SP<const mesh_type>& mesh,
    const char* class_name)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(std::string(class_name)
                                  + " requires a non-null mesh.");
    }

    auto map = mesh->overlap_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(std::string(class_name)
                               + " requires an assembled mesh with an overlap-cell map.");
    }

    return map;
}

/**
 * @brief Validate that a cell local ID is in range.
 *
 * Checks are only active in debug builds or when
 * SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS is defined.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param cell_lid Cell local ID to validate.
 * @throws std::out_of_range if the ID is negative or exceeds the local cell count.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
void CellFieldBase<Pack, Derived, StorageVector>::check_cell_lid(
    local_ordinal_type cell_lid) const
{
    CHECK_BOUNDS(cell_lid, 0, d_mesh->num_local_cells());
}

/**
 * @brief Verify that the owned and overlap map sizes match the mesh and that
 *        local cell IDs correspond to the overlap rows.
 *
 * Only active in debug builds or when SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS
 * is defined.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param class_name Class name used in error messages.
 * @throws std::runtime_error if any invariant is violated.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
void CellFieldBase<Pack, Derived, StorageVector>::check_cell_row_invariant(
    const char* class_name) const
{
    CHECK(d_data.getMap()->getLocalNumElements() == d_mesh->num_owned_cells());
    CHECK(d_overlap_data.getMap()->getLocalNumElements() == d_mesh->num_local_cells());
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)

    const auto invalid_row =
        Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    const auto num_owned = d_mesh->num_owned_cells();
    for (size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);

        const auto tpetra_gid =
            cell < num_owned
                ? d_mesh->tpetra_gid_offset() + static_cast<global_ordinal_type>(cell)
                : d_mesh->mesh_gid_to_tpetra_gid(d_mesh->cell_global_id(cell_lid));

        const auto local_row =
            d_overlap_data.getMap()->getLocalElement(tpetra_gid);
        if (local_row == invalid_row)
        {
            throw std::runtime_error(std::string(class_name)
                                   + " overlap map is missing a local cell.");
        }
        if (local_row != cell_lid)
        {
            throw std::runtime_error(std::string(class_name)
                                   + " requires local cell IDs to match overlap rows.");
        }

        if (d_mesh->is_owned_cell(cell_lid))
        {
            const auto owned_row =
                d_data.getMap()->getLocalElement(tpetra_gid);
            if (owned_row == invalid_row)
            {
                throw std::runtime_error(std::string(class_name)
                                       + " owned map is missing an owned cell.");
            }
            if (owned_row != cell_lid)
            {
                throw std::runtime_error(std::string(class_name)
                                       + " requires owned cell IDs to match owned rows.");
            }
        }
    }
#endif
}

/**
 * @brief Look up the owned Tpetra row index for a cell local ID.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param cell_lid Cell local ID.
 * @return Local row index in the owned data vector.
 * @throws std::out_of_range if @p cell_lid is out of bounds or not locally owned.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::owned_row_for_cell(
    local_ordinal_type cell_lid) const -> local_ordinal_type
{
    check_cell_lid(cell_lid);
    if (!d_mesh->is_owned_cell(cell_lid))
    {
        throw std::out_of_range("Cell local id is not owned by this rank: "
                              + std::to_string(cell_lid));
    }

    return cell_lid;
}

/**
 * @brief Look up the owned Tpetra row index for a cell global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param cell_gid Cell global ID.
 * @return Local row index in the owned data vector.
 * @throws std::out_of_range if @p cell_gid is not a valid mesh GID or not owned.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::owned_row_for_global_cell(
    global_ordinal_type cell_gid) const -> local_ordinal_type
{
    const auto tpetra_gid = d_mesh->mesh_gid_to_tpetra_gid(cell_gid);
    if (tpetra_gid == invalid_id<global_ordinal_type>())
    {
        throw std::out_of_range("Cell global id is not a valid mesh GID: "
                              + std::to_string(cell_gid));
    }
    const auto owned_row = d_data.getMap()->getLocalElement(tpetra_gid);
    if (owned_row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    {
        throw std::out_of_range("Cell global id is not owned by this rank: "
                              + std::to_string(cell_gid));
    }

    return owned_row;
}

/**
 * @brief Look up the overlap (local) Tpetra row index for a cell local ID.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param cell_lid Cell local ID.
 * @return Local row index in the overlap data vector.
 * @throws std::out_of_range if @p cell_lid is out of bounds.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::local_row_for_cell(
    local_ordinal_type cell_lid) const -> local_ordinal_type
{
    check_cell_lid(cell_lid);
    return cell_lid;
}

/**
 * @brief Look up the overlap (local) Tpetra row index for a cell global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 * @param cell_gid Cell global ID.
 * @return Local row index in the overlap data vector.
 * @throws std::out_of_range if @p cell_gid is not a valid mesh GID or not a local cell.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::local_row_for_global_cell(
    global_ordinal_type cell_gid) const -> local_ordinal_type
{
    const auto tpetra_gid = d_mesh->mesh_gid_to_tpetra_gid(cell_gid);
    if (tpetra_gid == invalid_id<global_ordinal_type>())
    {
        throw std::out_of_range("Cell global id is not a valid mesh GID: "
                              + std::to_string(cell_gid));
    }
    const auto local_row = d_overlap_data.getMap()->getLocalElement(tpetra_gid);
    if (local_row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    {
        throw std::out_of_range("Cell global id is not a local cell: "
                              + std::to_string(cell_gid));
    }

    return local_row;
}

} // namespace SimpleFluid
