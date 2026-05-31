/**
 * @file CellFieldBase.hh
 * @brief Shared CRTP infrastructure for cell-centered field classes.
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

    std::size_t num_owned_cells() const
    {
        return d_data.getMap()->getLocalNumElements();
    }

    std::size_t num_local_cells() const
    {
        return d_overlap_data.getMap()->getLocalNumElements();
    }

    void sync_ghosts()
    {
        d_overlap_data.doImport(d_data, *d_owned_to_overlap_import, Tpetra::INSERT);
    }

protected:
    CellFieldBase(SP<const mesh_type> mesh,
                  std::string name,
                  bool zero_out,
                  const char* class_name);

    CellFieldBase(SP<const mesh_type> mesh,
                  std::string name,
                  std::size_t num_components,
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

template<TpetraTypePack Pack, class Derived, class StorageVector>
CellFieldBase<Pack, Derived, StorageVector>::CellFieldBase(
    SP<const mesh_type> mesh,
    std::string name,
    std::size_t num_components,
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

template<TpetraTypePack Pack, class Derived, class StorageVector>
void CellFieldBase<Pack, Derived, StorageVector>::check_cell_lid(
    local_ordinal_type cell_lid) const
{
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)
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
#else
    (void)cell_lid;
#endif
}

template<TpetraTypePack Pack, class Derived, class StorageVector>
void CellFieldBase<Pack, Derived, StorageVector>::check_cell_row_invariant(
    const char* class_name) const
{
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)
    if (d_data.getMap()->getLocalNumElements() != d_mesh->num_owned_cells())
    {
        throw std::runtime_error(std::string(class_name)
                               + " owned map size does not match the owned-cell count.");
    }
    if (d_overlap_data.getMap()->getLocalNumElements() != d_mesh->num_local_cells())
    {
        throw std::runtime_error(std::string(class_name)
                               + " overlap map size does not match the local-cell count.");
    }

    const auto invalid_row =
        Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    const auto num_owned = d_mesh->num_owned_cells();
    for (std::size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);

        const auto tpetra_gid =
            cell < num_owned
                ? d_mesh->owned_cell_tpetra_gids()[cell]
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
#else
    (void)class_name;
#endif
}

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

template<TpetraTypePack Pack, class Derived, class StorageVector>
auto CellFieldBase<Pack, Derived, StorageVector>::local_row_for_cell(
    local_ordinal_type cell_lid) const -> local_ordinal_type
{
    check_cell_lid(cell_lid);
    return cell_lid;
}

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
