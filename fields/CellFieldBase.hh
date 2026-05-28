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
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Shared storage, map, row-cache, and ghost-sync logic for cell fields.
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
                  const char* class_name)
        : d_name(std::move(name)),
          d_mesh(std::move(mesh)),
          d_data(require_owned_map(d_mesh, class_name), zero_out),
          d_overlap_data(require_overlap_map(d_mesh, class_name), false),
          d_owned_to_overlap_import(
              Teuchos::rcp(new import_type(d_data.getMap(), d_overlap_data.getMap())))
    {
        cache_cell_rows(class_name);
        if (zero_out)
        {
            sync_ghosts();
        }
    }

    CellFieldBase(SP<const mesh_type> mesh,
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
        cache_cell_rows(class_name);
        if (zero_out)
        {
            sync_ghosts();
        }
    }

    static RCP<const map_type> require_owned_map(
        const SP<const mesh_type>& mesh,
        const char* class_name)
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

    static RCP<const map_type> require_overlap_map(
        const SP<const mesh_type>& mesh,
        const char* class_name)
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

    void check_cell_lid(local_ordinal_type cell_lid) const
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

    void cache_cell_rows(const char* class_name)
    {
        d_owned_row_by_cell_lid.assign(
            d_mesh->num_local_cells(),
            Teuchos::OrdinalTraits<local_ordinal_type>::invalid());
        d_local_row_by_cell_lid.assign(
            d_mesh->num_local_cells(),
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
                throw std::runtime_error(std::string(class_name)
                                       + " overlap map is missing a local cell.");
            }

            if (d_mesh->is_owned_cell(cell_lid))
            {
                d_owned_row_by_cell_lid[cell] =
                    d_data.getMap()->getLocalElement(cell_gid);
                if (d_owned_row_by_cell_lid[cell]
                    == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
                {
                    throw std::runtime_error(std::string(class_name)
                                           + " owned map is missing an owned cell.");
                }
            }
        }
    }

    local_ordinal_type owned_row_for_cell(local_ordinal_type cell_lid) const
    {
        check_cell_lid(cell_lid);
        if (!d_mesh->is_owned_cell(cell_lid))
        {
            throw std::out_of_range("Cell local id is not owned by this rank: "
                                  + std::to_string(cell_lid));
        }

        const auto row = d_owned_row_by_cell_lid[static_cast<std::size_t>(cell_lid)];
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)
        if (row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
        {
            throw std::runtime_error("Cached owned row is invalid for cell: "
                                   + std::to_string(cell_lid));
        }
#endif
        return row;
    }

    local_ordinal_type owned_row_for_global_cell(global_ordinal_type cell_gid) const
    {
        const auto owned_row = d_data.getMap()->getLocalElement(cell_gid);
        if (owned_row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
        {
            throw std::out_of_range("Cell global id is not owned by this rank: "
                                  + std::to_string(cell_gid));
        }

        return owned_row;
    }

    local_ordinal_type local_row_for_cell(local_ordinal_type cell_lid) const
    {
        check_cell_lid(cell_lid);
        const auto row = d_local_row_by_cell_lid[static_cast<std::size_t>(cell_lid)];
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)
        if (row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
        {
            throw std::out_of_range("Cell local id is not local to this rank: "
                                  + std::to_string(cell_lid));
        }
#endif
        return row;
    }

    local_ordinal_type local_row_for_global_cell(global_ordinal_type cell_gid) const
    {
        const auto local_row = d_overlap_data.getMap()->getLocalElement(cell_gid);
        if (local_row == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
        {
            throw std::out_of_range("Cell global id is not local to this rank: "
                                  + std::to_string(cell_gid));
        }

        return local_row;
    }

    std::string d_name;
    SP<const mesh_type> d_mesh;
    vector_type d_data;
    vector_type d_overlap_data;
    RCP<const import_type> d_owned_to_overlap_import;
    std::vector<local_ordinal_type> d_owned_row_by_cell_lid;
    std::vector<local_ordinal_type> d_local_row_by_cell_lid;
};

} // namespace SimpleFluid
