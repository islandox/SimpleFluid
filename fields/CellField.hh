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

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
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
{
public:
    using mesh_type = Mesh<Pack>;
    using vector_type = typename Pack::vector_type;
    using map_type = typename Pack::map_type;
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

    vector_type& data() noexcept { return d_data; }
    const vector_type& data() const noexcept { return d_data; }

    vector_type& vector() noexcept { return d_data; }
    const vector_type& vector() const noexcept { return d_data; }

    std::size_t num_owned_cells() const
    {
        return d_data.getMap()->getLocalNumElements();
    }

    void put_scalar(const scalar_type& value) { d_data.putScalar(value); }

    scalar_type value(local_ordinal_type cell_lid) const;
    scalar_type global_value(global_ordinal_type cell_gid) const;

    void set_value(local_ordinal_type cell_lid, const scalar_type& value);
    void set_global_value(global_ordinal_type cell_gid, const scalar_type& value);

    void sum_into_value(local_ordinal_type cell_lid, const scalar_type& value);
    void sum_into_global_value(global_ordinal_type cell_gid, const scalar_type& value);

    bool is_owned_cell(local_ordinal_type cell_lid) const;
    bool is_owned_global_cell(global_ordinal_type cell_gid) const;

private:
    static RCP<const map_type> require_owned_map(const SP<const mesh_type>& mesh);

    local_ordinal_type owned_row_for_cell(local_ordinal_type cell_lid) const;
    local_ordinal_type owned_row_for_global_cell(global_ordinal_type cell_gid) const;
    void check_cell_lid(local_ordinal_type cell_lid) const;

    std::string d_name;
    SP<const mesh_type> d_mesh;
    vector_type d_data;
};

template<TpetraTypePack Pack>
CellField<Pack>::CellField(SP<const mesh_type> mesh,
                           std::string name,
                           bool zero_out)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_data(require_owned_map(d_mesh), zero_out)
{
}

template<TpetraTypePack Pack>
CellField<Pack>::CellField(SP<const mesh_type> mesh,
                           const scalar_type& initial_value,
                           std::string name)
    : CellField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

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

template<TpetraTypePack Pack>
void CellField<Pack>::check_cell_lid(local_ordinal_type cell_lid) const
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        CHECK(cell_lid >= 0,
              "Cell local id cannot be negative: " + std::to_string(cell_lid),
              std::out_of_range);
    }

    CHECK(static_cast<std::size_t>(cell_lid) < d_mesh->num_local_cells(),
          "Cell local id is out of bounds: " + std::to_string(cell_lid),
          std::out_of_range);
}

template<TpetraTypePack Pack>
auto CellField<Pack>::owned_row_for_global_cell(global_ordinal_type cell_gid) const
    -> local_ordinal_type
{
    const auto owned_row = d_data.getMap()->getLocalElement(cell_gid);
    CHECK(owned_row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid(),
          "Cell global id is not owned by this rank: " + std::to_string(cell_gid),
          std::out_of_range);

    return owned_row;
}

template<TpetraTypePack Pack>
auto CellField<Pack>::owned_row_for_cell(local_ordinal_type cell_lid) const
    -> local_ordinal_type
{
    check_cell_lid(cell_lid);
    return cell_lid; // Assuming the local cell IDs are 0-based and contiguous for owned cells
}

template<TpetraTypePack Pack>
auto CellField<Pack>::value(local_ordinal_type cell_lid) const -> scalar_type
{
    return d_data.getData()[owned_row_for_cell(cell_lid)];
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
}

template<TpetraTypePack Pack>
void CellField<Pack>::set_global_value(global_ordinal_type cell_gid,
                                       const scalar_type& value)
{
    d_data.replaceLocalValue(owned_row_for_global_cell(cell_gid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_value(local_ordinal_type cell_lid,
                                     const scalar_type& value)
{
    d_data.sumIntoLocalValue(owned_row_for_cell(cell_lid), value);
}

template<TpetraTypePack Pack>
void CellField<Pack>::sum_into_global_value(global_ordinal_type cell_gid,
                                            const scalar_type& value)
{
    d_data.sumIntoLocalValue(owned_row_for_global_cell(cell_gid), value);
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    check_cell_lid(cell_lid);
    return d_mesh->is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
bool CellField<Pack>::is_owned_global_cell(global_ordinal_type cell_gid) const
{
    const auto row = d_data.getMap()->getLocalElement(cell_gid);
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

} // namespace SimpleFluid
