/**
 * @file VectorCellField.hh
 * @brief Three-component cell-centered vector field backed by Tpetra::MultiVector.
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
 * @brief Cell-centered 3D vector field stored as a Tpetra::MultiVector.
 *
 * The three MultiVector columns are ordered as x, y, z.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class VectorCellField
{
public:
    using mesh_type = Mesh<Pack>;
    using vector_type = typename Pack::multi_vector_type;
    using map_type = typename Pack::map_type;
    using import_type = typename Pack::import_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;

    static constexpr std::size_t num_components = 3;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    explicit VectorCellField(SP<const mesh_type> mesh,
                             std::string name = std::string(),
                             bool zero_out = true);

    VectorCellField(SP<const mesh_type> mesh,
                    const vec_type& initial_value,
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

    void put_scalar(const vec_type& value);
    void sync_ghosts();

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
    static RCP<const map_type> require_owned_map(const SP<const mesh_type>& mesh);
    static RCP<const map_type> require_overlap_map(const SP<const mesh_type>& mesh);

    static scalar_type component(const vec_type& value, std::size_t index);
    static void check_component(std::size_t component);

    local_ordinal_type owned_row_for_cell(local_ordinal_type cell_lid) const;
    local_ordinal_type local_row_for_cell(local_ordinal_type cell_lid) const;
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

template<TpetraTypePack Pack>
VectorCellField<Pack>::VectorCellField(SP<const mesh_type> mesh,
                                       std::string name,
                                       bool zero_out)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_data(require_owned_map(d_mesh), num_components, zero_out),
      d_overlap_data(require_overlap_map(d_mesh), num_components, false),
      d_owned_to_overlap_import(
          Teuchos::rcp(new import_type(d_data.getMap(), d_overlap_data.getMap())))
{
    cache_cell_rows();
    if (zero_out)
    {
        sync_ghosts();
    }
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
auto VectorCellField<Pack>::require_owned_map(const SP<const mesh_type>& mesh)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("VectorCellField requires a non-null mesh.");
    }

    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(
            "VectorCellField requires an assembled mesh with an owned-cell map.");
    }

    return map;
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::require_overlap_map(const SP<const mesh_type>& mesh)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("VectorCellField requires a non-null mesh.");
    }

    auto map = mesh->overlap_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(
            "VectorCellField requires an assembled mesh with an overlap-cell map.");
    }

    return map;
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::component(const vec_type& value,
                                      std::size_t index) -> scalar_type
{
    return index == 0 ? value.x : (index == 1 ? value.y : value.z);
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
void VectorCellField<Pack>::check_cell_lid(local_ordinal_type cell_lid) const
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

template<TpetraTypePack Pack>
void VectorCellField<Pack>::cache_cell_rows()
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
            throw std::runtime_error("VectorCellField overlap map is missing a local cell.");
        }

        if (d_mesh->is_owned_cell(cell_lid))
        {
            d_owned_row_by_cell_lid[cell] =
                d_data.getMap()->getLocalElement(cell_gid);
            if (d_owned_row_by_cell_lid[cell]
                == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
            {
                throw std::runtime_error("VectorCellField owned map is missing an owned cell.");
            }
        }
    }
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::owned_row_for_cell(local_ordinal_type cell_lid) const
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

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::local_row_for_cell(local_ordinal_type cell_lid) const
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

template<TpetraTypePack Pack>
void VectorCellField<Pack>::put_scalar(const vec_type& value)
{
    for (std::size_t component_id = 0; component_id < num_components; ++component_id)
    {
        d_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
        d_overlap_data.getVectorNonConst(component_id)->putScalar(
            component(value, component_id));
    }
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::sync_ghosts()
{
    d_overlap_data.doImport(d_data, *d_owned_to_overlap_import, Tpetra::INSERT);
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
    return d_data.getData(component_id)[owned_row_for_cell(cell_lid)];
}

template<TpetraTypePack Pack>
auto VectorCellField<Pack>::local_component_value(
    local_ordinal_type cell_lid,
    std::size_t component_id) const -> scalar_type
{
    check_component(component_id);
    return d_overlap_data.getData(component_id)[local_row_for_cell(cell_lid)];
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
    d_data.replaceLocalValue(owned_row_for_cell(cell_lid), component_id, value);
    d_overlap_data.replaceLocalValue(local_row_for_cell(cell_lid), component_id, value);
}

template<TpetraTypePack Pack>
void VectorCellField<Pack>::set_owned_component_value(
    local_ordinal_type cell_lid,
    std::size_t component_id,
    const scalar_type& value)
{
    check_component(component_id);
    d_data.replaceLocalValue(owned_row_for_cell(cell_lid), component_id, value);
}

template<TpetraTypePack Pack>
bool VectorCellField<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    check_cell_lid(cell_lid);
    return d_mesh->is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
bool VectorCellField<Pack>::is_local_cell(local_ordinal_type cell_lid) const
{
    check_cell_lid(cell_lid);
    const auto row = d_local_row_by_cell_lid[static_cast<std::size_t>(cell_lid)];
    return row != Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
}

} // namespace SimpleFluid
