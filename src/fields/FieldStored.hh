/**
 * @file FieldStored.hh
 * @brief Mesh-aware distributed storage for typed field descriptors.
 */

#pragma once

#include "fields/Field.hh"
#include "geometry/MeshHandle.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_CombineMode.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace SimpleFluid
{
namespace detail
{

template<class Value>
struct StoredFieldValueTraits
{
    static constexpr bool is_vector = false;
    static constexpr size_t components = 1;
    using scalar_type = Value;
};

template<class Scalar>
struct StoredFieldValueTraits<vec3<Scalar>>
{
    static constexpr bool is_vector = true;
    static constexpr size_t components = 3;
    using scalar_type = Scalar;
};

template<class Location>
inline constexpr bool supported_field_location =
    std::is_same_v<Location, CellLocation>
 || std::is_same_v<Location, FaceLocation>
 || std::is_same_v<Location, BoundaryFaceLocation>;

} // namespace detail

/**
 * @brief Distributed numerical storage for a typed field descriptor.
 *
 * Owned storage is authoritative. Overlap storage includes locally available
 * ghost entries and is refreshed explicitly with sync_ghosts().
 *
 * @tparam FieldType Field descriptor defining value and location types.
 * @tparam Pack Tpetra scalar, ordinal, map, and vector types.
 */
template<class FieldType, TpetraTypePack Pack = DefaultTpetraTypes>
class FieldStored
{
public:
    using descriptor_type = FieldType;
    using value_type = typename descriptor_type::value_type;
    using location_type = typename descriptor_type::location_type;
    using value_traits = detail::StoredFieldValueTraits<value_type>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;
    using import_type = typename Pack::import_type;
    using storage_type = std::conditional_t<
        value_traits::is_vector,
        typename Pack::multi_vector_type,
        typename Pack::vector_type>;

    static_assert(detail::supported_field_location<location_type>,
                  "FieldStored does not support this field location.");
    static_assert(std::is_same_v<typename value_traits::scalar_type,
                                 scalar_type>,
                  "Field value scalar must match the Tpetra pack scalar.");

    /**
     * @brief Allocate owned and overlap storage for a descriptor.
     * @param descriptor Field identity and value/location types.
     * @param mesh Mesh whose maps define the distributed layout.
     * @param zero_out Whether to initialize owned and ghost entries to zero.
     * @throws std::invalid_argument If @p mesh is null.
     */
    explicit FieldStored(
        descriptor_type descriptor,
        SP<const MeshHandle<Pack>> mesh,
        bool zero_out = true)
        : d_descriptor(std::move(descriptor)),
          d_mesh(require_mesh(std::move(mesh))),
          d_owned(make_storage(owned_map_for(*d_mesh), zero_out)),
          d_overlap(make_storage(overlap_map_for(*d_mesh), false)),
          d_import(Teuchos::rcp(new import_type(
              d_owned.getMap(), d_overlap.getMap())))
    {
        if (zero_out)
        {
            sync_ghosts();
        }
    }

    /** @brief Allocate storage and initialize every local entry. */
    FieldStored(descriptor_type descriptor,
                SP<const MeshHandle<Pack>> mesh,
                const value_type& initial_value)
        : FieldStored(
              std::move(descriptor), std::move(mesh), false)
    {
        put_value(initial_value);
    }

    const descriptor_type& descriptor() const noexcept
    {
        return d_descriptor;
    }

    size_t id() const noexcept { return d_descriptor.id(); }
    const std::string& name() const noexcept
    {
        return d_descriptor.name();
    }

    const MeshHandle<Pack>& mesh() const noexcept { return *d_mesh; }
    SP<const MeshHandle<Pack>> mesh_ptr() const noexcept { return d_mesh; }

    storage_type& data() noexcept { return d_owned; }
    const storage_type& data() const noexcept { return d_owned; }
    storage_type& owned_data() noexcept { return d_owned; }
    const storage_type& owned_data() const noexcept { return d_owned; }
    storage_type& overlap_data() noexcept { return d_overlap; }
    const storage_type& overlap_data() const noexcept { return d_overlap; }

    Teuchos::RCP<const map_type> map() const
    {
        return d_owned.getMap();
    }

    Teuchos::RCP<const map_type> overlap_map() const
    {
        return d_overlap.getMap();
    }

    size_t num_owned_entries() const noexcept
    {
        return d_owned.getMap()->getLocalNumElements();
    }

    size_t num_local_entries() const noexcept
    {
        return d_overlap.getMap()->getLocalNumElements();
    }

    /** @brief Import current owned values into overlap ghost storage. */
    void sync_ghosts()
    {
        d_overlap.doImport(d_owned, *d_import, Tpetra::REPLACE);
    }

    /** @brief Assign one value to every owned and overlap entry. */
    void put_value(const value_type& value)
    {
        if constexpr (value_traits::is_vector)
        {
            for (size_t component = 0;
                 component < value_traits::components;
                 ++component)
            {
                d_owned.getVectorNonConst(component)->putScalar(
                    value.component(component));
                d_overlap.getVectorNonConst(component)->putScalar(
                    value.component(component));
            }
        }
        else
        {
            d_owned.putScalar(value);
            d_overlap.putScalar(value);
        }
    }

    value_type value(local_ordinal_type local_id) const
    {
        return value_at(d_owned, owned_row(local_id));
    }

    value_type owned_value(local_ordinal_type local_id) const
    {
        return value(local_id);
    }

    value_type local_value(local_ordinal_type local_id) const
    {
        return value_at(d_overlap, overlap_row(local_id));
    }

    scalar_type component_value(local_ordinal_type local_id,
                                size_t component) const
    {
        check_component(component);
        if constexpr (value_traits::is_vector)
        {
            return d_owned.getData(component)[owned_row(local_id)];
        }
        else
        {
            if (component != 0)
            {
                throw std::out_of_range(
                    "Scalar field has only component zero.");
            }
            return value(local_id);
        }
    }

    /**
     * @brief Update an owned entry and its local overlap copy.
     * @throws std::out_of_range If the entry is not locally owned.
     */
    void set_value(local_ordinal_type local_id,
                   const value_type& value)
    {
        set_at(d_owned, owned_row(local_id), value);
        set_at(d_overlap, overlap_row(local_id), value);
    }

    /**
     * @brief Update only authoritative owned storage.
     *
     * Call sync_ghosts() before reading dependent overlap values.
     */
    void set_owned_value(local_ordinal_type local_id,
                         const value_type& value)
    {
        set_at(d_owned, owned_row(local_id), value);
    }

    void set_component_value(local_ordinal_type local_id,
                             size_t component,
                             scalar_type value)
    {
        check_component(component);
        if constexpr (value_traits::is_vector)
        {
            d_owned.replaceLocalValue(
                owned_row(local_id), component, value);
            d_overlap.replaceLocalValue(
                overlap_row(local_id), component, value);
        }
        else
        {
            if (component != 0)
            {
                throw std::out_of_range(
                    "Scalar field has only component zero.");
            }
            set_value(local_id, value);
        }
    }

    bool is_owned(local_ordinal_type local_id) const
    {
        return owned_row_or_invalid(local_id)
            != invalid_local_ordinal();
    }

    bool is_local(local_ordinal_type local_id) const
    {
        return overlap_row_or_invalid(local_id)
            != invalid_local_ordinal();
    }

private:
    static SP<const MeshHandle<Pack>> require_mesh(
        SP<const MeshHandle<Pack>> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "FieldStored requires a non-null mesh.");
        }
        return mesh;
    }

    static Teuchos::RCP<const map_type> owned_map_for(
        const MeshHandle<Pack>& mesh)
    {
        if constexpr (std::is_same_v<location_type, CellLocation>)
        {
            return mesh.owned_cell_map();
        }
        else if constexpr (std::is_same_v<location_type, FaceLocation>)
        {
            return mesh.owned_face_map();
        }
        else
        {
            return mesh.boundary_face_map();
        }
    }

    static Teuchos::RCP<const map_type> overlap_map_for(
        const MeshHandle<Pack>& mesh)
    {
        if constexpr (std::is_same_v<location_type, CellLocation>)
        {
            return mesh.overlap_cell_map();
        }
        else if constexpr (std::is_same_v<location_type, FaceLocation>)
        {
            return mesh.overlap_face_map();
        }
        else
        {
            return mesh.boundary_face_map();
        }
    }

    static storage_type make_storage(
        const Teuchos::RCP<const map_type>& map,
        bool zero_out)
    {
        if constexpr (value_traits::is_vector)
        {
            return storage_type(
                map, value_traits::components, zero_out);
        }
        else
        {
            return storage_type(map, zero_out);
        }
    }

    static constexpr local_ordinal_type invalid_local_ordinal()
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    global_ordinal_type global_id(local_ordinal_type local_id) const
    {
        if constexpr (std::is_same_v<location_type, CellLocation>)
        {
            return d_mesh->cell_global_id(local_id);
        }
        else
        {
            return d_mesh->face_global_id(local_id);
        }
    }

    local_ordinal_type owned_row_or_invalid(
        local_ordinal_type local_id) const
    {
        return d_owned.getMap()->getLocalElement(global_id(local_id));
    }

    local_ordinal_type overlap_row_or_invalid(
        local_ordinal_type local_id) const
    {
        return d_overlap.getMap()->getLocalElement(global_id(local_id));
    }

    local_ordinal_type owned_row(local_ordinal_type local_id) const
    {
        const auto row = owned_row_or_invalid(local_id);
        if (row == invalid_local_ordinal())
        {
            throw std::out_of_range(
                "Field entry is not locally owned.");
        }
        return row;
    }

    local_ordinal_type overlap_row(local_ordinal_type local_id) const
    {
        const auto row = overlap_row_or_invalid(local_id);
        if (row == invalid_local_ordinal())
        {
            throw std::out_of_range(
                "Field entry is not locally available.");
        }
        return row;
    }

    static void check_component(size_t component)
    {
        if (component >= value_traits::components)
        {
            throw std::out_of_range(
                "Field component index is out of range.");
        }
    }

    static value_type value_at(
        const storage_type& storage,
        local_ordinal_type row)
    {
        if constexpr (value_traits::is_vector)
        {
            return {
                storage.getData(0)[row],
                storage.getData(1)[row],
                storage.getData(2)[row]};
        }
        else
        {
            return storage.getData()[row];
        }
    }

    static void set_at(storage_type& storage,
                       local_ordinal_type row,
                       const value_type& value)
    {
        if constexpr (value_traits::is_vector)
        {
            for (size_t component = 0;
                 component < value_traits::components;
                 ++component)
            {
                storage.replaceLocalValue(
                    row, component, value.component(component));
            }
        }
        else
        {
            storage.replaceLocalValue(row, value);
        }
    }

    descriptor_type d_descriptor;
    SP<const MeshHandle<Pack>> d_mesh;
    storage_type d_owned;
    storage_type d_overlap;
    Teuchos::RCP<const import_type> d_import;
};

template<TpetraTypePack Pack = DefaultTpetraTypes>
using ScalarCellFieldStored =
    FieldStored<ScalarCellFieldDescriptor<Pack>, Pack>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorCellFieldStored =
    FieldStored<VectorCellFieldDescriptor<Pack>, Pack>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using ScalarFaceFieldStored =
    FieldStored<ScalarFaceFieldDescriptor<Pack>, Pack>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorFaceFieldStored =
    FieldStored<VectorFaceFieldDescriptor<Pack>, Pack>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using ScalarBoundaryFaceFieldStored =
    FieldStored<ScalarBoundaryFaceFieldDescriptor<Pack>, Pack>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorBoundaryFaceFieldStored =
    FieldStored<VectorBoundaryFaceFieldDescriptor<Pack>, Pack>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using AnyFieldStored = std::variant<
    SP<ScalarCellFieldStored<Pack>>,
    SP<VectorCellFieldStored<Pack>>,
    SP<ScalarFaceFieldStored<Pack>>,
    SP<VectorFaceFieldStored<Pack>>,
    SP<ScalarBoundaryFaceFieldStored<Pack>>,
    SP<VectorBoundaryFaceFieldStored<Pack>>>;

} // namespace SimpleFluid
