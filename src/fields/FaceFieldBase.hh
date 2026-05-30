/**
 * @file FaceFieldBase.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared CRTP infrastructure for face-centered field classes.
 * @version 0.1
 * @date 2026-05-31
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "geometry/Mesh.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Shared storage, owned-face map construction, row-cache, and
 *        ownership queries for face-centered fields.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam Derived CRTP derived class.
 * @tparam StorageVector Tpetra Vector or MultiVector storage type.
 */
template<TpetraTypePack Pack, class Derived, class StorageVector>
class FaceFieldBase
{
public:
    using mesh_type = Mesh<Pack>;
    using vector_type = StorageVector;
    using map_type = typename Pack::map_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    // -------- metadata --------

    /** @brief Return the field name. */
    const std::string& name() const noexcept { return d_name; }

    /** @brief Set the field name. */
    void set_name(std::string name) { d_name = std::move(name); }

    /** @brief Return a const reference to the mesh. */
    const mesh_type& mesh() const noexcept { return *d_mesh; }

    /** @brief Return the shared pointer to the mesh. */
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    /** @brief Return the Tpetra map for owned faces. */
    RCP<const map_type> map() const { return d_data.getMap(); }

    // -------- sizing --------

    /** @brief Number of faces owned by this rank. */
    std::size_t num_owned_faces() const noexcept
    {
        return d_owned_face_ids.size();
    }

    /**
     * @brief Vector of local face IDs for the owned faces.
     * @return Const reference to the ordered list of owned-face local IDs.
     */
    const std::vector<local_ordinal_type>& owned_face_ids() const noexcept
    {
        return d_owned_face_ids;
    }

    // -------- global-ID mapping --------

    /**
     * @brief Look up the global ID for an owned face.
     *
     * @param face_lid Local ID of the owned face.
     * @return Corresponding global ID.
     * @throws std::out_of_range if @p face_lid is not an owned face.
     */
    global_ordinal_type face_global_id(local_ordinal_type face_lid) const
    {
        return d_data.getMap()->getGlobalElement(
            owned_row_for_face(face_lid));
    }

    // -------- ownership queries --------

    /**
     * @brief Check whether a face local ID is owned by this rank.
     *
     * @param face_lid Local ID of the face to check.
     * @return true if the face is owned.
     */
    bool is_owned_face(local_ordinal_type face_lid) const
    {
        check_face_lid(face_lid);
        return d_face_lid_to_owned_row[static_cast<std::size_t>(face_lid)]
               != invalid_owned_row();
    }

    /**
     * @brief Check whether a face global ID is owned by this rank.
     *
     * @param face_gid Global ID of the face to check.
     * @return true if the face is owned.
     */
    bool is_owned_global_face(global_ordinal_type face_gid) const
    {
        return d_data.getMap()->getLocalElement(face_gid)
               != invalid_owned_row();
    }

protected:
    /**
     * @brief Construct the base with a name and mesh pointer.
     *
     * The derived class is responsible for constructing @ref d_data after
     * calling make_owned_face_map().
     *
     * @param name Field name.
     * @param mesh Shared pointer to the assembled mesh.
     */
    FaceFieldBase(std::string name, SP<const mesh_type> mesh)
        : d_name(std::move(name)),
          d_mesh(std::move(mesh))
    {
    }

    /**
     * @brief Build the Tpetra map for faces whose owner cell is locally
     *        owned.
     *
     * @param mesh Shared pointer to the assembled mesh.
     * @param class_name Name of the derived class (used in error messages).
     * @param[out] owned_face_ids Ordered list of owned-face local IDs.
     * @param[out] face_lid_to_owned_row Mapping from face local ID to
     *             owned row index.
     * @return RCP to the owned-face Tpetra map.
     * @throws std::invalid_argument if @p mesh is null.
     * @throws std::runtime_error if the mesh does not have an owned-cell map.
     */
    static RCP<const map_type> make_owned_face_map(
        const SP<const mesh_type>& mesh,
        const char* class_name,
        std::vector<local_ordinal_type>& owned_face_ids,
        std::vector<local_ordinal_type>& face_lid_to_owned_row);

    /** @brief Sentinel value for faces not owned by this rank. */
    static local_ordinal_type invalid_owned_row()
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    /**
     * @brief Look up the owned Tpetra row index for a given face local ID.
     *
     * @param face_lid Face local ID.
     * @return Local row index in the owned data vector.
     * @throws std::out_of_range if the face is not owned by this rank.
     */
    local_ordinal_type owned_row_for_face(
        local_ordinal_type face_lid) const;

    /**
     * @brief Look up the owned Tpetra row index for a given face global ID.
     *
     * @param face_gid Face global ID.
     * @return Local row index in the owned data vector.
     * @throws std::out_of_range if the face is not owned by this rank.
     */
    local_ordinal_type owned_row_for_global_face(
        global_ordinal_type face_gid) const;

    /**
     * @brief Validate that a face local ID is in range.
     *
     * @param face_lid Face local ID to validate.
     * @throws std::out_of_range if the ID is negative or exceeds the face
     *         count.
     */
    void check_face_lid(local_ordinal_type face_lid) const;

    std::string d_name;
    SP<const mesh_type> d_mesh;
    std::vector<local_ordinal_type> d_owned_face_ids;
    std::vector<local_ordinal_type> d_face_lid_to_owned_row;
    vector_type d_data;
};

// ========================================================================
//  Out-of-line definitions
// ========================================================================

template<TpetraTypePack Pack, class Derived, class StorageVector>
auto FaceFieldBase<Pack, Derived, StorageVector>::make_owned_face_map(
    const SP<const mesh_type>& mesh,
    const char* class_name,
    std::vector<local_ordinal_type>& owned_face_ids,
    std::vector<local_ordinal_type>& face_lid_to_owned_row)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(
            std::string(class_name) + " requires a non-null mesh.");
    }

    if (mesh->owned_cell_map() == Teuchos::null)
    {
        throw std::runtime_error(
            std::string(class_name)
            + " requires an assembled mesh with an owned-cell map.");
    }

    owned_face_ids.clear();
    owned_face_ids.reserve(mesh->num_faces());
    face_lid_to_owned_row.assign(mesh->num_faces(), invalid_owned_row());

    for (std::size_t fid = 0; fid < mesh->num_faces(); ++fid)
    {
        const auto face_lid =
            detail::checked_size_to_ordinal<local_ordinal_type>(
                fid, "face local id");
        const auto owner = mesh->owner_cell(face_lid);

        if (mesh->is_owned_cell(owner))
        {
            const auto owned_row =
                detail::checked_size_to_ordinal<local_ordinal_type>(
                    owned_face_ids.size(), "owned face row");
            owned_face_ids.push_back(face_lid);
            face_lid_to_owned_row[fid] = owned_row;
        }
    }

    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;
    const auto comm = Tpetra::getDefaultComm();

    return Teuchos::rcp(new map_type(invalid_global_size,
                                     owned_face_ids.size(),
                                     index_base,
                                     comm));
}

template<TpetraTypePack Pack, class Derived, class StorageVector>
void FaceFieldBase<Pack, Derived, StorageVector>::check_face_lid(
    local_ordinal_type face_lid) const
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (face_lid < 0)
        {
            throw std::out_of_range(
                "Face local id cannot be negative: "
                + std::to_string(face_lid));
        }
    }

    if (static_cast<std::size_t>(face_lid)
        >= d_face_lid_to_owned_row.size())
    {
        throw std::out_of_range(
            "Face local id is out of bounds: "
            + std::to_string(face_lid));
    }
}

template<TpetraTypePack Pack, class Derived, class StorageVector>
auto FaceFieldBase<Pack, Derived, StorageVector>::owned_row_for_face(
    local_ordinal_type face_lid) const -> local_ordinal_type
{
    check_face_lid(face_lid);

    const auto owned_row =
        d_face_lid_to_owned_row[static_cast<std::size_t>(face_lid)];
    if (owned_row == invalid_owned_row())
    {
        throw std::out_of_range(
            "Face local id is not owned by this rank: "
            + std::to_string(face_lid));
    }

    return owned_row;
}

template<TpetraTypePack Pack, class Derived, class StorageVector>
auto FaceFieldBase<Pack, Derived, StorageVector>::owned_row_for_global_face(
    global_ordinal_type face_gid) const -> local_ordinal_type
{
    const auto owned_row = d_data.getMap()->getLocalElement(face_gid);
    if (owned_row == invalid_owned_row())
    {
        throw std::out_of_range(
            "Face global id is not owned by this rank: "
            + std::to_string(face_gid));
    }

    return owned_row;
}

} // namespace SimpleFluid
