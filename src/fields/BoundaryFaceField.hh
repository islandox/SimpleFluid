/**
 * @file BoundaryFaceField.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Scalar field restricted to boundary faces.
 * @version 0.1
 * @date 2026-05-30
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
 * @brief Scalar field defined only on boundary faces owned by locally owned
 *        cells.
 *
 * Unlike FaceField, which stores a value for every face whose owner cell is
 * locally owned, BoundaryFaceField restricts storage to faces that lie on a
 * physical boundary (i.e. those for which
 * Mesh::is_boundary_face() returns true).
 *
 * Values are stored in a Tpetra vector whose map is built from the subset of
 * boundary faces whose owner cell is locally owned.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoundaryFaceField
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

    /**
     * @brief Construct a boundary-face field over owned boundary faces.
     *
     * @param mesh Shared pointer to an assembled mesh.
     * @param name Optional field name for I/O.
     * @param zero_out If true, initialize all entries to zero.
     * @throws std::invalid_argument if @p mesh is null.
     * @throws std::runtime_error if the mesh does not have an owned-cell map.
     */
    explicit BoundaryFaceField(SP<const mesh_type> mesh,
                               std::string name = std::string(),
                               bool zero_out = true);

    /**
     * @brief Construct a boundary-face field initialized with a uniform
     *        value.
     *
     * @param mesh Shared pointer to an assembled mesh.
     * @param initial_value Scalar value to fill all owned boundary-face
     *        entries.
     * @param name Optional field name for I/O.
     */
    BoundaryFaceField(SP<const mesh_type> mesh,
                      const scalar_type& initial_value,
                      std::string name = std::string());

    // -------- metadata --------

    /** @brief Return the field name. */
    const std::string& name() const noexcept { return d_name; }

    /** @brief Set the field name. */
    void set_name(std::string name) { d_name = std::move(name); }

    /** @brief Return a const reference to the mesh. */
    const mesh_type& mesh() const noexcept { return *d_mesh; }

    /** @brief Return the shared pointer to the mesh. */
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    /** @brief Return the Tpetra map for owned boundary faces. */
    RCP<const map_type> map() const { return d_data.getMap(); }

    // -------- raw data access --------

    /** @brief Return a mutable reference to the underlying Tpetra vector. */
    vector_type& data() noexcept { return d_data; }

    /** @brief Return a const reference to the underlying Tpetra vector. */
    const vector_type& data() const noexcept { return d_data; }

    // -------- sizing --------

    /**
     * @brief Number of boundary faces owned by this rank.
     * @return Count of boundary faces stored in the field.
     */
    std::size_t num_owned_boundary_faces() const noexcept
    {
        return d_owned_boundary_face_ids.size();
    }

    /**
     * @brief Vector of local face IDs for the owned boundary faces.
     * @return Const reference to the ordered list of boundary-face local
     *         IDs.
     */
    const std::vector<local_ordinal_type>& owned_boundary_face_ids() const noexcept
    {
        return d_owned_boundary_face_ids;
    }

    // -------- bulk operations --------

    /** @brief Set all owned boundary-face entries to @p value. */
    void put_scalar(const scalar_type& value) { d_data.putScalar(value); }

    // -------- per-face access --------

    /**
     * @brief Read the value stored at a boundary face.
     *
     * @param face_lid Local ID of the boundary face.
     * @return Stored scalar value.
     * @throws std::out_of_range if @p face_lid is out of bounds or not an
     *         owned boundary face.
     */
    scalar_type value(local_ordinal_type face_lid) const;

    /**
     * @brief Read the value stored at a boundary face by boundary ID and in-patch ID.
     *
     * @param boundary_id Boundary patch ID.
     * @param in_patch_id Local ID of the face within the boundary patch.
     * @return Stored scalar value.
     * @throws std::out_of_range if the boundary patch is not found or if the in-patch ID is out of bounds for the patch.
     */
    scalar_type value(int boundary_id, local_ordinal_type in_patch_id) const;

    /**
     * @brief Write a value to a boundary face.
     *
     * @param face_lid Local ID of the boundary face.
     * @param value Scalar value to store.
     * @throws std::out_of_range if @p face_lid is out of bounds or not an
     *         owned boundary face.
     */
    void set_value(local_ordinal_type face_lid, const scalar_type& value);

    /**
     * @brief Write a value to a boundary face by boundary ID and in-patch ID.
     *
     * @param boundary_id Boundary patch ID.
     * @param in_patch_id Local ID of the face within the boundary patch.
     * @param value Scalar value to store.
     * @throws std::out_of_range if the boundary patch is not found or if the in-patch ID is out of bounds for the patch.
     */
    void set_value(int boundary_id, local_ordinal_type in_patch_id, const scalar_type& value);

    // -------- queries --------

    /**
     * @brief Check whether a face is an owned boundary face.
     *
     * @param face_lid Local ID of the face to check.
     * @return true if @p face_lid refers to a boundary face owned by this
     *         rank.
     */
    bool is_owned_boundary_face(local_ordinal_type face_lid) const;

private:
    /**
     * @brief Build the Tpetra map for owned boundary faces.
     *
     * Iterates over all mesh faces and includes only those for which
     * Mesh::is_boundary_face() is true and the owner cell is locally
     * owned.
     *
     * @param mesh Shared pointer to the assembled mesh.
     * @param[out] owned_boundary_face_ids Ordered list of boundary-face
     *             local IDs.
     * @param[out] face_lid_to_owned_row Mapping from face local ID to
     *             owned row index (or invalid_owned_row() if not an owned
     *             boundary face).
     * @return RCP to the owned-boundary-face Tpetra map.
     */
    static RCP<const map_type> make_boundary_face_map(
        const SP<const mesh_type>& mesh,
        std::vector<local_ordinal_type>& owned_boundary_face_ids,
        std::vector<local_ordinal_type>& face_lid_to_owned_row);

    /** @brief Sentinel value for faces not stored in this field. */
    static local_ordinal_type invalid_owned_row()
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    /**
     * @brief Look up the owned Tpetra row index for a boundary face.
     *
     * @param face_lid Local ID of the boundary face.
     * @return Local row index in the owned data vector.
     * @throws std::out_of_range if @p face_lid is not an owned boundary
     *         face.
     */
    local_ordinal_type owned_row_for_face(local_ordinal_type face_lid) const;

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
    std::vector<local_ordinal_type> d_owned_boundary_face_ids;
    std::vector<local_ordinal_type> d_face_lid_to_owned_row;
    vector_type d_data;
};

// ========================================================================
//  Out-of-line definitions
// ========================================================================

template<TpetraTypePack Pack>
BoundaryFaceField<Pack>::BoundaryFaceField(SP<const mesh_type> mesh,
                                           std::string name,
                                           bool zero_out)
    : d_name(std::move(name)),
      d_mesh(std::move(mesh)),
      d_owned_boundary_face_ids(),
      d_face_lid_to_owned_row(),
      d_data(make_boundary_face_map(d_mesh,
                                    d_owned_boundary_face_ids,
                                    d_face_lid_to_owned_row),
             zero_out)
{
}

template<TpetraTypePack Pack>
BoundaryFaceField<Pack>::BoundaryFaceField(SP<const mesh_type> mesh,
                                           const scalar_type& initial_value,
                                           std::string name)
    : BoundaryFaceField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

template<TpetraTypePack Pack>
auto BoundaryFaceField<Pack>::make_boundary_face_map(
    const SP<const mesh_type>& mesh,
    std::vector<local_ordinal_type>& owned_boundary_face_ids,
    std::vector<local_ordinal_type>& face_lid_to_owned_row)
    -> RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "BoundaryFaceField requires a non-null mesh.");
    }

    if (mesh->owned_cell_map() == Teuchos::null)
    {
        throw std::runtime_error(
            "BoundaryFaceField requires an assembled mesh with an "
            "owned-cell map.");
    }

    std::size_t num_boundary_faces = 0;
    for (const auto& [patch_id, boundary_patch] : mesh->boundary_patches())
    {
        (void)patch_id;
        num_boundary_faces += boundary_patch.face_lids.size();
    }

    owned_boundary_face_ids.clear();
    owned_boundary_face_ids.reserve(num_boundary_faces);
    face_lid_to_owned_row.assign(mesh->num_faces(), invalid_owned_row());

    for (const auto& [patch_id, boundary_patch] : mesh->boundary_patches())
    {
        (void)patch_id;
        for (auto fid : boundary_patch.face_lids)
        {
            const auto owner = mesh->owner_cell(fid);
            if (!mesh->is_owned_cell(owner))
            {
                continue;
            }

            const auto owned_row =
                detail::checked_size_to_ordinal<local_ordinal_type>(
                    owned_boundary_face_ids.size(), "boundary-face owned row");
            owned_boundary_face_ids.push_back(fid);
            face_lid_to_owned_row[static_cast<std::size_t>(fid)] = owned_row;
        }
    }

    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;
    const auto comm = Tpetra::getDefaultComm();

    return Teuchos::rcp(new map_type(invalid_global_size,
                                     owned_boundary_face_ids.size(),
                                     index_base,
                                     comm));
}

template<TpetraTypePack Pack>
void BoundaryFaceField<Pack>::check_face_lid(
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

template<TpetraTypePack Pack>
auto BoundaryFaceField<Pack>::owned_row_for_face(
    local_ordinal_type face_lid) const -> local_ordinal_type
{
    check_face_lid(face_lid);

    const auto owned_row =
        d_face_lid_to_owned_row[static_cast<std::size_t>(face_lid)];
    if (owned_row == invalid_owned_row())
    {
        throw std::out_of_range(
            "Face local id is not an owned boundary face: "
            + std::to_string(face_lid));
    }

    return owned_row;
}

template<TpetraTypePack Pack>
auto BoundaryFaceField<Pack>::value(local_ordinal_type face_lid) const
    -> scalar_type
{
    return d_data.getData()[owned_row_for_face(face_lid)];
}

template<TpetraTypePack Pack>
auto BoundaryFaceField<Pack>::value(int patch_id, local_ordinal_type in_patch_id) const
    -> scalar_type
{
    const auto& face_patch = d_mesh->boundary_face_patch(patch_id);
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (in_patch_id < 0)
        {
            throw std::out_of_range(
                "In-patch ID cannot be negative: "
                + std::to_string(in_patch_id));
        }
    }

    const auto patch_index = static_cast<std::size_t>(in_patch_id);
    if (patch_index >= face_patch.face_lids.size())
    {
        throw std::out_of_range("In-patch ID is out of bounds for the specified boundary patch.");
    }

    return value(face_patch.face_lids[patch_index]);
}

template<TpetraTypePack Pack>
void BoundaryFaceField<Pack>::set_value(local_ordinal_type face_lid,
                                        const scalar_type& value)
{
    d_data.replaceLocalValue(owned_row_for_face(face_lid), value);
}

template<TpetraTypePack Pack>
void BoundaryFaceField<Pack>::set_value(int patch_id,
                                        local_ordinal_type in_patch_id,
                                        const scalar_type& value)
{
    const auto& face_patch = d_mesh->boundary_face_patch(patch_id);
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (in_patch_id < 0)
        {
            throw std::out_of_range(
                "In-patch ID cannot be negative: "
                + std::to_string(in_patch_id));
        }
    }

    const auto patch_index = static_cast<std::size_t>(in_patch_id);
    if (patch_index >= face_patch.face_lids.size())
    {
        throw std::out_of_range("In-patch ID is out of bounds for the specified boundary patch.");
    }

    set_value(face_patch.face_lids[patch_index], value);
}

template<TpetraTypePack Pack>
bool BoundaryFaceField<Pack>::is_owned_boundary_face(
    local_ordinal_type face_lid) const
{
    check_face_lid(face_lid);
    return d_face_lid_to_owned_row[static_cast<std::size_t>(face_lid)]
           != invalid_owned_row();
}

} // namespace SimpleFluid
