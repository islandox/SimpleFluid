/**
 * @file FaceField.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief face-centered scalar field backed by Tpetra::Vector
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "fields/FaceFieldBase.hh"

#include <cstddef>
#include <string>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Scalar field defined on mesh faces owned by locally owned cells.
 *
 * Stores values in a Tpetra vector. A face is owned by this field when its
 * owner cell is locally owned by the mesh.
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class FaceField : public FaceFieldBase<Pack,
                                        FaceField<Pack>,
                                        typename Pack::vector_type>
{
public:
    using base_type = FaceFieldBase<Pack,
                                     FaceField<Pack>,
                                     typename Pack::vector_type>;
    using mesh_type = typename base_type::mesh_type;
    using vector_type = typename base_type::vector_type;
    using map_type = typename base_type::map_type;
    using scalar_type = typename base_type::scalar_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using global_ordinal_type = typename base_type::global_ordinal_type;

    template <class T>
    using RCP = typename base_type::template RCP<T>;

    explicit FaceField(SP<const mesh_type> mesh,
                       std::string name = std::string(),
                       bool zero_out = true);

    FaceField(SP<const mesh_type> mesh,
              const scalar_type& initial_value,
              std::string name = std::string());

    using base_type::name;
    using base_type::set_name;
    using base_type::mesh;
    using base_type::mesh_ptr;
    using base_type::map;
    using base_type::num_owned_faces;
    using base_type::owned_face_ids;
    using base_type::face_global_id;
    using base_type::is_owned_face;
    using base_type::is_owned_global_face;

    vector_type& data() noexcept { return this->d_data; }
    const vector_type& data() const noexcept { return this->d_data; }

    vector_type& vector() noexcept { return this->d_data; }
    const vector_type& vector() const noexcept { return this->d_data; }

    void put_scalar(const scalar_type& value) { this->d_data.putScalar(value); }

    scalar_type value(local_ordinal_type face_lid) const;
    scalar_type global_value(global_ordinal_type face_gid) const;

    void set_value(local_ordinal_type face_lid, const scalar_type& value);
    void set_global_value(global_ordinal_type face_gid, const scalar_type& value);

    void sum_into_value(local_ordinal_type face_lid, const scalar_type& value);
    void sum_into_global_value(global_ordinal_type face_gid, const scalar_type& value);
};

/**
 * @brief Construct a face field over faces owned by locally owned cells.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param name Optional field name for I/O.
 * @param zero_out If true, initialize all entries to zero.
 * @throws std::invalid_argument if the mesh is null.
 * @throws std::runtime_error if the mesh does not have an owned-cell map.
 */
template<TpetraTypePack Pack>
FaceField<Pack>::FaceField(SP<const mesh_type> mesh,
                           std::string name,
                           bool zero_out)
    : base_type(std::move(name), mesh)
{
    this->d_data = vector_type(
        base_type::make_owned_face_map(
            mesh, "FaceField",
            this->d_owned_face_ids, this->d_face_lid_to_owned_row),
        zero_out);
}

/**
 * @brief Construct a face field initialized with a uniform value.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to an assembled mesh.
 * @param initial_value Scalar value to fill all owned face entries.
 * @param name Optional field name for I/O.
 */
template<TpetraTypePack Pack>
FaceField<Pack>::FaceField(SP<const mesh_type> mesh,
                           const scalar_type& initial_value,
                           std::string name)
    : FaceField(std::move(mesh), std::move(name), false)
{
    put_scalar(initial_value);
}

template<TpetraTypePack Pack>
auto FaceField<Pack>::value(local_ordinal_type face_lid) const -> scalar_type
{
    return this->d_data.getData()[this->owned_row_for_face(face_lid)];
}

template<TpetraTypePack Pack>
auto FaceField<Pack>::global_value(global_ordinal_type face_gid) const -> scalar_type
{
    return this->d_data.getData()[this->owned_row_for_global_face(face_gid)];
}

template<TpetraTypePack Pack>
void FaceField<Pack>::set_value(local_ordinal_type face_lid,
                                const scalar_type& value)
{
    this->d_data.replaceLocalValue(this->owned_row_for_face(face_lid), value);
}

template<TpetraTypePack Pack>
void FaceField<Pack>::set_global_value(global_ordinal_type face_gid,
                                       const scalar_type& value)
{
    this->d_data.replaceLocalValue(
        this->owned_row_for_global_face(face_gid), value);
}

template<TpetraTypePack Pack>
void FaceField<Pack>::sum_into_value(local_ordinal_type face_lid,
                                     const scalar_type& value)
{
    this->d_data.sumIntoLocalValue(this->owned_row_for_face(face_lid), value);
}

template<TpetraTypePack Pack>
void FaceField<Pack>::sum_into_global_value(global_ordinal_type face_gid,
                                            const scalar_type& value)
{
    this->d_data.sumIntoLocalValue(
        this->owned_row_for_global_face(face_gid), value);
}

} // namespace SimpleFluid
