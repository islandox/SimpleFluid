/**
 * @file FvmOperators.hh
 * @brief Finite-volume helper operators for cell and face fields.
 */
#pragma once

#include "fields/CellField.hh"
#include "fields/FaceField.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FvmOperators
{

namespace detail
{

inline MeshUtils::Vec3 solve_3x3(std::array<std::array<real_t, 3>, 3> a,
                                 MeshUtils::Vec3 b)
{
    std::array<real_t, 3> rhs{b.x, b.y, b.z};

    for (std::size_t pivot = 0; pivot < 3; ++pivot)
    {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < 3; ++row)
        {
            if (std::abs(a[row][pivot]) > std::abs(a[best][pivot]))
            {
                best = row;
            }
        }

        if (std::abs(a[best][pivot]) < 1.0e-14)
        {
            return {};
        }

        if (best != pivot)
        {
            std::swap(a[best], a[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }

        const auto inv = 1.0 / a[pivot][pivot];
        for (std::size_t col = pivot; col < 3; ++col)
        {
            a[pivot][col] *= inv;
        }
        rhs[pivot] *= inv;

        for (std::size_t row = 0; row < 3; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const auto factor = a[row][pivot];
            for (std::size_t col = pivot; col < 3; ++col)
            {
                a[row][col] -= factor * a[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    return {rhs[0], rhs[1], rhs[2]};
}

} // namespace detail

template<TpetraTypePack Pack>
std::vector<typename Mesh<Pack>::Vec3>
cell_gradient(const CellField<Pack>& field)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;

    const auto& mesh = field.mesh();
    std::vector<typename mesh_type::Vec3> gradients(mesh.num_owned_cells());

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto phi_p = field.value(cell_lid);
        const auto& center_p = mesh.cell_centroid(cell_lid);

        std::array<std::array<real_t, 3>, 3> normal{};
        typename mesh_type::Vec3 rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            const auto d = mesh.cell_centroid(other) - center_p;
            const auto phi_delta = field.local_value(other) - phi_p;

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][0] += d.y * d.x;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][0] += d.z * d.x;
            normal[2][1] += d.z * d.y;
            normal[2][2] += d.z * d.z;

            rhs.x += d.x * phi_delta;
            rhs.y += d.y * phi_delta;
            rhs.z += d.z * phi_delta;
        }

        gradients[owned] = detail::solve_3x3(normal, rhs);
    }

    return gradients;
}

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence(const Mesh<Pack>& mesh, const FaceField<Pack>& flux)
{
    using local_ordinal_type = typename Mesh<Pack>::local_ordinal_type;

    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        auto balance = typename Pack::scalar_type{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!flux.is_owned_face(face_lid))
            {
                continue;
            }

            const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
            balance += sign * flux.value(face_lid) * mesh.face_area(face_lid);
        }

        divergence[owned] = balance / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
identity_matrix(const Teuchos::RCP<const typename Pack::map_type>& map,
                typename Pack::scalar_type diagonal = 1.0)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(map, 1));
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(
            static_cast<typename Pack::local_ordinal_type>(row));
        Teuchos::Array<global_ordinal_type> cols{gid};
        Teuchos::Array<typename Pack::scalar_type> vals{diagonal};
        matrix->insertGlobalValues(gid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
diffusion_matrix(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        Teuchos::Array<global_ordinal_type> cols;
        Teuchos::Array<scalar_type> vals;
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            const auto distance =
                (mesh.cell_centroid(other) - mesh.cell_centroid(cell_lid)).norm();
            if (distance <= 0.0)
            {
                throw std::runtime_error("Cannot assemble diffusion across coincident cells.");
            }

            const auto coeff = diffusivity * mesh.face_area(face_lid) / distance;
            diagonal += coeff;
            cols.push_back(mesh.cell_global_id(other));
            vals.push_back(-coeff);
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

} // namespace SimpleFluid::FvmOperators
