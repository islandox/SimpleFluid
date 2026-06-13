/**
 * @file FvmCellOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cell-centered finite-volume gradient and divergence operators.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/OperatorDetails.hh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FVM
{

/**
 * @brief Compute a least-squares cell-centered gradient for every owned
 *        cell.
 *
 * @tparam Pack The Tpetra type pack.
 * @param field Scalar cell field whose gradient is computed.
 * @param[out] gradients Vector cell field to receive the gradient at
 *        each owned cell.
 */
template<TpetraTypePack Pack>
void cell_gradient(const CellField<Pack>& field,
                   VectorCellField<Pack>& gradients)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto phi_p = field.value(cell_lid);

        std::array<std::array<real_t, 3>, 3> normal{};
        typename mesh_type::Vec3 rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto d = mesh.cell_center_vector(face_lid, cell_lid);
            const auto phi_delta = field.local_value(other) - phi_p;

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][2] += d.z * d.z;

            rhs.x += d.x * phi_delta;
            rhs.y += d.y * phi_delta;
            rhs.z += d.z * phi_delta;
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        gradients.set_owned_value(
            cell_lid, detail::solve_3x3(normal, rhs));
    }
}

/**
 * @brief Compute least-squares cell-centered gradients for each component
 *        of a vector field.
 *
 * The tensor rows are component-major: gradients.value(cell)[0] is the
 * gradient of the x component, gradients.value(cell)[1] of y, and
 * gradients.value(cell)[2] of z.
 *
 * @tparam Pack The Tpetra type pack.
 * @param field Vector cell field whose component gradients are computed.
 * @param[out] gradients Tensor cell field to receive one 3x3 gradient per
 *        owned cell.
 */
template<TpetraTypePack Pack>
void cell_gradient(const VectorCellField<Pack>& field,
                   TensorCellField<Pack>& gradients)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;
    using tensor_type = typename TensorCellField<Pack>::tensor_type;

    const auto& mesh = field.mesh();
    if (&gradients.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_gradient requires input and output fields on one mesh.");
    }

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value_p = field.value(cell_lid);

        std::array<std::array<real_t, 3>, 3> normal{};
        tensor_type rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            const auto d = mesh.cell_center_vector(face_lid, cell_lid);
            const auto value_delta = field.local_value(other) - value_p;

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][2] += d.z * d.z;

            for (size_t component = 0;
                 component < VectorCellField<Pack>::num_components;
                 ++component)
            {
                const auto delta = value_delta.component(component);
                rhs[component].x += d.x * delta;
                rhs[component].y += d.y * delta;
                rhs[component].z += d.z * delta;
            }
        }

        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];
        tensor_type gradient{};
        for (size_t component = 0;
             component < VectorCellField<Pack>::num_components;
             ++component)
        {
            auto component_normal = normal;
            gradient[component] =
                detail::solve_3x3(component_normal, rhs[component]);
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
}

/**
 * @brief Compute the net flux balance (sum of signed face fluxes) for a
 *        single cell from a FaceField.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_fluxes FaceField of scalar fluxes.
 * @param cell_lid Local ID of the cell whose balance is computed.
 * @return Sum of outward-positive fluxes around @p cell_lid.
 */
template<TpetraTypePack Pack>
typename Pack::scalar_type cell_flux_balance(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid)
{
    typename Pack::scalar_type balance = 0.0;
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        if (!face_fluxes.is_owned_face(face_lid))
        {
            continue;
        }

        const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
        balance += sign * face_fluxes.value(face_lid);
    }

    return balance;
}

/**
 * @brief Compute the volume-normalized divergence at every owned cell
 *        from a FaceField of pre-computed face fluxes.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_fluxes FaceField of scalar fluxes.
 * @return Vector of divergence values indexed by owned-cell local ID.
 */
template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes)
{
    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        divergence[owned] = cell_flux_balance(mesh, face_fluxes, cell_lid)
                          / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

} // namespace SimpleFluid::FVM
