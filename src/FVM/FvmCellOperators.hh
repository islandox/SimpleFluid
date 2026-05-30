/**
 * @file FvmCellOperators.hh
 * @brief Cell-centered finite-volume gradient and divergence operators.
 */
#pragma once

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorFaceField.hh"
#include "FVM/FvmOperatorDetails.hh"

#include <array>
#include <cstddef>
#include <vector>

namespace SimpleFluid::FvmOperators
{

template<TpetraTypePack Pack>
void cell_gradient(const CellField<Pack>& field,
                   std::vector<typename Mesh<Pack>::Vec3>& gradients)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;

    const auto& mesh = field.mesh();
    gradients.assign(mesh.num_owned_cells(), typename mesh_type::Vec3{});

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
        gradients[owned] = detail::solve_3x3(normal, rhs);
    }
}

template<TpetraTypePack Pack>
std::vector<typename Mesh<Pack>::Vec3>
cell_gradient(const CellField<Pack>& field)
{
    std::vector<typename Mesh<Pack>::Vec3> gradients;
    cell_gradient(field, gradients);

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
typename Pack::scalar_type cell_flux_balance(
    const Mesh<Pack>& mesh,
    const std::vector<typename Pack::scalar_type>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid)
{
    if (face_fluxes.size() != mesh.num_faces())
    {
        throw std::invalid_argument("cell_flux_balance received the wrong face-flux size.");
    }

    typename Pack::scalar_type balance = 0.0;
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
        balance += sign * face_fluxes[static_cast<std::size_t>(face_lid)];
    }

    return balance;
}

template<TpetraTypePack Pack>
typename Pack::scalar_type cell_flux_balance(
    const Mesh<Pack>& mesh,
    const VectorFaceField<Pack>& face_velocity,
    typename Pack::local_ordinal_type cell_lid)
{
    if (&face_velocity.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "cell_flux_balance requires a face-velocity field on the input mesh.");
    }

    typename Pack::scalar_type balance = 0.0;
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        if (!face_velocity.is_owned_face(face_lid))
        {
            continue;
        }

        const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
        balance += sign
                 * face_velocity.value(face_lid).dot(mesh.face_normal(face_lid))
                 * mesh.face_area(face_lid);
    }

    return balance;
}

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes(
    const Mesh<Pack>& mesh,
    const std::vector<typename Pack::scalar_type>& face_fluxes)
{
    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        divergence[owned] = cell_flux_balance(mesh, face_fluxes, cell_lid)
                          / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes(
    const Mesh<Pack>& mesh,
    const VectorFaceField<Pack>& face_velocity)
{
    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        divergence[owned] = cell_flux_balance(mesh, face_velocity, cell_lid)
                          / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

} // namespace SimpleFluid::FvmOperators
