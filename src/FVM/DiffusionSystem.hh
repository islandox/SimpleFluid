/**
 * @file DiffusionSystem.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Scalar and vector diffusion system assembly (orthogonal two-point flux).
 * @version 0.1
 * @date 2026-06-04
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include "utils/export_helper.hh"

#include "geometry/MeshConcepts.hh"
#include "equations/BoundaryConditions.hh"
#include "FVM/details/DiffusionSystemImpl.hh"
#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/FaceStencilMatrix.hh"
#include "fields/CellField.hh"
#include "geometry/MeshHandle.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::FVM
{

/**
 * @brief Holds an assembled scalar diffusion matrix and right-hand side.
 *
 * The operator represents @f$-\nabla\cdot(\Gamma\nabla\phi)@f$ integrated
 * over each control volume.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack>
struct DiffusionSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::vector_type> rhs;
};

/**
 * @brief Holds a shared diffusion matrix and three-component right-hand side.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack>
struct VectorDiffusionSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::multi_vector_type> rhs;
};

/**
 * @brief Assemble a scalar orthogonal diffusion system with source and
 *        boundary contributions.
 *
 * The source term is interpreted per unit volume. Dirichlet conditions
 * contribute a diagonal coefficient and RHS value; Neumann values are
 * interpreted as outward normal gradients and contribute
 * @f$\Gamma g_f |S_f|@f$ to the RHS.
 *
 * @tparam Pack The Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning BoundaryCondition
 *         for (batch id, in-batch face id).
 * @tparam SourceProvider Callable returning scalar source for a cell LID.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @param boundary_condition Boundary-condition provider.
 * @param right_hand_source Source provider.
 * @return Matrix/RHS pair for the scalar diffusion problem.
 */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider,
         class SourceProvider>
DiffusionSystem<Pack>
diffusion_system(
    const MeshType& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source)
{
    return detail::diffusion_system_impl<Pack>(
        mesh,
        diffusivity,
        std::move(boundary_condition),
        std::move(right_hand_source));
}

/** @brief Assemble mapped scalar diffusion with a zero source. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider>
DiffusionSystem<Pack>
diffusion_system(
    const MeshType& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    auto zero_source = [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };
    return detail::diffusion_system_impl<Pack>(
        mesh,
        diffusivity,
        std::move(boundary_condition),
        zero_source);
}

/** @brief Assemble vector diffusion on a mapped mesh interface. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider,
         class SourceProvider>
VectorDiffusionSystem<Pack>
vector_diffusion_system(
    const MeshType& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition,
    SourceProvider right_hand_source)
{
    return detail::vector_diffusion_system_impl<Pack>(
        mesh,
        diffusivity,
        std::move(boundary_condition),
        std::move(right_hand_source));
}

/** @brief Assemble mapped vector diffusion with a zero source. */
template<TpetraTypePack Pack,
         SimpleFluid::MeshType MeshType,
         class BoundaryConditionProvider>
VectorDiffusionSystem<Pack>
vector_diffusion_system(
    const MeshType& mesh,
    typename Pack::scalar_type diffusivity,
    BoundaryConditionProvider boundary_condition)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = vec3<typename Pack::scalar_type>;
    auto zero_source = [](local_ordinal_type) -> vec_type
    {
        return {};
    };
    return detail::vector_diffusion_system_impl<Pack>(
        mesh,
        diffusivity,
        std::move(boundary_condition),
        zero_source);
}

} // namespace SimpleFluid::FVM
