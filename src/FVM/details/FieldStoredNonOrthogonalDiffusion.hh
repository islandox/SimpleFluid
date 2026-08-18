/**
 * @file FVM/details/FieldStoredNonOrthogonalDiffusion.hh
 * @brief Non-orthogonal steady diffusion for mapped meshes and FieldStored.
 */

#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/DiffusionSystem.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/OperatorDetails.hh"
#include "fields/FieldStored.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM::detail
{

template<class Scalar> struct StoredDiffusionWeights
{
    Scalar implicit{};
    Scalar explicit_{};
};

/** Collectively validate mapped non-orthogonal diffusion inputs. */
template<TpetraTypePack Pack, class MeshType>
StoredDiffusionWeights<typename Pack::scalar_type> validate_stored_diffusion_selection(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field, std::string_view context)
{
    using scalar_type = typename Pack::scalar_type;

    int treatment_state = 3;
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            treatment_state = 0;
            break;
        case NonOrthogonalTreatment::Implicit:
            treatment_state = 1;
            break;
        case NonOrthogonalTreatment::Hybrid:
            treatment_state = 2;
            break;
    }
    const int correction_state = correction_field == nullptr ? 0 : correction_field->mesh_ptr().get() == &mesh ? 1 : 2;
    const std::array<int, 5> local_state{!std::isfinite(diffusivity) || diffusivity < scalar_type{} ? 1 : 0,
        treatment_state, -treatment_state, correction_state, -correction_state};
    auto global_state = local_state;
    const auto communicator = mesh.owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
    {
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_state.size()), local_state.data(),
            global_state.data());
    }

    const auto prefix = std::string(context);
    if (global_state[0] != 0)
    {
        throw std::invalid_argument(prefix + " requires finite non-negative diffusivity.");
    }
    if (global_state[1] == 3)
    {
        throw std::invalid_argument(prefix + " received an unknown non-orthogonal treatment.");
    }
    if (-global_state[2] != global_state[1])
    {
        throw std::invalid_argument(prefix + " requires every rank to use the same treatment.");
    }
    if (-global_state[4] != global_state[3])
    {
        throw std::invalid_argument(prefix + " requires every rank to supply the same correction-field category.");
    }
    if (global_state[3] == 2)
    {
        throw std::invalid_argument(prefix + " requires the correction field on the target mesh.");
    }

    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            return {scalar_type{}, scalar_type{1}};
        case NonOrthogonalTreatment::Implicit:
            return {scalar_type{1}, scalar_type{}};
        case NonOrthogonalTreatment::Hybrid:
            return {scalar_type{0.5}, scalar_type{0.5}};
    }
    throw std::invalid_argument(prefix + " received an unknown non-orthogonal treatment.");
}

/** Evaluate interior least-squares gradients and synchronize overlap values. */
template<TpetraTypePack Pack, class MeshType, class Stencils>
void evaluate_stored_diffusion_gradients(const ScalarCellFieldStored<Pack, MeshType>& field, const Stencils& stencils,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename MeshType::Vec3;

    if (field.mesh_ptr().get() != gradients.mesh_ptr().get() || stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument("Stored diffusion-gradient stencils are incompatible with the mesh.");
    }
    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        vec_type gradient{};
        for (const auto& entry : stencils[owned])
        {
            gradient = gradient + entry.coefficient * field.local_value(entry.cell_lid);
        }
        gradients.set_owned_value(static_cast<local_ordinal_type>(owned), gradient);
    }
    gradients.sync_ghosts();
}

/** Add the lagged tangential diffusion contribution to a mapped RHS. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition>
void add_stored_explicit_diffusion_correction(const ScalarCellFieldStored<Pack, MeshType>& correction_field,
    typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = correction_field.mesh();
    if (diffusivity <= scalar_type{} || correction_weight == scalar_type{})
    {
        return;
    }
    if (!rhs.getMap()->isSameAs(*mesh.owned_cell_map()))
    {
        throw std::invalid_argument("Mapped non-orthogonal correction requires an owned-cell RHS.");
    }

    const auto stencils = least_squares_gradient_stencils(mesh);
    VectorCellFieldStored<Pack, MeshType> gradients(
        correction_field.mesh_ptr(), "stored_diffusion_non_orthogonal_gradient");
    evaluate_stored_diffusion_gradients(correction_field, stencils, gradients);
    const auto boundary_locations = boundary_face_locations(mesh);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto gradient = gradients.local_value(cell_lid);
            typename MeshType::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                gradient = (gradient + gradients.local_value(other)) / scalar_type{2};
                direction = mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto face_index = packed_face_local_id(mesh, face_lid);
                if (face_index >= boundary_locations.size() || !boundary_locations[face_index].active)
                {
                    continue;
                }
                const auto location = boundary_locations[face_index];
                if (boundary_condition(location.batch_id, location.in_batch_id).type !=
                    BoundaryConditionType::Dirichlet)
                {
                    continue;
                }
                direction = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }

            const auto tangential_area =
                non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid), direction);
            rhs.sumIntoLocalValue(cell_lid, correction_weight * diffusivity * gradient.dot(tangential_area));
        }
    }
}

/** Assemble the implicit fraction of mapped steady non-orthogonal diffusion. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class Source>
DiffusionSystem<Pack> stored_implicit_diffusion_system(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    BoundaryCondition boundary_condition, Source right_hand_source, typename Pack::scalar_type implicit_weight,
    const ScalarCellFieldStored<Pack, MeshType>* partition_correction_field)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (!std::isfinite(implicit_weight) || implicit_weight < scalar_type{} || implicit_weight > scalar_type{1})
    {
        throw std::invalid_argument("Mapped non-orthogonal implicit weight must be finite and in [0, 1].");
    }

    const auto gradient_stencils = least_squares_gradient_stencils(mesh);
    const auto boundary_locations = boundary_face_locations(mesh);
    std::unique_ptr<VectorCellFieldStored<Pack, MeshType>> partition_gradients;
    if (implicit_weight > scalar_type{} && partition_correction_field != nullptr)
    {
        partition_gradients = std::make_unique<VectorCellFieldStored<Pack, MeshType>>(
            partition_correction_field->mesh_ptr(), "stored_partition_diffusion_non_orthogonal_gradient");
        evaluate_stored_diffusion_gradients(*partition_correction_field, gradient_stencils, *partition_gradients);
    }

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 32));
    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        row_values.clear();
        auto rhs_value = static_cast<scalar_type>(mesh.cell_volume(cell_lid)) * right_hand_source(cell_lid);

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type face_gradient_weight,
                                              const typename MeshType::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{} || face_gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -implicit_weight * diffusivity * face_gradient_weight;
            for (const auto& entry : gradient_stencils[static_cast<size_t>(gradient_cell_lid)])
            {
                add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto coefficient = interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                add_matrix_entry(row_values, other, -coefficient);

                const auto tangential_area = non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, tangential_area);
                }
                else
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    if (partition_gradients == nullptr)
                    {
                        throw std::logic_error(
                            "Mapped partition-face implicit diffusion requires synchronized gradients.");
                    }
                    rhs_value += implicit_weight * diffusivity * scalar_type{0.5} *
                                 partition_gradients->local_value(other).dot(tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            const auto face_index = packed_face_local_id(mesh, face_lid);
            if (face_index >= boundary_locations.size() || !boundary_locations[face_index].active)
            {
                continue;
            }
            const auto location = boundary_locations[face_index];
            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, cell_lid, diffusivity);
                if (coefficient > scalar_type{})
                {
                    add_matrix_entry(row_values, cell_lid, coefficient);
                    rhs_value += coefficient * condition.value;
                }
                const auto tangential_area =
                    non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                        mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(cell_lid, scalar_type{1}, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs_value += diffusivity * condition.value * static_cast<scalar_type>(mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error("Mapped non-orthogonal diffusion does not implement Robin boundaries.");
            }
        }

        row_values.ensure(cell_lid);
        const auto count = SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(
            row_values.size(), "mapped diffusion row entry count");
        matrix->insertLocalValues(cell_lid, Teuchos::arrayView(row_values.column_data(), count),
            Teuchos::arrayView(row_values.value_data(), count));
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/** Assemble mapped steady diffusion for a selected treatment. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class Source>
DiffusionSystem<Pack> stored_non_orthogonal_diffusion_system(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition, Source right_hand_source,
    NonOrthogonalTreatment treatment, const ScalarCellFieldStored<Pack, MeshType>* correction_field)
{
    using scalar_type = typename Pack::scalar_type;
    const auto weights = validate_stored_diffusion_selection<Pack>(
        mesh, diffusivity, treatment, correction_field, "non_orthogonal_diffusion_system");

    if (weights.implicit == scalar_type{})
    {
        auto system = diffusion_system<Pack>(mesh, diffusivity, boundary_condition, right_hand_source);
        if (correction_field != nullptr && weights.explicit_ > scalar_type{})
        {
            add_stored_explicit_diffusion_correction<Pack>(
                *correction_field, diffusivity, boundary_condition, *system.rhs, weights.explicit_);
        }
        return system;
    }

    auto system = stored_implicit_diffusion_system<Pack>(
        mesh, diffusivity, boundary_condition, right_hand_source, weights.implicit, correction_field);
    if (correction_field != nullptr && weights.explicit_ > scalar_type{})
    {
        add_stored_explicit_diffusion_correction<Pack>(
            *correction_field, diffusivity, boundary_condition, *system.rhs, weights.explicit_);
    }
    return system;
}

/** Solve mapped steady non-orthogonal diffusion using FieldStored data. */
template<TpetraTypePack Pack, class MeshType>
bool solve_stored_non_orthogonal_diffusion(const MeshType& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    ScalarCellFieldStored<Pack, MeshType>& solution, NonOrthogonalTreatment treatment, int nNonOrthogonalCorrectors,
    const LinearSolverOptions& linear_options)
{
    using scalar_type = typename Pack::scalar_type;

    if (solution.mesh_ptr().get() != &mesh)
    {
        throw std::invalid_argument("solve_non_orthogonal_diffusion requires solution on the target mesh.");
    }
    if (nNonOrthogonalCorrectors < 0)
    {
        throw std::invalid_argument("nNonOrthogonalCorrectors cannot be negative.");
    }

    auto solve_system = [&](DiffusionSystem<Pack> system)
    {
        solution.owned_data().putScalar(scalar_type{});
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto converged = solve_linear_system<Pack>(matrix, *system.rhs, solution.owned_data(), linear_options);
        solution.sync_ghosts();
        return converged;
    };

    if (treatment == NonOrthogonalTreatment::Implicit)
    {
        return solve_system(stored_non_orthogonal_diffusion_system<Pack>(
            mesh, diffusivity, boundary_condition, right_hand_source, treatment, &solution));
    }
    if (treatment != NonOrthogonalTreatment::Explicit && treatment != NonOrthogonalTreatment::Hybrid)
    {
        throw std::invalid_argument("Unknown NonOrthogonalTreatment value.");
    }

    for (int corrector = 0; corrector <= nNonOrthogonalCorrectors; ++corrector)
    {
        DiffusionSystem<Pack> system;
        if (treatment == NonOrthogonalTreatment::Explicit)
        {
            system = stored_non_orthogonal_diffusion_system<Pack>(mesh, diffusivity, boundary_condition,
                right_hand_source, treatment, corrector == 0 ? nullptr : &solution);
        }
        else if (corrector == 0)
        {
            system = stored_implicit_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source, scalar_type{0.5}, &solution);
        }
        else
        {
            system = stored_non_orthogonal_diffusion_system<Pack>(
                mesh, diffusivity, boundary_condition, right_hand_source, treatment, &solution);
        }
        if (!solve_system(std::move(system)))
        {
            return false;
        }
    }
    return true;
}

} // namespace SimpleFluid::FVM::detail
