/**
 * @file Equation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Composable finite-volume equation and assembly implementation.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "FVM/EquationOperators.hh"
#include "FVM/details/OperatorDetails.hh"
#include "equations/AssembledEquation.hh"
#include "equations/BoundaryConditions.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_Array.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Composable finite-volume equation for a cell-centered unknown.
 *
 * Terms are collected independently on the left- and right-hand sides, then
 * assembled into a Tpetra matrix and scalar or multi-vector right-hand side.
 * Gradient and divergence terms are reserved for coupled block assembly.
 *
 * @tparam StoredField Scalar or vector cell field storage type.
 * @tparam Pack Tpetra type pack used by the field and linear system.
 */
template<class StoredField, TpetraTypePack Pack = DefaultTpetraTypes>
class Equation
{
public:
    using field_type = StoredField;
    using value_traits = typename field_type::value_traits;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using rhs_type = typename field_type::storage_type;
    using operator_type = FVM::OperatorVariant<Pack>;
    using boundary_type_provider =
        std::function<BoundaryConditionType(int)>;
    using boundary_value_provider =
        std::function<scalar_type(int, size_t, size_t)>;

    static_assert(std::is_same_v<
                      typename field_type::location_type,
                      CellLocation>,
                  "Generic Equation currently supports cell unknowns.");

    /**
     * @brief Construct an equation with homogeneous Neumann boundaries.
     * @param unknown Non-null field updated by the assembled equation.
     * @throws std::invalid_argument If @p unknown is null.
     */
    explicit Equation(SP<field_type> unknown)
        : d_unknown(require_unknown(std::move(unknown))),
          d_boundary_type(
              [](int)
              {
                  return BoundaryConditionType::Neumann;
              }),
          d_boundary_value(
              [](int, size_t, size_t)
              {
                  return scalar_type{};
              })
    {
    }

    const std::string& name() const noexcept
    {
        return d_unknown->name();
    }

    field_type& unknown() noexcept { return *d_unknown; }
    const field_type& unknown() const noexcept { return *d_unknown; }

    /**
     * @brief Add and validate an implicit operator term.
     * @return This equation for fluent construction.
     */
    Equation& add_lhs(operator_type term)
    {
        FVM::validate_operator(term, value_traits::components);
        d_lhs.push_back(std::move(term));
        return *this;
    }

    /**
     * @brief Add and validate an explicit source-side operator term.
     * @return This equation for fluent construction.
     */
    Equation& add_rhs(operator_type term)
    {
        FVM::validate_operator(term, value_traits::components);
        d_rhs.push_back(std::move(term));
        return *this;
    }

    const std::vector<operator_type>& lhs_terms() const noexcept
    {
        return d_lhs;
    }

    const std::vector<operator_type>& rhs_terms() const noexcept
    {
        return d_rhs;
    }

    /**
     * @brief Set callbacks used to assemble boundary contributions.
     *
     * The value provider receives batch ID, in-batch face index, and component
     * index.
     *
     * @throws std::invalid_argument If either provider is empty.
     */
    Equation& set_boundary_providers(
        boundary_type_provider type_provider,
        boundary_value_provider value_provider)
    {
        if (!type_provider || !value_provider)
        {
            throw std::invalid_argument(
                "Equation boundary providers must be callable.");
        }
        d_boundary_type = std::move(type_provider);
        d_boundary_value = std::move(value_provider);
        return *this;
    }

    /**
     * @brief Assemble interior operators and boundary conditions.
     * @param options Linear solver configuration retained by the result.
     * @return Matrix, right-hand side, and solution field ready to solve.
     */
    AssembledEquation<field_type, Pack> assemble(
        LinearSolverOptions options = {}) const
    {
        const auto& mesh = d_unknown->mesh();
        auto matrix = Teuchos::rcp(new matrix_type(
            mesh.owned_cell_map(), mesh.overlap_cell_map(), 16));
        auto rhs = make_rhs(mesh.owned_cell_map());

        Teuchos::Array<local_ordinal_type> columns;
        Teuchos::Array<scalar_type> values;
        columns.reserve(32);
        values.reserve(32);

        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            columns.clear();
            values.clear();
            scalar_type diagonal = {};
            std::array<scalar_type, value_traits::components>
                rhs_values{};

            assemble_cell_terms(
                mesh, cell_lid, d_lhs,
                diagonal, columns, values, rhs_values, true);
            assemble_cell_terms(
                mesh, cell_lid, d_rhs,
                diagonal, columns, values, rhs_values, false);

            columns.push_back(cell_lid);
            values.push_back(diagonal);
            matrix->insertLocalValues(
                cell_lid, columns(), values());
            replace_rhs(*rhs, cell_lid, rhs_values);
        }

        assemble_boundary_terms(*matrix, *rhs);
        matrix->fillComplete();
        return AssembledEquation<field_type, Pack>(
            d_unknown, matrix, rhs, options);
    }

private:
    static SP<field_type> require_unknown(SP<field_type> unknown)
    {
        if (!unknown)
        {
            throw std::invalid_argument(
                "Equation requires a non-null unknown field.");
        }
        return unknown;
    }

    static Teuchos::RCP<rhs_type> make_rhs(
        const Teuchos::RCP<const typename Pack::map_type>& map)
    {
        if constexpr (value_traits::is_vector)
        {
            return Teuchos::rcp(new rhs_type(
                map, value_traits::components, true));
        }
        else
        {
            return Teuchos::rcp(new rhs_type(map, true));
        }
    }

    /** @brief Accumulate one matrix row from a sequence of operator terms. */
    template<class MeshType>
    void assemble_cell_terms(
        const MeshType& mesh,
        local_ordinal_type cell_lid,
        const std::vector<operator_type>& terms,
        scalar_type& diagonal,
        Teuchos::Array<local_ordinal_type>& columns,
        Teuchos::Array<scalar_type>& values,
        std::array<scalar_type, value_traits::components>& rhs_values,
        bool lhs) const
    {
        for (const auto& term : terms)
        {
            std::visit(
                [&](const auto& value)
                {
                    using Term = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<
                                      Term,
                                      FVM::TransientOperator<Pack>>)
                    {
                        if (!lhs)
                        {
                            throw std::invalid_argument(
                                "TransientOperator belongs on the LHS.");
                        }
                        const auto coefficient =
                            mesh.cell_volume(cell_lid) / value.time_step;
                        diagonal += coefficient;
                        for (size_t component = 0;
                             component < value_traits::components;
                             ++component)
                        {
                            rhs_values[component] +=
                                coefficient
                              * value.old_value(cell_lid, component);
                        }
                    }
                    else if constexpr (std::is_same_v<
                                           Term,
                                           FVM::ConvectionOperator<Pack>>)
                    {
                        if (!lhs)
                        {
                            throw std::invalid_argument(
                                "ConvectionOperator belongs on the LHS.");
                        }
                        for (const auto face_lid : mesh.faces(cell_lid))
                        {
                            if (!mesh.is_interior_face(face_lid))
                            {
                                continue;
                            }
                            const auto owner_flux = value.face_flux(face_lid);
                            const auto outward_flux =
                                mesh.owner_cell(face_lid) == cell_lid
                              ? owner_flux
                              : -owner_flux;
                            if (outward_flux >= scalar_type{})
                            {
                                diagonal += outward_flux;
                            }
                            else if (mesh.is_interior_face(face_lid))
                            {
                                columns.push_back(
                                    mesh.opposite_or_periodic_neighbor_cell(
                                        face_lid, cell_lid));
                                values.push_back(outward_flux);
                            }
                        }
                    }
                    else if constexpr (std::is_same_v<
                                           Term,
                                           FVM::DiffusionOperator<Pack>>)
                    {
                        if (!lhs)
                        {
                            throw std::invalid_argument(
                                "DiffusionOperator belongs on the LHS.");
                        }
                        for (const auto face_lid : mesh.faces(cell_lid))
                        {
                            if (!mesh.is_interior_face(face_lid))
                            {
                                continue;
                            }
                            const auto other =
                                mesh.opposite_or_periodic_neighbor_cell(
                                    face_lid, cell_lid);
                            const auto coefficient =
                                FVM::detail::
                                    interior_diffusion_coefficient(
                                        mesh, face_lid, cell_lid,
                                        other, value.diffusivity);
                            diagonal += coefficient;
                            columns.push_back(other);
                            values.push_back(-coefficient);
                        }
                    }
                    else if constexpr (std::is_same_v<
                                           Term,
                                           FVM::SourceOperator<Pack>>)
                    {
                        if (lhs)
                        {
                            throw std::invalid_argument(
                                "SourceOperator belongs on the RHS.");
                        }
                        const auto volume = mesh.cell_volume(cell_lid);
                        for (size_t component = 0;
                             component < value_traits::components;
                             ++component)
                        {
                            rhs_values[component] +=
                                volume
                              * value.value(cell_lid, component);
                        }
                    }
                    else
                    {
                        throw std::logic_error(
                            "Gradient and divergence terms require a "
                            "coupled block assembler.");
                    }
                },
                term);
        }
    }

    /** @brief Add owned boundary-face contributions to matrix and RHS. */
    void assemble_boundary_terms(matrix_type& matrix,
                                 rhs_type& rhs) const
    {
        const auto& mesh = d_unknown->mesh();
        const auto* diffusion = find_diffusion();
        const auto* convection = find_convection();
        for (const auto& [batch_id, batch] : mesh.boundary_batches())
        {
            const auto type = d_boundary_type(batch_id);
            for (size_t in_batch = 0;
                 in_batch < batch.face_lids.size();
                 ++in_batch)
            {
                const auto face_lid = batch.face_lids[in_batch];
                if (!mesh.is_owned_face(face_lid)
                    || !mesh.is_boundary_face(face_lid))
                {
                    continue;
                }
                const auto owner = mesh.owner_cell(face_lid);
                std::array<scalar_type, value_traits::components>
                    additions{};

                if (diffusion != nullptr)
                {
                    if (type == BoundaryConditionType::Dirichlet
                        || type == BoundaryConditionType::NoSlip)
                    {
                        const auto coefficient =
                            FVM::detail::
                                boundary_diffusion_coefficient(
                                    mesh, face_lid, owner,
                                    diffusion->diffusivity);
                        local_ordinal_type column = owner;
                        matrix.sumIntoLocalValues(
                            owner,
                            Teuchos::arrayView(&column, 1),
                            Teuchos::arrayView(&coefficient, 1));
                        for (size_t component = 0;
                             component < value_traits::components;
                             ++component)
                        {
                            additions[component] +=
                                coefficient
                              * d_boundary_value(
                                    batch_id, in_batch, component);
                        }
                    }
                    else if (type == BoundaryConditionType::Neumann)
                    {
                        for (size_t component = 0;
                             component < value_traits::components;
                             ++component)
                        {
                            additions[component] +=
                                diffusion->diffusivity
                              * d_boundary_value(
                                    batch_id, in_batch, component)
                              * mesh.face_area(face_lid);
                        }
                    }
                    else if (type == BoundaryConditionType::Robin)
                    {
                        throw std::runtime_error(
                            "Generic Equation does not implement Robin "
                            "boundary conditions.");
                    }
                }

                if (convection != nullptr)
                {
                    const auto outward_flux =
                        convection->face_flux(face_lid);
                    if (outward_flux >= scalar_type{})
                    {
                        local_ordinal_type column = owner;
                        matrix.sumIntoLocalValues(
                            owner,
                            Teuchos::arrayView(&column, 1),
                            Teuchos::arrayView(&outward_flux, 1));
                    }
                    else
                    {
                        for (size_t component = 0;
                             component < value_traits::components;
                             ++component)
                        {
                            additions[component] -=
                                outward_flux
                              * d_boundary_value(
                                    batch_id, in_batch, component);
                        }
                    }
                }

                sum_rhs(rhs, owner, additions);
            }
        }
    }

    const FVM::DiffusionOperator<Pack>* find_diffusion() const
    {
        for (const auto& term : d_lhs)
        {
            if (const auto* value =
                    std::get_if<FVM::DiffusionOperator<Pack>>(&term))
            {
                return value;
            }
        }
        return nullptr;
    }

    const FVM::ConvectionOperator<Pack>* find_convection() const
    {
        for (const auto& term : d_lhs)
        {
            if (const auto* value =
                    std::get_if<FVM::ConvectionOperator<Pack>>(&term))
            {
                return value;
            }
        }
        return nullptr;
    }

    static void replace_rhs(
        rhs_type& rhs,
        local_ordinal_type row,
        const std::array<scalar_type,
                         value_traits::components>& values)
    {
        apply_rhs(
            rhs, row, values,
            [](auto& target, auto... arguments)
            {
                target.replaceLocalValue(arguments...);
            });
    }

    static void sum_rhs(
        rhs_type& rhs,
        local_ordinal_type row,
        const std::array<scalar_type,
                         value_traits::components>& values)
    {
        apply_rhs(
            rhs, row, values,
            [](auto& target, auto... arguments)
            {
                target.sumIntoLocalValue(arguments...);
            });
    }

    template<class Operation>
    static void apply_rhs(
        rhs_type& rhs,
        local_ordinal_type row,
        const std::array<scalar_type,
                         value_traits::components>& values,
        Operation operation)
    {
        if constexpr (value_traits::is_vector)
        {
            for (size_t component = 0;
                 component < value_traits::components;
                 ++component)
            {
                operation(
                    rhs, row, component, values[component]);
            }
        }
        else
        {
            operation(rhs, row, values[0]);
        }
    }

    SP<field_type> d_unknown;
    std::vector<operator_type> d_lhs;
    std::vector<operator_type> d_rhs;
    boundary_type_provider d_boundary_type;
    boundary_value_provider d_boundary_value;
};

} // namespace SimpleFluid
