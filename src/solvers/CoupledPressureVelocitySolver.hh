/**
 * @file CoupledPressureVelocitySolver.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Monolithic velocity-pressure assembly and Schur-preconditioned solve.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/OperatorDetails.hh"
#include "solvers/BelosLinearSolver.hh"

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <TpetraExt_MatrixMatrix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Monolithic coupled momentum-pressure linear system.
 *
 * Stores the block matrix, RHS, momentum sub-block, gradient/divergence
 * operators, pressure stabilization, and Schur complement approximation.
 *
 * @tparam Pack Tpetra type pack providing matrix, map, and vector types.
 */
template<TpetraTypePack Pack>
struct CoupledPressureVelocitySystem
{
    using matrix_type = typename Pack::matrix_type;
    using map_type = typename Pack::map_type;
    using vector_type = typename Pack::vector_type;

    Teuchos::RCP<const map_type> map;
    Teuchos::RCP<const map_type> overlap_map;
    Teuchos::RCP<matrix_type> matrix;
    Teuchos::RCP<vector_type> rhs;
    Teuchos::RCP<matrix_type> momentum;
    std::array<Teuchos::RCP<matrix_type>, 3> gradient;
    std::array<Teuchos::RCP<matrix_type>, 3> divergence;
    Teuchos::RCP<matrix_type> pressure_stabilization;
    Teuchos::RCP<matrix_type> schur;
};

/**
 * @brief Convergence result from a monolithic pressure-velocity solve.
 *
 * @tparam Scalar Floating-point type for tolerance values.
 */
template<class Scalar>
struct CoupledPressureVelocityResult
{
    bool converged = false;
    int iterations = 0;
    Scalar achieved_tolerance = {};
};

namespace detail
{

template<class Column, class Scalar>
void add_entry(std::unordered_map<Column, Scalar>& row,
               Column column,
               Scalar value)
{
    row[column] += value;
}

template<TpetraTypePack Pack>
auto make_coupled_maps(const Mesh<Pack>& mesh)
    -> std::pair<Teuchos::RCP<const typename Pack::map_type>,
                 Teuchos::RCP<const typename Pack::map_type>>
{
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using map_type = typename Pack::map_type;

    constexpr global_ordinal_type block_size = 4;
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const auto comm = mesh.owned_cell_map()->getComm();

    auto owned = Teuchos::rcp(new map_type(
        invalid_global_size,
        block_size * mesh.num_owned_cells(),
        global_ordinal_type{0},
        comm));

    std::vector<global_ordinal_type> overlap_gids;
    overlap_gids.reserve(block_size * mesh.num_local_cells());
    for (size_t cell = 0; cell < mesh.num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto cell_gid =
            mesh.overlap_cell_map()->getGlobalElement(cell_lid);
        for (global_ordinal_type component = 0;
             component < block_size;
             ++component)
        {
            overlap_gids.push_back(block_size * cell_gid + component);
        }
    }

    auto overlap = Teuchos::rcp(new map_type(
        invalid_global_size,
        overlap_gids.data(),
        static_cast<local_ordinal_type>(overlap_gids.size()),
        global_ordinal_type{0},
        comm));
    return {owned, overlap};
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
scaled_gradient_matrix(
    const typename Pack::matrix_type& gradient,
    const typename Pack::vector_type& inverse_diagonal)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;

    auto scaled = Teuchos::rcp(new matrix_type(
        gradient.getRowMap(), gradient.getColMap(),
        gradient.getLocalMaxNumRowEntries()));
    const auto diagonal = inverse_diagonal.getData();
    for (size_t row = 0; row < gradient.getLocalNumRows(); ++row)
    {
        typename matrix_type::local_inds_host_view_type columns;
        typename matrix_type::values_host_view_type values;
        gradient.getLocalRowView(
            static_cast<local_ordinal_type>(row), columns, values);

        Teuchos::Array<local_ordinal_type> copied_columns(columns.extent(0));
        Teuchos::Array<scalar_type> copied_values(values.extent(0));
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            copied_columns[entry] = columns[entry];
            copied_values[entry] = diagonal[row] * values[entry];
        }
        scaled->insertLocalValues(
            static_cast<local_ordinal_type>(row),
            copied_columns(), copied_values());
    }
    scaled->fillComplete();
    return scaled;
}

template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
build_schur_approximation(
    const typename Pack::matrix_type& momentum,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& gradient,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& divergence,
    const typename Pack::matrix_type& pressure_stabilization)
{
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using vector_type = typename Pack::vector_type;

    vector_type inverse_diagonal(momentum.getRowMap(), true);
    momentum.getLocalDiagCopy(inverse_diagonal);
    auto diagonal = inverse_diagonal.getDataNonConst();
    for (size_t row = 0; row < diagonal.size(); ++row)
    {
        if (std::abs(diagonal[row]) <= scalar_type{1.0e-30})
        {
            throw std::runtime_error(
                "Coupled pressure-velocity momentum diagonal is singular.");
        }
        diagonal[row] = scalar_type{1} / diagonal[row];
    }

    Teuchos::RCP<matrix_type> accumulated;
    for (size_t component = 0; component < gradient.size(); ++component)
    {
        auto scaled =
            scaled_gradient_matrix<Pack>(*gradient[component],
                                         inverse_diagonal);
        auto product = Teuchos::rcp(new matrix_type(
            divergence[component]->getRowMap(), 32));
        Tpetra::MatrixMatrix::Multiply(
            *divergence[component], false,
            *scaled, false,
            *product, true);

        if (accumulated.is_null())
        {
            accumulated = std::move(product);
        }
        else
        {
            Teuchos::RCP<matrix_type> sum;
            Tpetra::MatrixMatrix::Add(
                *accumulated, false, scalar_type{1},
                *product, false, scalar_type{1},
                sum);
            if (!sum->isFillComplete())
            {
                sum->fillComplete();
            }
            accumulated = std::move(sum);
        }
    }

    auto schur = Teuchos::rcp(new matrix_type(
        accumulated->getRowMap(), 32));
    const auto accumulated_col_map = accumulated->getColMap();
    const auto stabilization_col_map =
        pressure_stabilization.getColMap();
    for (size_t row = 0; row < accumulated->getLocalNumRows(); ++row)
    {
        const auto local_row = static_cast<local_ordinal_type>(row);
        const auto global_row =
            accumulated->getRowMap()->getGlobalElement(local_row);
        if (global_row == global_ordinal_type{0})
        {
            Teuchos::Array<global_ordinal_type> columns{global_row};
            Teuchos::Array<scalar_type> values{scalar_type{1}};
            schur->insertGlobalValues(global_row, columns(), values());
            continue;
        }

        std::unordered_map<global_ordinal_type, scalar_type> row_values;
        typename matrix_type::local_inds_host_view_type local_columns;
        typename matrix_type::values_host_view_type local_values;
        accumulated->getLocalRowView(
            local_row, local_columns, local_values);
        for (size_t entry = 0;
             entry < local_columns.extent(0);
             ++entry)
        {
            row_values[accumulated_col_map->getGlobalElement(
                local_columns[entry])] -= local_values[entry];
        }

        pressure_stabilization.getLocalRowView(
            local_row, local_columns, local_values);
        for (size_t entry = 0;
             entry < local_columns.extent(0);
             ++entry)
        {
            row_values[stabilization_col_map->getGlobalElement(
                local_columns[entry])] += local_values[entry];
        }

        scalar_type row_scale = scalar_type{1};
        for (const auto& [column, value] : row_values)
        {
            static_cast<void>(column);
            row_scale = std::max(row_scale, std::abs(value));
        }
        row_values[global_row] += scalar_type{1.0e-10} * row_scale;

        Teuchos::Array<global_ordinal_type> columns;
        Teuchos::Array<scalar_type> values;
        columns.reserve(row_values.size());
        values.reserve(row_values.size());
        for (const auto& [column, value] : row_values)
        {
            columns.push_back(column);
            values.push_back(value);
        }
        schur->insertGlobalValues(global_row, columns(), values());
    }
    schur->fillComplete();
    return schur;
}

template<TpetraTypePack Pack>
class CoupledSchurPreconditioner final
    : public Pack::operator_type
{
public:
    using map_type = typename Pack::map_type;
    using matrix_type = typename Pack::matrix_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using operator_type = typename Pack::operator_type;
    using scalar_type = typename Pack::scalar_type;
    using momentum_preconditioner_type =
        Ifpack2::Preconditioner<
            scalar_type,
            typename Pack::local_ordinal_type,
            typename Pack::global_ordinal_type,
            typename Pack::node_type>;

    CoupledSchurPreconditioner(
        Teuchos::RCP<const map_type> coupled_map,
        Teuchos::RCP<const map_type> cell_map,
        Teuchos::RCP<const matrix_type> momentum,
        std::array<Teuchos::RCP<matrix_type>, 3> gradient,
        Teuchos::RCP<matrix_type> schur)
        : d_coupled_map(std::move(coupled_map)),
          d_cell_map(std::move(cell_map)),
          d_gradient(std::move(gradient))
    {
        Teuchos::ParameterList momentum_parameters;
        momentum_parameters.set("relaxation: type", "Jacobi");
        momentum_parameters.set("relaxation: sweeps", 1);
        d_momentum_preconditioner =
            Ifpack2::Factory::create("RELAXATION", momentum);
        d_momentum_preconditioner->setParameters(momentum_parameters);
        d_momentum_preconditioner->initialize();
        d_momentum_preconditioner->compute();

        Teuchos::ParameterList pressure_parameters;
        pressure_parameters.set("verbosity", "none");
        pressure_parameters.set("coarse: max size", 64);
        pressure_parameters.set("smoother: type", "RELAXATION");
        pressure_parameters.sublist("smoother: params").set(
            "relaxation: type", "Jacobi");
        pressure_parameters.set("coarse: type", "RELAXATION");
        pressure_parameters.sublist("coarse: params").set(
            "relaxation: type", "Jacobi");
        pressure_parameters.sublist("coarse: params").set(
            "relaxation: sweeps", 4);
        Teuchos::RCP<operator_type> pressure_operator = schur;
        d_pressure_preconditioner =
            MueLu::CreateTpetraPreconditioner(pressure_operator,
                                               pressure_parameters);
    }

    Teuchos::RCP<const map_type> getDomainMap() const override
    {
        return d_coupled_map;
    }

    Teuchos::RCP<const map_type> getRangeMap() const override
    {
        return d_coupled_map;
    }

    void apply(
        const multi_vector_type& input,
        multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha = scalar_type{1},
        scalar_type beta = scalar_type{0}) const override
    {
        if (mode != Teuchos::NO_TRANS)
        {
            throw std::invalid_argument(
                "CoupledSchurPreconditioner does not support transpose apply.");
        }

        const auto num_vectors = input.getNumVectors();
        multi_vector_type pressure_rhs(d_cell_map, num_vectors, true);
        multi_vector_type pressure_solution(d_cell_map, num_vectors, true);
        multi_vector_type velocity_rhs(
            d_cell_map, 3 * num_vectors, true);
        multi_vector_type velocity_solution(
            d_cell_map, 3 * num_vectors, true);

        auto input_view = input.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto pressure_rhs_view =
            pressure_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
        auto velocity_rhs_view =
            velocity_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
        for (size_t cell = 0;
             cell < d_cell_map->getLocalNumElements();
             ++cell)
        {
            for (size_t vector = 0; vector < num_vectors; ++vector)
            {
                pressure_rhs_view(cell, vector) =
                    input_view(4 * cell + 3, vector);
                for (size_t component = 0; component < 3; ++component)
                {
                    velocity_rhs_view(cell, 3 * vector + component) =
                        input_view(4 * cell + component, vector);
                }
            }
        }

        d_pressure_preconditioner->apply(
            pressure_rhs, pressure_solution);

        multi_vector_type gradient_value(
            d_cell_map, num_vectors, true);
        for (size_t component = 0; component < d_gradient.size(); ++component)
        {
            gradient_value.putScalar(scalar_type{});
            d_gradient[component]->apply(
                pressure_solution, gradient_value);
            const auto gradient_view =
                gradient_value.getLocalViewHost(Tpetra::Access::ReadOnly);
            velocity_rhs_view =
                velocity_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
            for (size_t cell = 0;
                 cell < d_cell_map->getLocalNumElements();
                 ++cell)
            {
                for (size_t vector = 0; vector < num_vectors; ++vector)
                {
                    velocity_rhs_view(
                        cell, 3 * vector + component) -=
                        gradient_view(cell, vector);
                }
            }
        }

        d_momentum_preconditioner->apply(
            velocity_rhs, velocity_solution);

        multi_vector_type result(d_coupled_map, num_vectors, true);
        auto result_view =
            result.getLocalViewHost(Tpetra::Access::ReadWrite);
        const auto pressure_view =
            pressure_solution.getLocalViewHost(Tpetra::Access::ReadOnly);
        const auto velocity_view =
            velocity_solution.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t cell = 0;
             cell < d_cell_map->getLocalNumElements();
             ++cell)
        {
            for (size_t vector = 0; vector < num_vectors; ++vector)
            {
                for (size_t component = 0; component < 3; ++component)
                {
                    result_view(4 * cell + component, vector) =
                        velocity_view(cell, 3 * vector + component);
                }
                result_view(4 * cell + 3, vector) =
                    pressure_view(cell, vector);
            }
        }

        output.update(alpha, result, beta);
    }

private:
    Teuchos::RCP<const map_type> d_coupled_map;
    Teuchos::RCP<const map_type> d_cell_map;
    std::array<Teuchos::RCP<matrix_type>, 3> d_gradient;
    Teuchos::RCP<momentum_preconditioner_type>
        d_momentum_preconditioner;
    Teuchos::RCP<operator_type> d_pressure_preconditioner;
};

} // namespace detail

template<TpetraTypePack Pack = DefaultTpetraTypes>
class CoupledPressureVelocitySolver
{
public:
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using face_flux_field_type = FaceField<Pack>;
    using mesh_type = Mesh<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using vector_type = typename Pack::vector_type;
    using momentum_system_type = FVM::VectorTransportSystem<Pack>;
    using system_type = CoupledPressureVelocitySystem<Pack>;
    using result_type = CoupledPressureVelocityResult<scalar_type>;

    explicit CoupledPressureVelocitySolver(SP<const mesh_type> mesh)
        : d_mesh(EquationValidation::require_non_null_mesh(
              std::move(mesh), "CoupledPressureVelocitySolver"))
    {
    }

    system_type assemble(
        const IncompressibleMomentumEquation<Pack>& momentum_equation,
        const velocity_field_type& velocity,
        const field_type& pressure,
        const face_flux_field_type& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options) const
    {
        EquationValidation::require_mesh_match(
            *d_mesh, velocity, "CoupledPressureVelocitySolver");
        EquationValidation::require_mesh_match(
            *d_mesh, pressure, "CoupledPressureVelocitySolver");

        const auto* correction_field =
            time_options.non_orthogonal_treatment
                == FVM::NonOrthogonalTreatment::Implicit
          ? nullptr
          : &velocity;
        const auto momentum = momentum_equation.assemble_system(
            velocity,
            face_fluxes,
            velocity_boundary_cache,
            time_options,
            correction_field);
        return assemble_coupled_system(
            momentum,
            velocity,
            pressure,
            velocity_boundary_cache,
            boundary_conditions,
            time_options);
    }

    system_type assemble(
        const BoussinesqMomentumEquation<Pack>& momentum_equation,
        const velocity_field_type& velocity,
        const field_type& pressure,
        const field_type& temperature,
        const face_flux_field_type& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options,
        const MaterialPropertyFields<Pack>* material = nullptr,
        scalar_type reference_density = scalar_type{1},
        bool density_feedback_enabled = false) const
    {
        EquationValidation::require_mesh_match(
            *d_mesh, velocity, "CoupledPressureVelocitySolver");
        EquationValidation::require_mesh_match(
            *d_mesh, pressure, "CoupledPressureVelocitySolver");
        EquationValidation::require_mesh_match(
            *d_mesh, temperature, "CoupledPressureVelocitySolver");

        const auto* correction_field =
            time_options.non_orthogonal_treatment
                == FVM::NonOrthogonalTreatment::Implicit
          ? nullptr
          : &velocity;
        typename BoussinesqMomentumEquation<Pack>::system_type momentum;
        if (material != nullptr)
        {
            auto zero_source =
                [](local_ordinal_type)
                    -> typename velocity_field_type::vec_type
            {
                return {};
            };
            momentum = momentum_equation.assemble_physical_system(
                velocity,
                face_fluxes,
                temperature,
                velocity_boundary_cache,
                time_options,
                *material,
                reference_density,
                density_feedback_enabled,
                zero_source,
                correction_field);
        }
        else
        {
            momentum = momentum_equation.assemble_system(
                velocity, face_fluxes, temperature,
                velocity_boundary_cache, time_options,
                correction_field);
        }

        return assemble_coupled_system(
            momentum,
            velocity,
            pressure,
            velocity_boundary_cache,
            boundary_conditions,
            time_options);
    }

private:
    system_type assemble_coupled_system(
        const momentum_system_type& momentum,
        const velocity_field_type& velocity,
        const field_type& pressure,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options) const
    {
        auto [coupled_map, coupled_overlap_map] =
            detail::make_coupled_maps(*d_mesh);
        auto coupled_matrix = Teuchos::rcp(new matrix_type(
            coupled_map, coupled_overlap_map, 64));
        auto coupled_rhs = Teuchos::rcp(
            new vector_type(coupled_map, true));

        std::array<Teuchos::RCP<matrix_type>, 3> gradient;
        std::array<Teuchos::RCP<matrix_type>, 3> divergence;
        for (size_t component = 0; component < 3; ++component)
        {
            gradient[component] = Teuchos::rcp(new matrix_type(
                d_mesh->owned_cell_map(),
                d_mesh->overlap_cell_map(), 16));
            divergence[component] = Teuchos::rcp(new matrix_type(
                d_mesh->owned_cell_map(),
                d_mesh->overlap_cell_map(), 16));
        }
        auto pressure_stabilization = Teuchos::rcp(new matrix_type(
            d_mesh->owned_cell_map(),
            d_mesh->overlap_cell_map(), 32));

        const auto boundary_locations =
            FVM::detail::boundary_face_locations(*d_mesh);
        const auto gradient_stencils =
            FVM::detail::least_squares_gradient_stencils(*d_mesh);
        const auto momentum_rhs = momentum.rhs->getLocalViewHost(
            Tpetra::Access::ReadOnly);
        const auto momentum_col_map = momentum.matrix->getColMap();

        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto cell_gid =
                d_mesh->owned_cell_map()->getGlobalElement(cell_lid);

            std::array<
                std::unordered_map<local_ordinal_type, scalar_type>, 3>
                gradient_rows;
            std::array<
                std::unordered_map<local_ordinal_type, scalar_type>, 3>
                divergence_rows;
            std::unordered_map<local_ordinal_type, scalar_type>
                stabilization_row;
            std::array<scalar_type, 3> momentum_boundary_rhs{};
            scalar_type continuity_rhs = {};

            for (const auto face_lid : d_mesh->faces(cell_lid))
            {
                const auto area =
                    d_mesh->face_area_vector_outward(face_lid, cell_lid);
                const std::array<scalar_type, 3> area_components{
                    area.x, area.y, area.z};

                if (d_mesh->is_interior_face(face_lid))
                {
                    const auto other =
                        d_mesh->opposite_or_periodic_neighbor_cell(
                            face_lid, cell_lid);
                    for (size_t component = 0; component < 3; ++component)
                    {
                        detail::add_entry(
                            gradient_rows[component], cell_lid,
                            scalar_type{0.5}
                                * area_components[component]);
                        detail::add_entry(
                            gradient_rows[component], other,
                            scalar_type{0.5}
                                * area_components[component]);
                        detail::add_entry(
                            divergence_rows[component], cell_lid,
                            scalar_type{0.5}
                                * area_components[component]);
                        detail::add_entry(
                            divergence_rows[component], other,
                            scalar_type{0.5}
                                * area_components[component]);
                    }

                    const auto center_delta =
                        d_mesh->cell_center_vector(face_lid, cell_lid);
                    const auto distance_squared =
                        center_delta.dot(center_delta);
                    if (distance_squared > scalar_type{})
                    {
                        const auto direct_weight =
                            area.dot(center_delta) / distance_squared;
                        detail::add_entry(
                            stabilization_row, cell_lid,
                            time_options.time_step * direct_weight);
                        detail::add_entry(
                            stabilization_row, other,
                            -time_options.time_step * direct_weight);

                        auto add_gradient_stencil =
                            [&](local_ordinal_type gradient_cell)
                        {
                            if (!d_mesh->is_owned_cell(gradient_cell)
                                || static_cast<size_t>(gradient_cell)
                                   >= gradient_stencils.size())
                            {
                                return;
                            }
                            for (const auto& entry :
                                 gradient_stencils[
                                     static_cast<size_t>(
                                         gradient_cell)])
                            {
                                detail::add_entry(
                                    stabilization_row,
                                    entry.cell_lid,
                                    scalar_type{0.5}
                                    * time_options.time_step
                                    * entry.coefficient.dot(area));
                            }
                        };
                        add_gradient_stencil(cell_lid);
                        add_gradient_stencil(other);
                    }
                    continue;
                }

                if (static_cast<size_t>(face_lid)
                        >= boundary_locations.size()
                    || !boundary_locations[
                            static_cast<size_t>(face_lid)].active)
                {
                    continue;
                }

                const auto location =
                    boundary_locations[static_cast<size_t>(face_lid)];
                const auto batch_name =
                    d_mesh->boundary_batch_name(location.batch_id);
                const auto pressure_iter =
                    boundary_conditions.pressure.find(batch_name);
                const auto pressure_condition =
                    pressure_iter == boundary_conditions.pressure.end()
                  ? BoundaryCondition{}
                  : pressure_iter->second;

                if (pressure_condition.type
                    == BoundaryConditionType::Dirichlet)
                {
                    for (size_t component = 0; component < 3; ++component)
                    {
                        momentum_boundary_rhs[component] -=
                            pressure_condition.value
                            * area_components[component];
                    }
                }
                else
                {
                    for (size_t component = 0; component < 3; ++component)
                    {
                        detail::add_entry(
                            gradient_rows[component], cell_lid,
                            area_components[component]);
                        momentum_boundary_rhs[component] -=
                            pressure_condition.value
                            * d_mesh->cell_to_face_distance(
                                face_lid, cell_lid)
                            * area_components[component];
                    }
                }

                const auto velocity_type =
                    velocity_boundary_cache.type.at(location.batch_id);
                const auto prescribed =
                    velocity_type == BoundaryConditionType::Slip
                  ? FVM::detail::slip_face_velocity(
                        velocity, face_lid)
                  : velocity_boundary_cache.value.at(
                        location.batch_id)[location.in_batch_id];
                continuity_rhs -= prescribed.dot(area);
            }

            for (size_t component = 0; component < 3; ++component)
            {
                Teuchos::Array<local_ordinal_type> columns;
                Teuchos::Array<scalar_type> values;
                columns.reserve(gradient_rows[component].size());
                values.reserve(gradient_rows[component].size());
                for (const auto& [column, value] :
                     gradient_rows[component])
                {
                    columns.push_back(
                        static_cast<local_ordinal_type>(column));
                    values.push_back(value);
                }
                gradient[component]->insertLocalValues(
                    cell_lid, columns(), values());

                columns.clear();
                values.clear();
                for (const auto& [column, value] :
                     divergence_rows[component])
                {
                    columns.push_back(
                        static_cast<local_ordinal_type>(column));
                    values.push_back(value);
                }
                divergence[component]->insertLocalValues(
                    cell_lid, columns(), values());
            }

            {
                Teuchos::Array<local_ordinal_type> columns;
                Teuchos::Array<scalar_type> values;
                columns.reserve(stabilization_row.size());
                values.reserve(stabilization_row.size());
                for (const auto& [column, value] : stabilization_row)
                {
                    columns.push_back(column);
                    values.push_back(value);
                }
                pressure_stabilization->insertLocalValues(
                    cell_lid, columns(), values());
            }

            typename matrix_type::local_inds_host_view_type
                momentum_columns;
            typename matrix_type::values_host_view_type momentum_values;
            momentum.matrix->getLocalRowView(
                cell_lid, momentum_columns, momentum_values);
            for (size_t component = 0; component < 3; ++component)
            {
                Teuchos::Array<global_ordinal_type> columns;
                Teuchos::Array<scalar_type> values;
                columns.reserve(momentum_columns.extent(0)
                                + gradient_rows[component].size());
                values.reserve(columns.capacity());
                for (size_t entry = 0;
                     entry < momentum_columns.extent(0);
                     ++entry)
                {
                    const auto column_cell_gid =
                        momentum_col_map->getGlobalElement(
                            momentum_columns[entry]);
                    columns.push_back(
                        4 * column_cell_gid
                        + static_cast<global_ordinal_type>(component));
                    values.push_back(momentum_values[entry]);
                }
                for (const auto& [column, value] :
                     gradient_rows[component])
                {
                    const auto column_cell_gid =
                        d_mesh->overlap_cell_map()->getGlobalElement(
                            static_cast<local_ordinal_type>(column));
                    columns.push_back(4 * column_cell_gid + 3);
                    values.push_back(value);
                }

                const auto coupled_row =
                    4 * cell_gid
                    + static_cast<global_ordinal_type>(component);
                coupled_matrix->insertGlobalValues(
                    coupled_row, columns(), values());
                coupled_rhs->replaceGlobalValue(
                    coupled_row,
                    momentum_rhs(cell_lid, component)
                    + momentum_boundary_rhs[component]);
            }

            const auto pressure_row = 4 * cell_gid + 3;
            if (cell_gid == global_ordinal_type{0})
            {
                Teuchos::Array<global_ordinal_type> columns{pressure_row};
                Teuchos::Array<scalar_type> values{scalar_type{1}};
                coupled_matrix->insertGlobalValues(
                    pressure_row, columns(), values());
                coupled_rhs->replaceGlobalValue(
                    pressure_row, scalar_type{});
            }
            else
            {
                Teuchos::Array<global_ordinal_type> columns;
                Teuchos::Array<scalar_type> values;
                for (size_t component = 0; component < 3; ++component)
                {
                    for (const auto& [column, value] :
                         divergence_rows[component])
                    {
                        const auto column_cell_gid =
                            d_mesh->overlap_cell_map()->getGlobalElement(
                                static_cast<local_ordinal_type>(column));
                        columns.push_back(
                            4 * column_cell_gid
                            + static_cast<global_ordinal_type>(component));
                        values.push_back(value);
                    }
                }
                for (const auto& [column, value] : stabilization_row)
                {
                    const auto column_cell_gid =
                        d_mesh->overlap_cell_map()->getGlobalElement(
                            column);
                    columns.push_back(4 * column_cell_gid + 3);
                    values.push_back(value);
                }
                coupled_matrix->insertGlobalValues(
                    pressure_row, columns(), values());
                coupled_rhs->replaceGlobalValue(
                    pressure_row, continuity_rhs);
            }
        }

        for (size_t component = 0; component < 3; ++component)
        {
            gradient[component]->fillComplete();
            divergence[component]->fillComplete();
        }
        pressure_stabilization->fillComplete();
        coupled_matrix->fillComplete(coupled_map, coupled_map);

        auto schur = detail::build_schur_approximation<Pack>(
            *momentum.matrix, gradient, divergence,
            *pressure_stabilization);
        return {
            coupled_map,
            coupled_overlap_map,
            coupled_matrix,
            coupled_rhs,
            momentum.matrix,
            std::move(gradient),
            std::move(divergence),
            std::move(pressure_stabilization),
            std::move(schur)};
    }

public:
    result_type solve(
        const system_type& system,
        velocity_field_type& velocity,
        field_type& pressure,
        const LinearSolverOptions& options) const
    {
        using multi_vector_type = typename Pack::multi_vector_type;
        using operator_type = typename Pack::operator_type;
        using problem_type =
            Belos::LinearProblem<scalar_type,
                                 multi_vector_type,
                                 operator_type>;
        using solver_type =
            Belos::BlockGmresSolMgr<scalar_type,
                                    multi_vector_type,
                                    operator_type>;

        auto solution = Teuchos::rcp(
            new vector_type(system.map, true));
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto cell_gid =
                d_mesh->owned_cell_map()->getGlobalElement(cell_lid);
            const auto value = velocity.value(cell_lid);
            solution->replaceGlobalValue(4 * cell_gid, value.x);
            solution->replaceGlobalValue(4 * cell_gid + 1, value.y);
            solution->replaceGlobalValue(4 * cell_gid + 2, value.z);
            solution->replaceGlobalValue(
                4 * cell_gid + 3, pressure.value(cell_lid));
        }

        auto preconditioner = Teuchos::rcp(
            new detail::CoupledSchurPreconditioner<Pack>(
                system.map,
                d_mesh->owned_cell_map(),
                system.momentum,
                system.gradient,
                system.schur));
        Teuchos::RCP<const operator_type> matrix = system.matrix;
        Teuchos::RCP<const operator_type> right_preconditioner =
            preconditioner;
        auto solution_mv =
            Teuchos::rcp_implicit_cast<multi_vector_type>(solution);
        auto rhs_mv =
            Teuchos::rcp_implicit_cast<const multi_vector_type>(
                system.rhs);
        auto problem = Teuchos::rcp(
            new problem_type(matrix, solution_mv, rhs_mv));
        problem->setRightPrec(right_preconditioner);
        if (!problem->setProblem())
        {
            throw std::runtime_error(
                "Coupled pressure-velocity Belos problem setup failed.");
        }

        auto parameters = Teuchos::rcp(new Teuchos::ParameterList());
        parameters->set("Maximum Iterations", options.max_iterations);
        parameters->set("Convergence Tolerance", options.tolerance);
        parameters->set("Verbosity", options.verbosity);
        parameters->set("Flexible Gmres", true);
        parameters->set("Block Size", 1);
        parameters->set(
            "Num Blocks",
            std::max(
                1,
                std::min(
                    options.max_iterations,
                    static_cast<int>(
                        system.map->getGlobalNumElements()))));
        auto solver = Teuchos::rcp(
            new solver_type(problem, parameters));
        const auto converged =
            solver->solve() == Belos::Converged;

        const auto solution_view =
            solution->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            velocity.set_owned_value(
                cell_lid,
                {solution_view(4 * owned, 0),
                 solution_view(4 * owned + 1, 0),
                 solution_view(4 * owned + 2, 0)});
            pressure.set_owned_value(
                cell_lid, solution_view(4 * owned + 3, 0));
        }
        d_mesh->sync_periodic_boundaries(velocity);
        d_mesh->sync_periodic_boundaries(pressure);

        return {
            converged,
            solver->getNumIters(),
            solver->achievedTol()};
    }

private:
    SP<const mesh_type> d_mesh;
};

} // namespace SimpleFluid
